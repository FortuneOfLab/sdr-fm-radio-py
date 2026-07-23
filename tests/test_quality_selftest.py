"""quality_selftest loader and CLI-validation tests (PR #5 and PR #1)."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.io import wavfile

from fm_radio.quality_selftest import (
    EPS, _load_iq_wav, _load_stereo_wav, _read_wav_window, _resample,
    _to_float_audio,
)


FS = 48000


def _write_wav(path, data, fs=FS):
    wavfile.write(str(path), fs, data)
    return str(path)


def _load_iq_reference(path, target_fs, duration_s):
    """The pre-mmap eager algorithm, kept verbatim as the oracle."""
    fs, raw = wavfile.read(path)
    x = _to_float_audio(np.asarray(raw))
    i = x[:, 0]
    q = x[:, 1]
    if duration_s is not None and duration_s > 0:
        n = int(duration_s * fs)
        i = i[:n]
        q = q[:n]
    i = _resample(i, fs, target_fs)
    q = _resample(q, fs, target_fs)
    n = min(i.size, q.size)
    iq = i[:n].astype(np.float32) + 1j * q[:n].astype(np.float32)
    peak = float(np.max(np.abs(iq)) + EPS)
    return (0.95 * iq / peak).astype(np.complex64)


@pytest.fixture
def int16_stereo_wav(tmp_path, rng):
    data = (rng.standard_normal((FS, 2)) * 8000).astype(np.int16)
    return _write_wav(tmp_path / "s16.wav", data)


@pytest.mark.parametrize("duration", [None, 0.25, 10.0])
def test_iq_loader_matches_eager_reference(int16_stereo_wav, duration):
    new = _load_iq_wav(int16_stereo_wav, FS, duration)
    ref = _load_iq_reference(int16_stereo_wav, FS, duration)
    assert np.array_equal(new, ref)


def test_iq_loader_rejects_mono(tmp_path, rng):
    mono = (rng.standard_normal(FS) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "mono.wav", mono)
    with pytest.raises(ValueError):
        _load_iq_wav(path, FS, 1.0)


def test_stereo_loader_handles_mono_and_float(tmp_path, rng):
    mono = (rng.standard_normal(FS) * 0.5).astype(np.float32)
    path = _write_wav(tmp_path / "monof.wav", mono)
    left, right = _load_stereo_wav(path, FS, 0.5)
    assert np.array_equal(left, right)
    assert left.size > 0


def test_read_window_bounds_memory_to_the_window(tmp_path, rng):
    data = (rng.standard_normal((FS, 2)) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "w.wav", data)
    fs, raw = _read_wav_window(path, 0.1)
    assert fs == FS
    assert raw.shape[0] == int(0.1 * FS)


def test_read_window_duration_beyond_file_is_clamped(tmp_path, rng):
    data = (rng.standard_normal((100, 2)) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "short.wav", data)
    _fs, raw = _read_wav_window(path, 100.0)
    assert raw.shape[0] == 100


def test_build_mpx_clock_ppm_detunes_pilot():
    from fm_radio.quality_selftest import _build_mpx
    fs_c = 192000
    silence = np.zeros(FS, dtype=np.float32)  # 1 s => 1 Hz FFT bins
    mpx = _build_mpx(silence, silence, FS, fs_c, pilot_amp=0.1,
                     enable_preemphasis=False, preemphasis_tau_s=50e-6,
                     dsb_phase_deg=0.0, clock_ppm=1000.0)
    spec = np.abs(np.fft.rfft(mpx))
    peak_hz = np.argmax(spec) * fs_c / mpx.size
    # 1000 ppm on 19 kHz = +19 Hz.
    assert abs(peak_hz - 19019.0) < 2.0


def test_modulate_carrier_offset_shifts_instantaneous_frequency():
    from fm_radio.quality_selftest import _fm_modulate_iq
    fs_c, fs_iq = 192000, 1024000
    mpx = np.zeros(fs_c // 10, dtype=np.float32)  # unmodulated carrier
    iq = _fm_modulate_iq(mpx, fs_c, fs_iq, 75_000.0, None,
                         carrier_offset_hz=50_000.0)
    freq = np.angle(iq[1:] * np.conj(iq[:-1]))
    measured_hz = float(np.mean(freq)) * fs_iq / (2 * np.pi)
    assert abs(measured_hz - 50_000.0) < 100.0


def test_modulate_multipath_adds_expected_echo():
    from fm_radio.quality_selftest import _fm_modulate_iq
    fs_c, fs_iq = 192000, 1024000
    mpx = np.zeros(fs_c // 10, dtype=np.float32)  # unmodulated carrier
    iq = _fm_modulate_iq(mpx, fs_c, fs_iq, 75_000.0, None,
                         multipath_delay_us=3.0, multipath_gain=0.25,
                         multipath_phase_deg=60.0)
    d = int(round(3.0e-6 * fs_iq))
    expected = abs(1.0 + 0.25 * np.exp(1j * np.deg2rad(60.0)))
    # Before the echo arrives: direct path only; after: combined.
    assert np.allclose(np.abs(iq[:d]), 1.0, atol=1e-5)
    assert np.allclose(np.abs(iq[d + 1:]), expected, atol=1e-5)


def test_modulate_multipath_delay_beyond_signal_is_harmless():
    # Codex repro: --duration 0.01 --multipath-delay-us 10001 makes the
    # echo delay exceed the generated IQ length; the echo simply never
    # arrives (previously raised ValueError on a negative-slice assign).
    from fm_radio.quality_selftest import _fm_modulate_iq
    fs_c, fs_iq = 192000, 1024000
    mpx = np.zeros(fs_c // 100, dtype=np.float32)  # 10 ms
    iq = _fm_modulate_iq(mpx, fs_c, fs_iq, 75_000.0, None,
                         multipath_delay_us=10_001.0, multipath_gain=0.25)
    assert np.all(np.isfinite(iq))
    assert np.allclose(np.abs(iq), 1.0, atol=1e-5)  # direct path only


def test_thdn_median_resists_localised_transient():
    """A short burst must not dominate the windowed-median THD+N.

    This is the B2 hardening: the previous whole-signal single-FFT
    metric let one contaminated edge region swing the reading by over
    10 dB depending on the total duration.
    """
    from fm_radio.quality_selftest import _thdn_db, _thdn_db_single
    fs = 48000
    t = np.arange(4 * fs) / fs
    clean = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float64)
    dirty = clean.copy()
    # 100 ms loud broadband burst in the middle of the signal.
    rng = np.random.default_rng(3)
    burst = rng.standard_normal(fs // 10) * 0.5
    dirty[2 * fs:2 * fs + burst.size] += burst

    whole = _thdn_db_single(dirty, fs, 1000.0)
    windowed = _thdn_db(dirty, fs, 1000.0)
    reference = _thdn_db(clean, fs, 1000.0)
    # Whole-signal FFT is dominated by the burst; the median is not.
    assert whole > -20.0
    assert windowed < -50.0
    assert abs(windowed - reference) < 6.0


def test_snr_median_resists_localised_transient():
    from fm_radio.quality_selftest import _snr_db, _snr_db_single
    fs = 48000
    t = np.arange(4 * fs) / fs
    ref = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float64)
    rng = np.random.default_rng(4)
    x = ref + rng.standard_normal(ref.size) * 1e-3
    x_dirty = x.copy()
    x_dirty[2 * fs:2 * fs + fs // 10] += 0.5
    whole = _snr_db_single(ref, x_dirty)
    windowed = _snr_db(ref, x_dirty)
    clean_windowed = _snr_db(ref, x)
    assert windowed > whole + 10.0
    assert abs(windowed - clean_windowed) < 3.0


def test_main_rejects_duration_not_exceeding_warmup(monkeypatch):
    import fm_radio.quality_selftest as qs
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--duration", "0.5", "--warmup-s", "1.0",
        "--cnr-db", "35",
    ])
    with pytest.raises(SystemExit):
        qs.main()


def test_synthetic_runner_uses_variant_dsp_offset(monkeypatch):
    """Synthetic runners must pick the variant's DSP offset, untrimmed.

    Codex repro from the PR #24 review: with MAIN_DEMOD_USE_PLL=True
    the synthetic default was overwritten with the discriminator's
    316 deg instead of the PLL chain's 285 deg, breaking the PLL A/B
    semantics.  The default must be variant-aware and must never
    include the hardware phase trim (synthetic IQ has no tuner).
    """
    import fm_radio.demodulator as dmod
    import fm_radio.quality_selftest as qs

    captured = {}

    class Spy(qs.FMDemodulator):
        def demodulate(self, composite):
            captured["deg"] = float(
                np.rad2deg(self.subcarrier_phase_offset_rad)
            )
            return super().demodulate(composite)

    monkeypatch.setattr(qs, "FMDemodulator", Spy)
    iq = np.exp(1j * 0.001 * np.arange(40_000)).astype(np.complex64)

    for use_pll, expect in ((False, 316.0), (True, 285.0)):
        monkeypatch.setattr(dmod, "MAIN_DEMOD_USE_PLL", use_pll)
        captured.clear()
        qs._run_demod_from_iq(iq)
        assert abs(captured["deg"] - expect) < 0.01, (use_pll, captured)


@pytest.mark.slow
@pytest.mark.parametrize("cnr_db", [35.0, None])
def test_hifi_tx_matches_legacy_floors_at_1k(cnr_db):
    """The analytic reference modulator reproduces the legacy floors.

    The hifi TX synthesizes every MPX component at the IQ rate with no
    resampling; measuring the same scenario (noisy AND noiseless)
    within ~2 dB of the legacy two-stage-resampled TX proves the
    measurement floor (Sep ~30 dB, THD ~-38 dB at 1 kHz) belongs to
    the RECEIVER, not to resampler images in the test transmitter -
    the fact that unlocked the separation-vs-frequency analysis.
    Both separation directions and both channels' THD+N are held.
    """
    from fm_radio.quality_selftest import evaluate_quality
    np.random.seed(0)
    legacy = evaluate_quality(duration_s=3.0, tone_hz=1000.0, cnr_db=cnr_db,
                              pilot_amp=0.10, freq_dev_hz=75_000.0,
                              warmup_s=0.8)
    np.random.seed(0)
    hifi = evaluate_quality(duration_s=3.0, tone_hz=1000.0, cnr_db=cnr_db,
                            pilot_amp=0.10, freq_dev_hz=75_000.0,
                            warmup_s=0.8, hifi_tx=True)
    assert abs(hifi.separation_l_to_r_db - legacy.separation_l_to_r_db) < 2.5
    assert abs(hifi.separation_r_to_l_db - legacy.separation_r_to_l_db) < 2.5
    assert abs(hifi.thdn_left_db - legacy.thdn_left_db) < 3.0
    assert abs(hifi.thdn_right_db - legacy.thdn_right_db) < 3.0
    assert hifi.separation_l_to_r_db > 24.0
    assert hifi.separation_r_to_l_db > 24.0


def test_hifi_tx_rejects_unsupported_modes():
    from fm_radio.quality_selftest import evaluate_quality
    import pytest as _pytest
    with _pytest.raises(ValueError):
        evaluate_quality(duration_s=1.0, tone_hz=1000.0, cnr_db=None,
                         pilot_amp=0.10, freq_dev_hz=75_000.0,
                         warmup_s=0.3, hifi_tx=True, path="composite")



def _spy_eval(monkeypatch, captured):
    """Replace evaluate_quality with a capture stub (no DSP runs)."""
    import types
    import fm_radio.quality_selftest as qs

    def fake(**kwargs):
        captured.append(kwargs)
        return types.SimpleNamespace(
            thdn_left_db=0.0, thdn_right_db=0.0,
            snr_left_db=0.0, snr_right_db=0.0,
            separation_l_to_r_db=0.0, separation_r_to_l_db=0.0,
            blend_mean=1.0, blend_min=1.0, blend_max=1.0,
        )

    monkeypatch.setattr(qs, "evaluate_quality", fake)


def test_cli_passes_hifi_tx_to_evaluate_quality(monkeypatch):
    """Codex P1 repro: --hifi-tx must reach evaluate_quality via main().

    The flag existed but was missing from eval_kwargs, so the CLI
    silently ran the legacy TX.  Verified at the CLI boundary with the
    evaluation stubbed out (no DSP execution).
    """
    import sys
    import fm_radio.quality_selftest as qs
    captured = []
    _spy_eval(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--duration", "1", "--cnr-db", "-1", "--hifi-tx",
    ])
    qs.main()
    assert len(captured) == 1
    assert captured[0]["hifi_tx"] is True
    assert captured[0]["cnr_db"] is None  # -1 -> noiseless

    captured.clear()
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--duration", "1",
    ])
    qs.main()
    assert captured[0]["hifi_tx"] is False
    assert captured[0]["cnr_db"] == 40.0  # sentinel default resolves to 40


def test_cli_sep_sweep_passes_clock_ppm_and_noiseless_default(monkeypatch):
    """Codex P1 repro: --sep-sweep must honour --clock-ppm, and its
    omitted-cnr default must be NOISELESS (normal eval keeps 40 dB)."""
    import sys
    import fm_radio.quality_selftest as qs
    captured = []
    _spy_eval(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--sep-sweep", "--sep-freqs", "1000,3000",
        "--duration", "1", "--clock-ppm", "200",
    ])
    qs.main()
    assert len(captured) == 2
    for kw in captured:
        assert kw["clock_ppm"] == 200.0
        assert kw["cnr_db"] is None       # omitted -> noiseless for sweep
        assert kw["hifi_tx"] is True
        assert kw["hifi_constant_mod"] is True

    captured.clear()
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--sep-sweep", "--sep-freqs", "1000",
        "--duration", "1", "--cnr-db", "40",
    ])
    qs.main()
    assert captured[0]["cnr_db"] == 40.0  # explicit value wins
