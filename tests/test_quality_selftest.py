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


def test_main_rejects_duration_not_exceeding_warmup(monkeypatch):
    import fm_radio.quality_selftest as qs
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--duration", "0.5", "--warmup-s", "1.0",
        "--cnr-db", "35",
    ])
    with pytest.raises(SystemExit):
        qs.main()
