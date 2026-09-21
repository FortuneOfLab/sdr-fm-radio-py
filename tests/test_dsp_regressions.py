"""Numerical regressions from the demodulation/audio-quality review."""

import numpy as np
import pytest

from fm_radio.demodulator import FMDemodulator, FMDemodulatorLight
from fm_radio.filters import SideNoiseReducer, StatefulResampler
from fm_radio.quality_selftest import _build_mpx, _synthesize_iq_tone


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_stereo_zero_startup_does_not_disable_noise_reduction(cls):
    d = cls()
    # No forced blend: startup naturally starts above the adaptation gate.
    for _ in range(3):
        d.demodulate(np.zeros(3072, dtype=np.float32))
    assert d.side_nr.noise_floor is None
    rng = np.random.default_rng(42)
    for i in range(200):
        t = (np.arange(3072) + i * 3072) / d.composite_rate
        mpx = (0.1 * np.cos(2 * np.pi * 19000 * t)
               + 0.001 * rng.standard_normal(t.size)).astype(np.float32)
        left, right = d.demodulate(mpx)
        assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    mask = d.side_nr.band_mask > 0
    assert d.blend_factor > 0.95
    assert np.median(d.side_nr.noise_floor[mask]) > 1e-12
    assert np.median(d.side_nr.prev_gain[mask]) < 0.5


def test_noise_reduction_recovers_empty_bins_after_sparse_tone():
    nr = SideNoiseReducer(48000, alpha_floor=0.3, noise_decay_db_per_sec=6)
    # Exact periodic samples leave spectral nulls even with a Hann window.
    tone = np.tile(np.array([0.02, 0, -0.02, 0], dtype=np.float32), 12000)
    nr.process(tone)
    empty = (nr.noise_floor <= 1e-18) & (nr.band_mask > 0)
    assert np.any(empty)
    rng = np.random.default_rng(51)
    # The first mixed tone/noise frame seeds a small floor; allow the
    # configured 6 dB/s upward leakage to settle. Frequency smoothing
    # also shares gain with adjacent tone-era bins, so do not require
    # these isolated bins to reach alpha_floor exactly.
    noise = (0.02 * rng.standard_normal(8 * 48000)).astype(np.float32)
    y = nr.process(noise)
    assert np.all(np.isfinite(y))
    assert np.median(nr.noise_floor[empty]) > 1e-4
    assert np.median(nr.prev_gamma[empty]) < 2.0
    assert np.median(nr.prev_gain[empty]) < 0.9


@pytest.mark.parametrize("hop", [256, 512])
@pytest.mark.parametrize("bypass", [False, True])
def test_side_ola_has_unity_gain_at_every_hop_position(hop, bypass):
    rng = np.random.default_rng(8)
    x = rng.standard_normal(20000).astype(np.float32)
    nr = SideNoiseReducer(48000, frame=1024, hop=hop, alpha_floor=1.0,
                          gain_freq_smooth_bins=1)
    chunks = [nr.process(x[i:i + 777], bypass=bypass)
              for i in range(0, x.size, 777)]
    y = np.concatenate(chunks)
    # Startup lacks preceding windows; compare fully overlapped samples.
    np.testing.assert_allclose(y[1024:], x[1024:y.size], atol=8e-7, rtol=1e-6)


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_audio_rate_preserves_tone_pitch_and_stream_grid(cls, rate):
    fs = 192000
    t = np.arange(fs) / fs
    mpx = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    def run(block):
        d = cls(final_audio_rate=rate, stereo=False)
        y = np.concatenate([d.demodulate(mpx[i:i + block])[0]
                            for i in range(0, mpx.size, block)])
        # The streaming tail is held until future samples arrive:
        # the resampler holds its half length, and the mid/side NR
        # tail - which is in the chain in every mode, bypassed or
        # not - holds between frame - hop and frame samples, because
        # it emits whole hops.
        tail = d._audio_resampler_l._half_len * rate / fs
        nr = d.side_nr
        held = rate - y.size
        assert tail + nr.frame - nr.hop - 1 <= held <= tail + nr.frame + 1
        return y

    small, whole = run(997), run(fs)
    np.testing.assert_allclose(small, whole, atol=2e-7, rtol=2e-5)
    # Half a second gives exact 2 Hz bins at both supported rates.
    steady = small[-rate // 2:]
    f = np.fft.rfftfreq(steady.size, 1 / rate)
    peak = f[np.argmax(np.abs(np.fft.rfft(steady)))]
    assert abs(peak - 1000) < 2


def test_mpx_subcarrier_crosses_positive_at_both_pilot_zeros():
    # Independent time-domain oracle from BS.450 2.2.2.5.  A=-B>0
    # removes the mono component; fs/19k=40 gives exact pilot-zero indices.
    fs = 760000
    left = np.ones(400, dtype=np.float32) * 0.2
    mpx = _build_mpx(left, -left, fs, fs, 0.1, False, 50e-6, 0.0)
    # Remove the known pilot before checking the subcarrier slope, so
    # the oracle does not depend on the pilot/side amplitude ratio.
    mpx = mpx - 0.1 * np.cos(2 * np.pi * 19000 * np.arange(mpx.size) / fs)
    zeros = np.arange(10, 390, 20)
    assert np.max(np.abs(mpx[zeros])) < 1e-6
    assert np.all(mpx[zeros + 1] - mpx[zeros - 1] > 0)


def test_analytic_iq_transmitter_uses_broadcast_subcarrier_phase():
    fs, deviation, tone_hz = 1024000, 75000, 1000
    iq = _synthesize_iq_tone(0.05, fs, tone_hz, 0.2, -0.2, 0.1,
                             deviation, enable_preemphasis=False)
    t = np.arange(iq.size) / fs
    # Deliberately independent of the transmitter's helper/phase constants.
    expected = (0.1 * np.cos(2 * np.pi * 19000 * t)
                - 0.18 * np.sin(2 * np.pi * tone_hz * t)
                * np.sin(2 * np.pi * 38000 * t))
    measured = np.angle(iq[1:] * np.conj(iq[:-1])) * fs / (2 * np.pi * deviation)
    np.testing.assert_allclose(measured, (expected[1:] + expected[:-1]) / 2,
                               atol=1e-10)


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
@pytest.mark.parametrize("left_only", [True, False])
def test_default_receiver_separates_standard_mpx_without_phase_retuning(cls, left_only):
    # Use literal BS.450 sin/sin convention, not the shared synthesizer.
    # No offset override: this guards the actual production operating point.
    d = cls()
    d.side_nr_enabled = False
    d.iq_phase_correction_enabled = False
    ls, rs = [], []
    for i in range(70):
        t = (np.arange(3072) + i * 3072) / 192000
        tone = 0.2 * np.sin(2 * np.pi * 1000 * t)
        mpx = (tone + (1 if left_only else -1) * tone * np.sin(2 * np.pi * 38000 * t)
               + 0.1 * np.sin(2 * np.pi * 19000 * t))
        l, r = d.demodulate(mpx.astype(np.float32))
        ls.append(l)
        rs.append(r)
    l, r = np.concatenate(ls)[24000:], np.concatenate(rs)[24000:]
    main, leak = (l, r) if left_only else (r, l)
    separation = 20 * np.log10(np.std(main) / np.std(leak))
    assert separation > 30, separation


@pytest.mark.parametrize("cls,fs", [(FMDemodulator, 1024000),
                                  (FMDemodulatorLight, 250000)])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_iq_through_audio_has_no_gaps_across_blocks(cls, fs, rate):
    block = 16384
    iq = _synthesize_iq_tone((block * 12 + 317) / fs, fs, 1000,
                             0.5, 0.3, 0.1, 75000).astype(np.complex64)

    def run(size):
        d = cls(final_audio_rate=rate, stereo=False)
        d.side_nr_enabled = False
        composites, audio = [], []
        for i in range(0, iq.size, size):
            c = d.process_iq_samples(iq[i:i + size])
            composites.append(c)
            audio.append(d.demodulate(c)[0])
        return np.concatenate(composites), np.concatenate(audio)

    comp_blocks, audio_blocks = run(block)
    comp_one, audio_one = run(iq.size)
    assert comp_blocks.size == comp_one.size
    np.testing.assert_allclose(comp_blocks, comp_one, atol=1e-7, rtol=1e-6)
    assert audio_blocks.size == audio_one.size
    np.testing.assert_allclose(audio_blocks, audio_one, atol=2e-7, rtol=2e-5)


def test_resampler_rejects_pending_output_outside_history():
    r = StatefulResampler(3, 16, emit_align=4)
    # Construction now rejects 640. The runtime index guard still
    # protects a stream whose public configuration was changed later.
    r.emit_align = 640
    x = np.random.default_rng(21).standard_normal(16384).astype(np.float32)
    first = r.process(x)
    before = r._in_total
    with pytest.raises(ValueError, match="history no longer covers"):
        r.process(x)
    assert r._in_total == before  # the rejected block was not consumed
    r.reset()
    np.testing.assert_array_equal(r.process(x), first)


@pytest.mark.parametrize("up,down,align", [
    (1, 4, 2), (1, 4, 4), (1, 6, 4), (3, 16, 22), (3, 16, 640),
])
def test_resampler_rejects_truncated_fir_support_at_construction(up, down, align):
    with pytest.raises(ValueError, match="retained FIR history support"):
        StatefulResampler(up, down, emit_align=align)


@pytest.mark.parametrize("up,down,align", [
    (1, 4, 1), (1, 6, 1), (3, 16, 4), (3, 16, 21),
    (96, 125, 4), (147, 640, 1),
])
def test_resampler_accepted_alignment_preserves_values(up, down, align):
    from scipy.signal import resample_poly

    x = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
    r = StatefulResampler(up, down, emit_align=align)
    streamed = np.concatenate([r.process(x[i:i + 100])
                               for i in range(0, x.size, 100)])
    whole = StatefulResampler(up, down, emit_align=align).process(x)
    assert streamed.size == whole.size and streamed.size > 0
    np.testing.assert_array_equal(streamed, whole)
    # Independent reference also checks values, not just sample counts.
    np.testing.assert_allclose(streamed, resample_poly(x, up, down)[:streamed.size],
                               atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"final_audio_rate": 44056},
    {"iq_sample_rate": 1024001},
    {"final_audio_rate": 0},
    {"composite_rate": float("nan")},
    {"iq_sample_rate": float("inf")},
])
def test_expensive_or_invalid_rates_are_rejected_before_filter_design(kwargs, monkeypatch):
    import fm_radio.demodulator as module

    def unexpected_filter(*args, **kw):
        pytest.fail("rate validation must precede filter allocation")

    monkeypatch.setattr(module.FIRFilter, "lowpass", unexpected_filter)
    with pytest.raises(ValueError, match="resampling ratio|finite and positive"):
        FMDemodulator(**kwargs)


def test_trained_nr_retains_power_evidence_across_silence():
    nr = SideNoiseReducer(48000, alpha_floor=0.3, noise_decay_db_per_sec=6)
    rng = np.random.default_rng(58)
    nr.process((0.02 * rng.standard_normal(3 * 48000)).astype(np.float32))
    # Flush the mixed frames, then capture the settled pre-silence model.
    nr.process(np.zeros(4096, dtype=np.float32))
    attrs = ("noise_floor", "power_smooth", "prev_gain", "prev_gamma")
    saved = {key: getattr(nr, key).copy() for key in attrs}
    out = nr.process(np.zeros(2 * 48000, dtype=np.float32))
    assert np.all(out == 0)
    for key in attrs:
        np.testing.assert_array_equal(getattr(nr, key), saved[key])
    mask = nr.band_mask > 0
    recovery = (0.02 * rng.standard_normal(12000)).astype(np.float32)
    nr.process(recovery[:1024])
    drop_db = 10 * np.log10(np.median(nr.noise_floor[mask])
                           / np.median(saved["noise_floor"][mask]))
    assert drop_db > -3, drop_db
    nr.process(recovery[1024:])
    assert np.median(nr.prev_gain[mask]) < 0.5
