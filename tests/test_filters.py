"""Streaming-filter state management and reconstruction tests.

Covers the defects fixed in PR #1 (filter reset on re-tune), PR #1's
StatefulResampler continuity, and the SideNoiseReducer overlap-add
reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sg

from fm_radio.filters import (
    LowpassFilter, BandpassFilter, FIRFilter, NotchFilter,
    DeemphasisIIRFilter, StatefulResampler, SideNoiseReducer, StreamAligner,
)


def _warm(filt_apply, rng, n_blocks=5, size=4096):
    for _ in range(n_blocks):
        filt_apply(rng.standard_normal(size).astype(np.float32))


def test_lowpass_reset_gives_zero_output_for_zero_input(rng):
    lp = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    _warm(lp.apply, rng)
    assert np.linalg.norm(lp.zi) > 0
    lp.reset()
    out = lp.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_bandpass_reset_gives_zero_output_for_zero_input(rng):
    bp = BandpassFilter(order=9, lowcut=23000, highcut=53000, sample_rate=192000)
    _warm(bp.apply, rng)
    bp.reset()
    out = bp.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_notch_reset_gives_zero_output_for_zero_input(rng):
    nf = NotchFilter(freq=19000, Q=30, sample_rate=192000)
    _warm(nf.apply, rng)
    nf.reset()
    out = nf.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_deemphasis_reset_gives_zero_output_for_zero_input():
    de = DeemphasisIIRFilter(sample_rate=48000, tau=50e-6)
    de.process(np.ones(1024, dtype=np.float64))
    assert de.prev_output != 0.0
    de.reset()
    out = de.process(np.zeros(1024, dtype=np.float64))
    assert np.allclose(out, 0.0)


def test_deemphasis_matches_analog_curve():
    """The 1p1z de-emphasis must track analog 1/(1+j2pift) within 0.15 dB.

    The old matched-Z one-pole under-attenuated by up to +1.4 dB at
    15 kHz, which against a real broadcast's analog pre-emphasis is an
    audible HF brightness error.  DC gain must be exactly unity.
    """
    fs, tau = 48000.0, 50e-6
    de = DeemphasisIIRFilter(sample_rate=fs, tau=tau)
    assert abs((de.b0 + de.b1) / (1.0 - de.a1) - 1.0) < 1e-9  # DC gain
    f = np.linspace(20.0, 15000.0, 400)
    z = np.exp(1j * 2 * np.pi * f / fs)
    h = (de.b0 + de.b1 / z) / (1 - de.a1 / z)
    target = 1.0 / (1.0 + 1j * 2 * np.pi * f * tau)
    err_db = 20 * np.log10(np.abs(h)) - 20 * np.log10(np.abs(target))
    assert np.max(np.abs(err_db)) < 0.15, np.max(np.abs(err_db))
    assert abs(de.a1) < 1.0  # stable


def test_lowpass_streaming_matches_oneshot(rng):
    lp_stream = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    lp_ref = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    x = rng.standard_normal(8192)
    y_stream = np.concatenate(
        [lp_stream.apply(x[i:i + 1024]) for i in range(0, x.size, 1024)]
    )
    y_ref = lp_ref.apply(x)
    assert np.allclose(y_stream, y_ref, atol=1e-10)


def test_fir_streaming_matches_oneshot_for_arbitrary_blocks(rng):
    """Overlap-save streaming must equal lfilter(taps, 1, x) exactly.

    The stereo matrix relies on the FIR bank being sample-exact across
    block boundaries; block sizes include tiny (< ntaps) chunks.
    """
    f = FIRFilter.lowpass(321, 15000, 192000, stop=18500)
    x = rng.standard_normal(50_000)
    segs, pos = [], 0
    for n in (3072, 1000, 7, 8000, 3072, 30_000):
        segs.append(f.apply(x[pos:pos + n]))
        pos += n
    streamed = np.concatenate(segs)
    oneshot = sg.lfilter(f.taps, [1.0], x[:pos])
    assert streamed.size == pos
    assert np.max(np.abs(streamed - oneshot)) < 1e-12


def test_fir_bank_response_passband_flat_pilot_dead(rng):
    """15 kHz bank filter: flat to 15 k, ~-100 dB at the 19 kHz pilot."""
    f = FIRFilter.lowpass(321, 15000, 192000, stop=18500)
    w, h = sg.freqz(f.taps, worN=8192, fs=192000)

    def at(freq):
        return 20 * np.log10(np.abs(h[int(np.argmin(np.abs(w - freq)))]))

    assert abs(at(15000)) < 0.1     # passband edge still flat
    assert at(19000) < -95.0        # pilot image dead
    assert f.group_delay_samples == 160


def test_fir_reset_gives_zero_output_for_zero_input(rng):
    f = FIRFilter.lowpass(321, 15000, 192000)
    _warm(f.apply, rng)
    assert np.linalg.norm(f._state) > 0
    f.reset()
    out = f.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_fir_state_is_compact_and_owns_its_memory(rng):
    """The carried state must be ntaps-1 samples that OWN their buffer.

    A view of the extended block would pin the whole previous input
    (state + block) alive through ``.base`` until the next call - an
    arbitrarily large hidden allocation when large blocks are passed
    through the arbitrary-block-size API.
    """
    f = FIRFilter.lowpass(321, 15000, 192000)
    big = rng.standard_normal(500_000)
    f.apply(big)
    assert f._state.size == f.taps.size - 1
    assert f._state.base is None


def test_fir_boundary_block_sizes_match_oneshot(rng):
    """Empty, 1-sample, sub-state-length and beyond-hop blocks are exact."""
    f = FIRFilter.lowpass(321, 15000, 192000)
    x = rng.standard_normal(30_000)
    # 0 (empty), 1, < ntaps-1 (5, 100, 319), == ntaps (321), > hop
    # (hop = nfft - 320 = 3776 for nfft 4096 -> 12_000 spans multiple
    # transforms), remainder
    sizes = (0, 1, 5, 100, 319, 321, 12_000, 3776)
    segs, pos = [], 0
    for n in sizes:
        y = f.apply(x[pos:pos + n])
        assert y.size == n
        segs.append(y)
        pos += n
    segs.append(f.apply(x[pos:]))
    streamed = np.concatenate(segs)
    oneshot = sg.lfilter(f.taps, [1.0], x)
    assert np.max(np.abs(streamed - oneshot)) < 1e-12


def test_fir_reset_matches_fresh_instance(rng):
    f = FIRFilter.lowpass(321, 15000, 192000)
    f.apply(rng.standard_normal(10_000))
    f.reset()
    g = FIRFilter.lowpass(321, 15000, 192000)
    x = rng.standard_normal(8_192)
    assert np.array_equal(f.apply(x), g.apply(x))
    assert np.array_equal(f._state, g._state)


def test_stateful_resampler_matches_oneshot_exactly(rng):
    """Streamed output must equal one-shot resample_poly sample-for-sample.

    The resampler holds back the trailing half-filter-length outputs of
    each block until their FIR support is fully received, so the
    streamed output is a truncated prefix of the one-shot result — but
    every emitted sample must match exactly (no block-edge transients;
    the pre-fix version failed this with errors up to ~0.17 at every
    block end).
    """
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0))
    n_block = 16384
    x = rng.standard_normal(n_block * 4)
    y_stream = np.concatenate(
        [r.process(x[i:i + n_block]) for i in range(0, x.size, n_block)]
    )
    y_ref = sg.resample_poly(x, up, down, window=("kaiser", 10.0))
    # Held-back tail: half_len inputs => half_len*up/down = 30 outputs.
    assert y_ref.size - y_stream.size == 30
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_emit_align_keeps_blocks_decimatable(rng):
    """With emit_align=N every emitted block size is a multiple of N.

    The composite->audio stage decimates each block with a stateless
    resample_poly(1, 4); non-multiple-of-4 composite blocks would shift
    its per-block output grid and add a fractional-sample phase jump at
    every boundary (measured as a THD+N regression from -25.6 to
    -19.1 dB before this alignment existed).
    """
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0), emit_align=4)
    n_block = 16384
    x = rng.standard_normal(n_block * 4)
    segs = [r.process(x[i:i + n_block]) for i in range(0, x.size, n_block)]
    assert all(seg.size % 4 == 0 for seg in segs)
    # Alignment must not break exactness: still a prefix of one-shot.
    y_stream = np.concatenate(segs)
    y_ref = sg.resample_poly(x, up, down, window=("kaiser", 10.0))
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_light_ratio_is_exact(rng):
    """96/125 (light chain) with 16384-sample blocks must be exact.

    16384 is not a multiple of the 96/125 polyphase grid period (125),
    so the fixed-length-tail implementation emitted every block with a
    fractional-phase offset from the global grid - order-of-signal
    errors (max ~4.6 on unit-variance wideband input).  The variable-
    length tail keeps ext grid-aligned for arbitrary block sizes.
    """
    r = StatefulResampler(96, 125)
    x = rng.standard_normal(16384 * 4)
    y_stream = np.concatenate(
        [r.process(x[i:i + 16384]) for i in range(0, x.size, 16384)]
    )
    y_ref = sg.resample_poly(x, 96, 125)
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_arbitrary_block_sizes_are_exact(rng):
    """Mixed, grid-unaligned block sizes must still be an exact prefix."""
    r = StatefulResampler(3, 16, window=("kaiser", 10.0))
    sizes = [1000, 16384, 7, 12345, 500, 9000]
    x = rng.standard_normal(sum(sizes))
    pos, segs = 0, []
    for s in sizes:
        segs.append(r.process(x[pos:pos + s]))
        pos += s
    y_stream = np.concatenate(segs)
    y_ref = sg.resample_poly(x, 3, 16, window=("kaiser", 10.0))
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_reset_gives_repeatable_stream(rng):
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0))
    x = rng.standard_normal(16384)
    y1 = r.process(x)
    r.reset()
    y2 = r.process(x)
    assert np.array_equal(y1, y2)


def test_side_nr_passthrough_is_exact(rng):
    # alpha_floor=1.0 clamps the DD-Wiener gain (always <= 1) to exactly 1
    # in every bin: pure analysis/synthesis reconstruction.
    nr = SideNoiseReducer(
        sample_rate=48000, frame=1024, hop=256, alpha_floor=1.0, beta=1.0,
    )
    n = 48000
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    out = []
    i = 0
    while i < n:
        step = int(rng.integers(50, 500))
        y = nr.process(x[i:i + step])
        if y.size:
            out.append(y)
        i += step
    y = np.concatenate(out)
    lat = nr.latency_samples
    m = y.size - 2 * lat
    assert m > 0
    assert np.allclose(y[lat:lat + m], x[lat:lat + m], atol=1e-5)


def test_side_nr_adapt_false_is_exact_passthrough(rng):
    """adapt=False must be a unity-gain OLA passthrough from stream
    start, independent of block sizes (same contract as the
    alpha_floor=1 case, but with NO FFT and NO model updates)."""
    nr = SideNoiseReducer(sample_rate=48000, frame=1024, hop=256)
    n = 48000
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    out = []
    i = 0
    while i < n:
        step = int(rng.integers(50, 500))
        y = nr.process(x[i:i + step], adapt=False)
        if y.size:
            out.append(y)
        i += step
    y = np.concatenate(out)
    lat = nr.latency_samples
    m = y.size - 2 * lat
    assert m > 0
    assert np.allclose(y[lat:lat + m], x[lat:lat + m], atol=1e-5)
    # no model was ever created
    assert nr.noise_floor is None
    assert nr.power_smooth is None


def test_side_nr_adapt_false_freezes_model_and_advances_time(rng):
    """Long mono (adapt=False silence) must not collapse the NR model.

    Codex P1 on PR #31: feeding the mono path's artificial side = 0
    through the ADAPTING tracker collapsed the minimum-statistics
    floor (measured -35 dB after 1 s, -144 dB after 4 s, -289 dB
    after 8 s of mono) and, because recovery is bounded by
    SIDE_NR_NOISE_DECAY_DB_PER_SEC, pinned the NR at unity gain for
    30-50 s after stereo re-entry.  With adapt=False the spectral
    model must stay BIT-identical across 8 s of silence while the
    output timeline advances, and post-re-entry suppression must
    match a continuously-adapting control immediately.
    """
    fs = 48000
    noise = (rng.standard_normal(4 * fs) * 0.01).astype(np.float32)
    nr = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
    nr.process(noise)                        # learn the floor
    floor = nr.noise_floor.copy()
    psm = nr.power_smooth.copy()
    pg = nr.prev_gain.copy()
    pgm = nr.prev_gamma.copy()

    n_zero = 8 * fs
    out_z = nr.process(np.zeros(n_zero, dtype=np.float32), adapt=False,
                       bypass=True)                  # mono semantics
    assert abs(out_z.size - n_zero) <= nr.frame      # timeline advanced
    assert np.array_equal(nr.noise_floor, floor)     # model bit-frozen
    assert np.array_equal(nr.power_smooth, psm)
    assert np.array_equal(nr.prev_gain, pg)
    assert np.array_equal(nr.prev_gamma, pgm)

    # re-entry: suppression in the first second matches the control
    x2 = (rng.standard_normal(fs) * 0.01).astype(np.float32)
    y2 = nr.process(x2)
    ctrl = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
    ctrl.process(noise)
    y2c = ctrl.process(x2)

    def rms(v):
        return float(np.sqrt(np.mean(v[-fs // 2:] ** 2)))

    sup = 20 * np.log10(rms(y2) / rms(x2) + 1e-15)
    sup_ctrl = 20 * np.log10(rms(y2c) / rms(x2) + 1e-15)
    assert sup < -2.0, sup                   # NR active immediately
    assert abs(sup - sup_ctrl) < 1.0, (sup, sup_ctrl)


def test_side_nr_untrained_mono_start_does_not_poison_floor(rng):
    """Codex P1 round 2 on PR #31: sample provenance, untrained case.

    A reducer built in mono (noise_floor=None) processes silence with
    adapt=False; on switching to adaptive stereo, the input buffer
    still holds up to frame-1 mono-era samples.  Without provenance
    tracking the FIRST adaptive frame - measured 1014/1024 samples of
    silence - initialised the floor at -93 dB and pinned the NR at
    unity gain for ~12 s.  Contract: frames containing ANY
    adapt=False-origin sample are unity OLA and must not initialise
    or update the model; adaptation starts at the first
    fully-adaptive frame, so the floor and first-second suppression
    match a fresh continuously-adaptive control.
    """
    fs = 48000
    for blocks in ("mono-like", "arbitrary"):
        nr = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
        n_zero = 8 * fs
        if blocks == "mono-like":
            z = np.zeros(768, dtype=np.float32)
            for _ in range(n_zero // 768):
                nr.process(z, adapt=False, bypass=True)
        else:
            i = 0
            while i < n_zero:
                step = int(rng.integers(50, 900))
                nr.process(np.zeros(step, dtype=np.float32), adapt=False,
                           bypass=True)
                i += step
        assert nr.noise_floor is None
        assert nr._nonadapt_pending == nr.in_buf.size > 0

        noise = (rng.standard_normal(fs) * 0.01).astype(np.float32)
        # feed just enough adaptive samples that frames still contain
        # mono-era residue: the model must stay uninitialised
        n_taint = max(0, nr.frame - nr.in_buf.size)  # completes 1st frame
        nr.process(noise[:n_taint + 1])
        assert nr.noise_floor is None      # mixed frame did not initialise
        y = nr.process(noise[n_taint + 1:])

        ctrl = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
        yc = ctrl.process(noise)

        def rms(v):
            return float(np.sqrt(np.mean(v[-fs // 2:] ** 2)))

        assert nr.noise_floor is not None  # initialised from clean frames
        floor_db = 10 * np.log10(float(np.median(nr.noise_floor)))
        ctrl_db = 10 * np.log10(float(np.median(ctrl.noise_floor)))
        assert abs(floor_db - ctrl_db) < 3.0, (blocks, floor_db, ctrl_db)
        sup = 20 * np.log10(rms(y) / rms(noise) + 1e-15)
        sup_ctrl = 20 * np.log10(rms(yc) / rms(noise) + 1e-15)
        assert sup < -2.0, (blocks, sup)
        assert abs(sup - sup_ctrl) < 1.0, (blocks, sup, sup_ctrl)


def test_side_nr_freeze_mode_keeps_suppressing(rng):
    """Freeze mode (adapt=False, bypass=False) on a TRAINED reducer.

    Codex P1-2 round 4: the old adapt=False was a unity bypass, so
    closing the stereo low-blend gate dropped the learned suppression
    in one hop (+6.5 dB side-noise step exactly when reception
    degrades).  Freeze mode must keep the gain computation running
    against the FROZEN floor: continued broadband input stays
    suppressed at the adaptive level while only the learned floor is
    bit-frozen (the fast gain state - power_smooth, prev_gain,
    prev_gamma - keeps tracking the content).
    """
    fs = 48000
    noise = (rng.standard_normal(4 * fs) * 0.01).astype(np.float32)
    nr = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
    y_train = nr.process(noise)
    floor = nr.noise_floor.copy()

    x2 = (rng.standard_normal(fs) * 0.01).astype(np.float32)
    y_frozen = nr.process(x2, adapt=False, bypass=False)

    ctrl = SideNoiseReducer(sample_rate=fs, frame=1024, hop=256)
    ctrl.process(noise)
    y_ctrl = ctrl.process(x2)

    def rms(v):
        return float(np.sqrt(np.mean(v[-fs // 2:] ** 2)))

    assert np.array_equal(nr.noise_floor, floor)      # floor frozen
    sup = 20 * np.log10(rms(y_frozen) / rms(x2) + 1e-15)
    sup_ctrl = 20 * np.log10(rms(y_ctrl) / rms(x2) + 1e-15)
    assert sup < -2.0, sup                            # still suppressing
    assert abs(sup - sup_ctrl) < 1.0, (sup, sup_ctrl)


def test_side_nr_reset_clears_provenance_counter(rng):
    nr = SideNoiseReducer(sample_rate=48000, frame=1024, hop=256)
    nr.process(np.zeros(4096, dtype=np.float32), adapt=False)
    assert nr._nonadapt_pending > 0
    nr.reset()
    assert nr._nonadapt_pending == 0
    assert nr.in_buf.size == 0


def test_side_nr_reduces_stationary_noise(rng):
    fs = 48000
    nr = SideNoiseReducer(
        sample_rate=fs, frame=1024, hop=256, alpha_floor=0.15, beta=1.0,
    )
    n = 20 * fs
    x = (rng.standard_normal(n) * 0.02).astype(np.float32)
    out = []
    for i in range(0, n, 480):
        y = nr.process(x[i:i + 480])
        if y.size:
            out.append(y)
    y = np.concatenate(out)
    # Compare in-band (NR acts on 1.5-15 kHz) after settling.
    sos = sg.butter(8, [1500 / (fs / 2), 15000 / (fs / 2)], btype="band",
                    output="sos")
    y_in = sg.sosfilt(sos, y[-5 * fs:])
    x_in = sg.sosfilt(sos, x[-5 * fs:])
    red_db = 20 * np.log10(
        np.sqrt(np.mean(y_in[fs:] ** 2)) / np.sqrt(np.mean(x_in[fs:] ** 2))
    )
    assert red_db < -8.0, f"in-band reduction only {red_db:.1f} dB"


def test_side_nr_preserves_tone_while_reducing_noise(rng):
    """A stationary tone in broadband noise keeps its amplitude.

    The minimum-statistics floor used to climb into a sustained tone's
    own bin (the tone never pauses) and pin its gain at alpha_floor;
    the local-median tracker clamp prevents that.  The broadband noise
    between tones must still be reduced.
    """
    fs = 48000
    nr = SideNoiseReducer(
        sample_rate=fs, frame=1024, hop=256, alpha_floor=0.30, beta=1.0,
        noise_decay_db_per_sec=6.0,
    )
    n = 20 * fs
    t = np.arange(n) / fs
    tone_amp = 0.05
    tone = (tone_amp * np.sin(2 * np.pi * 5000.0 * t)).astype(np.float32)
    noise = (rng.standard_normal(n) * 0.005).astype(np.float32)
    x = tone + noise
    out = []
    for i in range(0, n, 480):
        y = nr.process(x[i:i + 480])
        if y.size:
            out.append(y)
    y = np.concatenate(out)
    # Measure the tone amplitude in the last 5 s by single-bin correlation.
    seg = y[-5 * fs:]
    tt = np.arange(seg.size) / fs
    amp = np.abs(np.mean(seg * np.exp(-1j * 2 * np.pi * 5000.0 * tt)) * 2)
    loss_db = 20 * np.log10(amp / tone_amp)
    # Without protection this measured ~-10 dB (alpha_floor).
    assert loss_db > -1.5, f"tone lost {loss_db:.1f} dB"


def test_side_nr_beta_zero_produces_finite_output(rng):
    # --side-nr-beta 0 is reachable from the CLI; xi=0 with beta=0 used
    # to produce 0/0 = NaN gains that silenced the side channel.
    nr = SideNoiseReducer(
        sample_rate=48000, frame=1024, hop=256, alpha_floor=0.3, beta=0.0,
    )
    x = (rng.standard_normal(48000) * 0.05).astype(np.float32)
    out = []
    for i in range(0, x.size, 480):
        y = nr.process(x[i:i + 480])
        if y.size:
            out.append(y)
    y = np.concatenate(out)
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y ** 2)) > 0  # not silenced


def test_side_nr_reset_restores_initial_state(rng):
    nr = SideNoiseReducer(sample_rate=48000, frame=1024, hop=256)
    nr.process(rng.standard_normal(4096).astype(np.float32))
    assert nr.noise_floor is not None
    nr.reset()
    assert nr.noise_floor is None
    assert nr.in_buf.size == 0


def test_stream_aligner_preserves_order_and_counts():
    al = StreamAligner()
    a = np.arange(10, dtype=np.float32)
    b = np.arange(10, 20, dtype=np.float32)
    out1 = al.feed_and_take(a, 4)
    out2 = al.feed_and_take(b, 10)
    assert np.array_equal(out1, a[:4])
    assert np.array_equal(out2, np.arange(4, 14, dtype=np.float32))
