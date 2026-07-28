"""Demodulator DSP correctness tests.

Covers block-continuity of the IQ lowpass (PR #4), the analytic
heterodyne pilot path (PR #6), the discriminator main demod (PR #7),
mode-coupled subcarrier offsets, and state reset on re-tune.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sg

import fm_radio.demodulator as dm
from fm_radio.constants import COMPOSITE_RATE, SDR_BLOCK_SIZE
from fm_radio.demodulator import FMDemodulator, FMDemodulatorLight


def _random_iq(rng, n):
    return (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64) * 0.3


def test_iq_lowpass_blockwise_matches_oneshot(rng):
    demod = FMDemodulator(stereo=True)
    n = SDR_BLOCK_SIZE * 4
    x = _random_iq(rng, n)
    zi = np.zeros_like(demod._iq_zi)
    y_blocks = []
    for i in range(0, n, SDR_BLOCK_SIZE):
        yb, zi = sg.sosfilt(demod.iq_sos, x[i:i + SDR_BLOCK_SIZE], zi=zi)
        y_blocks.append(yb)
    y_stream = np.concatenate(y_blocks)
    y_ref, _ = sg.sosfilt(demod.iq_sos, x, zi=np.zeros_like(demod._iq_zi))
    assert np.max(np.abs(y_stream - y_ref)) < 1e-9


def _fm_iq(n):
    """Deterministic FM-modulated IQ with realistic bounded deviation.

    Random complex noise is unsuitable for discriminator equivalence
    tests: its instantaneous frequency rides the +-pi branch cut, where
    1e-9 numeric differences flip into 2*pi jumps.  A real FM signal
    keeps |freq| well inside (-pi, pi).
    """
    fs = 1.024e6
    t = np.arange(n) / fs
    mpx = (
        0.30 * np.sin(2 * np.pi * 3_000.0 * t)
        + 0.15 * np.sin(2 * np.pi * 38_000.0 * t)
        + 0.05 * np.sin(2 * np.pi * 53_000.0 * t)
    )
    phase = np.cumsum(mpx)  # amplitudes are directly rad/sample
    return np.exp(1j * phase).astype(np.complex64)


def test_composite_is_block_size_invariant():
    """The IQ->composite chain must be stateful end to end.

    IQ lowpass state, discriminator carry-over and the resampler's
    held-back emission window together must make block-wise processing
    match one-shot processing sample-for-sample.
    """
    n = SDR_BLOCK_SIZE * 4
    x = _fm_iq(n)

    # The DC blocker is a stateful LTI filter, so - unlike the old
    # per-block EMA - it is exactly block-size invariant and stays
    # ACTIVE for this comparison.
    d_one = FMDemodulator(stereo=True)
    comp_one = d_one.process_iq_samples(x)

    d_blk = FMDemodulator(stereo=True)
    comp_blk = np.concatenate([
        d_blk.process_iq_samples(x[i:i + SDR_BLOCK_SIZE])
        for i in range(0, n, SDR_BLOCK_SIZE)
    ])

    assert comp_blk.size == comp_one.size
    assert np.allclose(comp_blk, comp_one, atol=1e-5)


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_dc_blocker_removes_static_dc(cls):
    """The blocker must null a static LO-leak DC (standard AND light).

    Structural: the numerator [1, -1] gives an EXACT zero at 0 Hz,
    the pole sits in (0, 1), the state stays complex128 (the pole at
    1 - 2*pi*fc/fs is precision-critical), and the design time
    constant is 1/(2*pi*0.1 Hz) ~ 1.59 s at each variant's own IQ
    rate.  Behavioural: a pure DC input through the demodulator's own
    _remove_dc decays exponentially and is gone - not merely
    attenuated - after ~7.5 time constants.  Streamed in blocks to
    exercise the carried state.
    """
    d = cls(stereo=True)
    sos = d._dc_sos
    assert sos[0, 0] + sos[0, 1] == 0.0  # exact null at DC
    rho = -sos[0, 4]
    assert 0.0 < rho < 1.0
    tau = 1.0 / ((1.0 - rho) * d.iq_sample_rate)
    assert abs(tau - 1.5915) < 0.02, tau
    assert d._dc_zi.dtype == np.complex128

    dc = np.complex64(0.05 + 0.03j)
    block_n = 16384
    block = np.full(block_n, dc, dtype=np.complex64)
    n_blocks = int(12.0 * d.iq_sample_rate / block_n)  # ~7.5 tau
    last = None
    for _ in range(n_blocks):
        last = d._remove_dc(block)
    assert np.all(np.isfinite(last))
    assert np.all(np.isfinite(d._dc_zi))
    residual = float(np.mean(np.abs(last)))
    assert residual < 1e-3 * abs(dc), residual


@pytest.mark.parametrize("cls,fs", [(FMDemodulator, 1.024e6),
                                    (FMDemodulatorLight, 0.25e6)])
def test_iq_path_returns_to_complex64_after_dc_blocker(cls, fs, rng):
    """Both variants must cast back to complex64 right after the
    blocker: the float64 blocker STATE is precision-critical, the
    discriminator is not (composite delta of the cast: 8.9e-8 max),
    and keeping the light chain in complex128 cost ~10% of its block
    budget (codex P2 on PR #30)."""
    d = cls(stereo=True)
    x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
         ).astype(np.complex64)
    d.process_iq_samples(x)
    assert d._disc_last is not None
    assert d._disc_last.dtype == np.complex64


def test_fir_bank_shares_one_group_delay_and_no_mono_delay():
    """Every mono/side bank filter must have the identical tap count.

    The stereo matrix subtracts the side from the mono path sample for
    sample; the linear-phase bank guarantees alignment ONLY if all
    seven filters share one group delay, in which case no mono delay
    compensation may remain.
    """
    for d in (FMDemodulator(stereo=True), FMDemodulatorLight(stereo=True)):
        bank = (d.lp_mono, d.lp_lr_base, d.lp_lr_base_q,
                d.lp_lr_low, d.lp_lr_low_q, d.lp_lr_mid, d.lp_lr_mid_q)
        sizes = {f.taps.size for f in bank}
        assert len(sizes) == 1, sizes
        assert d.mono_delay_samples == 0


def test_final_audio_lowpass_is_common_and_reconverges_after_stereo(rng):
    """LOCAL L/R chain contract of the final band limit.

    Guarantees only: identical taps on L and R (so the filter cannot
    degrade channel separation), and that mono operation advances the
    audio resamplers, final LPFs and de-emphasis with the same input,
    so L/R sample counts, output grid and filter states re-match after
    genuinely divergent stereo history.

    Side NR is disabled here to ISOLATE that local contract: with the
    shared mid/side NR tail (issue #29) the first mono blocks after a
    stereo -> mono switch legitimately return L != R while the NR
    flushes the previous side content - the end-to-end switch
    behaviour has its own test
    (test_mono_stereo_switches_are_continuous).
    """
    d = FMDemodulator(stereo=True)
    d.side_nr_enabled = False
    assert np.array_equal(d.lp_audio_l.taps, d.lp_audio_r.taps)
    assert d.lp_audio_l is not d.lp_audio_r

    # 1) Stereo blocks with random composite: the side path is nonzero,
    #    so L != R and the per-channel states genuinely diverge.
    for _ in range(4):
        d.demodulate(rng.standard_normal(3072).astype(np.float64) * 0.1)
    assert not np.array_equal(d.lp_audio_l._state, d.lp_audio_r._state)

    # 2) Switch to mono, 3) process standard blocks (3072 samples).
    # The mono path now returns (left-chain, right-chain) outputs, so
    # the FIRST block still carries each chain's divergent-history
    # transient; by the second block every state has re-matched and
    # the returned channels must be bit-identical.
    d.stereo = False
    d.demodulate(rng.standard_normal(3072).astype(np.float64) * 0.1)
    left, right = d.demodulate(rng.standard_normal(3072).astype(np.float64) * 0.1)

    # 4) L/R chains re-matched.
    rl, rr = d._audio_resampler_l, d._audio_resampler_r
    assert rl._in_total == rr._in_total
    assert rl._out_emitted == rr._out_emitted
    assert np.array_equal(rl._prev_tail, rr._prev_tail)
    assert np.array_equal(d.lp_audio_l._state, d.lp_audio_r._state)
    assert np.linalg.norm(d.lp_audio_l._state) > 0
    # De-emphasis is IIR: the divergent-history residue decays as
    # pole^n (~1e-100 after one 768-sample block) - tolerance covers it.
    assert abs(d.deemph_left.prev_input - d.deemph_right.prev_input) < 1e-12
    assert abs(d.deemph_left.prev_output - d.deemph_right.prev_output) < 1e-12
    assert np.array_equal(left, right)


def test_mono_stereo_switches_are_continuous():
    """End-to-end mono <-> stereo switch contract (issue #29).

    On main the NR chain froze during mono, so ~2 blocks of stale
    pre-switch side audio replayed on stereo re-entry (measured side
    RMS 0.31 / 0.19 / floor).  With the shared NR tail
    (_apply_side_nr) plus stereo re-acquisition
    (_reset_stereo_side_state), the stereo -> mono switch FLUSHES the
    held side smoothly instead of dropping it, and re-entry side
    stays at pilot-re-lock-noise level from block 0 (worst case
    measured 4e-4 with the blend forced open; production adaptive
    blend measures ~5e-5).  Both modes share one output latency, so
    the timeline no longer jumps ~16 ms at each switch.
    """
    fs_c = 192_000
    n_blk = 3072

    def stereo_comp(n, p0):
        t = (np.arange(n) + p0) / fs_c
        lmr = 0.45 * np.sin(2 * np.pi * 700.0 * t)
        return (lmr * np.cos(2 * np.pi * 38_000.0 * t)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * t))

    def mono_comp(n, p0):
        t = (np.arange(n) + p0) / fs_c
        return (0.45 * np.sin(2 * np.pi * 300.0 * t)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * t))

    def side_rms(l, r):
        s = 0.5 * (l.astype(np.float64) - r.astype(np.float64))
        return float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0

    d = FMDemodulator(stereo=True)
    d.force_blend_factor = 1.0          # worst case: no blend protection
    d.subcarrier_phase_offset_rad = 0.0
    pos = 0
    for _ in range(40):
        d.demodulate(stereo_comp(n_blk, pos))
        pos += n_blk

    # stereo -> mono: the held side is flushed smoothly, then drains.
    d.stereo = False
    flush = []
    for _ in range(4):
        l, r = d.demodulate(mono_comp(n_blk, pos))
        pos += n_blk
        flush.append(side_rms(l, r))
    assert flush[0] > 0.05, flush       # flushed, not dropped
    assert flush[3] < 1e-4, flush       # fully drained within ~4 blocks
    for _ in range(36):
        d.demodulate(mono_comp(n_blk, pos))
        pos += n_blk

    # mono -> stereo: no stale side from block 0 (main: 0.31 / 0.19).
    d.stereo = True
    for b in range(5):
        l, r = d.demodulate(mono_comp(n_blk, pos))
        pos += n_blk
        assert side_rms(l, r) < 5e-3, (b, side_rms(l, r))

    # Mode-independent output latency: equal emitted counts for equal
    # input (main: the mono path skipped the NR chain's ~16 ms).
    comp = mono_comp(n_blk * 8, 0)
    d_st = FMDemodulator(stereo=True)
    d_st.subcarrier_phase_offset_rad = 0.0
    d_mo = FMDemodulator(stereo=False)
    n_st = sum(d_st.demodulate(comp[i:i + n_blk])[0].size
               for i in range(0, comp.size, n_blk))
    n_mo = sum(d_mo.demodulate(comp[i:i + n_blk])[0].size
               for i in range(0, comp.size, n_blk))
    assert n_st == n_mo


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_mono_and_stereo_share_emission_schedule(cls):
    """Per-block audio emission must be identical in mono and stereo.

    Mode-independent output latency (issue #29): for 3072-sample
    composite blocks the expected schedule is [0, 512, 768, 768, ...]
    with the shared NR tail priming, for BOTH variants and BOTH
    modes.  Compared per block, not just in total.
    """
    fs_c = 192_000
    n_blk = 3072
    t = np.arange(n_blk * 8) / fs_c
    comp = (0.3 * np.sin(2 * np.pi * 1000.0 * t)
            + 0.1 * np.cos(2 * np.pi * 19_000.0 * t))
    d_st = cls(stereo=True)
    d_st.subcarrier_phase_offset_rad = 0.0
    d_mo = cls(stereo=False)
    sched_st = [d_st.demodulate(comp[i:i + n_blk])[0].size
                for i in range(0, comp.size, n_blk)]
    sched_mo = [d_mo.demodulate(comp[i:i + n_blk])[0].size
                for i in range(0, comp.size, n_blk)]
    assert sched_st == sched_mo, (sched_st, sched_mo)
    assert sched_st[:3] == [0, 512, 768], sched_st
    assert all(s == 768 for s in sched_st[2:]), sched_st


def test_mode_transition_flag_edge_cases(rng):
    """_reset_stereo_side_state runs exactly when stereo processing
    resumes after ACTUAL mono processing - once per transition, not
    after reset(), and not on attribute toggles without intervening
    mono demodulation."""
    calls = []

    def make(stereo):
        dm = FMDemodulator(stereo=stereo)
        orig = dm._reset_stereo_side_state

        def spy():
            calls.append(1)
            orig()
        dm._reset_stereo_side_state = spy
        return dm

    comp = rng.standard_normal(3072) * 0.1

    # constructed mono -> mono blocks -> stereo: re-acquire exactly once
    dm = make(stereo=False)
    dm.demodulate(comp)
    dm.demodulate(comp)
    dm.stereo = True
    calls.clear()
    dm.demodulate(comp)
    assert calls == [1]
    dm.demodulate(comp)
    assert calls == [1]

    # mono -> reset() -> stereo: reset cleared everything, no re-acquire
    dm2 = make(stereo=True)
    dm2.stereo = False
    dm2.demodulate(comp)
    dm2.reset()
    dm2.stereo = True
    calls.clear()
    dm2.demodulate(comp)
    assert calls == []

    # attribute toggling with no mono demodulation in between
    dm3 = make(stereo=True)
    dm3.demodulate(comp)
    dm3.stereo = False
    dm3.stereo = True
    calls.clear()
    dm3.demodulate(comp)
    assert calls == []


def test_discriminator_is_default_and_pll_selectable(monkeypatch):
    # Constructed demodulators carry the hardware phase trim on top of
    # each variant's DSP-intrinsic offset (synthetic paths override the
    # attribute with the untrimmed DSP value).
    trim = dm.HARDWARE_SUBCARRIER_PHASE_TRIM_DEG
    d = FMDemodulator(stereo=True)
    assert d.use_pll_demod is False
    assert abs(np.rad2deg(d.subcarrier_phase_offset_rad)
               - (dm.STEREO_SUBCARRIER_PHASE_OFFSET_DEG + trim)) < 0.01

    monkeypatch.setattr(dm, "MAIN_DEMOD_USE_PLL", True)
    d_pll = FMDemodulator(stereo=True)
    assert d_pll.use_pll_demod is True
    assert abs(np.rad2deg(d_pll.subcarrier_phase_offset_rad)
               - (dm.STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL + trim)) < 0.01


def test_light_demodulator_keeps_its_operating_point():
    d = FMDemodulatorLight(stereo=True)
    trim = dm.HARDWARE_SUBCARRIER_PHASE_TRIM_DEG
    assert abs(np.rad2deg(d.subcarrier_phase_offset_rad)
               - (dm.STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT + trim)) < 0.01


def test_pll_mode_produces_finite_composite(rng, monkeypatch):
    monkeypatch.setattr(dm, "MAIN_DEMOD_USE_PLL", True)
    d = FMDemodulator(stereo=True)
    comp = d.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
    assert np.all(np.isfinite(comp))


def test_pilot_heterodyne_tracks_offset_pilot_exactly():
    """Noise-free pilot with carrier offset: phase error must be ~zero.

    The pre-PR#6 FFT-Hilbert path had up to ~12 deg block-edge error in
    this exact scenario.
    """
    fs = COMPOSITE_RATE
    n_block = 3072
    n_blocks = 40
    n = n_block * n_blocks
    f_pilot = 19003.7
    t = np.arange(n) / fs
    true_phase = 2 * np.pi * f_pilot * t
    comp = (0.1 * np.cos(true_phase)).astype(np.float32)

    demod = FMDemodulator(stereo=True)
    est = []
    for i in range(0, n, n_block):
        phase, _resid = demod._estimate_pilot_phase(comp[i:i + n_block])
        est.append(phase)
    est = np.concatenate(est)

    settle = n_block * 20  # let the pilot PLL settle
    err = np.angle(np.exp(1j * (est[settle:] - true_phase[settle:])))
    err = err - np.median(err)
    assert np.rad2deg(np.max(np.abs(err))) < 0.05


def test_pilot_power_scaling_matches_real_bandpass_convention():
    """2*mean(|residual|^2) must equal A^2/2 for a pilot of amplitude A."""
    fs = COMPOSITE_RATE
    n_block = 3072
    amp = 0.1
    t = np.arange(n_block * 30) / fs
    comp = (amp * np.cos(2 * np.pi * 19000.0 * t)).astype(np.float32)

    demod = FMDemodulator(stereo=True)
    powers = []
    for i in range(0, comp.size, n_block):
        _phase, resid = demod._estimate_pilot_phase(comp[i:i + n_block])
        powers.append(2.0 * float(np.mean(np.abs(resid) ** 2)))
    measured = np.mean(powers[10:])
    expected = amp ** 2 / 2.0
    assert abs(measured - expected) / expected < 0.05


def test_reset_clears_all_streaming_state(rng):
    demod = FMDemodulator(stereo=True)
    # Warm every path with random IQ.
    for _ in range(5):
        comp = demod.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
        demod.demodulate(comp)
    demod.reset()
    assert np.all(demod._iq_zi == 0)
    assert np.all(demod._pilot_lp_zi == 0)
    assert np.all(demod._dc_zi == 0)
    assert demod._disc_last is None

    d_light = FMDemodulatorLight(stereo=True)
    for _ in range(3):
        d_light.demodulate(d_light.process_iq_samples(_random_iq(rng, 4096)))
    d_light.reset()
    assert np.all(d_light._dc_zi == 0)
    # Zero composite in -> zero audio out (no leakage from the warm state).
    left, right = demod.demodulate(np.zeros(3072, dtype=np.float32))
    if left.size:
        assert np.allclose(left, 0.0, atol=1e-12)
        assert np.allclose(right, 0.0, atol=1e-12)


def test_mono_audio_is_block_size_invariant():
    """The composite->audio path must be stateful end to end (B1).

    The mono path (mono lowpass -> notches -> audio decimation ->
    de-emphasis -> shared NR tail) contains only block-invariant
    streaming state (stateful filters plus the NR's temporal STFT/OLA
    machinery, which runs adapt-frozen in mono), so block-wise
    processing must match one-shot processing sample-for-sample.  The
    pre-B1 stateless per-block resample_poly failed this with zero-pad
    edge transients at every 16 ms block boundary.
    """
    fs = COMPOSITE_RATE
    n_block = 3072
    n = n_block * 8
    t = np.arange(n) / fs
    comp = (0.3 * np.sin(2 * np.pi * 1000.0 * t)
            + 0.05 * np.sin(2 * np.pi * 9000.0 * t)).astype(np.float32)

    d_one = FMDemodulator(stereo=False)
    audio_one, _ = d_one.demodulate(comp)

    d_blk = FMDemodulator(stereo=False)
    audio_blk = np.concatenate([
        d_blk.demodulate(comp[i:i + n_block])[0]
        for i in range(0, n, n_block)
    ])

    assert audio_blk.size == audio_one.size
    assert np.allclose(audio_blk, audio_one, atol=1e-6)


def test_mono_to_stereo_switch_keeps_lr_aligned():
    """Codex repro (PR #12 review): mono blocks then stereo=True.

    The mono path must advance the right-channel resampler in lockstep,
    otherwise the first stereo block after the switch emits mismatched
    L/R lengths and the mid/side recombination raises ValueError.
    """
    fs = COMPOSITE_RATE
    n_block = 3072
    t = np.arange(n_block) / fs
    comp = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    d = FMDemodulator(stereo=False)
    assert d.side_nr_enabled  # the default, and the crashing config
    for _ in range(3):
        d.demodulate(comp)
    d.stereo = True
    for _ in range(4):
        left, right = d.demodulate(comp)  # must not raise
        assert left.shape == right.shape

    # And the round trip back to mono stays healthy too.
    d.stereo = False
    for _ in range(2):
        left, right = d.demodulate(comp)
        assert left.shape == right.shape


def test_light_demod_immune_to_accumulated_phase():
    """Light demod output must not degrade as absolute phase accumulates.

    The old angle->unwrap->diff kept the unwrapped phase in float32;
    under a carrier offset it grows as 2*pi*df*t, and once it reaches
    ~1e6 rad the float32 spacing (~0.06 rad) dwarfs the per-sample
    step.  Reproducing that pipeline on a pure 50 kHz offset carrier
    measured 0.031 rad/sample of quantisation noise after 4 s (and
    growing); the conj-product discriminator measures 8e-5 and is
    time-invariant.  A constant-frequency input must give a constant
    composite: assert the late-time deviation stays tiny.
    """
    fs = 0.25e6
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    x = np.exp(1j * 2 * np.pi * 50e3 * t).astype(np.complex64)
    d = FMDemodulatorLight(stereo=False)
    comp = np.concatenate([
        d.process_iq_samples(x[i:i + 16384]) for i in range(0, n, 16384)
    ])
    tail = comp[-int(0.5 * fs * 96 / 125):]
    assert float(np.std(tail)) < 1e-3, float(np.std(tail))


def test_light_composite_is_block_size_invariant():
    """The light IQ->composite chain must be block-size invariant.

    The light chain's 96/125 resampler ratio has a polyphase grid
    period of 125, which SDR blocks (16384) never align to; before the
    variable-tail resampler fix, every light block was emitted with a
    fractional-phase offset (B7).
    """
    fs = 0.25e6
    n_block = 16384
    n = n_block * 4
    t = np.arange(n) / fs
    mpx = (0.30 * np.sin(2 * np.pi * 3_000.0 * t)
           + 0.10 * np.sin(2 * np.pi * 38_000.0 * t))
    x = np.exp(1j * np.cumsum(mpx)).astype(np.complex64)

    d_one = FMDemodulatorLight(stereo=True)
    comp_one = d_one.process_iq_samples(x)

    d_blk = FMDemodulatorLight(stereo=True)
    comp_blk = np.concatenate([
        d_blk.process_iq_samples(x[i:i + n_block])
        for i in range(0, n, n_block)
    ])

    assert comp_blk.size == comp_one.size
    assert np.allclose(comp_blk, comp_one, atol=1e-5)


def test_light_demodulator_end_to_end_is_finite(rng):
    d = FMDemodulatorLight(stereo=True)
    x = _random_iq(rng, 4096)
    for _ in range(5):
        comp = d.process_iq_samples(x)
        left, right = d.demodulate(comp)
    assert comp.size % 4 == 0  # emit_align invariant holds for light too
    assert np.all(np.isfinite(comp))
    assert np.all(np.isfinite(left))


def test_demodulate_returns_matched_stereo_pair(rng):
    demod = FMDemodulator(stereo=True)
    comp = demod.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
    left, right = demod.demodulate(comp)
    assert left.shape == right.shape
    assert left.dtype == np.float32
    assert np.all(np.isfinite(left))
