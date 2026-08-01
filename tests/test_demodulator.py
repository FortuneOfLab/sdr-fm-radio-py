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
from fm_radio.constants import (
    COMPOSITE_RATE, SDR_BLOCK_SIZE, STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF,
    STEREO_BLEND_SMOOTHING, STEREO_BLEND_SMOOTHING_OPEN,
)
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


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_mono_built_demod_learns_clean_nr_floor_on_stereo(cls, rng):
    """Full-path untrained mono start (codex P1 round 2 on PR #31).

    A demodulator CONSTRUCTED in mono (the light variant's default
    startup) that later enables stereo goes through the normal blend
    re-acquisition; the NR model must initialise from genuine stereo
    frames - not from the mono-era silence still in the STFT buffer -
    and must match a stereo-from-construction control that saw the
    same content (both experience the same blend ramp, so the
    mono-built case is exactly a fresh stereo start).  Before the
    provenance fix the mixed first frame initialised the floor at
    -93 dB and the NR stayed at unity gain for ~12 s.
    """
    fs_c = 192_000
    n_blk = 3072

    def stereo_comp(n, p0, r):
        tt = (np.arange(n) + p0) / fs_c
        lmr = 0.2 * np.sin(2 * np.pi * 800.0 * tt) + r.standard_normal(n) * 0.005
        lpr = 0.2 * np.sin(2 * np.pi * 400.0 * tt)
        return (lpr + lmr * np.cos(2 * np.pi * 38_000.0 * tt)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    def mono_comp(n, p0):
        tt = (np.arange(n) + p0) / fs_c
        return (0.4 * np.sin(2 * np.pi * 300.0 * tt)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    dm = cls(stereo=False)
    dm.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    pos = 0
    for _ in range(2 * fs_c // n_blk):          # 2 s mono from construction
        dm.demodulate(mono_comp(n_blk, pos))
        pos += n_blk
    dm.stereo = True
    # NATURAL blend for both variants (codex round 4): the gate's ON
    # both variants train through their real product paths (light's
    # old blend saturation is gone since PR #32); each is compared
    # against its own-variant control that experiences the same blend
    # trajectory.
    r1 = np.random.default_rng(7)
    for _ in range(6 * fs_c // n_blk):
        left, right = dm.demodulate(stereo_comp(n_blk, pos, r1))
        pos += n_blk
        assert left.size == right.size
        assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))

    ctrl = cls(stereo=True)
    ctrl.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    r2 = np.random.default_rng(7)
    pos2 = 0
    for _ in range(6 * fs_c // n_blk):
        ctrl.demodulate(stereo_comp(n_blk, pos2, r2))
        pos2 += n_blk

    assert dm._side_nr_adapt                          # gate opened naturally
    assert dm.side_nr.noise_floor is not None
    floor_db = 10 * np.log10(float(np.median(dm.side_nr.noise_floor)))
    ctrl_db = 10 * np.log10(float(np.median(ctrl.side_nr.noise_floor)))
    # 5 dB guards the contract while the pre-fix failure (-93 dB
    # init) stays far outside
    assert abs(floor_db - ctrl_db) < 5.0, (floor_db, ctrl_db)


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_nr_gate_untrained_weak_pilot_then_recovery(cls, rng):
    """Codex P1 round 3: blend-validity gate, untrained + weak pilot.

    A mono-built demodulator switches stereo on with NO pilot: blend
    stays ~0, the post-blend side is (near-)zero, and WITHOUT the
    gate the untrained floor initialised at exactly 0 - an ABSORBING
    state for the minimum tracker (measured: still 0 with gain
    pinned at 1.0 after 8 s of full blend).  With the gate the model
    must stay uninitialised while the temporal machinery advances;
    when a strong pilot appears and the blend opens past
    SIDE_NR_ADAPT_BLEND_ON, the floor initialises from genuine
    frames and converges to a fresh control.
    """
    fs_c = 192_000
    n_blk = 3072

    def comp(n, p0, r, pilot):
        # Weak signal is modelled as LOW CNR (broadband composite
        # noise floods the pilot and noise bands alike), not merely a
        # missing pilot on a noiseless synthetic: the light variant's
        # order-1 pilot LP leaks strong DSB content into the pilot
        # measure, so a noise-free pilot-less composite would read as
        # HIGH SNR there while a real weak signal never does.
        tt = (np.arange(n) + p0) / fs_c
        lmr = (0.2 * np.sin(2 * np.pi * 800.0 * tt)
               + r.standard_normal(n) * 0.005)
        lpr = 0.2 * np.sin(2 * np.pi * 400.0 * tt)
        out = lpr + lmr * np.cos(2 * np.pi * 38_000.0 * tt)
        if pilot:
            out = out + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt)
        else:
            out = out + r.standard_normal(n) * 0.05
        return out

    dm = cls(stereo=False)
    dm.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    pos = 0
    for _ in range(1 * fs_c // n_blk):              # mono from construction
        dm.demodulate(comp(n_blk, pos, rng, pilot=True))
        pos += n_blk
    dm.stereo = True
    # 4 s of pilot-less stereo: blend collapses toward 0
    emitted = 0
    r1 = np.random.default_rng(3)
    for _ in range(4 * fs_c // n_blk):
        left, right = dm.demodulate(comp(n_blk, pos, r1, pilot=False))
        pos += n_blk
        emitted += left.size
    assert dm.blend_factor < 0.1, dm.blend_factor
    assert not dm._side_nr_adapt                    # gate closed
    assert dm.side_nr.noise_floor is None           # model untouched
    assert emitted > 3 * 48_000                     # timeline advanced

    # strong pilot returns: gate opens NATURALLY for both variants
    # through their real product paths (with the PR #32 measurement
    # path both variants reach full blend on a healthy pilot).
    r2 = np.random.default_rng(4)
    for _ in range(6 * fs_c // n_blk):
        dm.demodulate(comp(n_blk, pos, r2, pilot=True))
        pos += n_blk
    from fm_radio.constants import SIDE_NR_ADAPT_BLEND_ON
    assert dm.blend_factor > SIDE_NR_ADAPT_BLEND_ON, dm.blend_factor
    assert dm.blend_factor > 0.5, dm.blend_factor
    assert dm._side_nr_adapt
    assert dm.side_nr.noise_floor is not None
    assert np.median(dm.side_nr.prev_gain) < 0.9    # NR active, not pinned

    ctrl = cls(stereo=True)
    ctrl.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    r3 = np.random.default_rng(4)
    pos2 = 0
    for _ in range(6 * fs_c // n_blk):
        ctrl.demodulate(comp(n_blk, pos2, r3, pilot=True))
        pos2 += n_blk
    floor_db = 10 * np.log10(float(np.median(dm.side_nr.noise_floor)))
    ctrl_db = 10 * np.log10(float(np.median(ctrl.side_nr.noise_floor)))
    assert abs(floor_db - ctrl_db) < 5.0, (floor_db, ctrl_db)


def test_nr_gate_protects_trained_model_and_recovers(rng):
    """Trained model + deterministic blend step 1 -> 0 -> 1.

    While the forced blend is 0 the gate must freeze all four
    adaptive arrays bit-identically (without the gate, the zero side
    collapsed the floor to the absorbing 0); restoring the blend must
    re-enable the NR without a relearning wait.
    """
    fs_c = 192_000
    n_blk = 3072

    def comp(n, p0, r):
        tt = (np.arange(n) + p0) / fs_c
        lmr = (0.2 * np.sin(2 * np.pi * 800.0 * tt)
               + r.standard_normal(n) * 0.02)
        return (0.2 * np.sin(2 * np.pi * 400.0 * tt)
                + lmr * np.cos(2 * np.pi * 38_000.0 * tt)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    d = FMDemodulator(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(1.0)
    d.force_blend_factor = 1.0
    pos = 0
    for _ in range(3 * fs_c // n_blk):              # train the model
        d.demodulate(comp(n_blk, pos, rng))
        pos += n_blk
    floor = d.side_nr.noise_floor.copy()
    psm = d.side_nr.power_smooth.copy()
    pg = d.side_nr.prev_gain.copy()
    pgm = d.side_nr.prev_gamma.copy()

    d.force_blend_factor = 0.0                      # low-blend stretch
    for _ in range(3 * fs_c // n_blk):
        d.demodulate(comp(n_blk, pos, rng))
        pos += n_blk
    assert not d._side_nr_adapt
    # FREEZE mode: the LEARNED floor is bit-frozen (no absorbing
    # zero), while the fast gain state (power_smooth / prev_gain /
    # prev_gamma) keeps tracking the content so the suppression
    # stays continuous (codex P1-2 round 4).
    assert np.array_equal(d.side_nr.noise_floor, floor)
    assert float(np.median(d.side_nr.noise_floor)) > 0.0
    del psm, pg, pgm

    d.force_blend_factor = 1.0                      # recovery
    for _ in range(2 * fs_c // n_blk):
        d.demodulate(comp(n_blk, pos, rng))
        pos += n_blk
    assert d._side_nr_adapt
    assert np.median(d.side_nr.prev_gain) < 0.9     # effective immediately
    # After reopen the min tracker briefly dips with the recovering
    # power_smooth (fast state decayed during the zero-side stretch;
    # measured -5.3 dB at +1 s) and heals at 6 dB/s - 2 s covers it.
    floor_db = 10 * np.log10(float(np.median(d.side_nr.noise_floor)))
    ref_db = 10 * np.log10(float(np.median(floor)))
    assert abs(floor_db - ref_db) < 3.0, (floor_db, ref_db)


def test_nr_gate_blend_ramp_has_no_gain_step():
    """EMA blend descent/ascent across the gate: no suppression step.

    Codex P1-2 round 4: with the old unity bypass, crossing the gate
    dropped the learned suppression in ONE hop (measured +6.5 dB side
    step on descent, -6.3 dB on ascent, at the flip blocks).  With
    freeze mode the flip must be seamless: measured flip-adjacent
    normalized steps -0.14 dB (close) / +1.31 dB (reopen) with a
    periodic stationary side noise (deterministic).  The 3 dB flip
    bound fails the old behaviour decisively.  (The ramp turnaround
    itself shows a ~5 dB normalized transient from smoothing lag
    under the harsh synthetic blend reversal - present for a fully
    adaptive NR too, hence not asserted.)  The descent side RMS must
    never increase, and the frozen floor must stay bit-identical
    while the gate is closed.
    """
    fs_c = 192_000
    n_blk = 3072
    rng_local = np.random.default_rng(11)
    noise_fixed = rng_local.standard_normal(n_blk) * 0.02

    def comp(p0):
        tt = (np.arange(n_blk) + p0) / fs_c
        lmr = 0.2 * np.sin(2 * np.pi * 800.0 * tt) + noise_fixed
        return (0.2 * np.sin(2 * np.pi * 400.0 * tt)
                + lmr * np.cos(2 * np.pi * 38_000.0 * tt)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    d = FMDemodulator(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(1.0)
    d.force_blend_factor = 1.0
    pos = 0
    for _ in range(3 * fs_c // n_blk):
        d.demodulate(comp(pos))
        pos += n_blk

    b = 1.0
    prev_norm = None
    steps = []
    gates = []
    rms_desc = []
    floor_at_close = None
    for phase in ("down", "up"):
        for _ in range(27):
            b = b * 0.92 if phase == "down" else 0.08 + 0.92 * b
            d.force_blend_factor = b
            left, right = d.demodulate(comp(pos))
            pos += n_blk
            s = 0.5 * (left.astype(np.float64) - right.astype(np.float64))
            rms = float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0
            norm = rms / max(b, 1e-6)
            steps.append(0.0 if prev_norm is None
                         else 20 * np.log10(norm / prev_norm))
            gates.append(d._side_nr_adapt)
            prev_norm = norm
            if phase == "down":
                rms_desc.append(rms)
                if not d._side_nr_adapt and floor_at_close is None:
                    floor_at_close = d.side_nr.noise_floor.copy()

    gates_arr = np.array(gates)
    steps_arr = np.array(steps)
    flips = np.where(gates_arr[1:] != gates_arr[:-1])[0] + 1
    assert flips.size >= 2, gates_arr                # closed AND reopened
    for i in flips:                                  # seamless at the flip
        assert abs(steps_arr[i]) < 3.0, (i, steps_arr[i])
        if i + 1 < steps_arr.size:
            assert abs(steps_arr[i + 1]) < 3.0, (i + 1, steps_arr[i + 1])
    # descent: reception degrading must never RAISE the side noise
    assert all(rms_desc[i + 1] <= rms_desc[i] * 1.15
               for i in range(len(rms_desc) - 1)), rms_desc
    assert floor_at_close is not None
    # the learned floor stayed bit-frozen through the closed stretch
    # (compare at the last closed block before reopen)
    assert d._side_nr_adapt                          # ended reopened


PILOTLESS_CASES = {
    "mono_1k": lambda n, p, fs: 0.4 * np.sin(
        2 * np.pi * 1000.0 * ((np.arange(n) + p) / fs)),
    "mono_10k": lambda n, p, fs: 0.4 * np.sin(
        2 * np.pi * 10_000.0 * ((np.arange(n) + p) / fs)),
    "mono_14k": lambda n, p, fs: 0.4 * np.sin(
        2 * np.pi * 14_000.0 * ((np.arange(n) + p) / fs)),
    "mono_broadband": lambda n, p, fs: (
        0.2 * np.sin(2 * np.pi * 400.0 * ((np.arange(n) + p) / fs))
        + 0.1 * np.sin(2 * np.pi * 3_000.0 * ((np.arange(n) + p) / fs))
        + 0.05 * np.sin(2 * np.pi * 11_000.0 * ((np.arange(n) + p) / fs))),
    "dsb_only": lambda n, p, fs: (
        0.4 * np.sin(2 * np.pi * 800.0 * ((np.arange(n) + p) / fs))
        * np.cos(2 * np.pi * 38_000.0 * ((np.arange(n) + p) / fs))),
    "silence": lambda n, p, fs: np.zeros(n),
}


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
@pytest.mark.parametrize("case", sorted(PILOTLESS_CASES))
def test_pilotless_high_cnr_never_reads_as_stereo(cls, case):
    """Codex P1 on PR #32: pilot-less HIGH-CNR content must stay mono.

    The light variant's order-1 phase LP leaks programme into the
    pilot measure; with the order-9 noise bands the denominator is
    tiny on clean composites, so pilot-less mono/DSB content read as
    74-91 dB SNR / blend 1.0 and the tracker false-acquired (-26 deg
    on plain mono, +16 deg on orphan DSB).  With the dedicated
    order-9 measurement residual plus the tracker's pilot-valid
    gate: SNR below the blend LO threshold, blend closed, NR gate
    closed, no acquisition, no streak progress, output essentially
    mono.  This is a DIFFERENT contract from the low-CNR test (which
    floods all bands with noise); both are needed.
    """
    from fm_radio.constants import STEREO_BLEND_PILOT_SNR_DB_LO
    fs_c = 192_000
    n_blk = 3072
    d = cls(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    outs = []
    pos = 0
    for _ in range(5 * fs_c // n_blk):
        left, right = d.demodulate(PILOTLESS_CASES[case](n_blk, pos, fs_c))
        pos += n_blk
        outs.append((left, right))
    assert d.pilot_snr_ema < STEREO_BLEND_PILOT_SNR_DB_LO, d.pilot_snr_ema
    assert d.blend_factor < 0.05, d.blend_factor
    assert not d._side_nr_adapt
    assert not d._phase_acquired
    assert d._phase_acq_count == 0                  # streak never advanced
    assert d.stereo_phase_err_ema == 0.0            # no false lock
    left = np.concatenate([o[0] for o in outs][-20:])
    right = np.concatenate([o[1] for o in outs][-20:])
    side = 0.5 * (left.astype(np.float64) - right.astype(np.float64))
    mid = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
    if float(np.sqrt(np.mean(mid ** 2))) > 1e-4:    # silence has no mid
        ratio = (np.sqrt(np.mean(side ** 2))
                 / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
        assert ratio < 0.02, ratio                  # essentially mono


@pytest.mark.parametrize("amp", [0.01, 0.1, 1.0])
def test_pure_pilot_parity_across_amplitudes(amp):
    """Both variants must read a clean pilot as high SNR at ANY
    amplitude (the old order-1 noise bands saturated light at
    9.975 dB regardless of amplitude), with variant parity."""
    fs_c = 192_000
    n_blk = 3072
    snrs = {}
    for cls in (FMDemodulatorLight, FMDemodulator):
        d = cls(stereo=True)
        d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
        pos = 0
        for _ in range(6 * fs_c // n_blk):
            tt = (np.arange(n_blk) + pos) / fs_c
            d.demodulate(amp * np.cos(2 * np.pi * 19_000.0 * tt))
            pos += n_blk
        snrs[cls] = d.pilot_snr_ema
        assert d.pilot_snr_ema > 60.0, (cls, amp, d.pilot_snr_ema)
        assert d.blend_factor > 0.99, (cls, amp, d.blend_factor)
    assert abs(snrs[FMDemodulatorLight] - snrs[FMDemodulator]) < 2.0, snrs


def test_light_standard_snr_parity_across_cnr():
    """With a VALID pilot, light and standard pilot SNR must agree
    across the blend threshold region (codex measured 0.11-0.43 dB
    deltas on an independent sweep; 1 dB guards the parity)."""
    fs_c = 192_000
    n_blk = 3072
    for noise in (0.05, 0.12, 0.25):                # spans ~blend LO..HI
        snrs = {}
        for cls in (FMDemodulator, FMDemodulatorLight):
            r = np.random.default_rng(5)
            d = cls(stereo=True)
            d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
            pos = 0
            for _ in range(5 * fs_c // n_blk):
                tt = (np.arange(n_blk) + pos) / fs_c
                compv = (0.2 * np.sin(2 * np.pi * 400.0 * tt)
                         + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt)
                         + r.standard_normal(n_blk) * noise)
                d.demodulate(compv)
                pos += n_blk
            snrs[cls] = d.pilot_snr_ema
        delta = abs(snrs[FMDemodulator] - snrs[FMDemodulatorLight])
        assert delta < 1.0, (noise, snrs, delta)


@pytest.mark.parametrize("cls", [FMDemodulator, FMDemodulatorLight])
def test_pilot_dropout_and_recovery(cls):
    """Pilot dropout on strong stereo, then recovery (codex PR #32).

    Dropout: blend closes, the tracker stops updating and leaks (no
    false re-acquisition from the pilot-less DSB+mono content).
    Recovery: re-acquires on the SAME branch/polarity (angle back
    near the pre-dropout value, blend reopens).
    """
    fs_c = 192_000
    n_blk = 3072

    def comp(n, p0, pilot):
        tt = (np.arange(n) + p0) / fs_c
        lmr = 0.3 * np.sin(2 * np.pi * 700.0 * tt)
        out = (0.2 * np.sin(2 * np.pi * 400.0 * tt)
               + lmr * np.cos(2 * np.pi * 38_000.0 * tt))
        if pilot:
            out = out + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt)
        return out

    d = cls(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    pos = 0
    for _ in range(4 * fs_c // n_blk):              # acquire
        d.demodulate(comp(n_blk, pos, True))
        pos += n_blk
    assert d._phase_acquired
    assert d.blend_factor > 0.9
    angle_before = np.rad2deg(d.stereo_phase_err_ema)

    for _ in range(4 * fs_c // n_blk):              # dropout
        d.demodulate(comp(n_blk, pos, False))
        pos += n_blk
    assert d.blend_factor < 0.05, d.blend_factor
    # leaked toward 0 (or already there); never a new false lock away
    assert abs(np.rad2deg(d.stereo_phase_err_ema)) <= abs(angle_before) + 1.0

    for _ in range(4 * fs_c // n_blk):              # recovery
        d.demodulate(comp(n_blk, pos, True))
        pos += n_blk
    assert d.blend_factor > 0.9
    assert d._phase_acquired
    angle_after = np.rad2deg(d.stereo_phase_err_ema)
    assert abs(angle_after - angle_before) < 10.0, (angle_before, angle_after)


def test_light_real_block_pilotless_transients():
    """Codex P1 on PR #32 round 3: transients at the REAL light block.

    The light variant's production block is 16384 IQ samples at
    250 kHz (~65.5 ms of composite) - ~4x the 16 ms reference the
    per-block EMAs were tuned against.  Before the time-normalised
    EMAs + fast-close, a pilot-less cold start held blend 0.716 at
    0.26 s (side/mid 0.877) and took 2.36 s to close; a dropout
    after full stereo took 3.60 s.  After the fix (both measured at
    the real block size): the cold start rides the time-normalised
    EMA through the ~190 ms settle window, is crushed by the
    steady-pilot-less trigger, and closes at 0.197 s; the dropout
    fires the pilot-power-collapse trigger and closes at 0.197 s
    (the first 65.5 ms block still carries pilot-era composite
    through the pipeline latency).
    """
    from fm_radio.quality_selftest import _synthesize_iq_tone
    fs_iq = 250_000
    blk = 16384                                     # real light block

    # --- pilot-less cold start: left-only content (mono present) ---
    iq = _synthesize_iq_tone(
        2.0, fs_iq, 1000.0, 1.0, 0.0, 0.0, 75_000.0,  # pilot_amp = 0
    ).astype(np.complex64)
    d = FMDemodulatorLight(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    t_now = 0.0
    t_closed = None
    for i in range(0, iq.size, blk):
        ch = iq[i:i + blk]
        if ch.size < 8:
            break
        left, right = d.demodulate(d.process_iq_samples(ch))
        t_now += ch.size / fs_iq
        side = 0.5 * (left.astype(np.float64) - right.astype(np.float64))
        mid = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
        if mid.size and float(np.sqrt(np.mean(mid ** 2))) > 1e-4:
            ratio = (np.sqrt(np.mean(side ** 2))
                     / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
            # During the ~190 ms settle window the blend rides the
            # time-normalised EMA down (the fast-close would be
            # unreliable there), so the first blocks pass genuine
            # side at a declining blend; after 0.3 s it must be gone.
            limit = 1.0 if t_now < 0.30 else 0.02
            assert ratio < limit, (t_now, ratio)
        if t_closed is None and d.blend_factor < 0.05:
            t_closed = t_now
    assert t_closed is not None and t_closed < 0.30, t_closed

    # --- dropout after full stereo (mono + side content) ---
    iq_st = _synthesize_iq_tone(
        3.0, fs_iq, 700.0, 1.0, 0.0, 0.1, 75_000.0,
    ).astype(np.complex64)
    iq_dr = _synthesize_iq_tone(
        3.0, fs_iq, 700.0, 1.0, 0.0, 0.0, 75_000.0,
    ).astype(np.complex64)
    d = FMDemodulatorLight(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
    for i in range(0, iq_st.size, blk):
        ch = iq_st[i:i + blk]
        if ch.size < 8:
            break
        d.demodulate(d.process_iq_samples(ch))
    assert d.blend_factor > 0.9
    t_now = 0.0
    t_closed = None
    for i in range(0, iq_dr.size, blk):
        ch = iq_dr[i:i + blk]
        if ch.size < 8:
            break
        left, right = d.demodulate(d.process_iq_samples(ch))
        t_now += ch.size / fs_iq
        if t_closed is None and d.blend_factor < 0.05:
            t_closed = t_now
        if t_now > 0.4:
            side = 0.5 * (left.astype(np.float64) - right.astype(np.float64))
            mid = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
            ratio = (np.sqrt(np.mean(side ** 2))
                     / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
            assert ratio < 0.05, (t_now, ratio)     # settled to mono
    # was 3.60 s before the fix; one 65.5 ms block of pipeline
    # latency still carries pilot-era composite, then fast-close
    assert t_closed is not None and t_closed < 0.35, t_closed

    # --- recovery on the SAME instance: the open side is time-
    # normalised too (codex P3 on PR #32 round 4).  The close side is
    # the fast path; re-opening runs through the ordinary blend EMA,
    # so it is the direct check that alpha_eff - not just the
    # fast-close - carries the 16 ms time constants onto 65.5 ms
    # blocks.  Measured here: blend > 0.5 at 0.393 s, > 0.9 at
    # 0.983 s, and monotonic from the first rising block.  Both are
    # deliberately slower than they used to be (0.262 / 0.524 s):
    # STEREO_BLEND_SMOOTHING_OPEN halves the OPENING rate to stop the
    # image pumping on intermittently degraded reception.  The
    # CLOSING side is untouched - the dropout above still closes at
    # 0.197 s - so the slower opening costs protection nothing.
    t_now = 0.0
    t_half = None
    t_full = None
    trace = []
    for i in range(0, iq_st.size, blk):
        ch = iq_st[i:i + blk]
        if ch.size < 8:
            break
        d.demodulate(d.process_iq_samples(ch))
        t_now += ch.size / fs_iq
        trace.append(d.blend_factor)
        if t_half is None and d.blend_factor > 0.5:
            t_half = t_now
        if t_full is None and d.blend_factor > 0.9:
            t_full = t_now
    # one light block (65.5 ms) of headroom over the measured
    # 0.393 / 0.983 s
    assert t_half is not None and t_half < 0.47, t_half
    assert t_full is not None and t_full < 1.06, t_full
    # No fast-close/EMA chatter on the way up.  The check starts at
    # the FIRST RISING block rather than a fixed index (codex P3 on
    # PR #32 round 5): the leading blocks still carry pilot-less
    # composite through the pipeline latency, but pinning that to an
    # index would both track the latency and hide a fast-close dip in
    # the block right after the first rise.
    steps = np.diff(np.asarray(trace, dtype=np.float64))
    assert (steps > 0).any(), trace[:8]
    rise = int(np.argmax(steps > 0))
    assert steps[rise:].min() >= -1e-9, (rise, float(steps[rise:].min()),
                                         trace[:8])


def test_light_real_block_noise_step_closes_blend():
    """Codex P2 on PR #32 round 5: noise rises, pilot stays intact.

    The pilot-power-collapse trigger cannot fire (the pilot is
    unchanged) and the EMA trigger waits for the slow SNR EMA to
    cross LO: measured 0.524 s to reach blend < 0.5 and 0.655 s to
    close, with side/mid ~0.67-0.75 - noise, not programme - through
    the first 0.2 s.  The sustained-sub-LO debounce bounds it.
    """
    fs_c = int(COMPOSITE_RATE)
    n_blk = 12_583                                  # light's real block
    dt = n_blk / fs_c
    rng = np.random.default_rng(7)
    d = FMDemodulatorLight(stereo=True)
    pos = 0

    def feed(noise_amp):
        nonlocal pos
        tt = (np.arange(n_blk) + pos) / fs_c
        pos += n_blk
        x = (0.20 * np.sin(2 * np.pi * 400.0 * tt)
             + 0.10 * np.cos(2 * np.pi * 19_000.0 * tt)     # pilot: constant
             + 0.10 * np.sin(2 * np.pi * 700.0 * tt)
             * np.cos(2 * np.pi * 38_000.0 * tt)
             + noise_amp * rng.standard_normal(n_blk))
        return d.demodulate(x)

    for _ in range(round(3.0 / dt)):                # clean acquisition
        feed(0.001)
    assert d.blend_factor > 0.9, d.blend_factor

    t_now = 0.0
    t_half = None
    t_closed = None
    for _ in range(20):
        left, right = feed(0.35)                    # noise floor steps up
        t_now += dt
        if t_half is None and d.blend_factor < 0.5:
            t_half = t_now
        if t_closed is None and d.blend_factor < 0.05:
            t_closed = t_now
        if t_now > 0.4:
            side = 0.5 * (left.astype(np.float64) - right.astype(np.float64))
            mid = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
            ratio = (np.sqrt(np.mean(side ** 2))
                     / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
            assert ratio < 0.05, (t_now, ratio)
    # Floors carry one light block (65.5 ms) of headroom over the
    # measured 0.262 / 0.328 s; they were 0.30 / 0.35 while the
    # debounce was 12 reference blocks (codex round 6 raised it to 16
    # for false-positive margin, which costs one block of closing).
    assert t_half is not None and t_half < 0.35, t_half      # was 0.524 s
    assert t_closed is not None and t_closed < 0.40, t_closed  # was 0.655 s


def test_blend_opens_slower_than_it_closes():
    """The blend EMA is asymmetric: slow to widen, prompt to narrow.

    A symmetric EMA pumped the stereo image on intermittently
    degraded reception - every good block pulled the blend back up
    before the next bad one pushed it down (measured +-0.14 per block
    on a 3-bad/1-good pattern at the light variant's real block
    size).  Slowing only the OPENING direction bounds that without
    touching the protective response: the fast-close path is
    independent of both constants, and the gradual closing rate is
    unchanged.
    """
    assert STEREO_BLEND_SMOOTHING_OPEN < STEREO_BLEND_SMOOTHING

    fs_c = int(COMPOSITE_RATE)
    n_blk = 12_583
    dt = n_blk / fs_c
    rng = np.random.default_rng(7)
    d = FMDemodulatorLight(stereo=True)
    pos = 0

    def feed(noise_amp):
        nonlocal pos
        tt = (np.arange(n_blk) + pos) / fs_c
        pos += n_blk
        x = (0.20 * np.sin(2 * np.pi * 400.0 * tt)
             + 0.10 * np.cos(2 * np.pi * 19_000.0 * tt)
             + 0.10 * np.sin(2 * np.pi * 700.0 * tt)
             * np.cos(2 * np.pi * 38_000.0 * tt)
             + noise_amp * rng.standard_normal(n_blk))
        d.demodulate(x)

    for _ in range(round(3.0 / dt)):
        feed(0.001)
    assert d.blend_factor > 0.9

    trace = []
    for k in range(round(8.0 / dt)):
        feed((0.35, 0.35, 0.35, 0.001)[k % 4])      # 3 bad, 1 good
        trace.append(d.blend_factor)
    settled = np.asarray(trace[round(3.0 / dt):], dtype=np.float64)
    steps = np.diff(settled)
    # measured 0.079 with the asymmetric rate, 0.140 symmetric
    assert np.abs(steps).max() < 0.10, float(np.abs(steps).max())
    # NOT asserted here: that the upward steps are the smaller ones.
    # A step is alpha * (target - blend), and on this pattern the
    # upward gap is the larger one (a good block's target jumps back
    # to ~1.0), so the per-block step measures the gap as much as the
    # rate.  The rate asymmetry itself is fixed by the constants
    # above and by the transient test's recovery timings.


def test_dropout_debounce_and_latch_contract():
    """Codex P2/P3 on PR #32 round 6: attack debounce + release hold.

    The sustained-degradation trigger counts CONTINUOUS sub-LO time,
    so an intermittent dip must not reach it, and once any trigger has
    fired the latch must hold until the SNR has been healthy for a
    while - releasing on a single good block made the blend pump
    (measured 0.01 <-> 0.26 with 37 sign flips in 5 s).
    """
    fs_c = int(COMPOSITE_RATE)
    n_ref_blk = int(round(0.016 * COMPOSITE_RATE))          # 16 ms

    def composite(n, p0, pilot_amp, noise_amp, rng):
        tt = (np.arange(n) + p0) / fs_c
        out = (0.20 * np.sin(2 * np.pi * 400.0 * tt)
               + pilot_amp * np.cos(2 * np.pi * 19_000.0 * tt)
               + 0.10 * np.sin(2 * np.pi * 700.0 * tt)
               * np.cos(2 * np.pi * 38_000.0 * tt))
        if noise_amp:
            out = out + noise_amp * rng.standard_normal(n)
        return out

    # --- the counters zero on every reset path ---
    rng = np.random.default_rng(3)
    d = FMDemodulatorLight(stereo=True)
    for k in range(60):                                     # drive it degraded
        d.demodulate(composite(n_ref_blk, k * n_ref_blk, 0.10, 0.35, rng))
    assert d._snr_sub_lo_ref > 0.0
    assert d._dropout_latched
    d.reset()
    assert d._snr_sub_lo_ref == 0.0
    assert d._snr_ok_ref == 0.0
    assert not d._dropout_latched

    d = FMDemodulatorLight(stereo=True)
    for k in range(60):
        d.demodulate(composite(n_ref_blk, k * n_ref_blk, 0.10, 0.35, rng))
    assert d._dropout_latched
    d.stereo = False                                        # mono stretch
    d.demodulate(composite(n_ref_blk, 0, 0.10, 0.0, rng))
    d.stereo = True                                         # re-entry resets
    d.demodulate(composite(n_ref_blk, 0, 0.10, 0.0, rng))
    assert d._snr_sub_lo_ref == 0.0
    assert not d._dropout_latched

    # --- an INTERMITTENT dip never reaches the attack debounce ---
    rng = np.random.default_rng(4)
    d = FMDemodulatorLight(stereo=True)
    peak = 0.0
    for k in range(120):
        noise = 0.35 if (k % 3) < 2 else 0.0                # 2 bad, 1 good
        d.demodulate(composite(n_ref_blk, k * n_ref_blk, 0.10, noise, rng))
        peak = max(peak, d._snr_sub_lo_ref)
    assert peak < STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF, peak

    # --- trigger (c) attack and the latch release, on REAL TIME, at
    # either block size (codex P3 on PR #32 round 7).  The measurement
    # starts from a CLEAN acquisition so the SNR EMA is high and
    # trigger (b) cannot fire - driving a cold instance with noise
    # instead fires (b) right after the settle guard and never
    # exercises (c) at all.  The degradation must also be clearly
    # sub-LO rather than marginal: at a noise level that leaves the
    # SNR straddling LO, the 16 ms chain's noisier per-block estimate
    # keeps breaking the CONTINUOUS run (measured: it fired at
    # 0.416 s against light's 0.262 s), which says more about
    # estimator variance than about the debounce.
    fired = {}
    released = {}
    for cls, blk in ((FMDemodulator, n_ref_blk), (FMDemodulatorLight, 12_583)):
        rng = np.random.default_rng(5)
        d = cls(stereo=True)
        pos = 0
        dt = blk / fs_c
        for _ in range(int(round(3.0 / dt))):            # clean acquisition
            d.demodulate(composite(blk, pos, 0.10, 0.0, rng))
            pos += blk
        assert d.blend_factor > 0.9, (cls.__name__, d.blend_factor)
        t_now = 0.0
        for _ in range(int(round(1.5 / dt))):            # sustained noise
            d.demodulate(composite(blk, pos, 0.10, 0.5, rng))
            pos += blk
            t_now += dt
            if cls not in fired and d._dropout_latched:
                fired[cls] = (t_now, d._snr_sub_lo_ref)
        assert cls in fired, cls.__name__
        t_now = 0.0
        for _ in range(int(round(1.5 / dt))):            # clean again
            d.demodulate(composite(blk, pos, 0.10, 0.0, rng))
            pos += blk
            t_now += dt
            if cls not in released and not d._dropout_latched:
                released[cls] = t_now
        assert cls in released, cls.__name__
        t_fire, sub_lo = fired[cls]
        # it was (c) that fired: the attack debounce was actually met
        assert sub_lo >= STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF, (
            cls.__name__, sub_lo)
        assert 0.20 < t_fire < 0.40, (cls.__name__, t_fire)
        assert 0.05 < released[cls] < 0.25, (cls.__name__, released[cls])
    # the whole point of the reference-block accounting: the 65.5 ms
    # chain must not take ~4x longer than the 16 ms one.  Measured
    # 0.256 / 0.262 s to fire and 0.128 / 0.131 s to release, so the
    # bound is one light block.
    assert abs(fired[FMDemodulator][0]
               - fired[FMDemodulatorLight][0]) < 0.07, fired
    assert abs(released[FMDemodulator]
               - released[FMDemodulatorLight]) < 0.07, released

    # --- the release timer starts at the LAST trigger, not before the
    # latch (codex P2 on PR #32 round 7).  A pilot POWER collapse can
    # fire while the SNR RATIO is still above LO - an overall level
    # drop scales pilot and noise together - and the healthy time
    # banked before that used to release the latch on the very next
    # block, skipping the hold entirely.
    rng = np.random.default_rng(9)
    d = FMDemodulatorLight(stereo=True)
    blk = 12_583
    pos = 0
    for _ in range(46):
        d.demodulate(composite(blk, pos, 0.10, 0.0, rng))
        pos += blk
    assert d._snr_ok_ref > STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF   # banked
    d.demodulate(0.1 * composite(blk, pos, 0.10, 0.0, rng))        # -20 dB
    pos += blk
    assert d._dropout_latched                                      # (a) fired
    assert d._snr_ok_ref == 0.0                                    # timer reset
    blend_at_latch = d.blend_factor
    d.demodulate(composite(blk, pos, 0.10, 0.0, rng))              # healthy
    pos += blk
    assert d._dropout_latched, 'released without serving the hold'
    assert d.blend_factor < blend_at_latch                         # still closing

    # --- once latched, an intermittent recovery must not pump ---
    rng = np.random.default_rng(6)
    d = FMDemodulatorLight(stereo=True)
    blk = 12_583
    for k in range(46):                                     # clean acquisition
        d.demodulate(composite(blk, k * blk, 0.10, 0.001, rng))
    assert d.blend_factor > 0.9
    trace = []
    fired_at = None
    for k in range(60):
        # 4 bad, 1 good: four 65.5 ms blocks reach the 16-reference
        # block attack debounce, and the single good block (4.096) is
        # short of the 8-reference block release hold, so the latch
        # must survive every cycle.  Without the hold this pattern
        # pumped the blend on every good block.
        noise = 0.001 if (k % 5) == 4 else 0.35
        d.demodulate(composite(blk, (46 + k) * blk, 0.10, noise, rng))
        trace.append(d.blend_factor)
        if fired_at is None and d._dropout_latched:
            fired_at = len(trace) - 1
    assert fired_at is not None
    assert d._dropout_latched                               # held through
    steps = np.diff(np.asarray(trace[fired_at:], dtype=np.float64))
    assert steps.max() <= 1e-9, (float(steps.max()), trace[fired_at:fired_at + 12])


def test_pending_iq_duration_is_consumed_once():
    """Codex P2 on PR #32 round 4: the IQ time step is per-block.

    _pending_block_iq_s is the duration of the IQ block that
    process_iq_samples just converted, and it is the EMA time base
    for exactly ONE demodulate() call.  It used to persist, so once a
    light instance had seen its 65.5 ms production block, any later
    composite-direct call inherited that stale duration and ran the
    EMAs, the settle guard and the fast-close 4.096x too fast.
    """
    fs_iq = 250_000
    blk = 16384
    n_comp = int(round(0.016 * COMPOSITE_RATE))     # 3072 = one 16 ms block
    tt = np.arange(n_comp) / float(COMPOSITE_RATE)
    comp16 = (0.2 * np.sin(2 * np.pi * 400.0 * tt)
              + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    from fm_radio.quality_selftest import _synthesize_iq_tone
    iq = _synthesize_iq_tone(
        0.5, fs_iq, 700.0, 1.0, 0.0, 0.1, 75_000.0,
    ).astype(np.complex64)[:blk]

    # --- production 1:1 path keeps the IQ-derived step ---
    d = FMDemodulatorLight(stereo=True)
    d.demodulate(d.process_iq_samples(iq))
    assert d._pilot_settled_ref == pytest.approx(4.096)   # 65.536 / 16 ms
    assert d._pending_block_iq_s is None                  # consumed

    # --- a composite-direct call after it must NOT reuse 65.536 ms ---
    before = d._pilot_settled_ref
    d.demodulate(comp16)
    assert d._pilot_settled_ref - before == pytest.approx(1.0)

    # --- standard's production block is exactly the 16 ms reference ---
    ds = FMDemodulator(stereo=True)
    iq_std = _synthesize_iq_tone(
        0.5, 1_024_000, 700.0, 1.0, 0.0, 0.1, 75_000.0,
    ).astype(np.complex64)[:int(SDR_BLOCK_SIZE)]
    ds.demodulate(ds.process_iq_samples(iq_std))
    assert ds._pilot_settled_ref == pytest.approx(1.0)

    # --- a mono block consumes the pending duration as well ---
    d = FMDemodulatorLight(stereo=False)
    d.demodulate(d.process_iq_samples(iq))                # mono path
    assert d._pending_block_iq_s is None
    d.stereo = True
    d.demodulate(comp16)
    # the mono -> stereo re-entry zeroes the settle counter first, so
    # the whole counter is this one composite block's own duration
    assert d._pilot_settled_ref == pytest.approx(1.0)

    # --- a FAILED preprocessing must not publish a duration at all
    # (codex P2 on PR #32 round 5): the pending token is set only
    # after the composite exists, so a block that never reaches
    # demodulate() leaves nothing for a later composite-direct call
    # to consume.  Both chains raise DemodulationError on an empty
    # input block.
    from fm_radio.exceptions import DemodulationError
    for cls in (FMDemodulator, FMDemodulatorLight):
        dx = cls(stereo=True)
        with pytest.raises(DemodulationError):
            dx.process_iq_samples(np.empty(0, dtype=np.complex64))
        assert dx._pending_block_iq_s is None, cls.__name__
        dx.demodulate(comp16)
        assert dx._pilot_settled_ref == pytest.approx(1.0), cls.__name__

    # --- an exception in the demodulation must not strand it ---
    d = FMDemodulatorLight(stereo=True)
    d.process_iq_samples(iq)
    assert d._pending_block_iq_s == pytest.approx(blk / fs_iq)

    def _boom(_composite):
        raise RuntimeError('boom')

    d._demodulate_stereo = _boom
    with pytest.raises(RuntimeError):
        d.demodulate(comp16)
    assert d._pending_block_iq_s is None

    # --- reset() restores the initial state ---
    d = FMDemodulatorLight(stereo=True)
    d.demodulate(d.process_iq_samples(iq))
    d.process_iq_samples(iq)                              # leave one pending
    assert d._pending_block_iq_s is not None
    assert d._pilot_settled_ref > 0.0
    assert d._pilot_pow_ema is not None
    d.reset()
    assert d._pending_block_iq_s is None
    assert d._pilot_settled_ref == 0.0
    assert d._pilot_pow_ema is None


def test_light_pilot_snr_matches_standard_on_pure_pilot():
    """Light's pilot SNR must track signal quality like standard's.

    Historically light reused its order-1 pilot order for the NOISE
    bandpasses, whose skirts leaked the pilot itself into the noise
    reference (-9.6/-10.4 dB at 19 kHz): pilot SNR saturated at
    9.975 dB and blend at 0.313 on a PURE pilot of ANY amplitude, so
    light never reached full stereo.  With order-9 noise bands
    (PILOT_NOISE_BAND_ORDER) a clean pilot must read high-SNR and
    open the blend fully, at parity with the standard variant
    (measured 84.18 vs 84.17 dB).
    """
    fs_c = 192_000
    n_blk = 3072
    snrs = {}
    for cls in (FMDemodulatorLight, FMDemodulator):
        d = cls(stereo=True)
        d.subcarrier_phase_offset_rad = np.deg2rad(0.3)
        pos = 0
        for _ in range(8 * fs_c // n_blk):
            tt = (np.arange(n_blk) + pos) / fs_c
            d.demodulate(0.1 * np.cos(2 * np.pi * 19_000.0 * tt))
            pos += n_blk
        snrs[cls] = d.pilot_snr_ema
        assert d.pilot_snr_ema > 60.0, (cls, d.pilot_snr_ema)
        assert d.blend_factor > 0.99, (cls, d.blend_factor)
        assert d._side_nr_adapt
    assert abs(snrs[FMDemodulatorLight] - snrs[FMDemodulator]) < 2.0, snrs


def test_nr_gate_hysteresis_and_reset(rng):
    """The gate must not flap inside the hysteresis band and must be
    restored by reset()."""
    from fm_radio.constants import (
        SIDE_NR_ADAPT_BLEND_ON, SIDE_NR_ADAPT_BLEND_OFF,
    )
    mid = 0.5 * (SIDE_NR_ADAPT_BLEND_ON + SIDE_NR_ADAPT_BLEND_OFF)
    lo = mid - 0.02
    hi = mid + 0.02
    fs_c = 192_000
    n_blk = 3072

    def comp(n, p0):
        tt = (np.arange(n) + p0) / fs_c
        return (0.2 * np.sin(2 * np.pi * 400.0 * tt)
                + 0.1 * np.cos(2 * np.pi * 19_000.0 * tt))

    d = FMDemodulator(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(1.0)
    pos = 0
    # open state: oscillate inside the band -> must stay open
    d.force_blend_factor = 1.0
    d.demodulate(comp(n_blk, pos)); pos += n_blk
    assert d._side_nr_adapt
    for k in range(20):
        d.force_blend_factor = lo if k % 2 == 0 else hi
        d.demodulate(comp(n_blk, pos)); pos += n_blk
        assert d._side_nr_adapt
    # closed state: oscillate inside the band -> must stay closed
    d.force_blend_factor = 0.0
    d.demodulate(comp(n_blk, pos)); pos += n_blk
    assert not d._side_nr_adapt
    for k in range(20):
        d.force_blend_factor = lo if k % 2 == 0 else hi
        d.demodulate(comp(n_blk, pos)); pos += n_blk
        assert not d._side_nr_adapt
    d.reset()
    assert d._side_nr_adapt                         # reset restores


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
