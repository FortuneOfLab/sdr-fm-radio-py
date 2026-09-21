"""End-to-end synthetic quality gates, clean and impaired.

Runs the full MPX -> FM IQ -> demodulator chain and asserts conservative
floors for the objective metrics.  The floors sit well below the
measured values (clean run at CNR=35: Sep ~43/57 dB, THD+N ~-37 dB,
SNR ~35 dB with pre-emphasis on) so they are robust across
platforms and RNG noise draws while still catching structural
regressions.  See the FLOORS comment below for the measurement
history across tuning changes.

The impaired scenarios exist because a pristine synthetic channel can
hide whole bug classes: the FFT-Hilbert block-edge defect fixed in
PR #6 was invisible with a pilot at exactly 19 000.0 Hz (integer
periodic in every block) and only appeared under a receiver/transmitter
clock mismatch.  Each scenario models one real-world impairment:

  clock-200ppm    pilot/subcarrier detuned by a worst-case cheap-dongle
                  crystal error (pilot at 19 003.8 Hz)
  tuning-30kHz    receiver tuning error: DC in the composite and
                  asymmetric sideband filtering in the IQ lowpass
  multipath       two-ray echo, 3 us / -12 dB / 60 deg

Marked slow: run explicitly with `pytest -m slow` or as part of CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from fm_radio.quality_selftest import evaluate_quality


BASE_KWARGS = dict(
    duration_s=3.0,
    tone_hz=1000.0,
    cnr_db=35.0,
    pilot_amp=0.10,
    freq_dev_hz=75_000.0,
    warmup_s=0.8,
)

SCENARIOS = {
    "clean": dict(),
    "clock-200ppm": dict(clock_ppm=200.0),
    "tuning-30kHz": dict(carrier_offset_hz=30_000.0),
    "multipath": dict(
        multipath_delay_us=3.0, multipath_gain=0.25, multipath_phase_deg=60.0,
    ),
}

# Measured values (2026-07, windowed-median metrics, neutral HF
# ceilings AND neutral blend-stability term, analog-exact pre-emphasis
# + analog-fitted de-emphasis): clean Sep ~43/57, THD -36.9, SNR 34.9.
# History: before the blend-stability neutralisation the same chain
# measured clean Sep 30.2/30.5 - the blend itself (0.95-0.997 on
# synthetic) capped separation at 20*log10((1+b)/(1-b)).  Earlier
# still: with the
# bilinear pre-emphasis + matched-Z de-emphasis mismatch these were
# Sep 29.3/30.7, THD -32.8, SNR 32.7; with the earlier 0.85/0.50 HF
# damping ceilings Sep 24.4/28.4, THD -31..-32.5, SNR 30.9-34.2.
# THD is duration-stable to ~0.5 dB (was swinging -18..-32 with the
# whole-signal single-FFT metric).
FLOORS = {
    "clean": dict(sep=18.0, thd=-20.0, snr=24.0),
    "clock-200ppm": dict(sep=18.0, thd=-20.0, snr=24.0),
    "tuning-30kHz": dict(sep=18.0, thd=-20.0, snr=24.0),
    "multipath": dict(sep=16.0, thd=-20.0, snr=24.0),
}


@pytest.mark.slow
def test_blend_snr_ramp_protects_weak_signal():
    """The pilot-SNR ramp must still open/close the blend correctly.

    With the jitter stability term neutral (PR #27) the SNR ramp is
    the ONLY weak-signal protection, so its behaviour is pinned:
    blend ~1 on a good signal, small under sustained weak signal, and
    closing within a bounded time after a good->weak SNR step.
    Statistics: per-block blend over 16384-sample IQ blocks, first
    0.5 s of each segment excluded as settling.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import _synthesize_iq_tone, _apply_channel
    from fm_radio.constants import SDR_SAMPLE_RATE, SDR_BLOCK_SIZE
    fs = int(SDR_SAMPLE_RATE)
    np.random.seed(0)
    clean = _synthesize_iq_tone(8.0, fs, 1000.0, 0.6, 0.6, 0.10, 75_000.0)
    half = int(4.0 * fs)
    good = _apply_channel(clean[:half], fs, 35.0)
    weak = _apply_channel(clean[half:], fs, 0.0)
    iq = np.concatenate([good, weak])
    d = FMDemodulator(stereo=True)
    blends = []
    for i in range(0, iq.size, SDR_BLOCK_SIZE):
        c = iq[i:i + SDR_BLOCK_SIZE]
        if c.size < 8:
            break
        d.demodulate(d.process_iq_samples(c))
        blends.append(d.blend_factor)
    blends = np.array(blends)
    blk_s = SDR_BLOCK_SIZE / fs
    good_tail = blends[int(3.0 / blk_s):int(4.0 / blk_s)]
    weak_tail = blends[int(7.0 / blk_s):]
    assert np.median(good_tail) > 0.95, np.median(good_tail)   # opens
    assert np.median(weak_tail) < 0.2, np.median(weak_tail)    # closes
    # closing speed: within 1.5 s of the step, blend below 0.5
    after = blends[int(5.5 / blk_s):int(6.0 / blk_s)]
    assert np.max(after) < 0.5, np.max(after)


@pytest.mark.slow
def test_front_end_separation_floor_under_realistic_conditions():
    """The 8-12 kHz region must stay clean with realistic impairments.

    Root cause of the historical "8-12 kHz dip": near-zero carrier
    offsets put the synthetic tone's discrete carrier line inside the
    DC remover's notch; the removed component intermodulates across
    the composite.  With a realistic carrier offset (this hardware
    measures ~60 Hz residual; 1237 Hz chosen off the synthetic tone
    comb) and a large injected LO-leak DC, the DC blocker must remove
    the DC with no separation cost: measured 47.3 dB at 10 kHz and
    48.8 dB at 12 kHz (NR and corrector off, fixed blend, DSP
    offset).  Per-frequency floors ~1 dB under the measured values;
    mutation-checked: bypassing _remove_dc entirely measures
    44.96/46.98 dB, which the floors reject.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import (
        _synthesize_iq_tone, _apply_channel, _stereo_separation_ls_db,
    )
    from fm_radio.constants import SDR_BLOCK_SIZE
    fs_a = 48_000
    floors = {10_000.0: 46.0, 12_000.0: 47.5}
    for tone in (10_000.0, 12_000.0):
        iq = _synthesize_iq_tone(
            5.0, 1_024_000, tone, 1.0, 0.0, 0.1, 75_000.0,
            constant_modulation=True,
        ).astype(np.complex64)
        iq = _apply_channel(iq, 1_024_000, None, carrier_offset_hz=1237.0)
        iq = (iq + np.complex64(0.05 + 0.03j)).astype(np.complex64)
        d = FMDemodulator(stereo=True)
        d.force_blend_factor = 1.0
        d.side_nr_enabled = False
        d.iq_phase_correction_enabled = False
        from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
        d.subcarrier_phase_offset_rad = np.deg2rad(
            STEREO_SUBCARRIER_PHASE_OFFSET_DEG)
        ls, rs = [], []
        for i in range(0, iq.size, SDR_BLOCK_SIZE):
            ch = iq[i:i + SDR_BLOCK_SIZE]
            if ch.size < 8:
                break
            l, r = d.demodulate(d.process_iq_samples(ch))
            ls.append(l.astype(np.float32))
            rs.append(r.astype(np.float32))
        left = np.concatenate(ls)[int(1.5 * fs_a):]
        right = np.concatenate(rs)[int(1.5 * fs_a):]
        sep = _stereo_separation_ls_db(left, right, max_lag=96)
        assert sep > floors[tone], (tone, sep)


@pytest.mark.slow
def test_hf_separation_maintained_at_14k():
    """The FIR bank's headline win must not regress.

    Canonical sweep conditions (hifi TX, constant modulation,
    noiseless): the IIR chain measured -3.7 dB at 14 kHz, the FIR
    bank 34.0/35.2 dB.  Floor at 30 dB keeps the structural
    improvement while tolerating measurement scatter; it also guards
    the final audio lowpass (a band limit encroaching below 15 kHz
    would show up here first).
    """
    np.random.seed(0)
    m = evaluate_quality(
        duration_s=4.0, tone_hz=14_000.0, cnr_db=None,
        pilot_amp=0.10, freq_dev_hz=75_000.0, warmup_s=0.8,
        hifi_tx=True, hifi_constant_mod=True,
    )
    assert m.separation_l_to_r_db > 30.0, m
    assert m.separation_r_to_l_db > 30.0, m


@pytest.mark.slow
def test_phase_corrector_recovers_large_static_error():
    """A -75 deg static subcarrier error must be FULLY corrected.

    Real multipath channels need corrections well beyond the original
    45 deg clamp (the reference station's raw estimates sit at ~-83
    deg).  Measured separation for this scenario (seeded) across the
    corrector's history:

        clamp 45:          18.85 / 20.21 dB  (severe loss)
        clamp 60:          23.06 / 25.75 dB  (partial: -15 deg residual)
        clamp 75:          24.36 / 27.98 dB  (recovery, slow approach)
        gated 4-quadrant:  30.28 / 30.51 dB  (direct acquisition at
                           -75, no clamp truncation of the estimate
                           distribution)

    The floors keep the pre-tracker discrimination (fail at clamp 60)
    and the tracker clears them by ~4-7 dB.
    """
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    np.random.seed(0)
    m = evaluate_quality(
        **BASE_KWARGS,
        subcarrier_phase_offset_deg=STEREO_SUBCARRIER_PHASE_OFFSET_DEG - 75.0,
    )
    assert m.separation_l_to_r_db > 23.5, m
    assert m.separation_r_to_l_db > 26.5, m


@pytest.mark.slow
def test_phase_tracker_never_acquires_on_mono_broadcast():
    """A mono broadcast (L=R) must never acquire a phase estimate.

    Codex repro from the PR #23 review: anisotropy alone is
    scale-invariant, so on a NOISELESS mono signal the tiny
    deterministic side-band residue (~-32 dB below mono) looked
    strongly 1-D and acquired a random angle on block 0.  The
    absolute-energy gate (side power within -18 dB of mono) blocks
    that; at CNR 20 the noise-dominated side is blocked by the
    anisotropy gate instead.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
    )
    fs = AUDIO_OUTPUT_RATE
    n = int(3.0 * fs)
    t = np.arange(n) / fs
    tone = (0.25 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    mpx = _build_mpx(tone, tone, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, 0.0)
    for cnr in (None, 20.0):
        np.random.seed(0)
        iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                             75_000.0, cnr)
        d = FMDemodulator(stereo=True)
        for i in range(0, iq.size, SDR_BLOCK_SIZE):
            c = iq[i:i + SDR_BLOCK_SIZE]
            if c.size < 8:
                break
            d.demodulate(d.process_iq_samples(c))
        assert not d._phase_acquired, cnr
        assert d.stereo_phase_err_ema == 0.0, cnr


def _side_and_mono(duration_s, width_db, fs):
    """A programme whose side/mono power ratio is *width_db*.

    L = m + d and R = m - d: a common tone and a difference tone,
    the second scaled to set how wide the programme is.  Two tones
    rather than one, so that what comes back in the side band can be
    told from what leaked out of the mono band.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    mono = np.sin(2 * np.pi * 997.0 * t)
    diff = (10.0 ** (width_db / 20.0)) * np.sin(2 * np.pi * 1499.0 * t)
    left, right = 0.5 * (mono + diff), 0.5 * (mono - diff)
    peak = max(np.abs(left).max(), np.abs(right).max())
    return ((0.7 * left / peak).astype(np.float32),
            (0.7 * right / peak).astype(np.float32))


def _run_gates(iq, block_size, side_over_noise_db=None, side_gate_db=None):
    """Demodulate, optionally with one of the gates opened right up."""
    from fm_radio import demodulator as module
    from fm_radio.demodulator import FMDemodulator

    was = (module.STEREO_PHASE_SIDE_OVER_NOISE_DB,
           module.STEREO_PHASE_SIDE_GATE_DB)
    if side_over_noise_db is not None:
        module.STEREO_PHASE_SIDE_OVER_NOISE_DB = float(side_over_noise_db)
    if side_gate_db is not None:
        module.STEREO_PHASE_SIDE_GATE_DB = float(side_gate_db)
    try:
        d = FMDemodulator(stereo=True)
        d.force_blend_factor = 1.0      # the tracker is what is on trial
        readings, over_mono, lefts, rights = [], [], [], []
        for i in range(0, iq.size, block_size):
            c = iq[i:i + block_size]
            if c.size < 8:
                break
            left, right = d.demodulate(d.process_iq_samples(c))
            readings.append(d.stereo_phase_side_over_noise_db)
            over_mono.append(d.stereo_phase_side_over_mono_db)
            if left.size:
                lefts.append(left)
                rights.append(right)
        return {
            "readings": np.asarray(readings),
            "side_over_mono_db": np.asarray(over_mono),
            "acquired": bool(d._phase_acquired),
            "angle_deg": float(np.rad2deg(d.stereo_phase_err_ema)),
            "left": np.concatenate(lefts) if lefts else np.zeros(0),
            "right": np.concatenate(rights) if rights else np.zeros(0),
        }
    finally:
        (module.STEREO_PHASE_SIDE_OVER_NOISE_DB,
         module.STEREO_PHASE_SIDE_GATE_DB) = was


def _tone_db(x, hz, fs, skip_s=0.8):
    """Coherent level of *hz* in x, in dB, past the settling head."""
    x = x[int(skip_s * fs):]
    n = x.size - (x.size % fs)
    assert n >= fs, "not enough audio to measure"
    t = np.arange(n) / fs
    amp = 2.0 * np.abs(np.mean(x[:n].astype(np.float64)
                               * np.exp(-2j * np.pi * hz * t)))
    return 20.0 * np.log10(amp + 1e-30)


@pytest.mark.slow
def test_noise_alone_stays_under_the_side_over_noise_gate():
    """The gate is a margin over measured noise; keep it one.

    The discriminator's noise rises as f^2, so the demodulated side
    band carries more noise than the mono band, and that noise is
    anisotropic - a stable pseudo-axis the tracker will lock to
    during silence if it is let.  STEREO_PHASE_SIDE_OVER_NOISE_DB is
    set from measured noise, so a change to the filters that moves
    the noise floor has to move it too.

    This is the lower half of the calibration only: that the gate
    clears the noise.  The other half - that it does not also
    refuse genuine content - is not a property of silence and is
    pinned where content exists, in
    test_which_gate_closes_on_a_near_mono_programme.

    Re-measured 2026-09-21 through the FIR path: silence at CNR
    45/35/25/15 reads med 22.6, max 24.8, so 1.2 dB under the gate.
    """
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
        STEREO_PHASE_SIDE_OVER_NOISE_DB,
    )
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq

    fs = AUDIO_OUTPUT_RATE
    quiet = np.zeros(int(2.0 * fs), dtype=np.float32)
    mpx = _build_mpx(quiet, quiet, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, 0.0)
    worst = -np.inf
    for cnr in (45.0, 35.0, 25.0, 15.0):
        np.random.seed(0)
        iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                             75_000.0, cnr)
        out = _run_gates(iq, SDR_BLOCK_SIZE)
        assert not out["acquired"], cnr
        worst = max(worst, float(out["readings"].max()))

    margin = STEREO_PHASE_SIDE_OVER_NOISE_DB - worst
    assert margin > 0.5, (
        "silence reaches the gate (worst %.1f dB, gate %.1f)"
        % (worst, STEREO_PHASE_SIDE_OVER_NOISE_DB))


#: (nominal width, rotation to correct, CNR, what happens).  The
#: nominal width is the programme's side/mono before pre-emphasis;
#: what the gates compare is measured in the test, because the two
#: are not the same number - pre-emphasis weights the two tones
#: differently and the side band's own noise puts a floor under it
#: (nominal -10 / -20 / -25 / -30 measure -9.4 / -18.5 / -23.0 /
#: -23.7 dB).  -20 and -25 are the pair that straddle
#: STEREO_PHASE_SIDE_GATE_DB, which is the point of this.
#:
#: Measured 2026-09-21, share of blocks each gate lets through:
#:
#:   nominal  CNR  rot   side/mono > -18   side/noise >= 26   acquires
#:      -10    35   30           100.0%             100.0%    yes
#:      -20    35   30            29.9%              99.4%    yes
#:      -20    60   35            30.6%              99.4%    yes
#:      -25    35   30             0.0%              94.9%    no
#:      -30    20   30             0.0%               7.6%    no
#: The last two numbers are how much material to run and how much
#: of it to leave out of the tone measurement.  A wide programme
#: hands the tracker every block and it converges in 0.1 s; at a
#: side/mono of -18.5 only 30% of blocks are informative and the
#: same 60 deg takes 1.8 s, so that case gets longer to do it in
#: and is measured after it has.
WIDTHS = [
    (-10.0, 30.0, 35.0, "acquires", 2.5, 0.8),
    (-20.0, 60.0, 35.0, "acquires", 5.0, 2.5),
    # Narrow programme on a clean signal: the PR #49 case, where the
    # side/mono gate closes on its own while the noise gate is wide
    # open on 95% of blocks.
    (-25.0, 30.0, 35.0, "side gate", 2.5, 0.8),
    # Narrow AND weak: both close, which is also correct.
    (-30.0, 30.0, 20.0, "both gates", 2.5, 0.8),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "width_db,rotation_deg,cnr_db,outcome,duration_s,skip_s", WIDTHS)
def test_which_gate_closes_on_a_near_mono_programme(
        width_db, rotation_deg, cnr_db, outcome, duration_s, skip_s):
    """On a clean signal it is the side/mono gate, not the noise one.

    A near-mono programme on a clean signal was read as evidence
    that the 26 dB side-over-noise gate is set too high (PR #49: the
    cleanest of three stations, pilot SNR 43 dB, never acquired).
    It is not that gate.  What a narrow programme fails is
    STEREO_PHASE_SIDE_GATE_DB - at a side/mono of -23 dB the noise
    gate still passes 95% of blocks and the side gate passes none -
    and there is next to nothing in the side band to take an angle
    from anyway.  Both close together only when the signal is weak
    as well, which is a different case and also correct.

    The acquiring cases hold the half of the noise-gate calibration
    that silence cannot show: genuine but narrow content reads a
    median of 28.7 dB, so a gate much above 26 would start refusing
    programme rather than noise.

    What a closed gate costs is bounded by how far the tracked
    angle's prior is from the truth: within +-7 deg on all four
    reference captures, which is 0.06 dB of side level.
    """
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
        STEREO_PHASE_SIDE_GATE_DB, STEREO_PHASE_SIDE_OVER_NOISE_DB,
    )
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq

    fs = AUDIO_OUTPUT_RATE
    left, right = _side_and_mono(duration_s, width_db, fs)
    mpx = _build_mpx(left, right, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, rotation_deg)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, cnr_db)

    shipped = _run_gates(iq, SDR_BLOCK_SIZE)
    assert shipped["acquired"] is (outcome == "acquires"), \
        "%s, angle %.1f deg" % (outcome, shipped["angle_deg"])

    # What the gates actually compare, rather than the nominal width.
    side_open = float(np.mean(
        shipped["side_over_mono_db"] > STEREO_PHASE_SIDE_GATE_DB))
    noise_open = float(np.mean(
        shipped["readings"] >= STEREO_PHASE_SIDE_OVER_NOISE_DB))

    # Whatever happens, it is not the noise gate that decides it:
    # the same run with that one out of the way comes out the same.
    opened = _run_gates(iq, SDR_BLOCK_SIZE, side_over_noise_db=6.0)
    assert opened["acquired"] is shipped["acquired"], (
        "the side-over-noise gate decided it after all")

    if outcome == "acquires":
        assert side_open > 0.0, "nothing passed the side gate"
        # Genuine content must clear the noise gate with room, or
        # that gate is refusing programme and not just noise.
        assert (float(np.median(shipped["readings"]))
                > STEREO_PHASE_SIDE_OVER_NOISE_DB), (
            "genuine content only reads %.1f dB against a %.1f dB gate"
            % (np.median(shipped["readings"]),
               STEREO_PHASE_SIDE_OVER_NOISE_DB))
        # And what it acquired is worth having: the side tone comes
        # back at the level it has with every gate out of the way.
        both = _run_gates(iq, SDR_BLOCK_SIZE, side_over_noise_db=6.0,
                          side_gate_db=-40.0)
        got = _tone_db(0.5 * (shipped["left"] - shipped["right"]), 1499.0,
                       fs, skip_s)
        best = _tone_db(0.5 * (both["left"] - both["right"]), 1499.0,
                        fs, skip_s)
        assert got > best - 0.3, "%.2f dB of the side band went" % (best - got)
        return

    assert shipped["angle_deg"] == 0.0, "it took an angle from nothing"
    assert side_open == 0.0, "the side gate passed %.1f%% of blocks" % (
        100.0 * side_open)
    if outcome == "side gate":
        # The noise gate is open on nearly every block here, so the
        # side/mono gate is the whole of the reason - and opening it
        # is what lets this programme through.
        assert noise_open > 0.8, (
            "the noise gate was shut too (%.1f%% open); this case is "
            "supposed to isolate the other one" % (100.0 * noise_open))
        instead = _run_gates(iq, SDR_BLOCK_SIZE, side_gate_db=-40.0)
        assert instead["acquired"], "the side/mono gate was not the one"
    else:
        # Narrow and weak: both shut, and neither alone explains it.
        assert noise_open < 0.5, (
            "the noise gate was open on %.1f%% of blocks" % (
                100.0 * noise_open))
        alone = _run_gates(iq, SDR_BLOCK_SIZE, side_gate_db=-40.0)
        assert not alone["acquired"], (
            "one gate alone was holding a narrow, weak signal back")


@pytest.mark.slow
def test_phase_tracker_acquires_correct_branch_at_boundary():
    """Acquisition at a true rotation of -88 deg must not swap L/R.

    Raw estimates on a station near the +-90 boundary straddle it and
    ~half wrap to +88-ish; a single-block acquisition would lock the
    wrong 180-deg branch (permanent L/R swap) with that probability.
    The doubled-angle circular mean over the acquisition streak is
    invariant to the wrap, so separation stays high and positive.
    """
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    np.random.seed(0)
    m = evaluate_quality(
        **BASE_KWARGS,
        subcarrier_phase_offset_deg=STEREO_SUBCARRIER_PHASE_OFFSET_DEG - 88.0,
    )
    assert m.separation_l_to_r_db > 24.0, m
    assert m.separation_r_to_l_db > 24.0, m


@pytest.mark.slow
def test_phase_tracker_branch_guard_blocks_then_admits():
    """The +-90 guard must PARK a low-confidence crossing, then admit.

    Field failure (2026-07-20 antenna capture): marginal blocks walked
    the tracker across the branch boundary, flipping L/R mid-session.
    This test drives the guard itself: the tracker is pre-seeded just
    inside the boundary (+89.9 deg) with ZERO recent confidence, and
    fed confident stereo whose axis lies beyond it (~+110 deg).  While
    the confidence EMA builds (first ~dozen blocks) the guard must park
    the angle at the boundary; once confidence exceeds
    STEREO_PHASE_BRANCH_CONF it must ADMIT the crossing and converge
    past +95 deg toward the true axis near +99 (the phase-true FIR
    chain maps a -100 deg TX dsb phase to ~+99; seeds 0-4 measure
    final ~99.0 / max ~101.1) - exercising both the block and the
    admit paths.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
    )
    fs = AUDIO_OUTPUT_RATE
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    tone = 0.25 * np.sin(2 * np.pi * 1000.0 * t)
    left = tone.astype(np.float32)
    right = (-tone).astype(np.float32)
    # dsb phase -45 deg maps to a ~+45 deg corrector demand in the
    # phase-true FIR chain (see the leak test); -100 deg lands the
    # axis just past +95.
    mpx = _build_mpx(left, right, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, -100.0)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, 35.0)
    d = FMDemodulator(stereo=True)
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    d.subcarrier_phase_offset_rad = np.deg2rad(
        STEREO_SUBCARRIER_PHASE_OFFSET_DEG)  # DSP value
    d._phase_acquired = True
    d.stereo_phase_err_ema = float(np.deg2rad(89.9))
    d._phase_conf = 0.0
    emas = []
    for i in range(0, iq.size, SDR_BLOCK_SIZE):
        c = iq[i:i + SDR_BLOCK_SIZE]
        if c.size < 8:
            break
        d.demodulate(d.process_iq_samples(c))
        emas.append(np.rad2deg(d.stereo_phase_err_ema))
    emas = np.array(emas)
    parked = np.sum((emas > 89.0) & (emas <= 90.05))
    assert parked >= 5, f"guard never parked at the boundary ({parked})"
    assert emas[-1] > 95.0, emas[-1]  # admitted after confidence built


@pytest.mark.slow
def test_phase_tracker_leaks_home_when_uninformed():
    """After acquisition, mono content must decay the angle toward 0.

    The hardware trim makes 0 the prior; holding a possibly wandered
    angle through a long uninformative stretch is worse than gliding
    home (a genuine offset re-converges within ~1 s of confident
    content returning).  Acquire at ~-45 deg via a static DSB phase,
    then feed mono content and assert the angle decays at roughly
    STEREO_PHASE_LEAK_DEG_PER_SEC.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
    )
    fs = AUDIO_OUTPUT_RATE
    n_st = int(2.0 * fs)
    n_mo = int(20.0 * fs)
    t = np.arange(n_st + n_mo) / fs
    tone = 0.25 * np.sin(2 * np.pi * 1000.0 * t)
    # Stereo lead-in, then SILENCE.  (A mono TONE would not do: the FM
    # chain's intermodulation products land in the side band coherently
    # and intermittently pass the gates - the same marginal-update
    # mechanism as the field wander - whereas the leak is defined by
    # gate-CLOSED blocks, which silence guarantees.)
    left = np.concatenate([tone[:n_st], np.zeros(n_mo)])
    right = np.concatenate([-tone[:n_st], np.zeros(n_mo)])
    mpx = _build_mpx(left.astype(np.float32), right.astype(np.float32),
                     fs, int(COMPOSITE_RATE), 0.10, True, 50e-6, -45.0)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, 35.0)
    d = FMDemodulator(stereo=True)
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    d.subcarrier_phase_offset_rad = np.deg2rad(
        STEREO_SUBCARRIER_PHASE_OFFSET_DEG)  # DSP value (no tuner)
    emas = []
    for i in range(0, iq.size, SDR_BLOCK_SIZE):
        c = iq[i:i + SDR_BLOCK_SIZE]
        if c.size < 8:
            break
        d.demodulate(d.process_iq_samples(c))
        emas.append(np.rad2deg(d.stereo_phase_err_ema))
    emas = np.array(emas)
    # angle right after the stereo lead-in vs at the end (the -45 deg
    # DSB phase maps to a ~+45 deg corrector demand in this chain's
    # sign convention; the test is sign-agnostic and only asserts
    # magnitude decay toward 0)
    at_switch = emas[int(2.2 / 0.016)]
    final = emas[-1]
    assert abs(at_switch) > 30.0, at_switch      # acquired away from 0
    assert abs(final) < abs(at_switch) - 5.0, (at_switch, final)
    assert np.sign(final) == np.sign(at_switch) or abs(final) < 2.0


@pytest.mark.slow
def test_phase_tracker_follows_drift_beyond_90_deg():
    """The tracker must follow a DSB phase drift through +-90 deg.

    The channel phase ramps -40 deg/s for 3 s (0 -> -120 deg).  Any
    clamped estimator saturates (a 75 deg clamp leaves a 45 deg
    residual at the end - roughly 8 dB of separation in the late
    windows), and past 90 deg the raw principal-axis estimate wraps
    to the opposite branch.  The continuity-based tracker follows the
    pi-periodic family and holds full separation throughout
    (measured 30.8 / 29.5 dB, floors well below).
    """
    np.random.seed(0)
    m = evaluate_quality(**BASE_KWARGS, dsb_phase_drift_deg_per_s=-40.0)
    assert m.separation_l_to_r_db > 24.0, m
    assert m.separation_r_to_l_db > 24.0, m
    assert m.thdn_left_db < -20.0, m


@pytest.mark.slow
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_synthetic_quality_floors(scenario):
    np.random.seed(0)  # _fm_modulate_iq uses the legacy global RNG
    m = evaluate_quality(**BASE_KWARGS, **SCENARIOS[scenario])
    floors = FLOORS[scenario]
    assert m.separation_l_to_r_db > floors["sep"], (scenario, m)
    assert m.separation_r_to_l_db > floors["sep"], (scenario, m)
    assert m.thdn_left_db < floors["thd"], (scenario, m)
    assert m.thdn_right_db < floors["thd"], (scenario, m)
    assert m.snr_left_db > floors["snr"], (scenario, m)
    assert m.snr_right_db > floors["snr"], (scenario, m)
    assert m.blend_mean > 0.8, (scenario, m)
