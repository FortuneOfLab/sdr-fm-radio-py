"""End-to-end synthetic quality gates, clean and impaired.

Runs the full MPX -> FM IQ -> demodulator chain and asserts conservative
floors for the objective metrics.  The floors sit below the measured
values (clean run at CNR=35: Sep 70.3/72.3 dB, THD+N -57.2 dB, SNR
30.4 dB with pre-emphasis on) so they are robust across platforms and
RNG noise draws while still catching structural regressions.  How far
below depends on the metric, taking the worse channel of each pair as
the floors do: separation by 12-20 dB, THD by 8-12, and SNR by 5-7 -
the thinnest of all being the 4.9 dB on dc-notch's right channel.
The SNR floor is the one this PR did not move; what limits the SNR
measurement is not the CNR (taking the noise away leaves clean at
30.373/30.377 dB against 30.369/30.363 with it) and has not been
looked into.  See the FLOORS comment below for the per-scenario
measurements and the history across tuning changes.

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
  dc-notch        the carrier tuned exactly to 0 Hz, where the DC
                  blocker's notch takes the synthetic signal's
                  carrier line and the removal intermodulates across
                  the composite

Marked slow - run explicitly with `pytest -m slow` or as part of CI -
with one exception: test_the_scenarios_are_not_measured_in_the_dc_notch
reads the module's own settings and runs no DSP, so it is left
unmarked and runs with the quick tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fm_radio.quality_selftest import evaluate_quality


#: Away from the DC blocker's notch, and away from the tone comb.
#: A synthetic signal is a handful of discrete lines, and at zero
#: carrier offset the carrier line sits in the notch (see
#: DC_BLOCK_CUTOFF_HZ, and the warning quality_selftest prints for
#: offsets inside it).  What that costs is not small: the clean
#: scenario measures 50.2/57.3 dB of separation and -40.3 dB THD
#: there against 70.3/72.3 and -57.2 here, so every number this file
#: used to record was of the notch rather than of the receiver.
#: 1237 Hz is the tooling's own recommendation, off the tone comb;
#: this hardware's residual offset is ~60 Hz and the two measure
#: alike.  The notch is still worth a scenario of its own - see
#: SCENARIOS - but not worth being the default.
A_REAL_OFFSET_HZ = 1237.0

BASE_KWARGS = dict(
    duration_s=3.0,
    tone_hz=1000.0,
    cnr_db=35.0,
    pilot_amp=0.10,
    freq_dev_hz=75_000.0,
    warmup_s=0.8,
    carrier_offset_hz=A_REAL_OFFSET_HZ,
)

SCENARIOS = {
    "clean": dict(),
    "clock-200ppm": dict(clock_ppm=200.0),
    "tuning-30kHz": dict(carrier_offset_hz=30_000.0),
    "multipath": dict(
        multipath_delay_us=3.0, multipath_gain=0.25, multipath_phase_deg=60.0,
    ),
    # Overrides the realistic offset on purpose: this one is the
    # pathology, kept so that a change which makes it worse is seen.
    # Its floors are its own, ~10 dB under what it measures.
    "dc-notch": dict(carrier_offset_hz=0.0),
}

# Measured 2026-09-21, off the DC notch (see A_REAL_OFFSET_HZ - the
# same measurements at 0 Hz are in the dc-notch row, and every number
# recorded here before that date was taken there):
#
#   scenario       sepL>R  sepR>L    thdL    thdR    snrL    snrR
#   clean            70.3    72.3   -57.2   -57.2    30.4    30.4
#   clock-200ppm     70.5    73.3   -57.2   -57.2    30.4    30.4
#   tuning-30kHz     56.2    63.5   -38.7   -39.4    35.3    31.2
#   multipath        37.3    37.2   -55.9   -55.9    31.1    31.1
#   dc-notch         50.2    57.3   -40.3   -40.9    31.1    28.9
#
# Both channels, because each floor is asserted against both.  The
# margin left on the worse one, per scenario and metric:
#
#   scenario         sep    thd    snr
#   clean           20.3   12.2    6.4
#   clock-200ppm    20.5   12.2    6.4
#   tuning-30kHz    16.2    8.7    7.2
#   multipath       12.2   10.9    7.1
#   dc-notch        15.2    8.3    4.9
#
# The separation floors sit 12-20 dB under those, which is where
# they were before relative to what was then being measured, and
# they now discriminate far harder: with the phase corrector
# disabled the clean scenario measures 5.0 dB of separation rather
# than 70.  The SNR floor of 24 dB is the one this PR did not move
# and the one with least room: 4.9 dB on dc-notch's right channel.
# Whatever sets that measurement, it is not the channel noise -
# clean measures 30.369/30.363 dB at CNR 35 and 30.373/30.377 with
# no noise at all - and nobody has yet looked into what does.
# History of the clean row, all at 0 Hz and so all of the notch:
# 2026-07 (windowed-median metrics, neutral HF ceilings and blend
# stability, analog-exact pre-emphasis) Sep ~43/57, THD -36.9, SNR
# 34.9; before the blend-stability neutralisation Sep 30.2/30.5 (the
# blend itself, 0.95-0.997 on synthetic, capped separation at
# 20*log10((1+b)/(1-b))); with the bilinear/matched-Z emphasis
# mismatch Sep 29.3/30.7, THD -32.8, SNR 32.7; with the earlier
# 0.85/0.50 HF damping ceilings Sep 24.4/28.4, THD -31..-32.5, SNR
# 30.9-34.2.  THD is duration-stable to ~0.5 dB (it swung -18..-32
# with the whole-signal single-FFT metric).
FLOORS = {
    "clean": dict(sep=50.0, thd=-45.0, snr=24.0),
    "clock-200ppm": dict(sep=50.0, thd=-45.0, snr=24.0),
    "tuning-30kHz": dict(sep=40.0, thd=-30.0, snr=24.0),
    "multipath": dict(sep=25.0, thd=-45.0, snr=24.0),
    "dc-notch": dict(sep=35.0, thd=-32.0, snr=24.0),
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

    Off the DC notch like the rest of the file, though this one
    measured the same either way (1.000 / 0.000 / 0.000 at both): the
    blend reads the pilot, and the notch takes the carrier line.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.quality_selftest import _synthesize_iq_tone, _apply_channel
    from fm_radio.constants import SDR_SAMPLE_RATE, SDR_BLOCK_SIZE
    fs = int(SDR_SAMPLE_RATE)
    np.random.seed(0)
    clean = _synthesize_iq_tone(8.0, fs, 1000.0, 0.6, 0.6, 0.10, 75_000.0)
    half = int(4.0 * fs)
    good = _apply_channel(clean[:half], fs, 35.0,
                          carrier_offset_hz=A_REAL_OFFSET_HZ)
    weak = _apply_channel(clean[half:], fs, 0.0,
                          carrier_offset_hz=A_REAL_OFFSET_HZ)
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
    bank 34.0/35.2 dB.  Both of those were measured at a zero carrier
    offset, in the DC blocker's notch; off it the same FIR bank
    measures 46.6/46.9 dB (2026-09-21, and 35.5/35.9 with the offset
    put back, which is the old number).  Floor at 40 dB keeps the
    structural improvement while tolerating measurement scatter; it
    also guards the final audio lowpass (a band limit encroaching
    below 15 kHz would show up here first).
    """
    np.random.seed(0)
    m = evaluate_quality(
        duration_s=4.0, tone_hz=14_000.0, cnr_db=None,
        pilot_amp=0.10, freq_dev_hz=75_000.0, warmup_s=0.8,
        hifi_tx=True, hifi_constant_mod=True,
        carrier_offset_hz=A_REAL_OFFSET_HZ,
    )
    assert m.separation_l_to_r_db > 40.0, m
    assert m.separation_r_to_l_db > 40.0, m


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

    All of those were measured at a zero carrier offset, inside the
    DC blocker's notch, so they are not comparable with what this
    measures now: off the notch the gated tracker takes the error out
    completely and the scenario reads 70.3 / 72.3 dB, against 5.0 /
    5.0 with the corrector disabled.  The floor is set from that pair
    rather than from the clamp history - the clamps are gone and
    cannot be re-measured - and it discriminates by 50 dB where the
    old one discriminated by 4-7.
    """
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    np.random.seed(0)
    m = evaluate_quality(
        **BASE_KWARGS,
        subcarrier_phase_offset_deg=STEREO_SUBCARRIER_PHASE_OFFSET_DEG - 75.0,
    )
    assert m.separation_l_to_r_db > 55.0, m
    assert m.separation_r_to_l_db > 55.0, m


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


def _side_and_mono(duration_s, width_db, fs, level_db=0.0):
    """A programme whose side/mono power ratio is *width_db*.

    L = m + d and R = m - d: a common tone and a difference tone,
    the second scaled to set how wide the programme is.  Two tones
    rather than one, so that what comes back in the side band can be
    told from what leaked out of the mono band.  *level_db* turns
    the whole thing down, which is how a quiet passage is made - the
    width stays and the deviation shrinks.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    scale = 10.0 ** (width_db / 20.0)
    mono = np.sin(2 * np.pi * 997.0 * t)
    diff = scale * np.sin(2 * np.pi * 1499.0 * t)
    # Scaled by the peak the two tones can reach rather than the one
    # they happen to reach in this many seconds: two tones beat, so
    # normalising by the observed peak would make the modulation
    # depth - and with it every level these tests read - depend on
    # how long the run is.
    gain = (0.7 * 10.0 ** (level_db / 20.0)) / (0.5 * (1.0 + scale))
    left, right = gain * 0.5 * (mono + diff), gain * 0.5 * (mono - diff)
    return left.astype(np.float32), right.astype(np.float32)


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
        readings, over_mono, axis, used = [], [], [], []
        lefts, rights = [], []
        for i in range(0, iq.size, block_size):
            c = iq[i:i + block_size]
            if c.size < 8:
                break
            left, right = d.demodulate(d.process_iq_samples(c))
            readings.append(d.stereo_phase_side_over_noise_db)
            over_mono.append(d.stereo_phase_side_over_mono_db)
            axis.append(d.stereo_phase_axis_deg)
            used.append(d.stereo_phase_informative)
            if left.size:
                lefts.append(left)
                rights.append(right)
        return {
            "readings": np.asarray(readings),
            "side_over_mono_db": np.asarray(over_mono),
            "axis_deg": np.asarray(axis),
            "informative": np.asarray(used),
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


def _a_programme(width_db, rotation_deg, cnr_db, duration_s=4.0,
                 level_db=0.0):
    """Synthetic FM carrying a programme of a known width and level."""
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE,
    )
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq

    left, right = _side_and_mono(duration_s, width_db,
                                 int(AUDIO_OUTPUT_RATE), level_db)
    mpx = _build_mpx(left, right, int(AUDIO_OUTPUT_RATE),
                     int(COMPOSITE_RATE), 0.10, True, 50e-6, rotation_deg)
    np.random.seed(0)
    return _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                           75_000.0, cnr_db,
                           carrier_offset_hz=A_REAL_OFFSET_HZ)


@pytest.mark.slow
def test_noise_alone_stays_under_the_side_over_noise_gate():
    """The gate is a margin over measured noise; keep it one.

    The discriminator's noise rises as f^2, so the demodulated side
    band carries more noise than the mono band, and that noise is
    anisotropic - a stable pseudo-axis the tracker will lock to
    during silence if it is let.  STEREO_PHASE_SIDE_OVER_NOISE_DB is
    set from measured noise, so a change to the filters that moves
    the noise floor has to move it too.

    This is the lower half of the calibration.  The other half -
    that it does not also refuse real programme - is not a property
    of silence and is pinned in
    test_a_quiet_passage_is_what_the_noise_gate_is_for.

    Re-measured 2026-09-21 through the FIR path: silence reads med
    22.7 / max 24.2 at CNR 45/35/25/15, and measures the same at a
    carrier offset of 0, 60 or 1237 Hz.  That is an observation and
    not a prediction - this fixture is silent audio, not a silent
    transmitter, and its pilot and carrier are discrete lines like
    any other - so it runs at A_REAL_OFFSET_HZ with the rest.
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
                             75_000.0, cnr,
                             carrier_offset_hz=A_REAL_OFFSET_HZ)
        out = _run_gates(iq, SDR_BLOCK_SIZE)
        assert not out["acquired"], cnr
        worst = max(worst, float(out["readings"].max()))

    margin = STEREO_PHASE_SIDE_OVER_NOISE_DB - worst
    assert margin > 0.5, (
        "silence reaches the gate (worst %.1f dB, gate %.1f)"
        % (worst, STEREO_PHASE_SIDE_OVER_NOISE_DB))


@pytest.mark.slow
def test_a_quiet_passage_is_what_the_noise_gate_is_for():
    """And it has to pass one, which is the other half of the margin.

    A quiet passage of stereo programme is where this gate does its
    work: the side band is still as wide as the mono one, so the
    side/mono gate is open, and what decides is whether there is
    enough of it above the noise.  Measured 2026-09-21 at 32 dB
    below full deviation, CNR 35: side/mono -13.9 dB (open),
    side/noise 27 dB, and it acquires.

    That leaves about a decibel above 26 before real programme
    starts being refused: the same passage against a 29 dB gate
    never acquires at all, and the tracker sits at the prior for
    the whole run.  Silence cannot show this, because silence is
    the half that says the gate must not be lower.
    """
    from fm_radio.constants import (
        SDR_BLOCK_SIZE, STEREO_PHASE_SIDE_GATE_DB,
    )

    iq = _a_programme(-10.0, 30.0, 35.0, level_db=-32.0)

    shipped = _run_gates(iq, SDR_BLOCK_SIZE)
    assert shipped["acquired"], "a quiet passage of stereo was refused"
    # The side/mono gate is open here, so what is being measured is
    # this gate and not that one.
    assert (float(np.median(shipped["side_over_mono_db"]))
            > STEREO_PHASE_SIDE_GATE_DB)

    raised = _run_gates(iq, SDR_BLOCK_SIZE, side_over_noise_db=29.0)
    assert not raised["acquired"], (
        "a gate 3 dB higher would still clear the noise, so what stops "
        "it being raised has to be this")


@pytest.mark.slow
def test_a_near_mono_programme_is_refused_by_the_side_gate():
    """Not by the noise gate, which is wide open on a clean signal.

    This is the case PR #49 asked about: a station that never
    acquires though its signal is excellent.  With the programme
    narrowed to a side/mono of -19.5 dB on a clean carrier, the
    side-over-noise reading is 45 dB - the noise gate passes every
    block - and STEREO_PHASE_SIDE_GATE_DB passes none.  Opening
    that one alone is enough to acquire; opening the noise gate
    alone changes nothing.

    Measured on the real captures, the two share the work depending
    on what is playing: of the blocks refused, the share failing
    only the side gate, only the noise gate, or both is 19/33/48%
    on optical 82.5, 10/71/18% on CATV 83.7, 28/14/58% on optical
    80.0 and 0/100/0% on antenna 91.6.  This test is about the
    clean, loud, near-mono corner, not about stations in general.
    """
    from fm_radio.constants import (
        SDR_BLOCK_SIZE, STEREO_PHASE_SIDE_GATE_DB,
        STEREO_PHASE_SIDE_OVER_NOISE_DB,
    )

    iq = _a_programme(-20.0, 30.0, 35.0)

    shipped = _run_gates(iq, SDR_BLOCK_SIZE)
    assert not shipped["acquired"], shipped["angle_deg"]
    assert shipped["angle_deg"] == 0.0, "it took an angle from nothing"

    side_open = float(np.mean(
        shipped["side_over_mono_db"] > STEREO_PHASE_SIDE_GATE_DB))
    noise_open = float(np.mean(
        shipped["readings"] >= STEREO_PHASE_SIDE_OVER_NOISE_DB))
    assert side_open == 0.0, "the side gate passed %.1f%%" % (100 * side_open)
    assert noise_open > 0.9, "the noise gate passed only %.1f%%" % (
        100 * noise_open)

    assert _run_gates(iq, SDR_BLOCK_SIZE, side_gate_db=-40.0)["acquired"], (
        "the side/mono gate was not what decided this")
    assert not _run_gates(iq, SDR_BLOCK_SIZE,
                          side_over_noise_db=6.0)["acquired"], (
        "the noise gate was deciding it after all")


@pytest.mark.slow
def test_a_block_reports_the_axis_it_saw_even_when_it_is_not_used():
    """The tracked angle is no evidence about a block it skipped.

    It is an EMA: while the gates are shut it holds what it had and
    leaks toward the prior, so reading it during a quiet stretch
    says what the loud stretch before said.  Deciding whether a
    refused block contained anything therefore needs the block's
    own estimate, which is why stereo_phase_axis_deg is taken on
    every block rather than only the ones that are used - it is
    what the reference-capture argument in
    STEREO_PHASE_SIDE_OVER_NOISE_DB's comment rests on.

    Content with a known rotation: the axis the blocks measure is
    the angle the tracker settles on.  Silence: no block is
    informative, the tracker stays at the prior, and an axis is
    still reported - the noise has one, which is the whole problem
    the gate exists for.
    """
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE, SDR_BLOCK_SIZE,
    )
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq

    fs = AUDIO_OUTPUT_RATE
    out = _run_gates(_a_programme(-10.0, 30.0, 35.0, duration_s=2.5),
                     SDR_BLOCK_SIZE)

    used = np.asarray(out["informative"])
    axis = np.asarray(out["axis_deg"])
    assert used.any(), "nothing was informative on wide stereo content"
    assert np.isfinite(axis[used]).all()
    # The blocks agree with each other, and with where the tracker
    # ended up: the second half, past the settling.
    settled = axis[used][len(axis[used]) // 2:]
    assert float(np.std(settled)) < 5.0, float(np.std(settled))
    assert abs(float(np.median(settled)) - out["angle_deg"]) < 2.0, (
        "%.1f deg of blocks against %.1f deg of tracker"
        % (np.median(settled), out["angle_deg"]))

    quiet = np.zeros(int(2.0 * fs), dtype=np.float32)
    mpx = _build_mpx(quiet, quiet, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, 0.0)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, 35.0, carrier_offset_hz=A_REAL_OFFSET_HZ)
    out = _run_gates(iq, SDR_BLOCK_SIZE)

    assert not np.asarray(out["informative"]).any(), "silence was used"
    assert out["angle_deg"] == 0.0
    assert np.isfinite(np.asarray(out["axis_deg"])).any(), (
        "the noise had no axis to report, which is not what the gate "
        "is there for")


@pytest.mark.slow
def test_phase_tracker_acquires_correct_branch_at_boundary():
    """Acquisition at a true rotation of -88 deg must not swap L/R.

    Raw estimates on a station near the +-90 boundary straddle it and
    ~half wrap to +88-ish; a single-block acquisition would lock the
    wrong 180-deg branch (permanent L/R swap) with that probability.
    The doubled-angle circular mean over the acquisition streak is
    invariant to the wrap, so separation stays high and positive.

    Off the DC notch this reads 70.3 / 72.3 dB against 1.0 / 1.0 with
    the corrector disabled, so the floor is set the same way as the
    one above (it was 24.0 when both were measured in the notch).
    """
    from fm_radio.constants import STEREO_SUBCARRIER_PHASE_OFFSET_DEG
    np.random.seed(0)
    m = evaluate_quality(
        **BASE_KWARGS,
        subcarrier_phase_offset_deg=STEREO_SUBCARRIER_PHASE_OFFSET_DEG - 88.0,
    )
    assert m.separation_l_to_r_db > 55.0, m
    assert m.separation_r_to_l_db > 55.0, m


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
    pi-periodic family and holds full separation throughout: off the
    DC notch this measures 58.7 / 59.4 dB against 6.3 / 6.2 with the
    corrector disabled.  The 30.8 / 29.5 dB in the history of this
    test, and its floor of 24, were measured at a zero carrier
    offset inside the notch.
    """
    np.random.seed(0)
    m = evaluate_quality(**BASE_KWARGS, dsb_phase_drift_deg_per_s=-40.0)
    assert m.separation_l_to_r_db > 45.0, m
    assert m.separation_r_to_l_db > 45.0, m
    assert m.thdn_left_db < -20.0, m


# Not slow any more, and not marked so: it stopped measuring
# anything when it stopped asserting the size of the pathology.
def test_the_scenarios_are_not_measured_in_the_dc_notch():
    """Every floor here but the dc-notch row was set off the notch.

    A default that drifted back to zero would take them all with it,
    and quietly: the floors sit far enough below the measurements
    that a notched run still clears several of them.  So the
    arrangement itself is asserted rather than measured - the
    default is outside the notch, and the one scenario that is
    inside it says so on purpose.

    Asserting the arrangement and not the gap between the two is
    deliberate.  Measured, that gap is 20 dB of separation and 17 of
    THD, but it is the size of a defect in the DC blocker: fixing
    that would shrink it, and a test that demanded it stay would
    fail on the improvement.  What the notch costs is watched by the
    dc-notch row's own floors, where it belongs.
    """
    from fm_radio.constants import DC_BLOCK_CUTOFF_HZ

    assert BASE_KWARGS["carrier_offset_hz"] == A_REAL_OFFSET_HZ
    assert abs(A_REAL_OFFSET_HZ) > 3.0 * DC_BLOCK_CUTOFF_HZ, (
        "the default offset is inside the notch transition")
    assert SCENARIOS["dc-notch"]["carrier_offset_hz"] == 0.0, (
        "the scenario that is supposed to sit in the notch does not")


@pytest.mark.slow
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_synthetic_quality_floors(scenario):
    np.random.seed(0)  # _fm_modulate_iq uses the legacy global RNG
    # Merged rather than passed as two mappings: two of the scenarios
    # set a carrier offset of their own, and theirs is the one that
    # counts.
    m = evaluate_quality(**{**BASE_KWARGS, **SCENARIOS[scenario]})
    floors = FLOORS[scenario]
    assert m.separation_l_to_r_db > floors["sep"], (scenario, m)
    assert m.separation_r_to_l_db > floors["sep"], (scenario, m)
    assert m.thdn_left_db < floors["thd"], (scenario, m)
    assert m.thdn_right_db < floors["thd"], (scenario, m)
    assert m.snr_left_db > floors["snr"], (scenario, m)
    assert m.snr_right_db > floors["snr"], (scenario, m)
    assert m.blend_mean > 0.8, (scenario, m)
