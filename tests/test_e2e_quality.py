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
    scale = 10.0 ** (width_db / 20.0)
    mono = np.sin(2 * np.pi * 997.0 * t)
    diff = scale * np.sin(2 * np.pi * 1499.0 * t)
    # Scaled by the peak the two tones can reach rather than the one
    # they happen to reach in this many seconds: two tones beat, so
    # normalising by the observed peak makes the modulation depth -
    # and with it every level these tests read - depend on how long
    # the run is.  Measured: the same -20 dB programme read a
    # side-over-noise median of 28.7 dB over 2.5 s and 26.4 over 5 s.
    gain = 0.7 / (0.5 * (1.0 + scale))
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


#: Long enough for the reading to have settled, and how much of it
#: to leave out.  With programme present the pilot-band noise
#: estimate takes about 3 s to come up, and until it has, every
#: side-over-noise reading is 4-5 dB high: the same -20 dB
#: programme reads a median of 28.8 dB over its first 2.5 s and
#: 25.3 dB after.  (On silence it settles at once, so the margin
#: test above is unaffected.)  Anything measured on a short run
#: here is measuring the transient.
SETTLED_FROM_S = 2.5
LONG_ENOUGH_S = 6.0


def _settled(out, block_size):
    """The readings from after the noise estimate has come up."""
    from fm_radio.constants import SDR_SAMPLE_RATE

    first = int(SETTLED_FROM_S / (block_size / float(SDR_SAMPLE_RATE)))
    return {k: (v[first:] if isinstance(v, np.ndarray) and v.ndim == 1
                and v.size > first else v)
            for k, v in out.items()}


def _a_programme(width_db, rotation_deg, cnr_db, duration_s=LONG_ENOUGH_S):
    """Synthetic FM carrying a programme of a known width."""
    from fm_radio.constants import (
        AUDIO_OUTPUT_RATE, COMPOSITE_RATE, SDR_SAMPLE_RATE,
    )
    from fm_radio.quality_selftest import _build_mpx, _fm_modulate_iq

    left, right = _side_and_mono(duration_s, width_db,
                                 int(AUDIO_OUTPUT_RATE))
    mpx = _build_mpx(left, right, int(AUDIO_OUTPUT_RATE),
                     int(COMPOSITE_RATE), 0.10, True, 50e-6, rotation_deg)
    np.random.seed(0)
    return _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                           75_000.0, cnr_db)


@pytest.mark.slow
def test_a_narrow_programme_still_gets_through_the_noise_gate():
    """The half of the calibration that silence cannot show.

    STEREO_PHASE_SIDE_OVER_NOISE_DB has to clear the noise (the
    test above) without also refusing programme, and the second
    bound is the tighter one: a genuinely stereo but narrow
    programme - side/mono -17 dB, CNR 35, the sort of thing a
    speech broadcast gives - reads a settled median of 25.6 dB,
    which is under the gate.  It acquires anyway, on the 15% of
    blocks that come in above it.  That leaves little room above
    26: at 29 the same programme never acquires at all and the
    tracker sits at the prior for the whole run.

    Measured 2026-09-21, settled blocks of a 6 s run.
    """
    from fm_radio.constants import SDR_BLOCK_SIZE

    iq = _a_programme(-20.0, 60.0, 35.0)

    shipped = _run_gates(iq, SDR_BLOCK_SIZE)
    assert shipped["acquired"], "a narrow programme was refused outright"
    # And the rotation it was given comes back out: this is a
    # programme worth tracking, not a technicality.
    both = _run_gates(iq, SDR_BLOCK_SIZE, side_over_noise_db=6.0,
                      side_gate_db=-40.0)
    from fm_radio.constants import AUDIO_OUTPUT_RATE
    fs = int(AUDIO_OUTPUT_RATE)
    got = _tone_db(0.5 * (shipped["left"] - shipped["right"]), 1499.0, fs,
                   SETTLED_FROM_S)
    best = _tone_db(0.5 * (both["left"] - both["right"]), 1499.0, fs,
                    SETTLED_FROM_S)
    assert got > best - 0.3, "%.2f dB of the side band went" % (best - got)

    raised = _run_gates(iq, SDR_BLOCK_SIZE, side_over_noise_db=29.0)
    assert not raised["acquired"], (
        "a gate 3 dB higher would still clear the noise, so what stops "
        "it being raised has to be this")


@pytest.mark.slow
def test_a_programme_with_no_side_band_left_is_refused_by_both_gates():
    """And by nothing else, which is what makes it explainable.

    Narrower still - side/mono -21 dB settled - and there is
    nothing in the side band but noise: the side/mono gate passes
    no blocks at all and the noise gate 3.7% of them, so the
    tracker holds the hardware-trim prior.

    Which of the two says no depends on when you ask.  A cold
    tracker acquires in the first seconds, and until the noise
    estimate has come up the noise gate is passing nearly
    everything - so acquisition is decided by the side/mono gate,
    and opening that one alone is enough.  Once settled both are
    shut.  Either way it is these two and nothing else.

    On the real captures both refuse blocks too, in proportions
    that depend on the programme - of the blocks turned away on
    optical 82.5, 19% fail only the side gate, 33% only the noise
    gate and 48% both.
    """
    from fm_radio.constants import (
        SDR_BLOCK_SIZE, STEREO_PHASE_SIDE_GATE_DB,
        STEREO_PHASE_SIDE_OVER_NOISE_DB,
    )

    iq = _a_programme(-25.0, 30.0, 35.0)

    shipped = _run_gates(iq, SDR_BLOCK_SIZE)
    assert not shipped["acquired"], shipped["angle_deg"]
    assert shipped["angle_deg"] == 0.0, "it took an angle from nothing"

    settled = _settled(shipped, SDR_BLOCK_SIZE)
    side_open = float(np.mean(
        settled["side_over_mono_db"] > STEREO_PHASE_SIDE_GATE_DB))
    noise_open = float(np.mean(
        settled["readings"] >= STEREO_PHASE_SIDE_OVER_NOISE_DB))
    assert side_open == 0.0, "the side gate passed %.1f%%" % (100 * side_open)
    assert noise_open < 0.2, "the noise gate passed %.1f%%" % (
        100 * noise_open)

    # Which of the two decides it is a question about the first
    # seconds, because that is when a cold tracker acquires: until
    # the noise estimate has come up the noise gate is passing
    # nearly everything, so the side/mono gate is the one saying no.
    # Opening it alone is enough to acquire; opening the other alone
    # is not, and nothing outside these two is involved.
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
    left, right = _side_and_mono(2.5, -10.0, fs)
    mpx = _build_mpx(left, right, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, 30.0)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, 35.0)
    out = _run_gates(iq, SDR_BLOCK_SIZE)

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
                         75_000.0, 35.0)
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
