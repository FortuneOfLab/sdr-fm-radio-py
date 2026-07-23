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
    np.random.seed(0)
    m = evaluate_quality(**BASE_KWARGS, subcarrier_phase_offset_deg=241.0)
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


@pytest.mark.slow
def test_phase_tracker_acquires_correct_branch_at_boundary():
    """Acquisition at a true rotation of -88 deg must not swap L/R.

    Raw estimates on a station near the +-90 boundary straddle it and
    ~half wrap to +88-ish; a single-block acquisition would lock the
    wrong 180-deg branch (permanent L/R swap) with that probability.
    The doubled-angle circular mean over the acquisition streak is
    invariant to the wrap, so separation stays high and positive.
    """
    np.random.seed(0)
    m = evaluate_quality(**BASE_KWARGS, subcarrier_phase_offset_deg=228.0)
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
    past +100 deg - exercising both the block and the admit paths.
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
    # dsb phase -45 deg maps to a ~+50 deg corrector demand in this
    # chain (see the leak test); -100 deg lands the axis near +110.
    mpx = _build_mpx(left, right, fs, int(COMPOSITE_RATE), 0.10, True,
                     50e-6, -100.0)
    np.random.seed(0)
    iq = _fm_modulate_iq(mpx, int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE),
                         75_000.0, 35.0)
    d = FMDemodulator(stereo=True)
    d.subcarrier_phase_offset_rad = np.deg2rad(316.0)  # DSP value
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
    assert emas[-1] > 100.0, emas[-1]  # admitted after confidence built


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
    d.subcarrier_phase_offset_rad = np.deg2rad(316.0)  # DSP value (no tuner)
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
