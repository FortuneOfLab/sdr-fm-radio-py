"""End-to-end synthetic quality gates, clean and impaired.

Runs the full MPX -> FM IQ -> demodulator chain and asserts conservative
floors for the objective metrics.  The floors sit well below the
measured values (clean run at CNR=35: Sep ~30/30.5 dB, THD+N ~-37 dB,
SNR ~35/37.5 dB with pre-emphasis on) so they are robust across
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
# ceilings, analog-exact pre-emphasis + analog-fitted de-emphasis):
# clean Sep 30.2/30.5, THD -37.4, SNR 34.9/37.5.  History: with the
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
