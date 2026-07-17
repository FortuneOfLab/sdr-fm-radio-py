"""End-to-end synthetic quality gates, clean and impaired.

Runs the full MPX -> FM IQ -> demodulator chain and asserts conservative
floors for the objective metrics.  The floors sit well below the
measured values (clean run at CNR=35: Sep ~24/28 dB, THD+N ~-28 dB,
SNR ~31-32 dB with pre-emphasis on) so they are robust across platforms
and RNG noise draws while still catching structural regressions.

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

# Measured values (2026-07): clean Sep 24.5/28.4, clock 25.8/26.3,
# tuning 33.5/34.0, multipath 23.3/26.0; THD -27..-28; SNR 30.5-33.6.
FLOORS = {
    "clean": dict(sep=18.0, thd=-20.0, snr=24.0),
    "clock-200ppm": dict(sep=18.0, thd=-20.0, snr=24.0),
    "tuning-30kHz": dict(sep=18.0, thd=-20.0, snr=24.0),
    "multipath": dict(sep=16.0, thd=-20.0, snr=24.0),
}


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
