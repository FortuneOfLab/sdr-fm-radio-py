"""End-to-end synthetic quality gate.

Runs the full MPX -> FM IQ -> demodulator chain and asserts conservative
floors for the objective metrics.  These floors are far enough below the
measured values (Sep ~24/27 dB, THD+N ~-25.6 dB, SNR ~27.5 dB at
CNR=35 dB) to be robust across platforms and RNG noise draws, while
still catching any structural regression of the stereo decoder.

Marked slow: run explicitly with `pytest -m slow` or as part of CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from fm_radio.quality_selftest import evaluate_quality


@pytest.mark.slow
def test_synthetic_quality_floors():
    np.random.seed(0)  # _fm_modulate_iq uses the legacy global RNG
    m = evaluate_quality(
        duration_s=3.0,
        tone_hz=1000.0,
        cnr_db=35.0,
        pilot_amp=0.10,
        freq_dev_hz=75_000.0,
        warmup_s=0.8,
    )
    assert m.separation_l_to_r_db > 18.0, m
    assert m.separation_r_to_l_db > 18.0, m
    assert m.thdn_left_db < -20.0, m
    assert m.thdn_right_db < -20.0, m
    assert m.snr_left_db > 24.0, m
    assert m.snr_right_db > 24.0, m
    assert m.blend_mean > 0.8, m
