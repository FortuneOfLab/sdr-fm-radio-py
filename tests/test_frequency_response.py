"""Audio-path frequency-response checks.

Uses the synthetic MPX -> FM IQ -> demod chain to measure the mono and
side magnitude responses and assert the filter chain behaves: flat
passband, de-emphasis rolloff, 15 kHz lowpass, a deep 19 kHz pilot
notch, and the side-NR stationary-tone attenuation vs bypass.

Marked slow (each probe tone runs a full demod pass).
"""

from __future__ import annotations

import numpy as np
import pytest

from fm_radio.quality_selftest import measure_frequency_response


def _db(gains: np.ndarray, freqs: np.ndarray, ref_hz: float) -> np.ndarray:
    ref = gains[int(np.argmin(np.abs(freqs - ref_hz)))]
    return 20.0 * np.log10((gains + 1e-12) / (ref + 1e-12))


def _at(freqs: np.ndarray, db: np.ndarray, f: float) -> float:
    return float(db[int(np.argmin(np.abs(freqs - f)))])


@pytest.mark.slow
def test_mono_path_deemphasis_and_lowpass():
    """De-emphasis rolloff (pre-emph off) + 15 kHz LPF + 19 kHz notch."""
    freqs = np.array([1000, 3000, 15000, 16000, 18000, 19000], dtype=float)
    # snap to a 2 s post-settle segment's bins
    resp = measure_frequency_response(
        freqs, modes=("mono",), duration_s=2.0, enable_preemphasis=False,
    )
    db = _db(resp["mono"], freqs, 1000.0)
    # 50 us de-emphasis: -3 dB near 3.2 kHz, so 3 kHz is a few dB down.
    assert -4.0 < _at(freqs, db, 3000) < -1.0
    # 15 kHz lowpass edge: still within a few dB at 15 k, deep past 18 k.
    assert _at(freqs, db, 16000) < -12.0
    assert _at(freqs, db, 18000) < -25.0
    # Pilot notch: 19 kHz must be crushed.
    assert _at(freqs, db, 19000) < -60.0


@pytest.mark.slow
def test_mono_passband_is_flat_low():
    freqs = np.array([100, 300, 1000], dtype=float)
    resp = measure_frequency_response(
        freqs, modes=("mono",), duration_s=2.0, enable_preemphasis=False,
    )
    db = _db(resp["mono"], freqs, 1000.0)
    assert abs(_at(freqs, db, 100)) < 1.0
    assert abs(_at(freqs, db, 300)) < 1.0


@pytest.mark.slow
def test_preemphasis_roundtrip_is_flat():
    """Pre-emph ON mono response must be flat within +-0.5 dB to 13 kHz.

    The synthetic analog pre-emphasis (exact FFT multiply) and the
    receiver's analog-fitted 1p1z de-emphasis must cancel.  Before the
    pair of fixes this showed a +3.5 dB bump at 13 kHz (+2.4 dB from
    the bilinear pre-emphasis over-boost, +1.1 dB from the matched-Z
    de-emphasis under-attenuation).
    """
    freqs = np.array([300, 1000, 3000, 7000, 11000, 13000], dtype=float)
    resp = measure_frequency_response(
        freqs, modes=("mono",), duration_s=2.0, enable_preemphasis=True,
    )
    db = _db(resp["mono"], freqs, 1000.0)
    assert np.max(np.abs(db)) < 0.5, db


@pytest.mark.slow
def test_side_nr_preserves_stationary_tone():
    """Side NR must pass a stationary side tone at near-bypass level.

    Without tonal protection the DD-Wiener minimum-statistics floor
    absorbed a sustained tone into the noise estimate and pinned its
    gain at alpha_floor (~-10 dB measured at 5 kHz).  The local-median
    clamp on the tracker input keeps the floor at the broadband level,
    so the tone survives within ~1 dB of the NR-off response.  Below
    1.5 kHz the NR band bypasses, so 1 kHz is a fair reference for
    both configs.
    """
    freqs = np.array([1000, 5000], dtype=float)
    on = measure_frequency_response(
        freqs, modes=("side",), duration_s=2.0,
        diag_kwargs={"side_nr_enable": True},
    )["side"]
    off = measure_frequency_response(
        freqs, modes=("side",), duration_s=2.0,
        diag_kwargs={"side_nr_enable": False},
    )["side"]
    # Reference both at 1 kHz (inside the NR bypass region).
    on_db = 20 * np.log10((on[1] + 1e-12) / (on[0] + 1e-12))
    off_db = 20 * np.log10((off[1] + 1e-12) / (off[0] + 1e-12))
    # The tone must sit within 1.5 dB of the bypass response (it sat
    # 7.5 dB below before the tonal protection).
    assert abs(on_db - off_db) < 1.5, (on_db, off_db)
