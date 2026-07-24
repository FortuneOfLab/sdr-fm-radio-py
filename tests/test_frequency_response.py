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
    # 15 kHz FIR lowpass: the passband is FLAT through 15 k (the old
    # Butterworth was already -3 dB there), the 15->18.5 kHz transition
    # has begun by 16 k, and 18 k is deep in the Kaiser stopband slope.
    assert _at(freqs, db, 16000) < -10.5
    assert _at(freqs, db, 18000) < -40.0
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


def _run_composite_direct(mpx: np.ndarray, fs_audio: int = 48_000):
    """Feed a synthetic composite straight into the stereo demod."""
    import logging
    logging.disable(logging.CRITICAL)
    from fm_radio.demodulator import FMDemodulator
    d = FMDemodulator(stereo=True)
    d.force_blend_factor = 1.0
    d.subcarrier_phase_offset_rad = 0.0
    ls, rs = [], []
    for i in range(0, mpx.size - 3071, 3072):
        l, r = d.demodulate(mpx[i:i + 3072])
        ls.append(l)
        rs.append(r)
    left = np.concatenate(ls).astype(np.float64)
    right = np.concatenate(rs).astype(np.float64)
    n0 = int(1.5 * fs_audio)
    return left[n0:], right[n0:]


def _tone_power_dbfs(x: np.ndarray, f0: float, fs: int = 48_000,
                     bw: float = 60.0) -> float:
    win = np.hanning(x.size)
    sp = np.abs(np.fft.rfft(x * win)) ** 2
    fr = np.fft.rfftfreq(x.size, 1.0 / fs)
    m = (fr > f0 - bw) & (fr < f0 + bw)
    return float(10 * np.log10(
        sp[m].sum() / (win.sum() / 2) ** 2 * 2 + 1e-30
    ))


@pytest.mark.slow
def test_out_of_band_composite_never_reaches_audio():
    """Composite 20.5-22 k / 54-56.5 k content must not reach the audio.

    The raw-composite demod's ideal-bandpass equivalence holds in the
    0-15 kHz target band only: through the bank FIR's 15-18.5 kHz
    transition, composite tones at 38 -/+ 17 kHz (21 k and 55 k - the
    codex-identified leak regions) demodulate to 17 kHz side content.
    The final common audio lowpass must crush them (measured
    -130 dBFS for a 0.2-amplitude composite tone).
    """
    fs_c = 192_000
    n = int(3.0 * fs_c)
    t = np.arange(n) / fs_c
    pilot = 0.1 * np.cos(2 * np.pi * 19_000.0 * t)
    for fc in (21_000.0, 55_000.0):
        mpx = pilot + 0.2 * np.cos(2 * np.pi * fc * t)
        left, right = _run_composite_direct(mpx)
        side = 0.5 * (left - right)
        leak = _tone_power_dbfs(side, 17_000.0)
        assert leak < -90.0, (fc, leak)


@pytest.mark.slow
def test_side_response_above_15k_is_suppressed_at_full_blend():
    """Direct side response at 16/17 kHz vs the 14 kHz reference.

    16 kHz sits at the start of the final lowpass transition (measured
    -17.8 dB rel 14 k), 17 kHz is deep in its stopband (measured
    -107.9 dB rel 14 k).  Guards the audio band limit end to end at
    fixed blend = 1.
    """
    fs_c = 192_000
    n = int(3.0 * fs_c)
    t = np.arange(n) / fs_c
    pilot = 0.1 * np.cos(2 * np.pi * 19_000.0 * t)
    levels = {}
    for fa in (14_000.0, 16_000.0, 17_000.0):
        lmr = 0.45 * np.sin(2 * np.pi * fa * t)
        mpx = pilot + lmr * np.cos(2 * np.pi * 38_000.0 * t)
        left, right = _run_composite_direct(mpx)
        levels[fa] = _tone_power_dbfs(0.5 * (left - right), fa)
    assert levels[16_000.0] - levels[14_000.0] < -12.0, levels
    assert levels[17_000.0] - levels[14_000.0] < -60.0, levels


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
