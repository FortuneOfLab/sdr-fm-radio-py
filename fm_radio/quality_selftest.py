#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) [2025] FortuneOfLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Self-test for FM demodulation quality using synthetic IQ.

This module synthesizes known stereo audio -> FM multiplex -> IQ, then runs the
current demodulator pipeline and reports objective metrics:
  - THD+N (single tone, per channel)
  - SNR versus best-fit reference
  - Stereo separation (left-only / right-only excitation)

Usage:
  python -m fm_radio.quality_selftest
  python -m fm_radio.quality_selftest --duration 6 --cnr-db 35 --tone-hz 1000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

from fm_radio.constants import (
    AUDIO_OUTPUT_RATE,
    COMPOSITE_RATE,
    SDR_BLOCK_SIZE,
    SDR_SAMPLE_RATE,
    SIDE_NR_ENABLE,
)
from fm_radio.demodulator import FMDemodulator


EPS = 1e-12


@dataclass
class QualityMetrics:
    thdn_left_db: float
    thdn_right_db: float
    snr_left_db: float
    snr_right_db: float
    separation_l_to_r_db: float
    separation_r_to_l_db: float
    blend_mean: float
    blend_min: float
    blend_max: float


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + EPS))


def _resample(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    ratio = Fraction(int(dst_fs), int(src_fs)).limit_denominator()
    return signal.resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32)
    if np.issubdtype(x.dtype, np.integer):
        scale = float(np.iinfo(x.dtype).max)
        return (x.astype(np.float32) / max(scale, 1.0)).astype(np.float32)
    return x.astype(np.float32)


def _read_wav_window(path: str, duration_s: float | None) -> tuple[int, np.ndarray]:
    """Read up to ``duration_s`` seconds of a WAV without loading it whole.

    Uses scipy's mmap mode and slices the requested window *before* any
    dtype conversion, so peak memory is bounded by the window size, not
    the file size.  IQ captures can exceed 4 GB now that the receiver
    rotates long recordings; the previous eager read materialised the
    entire file as int16 and again as float32 regardless of
    ``--duration``.  Falls back to an eager read for WAV subtypes mmap
    cannot handle (e.g. 24-bit packed).
    """
    try:
        fs, raw = wavfile.read(path, mmap=True)
    except Exception:
        fs, raw = wavfile.read(path)
    if duration_s is not None and duration_s > 0:
        n = int(duration_s * fs)
        raw = raw[:n]
    return fs, np.asarray(raw)


def _load_stereo_wav(path: str, target_fs: int, duration_s: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    fs, raw = _read_wav_window(path, duration_s)
    x = _to_float_audio(raw)
    if x.ndim == 1:
        left = x
        right = x
    else:
        left = x[:, 0]
        right = x[:, 1] if x.shape[1] > 1 else x[:, 0]

    left = _resample(left, fs, target_fs)
    right = _resample(right, fs, target_fs)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    peak = float(max(np.max(np.abs(left)), np.max(np.abs(right)), EPS))
    if peak > 0:
        left = (0.8 * left / peak).astype(np.float32)
        right = (0.8 * right / peak).astype(np.float32)
    return left, right


def _load_iq_wav(path: str, target_fs: int, duration_s: float | None = None) -> np.ndarray:
    fs, raw = _read_wav_window(path, duration_s)
    if raw.ndim < 2 or raw.shape[1] < 2:
        raise ValueError("IQ wav must have at least 2 channels (I=ch0, Q=ch1).")
    x = _to_float_audio(raw)
    i = x[:, 0]
    q = x[:, 1]
    i = _resample(i, fs, target_fs)
    q = _resample(q, fs, target_fs)
    n = min(i.size, q.size)
    iq = i[:n].astype(np.float32) + 1j * q[:n].astype(np.float32)
    peak = float(np.max(np.abs(iq)) + EPS)
    return (0.95 * iq / peak).astype(np.complex64)


def _make_stereo_tone(
    duration_s: float,
    fs_audio: int,
    tone_hz: float,
    left_amp: float,
    right_amp: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(duration_s * fs_audio)
    t = np.arange(n, dtype=np.float64) / fs_audio
    left = left_amp * np.sin(2.0 * np.pi * tone_hz * t)
    right = right_amp * np.sin(2.0 * np.pi * tone_hz * t)
    return left.astype(np.float32), right.astype(np.float32)


def _preemphasis(x: np.ndarray, fs: int, tau_s: float) -> np.ndarray:
    # H(s) = 1 + s*tau, bilinear transformed.
    b, a = signal.bilinear([tau_s, 1.0], [1.0], fs=fs)
    return signal.lfilter(b, a, x).astype(np.float32)


def _build_mpx(
    left_audio: np.ndarray,
    right_audio: np.ndarray,
    fs_audio: int,
    fs_composite: int,
    pilot_amp: float,
    enable_preemphasis: bool,
    preemphasis_tau_s: float,
    dsb_phase_deg: float,
    clock_ppm: float = 0.0,
) -> np.ndarray:
    if enable_preemphasis:
        left_audio = _preemphasis(left_audio, fs_audio, preemphasis_tau_s)
        right_audio = _preemphasis(right_audio, fs_audio, preemphasis_tau_s)

    left = _resample(left_audio, fs_audio, fs_composite).astype(np.float64)
    right = _resample(right_audio, fs_audio, fs_composite).astype(np.float64)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    t = np.arange(n, dtype=np.float64) / fs_composite
    # Broadcast-like MPX scaling:
    # mono (L+R) at ~45%, stereo DSB (L-R) at ~45%, pilot at ~10%.
    lpr = 0.45 * (left + right)
    lmr = 0.45 * (left - right)
    # clock_ppm models a transmitter/receiver sample-clock mismatch by
    # scaling the pilot and subcarrier frequencies while keeping their
    # exact 2:1 coherence.  A real broadcast pilot is never at exactly
    # 19 000.000 Hz relative to the receiver clock — and an exactly
    # periodic pilot is the one condition under which several classes
    # of block-processing defects are invisible (the FFT-Hilbert edge
    # bug fixed in PR #6 was undetectable at 0 ppm).  Audio-band
    # frequencies are left unscaled: sub-Hz shifts of the test tone
    # are irrelevant to every metric.
    scale = 1.0 + clock_ppm * 1e-6
    pilot = pilot_amp * np.cos(2.0 * np.pi * 19_000.0 * scale * t)
    dsb_phase_rad = np.deg2rad(dsb_phase_deg)
    dsb = lmr * np.cos(2.0 * np.pi * 38_000.0 * scale * t + dsb_phase_rad)
    mpx = lpr + dsb + pilot
    return mpx.astype(np.float32)


def _fm_modulate_iq(
    mpx: np.ndarray,
    fs_composite: int,
    fs_iq: int,
    freq_dev_hz: float,
    cnr_db: float | None,
    carrier_offset_hz: float = 0.0,
    multipath_delay_us: float = 0.0,
    multipath_gain: float = 0.0,
    multipath_phase_deg: float = 0.0,
) -> np.ndarray:
    mpx_iq = _resample(mpx, fs_composite, fs_iq).astype(np.float64)
    k = 2.0 * np.pi * freq_dev_hz / fs_iq
    phase = np.cumsum(k * mpx_iq, dtype=np.float64)
    iq = np.exp(1j * phase).astype(np.complex64)

    # Receiver tuning error: shifts the whole FM signal inside the IQ
    # lowpass passband (appears as a DC term in the demodulated
    # composite and exercises asymmetric filtering of the sidebands).
    if carrier_offset_hz:
        n = np.arange(iq.size, dtype=np.float64)
        iq = (iq * np.exp(
            1j * 2.0 * np.pi * carrier_offset_hz * n / fs_iq
        )).astype(np.complex64)

    # Two-ray multipath: direct path plus one delayed, attenuated,
    # phase-rotated echo.  Delay is rounded to whole IQ samples
    # (~0.98 us at 1.024 Msps).  An echo delayed beyond the generated
    # signal never arrives within the window, so the echo stays zero
    # (guard needed: iq[:size - d] with d > size is a *non-empty*
    # negative slice and would raise on assignment to the empty
    # echo[d:]).
    if multipath_gain and multipath_delay_us > 0.0:
        d = max(1, int(round(multipath_delay_us * 1e-6 * fs_iq)))
        echo = np.zeros_like(iq)
        if d < iq.size:
            echo[d:] = iq[:iq.size - d]
        coeff = np.complex64(
            multipath_gain * np.exp(1j * np.deg2rad(multipath_phase_deg))
        )
        iq = (iq + coeff * echo).astype(np.complex64)

    if cnr_db is None:
        return iq
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = signal_power / (10.0 ** (cnr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = (np.random.randn(iq.size) + 1j * np.random.randn(iq.size)) * sigma
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def _run_demod_from_iq(
    iq: np.ndarray,
    fixed_blend: float | None = None,
    disable_iq_phase_correction: bool = False,
    mono_delay_samples: int | None = None,
    subcarrier_phase_offset_deg: float | None = None,
    demod_diag: bool = False,
    demod_diag_interval: int | None = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demod = FMDemodulator(stereo=True)
    if fixed_blend is not None:
        demod.force_blend_factor = float(np.clip(fixed_blend, 0.0, 1.0))
    if disable_iq_phase_correction:
        demod.iq_phase_correction_enabled = False
    if mono_delay_samples is not None and int(mono_delay_samples) >= 0:
        demod.mono_delay_samples = int(mono_delay_samples)
        demod._mono_delay_state = np.zeros(demod.mono_delay_samples, dtype=np.float32)
    if subcarrier_phase_offset_deg is not None:
        demod.subcarrier_phase_offset_rad = np.deg2rad(float(subcarrier_phase_offset_deg))
    if demod_diag:
        demod.diag_enable = True
    if demod_diag_interval is not None and demod_diag_interval > 0:
        demod.diag_log_interval_blocks = int(demod_diag_interval)
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    blend_hist: list[float] = []
    for i in range(0, iq.size, SDR_BLOCK_SIZE):
        chunk = iq[i:i + SDR_BLOCK_SIZE]
        if chunk.size < 8:
            break
        composite = demod.process_iq_samples(chunk)
        left, right = demod.demodulate(composite)
        left_chunks.append(left.astype(np.float32))
        right_chunks.append(right.astype(np.float32))
        blend_hist.append(float(demod.blend_factor))
    if not left_chunks:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.concatenate(left_chunks),
        np.concatenate(right_chunks),
        np.asarray(blend_hist, dtype=np.float32),
    )


def _run_demod_diag_iq(
    iq: np.ndarray,
    fixed_blend: float | None = None,
    disable_iq_phase_correction: bool = False,
    mono_delay_samples: int | None = None,
    subcarrier_phase_offset_deg: float | None = None,
    lr_high_max_gain: float | None = None,
    lr_super_high_max_gain: float | None = None,
    side_nr_enable: bool | None = None,
    side_nr_alpha_floor: float | None = None,
    side_nr_beta: float | None = None,
) -> dict[str, np.ndarray]:
    """Run the demodulator and capture per-block diagnostics.

    Returns a dict with keys ``left``, ``right``, ``blend``, ``pilot_snr_db``
    where blend / pilot_snr_db are one entry per processed block.
    """
    demod = FMDemodulator(stereo=True)
    if fixed_blend is not None:
        demod.force_blend_factor = float(np.clip(fixed_blend, 0.0, 1.0))
    if disable_iq_phase_correction:
        demod.iq_phase_correction_enabled = False
    if mono_delay_samples is not None and int(mono_delay_samples) >= 0:
        demod.mono_delay_samples = int(mono_delay_samples)
        demod._mono_delay_state = np.zeros(demod.mono_delay_samples, dtype=np.float32)
    if subcarrier_phase_offset_deg is not None:
        demod.subcarrier_phase_offset_rad = np.deg2rad(float(subcarrier_phase_offset_deg))
    if lr_high_max_gain is not None:
        demod.lr_high_max_gain = float(lr_high_max_gain)
    if lr_super_high_max_gain is not None:
        demod.lr_super_high_max_gain = float(lr_super_high_max_gain)
    if side_nr_enable is not None:
        demod.side_nr_enabled = bool(side_nr_enable)
    if side_nr_alpha_floor is not None:
        demod.side_nr.alpha_floor = float(side_nr_alpha_floor)
    if side_nr_beta is not None:
        demod.side_nr.beta = float(side_nr_beta)

    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    blend_hist: list[float] = []
    snr_hist: list[float] = []
    for i in range(0, iq.size, SDR_BLOCK_SIZE):
        chunk = iq[i:i + SDR_BLOCK_SIZE]
        if chunk.size < 8:
            break
        composite = demod.process_iq_samples(chunk)
        left, right = demod.demodulate(composite)
        left_chunks.append(left.astype(np.float32))
        right_chunks.append(right.astype(np.float32))
        blend_hist.append(float(demod.blend_factor))
        snr = demod.pilot_snr_ema
        snr_hist.append(float(snr) if snr is not None else float("nan"))
    if not left_chunks:
        empty_f = np.zeros(0, dtype=np.float32)
        return {
            "left": empty_f, "right": empty_f,
            "blend": empty_f, "pilot_snr_db": empty_f,
        }
    return {
        "left": np.concatenate(left_chunks),
        "right": np.concatenate(right_chunks),
        "blend": np.asarray(blend_hist, dtype=np.float32),
        "pilot_snr_db": np.asarray(snr_hist, dtype=np.float32),
    }


def _band_rms_percentiles(
    x: np.ndarray, fs: int, lo_hz: float, hi_hz: float,
    win_ms: float = 50.0,
) -> tuple[float, float, float]:
    """Short-time band-RMS percentiles (p10/p50/p90) of *x* in [lo,hi] Hz.

    Used as a noise-floor estimator: the p10 of a high-frequency band is
    dominated by noise (program content's HF energy is sparse).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    nyq = fs / 2.0
    lo = max(1e-6, min(lo_hz, nyq - 1.0)) / nyq
    hi = max(lo + 1e-6, min(hi_hz, nyq - 1.0)) / nyq
    sos = signal.butter(4, [lo, hi], btype="band", output="sos")
    y = signal.sosfilt(sos, x)
    win = max(1, int(win_ms * 1e-3 * fs))
    nb = y.size // win
    if nb < 4:
        rms = np.sqrt(np.mean(y ** 2) + EPS)
        return float(rms), float(rms), float(rms)
    blocks = y[: nb * win].reshape(nb, win)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + EPS)
    p10 = float(np.percentile(rms, 10))
    p50 = float(np.percentile(rms, 50))
    p90 = float(np.percentile(rms, 90))
    return p10, p50, p90


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def _noise_metrics(left: np.ndarray, right: np.ndarray, fs: int,
                    hf_lo_hz: float = 10000.0, hf_hi_hz: float = 14500.0,
                    ) -> dict[str, float]:
    """Audio-side noise metrics used to compare demod configurations."""
    n = min(left.size, right.size)
    if n <= 0:
        return {}
    left = left[:n]
    right = right[:n]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    p10_mid, p50_mid, p90_mid = _band_rms_percentiles(mid, fs, hf_lo_hz, hf_hi_hz)
    p10_side, p50_side, p90_side = _band_rms_percentiles(side, fs, hf_lo_hz, hf_hi_hz)
    return {
        "mid_hf_p10_db": _db(p10_mid),
        "mid_hf_p50_db": _db(p50_mid),
        "side_hf_p10_db": _db(p10_side),
        "side_hf_p50_db": _db(p50_side),
        "side_hf_p90_db": _db(p90_side),
        "side_hf_p10_lin": p10_side,
        "mid_hf_p10_lin": p10_mid,
    }


def _run_demod_from_composite(
    composite: np.ndarray,
    fixed_blend: float | None = None,
    disable_iq_phase_correction: bool = False,
    mono_delay_samples: int | None = None,
    subcarrier_phase_offset_deg: float | None = None,
    demod_diag: bool = False,
    demod_diag_interval: int | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demod = FMDemodulator(stereo=True)
    if fixed_blend is not None:
        demod.force_blend_factor = float(np.clip(fixed_blend, 0.0, 1.0))
    if disable_iq_phase_correction:
        demod.iq_phase_correction_enabled = False
    if mono_delay_samples is not None and int(mono_delay_samples) >= 0:
        demod.mono_delay_samples = int(mono_delay_samples)
        demod._mono_delay_state = np.zeros(demod.mono_delay_samples, dtype=np.float32)
    if subcarrier_phase_offset_deg is not None:
        demod.subcarrier_phase_offset_rad = np.deg2rad(float(subcarrier_phase_offset_deg))
    if demod_diag:
        demod.diag_enable = True
    if demod_diag_interval is not None and demod_diag_interval > 0:
        demod.diag_log_interval_blocks = int(demod_diag_interval)

    ratio = Fraction(int(COMPOSITE_RATE), int(SDR_SAMPLE_RATE)).limit_denominator()
    composite_block = max(256, int(SDR_BLOCK_SIZE * ratio.numerator / ratio.denominator))

    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    blend_hist: list[float] = []
    for i in range(0, composite.size, composite_block):
        chunk = composite[i:i + composite_block]
        if chunk.size < 8:
            break
        left, right = demod.demodulate(chunk)
        left_chunks.append(left.astype(np.float32))
        right_chunks.append(right.astype(np.float32))
        blend_hist.append(float(demod.blend_factor))
    if not left_chunks:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.concatenate(left_chunks),
        np.concatenate(right_chunks),
        np.asarray(blend_hist, dtype=np.float32),
    )


def _align_and_fit(ref: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if ref.size == 0 or x.size == 0:
        return np.zeros(0), np.zeros(0)

    n = min(ref.size, x.size)
    ref_w = ref[:n]
    x_w = x[:n]
    corr = signal.correlate(x_w, ref_w, mode="full", method="fft")
    lags = signal.correlation_lags(x_w.size, ref_w.size, mode="full")
    valid = np.abs(lags) <= max_lag
    best_lag = int(lags[valid][np.argmax(corr[valid])])

    if best_lag >= 0:
        x_a = x[best_lag:]
        ref_a = ref[:x_a.size]
    else:
        ref_a = ref[-best_lag:]
        x_a = x[:ref_a.size]

    m = min(ref_a.size, x_a.size)
    ref_a = ref_a[:m]
    x_a = x_a[:m]
    if m == 0:
        return np.zeros(0), np.zeros(0)

    a = float(np.dot(x_a, ref_a) / (np.dot(ref_a, ref_a) + EPS))
    b = float(np.mean(x_a - a * ref_a))
    ref_fit = a * ref_a + b
    return ref_fit.astype(np.float64), x_a.astype(np.float64)


# Robust-metric windowing: THD+N and SNR are computed per window and the
# MEDIAN across windows is reported.  A whole-signal single-FFT metric
# lets one localised transient (settle dynamics reaching past the warmup
# trim, an end-of-stream edge) dominate the reading: the same DSP
# measured -18 to -25 dB THD+N depending only on where the measurement
# window edges landed.  The median across 1 s windows is insensitive to
# a contaminated window and to the total duration.
METRIC_WIN_S = 1.0
METRIC_HOP_S = 0.5


def _windowed_median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def _iter_windows(n: int, fs: int):
    win = int(METRIC_WIN_S * fs)
    hop = int(METRIC_HOP_S * fs)
    if n < win + hop:
        yield 0, n  # too short to window: single segment
        return
    for start in range(0, n - win + 1, hop):
        yield start, start + win


def _snr_db_single(ref_fit: np.ndarray, x: np.ndarray) -> float:
    err = x - ref_fit
    return 10.0 * np.log10((np.mean(ref_fit ** 2) + EPS) / (np.mean(err ** 2) + EPS))


def _snr_db(ref_fit: np.ndarray, x: np.ndarray) -> float:
    """Median of per-window SNR (see METRIC_WIN_S note above)."""
    n = min(ref_fit.size, x.size)
    if n == 0:
        return float("nan")
    vals = [
        _snr_db_single(ref_fit[a:b], x[a:b])
        for a, b in _iter_windows(n, AUDIO_OUTPUT_RATE)
    ]
    return _windowed_median(vals)


def _thdn_db_single(x: np.ndarray, fs: int, tone_hz: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < fs // 2:
        return float("nan")
    w = np.hanning(n)
    xw = (x - np.mean(x)) * w
    spec = np.fft.rfft(xw)
    power = np.abs(spec) ** 2
    k0 = int(round(tone_hz * n / fs))
    k0 = max(1, min(k0, power.size - 2))
    # The Hann main lobe is 4 bins wide; +-3 bins covers it with margin
    # so fundamental leakage is not mis-counted as noise (the previous
    # +-1 bin clipped part of the main lobe and biased the floor).
    fund_bins = slice(max(0, k0 - 3), min(power.size, k0 + 4))
    fund_power = float(np.sum(power[fund_bins]) + EPS)
    total_power = float(np.sum(power[1:]) + EPS)
    noise_power = max(total_power - fund_power, EPS)
    return 10.0 * np.log10(noise_power / fund_power)


def _thdn_db(x: np.ndarray, fs: int, tone_hz: float) -> float:
    """Median of per-window THD+N (see METRIC_WIN_S note above)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < fs // 2:
        return float("nan")
    vals = [
        _thdn_db_single(x[a:b], fs, tone_hz)
        for a, b in _iter_windows(x.size, fs)
    ]
    return _windowed_median(vals)


def _align_pair(ref: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if ref.size == 0 or x.size == 0:
        return np.zeros(0), np.zeros(0)

    n = min(ref.size, x.size)
    ref_w = ref[:n]
    x_w = x[:n]
    corr = signal.correlate(x_w, ref_w, mode="full", method="fft")
    lags = signal.correlation_lags(x_w.size, ref_w.size, mode="full")
    valid = np.abs(lags) <= max_lag
    best_lag = int(lags[valid][np.argmax(corr[valid])])

    if best_lag >= 0:
        x_a = x[best_lag:]
        ref_a = ref[:x_a.size]
    else:
        ref_a = ref[-best_lag:]
        x_a = x[:ref_a.size]

    m = min(ref_a.size, x_a.size)
    return ref_a[:m].astype(np.float64), x_a[:m].astype(np.float64)


def _stereo_separation_ls_db(main: np.ndarray, leak: np.ndarray, max_lag: int) -> float:
    main_a, leak_a = _align_pair(main, leak, max_lag)
    if main_a.size == 0 or leak_a.size == 0:
        return float("nan")
    main_zm = main_a - np.mean(main_a)
    leak_zm = leak_a - np.mean(leak_a)
    k = float(np.dot(leak_zm, main_zm) / (np.dot(main_zm, main_zm) + EPS))
    return 20.0 * np.log10(1.0 / (abs(k) + EPS))


# ----------------------------------------------------------------------
# Frequency-response sweep
# ----------------------------------------------------------------------

def _sweep_tone_gain(
    freq_hz: float,
    mode: str,                       # "mono" (L=R) or "side" (L=-R)
    duration_s: float,
    fs_audio: int,
    fs_composite: int,
    fs_iq: int,
    freq_dev_hz: float,
    amp: float,
    enable_preemphasis: bool,
    preemphasis_tau_s: float,
    diag_kwargs: dict,
) -> float:
    """Excite a single audio tone and return output/input amplitude ratio.

    ``mode`` selects which stereo component carries the tone: ``mono``
    puts it in L+R (L=R), ``side`` in L-R (L=-R).  The excited component
    is recovered from the demodulated (L+R) or (L-R) and its amplitude at
    ``freq_hz`` is measured by single-bin correlation.  IQ is generated
    noiseless (cnr_db=None) so the ratio is the pure filter magnitude
    response of that path.  A single tone at a time avoids the FM
    intermodulation that corrupts a multitone probe.
    """
    n = int(duration_s * fs_audio)
    t = np.arange(n, dtype=np.float64) / fs_audio
    tone = (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    left = tone
    right = tone if mode == "mono" else (-tone)
    mpx = _build_mpx(
        left, right, fs_audio, fs_composite, 0.10,
        enable_preemphasis=enable_preemphasis,
        preemphasis_tau_s=preemphasis_tau_s, dsb_phase_deg=0.0,
    )
    iq = _fm_modulate_iq(mpx, fs_composite, fs_iq, freq_dev_hz, None)
    diag = _run_demod_diag_iq(iq, **diag_kwargs)
    lo, ro = diag["left"], diag["right"]
    settle = int(1.2 * fs_audio)
    lo = lo[settle:]
    ro = ro[settle:]
    m = min(lo.size, ro.size)
    if m <= 8:
        return float("nan")
    lo = lo[:m].astype(np.float64)
    ro = ro[:m].astype(np.float64)
    out = (lo + ro) if mode == "mono" else (lo - ro)
    tt = np.arange(m, dtype=np.float64) / fs_audio
    corr = np.mean(out * np.exp(-1j * 2.0 * np.pi * freq_hz * tt)) * 2.0
    # Input amplitude of the excited component is 2*amp for both L+R and
    # L-R (L=R gives L+R=2·tone; L=-R gives L-R=2·tone).
    return float(np.abs(corr) / (2.0 * amp))


def measure_frequency_response(
    freqs_hz: np.ndarray,
    modes: tuple[str, ...] = ("mono", "side"),
    duration_s: float = 2.0,
    amp: float = 0.25,
    freq_dev_hz: float = 75_000.0,
    enable_preemphasis: bool = True,
    preemphasis_tau_s: float = 50e-6,
    diag_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Measure the mono/side audio magnitude response over ``freqs_hz``.

    Returns a dict keyed by mode, each an array of linear gains aligned
    with ``freqs_hz`` (snap the frequencies to FFT-bin centres upstream
    for cleanest results).
    """
    fs_audio = AUDIO_OUTPUT_RATE
    fs_composite = int(COMPOSITE_RATE)
    fs_iq = int(SDR_SAMPLE_RATE)
    diag_kwargs = dict(diag_kwargs or {})
    out: dict[str, np.ndarray] = {}
    for mode in modes:
        gains = [
            _sweep_tone_gain(
                float(f), mode, duration_s, fs_audio, fs_composite, fs_iq,
                freq_dev_hz, amp, enable_preemphasis, preemphasis_tau_s,
                diag_kwargs,
            )
            for f in freqs_hz
        ]
        out[mode] = np.asarray(gains, dtype=np.float64)
    return out


def _default_sweep_freqs(fs_audio: int, duration_s: float) -> np.ndarray:
    """Log-spaced probe frequencies snapped to FFT-bin centres."""
    n = int(duration_s * fs_audio)
    raw = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000,
           7000, 9000, 11000, 13000, 14000, 15000, 16000, 17000, 18000,
           19000, 20000]
    # Snap each to the nearest FFT bin of the post-settle segment so the
    # single-bin correlation has no scalloping loss.
    seg = n - int(1.2 * fs_audio)
    snapped = sorted({round(f * seg / fs_audio) * fs_audio / seg for f in raw})
    return np.asarray([f for f in snapped if 0 < f < fs_audio / 2],
                      dtype=np.float64)


def evaluate_quality(
    duration_s: float,
    tone_hz: float,
    cnr_db: float | None,
    pilot_amp: float,
    freq_dev_hz: float,
    fixed_blend: float | None = None,
    path: str = "full",
    warmup_s: float = 0.5,
    enable_preemphasis: bool = True,
    preemphasis_tau_s: float = 50e-6,
    dsb_phase_deg: float = 0.0,
    source_lr: tuple[np.ndarray, np.ndarray] | None = None,
    disable_iq_phase_correction: bool = False,
    mono_delay_samples: int | None = None,
    subcarrier_phase_offset_deg: float | None = None,
    demod_diag: bool = False,
    demod_diag_interval: int | None = None,
    clock_ppm: float = 0.0,
    carrier_offset_hz: float = 0.0,
    multipath_delay_us: float = 0.0,
    multipath_gain: float = 0.0,
    multipath_phase_deg: float = 0.0,
) -> QualityMetrics:
    fs_audio = AUDIO_OUTPUT_RATE
    fs_composite = int(COMPOSITE_RATE)
    fs_iq = int(SDR_SAMPLE_RATE)
    max_lag = int(0.2 * fs_audio)
    settle = int(max(0.0, warmup_s) * fs_audio)
    # Fixed tail guard: the last fraction of the aligned stream contains
    # end-of-stream edge effects (final partial processing blocks,
    # held-back resampler tails) whose inclusion depends on the total
    # duration; trimming a fixed amount makes the measurement segment
    # deterministic w.r.t. duration.
    tail_guard = int(0.25 * fs_audio)

    def _trim(seg: np.ndarray) -> np.ndarray:
        if seg.size > settle + tail_guard:
            return seg[settle:seg.size - tail_guard]
        return seg[settle:]

    impairments = dict(
        carrier_offset_hz=carrier_offset_hz,
        multipath_delay_us=multipath_delay_us,
        multipath_gain=multipath_gain,
        multipath_phase_deg=multipath_phase_deg,
    )

    if source_lr is None:
        left_ref, right_ref = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.6, 0.6)
        calc_thdn = True
    else:
        left_ref, right_ref = source_lr
        calc_thdn = False
    mpx = _build_mpx(
        left_ref, right_ref, fs_audio, fs_composite, pilot_amp,
        enable_preemphasis=enable_preemphasis, preemphasis_tau_s=preemphasis_tau_s,
        dsb_phase_deg=dsb_phase_deg, clock_ppm=clock_ppm,
    )
    if path == "composite":
        left_out, right_out, blend_hist = _run_demod_from_composite(
            mpx, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )
    else:
        iq = _fm_modulate_iq(mpx, fs_composite, fs_iq, freq_dev_hz, cnr_db,
                               **impairments)
        left_out, right_out, blend_hist = _run_demod_from_iq(
            iq, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )

    left_ref_fit, left_x = _align_and_fit(left_ref, left_out, max_lag)
    right_ref_fit, right_x = _align_and_fit(right_ref, right_out, max_lag)

    left_ref_fit = _trim(left_ref_fit)
    left_x = _trim(left_x)
    right_ref_fit = _trim(right_ref_fit)
    right_x = _trim(right_x)

    snr_l = _snr_db(left_ref_fit, left_x)
    snr_r = _snr_db(right_ref_fit, right_x)
    thdn_l = _thdn_db(left_x, fs_audio, tone_hz) if calc_thdn else float("nan")
    thdn_r = _thdn_db(right_x, fs_audio, tone_hz) if calc_thdn else float("nan")

    # 2) Separation: left-only then right-only.
    if source_lr is None:
        l_only, _ = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.7, 0.0)
        _, r_only = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.0, 0.7)
    else:
        l_only = left_ref
        r_only = right_ref
    r_zero = np.zeros_like(l_only)
    mpx_l = _build_mpx(
        l_only, r_zero, fs_audio, fs_composite, pilot_amp,
        enable_preemphasis=enable_preemphasis, preemphasis_tau_s=preemphasis_tau_s,
        dsb_phase_deg=dsb_phase_deg, clock_ppm=clock_ppm,
    )
    if path == "composite":
        l_main, r_leak, _ = _run_demod_from_composite(
            mpx_l, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )
    else:
        iq_l = _fm_modulate_iq(mpx_l, fs_composite, fs_iq, freq_dev_hz, cnr_db,
                               **impairments)
        l_main, r_leak, _ = _run_demod_from_iq(
            iq_l, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )
    sep_l2r = _stereo_separation_ls_db(_trim(l_main), _trim(r_leak), max_lag)

    l_zero = np.zeros_like(r_only)
    mpx_r = _build_mpx(
        l_zero, r_only, fs_audio, fs_composite, pilot_amp,
        enable_preemphasis=enable_preemphasis, preemphasis_tau_s=preemphasis_tau_s,
        dsb_phase_deg=dsb_phase_deg, clock_ppm=clock_ppm,
    )
    if path == "composite":
        l_leak, r_main, _ = _run_demod_from_composite(
            mpx_r, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )
    else:
        iq_r = _fm_modulate_iq(mpx_r, fs_composite, fs_iq, freq_dev_hz, cnr_db,
                               **impairments)
        l_leak, r_main, _ = _run_demod_from_iq(
            iq_r, fixed_blend=fixed_blend,
            disable_iq_phase_correction=disable_iq_phase_correction,
            mono_delay_samples=mono_delay_samples,
            subcarrier_phase_offset_deg=subcarrier_phase_offset_deg,
            demod_diag=demod_diag, demod_diag_interval=demod_diag_interval,
        )
    sep_r2l = _stereo_separation_ls_db(_trim(r_main), _trim(l_leak), max_lag)

    return QualityMetrics(
        thdn_left_db=thdn_l,
        thdn_right_db=thdn_r,
        snr_left_db=snr_l,
        snr_right_db=snr_r,
        separation_l_to_r_db=sep_l2r,
        separation_r_to_l_db=sep_r2l,
        blend_mean=float(np.mean(blend_hist)) if blend_hist.size else float("nan"),
        blend_min=float(np.min(blend_hist)) if blend_hist.size else float("nan"),
        blend_max=float(np.max(blend_hist)) if blend_hist.size else float("nan"),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FM demodulation quality self-test")
    p.add_argument("--duration", type=float, default=6.0, help="Test duration in seconds")
    p.add_argument("--tone-hz", type=float, default=1000.0, help="Tone frequency for THD/Sep")
    p.add_argument(
        "--source-wav", type=str, default="",
        help="Stereo WAV source for known-material injection (disables THD+N)",
    )
    p.add_argument(
        "--iq-wav", type=str, default="",
        help="Measured IQ WAV (I=ch0,Q=ch1). Runs diagnostics-only path.",
    )
    p.add_argument("--cnr-db", type=float, default=40.0,
                   help="Carrier-to-noise ratio in dB (set <0 to disable noise)")
    p.add_argument("--pilot-amp", type=float, default=0.10, help="Pilot amplitude in MPX")
    p.add_argument("--freq-dev-hz", type=float, default=75_000.0, help="FM frequency deviation")
    p.add_argument(
        "--path", choices=("full", "composite"), default="full",
        help="Evaluation path: full=MPX->FM IQ->demod, composite=MPX->stereo demod only",
    )
    p.add_argument(
        "--warmup-s", type=float, default=0.8,
        help="Seconds to discard at head for loop/filter settling",
    )
    p.add_argument(
        "--preemphasis", dest="preemphasis", action="store_true", default=True,
        help="Apply pre-emphasis to synthetic L/R before MPX synthesis "
             "(default: on — real broadcasts always transmit with "
             "pre-emphasis, and the receiver always de-emphasises)",
    )
    p.add_argument(
        "--no-preemphasis", dest="preemphasis", action="store_false",
        help="Disable pre-emphasis in the synthetic MPX",
    )
    p.add_argument(
        "--preemphasis-tau-us", type=float, default=50.0,
        help="Pre-emphasis tau in microseconds",
    )
    p.add_argument(
        "--clock-ppm", type=float, default=0.0,
        help="Pilot/subcarrier clock error in ppm (scales 19k/38k while "
             "keeping 2:1 coherence). Real pilots are never at exactly "
             "19 kHz relative to the receiver clock; ~200 ppm is a "
             "worst-case cheap-dongle crystal.",
    )
    p.add_argument(
        "--carrier-offset-hz", type=float, default=0.0,
        help="Receiver tuning error in Hz (shifts the FM signal inside "
             "the IQ lowpass; appears as DC in the composite)",
    )
    p.add_argument(
        "--multipath-delay-us", type=float, default=0.0,
        help="Two-ray multipath echo delay in microseconds (0=off)",
    )
    p.add_argument(
        "--multipath-gain", type=float, default=0.0,
        help="Two-ray multipath echo amplitude relative to direct path",
    )
    p.add_argument(
        "--multipath-phase-deg", type=float, default=0.0,
        help="Two-ray multipath echo carrier phase in degrees",
    )
    p.add_argument(
        "--fixed-blend", type=float, default=-1.0,
        help="Set 0.0-1.0 to bypass adaptive blend (negative value disables)",
    )
    p.add_argument(
        "--disable-iq-phase-correction", action="store_true",
        help="Disable I/Q phase rotation correction in LR synchronous demod",
    )
    p.add_argument(
        "--mono-delay-samples", type=int, default=-1,
        help="Mono delay override in composite samples (-1=default)",
    )
    p.add_argument(
        "--subcarrier-phase-offset-deg", type=float, default=float("nan"),
        help="Subcarrier phase offset override in degrees",
    )
    p.add_argument(
        "--demod-diag", action="store_true",
        help="Enable demodulator StereoDiag logging during self-test",
    )
    p.add_argument(
        "--demod-diag-interval", type=int, default=0,
        help="StereoDiag log interval in composite blocks (0=default)",
    )
    p.add_argument(
        "--dsb-phase-deg", type=float, default=0.0,
        help="Phase offset (degrees) for synthetic 38kHz DSB (L-R) generation",
    )
    p.add_argument(
        "--sweep-dsb-phase", action="store_true",
        help="Sweep DSB phase candidates and report the best separation score",
    )
    p.add_argument(
        "--sweep-candidates", type=str, default="0,90,180,270",
        help="Comma-separated DSB phase candidates in degrees for sweep mode",
    )
    p.add_argument(
        "--sweep-response", action="store_true",
        help="Measure the mono/side audio magnitude response (single-tone "
             "sweep, noiseless) and print a dB table.  Honours --preemphasis "
             "/ --no-preemphasis and the --side-nr* flags.",
    )
    p.add_argument(
        "--sweep-freqs", type=str, default="",
        help="Comma-separated probe frequencies in Hz for --sweep-response "
             "(default: log-spaced 50 Hz–20 kHz)",
    )
    p.add_argument(
        "--sweep-ref-hz", type=float, default=1000.0,
        help="Reference frequency (Hz) that --sweep-response normalises to 0 dB",
    )
    p.add_argument(
        "--play", action="store_true",
        help="Play demodulated audio via default audio device (IQ WAV mode)",
    )
    p.add_argument(
        "--out-wav", type=str, default="",
        help="Save demodulated audio to WAV file (IQ WAV mode)",
    )
    p.add_argument(
        "--noise-diag", dest="noise_diag", action="store_true", default=True,
        help="(IQ WAV mode) Run stereo + forced-mono demod, report excess "
             "stereo noise (default: on)",
    )
    p.add_argument(
        "--no-noise-diag", dest="noise_diag", action="store_false",
        help="Disable noise diagnostics (skips the second forced-mono pass)",
    )
    p.add_argument(
        "--noise-hf-lo-hz", type=float, default=10000.0,
        help="Lower edge of HF noise-floor band for stereo noise diagnostics",
    )
    p.add_argument(
        "--noise-hf-hi-hz", type=float, default=14500.0,
        help="Upper edge of HF noise-floor band for stereo noise diagnostics",
    )
    p.add_argument(
        "--noise-csv", type=str, default="",
        help="Append a CSV summary row of noise diagnostics to this file",
    )
    p.add_argument(
        "--noise-tag", type=str, default="",
        help="Free-form tag emitted in the CSV row (identifies the run)",
    )
    p.add_argument(
        "--lr-high-max-gain", type=float, default=float("nan"),
        help="Override LR_HIGH_MAX_GAIN ceiling for 7-12 kHz L-R "
             "(NaN=use constant). Lower values reduce HF stereo noise.",
    )
    p.add_argument(
        "--lr-super-high-max-gain", type=float, default=float("nan"),
        help="Override LR_SUPER_HIGH_MAX_GAIN ceiling for 12-15 kHz L-R "
             "(NaN=use constant). Lower values reduce HF stereo noise.",
    )
    p.add_argument(
        "--side-nr", dest="side_nr", action="store_true", default=None,
        help="Enable side-channel STFT noise reducer (overrides constant)",
    )
    p.add_argument(
        "--no-side-nr", dest="side_nr", action="store_false",
        help="Disable side-channel STFT noise reducer",
    )
    p.add_argument(
        "--side-nr-alpha-floor", type=float, default=float("nan"),
        help="Side NR minimum Wiener gain in linear units "
             "(0.15≈-16dB max attenuation). Higher = gentler.",
    )
    p.add_argument(
        "--side-nr-beta", type=float, default=float("nan"),
        help="Side NR over-subtraction factor (1.0=pure Wiener, "
             ">1=more aggressive)",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    cnr_db = None if args.cnr_db < 0 else float(args.cnr_db)
    fixed_blend = None if args.fixed_blend < 0.0 else float(args.fixed_blend)

    # Reject configurations that would produce zero post-warmup samples.
    # ``evaluate_quality`` and the IQ-WAV diagnostics path otherwise emit
    # NaN metrics and an empty WAV without surfacing the problem.
    if float(args.duration) <= float(args.warmup_s):
        raise SystemExit(
            f"--duration ({args.duration:.3f} s) must be greater than "
            f"--warmup-s ({args.warmup_s:.3f} s); the warmup window "
            f"would consume the entire test."
        )
    eval_kwargs = dict(
        duration_s=float(args.duration),
        tone_hz=float(args.tone_hz),
        cnr_db=cnr_db,
        pilot_amp=float(args.pilot_amp),
        freq_dev_hz=float(args.freq_dev_hz),
        fixed_blend=fixed_blend,
        path=str(args.path),
        warmup_s=float(args.warmup_s),
        enable_preemphasis=bool(args.preemphasis),
        preemphasis_tau_s=float(args.preemphasis_tau_us) * 1e-6,
        disable_iq_phase_correction=bool(args.disable_iq_phase_correction),
        mono_delay_samples=(None if int(args.mono_delay_samples) < 0 else int(args.mono_delay_samples)),
        subcarrier_phase_offset_deg=(
            None if np.isnan(float(args.subcarrier_phase_offset_deg))
            else float(args.subcarrier_phase_offset_deg)
        ),
        demod_diag=bool(args.demod_diag),
        demod_diag_interval=(None if int(args.demod_diag_interval) <= 0 else int(args.demod_diag_interval)),
        clock_ppm=float(args.clock_ppm),
        carrier_offset_hz=float(args.carrier_offset_hz),
        multipath_delay_us=float(args.multipath_delay_us),
        multipath_gain=float(args.multipath_gain),
        multipath_phase_deg=float(args.multipath_phase_deg),
    )
    if args.source_wav:
        eval_kwargs["source_lr"] = _load_stereo_wav(
            args.source_wav, AUDIO_OUTPUT_RATE, float(args.duration),
        )

    if args.iq_wav:
        if args.demod_diag:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        iq = _load_iq_wav(args.iq_wav, int(SDR_SAMPLE_RATE), float(args.duration))
        common_overrides = dict(
            disable_iq_phase_correction=bool(args.disable_iq_phase_correction),
            mono_delay_samples=(
                None if int(args.mono_delay_samples) < 0 else int(args.mono_delay_samples)
            ),
            subcarrier_phase_offset_deg=(
                None if np.isnan(float(args.subcarrier_phase_offset_deg))
                else float(args.subcarrier_phase_offset_deg)
            ),
            lr_high_max_gain=(
                None if np.isnan(float(args.lr_high_max_gain))
                else float(args.lr_high_max_gain)
            ),
            lr_super_high_max_gain=(
                None if np.isnan(float(args.lr_super_high_max_gain))
                else float(args.lr_super_high_max_gain)
            ),
            side_nr_enable=args.side_nr,
            side_nr_alpha_floor=(
                None if np.isnan(float(args.side_nr_alpha_floor))
                else float(args.side_nr_alpha_floor)
            ),
            side_nr_beta=(
                None if np.isnan(float(args.side_nr_beta))
                else float(args.side_nr_beta)
            ),
        )
        stereo_diag = _run_demod_diag_iq(
            iq, fixed_blend=fixed_blend, **common_overrides,
        )
        left = stereo_diag["left"]
        right = stereo_diag["right"]
        blend = stereo_diag["blend"]
        snr_hist = stereo_diag["pilot_snr_db"]

        s = int(max(0.0, float(args.warmup_s)) * AUDIO_OUTPUT_RATE)
        left_s = left[s:]
        right_s = right[s:]
        n = min(left_s.size, right_s.size)
        left_s = left_s[:n]
        right_s = right_s[:n]
        corr = float(np.corrcoef(left_s, right_s)[0, 1]) if n > 8 else float("nan")
        rms_l = _rms(left_s)
        rms_r = _rms(right_s)
        side = _rms(left_s - right_s)
        mono = _rms(left_s + right_s)
        print("FM Quality Self-Test (Measured IQ Diagnostics)")
        print(f"Samples(out): {n}")
        print(f"RMS L/R: {rms_l:.6f} / {rms_r:.6f}")
        print(f"L-R / L+R RMS ratio: {side/(mono+EPS):.4f}")
        print(f"L/R correlation: {corr:.4f}")
        if blend.size:
            print(
                f"Blend avg/min/max: {float(np.mean(blend)):.3f} / "
                f"{float(np.min(blend)):.3f} / {float(np.max(blend)):.3f}"
            )
        else:
            print("Blend avg/min/max: nan / nan / nan")
        finite_snr = snr_hist[np.isfinite(snr_hist)] if snr_hist.size else snr_hist
        if finite_snr.size:
            print(
                f"Pilot SNR p10/p50/avg/max [dB]: "
                f"{float(np.percentile(finite_snr, 10)):.2f} / "
                f"{float(np.median(finite_snr)):.2f} / "
                f"{float(np.mean(finite_snr)):.2f} / "
                f"{float(np.max(finite_snr)):.2f}"
            )
        else:
            print("Pilot SNR p10/p50/avg/max [dB]: nan / nan / nan / nan")

        # --- Audio-side stereo noise diagnostics ---
        side_hf_p10_db = float("nan")
        mid_hf_p10_db = float("nan")
        listen_penalty_db = float("nan")
        if args.noise_diag and n > 0:
            stereo_metrics = _noise_metrics(
                left_s, right_s, AUDIO_OUTPUT_RATE,
                hf_lo_hz=float(args.noise_hf_lo_hz),
                hf_hi_hz=float(args.noise_hf_hi_hz),
            )
            mid_hf_p10_db = stereo_metrics.get("mid_hf_p10_db", float("nan"))
            side_hf_p10_db = stereo_metrics.get("side_hf_p10_db", float("nan"))
            mid_lin = stereo_metrics.get("mid_hf_p10_lin", float("nan"))
            side_lin = stereo_metrics.get("side_hf_p10_lin", float("nan"))
            # HF noise penalty when listening in stereo vs mono.
            # Mono listening hears mid HF only; stereo listening hears
            # mid+/-side, so per-speaker HF power is mid^2 + side^2
            # (assuming side noise is uncorrelated with mid).
            if (np.isfinite(mid_lin) and np.isfinite(side_lin)
                    and mid_lin > 1e-12):
                ratio = (side_lin ** 2) / (mid_lin ** 2)
                listen_penalty_db = 10.0 * np.log10(1.0 + ratio)
            print(
                "Audio HF noise floor [dB] (band "
                f"{args.noise_hf_lo_hz/1e3:.1f}-{args.noise_hf_hi_hz/1e3:.1f}kHz, p10):"
            )
            print(
                f"  mid  (L+R)/2 = {mid_hf_p10_db:.2f}   "
                f"side (L-R)/2 = {side_hf_p10_db:.2f}   "
                f"(side - mid = {side_hf_p10_db - mid_hf_p10_db:+.2f} dB)"
            )
            print(
                f"Stereo listening HF penalty vs mono: "
                f"{listen_penalty_db:+.2f} dB"
            )
        print("Reference metrics (THD+N/SNR/separation) require synthetic or source-wav mode.")

        if args.noise_csv:
            import os as _os
            tag = args.noise_tag if args.noise_tag else _os.path.basename(args.iq_wav)
            header = (
                "tag,duration_s,blend_avg,pilot_snr_p10_db,pilot_snr_med_db,"
                "side_lr_ratio,mid_hf_p10_db,side_hf_p10_db,listen_penalty_db\n"
            )
            row = (
                f"{tag},{float(args.duration):.2f},"
                f"{(float(np.mean(blend)) if blend.size else float('nan')):.3f},"
                f"{(float(np.percentile(finite_snr,10)) if finite_snr.size else float('nan')):.2f},"
                f"{(float(np.median(finite_snr)) if finite_snr.size else float('nan')):.2f},"
                f"{side/(mono+EPS):.4f},"
                f"{mid_hf_p10_db:.2f},{side_hf_p10_db:.2f},{listen_penalty_db:.2f}\n"
            )
            need_header = (not _os.path.exists(args.noise_csv)) or (
                _os.path.getsize(args.noise_csv) == 0
            )
            with open(args.noise_csv, "a", encoding="utf-8") as f:
                if need_header:
                    f.write(header)
                f.write(row)
            print(f"CSV: appended row to {args.noise_csv}")

        if args.out_wav or args.play:
            import scipy.io.wavfile as wavfile
            stereo = np.stack([left, right], axis=-1)
            peak = float(np.max(np.abs(stereo))) + 1e-10
            stereo_16 = (stereo / peak * 0.9 * 32767).astype(np.int16)

            import os, subprocess
            play_path = args.out_wav if args.out_wav else os.path.join(os.getcwd(), "_selftest_play.wav")
            wavfile.write(play_path, int(AUDIO_OUTPUT_RATE), stereo_16)
            print(f"Saved: {play_path}")

            if args.play:
                print(f"Playing {n/AUDIO_OUTPUT_RATE:.1f}s audio ...")
                subprocess.run(
                    ["cmd", "/c", "start", "", play_path],
                    check=False,
                )
        return

    if args.sweep_response:
        if args.sweep_freqs.strip():
            freqs = np.asarray(
                [float(x) for x in args.sweep_freqs.split(",") if x.strip()],
                dtype=np.float64,
            )
        else:
            freqs = _default_sweep_freqs(AUDIO_OUTPUT_RATE, float(args.duration))
        diag_kwargs: dict = {}
        if args.side_nr is not None:
            diag_kwargs["side_nr_enable"] = bool(args.side_nr)
        if not np.isnan(float(args.side_nr_alpha_floor)):
            diag_kwargs["side_nr_alpha_floor"] = float(args.side_nr_alpha_floor)
        if not np.isnan(float(args.side_nr_beta)):
            diag_kwargs["side_nr_beta"] = float(args.side_nr_beta)
        resp = measure_frequency_response(
            freqs, duration_s=float(args.duration),
            freq_dev_hz=float(args.freq_dev_hz),
            enable_preemphasis=bool(args.preemphasis),
            preemphasis_tau_s=float(args.preemphasis_tau_us) * 1e-6,
            diag_kwargs=diag_kwargs,
        )
        ref_hz = float(args.sweep_ref_hz)
        ref_idx = int(np.argmin(np.abs(freqs - ref_hz)))
        print("FM Quality Self-Test (Frequency Response)")
        print(
            f"pre-emphasis={'on' if args.preemphasis else 'off'} "
            f"side_nr={diag_kwargs.get('side_nr_enable', SIDE_NR_ENABLE)} "
            f"ref={freqs[ref_idx]:.0f}Hz (0 dB)"
        )
        header = "freq_hz," + ",".join(f"{m}_db" for m in resp)
        print(header)
        rows_out: list[str] = [header]
        for i, f in enumerate(freqs):
            cells = []
            for m in resp:
                g = resp[m][i]
                ref = resp[m][ref_idx]
                db = 20.0 * np.log10((g + 1e-12) / (ref + 1e-12))
                cells.append(f"{db:.2f}")
            line = f"{f:.0f}," + ",".join(cells)
            print(line)
            rows_out.append(line)
        if args.noise_csv:
            with open(args.noise_csv, "w", encoding="utf-8") as f:
                f.write("\n".join(rows_out) + "\n")
            print(f"CSV: wrote frequency response to {args.noise_csv}")
        return

    if args.sweep_dsb_phase:
        phases = [float(x.strip()) for x in args.sweep_candidates.split(",") if x.strip()]
        rows: list[tuple[float, QualityMetrics]] = []
        for phase_deg in phases:
            m = evaluate_quality(dsb_phase_deg=phase_deg, **eval_kwargs)
            rows.append((phase_deg, m))
        rows.sort(key=lambda r: 0.5 * (r[1].separation_l_to_r_db + r[1].separation_r_to_l_db), reverse=True)
        print("FM Quality Self-Test (DSB phase sweep)")
        print("phase_deg,sep_l2r_db,sep_r2l_db,snr_l_db,thdn_l_db,blend_avg")
        for phase_deg, m in rows:
            print(
                f"{phase_deg:.1f},{m.separation_l_to_r_db:.2f},{m.separation_r_to_l_db:.2f},"
                f"{m.snr_left_db:.2f},{m.thdn_left_db:.2f},{m.blend_mean:.3f}"
            )
        best_phase, best = rows[0]
        print(f"Best phase: {best_phase:.1f} deg")
        print(
            f"Best separation avg: "
            f"{0.5 * (best.separation_l_to_r_db + best.separation_r_to_l_db):.2f} dB"
        )
        return

    m = evaluate_quality(dsb_phase_deg=float(args.dsb_phase_deg), **eval_kwargs)
    print("FM Quality Self-Test")
    print(f"THD+N L: {m.thdn_left_db:.2f} dB")
    print(f"THD+N R: {m.thdn_right_db:.2f} dB")
    print(f"SNR L:   {m.snr_left_db:.2f} dB")
    print(f"SNR R:   {m.snr_right_db:.2f} dB")
    print(f"Sep L->R: {m.separation_l_to_r_db:.2f} dB")
    print(f"Sep R->L: {m.separation_r_to_l_db:.2f} dB")
    print(f"Blend avg/min/max: {m.blend_mean:.3f} / {m.blend_min:.3f} / {m.blend_max:.3f}")


if __name__ == "__main__":
    main()
