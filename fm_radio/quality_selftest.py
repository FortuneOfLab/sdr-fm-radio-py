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
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.signal as signal

from fm_radio.constants import (
    AUDIO_OUTPUT_RATE,
    COMPOSITE_RATE,
    SDR_BLOCK_SIZE,
    SDR_SAMPLE_RATE,
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


def _build_mpx(
    left_audio: np.ndarray,
    right_audio: np.ndarray,
    fs_audio: int,
    fs_composite: int,
    pilot_amp: float,
) -> np.ndarray:
    left = _resample(left_audio, fs_audio, fs_composite).astype(np.float64)
    right = _resample(right_audio, fs_audio, fs_composite).astype(np.float64)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    t = np.arange(n, dtype=np.float64) / fs_composite
    lpr = left + right
    lmr = left - right
    pilot = pilot_amp * np.cos(2.0 * np.pi * 19_000.0 * t)
    dsb = lmr * np.cos(2.0 * np.pi * 38_000.0 * t)
    mpx = lpr + dsb + pilot
    peak = float(np.max(np.abs(mpx)) + EPS)
    return (0.9 * mpx / peak).astype(np.float32)


def _fm_modulate_iq(
    mpx: np.ndarray,
    fs_composite: int,
    fs_iq: int,
    freq_dev_hz: float,
    cnr_db: float | None,
) -> np.ndarray:
    mpx_iq = _resample(mpx, fs_composite, fs_iq).astype(np.float64)
    k = 2.0 * np.pi * freq_dev_hz / fs_iq
    phase = np.cumsum(k * mpx_iq, dtype=np.float64)
    iq = np.exp(1j * phase).astype(np.complex64)
    if cnr_db is None:
        return iq
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = signal_power / (10.0 ** (cnr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = (np.random.randn(iq.size) + 1j * np.random.randn(iq.size)) * sigma
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def _run_demod(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demod = FMDemodulator(stereo=True)
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


def _snr_db(ref_fit: np.ndarray, x: np.ndarray) -> float:
    err = x - ref_fit
    return 10.0 * np.log10((np.mean(ref_fit ** 2) + EPS) / (np.mean(err ** 2) + EPS))


def _thdn_db(x: np.ndarray, fs: int, tone_hz: float) -> float:
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
    fund_bins = slice(max(0, k0 - 1), min(power.size, k0 + 2))
    fund_power = float(np.sum(power[fund_bins]) + EPS)
    total_power = float(np.sum(power[1:]) + EPS)
    noise_power = max(total_power - fund_power, EPS)
    return 10.0 * np.log10(noise_power / fund_power)


def _tone_amplitude(x: np.ndarray, fs: int, tone_hz: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 16:
        return 0.0
    t = np.arange(n, dtype=np.float64) / fs
    s = np.sin(2.0 * np.pi * tone_hz * t)
    c = np.cos(2.0 * np.pi * tone_hz * t)
    a_s = 2.0 * np.dot(x, s) / n
    a_c = 2.0 * np.dot(x, c) / n
    return float(np.sqrt(a_s * a_s + a_c * a_c))


def _stereo_separation_db(main: np.ndarray, leak: np.ndarray, fs: int, tone_hz: float) -> float:
    a_main = _tone_amplitude(main, fs, tone_hz)
    a_leak = _tone_amplitude(leak, fs, tone_hz)
    return 20.0 * np.log10((a_main + EPS) / (a_leak + EPS))


def evaluate_quality(
    duration_s: float,
    tone_hz: float,
    cnr_db: float | None,
    pilot_amp: float,
    freq_dev_hz: float,
) -> QualityMetrics:
    fs_audio = AUDIO_OUTPUT_RATE
    fs_composite = int(COMPOSITE_RATE)
    fs_iq = int(SDR_SAMPLE_RATE)
    max_lag = int(0.2 * fs_audio)
    settle = int(0.5 * fs_audio)

    # 1) Equal L/R tone: THD+N and SNR.
    left_ref, right_ref = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.6, 0.6)
    mpx = _build_mpx(left_ref, right_ref, fs_audio, fs_composite, pilot_amp)
    iq = _fm_modulate_iq(mpx, fs_composite, fs_iq, freq_dev_hz, cnr_db)
    left_out, right_out, blend_hist = _run_demod(iq)

    left_ref_fit, left_x = _align_and_fit(left_ref, left_out, max_lag)
    right_ref_fit, right_x = _align_and_fit(right_ref, right_out, max_lag)

    left_ref_fit = left_ref_fit[settle:]
    left_x = left_x[settle:]
    right_ref_fit = right_ref_fit[settle:]
    right_x = right_x[settle:]

    snr_l = _snr_db(left_ref_fit, left_x)
    snr_r = _snr_db(right_ref_fit, right_x)
    thdn_l = _thdn_db(left_x, fs_audio, tone_hz)
    thdn_r = _thdn_db(right_x, fs_audio, tone_hz)

    # 2) Separation: left-only then right-only.
    l_only, r_zero = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.7, 0.0)
    iq_l = _fm_modulate_iq(
        _build_mpx(l_only, r_zero, fs_audio, fs_composite, pilot_amp),
        fs_composite, fs_iq, freq_dev_hz, cnr_db,
    )
    l_main, r_leak, _ = _run_demod(iq_l)
    sep_l2r = _stereo_separation_db(l_main[settle:], r_leak[settle:], fs_audio, tone_hz)

    l_zero, r_only = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.0, 0.7)
    iq_r = _fm_modulate_iq(
        _build_mpx(l_zero, r_only, fs_audio, fs_composite, pilot_amp),
        fs_composite, fs_iq, freq_dev_hz, cnr_db,
    )
    l_leak, r_main, _ = _run_demod(iq_r)
    sep_r2l = _stereo_separation_db(r_main[settle:], l_leak[settle:], fs_audio, tone_hz)

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
    p.add_argument("--cnr-db", type=float, default=40.0,
                   help="Carrier-to-noise ratio in dB (set <0 to disable noise)")
    p.add_argument("--pilot-amp", type=float, default=0.10, help="Pilot amplitude in MPX")
    p.add_argument("--freq-dev-hz", type=float, default=75_000.0, help="FM frequency deviation")
    return p


def main() -> None:
    args = _parser().parse_args()
    cnr_db = None if args.cnr_db < 0 else float(args.cnr_db)
    m = evaluate_quality(
        duration_s=float(args.duration),
        tone_hz=float(args.tone_hz),
        cnr_db=cnr_db,
        pilot_amp=float(args.pilot_amp),
        freq_dev_hz=float(args.freq_dev_hz),
    )
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
