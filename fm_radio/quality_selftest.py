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
from scipy.io import wavfile

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


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32)
    if np.issubdtype(x.dtype, np.integer):
        scale = float(np.iinfo(x.dtype).max)
        return (x.astype(np.float32) / max(scale, 1.0)).astype(np.float32)
    return x.astype(np.float32)


def _load_stereo_wav(path: str, target_fs: int, duration_s: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    fs, raw = wavfile.read(path)
    x = _to_float_audio(np.asarray(raw))
    if x.ndim == 1:
        left = x
        right = x
    else:
        left = x[:, 0]
        right = x[:, 1] if x.shape[1] > 1 else x[:, 0]

    if duration_s is not None and duration_s > 0:
        n = int(duration_s * fs)
        left = left[:n]
        right = right[:n]
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
    fs, raw = wavfile.read(path)
    x = _to_float_audio(np.asarray(raw))
    if x.ndim < 2 or x.shape[1] < 2:
        raise ValueError("IQ wav must have at least 2 channels (I=ch0, Q=ch1).")
    i = x[:, 0]
    q = x[:, 1]
    if duration_s is not None and duration_s > 0:
        n = int(duration_s * fs)
        i = i[:n]
        q = q[:n]
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
    pilot = pilot_amp * np.cos(2.0 * np.pi * 19_000.0 * t)
    dsb_phase_rad = np.deg2rad(dsb_phase_deg)
    dsb = lmr * np.cos(2.0 * np.pi * 38_000.0 * t + dsb_phase_rad)
    mpx = lpr + dsb + pilot
    return mpx.astype(np.float32)


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


def _run_demod_from_iq(
    iq: np.ndarray, fixed_blend: float | None = None, disable_phase_align: bool = False,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demod = FMDemodulator(stereo=True)
    if fixed_blend is not None:
        demod.force_blend_factor = float(np.clip(fixed_blend, 0.0, 1.0))
    if disable_phase_align:
        demod.phase_align_enabled = False
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


def _run_demod_from_composite(
    composite: np.ndarray, fixed_blend: float | None = None, disable_phase_align: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demod = FMDemodulator(stereo=True)
    if fixed_blend is not None:
        demod.force_blend_factor = float(np.clip(fixed_blend, 0.0, 1.0))
    if disable_phase_align:
        demod.phase_align_enabled = False

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


def evaluate_quality(
    duration_s: float,
    tone_hz: float,
    cnr_db: float | None,
    pilot_amp: float,
    freq_dev_hz: float,
    fixed_blend: float | None = None,
    path: str = "full",
    warmup_s: float = 0.5,
    enable_preemphasis: bool = False,
    preemphasis_tau_s: float = 50e-6,
    dsb_phase_deg: float = 0.0,
    source_lr: tuple[np.ndarray, np.ndarray] | None = None,
    disable_phase_align: bool = False,
) -> QualityMetrics:
    fs_audio = AUDIO_OUTPUT_RATE
    fs_composite = int(COMPOSITE_RATE)
    fs_iq = int(SDR_SAMPLE_RATE)
    max_lag = int(0.2 * fs_audio)
    settle = int(max(0.0, warmup_s) * fs_audio)

    if source_lr is None:
        left_ref, right_ref = _make_stereo_tone(duration_s, fs_audio, tone_hz, 0.6, 0.6)
        calc_thdn = True
    else:
        left_ref, right_ref = source_lr
        calc_thdn = False
    mpx = _build_mpx(
        left_ref, right_ref, fs_audio, fs_composite, pilot_amp,
        enable_preemphasis=enable_preemphasis, preemphasis_tau_s=preemphasis_tau_s,
        dsb_phase_deg=dsb_phase_deg,
    )
    if path == "composite":
        left_out, right_out, blend_hist = _run_demod_from_composite(
            mpx, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )
    else:
        iq = _fm_modulate_iq(mpx, fs_composite, fs_iq, freq_dev_hz, cnr_db)
        left_out, right_out, blend_hist = _run_demod_from_iq(
            iq, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )

    left_ref_fit, left_x = _align_and_fit(left_ref, left_out, max_lag)
    right_ref_fit, right_x = _align_and_fit(right_ref, right_out, max_lag)

    left_ref_fit = left_ref_fit[settle:]
    left_x = left_x[settle:]
    right_ref_fit = right_ref_fit[settle:]
    right_x = right_x[settle:]

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
        dsb_phase_deg=dsb_phase_deg,
    )
    if path == "composite":
        l_main, r_leak, _ = _run_demod_from_composite(
            mpx_l, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )
    else:
        iq_l = _fm_modulate_iq(mpx_l, fs_composite, fs_iq, freq_dev_hz, cnr_db)
        l_main, r_leak, _ = _run_demod_from_iq(
            iq_l, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )
    sep_l2r = _stereo_separation_ls_db(l_main[settle:], r_leak[settle:], max_lag)

    l_zero = np.zeros_like(r_only)
    mpx_r = _build_mpx(
        l_zero, r_only, fs_audio, fs_composite, pilot_amp,
        enable_preemphasis=enable_preemphasis, preemphasis_tau_s=preemphasis_tau_s,
        dsb_phase_deg=dsb_phase_deg,
    )
    if path == "composite":
        l_leak, r_main, _ = _run_demod_from_composite(
            mpx_r, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )
    else:
        iq_r = _fm_modulate_iq(mpx_r, fs_composite, fs_iq, freq_dev_hz, cnr_db)
        l_leak, r_main, _ = _run_demod_from_iq(
            iq_r, fixed_blend=fixed_blend, disable_phase_align=disable_phase_align,
        )
    sep_r2l = _stereo_separation_ls_db(r_main[settle:], l_leak[settle:], max_lag)

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
        "--preemphasis", action="store_true",
        help="Apply pre-emphasis to synthetic L/R before MPX synthesis",
    )
    p.add_argument(
        "--preemphasis-tau-us", type=float, default=50.0,
        help="Pre-emphasis tau in microseconds",
    )
    p.add_argument(
        "--fixed-blend", type=float, default=-1.0,
        help="Set 0.0-1.0 to bypass adaptive blend (negative value disables)",
    )
    p.add_argument(
        "--disable-phase-align", action="store_true",
        help="Disable mono/LR phase-alignment correction in demodulator",
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
    return p


def main() -> None:
    args = _parser().parse_args()
    cnr_db = None if args.cnr_db < 0 else float(args.cnr_db)
    fixed_blend = None if args.fixed_blend < 0.0 else float(args.fixed_blend)
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
        disable_phase_align=bool(args.disable_phase_align),
    )
    if args.source_wav:
        eval_kwargs["source_lr"] = _load_stereo_wav(
            args.source_wav, AUDIO_OUTPUT_RATE, float(args.duration),
        )

    if args.iq_wav:
        iq = _load_iq_wav(args.iq_wav, int(SDR_SAMPLE_RATE), float(args.duration))
        left, right, blend = _run_demod_from_iq(
            iq, fixed_blend=fixed_blend, disable_phase_align=bool(args.disable_phase_align),
        )
        s = int(max(0.0, float(args.warmup_s)) * AUDIO_OUTPUT_RATE)
        left = left[s:]
        right = right[s:]
        n = min(left.size, right.size)
        left = left[:n]
        right = right[:n]
        corr = float(np.corrcoef(left, right)[0, 1]) if n > 8 else float("nan")
        rms_l = _rms(left)
        rms_r = _rms(right)
        side = _rms(left - right)
        mono = _rms(left + right)
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
        print("Reference metrics (THD+N/SNR/separation) require synthetic or source-wav mode.")
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
