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
"""FM demodulator classes (Standard and Light versions).

Class hierarchy:
    FMDemodulatorInterface (ABC)
      -> BaseFMDemodulator (shared filters, demodulation, resampling)
           -> FMDemodulator      (PLL-based IQ processing)
           -> FMDemodulatorLight (phase-differentiation IQ processing)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from fractions import Fraction

import numpy as np
import scipy.signal as signal

from fm_radio.interfaces import FMDemodulatorInterface
from fm_radio.exceptions import DemodulationError
from fm_radio.filters import LowpassFilter, BandpassFilter, NotchFilter, DeemphasisIIRFilter
from fm_radio.pll import PLL
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_SAMPLE_RATE_LIGHT,
    MAIN_PLL_KP, MAIN_PLL_KI, PILOT_PLL_KP, PILOT_PLL_KI,
    IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF,
    MONO_LOWPASS_ORDER, MONO_LOWPASS_ORDER_LIGHT, MONO_LOWPASS_CUTOFF,
    LR_BASE_LOWPASS_CUTOFF, LR_HIGH_SPLIT_CUTOFF, LR_HIGH_SUPER_SPLIT_CUTOFF,
    LR_HIGH_MIN_GAIN, LR_SUPER_HIGH_MIN_GAIN,
    LR_HIGH_GATE_THRESHOLD, LR_HIGH_GATE_KNEE_MULT,
    LR_HIGH_GATE_MIN_GAIN, LR_HIGH_GATE_SMOOTHING,
    STEREO_HIGH_GATE_SNR_ASSIST_ENABLE,
    STEREO_HIGH_GATE_SNR_ASSIST_DB_LO, STEREO_HIGH_GATE_SNR_ASSIST_DB_HI,
    STEREO_HIGH_GATE_SNR_ASSIST_MAX, STEREO_HIGH_GATE_SNR_FLOOR_BOOST_MAX,
    PILOT_BANDPASS_ORDER, PILOT_BANDPASS_ORDER_LIGHT,
    PILOT_BANDPASS_LOW, PILOT_BANDPASS_HIGH,
    STEREO_PILOT_PHASE_MODE, STEREO_PILOT_RESIDUAL_CENTER_HZ,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG,
    STEREO_MONO_DELAY_SAMPLES,
    STEREO_LR_SIDE_RATIO_CAP_ENABLE, STEREO_LR_SIDE_RATIO_CAP_TARGET,
    STEREO_LR_SIDE_RATIO_CAP_MIN_GAIN,
    STEREO_LR_SIDE_RATIO_CAP_ATTACK, STEREO_LR_SIDE_RATIO_CAP_RELEASE,
    STEREO_PHASE_ERR_SMOOTHING, STEREO_PHASE_ERR_LIMIT_DEG, STEREO_IQ_PHASE_CORRECTION_ENABLE,
    STEREO_LEGACY_LR_DEMOD_ENABLE,
    PILOT_NOISE_BAND1_LOW, PILOT_NOISE_BAND1_HIGH,
    PILOT_NOISE_BAND2_LOW, PILOT_NOISE_BAND2_HIGH,
    LR_BANDPASS_ORDER, LR_BANDPASS_ORDER_LIGHT,
    LR_BANDPASS_LOW, LR_BANDPASS_HIGH,
    STEREO_LR_DEMOD_GAIN,
    STEREO_MONO_LR_PHASE_ALIGN_COH_MIN,
    STEREO_MONO_LR_PHASE_ALIGN_SIDE_RATIO_MIN,
    STEREO_MONO_LR_PHASE_ALIGN_SIDE_RATIO_MAX,
    STEREO_MONO_LR_PHASE_ALIGN_LIMIT_DEG,
    STEREO_MONO_LR_PHASE_ALIGN_SMOOTHING, STEREO_MONO_LR_PHASE_ALIGN_DECAY,
    STEREO_PHASE_ALIGN_ENABLE, STEREO_DIAG_ENABLE, STEREO_DIAG_LOG_INTERVAL_BLOCKS,
    DEEMPHASIS_TAU, DC_OFFSET_ALPHA,
    AUDIO_OUTPUT_RATE, COMPOSITE_RATE, LIGHT_COMPOSITE_SCALE,
    STANDARD_RESAMPLE_KAISER_BETA,
    STEREO_BLEND_PILOT_SNR_DB_HI, STEREO_BLEND_PILOT_SNR_DB_LO,
    STEREO_BLEND_PILOT_SNR_EMA_ALPHA,
    STEREO_BLEND_PILOT_JITTER_EMA_ALPHA,
    STEREO_BLEND_PILOT_JITTER_REF_DB,
    STEREO_BLEND_STABILITY_MIN_FACTOR,
    STEREO_BLEND_STABILITY_MIN_FACTOR_RESIDUAL,
    STEREO_BLEND_SMOOTHING,
    PILOT_NOTCH_FREQ, PILOT_NOTCH_Q,
)


class BaseFMDemodulator(FMDemodulatorInterface):
    """Base class for FM demodulators.

    Consolidates the shared logic of FMDemodulator and FMDemodulatorLight:
      - Filter initialisation (mono lowpass, pilot bandpass, L-R bandpass,
        de-emphasis) with configurable filter orders.
      - Resampling ratio calculation (IQ -> composite, composite -> audio).
      - Stereo/mono demodulation from the composite signal.
      - DC offset tracking.

    Subclasses must implement:
      - ``process_iq_samples``: convert raw IQ to composite signal.
      - ``_reset_subclass``: reset subclass-specific state.
    """

    def __init__(self, iq_sample_rate: float, composite_rate: float,
                 final_audio_rate: float, stereo: bool,
                 mono_order: int, pilot_order: int, lr_order: int,
                 logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.iq_sample_rate = iq_sample_rate
        self.composite_rate = composite_rate
        self.final_audio_rate = final_audio_rate
        self.stereo = stereo

        # --- Pilot PLL (shared by both standard and light) ---
        self.pilot_pll = PLL(Kp=PILOT_PLL_KP, Ki=PILOT_PLL_KI, return_phase=True)

        # --- Filters (order varies between standard and light) ---
        self.lp_mono = LowpassFilter(
            order=mono_order, cutoff=MONO_LOWPASS_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_base = LowpassFilter(
            order=mono_order, cutoff=LR_BASE_LOWPASS_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_base_q = LowpassFilter(
            order=mono_order, cutoff=LR_BASE_LOWPASS_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_low = LowpassFilter(
            order=mono_order, cutoff=LR_HIGH_SPLIT_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_mono_align = LowpassFilter(
            order=mono_order, cutoff=LR_HIGH_SPLIT_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_mid = LowpassFilter(
            order=mono_order, cutoff=LR_HIGH_SUPER_SPLIT_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_mid_q = LowpassFilter(
            order=mono_order, cutoff=LR_HIGH_SUPER_SPLIT_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.lp_lr_low_q = LowpassFilter(
            order=mono_order, cutoff=LR_HIGH_SPLIT_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.bp_pilot = BandpassFilter(
            order=pilot_order, lowcut=PILOT_BANDPASS_LOW,
            highcut=PILOT_BANDPASS_HIGH, sample_rate=self.composite_rate,
        )
        self.bp_pilot_noise_1 = BandpassFilter(
            order=pilot_order, lowcut=PILOT_NOISE_BAND1_LOW,
            highcut=PILOT_NOISE_BAND1_HIGH, sample_rate=self.composite_rate,
        )
        self.bp_pilot_noise_2 = BandpassFilter(
            order=pilot_order, lowcut=PILOT_NOISE_BAND2_LOW,
            highcut=PILOT_NOISE_BAND2_HIGH, sample_rate=self.composite_rate,
        )
        self.bp_lr = BandpassFilter(
            order=lr_order, lowcut=LR_BANDPASS_LOW,
            highcut=LR_BANDPASS_HIGH, sample_rate=self.composite_rate,
        )

        # --- De-emphasis ---
        self.deemph_left = DeemphasisIIRFilter(
            sample_rate=self.final_audio_rate, tau=DEEMPHASIS_TAU,
        )
        self.deemph_right = DeemphasisIIRFilter(
            sample_rate=self.final_audio_rate, tau=DEEMPHASIS_TAU,
        )

        # --- Pilot tone notch filter (19 kHz removal) ---
        self.notch_pilot_l = NotchFilter(
            freq=PILOT_NOTCH_FREQ, Q=PILOT_NOTCH_Q,
            sample_rate=self.composite_rate,
        )
        self.notch_pilot_l2 = NotchFilter(
            freq=PILOT_NOTCH_FREQ, Q=PILOT_NOTCH_Q,
            sample_rate=self.composite_rate,
        )
        self.notch_pilot_r = NotchFilter(
            freq=PILOT_NOTCH_FREQ, Q=PILOT_NOTCH_Q,
            sample_rate=self.composite_rate,
        )
        self.notch_pilot_r2 = NotchFilter(
            freq=PILOT_NOTCH_FREQ, Q=PILOT_NOTCH_Q,
            sample_rate=self.composite_rate,
        )

        # --- DC offset tracking ---
        self.dc_offset = 0.0
        self.dc_alpha = DC_OFFSET_ALPHA

        # --- Adaptive stereo blend ---
        # blend_factor: 1.0 = full stereo, 0.0 = full mono
        self.blend_factor: float = 1.0
        self.force_blend_factor: float | None = None
        self.pilot_snr_ema: float | None = None
        self.pilot_jitter_ema: float = 0.0
        self.stereo_phase_err_ema: float = 0.0
        self.mono_lr_phase_align_ema: float = 0.0
        self.lr_high_gate_gain: float = 1.0
        self.pilot_phase_mode: str = str(STEREO_PILOT_PHASE_MODE).strip().lower()
        if self.pilot_phase_mode not in ("classic", "residual", "hilbert"):
            self.pilot_phase_mode = "classic"
        self.pilot_residual_center_hz: float = float(STEREO_PILOT_RESIDUAL_CENTER_HZ)
        self.subcarrier_phase_offset_rad: float = np.deg2rad(STEREO_SUBCARRIER_PHASE_OFFSET_DEG)
        self.mono_delay_samples: int = max(0, int(STEREO_MONO_DELAY_SAMPLES))
        self._mono_delay_state: np.ndarray = np.zeros(self.mono_delay_samples, dtype=np.float32)
        self._pilot_phase_last: float | None = None
        self._pilot_mix_phase: float = 0.0
        self.lr_side_cap_gain: float = 1.0
        self.phase_align_enabled: bool = STEREO_PHASE_ALIGN_ENABLE
        self.iq_phase_correction_enabled: bool = STEREO_IQ_PHASE_CORRECTION_ENABLE
        self.legacy_lr_demod_enabled: bool = STEREO_LEGACY_LR_DEMOD_ENABLE
        self.high_gate_enabled: bool = True
        self.diag_enable: bool = STEREO_DIAG_ENABLE
        self.diag_log_interval_blocks: int = STEREO_DIAG_LOG_INTERVAL_BLOCKS
        self._diag_counter: int = 0

        # --- Resample ratios ---
        # IQ sample rate -> composite rate
        ratio = Fraction(
            int(self.composite_rate), int(self.iq_sample_rate),
        ).limit_denominator()
        self.up = ratio.numerator
        self.down = ratio.denominator
        # Composite rate -> final audio rate
        self._resample_up = 1
        self._resample_down = max(1, int(self.composite_rate / self.final_audio_rate))

    # ------------------------------------------------------------------
    # Shared demodulation pipeline
    # ------------------------------------------------------------------

    def demodulate(self, composite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate stereo or mono audio signals from the composite signal.

        Args:
            composite (ndarray): Composite signal.

        Returns:
            tuple: (left_channel, right_channel)
        """
        if self.stereo:
            return self._demodulate_stereo(composite)
        else:
            return self._demodulate_mono(composite)

    def _estimate_pilot_phase(self, pilot_complex: np.ndarray) -> np.ndarray:
        """Estimate pilot phase from analytic pilot signal."""
        mode = self.pilot_phase_mode
        if mode == "classic":
            pilot_phase = self.pilot_pll.process(pilot_complex).astype(np.float64, copy=False)
            if pilot_phase.size:
                self._pilot_phase_last = float(pilot_phase[-1])
            return pilot_phase

        if mode == "residual":
            n = np.arange(pilot_complex.size, dtype=np.float64)
            w0 = 2.0 * np.pi * self.pilot_residual_center_hz / self.composite_rate
            mix_phase = self._pilot_mix_phase + w0 * n
            mix_phase_wrapped = np.mod(mix_phase, 2.0 * np.pi)
            residual_in = pilot_complex * np.exp(-1j * mix_phase_wrapped)
            residual_phase = self.pilot_pll.process(
                np.asarray(residual_in, dtype=np.complex64)
            ).astype(np.float64, copy=False)
            pilot_phase = residual_phase + mix_phase
            self._pilot_mix_phase = float(
                np.mod(self._pilot_mix_phase + w0 * pilot_complex.size, 2.0 * np.pi)
            )
            if pilot_phase.size:
                if self._pilot_phase_last is None:
                    pilot_phase = np.unwrap(pilot_phase)
                else:
                    pilot_phase = np.unwrap(
                        np.concatenate(([self._pilot_phase_last], pilot_phase))
                    )[1:]
                self._pilot_phase_last = float(pilot_phase[-1])
            return pilot_phase

        raw_phase = np.angle(
            np.asarray(pilot_complex, dtype=np.complex64)
        ).astype(np.float64, copy=False)
        if raw_phase.size == 0:
            return raw_phase
        if self._pilot_phase_last is None:
            pilot_phase = np.unwrap(raw_phase)
        else:
            pilot_phase = np.unwrap(np.concatenate(([self._pilot_phase_last], raw_phase)))[1:]
        self._pilot_phase_last = float(pilot_phase[-1])
        return pilot_phase

    def _apply_mono_delay(self, mono: np.ndarray) -> np.ndarray:
        """Delay mono path to compensate LR path group delay."""
        delay = self.mono_delay_samples
        mono_f32 = np.asarray(mono, dtype=np.float32)
        if delay <= 0:
            return mono_f32.astype(np.float64, copy=False)
        if self._mono_delay_state.size != delay:
            self._mono_delay_state = np.zeros(delay, dtype=np.float32)
        stacked = np.concatenate((self._mono_delay_state, mono_f32))
        out = stacked[:mono_f32.size]
        self._mono_delay_state = stacked[mono_f32.size:]
        return out.astype(np.float64, copy=False)

    def _demodulate_stereo(self, composite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stereo demodulation with adaptive blend.

        When the 19 kHz pilot signal is strong the output is full stereo.
        As the pilot weakens (weak signal / multipath) the output smoothly
        transitions toward mono, improving the signal-to-noise ratio.

        Process flow:
          1. Extract mono signal via lowpass filter.
          2. Extract pilot signal using bandpass filter and Hilbert transform.
          3. Measure pilot SNR/stability and update blend factor (EMA-smoothed).
          4. Generate subcarrier from pilot phase.
          5. Extract and baseband LR signal, scaled by blend factor.
          6. Combine channels (L = mono + LR, R = mono - LR).
          7. Resample and apply de-emphasis.

        Returns:
            tuple: (left_channel_audio, right_channel_audio)
        """
        mono_raw = self.lp_mono.apply(composite)
        mono = self._apply_mono_delay(mono_raw)
        pilot_signal = self.bp_pilot.apply(composite)
        pilot_complex = signal.hilbert(pilot_signal.astype(np.float32))
        pilot_phase = self._estimate_pilot_phase(pilot_complex)

        # --- Adaptive stereo blend based on pilot SNR ---
        pilot_power = float(np.mean(pilot_signal ** 2))
        pilot_noise_1 = self.bp_pilot_noise_1.apply(composite)
        pilot_noise_2 = self.bp_pilot_noise_2.apply(composite)
        noise_power_1 = float(np.mean(pilot_noise_1 ** 2))
        noise_power_2 = float(np.mean(pilot_noise_2 ** 2))
        noise_power = 0.5 * (noise_power_1 + noise_power_2)
        snr_db = 10.0 * np.log10((pilot_power + 1e-12) / (noise_power + 1e-12))
        if self.pilot_snr_ema is None:
            self.pilot_snr_ema = snr_db
        snr_alpha = STEREO_BLEND_PILOT_SNR_EMA_ALPHA
        self.pilot_snr_ema = snr_alpha * snr_db + (1.0 - snr_alpha) * self.pilot_snr_ema
        snr_jitter = abs(snr_db - self.pilot_snr_ema)
        jitter_alpha = STEREO_BLEND_PILOT_JITTER_EMA_ALPHA
        self.pilot_jitter_ema = (
            jitter_alpha * snr_jitter + (1.0 - jitter_alpha) * self.pilot_jitter_ema
        )

        snr_for_blend = self.pilot_snr_ema if self.pilot_snr_ema is not None else snr_db
        lo = STEREO_BLEND_PILOT_SNR_DB_LO
        hi = STEREO_BLEND_PILOT_SNR_DB_HI
        if snr_for_blend >= hi:
            snr_score = 1.0
        elif snr_for_blend <= lo:
            snr_score = 0.0
        else:
            snr_score = (snr_for_blend - lo) / (hi - lo)
        jitter_ref = max(STEREO_BLEND_PILOT_JITTER_REF_DB, 1e-6)
        stability = np.clip(1.0 - (self.pilot_jitter_ema / jitter_ref), 0.0, 1.0)
        stability_min_factor = STEREO_BLEND_STABILITY_MIN_FACTOR
        if self.pilot_phase_mode == "residual":
            stability_min_factor = max(
                stability_min_factor, STEREO_BLEND_STABILITY_MIN_FACTOR_RESIDUAL,
            )
        stability_factor = (
            stability_min_factor
            + (1.0 - stability_min_factor) * stability
        )
        target = snr_score * stability_factor
        # EMA smoothing to avoid abrupt transitions
        alpha = STEREO_BLEND_SMOOTHING
        self.blend_factor = alpha * target + (1.0 - alpha) * self.blend_factor
        if self.force_blend_factor is not None:
            self.blend_factor = float(np.clip(self.force_blend_factor, 0.0, 1.0))

        sub_phase = 2.0 * pilot_phase + self.subcarrier_phase_offset_rad
        subcarrier_i = np.cos(sub_phase)
        subcarrier_q = np.sin(sub_phase)
        lr_band = self.bp_lr.apply(composite)
        if self.legacy_lr_demod_enabled:
            lr_demod_i = lr_band * subcarrier_i * STEREO_LR_DEMOD_GAIN
            self.stereo_phase_err_ema = 0.0
            lr_base_full = self.lp_lr_base.apply(lr_demod_i)
            lr_base_low = self.lp_lr_low.apply(lr_demod_i)
            lr_base_mid = self.lp_lr_mid.apply(lr_demod_i)
        else:
            lr_demod_i = lr_band * subcarrier_i * STEREO_LR_DEMOD_GAIN
            lr_demod_q = lr_band * subcarrier_q * STEREO_LR_DEMOD_GAIN
            lr_base_full_i = self.lp_lr_base.apply(lr_demod_i)
            lr_base_full_q = self.lp_lr_base_q.apply(lr_demod_q)
            cov_iq = float(np.mean(lr_base_full_i * lr_base_full_q))
            var_i = float(np.mean(lr_base_full_i ** 2))
            var_q = float(np.mean(lr_base_full_q ** 2))
            if self.iq_phase_correction_enabled:
                phase_err_raw = 0.5 * np.arctan2(2.0 * cov_iq, var_i - var_q + 1e-12)
                phase_lim = np.deg2rad(STEREO_PHASE_ERR_LIMIT_DEG)
                phase_err_raw = float(np.clip(phase_err_raw, -phase_lim, phase_lim))
                phase_alpha = STEREO_PHASE_ERR_SMOOTHING
                self.stereo_phase_err_ema = (
                    phase_alpha * phase_err_raw + (1.0 - phase_alpha) * self.stereo_phase_err_ema
                )
                cph = np.cos(self.stereo_phase_err_ema)
                sph = np.sin(self.stereo_phase_err_ema)
            else:
                self.stereo_phase_err_ema = 0.0
                cph = 1.0
                sph = 0.0
            lr_base_full = lr_base_full_i * cph + lr_base_full_q * sph

            lr_base_low_i = self.lp_lr_low.apply(lr_demod_i)
            lr_base_low_q = self.lp_lr_low_q.apply(lr_demod_q)
            lr_base_low = lr_base_low_i * cph + lr_base_low_q * sph
            lr_base_mid_i = self.lp_lr_mid.apply(lr_demod_i)
            lr_base_mid_q = self.lp_lr_mid_q.apply(lr_demod_q)
            lr_base_mid = lr_base_mid_i * cph + lr_base_mid_q * sph
        lr_base_midhigh = lr_base_mid - lr_base_low
        lr_base_super = lr_base_full - lr_base_mid

        gate_assist = 0.0
        gate_floor = LR_HIGH_GATE_MIN_GAIN
        if self.high_gate_enabled:
            high_rms = float(np.sqrt(np.mean(lr_base_super ** 2) + 1e-12))
            gate_thr = LR_HIGH_GATE_THRESHOLD
            gate_knee = max(gate_thr * LR_HIGH_GATE_KNEE_MULT, gate_thr + 1e-12)
            if high_rms <= gate_thr:
                gate_target = LR_HIGH_GATE_MIN_GAIN
            elif high_rms >= gate_knee:
                gate_target = 1.0
            else:
                t = (high_rms - gate_thr) / (gate_knee - gate_thr)
                gate_target = LR_HIGH_GATE_MIN_GAIN + (1.0 - LR_HIGH_GATE_MIN_GAIN) * t
            if STEREO_HIGH_GATE_SNR_ASSIST_ENABLE and self.pilot_snr_ema is not None:
                s_lo = STEREO_HIGH_GATE_SNR_ASSIST_DB_LO
                s_hi = max(STEREO_HIGH_GATE_SNR_ASSIST_DB_HI, s_lo + 1e-6)
                if self.pilot_snr_ema <= s_lo:
                    snr_open = 0.0
                elif self.pilot_snr_ema >= s_hi:
                    snr_open = 1.0
                else:
                    snr_open = (
                        (self.pilot_snr_ema - s_lo) / (s_hi - s_lo)
                    )
                gate_assist = snr_open * STEREO_HIGH_GATE_SNR_ASSIST_MAX
                gate_assist = float(np.clip(gate_assist, 0.0, 1.0))
                gate_floor = LR_HIGH_GATE_MIN_GAIN + (
                    np.clip(snr_open, 0.0, 1.0) * STEREO_HIGH_GATE_SNR_FLOOR_BOOST_MAX
                )
                gate_floor = float(np.clip(gate_floor, LR_HIGH_GATE_MIN_GAIN, 1.0))
                gate_target = max(gate_target, gate_floor)
                gate_target = gate_target + gate_assist * (1.0 - gate_target)
            gate_alpha = LR_HIGH_GATE_SMOOTHING
            self.lr_high_gate_gain = (
                gate_alpha * gate_target + (1.0 - gate_alpha) * self.lr_high_gate_gain
            )
        else:
            self.lr_high_gate_gain = 1.0

        midhigh_gain = LR_HIGH_MIN_GAIN + (1.0 - LR_HIGH_MIN_GAIN) * self.blend_factor
        super_gain = LR_SUPER_HIGH_MIN_GAIN + (1.0 - LR_SUPER_HIGH_MIN_GAIN) * self.blend_factor
        lr_shaped = (
            lr_base_low
            + midhigh_gain * lr_base_midhigh
            + (super_gain * self.lr_high_gate_gain) * lr_base_super
        )

        # Align LR phase to mono when coherence is high (helps separation).
        mono_align = self.lp_mono_align.apply(mono)
        mono_a = signal.hilbert(mono_align.astype(np.float32))
        lr_a = signal.hilbert(lr_base_low.astype(np.float32))
        cross = np.mean(lr_a * np.conj(mono_a))
        p_mono = float(np.mean(np.abs(mono_a) ** 2))
        p_lr = float(np.mean(np.abs(lr_a) ** 2))
        coh = abs(cross) / (np.sqrt(p_mono * p_lr) + 1e-12)
        side_ratio_align = float(np.sqrt(p_lr + 1e-12) / np.sqrt(p_mono + 1e-12))
        align_trusted = (
            coh >= STEREO_MONO_LR_PHASE_ALIGN_COH_MIN
            and side_ratio_align >= STEREO_MONO_LR_PHASE_ALIGN_SIDE_RATIO_MIN
            and side_ratio_align <= STEREO_MONO_LR_PHASE_ALIGN_SIDE_RATIO_MAX
        )
        if self.phase_align_enabled:
            if align_trusted:
                align_raw = float(np.angle(cross))
                align_lim = np.deg2rad(STEREO_MONO_LR_PHASE_ALIGN_LIMIT_DEG)
                align_raw = float(np.clip(align_raw, -align_lim, align_lim))
                align_alpha = STEREO_MONO_LR_PHASE_ALIGN_SMOOTHING
                self.mono_lr_phase_align_ema = (
                    align_alpha * align_raw
                    + (1.0 - align_alpha) * self.mono_lr_phase_align_ema
                )
            else:
                decay = np.clip(STEREO_MONO_LR_PHASE_ALIGN_DECAY, 0.0, 1.0)
                self.mono_lr_phase_align_ema *= (1.0 - decay)
            lr_shaped_a = signal.hilbert(lr_shaped.astype(np.float32))
            lr_shaped = np.real(
                lr_shaped_a * np.exp(-1j * self.mono_lr_phase_align_ema)
            ).astype(np.float64, copy=False)

        lr_baseband = lr_shaped * self.blend_factor
        side_ratio_now = float(
            np.sqrt(np.mean(lr_baseband ** 2) + 1e-12) / np.sqrt(np.mean(mono ** 2) + 1e-12)
        )
        if STEREO_LR_SIDE_RATIO_CAP_ENABLE:
            cap_target = max(STEREO_LR_SIDE_RATIO_CAP_TARGET, 1e-6)
            cap_min_gain = float(np.clip(STEREO_LR_SIDE_RATIO_CAP_MIN_GAIN, 0.0, 1.0))
            cap_gain_target = min(1.0, cap_target / max(side_ratio_now, 1e-12))
            cap_gain_target = max(cap_min_gain, cap_gain_target)
            cap_attack = float(np.clip(STEREO_LR_SIDE_RATIO_CAP_ATTACK, 0.0, 1.0))
            cap_release = float(np.clip(STEREO_LR_SIDE_RATIO_CAP_RELEASE, 0.0, 1.0))
            cap_alpha = cap_attack if cap_gain_target < self.lr_side_cap_gain else cap_release
            self.lr_side_cap_gain = (
                cap_alpha * cap_gain_target + (1.0 - cap_alpha) * self.lr_side_cap_gain
            )
            self.lr_side_cap_gain = float(np.clip(self.lr_side_cap_gain, cap_min_gain, 1.0))
            lr_baseband = lr_baseband * self.lr_side_cap_gain
        else:
            self.lr_side_cap_gain = 1.0

        left_channel = mono + lr_baseband
        right_channel = mono - lr_baseband

        if self.diag_enable:
            self._diag_counter += 1
            if self._diag_counter % max(1, self.diag_log_interval_blocks) == 0:
                phase_step = np.diff(np.unwrap(pilot_phase.astype(np.float64)))
                phase_jitter = float(np.std(phase_step - np.mean(phase_step))) if phase_step.size else 0.0
                self.logger.info(
                    "StereoDiag snr=%.2fdB blend=%.3f pilotP=%.6g noiseP=%.6g "
                    "phJit=%.6g lrBandRMS=%.6g lrBaseRMS=%.6g phaseIQ=%.3fdeg "
                    "monoLRAlign=%.3fdeg coh=%.3f side=%.3f align=%s iqCorr=%s legacyLR=%s highGate=%s "
                    "gateAssist=%.3f gateFloor=%.3f pilotMode=%s "
                    "monoDelay=%d scOff=%.1fdeg sideCap=%.3f",
                    snr_db, self.blend_factor, pilot_power, noise_power,
                    phase_jitter,
                    float(np.sqrt(np.mean(lr_band ** 2) + 1e-12)),
                    float(np.sqrt(np.mean(lr_baseband ** 2) + 1e-12)),
                    float(np.rad2deg(self.stereo_phase_err_ema)),
                    float(np.rad2deg(self.mono_lr_phase_align_ema)),
                    float(coh),
                    float(side_ratio_align),
                    "on" if self.phase_align_enabled else "off",
                    "on" if self.iq_phase_correction_enabled else "off",
                    "on" if self.legacy_lr_demod_enabled else "off",
                    "on" if self.high_gate_enabled else "off",
                    gate_assist,
                    gate_floor,
                    self.pilot_phase_mode,
                    self.mono_delay_samples,
                    float(np.rad2deg(self.subcarrier_phase_offset_rad)),
                    self.lr_side_cap_gain,
                )

        # Remove 19 kHz pilot tone leakage
        left_channel = self.notch_pilot_l.apply(left_channel)
        left_channel = self.notch_pilot_l2.apply(left_channel)
        right_channel = self.notch_pilot_r.apply(right_channel)
        right_channel = self.notch_pilot_r2.apply(right_channel)

        # Resample composite -> final audio
        left_48 = signal.resample_poly(
            left_channel.astype(np.float32),
            self._resample_up, self._resample_down,
        )
        right_48 = signal.resample_poly(
            right_channel.astype(np.float32),
            self._resample_up, self._resample_down,
        )
        left_48 = self.deemph_left.process(left_48)
        right_48 = self.deemph_right.process(right_48)
        return left_48, right_48

    def _demodulate_mono(self, composite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mono demodulation process.

        Args:
            composite (ndarray): Composite signal.

        Returns:
            tuple: (mono_channel, mono_channel) where both channels are identical.
        """
        mono = self.lp_mono.apply(composite)
        mono = self.notch_pilot_l.apply(mono)
        mono = self.notch_pilot_l2.apply(mono)
        mono_48 = signal.resample_poly(
            mono.astype(np.float32),
            self._resample_up, self._resample_down,
        )
        mono_48 = self.deemph_left.process(mono_48)
        return mono_48, mono_48

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset shared state and delegate to subclass."""
        self.pilot_pll.reset()
        self.dc_offset = 0.0
        self.blend_factor = 1.0
        self.pilot_snr_ema = None
        self.pilot_jitter_ema = 0.0
        self.stereo_phase_err_ema = 0.0
        self.mono_lr_phase_align_ema = 0.0
        self.lr_high_gate_gain = 1.0
        self.lr_side_cap_gain = 1.0
        if self.mono_delay_samples > 0:
            self._mono_delay_state = np.zeros(self.mono_delay_samples, dtype=np.float32)
        self._pilot_mix_phase = 0.0
        self._pilot_phase_last = None
        self._diag_counter = 0
        self._reset_subclass()

    @abstractmethod
    def _reset_subclass(self) -> None:
        """Reset subclass-specific state (called by ``reset()``)."""


# ======================================================================
# Concrete implementations
# ======================================================================

class FMDemodulator(BaseFMDemodulator):
    """Standard FM demodulator (PLL-based).

    Processes IQ samples through a PLL, IQ lowpass filter, resampling,
    stereo separation, and de-emphasis to generate the demodulated signal.
    """

    def __init__(
        self,
        iq_sample_rate: float = SDR_SAMPLE_RATE,
        composite_rate: float = COMPOSITE_RATE,
        final_audio_rate: float = AUDIO_OUTPUT_RATE,
        stereo: bool = True,
    ) -> None:
        super().__init__(
            iq_sample_rate=iq_sample_rate,
            composite_rate=composite_rate,
            final_audio_rate=final_audio_rate,
            stereo=stereo,
            mono_order=MONO_LOWPASS_ORDER,
            pilot_order=PILOT_BANDPASS_ORDER,
            lr_order=LR_BANDPASS_ORDER,
            logger_name='fm_receiver.FMDemodulator',
        )

        self.logger.info(
            f"Initializing FMDemodulator: IQ={iq_sample_rate/1e6:.3f}MHz, "
            f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
            f"Stereo={'enabled' if stereo else 'disabled'}"
        )

        # --- Standard-only: main PLL + IQ lowpass filter ---
        self.main_pll: PLL = PLL(Kp=MAIN_PLL_KP, Ki=MAIN_PLL_KI, return_phase=False)
        nyquist = self.iq_sample_rate / 2.0
        self.iq_b, self.iq_a = signal.butter(
            IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF / nyquist, btype="low",
        )

    def process_iq_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """Apply DC offset correction, lowpass filtering, PLL demodulation,
        and resampling to generate the composite signal.

        Args:
            iq_samples: Input IQ samples.

        Returns:
            Composite signal after resampling.
        """
        try:
            self.dc_offset = (
                self.dc_alpha * np.mean(iq_samples)
                + (1 - self.dc_alpha) * self.dc_offset
            )
            iq_processed = iq_samples - self.dc_offset
            iq_processed = np.asarray(iq_processed, dtype=np.complex64, copy=False)
            iq_filtered = signal.lfilter(self.iq_b, self.iq_a, iq_processed)
            main_output = self.main_pll.process(iq_filtered)
            composite = signal.resample_poly(
                main_output, up=self.up, down=self.down,
                window=("kaiser", STANDARD_RESAMPLE_KAISER_BETA),
            )
            return composite.astype(np.float32, copy=False)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing IQ samples: {e}", exc_info=True)
            raise DemodulationError(f"Error processing IQ samples: {e}") from e

    def _reset_subclass(self) -> None:
        """Reset main PLL state."""
        self.main_pll.reset()


class FMDemodulatorLight(BaseFMDemodulator):
    """Light FM demodulator (phase-differentiation-based).

    Implements a simplified process to reduce computational load.
    Uses phase differentiation for FM demodulation instead of a full PLL.
    """

    def __init__(
        self,
        iq_sample_rate: float = SDR_SAMPLE_RATE_LIGHT,
        composite_rate: float = COMPOSITE_RATE,
        final_audio_rate: float = AUDIO_OUTPUT_RATE,
        stereo: bool = True,
    ) -> None:
        super().__init__(
            iq_sample_rate=iq_sample_rate,
            composite_rate=composite_rate,
            final_audio_rate=final_audio_rate,
            stereo=stereo,
            mono_order=MONO_LOWPASS_ORDER_LIGHT,
            pilot_order=PILOT_BANDPASS_ORDER_LIGHT,
            lr_order=LR_BANDPASS_ORDER_LIGHT,
            logger_name='fm_receiver.FMDemodulatorLight',
        )

        self.logger.info(
            f"Initializing FMDemodulatorLight: IQ={iq_sample_rate/1e6:.3f}MHz, "
            f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
            f"Stereo={'enabled' if stereo else 'disabled'}"
        )

        # --- Light-only: phase tracking for differentiation ---
        self.last_phase: float | None = None

    def process_iq_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """Apply DC offset correction, phase extraction/differentiation,
        and resampling to generate the composite signal.

        Args:
            iq_samples: Input IQ samples.

        Returns:
            Composite signal after resampling.
        """
        try:
            self.dc_offset = (
                self.dc_alpha * np.mean(iq_samples)
                + (1 - self.dc_alpha) * self.dc_offset
            )
            iq_processed = iq_samples - self.dc_offset
            current_phase = np.angle(iq_processed)
            if self.last_phase is None:
                phase = np.unwrap(current_phase)
            else:
                phase = np.unwrap(np.concatenate(([self.last_phase], current_phase)))[1:]
            fm_demod = np.diff(phase, prepend=phase[0])
            self.last_phase = phase[-1]
            composite = (
                signal.resample_poly(fm_demod, up=self.up, down=self.down)
                * LIGHT_COMPOSITE_SCALE
            )
            return np.asarray(composite, dtype=np.float32, copy=False)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing IQ samples (Light): {e}", exc_info=True)
            raise DemodulationError(f"Error processing IQ samples (Light): {e}") from e

    def _reset_subclass(self) -> None:
        """Reset phase tracking state."""
        self.last_phase = None
