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
from fm_radio.filters import (
    LowpassFilter, BandpassFilter, NotchFilter, DeemphasisIIRFilter,
    StatefulResampler, SideNoiseReducer, StreamAligner,
)
from fm_radio.pll import PLL
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_SAMPLE_RATE_LIGHT,
    MAIN_DEMOD_USE_PLL,
    MAIN_PLL_KP, MAIN_PLL_KI, PILOT_PLL_KP, PILOT_PLL_KI,
    IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF,
    MONO_LOWPASS_ORDER, MONO_LOWPASS_ORDER_LIGHT, MONO_LOWPASS_CUTOFF,
    LR_BASE_LOWPASS_CUTOFF, LR_HIGH_SPLIT_CUTOFF, LR_HIGH_SUPER_SPLIT_CUTOFF,
    LR_HIGH_MIN_GAIN, LR_HIGH_MAX_GAIN,
    LR_SUPER_HIGH_MIN_GAIN, LR_SUPER_HIGH_MAX_GAIN,
    PILOT_BANDPASS_ORDER, PILOT_BANDPASS_ORDER_LIGHT,
    PILOT_BANDPASS_LOW, PILOT_BANDPASS_HIGH,
    STEREO_PILOT_RESIDUAL_CENTER_HZ,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG,
    HARDWARE_SUBCARRIER_PHASE_TRIM_DEG,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT,
    STEREO_MONO_DELAY_SAMPLES,
    STEREO_LR_SIDE_RATIO_CAP_ENABLE, STEREO_LR_SIDE_RATIO_CAP_TARGET,
    STEREO_LR_SIDE_RATIO_CAP_MIN_GAIN,
    STEREO_LR_SIDE_RATIO_CAP_ATTACK, STEREO_LR_SIDE_RATIO_CAP_RELEASE,
    STEREO_PHASE_ERR_SMOOTHING, STEREO_PHASE_ANISO_GATE,
    STEREO_PHASE_SIDE_GATE_DB, STEREO_PHASE_ACQUIRE_BLOCKS,
    STEREO_PHASE_CONF_ANISO, STEREO_PHASE_BRANCH_CONF,
    STEREO_PHASE_SIDE_OVER_NOISE_DB, STEREO_PHASE_NOISE_CONF_RAMP_DB,
    STEREO_PHASE_LEAK_DEG_PER_SEC,
    STEREO_IQ_PHASE_CORRECTION_ENABLE,
    PILOT_NOISE_BAND1_LOW, PILOT_NOISE_BAND1_HIGH,
    PILOT_NOISE_BAND2_LOW, PILOT_NOISE_BAND2_HIGH,
    LR_BANDPASS_ORDER, LR_BANDPASS_ORDER_LIGHT,
    LR_BANDPASS_LOW, LR_BANDPASS_HIGH,
    STEREO_LR_DEMOD_GAIN,
    STEREO_DIAG_ENABLE, STEREO_DIAG_LOG_INTERVAL_BLOCKS,
    DEEMPHASIS_TAU, DC_OFFSET_ALPHA,
    AUDIO_OUTPUT_RATE, COMPOSITE_RATE, LIGHT_COMPOSITE_SCALE,
    STANDARD_RESAMPLE_KAISER_BETA,
    STEREO_BLEND_PILOT_SNR_DB_HI, STEREO_BLEND_PILOT_SNR_DB_LO,
    STEREO_BLEND_PILOT_SNR_EMA_ALPHA,
    STEREO_BLEND_PILOT_JITTER_EMA_ALPHA,
    STEREO_BLEND_PILOT_JITTER_REF_DB,
    STEREO_BLEND_STABILITY_MIN_FACTOR,
    STEREO_BLEND_SMOOTHING,
    STEREO_HF_BLEND_PILOT_SNR_DB_HI, STEREO_HF_BLEND_PILOT_SNR_DB_LO,
    PILOT_NOTCH_FREQ, PILOT_NOTCH_Q,
    SIDE_NR_ENABLE, SIDE_NR_FRAME, SIDE_NR_HOP,
    SIDE_NR_ALPHA_FLOOR, SIDE_NR_BETA,
    SIDE_NR_NOISE_DECAY_DB_PER_SEC,
    SIDE_NR_TONE_PROTECT_DB, SIDE_NR_TONE_PROTECT_MED_BINS,
    SIDE_NR_LO_HZ, SIDE_NR_HI_HZ,
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
                 logger_name: str,
                 subcarrier_phase_offset_deg: float = STEREO_SUBCARRIER_PHASE_OFFSET_DEG):
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
        # --- Analytic pilot extraction (heterodyne + lowpass) ---
        # The pilot is extracted by mixing the composite down by the
        # pilot centre frequency with a phase-continuous carrier, then
        # lowpassing the complex baseband with carried filter state.
        # This replaces the previous real bandpass + per-block FFT
        # Hilbert: signal.hilbert assumes a periodic block, so any
        # carrier offset made the pilot non-periodic in the block and
        # produced phase errors of up to ~12 deg at every block edge
        # (x2 at the 38 kHz subcarrier).  The heterodyne path is
        # stateful end to end and has no block-boundary artefacts.
        # Cutoff = half the old bandpass width keeps the same
        # equivalent noise bandwidth, so pilot SNR scaling and the
        # tuned blend thresholds are preserved.
        pilot_lp_cutoff = 0.5 * (PILOT_BANDPASS_HIGH - PILOT_BANDPASS_LOW)
        self.pilot_lp_sos: np.ndarray = signal.butter(
            pilot_order, pilot_lp_cutoff / (self.composite_rate / 2.0),
            btype="low", output="sos",
        )
        self._pilot_lp_zi: np.ndarray = np.zeros(
            (self.pilot_lp_sos.shape[0], 2), dtype=np.complex128,
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
        self.stereo_phase_aniso: float = 0.0
        self._phase_acquired: bool = False
        self._phase_acq_acc: complex = 0j
        self._phase_acq_count: int = 0
        self._phase_conf: float = 0.0
        self.stereo_phase_side_over_noise_db: float = 0.0
        self.pilot_residual_center_hz: float = float(STEREO_PILOT_RESIDUAL_CENTER_HZ)
        # The per-variant offset constants are DSP-intrinsic (tuned on
        # synthetic IQ, which never passes through the tuner).  Real
        # hardware adds the front-end's 19k/38k phase characteristic on
        # top, so the hardware trim is applied here for every variant.
        # Synthetic paths in quality_selftest override
        # subcarrier_phase_offset_rad directly with the DSP value.
        self.subcarrier_phase_offset_rad: float = np.deg2rad(
            subcarrier_phase_offset_deg + HARDWARE_SUBCARRIER_PHASE_TRIM_DEG
        )
        self.mono_delay_samples: int = max(0, int(STEREO_MONO_DELAY_SAMPLES))
        self._mono_delay_state: np.ndarray = np.zeros(self.mono_delay_samples, dtype=np.float32)
        self._pilot_phase_last: float | None = None
        self._pilot_mix_phase: float = 0.0
        self.lr_side_cap_gain: float = 1.0
        self.iq_phase_correction_enabled: bool = STEREO_IQ_PHASE_CORRECTION_ENABLE
        # HF L-R band gain ceilings (apply additional damping at full blend
        # to trade stereo width above 7 kHz for HF noise reduction).
        self.lr_high_max_gain: float = float(LR_HIGH_MAX_GAIN)
        self.lr_super_high_max_gain: float = float(LR_SUPER_HIGH_MAX_GAIN)

        # --- Side-channel STFT noise reduction (post de-emphasis) ---
        self.side_nr_enabled: bool = bool(SIDE_NR_ENABLE)
        self.side_nr = SideNoiseReducer(
            sample_rate=self.final_audio_rate,
            frame=int(SIDE_NR_FRAME),
            hop=int(SIDE_NR_HOP),
            alpha_floor=float(SIDE_NR_ALPHA_FLOOR),
            beta=float(SIDE_NR_BETA),
            noise_decay_db_per_sec=float(SIDE_NR_NOISE_DECAY_DB_PER_SEC),
            tone_protect_db=float(SIDE_NR_TONE_PROTECT_DB),
            tone_protect_med_bins=int(SIDE_NR_TONE_PROTECT_MED_BINS),
            lo_hz=float(SIDE_NR_LO_HZ),
            hi_hz=float(SIDE_NR_HI_HZ),
        )
        self.side_nr_mid_aligner = StreamAligner()
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
        # Composite -> audio decimators with carried state.  The
        # previous per-block stateless resample_poly zero-padded both
        # edges of every 16 ms block (last member of the block-transient
        # bug family fixed in PR #4 / PR #9).  StatefulResampler's
        # alignment precondition (block sizes multiple of down) is
        # guaranteed by the IQ resampler's emit_align=_resample_down.
        # The left instance also serves the mono path so mono<->stereo
        # switches stay continuous on the primary channel.
        self._audio_resampler_l = StatefulResampler(
            self._resample_up, self._resample_down,
        )
        self._audio_resampler_r = StatefulResampler(
            self._resample_up, self._resample_down,
        )

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

    def _estimate_pilot_phase(
        self, composite: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate pilot phase via heterodyne + stateful lowpass.

        The real composite is mixed down by the pilot centre frequency
        (phase-continuous across blocks via ``_pilot_mix_phase``) and
        lowpassed with carried state, yielding the complex pilot
        residual directly for the PLL.  For a pilot of amplitude A the
        residual amplitude is A/2 (the mix splits the real cosine into
        two lines and the lowpass keeps one).

        Returns:
            Tuple of (pilot_phase, pilot_residual).
        """
        n = np.arange(composite.size, dtype=np.float64)
        w0 = 2.0 * np.pi * self.pilot_residual_center_hz / self.composite_rate
        mix_phase = self._pilot_mix_phase + w0 * n
        mix_phase_wrapped = np.mod(mix_phase, 2.0 * np.pi)
        mixed = composite * np.exp(-1j * mix_phase_wrapped)
        residual_in, self._pilot_lp_zi = signal.sosfilt(
            self.pilot_lp_sos, mixed, zi=self._pilot_lp_zi,
        )
        residual_phase = self.pilot_pll.process(
            residual_in.astype(np.complex64, copy=False)
        ).astype(np.float64, copy=False)
        pilot_phase = residual_phase + mix_phase
        self._pilot_mix_phase = float(
            np.mod(self._pilot_mix_phase + w0 * composite.size, 2.0 * np.pi)
        )
        if pilot_phase.size:
            if self._pilot_phase_last is None:
                pilot_phase = np.unwrap(pilot_phase)
            else:
                pilot_phase = np.unwrap(
                    np.concatenate(([self._pilot_phase_last], pilot_phase))
                )[1:]
            self._pilot_phase_last = float(pilot_phase[-1])
        return pilot_phase, residual_in

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
        pilot_phase, pilot_residual = self._estimate_pilot_phase(composite)

        # --- Adaptive stereo blend based on pilot SNR ---
        # The analytic residual has amplitude A/2 for a pilot of
        # amplitude A, so 2*mean(|residual|^2) equals the mean square
        # of the old real-bandpassed pilot (A^2/2), preserving the SNR
        # scale that the blend thresholds were tuned against.
        pilot_power = 2.0 * float(np.mean(np.abs(pilot_residual) ** 2))
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
        lr_demod_i = lr_band * subcarrier_i * STEREO_LR_DEMOD_GAIN
        lr_demod_q = lr_band * subcarrier_q * STEREO_LR_DEMOD_GAIN
        lr_base_full_i = self.lp_lr_base.apply(lr_demod_i)
        lr_base_full_q = self.lp_lr_base_q.apply(lr_demod_q)
        cov_iq = float(np.mean(lr_base_full_i * lr_base_full_q))
        var_i = float(np.mean(lr_base_full_i ** 2))
        var_q = float(np.mean(lr_base_full_q ** 2))
        if self.iq_phase_correction_enabled:
            # Anisotropy of the (I, Q) covariance: 1.0 for a perfectly
            # 1-D signal, ~0 for isotropic noise.  Gates the tracker so
            # mono programme (no side information) cannot random-walk
            # the angle across the 180-deg branch boundary.
            denom = var_i + var_q
            aniso = float(
                np.sqrt((var_i - var_q) ** 2 + 4.0 * cov_iq * cov_iq)
                / (denom + 1e-18)
            )
            self.stereo_phase_aniso = aniso
            # Absolute-energy gate: anisotropy is scale-invariant, so
            # on a MONO broadcast the tiny deterministic residue in the
            # side band (filter transients, pilot-harmonic products)
            # can look strongly 1-D while sitting ~-32 dB below the
            # mono programme.  Require the demodulated side power to be
            # within STEREO_PHASE_SIDE_GATE_DB of the mono power before
            # trusting the estimate (real stereo music: p5 = -11 dB;
            # noise-dominated side fails the anisotropy gate instead).
            mono_pow = float(np.mean(mono ** 2))
            side_gate = mono_pow * (10.0 ** (STEREO_PHASE_SIDE_GATE_DB / 10.0))
            # Side-over-noise gate: the discriminator's noise spectrum
            # rises as f^2, so the demodulated side band carries MORE
            # noise power than the mono band (during silence side/mono
            # measured +5 dB - the mono-relative gate inverts) and that
            # band noise is intrinsically ANISOTROPIC (the parabola is
            # asymmetric about 38 kHz), presenting a stable pseudo-axis
            # at aniso ~0.5 that overlaps genuine content.  The pilot
            # noise-band estimate predicts the side-band noise via a
            # chain constant: noise-only measures +16..17.5 dB
            # (synthetic) / +2..7 dB (hardware) above noise_power,
            # genuine stereo content +24.5 dB and up, so requiring
            # STEREO_PHASE_SIDE_OVER_NOISE_DB (21 dB) blocks the
            # pseudo-axis in both regimes while passing real content.
            noise_ref = noise_power * (
                10.0 ** (STEREO_PHASE_SIDE_OVER_NOISE_DB / 10.0)
            )
            self.stereo_phase_side_over_noise_db = float(
                10.0 * np.log10((denom + 1e-30) / (noise_power + 1e-30))
            )
            informative = (
                denom > 1e-18
                and mono_pow > 1e-18
                and denom >= side_gate
                and denom > noise_ref
                and aniso >= STEREO_PHASE_ANISO_GATE
            )
            if informative:
                beta_pa = 0.5 * np.arctan2(2.0 * cov_iq, var_i - var_q + 1e-12)
                if not self._phase_acquired:
                    # Acquisition: require STEREO_PHASE_ACQUIRE_BLOCKS
                    # CONSECUTIVE informative blocks (rejects start-up
                    # filter transients) and initialise from the
                    # doubled-angle circular mean 0.5*arg(sum(e^{j2b}))
                    # over the streak.  The doubled domain is invariant
                    # to the +-90 deg wrap of individual raw estimates:
                    # on a station whose true rotation sits near the
                    # boundary (the reference station is at ~-83 deg,
                    # with ~20% of raws wrapping to +88-ish), a single
                    # -block init would lock the wrong 180-deg branch
                    # (a permanent L/R swap) with that probability.
                    # The circular mean lands on the true axis and the
                    # (-90, 90] representative implements the cold-
                    # start convention |true rotation| < 90 deg - the
                    # only resolvable assumption, per the FM standard's
                    # pilot phase convention.
                    self._phase_acq_acc += complex(
                        np.cos(2.0 * beta_pa), np.sin(2.0 * beta_pa)
                    )
                    self._phase_acq_count += 1
                    if self._phase_acq_count >= STEREO_PHASE_ACQUIRE_BLOCKS:
                        self.stereo_phase_err_ema = float(
                            0.5 * np.arctan2(
                                self._phase_acq_acc.imag,
                                self._phase_acq_acc.real,
                            )
                        )
                        self._phase_acquired = True
                else:
                    # Tracking: the estimator is pi-periodic, so take
                    # the innovation to the NEAREST candidate in the
                    # {beta_pa + k*pi} family.  Continuity resolves the
                    # 180-deg branch, so corrections beyond +-90 deg
                    # stay locked instead of swapping L/R (the old
                    # clamp saturated at 75 deg and truncated the
                    # reference station's ~-83 deg demand).
                    #
                    # Confidence weighting: on weak / near-mono
                    # programme the gates pass MARGINAL blocks whose
                    # axis is content- and leakage-driven, and a field
                    # capture showed those walking the tracker ~74 deg
                    # and across the branch boundary (a mid-session
                    # L/R flip).  The innovation is scaled by
                    # w = (aniso - gate) / (conf - gate) clipped to
                    # [0, 1]: confident blocks (aniso >= 0.6, typical
                    # of genuine stereo, measured 0.75-0.89) track at
                    # full speed, marginal ones barely move the state.
                    w = (aniso - STEREO_PHASE_ANISO_GATE) / max(
                        STEREO_PHASE_CONF_ANISO - STEREO_PHASE_ANISO_GATE,
                        1e-6,
                    )
                    # The noise pseudo-axis reaches aniso ~0.5, inside
                    # the ramp above, so anisotropy alone still grants
                    # it substantial weight.  Multiply by the margin
                    # over the side-over-noise gate (0 at the gate, 1
                    # at gate + ramp): blocks that BARELY pass the
                    # noise gate are nearly weightless, so the rare
                    # pseudo-axis leakage cannot outrun the
                    # gate-closed leak toward 0.
                    w_noise = (
                        self.stereo_phase_side_over_noise_db
                        - STEREO_PHASE_SIDE_OVER_NOISE_DB
                    ) / max(STEREO_PHASE_NOISE_CONF_RAMP_DB, 1e-6)
                    w = float(np.clip(w, 0.0, 1.0)) * float(
                        np.clip(w_noise, 0.0, 1.0)
                    )
                    self._phase_conf = 0.9 * self._phase_conf + 0.1 * w
                    dd = beta_pa - self.stereo_phase_err_ema
                    dd = (dd + np.pi / 2.0) % np.pi - np.pi / 2.0
                    ema_old = self.stereo_phase_err_ema
                    ema = ema_old + STEREO_PHASE_ERR_SMOOTHING * w * dd
                    # Branch guard: crossing +-90 deg flips the
                    # nearest-representative branch (an L/R polarity
                    # flip).  With the hardware trim every legitimate
                    # operating point sits near 0, so a crossing is
                    # only accepted when the recent confidence EMA is
                    # high (sustained genuine-stereo tracking, e.g. a
                    # real channel drift); low-confidence wander is
                    # halted at the boundary instead.
                    lim = np.pi / 2.0
                    if self._phase_conf < STEREO_PHASE_BRANCH_CONF:
                        if ema_old < lim and ema >= lim:
                            ema = lim - 1e-6
                        elif ema_old > -lim and ema <= -lim:
                            ema = -lim + 1e-6
                    # Wrap the tracked angle into (-180, 180].
                    self.stereo_phase_err_ema = float(
                        (ema + np.pi) % (2.0 * np.pi) - np.pi
                    )
            else:
                if not self._phase_acquired:
                    # Acquisition demands CONSECUTIVE informative
                    # blocks; a break restarts the streak and its
                    # accumulator.
                    self._phase_acq_acc = 0j
                    self._phase_acq_count = 0
                else:
                    # Uninformed (gate closed) after acquisition: leak
                    # the tracked angle toward 0 - the hardware-trim
                    # prior - instead of holding a possibly wandered
                    # value.  With no side information the prior is the
                    # best estimate, and a genuine offset re-converges
                    # within ~1 s of confident content returning.
                    leak = np.deg2rad(STEREO_PHASE_LEAK_DEG_PER_SEC) * (
                        composite.size / float(self.composite_rate)
                    )
                    e = self.stereo_phase_err_ema
                    if abs(e) <= leak:
                        self.stereo_phase_err_ema = 0.0
                    else:
                        self.stereo_phase_err_ema = float(
                            e - np.sign(e) * leak
                        )
            cph = np.cos(self.stereo_phase_err_ema)
            sph = np.sin(self.stereo_phase_err_ema)
        else:
            self.stereo_phase_err_ema = 0.0
            self.stereo_phase_aniso = 0.0
            self.stereo_phase_side_over_noise_db = 0.0
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

        # Adaptive HF blend: ramp the configured MAX_GAIN ceilings up to 1.0
        # as pilot SNR improves, so HF stereo width is preserved on strong
        # signals. snr_open in [0, 1].
        s_lo_hf = STEREO_HF_BLEND_PILOT_SNR_DB_LO
        s_hi_hf = max(STEREO_HF_BLEND_PILOT_SNR_DB_HI, s_lo_hf + 1e-6)
        if self.pilot_snr_ema is None:
            snr_open_hf = 0.0
        elif self.pilot_snr_ema >= s_hi_hf:
            snr_open_hf = 1.0
        elif self.pilot_snr_ema <= s_lo_hf:
            snr_open_hf = 0.0
        else:
            snr_open_hf = (self.pilot_snr_ema - s_lo_hf) / (s_hi_hf - s_lo_hf)
        midhigh_max_cfg = max(LR_HIGH_MIN_GAIN, float(self.lr_high_max_gain))
        super_max_cfg = max(LR_SUPER_HIGH_MIN_GAIN, float(self.lr_super_high_max_gain))
        midhigh_max = midhigh_max_cfg + (1.0 - midhigh_max_cfg) * snr_open_hf
        super_max = super_max_cfg + (1.0 - super_max_cfg) * snr_open_hf
        midhigh_gain = LR_HIGH_MIN_GAIN + (midhigh_max - LR_HIGH_MIN_GAIN) * self.blend_factor
        super_gain = LR_SUPER_HIGH_MIN_GAIN + (super_max - LR_SUPER_HIGH_MIN_GAIN) * self.blend_factor
        lr_shaped = (
            lr_base_low
            + midhigh_gain * lr_base_midhigh
            + super_gain * lr_base_super
        )

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
                    "phJit=%.6g lrBandRMS=%.6g lrBaseRMS=%.6g phaseIQ=%.3fdeg aniso=%.2f sideNoise=%.1fdB conf=%.2f "
                    "iqCorr=%s "
                    "monoDelay=%d scOff=%.1fdeg sideCap=%.3f",
                    snr_db, self.blend_factor, pilot_power, noise_power,
                    phase_jitter,
                    float(np.sqrt(np.mean(lr_band ** 2) + 1e-12)),
                    float(np.sqrt(np.mean(lr_baseband ** 2) + 1e-12)),
                    float(np.rad2deg(self.stereo_phase_err_ema)),
                    self.stereo_phase_aniso,
                    self.stereo_phase_side_over_noise_db,
                    self._phase_conf,
                    "on" if self.iq_phase_correction_enabled else "off",
                    self.mono_delay_samples,
                    float(np.rad2deg(self.subcarrier_phase_offset_rad)),
                    self.lr_side_cap_gain,
                )

        # Remove 19 kHz pilot tone leakage
        left_channel = self.notch_pilot_l.apply(left_channel)
        left_channel = self.notch_pilot_l2.apply(left_channel)
        right_channel = self.notch_pilot_r.apply(right_channel)
        right_channel = self.notch_pilot_r2.apply(right_channel)

        # Resample composite -> final audio (stateful across blocks)
        left_48 = self._audio_resampler_l.process(
            left_channel.astype(np.float32),
        )
        right_48 = self._audio_resampler_r.process(
            right_channel.astype(np.float32),
        )
        left_48 = self.deemph_left.process(left_48)
        right_48 = self.deemph_right.process(right_48)

        if self.side_nr_enabled:
            mid = (0.5 * (left_48 + right_48)).astype(np.float32)
            side = (0.5 * (left_48 - right_48)).astype(np.float32)
            side_clean = self.side_nr.process(side)
            mid_aligned = self.side_nr_mid_aligner.feed_and_take(
                mid, side_clean.size,
            )
            n = min(mid_aligned.size, side_clean.size)
            mid_aligned = mid_aligned[:n]
            side_clean = side_clean[:n]
            left_48 = (mid_aligned + side_clean).astype(np.float32)
            right_48 = (mid_aligned - side_clean).astype(np.float32)
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
        mono_f32 = mono.astype(np.float32)
        mono_48 = self._audio_resampler_l.process(mono_f32)
        # Advance the right-channel chain in lockstep with the same
        # input: the stereo path uses both resamplers, so if only the
        # left one progressed during mono operation, a later
        # mono -> stereo switch would resume with the two resamplers at
        # different global emission positions and the first stereo
        # block would return mismatched L/R lengths (breaking the
        # mid/side recombination downstream).
        right_48 = self._audio_resampler_r.process(mono_f32)
        mono_48 = self.deemph_left.process(mono_48)
        self.deemph_right.process(right_48)
        return mono_48, mono_48

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset shared state and delegate to subclass.

        Clears all streaming-filter state (SOS lowpass / bandpass / notch
        and the de-emphasis IIR) so that a re-tune does not leak the
        previous station's audio into the first composite block of the
        new one.
        """
        self.pilot_pll.reset()
        self.dc_offset = 0.0
        self.blend_factor = 1.0
        self.pilot_snr_ema = None
        self.pilot_jitter_ema = 0.0
        self.stereo_phase_err_ema = 0.0
        self.stereo_phase_aniso = 0.0
        self.stereo_phase_side_over_noise_db = 0.0
        self._phase_acquired = False
        self._phase_acq_acc = 0j
        self._phase_acq_count = 0
        self._phase_conf = 0.0
        self.lr_side_cap_gain = 1.0
        if self.mono_delay_samples > 0:
            self._mono_delay_state = np.zeros(self.mono_delay_samples, dtype=np.float32)
        self._pilot_mix_phase = 0.0
        self._pilot_phase_last = None
        self._diag_counter = 0
        # Clear streaming-filter states so the first block after a
        # re-tune starts from zero initial conditions.
        for filt in (
            self.lp_mono,
            self.lp_lr_base, self.lp_lr_base_q,
            self.lp_lr_low, self.lp_lr_low_q,
            self.lp_lr_mid, self.lp_lr_mid_q,
            self.bp_pilot_noise_1, self.bp_pilot_noise_2,
            self.bp_lr,
            self.notch_pilot_l, self.notch_pilot_l2,
            self.notch_pilot_r, self.notch_pilot_r2,
            self.deemph_left, self.deemph_right,
        ):
            filt.reset()
        self._pilot_lp_zi = np.zeros_like(self._pilot_lp_zi)
        self._audio_resampler_l.reset()
        self._audio_resampler_r.reset()
        self.side_nr.reset()
        self.side_nr_mid_aligner.reset()
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
            # The subcarrier operating point depends on the main demod:
            # the PLL chain carries a -30.7 deg 19k/38k phase
            # inconsistency that the discriminator does not, so each
            # mode has its own tuned offset.
            subcarrier_phase_offset_deg=(
                STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL
                if MAIN_DEMOD_USE_PLL
                else STEREO_SUBCARRIER_PHASE_OFFSET_DEG
            ),
        )

        self.logger.info(
            f"Initializing FMDemodulator: IQ={iq_sample_rate/1e6:.3f}MHz, "
            f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
            f"Stereo={'enabled' if stereo else 'disabled'}"
        )

        # --- Standard-only: main FM demod + IQ lowpass filter ---
        # Discriminator by default; the PLL is kept selectable via
        # MAIN_DEMOD_USE_PLL for A/B comparison (see constants.py for
        # the measured response difference).
        self.use_pll_demod: bool = bool(MAIN_DEMOD_USE_PLL)
        self.main_pll: PLL = PLL(Kp=MAIN_PLL_KP, Ki=MAIN_PLL_KI, return_phase=False)
        # Last filtered IQ sample carried across blocks so the
        # discriminator's first phase difference is block-continuous.
        self._disc_last: np.ndarray | None = None
        nyquist = self.iq_sample_rate / 2.0
        # SOS form with carried filter state.  A stateless per-block
        # lfilter here resets the IIR internal state at every 16 ms
        # block boundary, producing a start-of-block transient of up to
        # ~75 deg phase error at the PLL input — audible as periodic
        # impulse noise after FM demodulation.
        self.iq_sos: np.ndarray = signal.butter(
            IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF / nyquist, btype="low",
            output="sos",
        )
        # Complex state: the filter runs on complex64 IQ samples.
        self._iq_zi: np.ndarray = np.zeros(
            (self.iq_sos.shape[0], 2), dtype=np.complex128,
        )
        self._iq_resampler = StatefulResampler(
            self.up, self.down,
            window=("kaiser", STANDARD_RESAMPLE_KAISER_BETA),
            # Keep every emitted composite block a multiple of the
            # composite->audio decimation factor so the downstream
            # per-block resample_poly stays on a consistent output grid.
            emit_align=self._resample_down,
        )

    def process_iq_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """Apply DC offset correction, lowpass filtering, FM demodulation
        (discriminator or PLL), and resampling to generate the composite.

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
            iq_filtered, self._iq_zi = signal.sosfilt(
                self.iq_sos, iq_processed, zi=self._iq_zi,
            )
            iq_filtered = iq_filtered.astype(np.complex64, copy=False)
            if self.use_pll_demod:
                main_output = self.main_pll.process(iq_filtered)
            else:
                # Arctan discriminator: instantaneous frequency in
                # rad/sample, same scale as the PLL's freq output but
                # exactly flat over the MPX band.  The previous block's
                # last sample seeds the first difference so the output
                # is continuous across block boundaries.
                if self._disc_last is None:
                    prev = iq_filtered[:1]
                else:
                    prev = self._disc_last
                ext = np.concatenate((prev, iq_filtered))
                main_output = np.angle(
                    ext[1:] * np.conj(ext[:-1])
                ).astype(np.float32, copy=False)
                self._disc_last = iq_filtered[-1:].copy()
            composite = self._iq_resampler.process(main_output)
            return composite.astype(np.float32, copy=False)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing IQ samples: {e}", exc_info=True)
            raise DemodulationError(f"Error processing IQ samples: {e}") from e

    def _reset_subclass(self) -> None:
        """Reset main demod state (PLL, discriminator, filters)."""
        self.main_pll.reset()
        self._iq_resampler.reset()
        self._iq_zi = np.zeros_like(self._iq_zi)
        self._disc_last = None


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
            # The light demodulator's old pilot bandpass (order 1) had a
            # different static phase than the standard order-9 one, so its
            # tuned operating point maps to a different offset here.
            subcarrier_phase_offset_deg=STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT,
        )

        self.logger.info(
            f"Initializing FMDemodulatorLight: IQ={iq_sample_rate/1e6:.3f}MHz, "
            f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
            f"Stereo={'enabled' if stereo else 'disabled'}"
        )

        # --- Light-only: discriminator state (previous IQ sample) ---
        self._disc_last: np.ndarray | None = None
        self._iq_resampler = StatefulResampler(
            self.up, self.down, emit_align=self._resample_down,
        )

    def process_iq_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """Apply DC offset correction, arctan discrimination, and
        resampling to generate the composite signal.

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
            # Arctan discriminator, same form as the standard chain:
            # angle(x[n]*conj(x[n-1])) is the wrapped per-sample phase
            # step - identical to the previous angle->unwrap->diff
            # whenever the true step is within +-pi (unwrap's own
            # assumption), but it never accumulates absolute phase.
            # The old path kept the unwrapped phase in float32; under a
            # carrier offset it grows as 2*pi*df*t without bound, and
            # once it reaches ~1e6 rad the float32 spacing (~0.06 rad)
            # dwarfs the per-sample step, quantising the audio in long
            # light-mode sessions.  The previous block's last sample
            # seeds the first difference (stream start: zero).
            if self._disc_last is None:
                prev = iq_processed[:1]
            else:
                prev = self._disc_last
            ext = np.concatenate((prev, iq_processed))
            fm_demod = np.angle(
                ext[1:] * np.conj(ext[:-1])
            ).astype(np.float32, copy=False)
            self._disc_last = iq_processed[-1:].copy()
            composite = (
                self._iq_resampler.process(fm_demod) * LIGHT_COMPOSITE_SCALE
            )
            return np.asarray(composite, dtype=np.float32, copy=False)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error processing IQ samples (Light): {e}", exc_info=True)
            raise DemodulationError(f"Error processing IQ samples (Light): {e}") from e

    def _reset_subclass(self) -> None:
        """Reset discriminator state."""
        self._disc_last = None
        self._iq_resampler.reset()
