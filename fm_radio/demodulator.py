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

import logging
from abc import abstractmethod
from fractions import Fraction

import numpy as np
import scipy.signal as signal

from fm_radio.interfaces import FMDemodulatorInterface
from fm_radio.filters import LowpassFilter, BandpassFilter, DeemphasisIIRFilter
from fm_radio.pll import PLL
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_SAMPLE_RATE_LIGHT,
    MAIN_PLL_KP, MAIN_PLL_KI, PILOT_PLL_KP, PILOT_PLL_KI,
    IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF,
    MONO_LOWPASS_ORDER, MONO_LOWPASS_ORDER_LIGHT, MONO_LOWPASS_CUTOFF,
    PILOT_BANDPASS_ORDER, PILOT_BANDPASS_ORDER_LIGHT,
    PILOT_BANDPASS_LOW, PILOT_BANDPASS_HIGH,
    LR_BANDPASS_ORDER, LR_BANDPASS_ORDER_LIGHT,
    LR_BANDPASS_LOW, LR_BANDPASS_HIGH,
    DEEMPHASIS_TAU, DC_OFFSET_ALPHA,
    AUDIO_OUTPUT_RATE, COMPOSITE_RATE, LIGHT_COMPOSITE_SCALE,
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
        self.lp_base = LowpassFilter(
            order=mono_order, cutoff=MONO_LOWPASS_CUTOFF,
            sample_rate=self.composite_rate,
        )
        self.bp_pilot = BandpassFilter(
            order=pilot_order, lowcut=PILOT_BANDPASS_LOW,
            highcut=PILOT_BANDPASS_HIGH, sample_rate=self.composite_rate,
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

        # --- DC offset tracking ---
        self.dc_offset = 0.0
        self.dc_alpha = DC_OFFSET_ALPHA

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

    def demodulate(self, composite):
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

    def _demodulate_stereo(self, composite):
        """Stereo demodulation process.

        Process flow:
          1. Extract mono signal via lowpass filter.
          2. Extract pilot signal using bandpass filter and Hilbert transform.
          3. Generate subcarrier from pilot phase.
          4. Extract and baseband LR signal.
          5. Combine channels (L = mono + LR, R = mono - LR).
          6. Resample and apply de-emphasis.

        Returns:
            tuple: (left_channel_audio, right_channel_audio)
        """
        mono = self.lp_mono.apply(composite)
        pilot_signal = self.bp_pilot.apply(composite)
        pilot_complex = signal.hilbert(pilot_signal.astype(np.float32))
        pilot_phase = self.pilot_pll.process(pilot_complex)
        subcarrier = np.cos(2.0 * pilot_phase)
        lr_band = self.bp_lr.apply(composite)
        lr_demodulated = lr_band * subcarrier
        lr_baseband = self.lp_base.apply(lr_demodulated)
        left_channel = mono + lr_baseband
        right_channel = mono - lr_baseband

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

    def _demodulate_mono(self, composite):
        """Mono demodulation process.

        Args:
            composite (ndarray): Composite signal.

        Returns:
            tuple: (mono_channel, mono_channel) where both channels are identical.
        """
        mono = self.lp_mono.apply(composite)
        mono_48 = signal.resample_poly(
            mono.astype(np.float32),
            self._resample_up, self._resample_down,
        )
        mono_48 = self.deemph_left.process(mono_48)
        return mono_48, mono_48

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset shared state and delegate to subclass."""
        self.pilot_pll.reset()
        self.dc_offset = 0.0
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

    def __init__(self, iq_sample_rate=SDR_SAMPLE_RATE, composite_rate=COMPOSITE_RATE,
                 final_audio_rate=AUDIO_OUTPUT_RATE, stereo=True):
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
        self.main_pll = PLL(Kp=MAIN_PLL_KP, Ki=MAIN_PLL_KI, return_phase=False)
        nyquist = self.iq_sample_rate / 2.0
        self.iq_b, self.iq_a = signal.butter(
            IQ_LOWPASS_ORDER, IQ_LOWPASS_CUTOFF / nyquist, btype="low",
        )

    def process_iq_samples(self, iq_samples):
        """Apply DC offset correction, lowpass filtering, PLL demodulation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
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
            composite = signal.resample_poly(main_output, up=self.up, down=self.down)
            return composite.astype(np.float32, copy=False)
        except Exception as e:
            self.logger.error(f"Error processing IQ samples: {e}", exc_info=True)
            raise

    def _reset_subclass(self):
        """Reset main PLL state."""
        self.main_pll.reset()


class FMDemodulatorLight(BaseFMDemodulator):
    """Light FM demodulator (phase-differentiation-based).

    Implements a simplified process to reduce computational load.
    Uses phase differentiation for FM demodulation instead of a full PLL.
    """

    def __init__(self, iq_sample_rate=SDR_SAMPLE_RATE_LIGHT, composite_rate=COMPOSITE_RATE,
                 final_audio_rate=AUDIO_OUTPUT_RATE, stereo=True):
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
        self.last_phase = None

    def process_iq_samples(self, iq_samples):
        """Apply DC offset correction, phase extraction/differentiation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
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
        except Exception as e:
            self.logger.error(f"Error processing IQ samples (Light): {e}", exc_info=True)
            raise

    def _reset_subclass(self):
        """Reset phase tracking state."""
        self.last_phase = None
