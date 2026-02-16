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
"""FM demodulator classes (Standard and Light versions)."""

import logging
from fractions import Fraction

import numpy as np
import scipy.signal as signal

from fm_radio.filters import LowpassFilter, BandpassFilter, DeemphasisIIRFilter
from fm_radio.pll import PLL


class FMDemodulator:
    """
    Standard FM demodulation class

    Processes IQ samples through PLL, filters, resampling, stereo separation,
    and de-emphasis to generate the demodulated signal.
    """
    def __init__(self, iq_sample_rate=1.024e6, composite_rate=192000,
                 final_audio_rate=48000, stereo=True):
        self.logger = logging.getLogger('fm_receiver.FMDemodulator')
        self.iq_sample_rate = iq_sample_rate
        self.composite_rate = composite_rate
        self.final_audio_rate = final_audio_rate
        self.stereo = stereo

        self.logger.info(f"Initializing FMDemodulator: IQ={iq_sample_rate/1e6:.3f}MHz, "
                        f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
                        f"Stereo={'enabled' if stereo else 'disabled'}")

        # PLL for main signal demodulation (frequency output)
        self.main_pll = PLL(Kp=0.12926, Ki=0.0208844, return_phase=False)
        # PLL for pilot signal demodulation (phase output)
        self.pilot_pll = PLL(Kp=0.0432, Ki=0.000116, return_phase=True)

        nyquist = self.iq_sample_rate / 2.0
        # Lowpass filter for IQ samples (pass frequencies below 200 kHz)
        self.iq_b, self.iq_a = signal.butter(5, 200e3 / nyquist, btype="low")
        # Lowpass filter for mono signal (15 kHz cutoff)
        self.lp_mono = LowpassFilter(order=15, cutoff=15000.0, sample_rate=self.composite_rate)
        # Baseband filter for stereo separation
        self.lp_base = LowpassFilter(order=15, cutoff=15000.0, sample_rate=self.composite_rate)
        # Bandpass filter for extracting pilot signal (17-21 kHz)
        self.bp_pilot = BandpassFilter(order=5, lowcut=17000.0, highcut=21000.0, sample_rate=self.composite_rate)
        # Bandpass filter for extracting LR signal (23-53 kHz)
        self.bp_lr = BandpassFilter(order=15, lowcut=23000.0, highcut=53000.0, sample_rate=self.composite_rate)
        # De-emphasis filters for left and right channels
        self.deemph_left = DeemphasisIIRFilter(sample_rate=self.final_audio_rate, tau=50e-6)
        self.deemph_right = DeemphasisIIRFilter(sample_rate=self.final_audio_rate, tau=50e-6)
        self.dc_offset = 0.0  # For DC offset correction
        self.dc_alpha = 0.01  # Smoothing coefficient for DC offset

        # Calculate resampling ratio (as a close integer ratio)
        ratio = Fraction(int(self.composite_rate), int(self.iq_sample_rate)).limit_denominator()
        self.up = ratio.numerator
        self.down = ratio.denominator
        # Precompute composite -> final audio integer resample factors
        self._resample_up = 1
        self._resample_down = max(1, int(self.composite_rate / self.final_audio_rate))

    def process_iq_samples(self, iq_samples):
        """
        Apply DC offset correction, lowpass filtering, PLL demodulation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
        """
        try:
            self.dc_offset = self.dc_alpha * np.mean(iq_samples) + (1 - self.dc_alpha) * self.dc_offset
            iq_processed = iq_samples - self.dc_offset
            # ensure complex64 for processing pipeline
            iq_processed = np.asarray(iq_processed, dtype=np.complex64, copy=False)
            iq_filtered = signal.lfilter(self.iq_b, self.iq_a, iq_processed)
            main_output = self.main_pll.process(iq_filtered)
            composite = signal.resample_poly(main_output, up=self.up, down=self.down)
            # ensure float32 to avoid later casts
            return composite.astype(np.float32, copy=False)
        except Exception as e:
            self.logger.error(f"Error processing IQ samples: {e}", exc_info=True)
            raise

    def demodulate(self, composite):
        """
        Generate stereo or mono audio signals from the composite signal.

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
        """
        Stereo demodulation process

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

        # precompute resample ratio from composite_rate -> final_audio_rate
        self._resample_up = 1
        self._resample_down = int(self.composite_rate / self.final_audio_rate)
        if self._resample_down < 1:
            self._resample_down = 1
        # replace samplerate.resample with resample_poly using integer downsample
        left_48 = signal.resample_poly(left_channel.astype(np.float32), self._resample_up, self._resample_down)
        right_48 = signal.resample_poly(right_channel.astype(np.float32), self._resample_up, self._resample_down)
        left_48 = self.deemph_left.process(left_48)
        right_48 = self.deemph_right.process(right_48)
        return left_48, right_48

    def _demodulate_mono(self, composite):
        """
        Mono demodulation process

        Args:
            composite (ndarray): Composite signal.

        Returns:
            tuple: (mono_channel, mono_channel) where both channels are identical.
        """
        mono = self.lp_mono.apply(composite)
        # replace samplerate.resample with resample_poly (composite -> final audio)
        mono_48 = signal.resample_poly(mono.astype(np.float32), self._resample_up, self._resample_down)
        mono_48 = self.deemph_left.process(mono_48)
        return mono_48, mono_48

    def reset(self):
        """Reset PLL states and DC offset."""
        self.main_pll.reset()
        self.pilot_pll.reset()
        self.dc_offset = 0.0


class FMDemodulatorLight:
    """
    Light version of the FM demodulation class

    Implements a simplified process to reduce computational load.
    Uses phase differentiation for FM demodulation, resampling,
    de-emphasis, and stereo/mono separation.
    """
    def __init__(self, iq_sample_rate=0.25e6, composite_rate=192000,
                 final_audio_rate=48000, stereo=True):
        self.logger = logging.getLogger('fm_receiver.FMDemodulatorLight')
        self.iq_sample_rate = iq_sample_rate
        self.composite_rate = composite_rate
        self.final_audio_rate = final_audio_rate
        self.stereo = stereo

        self.logger.info(f"Initializing FMDemodulatorLight: IQ={iq_sample_rate/1e6:.3f}MHz, "
                        f"Composite={composite_rate/1e3:.0f}kHz, Audio={final_audio_rate/1e3:.0f}kHz, "
                        f"Stereo={'enabled' if stereo else 'disabled'}")

        # PLL for pilot signal demodulation (phase output)
        self.pilot_pll = PLL(Kp=0.0432, Ki=0.000116, return_phase=True)

        # Use lower order filters for reduced computation
        self.lp_mono = LowpassFilter(order=1, cutoff=15000.0, sample_rate=self.composite_rate)
        self.lp_base = LowpassFilter(order=1, cutoff=15000.0, sample_rate=self.composite_rate)
        self.bp_pilot = BandpassFilter(order=1, lowcut=17000.0, highcut=21000.0, sample_rate=self.composite_rate)
        self.bp_lr = BandpassFilter(order=1, lowcut=23000.0, highcut=53000.0, sample_rate=self.composite_rate)
        self.deemph_left = DeemphasisIIRFilter(sample_rate=self.final_audio_rate, tau=50e-6)
        self.deemph_right = DeemphasisIIRFilter(sample_rate=self.final_audio_rate, tau=50e-6)
        self.dc_offset = 0.0
        self.dc_alpha = 0.01
        self.last_phase = None  # Save previous phase for differentiation

        ratio = Fraction(int(self.composite_rate), int(self.iq_sample_rate)).limit_denominator()
        self.up = ratio.numerator
        self.down = ratio.denominator
        # Precompute composite -> final audio integer resample factors
        self._resample_up = 1
        self._resample_down = max(1, int(self.composite_rate / self.final_audio_rate))

    def process_iq_samples(self, iq_samples):
        """
        Apply DC offset correction, phase extraction/differentiation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
        """
        try:
            self.dc_offset = self.dc_alpha * np.mean(iq_samples) + (1 - self.dc_alpha) * self.dc_offset
            iq_processed = iq_samples - self.dc_offset
            current_phase = np.angle(iq_processed)
            if self.last_phase is None:
                phase = np.unwrap(current_phase)
            else:
                phase = np.unwrap(np.concatenate(([self.last_phase], current_phase)))[1:]
            fm_demod = np.diff(phase, prepend=phase[0])
            self.last_phase = phase[-1]
            composite = signal.resample_poly(fm_demod, up=self.up, down=self.down) * 0.35
            # ensure float32 for downstream pipeline
            return np.asarray(composite, dtype=np.float32, copy=False)
        except Exception as e:
            self.logger.error(f"Error processing IQ samples (Light): {e}", exc_info=True)
            raise

    def demodulate(self, composite):
        """
        Generate stereo or mono audio signals from the composite signal.

        Returns:
            tuple: (left_channel, right_channel)
        """
        if self.stereo:
            return self._demodulate_stereo(composite)
        else:
            return self._demodulate_mono(composite)

    def _demodulate_stereo(self, composite):
        """
        Light version stereo demodulation process

        Process flow:
          1. Extract mono signal via lowpass filter.
          2. Extract pilot signal and generate complex pilot using Hilbert transform.
          3. Generate subcarrier from pilot phase.
          4. Demodulate LR signal and baseband it.
          5. Generate left and right channels, then resample and apply de-emphasis.

        Returns:
            tuple: (left_channel_audio, right_channel_audio)
        """
        mono = self.lp_mono.apply(composite)
        pilot = self.bp_pilot.apply(composite)
        pilot_complex = signal.hilbert(pilot)
        pilot_phase = self.pilot_pll.process(pilot_complex)
        subcarrier = np.cos(2.0 * pilot_phase)
        lr_band = self.bp_lr.apply(composite)
        lr_demodulated = lr_band * subcarrier
        lr_baseband = self.lp_base.apply(lr_demodulated)
        left_channel = mono + lr_baseband
        right_channel = mono - lr_baseband
        left_48 = signal.resample_poly(left_channel.astype(np.float32), self._resample_up, self._resample_down)
        right_48 = signal.resample_poly(right_channel.astype(np.float32), self._resample_up, self._resample_down)
        left_48 = self.deemph_left.process(left_48)
        right_48 = self.deemph_right.process(right_48)
        return left_48, right_48

    def _demodulate_mono(self, composite):
        """
        Light version mono demodulation process

        Returns:
            tuple: (mono_channel, mono_channel)
        """
        mono = self.lp_mono.apply(composite)
        mono_48 = signal.resample_poly(mono.astype(np.float32), self._resample_up, self._resample_down)
        mono_48 = self.deemph_left.process(mono_48)
        return mono_48, mono_48

    def reset(self):
        """Reset DC offset and phase information."""
        self.dc_offset = 0.0
        self.last_phase = None
