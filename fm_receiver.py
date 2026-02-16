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
"""
FM Receiver System

This program receives FM broadcast signals using RTL-SDR,
performs FM demodulation via PLL and various filter processes,
and outputs/records audio using PyAudio.

Usage Examples:
  - Standard mode (no logging):
      $ python fm_receiver.py
  - Light mode:
      $ python fm_receiver.py --light
  - With logging enabled:
      $ python fm_receiver.py --log
      $ python fm_receiver.py --verbose  (or -v)
  - With debug logging:
      $ python fm_receiver.py --debug
  - Save logs to file:
      $ python fm_receiver.py --log-file fm_receiver.log
  - Combine options:
      $ python fm_receiver.py --light --debug --log-file debug.log

Command Examples (during execution):
  'list'              : Show available stations
  'stereo on'         : Enable stereo demodulation
  'stereo off' / 'mono': Enable mono demodulation
  'record start'      : Start recording (file name auto-generated)
  'record stop'       : Stop recording
  'agc on'            : Enable automatic gain control (AGC)
  'agc off'           : Enable manual gain control
  'gain <value>'      : Set manual gain (only when AGC is disabled)
  '<station_num>' or '<freq_MHz>' : Tune to the specified station
  'q'                 : Quit the program
"""

import threading
import queue
import math
import time
import sys
import wave
import logging
from fractions import Fraction
from collections import deque

import numpy as np
import scipy.signal as signal
import pyaudio
import samplerate
from rtlsdr import RtlSdr
from numba import njit

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the FM receiver system.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path. If None, logs to console only.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

# Initialize logger
logger = logging.getLogger('fm_receiver')


# --------------------------------------------------
# Deemphasis IIR Filter Class (Numba optimized)
# --------------------------------------------------
@njit
def deemphasis_iir_process_numba(x, alpha, prev_output):
    """
    Numba-optimized de-emphasis IIR filter processing.

    Args:
        x (ndarray): Input audio signal array.
        alpha (float): Filter coefficient.
        prev_output (float): Previous output sample for filter state.

    Returns:
        tuple: (y, prev) where y is filtered output array and prev is last output value.
    """
    y = np.empty_like(x)
    prev = prev_output
    for i in range(x.shape[0]):
        y[i] = (1.0 - alpha) * x[i] + alpha * prev
        prev = y[i]
    return y, prev


class DeemphasisIIRFilter:
    """
    IIR filter for de-emphasis processing in FM broadcasting.

    Args:
        sample_rate (float): Audio signal sample rate.
        tau (float): Time constant (e.g., 50e-6 seconds).
    """
    def __init__(self, sample_rate, tau):
        self.sample_rate = sample_rate
        self.tau = tau
        # Calculate filter coefficient alpha using an exponential function
        self.alpha = math.exp(-1.0 / (sample_rate * tau))
        self.prev_output = 0.0

    def process(self, x):
        """
        Apply de-emphasis processing to the input signal using a Numba-optimized loop.

        Args:
            x (ndarray): Input audio signal array.

        Returns:
            ndarray: Audio signal after filtering.
        """
        y, self.prev_output = deemphasis_iir_process_numba(x, self.alpha, self.prev_output)
        return y


# --------------------------------------------------
# Filter Classes (LowpassFilter & BandpassFilter)
# --------------------------------------------------
class LowpassFilter:
    """
    Butterworth lowpass filter (streaming using lfilter with state)
    """
    def __init__(self, order, cutoff, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, cutoff / nyquist, btype="low", analog=False)
        # filter state for streaming: length = max(len(a), len(b)) - 1
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)

    def apply(self, data):
        """
        Apply the filter to streaming chunk, preserving state.
        """
        y, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        return y


class BandpassFilter:
    """
    Butterworth bandpass filter (streaming using lfilter with state)
    """
    def __init__(self, order, lowcut, highcut, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, [lowcut / nyquist, highcut / nyquist],
                                        btype="band", analog=False)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)

    def apply(self, data):
        """
        Apply the bandpass filter to streaming chunk, preserving state.
        """
        y, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        return y


# --------------------------------------------------
# PLL Class (Standard Version) and Numba Optimized Function
# --------------------------------------------------
@njit
def pll_demodulate(iq_samples, Kp, Ki, state):
    """
    Numba optimized FM demodulation using PLL

    Args:
        iq_samples (ndarray): Array of complex IQ samples.
        Kp (float): Proportional gain of the PLL.
        Ki (float): Integral gain of the PLL.
        state (ndarray): Array holding [phase, integrator] state.

    Returns:
        tuple: (phase_out, freq_out)
            phase_out (ndarray): Phase after processing each sample.
            freq_out (ndarray): Frequency estimate after each sample.
    """
    n = iq_samples.shape[0]
    phase_out = np.empty(n, dtype=np.float32)
    freq_out = np.empty(n, dtype=np.float32)
    errors = np.empty(n, dtype=np.float32)
    phase = state[0]
    integrator = state[1]
    for i in range(n):
        sample = iq_samples[i]
        vco_real = math.cos(phase)
        vco_imag = math.sin(phase)
        prod_real = sample.real * vco_real + sample.imag * vco_imag
        prod_imag = -sample.real * vco_imag + sample.imag * vco_real
        error = math.atan2(prod_imag, prod_real)
        errors[i] = error
        integrator += Ki * error
        freq_est = Kp * error + integrator
        phase += freq_est
        phase = np.mod(phase, 2 * math.pi)
        phase_out[i] = phase
        freq_out[i] = freq_est
    state[0] = phase
    state[1] = integrator
    return phase_out, freq_out


class PLL:
    """
    FM demodulation using a Phase Locked Loop (PLL).

    Extracts phase or frequency information from IQ samples.
    """
    def __init__(self, Kp, Ki, return_phase=False):
        self._Kp = Kp
        self._Ki = Ki
        self.return_phase = return_phase
        self.state = np.zeros(2, dtype=np.float32)  # [phase, integrator]
        self.last_freq = 0.0

    def process(self, iq_samples):
        """
        Process IQ samples with the PLL to generate demodulated signal.

        Args:
            iq_samples (ndarray): Array of IQ samples.

        Returns:
            ndarray: Demodulated signal (phase or frequency).
        """
        phase_out, freq_out = pll_demodulate(iq_samples, self._Kp, self._Ki, self.state)
        if freq_out.size > 0:
            self.last_freq = freq_out[-1]
        return phase_out if self.return_phase else freq_out

    def get_last_freq(self):
        """Retrieve the last frequency estimate."""
        return self.last_freq

    def reset(self):
        """Reset PLL state (phase and integrator)."""
        self.state[:] = 0.0

    def set_Kp(self, Kp):
        """Set the proportional gain."""
        self._Kp = Kp

    def get_Kp(self):
        """Get the proportional gain."""
        return self._Kp

    def set_Ki(self, Ki):
        """Set the integral gain."""
        self._Ki = Ki

    def get_Ki(self):
        """Get the integral gain."""
        return self._Ki


# --------------------------------------------------
# SDRReceiver Class
# --------------------------------------------------
class SDRReceiver:
    """
    Receiver class using RTL-SDR

    Retrieves samples from the RTL-SDR device and asynchronously stores them in a queue.
    """
    def __init__(self, sample_rate=1.024e6, center_freq=80e6, block_size=16384):
        self.logger = logging.getLogger('fm_receiver.SDRReceiver')
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.block_size = block_size
        self.data_queue = queue.Queue(maxsize=20)

        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.set_manual_gain_enabled(False)
            self.manual_gain = False
            self.sdr.set_gain(0)
            self.logger.info(f"SDR initialized: sample_rate={sample_rate/1e6:.3f}MHz, center_freq={center_freq/1e6:.1f}MHz")
        except Exception as e:
            self.logger.error(f"Failed to initialize RTL-SDR device: {e}")
            raise

        try:
            # Disable direct_sampling if available
            self.sdr.direct_sampling = 0
            self.logger.debug("Direct sampling disabled")
        except Exception as e:
            self.logger.warning(f"Failed to disable direct_sampling (may not be supported): {e}")

    def set_center_frequency(self, freq):
        """Change the center frequency."""
        try:
            self.center_freq = freq
            self.sdr.center_freq = freq
            self.logger.info(f"Center frequency set to {freq/1e6:.1f} MHz")
        except Exception as e:
            self.logger.error(f"Failed to set center frequency to {freq/1e6:.1f} MHz: {e}")
            raise

    def get_center_frequency(self):
        """Retrieve the current center frequency."""
        return self.sdr.center_freq

    def set_gain(self, gain):
        """Set gain value (for manual mode)."""
        try:
            self.sdr.set_gain(gain)
            self.logger.info(f"Gain set to {gain:.1f} dB")
        except Exception as e:
            self.logger.error(f"Failed to set gain to {gain:.1f} dB: {e}")
            raise

    def get_gain(self):
        """Retrieve the current gain value."""
        return self.sdr.get_gain()

    def set_manual_gain_mode(self, manual):
        """
        Set manual gain mode.

        Args:
            manual (bool): True for manual mode, False for AGC.
        """
        try:
            self.manual_gain = manual
            self.sdr.set_manual_gain_enabled(manual)
            mode = "manual" if manual else "AGC"
            self.logger.info(f"Gain mode set to {mode}")
        except Exception as e:
            self.logger.error(f"Failed to set gain mode: {e}")
            raise

    def callback(self, iq_samples, sdr_obj):
        """
        Callback to store received IQ samples in the data queue.

        Args:
            iq_samples (ndarray): Received IQ samples.
            sdr_obj: SDR object (unused).
        """
        try:
            # Convert to numpy array allowing a copy if necessary (NumPy 2.x compatibility).
            iq = np.asarray(iq_samples, dtype=np.complex64)
            self.data_queue.put(iq, block=False)
        except queue.Full:
            # Discard sample if queue is full.
            self.logger.debug("SDR data queue full, dropping samples")
        except Exception as e:
            # Drop this buffer if conversion fails to avoid crashing the SDR ctypes callback.
            self.logger.error(f"Error in SDR callback: {e}", exc_info=True)

    def start(self):
        """Start asynchronous sample retrieval."""
        try:
            self.logger.info("Starting SDR async read")
            self.sdr.read_samples_async(self.callback, num_samples=self.block_size)
        except Exception as e:
            self.logger.error(f"Failed to start SDR async read: {e}")
            raise

    def stop(self):
        """Stop asynchronous sample retrieval and close SDR."""
        try:
            self.logger.info("Stopping SDR async read")
            self.sdr.cancel_read_async()
        except Exception as e:
            self.logger.warning(f"Error canceling async read: {e}")

        try:
            self.sdr.close()
            self.logger.info("SDR closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing SDR: {e}")


# --------------------------------------------------
# FMDemodulator Class (Standard Version)
# --------------------------------------------------
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


# --------------------------------------------------
# FMDemodulatorLight Class (Light Version)
# --------------------------------------------------
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


# --------------------------------------------------
# AudioOutput Class
# --------------------------------------------------
class AudioOutput:
    """
    Audio output and recording management class

    Uses PyAudio for audio output and recording.
    """
    def __init__(self, output_rate=48000, frames_per_buffer=1024):
        self.logger = logging.getLogger('fm_receiver.AudioOutput')
        self.output_rate = output_rate
        self.frames_per_buffer = frames_per_buffer
        self.audio_buffer_queue = queue.Queue(maxsize=50)
        self.recording = False
        self.record_wave = None
        self.record_lock = threading.Lock()
        # replace large concatenating buffer with deque of numpy arrays
        self._buffer_deque = deque()
        self._buffer_len = 0  # total number of float32 samples in deque

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=int(self.output_rate),
                output=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            self.logger.info(f"Audio output initialized: rate={output_rate}Hz, buffer={frames_per_buffer}")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio output: {e}")
            raise

    def callback(self, in_data, frame_count, time_info, status):
        try:
            if status:
                self.logger.warning(f"Audio callback status: {status}")

            requested_samples = frame_count * 2  # stereo interleaved samples
            # fill deque from queue (avoid concatenation)
            while self._buffer_len < requested_samples:
                try:
                    left, right = self.audio_buffer_queue.get_nowait()
                    stereo = np.empty((left.size + right.size,), dtype=np.float32)
                    stereo[0::2] = left
                    stereo[1::2] = right
                    self._buffer_deque.append(stereo)
                    self._buffer_len += stereo.size
                except queue.Empty:
                    break

            out = np.empty((requested_samples,), dtype=np.float32)
            filled = 0
            while filled < requested_samples and self._buffer_deque:
                chunk = self._buffer_deque[0]
                need = requested_samples - filled
                if chunk.size <= need:
                    out[filled:filled + chunk.size] = chunk
                    filled += chunk.size
                    self._buffer_deque.popleft()
                    self._buffer_len -= chunk.size
                else:
                    out[filled:filled + need] = chunk[:need]
                    # keep remainder in deque (slice shares no memory — acceptable)
                    self._buffer_deque[0] = chunk[need:]
                    self._buffer_len -= need
                    filled += need

            if filled < requested_samples:
                out[filled:requested_samples] = 0.0
                if filled == 0:
                    self.logger.debug("Audio buffer underrun")

            return (out.tobytes(), pyaudio.paContinue)
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}", exc_info=True)
            # Return silence to avoid crashing the audio stream
            silence = np.zeros(frame_count * 2, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)

    def enqueue_audio(self, left, right):
        try:
            left32 = np.asarray(left, dtype=np.float32, copy=False)
            right32 = np.asarray(right, dtype=np.float32, copy=False)
            self.audio_buffer_queue.put((left32, right32), timeout=0.01)
        except queue.Full:
            self.logger.debug("Audio buffer queue full, dropping audio data")
        except Exception as e:
            self.logger.error(f"Error enqueueing audio: {e}", exc_info=True)

    def start_recording(self, filename, channels=2):
        """
        Start recording audio.

        Args:
            filename (str): Filename to save the WAV file.
            channels (int): Number of channels.
        """
        with self.record_lock:
            if self.recording:
                self.logger.warning("Already recording")
                print("Already recording.")
                return
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(int(self.output_rate))
                self.record_wave = wf
                self.recording = True
                self.logger.info(f"Recording started: {filename}")
                print(f"Recording started: {filename}")
            except Exception as e:
                self.logger.error(f"Recording start failed: {e}", exc_info=True)
                print("Recording start failed:", e)

    def stop_recording(self):
        """Stop recording and close the file."""
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                try:
                    self.record_wave.close()
                    self.logger.info("Recording stopped")
                    print("Recording stopped.")
                except Exception as e:
                    self.logger.error(f"Error closing recording file: {e}", exc_info=True)
                finally:
                    self.record_wave = None
                    self.recording = False
            else:
                self.logger.debug("stop_recording called but not currently recording")
                print("Not currently recording.")

    def record(self, stereo_audio):
        """
        Write audio data to file if recording.

        Args:
            stereo_audio (ndarray): Stereo audio data.
        """
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                try:
                    # Clip to [-1,1] before converting to int16 to avoid overflow/distortion
                    clipped = np.clip(stereo_audio, -1.0, 1.0)
                    int16_audio = np.int16(clipped * 32767)
                    self.record_wave.writeframes(int16_audio.tobytes())
                except Exception as e:
                    self.logger.error(f"Error writing audio to file: {e}", exc_info=True)

    def cleanup(self):
        """Stop audio stream and terminate PyAudio instance."""
        try:
            # Stop recording if active
            if self.recording:
                self.logger.info("Stopping active recording during cleanup")
                self.stop_recording()

            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
            self.logger.info("Audio output cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during audio cleanup: {e}", exc_info=True)


# --------------------------------------------------
# CommandLineInterface Class
# --------------------------------------------------
class CommandLineInterface(threading.Thread):
    """
    Thread for handling command line input

    Receives user commands and controls FMReceiverController functions.
    """
    def __init__(self, controller):
        super().__init__(daemon=True)
        self.controller = controller

    def run(self):
        while not self.controller.quit_event.is_set():
            print("\nEnter command:")
            print("  'list' -> show station list")
            print("  'stereo on/off' or 'mono' -> toggle stereo demodulation")
            print("  'record start' -> start recording with auto-generated filename")
            print("  'record stop' -> stop recording")
            print("  'agc on' -> enable AGC")
            print("  'agc off' -> disable AGC (manual mode)")
            print("  'gain <value>' -> set manual gain")
            print("  <station_num> or <freq_MHz> -> tune")
            print("  'q' -> quit")
            cmd = input().strip().lower()

            if cmd == 'q':
                print("Exiting command input...")
                self.controller.quit_event.set()
                break
            elif cmd == 'list':
                print("Available stations:")
                for i, (name, freq) in enumerate(self.controller.stations_list, start=1):
                    print(f"{i}: {name} ({freq/1e6:.1f} MHz)")
            elif cmd in ("stereo on", "stereo"):
                if hasattr(self.controller.fm_demodulator, 'stereo'):
                    self.controller.fm_demodulator.stereo = True
                    print("Stereo demodulation enabled.")
                else:
                    print("Stereo demodulation not supported.")
            elif cmd in ("stereo off", "mono"):
                if hasattr(self.controller.fm_demodulator, 'stereo'):
                    self.controller.fm_demodulator.stereo = False
                    print("Mono demodulation enabled.")
                else:
                    print("Stereo demodulation not supported.")
            elif cmd == "record start":
                current_time = time.strftime("%Y%m%d_%H%M%S")
                freq = self.controller.sdr_receiver.get_center_frequency() / 1e6
                filename = f"{current_time}_{freq:.1f}MHz.wav"
                self.controller.audio_output.start_recording(filename)
            elif cmd == "record stop":
                self.controller.audio_output.stop_recording()
            elif cmd.startswith("agc"):
                tokens = cmd.split()
                if len(tokens) == 2:
                    if tokens[1] in ["on"]:
                        self.controller.sdr_receiver.set_manual_gain_mode(False)
                        print("Automatic gain control enabled.")
                    elif tokens[1] in ["off"]:
                        self.controller.sdr_receiver.set_manual_gain_mode(True)
                        print(f"Manual gain control enabled. Current gain: {self.controller.sdr_receiver.get_gain():.1f}")
                    else:
                        print("Invalid agc command format.")
                elif len(tokens) == 3:
                    if tokens[1] in ["off"]:
                        try:
                            gain_value = float(tokens[2])
                            self.controller.sdr_receiver.set_manual_gain_mode(True)
                            self.controller.sdr_receiver.set_gain(gain_value)
                            print(f"Manual gain control enabled. Gain set to {gain_value:.1f}")
                        except ValueError:
                            print("Invalid gain value.")
                    else:
                        print("Invalid gain command format.")
                else:
                    print("Invalid agc command format.")
            elif cmd.startswith("gain"):
                tokens = cmd.split()
                if len(tokens) == 2:
                    try:
                        gain_value = float(tokens[1])
                        if self.controller.sdr_receiver.manual_gain:
                            self.controller.sdr_receiver.set_gain(gain_value)
                            print(f"Manual gain set to {gain_value:.1f}")
                        else:
                            print("Automatic gain control is enabled. Please disable it to set manual gain.")
                    except ValueError:
                        print("Invalid gain command.")
                else:
                    print("Invalid gain command format.")
            else:
                try:
                    if cmd.isdigit():
                        idx = int(cmd) - 1
                        if 0 <= idx < len(self.controller.stations_list):
                            new_freq = self.controller.stations_list[idx][1]
                            self.controller.sdr_receiver.set_center_frequency(new_freq)
                            print(f"Tuned to {self.controller.stations_list[idx][0]} ({new_freq/1e6:.1f} MHz).")
                            self.controller.flush_data_queue()
                            self.controller.fm_demodulator.reset()
                            if self.controller.audio_output.recording:
                                self.controller.audio_output.stop_recording()
                        else:
                            print("Invalid station number.")
                    else:
                        freq_val = float(cmd)
                        new_freq = freq_val * 1e6
                        self.controller.sdr_receiver.set_center_frequency(new_freq)
                        print(f"Tuned to {new_freq/1e6:.1f} MHz.")
                        self.controller.flush_data_queue()
                        self.controller.fm_demodulator.reset()
                        if self.controller.audio_output.recording:
                            self.controller.audio_output.stop_recording()
                except ValueError:
                    print("Unknown command.")


# --------------------------------------------------
# FMReceiverController Class
# --------------------------------------------------
class FMReceiverController:
    """
    FM Receiver Controller

    Integrates SDR reception, FM demodulation, audio output, and command input.
    The 'light' parameter selects between the standard and light demodulation versions.
    """
    def __init__(self, light=False):
        self.logger = logging.getLogger('fm_receiver.FMReceiverController')
        self.light = light
        self.quit_event = threading.Event()
        # Predefined station list (station name and frequency)
        self.stations = {
            "bayfm":        78.0e6,
            "NACK5":        79.5e6,
            "TOKYO FM":     80.0e6,
            "J-WAVE":       81.3e6,
            "NHK-FM":       82.5e6,
            "Fm yokohama":  84.7e6,
            "InterFM":      89.7e6,
            "JOKR":         90.5e6,
            "JOQR":         91.6e6,
            "JOLF":         93.0e6,
        }
        self.stations_list = sorted(self.stations.items(), key=lambda x: x[1])

        try:
            # Select demodulator version based on 'light' parameter
            if self.light:
                self.logger.info("Initializing FM Receiver in Light mode")
                # Initialize SDR receiver
                self.sdr_receiver = SDRReceiver(sample_rate=0.25e6, center_freq=80e6)
                self.fm_demodulator = FMDemodulatorLight(
                    iq_sample_rate=self.sdr_receiver.sample_rate,
                    final_audio_rate=48000,
                    stereo=False
                )
            else:
                self.logger.info("Initializing FM Receiver in Standard mode")
                # Initialize SDR receiver
                self.sdr_receiver = SDRReceiver(sample_rate=1.024e6, center_freq=80e6)
                self.fm_demodulator = FMDemodulator(
                    iq_sample_rate=self.sdr_receiver.sample_rate,
                    final_audio_rate=48000,
                    stereo=True
                )
            # AudioOutput instance manages its own internal queue
            self.audio_output = AudioOutput(output_rate=48000, frames_per_buffer=1024)
            # Start command line interface
            self.cmd_interface = CommandLineInterface(self)
            self.threads = []
            self.logger.info("FM Receiver Controller initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize FM Receiver Controller: {e}", exc_info=True)
            raise

    def flush_data_queue(self):
        """Clear any unprocessed samples from the SDR data queue."""
        while not self.sdr_receiver.data_queue.empty():
            try:
                self.sdr_receiver.data_queue.get_nowait()
            except queue.Empty:
                break

    def processing_thread(self):
        """
        Retrieve IQ samples from SDR, perform FM demodulation and audio conversion,
        then add the resulting audio data to the output queue via AudioOutput.
        """
        self.logger.info("Processing thread started")
        try:
            while not self.quit_event.is_set():
                try:
                    iq_samples = self.sdr_receiver.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error getting IQ samples from queue: {e}")
                    continue

                try:
                    composite = self.fm_demodulator.process_iq_samples(iq_samples)
                    left, right = self.fm_demodulator.demodulate(composite)
                    # Use AudioOutput method to enqueue audio data
                    self.audio_output.enqueue_audio(left, right)

                    # Check recording status with lock
                    with self.audio_output.record_lock:
                        is_recording = self.audio_output.recording

                    if is_recording:
                        stereo = np.empty((len(left) * 2,), dtype=np.float32)
                        stereo[0::2] = left
                        stereo[1::2] = right
                        self.audio_output.record(stereo)
                except Exception as e:
                    self.logger.error(f"Error in processing thread: {e}", exc_info=True)
                    # Continue processing even if one block fails
                    continue
        except Exception as e:
            self.logger.critical(f"Fatal error in processing thread: {e}", exc_info=True)
        finally:
            self.logger.info("Processing thread stopped")

    def start(self):
        """Start all threads and begin the main loop."""
        try:
            self.logger.info("Starting FM Receiver Controller")
            self.cmd_interface.start()

            sdr_thread = threading.Thread(target=self.sdr_receiver.start, daemon=True)
            sdr_thread.start()
            self.threads.append(sdr_thread)

            proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
            proc_thread.start()
            self.threads.append(proc_thread)

            if self.light:
                print("FM Receiver (Light) started.")
                print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
            else:
                print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Composite: {self.fm_demodulator.composite_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
                print(f"Default station: {self.sdr_receiver.get_center_frequency()/1e6:.1f} MHz")
                print("Stereo demodulation enabled.")
                print("Commands: q, list, <freq>, stereo on/off, record start/stop, agc on/off, gain <value>, etc.")

            self.logger.info("FM Receiver started successfully, entering main loop")

            try:
                while not self.quit_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                self.quit_event.set()
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                self.quit_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources."""
        try:
            self.logger.info("Cleaning up FM Receiver Controller")
            self.sdr_receiver.stop()
            self.audio_output.cleanup()
            self.logger.info("FM Receiver cleanup completed")
            print("Exiting FM Receiver.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            print("Error during cleanup - see log for details.")


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == '__main__':
    # Parse command line arguments
    light_mode = False
    enable_logging = False
    log_level = logging.INFO
    log_file = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--light':
            light_mode = True
        elif arg in ('--log', '--verbose', '-v'):
            enable_logging = True
        elif arg == '--debug':
            enable_logging = True
            log_level = logging.DEBUG
        elif arg == '--log-file' and i + 1 < len(sys.argv):
            log_file = sys.argv[i + 1]
            enable_logging = True

    # Setup logging only if requested
    if enable_logging:
        setup_logging(log_level=log_level, log_file=log_file)
        logger.info("=" * 60)
        logger.info("FM Receiver System Starting")
        logger.info("=" * 60)
    else:
        # Disable all logging by setting to CRITICAL+1
        logging.disable(logging.CRITICAL)

    try:
        controller = FMReceiverController(light=light_mode)
        controller.start()
    except Exception as e:
        if enable_logging:
            logger.critical(f"Failed to start FM Receiver: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        if enable_logging:
            print("Check the log for details.")
        sys.exit(1)
