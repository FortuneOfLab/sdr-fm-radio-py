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
  - Standard mode:
      $ python fm_receiver.py
  - Light mode:
      $ python fm_receiver.py --light

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
from fractions import Fraction

import numpy as np
import scipy.signal as signal
import pyaudio
import samplerate
from rtlsdr import RtlSdr
from numba import njit


# --------------------------------------------------
# Deemphasis IIR Filter Class
# --------------------------------------------------
class DeemphasisIIRFilter:
    """
    IIR filter for de-emphasis processing in FM broadcasting

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
        Apply de-emphasis processing to the input signal.

        Args:
            x (ndarray): Input audio signal array.

        Returns:
            ndarray: Audio signal after filtering.
        """
        y = np.empty_like(x)
        for i in range(len(x)):
            y[i] = (1 - self.alpha) * x[i] + self.alpha * self.prev_output
            self.prev_output = y[i]
        return y


# --------------------------------------------------
# Filter Classes (LowpassFilter & BandpassFilter)
# --------------------------------------------------
class LowpassFilter:
    """
    Butterworth lowpass filter

    Passes frequency components below the specified cutoff frequency.
    """
    def __init__(self, order, cutoff, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, cutoff / nyquist, btype="low", analog=False)

    def apply(self, data):
        """
        Apply the filter to the signal data.

        Args:
            data (ndarray): Signal data before filtering.

        Returns:
            ndarray: Filtered signal data.
        """
        return signal.filtfilt(self.b, self.a, data)


class BandpassFilter:
    """
    Butterworth bandpass filter

    Passes only the frequency components within the specified range.
    """
    def __init__(self, order, lowcut, highcut, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, [lowcut / nyquist, highcut / nyquist],
                                        btype="band", analog=False)

    def apply(self, data):
        """
        Apply the filter to the signal data.

        Args:
            data (ndarray): Input signal.

        Returns:
            ndarray: Signal after bandpass filtering.
        """
        return signal.filtfilt(self.b, self.a, data)


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
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.block_size = block_size
        self.data_queue = queue.Queue(maxsize=20)

        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rate
        self.sdr.center_freq = self.center_freq
        self.sdr.set_manual_gain_enabled(False)
        self.manual_gain = True
        self.sdr.set_gain(0)
        try:
            # Disable direct_sampling if available
            self.sdr.direct_sampling = 0
        except Exception as e:
            print("Failed to disable direct_sampling:", e)

    def set_center_frequency(self, freq):
        """Change the center frequency."""
        self.center_freq = freq
        self.sdr.center_freq = freq

    def get_center_frequency(self):
        """Retrieve the current center frequency."""
        return self.sdr.center_freq

    def set_gain(self, gain):
        """Set gain value (for manual mode)."""
        self.sdr.set_gain(gain)

    def get_gain(self):
        """Retrieve the current gain value."""
        return self.sdr.get_gain()

    def set_manual_gain_mode(self, manual):
        """
        Set manual gain mode.

        Args:
            manual (bool): True for manual mode, False for AGC.
        """
        self.manual_gain = manual
        self.sdr.set_manual_gain_enabled(manual)

    def callback(self, iq_samples, sdr_obj):
        """
        Callback to store received IQ samples in the data queue.

        Args:
            iq_samples (ndarray): Received IQ samples.
            sdr_obj: SDR object (unused).
        """
        try:
            self.data_queue.put(iq_samples, block=False)
        except queue.Full:
            # Discard sample if queue is full.
            pass

    def start(self):
        """Start asynchronous sample retrieval."""
        self.sdr.read_samples_async(self.callback, num_samples=self.block_size)

    def stop(self):
        """Stop asynchronous sample retrieval and close SDR."""
        try:
            self.sdr.cancel_read_async()
        except Exception:
            pass
        self.sdr.close()


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
        self.iq_sample_rate = iq_sample_rate
        self.composite_rate = composite_rate
        self.final_audio_rate = final_audio_rate
        self.stereo = stereo

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

    def process_iq_samples(self, iq_samples):
        """
        Apply DC offset correction, lowpass filtering, PLL demodulation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
        """
        self.dc_offset = self.dc_alpha * np.mean(iq_samples) + (1 - self.dc_alpha) * self.dc_offset
        iq_processed = iq_samples - self.dc_offset
        iq_filtered = signal.lfilter(self.iq_b, self.iq_a, iq_processed)
        main_output = self.main_pll.process(iq_filtered)
        composite = signal.resample_poly(main_output, up=self.up, down=self.down)
        return composite

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

        left_48 = samplerate.resample(left_channel, 0.25, converter_type="sinc_best")
        right_48 = samplerate.resample(right_channel, 0.25, converter_type="sinc_best")
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
        mono_48 = samplerate.resample(mono, 0.25, converter_type="sinc_best")
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
        self.iq_sample_rate = iq_sample_rate
        self.composite_rate = composite_rate
        self.final_audio_rate = final_audio_rate
        self.stereo = stereo

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

    def process_iq_samples(self, iq_samples):
        """
        Apply DC offset correction, phase extraction/differentiation,
        and resampling to generate the composite signal.

        Args:
            iq_samples (ndarray): Input IQ samples.

        Returns:
            ndarray: Composite signal after resampling.
        """
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
        return composite

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
        left_48 = samplerate.resample(left_channel, 0.25, converter_type="sinc_fastest")
        right_48 = samplerate.resample(right_channel, 0.25, converter_type="sinc_fastest")
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
        mono_48 = samplerate.resample(mono, 0.25, converter_type="sinc_fastest")
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
    def __init__(self, output_rate=48000):
        self.output_rate = output_rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=int(self.output_rate),
            output=True
        )
        self.recording = False
        self.record_wave = None
        self.record_lock = threading.Lock()

    def play(self, left, right):
        """
        Combine left and right channels and play the audio.

        Args:
            left (ndarray): Left channel audio.
            right (ndarray): Right channel audio.
        """
        stereo = np.empty((len(left) * 2,), dtype=np.float32)
        stereo[0::2] = left
        stereo[1::2] = right
        try:
            self.stream.write(stereo.tobytes())
        except Exception as e:
            print("Audio output error:", e)

    def start_recording(self, filename, channels=2):
        """
        Start recording audio.

        Args:
            filename (str): Filename to save the WAV file.
            channels (int): Number of channels.
        """
        with self.record_lock:
            if self.recording:
                print("Already recording.")
                return
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(int(self.output_rate))
                self.record_wave = wf
                self.recording = True
                print(f"Recording started: {filename}")
            except Exception as e:
                print("Recording start failed:", e)

    def stop_recording(self):
        """Stop recording and close the file."""
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                try:
                    self.record_wave.close()
                except Exception:
                    pass
                self.record_wave = None
                self.recording = False
                print("Recording stopped.")
            else:
                print("Not currently recording.")

    def record(self, stereo_audio):
        """
        Write audio data to file if recording.

        Args:
            stereo_audio (ndarray): Stereo audio data.
        """
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                int16_audio = np.int16(stereo_audio * 32767)
                self.record_wave.writeframes(int16_audio.tobytes())

    def cleanup(self):
        """Stop audio stream and terminate PyAudio instance."""
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()


# --------------------------------------------------
# AudioOutputThread Class
# --------------------------------------------------
class AudioOutputThread(threading.Thread):
    """
    Background thread for audio playback

    Retrieves data from the audio buffer and plays it using AudioOutput.
    """
    def __init__(self, audio_output, audio_buffer_queue, update_interval=0.01):
        super().__init__(daemon=True)
        self.audio_output = audio_output
        self.audio_buffer_queue = audio_buffer_queue
        self.update_interval = update_interval
        self.quit_flag = False

    def run(self):
        while not self.quit_flag:
            try:
                left, right = self.audio_buffer_queue.get(timeout=self.update_interval)
                self.audio_output.play(left, right)
            except queue.Empty:
                continue

    def stop(self):
        """Set flag to stop the thread."""
        self.quit_flag = True


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
# FMReceiverController Class (Unified Version)
# --------------------------------------------------
class FMReceiverController:
    """
    FM Receiver Controller

    Integrates SDR reception, FM demodulation, audio output, and command input.
    The 'light' parameter selects between the standard and light demodulation versions.
    """
    def __init__(self, light=False):
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
        # Select demodulator version based on 'light' parameter
        if self.light:
            # Initialize SDR receiver
            self.sdr_receiver = SDRReceiver(sample_rate=0.25e6, center_freq=80e6)
            self.fm_demodulator = FMDemodulatorLight(
                iq_sample_rate=self.sdr_receiver.sample_rate,
                final_audio_rate=48000,
                stereo=False
            )
        else:
            # Initialize SDR receiver
            self.sdr_receiver = SDRReceiver(sample_rate=1.024e6, center_freq=80e6)
            self.fm_demodulator = FMDemodulator(
                iq_sample_rate=self.sdr_receiver.sample_rate,
                final_audio_rate=48000,
                stereo=True
            )
        # Initialize audio output
        self.audio_output = AudioOutput(output_rate=48000)
        self.audio_buffer_queue = queue.Queue(maxsize=50)
        # Start command line interface
        self.cmd_interface = CommandLineInterface(self)
        # Initialize audio output thread
        self.audio_output_thread = AudioOutputThread(self.audio_output, self.audio_buffer_queue, update_interval=0.01)
        self.threads = []

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
        then send the audio data to the output buffer.
        """
        while not self.quit_event.is_set():
            try:
                iq_samples = self.sdr_receiver.data_queue.get(timeout=1)
            except queue.Empty:
                continue
            composite = self.fm_demodulator.process_iq_samples(iq_samples)
            left, right = self.fm_demodulator.demodulate(composite)
            try:
                self.audio_buffer_queue.put((left, right), timeout=0.01)
            except queue.Full:
                continue
            if self.audio_output.recording:
                stereo = np.empty((len(left) * 2,), dtype=np.float32)
                stereo[0::2] = left
                stereo[1::2] = right
                self.audio_output.record(stereo)

    def start(self):
        """Start all threads and begin the main loop."""
        self.cmd_interface.start()

        sdr_thread = threading.Thread(target=self.sdr_receiver.start, daemon=True)
        sdr_thread.start()
        self.threads.append(sdr_thread)

        proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        proc_thread.start()
        self.threads.append(proc_thread)

        self.audio_output_thread.start()

        if self.light:
            print("FM Receiver (Light) started.")
            print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
        else:
            print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Composite: {self.fm_demodulator.composite_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
            print(f"Default station: {self.sdr_receiver.get_center_frequency()/1e6:.1f} MHz")
            print("Stereo demodulation enabled.")
            print("Commands: q, list, <freq>, stereo on/off, record start/stop, agc on/off, gain <value>, etc.")

        try:
            while not self.quit_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.quit_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources."""
        self.audio_output_thread.stop()
        self.sdr_receiver.stop()
        self.audio_output.cleanup()
        print("Exiting FM Receiver.")


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--light':
        controller = FMReceiverController(light=True)
    else:
        controller = FMReceiverController()
    controller.start()
