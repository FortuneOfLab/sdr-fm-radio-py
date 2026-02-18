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
"""FM Receiver Controller - integrates all subsystems."""

from __future__ import annotations

import queue
import time
import threading
import logging

import numpy as np

from fm_radio.sdr_receiver import SDRReceiver
from fm_radio.demodulator import FMDemodulator, FMDemodulatorLight
from fm_radio.audio_output import AudioOutput
from fm_radio.cli import CommandLineInterface
from fm_radio.exceptions import SDRDeviceError, AudioOutputError
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_SAMPLE_RATE_LIGHT, SDR_CENTER_FREQ_DEFAULT,
    AUDIO_OUTPUT_RATE, AUDIO_FRAMES_PER_BUFFER,
)


class FMReceiverController:
    """
    FM Receiver Controller

    Integrates SDR reception, FM demodulation, audio output, and command input.
    The 'light' parameter selects between the standard and light demodulation versions.
    """
    def __init__(self, light: bool = False) -> None:
        self.logger: logging.Logger = logging.getLogger('fm_receiver.FMReceiverController')
        self.light: bool = light
        self.quit_event: threading.Event = threading.Event()
        # Predefined station list (station name and frequency)
        self.stations: dict[str, float] = {
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
        self.stations_list: list[tuple[str, float]] = sorted(
            self.stations.items(), key=lambda x: x[1],
        )

        try:
            # Select demodulator version based on 'light' parameter
            if self.light:
                self.logger.info("Initializing FM Receiver in Light mode")
                # Initialize SDR receiver
                self.sdr_receiver = SDRReceiver(sample_rate=SDR_SAMPLE_RATE_LIGHT, center_freq=SDR_CENTER_FREQ_DEFAULT)
                self.fm_demodulator: FMDemodulator | FMDemodulatorLight = FMDemodulatorLight(
                    iq_sample_rate=self.sdr_receiver.sample_rate,
                    final_audio_rate=AUDIO_OUTPUT_RATE,
                    stereo=False
                )
            else:
                self.logger.info("Initializing FM Receiver in Standard mode")
                # Initialize SDR receiver
                self.sdr_receiver = SDRReceiver(sample_rate=SDR_SAMPLE_RATE, center_freq=SDR_CENTER_FREQ_DEFAULT)
                self.fm_demodulator = FMDemodulator(
                    iq_sample_rate=self.sdr_receiver.sample_rate,
                    final_audio_rate=AUDIO_OUTPUT_RATE,
                    stereo=True
                )
            # AudioOutput instance manages its own internal queue
            self.audio_output: AudioOutput = AudioOutput(
                output_rate=AUDIO_OUTPUT_RATE, frames_per_buffer=AUDIO_FRAMES_PER_BUFFER,
            )
            # Start command line interface
            self.cmd_interface: CommandLineInterface = CommandLineInterface(self)
            self.threads: list[threading.Thread] = []
            self.logger.info("FM Receiver Controller initialized successfully")
        except (SDRDeviceError, AudioOutputError) as e:
            self.logger.error(f"Failed to initialize FM Receiver Controller: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Facade API — public interface for CLI and external consumers
    # ------------------------------------------------------------------

    def get_stations_list(self) -> list[tuple[str, float]]:
        """Return the sorted list of (station_name, frequency_hz) tuples."""
        return self.stations_list

    def tune(self, freq_hz: float) -> None:
        """Tune to a new frequency.

        Sets the SDR center frequency, flushes stale IQ data, resets
        demodulator state, and stops any active recording.

        Args:
            freq_hz: Target frequency in Hz.
        """
        self.sdr_receiver.set_center_frequency(freq_hz)
        self._flush_data_queue()
        self.fm_demodulator.reset()
        if self.audio_output.recording:
            self.audio_output.stop_recording()

    def get_frequency(self) -> float:
        """Return the current center frequency in Hz."""
        return self.sdr_receiver.get_center_frequency()

    def set_stereo(self, enabled: bool) -> bool:
        """Set stereo/mono demodulation mode.

        Args:
            enabled: True for stereo, False for mono.

        Returns:
            True if stereo mode is supported, False otherwise.
        """
        if hasattr(self.fm_demodulator, 'stereo'):
            self.fm_demodulator.stereo = enabled
            return True
        return False

    def start_recording(self, filename: str) -> None:
        """Start recording audio to a WAV file.

        Args:
            filename: Output WAV file path.
        """
        self.audio_output.start_recording(filename)

    def stop_recording(self) -> None:
        """Stop the current recording session."""
        self.audio_output.stop_recording()

    def is_recording(self) -> bool:
        """Return True if currently recording."""
        return self.audio_output.recording

    def set_agc_mode(self, enabled: bool) -> None:
        """Enable or disable automatic gain control.

        Args:
            enabled: True to enable AGC, False for manual gain mode.
        """
        self.sdr_receiver.set_manual_gain_mode(not enabled)

    def get_gain(self) -> float:
        """Return the current gain value in dB."""
        return self.sdr_receiver.get_gain()

    def set_gain(self, gain: float) -> None:
        """Set the manual gain value in dB.

        Args:
            gain: Gain value in dB.
        """
        self.sdr_receiver.set_gain(gain)

    def is_manual_gain(self) -> bool:
        """Return True if manual gain mode is active."""
        return self.sdr_receiver.manual_gain

    def set_software_agc(self, enabled: bool) -> None:
        """Enable or disable software AGC in the demodulator.

        Args:
            enabled: True to enable, False to disable.
        """
        self.fm_demodulator.agc_enabled = enabled

    def is_software_agc(self) -> bool:
        """Return True if software AGC is enabled."""
        return self.fm_demodulator.agc_enabled

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _flush_data_queue(self) -> None:
        """Clear any unprocessed samples from the SDR data queue."""
        while not self.sdr_receiver.data_queue.empty():
            try:
                self.sdr_receiver.data_queue.get_nowait()
            except queue.Empty:
                break

    def processing_thread(self) -> None:
        """Retrieve IQ samples from SDR, perform FM demodulation and audio conversion,
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

    def start(self) -> None:
        """Start all threads and begin the main loop."""
        try:
            self.logger.info("Starting FM Receiver Controller")

            # Explicitly enable both AGC layers at startup
            self.set_agc_mode(True)
            self.set_software_agc(True)

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
                print("Commands: q, list, <freq>, stereo on/off, record start/stop, agc on/off, sagc on/off, gain <value>, etc.")
            print("Hardware AGC: ON, Software AGC: ON")

            # Start CLI thread after startup messages to avoid interleaving
            self.cmd_interface.start()

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

    def cleanup(self) -> None:
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
