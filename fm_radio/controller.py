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
from fm_radio.auto_gain import AutoGainController
from fm_radio.exceptions import SDRDeviceError, AudioOutputError
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_SAMPLE_RATE_LIGHT, SDR_CENTER_FREQ_DEFAULT,
    AUDIO_OUTPUT_RATE, AUDIO_FRAMES_PER_BUFFER,
)


# Per-block timing budget. SDR delivers a 16384-sample block every
# ~16 ms (at 1.024 Msps) so any block taking longer than that risks
# backing up the SDR data_queue.
_BLOCK_BUDGET_SEC: float = 0.016
# Slow-block log threshold: log immediately if a block exceeds this.
_SLOW_BLOCK_THRESHOLD_SEC: float = 0.020
# Periodic summary interval (real time, seconds).
_PROFILE_SUMMARY_INTERVAL_SEC: float = 60.0


class _BlockProfiler:
    """Lightweight per-block timing profiler for the processing loop.

    Tracks per-block processing time and SDR queue depth.  Logs a
    summary every ``_PROFILE_SUMMARY_INTERVAL_SEC`` and warns
    immediately on any block exceeding ``_SLOW_BLOCK_THRESHOLD_SEC``.
    """

    def __init__(self, logger: logging.Logger, q_max_capacity: int) -> None:
        self._log = logger
        self._q_capacity = q_max_capacity
        self._t0_session = time.perf_counter()
        self._t_last_summary = self._t0_session
        # Window stats (reset every summary)
        self._win_blocks = 0
        self._win_sum_dt = 0.0
        self._win_max_dt = 0.0
        self._win_slow_blocks = 0
        self._win_q_max = 0
        # Cumulative
        self._tot_blocks = 0
        self._tot_slow_blocks = 0

    def record(
        self, dt_sec: float, q_depth: int,
        stage_times: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        """Record one block's timing.

        ``stage_times`` is ``(agc, process_iq, demodulate, enqueue, record)``
        in seconds and is included in the SLOW BLOCK warning so the
        offending phase can be identified post-mortem.
        """
        self._win_blocks += 1
        self._tot_blocks += 1
        self._win_sum_dt += dt_sec
        if dt_sec > self._win_max_dt:
            self._win_max_dt = dt_sec
        if q_depth > self._win_q_max:
            self._win_q_max = q_depth
        if dt_sec >= _SLOW_BLOCK_THRESHOLD_SEC:
            self._win_slow_blocks += 1
            self._tot_slow_blocks += 1
            elapsed = time.perf_counter() - self._t0_session
            stages = ""
            if stage_times is not None:
                ag, pi, dm, eq, rc = stage_times
                stages = (
                    f" stages_ms=[agc:{ag*1000:.1f} "
                    f"process_iq:{pi*1000:.1f} demod:{dm*1000:.1f} "
                    f"enqueue:{eq*1000:.1f} record:{rc*1000:.1f}]"
                )
            self._log.warning(
                "BlockProfile: SLOW BLOCK dt=%.1fms q_depth=%d/%d "
                "session_t=%.1fs (%.2fmin) total_slow=%d%s",
                dt_sec * 1000.0, q_depth, self._q_capacity,
                elapsed, elapsed / 60.0, self._tot_slow_blocks,
                stages,
            )

        now = time.perf_counter()
        if now - self._t_last_summary >= _PROFILE_SUMMARY_INTERVAL_SEC:
            blocks = max(self._win_blocks, 1)
            avg_ms = self._win_sum_dt * 1000.0 / blocks
            elapsed = now - self._t0_session
            self._log.info(
                "BlockProfile: t=%.0fs (%.1fmin) blocks=%d avg=%.2fms "
                "max=%.1fms slow_in_window=%d q_max=%d/%d total_slow=%d",
                elapsed, elapsed / 60.0,
                self._win_blocks, avg_ms, self._win_max_dt * 1000.0,
                self._win_slow_blocks, self._win_q_max, self._q_capacity,
                self._tot_slow_blocks,
            )
            self._t_last_summary = now
            self._win_blocks = 0
            self._win_sum_dt = 0.0
            self._win_max_dt = 0.0
            self._win_slow_blocks = 0
            self._win_q_max = 0


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
            # Auto gain controller (replaces hardware AGC)
            self.auto_gain: AutoGainController = AutoGainController(self.sdr_receiver)
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
        self.auto_gain.reset_counters()
        if self.audio_output.recording:
            self.audio_output.stop_recording()
        if self.sdr_receiver.iq_recording:
            self.sdr_receiver.stop_iq_recording()

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
        # Capture context for the metadata sidecar; the audio subsystem
        # itself does not know the tuner state.
        try:
            metadata = {
                "center_freq_hz": float(self.sdr_receiver.get_center_frequency()),
                "gain_db": float(self.sdr_receiver.get_gain()),
            }
        except Exception:
            metadata = None
        self.audio_output.start_recording(filename, metadata=metadata)

    def stop_recording(self) -> None:
        """Stop the current recording session."""
        self.audio_output.stop_recording()

    def is_recording(self) -> bool:
        """Return True if currently recording."""
        return self.audio_output.recording

    def start_iq_recording(self, filename: str) -> None:
        """Start recording raw IQ samples to a 2-channel WAV file."""
        self.sdr_receiver.start_iq_recording(filename)

    def stop_iq_recording(self) -> None:
        """Stop the current IQ recording session."""
        self.sdr_receiver.stop_iq_recording()

    def is_iq_recording(self) -> bool:
        """Return True if raw IQ recording is active."""
        return self.sdr_receiver.iq_recording

    def set_agc_mode(self, enabled: bool) -> None:
        """Enable or disable automatic gain control.

        When enabled, the auto gain controller monitors IQ peak
        amplitude and adjusts RTL-SDR hardware gain automatically.
        When disabled, the user controls gain manually via CLI.

        Args:
            enabled: True to enable auto gain, False for manual mode.
        """
        if enabled:
            self.auto_gain.enable()
        else:
            self.auto_gain.disable()

    def get_gain(self) -> float:
        """Return the current gain value in dB."""
        return self.sdr_receiver.get_gain()

    def set_gain(self, gain: float) -> None:
        """Set the manual gain value in dB.

        Only effective when auto gain is disabled.

        Args:
            gain: Gain value in dB.
        """
        self.auto_gain.set_gain_manual(gain)

    def is_manual_gain(self) -> bool:
        """Return True if manual gain mode is active (auto gain disabled)."""
        return not self.auto_gain.enabled

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

    def _prewarm_jit(self) -> None:
        """Trigger Numba JIT compile and FFT plan caches for hot demod paths.

        Without this the very first SDR block takes ~1.3 s to process
        (Numba compiling pll_demodulate / deemphasis_iir, scipy / numpy
        building FFT plans, scipy.signal.resample_poly designing its FIR
        filter).  That single pause overflows the SDR data_queue and
        loses the leading ~50 ms of audio.  We trigger the same code
        paths here, before the SDR async thread starts, so JIT cost is
        paid against an idle queue rather than a live RF stream.

        Several iterations are performed so caches that are populated
        on the second call (some scipy plans) are also warm by the time
        real samples arrive.  The demodulator state is reset afterwards
        so the first real block starts from scratch.
        """
        self.logger.info("JIT pre-warming demodulator paths...")
        t0 = time.perf_counter()
        block_size = self.sdr_receiver.block_size
        dummy_iq = np.zeros(block_size, dtype=np.complex64)
        try:
            for _ in range(3):
                composite = self.fm_demodulator.process_iq_samples(dummy_iq)
                self.fm_demodulator.demodulate(composite)
            self.fm_demodulator.reset()
        except Exception as e:
            # Pre-warm should never fail the boot — the demod will still
            # JIT lazily on first real sample.
            self.logger.warning(
                f"JIT pre-warm failed (non-fatal, falling back to lazy "
                f"compile): {e}",
                exc_info=True,
            )
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.logger.info(f"JIT pre-warming complete in {dt_ms:.1f} ms")

    def processing_thread(self) -> None:
        """Retrieve IQ samples from SDR, perform FM demodulation and audio conversion,
        then add the resulting audio data to the output queue via AudioOutput.
        """
        self.logger.info("Processing thread started")
        profiler = _BlockProfiler(
            self.logger, self.sdr_receiver.data_queue.maxsize,
        )
        try:
            while not self.quit_event.is_set():
                try:
                    iq_samples = self.sdr_receiver.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error getting IQ samples from queue: {e}")
                    continue

                # Snapshot queue depth at the moment we pulled this block.
                q_depth_after_get = self.sdr_receiver.data_queue.qsize()
                t_block_start = time.perf_counter()
                t_agc = t_proc = t_demod = t_enq = t_rec = t_block_start

                try:
                    # Auto gain adjustment (before demodulation)
                    self.auto_gain.update(iq_samples)
                    t_agc = time.perf_counter()

                    composite = self.fm_demodulator.process_iq_samples(iq_samples)
                    t_proc = time.perf_counter()

                    left, right = self.fm_demodulator.demodulate(composite)
                    t_demod = time.perf_counter()

                    # Use AudioOutput method to enqueue audio data
                    self.audio_output.enqueue_audio(left, right)
                    t_enq = time.perf_counter()

                    # Check recording status without acquiring record_lock:
                    # the worker thread holds record_lock for the duration
                    # of each writeframes() (which is exactly the disk
                    # stall we are trying to keep off this thread).  A
                    # bool read is atomic in CPython, and audio_output.
                    # record() re-validates self.recording internally
                    # before any further work.
                    if self.audio_output.recording:
                        stereo = np.empty((len(left) * 2,), dtype=np.float32)
                        stereo[0::2] = left
                        stereo[1::2] = right
                        self.audio_output.record(stereo)
                    t_rec = time.perf_counter()
                except Exception as e:
                    self.logger.error(f"Error in processing thread: {e}", exc_info=True)
                    # Continue processing even if one block fails
                    continue
                finally:
                    t_end = time.perf_counter()
                    profiler.record(
                        t_end - t_block_start,
                        q_depth_after_get,
                        stage_times=(
                            t_agc - t_block_start,
                            t_proc - t_agc,
                            t_demod - t_proc,
                            t_enq - t_demod,
                            t_rec - t_enq,
                        ),
                    )
        except Exception as e:
            self.logger.critical(f"Fatal error in processing thread: {e}", exc_info=True)
        finally:
            self.logger.info("Processing thread stopped")

    def start(self) -> None:
        """Start all threads and begin the main loop."""
        try:
            self.logger.info("Starting FM Receiver Controller")

            # Pre-compile Numba / FFT paths before the SDR delivers
            # samples so the first block does not stall the realtime
            # path while JIT compilation runs.
            self._prewarm_jit()

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
                print("Commands: q, list, <freq>, stereo on/off, record start/stop, iqrec start/stop, agc on/off, gain <value>, etc.")
            print("Auto gain control: ON")

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
            self.auto_gain.stop()
            self.logger.info("FM Receiver cleanup completed")
            print("Exiting FM Receiver.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            print("Error during cleanup - see log for details.")
