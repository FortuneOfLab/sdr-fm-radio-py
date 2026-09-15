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
import sys
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
from fm_radio.stations import (
    Station, load_stations, favorites, search, in_area, nearest,
)
from fm_radio.telemetry import StatusSnapshot, TelemetryPublisher, peak_dbfs
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
# Least time between warnings about a snapshot that will not build.  A
# failure that persists is one problem, not one problem per block.
_TELEMETRY_WARN_INTERVAL_SEC: float = 60.0
# How long cleanup waits for a thread to notice it should stop.  The
# processing thread blocks on the SDR queue for at most a second, so this
# only has to outlast that; a thread that is still going after it is not
# going to be waited for indefinitely.
_THREAD_JOIN_TIMEOUT_SEC: float = 3.0


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

    @property
    def window_avg_ms(self) -> float:
        """Mean block time since the last summary, in milliseconds."""
        return self._win_sum_dt * 1000.0 / max(self._win_blocks, 1)

    @property
    def window_max_ms(self) -> float:
        """Longest block since the last summary, in milliseconds."""
        return self._win_max_dt * 1000.0

    @property
    def slow_blocks(self) -> int:
        """Blocks over the slow-block threshold since the receiver started."""
        return self._tot_slow_blocks

    @property
    def uptime_sec(self) -> float:
        """Seconds since the processing thread started."""
        return time.perf_counter() - self._t0_session

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
    def __init__(self, light: bool = False,
                 stations_path: str | None = None) -> None:
        self.logger: logging.Logger = logging.getLogger('fm_receiver.FMReceiverController')
        self.light: bool = light
        self.quit_event: threading.Event = threading.Event()
        # Nationwide catalogue (bundled snapshot + the user's stations.toml)
        # and the short preset list the CLI tunes by number.  Loading
        # problems are printed as well as logged: logging is off unless
        # --log was passed, and a station file that was silently ignored is
        # exactly the kind of thing the user needs to hear about.
        self.catalogue: list[Station] = load_stations(
            stations_path, warn=self._warn_station_config)
        self.presets: list[Station] = favorites(self.catalogue)
        if not self.catalogue:
            self.logger.warning(
                "Station catalogue is empty; tune by frequency instead")
        else:
            self.logger.info("Station catalogue: %d transmitters, %d presets",
                             len(self.catalogue), len(self.presets))

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
            # Latest-value slot the processing thread publishes state to;
            # see fm_radio.telemetry for why it is rate-limited rather than
            # written every block.
            # The generation a snapshot is judged against comes from the
            # tuner, so a block captured before a retune can never be
            # published as the state of what the receiver moved to.
            self.telemetry: TelemetryPublisher = TelemetryPublisher(
                current_generation=lambda: self.sdr_receiver.tuning_generation)
            # Naming the tuned station means scanning the catalogue, which
            # only has a different answer when the frequency changes.  The
            # cache is written and read on the processing thread only.
            self._station_name_cache: tuple[float, str] = (float("nan"), "")
            # Rate limiting for the warning about a snapshot that will not
            # build; both are touched only from the processing thread.
            self._telemetry_failures: int = 0
            self._telemetry_last_warn: float | None = None
            # Start command line interface
            self.cmd_interface: CommandLineInterface = CommandLineInterface(self)
            self.threads: list[threading.Thread] = []
            self.logger.info("FM Receiver Controller initialized successfully")
        except (SDRDeviceError, AudioOutputError) as e:
            self.logger.error(f"Failed to initialize FM Receiver Controller: {e}", exc_info=True)
            raise

    @staticmethod
    def _warn_station_config(message: str) -> None:
        """Put a station-list problem in front of the user, log or no log."""
        print(f"Station list: {message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Facade API — public interface for CLI and external consumers
    # ------------------------------------------------------------------

    def get_stations_list(self) -> list[tuple[str, float]]:
        """Return the preset stations as (station_name, frequency_hz) tuples.

        These are the favourites — the entries the CLI lists and tunes by
        number.  The full catalogue is available via :meth:`get_catalogue`.
        """
        return [(s.name, s.freq_hz) for s in self.presets]

    def get_catalogue(self) -> list[Station]:
        """Return every known transmitter, sorted by area then frequency."""
        return self.catalogue

    def search_stations(self, query: str) -> list[Station]:
        """Return catalogue entries matching *query*.

        Matches the brand name, the legal name, the transmitter site, the
        area and the frequency in MHz, case-insensitively.
        """
        return search(self.catalogue, query)

    def stations_in_area(self, area: str) -> list[Station]:
        """Return every catalogue entry in *area* (e.g. ``"関東"``)."""
        return in_area(self.catalogue, area)

    def current_station(self) -> Station | None:
        """Return the catalogue entry the tuner is currently sitting on."""
        return nearest(self.catalogue, self.get_frequency())

    def get_status(self) -> StatusSnapshot | None:
        """Return the current receiver state, or None if there is not one.

        The snapshot is produced by the processing thread at roughly 20 Hz
        and is a plain immutable value: reading it neither blocks that thread
        nor reaches into any of its objects.

        None means there is no current snapshot. That covers three
        situations, which this return value does not distinguish between:

        * Nothing has been published yet — the SDR is still starting, or the
          receiver was only just constructed.
        * Everything published so far was captured under a previous tuning,
          which retuning invalidates immediately.
        * Every attempt to build one has failed. A failure backs off by a
          publish interval and is reported in the log, so this is the one
          case that lasts.

        It clears on the next successful publish: normally within one publish
        interval of the receiver having audio to describe, 50 ms by default,
        but not on any guaranteed schedule.
        """
        return self.telemetry.latest

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

    def _publish_status(self, iq_samples: np.ndarray, left: np.ndarray,
                        right: np.ndarray, profiler: "_BlockProfiler",
                        block_dt_sec: float, q_depth: int, now: float,
                        generation: int) -> None:
        """Build and publish a snapshot, absorbing anything that goes wrong."""
        try:
            self.telemetry.publish(
                self._build_snapshot(iq_samples, left, right, profiler,
                                     block_dt_sec, q_depth, now),
                generation,
            )
        except Exception as e:
            # Telemetry is for looking at, never a reason to interrupt the
            # audio it is describing.  Back off on the same schedule as a
            # successful publish: a snapshot that cannot be built now will
            # not build on the next block either, and retrying every block
            # would turn one broken snapshot into a failure per block.
            self.telemetry.defer(now)
            self._report_telemetry_failure(e, now)
        else:
            # A snapshot got through, so the quiet period the last fault
            # earned is over: whatever fails next is a new fault and has to
            # be reported rather than hidden behind the old one.
            self._telemetry_last_warn = None

    def _report_telemetry_failure(self, exc: Exception, now: float) -> None:
        """Log a snapshot failure, at most once per warning interval."""
        self._telemetry_failures += 1
        last = self._telemetry_last_warn
        if last is not None and now - last < _TELEMETRY_WARN_INTERVAL_SEC:
            return
        self._telemetry_last_warn = now
        self.logger.warning(
            "Telemetry snapshot failed (%d since start): %s",
            self._telemetry_failures, exc, exc_info=True)

    def _station_name_for(self, freq_hz: float) -> str:
        """Name the station at *freq_hz*, reusing the last lookup.

        nearest() walks the whole catalogue - ~270 us over 983 transmitters,
        which is most of what a snapshot would otherwise cost - and the
        answer only changes when the tuner moves.
        """
        cached_freq, cached_name = self._station_name_cache
        if freq_hz == cached_freq:
            return cached_name
        station = nearest(self.catalogue, freq_hz)
        name = station.name if station else ""
        self._station_name_cache = (freq_hz, name)
        return name

    def _build_snapshot(self, iq_samples: np.ndarray, left: np.ndarray,
                        right: np.ndarray, profiler: "_BlockProfiler",
                        block_dt_sec: float, q_depth: int,
                        now: float) -> StatusSnapshot:
        """Capture the receiver's state for whatever is watching it.

        Called from the processing thread, only on a block where the
        publisher is due.  The three measurements taken here — IQ peak and
        the two audio levels — are the only work this adds to the realtime
        path; everything else is reading a value the receiver already keeps.
        """
        demod = self.fm_demodulator
        audio = self.audio_output
        sdr = self.sdr_receiver
        freq_hz = float(sdr.get_center_frequency())
        station_name = self._station_name_for(freq_hz)

        iq_peak = float(np.max(np.abs(iq_samples))) if iq_samples.size else 0.0

        return StatusSnapshot(
            freq_hz=freq_hz,
            station=station_name,
            gain_db=float(sdr.get_gain()),
            auto_gain=self.auto_gain.enabled,
            iq_peak=iq_peak,

            # Read directly rather than through getattr defaults: both
            # demodulators define these, and a default would turn a renamed
            # attribute into a snapshot that quietly reports zero.
            stereo=bool(demod.stereo),
            blend_factor=float(demod.blend_factor),
            pilot_snr_db=demod.pilot_snr_ema,
            pilot_jitter_db=float(demod.pilot_jitter_ema),
            side_nr_enabled=bool(demod.side_nr_enabled),

            level_left_dbfs=peak_dbfs(left),
            level_right_dbfs=peak_dbfs(right),

            block_ms=block_dt_sec * 1000.0,
            block_ms_avg=profiler.window_avg_ms,
            block_ms_max=profiler.window_max_ms,
            block_budget_ms=_BLOCK_BUDGET_SEC * 1000.0,
            sdr_queue=q_depth,
            sdr_queue_max=sdr.data_queue.maxsize,
            slow_blocks=profiler.slow_blocks,
            iq_drops=sdr.dropped_blocks,
            audio_drops=audio.dropped_blocks,
            audio_underruns=audio.underruns,

            recording_audio=audio.recording,
            recording_iq=sdr.iq_recording,

            uptime_sec=profiler.uptime_sec,
            timestamp=now,
        )

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
                    # The generation comes with the block rather than being
                    # read here: a retune between the two would tag samples
                    # from the old station as belonging to the new one.
                    generation, iq_samples = self.sdr_receiver.data_queue.get(
                        timeout=1)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error getting IQ samples from queue: {e}")
                    continue

                # Snapshot queue depth at the moment we pulled this block.
                q_depth_after_get = self.sdr_receiver.data_queue.qsize()
                t_block_start = time.perf_counter()
                t_agc = t_proc = t_demod = t_enq = t_rec = t_block_start
                block_ok = False

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
                    block_ok = True
                except Exception as e:
                    self.logger.error(f"Error in processing thread: {e}", exc_info=True)
                    # Continue processing even if one block fails
                    continue
                finally:
                    t_end = time.perf_counter()
                    block_dt = t_end - t_block_start
                    profiler.record(
                        block_dt,
                        q_depth_after_get,
                        stage_times=(
                            t_agc - t_block_start,
                            t_proc - t_agc,
                            t_demod - t_proc,
                            t_enq - t_demod,
                            t_rec - t_enq,
                        ),
                    )
                    # Only a block that made it all the way through has
                    # audio of its own to report; a failed one would be
                    # published carrying the previous block's levels against
                    # this block's timestamp.  Publishing is also
                    # rate-limited, so most blocks pay one comparison here
                    # and nothing else.
                    if block_ok and self.telemetry.due(t_end):
                        self._publish_status(
                            iq_samples, left, right, profiler, block_dt,
                            q_depth_after_get, t_end, generation,
                        )
        except Exception as e:
            self.logger.critical(f"Fatal error in processing thread: {e}", exc_info=True)
        finally:
            self.logger.info("Processing thread stopped")

    def start_background(self) -> None:
        """Start the SDR and processing threads and return.

        Everything :meth:`start` does except run a user interface, so a front
        end with its own event loop can take that part over.  The caller owns
        :meth:`cleanup` from here on.
        """
        self.logger.info("Starting FM Receiver Controller")

        # Pre-compile Numba / FFT paths before the SDR delivers samples so
        # the first block does not stall the realtime path while JIT
        # compilation runs.
        self._prewarm_jit()

        sdr_thread = threading.Thread(target=self.sdr_receiver.start, daemon=True)
        sdr_thread.start()
        self.threads.append(sdr_thread)

        proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        proc_thread.start()
        self.threads.append(proc_thread)

        self.logger.info("FM Receiver started successfully")

    def _announce(self) -> None:
        """Print the banner the command line starts with."""
        if self.light:
            print("FM Receiver (Light) started.")
            print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
        else:
            print(f"SDR sample_rate: {self.sdr_receiver.sample_rate:.0f} Hz, Composite: {self.fm_demodulator.composite_rate:.0f} Hz, Audio: {self.audio_output.output_rate} Hz")
            station = self.current_station()
            print(f"Default station: "
                  f"{self.sdr_receiver.get_center_frequency()/1e6:.1f} MHz"
                  + (f" ({station.name})" if station else ""))
            print("Stereo demodulation enabled.")
            print("Commands: q, list, <freq>, stereo on/off, record start/stop, iqrec start/stop, agc on/off, gain <value>, etc.")
        print("Auto gain control: ON")

    def start(self) -> None:
        """Start the receiver and run the command line until it quits."""
        try:
            self.start_background()
            self._announce()

            # Start CLI thread after startup messages to avoid interleaving
            self.cmd_interface.start()
            self.logger.info("Entering main loop")

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
        """Stop the receiver and release what it was using.

        Sets ``quit_event`` first and waits for the threads it started: the
        processing thread hands blocks to the audio output, and closing that
        underneath it would be using a stream that has already gone.  Safe to
        call twice, and safe to call on a receiver that never fully started.
        """
        try:
            self.logger.info("Cleaning up FM Receiver Controller")
            # Whoever is shutting us down may not have asked the threads to
            # stop - a window that failed to open, for one - and everything
            # below is something they are still using.
            self.quit_event.set()
            self.sdr_receiver.stop()
            self._join_threads()
            self.audio_output.cleanup()
            self.auto_gain.stop()
            self.logger.info("FM Receiver cleanup completed")
            print("Exiting FM Receiver.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            print("Error during cleanup - see log for details.")

    def _join_threads(self) -> None:
        """Wait for the started threads, warning about any that will not stop."""
        for thread in self.threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                self.logger.warning(
                    "%s did not stop within %.0f s; carrying on with cleanup",
                    thread.name, _THREAD_JOIN_TIMEOUT_SEC)
        self.threads = [t for t in self.threads if t.is_alive()]
