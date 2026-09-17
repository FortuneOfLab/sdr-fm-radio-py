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

import os
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
from fm_radio.device_worker import TUNE, DeviceWorker, Request
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
        # Why the receiver stopped, when it stopped on its own.  Read by
        # whatever is showing the receiver to a person: the command line
        # prints it on the way out, the window puts it on the health line.
        self.device_failure: str | None = None
        # cleanup() can be asked for from two places at once: the window
        # starts one when the device goes, and closing the window starts
        # another.  It is idempotent, but only one at a time.
        self._cleanup_lock: threading.Lock = threading.Lock()
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
            # One thread for every write to the device, so that no
            # window and no command line ever waits for USB.
            self.device_worker: DeviceWorker = DeviceWorker(self.logger)
            self.auto_gain: AutoGainController = AutoGainController(
                self.sdr_receiver, self.device_worker)
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

    def tune(self, freq_hz: float) -> "Request":
        """Ask for a new frequency, and come straight back.

        The write itself is 60 ms of USB on a device that is answering
        and forever on one that is not, so it happens on the device
        worker rather than on whichever thread asked - the window asks
        from the thread that draws it.

        Everything that follows a tune - the stale IQ, the demodulator
        state, the AGC counters, a recording that was of a different
        station - happens on the worker too, after the write, so a
        caller who does wait sees a receiver that has finished moving.

        Args:
            freq_hz: Target frequency in Hz.

        Returns:
            The request, for a caller that wants to know how it went.
            The window does not; the command line waits a moment.
        """
        return self.device_worker.submit(
            TUNE, f"Tuned to {freq_hz / 1e6:.1f} MHz",
            lambda: self._tune_now(freq_hz))

    def _tune_now(self, freq_hz: float) -> None:
        """Change frequency and settle everything behind it.

        Runs on the device worker.  The order matters: nothing downstream
        should see a block from the old station tagged with the new one,
        which is what the generation on each block and the flush here are
        between them for.

        The recordings stop taking samples before the frequency moves,
        not after.  A recording is a file that claims a station, and the
        SDR starts delivering the new one the moment the write lands -
        stopping afterwards leaves however many blocks fell in the gap
        recorded under the wrong name.  Stopping first can cost the last
        fraction of a second of the old station, and if the write then
        fails the recording has ended for a station the receiver never
        left; a short file is a smaller harm than a wrong one.
        """
        taken = self._stop_the_recordings_taking_samples()
        try:
            self.sdr_receiver.set_center_frequency(freq_hz)
            self._flush_data_queue()
            self.fm_demodulator.reset()
            self.auto_gain.reset_counters()
        finally:
            self._close_the_recordings(taken)

    def _stop_the_recordings_taking_samples(self) -> tuple[bool, bool]:
        """Shut the door on both recordings, and come straight back.

        A flag under a lock each; the flush, the close and the sidecar
        are left for :meth:`_close_the_recordings`.

        Returns:
            Which of the two - audio, IQ - this call took, and therefore
            which of them are owed a finish.
        """
        return (self.audio_output.begin_stopping_the_recording(),
                self.sdr_receiver.begin_stopping_the_iq_recording())

    def _close_the_recordings(self, taken: tuple[bool, bool]) -> None:
        """Finish the recordings *taken* closed, on a thread of its own.

        Closing one is a flush handshake with its worker and takes up to
        fifteen seconds.  That is not the device worker's to spend: every
        other write would queue behind it, and the gain the AGC wants
        next is not worth a quarter of a minute.

        Nobody waits for this here.  Shutdown does, through the
        finalising flags on the two recorders, which is what keeps a file
        from being left half written.
        """
        if not any(taken):
            return
        try:
            threading.Thread(target=self._close_the_recordings_now,
                             args=(taken,), name="RecordingClose",
                             daemon=True).start()
        except RuntimeError as e:               # pragma: no cover - guard
            # No thread to be had.  Slow is better than a file left open
            # and a finalising flag nobody will ever clear.
            self.logger.error("Closing a recording on this thread: %s", e)
            self._close_the_recordings_now(taken)

    def _close_the_recordings_now(self, taken: tuple[bool, bool]) -> None:
        audio, iq = taken
        if audio:
            try:
                self.audio_output.finish_stopping_the_recording()
            except Exception as e:              # pragma: no cover - guard
                self.logger.error(
                    "Could not close the recording after tuning: %s", e,
                    exc_info=True)
        if iq:
            try:
                self.sdr_receiver.finish_stopping_the_iq_recording()
            except Exception as e:              # pragma: no cover - guard
                self.logger.error(
                    "Could not close the IQ recording after tuning: %s", e,
                    exc_info=True)

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

    def set_agc_mode(self, enabled: bool) -> "Request | None":
        """Enable or disable automatic gain control.

        When enabled, the auto gain controller monitors IQ peak
        amplitude and adjusts RTL-SDR hardware gain automatically.
        When disabled, the user controls gain manually via CLI.

        Args:
            enabled: True to enable auto gain, False for manual mode.

        Returns:
            The device write this asked for, or None when it asked for
            none.  The window watches it; nothing has to.
        """
        if enabled:
            return self.auto_gain.enable()
        return self.auto_gain.disable()

    def get_gain(self) -> float:
        """Return the current gain value in dB."""
        return self.sdr_receiver.get_gain()

    def set_gain(self, gain: float) -> "Request | None":
        """Set the manual gain value in dB.

        Only effective when auto gain is disabled.

        Args:
            gain: Gain value in dB.

        Returns:
            The device write this asked for, or None when auto gain is on
            and the request was refused.
        """
        return self.auto_gain.set_gain_manual(gain)

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

        sdr_thread = threading.Thread(target=self._run_sdr_thread,
                                      name="SDRThread", daemon=True)
        sdr_thread.start()
        self.threads.append(sdr_thread)

        proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        proc_thread.start()
        self.threads.append(proc_thread)

        self.logger.info("FM Receiver started successfully")

    def _run_sdr_thread(self) -> None:
        """Read from the SDR, and end the receiver if the device goes.

        A bare thread target turns a device that has been unplugged into a
        traceback on stderr and nothing else: the processing thread goes on
        waiting for a queue nobody fills, the output underruns at fifty
        callbacks a second, and a window shows the last reading it had.
        Nothing is coming back from here, so the rest is told.
        """
        try:
            self.sdr_receiver.start()
        except SDRDeviceError as e:
            self._device_is_gone(str(e))
        except Exception as e:                  # pragma: no cover - guard
            self.logger.critical(
                "Unexpected failure in the SDR thread: %s", e, exc_info=True)
            self._device_is_gone(str(e))

    def _device_is_gone(self, why: str) -> None:
        """Record why the samples stopped, and ask everything else to stop.

        The stopping comes first and every kind of telling second.  Both
        kinds can fail on a handle that has been closed underneath them -
        print raises BrokenPipeError, and a log handler on a closed file
        raises ValueError - and a receiver that keeps running because it
        could not announce that it had stopped is worse than one that
        stops quietly.  They are also separate from each other: a log that
        cannot be written is no reason not to try the console.
        """
        self.device_failure = why
        self.quit_event.set()
        self._log_the_device_is_gone(why)
        self._say_the_device_is_gone(why)

    def _log_the_device_is_gone(self, why: str) -> None:
        """Write the reason down, if there is anywhere left to write it."""
        try:
            self.logger.error("The SDR stopped delivering samples: %s", why)
        except Exception:               # pragma: no cover - last resort
            # Nowhere left to report a logging failure to.  The console
            # notice is tried next and may still get through.
            pass

    def _say_the_device_is_gone(self, why: str) -> None:
        """Tell whoever is watching, if there is anywhere left to tell.

        Said from the SDR thread rather than left to cleanup, because
        cleanup has a few bounded waits in it and the person who has just
        pulled a cable should be told before being made to wait.  It has
        already been logged, so losing this costs nothing that matters.
        """
        try:
            print(f"\nSDR disconnected: {why}")
            print("Stopping the receiver.")
        except Exception as e:
            self.logger.debug(
                "Could not print the disconnect notice: %s", e)

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
            self._leave_past_the_blocked_reader()

    def _leave_past_the_blocked_reader(self) -> None:
        """Go without waiting for a command that is never coming.

        The command thread spends its life inside input(), and there is no
        portable way to wake one blocked on a console.  Letting the
        interpreter finalise around it aborts the process with
        "_enter_buffered_busy: could not acquire lock for <stdin>" - after
        a clean shutdown, which makes the shutdown look like it failed.
        A person who has just unplugged their radio should not be shown a
        fatal error for it.

        cleanup() has already run, so Python's own finalisation has
        nothing left to do for us: the device is closed, the audio stream
        is closed, and every recording has been flushed and closed.  The
        status says whether the receiver was asked to stop or stopped
        because the device went - and so does the entry point, for the
        times the command thread has already gone and this does nothing.
        """
        if not self.cmd_interface.is_alive():
            return
        # Every one of these can fail on a closed pipe, and none of them is
        # a reason to stay.
        for flush in (sys.stdout.flush, sys.stderr.flush,
                      *(h.flush for h in logging.getLogger().handlers)):
            try:
                flush()
            except Exception:           # pragma: no cover - best effort
                pass
        os._exit(1 if self.device_failure else 0)

    def cleanup(self) -> None:
        """Stop the receiver and release what it was using.

        Sets ``quit_event`` first and waits for the threads it started: the
        processing thread hands blocks to the audio output, and closing that
        underneath it would be using a stream that has already gone.  Safe to
        call twice, and safe to call on a receiver that never fully started.
        """
        with self._cleanup_lock:
            self._cleanup()

    def _cleanup(self) -> None:
        """Body of :meth:`cleanup`; the caller holds ``_cleanup_lock``."""
        try:
            self.logger.info("Cleaning up FM Receiver Controller")
            # Whoever is shutting us down may not have asked the threads to
            # stop - a window that failed to open, for one - and everything
            # below is something they are still using.
            self.quit_event.set()
            # The gain worker writes to the SDR from its own thread, so it
            # goes before the device it writes to.  Each resource also
            # refuses use once closed, because a bounded join cannot promise
            # that every thread has finished.
            # The writer goes before the device it writes to, and the
            # writer is one thread for all of them now.
            self.device_worker.stop()
            self.auto_gain.stop()
            self.sdr_receiver.stop()
            self._join_threads()
            self.audio_output.cleanup()
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
