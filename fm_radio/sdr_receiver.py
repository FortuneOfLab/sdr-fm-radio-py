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
"""SDR receiver class using RTL-SDR."""

from __future__ import annotations

import os
import queue
import logging
import time
import wave
import threading

import numpy as np
from rtlsdr import RtlSdr

from fm_radio import recording_meta
from fm_radio.device_handle import DeviceHandle
from fm_radio.interfaces import SDRReceiverInterface
from fm_radio.exceptions import SDRDeviceError, RecordingError
from fm_radio.constants import (
    SDR_SAMPLE_RATE, SDR_CENTER_FREQ_DEFAULT,
    SDR_BLOCK_SIZE, SDR_QUEUE_MAXSIZE,
    IQ_RECORD_QUEUE_MAXSIZE, IQ_RECORD_ROTATE_THRESHOLD_BYTES,
)


# Sentinels placed in the IQ-recording queue to wake the worker.
_IQ_RECORD_WORKER_SHUTDOWN = object()
_IQ_RECORD_FLUSH_SENTINEL = object()

#: Longest shutdown waits for an IQ recording that is being closed
#: elsewhere.  stop_iq_recording is itself bounded at about fifteen
#: seconds; past this something is wrong and the log should say so.
_IQ_CLOSE_TIMEOUT_SEC: float = 20.0

#: Longest a new IQ recording waits for the one before it to finish
#: closing; see AudioOutput._PREVIOUS_CLOSE_WAIT_SEC.  The wait is on
#: whichever thread pressed the button, so it is short.
_PREVIOUS_IQ_CLOSE_WAIT_SEC: float = 2.0



class _ReadyIQRecording:
    """An IQ file that is open and waiting to be recorded into.

    Carries the gain as well, because reading it is a USB transfer
    and the install is not the place for one; it is read while this
    is being prepared, with nothing held.
    """

    __slots__ = ("path", "wave", "gain_db")

    def __init__(self, path: str, wave_file, gain_db: float | None) -> None:
        self.path = path
        self.wave = wave_file
        self.gain_db = gain_db


class SDRReceiver(SDRReceiverInterface):
    """
    Receiver class using RTL-SDR

    Retrieves samples from the RTL-SDR device and asynchronously stores them in a queue.
    """
    def __init__(
        self,
        sample_rate: float = SDR_SAMPLE_RATE,
        center_freq: float = SDR_CENTER_FREQ_DEFAULT,
        block_size: int = SDR_BLOCK_SIZE,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger('fm_receiver.SDRReceiver')
        self.sample_rate: float = sample_rate
        self.center_freq: float = center_freq
        self.block_size: int = block_size
        # Each entry is (tuning generation, IQ block).  The generation is
        # what makes it possible to tell, later in the pipeline, which
        # tuning a block was captured under: by the time the processing
        # thread gets to one, the receiver may already have retuned.
        self.data_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(
            maxsize=SDR_QUEUE_MAXSIZE)
        # Bumped by set_center_frequency, read by the SDR callback and by
        # anything that wants to know whether a block is still current.
        self._tuning_generation: int = 0
        # The frequency the current generation stands for.  Changed
        # with the generation, under the same lock, and read with it:
        # see the_tuning_and_its_frequency.
        self._generation_freq_hz: float = float(center_freq)
        self._tuning_lock: threading.Lock = threading.Lock()
        # Last values seen from the device, handed out in place of touching
        # it when it is closed or busy.  Written under the lock, read
        # without one: a reading a block old on a display costs nothing,
        # and stalling the processing thread behind a USB write does.
        self._last_freq_hz: float = float(center_freq)
        self._last_gain_db: float = 0.0
        # Serialises stop() against itself, so its steps happen once
        # however many times cleanup runs.
        self._stop_lock: threading.Lock = threading.Lock()
        self.iq_recording: bool = False
        # Set for as long as a recording is being closed: the flag above
        # goes down first and the file stays open for the flush, the
        # worker handshake, the close and the sidecar - up to fifteen
        # seconds of a file that is still being written while nothing
        # calls it a recording.  Shutdown waits for this, not for that.
        self._iq_finalising: threading.Event = threading.Event()
        self.iq_record_wave: wave.Wave_write | None = None
        # ``iq_record_lock`` guards self.iq_record_wave (file open/close vs
        # in-flight writeframes inside the worker).  ``_iq_enqueue_lock``
        # guards the atomic pair (self.iq_recording check + queue.put_nowait)
        # in the SDR callback so a stop_iq_recording flag flip cannot race
        # past an in-progress callback enqueue.  The worker never takes
        # ``_iq_enqueue_lock`` so disk-write stalls cannot back-pressure
        # the realtime callback.
        self.iq_record_lock: threading.Lock = threading.Lock()
        self._iq_enqueue_lock: threading.Lock = threading.Lock()
        # Somewhere for a second reader to be handed the same blocks;
        # see watch_the_blocks.  None when nobody is watching, which
        # is nearly always, and read on the realtime path.
        self._tap: "queue.Queue | None" = None
        # Only the check-and-set and the compare-and-clear below; the
        # callback never takes it, and reads the attribute once.
        self._tap_lock: threading.Lock = threading.Lock()
        # Serialises concurrent ``start_iq_recording`` callers so only
        # one reaches ``wave.open`` (the file-truncating step).
        # Distinct from ``_iq_enqueue_lock`` so the SDR callback is
        # never blocked during the slow file open.
        # Reentrant: start_iq_recording takes it and then calls
        # install, which takes it again to decide against a teardown.
        self._iq_start_lock: threading.RLock = threading.RLock()
        # One recording after another, counted; see AudioOutput.
        self._iq_record_session: int = 0
        self._iq_sidecar_lock: threading.Lock = threading.Lock()
        # Cumulative count of dropped IQ blocks (queue full). Bumped from
        # the SDR callback thread; only read for diagnostic logging.
        self._dropped_count: int = 0
        # Async IQ-recording worker: keeps the SDR callback off disk
        # writeframes (~4 MB/s of data, with occasional 100-1000 ms
        # OS-level stalls), so callback stalls cannot back up the
        # rtlsdr internal buffer and lose IQ samples.
        self._iq_record_q: queue.Queue[object] = queue.Queue(
            maxsize=IQ_RECORD_QUEUE_MAXSIZE,
        )
        self._iq_flush_event: threading.Event = threading.Event()
        self._iq_record_drop_count: int = 0
        # State for 4-GiB WAV rotation (set in start_iq_recording, used
        # by the worker).  ``_iq_record_base_path`` is the path the
        # caller supplied; ``_iq_record_part_index`` is 0 for the first
        # file, 1+ for rotated continuations; ``_iq_record_bytes_written``
        # tracks the data chunk size of the current file so we can
        # rotate before crossing the 2^32-byte WAV header limit.
        self._iq_record_base_path: str | None = None
        self._iq_record_meta: dict | None = None
        self._iq_record_part_index: int = 0
        self._iq_record_bytes_written: int = 0
        self._iq_record_worker_stop: threading.Event = threading.Event()
        self._iq_record_worker: threading.Thread = threading.Thread(
            target=self._iq_record_worker_loop,
            name='IQRecordWorker',
            daemon=True,
        )
        self._iq_record_worker.start()

        try:
            self.sdr: RtlSdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.set_manual_gain_enabled(False)
            self.manual_gain: bool = False
            self.sdr.set_gain(0)
            # Everything about when this device may be touched and when it
            # is freed lives in here, including the close that pyrtlsdr
            # makes for itself when a call fails.
            self.handle: DeviceHandle = DeviceHandle(self.sdr, self.logger)
            self.logger.info(f"SDR initialized: sample_rate={sample_rate/1e6:.3f}MHz, center_freq={center_freq/1e6:.1f}MHz")
        except OSError as e:
            self.logger.error(f"Failed to initialize RTL-SDR device: {e}")
            raise SDRDeviceError(f"Failed to initialize RTL-SDR device: {e}") from e

        try:
            # Disable direct_sampling if available
            self.sdr.direct_sampling = 0
            self.logger.debug("Direct sampling disabled")
        except OSError as e:
            self.logger.warning(f"Failed to disable direct_sampling (may not be supported): {e}")

    @property
    def tuning_generation(self) -> int:
        """Counter identifying the current tuning.

        Every IQ block carries the value this had when it was captured, so
        a block from before a retune can be recognised as such however long
        it sat in the queue.
        """
        return self._tuning_generation

    def the_tuning_and_its_frequency(self) -> tuple[int, float]:
        """The current generation and the frequency that generation means.

        Read together, under the lock that changes them together, so
        that a caller cannot pair one tuning's samples with another
        tuning's frequency.  ``center_freq`` on its own cannot be
        used for this: it is set before the hardware write, so a
        picture built from a block of the old station during those
        40-200 ms would be labelled with the new one.
        """
        with self._tuning_lock:
            return self._tuning_generation, self._generation_freq_hz

    def set_center_frequency(self, freq: float) -> None:
        """Change the center frequency."""
        try:
            if not self.handle.usable:
                self.logger.debug(
                    "Ignoring tune to %.1f MHz: the SDR is closed", freq / 1e6)
                return
            with self.handle.held():
                if not self.handle.usable:
                    return              # closed while we waited for the lock
                self.center_freq = freq
                self.sdr.center_freq = freq
                self._last_freq_hz = float(freq)
            # Bumped after the hardware change, never before: a block
            # captured on the new frequency but enqueued before this point
            # is then treated as belonging to the old tuning, which loses a
            # snapshot rather than showing one against the wrong station.
            # The frequency the new generation stands for goes in with it,
            # so that the two can be read as one thing.
            with self._tuning_lock:
                self._tuning_generation += 1
                self._generation_freq_hz = float(freq)
            self.logger.info(f"Center frequency set to {freq/1e6:.1f} MHz")
        except OSError as e:
            self.logger.error(f"Failed to set center frequency to {freq/1e6:.1f} MHz: {e}")
            raise SDRDeviceError(f"Failed to set center frequency: {e}") from e
        finally:
            # Whatever happened above - a write, a refusal, a failure that
            # closed the device from inside pyrtlsdr - the device lock is
            # free again now, which may be the thing a deferred close was
            # waiting for.
            self.handle.retry_pending_close()

    def watch_the_blocks(self, depth: int = 1) -> "queue.Queue":
        """Be handed the same IQ blocks the demodulator gets.

        For something that wants to look at the samples without
        taking them: a band scan, most obviously, which runs while
        the receiver is still playing.  The blocks are the same
        arrays, so a watcher reads them and does not write to them.

        One watcher at a time, and a second one is an error rather
        than a quiet takeover: the one already there would simply
        stop being fed, and would find out as a hop that timed out
        with nothing to say why.  Blocks are dropped rather than
        queued when the watcher is behind.

        Returns:
            The queue to read ``(generation, block)`` from.  Hand it
            back to :meth:`stop_watching` when finished with it.

        Raises:
            RuntimeError: if somebody is already watching.
        """
        tap: "queue.Queue" = queue.Queue(maxsize=max(1, int(depth)))
        with self._tap_lock:
            if self._tap is not None:
                raise RuntimeError(
                    "something is already watching the blocks")
            self._tap = tap
        return tap

    def stop_watching(self, tap: "queue.Queue") -> None:
        """Stop feeding *tap*, unless somebody else has taken over.

        A block can still land in *tap* just after this: the
        callback reads the attribute once and may have read it
        already.  Harmless - the owner is about to drop the queue -
        and worth more than a lock on the realtime path.
        """
        with self._tap_lock:
            if self._tap is tap:
                self._tap = None

    def get_center_frequency(self) -> float:
        """Return the centre frequency in Hz, or the last one read.

        The processing thread calls this for every telemetry snapshot, so it
        never waits: a closed device or a write in progress gets the cached
        value instead.  Reading a closed handle is undefined behaviour in
        librtlsdr, and waiting behind a 40-200 ms write would cost blocks.
        """
        if (not self.handle.usable
                or not self.handle.device_lock.acquire(blocking=False)):
            return self._last_freq_hz
        try:
            if not self.handle.usable:
                return self._last_freq_hz    # closed between the two checks
            self._last_freq_hz = float(self.sdr.center_freq)
            return self._last_freq_hz
        except OSError as e:
            self.logger.debug("Could not read the centre frequency: %s", e)
            return self._last_freq_hz
        finally:
            self.handle.device_lock.release()

    @property
    def closed(self) -> bool:
        """True once anybody has decided the device is going."""
        return self.handle.closing.is_set()

    def set_gain(self, gain: float) -> None:
        """Set gain value (for manual mode).

        Does nothing once the device is closed: the gain worker runs on its
        own thread and may still have a write queued when shutdown begins.
        """
        try:
            if not self.handle.usable:
                self.logger.debug(
                    "Ignoring gain %.1f dB: the SDR is closed", gain)
                return
            with self.handle.held():
                if not self.handle.usable:
                    return              # closed while we waited for the lock
                self.sdr.set_gain(gain)
                self._last_gain_db = float(gain)
            self.logger.info(f"Gain set to {gain:.1f} dB")
        except OSError as e:
            self.logger.error(f"Failed to set gain to {gain:.1f} dB: {e}")
            raise SDRDeviceError(f"Failed to set gain: {e}") from e
        finally:
            self.handle.retry_pending_close()

    def get_gain(self) -> float:
        """Return the gain in dB, or the last one read.

        Same bargain as :meth:`get_center_frequency`.
        """
        if (not self.handle.usable
                or not self.handle.device_lock.acquire(blocking=False)):
            return self._last_gain_db
        try:
            if not self.handle.usable:
                return self._last_gain_db
            self._last_gain_db = float(self.sdr.get_gain())
            return self._last_gain_db
        except OSError as e:
            self.logger.debug("Could not read the gain: %s", e)
            return self._last_gain_db
        finally:
            self.handle.device_lock.release()

    def set_manual_gain_mode(self, manual: bool) -> None:
        """
        Set manual gain mode.

        Args:
            manual (bool): True for manual mode, False for AGC.
        """
        try:
            self.manual_gain = manual
            if not self.handle.usable:
                # Before the lock, not after it: this is called from the
                # window, and a receiver that is closed has nothing to
                # queue behind.
                self.logger.debug(
                    "Ignoring gain mode %s: the SDR is closed",
                    "manual" if manual else "AGC")
                return
            with self.handle.held():
                if not self.handle.usable:
                    return
                self.sdr.set_manual_gain_enabled(manual)
            mode = "manual" if manual else "AGC"
            self.logger.info(f"Gain mode set to {mode}")
        except OSError as e:
            self.logger.error(f"Failed to set gain mode: {e}")
            raise SDRDeviceError(f"Failed to set gain mode: {e}") from e
        finally:
            self.handle.retry_pending_close()

    def callback(self, iq_samples: np.ndarray, sdr_obj: RtlSdr) -> None:
        """Callback to store received IQ samples in the data queue.

        Runs on the rtlsdr library's internal thread; must return
        promptly or libusb back-pressure causes IQ loss.  The IQ-WAV
        write is therefore handed off to a worker via a bounded queue
        (the realtime path itself only does a non-blocking put).

        Args:
            iq_samples: Received IQ samples.
            sdr_obj: SDR object (unused).
        """
        # Which tuning these samples came from, recorded before anything is
        # done with them.  The conversion below copies - pyrtlsdr hands over
        # complex128 - and a retune during that copy would otherwise stamp
        # the old station's samples with the new tuning, putting them past
        # the queue flush that tune() had just done.
        generation = self._tuning_generation
        try:
            # Convert to numpy array allowing a copy if necessary (NumPy 2.x compatibility).
            iq = np.asarray(iq_samples, dtype=np.complex64)
            self.data_queue.put((generation, iq), block=False)

            # And to whoever else is watching.  data_queue is a work
            # queue - whatever takes a block from it is the only thing
            # that gets it - so a second reader sharing it would be
            # taking blocks out of the demodulator's stream.  This
            # hands over the same array, which nobody writes to.
            # Dropped rather than waited for: this is the realtime
            # path, and a watcher that is behind wants the next block
            # rather than an old one.
            tap = self._tap
            if tap is not None:
                try:
                    tap.put_nowait((generation, iq))
                except queue.Full:
                    pass

            # Hand the same array to the IQ-recording worker if active.
            # The pair (flag check, put_nowait) is atomic under
            # _iq_enqueue_lock so stop_iq_recording cannot race past
            # an in-progress callback put.
            with self._iq_enqueue_lock:
                if self.iq_recording:
                    try:
                        self._iq_record_q.put_nowait(iq)
                    except queue.Full:
                        self._iq_record_drop_count += 1
                        self.logger.warning(
                            "IQ record queue full, dropping IQ block "
                            "(drops=%d, qsize=%d/%d) — disk write fell "
                            "behind realtime",
                            self._iq_record_drop_count,
                            self._iq_record_q.qsize(),
                            self._iq_record_q.maxsize,
                        )
        except queue.Full:
            # Discard sample if queue is full.  This produces audible
            # dropouts in the demodulated audio so we promote it to
            # WARNING for visibility during diagnostics.
            self._dropped_count += 1
            self.logger.warning(
                "SDR data queue full, dropping samples (total dropped=%d, "
                "qsize=%d/%d)",
                self._dropped_count,
                self.data_queue.qsize(),
                self.data_queue.maxsize,
            )
        except Exception as e:
            # Drop this buffer if conversion fails to avoid crashing the SDR ctypes callback.
            self.logger.error(f"Error in SDR callback: {e}", exc_info=True)

    @property
    def dropped_blocks(self) -> int:
        """IQ blocks the SDR callback discarded because the queue was full."""
        return self._dropped_count

    @property
    def iq_record_drops(self) -> int:
        """IQ blocks dropped by the recording queue in this session."""
        return self._iq_record_drop_count

    def start_iq_recording(self, filename: str) -> None:
        """Start recording raw IQ samples to a 2-channel WAV file (async).

        Thread safety design:
          - ``_iq_start_lock`` serialises concurrent
            ``start_iq_recording`` callers, so only the winner reaches
            ``wave.open`` and losers see ``self.iq_recording = True``
            and return without touching their target path.  The SDR
            callback does not take this lock, so the slow
            ``wave.open`` call inside it does not back-pressure the
            realtime path.
          - ``_iq_enqueue_lock`` is taken only briefly inside that,
            to atomically drain stale queue items, install the wave
            handle and flip ``self.iq_recording``.
        """
        with self._iq_start_lock:
            if self.iq_recording:
                self.logger.warning(
                    "IQ recording already active; ignoring duplicate "
                    "start_iq_recording for %s", filename,
                )
                return
            ready = self.prepare_an_iq_recording(filename)
            try:
                session = self.install_a_prepared_iq_recording(ready)
            except BaseException:
                self.discard_a_prepared_iq_recording(ready)
                raise
        self.finish_starting_the_iq_recording(session)

    def prepare_an_iq_recording(self, filename: str) -> "_ReadyIQRecording":
        """Get a file ready to record IQ into, without starting anything.

        Mirrors AudioOutput.prepare_a_recording: everything slow is
        here - the wait for the recording before this one, and the
        open - so a caller that has to decide under a lock whether to
        go ahead holds nothing while it happens.

        Raises:
            RecordingError: The receiver has been stopped, the one
                before this has not finished closing, or the file
                would not open.
        """
        if not self.handle.usable:
            # The worker that would write the blocks is gone, and no
            # blocks are coming anyway.  Opening the file here would
            # leave one behind with iq_recording set against nothing.
            self.logger.warning(
                "Ignoring start_iq_recording for %s: the receiver has "
                "been stopped", filename)
            raise RecordingError(
                "Cannot start IQ recording: the receiver has been stopped")

        self._let_the_last_iq_recording_go(filename)

        # Made, not opened; see AudioOutput.prepare_a_recording.  The
        # file wave.open would truncate can be one that is being
        # recorded into at this moment.
        try:
            os.close(os.open(filename,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError as e:
            self.logger.error(
                "IQ recording start failed: %s exists", filename)
            raise RecordingError(
                f"IQ recording start failed: {filename} already exists",
            ) from e
        except OSError as e:
            self.logger.error(
                f"IQ recording start failed: {e}", exc_info=True,
            )
            raise RecordingError(
                f"IQ recording start failed: {e}",
            ) from e

        try:
            wf = wave.open(filename, 'wb')
        except (OSError, wave.Error) as e:
            self.logger.error(
                f"IQ recording start failed: {e}", exc_info=True,
            )
            self._remove_the_file_we_made(filename)
            raise RecordingError(
                f"IQ recording start failed: {e}",
            ) from e

        try:
            wf.setnchannels(2)           # I/Q
            wf.setsampwidth(2)           # int16
            wf.setframerate(int(self.sample_rate))
        except Exception as e:
            # See AudioOutput._shut_a_file_that_never_started.
            self.logger.error(
                f"IQ recording start failed: {e}", exc_info=True,
            )
            self._shut_a_file_that_never_started(wf, filename)
            raise RecordingError(
                f"IQ recording start failed: {e}",
            ) from e

        # Here rather than in the install: this is a control transfer
        # to the device when the device lock is free, and the install
        # runs with the caller's lock held.
        try:
            gain_db = float(self.get_gain())
        except Exception:
            gain_db = None
        return _ReadyIQRecording(filename, wf, gain_db)

    def _shut_a_file_that_never_started(self, wave_file,
                                        filename: str) -> None:
        """Close a handle with no header on it, and take the file back."""
        try:
            wave_file.close()
        except Exception as e:
            self.logger.debug("Could not close %s cleanly: %s", filename, e)
        self._remove_the_file_we_made(filename)

    def _remove_the_file_we_made(self, filename: str) -> None:
        """Take back a file this class created and is not going to use."""
        try:
            os.remove(filename)
        except OSError as e:                    # pragma: no cover - guard
            self.logger.debug("Could not remove %s: %s", filename, e)

    def install_a_prepared_iq_recording(self,
                                        ready: "_ReadyIQRecording") -> int:
        """Put a prepared file in and start recording IQ into it.

        Under ``_iq_start_lock``, which the teardown also takes, so a
        recording cannot be installed into a receiver that has been
        stopped and leave a handle nothing will close.  Nothing here
        touches a file or the device: the gain was read while the
        recording was being prepared, and the sidecar and the log are
        :meth:`finish_starting_the_iq_recording`, afterwards.

        Returns:
            The session number for that call.

        Raises:
            RecordingError: The receiver was stopped while this was
                being prepared.
        """
        with self._iq_start_lock:
            if not self.handle.usable:
                raise RecordingError(
                    "Cannot start IQ recording: the receiver has been "
                    "stopped")
            # Atomic install w.r.t. the SDR callback's enqueue path.
            with self._iq_enqueue_lock:
                self._drain_iq_record_queue()
                self._iq_flush_event.clear()
                self._iq_record_drop_count = 0
                with self.iq_record_lock:
                    self.iq_record_wave = ready.wave
                    self._iq_record_base_path = ready.path
                    self._iq_record_part_index = 0
                    self._iq_record_bytes_written = 0
                self.iq_recording = True

            self._iq_record_session += 1
            session = self._iq_record_session
            self._iq_record_meta = {
                "type": "iq",
                "file": recording_meta.part_list(
                    ready.path, 0, self._make_rotated_iq_path,
                )[0],
                "sample_rate_hz": int(self.sample_rate),
                "center_freq_hz": float(self.center_freq),
                "gain_db": ready.gain_db,
                "started_at": recording_meta.now_iso(),
            }
        return session

    def finish_starting_the_iq_recording(self, session: int) -> None:
        """Write the sidecar and say so; see AudioOutput for why here."""
        with self._iq_sidecar_lock:
            if session != self._iq_record_session:
                self.logger.debug(
                    "Not writing the sidecar for IQ recording %d; the "
                    "current one is %d", session, self._iq_record_session)
                return
            path = self._iq_record_base_path
            if path is None or not self._iq_record_meta:
                return
            recording_meta.write_sidecar(path, self._iq_record_meta,
                                         self.logger)
        self.logger.info(f"IQ recording started: {path}")

    def discard_a_prepared_iq_recording(self,
                                        ready: "_ReadyIQRecording") -> None:
        """Give back a file that is not going to be recorded into."""
        try:
            ready.wave.close()
        except Exception as e:                  # pragma: no cover - guard
            self.logger.debug("Could not close %s: %s", ready.path, e)
        # Only ever the file prepare_an_iq_recording created.
        self._remove_the_file_we_made(ready.path)

    def _let_the_last_iq_recording_go(self, filename: str) -> None:
        """Wait for an IQ recording that is still closing, or refuse this.

        Mirrors AudioOutput._let_the_last_recording_go: one that is
        being closed still owns the wave handle, and would close this
        one instead of its own.

        Raises:
            RecordingError: The last one is still closing after
                ``_PREVIOUS_IQ_CLOSE_WAIT_SEC``.
        """
        if not self._iq_finalising.is_set():
            return
        self.logger.info(
            "Waiting for the previous IQ recording to finish closing before "
            "starting %s", filename)
        if self.wait_for_the_iq_recording_to_close(
                _PREVIOUS_IQ_CLOSE_WAIT_SEC):
            return
        self.logger.error(
            "Refusing to start %s: the previous IQ recording has been "
            "closing for %.0f s", filename, _PREVIOUS_IQ_CLOSE_WAIT_SEC)
        raise RecordingError(
            "Cannot start IQ recording: the previous recording is still "
            "closing")

    def stop_iq_recording(self) -> None:
        """Stop IQ recording, flush pending writes, and close the file.

        Mirrors AudioOutput.stop_recording: the wave file is only
        closed after the worker has actually written every block that
        was queued before stop_iq_recording was called, via a flush
        sentinel + Event handshake.
        """
        if self.begin_stopping_the_iq_recording():
            self.finish_stopping_the_iq_recording()

    def begin_stopping_the_iq_recording(self) -> bool:
        """Stop taking IQ for the recording, and nothing else.

        Mirrors AudioOutput.begin_stopping_the_recording: the half that
        has to happen before the tuner moves, so that no sample of the
        new station reaches the old station's file.

        Returns:
            True when this call took the recording and owes it a
            ``finish_stopping_the_iq_recording``.
        """
        with self._iq_enqueue_lock:
            if not self.iq_recording:
                self.logger.debug(
                    "stop_iq_recording called but not currently recording",
                )
                return False
            # Stop further enqueues from the SDR callback.  Because we
            # hold _iq_enqueue_lock, any callback that already passed
            # the flag check has also completed its put before us.
            self.iq_recording = False
            # From here the file is open and nothing calls it a
            # recording.  Anybody who needs it finished waits on this.
            self._iq_finalising.set()
        return True

    def finish_stopping_the_iq_recording(self) -> None:
        """Flush what was queued, close the file and write the sidecar."""
        try:
            self._finish_the_iq_recording()
        finally:
            self._iq_finalising.clear()

    def _finish_the_iq_recording(self) -> None:
        """Flush what is queued, close the file and write the sidecar."""

        # Push a flush sentinel.  The worker writes every block before
        # the sentinel and only then sets _iq_flush_event.
        self._iq_flush_event.clear()
        if not self._iq_record_worker.is_alive():
            # Nothing is left to answer the sentinel, so waiting for one
            # would just be fifteen seconds of nothing happening.
            self.logger.warning(
                "The IQ record worker is no longer running; closing the "
                "file without waiting for a flush",
            )
        else:
            try:
                self._iq_record_q.put(_IQ_RECORD_FLUSH_SENTINEL, timeout=5.0)
            except queue.Full:
                self.logger.warning(
                    "Could not enqueue IQ flush sentinel; tail blocks may "
                    "be lost",
                )
            else:
                if not self._iq_flush_event.wait(timeout=10.0):
                    self.logger.warning(
                        "IQ recording flush did not complete within 10 s; "
                        "closing anyway",
                    )

        with self.iq_record_lock:
            parts_count = self._iq_record_part_index + 1
            base_path = self._iq_record_base_path
            part_index = self._iq_record_part_index
            drops = self._iq_record_drop_count
            if self.iq_record_wave is not None:
                try:
                    self.iq_record_wave.close()
                    self.logger.info(
                        "IQ recording stopped (drops during session: %d, "
                        "parts: %d)",
                        self._iq_record_drop_count, parts_count,
                    )
                except (OSError, wave.Error) as e:
                    self.logger.error(
                        f"Error closing IQ recording file: {e}", exc_info=True,
                    )
                finally:
                    self.iq_record_wave = None

        # Finalise the metadata sidecar with the session outcome; see
        # AudioOutput for why the session is moved on afterwards.
        with self._iq_sidecar_lock:
            if base_path is not None and self._iq_record_meta:
                self._iq_record_meta.update({
                    "stopped_at": recording_meta.now_iso(),
                    "parts": recording_meta.part_list(
                        base_path, part_index, self._make_rotated_iq_path,
                    ),
                    "dropped_blocks": int(drops),
                })
                recording_meta.write_sidecar(
                    base_path, self._iq_record_meta, self.logger,
                )
            self._iq_record_session += 1

    # ------------------------------------------------------------------
    # IQ recording worker (runs disk writes off the SDR callback thread)
    # ------------------------------------------------------------------

    def _drain_iq_record_queue(self) -> None:
        """Drop everything currently sitting in the IQ record queue."""
        try:
            while True:
                self._iq_record_q.get_nowait()
        except queue.Empty:
            pass

    @staticmethod
    def _make_rotated_iq_path(base_path: str, part_index: int) -> str:
        """Build the rotated-file path ``foo.partNNN.wav`` from ``foo.wav``."""
        root, ext = os.path.splitext(base_path)
        return f"{root}.part{part_index:03d}{ext}"

    def _rotate_iq_recording(self, current_bytes: int) -> bool:
        """Close the current IQ-WAV and open the next part file.

        Caller must hold ``self.iq_record_lock``.  On success, swaps
        ``self.iq_record_wave`` to the new file and resets the byte
        counter.  On any failure leaves ``self.iq_record_wave = None``
        so subsequent worker iterations drop chunks rather than crash;
        the user's ``stop_iq_recording`` will still close cleanly.

        Returns:
            True if rotation succeeded, False otherwise.
        """
        prev_wave = self.iq_record_wave
        prev_path = (
            self._make_rotated_iq_path(
                self._iq_record_base_path, self._iq_record_part_index,
            ) if self._iq_record_part_index > 0
            else self._iq_record_base_path
        )
        try:
            if prev_wave is not None:
                prev_wave.close()
        except (OSError, wave.Error) as e:
            self.logger.error(
                "Error closing IQ part file %s during rotation: %s",
                prev_path, e, exc_info=True,
            )
        # Bump part index and open the next file.
        self._iq_record_part_index += 1
        next_path = self._make_rotated_iq_path(
            self._iq_record_base_path, self._iq_record_part_index,
        )
        try:
            wf = wave.open(next_path, 'wb')
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(self.sample_rate))
        except (OSError, wave.Error) as e:
            self.logger.error(
                "Failed to open rotated IQ part file %s: %s — recording "
                "will stop accepting new blocks", next_path, e,
                exc_info=True,
            )
            self.iq_record_wave = None
            return False
        self.iq_record_wave = wf
        self._iq_record_bytes_written = 0
        self.logger.info(
            "IQ recording rotated to part %d: %s "
            "(previous part wrote ~%.2f GB)",
            self._iq_record_part_index, next_path,
            current_bytes / 1e9,
        )
        return True

    def _iq_record_worker_loop(self) -> None:
        while not self._iq_record_worker_stop.is_set():
            try:
                item = self._iq_record_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _IQ_RECORD_WORKER_SHUTDOWN:
                break
            if item is _IQ_RECORD_FLUSH_SENTINEL:
                # Every block queued before this sentinel has now been
                # written (queue is FIFO; writes happen on this thread).
                self._iq_flush_event.set()
                continue
            iq = item
            # Catch *any* exception so the worker thread does not die
            # mid-recording — historically struct.error from the wave
            # module's 4-GiB header limit killed this thread, leaving
            # subsequent IQ blocks silently dropped.  Rotation below
            # prevents that root cause but the broad catch is kept as
            # a safety net for unforeseen corner cases.
            try:
                # Each IQ block becomes ``size * 2 channels * 2 bytes``.
                chunk_bytes = int(iq.size) * 2 * 2
                with self.iq_record_lock:
                    if self.iq_record_wave is None:
                        # File already closed; drop the block.
                        continue
                    # Rotate before writing if this chunk would push
                    # the current file past the WAV 4-GiB limit.
                    if (self._iq_record_bytes_written + chunk_bytes
                            > IQ_RECORD_ROTATE_THRESHOLD_BYTES):
                        if not self._rotate_iq_recording(
                            self._iq_record_bytes_written,
                        ):
                            continue
                    clipped_i = np.clip(iq.real, -1.0, 1.0)
                    clipped_q = np.clip(iq.imag, -1.0, 1.0)
                    iq_interleaved = np.empty(iq.size * 2, dtype=np.int16)
                    iq_interleaved[0::2] = np.int16(clipped_i * 32767.0)
                    iq_interleaved[1::2] = np.int16(clipped_q * 32767.0)
                    self.iq_record_wave.writeframes(iq_interleaved.tobytes())
                    self._iq_record_bytes_written += chunk_bytes
            except Exception as e:
                self.logger.error(
                    "Unexpected error writing IQ to file: %s", e,
                    exc_info=True,
                )
                # Drop the failed block and keep the worker alive; the
                # CLI can still call stop_iq_recording later.

    def start(self) -> None:
        """Start asynchronous sample retrieval.

        Does nothing once the receiver is closed.  The handle decides
        that - see :meth:`~fm_radio.device_handle.DeviceHandle.begin_reading`
        - so a read either arms in time to be cancelled or never touches
        the device at all.
        """
        if not self.handle.begin_reading():
            self.logger.info(
                "Not starting the async read: the receiver is closed")
            return
        try:
            self.logger.info("Starting SDR async read")
            self.sdr.read_samples_async(self.callback, num_samples=self.block_size)
        except OSError as e:
            self.logger.error(f"Failed to start SDR async read: {e}")
            # The read is over and pyrtlsdr has asked for the device to be
            # closed on the way out.  Whether that close could be made yet
            # or had to be deferred, this receiver is finished: refuse
            # writes from here rather than leaving it looking open.
            self.handle.closing.set()
            if not getattr(self.sdr, "device_opened", True):
                self.handle.note_closed_by_driver()
            raise SDRDeviceError(f"Failed to start SDR async read: {e}") from e
        finally:
            self.handle.finished_reading()

    def stop(self) -> None:
        """Stop the recording, then ask for the device to be closed.

        Marks the device on its way out first, so a write already in
        flight from another thread is dropped rather than landing on a
        handle that is about to go.  What that closing then involves -
        cancelling the async read, waiting for it and for the writes,
        keeping the request when either will not let go - is the handle's
        to decide, and it decides the same way for every caller.  Safe to
        call more than once.
        """
        with self._stop_lock:
            self.handle.closing.set()
            # An IQ recording that is part way through starting finishes
            # installing itself before the teardown below, so it is closed
            # properly instead of being left behind with a worker that has
            # already gone.  One that starts after this finds the handle
            # no longer usable and never opens a file at all.
            with self._iq_start_lock:
                # A recording being closed somewhere else is still an
                # open file; the flag went down when the closing started.
                if self._iq_finalising.is_set():
                    self.logger.info(
                        "Waiting for an IQ recording that is still closing")
                    if not self.wait_for_the_iq_recording_to_close(
                            _IQ_CLOSE_TIMEOUT_SEC):
                        self.logger.error(
                            "An IQ recording has not finished closing after "
                            "%.0f s; the file may be left unfinished",
                            _IQ_CLOSE_TIMEOUT_SEC)
                self.stop_iq_recording()
                self._stop_iq_record_worker()

            self.handle.close()

    @property
    def iq_finalising(self) -> bool:
        """True while an IQ recording is being closed but is no longer one."""
        return self._iq_finalising.is_set()

    def wait_for_the_iq_recording_to_close(self, timeout: float) -> bool:
        """Wait for an IQ recording that is being closed somewhere else."""
        deadline = time.monotonic() + timeout
        while self._iq_finalising.is_set():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _stop_iq_record_worker(self) -> None:
        """Wake the IQ-recording worker and wait briefly for it to exit."""
        self._iq_record_worker_stop.set()
        try:
            self._iq_record_q.put_nowait(_IQ_RECORD_WORKER_SHUTDOWN)
        except queue.Full:
            pass
        if self._iq_record_worker.is_alive():
            self._iq_record_worker.join(timeout=1.0)

