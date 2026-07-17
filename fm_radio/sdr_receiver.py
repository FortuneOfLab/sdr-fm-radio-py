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
        self.data_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=SDR_QUEUE_MAXSIZE)
        self.iq_recording: bool = False
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
        # Serialises concurrent ``start_iq_recording`` callers so only
        # one reaches ``wave.open`` (the file-truncating step).
        # Distinct from ``_iq_enqueue_lock`` so the SDR callback is
        # never blocked during the slow file open.
        self._iq_start_lock: threading.Lock = threading.Lock()
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

    def set_center_frequency(self, freq: float) -> None:
        """Change the center frequency."""
        try:
            self.center_freq = freq
            self.sdr.center_freq = freq
            self.logger.info(f"Center frequency set to {freq/1e6:.1f} MHz")
        except OSError as e:
            self.logger.error(f"Failed to set center frequency to {freq/1e6:.1f} MHz: {e}")
            raise SDRDeviceError(f"Failed to set center frequency: {e}") from e

    def get_center_frequency(self) -> float:
        """Retrieve the current center frequency."""
        return self.sdr.center_freq

    def set_gain(self, gain: float) -> None:
        """Set gain value (for manual mode)."""
        try:
            self.sdr.set_gain(gain)
            self.logger.info(f"Gain set to {gain:.1f} dB")
        except OSError as e:
            self.logger.error(f"Failed to set gain to {gain:.1f} dB: {e}")
            raise SDRDeviceError(f"Failed to set gain: {e}") from e

    def get_gain(self) -> float:
        """Retrieve the current gain value."""
        return self.sdr.get_gain()

    def set_manual_gain_mode(self, manual: bool) -> None:
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
        except OSError as e:
            self.logger.error(f"Failed to set gain mode: {e}")
            raise SDRDeviceError(f"Failed to set gain mode: {e}") from e

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
        try:
            # Convert to numpy array allowing a copy if necessary (NumPy 2.x compatibility).
            iq = np.asarray(iq_samples, dtype=np.complex64)
            self.data_queue.put(iq, block=False)

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

            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(2)           # I/Q
                wf.setsampwidth(2)           # int16
                wf.setframerate(int(self.sample_rate))
            except (OSError, wave.Error) as e:
                self.logger.error(
                    f"IQ recording start failed: {e}", exc_info=True,
                )
                raise RecordingError(
                    f"IQ recording start failed: {e}",
                ) from e

            # Atomic install w.r.t. the SDR callback's enqueue path.
            with self._iq_enqueue_lock:
                self._drain_iq_record_queue()
                self._iq_flush_event.clear()
                self._iq_record_drop_count = 0
                with self.iq_record_lock:
                    self.iq_record_wave = wf
                    self._iq_record_base_path = filename
                    self._iq_record_part_index = 0
                    self._iq_record_bytes_written = 0
                self.iq_recording = True

            # Metadata sidecar (CLI-thread only, never the SDR callback).
            try:
                gain_db = float(self.sdr.get_gain())
            except Exception:
                gain_db = None
            self._iq_record_meta = {
                "type": "iq",
                "file": recording_meta.part_list(
                    filename, 0, self._make_rotated_iq_path,
                )[0],
                "sample_rate_hz": int(self.sample_rate),
                "center_freq_hz": float(self.center_freq),
                "gain_db": gain_db,
                "started_at": recording_meta.now_iso(),
            }
            recording_meta.write_sidecar(
                filename, self._iq_record_meta, self.logger,
            )
            self.logger.info(f"IQ recording started: {filename}")

    def stop_iq_recording(self) -> None:
        """Stop IQ recording, flush pending writes, and close the file.

        Mirrors AudioOutput.stop_recording: the wave file is only
        closed after the worker has actually written every block that
        was queued before stop_iq_recording was called, via a flush
        sentinel + Event handshake.
        """
        with self._iq_enqueue_lock:
            if not self.iq_recording:
                self.logger.debug(
                    "stop_iq_recording called but not currently recording",
                )
                return
            # Stop further enqueues from the SDR callback.  Because we
            # hold _iq_enqueue_lock, any callback that already passed
            # the flag check has also completed its put before us.
            self.iq_recording = False

        # Push a flush sentinel.  The worker writes every block before
        # the sentinel and only then sets _iq_flush_event.
        self._iq_flush_event.clear()
        try:
            self._iq_record_q.put(_IQ_RECORD_FLUSH_SENTINEL, timeout=5.0)
        except queue.Full:
            self.logger.warning(
                "Could not enqueue IQ flush sentinel; tail blocks may be lost",
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

        # Finalise the metadata sidecar with the session outcome.
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
        """Start asynchronous sample retrieval."""
        try:
            self.logger.info("Starting SDR async read")
            self.sdr.read_samples_async(self.callback, num_samples=self.block_size)
        except OSError as e:
            self.logger.error(f"Failed to start SDR async read: {e}")
            raise SDRDeviceError(f"Failed to start SDR async read: {e}") from e

    def stop(self) -> None:
        """Stop asynchronous sample retrieval and close SDR."""
        self.stop_iq_recording()

        # Shut down the IQ record worker.
        self._iq_record_worker_stop.set()
        try:
            self._iq_record_q.put_nowait(_IQ_RECORD_WORKER_SHUTDOWN)
        except queue.Full:
            pass
        if self._iq_record_worker.is_alive():
            self._iq_record_worker.join(timeout=1.0)

        try:
            self.logger.info("Stopping SDR async read")
            self.sdr.cancel_read_async()
        except OSError as e:
            self.logger.warning(f"Error canceling async read: {e}")

        try:
            self.sdr.close()
            self.logger.info("SDR closed successfully")
        except OSError as e:
            self.logger.error(f"Error closing SDR: {e}")
