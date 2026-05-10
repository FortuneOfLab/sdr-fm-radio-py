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
"""Audio output and recording management."""

from __future__ import annotations

import os
import queue
import time
import wave
import threading
import logging
from collections import deque

import numpy as np
import pyaudio

from fm_radio.interfaces import AudioOutputInterface
from fm_radio.exceptions import AudioOutputError, RecordingError
from fm_radio.constants import (
    AUDIO_OUTPUT_RATE, AUDIO_FRAMES_PER_BUFFER, AUDIO_QUEUE_MAXSIZE,
    AUDIO_CHANNELS, AUDIO_ENQUEUE_TIMEOUT,
    RECORD_SAMPLE_WIDTH, RECORD_MAX_INT16,
    RECORD_QUEUE_MAXSIZE, AUDIO_RECORD_ROTATE_THRESHOLD_BYTES,
)


# Sentinel placed in the recording queue to wake the worker for shutdown.
_RECORD_WORKER_SHUTDOWN = object()
# Sentinel placed in the recording queue to mark the end of a session.
# When the worker reaches it, every preceding chunk has been written;
# stop_recording() can then safely close the wave file.
_RECORD_FLUSH_SENTINEL = object()


class AudioOutput(AudioOutputInterface):
    """
    Audio output and recording management class

    Uses PyAudio for audio output and recording.
    """
    def __init__(
        self,
        output_rate: int = AUDIO_OUTPUT_RATE,
        frames_per_buffer: int = AUDIO_FRAMES_PER_BUFFER,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger('fm_receiver.AudioOutput')
        self.output_rate: int = output_rate
        self.frames_per_buffer: int = frames_per_buffer
        self.audio_buffer_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(
            maxsize=AUDIO_QUEUE_MAXSIZE,
        )
        self.recording: bool = False
        self.record_wave: wave.Wave_write | None = None
        self.record_lock: threading.Lock = threading.Lock()
        self._buffer_deque: deque[np.ndarray] = deque()
        self._buffer_len: int = 0

        # Asynchronous recording worker: keeps the realtime processing
        # thread off disk I/O.  WAV writes are buffered through Python's
        # io stack and the OS page cache, both of which can stall for
        # 100-1000 ms (Defender scans, dirty-page flush, sync apps);
        # those stalls happen here in the worker rather than in
        # FMReceiverController.processing_thread.
        self._record_q: queue.Queue[object] = queue.Queue(
            maxsize=RECORD_QUEUE_MAXSIZE,
        )
        # Serialises record() / start_recording / stop_recording with
        # respect to the ``recording`` flag and the queue.put for the
        # flush sentinel.  Held only for fast operations (flag check +
        # queue.put_nowait); the worker does NOT take this lock so
        # disk-write stalls in the worker do not propagate here.
        self._enqueue_lock: threading.Lock = threading.Lock()
        # Serialises concurrent ``start_recording`` calls so only one
        # thread reaches ``wave.open`` per session.  Distinct from
        # ``_enqueue_lock`` so the realtime path (``record()``) is
        # never blocked during the slow file open.
        self._start_lock: threading.Lock = threading.Lock()
        # Set by the worker when it reaches the flush sentinel so
        # stop_recording() can close the wave file only after every
        # queued chunk has actually been written.
        self._flush_event: threading.Event = threading.Event()
        self._record_worker_stop: threading.Event = threading.Event()
        self._record_drop_count: int = 0
        # State for 4-GiB WAV rotation (set in start_recording, used
        # by the worker).  At 48 kHz / 16-bit / 2 ch this only matters
        # for ~6+ hour recordings, but the underlying wave.writeframes
        # crash mode is identical to the IQ recording path.
        self._record_base_path: str | None = None
        self._record_part_index: int = 0
        self._record_bytes_written: int = 0
        self._record_channels: int = AUDIO_CHANNELS
        self._record_worker: threading.Thread = threading.Thread(
            target=self._record_worker_loop,
            name='AudioRecordWorker',
            daemon=True,
        )
        self._record_worker.start()

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=AUDIO_CHANNELS,
                rate=int(self.output_rate),
                output=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            self.logger.info(f"Audio output initialized: rate={output_rate}Hz, buffer={frames_per_buffer}")
        except OSError as e:
            self.logger.error(f"Failed to initialize audio output: {e}")
            raise AudioOutputError(f"Failed to initialize audio output: {e}") from e

    def callback(
        self, in_data: bytes | None, frame_count: int, time_info: dict, status: int,
    ) -> tuple[bytes, int]:
        try:
            if status:
                self.logger.warning(f"Audio callback status: {status}")

            requested_samples = frame_count * AUDIO_CHANNELS  # stereo interleaved samples
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
            silence = np.zeros(frame_count * AUDIO_CHANNELS, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)

    def enqueue_audio(self, left: np.ndarray, right: np.ndarray) -> None:
        try:
            left32 = np.asarray(left, dtype=np.float32, copy=False)
            right32 = np.asarray(right, dtype=np.float32, copy=False)
            self.audio_buffer_queue.put((left32, right32), timeout=AUDIO_ENQUEUE_TIMEOUT)
        except queue.Full:
            self.logger.debug("Audio buffer queue full, dropping audio data")
        except Exception as e:
            self.logger.error(f"Error enqueueing audio: {e}", exc_info=True)

    def start_recording(self, filename: str, channels: int = 2) -> None:
        """Start recording audio (asynchronous via worker thread).

        Thread safety design:
          - ``_start_lock`` serialises concurrent ``start_recording``
            callers, so only the winner reaches ``wave.open`` and
            losers see ``self.recording = True`` and return without
            touching their target path.  The realtime ``record()``
            path does not take this lock, so the slow ``wave.open``
            call inside it does not block audio writes.
          - ``_enqueue_lock`` is taken only briefly inside that, to
            atomically drain stale queue items, install the wave
            handle and flip ``self.recording``.

        Args:
            filename: Filename to save the WAV file.
            channels: Number of channels.
        """
        # Serialise all start callers — the wave.open below is the
        # only place that can truncate the target file, and we want
        # exactly one caller per session to reach it.
        with self._start_lock:
            if self.recording:
                self.logger.warning(
                    "Already recording; ignoring duplicate "
                    "start_recording for %s", filename,
                )
                return

            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(RECORD_SAMPLE_WIDTH)
                wf.setframerate(int(self.output_rate))
            except (OSError, wave.Error) as e:
                self.logger.error(
                    f"Recording start failed: {e}", exc_info=True,
                )
                raise RecordingError(f"Recording start failed: {e}") from e

            # Atomic install w.r.t. the realtime ``record()`` path.
            with self._enqueue_lock:
                self._drain_record_queue()
                self._flush_event.clear()
                self._record_drop_count = 0
                with self.record_lock:
                    self.record_wave = wf
                    self._record_base_path = filename
                    self._record_part_index = 0
                    self._record_bytes_written = 0
                    self._record_channels = channels
                self.recording = True
            self.logger.info(f"Recording started: {filename}")

    def stop_recording(self) -> None:
        """Stop recording, flush pending writes, and close the file.

        The wave file is closed only after the worker has actually
        written every chunk that was queued before stop_recording was
        called: a flush sentinel is pushed through the queue and the
        worker sets ``_flush_event`` once it pops the sentinel, by
        which point all preceding chunks have been processed under
        ``record_lock``.
        """
        with self._enqueue_lock:
            if not self.recording:
                self.logger.debug("stop_recording called but not currently recording")
                return
            # Stop further enqueues from the realtime path.  Because we
            # hold _enqueue_lock, any record() that has already passed
            # its flag check has also completed its put before us.
            self.recording = False

        # Push a flush sentinel.  The worker writes every chunk before
        # the sentinel and then sets _flush_event.
        self._flush_event.clear()
        try:
            self._record_q.put(_RECORD_FLUSH_SENTINEL, timeout=5.0)
        except queue.Full:
            self.logger.warning(
                "Could not enqueue flush sentinel; tail chunks may be lost",
            )
        else:
            if not self._flush_event.wait(timeout=10.0):
                self.logger.warning(
                    "Recording flush did not complete within 10 s; "
                    "closing anyway",
                )

        with self.record_lock:
            parts_count = self._record_part_index + 1
            if self.record_wave is not None:
                try:
                    self.record_wave.close()
                    self.logger.info(
                        "Recording stopped (drops during session: %d, "
                        "parts: %d)",
                        self._record_drop_count, parts_count,
                    )
                except (OSError, wave.Error) as e:
                    self.logger.error(
                        f"Error closing recording file: {e}", exc_info=True,
                    )
                finally:
                    self.record_wave = None

    def record(self, stereo_audio: np.ndarray) -> None:
        """Hand stereo audio to the recording worker.

        Non-blocking from the realtime processing thread's point of view:
        ``_enqueue_lock`` is taken only briefly to atomically pair the
        ``self.recording`` check with the queue.put_nowait, and the
        worker does not take this lock.  The slow disk-write happens
        in the worker thread.  When the queue is full (sustained disk
        stall longer than RECORD_QUEUE_MAXSIZE chunks) the chunk is
        dropped and a warning is logged.

        Args:
            stereo_audio: Stereo audio data.
        """
        with self._enqueue_lock:
            if not self.recording:
                return
            try:
                self._record_q.put_nowait(stereo_audio)
            except queue.Full:
                self._record_drop_count += 1
                self.logger.warning(
                    "Record queue full, dropping chunk (drops=%d, qsize=%d/%d) "
                    "— disk write fell behind realtime",
                    self._record_drop_count,
                    self._record_q.qsize(),
                    self._record_q.maxsize,
                )

    # ------------------------------------------------------------------
    # Recording worker (runs disk writes off the realtime thread)
    # ------------------------------------------------------------------

    def _drain_record_queue(self) -> None:
        """Drop everything currently sitting in the record queue."""
        try:
            while True:
                self._record_q.get_nowait()
        except queue.Empty:
            pass

    @staticmethod
    def _make_rotated_record_path(base_path: str, part_index: int) -> str:
        """Build the rotated-file path ``foo.partNNN.wav`` from ``foo.wav``."""
        root, ext = os.path.splitext(base_path)
        return f"{root}.part{part_index:03d}{ext}"

    def _rotate_record(self, current_bytes: int) -> bool:
        """Close current audio WAV and open the next part file.

        Caller must hold ``self.record_lock``.  Mirror of
        ``SDRReceiver._rotate_iq_recording`` — same crash mode (wave's
        4-GiB header limit), same fix.

        Returns:
            True if rotation succeeded, False otherwise.
        """
        prev_wave = self.record_wave
        prev_path = (
            self._make_rotated_record_path(
                self._record_base_path, self._record_part_index,
            ) if self._record_part_index > 0
            else self._record_base_path
        )
        try:
            if prev_wave is not None:
                prev_wave.close()
        except (OSError, wave.Error) as e:
            self.logger.error(
                "Error closing audio part file %s during rotation: %s",
                prev_path, e, exc_info=True,
            )
        self._record_part_index += 1
        next_path = self._make_rotated_record_path(
            self._record_base_path, self._record_part_index,
        )
        try:
            wf = wave.open(next_path, 'wb')
            # Channel count was set on the first file (typically 2);
            # preserve by querying the previous file's channel count
            # via the base path.  We capture it from the first wave_open
            # at start time to avoid re-reading the closed file.
            wf.setnchannels(self._record_channels)
            wf.setsampwidth(RECORD_SAMPLE_WIDTH)
            wf.setframerate(int(self.output_rate))
        except (OSError, wave.Error) as e:
            self.logger.error(
                "Failed to open rotated audio part file %s: %s — recording "
                "will stop accepting new chunks", next_path, e, exc_info=True,
            )
            self.record_wave = None
            return False
        self.record_wave = wf
        self._record_bytes_written = 0
        self.logger.info(
            "Audio recording rotated to part %d: %s "
            "(previous part wrote ~%.2f GB)",
            self._record_part_index, next_path,
            current_bytes / 1e9,
        )
        return True

    def _record_worker_loop(self) -> None:
        while not self._record_worker_stop.is_set():
            try:
                item = self._record_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _RECORD_WORKER_SHUTDOWN:
                break
            if item is _RECORD_FLUSH_SENTINEL:
                # Every chunk queued before this sentinel has now been
                # written (queue is FIFO and the writeframes path above
                # is on this same thread), so stop_recording() may
                # safely close the file.
                self._flush_event.set()
                continue
            stereo_audio = item
            # Catch any exception so the worker thread does not die
            # mid-recording (e.g. the historical struct.error from the
            # wave module's 4-GiB header limit).  Rotation below
            # prevents that root cause but the broad catch is kept as
            # a safety net.
            try:
                # int16 PCM size: each sample is 2 bytes regardless of
                # interleaving; ``stereo_audio`` already encodes both
                # channels per sample slot in flat float32 form.
                chunk_bytes = int(stereo_audio.size) * 2
                with self.record_lock:
                    if self.record_wave is None:
                        # Wave file already closed (e.g. after a flush
                        # sentinel) — drop the chunk.
                        continue
                    if (self._record_bytes_written + chunk_bytes
                            > AUDIO_RECORD_ROTATE_THRESHOLD_BYTES):
                        if not self._rotate_record(
                            self._record_bytes_written,
                        ):
                            continue
                    clipped = np.clip(stereo_audio, -1.0, 1.0)
                    int16_audio = np.int16(clipped * RECORD_MAX_INT16)
                    self.record_wave.writeframes(int16_audio.tobytes())
                    self._record_bytes_written += chunk_bytes
            except Exception as e:
                self.logger.error(
                    "Unexpected error writing audio to file: %s", e,
                    exc_info=True,
                )

    def cleanup(self) -> None:
        """Stop audio stream and terminate PyAudio instance."""
        try:
            # Stop recording if active
            if self.recording:
                self.logger.info("Stopping active recording during cleanup")
                self.stop_recording()

            # Tell the recording worker to exit.
            self._record_worker_stop.set()
            try:
                self._record_q.put_nowait(_RECORD_WORKER_SHUTDOWN)
            except queue.Full:
                pass
            if self._record_worker.is_alive():
                self._record_worker.join(timeout=1.0)

            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
            self.logger.info("Audio output cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during audio cleanup: {e}", exc_info=True)
