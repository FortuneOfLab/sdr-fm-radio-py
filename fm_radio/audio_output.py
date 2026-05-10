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
    RECORD_QUEUE_MAXSIZE,
)


# Sentinel placed in the recording queue to wake the worker for shutdown.
_RECORD_WORKER_SHUTDOWN = object()


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
        self._record_worker_stop: threading.Event = threading.Event()
        self._record_drop_count: int = 0
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

        Args:
            filename: Filename to save the WAV file.
            channels: Number of channels.
        """
        with self.record_lock:
            if self.recording:
                self.logger.warning("Already recording")
                return
            # Drop any stale items left in the queue from a previous
            # session so the new recording starts clean.
            self._drain_record_queue()
            self._record_drop_count = 0
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(RECORD_SAMPLE_WIDTH)
                wf.setframerate(int(self.output_rate))
                self.record_wave = wf
                self.recording = True
                self.logger.info(f"Recording started: {filename}")
            except (OSError, wave.Error) as e:
                self.logger.error(f"Recording start failed: {e}", exc_info=True)
                raise RecordingError(f"Recording start failed: {e}") from e

    def stop_recording(self) -> None:
        """Stop recording, drain pending writes, and close the file."""
        with self.record_lock:
            if not self.recording:
                self.logger.debug("stop_recording called but not currently recording")
                return
            # Stop further enqueues from the realtime path immediately.
            self.recording = False

        # Wait (up to 5 s) for the worker to flush any backlog before
        # closing the file.  The worker checks the queue every 0.2 s.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if self._record_q.empty():
                break
            time.sleep(0.02)

        with self.record_lock:
            if self.record_wave is not None:
                try:
                    self.record_wave.close()
                    self.logger.info(
                        "Recording stopped (drops during session: %d)",
                        self._record_drop_count,
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
        the queue is bounded, and the slow disk-write happens in the
        worker thread.  When the queue is full (sustained disk stall
        longer than RECORD_QUEUE_MAXSIZE chunks) the chunk is dropped
        and a warning is logged.

        Args:
            stereo_audio: Stereo audio data.
        """
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

    def _record_worker_loop(self) -> None:
        while not self._record_worker_stop.is_set():
            try:
                item = self._record_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _RECORD_WORKER_SHUTDOWN:
                break
            stereo_audio = item
            # Gate writes on the wave file's open/closed state, NOT on
            # self.recording.  stop_recording() flips self.recording to
            # False *before* draining the queue so the realtime path
            # stops enqueueing; if the worker also stopped writing on
            # that flag, queued tail chunks would be discarded instead
            # of flushed.  The wave file remains open until close() is
            # called after the drain completes, which is the right
            # gate.
            with self.record_lock:
                if self.record_wave is None:
                    continue
                try:
                    clipped = np.clip(stereo_audio, -1.0, 1.0)
                    int16_audio = np.int16(clipped * RECORD_MAX_INT16)
                    self.record_wave.writeframes(int16_audio.tobytes())
                except (OSError, wave.Error) as e:
                    self.logger.error(
                        f"Error writing audio to file: {e}", exc_info=True,
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
