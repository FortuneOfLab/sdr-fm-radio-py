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

from fm_radio import recording_meta
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

# Least time between underrun lines.  A stream nobody is feeding underruns
# on every callback, and the interesting part is that it is happening at
# all, not each of the fifty a second.
_UNDERRUN_LOG_INTERVAL_SEC: float = 5.0

#: Longest shutdown waits for a recording that is being closed elsewhere.
#: stop_recording is itself bounded at about fifteen seconds; past this
#: something is wrong and the log should say so rather than the process
#: hanging on the way out.
_RECORDING_CLOSE_TIMEOUT_SEC: float = 20.0

#: Longest a new recording waits for the one before it to finish
#: closing.  Shutdown can afford the twenty seconds above; this wait
#: falls on whichever thread pressed the button, so past this, refusing
#: says more than freezing.  A close that is going normally is over in
#: the time it takes to write what is queued.
_PREVIOUS_CLOSE_WAIT_SEC: float = 2.0
# Sentinel placed in the recording queue to mark the end of a session.
# When the worker reaches it, every preceding chunk has been written;
# stop_recording() can then safely close the wave file.
_RECORD_FLUSH_SENTINEL = object()


class _ReadyRecording:
    """A file that is open and waiting to be recorded into.

    Made by :meth:`AudioOutput.prepare_a_recording`, and then either
    installed or discarded.  Nothing about the receiver has changed
    while one of these exists.
    """

    __slots__ = ("path", "wave", "channels", "metadata")

    def __init__(self, path: str, wave_file, channels: int,
                 metadata: dict | None) -> None:
        self.path = path
        self.wave = wave_file
        self.channels = channels
        self.metadata = metadata


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
        # Reentrant: start_recording takes it and then calls install,
        # which takes it again to make its decision against cleanup.
        self._start_lock: threading.RLock = threading.RLock()
        # One recording after another, counted.  A sidecar written for
        # an older number has been overtaken - by the stop that ended
        # that recording, or by the next one - and is thrown away
        # rather than written over what came after it.
        self._record_session: int = 0
        # Held for the sidecar write alone, so two of them cannot
        # interleave inside the file.
        self._sidecar_lock: threading.Lock = threading.Lock()
        # Set by the worker when it reaches the flush sentinel so
        # stop_recording() can close the wave file only after every
        # queued chunk has actually been written.
        self._flush_event: threading.Event = threading.Event()
        self._record_worker_stop: threading.Event = threading.Event()
        self._record_drop_count: int = 0
        # Output-side health, read by telemetry.  Both are plain counters
        # bumped from the thread that noticed the problem: the processing
        # thread for a dropped block, the PortAudio callback for an
        # underrun.  Only ever incremented and read, so no lock is needed.
        self._enqueue_drop_count: int = 0
        self._underrun_count: int = 0
        # When nothing is feeding the output any more - the SDR unplugged,
        # say - every callback underruns, and one debug line each is fifty
        # a second for as long as the process lives.  The count is what
        # matters; the lines are a sample of it.
        self._underrun_last_logged: float = 0.0
        self._underrun_logged_at: int = 0
        # Set for as long as a recording is being closed: the flag above
        # goes down first and the file stays open for the flush, the
        # worker handshake, the close and the sidecar - up to fifteen
        # seconds of a file that is still being written while nothing
        # calls it a recording.  Shutdown waits for this, not for that.
        self._finalising: threading.Event = threading.Event()
        # Set by cleanup().  A bounded join cannot promise that the thread
        # feeding us has stopped, so the stream defends itself rather than
        # trusting that nobody is left to call in.
        self._closed: threading.Event = threading.Event()
        # Makes "is it still open?" and "here is a block" a single step with
        # respect to cleanup, which would otherwise be free to close the
        # stream between the two.
        self._close_lock: threading.Lock = threading.Lock()
        # State for 4-GiB WAV rotation (set in start_recording, used
        # by the worker).  At 48 kHz / 16-bit / 2 ch this only matters
        # for ~6+ hour recordings, but the underlying wave.writeframes
        # crash mode is identical to the IQ recording path.
        self._record_base_path: str | None = None
        self._record_part_index: int = 0
        self._record_bytes_written: int = 0
        self._record_channels: int = AUDIO_CHANNELS
        self._record_meta: dict | None = None
        self._record_worker: threading.Thread = threading.Thread(
            target=self._record_worker_loop,
            name='AudioRecordWorker',
            daemon=True,
        )
        self._record_worker.start()

        # Whether the sound card has been told to start asking for
        # buffers.  Set once, by the first block of audio; read on the
        # realtime path, which is why it is a plain flag and the lock
        # below is only taken on the one call that changes it.
        self._playing: bool = False
        self._play_lock: threading.Lock = threading.Lock()

        # Opened but not started.  A stream that is running is a sound
        # card asking for a buffer every few milliseconds, and until
        # the receiver has produced anything there is nothing to give
        # it: each of those is an underrun, and on this machine there
        # were forty-three of them before the first block arrived -
        # through the JIT pre-warm, which is the better part of a
        # second of nothing.  See _play_from_now_on.
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=AUDIO_CHANNELS,
                rate=int(self.output_rate),
                output=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.callback,
                start=False,
            )
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
                # Padding with silence at all is an underrun: a partly filled
                # callback is an audible gap just the same, and counting only
                # the completely empty ones hides the onset of the problem.
                out[filled:requested_samples] = 0.0
                self._underrun_count += 1
                now = time.monotonic()
                if (now - self._underrun_last_logged
                        >= _UNDERRUN_LOG_INTERVAL_SEC):
                    since = self._underrun_count - self._underrun_logged_at
                    self._underrun_last_logged = now
                    self._underrun_logged_at = self._underrun_count
                    self.logger.debug(
                        "Audio buffer underrun (%d of %d frames; %d since "
                        "the last of these)",
                        requested_samples - filled, requested_samples, since)

            return (out.tobytes(), pyaudio.paContinue)
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}", exc_info=True)
            # Return silence to avoid crashing the audio stream
            silence = np.zeros(frame_count * AUDIO_CHANNELS, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)

    @property
    def closed(self) -> bool:
        """True once :meth:`cleanup` has run; the stream is gone after that."""
        return self._closed.is_set()

    def _play_from_now_on(self) -> None:
        """Start the stream, now that there is a block to play.

        Called once the first block is in the queue, so that the first
        buffer the card asks for is already waiting: starting the
        stream before that would cost exactly the underrun this is
        here to avoid.

        Called from the realtime path, so the flag is checked outside
        the lock and again inside it: two threads can both see a
        stopped stream, and only one of them should start it.
        """
        with self._play_lock:
            if self._playing or self._closed.is_set():
                return
            try:
                self.stream.start_stream()
            except Exception as e:             # pragma: no cover - guard
                self.logger.error("Could not start the audio stream: %s", e,
                                  exc_info=True)
                return
            self._playing = True
            self.logger.info("Audio output started")

    def enqueue_audio(self, left: np.ndarray, right: np.ndarray) -> None:
        """Hand a block to the output, unless it has been closed.

        The check and the hand-off are one step: cleanup() takes the same
        lock before it sets the flag, so it cannot close the stream between
        them and leave a block queued against one that has gone.  The lock
        is uncontended on the realtime path and held only for a put with a
        10 ms ceiling.
        """
        with self._close_lock:
            if self._closed.is_set():
                # A block that arrived after shutdown has nowhere to go, and
                # the stream behind this queue has already been closed.
                return
            self._enqueue_locked(left, right)

    def _enqueue_locked(self, left: np.ndarray, right: np.ndarray) -> None:
        """Body of :meth:`enqueue_audio`; caller holds ``_close_lock``."""
        try:
            left32 = np.asarray(left, dtype=np.float32, copy=False)
            right32 = np.asarray(right, dtype=np.float32, copy=False)
            self.audio_buffer_queue.put((left32, right32), timeout=AUDIO_ENQUEUE_TIMEOUT)
        except queue.Full:
            self._enqueue_drop_count += 1
            self.logger.debug("Audio buffer queue full, dropping audio data")
            return
        except Exception as e:
            self.logger.error(f"Error enqueueing audio: {e}", exc_info=True)
            return
        # After the block is in, not before: starting the stream is
        # telling the card to ask, and the first thing it asks for
        # should already be waiting.
        if not self._playing:
            self._play_from_now_on()

    @property
    def dropped_blocks(self) -> int:
        """Audio blocks discarded because the output queue was full."""
        return self._enqueue_drop_count

    @property
    def underruns(self) -> int:
        """Callbacks that had to pad the output with silence.

        Counts a partly filled callback as well as a completely empty one:
        both are a gap in the audio.
        """
        return self._underrun_count

    @property
    def record_drops(self) -> int:
        """Chunks dropped by the recording queue in this session."""
        return self._record_drop_count

    def start_recording(self, filename: str, channels: int = 2,
                        metadata: dict | None = None) -> None:
        """Start recording audio (asynchronous via worker thread).

        Getting the file ready and putting it in are separate - see
        :meth:`prepare_a_recording` - and this does both, for a caller
        with nothing to decide in between.

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
            metadata: Extra key/value pairs (e.g. centre frequency,
                gain) merged into the ``.json`` metadata sidecar.
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
            ready = self.prepare_a_recording(filename, channels, metadata)
            try:
                session = self.install_a_prepared_recording(ready)
            except BaseException:
                # The file is open and nothing is going to write to it.
                self.discard_a_prepared_recording(ready)
                raise
        self.finish_starting_the_recording(session)

    def prepare_a_recording(self, filename: str, channels: int = 2,
                            metadata: dict | None = None
                            ) -> "_ReadyRecording":
        """Get a file ready to record into, without starting anything.

        Everything slow is here: waiting for the recording before this
        one to finish closing, and opening the file.  None of it
        changes what the receiver is doing, so a caller that has to
        decide under a lock whether to go ahead can do this first and
        hold nothing while it happens.

        Args:
            filename: Filename to save the WAV file.
            channels: Number of channels.
            metadata: Extra key/value pairs for the sidecar.

        Returns:
            What :meth:`install_a_prepared_recording` needs, or what
            :meth:`discard_a_prepared_recording` takes back.

        Raises:
            RecordingError: The output is closed, the one before this
                has not finished closing, or the file would not open.
        """
        if self._closed.is_set():
            # Same reasoning as SDRReceiver.start_iq_recording: the
            # worker that would write the chunks has gone, and no
            # audio is coming to write.
            self.logger.warning(
                "Ignoring start_recording for %s: the audio output is "
                "closed", filename)
            raise RecordingError(
                "Cannot start recording: the audio output is closed")

        self._let_the_last_recording_go(filename)

        # Before anything is made: a channel count the wave module will
        # not take is a failure that should cost nothing, rather than
        # one that leaves a file to be taken back.
        if int(channels) < 1:
            raise RecordingError(
                f"Recording start failed: {channels} channels")

        # Made, not opened: wave.open truncates, and the file it would
        # truncate can be the one a recording is being written to right
        # now.  Creating it exclusively means the only file this can
        # ever empty is the one it just made, which is also the only
        # one discard_a_prepared_recording is entitled to remove.
        try:
            os.close(os.open(filename,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError as e:
            self.logger.error("Recording start failed: %s exists", filename)
            raise RecordingError(
                f"Recording start failed: {filename} already exists") from e
        except OSError as e:
            self.logger.error(
                f"Recording start failed: {e}", exc_info=True,
            )
            raise RecordingError(f"Recording start failed: {e}") from e

        try:
            wf = wave.open(filename, 'wb')
        except (OSError, wave.Error) as e:
            self.logger.error(
                f"Recording start failed: {e}", exc_info=True,
            )
            self._remove_the_file_we_made(filename)
            raise RecordingError(f"Recording start failed: {e}") from e

        try:
            wf.setnchannels(channels)
            wf.setsampwidth(RECORD_SAMPLE_WIDTH)
            wf.setframerate(int(self.output_rate))
        except Exception as e:
            # The handle is open and has no header; closing it is what
            # lets go of the file, and the file has to go before the
            # next attempt at the same name can make it.
            self.logger.error(
                f"Recording start failed: {e}", exc_info=True,
            )
            self._shut_a_file_that_never_started(wf, filename)
            raise RecordingError(f"Recording start failed: {e}") from e
        return _ReadyRecording(filename, wf, channels, metadata)

    def _shut_a_file_that_never_started(self, wave_file,
                                        filename: str) -> None:
        """Close a handle with no header on it, and take the file back.

        ``Wave_write.close`` raises when it has no parameters to write
        a header from, but it lets go of the file underneath on its way
        out regardless - which is the part that matters, because on
        Windows a file that is still open cannot be removed.
        """
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

    def install_a_prepared_recording(self, ready: "_ReadyRecording") -> int:
        """Put a prepared file in and start recording into it.

        The quick half: a queue drained, a handle installed, a flag
        raised.  No file is opened, written or closed, so a caller
        holding a lock across this is holding it for a few dozen
        assignments.  The sidecar is
        :meth:`finish_starting_the_recording`, afterwards and outside
        whatever lock the caller is holding.

        Taken under ``_start_lock``, which cleanup also takes: a
        recording cannot be installed into an output that has been
        shut down, leaving a handle nothing will ever close.

        Returns:
            The session number to hand to
            :meth:`finish_starting_the_recording`.

        Raises:
            RecordingError: The output was closed while this was being
                prepared.
        """
        with self._start_lock:
            if self._closed.is_set():
                raise RecordingError(
                    "Cannot start recording: the audio output is closed")
            # Atomic install w.r.t. the realtime ``record()`` path.
            with self._enqueue_lock:
                self._drain_record_queue()
                self._flush_event.clear()
                self._record_drop_count = 0
                with self.record_lock:
                    self.record_wave = ready.wave
                    self._record_base_path = ready.path
                    self._record_part_index = 0
                    self._record_bytes_written = 0
                    self._record_channels = ready.channels
                self.recording = True

            self._record_session += 1
            session = self._record_session
            self._record_meta = {
                "type": "audio",
                "file": recording_meta.part_list(
                    ready.path, 0, self._make_rotated_record_path,
                )[0],
                "sample_rate_hz": int(self.output_rate),
                "channels": int(ready.channels),
                "started_at": recording_meta.now_iso(),
            }
            if ready.metadata:
                self._record_meta.update(ready.metadata)
        return session

    def finish_starting_the_recording(self, session: int) -> None:
        """Write the sidecar and say so, for the recording *session* began.

        Everything about starting a recording that touches a file: the
        sidecar, which is opened, written and closed, and the log line,
        which on this program's settings goes to a handler that writes
        one too.  Neither is for under a lock anybody else wants, and
        the caller is expected to have let go of its own first.

        A session that has since been overtaken - stopped, or replaced
        by a later recording - writes nothing: the outcome the stop
        recorded is the last word, and a start that was slow to get
        here must not put its own beginning back over it.
        """
        with self._sidecar_lock:
            if session != self._record_session:
                self.logger.debug(
                    "Not writing the sidecar for recording %d; the current "
                    "one is %d", session, self._record_session)
                return
            path = self._record_base_path
            meta = getattr(self, "_record_meta", None)
            if path is None or not meta:
                return
            recording_meta.write_sidecar(path, meta, self.logger)
        self.logger.info(f"Recording started: {path}")

    def discard_a_prepared_recording(self, ready: "_ReadyRecording") -> None:
        """Give back a file that is not going to be recorded into.

        The caller decided against it after the file was made, so the
        file goes too: an empty WAV left behind is a recording that
        never happened, and it would be the only trace of one.
        """
        try:
            ready.wave.close()
        except Exception as e:                  # pragma: no cover - guard
            self.logger.debug("Could not close %s: %s", ready.path, e)
        # Only ever the file prepare_a_recording created: it refuses to
        # take over one that was already there, so this cannot remove
        # somebody else's.
        self._remove_the_file_we_made(ready.path)

    def _let_the_last_recording_go(self, filename: str) -> None:
        """Wait for a recording that is still closing, or refuse this one.

        A recording being closed is no longer a recording, but it still
        owns the wave handle and the queue: it closes whatever is
        installed by the time it reaches them, which would be the one
        about to be started here.  Tuning starts that close on a thread
        of its own, so the two can easily be a button press apart.

        Args:
            filename: What is being started, for the log.

        Raises:
            RecordingError: The last one is still closing after
                ``_PREVIOUS_CLOSE_WAIT_SEC``.
        """
        if not self._finalising.is_set():
            return
        self.logger.info(
            "Waiting for the previous recording to finish closing before "
            "starting %s", filename)
        if self.wait_for_the_recording_to_close(_PREVIOUS_CLOSE_WAIT_SEC):
            return
        self.logger.error(
            "Refusing to start %s: the previous recording has been closing "
            "for %.0f s", filename, _PREVIOUS_CLOSE_WAIT_SEC)
        raise RecordingError(
            "Cannot start recording: the previous recording is still "
            "closing")

    def stop_recording(self) -> None:
        """Stop recording, flush pending writes, and close the file.

        The wave file is closed only after the worker has actually
        written every chunk that was queued before stop_recording was
        called: a flush sentinel is pushed through the queue and the
        worker sets ``_flush_event`` once it pops the sentinel, by
        which point all preceding chunks have been processed under
        ``record_lock``.
        """
        if self.begin_stopping_the_recording():
            self.finish_stopping_the_recording()

    def begin_stopping_the_recording(self) -> bool:
        """Stop taking audio for the recording, and nothing else.

        The two halves of stopping cost very different amounts.  This
        one is a flag under a lock and returns at once; the other is a
        handshake with the worker and can take fifteen seconds.  A
        caller who must not carry on putting the new station into the
        old station's file - tuning - needs this half to have happened
        before it goes on, and can leave the other to a thread.

        Returns:
            True when this call is the one that took the recording, and
            therefore owes it a ``finish_stopping_the_recording``.
            False when there was nothing to stop.
        """
        with self._enqueue_lock:
            if not self.recording:
                self.logger.debug("stop_recording called but not currently recording")
                return False
            # Stop further enqueues from the realtime path.  Because we
            # hold _enqueue_lock, any record() that has already passed
            # its flag check has also completed its put before us.
            self.recording = False
            # From here the file is open and nothing calls it a
            # recording.  Anybody who needs it finished waits on this.
            self._finalising.set()
        return True

    def finish_stopping_the_recording(self) -> None:
        """Flush what was queued, close the file and write the sidecar.

        The slow half, for whoever ``begin_stopping_the_recording`` gave
        the recording to.  Whatever happens, the file stops being called
        one that is closing, or shutdown would wait out its timeout.
        """
        try:
            self._finish_the_recording()
        finally:
            self._finalising.clear()

    def _finish_the_recording(self) -> None:
        """Flush what is queued, close the file and write the sidecar."""

        # Push a flush sentinel.  The worker writes every chunk before
        # the sentinel and then sets _flush_event.
        self._flush_event.clear()
        if not self._record_worker.is_alive():
            # Nothing is left to answer the sentinel, so waiting for one
            # would just be fifteen seconds of nothing happening.
            self.logger.warning(
                "The record worker is no longer running; closing the file "
                "without waiting for a flush",
            )
        else:
            try:
                self._record_q.put(_RECORD_FLUSH_SENTINEL, timeout=5.0)
            except queue.Full:
                self.logger.warning(
                    "Could not enqueue flush sentinel; tail chunks may be "
                    "lost",
                )
            else:
                if not self._flush_event.wait(timeout=10.0):
                    self.logger.warning(
                        "Recording flush did not complete within 10 s; "
                        "closing anyway",
                    )

        with self.record_lock:
            parts_count = self._record_part_index + 1
            base_path = self._record_base_path
            part_index = self._record_part_index
            drops = self._record_drop_count
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

        # Finalise the metadata sidecar with the session outcome.  Under
        # the same lock as the one the start writes through, and last:
        # the session is moved on first, so a start still on its way to
        # writing its own beginning finds itself overtaken and leaves
        # this alone.
        with self._sidecar_lock:
            if base_path is not None and getattr(self, "_record_meta", None):
                self._record_meta.update({
                    "stopped_at": recording_meta.now_iso(),
                    "parts": recording_meta.part_list(
                        base_path, part_index, self._make_rotated_record_path,
                    ),
                    "dropped_chunks": int(drops),
                })
                recording_meta.write_sidecar(
                    base_path, self._record_meta, self.logger,
                )
            self._record_session += 1

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

    @property
    def finalising(self) -> bool:
        """True while a recording is being closed but is no longer one."""
        return self._finalising.is_set()

    def wait_for_the_recording_to_close(self, timeout: float) -> bool:
        """Wait for a recording that is being closed somewhere else.

        Returns False if it is still going by then, which is the caller's
        cue to say so rather than to keep waiting.
        """
        deadline = time.monotonic() + timeout
        while self._finalising.is_set():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def cleanup(self) -> None:
        """Stop the audio stream and terminate PyAudio.

        Refuses further audio first: whoever was feeding this may still be
        running, and everything below is about to go away.  Safe to call
        more than once.
        """
        # Under the lock: a block already on its way in finishes being
        # queued before the flag goes up, and one that has not started sees
        # the flag rather than the stream disappearing under it.
        with self._close_lock:
            self._closed.set()
        try:
            # A recording part way through starting finishes installing
            # itself before the teardown, so it is closed properly rather
            # than left behind with a worker that has already gone.  One
            # that starts after this is refused.
            with self._start_lock:
                # A recording being closed somewhere else is still an
                # open file: the flag went down when the closing started,
                # not when it finished, so asking the flag would skip it
                # and leave the file unfinished.
                if self._finalising.is_set():
                    self.logger.info(
                        "Waiting for a recording that is still closing")
                    if not self.wait_for_the_recording_to_close(
                            _RECORDING_CLOSE_TIMEOUT_SEC):
                        self.logger.error(
                            "A recording has not finished closing after "
                            "%.0f s; the file may be left unfinished",
                            _RECORDING_CLOSE_TIMEOUT_SEC)
                # Stop recording if active
                if self.recording:
                    self.logger.info(
                        "Stopping active recording during cleanup")
                    self.stop_recording()

                # Tell the recording worker to exit.
                self._record_worker_stop.set()
                try:
                    self._record_q.put_nowait(_RECORD_WORKER_SHUTDOWN)
                except queue.Full:
                    pass
                if self._record_worker.is_alive():
                    self._record_worker.join(timeout=1.0)

            # A stream that was never started has nothing to stop, and
            # PortAudio is entitled to object to being asked.
            if self._playing:
                self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
            self.logger.info("Audio output cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during audio cleanup: {e}", exc_info=True)
