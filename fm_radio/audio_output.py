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

import queue
import wave
import threading
import logging
from collections import deque

import numpy as np
import pyaudio

from fm_radio.interfaces import AudioOutputInterface
from fm_radio.constants import (
    AUDIO_OUTPUT_RATE, AUDIO_FRAMES_PER_BUFFER, AUDIO_QUEUE_MAXSIZE,
    AUDIO_CHANNELS, AUDIO_ENQUEUE_TIMEOUT,
    RECORD_SAMPLE_WIDTH, RECORD_MAX_INT16,
)


class AudioOutput(AudioOutputInterface):
    """
    Audio output and recording management class

    Uses PyAudio for audio output and recording.
    """
    def __init__(self, output_rate=AUDIO_OUTPUT_RATE, frames_per_buffer=AUDIO_FRAMES_PER_BUFFER):
        self.logger = logging.getLogger('fm_receiver.AudioOutput')
        self.output_rate = output_rate
        self.frames_per_buffer = frames_per_buffer
        self.audio_buffer_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self.recording = False
        self.record_wave = None
        self.record_lock = threading.Lock()
        # replace large concatenating buffer with deque of numpy arrays
        self._buffer_deque = deque()
        self._buffer_len = 0  # total number of float32 samples in deque

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
        except Exception as e:
            self.logger.error(f"Failed to initialize audio output: {e}")
            raise

    def callback(self, in_data, frame_count, time_info, status):
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

    def enqueue_audio(self, left, right):
        try:
            left32 = np.asarray(left, dtype=np.float32, copy=False)
            right32 = np.asarray(right, dtype=np.float32, copy=False)
            self.audio_buffer_queue.put((left32, right32), timeout=AUDIO_ENQUEUE_TIMEOUT)
        except queue.Full:
            self.logger.debug("Audio buffer queue full, dropping audio data")
        except Exception as e:
            self.logger.error(f"Error enqueueing audio: {e}", exc_info=True)

    def start_recording(self, filename, channels=2):
        """
        Start recording audio.

        Args:
            filename (str): Filename to save the WAV file.
            channels (int): Number of channels.
        """
        with self.record_lock:
            if self.recording:
                self.logger.warning("Already recording")
                print("Already recording.")
                return
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(RECORD_SAMPLE_WIDTH)
                wf.setframerate(int(self.output_rate))
                self.record_wave = wf
                self.recording = True
                self.logger.info(f"Recording started: {filename}")
                print(f"Recording started: {filename}")
            except Exception as e:
                self.logger.error(f"Recording start failed: {e}", exc_info=True)
                print("Recording start failed:", e)

    def stop_recording(self):
        """Stop recording and close the file."""
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                try:
                    self.record_wave.close()
                    self.logger.info("Recording stopped")
                    print("Recording stopped.")
                except Exception as e:
                    self.logger.error(f"Error closing recording file: {e}", exc_info=True)
                finally:
                    self.record_wave = None
                    self.recording = False
            else:
                self.logger.debug("stop_recording called but not currently recording")
                print("Not currently recording.")

    def record(self, stereo_audio):
        """
        Write audio data to file if recording.

        Args:
            stereo_audio (ndarray): Stereo audio data.
        """
        with self.record_lock:
            if self.recording and self.record_wave is not None:
                try:
                    # Clip to [-1,1] before converting to int16 to avoid overflow/distortion
                    clipped = np.clip(stereo_audio, -1.0, 1.0)
                    int16_audio = np.int16(clipped * RECORD_MAX_INT16)
                    self.record_wave.writeframes(int16_audio.tobytes())
                except Exception as e:
                    self.logger.error(f"Error writing audio to file: {e}", exc_info=True)

    def cleanup(self):
        """Stop audio stream and terminate PyAudio instance."""
        try:
            # Stop recording if active
            if self.recording:
                self.logger.info("Stopping active recording during cleanup")
                self.stop_recording()

            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
            self.logger.info("Audio output cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during audio cleanup: {e}", exc_info=True)
