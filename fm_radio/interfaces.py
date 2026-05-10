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
"""Abstract interfaces for FM receiver subsystems."""

from abc import ABC, abstractmethod

import numpy as np


class FMDemodulatorInterface(ABC):
    """Abstract interface for FM demodulators.

    Defines the contract that all FM demodulator implementations must fulfill.
    Concrete implementations include standard PLL-based and lightweight
    phase-differentiation-based demodulators.
    """

    @abstractmethod
    def process_iq_samples(self, iq_samples: np.ndarray) -> np.ndarray:
        """Convert raw IQ samples into a composite baseband signal.

        Args:
            iq_samples: Complex IQ samples from the SDR receiver.

        Returns:
            Composite baseband signal (float32 array).
        """

    @abstractmethod
    def demodulate(self, composite: np.ndarray) -> tuple:
        """Generate stereo or mono audio from the composite signal.

        Args:
            composite: Composite baseband signal.

        Returns:
            Tuple of (left_channel, right_channel) audio arrays.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (PLL, filters, DC offset, etc.)."""


class SDRReceiverInterface(ABC):
    """Abstract interface for SDR receivers.

    Defines the contract for hardware abstraction of SDR devices.
    """

    @abstractmethod
    def set_center_frequency(self, freq: float) -> None:
        """Change the center (tuning) frequency.

        Args:
            freq: Center frequency in Hz.
        """

    @abstractmethod
    def get_center_frequency(self) -> float:
        """Retrieve the current center frequency.

        Returns:
            Current center frequency in Hz.
        """

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """Set the receiver gain value.

        Args:
            gain: Gain in dB.
        """

    @abstractmethod
    def get_gain(self) -> float:
        """Retrieve the current gain value.

        Returns:
            Current gain in dB.
        """

    @abstractmethod
    def set_manual_gain_mode(self, manual: bool) -> None:
        """Toggle between manual gain and AGC mode.

        Args:
            manual: True for manual gain, False for AGC.
        """

    @abstractmethod
    def start(self) -> None:
        """Start asynchronous sample retrieval."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sample retrieval and release resources."""


class AudioOutputInterface(ABC):
    """Abstract interface for audio output subsystems.

    Defines the contract for audio playback and recording.
    """

    @abstractmethod
    def enqueue_audio(self, left: np.ndarray, right: np.ndarray) -> None:
        """Enqueue stereo audio data for playback.

        Args:
            left: Left channel audio samples (float32).
            right: Right channel audio samples (float32).
        """

    @abstractmethod
    def start_recording(self, filename: str, channels: int = 2) -> None:
        """Start recording audio to a WAV file.

        Args:
            filename: Output WAV file path.
            channels: Number of audio channels.
        """

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop the current recording session."""

    @abstractmethod
    def record(self, stereo_audio: np.ndarray) -> None:
        """Write audio data to the recording file.

        Args:
            stereo_audio: Interleaved stereo audio data.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Stop audio streams and release resources."""
