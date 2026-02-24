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
"""Filter classes for FM signal processing."""

from __future__ import annotations

import math

import numpy as np
import scipy.signal as signal
from numba import njit


# --------------------------------------------------
# Deemphasis IIR Filter Class (Numba optimized)
# --------------------------------------------------
@njit
def deemphasis_iir_process_numba(
    x: np.ndarray, alpha: float, prev_output: float,
) -> tuple[np.ndarray, float]:
    """Numba-optimized de-emphasis IIR filter processing.

    Args:
        x: Input audio signal array.
        alpha: Filter coefficient.
        prev_output: Previous output sample for filter state.

    Returns:
        Tuple of (filtered output array, last output value).
    """
    y = np.empty_like(x)
    prev = prev_output
    for i in range(x.shape[0]):
        y[i] = (1.0 - alpha) * x[i] + alpha * prev
        prev = y[i]
    return y, prev


class DeemphasisIIRFilter:
    """IIR filter for de-emphasis processing in FM broadcasting.

    Args:
        sample_rate: Audio signal sample rate.
        tau: Time constant (e.g., 50e-6 seconds).
    """

    def __init__(self, sample_rate: float, tau: float) -> None:
        self.sample_rate: float = sample_rate
        self.tau: float = tau
        self.alpha: float = math.exp(-1.0 / (sample_rate * tau))
        self.prev_output: float = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply de-emphasis processing to the input signal.

        Args:
            x: Input audio signal array.

        Returns:
            Audio signal after filtering.
        """
        y, self.prev_output = deemphasis_iir_process_numba(x, self.alpha, self.prev_output)
        return y


# --------------------------------------------------
# Filter Classes (LowpassFilter & BandpassFilter)
# --------------------------------------------------
class LowpassFilter:
    """Butterworth lowpass filter (streaming using lfilter with state)."""

    def __init__(self, order: int, cutoff: float, sample_rate: float) -> None:
        nyquist = sample_rate / 2.0
        self.sos: np.ndarray = signal.butter(
            order, cutoff / nyquist, btype="low", analog=False, output="sos",
        )
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y


class BandpassFilter:
    """Butterworth bandpass filter (streaming using lfilter with state)."""

    def __init__(self, order: int, lowcut: float, highcut: float, sample_rate: float) -> None:
        nyquist = sample_rate / 2.0
        self.sos: np.ndarray = signal.butter(
            order, [lowcut / nyquist, highcut / nyquist], btype="band",
            analog=False, output="sos",
        )
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the bandpass filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y


class NotchFilter:
    """IIR notch (band-reject) filter (streaming using lfilter with state)."""

    def __init__(self, freq: float, Q: float, sample_rate: float) -> None:
        b, a = signal.iirnotch(freq, Q, sample_rate)
        self.sos: np.ndarray = signal.tf2sos(b, a)
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the notch filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y


# --------------------------------------------------
# Stateful polyphase resampler (overlap-save)
# --------------------------------------------------
class StatefulResampler:
    """Overlap-save wrapper for ``scipy.signal.resample_poly``.

    ``resample_poly`` is stateless: each call zero-pads the block edges,
    producing boundary transients that manifest as impulse noise in
    downstream stereo L-R processing.  This class keeps the tail of the
    previous input block and prepends it to the current block so that
    the polyphase filter sees a continuous stream.
    """

    def __init__(self, up: int, down: int, window: object = None) -> None:
        self.up: int = int(up)
        self.down: int = int(down)
        self.window: object = window
        # Half-length of the internal polyphase FIR (scipy default)
        half_len: int = 10 * max(self.up, self.down)
        self._overlap: int = half_len * 2
        self._prev_tail: np.ndarray | None = None

    # ----- public API -----

    def process(self, x: np.ndarray) -> np.ndarray:
        """Resample *x* while maintaining block-to-block continuity."""
        if self._prev_tail is None:
            self._prev_tail = x[-self._overlap:].copy()
            return self._resample(x)

        extended = np.concatenate([self._prev_tail, x])
        self._prev_tail = x[-self._overlap:].copy()
        y_ext = self._resample(extended)
        overlap_out = int(np.round(self._overlap * self.up / self.down))
        return y_ext[overlap_out:]

    def reset(self) -> None:
        """Discard saved tail so the next call starts fresh."""
        self._prev_tail = None

    # ----- internal -----

    def _resample(self, x: np.ndarray) -> np.ndarray:
        if self.window is not None:
            return signal.resample_poly(x, self.up, self.down, window=self.window)
        return signal.resample_poly(x, self.up, self.down)
