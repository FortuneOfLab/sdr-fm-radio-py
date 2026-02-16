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

import math

import numpy as np
import scipy.signal as signal
from numba import njit


# --------------------------------------------------
# Deemphasis IIR Filter Class (Numba optimized)
# --------------------------------------------------
@njit
def deemphasis_iir_process_numba(x, alpha, prev_output):
    """
    Numba-optimized de-emphasis IIR filter processing.

    Args:
        x (ndarray): Input audio signal array.
        alpha (float): Filter coefficient.
        prev_output (float): Previous output sample for filter state.

    Returns:
        tuple: (y, prev) where y is filtered output array and prev is last output value.
    """
    y = np.empty_like(x)
    prev = prev_output
    for i in range(x.shape[0]):
        y[i] = (1.0 - alpha) * x[i] + alpha * prev
        prev = y[i]
    return y, prev


class DeemphasisIIRFilter:
    """
    IIR filter for de-emphasis processing in FM broadcasting.

    Args:
        sample_rate (float): Audio signal sample rate.
        tau (float): Time constant (e.g., 50e-6 seconds).
    """
    def __init__(self, sample_rate, tau):
        self.sample_rate = sample_rate
        self.tau = tau
        # Calculate filter coefficient alpha using an exponential function
        self.alpha = math.exp(-1.0 / (sample_rate * tau))
        self.prev_output = 0.0

    def process(self, x):
        """
        Apply de-emphasis processing to the input signal using a Numba-optimized loop.

        Args:
            x (ndarray): Input audio signal array.

        Returns:
            ndarray: Audio signal after filtering.
        """
        y, self.prev_output = deemphasis_iir_process_numba(x, self.alpha, self.prev_output)
        return y


# --------------------------------------------------
# Filter Classes (LowpassFilter & BandpassFilter)
# --------------------------------------------------
class LowpassFilter:
    """
    Butterworth lowpass filter (streaming using lfilter with state)
    """
    def __init__(self, order, cutoff, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, cutoff / nyquist, btype="low", analog=False)
        # filter state for streaming: length = max(len(a), len(b)) - 1
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)

    def apply(self, data):
        """
        Apply the filter to streaming chunk, preserving state.
        """
        y, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        return y


class BandpassFilter:
    """
    Butterworth bandpass filter (streaming using lfilter with state)
    """
    def __init__(self, order, lowcut, highcut, sample_rate):
        nyquist = sample_rate / 2.0
        self.b, self.a = signal.butter(order, [lowcut / nyquist, highcut / nyquist],
                                        btype="band", analog=False)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)

    def apply(self, data):
        """
        Apply the bandpass filter to streaming chunk, preserving state.
        """
        y, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        return y
