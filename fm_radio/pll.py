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
"""Phase-Locked Loop (PLL) for FM demodulation."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit
def pll_demodulate(
    iq_samples: np.ndarray, Kp: float, Ki: float, state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba optimized FM demodulation using PLL

    Args:
        iq_samples (ndarray): Array of complex IQ samples.
        Kp (float): Proportional gain of the PLL.
        Ki (float): Integral gain of the PLL.
        state (ndarray): Array holding [phase, integrator] state.

    Returns:
        tuple: (phase_out, freq_out)
            phase_out (ndarray): Phase after processing each sample.
            freq_out (ndarray): Frequency estimate after each sample.
    """
    n = iq_samples.shape[0]
    phase_out = np.empty(n, dtype=np.float32)
    freq_out = np.empty(n, dtype=np.float32)
    errors = np.empty(n, dtype=np.float32)
    phase = state[0]
    integrator = state[1]
    for i in range(n):
        sample = iq_samples[i]
        vco_real = math.cos(phase)
        vco_imag = math.sin(phase)
        prod_real = sample.real * vco_real + sample.imag * vco_imag
        prod_imag = -sample.real * vco_imag + sample.imag * vco_real
        error = math.atan2(prod_imag, prod_real)
        errors[i] = error
        integrator += Ki * error
        freq_est = Kp * error + integrator
        phase += freq_est
        phase = np.mod(phase, 2 * math.pi)
        phase_out[i] = phase
        freq_out[i] = freq_est
    state[0] = phase
    state[1] = integrator
    return phase_out, freq_out


class PLL:
    """FM demodulation using a Phase Locked Loop (PLL).

    Extracts phase or frequency information from IQ samples.
    """

    def __init__(self, Kp: float, Ki: float, return_phase: bool = False) -> None:
        self._Kp: float = Kp
        self._Ki: float = Ki
        self.return_phase: bool = return_phase
        self.state: np.ndarray = np.zeros(2, dtype=np.float32)  # [phase, integrator]
        self.last_freq: float = 0.0

    def process(self, iq_samples: np.ndarray) -> np.ndarray:
        """Process IQ samples with the PLL to generate demodulated signal.

        Args:
            iq_samples: Array of IQ samples.

        Returns:
            Demodulated signal (phase or frequency).
        """
        phase_out, freq_out = pll_demodulate(iq_samples, self._Kp, self._Ki, self.state)
        if freq_out.size > 0:
            self.last_freq = freq_out[-1]
        return phase_out if self.return_phase else freq_out

    def get_last_freq(self) -> float:
        """Retrieve the last frequency estimate."""
        return self.last_freq

    def reset(self) -> None:
        """Reset PLL state (phase and integrator)."""
        self.state[:] = 0.0

    def set_Kp(self, Kp: float) -> None:
        """Set the proportional gain."""
        self._Kp = Kp

    def get_Kp(self) -> float:
        """Get the proportional gain."""
        return self._Kp

    def set_Ki(self, Ki: float) -> None:
        """Set the integral gain."""
        self._Ki = Ki

    def get_Ki(self) -> float:
        """Get the integral gain."""
        return self._Ki
