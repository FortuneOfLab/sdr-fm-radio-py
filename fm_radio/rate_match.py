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
"""Keeping the audio the receiver makes in step with the card that plays it.

The receiver makes exactly 768 frames of audio for every 16384 IQ
samples, so the audio comes at whatever rate the samples really arrive
at - and measured on the hardware that was 200 to 500 ppm short of
1.024 Msps, with USB transfers lost and nothing to report them.  The
card takes 48 000 frames a second by its own clock regardless.  The
difference comes out of the cushion of audio waiting for the card,
1300 frames a minute at worst, and when that is gone every late block
is a gap.

Two parts:

- :class:`FractionalResampler` turns a block into slightly more or
  slightly fewer frames - a ratio within a couple of thousandths of 1 -
  with a windowed-sinc kernel, carrying its state from block to block
  so the joins are not heard.
- :class:`RateMatcher` decides the ratio from how much audio is
  waiting: a PI loop that holds the cushion at a target.  The integral
  ends up holding the real mismatch, so it is kept across retunes.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class FractionalResampler:
    """Resample stereo audio by a ratio close to 1, block by block.

    ``ratio`` is output frames per input frame.  The kernel is a
    Kaiser-windowed sinc, ``taps`` long, as a function of where the
    output falls between two input frames - and that function is
    fitted with a polynomial of degree ``degree`` in the fraction (a
    Farrow structure).  Each output is then the input window under it
    times ``degree + 1`` fixed filters, one matrix product for the
    whole block, combined by Horner's rule on the fraction.  Measured
    against an ideal resampled tone: within 0.0005 dB from 100 Hz to
    15 kHz and the residual 89 dB down, at 155 us a 768-frame block -
    where tabulating the kernel at 512 phases and interpolating cost
    400 us for the same accuracy.

    At 48 kHz the passband is flat to 18 kHz, above anything the
    receiver makes (its final low-pass is 15 kHz), so a ratio this
    close to 1 neither colours the audio nor folds anything back.

    The output is delayed by ``taps // 2`` input frames, once, at the
    start: 0.33 ms with the defaults.
    """

    def __init__(self, taps: int = 32, degree: int = 5,
                 cutoff: float = 0.9, beta: float = 8.0,
                 fit_points: int = 513) -> None:
        if taps % 2:
            raise ValueError("taps must be even")
        self.taps = int(taps)
        self.half = self.taps // 2
        self.degree = int(degree)
        # The kernel for an output a fraction f of the way from one
        # input frame to the next, sampled across f and fitted.
        frac = np.linspace(0.0, 1.0, int(fit_points))
        offsets = ((np.arange(self.taps) - self.half + 1)[None, :]
                   - frac[:, None])
        window = np.i0(beta * np.sqrt(np.clip(
            1.0 - (offsets / self.half) ** 2, 0.0, None))) / np.i0(beta)
        kernel = cutoff * np.sinc(cutoff * offsets) * window
        kernel /= kernel.sum(axis=1, keepdims=True)    # unity at DC
        powers = np.vander(frac, self.degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(powers, kernel, rcond=None)
        #: How far the polynomial strays from the kernel it stands for.
        self.fit_error = float(np.abs(powers @ coeffs - kernel).max())
        self._coeffs_t = coeffs.T.astype(np.float32).copy()  # (taps, deg+1)
        self.reset()

    def reset(self) -> None:
        """Forget the audio so far, as after a jump in it."""
        self._history = np.zeros((2, self.taps), dtype=np.float32)
        # Where the next output falls, in frames of the buffer that the
        # next block will be appended to.
        self._t = float(self.half)

    def process(self, left: np.ndarray, right: np.ndarray,
                ratio: float) -> "tuple[np.ndarray, np.ndarray]":
        """Resample one block; returns the left and right channels."""
        n = int(left.size)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty
        buf = np.empty((2, self.taps + n), dtype=np.float32)
        buf[:, :self.taps] = self._history
        buf[0, self.taps:] = left
        buf[1, self.taps:] = right
        length = buf.shape[1]
        step = 1.0 / float(ratio)
        last = length - self.half - 1          # the latest an output can fall
        count = (int((last - self._t) // step) + 1
                 if self._t <= last else 0)
        t = self._t + step * np.arange(count)
        base = t.astype(np.int64)
        frac = (t - base).astype(np.float32)
        windows = sliding_window_view(buf, self.taps, axis=1)[
            :, base - self.half + 1]           # (2, count, taps)
        terms = windows @ self._coeffs_t       # (2, count, degree + 1)
        out = terms[..., self.degree]
        for m in range(self.degree - 1, -1, -1):
            out = out * frac + terms[..., m]
        # Carry on from where this block left off, in the coordinates
        # of the next buffer: it starts with the last `taps` frames.
        self._t = self._t + step * count - (length - self.taps)
        self._history = buf[:, -self.taps:].copy()
        return (np.ascontiguousarray(out[0], dtype=np.float32),
                np.ascontiguousarray(out[1], dtype=np.float32))


class RateMatcher:
    """A PI loop from frames waiting to a resampling ratio.

    ``update`` is given the frames on hand each time a block is about
    to be queued, and how many frames that block is.  The level is
    smoothed over ``smoothing_sec`` - it swings by a block and a card
    buffer between readings - and the ratio is set from how far the
    smoothed level is from ``target``: more frames out when there are
    too few waiting, fewer when there are too many.

    The proportional gain sets how quickly a step is taken back (time
    constant ``1 / (rate * 1e-6 * kp)``: 42 s with the defaults); the
    integral learns the steady mismatch, so that the level settles on
    the target rather than below it.  Both are clamped to
    ``max_ppm`` - two parts in a thousand at most is 3.5 cents of
    pitch, which nobody hears.
    """

    def __init__(self, rate: float, target_frames: float,
                 kp_ppm_per_frame: float = 0.5,
                 integral_sec: float = 120.0,
                 smoothing_sec: float = 5.0,
                 max_ppm: float = 2000.0) -> None:
        self.rate = float(rate)
        self.target = float(target_frames)
        self.kp = float(kp_ppm_per_frame)
        self.integral_sec = float(integral_sec)
        self.smoothing_sec = float(smoothing_sec)
        self.max_ppm = float(max_ppm)
        self.level: float | None = None
        self.integral_ppm: float = 0.0
        self.ppm: float = 0.0

    def ratio(self) -> float:
        """Output frames per input frame, as last decided."""
        return 1.0 + self.ppm * 1e-6

    def restart_level(self) -> None:
        """Measure the level afresh; the integral is kept.

        For after a gap in the audio - a hold, a retune's refill - when
        the smoothed level describes a cushion that is no longer there.
        """
        self.level = None

    def update(self, on_hand: float, block_frames: int) -> float:
        """Take a reading and return the ratio for the block now going in.

        ``on_hand`` is read just before the block goes in: the low point
        of a level that rises by a block at every enqueue and falls by
        the same between them.  Half a block is added so that what is
        held at the target is the mean the card sees, not the trough.
        """
        dt = max(block_frames, 0) / self.rate
        reading = float(on_hand) + 0.5 * max(block_frames, 0)
        if self.level is None:
            self.level = reading
        else:
            alpha = min(1.0, dt / self.smoothing_sec)
            self.level += alpha * (reading - self.level)
        error = self.level - self.target          # frames too many
        proportional = -self.kp * error
        self.integral_ppm += proportional * dt / self.integral_sec
        self.integral_ppm = float(np.clip(self.integral_ppm,
                                          -self.max_ppm, self.max_ppm))
        self.ppm = float(np.clip(proportional + self.integral_ppm,
                                 -self.max_ppm, self.max_ppm))
        return self.ratio()
