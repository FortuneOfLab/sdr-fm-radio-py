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
"""What the band looks like around the station being received.

One picture of the RF spectrum, made from the same IQ the demodulator
gets, so it costs nothing to capture and shows exactly what the receiver
is working with: the station in the middle, its neighbours either side,
and whatever else is in the sample rate.

The cost is why this is not done every block.  A block arrives every
16 ms and a display cannot use more than about ten pictures a second, so
one block in six is turned into a :class:`SpectrumFrame` and the rest go
straight to the demodulator untouched.  What that one block costs is
measured in ``tests/test_spectrum_benchmark.py``.

The averaging is Welch's: the block is cut into segments, each is
windowed and transformed, and the power spectra are averaged.  One
transform of the whole block would have far finer resolution than a few
hundred pixels can show and a noise floor that jumps around by several
dB between frames; averaging eight short ones costs a fraction as much
and gives a floor that sits still.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: How often a picture is worth making.  Ten a second is past the point
#: where a waterfall looks smooth, and it is a sixth of the block rate.
DEFAULT_SPECTRUM_INTERVAL_SEC: float = 0.1

#: Samples per transform.  At 1.024 Msps this is 1 kHz per bin before
#: the display groups them, which resolves a station's skirts without
#: paying for resolution nothing can show.
SEGMENT_SAMPLES: int = 1024

#: How many segments go into one picture.  Eight takes the noise floor
#: down by about five dB, which is enough for it to sit still, and the
#: arithmetic is a fifth of what using every segment in the block would
#: cost - which matters, because this runs on the thread with the
#: sixteen millisecond budget.
SEGMENTS_PER_FRAME: int = 8

#: How many points the display gets.  More than a plot is wide, few
#: enough that the whole frame is a few kilobytes.
DISPLAY_BINS: int = 512

#: Where the scale bottoms out.  Nothing useful lives below this, and a
#: floor keeps a silent band from stretching the plot to -inf.
FLOOR_DBFS: float = -120.0


@dataclasses.dataclass(frozen=True)
class SpectrumFrame:
    """One picture of the band, ready to draw.

    Frozen and self-contained, like :class:`~fm_radio.telemetry.
    StatusSnapshot`: the processing thread hands it over by rebinding a
    single attribute, and whoever draws it holds it for as long as it
    takes without anything changing underneath.

    Attributes:
        center_hz: The frequency the middle of the picture is on.
        span_hz: How wide the picture is - the sample rate, because
            that is what the receiver can see.
        dbfs: Power in each bin, lowest frequency first, full scale at
            0 dB and never below ``FLOOR_DBFS``.
        timestamp: ``time.perf_counter()`` when the block was processed.
    """

    center_hz: float
    span_hz: float
    dbfs: tuple[float, ...]
    timestamp: float

    @property
    def bin_hz(self) -> float:
        """How much of the band each point covers."""
        return self.span_hz / len(self.dbfs) if self.dbfs else 0.0

    def frequencies_hz(self) -> np.ndarray:
        """The frequency at the middle of each point, for an x axis."""
        count = len(self.dbfs)
        if count == 0:
            return np.zeros(0, dtype=np.float64)
        edges = np.arange(count, dtype=np.float64)
        return (self.center_hz - self.span_hz / 2.0
                + (edges + 0.5) * self.span_hz / count)


class SpectrumMaker:
    """Turns IQ blocks into pictures, reusing everything it can.

    The window and the bin edges depend only on the block size and the
    sample rate, so they are worked out once and kept.  Being called
    with a different block size is normal - the last block of a session
    can be short - and simply makes a new window.

    Args:
        sample_rate_hz: What the IQ was captured at; the width of the
            picture.
        display_bins: How many points to hand the display.
    """

    def __init__(self, sample_rate_hz: float,
                 display_bins: int = DISPLAY_BINS) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.display_bins = int(display_bins)
        self._window: np.ndarray | None = None
        self._window_samples: int = 0
        self._grouping: np.ndarray | None = None
        self._grouped_from: int = 0
        self._segments: np.ndarray | None = None
        self._segments_from: int = 0

    def frame(self, iq_samples: np.ndarray, center_hz: float,
              timestamp: float) -> SpectrumFrame:
        """One picture from one block.

        Args:
            iq_samples: The block, complex.
            center_hz: What the receiver is tuned to.
            timestamp: ``time.perf_counter()`` for the block.

        Returns:
            The frame, with ``display_bins`` points - or fewer, if the
            block was too short to fill them.
        """
        power = self._average_power(iq_samples)
        if power.size == 0:
            return SpectrumFrame(float(center_hz), self.sample_rate_hz, (),
                                 float(timestamp))
        grouped = self._group(power)
        dbfs = 10.0 * np.log10(np.maximum(grouped, 1e-20))
        np.maximum(dbfs, FLOOR_DBFS, out=dbfs)
        return SpectrumFrame(float(center_hz), self.sample_rate_hz,
                             tuple(float(v) for v in dbfs), float(timestamp))

    # ------------------------------------------------------------------
    # The arithmetic
    # ------------------------------------------------------------------

    def _average_power(self, iq_samples: np.ndarray) -> np.ndarray:
        """Welch's average, lowest frequency first.

        Eight segments, spread across the whole block rather than
        taken from the front of it, so the picture is of the block and
        not of its first few milliseconds.

        All of them go through one call: eight transforms asked for
        one at a time cost several times what one transform of an
        eight-row array does.
        """
        samples = np.ascontiguousarray(iq_samples)
        if samples.size < SEGMENT_SAMPLES:
            return np.zeros(0, dtype=np.float64)
        window = self._the_window(SEGMENT_SAMPLES)
        segments = samples[self._the_segments(samples.size)]
        spectra = np.fft.fftshift(
            np.fft.fft(segments * window, axis=-1), axes=-1)
        power = np.mean(spectra.real ** 2 + spectra.imag ** 2, axis=0)
        # Normalised so a full-scale tone reads 0 dB whatever the
        # segment length or the window happen to be.
        return power / (float(np.sum(window)) ** 2)

    def _the_segments(self, points: int) -> np.ndarray:
        """Where the segments start, as one index array to slice with.

        Worked out once per block size: the same block arrives over and
        over, and building this is more arithmetic than using it.
        """
        if self._segments is None or self._segments_from != points:
            room = points - SEGMENT_SAMPLES
            wanted = min(SEGMENTS_PER_FRAME, room // (SEGMENT_SAMPLES // 2) + 1)
            starts = np.linspace(0, room, wanted).astype(np.intp)
            self._segments = (starts[:, None]
                              + np.arange(SEGMENT_SAMPLES, dtype=np.intp))
            self._segments_from = points
        return self._segments

    def _the_window(self, samples: int) -> np.ndarray:
        if self._window is None or self._window_samples != samples:
            self._window = np.hanning(samples).astype(np.float64)
            self._window_samples = samples
        return self._window

    def _group(self, power: np.ndarray) -> np.ndarray:
        """Down to ``display_bins`` points, keeping the peaks.

        The peak rather than the mean: a pilot tone or a narrow carrier
        is one bin wide out of hundreds, and averaging it with its
        neighbours is how it disappears from the picture.
        """
        if power.size <= self.display_bins:
            return power
        edges = self._the_grouping(power.size)
        return np.maximum.reduceat(power, edges)

    def _the_grouping(self, points: int) -> np.ndarray:
        if self._grouping is None or self._grouped_from != points:
            self._grouping = np.linspace(
                0, points, self.display_bins, endpoint=False).astype(np.intp)
            self._grouped_from = points
        return self._grouping
