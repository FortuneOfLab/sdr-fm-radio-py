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
"""The band, drawn: a spectrum above a waterfall.

Both are of the same :class:`~fm_radio.spectrum.SpectrumFrame`, which
the processing thread makes ten times a second out of the IQ the
demodulator is already working on.  This module only draws; it computes
nothing and asks the receiver for nothing but the latest frame.

pyqtgraph is optional.  Without it the window comes up with everything
else and a line where this would have been, because a spectrum is worth
having and not worth refusing to start over.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

try:                                            # pragma: no cover - optional
    import pyqtgraph
except Exception as _e:                         # pragma: no cover - optional
    pyqtgraph = None
    _why_not = _e
else:                                           # pragma: no cover - optional
    _why_not = None

#: How many frames the waterfall remembers.  At ten a second this is
#: about twelve seconds of history, which is long enough to see a
#: station fade and short enough to fit above the controls.
HISTORY_FRAMES: int = 120

#: The range the colours and the plot cover.  Fixed rather than fitted
#: to each frame: a waterfall whose colours mean something different
#: from one second to the next cannot be read across time, which is the
#: only thing it is for.  Signals on this receiver sit around -20 dBFS
#: and the floor around -75, so this holds both with room either side.
TOP_DBFS: float = 0.0
BOTTOM_DBFS: float = -100.0

#: What the waterfall is coloured with.  Greyscale is the default and
#: it wastes the eye's ability to tell colours apart far better than it
#: tells shades of grey apart - on grey, a station and its noise floor
#: look like two similar greys.
COLOUR_MAP: str = "viridis"

#: The part of the scale the colours are spread over.  Narrower than
#: the plot's axis on purpose: measured on this receiver, a station
#: sits around -20 dBFS and the noise floor around -75, and spreading
#: the colours over the full hundred dB puts all of that in the bottom
#: half of one colour.  The plot above keeps the absolute scale; this
#: is the same data stretched to be legible.
COLOUR_TOP_DBFS: float = -20.0
COLOUR_BOTTOM_DBFS: float = -90.0


def is_available() -> bool:
    """True when there is something to draw a spectrum with."""
    return pyqtgraph is not None


def why_not() -> str:
    """Why there is no spectrum, for the label that takes its place."""
    return f"install pyqtgraph for the spectrum ({_why_not})"


class BandView(QWidget):
    """A spectrum and a waterfall of whatever the receiver can see.

    Args:
        parent: The usual.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not is_available():
            layout.addWidget(QLabel(why_not(), self))
            self._curve = None
            self._waterfall = None
            self._history = None
            return

        self._plot = pyqtgraph.PlotWidget(parent=self)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setLabel("left", "dBFS")
        self._plot.setYRange(BOTTOM_DBFS, TOP_DBFS)
        self._plot.setMaximumHeight(140)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = self._plot.plot([], [])
        # Where the receiver is actually listening, which is not
        # otherwise obvious: an FM station is wider than the picture in
        # light mode, so the middle is not marked by the signal itself.
        self._tuned_to = pyqtgraph.InfiniteLine(
            angle=90, movable=False,
            pen=pyqtgraph.mkPen("#ff9800", width=1,
                                style=pyqtgraph.QtCore.Qt.DashLine))
        self._plot.addItem(self._tuned_to)
        layout.addWidget(self._plot)

        self._fall = pyqtgraph.PlotWidget(parent=self)
        self._fall.setMouseEnabled(x=False, y=False)
        self._fall.setMenuEnabled(False)
        self._fall.hideButtons()
        self._fall.hideAxis("left")
        self._fall.setLabel("bottom", "MHz")
        self._fall.setMaximumHeight(160)
        self._waterfall = pyqtgraph.ImageItem()
        self._waterfall.setLevels((COLOUR_BOTTOM_DBFS, COLOUR_TOP_DBFS))
        self._waterfall.setColorMap(pyqtgraph.colormap.get(COLOUR_MAP))
        self._fall.addItem(self._waterfall)
        layout.addWidget(self._fall)

        # Newest at the top, so the history falls away below it, and
        # filled with the floor rather than zeros: an empty waterfall
        # should look like an empty band, not a full-scale one.
        self._history: np.ndarray | None = None
        self._bins: int = 0
        self._span_hz: float = 0.0
        self._center_hz: float = 0.0

    # ------------------------------------------------------------------

    def show_the_frame(self, frame) -> None:
        """Draw *frame*, or clear the picture when there is not one.

        Called from the window's refresh, which runs on the Qt thread
        at its own rate: frames it misses are simply not drawn, and
        frames it asks for twice are drawn twice.
        """
        if self._curve is None:
            return                              # no pyqtgraph, nothing to do
        if frame is None or not frame.dbfs:
            self._curve.setData([], [])
            return
        dbfs = np.asarray(frame.dbfs, dtype=np.float32)
        self._curve.setData(frame.frequencies_hz() / 1e6, dbfs)
        self._tuned_to.setPos(frame.center_hz / 1e6)
        self._remember(frame, dbfs)

    def _remember(self, frame, dbfs: np.ndarray) -> None:
        """Push one row into the waterfall and redraw it."""
        if (self._history is None or self._bins != dbfs.size
                or self._span_hz != frame.span_hz
                or self._center_hz != frame.center_hz):
            # A different band, or a different shape of picture: what
            # is on screen is of somewhere else and does not belong
            # above what comes next.
            self._history = np.full((HISTORY_FRAMES, dbfs.size),
                                    BOTTOM_DBFS, dtype=np.float32)
            self._bins = dbfs.size
            self._span_hz = frame.span_hz
            self._center_hz = frame.center_hz
            self._place_the_waterfall(frame)
        self._history[:-1] = self._history[1:]
        self._history[-1] = dbfs
        # ImageItem takes (x, y); the history is (time, frequency), so
        # it goes in transposed and time runs left to right internally.
        self._waterfall.setImage(self._history.T, autoLevels=False,
                                 levels=(COLOUR_BOTTOM_DBFS,
                                         COLOUR_TOP_DBFS))

    def _place_the_waterfall(self, frame) -> None:
        """Put the image where its frequencies say it belongs."""
        left = (frame.center_hz - frame.span_hz / 2.0) / 1e6
        self._waterfall.setRect(left, 0.0, frame.span_hz / 1e6,
                                float(HISTORY_FRAMES))
        self._fall.setXRange(left, left + frame.span_hz / 1e6, padding=0)
        # And to the height of the history, or the image sits in a
        # corner of a plot that has auto-ranged to something else.
        self._fall.setYRange(0.0, float(HISTORY_FRAMES), padding=0)
        # Installing an image is enough to make a view range itself
        # again, and it does not know that the rest of the history is
        # coming.  Both ranges are ours now.
        self._fall.getPlotItem().getViewBox().disableAutoRange()
        self._plot.setXRange(left, left + frame.span_hz / 1e6, padding=0)
