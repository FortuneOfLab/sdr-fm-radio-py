"""The spectrum and waterfall widget: what it draws, and when it does not.

Driven with frames built by hand rather than by a receiver, because
what is being checked is the drawing - that the curve gets the
frequencies the frame says, that the waterfall scrolls, and that a
frame from a different band does not appear above one from this band.

Qt runs offscreen (see the ``qt_app`` fixture), so these need no
display.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets", reason="the GUI is optional")

from fm_radio.gui import band_view                            # noqa: E402
from fm_radio.gui.band_view import (                          # noqa: E402
    BOTTOM_DBFS, HISTORY_FRAMES, BandView,
)
from fm_radio.spectrum import SpectrumFrame                   # noqa: E402

pyqtgraph = pytest.importorskip("pyqtgraph",
                                reason="the spectrum is optional")


def a_frame(center_hz: float = 80.0e6, span_hz: float = 1.024e6,
            bins: int = 8, peak_at: int | None = None,
            timestamp: float = 1.0) -> SpectrumFrame:
    """A small frame, with one point louder than the rest if asked."""
    dbfs = [-90.0] * bins
    if peak_at is not None:
        dbfs[peak_at] = -10.0
    return SpectrumFrame(center_hz, span_hz, tuple(dbfs), timestamp)


@pytest.fixture
def view(qt_app):
    widget = BandView()
    try:
        yield widget
    finally:
        widget.deleteLater()


# ----------------------------------------------------------------------
# The curve
# ----------------------------------------------------------------------

def test_the_curve_gets_the_frequencies_the_frame_says(view):
    frame = a_frame(peak_at=5)

    view.show_the_frame(frame)

    x, y = view._curve.getData()
    assert len(x) == len(frame.dbfs)
    np.testing.assert_allclose(x, frame.frequencies_hz() / 1e6)
    np.testing.assert_allclose(y, np.asarray(frame.dbfs, dtype=np.float32))


def test_no_frame_leaves_nothing_drawn(view):
    view.show_the_frame(a_frame(peak_at=1))

    view.show_the_frame(None)

    x, _y = view._curve.getData()
    assert x is None or len(x) == 0


def test_an_empty_frame_leaves_nothing_drawn(view):
    """A block too short to transform gives one of these."""
    view.show_the_frame(SpectrumFrame(80.0e6, 1.024e6, (), 1.0))

    x, _y = view._curve.getData()
    assert x is None or len(x) == 0


# ----------------------------------------------------------------------
# The waterfall
# ----------------------------------------------------------------------

def test_the_waterfall_starts_at_the_floor(view):
    """An empty history should look like an empty band, not a full one."""
    view.show_the_frame(a_frame(peak_at=2))

    # Every row but the newest is still the floor.
    assert np.all(view._history[:-1] == BOTTOM_DBFS)


def test_the_newest_row_is_the_frame_just_given(view):
    frame = a_frame(peak_at=3)

    view.show_the_frame(frame)

    np.testing.assert_allclose(view._history[-1],
                               np.asarray(frame.dbfs, dtype=np.float32))


def test_the_history_scrolls(view):
    """The row that was newest moves down, and the new one takes its place."""
    first = a_frame(peak_at=1)
    second = a_frame(peak_at=6)

    view.show_the_frame(first)
    view.show_the_frame(second)

    np.testing.assert_allclose(view._history[-1],
                               np.asarray(second.dbfs, dtype=np.float32))
    np.testing.assert_allclose(view._history[-2],
                               np.asarray(first.dbfs, dtype=np.float32))


def test_the_history_is_bounded(view):
    for i in range(HISTORY_FRAMES * 2):
        view.show_the_frame(a_frame(peak_at=i % 8, timestamp=float(i)))

    assert view._history.shape == (HISTORY_FRAMES, 8)


def test_a_different_band_starts_a_new_waterfall(view):
    """What is on screen is of somewhere else.

    Retuning is exactly this, and leaving the old station's history
    above the new one's would read as one band with a seam in it.
    """
    view.show_the_frame(a_frame(center_hz=80.0e6, peak_at=1))
    view.show_the_frame(a_frame(center_hz=80.0e6, peak_at=1))

    view.show_the_frame(a_frame(center_hz=81.3e6, peak_at=6))

    assert np.all(view._history[:-1] == BOTTOM_DBFS), (
        "the old station's history was kept")


def test_a_different_span_starts_a_new_waterfall(view):
    """Light mode and full mode do not see the same width of band."""
    view.show_the_frame(a_frame(span_hz=1.024e6, peak_at=1))

    view.show_the_frame(a_frame(span_hz=0.25e6, peak_at=1))

    assert np.all(view._history[:-1] == BOTTOM_DBFS)


def test_a_different_number_of_points_starts_a_new_waterfall(view):
    view.show_the_frame(a_frame(bins=8, peak_at=1))

    view.show_the_frame(a_frame(bins=16, peak_at=1))

    assert view._history.shape == (HISTORY_FRAMES, 16)


def test_the_waterfall_sits_where_its_frequencies_say(view):
    frame = a_frame(center_hz=80.0e6, span_hz=1.024e6, peak_at=1)

    view.show_the_frame(frame)

    rect = view._waterfall.boundingRect()
    left = (frame.center_hz - frame.span_hz / 2) / 1e6
    assert view._waterfall.mapRectToParent(rect).left() == pytest.approx(
        left, abs=1e-6)


# ----------------------------------------------------------------------
# Without pyqtgraph
# ----------------------------------------------------------------------

def test_the_window_still_comes_up_without_pyqtgraph(qt_app, monkeypatch):
    """A spectrum is worth having and not worth refusing to start over."""
    monkeypatch.setattr(band_view, "pyqtgraph", None)

    widget = BandView()
    try:
        widget.show_the_frame(a_frame(peak_at=1))   # must not raise

        assert widget._curve is None
        assert band_view.is_available() is False
    finally:
        widget.deleteLater()
