"""The spectrum and waterfall widget: what it draws, and when it does not.

Driven with frames built by hand rather than by a receiver, because
what is being checked is the drawing - that the curve gets the
frequencies the frame says, that the waterfall scrolls, and that a
frame from a different band does not appear above one from this band.

Qt runs offscreen (see the ``qt_app`` fixture), so these need no
display.
"""

from __future__ import annotations

import itertools

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


#: Ten a second, as the receiver makes them.  Frames made one after
#: another carry different times, and the view uses that to tell a
#: new one from the same one shown again - so a helper that stamped
#: them all alike would be testing something that never happens.
_a_tenth_of_a_second = itertools.count(1.0, 0.1)


def a_frame(center_hz: float = 80.0e6, span_hz: float = 1.024e6,
            bins: int = 8, peak_at: int | None = None,
            timestamp: float | None = None) -> SpectrumFrame:
    """A small frame, with one point louder than the rest if asked.

    Each one is later than the last unless a time is given.
    """
    dbfs = [-90.0] * bins
    if peak_at is not None:
        dbfs[peak_at] = -10.0
    when = next(_a_tenth_of_a_second) if timestamp is None else timestamp
    return SpectrumFrame(center_hz, span_hz, tuple(dbfs), when)


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


@pytest.mark.parametrize("nothing", [
    None,
    SpectrumFrame(80.0e6, 1.024e6, (), 1.0),
])
def test_nothing_to_show_takes_the_whole_picture_down(view, nothing):
    """Not just the trace: the waterfall is of a station too.

    This is what a retune looks like from here - frames are held
    back until one of the new station has been made - and the
    frequency beside it has already moved.  A waterfall of the old
    one left up under the new name is the window saying the receiver
    is somewhere it is not.
    """
    view.show_the_frame(a_frame(center_hz=81.3e6, peak_at=1))
    assert view._history is not None, "nothing was drawn to begin with"

    view.show_the_frame(nothing)

    x, _y = view._curve.getData()
    assert x is None or len(x) == 0, "the trace is still up"
    assert view._history is None, "the history is still there"
    assert view._waterfall.image is None, "the waterfall is still up"
    assert not view._tuned_to.isVisible(), "it still marks a station"


def test_the_next_station_draws_from_an_empty_waterfall(view):
    """And what comes after the gap is only of the new one."""
    view.show_the_frame(a_frame(center_hz=81.3e6, bins=8, peak_at=1))
    view.show_the_frame(None)

    view.show_the_frame(a_frame(center_hz=80.0e6, bins=8, peak_at=6))

    assert view._history is not None
    assert view._center_hz == 80.0e6
    rows_with_anything_in = int(
        (view._history.max(axis=1) > BOTTOM_DBFS).sum())
    assert rows_with_anything_in == 1, (
        "%d rows carry a reading; only the new one should"
        % rows_with_anything_in)


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


def test_the_waterfall_covers_exactly_the_band_it_is_of(view):
    """All four edges, not just the left one.

    setRect works out its transform from the shape the image has when
    it is called, so calling it before the first image scaled one
    pixel up to the whole band and then let five hundred bins inherit
    that scale - the picture ended up four hundred thousand units
    wide, with a corner of it in the view and the rest outside.
    """
    frame = a_frame(center_hz=80.0e6, span_hz=1.024e6, bins=64, peak_at=1)

    view.show_the_frame(frame)

    placed = view._waterfall.mapRectToParent(view._waterfall.boundingRect())
    left = (frame.center_hz - frame.span_hz / 2) / 1e6
    assert placed.left() == pytest.approx(left, abs=1e-6)
    assert placed.width() == pytest.approx(frame.span_hz / 1e6, abs=1e-6)
    assert placed.top() == pytest.approx(0.0, abs=1e-6)
    assert placed.height() == pytest.approx(float(HISTORY_FRAMES), abs=1e-6)


def test_the_waterfall_is_still_placed_right_after_the_band_changes(view):
    """Retuning makes a new history, and a new transform with it."""
    view.show_the_frame(a_frame(center_hz=80.0e6, span_hz=1.024e6, bins=64))

    view.show_the_frame(a_frame(center_hz=81.3e6, span_hz=0.25e6, bins=32))

    placed = view._waterfall.mapRectToParent(view._waterfall.boundingRect())
    assert placed.left() == pytest.approx((81.3e6 - 0.125e6) / 1e6, abs=1e-6)
    assert placed.width() == pytest.approx(0.25, abs=1e-6)
    assert placed.height() == pytest.approx(float(HISTORY_FRAMES), abs=1e-6)


def where_on_screen(plot, mhz: float, row: float):
    """The pixel in *plot* that shows *mhz* at history row *row*."""
    from PySide6.QtCore import QPointF

    box = plot.getPlotItem().getViewBox()
    scene = box.mapViewToScene(QPointF(float(mhz), float(row)))
    return plot.mapFromScene(scene)


def test_a_loud_bin_is_drawn_in_a_different_colour(view, qt_app):
    """The point of a waterfall, asked of the pixels that show it.

    Of the two pixels that carry the claim, not of how many colours
    the widget has altogether: axis, labels and background make
    three on their own, so counting them passes with the waterfall
    hidden.  The picture was once placed so far outside the view
    that only its quietest corner showed, and every number involved
    was right.
    """
    bins, peak_at, rows = 32, 16, 20
    view.resize(400, 300)
    view.show()
    # Twenty rows, not one: a hundred and twenty of them share the
    # height of the picture, so a single row is about one pixel tall
    # and which pixel it lands on is a rounding question.
    for _ in range(rows):
        frame = a_frame(bins=bins, peak_at=peak_at)
        view.show_the_frame(frame)
    qt_app.processEvents()

    picture = view._fall.grab().toImage()
    megahertz = frame.frequencies_hz() / 1e6
    # The newest rows are the top ones; look at the middle of them.
    row = HISTORY_FRAMES - rows / 2.0
    loud = where_on_screen(view._fall, megahertz[peak_at], row)
    quiet = where_on_screen(view._fall, megahertz[peak_at // 2], row)

    for name, point in (("loud", loud), ("quiet", quiet)):
        assert picture.rect().contains(point), (
            "the %s bin is at %s, outside a %dx%d picture"
            % (name, point, picture.width(), picture.height()))

    loud_colour = picture.pixel(loud) & 0xFFFFFF
    quiet_colour = picture.pixel(quiet) & 0xFFFFFF
    background = picture.pixel(2, 2) & 0xFFFFFF

    assert loud_colour != quiet_colour, (
        "the loud bin and the floor are both #%06x" % loud_colour)
    assert loud_colour != background, "the loud bin is the background"
    assert quiet_colour != background, "the floor is the background"


def test_the_same_frame_shown_again_does_not_move_the_history(view):
    """The window refreshes at 50 ms; frames are made at 100.

    So half of what the window draws has been drawn already.  A row
    for each of those would run the history at twice the speed its
    scale claims, and twelve seconds of band would be six.
    """
    frame = a_frame(bins=8, peak_at=3)
    view.show_the_frame(frame)
    after_one = view._history.copy()

    for _ in range(20):
        view.show_the_frame(frame)

    assert np.array_equal(view._history, after_one), (
        "the picture moved on without a new frame")


def test_a_receiver_that_has_stopped_does_not_fill_the_picture(view):
    """The publisher holds its last frame and keeps handing it out.

    So a receiver that has stopped making them looks, from here,
    exactly like one making the same one over and over.  Twelve
    seconds later the whole picture would be that one reading - the
    waterfall saying the band looked like this all along, about a
    receiver that stopped after the first frame.
    """
    last = a_frame(bins=8, peak_at=3)

    for _ in range(HISTORY_FRAMES * 2):
        view.show_the_frame(last)

    rows_with_anything_in = int(
        (view._history.max(axis=1) > BOTTOM_DBFS).sum())
    assert rows_with_anything_in == 1, (
        "%d of %d rows carry the one reading there has been"
        % (rows_with_anything_in, HISTORY_FRAMES))


def test_a_later_frame_does_move_the_history(view):
    """The other half of it: a new frame is a new row."""
    view.show_the_frame(a_frame(bins=8, peak_at=3))
    after_one = view._history.copy()

    view.show_the_frame(a_frame(bins=8, peak_at=5))

    assert not np.array_equal(view._history, after_one), (
        "a new frame did not reach the picture")
    assert view._history[-1].argmax() == 5


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


# ----------------------------------------------------------------------
# The two halves are of the same band, and line up
# ----------------------------------------------------------------------

def plot_area(plot):
    """Where the drawing happens on screen, left and right."""
    box = plot.getPlotItem().getViewBox()
    placed = box.mapRectToScene(box.boundingRect())
    return placed.left(), placed.right()


def test_the_same_place_on_screen_is_the_same_frequency_in_both(view,
                                                                 qt_app):
    """The trace and the waterfall are read against each other.

    They are different widths the moment their left-hand scales are:
    a scale that sizes itself to "-50" is not the width of one that is
    empty, and then a peak in the trace sits over the wrong part of
    the waterfall.
    """
    view.resize(800, 420)
    view.show()
    view.show_the_frame(a_frame(bins=64, peak_at=32))
    qt_app.processEvents()

    assert plot_area(view._plot) == pytest.approx(plot_area(view._fall))


def test_the_two_halves_show_the_same_frequencies(view, qt_app):
    """Lining the areas up is only half of it; the ranges have to match."""
    view.resize(800, 420)
    view.show()
    view.show_the_frame(a_frame(center_hz=80.0e6, span_hz=1.024e6, bins=64))
    qt_app.processEvents()

    above = view._plot.getPlotItem().getViewBox().viewRange()[0]
    below = view._fall.getPlotItem().getViewBox().viewRange()[0]

    assert above == pytest.approx(below)


def test_they_stay_together_when_the_band_changes(view, qt_app):
    """Retuning moves both, because they are linked rather than set twice."""
    view.resize(800, 420)
    view.show()
    view.show_the_frame(a_frame(center_hz=80.0e6, span_hz=1.024e6, bins=64))
    qt_app.processEvents()

    view.show_the_frame(a_frame(center_hz=81.3e6, span_hz=0.25e6, bins=64))
    qt_app.processEvents()

    above = view._plot.getPlotItem().getViewBox().viewRange()[0]
    below = view._fall.getPlotItem().getViewBox().viewRange()[0]
    assert above == pytest.approx(below)
    assert above[0] == pytest.approx(81.3 - 0.125, abs=1e-6)
    assert plot_area(view._plot) == pytest.approx(plot_area(view._fall))
