"""The picture of the band: what is in it, and what it costs.

A spectrum is easy to draw and hard to check by eye, so these put
signals of a known size at a known place and ask where they came out.
The cost matters as much as the shape: this runs on the thread with the
sixteen millisecond budget.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fm_radio.spectrum import (
    DISPLAY_BINS, FLOOR_DBFS, SEGMENT_SAMPLES, SEGMENTS_PER_FRAME,
    SpectrumFrame, SpectrumMaker,
)

RATE = 1.024e6
BLOCK = 16384
CENTRE = 80.0e6


def tone(offset_hz: float, amplitude: float = 1.0, samples: int = BLOCK,
         rate: float = RATE) -> np.ndarray:
    """A complex tone *offset_hz* from the middle of the picture."""
    t = np.arange(samples, dtype=np.float64) / rate
    return (amplitude * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def noise(samples: int = BLOCK, amplitude: float = 1e-3,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * (rng.standard_normal(samples)
                         + 1j * rng.standard_normal(samples))
            ).astype(np.complex64)


def where(frame: SpectrumFrame, index: int) -> float:
    """How far that point is from the middle, in Hz."""
    return float(frame.frequencies_hz()[index] - frame.center_hz)


def peak_of(frame: SpectrumFrame) -> int:
    return int(np.argmax(np.asarray(frame.dbfs)))


# ----------------------------------------------------------------------
# Where things are
# ----------------------------------------------------------------------

def test_the_picture_covers_the_sample_rate():
    """What the receiver can see is what it was sampled at."""
    frame = SpectrumMaker(RATE).frame(noise(), CENTRE, 1.0)

    assert frame.span_hz == RATE
    assert frame.center_hz == CENTRE
    assert len(frame.dbfs) == DISPLAY_BINS
    assert frame.bin_hz == pytest.approx(RATE / DISPLAY_BINS)


def test_a_tone_comes_out_where_it_went_in():
    """The one thing a spectrum has to get right."""
    frame = SpectrumMaker(RATE).frame(tone(100e3), CENTRE, 1.0)

    assert where(frame, peak_of(frame)) == pytest.approx(100e3,
                                                         abs=2 * frame.bin_hz)


def test_a_tone_below_the_centre_comes_out_below_it():
    """Which half of the picture is which."""
    frame = SpectrumMaker(RATE).frame(tone(-200e3), CENTRE, 1.0)

    assert where(frame, peak_of(frame)) == pytest.approx(-200e3,
                                                         abs=2 * frame.bin_hz)


def test_the_frequencies_run_from_one_edge_to_the_other():
    frame = SpectrumMaker(RATE).frame(noise(), CENTRE, 1.0)
    freqs = frame.frequencies_hz()

    assert freqs[0] == pytest.approx(CENTRE - RATE / 2, abs=frame.bin_hz)
    assert freqs[-1] == pytest.approx(CENTRE + RATE / 2, abs=frame.bin_hz)
    assert np.all(np.diff(freqs) > 0), "the axis has to be in order"


# ----------------------------------------------------------------------
# How big they are
# ----------------------------------------------------------------------

def test_a_full_scale_tone_reads_near_zero_dbfs():
    """The scale is absolute, so a level on screen means something."""
    frame = SpectrumMaker(RATE).frame(tone(50e3), CENTRE, 1.0)

    assert max(frame.dbfs) == pytest.approx(0.0, abs=1.5)


def test_a_tone_forty_db_down_reads_forty_db_down():
    """And the distance between two of them means something too."""
    maker = SpectrumMaker(RATE)
    loud = max(maker.frame(tone(50e3), CENTRE, 1.0).dbfs)
    quiet = max(maker.frame(tone(50e3, 0.01), CENTRE, 1.0).dbfs)

    assert loud - quiet == pytest.approx(40.0, abs=1.0)


def test_the_scale_does_not_depend_on_the_sample_rate():
    """A tone is a tone whichever mode the receiver is in."""
    full = SpectrumMaker(RATE).frame(tone(50e3), CENTRE, 1.0)
    light = SpectrumMaker(0.25e6).frame(
        tone(50e3, rate=0.25e6), CENTRE, 1.0)

    assert max(full.dbfs) == pytest.approx(max(light.dbfs), abs=1.0)


def test_nothing_falls_below_the_floor():
    """An empty band would otherwise stretch the scale to -inf."""
    frame = SpectrumMaker(RATE).frame(
        np.zeros(BLOCK, dtype=np.complex64), CENTRE, 1.0)

    assert min(frame.dbfs) == FLOOR_DBFS
    assert max(frame.dbfs) == FLOOR_DBFS


def test_a_narrow_tone_survives_the_grouping():
    """Hundreds of bins become hundreds of points, keeping the peaks.

    A pilot is one bin wide out of a thousand.  Averaging it with its
    neighbours on the way to the display is how it disappears, so the
    grouping keeps the loudest of each group instead.
    """
    quiet = noise(amplitude=1e-3)
    frame = SpectrumMaker(RATE).frame(quiet + tone(19e3, 0.05), CENTRE, 1.0)

    at_the_pilot = max(
        frame.dbfs[i] for i in range(len(frame.dbfs))
        if abs(where(frame, i) - 19e3) < 3 * frame.bin_hz)
    floor = float(np.median(frame.dbfs))

    assert at_the_pilot - floor > 20.0, "the tone was averaged away"


# ----------------------------------------------------------------------
# What it does with an awkward block
# ----------------------------------------------------------------------

def test_a_block_too_short_to_transform_makes_an_empty_picture():
    """Rather than raising at the caller, which is the realtime path."""
    frame = SpectrumMaker(RATE).frame(
        np.zeros(SEGMENT_SAMPLES - 1, dtype=np.complex64), CENTRE, 1.0)

    assert frame.dbfs == ()
    assert frame.bin_hz == 0.0
    assert frame.frequencies_hz().size == 0


def test_a_block_of_exactly_one_segment_is_enough():
    frame = SpectrumMaker(RATE).frame(
        tone(100e3, samples=SEGMENT_SAMPLES), CENTRE, 1.0)

    assert len(frame.dbfs) == DISPLAY_BINS
    assert where(frame, peak_of(frame)) == pytest.approx(
        100e3, abs=2 * frame.bin_hz)


def test_a_block_of_a_different_size_is_taken_as_it_comes():
    """The last block of a session can be short."""
    maker = SpectrumMaker(RATE)
    maker.frame(tone(100e3), CENTRE, 1.0)

    frame = maker.frame(tone(100e3, samples=BLOCK // 2), CENTRE, 2.0)

    assert len(frame.dbfs) == DISPLAY_BINS
    assert where(frame, peak_of(frame)) == pytest.approx(
        100e3, abs=2 * frame.bin_hz)


def test_the_segments_are_spread_across_the_block():
    """Not taken from the front of it.

    A block is sixteen milliseconds; a picture of its first four would
    miss anything that only happened later.
    """
    maker = SpectrumMaker(RATE)
    quiet = np.zeros(BLOCK, dtype=np.complex64)
    late = quiet.copy()
    late[-SEGMENT_SAMPLES:] = tone(100e3, samples=SEGMENT_SAMPLES)

    frame = maker.frame(late, CENTRE, 1.0)

    assert max(frame.dbfs) > FLOOR_DBFS + 40, (
        "the end of the block was never looked at")


def test_the_number_of_segments_is_what_it_says():
    """The cost is this number times one transform, and no more."""
    maker = SpectrumMaker(RATE)
    maker.frame(noise(), CENTRE, 1.0)

    assert maker._segments.shape == (SEGMENTS_PER_FRAME, SEGMENT_SAMPLES)


# ----------------------------------------------------------------------
# What it costs
# ----------------------------------------------------------------------

def test_a_frame_costs_a_fraction_of_a_block():
    """It is made on the thread with the sixteen millisecond budget.

    Generous on purpose - a shared runner is not a quiet machine - but
    a frame that has grown to milliseconds is one to look at.
    """
    maker = SpectrumMaker(RATE)
    block = noise() + tone(100e3, 0.5)
    maker.frame(block, CENTRE, 1.0)          # warm the plans up

    best = min(_one_run(maker, block) for _ in range(3))

    assert best < 4.0, f"a frame took {best:.2f} ms"


def _one_run(maker: SpectrumMaker, block: np.ndarray, times: int = 20
             ) -> float:
    started = time.perf_counter()
    for _ in range(times):
        maker.frame(block, CENTRE, 1.0)
    return (time.perf_counter() - started) / times * 1e3


def test_the_work_is_reused_between_frames():
    """The window and the segment offsets are the same every block."""
    maker = SpectrumMaker(RATE)
    maker.frame(noise(), CENTRE, 1.0)
    window, segments, grouping = (maker._window, maker._segments,
                                  maker._grouping)

    maker.frame(noise(seed=1), CENTRE, 2.0)

    assert maker._window is window
    assert maker._segments is segments
    assert maker._grouping is grouping
