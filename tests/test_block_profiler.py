"""What the processing loop says about its own timing.

The numbers these choices were made from, measured on real hardware
over 60 s (the summary line of that run): 3740 blocks, average
10.24 ms of the 16 ms budget, 336 of them over the 20 ms threshold,
and a queue that never went past 2 of 80.  A warning per slow block
was 432 lines in 82 seconds of a receiver that was keeping up
perfectly, and the one event worth reading - a queue at 47 of 80,
recovering - was among them unread.
"""

from __future__ import annotations

import logging

import pytest

from fm_radio.controller import (
    _BACKLOG_LOG_INTERVAL_SEC,
    _BlockProfiler,
    _QUEUE_BACKLOG_SHARE,
    _SLOW_BLOCK_LOG_INTERVAL_SEC,
    _SLOW_BLOCK_THRESHOLD_SEC,
)

CAPACITY = 80
SLOW = _SLOW_BLOCK_THRESHOLD_SEC + 0.005
QUICK = 0.007          # the measured median, well under the budget


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def on(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def profiling(caplog):
    """A profiler on a hand-driven clock, and the lines it writes."""
    clock = Clock()
    logger = logging.getLogger("fm_receiver.test_profiler")
    profiler = _BlockProfiler(logger, CAPACITY, clock=clock)
    caplog.set_level(logging.INFO, logger=logger.name)
    return profiler, clock, caplog


def lines(caplog, what: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if what in r.getMessage()]


# ----------------------------------------------------------------------
# Slow blocks: a sample, and a count that loses nothing
# ----------------------------------------------------------------------

def test_a_run_of_slow_blocks_is_one_line_carrying_the_count(profiling):
    profiler, clock, caplog = profiling

    for _ in range(20):
        profiler.record(SLOW, q_depth=0)
        clock.on(0.016)

    said = lines(caplog, "SLOW BLOCK")
    assert len(said) == 1, "a line each is what buried the real one"
    assert "(the worst of 1 since the last of these)" in said[0]

    clock.on(_SLOW_BLOCK_LOG_INTERVAL_SEC)
    profiler.record(SLOW, q_depth=0)

    said = lines(caplog, "SLOW BLOCK")
    assert len(said) == 2
    assert "(the worst of 20 since the last of these)" in said[1], (
        "the nineteen that were not printed have to be in the next line")
    assert "total_slow=21" in said[1]


def test_the_line_is_about_the_worst_block_it_stands_for(profiling):
    """Not whichever block happened to be due when the line was.

    A 400 ms stall in one stage is exactly what a sample must not
    throw away: the summary would say max=400ms and nothing about
    where it went.
    """
    profiler, clock, caplog = profiling
    ordinary = (0.0, 0.005, 0.016, 0.0, 0.0)
    the_stall = (0.0, 0.001, 0.002, 0.400, 0.0)     # all of it in enqueue

    profiler.record(SLOW, q_depth=0, stage_times=ordinary)   # writes a line
    clock.on(0.5)
    profiler.record(0.404, q_depth=4, stage_times=the_stall)  # held back
    clock.on(0.5)
    for _ in range(10):
        profiler.record(SLOW, q_depth=0, stage_times=ordinary)
        clock.on(0.1)

    clock.on(_SLOW_BLOCK_LOG_INTERVAL_SEC)
    profiler.record(SLOW, q_depth=0, stage_times=ordinary)

    said = lines(caplog, "SLOW BLOCK")
    assert len(said) == 2
    assert "dt=404.0ms" in said[1], "the worst one was thrown away"
    assert "enqueue:400.0" in said[1], "and with it, where the time went"
    assert "q_depth=4/80" in said[1], "and what the queue was doing"
    assert "session_t=0.5s" in said[1], "and when it happened"


def test_a_stall_nothing_follows_is_still_reported(profiling):
    """The line is written when a slow block is due, or by the summary.

    A one-off stall on a receiver that then behaves would otherwise
    wait for the next slow block, which may never come.
    """
    profiler, clock, caplog = profiling
    the_stall = (0.0, 0.001, 0.002, 0.400, 0.0)

    profiler.record(0.404, q_depth=4, stage_times=the_stall)
    assert len(lines(caplog, "SLOW BLOCK")) == 1, (
        "the first one is not held back; hold it back to test the rest")

    clock.on(0.1)
    profiler.record(0.404, q_depth=4, stage_times=the_stall)
    assert len(lines(caplog, "SLOW BLOCK")) == 1, "held back, as it should be"

    clock.on(61.0)
    profiler.record(QUICK, q_depth=0)          # nothing slow about it

    said = lines(caplog, "SLOW BLOCK")
    assert len(said) == 2, "the summary let the stall out"
    assert "enqueue:400.0" in said[1]
    assert lines(caplog, "BlockProfile: t=")


def test_the_summary_counts_every_slow_block(profiling):
    """The lines are a sample; the counts are not."""
    profiler, clock, caplog = profiling

    for _ in range(50):
        profiler.record(SLOW, q_depth=0)
        clock.on(0.016)
    clock.on(61.0)
    profiler.record(QUICK, q_depth=0)

    assert profiler.slow_blocks == 50
    summary = lines(caplog, "BlockProfile: t=")
    assert len(summary) == 1
    assert "slow_in_window=50" in summary[0]
    assert "blocks=51" in summary[0]


def test_a_block_inside_the_budget_says_nothing(profiling):
    profiler, clock, caplog = profiling

    for _ in range(100):
        profiler.record(QUICK, q_depth=1)
        clock.on(0.016)

    assert lines(caplog, "SLOW BLOCK") == []
    assert lines(caplog, "FALLING BEHIND") == []


# ----------------------------------------------------------------------
# The queue backing up: the line worth having
# ----------------------------------------------------------------------

def test_a_queue_backing_up_is_said_at_once(profiling):
    """Even while the slow-block line is holding its tongue.

    This is the receiver failing to keep up - the samples past the
    end of the queue are dropped - and it has to be visible the
    moment it starts, not in a summary a minute later.
    """
    profiler, clock, caplog = profiling

    # Use up the slow-block line first, so the backlog line is the
    # only one that can still be written.
    profiler.record(SLOW, q_depth=0)
    clock.on(0.016)
    assert len(lines(caplog, "SLOW BLOCK")) == 1

    deep = int(CAPACITY * _QUEUE_BACKLOG_SHARE)
    profiler.record(SLOW, q_depth=deep)

    behind = lines(caplog, "FALLING BEHIND")
    assert len(behind) == 1
    assert "q_depth=%d/%d" % (deep, CAPACITY) in behind[0]
    assert "0.3s deep" in behind[0], behind[0]     # 20 blocks of 16 ms
    assert len(lines(caplog, "SLOW BLOCK")) == 1, "still rate limited"


def test_a_backlog_that_lasts_is_not_a_line_a_block(profiling):
    profiler, clock, caplog = profiling
    deep = CAPACITY // 2

    for _ in range(60):                      # about a second of blocks
        profiler.record(QUICK, q_depth=deep)
        clock.on(0.016)

    behind = lines(caplog, "FALLING BEHIND")
    assert 1 <= len(behind) <= 2, (
        "a second of backlog should be a line or two, not sixty")

    clock.on(_BACKLOG_LOG_INTERVAL_SEC)
    profiler.record(QUICK, q_depth=deep)
    assert len(lines(caplog, "FALLING BEHIND")) == len(behind) + 1


def test_a_queue_doing_its_job_is_not_a_backlog(profiling):
    """Two or three deep is the jitter the queue is there to absorb."""
    profiler, clock, caplog = profiling

    for depth in (1, 2, 3, 4, 5):
        profiler.record(QUICK, q_depth=depth)
        clock.on(0.016)

    assert lines(caplog, "FALLING BEHIND") == []
    assert profiler.window_max_ms == pytest.approx(QUICK * 1000.0)


def test_the_queue_is_said_in_this_mode_s_seconds(caplog):
    """Light mode's blocks are 65.5 ms, four times standard mode's.

    The same twenty blocks are 0.32 s of one and 1.31 s of the
    other, and a line that says 0.3 either way is telling a light
    mode listener their queue is a quarter of what it is.
    """
    clock = Clock()
    logger = logging.getLogger("fm_receiver.test_profiler_light")
    caplog.set_level(logging.INFO, logger=logger.name)
    profiler = _BlockProfiler(logger, CAPACITY,
                              block_interval_sec=16384 / 250000.0,
                              clock=clock)

    profiler.record(QUICK, q_depth=int(CAPACITY * _QUEUE_BACKLOG_SHARE))

    behind = lines(caplog, "FALLING BEHIND")
    assert len(behind) == 1
    assert "1.3s deep" in behind[0], behind[0]
