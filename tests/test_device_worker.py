"""The thread that writes to the SDR, and what it does with the asking.

A write is 30-200 ms of USB and sometimes forever, so the point of all of
this is that the caller does not wait for one.  What has to survive that
is the order things happen in, and the answer eventually coming back.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from fm_radio.device_worker import GAIN, GAIN_MODE, TUNE, DeviceWorker


@pytest.fixture
def worker():
    instance = DeviceWorker(logging.getLogger("test.device_worker"))
    try:
        yield instance
    finally:
        instance.stop()


def blocking(event: threading.Event, done: list | None = None, tag=None,
             started: threading.Event | None = None):
    """A write that does not return until the test says so.

    ``started`` is set once the worker is actually inside it, which is
    what lets a test put the worker in the one state that matters here:
    busy, with everything else piling up behind it.
    """
    def run() -> None:
        if started is not None:
            started.set()
        event.wait(10)
        if done is not None:
            done.append(tag)
    return run


def a_worker_that_is_busy(worker, done: list | None = None):
    """Give the worker something slow to be in the middle of."""
    hold = threading.Event()
    running = threading.Event()
    worker.submit(TUNE, "the one in flight",
                  blocking(hold, done, "in flight", running))
    assert running.wait(5), "the worker never started the first write"
    return hold


# ----------------------------------------------------------------------
# Nobody waits
# ----------------------------------------------------------------------

def test_asking_for_a_write_comes_straight_back(worker):
    """The whole point: 60 ms of USB is not the caller's to pay."""
    release = threading.Event()
    try:
        started = time.monotonic()
        request = worker.submit(TUNE, "tune to 80.0 MHz", blocking(release))
        elapsed = time.monotonic() - started

        assert elapsed < 0.1, f"submitting took {elapsed * 1e3:.0f} ms"
        assert not request.finished
    finally:
        release.set()


def test_a_write_that_never_returns_does_not_hold_the_caller(worker):
    """The case that matters: a device that has stopped answering."""
    wedged = threading.Event()
    try:
        worker.submit(TUNE, "tune to 80.0 MHz", blocking(wedged))
        started = time.monotonic()
        for i in range(20):
            worker.submit(GAIN, f"gain {i}", lambda: None)
        elapsed = time.monotonic() - started

        assert elapsed < 0.5, f"twenty asks took {elapsed * 1e3:.0f} ms"
    finally:
        wedged.set()


def test_the_answer_comes_back_afterwards(worker):
    """Not waiting is only useful if the outcome can still be had."""
    request = worker.submit(TUNE, "tune to 80.0 MHz", lambda: None)

    assert request.wait(5), "the write never finished"
    assert not request.failed
    assert worker.latest is request


def test_a_write_that_fails_says_so(worker):
    def explode() -> None:
        raise OSError("LIBUSB_ERROR_TIMEOUT")

    request = worker.submit(TUNE, "tune to 80.0 MHz", explode)

    assert request.wait(5)
    assert request.failed
    assert "LIBUSB_ERROR_TIMEOUT" in str(request.error)
    assert worker.latest is request


# ----------------------------------------------------------------------
# Order, and what is worth doing at all
# ----------------------------------------------------------------------

def test_writes_happen_in_the_order_they_were_asked_for(worker):
    done: list[str] = []
    hold = a_worker_that_is_busy(worker, done)

    for kind, tag in ((GAIN_MODE, "mode"), (GAIN, "gain"), (TUNE, "tune")):
        worker.submit(kind, tag, lambda t=tag: done.append(t))
    last = worker.submit(GAIN_MODE, "mode again",
                         lambda: done.append("mode again"))
    hold.set()

    assert last.wait(5)
    # "mode" is gone, replaced by the later one of its kind; what is left
    # is in the order it was asked for.
    assert done == ["in flight", "gain", "tune", "mode again"], done


def test_a_burst_of_the_same_kind_collapses_to_the_last(worker):
    """A held button is one intent expressed twenty times."""
    done: list[str] = []
    hold = a_worker_that_is_busy(worker, done)

    asked = [worker.submit(TUNE, f"tune {i}", lambda i=i: done.append(f"t{i}"))
             for i in range(20)]
    hold.set()

    assert asked[-1].wait(5)
    assert done == ["in flight", "t19"], done
    assert all(r.superseded for r in asked[:-1])
    assert not asked[-1].superseded


def test_coalescing_does_not_reorder_the_kinds(worker):
    """"Manual gain" then "30 dB" still has to happen that way round."""
    done: list[str] = []
    hold = a_worker_that_is_busy(worker, done)

    worker.submit(GAIN, "gain 10", lambda: done.append("gain 10"))
    worker.submit(GAIN_MODE, "manual", lambda: done.append("manual"))
    worker.submit(GAIN, "gain 30", lambda: done.append("gain 30"))
    last = worker.submit(TUNE, "tune 81.3", lambda: done.append("tune 81.3"))
    hold.set()

    assert last.wait(5)
    # The superseded gain is gone; the mode still precedes the gain that
    # replaced it, because that is the order they were asked for in.
    assert done == ["in flight", "manual", "gain 30", "tune 81.3"], done


def test_a_superseded_request_is_finished_not_left_hanging(worker):
    """Somebody may be waiting on it, and it is never going to happen."""
    hold = a_worker_that_is_busy(worker)
    dropped = worker.submit(GAIN, "gain 10", lambda: None)
    worker.submit(GAIN, "gain 30", lambda: None)
    hold.set()

    assert dropped.wait(5), "a superseded request never finished"
    assert dropped.superseded
    assert not dropped.failed


# ----------------------------------------------------------------------
# Stopping
# ----------------------------------------------------------------------

def test_stopping_does_not_write_what_was_still_queued(worker):
    """Shutdown has decided the device is going; a tune does not help."""
    done: list[str] = []
    hold = a_worker_that_is_busy(worker, done)
    queued = [worker.submit(GAIN, f"gain {i}", lambda: done.append("gain"))
              for i in range(3)]

    stopped = threading.Thread(target=worker.stop, daemon=True)
    stopped.start()
    hold.set()
    stopped.join(timeout=10)

    assert not stopped.is_alive()
    assert done == ["in flight"], done
    for request in queued:
        assert request.finished, "left somebody waiting on a dropped write"


def test_stopping_twice_is_allowed(worker):
    worker.stop()
    worker.stop()
    assert not worker.running


def test_a_request_made_after_stopping_does_not_hang_the_caller(worker):
    worker.stop()

    request = worker.submit(TUNE, "tune to 80.0 MHz", lambda: None)

    assert request.wait(2), "a request nobody will carry out never finished"


def test_a_wedged_write_does_not_hold_the_stop_for_ever(worker):
    """The bounded join: the thread is a daemon and the process may go."""
    wedged = threading.Event()
    try:
        worker.submit(TUNE, "tune to 80.0 MHz", blocking(wedged))
        time.sleep(0.1)                 # let it get inside the write

        started = time.monotonic()
        worker.stop(timeout=0.3)
        elapsed = time.monotonic() - started

        assert elapsed < 3.0, f"stop took {elapsed:.1f} s"
    finally:
        wedged.set()
