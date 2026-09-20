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

from fm_radio.device_worker import (
    GAIN, GAIN_MODE, RECORDING, TUNE, DeviceWorker,
)


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


def test_a_held_button_never_fills_the_queue(worker):
    """Coalescing on the way in is what bounds it.

    Coalescing only when the worker came back for more meant a burst
    could fill the queue while it was busy, and the request the user
    ended on - the only one worth making - was the one refused.
    """
    hold = a_worker_that_is_busy(worker)
    try:
        asked = [worker.submit(TUNE, f"tune {i}", lambda: None)
                 for i in range(200)]

        assert not any(r.failed for r in asked), (
            "refused a request instead of replacing an older one")
        assert all(r.superseded for r in asked[:-1])
        assert not asked[-1].finished, "the last one should still be waiting"
    finally:
        hold.set()


def test_only_one_of_each_kind_is_ever_waiting(worker):
    """What the queue holds, rather than what comes out of it."""
    hold = a_worker_that_is_busy(worker)
    try:
        for i in range(50):
            worker.submit(TUNE, f"tune {i}", lambda: None)
            worker.submit(GAIN, f"gain {i}", lambda: None)
            worker.submit(GAIN_MODE, f"mode {i}", lambda: None)

        with worker._gate:
            waiting = list(worker._waiting)
        assert len(waiting) == 3, [r.what for r in waiting]
        assert {r.kind for r in waiting} == {TUNE, GAIN, GAIN_MODE}
    finally:
        hold.set()


def test_asking_while_stopping_leaves_nobody_waiting(worker):
    """The check and the queueing are one step with respect to stopping.

    They were two, and a stop landing between them put the request into a
    queue nothing would ever drain again - a caller who waited on it
    waited for good.
    """
    asked: list = []
    keep_going = threading.Event()
    keep_going.set()

    def ask_until_told_to_stop() -> None:
        while keep_going.is_set():
            asked.append(worker.submit(GAIN, "gain", lambda: None))

    askers = [threading.Thread(target=ask_until_told_to_stop, daemon=True)
              for _ in range(4)]
    for thread in askers:
        thread.start()
    time.sleep(0.2)                     # let them get going

    worker.stop()
    keep_going.clear()
    for thread in askers:
        thread.join(timeout=10)

    assert asked, "the askers never asked for anything"
    unfinished = [r for r in asked if not r.finished]
    assert not unfinished, f"{len(unfinished)} of {len(asked)} left waiting"


def test_stopping_while_a_write_is_in_flight_still_releases_the_rest(worker):
    hold = a_worker_that_is_busy(worker)
    queued = [worker.submit(GAIN, f"gain {i}", lambda: None) for i in range(3)]

    worker.stop(timeout=0.3)
    hold.set()

    for request in queued:
        assert request.finished, "left somebody waiting on a dropped write"
        assert request.superseded


def test_a_request_taken_but_not_begun_is_not_written_after_the_stop(worker):
    """Off the queue is not yet at the device.

    Between the two the worker holds a request that ``stop`` can no
    longer find in the queue.  If that counted as begun, a write would
    land on the device after shutdown had been told there would be no
    more - which on the way out is a write to a handle that is being
    closed underneath it.
    """
    # Hold the worker inside a write first, so the patch below is in
    # place before it next looks at the queue.
    release = threading.Event()
    inside = threading.Event()
    worker.submit(GAIN, "the one in flight", blocking(release, started=inside))
    assert inside.wait(5), "the worker never started the first write"

    taken = threading.Event()
    go = threading.Event()
    real_next = worker._next

    def take_and_park():
        request = real_next()
        if request is not None:
            taken.set()
            go.wait(5)          # the stop happens in here
        return request

    worker._next = take_and_park

    written = []
    asked = worker.submit(TUNE, "Tuned to 80.0 MHz",
                          lambda: written.append("Tuned to 80.0 MHz"))
    release.set()
    assert taken.wait(5), "the worker never took the request off the queue"

    # Decided while the worker is parked with the request in hand, so
    # the stop is certainly first.  The join times out; that is the
    # point of the timeout.
    worker.stop(timeout=0.1)
    go.set()

    assert asked.wait(5), "nobody ever answered for it"
    assert asked.superseded, "answered as though it had been written"
    assert not asked.failed
    assert written == [], f"written to the device after the stop: {written}"


def test_a_write_that_had_begun_is_let_finish(worker):
    """The other side of the same boundary.

    A write that is inside the driver cannot be taken back, and pulling
    the handle out from under it is what this whole file exists to
    avoid.  Stop waits for it instead.
    """
    release = threading.Event()
    inside = threading.Event()
    done = []
    asked = worker.submit(TUNE, "Tuned to 80.0 MHz",
                          blocking(release, done, "tune", started=inside))
    assert inside.wait(5), "the write never started"

    stopped = threading.Event()
    threading.Thread(target=lambda: (worker.stop(timeout=5), stopped.set()),
                     daemon=True).start()

    assert not stopped.wait(0.2), "stop went past a write that was running"
    release.set()

    assert stopped.wait(5), "stop never returned"
    assert done == ["tune"], "the write did not finish"
    assert asked.finished and not asked.superseded and not asked.failed


# ----------------------------------------------------------------------
# Coalescing stops at an event
# ----------------------------------------------------------------------

def test_a_tune_before_an_event_is_not_taken_away_by_a_later_one(worker):
    """The order the queue runs in has to be the order that was asked for.

    Coalescing used to reach the whole queue, so a tune queued before a
    recording could be removed by a tune queued after it.  The worker
    still ran one thing at a time - it just ran the wrong set of
    things, and the recording ended up named for a station the
    receiver had been told to leave.
    """
    release = threading.Event()
    inside = threading.Event()
    done: list = []
    worker.submit(GAIN, "the one in flight", blocking(release, started=inside))
    assert inside.wait(5), "the worker never started"

    first = worker.submit(TUNE, "tune 81.3",
                          lambda: done.append("tune 81.3"))
    recording = worker.submit(RECORDING, "recording",
                              lambda: done.append("recording"))
    second = worker.submit(TUNE, "tune 82.5",
                           lambda: done.append("tune 82.5"))
    release.set()

    assert second.wait(5), "the last request never ran"
    assert not first.superseded, "the tune before the recording was taken away"
    assert done == ["tune 81.3", "recording", "tune 82.5"], done
    assert recording.finished and not recording.failed


def test_two_tunes_after_an_event_still_collapse(worker):
    """The coalescing that is worth having is still there."""
    release = threading.Event()
    inside = threading.Event()
    done: list = []
    worker.submit(GAIN, "the one in flight", blocking(release, started=inside))
    assert inside.wait(5), "the worker never started"

    worker.submit(RECORDING, "recording", lambda: done.append("recording"))
    first = worker.submit(TUNE, "tune 81.3", lambda: done.append("81.3"))
    last = worker.submit(TUNE, "tune 82.5", lambda: done.append("82.5"))
    release.set()

    assert last.wait(5)
    assert first.superseded, "the tune before it should have been replaced"
    assert done == ["recording", "82.5"], done


def test_two_events_are_two_events(worker):
    """Nothing coalesces a recording, in either direction."""
    release = threading.Event()
    inside = threading.Event()
    done: list = []
    worker.submit(GAIN, "the one in flight", blocking(release, started=inside))
    assert inside.wait(5), "the worker never started"

    worker.submit(RECORDING, "start", lambda: done.append("start"))
    last = worker.submit(RECORDING, "stop", lambda: done.append("stop"))
    release.set()

    assert last.wait(5)
    assert done == ["start", "stop"], done


# ----------------------------------------------------------------------
# Taking a request back
# ----------------------------------------------------------------------

def test_a_queued_request_can_be_taken_back(worker):
    """And whoever was waiting on it is released, not left there."""
    release = threading.Event()
    inside = threading.Event()
    done: list = []
    worker.submit(GAIN, "the one in flight", blocking(release, started=inside))
    assert inside.wait(5), "the worker never started"

    asked = worker.submit(RECORDING, "recording",
                          lambda: done.append("recording"))

    assert worker.cancel(asked) is True
    release.set()

    assert asked.wait(5), "the waiter was left there"
    assert asked.cancelled and not asked.failed
    assert done == [], "it ran anyway"


def test_a_request_that_has_begun_cannot_be_taken_back(worker):
    """It is already inside whatever it does; the caller has to settle."""
    release = threading.Event()
    inside = threading.Event()
    asked = worker.submit(RECORDING, "recording",
                          blocking(release, started=inside))
    assert inside.wait(5), "it never started"

    assert worker.cancel(asked) is False

    release.set()
    assert asked.wait(5)
    assert not asked.cancelled


def test_cancelling_nothing_is_allowed(worker):
    assert worker.cancel(None) is False
