"""Shutting down while something is still using what is being closed.

A bounded join cannot promise that every thread has finished, so these check
the two hazards that leaves: audio arriving after the output is closed, and a
gain write landing on a device that has just gone. Timing is forced with
events and deliberate delays rather than left to chance.
"""

from __future__ import annotations

import threading
import time
import wave

import numpy as np
import pytest

from conftest import FakeLibRtlSdr, RTLSDR_INACTIVE, RTLSDR_RUNNING

import fm_radio.device_handle as device_handle
from fm_radio.controller import FMReceiverController
from fm_radio.exceptions import RecordingError, SDRDeviceError


@pytest.fixture
def receiver(no_user_config):
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        try:
            instance.cleanup()
        except Exception:
            pass


def iq_block(controller):
    size = controller.sdr_receiver.block_size
    rng = np.random.default_rng(0)
    return (0.3 * (rng.standard_normal(size)
                   + 1j * rng.standard_normal(size))).astype(np.complex64)


# ----------------------------------------------------------------------
# Audio arriving after the output is closed
# ----------------------------------------------------------------------

def test_a_closed_output_refuses_audio(receiver):
    """The stream behind the queue is gone; a block has nowhere to go."""
    audio = receiver.audio_output
    left = np.zeros(192, dtype=np.float32)

    audio.cleanup()
    assert audio.closed

    audio.enqueue_audio(left, left)         # must not raise
    assert audio.audio_buffer_queue.empty()


def test_a_slow_block_cannot_reach_a_closed_output(receiver, monkeypatch):
    """The join is bounded, so a slow block outlives the wait for it.

    Demodulation here takes longer than the join allows, which is what a
    stalled machine does for real.
    """
    reached = []
    original_put = receiver.audio_output.audio_buffer_queue.put

    def counted(*args, **kwargs):
        reached.append(1)
        return original_put(*args, **kwargs)

    monkeypatch.setattr(receiver.audio_output.audio_buffer_queue, "put", counted)

    demodulating = threading.Event()
    release = threading.Event()
    original_demodulate = receiver.fm_demodulator.demodulate

    def stalled(composite):
        demodulating.set()
        release.wait(10)
        return original_demodulate(composite)

    monkeypatch.setattr(receiver.fm_demodulator, "demodulate", stalled)
    monkeypatch.setattr("fm_radio.controller._THREAD_JOIN_TIMEOUT_SEC", 0.2)

    receiver.sdr_receiver.data_queue.put(
        (receiver.sdr_receiver.tuning_generation, iq_block(receiver)))
    thread = threading.Thread(target=receiver.processing_thread, daemon=True)
    receiver.threads.append(thread)
    thread.start()
    assert demodulating.wait(10), "the block never started"

    receiver.cleanup()                      # gives up waiting, closes anyway
    before = len(reached)
    release.set()                           # ... and now the block finishes
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(reached) == before, "audio reached an output that was closed"


# ----------------------------------------------------------------------
# A gain write landing on a device that has gone
# ----------------------------------------------------------------------

def test_a_closed_device_refuses_writes(receiver):
    sdr = receiver.sdr_receiver
    sdr.stop()
    assert sdr.closed

    sdr.set_gain(30.0)                      # must not raise
    sdr.set_center_frequency(81.3e6)


def test_a_write_in_flight_finishes_before_the_device_closes(receiver,
                                                              monkeypatch):
    """The gain worker can be inside a USB call when shutdown starts.

    The flag alone cannot help there - the write is already past it - so the
    close waits behind the same lock.
    """
    device = receiver.sdr_receiver.sdr
    handle = receiver.sdr_receiver.handle
    closed = threading.Event()
    writing = threading.Event()
    after_close = []
    original_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        # Longer than the second auto_gain.stop() waits for its worker:
        # inside that, stopping the writer first is enough on its own and
        # the lock is never the thing being tested.
        time.sleep(1.5)
        if closed.is_set():
            after_close.append(gain)
        return original_set_gain(gain)

    # _real_close, not device.close: the handle keeps the original and
    # calls it directly, so device.close is not on the path that frees
    # anything.  Watching that one would be watching nothing.
    original_close = handle._real_close

    def marked_close():
        closed.set()
        return original_close()

    monkeypatch.setattr(device, "set_gain", slow_write)
    monkeypatch.setattr(handle, "_real_close", marked_close)

    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    receiver.cleanup()
    time.sleep(0.5)

    assert not after_close, f"the device was written to after close: {after_close}"


def test_the_gain_worker_stops_before_the_device_does(receiver, monkeypatch):
    """Ordering, as distinct from the lock: the writer goes first."""
    order = []
    monkeypatch.setattr(receiver.auto_gain, "stop",
                        lambda: order.append("gain worker"))
    monkeypatch.setattr(receiver.sdr_receiver, "stop",
                        lambda: order.append("sdr"))
    monkeypatch.setattr(receiver.audio_output, "cleanup",
                        lambda: order.append("audio"))

    receiver.cleanup()

    assert order == ["gain worker", "sdr", "audio"]


# ----------------------------------------------------------------------
# The endings cleanup has to survive
# ----------------------------------------------------------------------

def test_cleanup_is_idempotent(receiver):
    receiver.start_background()
    receiver.cleanup()
    receiver.cleanup()                      # must not raise
    assert receiver.audio_output.closed and receiver.sdr_receiver.closed


def test_cleanup_without_having_started(receiver):
    receiver.cleanup()
    assert receiver.quit_event.is_set()


def test_cleanup_after_a_partial_start(receiver, monkeypatch):
    """start_background() starts the SDR thread before the processing one."""
    real_thread = threading.Thread

    def fail_on_the_second(*args, **kwargs):
        if any("processing" in str(a) for a in kwargs.values()):
            raise RuntimeError("could not start the processing thread")
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", fail_on_the_second)
    with pytest.raises(RuntimeError):
        receiver.start_background()

    receiver.cleanup()                      # must not raise
    assert receiver.sdr_receiver.closed


# ----------------------------------------------------------------------
# A device write that never comes back
# ----------------------------------------------------------------------

def test_a_wedged_write_does_not_hold_cleanup_open(receiver, monkeypatch):
    """Waiting for an in-flight write has to be bounded.

    A USB write normally takes 40-200 ms, but a device that has stopped
    answering never returns from one.  The close waits behind the same lock
    the writers hold, so an unbounded wait there is a shutdown that never
    finishes.  It gives up instead - and gives up on closing too, rather
    than pulling the handle out from under a call that is still inside it.
    """
    device = receiver.sdr_receiver.sdr
    handle = receiver.sdr_receiver.handle
    writing = threading.Event()
    release = threading.Event()
    closes: list[str] = []
    real_close = handle._real_close

    def wedged_write(gain):
        writing.set()
        release.wait(30)                    # never set until the test is done

    monkeypatch.setattr(device, "set_gain", wedged_write)
    # The handle calls _real_close directly; device.close is not on the
    # path that frees anything.
    monkeypatch.setattr(handle, "_real_close",
                        lambda: (closes.append("close"), real_close()))
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.5)

    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    started = time.monotonic()
    try:
        receiver.cleanup()
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"cleanup took {elapsed:.1f} s behind a stuck write"
        assert not closes, "closed the device while a write was still inside it"
        assert receiver.sdr_receiver.closed, "later writes must still be refused"
    finally:
        release.set()                       # let the wedged worker unwind


def test_a_write_released_after_the_wait_still_finds_the_device_marked_closed(
        receiver, monkeypatch):
    """Giving up on the close must not give up on refusing later writes."""
    device = receiver.sdr_receiver.sdr
    writing = threading.Event()
    release = threading.Event()

    def wedged_write(gain):
        writing.set()
        release.wait(30)

    monkeypatch.setattr(device, "set_gain", wedged_write)
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5)
    receiver.sdr_receiver.stop()
    release.set()
    time.sleep(0.2)

    before = list(device.gain_calls)
    receiver.sdr_receiver.set_gain(40.0)    # must not raise, must not write
    assert device.gain_calls == before


# ----------------------------------------------------------------------
# Reading the device after it has gone
# ----------------------------------------------------------------------

class _TrackingDevice:
    """Wraps the fake device and notes any read made after it was closed."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.is_closed = False
        self.reads_after_close: list[str] = []

    @property
    def center_freq(self):
        if self.is_closed:
            self.reads_after_close.append("center_freq")
        return self._inner.center_freq

    @center_freq.setter
    def center_freq(self, value) -> None:
        self._inner.center_freq = value

    def get_gain(self) -> float:
        if self.is_closed:
            self.reads_after_close.append("get_gain")
        return self._inner.get_gain()

    def close(self) -> None:
        self.is_closed = True
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_slow_block_does_not_read_a_closed_device(receiver, monkeypatch):
    """The same overrun as the audio case, on the telemetry side.

    The block that outlived the join goes on to build a snapshot, and a
    snapshot asks the receiver for its frequency and gain.  Those are USB
    reads on a handle that closing has already freed.
    """
    sdr = receiver.sdr_receiver
    device = _TrackingDevice(sdr.sdr)
    sdr.sdr = device

    demodulating = threading.Event()
    release = threading.Event()
    original_demodulate = receiver.fm_demodulator.demodulate

    def stalled(composite):
        demodulating.set()
        release.wait(10)
        return original_demodulate(composite)

    monkeypatch.setattr(receiver.fm_demodulator, "demodulate", stalled)
    monkeypatch.setattr("fm_radio.controller._THREAD_JOIN_TIMEOUT_SEC", 0.2)

    sdr.data_queue.put((sdr.tuning_generation, iq_block(receiver)))
    thread = threading.Thread(target=receiver.processing_thread, daemon=True)
    receiver.threads.append(thread)
    thread.start()
    assert demodulating.wait(10), "the block never started"

    receiver.cleanup()                      # gives up waiting, closes anyway
    release.set()                           # ... and now the block finishes
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert device.reads_after_close == [], (
        f"read a device that was closed: {device.reads_after_close}")


class _GoneDevice:
    """A handle that has been freed: touching it at all is the bug."""

    @property
    def center_freq(self):
        raise AssertionError("read the frequency from a closed device")

    def get_gain(self):
        raise AssertionError("read the gain from a closed device")


def test_reads_fall_back_to_the_last_value_once_closed(receiver):
    """What a caller gets instead: the last reading, not an exception.

    The handle is swapped for one that refuses every read, because the
    fake device would happily answer after close and librtlsdr will not.
    """
    sdr = receiver.sdr_receiver
    sdr.set_center_frequency(81.3e6)
    gain_before = sdr.get_gain()

    sdr.stop()
    sdr.sdr = _GoneDevice()

    assert sdr.get_center_frequency() == pytest.approx(81.3e6)
    assert sdr.get_gain() == pytest.approx(gain_before)


def test_a_read_does_not_wait_behind_a_device_write(receiver):
    """These run on the processing thread, which has 16 ms a block.

    Taking the lock and waiting would put a 40-200 ms USB write in the
    middle of the realtime path; the cached value is worth more than an
    exactly current one.
    """
    sdr = receiver.sdr_receiver
    assert sdr.handle.device_lock.acquire()
    try:
        started = time.monotonic()
        sdr.get_center_frequency()
        sdr.get_gain()
        elapsed = time.monotonic() - started
    finally:
        sdr.handle.device_lock.release()

    assert elapsed < 0.1, f"a read waited {elapsed * 1e3:.0f} ms for the lock"


# ----------------------------------------------------------------------
# Closing the output between the check and the hand-off
# ----------------------------------------------------------------------

def test_cleanup_cannot_close_the_stream_mid_enqueue(receiver, monkeypatch):
    """The flag check and the put have to be one step.

    Between them, enqueue_audio has seen an open output and not yet handed
    the block over; cleanup arriving there closes the stream the block is
    about to be queued for.
    """
    audio = receiver.audio_output
    events: list[str] = []
    in_put = threading.Event()
    finish_put = threading.Event()
    original_put = audio.audio_buffer_queue.put

    def slow_put(*args, **kwargs):
        in_put.set()
        finish_put.wait(10)
        events.append("block queued")
        return original_put(*args, **kwargs)

    monkeypatch.setattr(audio.audio_buffer_queue, "put", slow_put)
    monkeypatch.setattr(audio.stream, "close",
                        lambda: events.append("stream closed"))

    block = np.zeros(192, dtype=np.float32)
    feeder = threading.Thread(target=audio.enqueue_audio, args=(block, block),
                              daemon=True)
    feeder.start()
    assert in_put.wait(5), "the block never reached the queue"

    closer = threading.Thread(target=audio.cleanup, daemon=True)
    closer.start()
    time.sleep(0.5)                         # every chance to get ahead
    finish_put.set()
    feeder.join(timeout=10)
    closer.join(timeout=10)

    assert events == ["block queued", "stream closed"], (
        f"the stream closed under an enqueue in progress: {events}")


# ----------------------------------------------------------------------
# Sampling has to stop even when the device cannot be closed
# ----------------------------------------------------------------------

def test_sampling_stops_even_when_the_device_cannot_be_closed(receiver,
                                                              monkeypatch):
    """Giving up on the close must not mean giving up on the read.

    read_samples_async does not return until it is cancelled, so a stop
    that skips the cancel leaves the receive thread running for the life
    of the process - still calling back, still filling the queue.

    Cancelling does not need the device lock the way closing does.
    rtlsdr_cancel_async only flips two fields on the device struct; it
    sends nothing over USB and frees nothing, so it is safe alongside a
    control-transfer write that is still in the air.  close() is the one
    that frees the handle that write is using.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    writing = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(device, "set_gain",
                        lambda gain: (writing.set(), release.wait(30)))
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.5)

    reader = running_reader(receiver)
    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    try:
        receiver.cleanup()

        reader.join(timeout=5)
        assert not reader.is_alive(), "the receive thread outlived cleanup"
        assert "cancel" in device.calls, "the async read was never cancelled"
        assert "close" not in device.calls, (
            "closed the device while a write was still inside it")
    finally:
        release.set()
        time.sleep(0.2)             # let the wedged write unwind


def significant(calls):
    """Drop the retries that did nothing.

    The cancel is repeated until the read has actually returned, so a run
    normally ends with a few attempts that found nothing to cancel.  They
    are not events, they are the loop noticing it is not finished.
    """
    return [c for c in calls if c != "cancel (no read running)"]


def running_reader(receiver):
    """Start the SDR thread and wait until the async read is under way."""
    sdr = receiver.sdr_receiver
    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert sdr.sdr.reading.wait(5), "the async read never started"
    return thread


def test_sampling_is_cancelled_before_the_device_is_closed(receiver):
    """Order matters: cancel_read_async dereferences the handle close frees."""
    sdr = receiver.sdr_receiver
    thread = running_reader(receiver)

    sdr.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert significant(sdr.sdr.calls) == ["read", "cancel", "close"]


def test_stopping_twice_does_not_touch_a_freed_handle(receiver):
    """cleanup runs twice on several paths, and the second must do nothing.

    pyrtlsdr guards close() with its own flag but passes the device
    pointer to rtlsdr_cancel_async without checking, so a second cancel
    after the close reads memory that has been freed.
    """
    sdr = receiver.sdr_receiver
    thread = running_reader(receiver)

    sdr.stop()
    sdr.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert significant(sdr.sdr.calls) == ["read", "cancel", "close"]


def test_a_driver_side_close_stops_the_cancel(receiver, monkeypatch):
    """pyrtlsdr closes the device itself when a control write fails."""
    device = receiver.sdr_receiver.sdr
    monkeypatch.setattr(device, "device_opened", False)
    device.calls.clear()

    receiver.sdr_receiver.stop()

    assert device.calls == [], (
        f"touched a handle the driver had already freed: {device.calls}")


# ----------------------------------------------------------------------
# A recording that starts after, or as, the receiver stops
# ----------------------------------------------------------------------

def test_a_recording_cannot_start_after_the_receiver_stopped(receiver,
                                                             tmp_path):
    """The worker that would write the blocks is gone, and so are the blocks."""
    sdr = receiver.sdr_receiver
    sdr.stop()
    path = tmp_path / "after_stop.wav"

    with pytest.raises(RecordingError):
        sdr.start_iq_recording(str(path))

    assert not sdr.iq_recording
    assert sdr.iq_record_wave is None
    assert not path.exists(), "left a file behind for a recording that cannot run"


def test_a_recording_starting_as_the_receiver_stops_is_not_left_behind(
        receiver, monkeypatch, tmp_path):
    """The overlap the flag check on its own cannot cover.

    A start that is already inside wave.open has passed every check there
    is.  Shutdown waits for it to finish installing itself and then tears
    it down properly, rather than racing past and leaving an open file
    with nothing to write to it.
    """
    sdr = receiver.sdr_receiver
    opening = threading.Event()
    release = threading.Event()
    real_open = wave.open

    def slow_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opening.set()
        release.wait(10)
        return handle

    monkeypatch.setattr("fm_radio.sdr_receiver.wave.open", slow_open)

    path = tmp_path / "overlap.wav"
    starter = threading.Thread(target=sdr.start_iq_recording,
                               args=(str(path),), daemon=True)
    starter.start()
    assert opening.wait(5), "the recording never reached wave.open"

    stopper = threading.Thread(target=sdr.stop, daemon=True)
    stopper.start()
    time.sleep(0.3)                 # every chance to race past
    release.set()
    starter.join(timeout=10)
    stopper.join(timeout=10)

    assert not starter.is_alive() and not stopper.is_alive()
    assert not sdr.iq_recording, "still marked as recording after shutdown"
    assert sdr.iq_record_wave is None, "a wave handle was left open"


def test_stopping_a_recording_does_not_wait_for_a_worker_that_is_gone(
        receiver, tmp_path):
    """Nothing is left to answer the flush sentinel, so nothing waits for one."""
    sdr = receiver.sdr_receiver
    sdr.start_iq_recording(str(tmp_path / "orphan.wav"))
    sdr._stop_iq_record_worker()
    assert not sdr._iq_record_worker.is_alive()

    started = time.monotonic()
    sdr.stop_iq_recording()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"waited {elapsed:.1f} s on a worker that had gone"
    assert not sdr.iq_recording
    assert sdr.iq_record_wave is None


def test_an_audio_recording_cannot_start_after_cleanup(receiver, tmp_path):
    """The same hole on the audio side, closed the same way."""
    audio = receiver.audio_output
    audio.cleanup()
    path = tmp_path / "after_cleanup.wav"

    with pytest.raises(RecordingError):
        audio.start_recording(str(path))

    assert not audio.recording
    assert audio.record_wave is None
    assert not path.exists()


# ----------------------------------------------------------------------
# The cancel that fails, and what pyrtlsdr does about it
# ----------------------------------------------------------------------
#
# rtlsdr_cancel_async returns an error whenever no read is running, and
# pyrtlsdr answers an error by closing the device and raising
# (rtlsdr.py:699-706).  That close lands outside _device_lock, which is
# the one place a close must never be.  The fake in conftest copies this.


def test_cleanup_before_the_read_begins_never_cancels(receiver):
    """The thread exists, the read does not: there is nothing to cancel.

    Asking anyway is what makes the C call fail, and a failed cancel takes
    the device with it.  The read that follows then goes through a handle
    that has already been freed.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    at_the_gate = threading.Event()
    gate = threading.Event()

    def late_start():
        at_the_gate.set()
        gate.wait(10)
        sdr.start()

    thread = threading.Thread(target=late_start, name="SDRThread", daemon=True)
    thread.start()
    assert at_the_gate.wait(5), "the SDR thread never started"

    receiver.cleanup()
    gate.set()                      # the thread only gets going now
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert "cancel failed" not in device.calls, (
        f"asked a device that was not reading to cancel: {device.calls}")
    assert "read" not in device.calls, (
        f"started a read after the device was closed: {device.calls}")
    assert device.calls.count("close") == 1, device.calls


def test_a_cancel_that_arrives_too_early_does_not_close_the_device(
        receiver, monkeypatch):
    """The window between arming the read and librtlsdr running it.

    A cancel landing in there fails, and pyrtlsdr's answer to a failed
    cancel is to close - from a thread that does not hold the device lock,
    on top of a gain write that is still using the handle.  Suppressing
    the wrapper's close is only half of it: the cancel itself did nothing,
    so the read starts regardless and has to be asked again.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    armed = threading.Event()
    let_the_read_begin = threading.Event()
    real_read = device.read_samples_async

    def read_samples_async(cb, num_samples=None):
        armed.set()
        let_the_read_begin.wait(10)     # armed, not yet running
        real_read(cb, num_samples)

    writing = threading.Event()
    release = threading.Event()
    closed_under_the_write: list[float] = []
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(10)
        if not device.device_opened:
            closed_under_the_write.append(gain)
        real_set_gain(gain)

    monkeypatch.setattr(device, "read_samples_async", read_samples_async)
    monkeypatch.setattr(device, "set_gain", slow_write)

    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert armed.wait(5), "the read never armed"
    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()
    # Past auto_gain's one-second join and well into the cancel attempts.
    time.sleep(1.5)

    assert device.device_opened, (
        f"a failed cancel closed the device: {device.calls}")
    assert not closed_under_the_write

    let_the_read_begin.set()        # the read finally reaches librtlsdr
    release.set()                   # and the write finally returns

    assert done.wait(20), "cleanup never finished"
    thread.join(timeout=10)

    assert not thread.is_alive(), "the read was never cancelled"
    assert not closed_under_the_write, (
        "the device closed under a write that was still in flight")
    assert device.calls.count("close") == 1, device.calls
    assert significant(device.calls)[-2:] == ["cancel", "close"], device.calls


def test_a_read_that_will_not_return_leaves_the_device_open(receiver,
                                                            monkeypatch):
    """Closing would free the handle the read is still going through."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    reading = threading.Event()
    release = threading.Event()

    def deaf_read(cb, num_samples=None):
        device.read_async_canceling = False
        device.calls.append("read")
        device.reading.set()
        reading.set()
        release.wait(30)            # ignores every cancel
        device.reading.clear()

    monkeypatch.setattr(device, "read_samples_async", deaf_read)
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert reading.wait(5)

    try:
        started = time.monotonic()
        receiver.cleanup()
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"cleanup took {elapsed:.1f} s"
        assert "close" not in device.calls, (
            "freed a handle a read was still going through")
        assert device.device_opened
    finally:
        release.set()
        thread.join(timeout=10)


# ----------------------------------------------------------------------
# Not resting on a flag the reader is free to clear
# ----------------------------------------------------------------------

def test_the_wrapper_is_not_used_when_the_c_call_is_reachable(receiver):
    """Every route through pyrtlsdr's cancel can end in a close."""
    sdr = receiver.sdr_receiver
    thread = running_reader(receiver)

    sdr.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "wrapper cancel" not in sdr.sdr.calls, sdr.sdr.calls


def test_a_cancel_never_rests_on_the_wrappers_flag(receiver, monkeypatch):
    """The reader clears read_async_canceling in the window we cancel in.

    pyrtlsdr's read_bytes_async sets that flag False on its way in
    (rtlsdr.py:599) and only then hands over to librtlsdr, so between the
    two the flag is down and the C side still says no read is running.
    Anything that sets the flag to keep a failed cancel from closing the
    device can be undone right there - and the close that follows lands
    outside _device_lock, on a gain write still using the handle.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    armed = threading.Event()
    flag_cleared = threading.Event()
    let_the_read_begin = threading.Event()
    real_read = device.read_samples_async

    def read_samples_async(cb, num_samples=None):
        armed.set()
        device.read_async_canceling = False     # rtlsdr.py:599
        flag_cleared.set()
        let_the_read_begin.wait(10)             # librtlsdr not reached yet
        real_read(cb, num_samples)

    writing = threading.Event()
    release = threading.Event()
    closed_under_the_write: list[float] = []
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(10)
        if not device.device_opened:
            closed_under_the_write.append(gain)
        real_set_gain(gain)

    monkeypatch.setattr(device, "read_samples_async", read_samples_async)
    monkeypatch.setattr(device, "set_gain", slow_write)

    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert armed.wait(5), "the read never armed"
    assert flag_cleared.wait(5), "the reader never cleared the flag"
    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()
    # Past auto_gain's one-second join and well into the cancel attempts,
    # every one of which finds no read running.
    time.sleep(1.5)

    assert device.device_opened, (
        f"a failed cancel closed the device: {device.calls}")
    assert "wrapper cancel" not in device.calls, device.calls
    assert not closed_under_the_write

    let_the_read_begin.set()        # the read finally reaches librtlsdr
    release.set()                   # and the write finally returns

    assert done.wait(20), "cleanup never finished"
    thread.join(timeout=10)

    assert not thread.is_alive(), "the read was never cancelled"
    assert not closed_under_the_write, (
        "the device closed under a write that was still in flight")
    assert "wrapper cancel" not in device.calls, device.calls
    assert device.calls.count("close") == 1, device.calls
    assert significant(device.calls)[-2:] == ["cancel", "close"], device.calls


def test_the_fallback_is_not_asked_about_a_read_it_cannot_cancel(receiver,
                                                                 monkeypatch):
    """Armed, but librtlsdr has not started the read: -2, and a close.

    pyrtlsdr's cancel closes the device whenever the C call fails, and the
    call fails in every state but RUNNING.  With no C entry point to fall
    back from, the only safe move is not to ask: an uncancelled read costs
    a handle the process releases on exit, where the close costs whatever
    write is in flight and lets the read start afterwards on a dead one.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.RAW_CANCEL_ASYNC", None)
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    armed = threading.Event()
    let_the_read_begin = threading.Event()
    real_read = device.read_samples_async

    def read_samples_async(cb, num_samples=None):
        armed.set()
        device.read_async_canceling = False
        let_the_read_begin.wait(10)     # librtlsdr not reached yet
        real_read(cb, num_samples)

    monkeypatch.setattr(device, "read_samples_async", read_samples_async)

    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert armed.wait(5), "the read never armed"

    try:
        started = time.monotonic()
        receiver.cleanup()
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"cleanup took {elapsed:.1f} s"
        assert "wrapper cancel" not in device.calls, device.calls
        assert "close" not in device.calls, device.calls
        assert device.device_opened
    finally:
        let_the_read_begin.set()
        device.cancelled.set()
        thread.join(timeout=10)


def test_a_callback_is_not_taken_as_proof_the_read_is_running(receiver,
                                                              monkeypatch):
    """A callback says one arrived, not that the read is still running.

    Between the two the read can have moved to CANCELING, where pyrtlsdr's
    cancel gets -2, closes the device and leaves the read - still
    unwinding - using a handle that has been freed.  There is no moment
    that can be shown from out here to be safe, so the wrapper is never
    asked at all.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.RAW_CANCEL_ASYNC", None)

    real_read = device.read_samples_async

    def slow_to_unwind(cb, num_samples=None):
        real_read(cb, num_samples)      # returns once cancelled
        time.sleep(1.0)                 # ... and is still unwinding

    monkeypatch.setattr(device, "read_samples_async", slow_to_unwind)
    thread = running_reader(receiver)
    sdr.callback(np.zeros(8, dtype=np.complex64), device)

    writing = threading.Event()
    release = threading.Event()
    closed_under_the_write: list[float] = []
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(10)
        if not device.device_opened:
            closed_under_the_write.append(gain)
        real_set_gain(gain)

    monkeypatch.setattr(device, "set_gain", slow_write)
    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()
    time.sleep(0.4)
    # Something else cancels: the read is no longer RUNNING by the time
    # any ask could land, though a callback certainly did arrive.
    FakeLibRtlSdr.rtlsdr_cancel_async(device)
    release.set()

    assert done.wait(30), "cleanup never finished"

    assert "wrapper cancel" not in device.calls, device.calls
    assert "cancel failed" not in device.calls, device.calls
    assert not closed_under_the_write
    assert device.device_opened, (
        "closed a device whose state could not be established")

    device.cancelled.set()
    thread.join(timeout=10)


# ----------------------------------------------------------------------
# One owner for the handle's lifetime
# ----------------------------------------------------------------------

def test_a_failed_read_cannot_close_beside_a_write(receiver, monkeypatch):
    """pyrtlsdr closes the device from inside a failed async read too.

    That is rtlsdr.py:601-603, on the reader thread, which holds nothing.
    The close waits for the write in flight instead of freeing the handle
    the write is still using.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    reading = threading.Event()
    fail_now = threading.Event()

    def failing_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        fail_now.wait(10)
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE
        device.calls.append("read failed")
        device.close()                  # rtlsdr.py:601-603
        raise OSError("LIBUSB_ERROR_IO: could not read")

    writing = threading.Event()
    release = threading.Event()
    closed_under_the_write: list[float] = []
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(10)
        if not device.device_opened:
            closed_under_the_write.append(gain)
        real_set_gain(gain)

    monkeypatch.setattr(device, "read_samples_async", failing_read)
    monkeypatch.setattr(device, "set_gain", slow_write)

    def read_and_swallow():
        try:
            sdr.start()
        except SDRDeviceError:
            pass                        # the point is the close, not the raise

    reader = threading.Thread(target=read_and_swallow, name="SDRThread",
                              daemon=True)
    reader.start()
    assert reading.wait(5), "the read never started"

    writer = threading.Thread(target=lambda: sdr.set_gain(30.0), daemon=True)
    writer.start()
    assert writing.wait(5), "the write never started"

    fail_now.set()                      # the read fails and asks to close
    time.sleep(0.5)

    assert device.device_opened, (
        "closed the device while a write was still inside it")
    assert not closed_under_the_write

    release.set()                       # the write returns
    writer.join(timeout=10)
    reader.join(timeout=10)

    assert not writer.is_alive() and not reader.is_alive()
    assert not closed_under_the_write, (
        "the write found the device closed under it")
    assert device.calls.count("close") == 1, device.calls


def test_a_close_waiting_on_a_wedged_write_gives_up(receiver, monkeypatch):
    """The same bargain as everywhere else, and bounded."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.4)

    writing = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(device, "set_gain",
                        lambda gain: (writing.set(), release.wait(30)))

    writer = threading.Thread(target=lambda: sdr.set_gain(30.0), daemon=True)
    writer.start()
    assert writing.wait(5)

    try:
        started = time.monotonic()
        device.close()                  # as pyrtlsdr would, from elsewhere
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, f"the close waited {elapsed:.1f} s"
        assert device.device_opened, "closed underneath a wedged write"
        assert "close" not in device.calls, device.calls
    finally:
        release.set()
        writer.join(timeout=10)




# ----------------------------------------------------------------------
# One owner for the handle's lifetime
# ----------------------------------------------------------------------

def test_a_close_cannot_land_between_the_check_and_the_c_call(receiver,
                                                              monkeypatch):
    """pyrtlsdr closes the device from inside a failed control write.

    That happens on whichever thread made the write, with no regard for
    the cancel loop, which by then may have satisfied itself that the
    handle is alive and be about to hand the pointer to librtlsdr.  Every
    close goes through one lock that the C call holds for its duration, so
    there is no gap between the two to land in.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    thread = running_reader(receiver)

    in_the_close = threading.Event()
    let_the_close_finish = threading.Event()
    real_close = sdr.handle._real_close

    def slow_close():
        in_the_close.set()
        let_the_close_finish.wait(10)
        real_close()

    monkeypatch.setattr(sdr.handle, "_real_close", slow_close)

    # rtlsdr.py:317 - a failed gain write closes the device itself.
    def failing_write(gain):
        device.calls.append("gain write failed")
        device.close()
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)

    def write_and_swallow():
        try:
            sdr.set_gain(30.0)
        except SDRDeviceError:
            pass                    # the point is the close, not the raise

    closer = threading.Thread(target=write_and_swallow, daemon=True)
    closer.start()
    assert in_the_close.wait(5), "the failing write never reached the close"

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()
    time.sleep(0.5)                 # the cancel loop is running by now

    let_the_close_finish.set()
    device.cancelled.set()          # closing makes the real read return

    assert done.wait(20), "cleanup never finished"
    closer.join(timeout=5)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert "cancel after close" not in device.calls, (
        f"handed librtlsdr a pointer that had been freed: {device.calls}")
    assert device.calls.count("close") == 1, device.calls
    assert "read" not in device.calls[device.calls.index("close"):], (
        f"a read started after the close: {device.calls}")


def test_a_close_cannot_slip_in_while_the_c_call_is_being_made(receiver,
                                                                monkeypatch):
    """The gap the liveness check leaves if the call is outside the lock.

    The check says the handle is alive; a failing write closes it; the
    call then hands librtlsdr a pointer to freed memory.  Held here with
    events, right at the moment the C function would be entered.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    thread = running_reader(receiver)

    at_the_call = threading.Event()
    close_done = threading.Event()
    real_raw = device_handle.RAW_CANCEL_ASYNC

    def stalled_raw(dev_p):
        at_the_call.set()
        close_done.wait(1.0)        # bounded: with the lock held it expires
        return real_raw(dev_p)

    monkeypatch.setattr("fm_radio.device_handle.RAW_CANCEL_ASYNC", stalled_raw)

    def close_from_a_failing_write():
        if not at_the_call.wait(10):
            return
        device.calls.append("gain write failed")
        device.close()              # rtlsdr.py:317
        close_done.set()

    closer = threading.Thread(target=close_from_a_failing_write, daemon=True)
    closer.start()

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()

    assert done.wait(20), "cleanup never finished"
    closer.join(timeout=10)
    device.cancelled.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert "cancel after close" not in device.calls, (
        f"handed librtlsdr a pointer that had been freed: {device.calls}")


def test_a_close_from_a_failed_read_waits_for_a_cancel_in_flight(receiver,
                                                                 monkeypatch):
    """The one interleaving only handle_lock covers.

    A read that fails closes the device from the reader thread
    (rtlsdr.py:601-603), and that close does not wait for the read to end -
    the read has ended, that is why pyrtlsdr is closing.  So the guard
    that holds every other close back is not in play here.  Meanwhile a
    cancel from another thread has satisfied itself that the handle is
    alive and is inside librtlsdr with the pointer.  handle_lock is what
    keeps the close out until that call returns.
    """
    sdr = receiver.sdr_receiver
    handle = sdr.handle
    device = sdr.sdr

    at_the_call = threading.Event()
    cancel_done = threading.Event()
    real_raw = device_handle.RAW_CANCEL_ASYNC

    def stalled_raw(dev_p):
        at_the_call.set()
        cancel_done.wait(1.0)       # bounded: with the lock held it expires
        return real_raw(dev_p)

    monkeypatch.setattr("fm_radio.device_handle.RAW_CANCEL_ASYNC", stalled_raw)

    reading = threading.Event()
    fail_now = threading.Event()

    def failing_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        fail_now.wait(10)
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE
        device.calls.append("read failed")
        device.close()              # rtlsdr.py:601-603, on this thread
        raise OSError("LIBUSB_ERROR_IO: could not read")

    monkeypatch.setattr(device, "read_samples_async", failing_read)

    def read_and_swallow():
        try:
            sdr.start()
        except SDRDeviceError:
            pass                    # the point is the close, not the raise

    reader = threading.Thread(target=read_and_swallow, name="SDRThread",
                              daemon=True)
    reader.start()
    assert reading.wait(5), "the read never started"

    canceller = threading.Thread(target=handle.stop_sampling, daemon=True)
    canceller.start()
    assert at_the_call.wait(5), "the cancel never reached librtlsdr"

    fail_now.set()                  # the read fails and asks to close
    reader.join(timeout=15)
    canceller.join(timeout=15)

    assert not reader.is_alive() and not canceller.is_alive()
    assert "cancel after close" not in device.calls, (
        f"freed the handle while librtlsdr had the pointer: {device.calls}")
    assert device.calls.count("close") == 1, device.calls
    assert not device.device_opened


def test_a_driver_side_close_is_not_closed_again(receiver, monkeypatch):
    """One owner means one close, whoever asked for it first."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    thread = running_reader(receiver)

    def failing_write(gain):
        device.close()
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)
    with pytest.raises(SDRDeviceError):
        sdr.set_gain(30.0)
    assert not device.device_opened

    device.cancelled.set()
    receiver.cleanup()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert device.calls.count("close") == 1, device.calls
    assert "cancel after close" not in device.calls, device.calls


# ----------------------------------------------------------------------
# A close asked for while librtlsdr is still reading
# ----------------------------------------------------------------------

def test_a_failing_write_does_not_close_under_a_running_read(receiver,
                                                             monkeypatch):
    """pyrtlsdr closes the device from inside a failed control write.

    librtlsdr is using that handle for as long as rtlsdr_read_async has
    not returned, and a gain write failing is no reason to pull it out
    from under the read.  The close cancels the read, waits for it, and
    only then frees the handle.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    reader = running_reader(receiver)

    def failing_write(gain):
        device.calls.append("gain write failed")
        device.close()                  # rtlsdr.py:317
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)

    with pytest.raises(SDRDeviceError):
        sdr.set_gain(30.0)

    reader.join(timeout=10)

    assert not reader.is_alive(), "the read never ended"
    assert "close during read" not in device.calls, device.calls
    assert significant(device.calls) == [
        "read", "gain write failed", "cancel", "close"], device.calls
    assert sdr.closed, "the receiver still looks open"


def test_a_read_that_will_not_end_defers_the_close(receiver, monkeypatch):
    """Waiting for the read has to be bounded like everything else."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    reading = threading.Event()
    release = threading.Event()

    def deaf_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        release.wait(30)                # ignores every cancel
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE

    monkeypatch.setattr(device, "read_samples_async", deaf_read)
    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert reading.wait(5)

    def failing_write(gain):
        device.calls.append("gain write failed")
        device.close()
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)

    try:
        started = time.monotonic()
        with pytest.raises(SDRDeviceError):
            sdr.set_gain(30.0)
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"the close waited {elapsed:.1f} s"
        assert "close" not in device.calls, device.calls
        assert device.device_opened
        assert sdr.handle.close_pending, "the close was forgotten rather than kept"
    finally:
        release.set()
        thread.join(timeout=10)


# ----------------------------------------------------------------------
# A close that could not be made when it was asked for
# ----------------------------------------------------------------------

def test_a_deferred_close_is_made_when_the_write_lets_go(receiver,
                                                         monkeypatch):
    """The read fails and asks to close; a write on another thread has the
    device.  The close is kept, not dropped, and made when the lock frees.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    reading = threading.Event()
    fail_now = threading.Event()

    def failing_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        fail_now.wait(10)
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE
        device.calls.append("read failed")
        device.close()                  # rtlsdr.py:601-603
        raise OSError("LIBUSB_ERROR_IO: could not read")

    writing = threading.Event()
    release = threading.Event()
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(10)
        real_set_gain(gain)

    monkeypatch.setattr(device, "read_samples_async", failing_read)
    monkeypatch.setattr(device, "set_gain", slow_write)

    def read_and_swallow():
        try:
            sdr.start()
        except SDRDeviceError:
            pass

    thread = threading.Thread(target=read_and_swallow, name="SDRThread",
                              daemon=True)
    thread.start()
    assert reading.wait(5), "the read never started"

    writer = threading.Thread(target=lambda: sdr.set_gain(30.0), daemon=True)
    writer.start()
    assert writing.wait(5), "the write never started"

    fail_now.set()
    thread.join(timeout=10)             # the reader gives up on the close

    assert not thread.is_alive()
    assert device.device_opened, "closed while a write was still inside it"
    assert sdr.handle.close_pending, "the close request was dropped"
    assert sdr.closed, "a failed read left the receiver looking open"

    release.set()                       # the write finishes
    writer.join(timeout=10)

    assert not writer.is_alive()
    assert not device.device_opened, "the deferred close was never made"
    assert device.calls.count("close") == 1, device.calls
    assert "close during read" not in device.calls, device.calls


def test_a_deferred_close_is_made_by_stop_if_nothing_else_does(receiver,
                                                               monkeypatch):
    """No further writes come, so the last chance is cleanup."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    release = threading.Event()
    holding = threading.Event()

    def hold_the_device():
        sdr.handle.device_lock.acquire()
        holding.set()
        release.wait(10)
        sdr.handle.device_lock.release()

    holder = threading.Thread(target=hold_the_device, daemon=True)
    holder.start()
    assert holding.wait(5)

    device.close()                      # deferred: the lock is held
    assert sdr.handle.close_pending
    assert device.device_opened

    release.set()
    holder.join(timeout=5)

    receiver.cleanup()

    assert not device.device_opened, "cleanup left a deferred close unmade"
    assert device.calls.count("close") == 1, device.calls


def test_a_deferred_close_is_made_only_once(receiver, monkeypatch):
    """Retried from several places, but the handle is freed one time."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    release = threading.Event()
    holding = threading.Event()

    def hold_the_device():
        sdr.handle.device_lock.acquire()
        holding.set()
        release.wait(10)
        sdr.handle.device_lock.release()

    holder = threading.Thread(target=hold_the_device, daemon=True)
    holder.start()
    assert holding.wait(5)

    device.close()
    device.close()                      # asked twice, deferred twice
    assert sdr.handle.close_pending

    release.set()
    holder.join(timeout=5)

    sdr.handle.retry_pending_close()
    sdr.handle.retry_pending_close()
    receiver.cleanup()
    receiver.cleanup()

    assert device.calls.count("close") == 1, device.calls


# ----------------------------------------------------------------------
# The receiver's own state, on every close path
# ----------------------------------------------------------------------

def test_a_close_before_any_read_still_closes_the_receiver(receiver,
                                                           monkeypatch):
    """A write can fail before the read has ever started.

    pyrtlsdr closes the device from inside it, and the receiver has to
    know: the handle is gone, so a later gain or frequency read must come
    from the cache rather than from a device that is not there.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr

    def failing_write(gain):
        device.close()                  # rtlsdr.py:317
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)

    with pytest.raises(SDRDeviceError):
        sdr.set_gain(30.0)

    assert not device.device_opened
    assert sdr.closed, "the receiver still reports itself open"

    device.calls.clear()
    monkeypatch.setattr(device, "set_gain", lambda gain: None)
    sdr.get_gain()
    sdr.get_center_frequency()
    sdr.set_gain(20.0)
    sdr.set_center_frequency(81.3e6)
    sdr.set_manual_gain_mode(True)

    assert device.calls == [], (
        f"used a handle that had been freed: {device.calls}")


def test_a_deferred_close_leaves_the_receiver_closed_too(receiver,
                                                         monkeypatch):
    """Even when the close itself has to wait, nothing new may reach it."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    release = threading.Event()
    holding = threading.Event()

    def hold_the_device():
        sdr.handle.device_lock.acquire()
        holding.set()
        release.wait(10)
        sdr.handle.device_lock.release()

    holder = threading.Thread(target=hold_the_device, daemon=True)
    holder.start()
    assert holding.wait(5)

    try:
        device.close()                  # deferred: the lock is held

        assert sdr.handle.close_pending
        assert device.device_opened
        assert sdr.closed, "a pending close left the receiver looking open"
    finally:
        release.set()
        holder.join(timeout=5)


# ----------------------------------------------------------------------
# The read ending is itself a chance to close
# ----------------------------------------------------------------------

def test_the_read_ending_makes_good_on_a_deferred_close(receiver,
                                                        monkeypatch):
    """The close was deferred because the read would not end.

    When it does end, that is the moment the close became possible, and
    nothing else is going to come along: writes are refused by then.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    reading = threading.Event()
    release = threading.Event()

    def deaf_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        release.wait(30)                # ignores every cancel
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE

    monkeypatch.setattr(device, "read_samples_async", deaf_read)
    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert reading.wait(5), "the read never started"

    def failing_write(gain):
        device.close()
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)
    with pytest.raises(SDRDeviceError):
        sdr.set_gain(30.0)

    assert sdr.handle.close_pending, "the close should have been deferred"
    assert device.device_opened

    release.set()                       # ... and now the read ends
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not sdr.handle.close_pending, "the deferred close was forgotten"
    assert not device.device_opened, "the deferred close was never made"
    assert device.calls.count("close") == 1, device.calls
    assert "close during read" not in device.calls, device.calls


def test_a_refused_write_still_makes_good_on_a_deferred_close(receiver,
                                                              monkeypatch):
    """Writes are refused once closing has begun, and a refusal is a
    perfectly good moment to try the close that could not be made."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)

    release = threading.Event()
    holding = threading.Event()

    def hold_the_device():
        sdr.handle.device_lock.acquire()
        holding.set()
        release.wait(10)
        sdr.handle.device_lock.release()

    holder = threading.Thread(target=hold_the_device, daemon=True)
    holder.start()
    assert holding.wait(5)

    device.close()                      # deferred
    assert sdr.handle.close_pending

    release.set()
    holder.join(timeout=5)

    sdr.set_gain(20.0)                  # refused, and retries the close

    assert not sdr.handle.close_pending
    assert not device.device_opened
    assert device.calls == ["close"], device.calls


def test_a_failing_write_retries_a_close_it_could_not_make(receiver,
                                                           monkeypatch):
    """The exception path is an exit too, and the read ends behind it."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    reading = threading.Event()
    release = threading.Event()
    ended = threading.Event()

    def deaf_read(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        release.wait(30)
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE
        ended.set()

    monkeypatch.setattr(device, "read_samples_async", deaf_read)
    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert reading.wait(5)

    def failing_write(gain):
        device.close()                  # deferred: the read will not end
        raise OSError("LIBUSB_ERROR_NO_DEVICE")

    monkeypatch.setattr(device, "set_gain", failing_write)
    with pytest.raises(SDRDeviceError):
        sdr.set_gain(30.0)
    assert sdr.handle.close_pending

    release.set()
    thread.join(timeout=10)
    assert ended.is_set()

    assert not device.device_opened
    assert device.calls.count("close") == 1, device.calls
    assert "close during read" not in device.calls, device.calls


def test_cleanup_finishes_after_a_deferred_close(receiver, monkeypatch):
    """Whatever happened above, shutdown still has to end."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 0.3)
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    release = threading.Event()
    holding = threading.Event()

    def hold_the_device():
        sdr.handle.device_lock.acquire()
        holding.set()
        release.wait(10)
        sdr.handle.device_lock.release()

    holder = threading.Thread(target=hold_the_device, daemon=True)
    holder.start()
    assert holding.wait(5)

    device.close()                      # deferred
    assert sdr.handle.close_pending

    try:
        started = time.monotonic()
        receiver.cleanup()              # the lock is still held
        elapsed = time.monotonic() - started

        assert elapsed < 15.0, f"cleanup took {elapsed:.1f} s"
        assert device.device_opened, "closed underneath the lock holder"
    finally:
        release.set()
        holder.join(timeout=5)

    receiver.cleanup()                  # now it can be made
    assert not device.device_opened
    assert device.calls.count("close") == 1, device.calls


# ----------------------------------------------------------------------
# What the window feels while a close is pending
# ----------------------------------------------------------------------

@pytest.fixture
def write_in_flight(receiver, monkeypatch):
    """A gain write holding the device, with a deferred close behind it."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    # Long enough that a waiting exit path would be unmistakable, short
    # enough that the test is not slow if one creeps back in.
    monkeypatch.setattr("fm_radio.device_handle.DEVICE_LOCK_TIMEOUT_SEC", 1.0)

    writing = threading.Event()
    release = threading.Event()
    real_set_gain = device.set_gain

    def slow_write(gain):
        writing.set()
        release.wait(30)
        real_set_gain(gain)

    monkeypatch.setattr(device, "set_gain", slow_write)
    writer = threading.Thread(target=lambda: sdr.set_gain(30.0), daemon=True)
    writer.start()
    assert writing.wait(5), "the write never started"

    device.close()                      # deferred: the write has the device
    assert sdr.handle.close_pending
    assert device.device_opened

    yield sdr, device, release, writer

    release.set()
    writer.join(timeout=10)


def test_operations_refused_during_a_pending_close_return_at_once(
        write_in_flight):
    """These are what the window calls, and it calls them on its own thread.

    The operation is refused - the receiver is closing - and the exit path
    tries the deferred close.  It must not wait for it: the device lock is
    held by a write that has nothing to do with this call, and a window
    that blocks on one is a window that has stopped answering.
    """
    sdr, device, _release, _writer = write_in_flight

    operations = [
        ("tune", lambda: sdr.set_center_frequency(81.3e6)),
        ("AGC toggle", lambda: sdr.set_manual_gain_mode(True)),
        ("manual gain", lambda: sdr.set_manual_gain_mode(False)),
        ("tune", lambda: sdr.set_center_frequency(82.5e6)),
        ("gain", lambda: sdr.set_gain(20.0)),
        ("read the gain", sdr.get_gain),
        ("read the frequency", sdr.get_center_frequency),
    ]
    slowest = 0.0
    for name, call in operations:
        started = time.monotonic()
        call()
        took = time.monotonic() - started
        assert took < 0.25, f"a refused {name} took {took:.3f} s"
        slowest = max(slowest, took)

    assert device.device_opened, "closed underneath the write"
    assert sdr.handle.close_pending, "the close should still be waiting its turn"


def test_a_repeated_tune_and_agc_toggle_never_queue_behind_the_write(
        write_in_flight):
    """The window can be clicked faster than one call per second."""
    sdr, device, _release, _writer = write_in_flight

    started = time.monotonic()
    for i in range(20):
        sdr.set_center_frequency(80e6 + i * 1e5)
        sdr.set_manual_gain_mode(i % 2 == 0)
    took = time.monotonic() - started

    assert took < 0.5, f"40 refused operations took {took:.3f} s"
    assert device.calls == [], f"reached the device while closing: {device.calls}"


def test_the_close_lands_when_the_write_lets_go(write_in_flight):
    """Not waiting is only safe if somebody still makes the close."""
    sdr, device, release, writer = write_in_flight

    sdr.set_center_frequency(81.3e6)    # refused, and does not wait
    assert sdr.handle.close_pending

    release.set()                       # the write finishes
    writer.join(timeout=10)

    assert not writer.is_alive()
    assert not sdr.handle.close_pending, "the deferred close was forgotten"
    assert not device.device_opened, "the deferred close was never made"
    assert device.calls.count("close") == 1, device.calls


def test_the_close_lands_once_however_many_operations_come_past(
        write_in_flight):
    """Every exit path tries it; the handle is still freed one time."""
    sdr, device, release, writer = write_in_flight

    release.set()
    writer.join(timeout=10)

    for i in range(10):
        sdr.set_center_frequency(80e6 + i * 1e5)
        sdr.set_manual_gain_mode(True)
        sdr.set_gain(20.0)

    assert device.calls.count("close") == 1, device.calls


def test_cleanup_is_bounded_with_a_close_still_pending(write_in_flight):
    """The write is still holding on, and shutdown still has to end."""
    sdr, device, release, writer = write_in_flight

    started = time.monotonic()
    sdr.stop()
    took = time.monotonic() - started

    assert took < 10.0, f"stop took {took:.1f} s"
    assert device.device_opened, "closed underneath the write"

    release.set()
    writer.join(timeout=10)

    assert not device.device_opened, "nothing made good on the close"
    assert device.calls.count("close") == 1, device.calls


# ----------------------------------------------------------------------
# A read that ends a moment too late
# ----------------------------------------------------------------------

@pytest.fixture
def read_that_ignores_cancels(receiver, monkeypatch):
    """A read that ignores every cancel until the test lets it go."""
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    monkeypatch.setattr("fm_radio.device_handle.SAMPLING_CANCEL_TIMEOUT_SEC",
                        0.3)

    reading = threading.Event()
    release = threading.Event()

    def ignores_the_cancel(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        reading.set()
        release.wait(30)
        device.reading.clear()
        device.async_status = RTLSDR_INACTIVE

    monkeypatch.setattr(device, "read_samples_async", ignores_the_cancel)
    thread = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    thread.start()
    assert reading.wait(5), "the read never started"

    yield sdr, device, release, thread

    release.set()
    thread.join(timeout=10)


def test_a_read_that_ends_after_stop_gave_up_is_still_closed(read_that_ignores_cancels):
    """Shutdown ran out of patience; the read finished a moment later.

    The close could not be made at the time and must not simply be
    dropped: the read ending is precisely the moment it became possible,
    and by then nothing else is coming - writes are refused and stop has
    already been and gone.
    """
    sdr, device, release, thread = read_that_ignores_cancels

    sdr.stop()

    assert sdr.handle.close_pending, "the close request was dropped"
    assert device.device_opened, "closed the handle the read was using"
    assert "close" not in device.calls, device.calls

    release.set()                       # ... and now the read ends
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not sdr.handle.close_pending, "the deferred close was forgotten"
    assert not device.device_opened, "the deferred close was never made"
    assert device.calls.count("close") == 1, device.calls
    assert "close during read" not in device.calls, device.calls


def test_a_read_that_never_ends_keeps_the_handle_open(read_that_ignores_cancels):
    """The other half of the bargain: no read, no close.

    Keeping the request must not turn into making it anyway.  A handle
    librtlsdr is still reading through outlives the process instead.
    """
    sdr, device, _release, thread = read_that_ignores_cancels

    started = time.monotonic()
    sdr.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"stop took {elapsed:.1f} s"
    assert sdr.handle.close_pending
    assert device.device_opened
    assert "close" not in device.calls, device.calls
    assert thread.is_alive(), "the read was supposed to still be going"

    # Every retry point in turn, while the read is still running.
    sdr.handle.retry_pending_close()
    sdr.set_gain(20.0)
    sdr.set_center_frequency(81.3e6)
    sdr.stop()

    assert device.device_opened, "closed underneath a running read"
    assert "close" not in device.calls, device.calls


def test_the_close_after_a_late_read_happens_once(read_that_ignores_cancels):
    """Several retry points fire as the read unwinds; one close."""
    sdr, device, release, thread = read_that_ignores_cancels

    sdr.stop()
    sdr.stop()
    assert sdr.handle.close_pending

    release.set()
    thread.join(timeout=10)

    sdr.handle.retry_pending_close()
    sdr.set_gain(20.0)
    sdr.stop()

    assert device.calls.count("close") == 1, device.calls


# ----------------------------------------------------------------------
# A stop asked for from inside the read
# ----------------------------------------------------------------------

@pytest.fixture
def read_with_a_callback(receiver, monkeypatch):
    """A read that calls back on its own thread, as librtlsdr does.

    rtlsdr_read_async invokes the callback from the thread that called it
    and keeps running, so anything the callback does happens while the
    read is still going - and the read cannot end until the callback
    returns.
    """
    sdr = receiver.sdr_receiver
    device = sdr.sdr
    in_the_callback = threading.Event()
    from_the_callback: list = []

    def read_and_call_back(cb, num_samples=None):
        device.calls.append("read")
        device.async_status = RTLSDR_RUNNING
        device.reading.set()
        try:
            cb(np.zeros(8, dtype=np.complex64), device)
            in_the_callback.set()
            for call in from_the_callback:
                call()
            device.cancelled.wait(10)   # the read ends only on a cancel
        finally:
            device.reading.clear()
            device.async_status = RTLSDR_INACTIVE

    monkeypatch.setattr(device, "read_samples_async", read_and_call_back)

    def start(*calls):
        from_the_callback.extend(calls)
        thread = threading.Thread(target=sdr.start, name="SDRThread",
                                  daemon=True)
        thread.start()
        return thread

    yield sdr, device, start, in_the_callback

    device.cancelled.set()


def test_a_stop_from_the_callback_does_not_close_under_the_read(
        read_with_a_callback):
    """The callback is on the reading thread, and the read is still going.

    rtlsdr_close waits for the async read to finish, and the read cannot
    finish while its own callback is inside the close - so closing here
    would be the callback waiting for itself.  The cancel goes out, the
    close waits for the read to return, and the read returning is what
    makes it.
    """
    sdr, device, start, in_the_callback = read_with_a_callback

    thread = start(sdr.stop)
    assert in_the_callback.wait(5), "the callback never ran"
    thread.join(timeout=10)

    assert not thread.is_alive(), "the read never returned"
    assert "close during read" not in device.calls, device.calls
    assert significant(device.calls) == ["read", "cancel", "close"], device.calls
    assert not device.device_opened
    assert not sdr.handle.close_pending


def test_a_stop_from_the_callback_returns_without_waiting(
        read_with_a_callback):
    """It is the read's own thread: waiting would be waiting for itself."""
    sdr, device, start, in_the_callback = read_with_a_callback
    took: list[float] = []

    def timed_stop():
        started = time.monotonic()
        sdr.stop()
        took.append(time.monotonic() - started)

    thread = start(timed_stop)
    assert in_the_callback.wait(5)
    thread.join(timeout=10)

    assert took and took[0] < 1.0, f"stop() from the callback took {took}"
    assert not thread.is_alive()


def test_a_stop_from_the_callback_closes_once_however_often_it_is_asked(
        read_with_a_callback):
    """Twice from the callback, then again from outside."""
    sdr, device, start, in_the_callback = read_with_a_callback

    thread = start(sdr.stop, sdr.stop)
    assert in_the_callback.wait(5)
    thread.join(timeout=10)

    sdr.stop()
    sdr.handle.retry_pending_close()

    assert device.calls.count("close") == 1, device.calls
    assert "close during read" not in device.calls, device.calls


# ----------------------------------------------------------------------
# A recording being closed while shutdown goes past
# ----------------------------------------------------------------------

def test_cleanup_waits_for_a_recording_that_is_still_closing(receiver,
                                                             tmp_path):
    """Tuning ends a recording; shutdown must not overtake the ending.

    stop_recording clears its flag first and then flushes, closes the
    file and writes the sidecar.  cleanup asked the flag, saw no
    recording, and came back with the file still open - the process then
    went, leaving a WAV nothing had finished.
    """
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "left_open.wav"))
    audio.record(np.zeros((2048, 2), dtype=np.float32))

    closing = threading.Event()
    finish = threading.Event()
    real_wait = audio._flush_event.wait

    def slow_flush(timeout=None):
        closing.set()
        finish.wait(10)
        return real_wait(0)

    audio._flush_event.wait = slow_flush
    threading.Thread(target=audio.stop_recording, daemon=True).start()
    assert closing.wait(5), "the recording never started closing"
    assert not audio.recording, "the flag is meant to be down by now"
    assert audio.finalising, "and the file still open"

    done = threading.Event()
    threading.Thread(target=lambda: (receiver.cleanup(), done.set()),
                     daemon=True).start()
    time.sleep(0.3)

    assert not done.is_set(), "cleanup went past a file that was still open"

    finish.set()

    assert done.wait(20), "cleanup never finished"
    assert audio.record_wave is None, "the file was left open"


def test_cleanup_does_not_wait_for_ever_on_a_recording(receiver, tmp_path,
                                                       monkeypatch):
    """Bounded, like every other wait on the way out."""
    monkeypatch.setattr("fm_radio.audio_output._RECORDING_CLOSE_TIMEOUT_SEC",
                        0.3)
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "stuck.wav"))

    stuck = threading.Event()
    audio._finalising.set()             # as though a close were under way

    try:
        started = time.monotonic()
        receiver.cleanup()
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"cleanup took {elapsed:.1f} s"
    finally:
        stuck.set()
        audio._finalising.clear()


def test_tuning_does_not_put_the_recording_close_on_the_device_worker(
        receiver, tmp_path):
    """Fifteen seconds of flush is not the device worker's to spend.

    Every other write queues behind it, and the gain the AGC wants next
    is not worth a quarter of a minute.
    """
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "while_tuning.wav"))

    closing = threading.Event()
    finish = threading.Event()
    real_wait = audio._flush_event.wait

    def slow_flush(timeout=None):
        closing.set()
        finish.wait(10)
        return real_wait(0)

    audio._flush_event.wait = slow_flush

    try:
        tuned = receiver.tune(81.3e6)
        assert tuned.wait(5), "the tune waited for the recording to close"
        assert closing.wait(5), "the recording was never closed"

        # The worker is free while that goes on.
        after = receiver.set_gain(20.0)
        if after is not None:
            assert after.wait(5), "a later write queued behind the recording"
    finally:
        finish.set()


def a_close_that_will_not_finish(audio):
    """Hold a recording open in its flush, and hand back the release.

    Everything after the flag goes down - the flush, the close, the
    sidecar - is the slow half, and these tests are about what has to
    have happened before it.
    """
    closing = threading.Event()
    finish = threading.Event()
    real_wait = audio._flush_event.wait

    def slow_flush(timeout=None):
        closing.set()
        finish.wait(10)
        return real_wait(0)

    audio._flush_event.wait = slow_flush
    return closing, finish


def test_tuning_shuts_the_recordings_before_the_frequency_moves(receiver,
                                                                tmp_path):
    """The door shuts first; the file is finished afterwards.

    The SDR starts delivering the new station the moment the write
    lands, so a recording that is still taking samples then takes some
    of them - into a file whose name and sidecar say the old station.
    """
    audio = receiver.audio_output
    sdr = receiver.sdr_receiver
    audio.start_recording(str(tmp_path / "before.wav"))
    sdr.start_iq_recording(str(tmp_path / "before_iq.wav"))
    closing, finish = a_close_that_will_not_finish(audio)

    when_the_write_went = {}
    real_set = sdr.set_center_frequency

    def watched(freq_hz):
        when_the_write_went["audio"] = audio.recording
        when_the_write_went["iq"] = sdr.iq_recording
        return real_set(freq_hz)

    sdr.set_center_frequency = watched

    try:
        tuned = receiver.tune(81.3e6)
        assert tuned.wait(5), "the tune never landed"

        assert when_the_write_went == {"audio": False, "iq": False},             when_the_write_went
        assert closing.wait(5), "the slow half never started"
        assert audio.finalising, "the file is meant to still be closing"
    finally:
        finish.set()
        sdr.set_center_frequency = real_set


def test_a_block_of_the_new_station_misses_the_old_recording(receiver,
                                                             tmp_path):
    """And the file proves it: only what was offered before the tune.

    The thread that finishes the file is held at its first line for the
    whole of this, which is the window the old order left open: the
    write has landed, the SDR is on the new station, and nothing has
    closed the door on the old station's file yet.
    """
    audio = receiver.audio_output
    path = tmp_path / "one_station.wav"
    audio.start_recording(str(path))
    audio.record(np.zeros((2048, 2), dtype=np.float32))

    let_it_close = threading.Event()
    real_close = receiver._close_the_recordings_now

    def held_close(*taken):
        assert let_it_close.wait(10), "the test never let the close start"
        return real_close(*taken)

    receiver._close_the_recordings_now = held_close

    try:
        tuned = receiver.tune(81.3e6)
        assert tuned.wait(5), "the tune never landed"

        # A block of the new station, offered the way the processing
        # thread offers one.  The door is shut and this has nowhere to
        # go, even though the file is still open behind it.
        audio.record(np.ones((2048, 2), dtype=np.float32))
    finally:
        let_it_close.set()

    assert audio.wait_for_the_recording_to_close(20),         "the recording never finished closing"
    with wave.open(str(path), "rb") as f:
        assert f.getnframes() == 2048,             "the new station is in the old station's file"


def test_a_tune_does_not_wait_for_the_recording_to_finish(receiver, tmp_path):
    """The fast half is a flag; the slow half is somebody else's thread."""
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "not_waited_for.wav"))
    closing, finish = a_close_that_will_not_finish(audio)

    try:
        tuned = receiver.tune(81.3e6)

        assert tuned.wait(5), "the tune waited for the recording to close"
        assert closing.wait(5), "the slow half never started"
        assert audio.finalising, "it was supposed to still be going"

        after = receiver.set_gain(20.0)
        if after is not None:
            assert after.wait(5), "a later write queued behind the recording"
    finally:
        finish.set()


def a_start_that_is_half_way(receiver, what):
    """Hold a recording between its metadata and its file being installed.

    The window the tune must not fit through: the frequency has been
    read for the sidecar, and nothing is yet calling itself a recording.
    """
    inside = threading.Event()
    go = threading.Event()
    if what == "audio":
        owner, name = receiver.audio_output, "start_recording"
    else:
        owner, name = receiver.sdr_receiver, "start_iq_recording"
    real = getattr(owner, name)

    def slow_start(filename, **kw):
        inside.set()
        assert go.wait(10), "the test never let the recording start"
        return real(filename, **kw)

    setattr(owner, name, slow_start)
    return inside, go


def test_a_tune_waits_for_a_recording_that_is_half_started(receiver, tmp_path):
    """Otherwise the file is named for a station it does not contain.

    Starting a recording reads the frequency for the sidecar, opens the
    file and only then calls itself a recording.  A tune that went
    through the middle of that found no recording to stop, moved, and
    left the new station being written into the old station's file.
    """
    audio = receiver.audio_output
    sdr = receiver.sdr_receiver
    was_on = sdr.get_center_frequency()
    inside, go = a_start_that_is_half_way(receiver, "audio")

    path = tmp_path / "half_started.wav"
    starting = receiver.start_recording(str(path))
    assert inside.wait(5), "the recording never started starting"

    when_the_write_went = {}
    real_set = sdr.set_center_frequency

    def watched(freq_hz):
        when_the_write_went["recording"] = audio.recording
        return real_set(freq_hz)

    sdr.set_center_frequency = watched

    try:
        tuned = receiver.tune(81.3e6)
        assert not tuned.wait(0.3), \
            "the tune went past a recording that was being installed"

        go.set()
        assert starting.wait(5), "the recording never finished starting"
        assert tuned.wait(5), "the tune never landed"

        # It started, and then the tune stopped it - in that order, and
        # before the frequency moved.  So the file is short, and what is
        # in it is the station its sidecar names.
        assert when_the_write_went == {"recording": False}, when_the_write_went
        assert audio._record_meta["center_freq_hz"] == pytest.approx(was_on)
    finally:
        sdr.set_center_frequency = real_set
        go.set()


def test_a_tune_waits_for_an_iq_recording_that_is_half_started(receiver,
                                                               tmp_path):
    """The IQ sidecar names a centre frequency too."""
    sdr = receiver.sdr_receiver
    inside, go = a_start_that_is_half_way(receiver, "iq")

    path = tmp_path / "half_started_iq.wav"
    starting = receiver.start_iq_recording(str(path))
    assert inside.wait(5), "the recording never started starting"

    when_the_write_went = {}
    real_set = sdr.set_center_frequency

    def watched(freq_hz):
        when_the_write_went["iq_recording"] = sdr.iq_recording
        return real_set(freq_hz)

    sdr.set_center_frequency = watched

    try:
        tuned = receiver.tune(81.3e6)
        assert not tuned.wait(0.3), \
            "the tune went past an IQ recording that was being installed"

        go.set()
        assert starting.wait(5), "the recording never finished starting"
        assert tuned.wait(5), "the tune never landed"

        assert when_the_write_went == {"iq_recording": False}, \
            when_the_write_went
    finally:
        sdr.set_center_frequency = real_set
        go.set()


def test_asking_for_a_recording_does_not_wait_for_the_tuner(
        receiver, tmp_path):
    """The whole point: the button comes back, whatever the tuner is doing.

    On a device that has stopped answering the frequency write never
    returns.  The recording is asked of the same worker, so it happens
    after the write - but the asking is over at once, and the thread
    that asked is the one drawing the window.
    """
    sdr = receiver.sdr_receiver
    stuck = threading.Event()
    writing = threading.Event()
    real_set = sdr.set_center_frequency

    def never_answers(freq_hz):
        writing.set()
        stuck.wait(10)
        return real_set(freq_hz)

    sdr.set_center_frequency = never_answers

    try:
        receiver.tune(81.3e6)
        assert writing.wait(5), "the write never started"

        started = time.monotonic()
        asked = receiver.start_recording(str(tmp_path / "later.wav"))
        waited = time.monotonic() - started

        assert waited < 1.0, f"the button was held for {waited:.1f} s"
        assert not asked.finished, "it cannot have happened yet"
        assert not receiver.audio_output.recording

        stuck.set()
        assert asked.wait(5), "the recording never started"
        assert receiver.audio_output.recording
    finally:
        stuck.set()
        sdr.set_center_frequency = real_set


def test_a_tune_that_fails_half_way_still_finishes_what_it_took(receiver,
                                                                tmp_path):
    """The debt is written down as it is taken, not after it all succeeds.

    The audio recording has been stopped and its file is open when the
    IQ side blows up.  If that skipped the finish, the WAV would be left
    for shutdown to wait twenty seconds for and then abandon.
    """
    audio = receiver.audio_output
    path = tmp_path / "half_way.wav"
    audio.start_recording(str(path))
    audio.record(np.zeros((2048, 2), dtype=np.float32))

    def blows_up():
        raise RuntimeError("the IQ side blew up")

    receiver.sdr_receiver.begin_stopping_the_iq_recording = blows_up

    asked = receiver.tune(81.3e6)
    assert asked.wait(5), "the tune never answered"
    assert asked.failed, "the tune was supposed to fail"

    assert audio.wait_for_the_recording_to_close(10), \
        "the recording was left half closed"
    assert audio.record_wave is None, "the file was left open"
    with wave.open(str(path), "rb") as f:
        assert f.getnframes() == 2048


def test_a_frequency_write_that_fails_still_finishes_what_it_took(receiver,
                                                                   tmp_path):
    """The same, one line later."""
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "write_failed.wav"))

    def blows_up(freq_hz):
        raise SDRDeviceError("the write blew up")

    receiver.sdr_receiver.set_center_frequency = blows_up

    asked = receiver.tune(81.3e6)
    assert asked.wait(5), "the tune never answered"
    assert asked.failed, "the tune was supposed to fail"

    assert audio.wait_for_the_recording_to_close(10), \
        "the recording was left half closed"
    assert audio.record_wave is None, "the file was left open"


def test_a_recording_is_named_for_the_station_it_will_contain(
        receiver, tmp_path, monkeypatch):
    """Not for the one the button was pressed on.

    The name claims a station as much as the sidecar does, and a tune
    can be in front of the recording on the worker.  Both are decided
    there, after the tune, so all three agree.
    """
    monkeypatch.chdir(tmp_path)
    sdr = receiver.sdr_receiver
    let_the_write_land = threading.Event()
    writing = threading.Event()
    real_set = sdr.set_center_frequency

    def slow_write(freq_hz):
        writing.set()
        assert let_the_write_land.wait(10), "the test never let it land"
        return real_set(freq_hz)

    sdr.set_center_frequency = slow_write

    try:
        tuned = receiver.tune(81.3e6)
        assert writing.wait(5), "the write never started"

        # Asked for while the receiver is still on 80.0, by a caller
        # that names nothing.
        asked = receiver.start_recording()
        let_the_write_land.set()

        assert tuned.wait(5) and asked.wait(5), "nothing landed"
        assert not asked.failed, asked.error
    finally:
        let_the_write_land.set()
        sdr.set_center_frequency = real_set

    assert "81.3MHz" in asked.result, asked.result
    assert receiver.audio_output._record_meta["center_freq_hz"] == \
        pytest.approx(81.3e6)


def a_close_held_at_the_door(receiver):
    """Hold the thread that finishes a recording, and hand back the key."""
    let_it_close = threading.Event()
    real_close = receiver._close_the_recordings_now

    def held_close(*shut):
        assert let_it_close.wait(10), "the test never let the close start"
        return real_close(*shut)

    receiver._close_the_recordings_now = held_close
    return let_it_close


def a_start_that_says_when_it_asks(owner, name):
    """Note when a start asks whether the last recording is still closing.

    That question is the whole fix, and it is asked before the new file
    is opened - so a test that waits for it knows the starter is still
    outside, whatever the scheduler is doing.
    """
    asked = threading.Event()
    real = getattr(owner, name)

    def watched(timeout):
        asked.set()
        return real(timeout)

    setattr(owner, name, watched)
    return asked


def test_a_recording_that_is_closing_does_not_close_the_next_one(receiver,
                                                                 tmp_path):
    """The handle belongs to the close until the close has finished.

    Tuning hands the flush and the close to a thread.  While that runs
    the flag is down and nothing calls it a recording, so a new one
    could be started - into the same handle field.  The old close then
    reached it and closed the new recording's file, leaving a window
    that still said "recording" and a WAV with nothing in it.
    """
    audio = receiver.audio_output
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    audio.start_recording(str(first))
    audio.record(np.zeros((2048, 2), dtype=np.float32))

    let_it_close = a_close_held_at_the_door(receiver)

    tuned = receiver.tune(81.3e6)
    assert tuned.wait(5), "the tune never landed"
    assert not audio.recording and audio.finalising, \
        "the first recording is meant to be closing"

    asked = a_start_that_says_when_it_asks(
        audio, "wait_for_the_recording_to_close")
    request = receiver.start_recording(str(second))

    assert asked.wait(5), \
        "the start never asked whether the last recording was still closing"
    assert not audio.recording, \
        "the second recording was installed under the first one's close"

    let_it_close.set()
    assert request.wait(10), "the second recording never started"
    assert not request.failed, request.error

    assert audio.recording, "the second recording is not running"
    assert audio.record_wave is not None, \
        "the first recording's close took the second one's file"

    audio.record(np.ones((2048, 2), dtype=np.float32))
    audio.stop_recording()

    with wave.open(str(second), "rb") as f:
        assert f.getnframes() == 2048, "nothing was written to the second file"
    with wave.open(str(first), "rb") as f:
        assert f.getnframes() == 2048, "the first file lost what was in it"


def test_an_iq_recording_that_is_closing_does_not_close_the_next_one(
        receiver, tmp_path):
    """The IQ side shares a handle field the same way."""
    sdr = receiver.sdr_receiver
    first = tmp_path / "first_iq.wav"
    second = tmp_path / "second_iq.wav"
    sdr.start_iq_recording(str(first))

    let_it_close = a_close_held_at_the_door(receiver)

    tuned = receiver.tune(81.3e6)
    assert tuned.wait(5), "the tune never landed"
    assert not sdr.iq_recording and sdr.iq_finalising, \
        "the first IQ recording is meant to be closing"

    asked = a_start_that_says_when_it_asks(
        sdr, "wait_for_the_iq_recording_to_close")
    request = receiver.start_iq_recording(str(second))

    assert asked.wait(5), \
        "the start never asked whether the last one was still closing"
    assert not sdr.iq_recording, \
        "the second IQ recording was installed under the first one's close"

    let_it_close.set()
    assert request.wait(10), "the second IQ recording never started"
    assert not request.failed, request.error

    assert sdr.iq_recording, "the second IQ recording is not running"
    assert sdr.iq_record_wave is not None, \
        "the first recording's close took the second one's file"


def test_a_recording_is_refused_when_the_last_one_will_not_finish_closing(
        receiver, tmp_path, monkeypatch):
    """The wait is on the thread that pressed the button, so it is bounded."""
    monkeypatch.setattr("fm_radio.audio_output._PREVIOUS_CLOSE_WAIT_SEC", 0.2)
    audio = receiver.audio_output
    audio.start_recording(str(tmp_path / "will_not_close.wav"))

    let_it_close = a_close_held_at_the_door(receiver)

    try:
        assert receiver.tune(81.3e6).wait(5), "the tune never landed"
        assert audio.finalising

        asked = receiver.start_recording(str(tmp_path / "refused.wav"))

        assert asked.wait(5), "the request never answered"
        assert isinstance(asked.error, RecordingError), asked.error
        assert audio.record_wave is not None, \
            "the refused recording took the closing one's handle"
    finally:
        let_it_close.set()


def test_an_iq_recording_is_refused_when_the_last_one_will_not_close(
        receiver, tmp_path, monkeypatch):
    """The same for IQ."""
    monkeypatch.setattr("fm_radio.sdr_receiver._PREVIOUS_IQ_CLOSE_WAIT_SEC",
                        0.2)
    sdr = receiver.sdr_receiver
    sdr.start_iq_recording(str(tmp_path / "will_not_close_iq.wav"))

    let_it_close = a_close_held_at_the_door(receiver)

    try:
        assert receiver.tune(81.3e6).wait(5), "the tune never landed"
        assert sdr.iq_finalising

        asked = receiver.start_iq_recording(str(tmp_path / "refused_iq.wav"))

        assert asked.wait(5), "the request never answered"
        assert isinstance(asked.error, RecordingError), asked.error
    finally:
        let_it_close.set()
