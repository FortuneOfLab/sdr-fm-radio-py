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

import fm_radio.sdr_receiver as fm_sdr
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

    original_close = device.close

    def marked_close():
        closed.set()
        return original_close()

    monkeypatch.setattr(device, "set_gain", slow_write)
    monkeypatch.setattr(device, "close", marked_close)

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
    writing = threading.Event()
    release = threading.Event()
    closes: list[str] = []

    def wedged_write(gain):
        writing.set()
        release.wait(30)                    # never set until the test is done

    monkeypatch.setattr(device, "set_gain", wedged_write)
    monkeypatch.setattr(device, "close", lambda: closes.append("close"))
    monkeypatch.setattr("fm_radio.sdr_receiver._DEVICE_LOCK_TIMEOUT_SEC", 0.5)

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
    monkeypatch.setattr("fm_radio.sdr_receiver._DEVICE_LOCK_TIMEOUT_SEC", 0.3)

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
    assert sdr._device_lock.acquire()
    try:
        started = time.monotonic()
        sdr.get_center_frequency()
        sdr.get_gain()
        elapsed = time.monotonic() - started
    finally:
        sdr._device_lock.release()

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
    monkeypatch.setattr("fm_radio.sdr_receiver._DEVICE_LOCK_TIMEOUT_SEC", 0.5)

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
    monkeypatch.setattr("fm_radio.sdr_receiver._SAMPLING_CANCEL_TIMEOUT_SEC",
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
    monkeypatch.setattr("fm_radio.sdr_receiver._RAW_CANCEL_ASYNC", None)
    monkeypatch.setattr("fm_radio.sdr_receiver._SAMPLING_CANCEL_TIMEOUT_SEC",
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
    monkeypatch.setattr("fm_radio.sdr_receiver._RAW_CANCEL_ASYNC", None)

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
    monkeypatch.setattr("fm_radio.sdr_receiver._DEVICE_LOCK_TIMEOUT_SEC", 0.4)

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
    real_close = sdr._real_close

    def slow_close():
        in_the_close.set()
        let_the_close_finish.wait(10)
        real_close()

    monkeypatch.setattr(sdr, "_real_close", slow_close)

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
    real_raw = fm_sdr._RAW_CANCEL_ASYNC

    def stalled_raw(dev_p):
        at_the_call.set()
        close_done.wait(1.0)        # bounded: with the lock held it expires
        return real_raw(dev_p)

    monkeypatch.setattr("fm_radio.sdr_receiver._RAW_CANCEL_ASYNC", stalled_raw)

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
