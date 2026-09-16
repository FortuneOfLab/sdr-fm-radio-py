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

from fm_radio.controller import FMReceiverController
from fm_radio.exceptions import RecordingError


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
    cancelled = threading.Event()
    writing = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(device, "cancel_read_async", lambda: cancelled.set())
    monkeypatch.setattr(device, "read_samples_async",
                        lambda cb, num_samples=None: cancelled.wait(30))
    monkeypatch.setattr(device, "set_gain",
                        lambda gain: (writing.set(), release.wait(30)))
    monkeypatch.setattr("fm_radio.sdr_receiver._DEVICE_LOCK_TIMEOUT_SEC", 0.5)

    reader = threading.Thread(target=sdr.start, name="SDRThread", daemon=True)
    reader.start()
    receiver.auto_gain._submit_async_gain(30.0)
    assert writing.wait(5), "the gain worker never started writing"

    try:
        receiver.cleanup()

        assert cancelled.is_set(), "the async read was never cancelled"
        reader.join(timeout=5)
        assert not reader.is_alive(), "the receive thread outlived cleanup"
    finally:
        release.set()
        cancelled.set()
        time.sleep(0.2)             # let the wedged write unwind


def test_sampling_is_cancelled_before_the_device_is_closed(receiver,
                                                           monkeypatch):
    """Order matters: cancel_read_async dereferences the handle close frees."""
    device = receiver.sdr_receiver.sdr
    order: list[str] = []
    monkeypatch.setattr(device, "cancel_read_async",
                        lambda: order.append("cancel"))
    monkeypatch.setattr(device, "close", lambda: order.append("close"))

    receiver.sdr_receiver.stop()

    assert order == ["cancel", "close"]


def test_stopping_twice_does_not_touch_a_freed_handle(receiver, monkeypatch):
    """cleanup runs twice on several paths, and the second must do nothing.

    pyrtlsdr guards close() with its own flag but passes the device
    pointer to rtlsdr_cancel_async without checking, so a second cancel
    after the close reads memory that has been freed.
    """
    device = receiver.sdr_receiver.sdr
    calls: list[str] = []
    monkeypatch.setattr(device, "cancel_read_async",
                        lambda: calls.append("cancel"))
    monkeypatch.setattr(device, "close", lambda: calls.append("close"))

    receiver.sdr_receiver.stop()
    receiver.sdr_receiver.stop()

    assert calls == ["cancel", "close"]


def test_a_driver_side_close_stops_the_cancel(receiver, monkeypatch):
    """pyrtlsdr closes the device itself when a control write fails."""
    device = receiver.sdr_receiver.sdr
    calls: list[str] = []
    monkeypatch.setattr(device, "device_opened", False, raising=False)
    monkeypatch.setattr(device, "cancel_read_async",
                        lambda: calls.append("cancel"))
    monkeypatch.setattr(device, "close", lambda: calls.append("close"))

    receiver.sdr_receiver.stop()

    assert calls == [], f"touched a handle the driver had already freed: {calls}"


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
