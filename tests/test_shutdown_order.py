"""Shutting down while something is still using what is being closed.

A bounded join cannot promise that every thread has finished, so these check
the two hazards that leaves: audio arriving after the output is closed, and a
gain write landing on a device that has just gone. Timing is forced with
events and deliberate delays rather than left to chance.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fm_radio.controller import FMReceiverController


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
