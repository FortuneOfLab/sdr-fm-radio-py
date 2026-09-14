"""The processing thread publishing its state.

These drive the real processing loop over fake hardware: IQ blocks are put
straight into the SDR queue, the loop is run until it has published, and the
snapshot is inspected.  What is being checked is the wiring and the cost, not
the DSP.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fm_radio.controller import FMReceiverController
from fm_radio.telemetry import SILENCE_DBFS


def iq_block(controller, amplitude: float = 0.3) -> np.ndarray:
    """One block of noise, the size the SDR would deliver."""
    size = controller.sdr_receiver.block_size
    rng = np.random.default_rng(0)
    return (amplitude * (rng.standard_normal(size)
                         + 1j * rng.standard_normal(size))).astype(np.complex64)


@pytest.fixture
def receiver(no_user_config):
    """A controller on fake hardware whose SDR thread is never started."""
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def run_blocks(controller, count: int, timeout: float = 10.0) -> None:
    """Feed *count* IQ blocks through the real processing loop."""
    block = iq_block(controller)
    for _ in range(count):
        controller.sdr_receiver.data_queue.put(block)

    # A previous call left the loop stopped; it has to be told to run again.
    controller.quit_event.clear()
    thread = threading.Thread(target=controller.processing_thread, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while (not controller.sdr_receiver.data_queue.empty()
           and time.monotonic() < deadline):
        time.sleep(0.005)
    controller.quit_event.set()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "processing thread did not stop"


# ----------------------------------------------------------------------

def test_no_status_before_the_first_block(receiver):
    assert receiver.get_status() is None


def test_a_snapshot_appears_once_blocks_are_processed(receiver):
    run_blocks(receiver, 4)
    status = receiver.get_status()
    assert status is not None
    assert status.freq_hz == receiver.get_frequency()
    assert status.sdr_queue_max == receiver.sdr_receiver.data_queue.maxsize
    assert status.block_budget_ms == pytest.approx(16.0)
    assert status.uptime_sec > 0.0


def test_the_snapshot_names_the_tuned_station(receiver):
    receiver.tune(80.0e6)
    run_blocks(receiver, 4)
    assert receiver.get_status().station == "TOKYO FM"


def test_levels_are_measured_from_the_demodulated_audio(receiver):
    run_blocks(receiver, 4)
    status = receiver.get_status()
    for level in (status.level_left_dbfs, status.level_right_dbfs):
        assert SILENCE_DBFS <= level <= 0.0


def test_block_timing_is_reported(receiver):
    run_blocks(receiver, 4)
    status = receiver.get_status()
    assert status.block_ms > 0.0
    assert status.block_ms_avg > 0.0
    assert status.block_ms_max >= status.block_ms_avg


def test_health_counters_start_clean(receiver):
    run_blocks(receiver, 4)
    status = receiver.get_status()
    assert status.iq_drops == 0
    assert status.audio_drops == 0
    assert status.slow_blocks >= 0


def test_recording_state_is_reported(receiver, tmp_path):
    receiver.start_recording(str(tmp_path / "audio.wav"))
    try:
        run_blocks(receiver, 4)
        assert receiver.get_status().recording_audio is True
    finally:
        receiver.stop_recording()


def test_a_block_that_is_not_due_does_not_publish(receiver):
    """Rate limiting is asserted against the interval, not against the clock.

    Counting snapshots from a wall-clock expectation would depend on how
    fast the machine runs the loop, which is exactly the kind of assertion
    that passes here and flakes on a slower runner.
    """
    receiver.telemetry.interval_sec = 3600.0
    run_blocks(receiver, 60)
    assert receiver.telemetry.published_count == 1


def test_every_block_publishes_when_the_interval_is_zero(receiver):
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 10)
    assert receiver.telemetry.published_count == 10


def test_tuning_drops_the_previous_snapshot(receiver):
    """The old snapshot describes the old station's pilot and blend."""
    run_blocks(receiver, 4)
    assert receiver.get_status() is not None
    receiver.tune(81.3e6)
    assert receiver.get_status() is None


def test_a_failing_snapshot_does_not_stop_the_audio(receiver, monkeypatch):
    """Telemetry is for looking at, never a reason to interrupt the audio."""
    calls = []

    def explode(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("snapshot boom")

    monkeypatch.setattr(receiver, "_build_snapshot", explode)
    run_blocks(receiver, 10)

    assert calls, "the snapshot was never attempted"
    assert receiver.get_status() is None
    # The loop kept going: every block was taken off the queue.
    assert receiver.sdr_receiver.data_queue.empty()


def test_the_snapshot_is_a_plain_value(receiver):
    run_blocks(receiver, 4)
    for value in receiver.get_status().as_dict().values():
        assert value is None or isinstance(value, (bool, int, float, str))


def test_the_station_lookup_is_not_repeated_per_snapshot(receiver, monkeypatch):
    """Scanning 983 transmitters every 50 ms was most of a snapshot's cost."""
    import fm_radio.controller as controller_module

    calls = []
    original = controller_module.nearest

    def counted(catalogue, freq_hz, *args, **kwargs):
        calls.append(freq_hz)
        return original(catalogue, freq_hz, *args, **kwargs)

    monkeypatch.setattr(controller_module, "nearest", counted)
    receiver.telemetry.interval_sec = 0.0       # publish on every block
    run_blocks(receiver, 12)

    assert receiver.telemetry.published_count == 12
    # One lookup for the frequency, however many snapshots were published.
    assert len(calls) == 1, f"{len(calls)} catalogue scans"


def test_retuning_refreshes_the_cached_station_name(receiver):
    receiver.tune(80.0e6)
    run_blocks(receiver, 2)
    assert receiver.get_status().station == "TOKYO FM"

    receiver.tune(81.3e6)
    run_blocks(receiver, 2)
    assert receiver.get_status().station == "J-WAVE"
