"""The processing thread publishing its state.

These drive the real processing loop over fake hardware: IQ blocks are put
straight into the SDR queue, the loop is run until it has published, and the
snapshot is inspected.  What is being checked is the wiring and the cost, not
the DSP.
"""

from __future__ import annotations

import contextlib
import sys
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


def enqueue(controller, block, generation: int | None = None) -> None:
    """Put one IQ block on the queue, stamped as the SDR would stamp it."""
    if generation is None:
        generation = controller.sdr_receiver.tuning_generation
    controller.sdr_receiver.data_queue.put((generation, block))


def tuned(controller, freq_hz: float, timeout: float = 5.0) -> None:
    """Tune, and wait for the write to have happened.

    ``tune()`` hands the write to the device worker and comes back at
    once, so anything that depends on the tuner having moved - the
    generation, what the next block counts as - has to wait for the
    request rather than for the call.
    """
    asked = controller.tune(freq_hz)
    assert asked.wait(timeout), f"the tune to {freq_hz / 1e6:.1f} MHz never landed"
    assert not asked.failed, f"the tune failed: {asked.error}"


def run_blocks(controller, count: int, timeout: float = 10.0) -> None:
    """Feed *count* IQ blocks through the real processing loop."""
    block = iq_block(controller)
    for _ in range(count):
        enqueue(controller, block)

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
    tuned(receiver, 80.0e6)
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
    """The old snapshot describes the old station's pilot and blend.

    Dropped when the tune lands, not when it is asked for: until the
    write has happened the radio really is still on the old station, and
    the old snapshot is the truth about it.
    """
    run_blocks(receiver, 4)
    assert receiver.get_status() is not None

    request = receiver.tune(81.3e6)

    assert request.wait(5), "the tune never landed"
    assert receiver.get_status() is None


def test_the_old_snapshot_stands_until_the_tune_lands(receiver, monkeypatch):
    """A tune that has been asked for is not a tune that has happened.

    The window would otherwise have nothing to show for as long as the
    write takes - 60 ms on a device that is answering - having been told
    the reading it has is stale when the receiver has not moved yet.
    """
    landed = threading.Event()
    original = receiver.sdr_receiver.set_center_frequency

    def slow_write(freq_hz):
        landed.wait(10)
        original(freq_hz)

    monkeypatch.setattr(receiver.sdr_receiver, "set_center_frequency",
                        slow_write)
    run_blocks(receiver, 4)
    before = receiver.get_status()
    assert before is not None

    request = receiver.tune(81.3e6)

    assert receiver.get_status() is before, (
        "dropped the old station's reading before leaving it")

    landed.set()
    assert request.wait(5)
    assert receiver.get_status() is None


def count_enqueued(receiver, monkeypatch) -> list:
    """Record every block handed to the audio output.

    An empty SDR queue only says the loop drained it; what matters is that
    audio kept reaching the output while telemetry was misbehaving.
    """
    enqueued = []
    original = receiver.audio_output.enqueue_audio

    def counted(left, right):
        enqueued.append(float(left.size))
        return original(left, right)

    monkeypatch.setattr(receiver.audio_output, "enqueue_audio", counted)
    return enqueued


def test_a_failing_snapshot_does_not_stop_the_audio(receiver, monkeypatch):
    """Telemetry is for looking at, never a reason to interrupt the audio."""
    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("snapshot boom")

    enqueued = count_enqueued(receiver, monkeypatch)
    monkeypatch.setattr(receiver, "_build_snapshot", explode)
    run_blocks(receiver, 10)

    assert attempts, "the snapshot was never attempted"
    assert receiver.get_status() is None
    assert len(enqueued) == 10, "audio stopped reaching the output"


def test_a_persistent_failure_is_not_retried_every_block(receiver, monkeypatch):
    """Without deferral one broken snapshot became a failure per block."""
    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("snapshot boom")

    receiver.telemetry.interval_sec = 3600.0
    monkeypatch.setattr(receiver, "_build_snapshot", explode)
    run_blocks(receiver, 10)

    assert len(attempts) == 1, f"{len(attempts)} attempts over 10 blocks"


def capture_telemetry_warnings(receiver, monkeypatch) -> list:
    """Record only the telemetry warning.

    The processing thread shares this logger with the block profiler, which
    warns about any block over 20 ms - something a cold first block does on
    its own. Counting every warning made this test depend on that.
    """
    warnings = []
    original = receiver.logger.warning

    def capture(message, *args, **kwargs):
        if isinstance(message, str) and message.startswith(
                "Telemetry snapshot failed"):
            warnings.append((message, args))
        return original(message, *args, **kwargs)

    monkeypatch.setattr(receiver.logger, "warning", capture)
    return warnings


def test_a_persistent_failure_warns_once_not_once_per_block(receiver,
                                                            monkeypatch):
    warnings = capture_telemetry_warnings(receiver, monkeypatch)
    monkeypatch.setattr(receiver, "_build_snapshot",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("snapshot boom")))
    receiver.telemetry.interval_sec = 0.0       # retry on every block
    run_blocks(receiver, 10)

    assert receiver._telemetry_failures == 10   # every block did try
    assert len(warnings) == 1, f"{len(warnings)} warnings for one fault"


def test_a_fault_after_a_recovery_is_warned_about(receiver, monkeypatch):
    """The quiet period belongs to the fault that earned it, not to the clock."""
    warnings = capture_telemetry_warnings(receiver, monkeypatch)
    state = {"failing": True}
    original = receiver._build_snapshot

    def sometimes(*args, **kwargs):
        if state["failing"]:
            raise RuntimeError("snapshot boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(receiver, "_build_snapshot", sometimes)
    receiver.telemetry.interval_sec = 0.0

    run_blocks(receiver, 2)                     # fault
    assert len(warnings) == 1
    state["failing"] = False
    run_blocks(receiver, 2)                     # recovery
    assert receiver.get_status() is not None
    state["failing"] = True
    run_blocks(receiver, 2)                     # a new fault

    assert len(warnings) == 2, "the second fault was hidden by the first"


def test_publishing_recovers_once_the_failure_clears(receiver, monkeypatch):
    """The deadline is armed, not the publisher switched off."""
    failing = {"yes": True}
    original = receiver._build_snapshot

    def sometimes(*args, **kwargs):
        if failing["yes"]:
            raise RuntimeError("snapshot boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(receiver, "_build_snapshot", sometimes)
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 4)
    assert receiver.get_status() is None

    failing["yes"] = False
    run_blocks(receiver, 4)
    assert receiver.get_status() is not None


def test_a_failed_block_does_not_publish_the_previous_block_audio(
        receiver, monkeypatch):
    """The failed block would otherwise carry the last block's levels."""
    calls = {"n": 0}
    demodulate = receiver.fm_demodulator.demodulate

    def fail_second(composite):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("demod boom")
        return demodulate(composite)

    monkeypatch.setattr(receiver.fm_demodulator, "demodulate", fail_second)
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 2)

    assert calls["n"] == 2, "the second block was never demodulated"
    assert receiver.telemetry.published_count == 1


def test_a_failing_first_block_publishes_nothing(receiver, monkeypatch):
    """There is no previous audio at all; this used to raise NameError."""
    warnings = []
    monkeypatch.setattr(receiver.logger, "warning",
                        lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(receiver.fm_demodulator, "demodulate",
                        lambda composite: (_ for _ in ()).throw(
                            RuntimeError("demod boom")))
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 2)

    assert receiver.telemetry.published_count == 0
    assert receiver.get_status() is None
    assert not warnings, f"unexpected warnings: {warnings}"


def test_a_block_that_is_not_due_builds_nothing(receiver, monkeypatch):
    """Deterministic stand-in for a timing assertion on the same property."""
    builds = []
    original = receiver._build_snapshot

    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(receiver, "_build_snapshot", counted)
    receiver.telemetry.interval_sec = 3600.0
    run_blocks(receiver, 30)

    assert len(builds) == 1, f"{len(builds)} snapshots built for 30 blocks"


def test_a_block_captured_before_a_retune_is_never_published(receiver):
    """The generation travels with the samples, not with the clock.

    Reading it after taking the block off the queue left a window: a retune
    landing in that window tagged the old station's samples as belonging to
    the new one, and the snapshot went out with the new frequency against
    the old station's IQ peak.

    Such a block is now dropped where it comes off the queue rather than
    demodulated and then kept off the display: it is the old station's
    audio, and there is nothing to be done with it.
    """
    # Waited for, both of them: tune() asks the device worker and comes
    # straight back, so reading the generation or queueing a block
    # without waiting races the write and the bump that follows it.
    tuned(receiver, 80.0e6)
    stale_generation = receiver.sdr_receiver.tuning_generation

    # tune() flushes the queue, so the block that matters is the one still in
    # flight in the SDR when the frequency changed: it lands after the flush,
    # carrying the tuning it was captured under.
    tuned(receiver, 81.3e6)
    assert receiver.sdr_receiver.tuning_generation != stale_generation
    enqueue(receiver, iq_block(receiver), stale_generation)

    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 0)                     # drain what is queued

    assert receiver._stale_blocks == 1, "the pre-retune block was processed"
    assert receiver.telemetry.published_count == 0
    assert receiver.get_status() is None, "the pre-retune block was published"


def test_a_block_captured_after_a_retune_is_published(receiver):
    tuned(receiver, 80.0e6)
    tuned(receiver, 81.3e6)
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 2)

    status = receiver.get_status()
    assert status is not None
    assert status.station == "J-WAVE"


def test_a_retune_while_a_block_is_in_flight_hides_its_snapshot(receiver):
    """The ordering is forced with events rather than left to chance.

    The existing retune test stops the loop first, so it cannot see this:
    a snapshot built for the old station and stored after the tuner moved
    used to come back as the current state of the new frequency.
    """
    built = threading.Event()
    retuned = threading.Event()
    original = receiver._build_snapshot

    def stalled(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        built.set()
        retuned.wait(5)                     # the tuner moves meanwhile
        return snapshot

    tuned(receiver, 80.0e6)
    receiver._build_snapshot = stalled
    enqueue(receiver, iq_block(receiver))
    receiver.quit_event.clear()
    thread = threading.Thread(target=receiver.processing_thread, daemon=True)
    thread.start()
    try:
        assert built.wait(10), "the snapshot was never built"
        tuned(receiver, 81.3e6)
        retuned.set()
        deadline = time.monotonic() + 5
        while (receiver.telemetry.published_count == 0
               and time.monotonic() < deadline):
            time.sleep(0.005)
    finally:
        receiver.quit_event.set()
        thread.join(timeout=5)

    assert receiver.telemetry.published_count == 1, "nothing was published"
    assert receiver.get_status() is None, "the old station's snapshot came back"
    assert receiver.get_frequency() == pytest.approx(81.3e6)


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
    # Retuning no longer re-arms the deadline, so publish on every block.
    receiver.telemetry.interval_sec = 0.0
    tuned(receiver, 80.0e6)
    run_blocks(receiver, 2)
    assert receiver.get_status().station == "TOKYO FM"

    tuned(receiver, 81.3e6)
    run_blocks(receiver, 2)
    assert receiver.get_status().station == "J-WAVE"


def test_a_retune_during_the_sdr_callback_does_not_publish_the_old_station(
        receiver, stalling_samples):
    """End to end through the real callback, not the test helper.

    The helper stamps the generation by hand, so it cannot see the callback
    stamping the wrong one. Here the retune lands inside the conversion the
    callback does, which is where the samples get their generation.
    """
    tuned(receiver, 80.0e6)
    was_on = receiver.sdr_receiver.tuning_generation
    receiver.telemetry.interval_sec = 0.0
    samples = stalling_samples(iq_block(receiver).astype(np.complex128))

    thread = threading.Thread(target=receiver.sdr_receiver.callback,
                              args=(samples, None), daemon=True)
    thread.start()
    try:
        assert samples.converting.wait(5), "the callback never started converting"
        # Released only once the retune has actually happened.  tune()
        # hands the write to the device worker and returns, so letting
        # the callback go on the strength of the call alone lets the
        # block through before the generation has moved - and then
        # there is nothing stale about it.
        tuned(receiver, 81.3e6)         # flushes, then this block lands
        assert receiver.sdr_receiver.tuning_generation != was_on
        samples.retuned.set()
    finally:
        thread.join(timeout=5)

    assert samples.retuned_in_time, (
        "the retune did not complete while the samples were being converted, "
        "so this run never built the ordering it is meant to test")
    assert receiver.sdr_receiver.data_queue.qsize() == 1
    run_blocks(receiver, 0)             # process what the callback queued

    assert receiver._stale_blocks == 1, "the pre-retune block was processed"
    assert receiver.telemetry.published_count == 0
    assert receiver.get_status() is None, "the pre-retune block was published"


# ----------------------------------------------------------------------
# Resetting the demodulator: which thread, and in what order
# ----------------------------------------------------------------------

class WatchedDemodulator:
    """Records resets and processed blocks, and says when enough happened.

    Both calls are made on the processing thread, so the list is the
    order that thread saw them in - which is the thing being pinned.
    """

    def __init__(self, demodulator):
        self.events: list[tuple[str, str]] = []
        self._want: int | None = None
        self._enough = threading.Event()
        self._real_reset = demodulator.reset
        self._real_process = demodulator.process_iq_samples
        demodulator.reset = self._reset
        demodulator.process_iq_samples = self._process

    def _note(self, what: str) -> None:
        self.events.append((what, threading.current_thread().name))
        if self._want is not None and len(self.events) >= self._want:
            self._enough.set()

    def _reset(self):
        self._note("reset")
        return self._real_reset()

    def _process(self, iq_samples):
        self._note("process")
        return self._real_process(iq_samples)

    def expect(self, count: int) -> None:
        """Arm for *count* events from here on.  Call before causing them."""
        self.events = []
        self._want = count
        self._enough.clear()

    def wait(self, timeout: float = 10.0) -> None:
        assert self._enough.wait(timeout), \
            f"only {len(self.events)} of {self._want}: {self.kinds}"

    @property
    def kinds(self) -> list[str]:
        return [what for what, _who in self.events]

    @property
    def threads(self) -> set[str]:
        return {who for _what, who in self.events}


@contextlib.contextmanager
def the_loop_running(controller, timeout: float = 10.0):
    """Run the real processing loop for the body of the with-statement."""
    controller.quit_event.clear()
    thread = threading.Thread(target=controller.processing_thread,
                              name="Processing", daemon=True)
    thread.start()
    try:
        yield
    finally:
        controller.quit_event.set()
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "processing thread did not stop"


def test_tuning_does_not_reset_the_demodulator_from_the_device_worker(
        receiver):
    """That state belongs to the thread that is using it.

    A reset from the worker lands in the middle of whatever block the
    processing thread has in hand: it clears the resampler history that
    the pending output of that block is measured against, and the block
    dies with "Resampler history no longer covers pending output".
    """
    watched = WatchedDemodulator(receiver.fm_demodulator)

    tuned(receiver, 81.3e6)

    assert watched.events == [], \
        f"the tune touched the demodulator: {watched.events}"


def test_the_first_block_of_a_new_tuning_is_reset_before_it_is_processed(
        receiver):
    """And the reset is on the processing thread, where it is safe."""
    tuned(receiver, 80.0e6)
    watched = WatchedDemodulator(receiver.fm_demodulator)

    with the_loop_running(receiver):
        watched.expect(2)                       # the loop's own first block
        enqueue(receiver, iq_block(receiver))
        watched.wait()

        tuned(receiver, 81.3e6)
        watched.expect(2)
        enqueue(receiver, iq_block(receiver))
        watched.wait()

    assert watched.kinds == ["reset", "process"], watched.kinds
    assert watched.threads == {"Processing"}, watched.threads


def test_the_blocks_after_the_first_are_not_reset_again(receiver):
    """One reset per tuning, not one per block."""
    tuned(receiver, 80.0e6)
    watched = WatchedDemodulator(receiver.fm_demodulator)

    with the_loop_running(receiver):
        watched.expect(2)
        enqueue(receiver, iq_block(receiver))
        watched.wait()

        watched.expect(4)
        for _ in range(4):
            enqueue(receiver, iq_block(receiver))
        watched.wait()

    assert watched.kinds == ["process"] * 4, watched.kinds


def test_a_stale_block_is_dropped_before_the_new_tuning_is_started(receiver):
    """The order, and what asking the other question first would cost.

    A block from before the retune is going to be thrown away, so it
    should not touch the demodulator on its way out.  Resetting for it
    first costs a reset that nothing uses, and leaves
    ``_generation_in_hand`` naming a tuning no block of which was ever
    demodulated - the real first block of the new station then resets
    again.  Two resets and a lie about which tuning the demodulator is
    set up for, in place of one reset and the truth.
    """
    tuned(receiver, 80.0e6)
    stale = receiver.sdr_receiver.tuning_generation
    tuned(receiver, 81.3e6)
    fresh = receiver.sdr_receiver.tuning_generation
    assert fresh != stale, "the two tunes were the same generation"

    watched = WatchedDemodulator(receiver.fm_demodulator)
    watched.expect(2)
    enqueue(receiver, iq_block(receiver), stale)    # the straggler
    enqueue(receiver, iq_block(receiver), fresh)    # the new station

    with the_loop_running(receiver):
        watched.wait()

    assert watched.kinds == ["reset", "process"], \
        f"the stale block reached the demodulator: {watched.kinds}"
    assert receiver._stale_blocks == 1


# ----------------------------------------------------------------------
# How clean the channel is
# ----------------------------------------------------------------------

def test_the_snapshot_says_how_steady_the_envelope_is(receiver):
    """An FM carrier holds its amplitude; noise does not."""
    run_blocks(receiver, 4)

    status = receiver.get_status()

    assert status.am_depth > 0.0, "the fake IQ is noise; it cannot be steady"


def test_the_envelope_is_measured_on_the_channel_not_the_band(receiver):
    """The raw block carries the neighbours too.

    On the real radio a frequency with nothing on it next to a loud
    one read 0.124 against the whole 1.024 MHz and 0.646 once the
    channel was all that was left - a clean-looking figure produced
    entirely by somebody else's transmission.  So the snapshot has
    to be measuring what the demodulator filtered, not what the SDR
    delivered.
    """
    demod = receiver.fm_demodulator
    handed_in = []
    measured = []

    real_process = demod.process_iq_samples
    module = sys.modules["fm_radio.controller"].multipath
    real_depth = module.am_depth

    def watched_process(iq):
        handed_in.append(id(iq))
        return real_process(iq)

    def watched_depth(iq):
        # What it was given, and whether that was the channel the
        # demodulator had just filtered.  Not the size: a filter is
        # the same length in as out, so every version of this passes
        # a size check.
        measured.append((id(iq), iq is demod.channel_iq))
        return real_depth(iq)

    demod.process_iq_samples = watched_process
    module.am_depth = watched_depth
    try:
        run_blocks(receiver, 4)
    finally:
        module.am_depth = real_depth

    assert measured, "nothing measured the envelope"
    raw = set(handed_in)
    for where, was_the_channel in measured:
        assert where not in raw, "the whole band was measured"
        assert was_the_channel, "something other than the channel was"


def test_a_block_that_cannot_be_measured_is_not_reported_as_clean(receiver):
    """The snapshot carries the not-measurable through as None.

    0.06 is a good station and 0.0 is a perfect one; neither is what
    a block of NaNs is, and this reading is the one thing that must
    not call the worst input the best reception.
    """
    block = iq_block(receiver)
    block[7] = complex(float("nan"), 0.0)
    enqueue(receiver, block)
    receiver.telemetry.interval_sec = 0.0
    run_blocks(receiver, 1)

    seen = [receiver.get_status().am_depth]
    receiver.fm_demodulator._channel_iq = np.full(
        16, complex(float("nan"), 0.0), dtype=np.complex64)
    run_blocks(receiver, 1)
    seen.append(receiver.get_status().am_depth)

    assert None in seen, "a block of NaNs was measured as %s" % seen


def test_a_demodulator_that_has_not_run_yet_reads_as_nothing(receiver):
    """channel_iq is None before the first block, and after a retune."""
    assert receiver.fm_demodulator.channel_iq is None

    run_blocks(receiver, 2)
    assert receiver.fm_demodulator.channel_iq is not None

    receiver.fm_demodulator.reset()
    assert receiver.fm_demodulator.channel_iq is None,         "the station just left is still in there"


# ----------------------------------------------------------------------
# The picture of the band
# ----------------------------------------------------------------------

def test_a_block_produces_a_picture_of_the_band(receiver):
    """Out of the same IQ the demodulator gets."""
    receiver.spectrum.interval_sec = 0.0
    run_blocks(receiver, 2)

    frame = receiver.get_spectrum()

    assert frame is not None, "no picture was published"
    assert frame.span_hz == receiver.sdr_receiver.sample_rate
    assert frame.center_hz == pytest.approx(
        receiver.sdr_receiver.center_freq)
    assert len(frame.dbfs) > 0


def test_the_picture_is_not_made_for_every_block(receiver):
    """Ten a second, not one per block: it costs more than a snapshot."""
    receiver.spectrum.interval_sec = 10.0
    run_blocks(receiver, 6)

    assert receiver.spectrum.published_count == 1, \
        "a picture was made for more than the first block"


def test_a_picture_made_during_a_retune_is_of_the_station_it_came_from(
        receiver):
    """The window between the attribute moving and the tuner moving.

    set_center_frequency writes the new frequency to the receiver's
    own attribute, then spends 40-200 ms on the hardware, then bumps
    the generation.  For all of that time the attribute says the new
    station while the samples are still of the old one - and the
    staleness check cannot catch it, because the generation has not
    moved either.  So the frequency has to come from the tuner
    paired with the generation, not from the attribute.
    """
    tuned(receiver, 80.0e6)
    receiver.spectrum.interval_sec = 0.0
    sdr = receiver.sdr_receiver
    was = sdr.tuning_generation
    writing = threading.Event()
    let_it_finish = threading.Event()
    real_device = sdr.sdr

    class HeldInTheWrite:
        """The device, with its tuning write held open."""

        def __getattr__(self, name):
            return getattr(real_device, name)

        def __setattr__(self, name, value):
            if name == "center_freq":
                writing.set()
                assert let_it_finish.wait(5), "the test never let go"
            setattr(real_device, name, value)

    sdr.sdr = HeldInTheWrite()
    asked = receiver.tune(81.3e6)
    try:
        assert writing.wait(5), "the write never started"
        # The state the finding is about, before anything is asked of it.
        assert sdr.center_freq == 81.3e6, "the attribute has not moved yet"
        assert sdr.tuning_generation == was, "the generation moved too soon"

        receiver._publish_spectrum(iq_block(receiver), time.monotonic(), was)
        frame = receiver.get_spectrum()
    finally:
        let_it_finish.set()
        asked.wait(5)
        sdr.sdr = real_device

    assert frame is not None, "no picture was published"
    assert frame.center_hz == pytest.approx(80.0e6), (
        "samples of 80.0 MHz published as %.1f MHz"
        % (frame.center_hz / 1e6))


def test_a_picture_from_before_a_retune_is_never_handed_out(receiver):
    """The same rule the snapshots follow, for the same reason.

    A picture of the station the receiver has just left is not one to
    show above a tuner that says it is somewhere else.
    """
    tuned(receiver, 80.0e6)
    receiver.spectrum.interval_sec = 0.0
    run_blocks(receiver, 2)
    assert receiver.get_spectrum() is not None

    tuned(receiver, 81.3e6)

    assert receiver.get_spectrum() is None, "the old station's picture"


def test_a_picture_that_cannot_be_built_costs_one_block(receiver):
    """Not one failure per block for ever afterwards."""
    receiver.spectrum.interval_sec = 0.0
    tries = []

    def refuse(*args, **kwargs):
        tries.append(1)
        raise ValueError("no picture today")

    receiver._spectrum_maker.frame = refuse
    run_blocks(receiver, 4)

    assert receiver.get_spectrum() is None
    assert len(tries) == 4, f"it stopped trying after {len(tries)}"


def test_a_picture_that_cannot_be_built_does_not_stop_the_block(receiver):
    """The audio matters more than the display."""
    receiver.spectrum.interval_sec = 0.0
    receiver.telemetry.interval_sec = 0.0

    def refuse(*args, **kwargs):
        raise ValueError("no picture today")

    receiver._spectrum_maker.frame = refuse
    run_blocks(receiver, 2)

    assert receiver.get_status() is not None, "the block died with the picture"
