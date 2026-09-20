"""Tests for the telemetry snapshot and the slot it is published through."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fm_radio.telemetry import (
    DEFAULT_PUBLISH_INTERVAL_SEC,
    SILENCE_DBFS,
    StatusSnapshot,
    TelemetryPublisher,
    peak_dbfs,
    to_dbfs,
)


def make_snapshot(**overrides) -> StatusSnapshot:
    """A snapshot with plausible values, so a test can vary one field."""
    defaults = dict(
        freq_hz=80.0e6, station="TOKYO FM", gain_db=28.0, auto_gain=True,
        iq_peak=0.62,
        stereo=True, blend_factor=1.0, pilot_snr_db=19.6, pilot_jitter_db=0.8,
        side_nr_enabled=True,
        am_depth=0.06,
        level_left_dbfs=-6.2, level_right_dbfs=-7.8,
        block_ms=4.2, block_ms_avg=4.0, block_ms_max=6.1, block_budget_ms=16.0,
        sdr_queue=1, sdr_queue_max=80, slow_blocks=0,
        iq_drops=0, audio_drops=0, audio_underruns=0,
        recording_audio=False, recording_iq=False,
        uptime_sec=767.0, timestamp=time.perf_counter(),
    )
    defaults.update(overrides)
    return StatusSnapshot(**defaults)


# ----------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------

def test_full_scale_is_zero_dbfs():
    assert to_dbfs(1.0) == pytest.approx(0.0)


def test_half_scale_is_about_minus_six_db():
    assert to_dbfs(0.5) == pytest.approx(-6.02, abs=0.01)


@pytest.mark.parametrize("value", [0.0, -0.5, float("nan"), float("-inf")])
def test_unusable_amplitudes_read_as_silence(value):
    assert to_dbfs(value) == SILENCE_DBFS


def test_very_quiet_is_floored_rather_than_minus_infinity():
    assert to_dbfs(1e-30) == SILENCE_DBFS


def test_peak_dbfs_uses_the_largest_magnitude():
    samples = np.array([0.1, -0.5, 0.25], dtype=np.float32)
    assert peak_dbfs(samples) == pytest.approx(to_dbfs(0.5))


def test_peak_dbfs_of_an_empty_block_is_silence():
    assert peak_dbfs(np.array([], dtype=np.float32)) == SILENCE_DBFS


# ----------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------

def test_a_snapshot_cannot_be_modified():
    snapshot = make_snapshot()
    with pytest.raises(Exception):
        snapshot.freq_hz = 81.3e6       # frozen dataclass


def test_healthy_while_inside_budget_and_nothing_dropped():
    assert make_snapshot().healthy


@pytest.mark.parametrize("overrides", [
    {"block_ms": 17.0},                 # over budget
    {"iq_drops": 1},
    {"audio_drops": 1},
])
def test_not_healthy_once_something_is_wrong(overrides):
    assert not make_snapshot(**overrides).healthy


@pytest.mark.parametrize("blend,expected", [(1.0, True), (0.51, True),
                                            (0.5, False), (0.0, False)])
def test_stereo_locked_follows_the_blend(blend, expected):
    assert make_snapshot(blend_factor=blend).stereo_locked is expected


@pytest.mark.parametrize("blend", [1.0, 0.6, 0.0])
def test_a_mono_snapshot_is_never_stereo_locked(blend):
    """blend_factor starts at 1.0 and the mono path never moves it."""
    assert make_snapshot(stereo=False, blend_factor=blend).stereo_locked is False


def test_as_dict_round_trips_every_field():
    snapshot = make_snapshot()
    assert StatusSnapshot(**snapshot.as_dict()) == snapshot


def test_pilot_snr_may_be_unknown():
    """A mono demodulator, or the first blocks after tuning, has no pilot."""
    assert make_snapshot(pilot_snr_db=None).pilot_snr_db is None


# ----------------------------------------------------------------------
# Publisher
# ----------------------------------------------------------------------

def test_nothing_is_published_before_the_first_block():
    assert TelemetryPublisher().latest is None


def test_the_first_block_is_always_due():
    assert TelemetryPublisher().due(time.perf_counter())


def test_publishing_makes_the_snapshot_readable():
    publisher = TelemetryPublisher()
    snapshot = make_snapshot()
    publisher.publish(snapshot, publisher.generation)
    assert publisher.latest is snapshot
    assert publisher.published_count == 1


def test_a_later_snapshot_replaces_the_earlier_one():
    """Stale state is dropped, not queued: a display wants the current value."""
    publisher = TelemetryPublisher()
    first = make_snapshot(timestamp=100.0, freq_hz=80.0e6)
    second = make_snapshot(timestamp=200.0, freq_hz=81.3e6)
    publisher.publish(first, publisher.generation)
    publisher.publish(second, publisher.generation)
    assert publisher.latest is second
    assert publisher.published_count == 2


def test_publishing_arms_the_next_interval():
    publisher = TelemetryPublisher(interval_sec=0.05)
    publisher.publish(make_snapshot(timestamp=10.0), publisher.generation)
    assert not publisher.due(10.0)
    assert not publisher.due(10.049)
    assert publisher.due(10.05)
    assert publisher.due(10.2)


def test_a_zero_interval_publishes_every_block():
    publisher = TelemetryPublisher(interval_sec=0.0)
    publisher.publish(make_snapshot(timestamp=10.0), publisher.generation)
    assert publisher.due(10.0)


def test_a_negative_interval_is_clamped():
    assert TelemetryPublisher(interval_sec=-1.0).interval_sec == 0.0


# ----------------------------------------------------------------------
# Generations
# ----------------------------------------------------------------------

def tunable(interval_sec: float = 0.0):
    """A publisher whose generation the test can move, as retuning does."""
    tuning = {"generation": 0}
    publisher = TelemetryPublisher(
        interval_sec=interval_sec,
        current_generation=lambda: tuning["generation"])
    return publisher, tuning


def test_a_snapshot_from_a_previous_generation_is_not_handed_out():
    """The race the generation exists for: capture, retune, then publish.

    A snapshot from samples captured before a retune would otherwise
    resurface as the current state of a station the receiver has left.
    """
    publisher, tuning = tunable()
    captured_under = publisher.generation       # the block's own generation
    tuning["generation"] += 1                   # the tuner moves
    publisher.publish(make_snapshot(), captured_under)

    assert publisher.latest is None
    assert publisher.published_count == 1       # it was published, not hidden


def test_a_snapshot_from_the_current_generation_is_handed_out():
    publisher, tuning = tunable()
    tuning["generation"] += 1
    snapshot = make_snapshot()
    publisher.publish(snapshot, publisher.generation)
    assert publisher.latest is snapshot


def test_a_published_snapshot_is_hidden_by_a_later_retune():
    publisher, tuning = tunable()
    publisher.publish(make_snapshot(), publisher.generation)
    assert publisher.latest is not None
    tuning["generation"] += 1
    assert publisher.latest is None


def test_the_generation_comes_from_the_supplied_source():
    publisher, tuning = tunable()
    tuning["generation"] = 7
    assert publisher.generation == 7


def test_deferring_arms_the_interval_without_publishing():
    """A snapshot that could not be built must still cost its interval."""
    publisher = TelemetryPublisher(interval_sec=0.05)
    publisher.defer(10.0)
    assert publisher.latest is None
    assert publisher.published_count == 0
    assert not publisher.due(10.049)
    assert publisher.due(10.05)


def test_the_default_interval_is_no_faster_than_a_display_needs():
    # 20 Hz: a GUI polling faster sees the same snapshot twice, which is
    # cheaper than producing one it cannot use.
    assert DEFAULT_PUBLISH_INTERVAL_SEC == pytest.approx(0.05)


def test_each_published_snapshot_is_internally_consistent():
    """Single-threaded: what is read back belongs to one block, not several.

    This says nothing about concurrency — the reader runs after the writer
    has finished.  See the test below for a reader on its own thread.
    """
    publisher = TelemetryPublisher(interval_sec=0.0)
    for block in range(500):
        publisher.publish(make_snapshot(timestamp=float(block),
                                        freq_hz=80.0e6 + block,
                                        station=f"station-{block}"),
                          publisher.generation)
        current = publisher.latest
        assert current is not None
        assert current.station == f"station-{int(current.freq_hz - 80.0e6)}"
        assert current.timestamp == current.freq_hz - 80.0e6


def test_a_concurrent_reader_only_ever_sees_whole_snapshots():
    """A reader thread running against a writer never sees a mixed snapshot.

    This exercises the slot under concurrency; it does not prove anything
    about the interpreter.  Passing means no inconsistent snapshot was
    observed in this run, which is why the reader's observation count is
    asserted: a reader that never ran would otherwise pass trivially.
    """
    publisher = TelemetryPublisher(interval_sec=0.0)
    publisher.publish(make_snapshot(timestamp=0.0, freq_hz=80.0e6,
                                    station="station-0"), publisher.generation)

    reading = threading.Event()
    saw_an_update = threading.Event()
    stop = threading.Event()
    observations = []
    inconsistent = []

    def reader():
        while not stop.is_set():
            current = publisher.latest
            if current is None:
                continue
            observations.append(1)
            reading.set()                       # the writer waits for this
            block = int(current.freq_hz - 80.0e6)
            if (current.station != f"station-{block}"
                    or current.timestamp != float(block)):
                inconsistent.append(current)
            if block > 0:
                saw_an_update.set()             # ... and for this

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    # Start publishing only once the reader has actually read something.
    # Waiting on a barrier instead would let the writer run to completion
    # before the reader is ever scheduled - which it does, deterministically,
    # with a long switch interval - and the test would then pass or fail on
    # how the interpreter happened to interleave rather than on the slot.
    assert reading.wait(10), "the reader never observed a snapshot"

    for block in range(1, 20000):
        publisher.publish(make_snapshot(timestamp=float(block),
                                        freq_hz=80.0e6 + block,
                                        station=f"station-{block}"),
                          publisher.generation)
        # Reading the initial snapshot 400,000 times says nothing about
        # reading one the writer produced.  Hand the reader the interpreter
        # until it has seen an update; wait() releases the GIL, which a busy
        # loop at a one-second switch interval would not.
        if not saw_an_update.is_set() and block % 500 == 0:
            saw_an_update.wait(0.05)
    stop.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert observations, "the reader never read"
    assert saw_an_update.is_set(), "the reader only ever saw the first snapshot"
    assert not inconsistent, f"{len(inconsistent)} mixed snapshots"


def test_a_concurrent_retune_hides_an_in_flight_snapshot():
    """Retuning while a snapshot is in flight must not publish it as current.

    The ordering is forced with events rather than left to chance: the
    snapshot is built, the tuner moves, and only then is it published.
    """
    publisher, tuning = tunable()
    built = threading.Event()
    retuned = threading.Event()

    def writer():
        generation = publisher.generation       # the block's generation
        snapshot = make_snapshot(station="old station")
        built.set()
        retuned.wait(5)                         # the tuner moves meanwhile
        publisher.publish(snapshot, generation)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    assert built.wait(5)
    tuning["generation"] += 1
    retuned.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert publisher.published_count == 1
    assert publisher.latest is None


def test_the_snapshot_holds_no_reference_into_the_receiver():
    """Plain values only: a reader cannot reach a filter or a queue."""
    for value in make_snapshot().as_dict().values():
        assert value is None or isinstance(value, (bool, int, float, str))


# ----------------------------------------------------------------------
# Cost on the realtime path
#
# Timing assertions live in tests/test_telemetry_benchmark.py, which is
# marked slow and excluded from the default run: a threshold on wall-clock
# time fails on a CI runner that was descheduled, whatever the code does.
# What is checked here is the property that makes the cost low - that a
# block which is not due does no work at all - which is deterministic.
# ----------------------------------------------------------------------

def test_a_block_that_is_not_due_touches_nothing_but_the_deadline():
    publisher = TelemetryPublisher(interval_sec=3600.0)
    publisher.publish(make_snapshot(timestamp=10.0), publisher.generation)

    before = publisher.latest
    for block in range(1000):
        assert not publisher.due(10.0 + block * 0.016)

    assert publisher.latest is before
    assert publisher.published_count == 1
