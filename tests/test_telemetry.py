"""Tests for the telemetry snapshot and the slot it is published through."""

from __future__ import annotations

import math
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
    publisher.publish(snapshot)
    assert publisher.latest is snapshot
    assert publisher.published_count == 1


def test_a_later_snapshot_replaces_the_earlier_one():
    """Stale state is dropped, not queued: a display wants the current value."""
    publisher = TelemetryPublisher()
    first = make_snapshot(timestamp=100.0, freq_hz=80.0e6)
    second = make_snapshot(timestamp=200.0, freq_hz=81.3e6)
    publisher.publish(first)
    publisher.publish(second)
    assert publisher.latest is second
    assert publisher.published_count == 2


def test_publishing_arms_the_next_interval():
    publisher = TelemetryPublisher(interval_sec=0.05)
    publisher.publish(make_snapshot(timestamp=10.0))
    assert not publisher.due(10.0)
    assert not publisher.due(10.049)
    assert publisher.due(10.05)
    assert publisher.due(10.2)


def test_a_zero_interval_publishes_every_block():
    publisher = TelemetryPublisher(interval_sec=0.0)
    publisher.publish(make_snapshot(timestamp=10.0))
    assert publisher.due(10.0)


def test_a_negative_interval_is_clamped():
    assert TelemetryPublisher(interval_sec=-1.0).interval_sec == 0.0


def test_reset_forgets_the_snapshot_and_publishes_again():
    publisher = TelemetryPublisher(interval_sec=10.0)
    publisher.publish(make_snapshot(timestamp=10.0))
    assert not publisher.due(10.1)
    publisher.reset()
    assert publisher.latest is None
    assert publisher.due(10.1)


def test_the_default_interval_is_no_faster_than_a_display_needs():
    # 20 Hz: a GUI polling faster sees the same snapshot twice, which is
    # cheaper than producing one it cannot use.
    assert DEFAULT_PUBLISH_INTERVAL_SEC == pytest.approx(0.05)


def test_readers_never_see_a_half_built_snapshot():
    """The slot holds an immutable object, so a reader gets all of it or none.

    Reading while a writer publishes repeatedly must never produce a
    snapshot whose fields come from different blocks.
    """
    publisher = TelemetryPublisher(interval_sec=0.0)
    for block in range(500):
        publisher.publish(make_snapshot(timestamp=float(block),
                                        freq_hz=80.0e6 + block,
                                        station=f"station-{block}"))
        current = publisher.latest
        assert current is not None
        assert current.station == f"station-{int(current.freq_hz - 80.0e6)}"
        assert current.timestamp == current.freq_hz - 80.0e6


def test_the_snapshot_holds_no_reference_into_the_receiver():
    """Plain values only: a reader cannot reach a filter or a queue."""
    for value in make_snapshot().as_dict().values():
        assert value is None or isinstance(value, (bool, int, float, str))


# ----------------------------------------------------------------------
# Cost on the realtime path
# ----------------------------------------------------------------------

def test_a_block_that_is_not_due_costs_almost_nothing():
    """Most blocks only ask whether they should publish."""
    publisher = TelemetryPublisher(interval_sec=3600.0)
    publisher.publish(make_snapshot(timestamp=time.perf_counter()))

    now = time.perf_counter()
    iterations = 20000
    start = time.perf_counter()
    for _ in range(iterations):
        publisher.due(now)
    per_call_us = (time.perf_counter() - start) / iterations * 1e6

    # The block budget is 16 ms; this has to be lost in the noise of it.
    assert per_call_us < 20.0, f"{per_call_us:.1f} us per block"


def test_taking_the_measurements_fits_well_inside_the_block_budget():
    """The IQ peak and the two audio levels are the only added work."""
    iq = (np.random.randn(16384) + 1j * np.random.randn(16384)).astype(np.complex64)
    left = np.random.randn(1536).astype(np.float32)
    right = np.random.randn(1536).astype(np.float32)

    def measure():
        return (float(np.max(np.abs(iq))), peak_dbfs(left), peak_dbfs(right))

    measure()
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        measure()
    per_snapshot_ms = (time.perf_counter() - start) / iterations * 1e3

    # A 16 ms budget, and this runs on at most one block in three.
    assert per_snapshot_ms < 1.6, f"{per_snapshot_ms:.3f} ms per snapshot"
