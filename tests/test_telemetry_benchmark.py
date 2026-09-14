"""What telemetry costs the processing thread, measured on the real path.

Marked slow and excluded from the default run. A threshold on wall-clock
time is not a correctness check: a CI runner that gets descheduled fails it
whatever the code does. These exist to produce a number on demand
(``pytest -m slow -s tests/test_telemetry_benchmark.py``) and to catch a
gross regression, so the bounds are an order of magnitude above what the
code does rather than a tight fit.

What the default run checks instead is the property that keeps the cost low:
that a block which is not due does no work. See
``test_a_block_that_is_not_due_touches_nothing_but_the_deadline`` in
tests/test_telemetry.py and ``test_a_block_that_is_not_due_builds_nothing``
in tests/test_controller_telemetry.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fm_radio.controller import FMReceiverController, _BlockProfiler
from fm_radio.telemetry import TelemetryPublisher

pytestmark = pytest.mark.slow

#: Bounds, in milliseconds, against the 16 ms per-block budget.  Measured at
#: 0.067 ms for a snapshot and 0.00018 ms for the due() check; these are far
#: enough above that only a real regression trips them.
SNAPSHOT_BUDGET_MS = 2.0
DUE_BUDGET_MS = 0.05


def measure(call, iterations: int) -> float:
    """Return the mean time of *call* in milliseconds."""
    call()                                          # warm up
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    return (time.perf_counter() - start) / iterations * 1e3


@pytest.fixture
def receiver(no_user_config):
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def test_building_a_snapshot_fits_inside_the_block_budget(receiver, capsys):
    """Measures the real _build_snapshot, not a copy of what it does."""
    size = receiver.sdr_receiver.block_size
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(size)
          + 1j * rng.standard_normal(size)).astype(np.complex64)
    left = rng.standard_normal(1536).astype(np.float32)
    right = rng.standard_normal(1536).astype(np.float32)
    profiler = _BlockProfiler(receiver.logger, 80)
    profiler.record(0.004, 1)
    now = time.perf_counter()

    per_call_ms = measure(
        lambda: receiver._build_snapshot(iq, left, right, profiler,
                                         0.004, 1, now), 300)

    with capsys.disabled():
        print(f"\n  _build_snapshot: {per_call_ms * 1e3:.1f} us "
              f"({per_call_ms / 16.0 * 100:.2f}% of a 16 ms block)")
    assert per_call_ms < SNAPSHOT_BUDGET_MS, f"{per_call_ms:.3f} ms"


def test_asking_whether_to_publish_costs_almost_nothing(capsys):
    publisher = TelemetryPublisher(interval_sec=3600.0)
    now = time.perf_counter()
    publisher.defer(now)

    per_call_ms = measure(lambda: publisher.due(now), 20000)

    with capsys.disabled():
        print(f"  due(): {per_call_ms * 1e3:.3f} us")
    assert per_call_ms < DUE_BUDGET_MS, f"{per_call_ms:.4f} ms"


def test_naming_the_station_stays_cached(receiver, capsys):
    """Without the cache this walked 983 transmitters per snapshot."""
    freq = receiver.get_frequency()
    receiver._station_name_for(freq)                # prime it

    per_call_ms = measure(lambda: receiver._station_name_for(freq), 20000)

    with capsys.disabled():
        print(f"  _station_name_for (cached): {per_call_ms * 1e3:.3f} us")
    assert per_call_ms < DUE_BUDGET_MS, f"{per_call_ms:.4f} ms"
