#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) [2025] FortuneOfLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Receiver state, published for something outside the realtime path to read.

The processing thread has a 16 ms budget per IQ block and everything worth
displaying already exists inside it — pilot SNR, the stereo blend, queue
depths, block timing.  Until now the only way to see any of it was to turn
logging on and read the log afterwards.

:class:`StatusSnapshot` is an immutable picture of that state and
:class:`TelemetryPublisher` is the one-writer, many-reader slot it is handed
through.  The contract is deliberately one-way: readers never reach into the
receiver, and the receiver never waits on a reader.

Cost.  Publishing is rate-limited (:data:`DEFAULT_PUBLISH_INTERVAL_SEC`,
20 Hz) because nothing watching this can use more.  Measured on the light
demodulator: a block that is not due pays 0.18 us for the comparison, and
building a snapshot costs 67 us — 0.4% of the 16 ms budget — which averages
to 0.02 ms per block, or 0.14% of it.  Nothing here allocates per block
beyond the snapshot itself.

Naming the tuned station is deliberately not part of that: scanning the 983
transmitters in the catalogue took 270 us of an early version's 322 us, so
the caller caches it against the frequency instead.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

#: How often the processing thread publishes.  A GUI polling faster than this
#: sees the same snapshot twice, which is cheaper than producing one it will
#: not use.
DEFAULT_PUBLISH_INTERVAL_SEC: float = 0.05

#: Level reported for digital silence, in dBFS.  Anything quieter than this
#: is not worth distinguishing on a meter.
SILENCE_DBFS: float = -120.0


def to_dbfs(amplitude: float) -> float:
    """Convert a linear amplitude in [0, 1] to dBFS, floored at silence."""
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(amplitude))


def peak_dbfs(samples: np.ndarray) -> float:
    """Peak level of *samples* in dBFS, or silence for an empty block."""
    if samples.size == 0:
        return SILENCE_DBFS
    return to_dbfs(float(np.max(np.abs(samples))))


@dataclass(frozen=True)
class StatusSnapshot:
    """Everything the receiver knows about itself at one instant.

    Frozen, and built from plain floats, ints and strings rather than from
    references into the receiver: a reader cannot reach back into a filter or
    a queue through it, and cannot see a half-updated value.

    ``pilot_snr_db`` is None before the stereo path has measured a pilot —
    on a mono demodulator, or in the first blocks after tuning.
    """

    # --- tuner -------------------------------------------------------
    freq_hz: float
    station: str                        # catalogue name, "" if unknown
    gain_db: float
    auto_gain: bool
    iq_peak: float                      # 0..1, clipping at 1

    # --- demodulation ------------------------------------------------
    stereo: bool                        # stereo requested
    blend_factor: float                 # 0 = mono, 1 = full stereo
    pilot_snr_db: float | None
    pilot_jitter_db: float
    side_nr_enabled: bool

    # --- audio -------------------------------------------------------
    level_left_dbfs: float
    level_right_dbfs: float

    # --- health ------------------------------------------------------
    block_ms: float                     # the block this snapshot came from
    block_ms_avg: float                 # since the last profiler summary
    block_ms_max: float
    block_budget_ms: float
    sdr_queue: int
    sdr_queue_max: int
    slow_blocks: int                    # cumulative, over the budget
    iq_drops: int                       # IQ blocks the SDR callback dropped
    audio_drops: int                    # audio blocks dropped on enqueue
    audio_underruns: int                # times the output ran dry

    # --- recording ---------------------------------------------------
    recording_audio: bool
    recording_iq: bool

    # --- bookkeeping -------------------------------------------------
    uptime_sec: float
    timestamp: float                    # time.perf_counter() at capture

    @property
    def healthy(self) -> bool:
        """True while the block time is inside its budget and nothing dropped."""
        return (self.block_ms <= self.block_budget_ms
                and self.iq_drops == 0
                and self.audio_drops == 0)

    @property
    def stereo_locked(self) -> bool:
        """True when the blend is open far enough to call the output stereo."""
        return self.blend_factor > 0.5

    def as_dict(self) -> dict[str, Any]:
        """Return the snapshot as plain values, for logging or serialising."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


class TelemetryPublisher:
    """A latest-value slot between the processing thread and its readers.

    One writer and any number of readers.  Publishing rebinds a single
    attribute to an immutable object, which is atomic under the GIL, so there
    is no lock on either side: a reader gets some complete snapshot, never a
    torn one, and a writer is never delayed by a reader that stopped reading.

    Snapshots are dropped rather than queued.  What a display wants is the
    current state, and a backlog of stale ones would only grow when the
    reader is already behind.
    """

    def __init__(self, interval_sec: float = DEFAULT_PUBLISH_INTERVAL_SEC) -> None:
        self.interval_sec: float = max(0.0, float(interval_sec))
        self._latest: StatusSnapshot | None = None
        self._next_due: float = 0.0
        self._published: int = 0

    def due(self, now: float | None = None) -> bool:
        """True if a snapshot should be published at *now*.

        Called once per block on the realtime path, so it is deliberately one
        comparison against a precomputed deadline.
        """
        return (now if now is not None else time.perf_counter()) >= self._next_due

    def publish(self, snapshot: StatusSnapshot) -> None:
        """Store *snapshot* as the current state and arm the next interval."""
        self._latest = snapshot
        self._published += 1
        self._next_due = snapshot.timestamp + self.interval_sec

    @property
    def latest(self) -> StatusSnapshot | None:
        """The most recent snapshot, or None before the first block."""
        return self._latest

    @property
    def published_count(self) -> int:
        """How many snapshots have been published, for tests and diagnostics."""
        return self._published

    def reset(self) -> None:
        """Forget the current snapshot and publish again on the next block."""
        self._latest = None
        self._next_due = 0.0
