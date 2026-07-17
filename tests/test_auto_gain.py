"""AutoGainController async-worker behaviour.

Covers the audio-dropout fixes: the realtime update() path must never
block on USB, stale AGC requests must not override manual settings, and
`agc off` must pin the gain at its current value.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fm_radio.auto_gain import AutoGainController
from fm_radio.constants import AGC_GAIN_TABLE, AGC_DEFAULT_GAIN_INDEX


USB_STALL_S = 0.15
DEFAULT_GAIN_DB = AGC_GAIN_TABLE[AGC_DEFAULT_GAIN_INDEX] / 10.0
CLIP_BLOCK = np.ones(1024, dtype=np.complex64) * 0.99


class SlowMockSdr:
    """SDR stub whose set_gain simulates a slow USB control transfer."""

    def __init__(self, stall_s: float = USB_STALL_S) -> None:
        self.stall_s = stall_s
        self.calls: list[float] = []

    def set_manual_gain_mode(self, manual: bool) -> None: ...

    def set_gain(self, gain_db: float) -> None:
        time.sleep(self.stall_s)
        self.calls.append(gain_db)


@pytest.fixture
def agc_pair():
    sdr = SlowMockSdr()
    agc = AutoGainController(sdr)
    yield sdr, agc
    agc.stop()


def _skip_warmup(agc: AutoGainController) -> None:
    agc._start_time = time.perf_counter() - 60.0


def _wait_idle(sdr: SlowMockSdr, timeout_s: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout_s
    stable_count = None
    while time.perf_counter() < deadline:
        if stable_count == len(sdr.calls):
            return
        stable_count = len(sdr.calls)
        time.sleep(3 * USB_STALL_S)


def test_update_never_blocks_on_usb(agc_pair):
    sdr, agc = agc_pair
    _skip_warmup(agc)
    worst = 0.0
    for _ in range(10):
        t0 = time.perf_counter()
        agc.update(CLIP_BLOCK)
        worst = max(worst, time.perf_counter() - t0)
    # Sync implementation would take >= USB_STALL_S on the firing call.
    assert worst < USB_STALL_S / 2


def test_agc_steps_down_on_sustained_clipping(agc_pair):
    sdr, agc = agc_pair
    _skip_warmup(agc)
    for _ in range(5):
        agc.update(CLIP_BLOCK)
    _wait_idle(sdr)
    expected = AGC_GAIN_TABLE[AGC_DEFAULT_GAIN_INDEX - 1] / 10.0
    assert sdr.calls[-1] == pytest.approx(expected)


def test_disable_without_argument_pins_current_gain(agc_pair):
    sdr, agc = agc_pair
    _skip_warmup(agc)
    for _ in range(3):  # queue an AGC step-down request
        agc.update(CLIP_BLOCK)
    agc.disable()  # immediately: pending request must not win
    _wait_idle(sdr)
    assert sdr.calls[-1] == pytest.approx(DEFAULT_GAIN_DB)


def test_disable_with_manual_gain_wins_over_pending_agc(agc_pair):
    sdr, agc = agc_pair
    _skip_warmup(agc)
    for _ in range(3):
        agc.update(CLIP_BLOCK)
    agc.disable(manual_gain_db=49.6)
    _wait_idle(sdr)
    assert sdr.calls[-1] == pytest.approx(49.6)


def test_warmup_suppresses_agc(agc_pair):
    sdr, agc = agc_pair  # fresh controller: warmup window active
    calls_before = len(sdr.calls)
    for _ in range(10):
        agc.update(CLIP_BLOCK)
    time.sleep(3 * USB_STALL_S)
    assert len(sdr.calls) == calls_before


def test_stop_joins_worker(agc_pair):
    sdr, agc = agc_pair
    agc.stop()
    assert not agc._gain_worker.is_alive()
