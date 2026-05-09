"""Automatic hardware gain controller for RTL-SDR.

Monitors IQ sample peak amplitude and adjusts the RTL-SDR hardware
gain register to avoid ADC clipping while maintaining good signal
strength.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import TYPE_CHECKING

import numpy as np

from fm_radio.constants import (
    AGC_GAIN_TABLE,
    AGC_DEFAULT_GAIN_INDEX,
    AGC_CLIP_THRESHOLD,
    AGC_WEAK_THRESHOLD,
    AGC_CLIP_COUNT,
    AGC_WEAK_COUNT,
    AGC_HOLDOFF_BLOCKS,
)

if TYPE_CHECKING:
    from fm_radio.sdr_receiver import SDRReceiver


class AutoGainController:
    """Software-controlled automatic gain using RTL-SDR manual gain steps.

    Replaces hardware AGC with a monitoring loop that runs in the
    processing thread.  When IQ samples clip (peak > threshold) for
    several consecutive blocks, gain is stepped down.  When the signal
    is weak for an extended period, gain is stepped up.

    Thread safety: a Lock protects all mutable state.  The actual
    ``sdr_receiver.set_gain()`` call is made outside the lock.
    """

    def __init__(self, sdr_receiver: SDRReceiver) -> None:
        self.logger = logging.getLogger('fm_receiver.AutoGainController')
        self._sdr = sdr_receiver
        self._lock = threading.Lock()

        # State
        self._enabled: bool = True
        self._gain_index: int = AGC_DEFAULT_GAIN_INDEX
        self._clip_counter: int = 0
        self._weak_counter: int = 0
        self._holdoff: int = 0

        # Apply initial gain: switch to manual gain mode
        self._sdr.set_manual_gain_mode(True)
        self._sdr.set_gain(AGC_GAIN_TABLE[self._gain_index] / 10.0)

    # ------------------------------------------------------------------
    # Public API (called from CLI / controller thread)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return True if auto gain control is active."""
        with self._lock:
            return self._enabled

    def enable(self) -> None:
        """Enable auto gain control (replaces hardware AGC)."""
        with self._lock:
            self._enabled = True
            self._clip_counter = 0
            self._weak_counter = 0
            self._holdoff = 0
        self._sdr.set_manual_gain_mode(True)
        self._sdr.set_gain(AGC_GAIN_TABLE[self._gain_index] / 10.0)
        self.logger.info("Auto gain control enabled")

    def disable(self, manual_gain_db: float | None = None) -> None:
        """Disable auto gain, optionally setting a fixed gain.

        Args:
            manual_gain_db: If provided, set this as the fixed manual
                gain (in dB).  If ``None``, keep current gain level.
        """
        with self._lock:
            self._enabled = False
            self._clip_counter = 0
            self._weak_counter = 0
            self._holdoff = 0
        if manual_gain_db is not None:
            self._sdr.set_gain(manual_gain_db)
        self.logger.info("Auto gain control disabled")

    def set_gain_manual(self, gain_db: float) -> None:
        """Set gain explicitly (for CLI ``gain <value>`` command).

        Only effective when auto gain is disabled.  Updates the
        internal gain index to the nearest valid step.

        Args:
            gain_db: Desired gain in dB.
        """
        with self._lock:
            if self._enabled:
                return
            # Snap to nearest valid gain step
            gain_tenths = int(round(gain_db * 10))
            best_idx = 0
            best_dist = abs(AGC_GAIN_TABLE[0] - gain_tenths)
            for i, g in enumerate(AGC_GAIN_TABLE):
                d = abs(g - gain_tenths)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            self._gain_index = best_idx
        self._sdr.set_gain(gain_db)

    def reset_counters(self) -> None:
        """Reset gain to default and clear counters (e.g., after tuning).

        Restores the gain index to the default level so the auto gain
        loop re-adjusts from scratch for the new station.
        """
        apply_gain: float | None = None
        with self._lock:
            self._clip_counter = 0
            self._weak_counter = 0
            self._holdoff = 0
            if self._enabled and self._gain_index != AGC_DEFAULT_GAIN_INDEX:
                self._gain_index = AGC_DEFAULT_GAIN_INDEX
                apply_gain = AGC_GAIN_TABLE[self._gain_index] / 10.0
        if apply_gain is not None:
            self._sdr.set_gain(apply_gain)
            self.logger.info(f"Gain reset to default {apply_gain:.1f} dB for new station")

    # ------------------------------------------------------------------
    # Processing loop (called from processing_thread)
    # ------------------------------------------------------------------

    def update(self, iq_samples: np.ndarray) -> None:
        """Monitor IQ peak amplitude and adjust gain if needed.

        Called once per IQ block from the processing thread, before
        demodulation.

        Args:
            iq_samples: Raw complex64 IQ samples from SDR.
        """
        apply_gain: float | None = None

        with self._lock:
            if not self._enabled:
                return

            if self._holdoff > 0:
                self._holdoff -= 1
                return

            peak = float(np.max(np.abs(iq_samples)))

            if peak > AGC_CLIP_THRESHOLD:
                self._clip_counter += 1
                self._weak_counter = 0
                if self._clip_counter >= AGC_CLIP_COUNT and self._gain_index > 0:
                    self._gain_index -= 1
                    self._clip_counter = 0
                    self._holdoff = AGC_HOLDOFF_BLOCKS
                    apply_gain = AGC_GAIN_TABLE[self._gain_index] / 10.0
            elif peak < AGC_WEAK_THRESHOLD:
                self._weak_counter += 1
                self._clip_counter = 0
                if (self._weak_counter >= AGC_WEAK_COUNT
                        and self._gain_index < len(AGC_GAIN_TABLE) - 1):
                    self._gain_index += 1
                    self._weak_counter = 0
                    self._holdoff = AGC_HOLDOFF_BLOCKS
                    apply_gain = AGC_GAIN_TABLE[self._gain_index] / 10.0
            else:
                self._clip_counter = 0
                self._weak_counter = 0

        # Apply gain change outside the lock.
        # The RTL-SDR set_gain() is a synchronous USB control transfer
        # which can block 50-200 ms; this blocking happens on the
        # processing thread so we instrument it to identify whether it
        # is the cause of audio dropouts.
        if apply_gain is not None:
            t0 = time.perf_counter()
            self._sdr.set_gain(apply_gain)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            level = (
                logging.WARNING if dt_ms >= 20.0 else logging.INFO
            )
            self.logger.log(
                level,
                "Auto gain adjusted to %.1f dB (step %d/%d) "
                "set_gain_blocked=%.1fms",
                apply_gain,
                self._gain_index,
                len(AGC_GAIN_TABLE) - 1,
                dt_ms,
            )
