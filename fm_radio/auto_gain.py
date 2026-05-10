"""Automatic hardware gain controller for RTL-SDR.

Monitors IQ sample peak amplitude and adjusts the RTL-SDR hardware
gain register to avoid ADC clipping while maintaining good signal
strength.
"""

from __future__ import annotations

import queue
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
    AGC_WARMUP_SEC,
)

if TYPE_CHECKING:
    from fm_radio.sdr_receiver import SDRReceiver


# Sentinel placed in the gain-request queue to signal worker shutdown.
_GAIN_WORKER_SHUTDOWN = object()


class AutoGainController:
    """Software-controlled automatic gain using RTL-SDR manual gain steps.

    Replaces hardware AGC with a monitoring loop that runs in the
    processing thread.  When IQ samples clip (peak > threshold) for
    several consecutive blocks, gain is stepped down.  When the signal
    is weak for an extended period, gain is stepped up.

    Thread safety: a Lock protects all mutable state.  Hardware
    ``sdr_receiver.set_gain()`` calls triggered from ``update()`` are
    dispatched to a dedicated worker thread; the synchronous USB
    control transfer (~40-200 ms per call) therefore never blocks the
    processing thread.  Calls from the CLI/controller thread
    (``enable``, ``disable``, ``reset_counters``, ``set_gain_manual``)
    remain synchronous since they are not on a real-time path.
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
        # Suppress AGC during the warmup window (Numba JIT compile and
        # filter settling can otherwise produce spurious fast firings).
        self._start_time: float = time.perf_counter()

        # Async gain-change worker: own all USB control transfers
        # triggered from the real-time processing thread.
        self._gain_q: queue.Queue[object] = queue.Queue(maxsize=4)
        self._gain_worker_stop: threading.Event = threading.Event()
        # Last value successfully applied to the SDR (in dB).  Updated
        # by the worker after each set_gain call and read by disable()
        # to pin the gain at "current" rather than letting an in-flight
        # AGC request continue to land afterwards.
        self._last_applied_gain: float = AGC_GAIN_TABLE[self._gain_index] / 10.0
        self._gain_worker: threading.Thread = threading.Thread(
            target=self._gain_worker_loop,
            name='AutoGainWorker',
            daemon=True,
        )
        self._gain_worker.start()

        # Apply initial gain: switch to manual gain mode
        self._sdr.set_manual_gain_mode(True)
        self._sdr.set_gain(self._last_applied_gain)

    # ------------------------------------------------------------------
    # Async USB worker (avoids blocking the processing thread)
    # ------------------------------------------------------------------

    def _submit_async_gain(self, gain_db: float) -> None:
        """Enqueue a gain-change request for the worker thread.

        Coalesces by draining stale pending requests so that only the
        most recent gain decision survives.  This way a burst of AGC
        firings collapses to whatever the final desired level is.
        """
        try:
            while True:
                self._gain_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._gain_q.put_nowait(float(gain_db))
        except queue.Full:
            self.logger.debug("Gain request queue full (should not happen)")

    def _gain_worker_loop(self) -> None:
        """Worker thread loop: applies pending gain requests via USB."""
        while not self._gain_worker_stop.is_set():
            try:
                item = self._gain_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _GAIN_WORKER_SHUTDOWN:
                break
            gain_db = float(item)
            t0 = time.perf_counter()
            try:
                self._sdr.set_gain(gain_db)
            except Exception as e:
                self.logger.error(
                    f"Failed to apply gain {gain_db:.1f} dB: {e}",
                    exc_info=True,
                )
                continue
            with self._lock:
                self._last_applied_gain = gain_db
            dt_ms = (time.perf_counter() - t0) * 1000.0
            level = logging.WARNING if dt_ms >= 50.0 else logging.INFO
            self.logger.log(
                level,
                "Auto gain applied %.1f dB (USB blocked %.1fms, async)",
                gain_db, dt_ms,
            )

    def stop(self) -> None:
        """Stop the async gain worker thread (called from cleanup)."""
        self._gain_worker_stop.set()
        try:
            self._gain_q.put_nowait(_GAIN_WORKER_SHUTDOWN)
        except queue.Full:
            pass
        if self._gain_worker.is_alive():
            self._gain_worker.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Public API (called from CLI / controller thread)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return True if auto gain control is active."""
        with self._lock:
            return self._enabled

    def enable(self) -> None:
        """Enable auto gain control (replaces hardware AGC).

        The desired gain is routed through ``_submit_async_gain`` so it
        cannot race against any USB transfer that might still be in
        flight in the worker.
        """
        # set_manual_gain_mode is only ever called on enable/disable
        # transitions (not on the realtime path) so a synchronous USB
        # call here is acceptable.
        self._sdr.set_manual_gain_mode(True)
        with self._lock:
            self._enabled = True
            self._clip_counter = 0
            self._weak_counter = 0
            self._holdoff = 0
            # Submit inside the lock for serialisation with update()
            # and disable() — see notes there.
            self._submit_async_gain(AGC_GAIN_TABLE[self._gain_index] / 10.0)
        self.logger.info("Auto gain control enabled")

    def disable(self, manual_gain_db: float | None = None) -> None:
        """Disable auto gain, optionally setting a fixed gain.

        After this call returns the device gain is pinned: any AGC
        request still in flight or queued by the worker is overridden.
        When ``manual_gain_db`` is None the gain is pinned at the
        last value the worker successfully applied to the SDR — this
        is what ``agc off`` from the CLI expects, so the gain does
        not silently change after the user disables AGC.

        Args:
            manual_gain_db: If provided, set this as the fixed manual
                gain (in dB).  If ``None``, pin to the last applied
                gain.
        """
        with self._lock:
            self._enabled = False
            self._clip_counter = 0
            self._weak_counter = 0
            self._holdoff = 0
            final_gain = (
                float(manual_gain_db) if manual_gain_db is not None
                else float(self._last_applied_gain)
            )
            # Submit while still holding the lock so this request is
            # ordered after any AGC submission from update() that ran
            # earlier (update() also submits inside the same lock) —
            # the worker will therefore drain the AGC value and end
            # at final_gain.
            self._submit_async_gain(final_gain)
        self.logger.info(
            "Auto gain control disabled, gain pinned at %.1f dB", final_gain,
        )

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
            # Submit inside the lock to serialise with update() etc.
            self._submit_async_gain(gain_db)

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
                # Submit inside the lock for serialisation with
                # update()/disable()/set_gain_manual()/enable().
                self._submit_async_gain(apply_gain)
        if apply_gain is not None:
            self.logger.info(f"Gain reset to default {apply_gain:.1f} dB for new station")

    # ------------------------------------------------------------------
    # Processing loop (called from processing_thread)
    # ------------------------------------------------------------------

    def update(self, iq_samples: np.ndarray) -> None:
        """Monitor IQ peak amplitude and adjust gain if needed.

        Called once per IQ block from the processing thread, before
        demodulation.  Gain change USB calls are dispatched to the
        async worker; this method itself is real-time safe.

        Args:
            iq_samples: Raw complex64 IQ samples from SDR.
        """
        apply_gain: float | None = None
        applied_step: int = 0

        with self._lock:
            if not self._enabled:
                return

            # Suppress AGC during startup warmup so Numba JIT compile
            # transients and filter settling don't trigger spurious
            # gain steps.
            if (time.perf_counter() - self._start_time) < AGC_WARMUP_SEC:
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
                    applied_step = self._gain_index
            elif peak < AGC_WEAK_THRESHOLD:
                self._weak_counter += 1
                self._clip_counter = 0
                if (self._weak_counter >= AGC_WEAK_COUNT
                        and self._gain_index < len(AGC_GAIN_TABLE) - 1):
                    self._gain_index += 1
                    self._weak_counter = 0
                    self._holdoff = AGC_HOLDOFF_BLOCKS
                    apply_gain = AGC_GAIN_TABLE[self._gain_index] / 10.0
                    applied_step = self._gain_index
            else:
                self._clip_counter = 0
                self._weak_counter = 0

            # Submit *inside* the lock so AGC's request is serialised
            # with disable()/set_gain_manual()/reset_counters(), all of
            # which also submit while holding self._lock.  Without this,
            # an AGC request decided just before disable() could land
            # in the worker queue *after* disable()'s pin and silently
            # change the gain post-disable.
            if apply_gain is not None:
                self._submit_async_gain(apply_gain)

        # Logging is outside the lock to avoid pulling logging-handler
        # latency under the realtime path.
        if apply_gain is not None:
            self.logger.info(
                "Auto gain requested %.1f dB (step %d/%d, async)",
                apply_gain, applied_step, len(AGC_GAIN_TABLE) - 1,
            )
