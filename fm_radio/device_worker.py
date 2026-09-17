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
"""The one thread that writes to the SDR.

Every write to an RTL-SDR is a USB control transfer.  Measured on the
hardware this was built for, a tune costs 60 ms and switching the gain
mode 30 ms, and a device that has stopped answering costs as long as the
process lives.  Whoever asks for a write should not be the one paying
that: the window asks from the thread that draws it, where 60 ms is a
visible stutter and an unanswered write is a frozen application.

So nothing writes to the device except this worker.  Callers hand it what
they want done and get back a :class:`Request` they are free to ignore -
the window ignores it and shows whatever the worker reports later, the
command line waits a moment for it because somebody just typed a command
and is looking at the prompt.

Requests are coalesced by kind: only the newest tune, the newest gain and
the newest gain mode are worth making, and a burst of them from a held
button or a dragged slider collapses to the one the user ended on.  Order
between kinds is kept, so "manual gain" followed by "30 dB" still happens
that way round.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

#: What a request is about.  Two requests of the same kind are the same
#: intent expressed twice, and only the newer one is worth carrying out.
TUNE = "tune"
GAIN = "gain"
GAIN_MODE = "gain mode"

#: Longest a write may take before it is worth saying so.  Ordinary ones
#: are 30-200 ms; past this the device is struggling and the log should
#: show it next to whatever else was happening.
_SLOW_WRITE_MS: float = 250.0

#: How long the worker waits for something to do before looking at
#: whether it has been asked to stop.
_IDLE_TIMEOUT_SEC: float = 0.2

_SHUTDOWN = object()


class Request:
    """One device write, and what became of it.

    Callers may wait on it, ask it afterwards, or drop it entirely.
    """

    _counter = 0
    _counter_lock = threading.Lock()

    def __init__(self, kind: str, what: str, run: Callable[[], None]) -> None:
        with Request._counter_lock:
            Request._counter += 1
            #: Increases with every request ever made, so a display can
            #: tell a new outcome from one it has already shown.
            self.serial: int = Request._counter
        self.kind = kind
        #: How to describe this to a person, in the log and on screen.
        self.what = what
        self._run = run
        self._done: threading.Event = threading.Event()
        #: What went wrong, if anything did.
        self.error: Exception | None = None
        #: True when a newer request of the same kind replaced this one
        #: before it was carried out.
        self.superseded: bool = False

    @property
    def finished(self) -> bool:
        """True once this has been carried out, replaced, or given up on."""
        return self._done.is_set()

    @property
    def failed(self) -> bool:
        """True when the write was attempted and did not work."""
        return self.error is not None

    def wait(self, timeout: float) -> bool:
        """Wait for this to finish.  False if it has not by then."""
        return self._done.wait(timeout)

    def _finish(self, error: Exception | None = None) -> None:
        self.error = error
        self._done.set()

    def _supersede(self) -> None:
        self.superseded = True
        self._done.set()

    def __repr__(self) -> str:                  # pragma: no cover - debug
        state = ("waiting" if not self.finished else
                 "superseded" if self.superseded else
                 f"failed: {self.error}" if self.failed else "done")
        return f"<Request {self.serial} {self.what} ({state})>"


class DeviceWorker:
    """Carries out device writes so that nobody else has to wait for them.

    Args:
        logger: Where the writes and their cost are reported.
        queue_size: How many requests may be waiting.  Small on purpose:
            they coalesce by kind, so a backlog means something is very
            wrong rather than that the user is busy.
    """

    def __init__(self, logger: logging.Logger, queue_size: int = 32) -> None:
        self.logger = logger
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._stop: threading.Event = threading.Event()
        self._latest: Request | None = None
        self._latest_lock: threading.Lock = threading.Lock()
        self._thread: threading.Thread = threading.Thread(
            target=self._loop, name="DeviceWorker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Asking for something
    # ------------------------------------------------------------------

    def submit(self, kind: str, what: str,
               run: Callable[[], None]) -> Request:
        """Ask for a device write, and come straight back.

        Args:
            kind: ``TUNE``, ``GAIN`` or ``GAIN_MODE``.  A newer request of
                the same kind replaces an older one that has not been
                carried out yet.
            what: How to describe it to a person.
            run: What to do on the worker thread.  Anything it raises
                becomes the request's error.

        Returns:
            The request, which the caller is free to ignore.
        """
        request = Request(kind, what, run)
        if self._stop.is_set():
            # Nothing is going to carry this out, and a caller that
            # chooses to wait would wait for ever.
            self.logger.debug("Not %s: the device worker has stopped", what)
            request._supersede()
            return request
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            # Nothing is draining, which a coalescing queue should make
            # impossible; say so rather than block the caller, which is
            # the one thing this exists to avoid.
            self.logger.error(
                "Device request queue full; dropping %s", what)
            request._finish(RuntimeError("the device worker is not keeping up"))
            self._remember(request)
        return request

    @property
    def latest(self) -> Request | None:
        """The last request that was carried out, for anything showing it."""
        with self._latest_lock:
            return self._latest

    def _remember(self, request: Request) -> None:
        with self._latest_lock:
            self._latest = request

    # ------------------------------------------------------------------
    # Doing it
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._take_a_batch()
            if batch is None:
                continue
            if batch is _SHUTDOWN:
                break
            for request in batch:
                if self._stop.is_set():
                    request._supersede()
                    continue
                self._carry_out(request)

    def _take_a_batch(self):
        """Wait for work, then take everything else already waiting.

        Taking them together is what makes coalescing possible: five
        tunes from a held button are in the queue at once, and only the
        last of them is worth 60 ms of USB.
        """
        try:
            first = self._queue.get(timeout=_IDLE_TIMEOUT_SEC)
        except queue.Empty:
            return None
        if first is _SHUTDOWN:
            return _SHUTDOWN
        waiting = [first]
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                self._drop(waiting)
                return _SHUTDOWN
            waiting.append(item)
        return self._newest_of_each_kind(waiting)

    @staticmethod
    def _newest_of_each_kind(waiting: list) -> list:
        """Drop every request a later one of the same kind replaces.

        The survivors keep the order of those later requests, so a gain
        mode asked for before a gain still happens before it.
        """
        newest: dict[str, Request] = {}
        for request in waiting:
            newest[request.kind] = request
        keep = set(id(request) for request in newest.values())
        out = []
        for request in waiting:
            if id(request) in keep:
                out.append(request)
            else:
                request._supersede()
        return out

    @staticmethod
    def _drop(waiting: list) -> None:
        for request in waiting:
            request._supersede()

    def _carry_out(self, request: Request) -> None:
        started = time.perf_counter()
        try:
            request._run()
        except Exception as e:
            self.logger.error("%s failed: %s", request.what, e, exc_info=True)
            request._finish(e)
            self._remember(request)
            return
        took_ms = (time.perf_counter() - started) * 1000.0
        level = logging.WARNING if took_ms >= _SLOW_WRITE_MS else logging.INFO
        self.logger.log(level, "%s (USB blocked %.1fms)", request.what, took_ms)
        request._finish()
        self._remember(request)

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------

    def stop(self, timeout: float = 1.0) -> None:
        """Stop writing and let the thread go.

        Anything still queued is dropped rather than written: shutdown
        has already decided the device is going, and a tune landing on
        the way out helps nobody.  Safe to call more than once.
        """
        self._stop.set()
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._drain()

    def _drain(self) -> None:
        """Release anybody waiting on a request that will never be made."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, Request):
                item._supersede()

    @property
    def running(self) -> bool:
        """True while the worker is still able to carry anything out."""
        return self._thread.is_alive() and not self._stop.is_set()
