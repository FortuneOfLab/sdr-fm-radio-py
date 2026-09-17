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
"""Ownership of one RTL-SDR device handle: its state, and its closing.

Everything about when the handle may be touched and when it is freed lives
here, in one object, rather than spread through the receiver that uses it.
That matters because the answer is not simple: pyrtlsdr closes the device
itself whenever a control write or the async read fails, from whichever
thread made the call, and librtlsdr keeps using the handle for as long as
``rtlsdr_read_async`` has not returned.  A close is therefore never just a
close, and having one object answer for it is what keeps the rules in one
readable place.

The rules, in short:

* Nothing touches the device once anybody has decided it is going, whether
  or not the close itself has happened yet (:attr:`DeviceHandle.usable`).
* A write holds :attr:`device_lock` for its whole USB call, and a close
  waits for that - so the handle is never freed under a write.
* The C cancel holds :attr:`handle_lock` for its call, and so does the
  close - so the pointer is never freed between a liveness check and the
  call that uses it.
* A close asked for while the read is running cancels the read and waits
  for it, bounded.  One asked for *by* the read - pyrtlsdr closing after a
  failed read, or the SDR callback asking to stop - cannot wait for it, so
  it only asks for the cancel and leaves the close to
  :meth:`DeviceHandle.finished_reading`.
* Every one of those waits can run out, and when it does the close is not
  made.  It is remembered, and the next operation to finish with the
  device makes it instead.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

try:
    # The C cancel, reached without pyrtlsdr's wrapper around it.  The
    # wrapper closes the device and raises when the call fails, and the
    # call fails whenever no read is running; see _ask_the_read_to_stop.
    from rtlsdr.librtlsdr import librtlsdr as _librtlsdr

    RAW_CANCEL_ASYNC = _librtlsdr.rtlsdr_cancel_async
except Exception:                       # pragma: no cover - layout differs
    RAW_CANCEL_ASYNC = None


#: Longest a close waits for a device write to finish before it gives up.
#: Ordinary writes take 40-200 ms; a write still going after this is one
#: the device is not answering, and closing underneath it is the very
#: thing the lock exists to prevent.
DEVICE_LOCK_TIMEOUT_SEC: float = 5.0

#: Longest the cancel waits for a close that is already under way.  Only a
#: close ever holds the handle lock, and rtlsdr_close is milliseconds; a
#: wedged write does not hold it, which is the point of keeping it separate
#: from the device lock.
HANDLE_LOCK_TIMEOUT_SEC: float = 1.0

#: Longest the handle keeps asking the async read to return.  A cancel that
#: arrives before librtlsdr has marked the read as running does nothing at
#: all, and the read then starts anyway, so one attempt is not enough.
SAMPLING_CANCEL_TIMEOUT_SEC: float = 2.0
SAMPLING_CANCEL_RETRY_SEC: float = 0.05


class DeviceHandle:
    """The device, and the question of whether it may still be used.

    Args:
        device: The ``RtlSdr`` this owns.  Its ``close`` is replaced, so
            pyrtlsdr's own close-on-failure arrives here too.
        logger: Where the decisions below are explained.
    """

    def __init__(self, device, logger: logging.Logger) -> None:
        self.device = device
        self.logger = logger

        # Set the moment anybody decides the device is going, which is not
        # the same as it having gone: a close can be asked for long before
        # it can be made.  Nothing new reaches the device from here on.
        self.closing: threading.Event = threading.Event()
        # Set when the handle has actually been freed.
        self.closed: threading.Event = threading.Event()
        # A close that was asked for and could not be made yet.
        self.close_pending: bool = False

        # Held for the length of each device write, and by every close.  A
        # write can be in progress on another thread when shutdown begins -
        # the USB call takes 40-200 ms - and a flag alone cannot stop one
        # that is already past it.
        #
        # Reentrant because two of the closes are made by the thread that
        # already holds it: the receiver's own stop, and pyrtlsdr closing
        # the device from inside a control write we made.  The third -
        # pyrtlsdr closing from inside a failed async read - comes from the
        # reader thread, which holds nothing and therefore waits, which is
        # the point.
        self.device_lock: threading.RLock = threading.RLock()
        # The inner one.  The C cancel holds it for the length of its call,
        # so a close cannot land between its liveness check and the call.
        self.handle_lock: threading.Lock = threading.Lock()
        self._alive: bool = True

        # Armed before the async read begins and cleared when it returns.
        # Read under _sampling_lock, which is what decides whether there is
        # a read to cancel at all: cancelling one that has not started is
        # not merely pointless, it makes librtlsdr return an error and
        # pyrtlsdr answer that by closing the device.
        self._sampling_lock: threading.Lock = threading.Lock()
        self.sampling_active: threading.Event = threading.Event()
        self._sampling_cancelled: threading.Event = threading.Event()
        # Whichever thread is inside read_samples_async.  A close asked for
        # from in there must not wait for the read to end: the read has
        # ended, that is why pyrtlsdr is closing.
        self._reading_thread: threading.Thread | None = None
        self._no_safe_cancel_reported: bool = False

        # Every close of this handle now goes through one place.  pyrtlsdr
        # closes the device itself whenever a control write fails
        # (rtlsdr.py:217, 317) or the async read does (rtlsdr.py:601-603),
        # from inside the call and so from whichever thread made it;
        # interposing here is what brings those under the same locks as the
        # C cancel, without having to hold one for the length of a write.
        self._real_close = device.close
        device.close = self.close

    # ------------------------------------------------------------------
    # Whether the device may be touched
    # ------------------------------------------------------------------

    @property
    def usable(self) -> bool:
        """False once anybody has decided the device is going.

        Deliberately not "is the handle still open": a handle on its way
        out is one nothing should be starting to use, and answering that
        question later than this is how a freed pointer gets dereferenced.
        """
        return not (self.closing.is_set() or self.closed.is_set())

    @contextlib.contextmanager
    def held(self):
        """Hold the device, and try a deferred close on the way out.

        Every operation that touches the device takes it through here, so
        the last one out always makes good on a close that was waiting for
        the device to be free.  Without that the retry would rest on a
        convention - that whoever holds the lock also remembers to try -
        and a close could sit pending with nothing left to trigger it.
        """
        self.device_lock.acquire()
        try:
            yield
        finally:
            self.device_lock.release()
            self.retry_pending_close()

    # ------------------------------------------------------------------
    # The read
    # ------------------------------------------------------------------

    def begin_reading(self) -> bool:
        """Arm the async read, unless the device is on its way out.

        The check and the arming are one step with respect to
        :meth:`stop_sampling`, so a read either arms in time to be
        cancelled or sees the flag and never touches the device - there is
        no third outcome where a thread starts reading through a handle
        that has already been freed.

        Returns:
            True when the caller may go ahead and start the read.
        """
        with self._sampling_lock:
            if not self.usable:
                return False
            self._reading_thread = threading.current_thread()
            self.sampling_active.set()
            return True

    def finished_reading(self) -> None:
        """The async read has returned; say so, and try a deferred close.

        In this order: a close deferred because the read would not end has
        been waiting for exactly this.
        """
        self.sampling_active.clear()
        self._reading_thread = None
        self.retry_pending_close()

    def stop_sampling(self) -> bool:
        """Ask the async read to return, and say whether it has.

        Three things about pyrtlsdr shape this.  It passes the device
        pointer to rtlsdr_cancel_async without checking whether the device
        is still open, so this must not run twice or after the close.
        rtlsdr_cancel_async returns an error whenever the read is not
        running - before it starts, and after it has been cancelled once -
        and the wrapper answers an error by closing the device and raising.
        That close would land outside :attr:`device_lock`, on top of a gain
        write still using the handle, which is the one thing the lock
        exists to stop.

        So: nothing is cancelled unless a read is actually armed, the ask
        goes straight to the C function rather than through the wrapper
        (see :meth:`_ask_the_read_to_stop`), and it is repeated until the
        read has returned, because a cancel landing in the moment between
        arming and librtlsdr marking the read as running does nothing at
        all and the read starts regardless.

        Returns:
            True when no read is running any more - including when one was
            never started.  False when one is still going, in which case
            the caller must not free the handle it is reading through.
        """
        with self._sampling_lock:
            if self.closed.is_set():
                return True
            if not getattr(self.device, "device_opened", True):
                # pyrtlsdr closes the device itself when a control write
                # fails.  There is nothing to cancel and nothing to close.
                self.logger.warning(
                    "The driver has already closed the SDR; nothing to "
                    "cancel")
                self.note_closed_by_driver()
                return True
            if self._sampling_cancelled.is_set():
                return not self.sampling_active.is_set()
            self._sampling_cancelled.set()
            # closing is set before this runs, so begin_reading cannot arm
            # after this point: what is armed now is all there will ever be.
            if not self.sampling_active.is_set():
                self.logger.info(
                    "No async read to cancel; the receiver never started one")
                return True

        if RAW_CANCEL_ASYNC is None or getattr(self.device, "dev_p", None) is None:
            # Nothing to retry: there is no ask that can be made safely.
            self._report_no_safe_cancel()
            return False

        self.logger.info("Stopping SDR async read")
        deadline = time.monotonic() + SAMPLING_CANCEL_TIMEOUT_SEC
        while True:
            if not getattr(self.device, "device_opened", True):
                self.logger.warning(
                    "The driver closed the SDR while we were cancelling; "
                    "not asking it again")
                self.note_closed_by_driver()
                return True
            self._ask_the_read_to_stop()
            if not self.sampling_active.is_set():
                return True
            if time.monotonic() >= deadline:
                self.logger.error(
                    "The async read has not returned %.1f s after being "
                    "cancelled", SAMPLING_CANCEL_TIMEOUT_SEC)
                return False
            time.sleep(SAMPLING_CANCEL_RETRY_SEC)

    def _ask_the_read_to_stop(self) -> None:
        """One attempt at cancelling, with no way for it to close anything.

        rtlsdr_cancel_async is called directly rather than through
        pyrtlsdr.  The wrapper closes the device and raises when the C call
        fails (rtlsdr.py:699-706), and the call fails whenever the read is
        not running - which includes the moment between read_bytes_async
        clearing the wrapper's own suppression flag (rtlsdr.py:599) and
        librtlsdr marking the read as started.  A close from there lands
        outside :attr:`device_lock`, on a handle a gain write is still
        using, so nothing here may rest on a flag the reader is free to
        clear.

        The C function writes two fields on the device struct and nothing
        else: no USB traffic, nothing freed, safe beside a control
        transfer.  A failure is simply a cancel that did nothing, and the
        caller asks again.
        """
        dev_p = getattr(self.device, "dev_p", None)
        if RAW_CANCEL_ASYNC is None or dev_p is None:
            self._report_no_safe_cancel()
            return
        # Under the handle lock, which every close takes: the pointer
        # cannot be freed between the check below and the call.
        if not self.handle_lock.acquire(timeout=HANDLE_LOCK_TIMEOUT_SEC):
            self.logger.error(
                "A close has held the device handle for %.1f s; not "
                "cancelling the read through a pointer that may already "
                "have been freed", HANDLE_LOCK_TIMEOUT_SEC)
            return
        try:
            if not self._alive:
                self.logger.debug(
                    "The handle is closed; there is nothing to cancel")
                return
            result = RAW_CANCEL_ASYNC(dev_p)
            if result < 0:
                self.logger.debug(
                    "rtlsdr_cancel_async returned %d; the read is not "
                    "running yet", result)
        finally:
            self.handle_lock.release()

    def _report_no_safe_cancel(self) -> None:
        """Say once that there is no way to stop the read without risk.

        pyrtlsdr's cancel closes the device whenever the C call fails, and
        the call fails in every state but RUNNING.  Nothing observable from
        out here says which state librtlsdr is in: a callback says one
        arrived, not that the read is still running, and in between the
        read can have moved to CANCELING - where the ask returns -2, the
        wrapper closes, and the read that is still unwinding is left using
        a handle that has been freed.  There is no moment that can be shown
        to be safe, so the wrapper is not used at all.

        A read that is never cancelled costs a handle the process releases
        on exit.  A close made at the wrong moment costs whatever is using
        the device right then.
        """
        if self._no_safe_cancel_reported:
            return
        self._no_safe_cancel_reported = True
        self.logger.error(
            "rtlsdr_cancel_async cannot be reached directly, and pyrtlsdr's "
            "cancel closes the device whenever it fails; leaving the async "
            "read running rather than risking that. The handle is released "
            "when the process exits.")

    # ------------------------------------------------------------------
    # The close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the device once, when nothing is left using it.

        This is what ``device.close`` now is, so it is what pyrtlsdr calls
        from inside a control write or a failed async read, and what the
        receiver calls on the way out.  It waits - bounded - for each of
        the things that must be true first, because for all of those
        callers the close is the point of the call.  The exception is a
        call from the reading thread, which cannot wait for the read
        without waiting for itself; see :meth:`_close_now`.

        :meth:`retry_pending_close` is the form that does not wait.
        """
        self._close_now(wait=True)

    def _close_now(self, wait: bool) -> None:
        """Body of the close.  See :meth:`close`."""
        if self.closed.is_set():
            return
        if not getattr(self.device, "device_opened", True):
            # The driver closed it for us on the way out of a call that
            # failed.  There is nothing left to free, and reaching for it
            # would be reaching into memory that has been given back.
            self.logger.warning(
                "The driver has already closed the SDR; nothing to close")
            self.note_closed_by_driver()
            return
        # Somebody has decided the device is going: this caller, or
        # pyrtlsdr from inside a call that failed.  Either way nothing new
        # should reach it from here, whether or not the close itself lands
        # this time - a handle that has been freed is worse than one that
        # is merely on its way out.
        self.closing.set()
        if threading.current_thread() is self._reading_thread:
            # The request comes from inside the read: pyrtlsdr closing
            # after rtlsdr_read_async returned an error, or something in
            # the SDR callback asking to stop.  From out here those look
            # the same, and only one of them is safe to act on, so neither
            # is.  This thread is the one that has to return before the
            # read can end, so it can neither wait for the read nor free a
            # handle librtlsdr may still be inside: rtlsdr_close waits for
            # the async read to finish, and the read cannot finish while
            # its own callback is stuck in the close.
            #
            # Ask for the cancel and leave.  finished_reading() makes the
            # close, once, when the read has actually returned.
            self._ask_the_read_to_stop()
            self._defer_close("the close was asked for from inside the read")
            return
        if self.sampling_active.is_set():
            if not wait:
                self._defer_close("the async read is still running")
                return
            if not self.stop_sampling():
                self._defer_close("the async read has not returned")
                return
        if wait:
            taken = self.device_lock.acquire(timeout=DEVICE_LOCK_TIMEOUT_SEC)
        else:
            taken = self.device_lock.acquire(blocking=False)
        if not taken:
            self._defer_close("a device write has not returned")
            return
        try:
            with self.handle_lock:
                if not self._alive:
                    return
                self._alive = False
                self.close_pending = False
                self.closed.set()
                try:
                    self._real_close()
                except OSError as e:
                    self.logger.error("Error closing SDR: %s", e)
                self.logger.info("SDR handle closed")
        finally:
            self.device_lock.release()

    def note_closed_by_driver(self) -> None:
        """Record a close the driver made for us, without making another."""
        self.closing.set()
        self.closed.set()
        self.close_pending = False
        with self.handle_lock:
            self._alive = False

    def _defer_close(self, because: str) -> None:
        """Remember a close that could not be made, and say so once."""
        if not self.close_pending:
            self.close_pending = True
            # A warning rather than an error: the handle is intact, the
            # request is kept, and the next operation to finish with the
            # device makes good on it.  The errors in here are for the
            # things that do not recover - a device that stops answering
            # a cancel, a cancel that cannot be reached at all.
            self.logger.warning(
                "Not closing the SDR: %s. The handle stays open and valid; "
                "the close is retried when the device is free, and the "
                "process releases the handle on exit.", because)

    def retry_pending_close(self) -> None:
        """The exit path of every device operation, and of the read.

        A close that could not be made when it was asked for is made here
        instead, the moment whatever was in the way lets go.  It never
        waits: whoever holds the device right now is inside one of these
        same operations and will come through here on the way out, so
        waiting would buy nothing - and these are the calls the window
        makes, where a wait is a frozen window.

        The last one out therefore makes the close, and an operation that
        was refused outright still comes past here on its way to
        returning, which is what covers a receiver nobody talks to again.
        """
        if self.close_pending and not self.closed.is_set():
            self._close_now(wait=False)
            if self.closed.is_set():
                self.logger.info("Made good on a close that was deferred")
