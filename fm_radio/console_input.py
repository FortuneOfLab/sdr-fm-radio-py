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
"""Reading a line from the console, and being able to stop waiting for one.

``input()`` blocks inside a C-level read, and a thread sitting there
cannot be woken by anything Python offers.  That is why shutdown used to
end in ``os._exit``: letting the interpreter finalise around such a
thread aborts the process with ``_enter_buffered_busy: could not acquire
lock for <stdin>``, which makes a clean shutdown look like a crash.

Neither platform needs that if the wait is arranged so it can end.

* Where ``select`` can watch stdin - every POSIX system - the reader
  waits on stdin *and* on a pipe of its own, and :meth:`stop` writes a
  byte to the pipe.  The read itself only happens once there is
  something to read, so it never blocks.
* On a Windows console, ``select`` cannot watch a console handle, so the
  reader keeps ``input()`` - which is worth keeping for the line editing
  and the IME that come with it - and :meth:`stop` pushes a Return into
  the console's own input buffer.  The read returns the way it would if
  somebody had pressed the key, and the reader then sees that it was
  asked to stop.
* Anywhere else - a pipe on Windows, a stdin that has no file
  descriptor - there is nothing to pull, and :attr:`can_be_stopped` says
  so.  The caller is expected to have a blunter instrument in reserve.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

#: Longest a stopped reader is given to notice.  The wake is immediate
#: on both paths; this is only how long the caller waits before
#: concluding that it did not arrive.
STOP_TIMEOUT_SEC: float = 1.0


class ConsoleReader:
    """One line at a time from the console, with a way to give up.

    Subclasses differ only in how the waiting is arranged.  All of them
    return None from :meth:`read_line` for both of the ways there is no
    line coming: stdin ended, or somebody called :meth:`stop`.
    """

    #: Whether :meth:`stop` can actually end a read that is under way.
    can_be_stopped: bool = False

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._stopped: threading.Event = threading.Event()

    @property
    def stopped(self) -> bool:
        """True once :meth:`stop` has been called."""
        return self._stopped.is_set()

    def read_line(self) -> str | None:       # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:
        """Ask a read in progress to give up.  Safe to call more than once."""
        self._stopped.set()

    def close(self) -> None:
        """Release whatever the reader is holding.  Safe to call twice."""


class PlainReader(ConsoleReader):
    """``input()``, and nothing that can interrupt it.

    The fallback for a platform and a stdin that offer no way in.  It
    reads correctly; it just cannot be stopped, and says so.
    """

    can_be_stopped = False

    def read_line(self) -> str | None:
        if self.stopped:
            return None
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        except ValueError:
            # stdin closed under the read, which is one of the ways this
            # ends when somebody else is tearing the process down.
            return None
        return None if self.stopped else line


class SelectReader(ConsoleReader):
    """Waits on stdin and on a pipe of its own, and reads only when ready.

    Because the read happens after ``select`` has said there is
    something there, it does not block, and the wait that does block can
    be ended by writing to the pipe.

    The buffering is this class's own: bytes are read straight from the
    file descriptor and split into lines here.  Going through
    ``sys.stdin.readline`` instead would leave the rest of a pasted
    block sitting in a buffer that ``select`` cannot see, and the next
    line would not arrive until the one after it was typed.

    Args:
        fd: The file descriptor to read from, usually stdin's.
        logger: Where a failure to wake is reported.
    """

    can_be_stopped = True

    def __init__(self, fd: int, logger: logging.Logger | None = None) -> None:
        super().__init__(logger)
        self._fd = fd
        self._pending: bytes = b""
        self._at_the_end: bool = False
        self._wake_r, self._wake_w = os.pipe()
        self._closed: threading.Event = threading.Event()

    def read_line(self) -> str | None:
        import select

        while True:
            line = self._take_a_line()
            if line is not None:
                return line
            if self._at_the_end or self.stopped or self._closed.is_set():
                return None
            try:
                ready, _, _ = select.select([self._fd, self._wake_r], [], [])
            except (OSError, ValueError):
                # stdin or the pipe has gone; either way there is no
                # line coming.
                return None
            if self.stopped:
                return None
            if self._fd not in ready:
                continue
            if not self._take_what_is_there():
                return self._take_a_line()      # whatever was left, or None

    def _take_a_line(self) -> str | None:
        """The next complete line in hand, or None when there is not one."""
        at = self._pending.find(b"\n")
        if at < 0:
            if self._at_the_end and self._pending:
                # stdin ended on a line nobody finished; it still counts.
                last, self._pending = self._pending, b""
                return self._decode(last)
            return None
        line, self._pending = self._pending[:at], self._pending[at + 1:]
        return self._decode(line)

    @staticmethod
    def _decode(line: bytes) -> str:
        return line.rstrip(b"\r").decode("utf-8", errors="replace")

    def _take_what_is_there(self) -> bool:
        """Read what is waiting.  False once stdin has ended."""
        try:
            chunk = os.read(self._fd, 4096)
        except OSError:
            chunk = b""
        if not chunk:
            self._at_the_end = True
            return False
        self._pending += chunk
        return True

    def stop(self) -> None:
        super().stop()
        try:
            os.write(self._wake_w, b"\n")
        except OSError as e:                # pragma: no cover - best effort
            self.logger.debug("Could not wake the console reader: %s", e)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:                 # pragma: no cover - best effort
                pass


class ConsoleWakeReader(ConsoleReader):
    """``input()`` on a Windows console, woken by a Return of our own.

    ``select`` cannot watch a console handle, and reading one by hand
    would mean reimplementing the line editing, the history and the IME
    that ``input()`` gets for nothing.  So the read stays as it is and
    :meth:`stop` writes a Return key event into the console's input
    buffer: the read ends the way it would have if the user had pressed
    the key, and what comes back is thrown away.

    Args:
        write_a_return: What to call to push the key event, for tests.
            The default talks to the console through ctypes.
        logger: Where a failed wake is reported.
    """

    can_be_stopped = True

    def __init__(self, write_a_return=None,
                 logger: logging.Logger | None = None) -> None:
        super().__init__(logger)
        self._write_a_return = write_a_return or _write_a_return_to_the_console

    def read_line(self) -> str | None:
        if self.stopped:
            return None
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        except ValueError:
            return None
        return None if self.stopped else line

    def stop(self) -> None:
        super().stop()
        try:
            self._write_a_return()
        except Exception as e:              # pragma: no cover - best effort
            self.logger.debug("Could not wake the console reader: %s", e)


# ----------------------------------------------------------------------
# The Windows console, through ctypes
# ----------------------------------------------------------------------

_STD_INPUT_HANDLE = -10
_KEY_EVENT = 0x0001
_VK_RETURN = 0x0D


def _console_input_handle():
    """The console's input handle, or None when stdin is not one."""
    if os.name != "nt":                     # pragma: no cover - not Windows
        return None
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    if handle in (0, None) or handle == ctypes.c_void_p(-1).value:
        return None
    mode = ctypes.c_uint32()
    # GetConsoleMode succeeds only on a console handle, which is exactly
    # the question: a piped stdin cannot be woken this way.
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return None
    return handle


def _write_a_return_to_the_console() -> None:  # pragma: no cover - Windows
    """Push a Return key press and release into the console's input."""
    import ctypes
    from ctypes import wintypes

    class _CHAR(ctypes.Union):
        _fields_ = [("UnicodeChar", wintypes.WCHAR),
                    ("AsciiChar", ctypes.c_char)]

    class _KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL),
                    ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD),
                    ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", _CHAR),
                    ("dwControlKeyState", wintypes.DWORD)]

    class _EVENT(ctypes.Union):
        _fields_ = [("KeyEvent", _KEY_EVENT_RECORD)]

    class _INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]

    handle = _console_input_handle()
    if handle is None:
        raise OSError("stdin is not a console")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    records = (_INPUT_RECORD * 2)()
    for i, down in enumerate((True, False)):
        records[i].EventType = _KEY_EVENT
        key = records[i].Event.KeyEvent
        key.bKeyDown = down
        key.wRepeatCount = 1
        key.wVirtualKeyCode = _VK_RETURN
        key.wVirtualScanCode = 0
        key.uChar.UnicodeChar = "\r"
        key.dwControlKeyState = 0
    written = wintypes.DWORD()
    if not kernel32.WriteConsoleInputW(handle, records, 2,
                                       ctypes.byref(written)):
        raise ctypes.WinError(ctypes.get_last_error())


# ----------------------------------------------------------------------
# Choosing one
# ----------------------------------------------------------------------

def console_reader(logger: logging.Logger | None = None) -> ConsoleReader:
    """The reader this platform and this stdin can offer.

    Args:
        logger: Passed to whichever reader is chosen.

    Returns:
        A reader that can be stopped where that is possible, and a
        :class:`PlainReader` where it is not.
    """
    log = logger or logging.getLogger(__name__)
    if os.name == "nt":
        if _console_input_handle() is not None:
            return ConsoleWakeReader(logger=log)
        log.debug("stdin is not a console; the reader cannot be woken")
        return PlainReader(log)
    fd = _stdin_fd()
    if fd is None:
        log.debug("stdin has no file descriptor; the reader cannot be woken")
        return PlainReader(log)
    return SelectReader(fd, log)


def _stdin_fd() -> int | None:
    """stdin's file descriptor, or None when it does not have one."""
    stream = sys.stdin
    if stream is None:
        return None
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return fd if isinstance(fd, int) and fd >= 0 else None
