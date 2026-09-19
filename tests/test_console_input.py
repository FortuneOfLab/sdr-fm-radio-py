"""Waiting for a line, and being able to stop waiting.

The point of all of this is that shutdown does not have to take the
process down under a thread that is blocked in a read. So what is
checked here is mostly the stopping: that a reader parked in a read
comes back when it is asked to, and that a reader which cannot be
stopped admits it rather than pretending.
"""

from __future__ import annotations

import builtins
import os
import sys
import threading
import time

import pytest

from fm_radio.console_input import (
    ConsoleWakeReader, PlainReader, SelectReader, console_reader,
)


def reading(reader, waiting):
    """Start a read and come back once it is actually waiting.

    *waiting* has to be set from inside the wait itself, not from around
    the call: a test that stops the reader before the read has got that
    far is answered by the check at the top of read_line, and never asks
    whether the waking works at all.
    """
    got: list = []

    def read():
        got.append(reader.read_line())

    thread = threading.Thread(target=read, name="Reading", daemon=True)
    thread.start()
    assert waiting.wait(5), "the read never reached the wait"
    return thread, got


def waits_are_announced(reader):
    """Make *reader* say when it is inside the wait, and hand back the event.

    Wrapping the wait rather than the read: the point is the moment
    after the reader has decided there is nothing to hand back and
    before anything has woken it.
    """
    waiting = threading.Event()
    real = reader._wait_for_something

    def announced():
        waiting.set()
        return real()

    reader._wait_for_something = announced
    return waiting


# ----------------------------------------------------------------------
# Watching a file descriptor, which is how POSIX does it
# ----------------------------------------------------------------------

@pytest.fixture
def pipe_reader():
    """A SelectReader over a pipe the test writes into."""
    if not hasattr(os, "fork") and sys.platform == "win32":
        pytest.skip("select cannot watch a pipe on Windows")
    read_fd, write_fd = os.pipe()
    reader = SelectReader(read_fd)
    try:
        yield reader, write_fd
    finally:
        reader.close()
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_a_line_that_arrives_is_returned(pipe_reader):
    reader, write_fd = pipe_reader
    os.write(write_fd, b"record start\n")

    assert reader.read_line() == "record start"


def test_a_read_that_is_stopped_comes_back_with_nothing(pipe_reader):
    """The whole reason this module exists.

    The read is known to be inside the wait before the stop, so what
    this measures is the waking, not the check on the way in.
    """
    reader, _write_fd = pipe_reader
    waiting = waits_are_announced(reader)
    thread, got = reading(reader, waiting)

    reader.stop()
    thread.join(timeout=5)

    assert not thread.is_alive(), "the read never came back"
    assert got == [None]


def test_a_line_left_over_is_not_handed_out_after_a_stop(pipe_reader):
    """A stop during shutdown must not dispatch what was already typed.

    Two commands can arrive in one chunk.  Returning the second after
    the receiver has been told to stop runs it against a receiver that
    is being taken apart.
    """
    reader, write_fd = pipe_reader
    os.write(write_fd, b"stereo off\nrecord start\n")
    assert reader.read_line() == "stereo off"

    reader.stop()

    assert reader.read_line() is None, "a queued command was still handed out"


def test_a_stopped_reader_does_not_start_another_read(pipe_reader):
    reader, write_fd = pipe_reader
    reader.stop()
    os.write(write_fd, b"q\n")

    assert reader.read_line() is None


def test_the_end_of_stdin_is_the_end_of_the_reader(pipe_reader):
    reader, write_fd = pipe_reader
    os.close(write_fd)

    assert reader.read_line() is None


def test_a_last_line_with_no_newline_still_counts(pipe_reader):
    """Somebody who pipes in a file that does not end in one."""
    reader, write_fd = pipe_reader
    os.write(write_fd, b"q")
    os.close(write_fd)

    assert reader.read_line() == "q"
    assert reader.read_line() is None


def test_two_lines_in_one_go_are_two_lines(pipe_reader):
    """A paste arrives as one chunk and must not hold the second line.

    This is why the buffering is the reader's own: going through
    sys.stdin.readline would leave the rest of the chunk in a buffer
    select cannot see, and the second line would wait for a third.
    """
    reader, write_fd = pipe_reader
    os.write(write_fd, b"stereo off\nagc off\n")

    assert reader.read_line() == "stereo off"
    assert reader.read_line() == "agc off"


def test_a_windows_line_ending_is_not_part_of_the_command(pipe_reader):
    reader, write_fd = pipe_reader
    os.write(write_fd, b"list\r\n")

    assert reader.read_line() == "list"


def test_a_line_that_is_not_utf8_does_not_kill_the_reader(pipe_reader):
    reader, write_fd = pipe_reader
    os.write(write_fd, b"search \xff\xfe\n")

    line = reader.read_line()
    assert line is not None and line.startswith("search ")


def test_closing_twice_is_allowed(pipe_reader):
    reader, _write_fd = pipe_reader
    reader.close()
    reader.close()


# ----------------------------------------------------------------------
# The line splitting, which is the reader's own
#
# select cannot watch a pipe on Windows, so the tests above do not run
# there.  These take the same reader and feed it by hand: no select, so
# they run everywhere, and what they check is the part that would
# otherwise only ever be exercised on one platform.
# ----------------------------------------------------------------------

@pytest.fixture
def fed_by_hand():
    """A SelectReader whose buffer the test fills directly."""
    read_fd, write_fd = os.pipe()
    reader = SelectReader(read_fd)

    def feed(data: bytes, end: bool = False):
        if data:
            os.write(write_fd, data)
            assert reader._take_what_is_there()
        if end:
            os.close(write_fd)
            reader._take_what_is_there()

    try:
        yield reader, feed
    finally:
        reader.close()
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_one_chunk_can_hold_more_than_one_line(fed_by_hand):
    """The reason the buffering is here and not in sys.stdin.

    A pasted block arrives as one read.  Handing it to
    sys.stdin.readline would leave the rest in a buffer select cannot
    see, and the second command would wait for a third to be typed.
    """
    reader, feed = fed_by_hand
    feed(b"stereo off\nagc off\ngain 20\n")

    assert reader._take_a_line() == "stereo off"
    assert reader._take_a_line() == "agc off"
    assert reader._take_a_line() == "gain 20"
    assert reader._take_a_line() is None


def test_a_line_already_in_the_buffer_is_dropped_on_a_stop(fed_by_hand):
    """The same as the pipe test above, where select cannot go.

    Two commands can arrive in one chunk.  Handing the second one out
    after the receiver has been told to stop runs it against a receiver
    that is being taken apart.
    """
    reader, feed = fed_by_hand
    feed(b"stereo off\nrecord start\n")

    assert reader.read_line() == "stereo off"

    reader.stop()

    assert reader.read_line() is None, "a queued command was still handed out"


def test_a_line_split_across_two_reads_is_one_line(fed_by_hand):
    reader, feed = fed_by_hand
    feed(b"rec")

    assert reader._take_a_line() is None, "half a line is not a line"

    feed(b"ord start\n")
    assert reader._take_a_line() == "record start"


def test_a_windows_line_ending_is_trimmed(fed_by_hand):
    reader, feed = fed_by_hand
    feed(b"list\r\n")

    assert reader._take_a_line() == "list"


def test_what_is_left_when_stdin_ends_is_the_last_line(fed_by_hand):
    reader, feed = fed_by_hand
    feed(b"q", end=True)

    assert reader._take_a_line() == "q"
    assert reader._take_a_line() is None


def test_bytes_that_are_not_utf8_come_back_as_something(fed_by_hand):
    reader, feed = fed_by_hand
    feed(b"search \xff\xfe\n")

    line = reader._take_a_line()
    assert line is not None and line.startswith("search ")


# ----------------------------------------------------------------------
# input(), woken by a Return of our own, which is how Windows does it
# ----------------------------------------------------------------------

@pytest.fixture
def console_wake(monkeypatch):
    """A ConsoleWakeReader over a stand-in console.

    ``input()`` waits for the key event the reader writes, as the real
    one does; the test never has to guess at timing.
    """
    key_pressed = threading.Event()
    waiting = threading.Event()
    written: list[str] = []

    def fake_input(prompt=""):
        # Set from inside the read, so a test knows the reader is
        # really parked there before it asks for a stop.
        waiting.set()
        assert key_pressed.wait(5), "nothing woke the read"
        return ""

    def write_a_return():
        written.append("return")
        key_pressed.set()

    monkeypatch.setattr(builtins, "input", fake_input)
    reader = ConsoleWakeReader(write_a_return=write_a_return)
    return reader, written, waiting


def test_the_console_read_is_woken_by_a_return(console_wake):
    reader, written, waiting = console_wake
    thread, got = reading(reader, waiting)

    reader.stop()
    thread.join(timeout=5)

    assert not thread.is_alive(), "the read never came back"
    assert written == ["return"], "no key event was written"
    assert got == [None], "the read passed off our own Return as a command"


def test_the_console_reader_hands_back_what_was_typed(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda: "81.3")
    reader = ConsoleWakeReader(write_a_return=lambda: None)

    assert reader.read_line() == "81.3"


def test_the_end_of_input_ends_the_console_reader(monkeypatch):
    def refuse():
        raise EOFError

    monkeypatch.setattr(builtins, "input", refuse)
    reader = ConsoleWakeReader(write_a_return=lambda: None)

    assert reader.read_line() is None


def test_a_wake_that_fails_does_not_escape(monkeypatch):
    """There is nothing to be done about it, and something else to try."""
    def refuse():
        raise OSError("no console here")

    monkeypatch.setattr(builtins, "input", lambda: "")
    reader = ConsoleWakeReader(write_a_return=refuse)

    reader.stop()                       # must not raise

    assert reader.stopped


# ----------------------------------------------------------------------
# The one that cannot be stopped, and says so
# ----------------------------------------------------------------------

def test_the_plain_reader_admits_it_cannot_be_stopped():
    assert PlainReader().can_be_stopped is False


def test_the_plain_reader_still_reads(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda: "q")

    assert PlainReader().read_line() == "q"


def test_the_plain_reader_throws_away_a_line_read_after_a_stop(monkeypatch):
    """It cannot end the read, but it will not act on what it returns."""
    reader = PlainReader()
    reader.stop()
    monkeypatch.setattr(builtins, "input", lambda: "record start")

    assert reader.read_line() is None


# ----------------------------------------------------------------------
# Choosing one
# ----------------------------------------------------------------------

def test_the_reader_for_this_platform_can_be_stopped_or_says_why(monkeypatch):
    """Whichever is chosen, can_be_stopped is the truth about it."""
    reader = console_reader()
    try:
        assert isinstance(reader, (SelectReader, ConsoleWakeReader,
                                   PlainReader))
        assert reader.can_be_stopped == (not isinstance(reader, PlainReader))
    finally:
        reader.close()


def test_a_stdin_with_no_descriptor_falls_back(monkeypatch):
    class _NoDescriptor:
        def fileno(self):
            raise OSError("no fileno here")

    monkeypatch.setattr(sys, "stdin", _NoDescriptor())
    if os.name == "nt":
        monkeypatch.setattr("fm_radio.console_input._console_input_handle",
                            lambda: None)

    reader = console_reader()
    try:
        assert isinstance(reader, PlainReader)
    finally:
        reader.close()


def test_a_stdin_that_is_gone_falls_back(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    if os.name == "nt":
        monkeypatch.setattr("fm_radio.console_input._console_input_handle",
                            lambda: None)

    reader = console_reader()
    try:
        assert isinstance(reader, PlainReader)
    finally:
        reader.close()


@pytest.mark.skipif(os.name != "nt", reason="the console path is Windows only")
def test_a_windows_console_gets_the_reader_that_can_be_woken(monkeypatch):
    monkeypatch.setattr("fm_radio.console_input._console_input_handle",
                        lambda: 7)

    assert isinstance(console_reader(), ConsoleWakeReader)


@pytest.mark.skipif(os.name != "nt", reason="the console path is Windows only")
def test_a_windows_pipe_gets_the_one_that_cannot(monkeypatch):
    monkeypatch.setattr("fm_radio.console_input._console_input_handle",
                        lambda: None)

    assert isinstance(console_reader(), PlainReader)


# ----------------------------------------------------------------------
# What the command line does with it
# ----------------------------------------------------------------------

class _ScriptedReader:
    """Hands out prepared lines, then whatever the test asks for."""

    can_be_stopped = True

    def __init__(self, lines):
        self.lines = list(lines)
        self.stopped = False
        self.closed = False
        self.waiting = threading.Event()

    def read_line(self):
        if self.stopped:
            return None
        if self.lines:
            return self.lines.pop(0)
        self.waiting.set()
        while not self.stopped:
            time.sleep(0.005)
        return None

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


@pytest.fixture
def light_controller(no_user_config):
    """A real controller, for its real CommandLineInterface."""
    from fm_radio.controller import FMReceiverController

    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        try:
            instance.cleanup()
        except Exception:
            pass


def test_the_command_line_stops_when_its_reader_does(light_controller):
    """No line coming means the thread is finished, not idle."""
    reader = _ScriptedReader([])
    cli = light_controller.cmd_interface
    cli.reader = reader

    cli.start()
    try:
        assert reader.waiting.wait(5), "the command line never waited"
        cli.stop_reading()
        cli.join(timeout=5)

        assert not cli.is_alive(), "the command line is still going"
        assert reader.closed, "the reader was not closed on the way out"
    finally:
        reader.stop()


def test_the_command_line_runs_what_it_is_given(light_controller):
    reader = _ScriptedReader(["stereo off", "q"])
    cli = light_controller.cmd_interface
    cli.reader = reader

    cli.start()
    cli.join(timeout=5)

    assert not cli.is_alive(), "'q' did not end the command line"
    assert light_controller.quit_event.is_set()


# ----------------------------------------------------------------------
# A pipe is a pipe, whoever ends up not using it
# ----------------------------------------------------------------------

def a_select_reader():
    """A SelectReader and the fds to close after it."""
    read_fd, write_fd = os.pipe()
    return SelectReader(read_fd), (read_fd, write_fd)


def descriptors_handed_back(monkeypatch):
    """Record every close, so a test can ask which fds really went.

    Asking ``reader.closed`` only asks whether something set a flag.
    The point of closing is the two file descriptors, and a flag can be
    set without them going anywhere.
    """
    given_back: list[int] = []
    real_close = os.close

    def watched(fd):
        given_back.append(fd)
        return real_close(fd)

    monkeypatch.setattr("fm_radio.console_input.os.close", watched)
    return given_back


def test_closing_hands_back_both_ends_of_the_wake_pipe(monkeypatch):
    reader, fds = a_select_reader()
    wake = (reader._wake_r, reader._wake_w)
    given_back = descriptors_handed_back(monkeypatch)

    reader.close()

    try:
        assert set(wake) <= set(given_back), \
            f"the wake pipe was not handed back: {given_back}"
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def test_a_wait_that_fails_still_hands_the_pipe_back(monkeypatch):
    """Giving up on the wait is not the same as having let go.

    select can fail - a stdin whose descriptor has been closed under
    it, a handle it will not take.  There is no line coming after that,
    but the pipe is still ours until close() says otherwise, and a
    reader that called itself closed on the way past would take the
    early return in close() and keep both descriptors for good.
    """
    reader, fds = a_select_reader()
    wake = (reader._wake_r, reader._wake_w)

    def refuse(*args, **kwargs):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr("select.select", refuse)

    assert reader.read_line() is None, "a failed wait produced a line"
    assert not reader.closed, "it called itself closed without closing"

    given_back = descriptors_handed_back(monkeypatch)
    reader.close()

    try:
        assert set(wake) <= set(given_back), \
            f"the wake pipe was left open after a failed wait: {given_back}"
        assert reader.closed
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def test_a_failed_wait_does_not_spin(monkeypatch):
    """It has to end the read, not go round again for ever."""
    reader, fds = a_select_reader()
    tries: list[int] = []

    def refuse(*args, **kwargs):
        tries.append(1)
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr("select.select", refuse)
    try:
        assert reader.read_line() is None
        assert reader.read_line() is None, "it tried again after giving up"

        assert len(tries) == 1, f"the wait was attempted {len(tries)} times"
    finally:
        reader.close()
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def test_the_command_line_makes_no_reader_until_it_runs(light_controller):
    """A window builds a controller and never starts this thread.

    Making the reader in __init__ took a pipe for a command line
    nobody was going to type at, and nothing ever closed it.
    """
    assert light_controller.cmd_interface.reader is None


def test_a_reader_that_was_never_used_is_closed_by_cleanup(light_controller,
                                                           monkeypatch):
    """The window's case: built, never run, and then shut down."""
    reader, fds = a_select_reader()
    wake = (reader._wake_r, reader._wake_w)
    light_controller.cmd_interface.reader = reader

    given_back = descriptors_handed_back(monkeypatch)
    light_controller.cleanup()

    try:
        assert set(wake) <= set(given_back), \
            f"the reader's pipe was left open: {given_back}"
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def test_a_reader_in_use_is_left_to_close_itself(light_controller):
    """Closing it under the thread reading from it would be worse.

    run() closes its own on the way out; this is only for the times
    there is no thread.
    """
    reader = _ScriptedReader([])
    cli = light_controller.cmd_interface
    cli.reader = reader

    cli.start()
    try:
        assert reader.waiting.wait(5), "the command line never waited"
        cli.close_reader()

        assert not reader.closed, "closed under the thread using it"
    finally:
        cli.stop_reading()
        cli.join(timeout=5)

    assert reader.closed, "run() did not close it on the way out"


def test_closing_a_reader_twice_hands_each_end_back_once(light_controller,
                                                          monkeypatch):
    reader, fds = a_select_reader()
    wake = (reader._wake_r, reader._wake_w)
    light_controller.cmd_interface.reader = reader
    given_back = descriptors_handed_back(monkeypatch)
    try:
        light_controller.cmd_interface.close_reader()
        light_controller.cmd_interface.close_reader()

        mine = [fd for fd in given_back if fd in wake]
        assert sorted(mine) == sorted(wake), \
            f"each end should go exactly once: {mine}"
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
