"""Losing the device while it is playing.

Unplugging an RTL-SDR makes rtlsdr_read_async return an error, and the
handle takes care of itself from there.  What these are about is the rest
of the receiver: nothing is coming back from the SDR thread, and nobody
downstream finds that out unless they are told.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time

import numpy as np
import pytest

from fm_radio.controller import FMReceiverController
from fm_radio.exceptions import SDRDeviceError


@pytest.fixture
def receiver(no_user_config):
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


@pytest.fixture
def unpluggable(receiver, monkeypatch):
    """A read that fails the way an unplugged device does."""
    device = receiver.sdr_receiver.sdr
    reading = threading.Event()
    unplug = threading.Event()

    def read_until_unplugged(cb, num_samples=None):
        device.calls.append("read")
        device.reading.set()
        reading.set()
        unplug.wait(30)
        device.reading.clear()
        # rtlsdr.py:601-603: the driver closes the device on the way out.
        device.close()
        raise OSError("LIBUSB_ERROR_NOT_FOUND (-5): Entity not found")

    monkeypatch.setattr(device, "read_samples_async", read_until_unplugged)
    return receiver, device, reading, unplug


# ----------------------------------------------------------------------
# The receiver is told
# ----------------------------------------------------------------------

def test_an_unplugged_device_stops_the_receiver(unpluggable):
    """quit_event is how everything else finds out, so it has to be set.

    Without it the processing thread waits on a queue nobody fills, the
    output underruns on every callback, and the command line sits at its
    prompt as though the radio were still playing.
    """
    receiver, _device, reading, unplug = unpluggable

    receiver.start_background()
    assert reading.wait(5), "the read never started"
    assert not receiver.quit_event.is_set()

    unplug.set()

    assert _within(5.0, receiver.quit_event.is_set), (
        "nothing told the receiver the device had gone")
    assert receiver.device_failure is not None
    assert "LIBUSB_ERROR_NOT_FOUND" in receiver.device_failure


def test_the_reason_reaches_the_person_watching(unpluggable, capsys):
    """Said at the time, not left to whatever prints last."""
    receiver, _device, reading, unplug = unpluggable

    receiver.start_background()
    assert reading.wait(5)
    unplug.set()
    assert _within(5.0, receiver.quit_event.is_set)

    printed = capsys.readouterr().out
    assert "SDR disconnected" in printed, printed
    assert "LIBUSB_ERROR_NOT_FOUND" in printed, printed


def test_the_threads_stop_after_the_device_goes(unpluggable):
    """The point of the flag: everything it gates comes to a halt."""
    receiver, _device, reading, unplug = unpluggable

    receiver.start_background()
    assert reading.wait(5)
    unplug.set()
    assert _within(5.0, receiver.quit_event.is_set)

    for thread in list(receiver.threads):
        thread.join(timeout=10)
        assert not thread.is_alive(), f"{thread.name} outlived the device"


def test_a_device_that_never_goes_leaves_the_receiver_alone(receiver):
    """The flag means something, so it must not be set for nothing."""
    receiver.start_background()
    time.sleep(0.2)

    assert not receiver.quit_event.is_set()
    assert receiver.device_failure is None


# ----------------------------------------------------------------------
# What the window does about it
# ----------------------------------------------------------------------

class _FakeController:
    """Enough of a controller for the window, with a device that can go."""

    def __init__(self, *, recording: bool = False) -> None:
        self.quit_event = threading.Event()
        self.device_failure: str | None = None
        self.center_freq = 80e6
        self.stereo_enabled = True
        self.recording = recording
        # Set when the window asks for the device to be released, so a test
        # can wait for it rather than guess how long it takes.
        self.cleaned_up = threading.Event()
        # Set once the flag has been cleared and the file is being closed;
        # cleared until the test lets that finish.
        self.finalising = threading.Event()
        self.finish = threading.Event()
        self.finish.set()
        self.cleanups: list[str] = []

    def cleanup(self) -> None:
        # In the order the real one does it: AudioOutput.stop_recording
        # clears the flag first and only then flushes the queue, closes
        # the wave file and writes the sidecar.  Asking "is it recording?"
        # after that point says no about a file still being written.
        self.cleanups.append(threading.current_thread().name)
        self.recording = False
        self.finalising.set()
        self.finish.wait(10)
        self.cleaned_up.set()


    def current_station(self):
        return None

    def get_status(self):
        return None

    def get_stations_list(self):
        return []

    def get_frequency(self) -> float:
        return self.center_freq

    def get_gain(self) -> float:
        return 20.0

    def is_manual_gain(self) -> bool:
        return False

    def is_recording(self) -> bool:
        return self.recording

    def is_iq_recording(self) -> bool:
        return False


def test_the_window_says_why_it_stopped(qt_app):
    """A window that keeps showing the last reading is a lie."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController()
    window = ReceiverWindow(controller)
    try:
        window.refresh()
        assert window._timer.isActive()

        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert "SDR disconnected" in window._health.text()
        # The driver's own words are long enough to push the window wider
        # than its controls need, so they go in the tooltip.
        assert "LIBUSB_ERROR_NOT_FOUND" in window._health.toolTip()
        assert "LIBUSB_ERROR_NOT_FOUND" not in window._health.text()
        # Nothing may still look like a radio that is playing.
        assert window._station.text() == "no device"
        assert window._mode.text() == "--"
        assert window._pilot.text() == "--"
        assert window._left.value() == 0 and window._right.value() == 0
        assert window._left_db.text() == "--"
        assert window._right_db.text() == "--"
        assert window._iq_peak.text() == "peak --"
        # Nothing in here can reach the device again.
        assert not window._down.isEnabled()
        assert not window._up.isEnabled()
        assert not window._presets.isEnabled()
        assert not window._auto_gain.isEnabled()
        assert not window._gain_slider.isEnabled()
        assert not window._record_audio.isEnabled()
        assert not window._record_iq.isEnabled()
        # And redrawing the same dead reading fifty times a second is not
        # worth the wake-ups.
        assert not window._timer.isActive()
    finally:
        window.close()
        app.processEvents()


# ----------------------------------------------------------------------
# The noise a stream makes when nobody is feeding it
# ----------------------------------------------------------------------

def test_underruns_are_counted_every_time_and_logged_now_and_then(receiver,
                                                                  caplog):
    """Every callback underruns once the samples stop.

    The count is what telemetry shows and what matters; a line each is
    fifty a second for as long as the process lives, which buried the one
    line that said why in 448 KB of log when this was measured on real
    hardware.
    """
    audio = receiver.audio_output
    frames = 2048

    with caplog.at_level("DEBUG", logger="fm_receiver.AudioOutput"):
        for _ in range(200):
            audio.callback(None, frames, None, None)

    assert audio.underruns == 200, "an underrun that is not counted is lost"
    lines = [r for r in caplog.records if "underrun" in r.message]
    assert len(lines) == 1, f"one line, not two hundred: {len(lines)}"
    assert "since the last of these" in lines[0].message


def test_underruns_are_logged_again_after_the_interval(receiver, caplog,
                                                       monkeypatch):
    """Now and then, not never: a stream that never recovers still says so."""
    monkeypatch.setattr("fm_radio.audio_output._UNDERRUN_LOG_INTERVAL_SEC",
                        0.0)
    audio = receiver.audio_output

    with caplog.at_level("DEBUG", logger="fm_receiver.AudioOutput"):
        for _ in range(5):
            audio.callback(None, 2048, None, None)

    lines = [r for r in caplog.records if "underrun" in r.message]
    assert len(lines) == 5


def test_a_filled_callback_is_not_an_underrun(receiver):
    """The counter still means what it meant."""
    audio = receiver.audio_output
    block = np.zeros(4096, dtype=np.float32)
    audio.enqueue_audio(block, block)

    audio.callback(None, 2048, None, None)

    assert audio.underruns == 0


def _within(seconds: float, predicate) -> bool:
    """True as soon as *predicate* is, or False once *seconds* have gone."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ----------------------------------------------------------------------
# Telling somebody, when there is nobody left to tell
# ----------------------------------------------------------------------

class _ClosedPipe:
    """Standard output after whatever was reading it has gone."""

    def write(self, *args) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


def test_a_closed_pipe_does_not_keep_the_receiver_running(receiver,
                                                          monkeypatch):
    """Stopping comes first, telling second.

    A receiver that goes on running because it could not announce that it
    had stopped is worse than one that stops quietly - and the reason is
    in the log either way.
    """
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    receiver._device_is_gone("LIBUSB_ERROR_NOT_FOUND (-5)")

    assert receiver.quit_event.is_set(), "a broken pipe stopped the shutdown"
    assert receiver.device_failure == "LIBUSB_ERROR_NOT_FOUND (-5)"


def test_a_closed_pipe_does_not_stop_the_sdr_thread_either(unpluggable,
                                                           monkeypatch):
    """The same thing where it actually happens: on the SDR thread."""
    receiver, _device, reading, unplug = unpluggable
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    receiver.start_background()
    assert reading.wait(5), "the read never started"
    unplug.set()

    assert _within(5.0, receiver.quit_event.is_set), (
        "the SDR thread died on the notice instead of stopping the receiver")
    for thread in list(receiver.threads):
        thread.join(timeout=10)
        assert not thread.is_alive(), f"{thread.name} outlived the device"


def test_the_reason_still_reaches_the_log(receiver, monkeypatch, caplog):
    """Losing the printed notice costs nothing that is not written down."""
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    with caplog.at_level("ERROR", logger="fm_receiver.FMReceiverController"):
        receiver._device_is_gone("LIBUSB_ERROR_NOT_FOUND (-5)")

    assert any("LIBUSB_ERROR_NOT_FOUND" in r.message for r in caplog.records)


def a_reader_that_cannot_be_woken(receiver, monkeypatch):
    """A command thread stuck in a read nothing can interrupt.

    Windows with a piped stdin, or any stdin without a file
    descriptor: stop_reading() has nowhere to write, the thread stays
    where it is, and os._exit is what is left.
    """
    asked: list[str] = []
    monkeypatch.setattr(receiver.cmd_interface, "is_alive", lambda: True)
    monkeypatch.setattr(receiver.cmd_interface, "stop_reading",
                        lambda: asked.append("stop"))
    monkeypatch.setattr(receiver.cmd_interface, "join",
                        lambda timeout=None: None)
    return asked


def a_reader_that_stops_when_asked(receiver, monkeypatch):
    """The ordinary case: the wait ends and the thread returns."""
    alive = [True]
    asked: list[str] = []

    def stop_reading():
        asked.append("stop")
        alive[0] = False            # the read ends and run() returns

    monkeypatch.setattr(receiver.cmd_interface, "is_alive",
                        lambda: alive[0])
    monkeypatch.setattr(receiver.cmd_interface, "stop_reading", stop_reading)
    monkeypatch.setattr(receiver.cmd_interface, "join",
                        lambda timeout=None: None)
    return asked


def test_a_command_line_that_stops_when_asked_is_not_exited_past(receiver,
                                                                 monkeypatch):
    """The whole point: an ordinary shutdown unwinds like any other.

    os._exit skips everything Python would do on the way out.  It was
    there because a thread blocked in input() cannot be woken - so the
    wait is arranged to be endable, and where it ends, this does
    nothing but end it.
    """
    import fm_radio.controller as controller_module

    left: list[int] = []
    monkeypatch.setattr(controller_module.os, "_exit", left.append)
    asked = a_reader_that_stops_when_asked(receiver, monkeypatch)
    receiver.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"

    receiver._leave_past_the_blocked_reader()

    assert asked == ["stop"], "the command line was never asked to stop"
    assert left == [], f"the process was taken down anyway: {left}"


def test_a_command_line_already_gone_is_not_asked_for_anything(receiver,
                                                               monkeypatch):
    """It often is: it saw quit_event, or stdin ended."""
    import fm_radio.controller as controller_module

    left: list[int] = []
    monkeypatch.setattr(controller_module.os, "_exit", left.append)
    asked: list[str] = []
    monkeypatch.setattr(receiver.cmd_interface, "is_alive", lambda: False)
    monkeypatch.setattr(receiver.cmd_interface, "stop_reading",
                        lambda: asked.append("stop"))

    receiver._leave_past_the_blocked_reader()

    assert asked == [] and left == []


def test_a_closed_pipe_does_not_keep_the_process_alive(receiver, monkeypatch):
    """The last flushes before leaving can fail on the same pipe."""
    import fm_radio.controller as controller_module

    left: list[int] = []
    monkeypatch.setattr(controller_module.os, "_exit", left.append)
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    monkeypatch.setattr(sys, "stderr", _ClosedPipe())
    a_reader_that_cannot_be_woken(receiver, monkeypatch)
    receiver.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"

    receiver._leave_past_the_blocked_reader()

    assert left == [1], f"a broken pipe stopped the exit: {left}"


def test_leaving_on_purpose_says_so(receiver, monkeypatch):
    """A quit is not a failure, whatever the command thread was doing."""
    import fm_radio.controller as controller_module

    left: list[int] = []
    monkeypatch.setattr(controller_module.os, "_exit", left.append)
    a_reader_that_cannot_be_woken(receiver, monkeypatch)

    receiver._leave_past_the_blocked_reader()

    assert left == [0]


def test_a_reader_that_will_not_stop_is_left_behind(receiver, monkeypatch,
                                                    caplog):
    """And the log says why the blunt instrument came out."""
    import fm_radio.controller as controller_module

    left: list[int] = []
    monkeypatch.setattr(controller_module.os, "_exit", left.append)
    monkeypatch.setattr("fm_radio.controller.READER_STOP_TIMEOUT_SEC", 0.05)
    asked = a_reader_that_cannot_be_woken(receiver, monkeypatch)

    with caplog.at_level("WARNING",
                         logger="fm_receiver.FMReceiverController"):
        receiver._leave_past_the_blocked_reader()

    assert asked == ["stop"], "it was not even asked"
    assert left == [0]
    assert any("cannot be interrupted" in r.message for r in caplog.records), \
        [r.message for r in caplog.records]


# ----------------------------------------------------------------------
# A device that goes before the window is built
# ----------------------------------------------------------------------

def test_the_window_opens_even_if_the_device_went_first(qt_app):
    """start_background() then an unplug, both before the window exists.

    The window is what is going to say why, so it has to survive being
    built into that situation.
    """
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController()
    controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"

    window = ReceiverWindow(controller)
    try:
        assert "SDR disconnected" in window._health.text()
        assert "LIBUSB_ERROR_NOT_FOUND" in window._health.toolTip()
        assert window._station.text() == "no device"
        assert not window._timer.isActive(), (
            "redrawing a dead reading fifty times a second")
        assert not window._down.isEnabled()
    finally:
        window.close()
        app.processEvents()


# ----------------------------------------------------------------------
# What the window does about the recording that was running
# ----------------------------------------------------------------------

def test_the_window_frees_the_device_without_being_closed(qt_app):
    """A recording must not sit half-written until somebody closes a window.

    The record buttons are disabled by then, so nothing else is going to
    end it, and the audio stream has nothing left to play.
    """
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert controller.cleaned_up.wait(5), (
            "the recording was left open until the window closed")
        assert not controller.recording
        # And the window is still there to be read.
        assert "SDR disconnected" in window._health.text()
    finally:
        window.close()
        app.processEvents()


def test_freeing_the_device_does_not_block_the_window(qt_app):
    """cleanup() has bounded waits; a frozen window explains nothing."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    started = threading.Event()
    release = threading.Event()

    def slow_cleanup() -> None:
        # The flag goes first, as it does in AudioOutput.stop_recording;
        # the file is still being closed for as long as this waits.
        controller.recording = False
        started.set()
        release.wait(10)
        controller.cleaned_up.set()

    controller.cleanup = slow_cleanup
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()            # must come straight back

        assert started.wait(5), "the release never started"
        assert not controller.cleaned_up.is_set(), (
            "refresh() waited for the cleanup it started")
        assert "SDR disconnected" in window._health.text()
    finally:
        release.set()
        if window._releasing is not None:
            window._releasing.join(timeout=10)
        window.close()
        app.processEvents()


def test_the_device_is_freed_once_however_often_refresh_runs(qt_app):
    """refresh() is on a timer and is called after anything that changes."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        for _ in range(5):
            window.refresh()

        assert controller.cleaned_up.wait(5)
        window._releasing.join(timeout=10)
        assert controller.cleanups == ["DeviceLossCleanup"], controller.cleanups
    finally:
        window.close()
        app.processEvents()


def test_two_cleanups_at_once_do_not_overlap(receiver):
    """The window starts one, closing the window starts another.

    They are the same call arriving twice, and the second waits for the
    first rather than tearing down beside it.
    """
    inside: list[str] = []
    overlapped: list[bool] = []
    original = receiver._cleanup

    def watched_cleanup() -> None:
        overlapped.append(bool(inside))
        inside.append(threading.current_thread().name)
        try:
            time.sleep(0.2)
            original()
        finally:
            inside.pop()

    receiver._cleanup = watched_cleanup
    threads = [threading.Thread(target=receiver.cleanup, daemon=True)
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not any(thread.is_alive() for thread in threads)
    assert overlapped == [False] * 4, "two cleanups ran at the same time"


# ----------------------------------------------------------------------
# When the log is the thing that cannot be written
# ----------------------------------------------------------------------

class _ClosedFile:
    """A stream whose file has been closed underneath it.

    Distinct from _ClosedPipe on purpose: logging swallows a handler
    failure by reporting it to stderr, and swallows only OSError while
    doing that.  A BrokenPipeError is an OSError and disappears; a
    ValueError from a closed file comes back out at whoever logged.
    """

    def write(self, *args) -> int:
        raise ValueError("I/O operation on closed file")

    def flush(self) -> None:
        raise ValueError("I/O operation on closed file")


@contextlib.contextmanager
def nowhere_to_log(receiver, monkeypatch):
    """A log handler on a closed file, and no stderr to complain to."""
    handler = logging.StreamHandler(_ClosedFile())
    receiver.logger.addHandler(handler)
    monkeypatch.setattr(receiver.logger, "propagate", False)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    monkeypatch.setattr(sys, "stderr", _ClosedFile())
    try:
        yield
    finally:
        receiver.logger.removeHandler(handler)


def test_a_log_that_cannot_be_written_does_not_keep_the_receiver_running(
        receiver, monkeypatch):
    """The state goes up before any kind of telling is attempted.

    Both kinds can fail on a handle closed underneath them, and a receiver
    that keeps running because it could not announce that it had stopped
    is worse than one that stops quietly.
    """
    with nowhere_to_log(receiver, monkeypatch):
        receiver._device_is_gone("LIBUSB_ERROR_NOT_FOUND (-5)")

    assert receiver.quit_event.is_set(), "a broken log stopped the shutdown"
    assert receiver.device_failure == "LIBUSB_ERROR_NOT_FOUND (-5)"


def test_a_log_that_cannot_be_written_still_lets_the_console_hear(
        receiver, monkeypatch, capsys):
    """One kind of telling failing is no reason not to try the other."""
    with nowhere_to_log(receiver, monkeypatch):
        receiver._device_is_gone("LIBUSB_ERROR_NOT_FOUND (-5)")

    printed = capsys.readouterr().out
    assert "SDR disconnected" in printed, printed


def test_a_broken_log_does_not_stop_the_sdr_thread_either(unpluggable,
                                                          monkeypatch):
    """The same thing where it actually happens: on the SDR thread."""
    receiver, _device, reading, unplug = unpluggable

    with nowhere_to_log(receiver, monkeypatch):
        receiver.start_background()
        assert reading.wait(5), "the read never started"
        unplug.set()

        assert _within(5.0, receiver.quit_event.is_set), (
            "the SDR thread died on the log instead of stopping the receiver")
        for thread in list(receiver.threads):
            thread.join(timeout=10)
            assert not thread.is_alive(), f"{thread.name} outlived the device"


def test_the_recording_line_does_not_outlive_the_recording(qt_app):
    """A status line saying "recording audio" about a closed file.

    Observed on hardware: everything else went to "--" and the recording
    line kept its last sentence, because the disconnected view returns
    before refresh() gets to the recording.
    """
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    window = ReceiverWindow(controller)
    try:
        window.refresh()
        assert "recording" in window._recording_status.text()

        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert controller.cleaned_up.wait(5)
        window._releasing.join(timeout=10)
        window.refresh()            # the timer would have brought this

        assert window._recording_status.text() == "", (
            "the window still claims to be recording")
        # A disabled button still shows that it is pressed in, which reads
        # as a recording that is running.
        assert not window._record_audio.isChecked()
        assert not window._record_iq.isChecked()
        assert not window._timer.isActive()
    finally:
        window.close()
        app.processEvents()


def test_the_recording_line_says_so_while_the_file_is_still_closing(qt_app):
    """For that moment there really is still a recording open."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    started = threading.Event()
    release = threading.Event()

    def slow_cleanup() -> None:
        # The flag goes first, as it does in AudioOutput.stop_recording;
        # the file is still being closed for as long as this waits.
        controller.recording = False
        started.set()
        release.wait(10)
        controller.cleaned_up.set()

    controller.cleanup = slow_cleanup
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()
        assert started.wait(5)

        assert window._recording_status.text() == "closing the recording"
        # Still refreshing, because the recording is still moving.
        assert window._timer.isActive()

        release.set()
        window._releasing.join(timeout=10)
        window.refresh()

        assert window._recording_status.text() == ""
        assert not window._timer.isActive()
    finally:
        release.set()
        window.close()
        app.processEvents()


def test_nothing_was_recording_and_the_window_goes_quiet_at_once(qt_app):
    """No recording to follow out, so no reason to keep refreshing."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController()
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert not window._timer.isActive()
        assert window._recording_status.text() == ""
    finally:
        window.close()
        app.processEvents()


def test_the_recording_line_outlasts_the_flag_that_goes_up_first(qt_app):
    """The receiver stops calling it a recording long before it is closed.

    AudioOutput.stop_recording clears its flag and only then flushes the
    queue, waits for the worker, closes the wave file and writes the
    sidecar - up to ten seconds of a file that is still being written
    while the receiver answers "not recording".  The window follows the
    release out instead of the flag.
    """
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    controller.finish.clear()           # hold it inside the finalising
    window = ReceiverWindow(controller)
    try:
        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert controller.finalising.wait(5), "the release never started"
        assert not controller.recording, (
            "the fake is meant to clear the flag before the file is closed")
        assert window._releasing.is_alive()

        window.refresh()                # the timer would have brought this

        assert window._recording_status.text() == "closing the recording", (
            "the window gave up on the recording while it was still closing")
        assert window._timer.isActive(), "and stopped looking"

        controller.finish.set()
        window._releasing.join(timeout=10)
        window.refresh()

        assert window._recording_status.text() == ""
        assert not window._record_audio.isChecked()
        assert not window._timer.isActive()
    finally:
        controller.finish.set()
        window.close()
        app.processEvents()


def test_a_recording_that_started_and_stopped_before_the_unplug(qt_app):
    """Nothing was recording when the device went, so nothing to follow."""
    from fm_radio.gui.main_window import ReceiverWindow

    app = qt_app
    controller = _FakeController(recording=True)
    window = ReceiverWindow(controller)
    try:
        window.refresh()
        controller.recording = False    # the user stopped it themselves
        window.refresh()

        controller.device_failure = "LIBUSB_ERROR_NOT_FOUND (-5)"
        window.refresh()

        assert window._recording_status.text() == ""
        assert not window._timer.isActive(), (
            "kept looking for a recording that was not there")
    finally:
        window.close()
        app.processEvents()
