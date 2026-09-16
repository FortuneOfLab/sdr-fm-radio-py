"""Losing the device while it is playing.

Unplugging an RTL-SDR makes rtlsdr_read_async return an error, and the
handle takes care of itself from there.  What these are about is the rest
of the receiver: nothing is coming back from the SDR thread, and nobody
downstream finds that out unless they are told.
"""

from __future__ import annotations

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

    def __init__(self) -> None:
        self.quit_event = threading.Event()
        self.device_failure: str | None = None
        self.center_freq = 80e6
        self.stereo_enabled = True


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
        return False

    def is_iq_recording(self) -> bool:
        return False


def test_the_window_says_why_it_stopped():
    """A window that keeps showing the last reading is a lie."""
    pytest.importorskip("PySide6.QtWidgets")
    from fm_radio.gui.main_window import ReceiverWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
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
