"""The status window: what it shows, and what it asks the receiver to do.

Driven against a stand-in controller rather than a real receiver, because
what is being checked is the window — that it reads state only through
``get_status()``, changes it only through the facade, and does not fight the
user for a control it is also updating.

Qt runs offscreen (see the ``qt_app`` fixture), so these need no display.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from fm_radio.telemetry import SILENCE_DBFS, StatusSnapshot

# QtWidgets, not just PySide6: the package imports fine on a machine
# without the system EGL/GL libraries it links against, and only fails when
# a Qt module is loaded.  Skipping there keeps the rest of the suite usable;
# CI installs those libraries so these actually run.
pytest.importorskip("PySide6.QtWidgets", reason="the GUI is optional")

from PySide6.QtCore import Qt                                  # noqa: E402

from fm_radio.device_worker import (                      # noqa: E402
    GAIN, GAIN_MODE, RECORDING, TUNE, DeviceWorker,
)
from fm_radio.exceptions import RecordingError            # noqa: E402
from fm_radio.gui.main_window import (                         # noqa: E402
    METER_FLOOR_DBFS, REFRESH_INTERVAL_MS, ReceiverWindow, _level_percent,
)


class FakeController:
    """The facade the window is allowed to use, and nothing else.

    Every call is recorded, so a test can assert that the window went
    through the facade rather than reaching into the receiver.
    """

    def __init__(self, status: StatusSnapshot | None = None) -> None:
        self.status = status
        # The real controller owns one of these; the window reads it to
        # find out how the writes it asked for went.
        self.device_worker = DeviceWorker(logging.getLogger("test.gui"))
        self.last_tune = None
        self.last_write = None
        self.gain_error = None
        self.mode_error = None
        self.frequency = 80.0e6
        self.gain = 28.0
        self.auto_gain = True
        self.recording = False
        self.iq_recording = False
        # Stopped but still being written, as the real one is between
        # the flag going down and the file being closed.  A test that
        # wants the file finished says so; nothing finishes by itself.
        self.finishing_audio = False
        self.finishing_iq = False
        self.recording_path = None
        self.calls: list[tuple] = []
        self.quit_event = _Event()
        self.station = _Station("TOKYO FM")
        self.presets = [("TOKYO FM", 80.0e6), ("J-WAVE", 81.3e6)]
        self.tune_error: Exception | None = None
        self.record_error: Exception | None = None
        # As in the controller: bumped by every stop, so a start asked
        # for before it knows it was overtaken.
        self.audio_wanted = 0
        self.iq_wanted = 0
        self.starting_audio = None
        self.starting_iq = None

    # --- reading ---
    def get_status(self):
        return self.status

    def get_frequency(self):
        return self.frequency

    def get_gain(self):
        return self.gain

    def is_manual_gain(self):
        return not self.auto_gain

    def is_recording(self):
        return self.recording

    def is_iq_recording(self):
        return self.iq_recording

    def is_finishing_a_recording(self):
        return self.finishing_audio

    def is_finishing_an_iq_recording(self):
        return self.finishing_iq

    def finished_the_recordings(self):
        """What the close thread getting there amounts to, for a test."""
        self.finishing_audio = False
        self.finishing_iq = False

    def current_station(self):
        return self.station

    def get_stations_list(self):
        return list(self.presets)

    def tuned(self, timeout: float = 5.0) -> None:
        """Wait for the tune that was asked for, as a test may.

        The window never does this; it carries on drawing and picks the
        outcome up on a later refresh.
        """
        assert self.last_tune is not None, "nothing asked for a tune"
        assert self.last_tune.wait(timeout), "the tune never landed"

    def settled(self, timeout: float = 5.0) -> None:
        """Wait for the last write of any kind this was asked for."""
        assert self.last_write is not None, "nothing asked for a write"
        assert self.last_write.wait(timeout), "the write never landed"

    # --- changing ---
    def tune(self, freq_hz):
        """Hand the write to the worker, as the real one does.

        Nothing is raised at the caller any more: a tune that fails does
        so on the worker thread, and the window hears about it through
        the request the worker keeps.
        """
        self.calls.append(("tune", freq_hz))

        def write():
            if self.tune_error is not None:
                raise self.tune_error
            self.frequency = freq_hz

        self.last_tune = self.device_worker.submit(
            TUNE, f"Tuned to {freq_hz / 1e6:.1f} MHz", write)
        return self.last_tune

    def set_agc_mode(self, enabled):
        """Switch Auto now and write later, as AutoGainController does.

        The real one flips ``_enabled`` under its own lock as the call
        goes through; only the USB write waits for the worker, so a
        refresh between the two finds Auto already on.  Turning it on
        asks for the mode and then a gain, and hands back the mode
        request; turning it off asks for a gain alone - the one that
        pins the gain where the AGC left it - and hands that back.
        """
        self.calls.append(("set_agc_mode", enabled))
        self.auto_gain = enabled

        def write_mode():
            if self.mode_error is not None:
                raise self.mode_error

        def write_gain():
            if self.gain_error is not None:
                raise self.gain_error

        if enabled:
            asked = self.device_worker.submit(
                GAIN_MODE, "Gain mode set to manual", write_mode)
            self.last_write = self.device_worker.submit(
                GAIN, f"Auto gain applied {self.gain:.1f} dB", write_gain)
        else:
            asked = self.device_worker.submit(
                GAIN, f"Gain set to {self.gain:.1f} dB", write_gain)
            self.last_write = asked
        return asked

    def set_gain(self, gain_db):
        self.calls.append(("set_gain", gain_db))

        def write():
            if self.gain_error is not None:
                raise self.gain_error
            self.gain = gain_db

        self.last_write = self.device_worker.submit(
            GAIN, f"Gain set to {gain_db:.1f} dB", write)
        return self.last_write

    def start_recording(self, path=None):
        """Ask the worker, as the real one does, and name it there.

        Three things the real one does that this has to do too: the
        name comes from the frequency the worker finds when it gets
        round to it, a start that a stop overtook does not install, and
        a second recording over a first is refused rather than reported
        as a success against a file nobody made.
        """
        self.calls.append(("start_recording", path))
        wanted = self.audio_wanted

        def start():
            if self.record_error is not None:
                raise self.record_error
            if wanted != self.audio_wanted:
                raise RecordingError(
                    "The recording was stopped before it started")
            if self.recording:
                raise RecordingError(
                    "Already recording; this recording was not started")
            made = path or f"recordings/{self.frequency / 1e6:.1f}MHz.wav"
            self.recording = True
            self.recording_path = made
            return made

        self.last_write = self.device_worker.submit(
            RECORDING, "Starting the recording", start)
        self.starting_audio = self.last_write
        return self.last_write

    def stop_recording(self):
        """Shut the door and come back, as the real one does.

        The file is not finished when this returns: the flag goes
        down, and finishing_audio stays up until a test says the close
        thread got there.
        """
        self.calls.append(("stop_recording",))
        self.audio_wanted += 1
        asked_for, self.starting_audio = self.starting_audio, None
        taken_back = self.device_worker.cancel(asked_for)
        was = self.recording
        self.recording = False
        if was:
            self.finishing_audio = True
        return was or taken_back

    def start_iq_recording(self, path=None):
        self.calls.append(("start_iq_recording", path))
        wanted = self.iq_wanted

        def start():
            if self.record_error is not None:
                raise self.record_error
            if wanted != self.iq_wanted:
                raise RecordingError(
                    "The IQ recording was stopped before it started")
            if self.iq_recording:
                raise RecordingError(
                    "Already recording; this IQ recording was not started")
            made = path or f"recordings/{self.frequency / 1e6:.1f}MHz_IQ.wav"
            self.iq_recording = True
            return made

        self.last_write = self.device_worker.submit(
            RECORDING, "Starting the IQ recording", start)
        self.starting_iq = self.last_write
        return self.last_write

    def stop_iq_recording(self):
        self.calls.append(("stop_iq_recording",))
        self.iq_wanted += 1
        asked_for, self.starting_iq = self.starting_iq, None
        taken_back = self.device_worker.cancel(asked_for)
        was = self.iq_recording
        self.iq_recording = False
        if was:
            self.finishing_iq = True
        return was or taken_back


class _Event:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class _Station:
    def __init__(self, name):
        self.name = name


def snapshot(**overrides) -> StatusSnapshot:
    defaults = dict(
        freq_hz=80.0e6, station="TOKYO FM", gain_db=28.0, auto_gain=True,
        iq_peak=0.62,
        stereo=True, blend_factor=1.0, pilot_snr_db=19.6, pilot_jitter_db=0.8,
        side_nr_enabled=True,
        level_left_dbfs=-6.2, level_right_dbfs=-7.8,
        block_ms=4.2, block_ms_avg=4.0, block_ms_max=6.1, block_budget_ms=16.0,
        sdr_queue=1, sdr_queue_max=80, slow_blocks=0,
        iq_drops=0, audio_drops=0, audio_underruns=0,
        recording_audio=False, recording_iq=False,
        uptime_sec=767.0, timestamp=1.0,
    )
    defaults.update(overrides)
    return StatusSnapshot(**defaults)


@pytest.fixture
def window(qt_app):
    """A window over a stand-in controller, closed afterwards."""
    built = []

    def _build(controller=None):
        controller = controller or FakeController(snapshot())
        instance = ReceiverWindow(controller)
        built.append(instance)
        return instance, controller

    yield _build
    for instance in built:
        instance._timer.stop()
        instance.close()


# ----------------------------------------------------------------------
# What it shows
# ----------------------------------------------------------------------

def test_the_frequency_and_station_come_from_the_snapshot(window):
    view, _ = window(FakeController(snapshot(freq_hz=81.3e6, station="J-WAVE")))
    assert view._frequency.text() == "81.3 MHz"
    assert view._station.text() == "J-WAVE"


@pytest.mark.parametrize("overrides,expected", [
    ({"stereo": True, "blend_factor": 1.0}, "STEREO"),
    ({"stereo": False, "blend_factor": 1.0}, "MONO"),
    ({"stereo": True, "blend_factor": 0.3}, "BLENDING (0.30)"),
])
def test_the_mode_distinguishes_mono_from_a_half_open_blend(
        window, overrides, expected):
    view, _ = window(FakeController(snapshot(**overrides)))
    assert view._mode.text() == expected


def test_an_unmeasured_pilot_reads_as_unknown(window):
    view, _ = window(FakeController(snapshot(pilot_snr_db=None)))
    assert view._pilot.text() == "--"


def test_the_levels_drive_the_meters(window):
    view, _ = window(FakeController(
        snapshot(level_left_dbfs=-6.0, level_right_dbfs=-30.0)))
    assert view._left_db.text() == "-6.0 dBFS"
    assert view._right_db.text() == "-30.0 dBFS"
    assert view._left.value() > view._right.value()


def test_silence_reads_as_unknown_rather_than_a_number(window):
    view, _ = window(FakeController(snapshot(level_left_dbfs=SILENCE_DBFS)))
    assert view._left_db.text() == "--"
    assert view._left.value() == 0


@pytest.mark.parametrize("dbfs,expected", [
    (0.0, 100), (METER_FLOOR_DBFS, 0), (METER_FLOOR_DBFS - 10, 0),
    (-30.0, 50), (10.0, 100),
])
def test_the_meter_scale_covers_its_range(dbfs, expected):
    assert _level_percent(dbfs) == expected


def test_the_health_line_reports_the_block_budget(window):
    view, _ = window(FakeController(snapshot(block_ms=4.2, iq_drops=0)))
    text = view._health.text()
    assert "healthy" in text and "4.2/16 ms" in text


def test_the_health_line_says_so_when_blocks_run_long(window):
    view, _ = window(FakeController(snapshot(block_ms=20.0)))
    assert "loaded" in view._health.text()


def test_without_a_snapshot_it_shows_the_tuner_not_stale_values(window):
    """Startup, and the moment after tuning: the snapshot is gone."""
    controller = FakeController(status=None)
    controller.frequency = 82.5e6
    controller.station = _Station("NHK-FM 東京")
    view, _ = window(controller)

    assert view._frequency.text() == "82.5 MHz"
    assert view._station.text() == "NHK-FM 東京"
    assert view._mode.text() == "--"
    assert view._pilot.text() == "--"
    assert view._left.value() == 0
    assert "waiting" in view._health.text()


def test_a_snapshot_arriving_later_replaces_the_placeholder(window):
    controller = FakeController(status=None)
    view, _ = window(controller)
    assert view._mode.text() == "--"

    controller.status = snapshot(station="J-WAVE")
    view.refresh()
    assert view._mode.text() == "STEREO"
    assert view._station.text() == "J-WAVE"


# ----------------------------------------------------------------------
# What it asks the receiver to do
# ----------------------------------------------------------------------

def test_the_arrows_tune_by_one_step(window):
    """Each step is asked for from where the receiver is, not where it was.

    The asking no longer waits for the write, so the arrows read back the
    frequency the receiver reports - which is the one it is still on
    until the tune lands.  Each click is therefore let land before the
    next, the way a person clicking would.
    """
    view, controller = window()
    for button in (view._up, view._down, view._down):
        button.click()
        controller.tuned()
        view.refresh()

    assert [c for c in controller.calls if c[0] == "tune"] == [
        ("tune", 80.1e6), ("tune", 80.0e6), ("tune", 79.9e6)]


def test_a_preset_tunes_to_its_frequency(window):
    view, controller = window()
    index = view._presets.findText("81.3  J-WAVE")
    assert index > 0
    view._presets.setCurrentIndex(index)
    view._presets.activated.emit(index)

    assert ("tune", 81.3e6) in controller.calls
    # The list goes back to its label rather than pretending to be a display.
    assert view._presets.currentIndex() == 0


def test_the_preset_label_itself_tunes_nothing(window):
    view, controller = window()
    view._presets.activated.emit(0)
    assert not [c for c in controller.calls if c[0] == "tune"]


def test_a_tuner_that_refuses_is_reported_not_raised(window):
    from fm_radio.exceptions import SDRDeviceError

    controller = FakeController(snapshot())
    controller.tune_error = SDRDeviceError("device gone")
    view, controller = window(controller)

    view._up.click()                    # must not raise
    controller.tuned()                  # the write fails on the worker
    view.refresh()                      # ... and the window hears about it

    assert "failed" in view._health.text(), view._health.text()
    assert "device gone" in view._health.text()


def test_a_failure_survives_the_next_refresh(window):
    """The health line is rewritten every 50 ms; a message must outlast that."""
    from fm_radio.exceptions import SDRDeviceError

    controller = FakeController(snapshot())
    controller.tune_error = SDRDeviceError("device gone")
    view, controller = window(controller)

    view._up.click()
    controller.tuned()
    view.refresh()
    view.refresh()
    assert "failed" in view._health.text(), view._health.text()


def test_a_failure_gives_way_to_the_health_line_eventually(window,
                                                            monkeypatch):
    from fm_radio.exceptions import SDRDeviceError
    from fm_radio.gui import main_window

    controller = FakeController(snapshot())
    controller.tune_error = SDRDeviceError("device gone")
    view, controller = window(controller)

    view._up.click()
    controller.tuned()
    view.refresh()
    assert "failed" in view._health.text(), view._health.text()

    clock = [main_window.time.monotonic() + main_window.NOTICE_SECONDS + 1]
    monkeypatch.setattr(main_window.time, "monotonic", lambda: clock[0])
    view.refresh()
    assert "healthy" in view._health.text()


def test_the_auto_gain_box_switches_the_receiver_and_the_slider(window):
    view, controller = window()
    view._auto_gain.setChecked(False)

    assert ("set_agc_mode", False) in controller.calls
    assert view._gain_slider.isEnabled()

    view._auto_gain.setChecked(True)
    assert ("set_agc_mode", True) in controller.calls
    assert not view._gain_slider.isEnabled()


def test_releasing_the_slider_sets_the_gain(window):
    view, controller = window()
    view._auto_gain.setChecked(False)
    view._gain_slider.setValue(384)
    view._gain_slider.sliderReleased.emit()

    assert ("set_gain", 38.4) in controller.calls


def test_the_slider_does_not_set_the_gain_while_auto_is_on(window):
    view, controller = window()
    view._gain_slider.setValue(384)
    view._gain_slider.sliderReleased.emit()

    assert not [c for c in controller.calls if c[0] == "set_gain"]


def test_recording_starts_and_stops_through_the_facade(window, tmp_path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    view, controller = window()

    view._record_audio.click()
    controller.settled()
    view.refresh()

    started = [c for c in controller.calls if c[0] == "start_recording"]
    assert len(started) == 1, controller.calls
    assert started[0][1] is None, "the window named the file itself"
    assert "recording audio" in view._recording_status.text()

    view._record_audio.click()
    assert ("stop_recording",) in controller.calls


def test_the_receiver_is_the_one_that_names_the_recording(window, tmp_path,
                                                          monkeypatch):
    """Not the window.

    The name says which station this is, and which station that will
    be is settled by the receiver - a tune may be in front of the
    recording on the worker.  The window passes no path and reads the
    one that comes back.
    """
    monkeypatch.chdir(tmp_path)
    view, controller = window()

    view._record_audio.click()
    controller.settled()

    assert controller.last_write.result.endswith("MHz.wav"), \
        controller.last_write.result


def test_iq_recording_asks_for_an_iq_file(window, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view, controller = window()
    view._record_iq.click()
    controller.settled()

    started = [c for c in controller.calls if c[0] == "start_iq_recording"]
    assert len(started) == 1 and started[0][1] is None
    assert "_IQ" in controller.last_write.result


def test_a_recording_that_will_not_start_releases_the_button(window,
                                                              tmp_path,
                                                              monkeypatch):
    """The failure arrives on a later refresh now, not out of the slot."""
    from fm_radio.exceptions import RecordingError

    monkeypatch.chdir(tmp_path)
    controller = FakeController(snapshot())
    controller.record_error = RecordingError("disk full")
    view, _ = window(controller)

    view._record_audio.click()          # must not raise
    controller.settled()
    view.refresh()

    assert not view._record_audio.isChecked()
    assert "disk full" in view._health.text(), view._health.text()
    assert view._recording_status.text() == ""


def test_a_recording_that_has_been_asked_for_keeps_the_button_down(window):
    """The receiver decides when it starts; until then it looks pressed.

    A button that springs back up while the request is still on the
    worker reads as a button that did nothing.
    """
    view, controller = window()
    release = held(controller)
    try:
        view._record_audio.click()
        view.refresh()

        assert view._record_audio.isChecked(), "the button sprang back up"
        assert not controller.is_recording(), "it is not recording yet"
        assert "starting audio" in view._recording_status.text(), \
            view._recording_status.text()
    finally:
        release.set()
    controller.settled()
    view.refresh()

    assert "recording audio" in view._recording_status.text()


def test_a_recording_that_fails_lets_the_button_go(window, tmp_path,
                                                   monkeypatch):
    """And the line that said it was starting goes with it."""
    from fm_radio.exceptions import RecordingError

    monkeypatch.chdir(tmp_path)
    controller = FakeController(snapshot())
    controller.record_error = RecordingError("disk full")
    view, _ = window(controller)

    view._record_audio.click()
    controller.settled()
    view.refresh()

    assert not view._record_audio.isChecked()
    assert view._recording_status.text() == "", \
        view._recording_status.text()


def test_a_recording_that_stops_by_itself_releases_the_button(window):
    """Rotation or an error can stop a recording without the user."""
    view, controller = window()
    controller.recording = True
    view.refresh()
    assert view._record_audio.isChecked()

    controller.recording = False        # the receiver stopped on its own
    view.refresh()
    assert not view._record_audio.isChecked()
    # ... and doing so must not have been read as a button press.
    assert not [c for c in controller.calls if c[0] == "stop_recording"]


# ----------------------------------------------------------------------
# Not fighting the user
# ----------------------------------------------------------------------

def test_a_refresh_does_not_move_a_slider_being_dragged(window):
    view, controller = window()
    view._auto_gain.setChecked(False)
    view._gain_slider.setSliderDown(True)
    view._gain_slider.setValue(100)

    controller.status = snapshot(gain_db=49.0, auto_gain=False)
    view.refresh()

    assert view._gain_slider.value() == 100, "the refresh took the slider"
    view._gain_slider.setSliderDown(False)


def test_a_refresh_moves_the_slider_when_it_is_not_being_dragged(window):
    view, controller = window()
    controller.status = snapshot(gain_db=38.4, auto_gain=True)
    view.refresh()
    assert view._gain_slider.value() == 384


def test_following_auto_gain_does_not_ask_the_receiver_to_change_it(window):
    """Showing a value must not be mistaken for the user setting one."""
    view, controller = window()
    controller.status = snapshot(gain_db=44.5, auto_gain=True)
    view.refresh()

    assert not [c for c in controller.calls if c[0] == "set_gain"]
    assert not [c for c in controller.calls if c[0] == "set_agc_mode"]


def test_following_the_receiver_into_manual_gain_does_not_echo_it_back(window):
    view, controller = window()
    controller.status = snapshot(auto_gain=False, gain_db=30.0)
    view.refresh()

    assert view._auto_gain.isChecked() is False
    assert view._gain_slider.isEnabled()
    assert not [c for c in controller.calls if c[0] == "set_agc_mode"]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------

def test_the_refresh_timer_runs_at_the_publish_rate(window):
    view, _ = window()
    assert view._timer.isActive()
    assert view._timer.interval() == REFRESH_INTERVAL_MS == 50


def test_closing_the_window_asks_the_receiver_to_stop(window):
    view, controller = window()
    view.close()

    assert controller.quit_event.was_set
    assert not view._timer.isActive()


def test_the_presets_come_from_the_catalogue(window):
    view, _ = window()
    # One label plus the favourites.
    assert view._presets.count() == 3
    assert view._presets.itemData(1) == 80.0e6


# ----------------------------------------------------------------------
# The optional dependency
# ----------------------------------------------------------------------

def test_without_pyside_the_entry_point_says_what_to_install(monkeypatch,
                                                              capsys):
    """The receiver does not need Qt; asking for the GUI without it should
    explain that rather than traceback."""
    import sys as _sys

    from fm_radio import gui

    monkeypatch.setitem(_sys.modules, "fm_radio.gui.main_window", None)
    assert gui.run(object()) == 1

    message = capsys.readouterr().err
    assert "pip install PySide6" in message
    assert "without --gui" in message


def test_run_passes_the_controller_to_the_window(monkeypatch):
    from fm_radio import gui
    from fm_radio.gui import main_window

    seen = []
    monkeypatch.setattr(
        main_window, "run_window",
        lambda controller, start=None: seen.append((controller, start)) or 0)
    controller = object()
    assert gui.run(controller) == 0
    assert seen == [(controller, None)]


def test_run_passes_the_switch_through_as_well(monkeypatch):
    """Whatever starts the receiver is the window's to call, when it is up."""
    from fm_radio import gui
    from fm_radio.gui import main_window

    seen = []
    monkeypatch.setattr(
        main_window, "run_window",
        lambda controller, start=None: seen.append((controller, start)) or 0)
    controller = object()

    def switch():
        pass

    assert gui.run(controller, switch) == 0
    assert seen == [(controller, switch)]


# ----------------------------------------------------------------------
# Every way of moving the slider
# ----------------------------------------------------------------------

def test_the_wheel_and_the_keyboard_set_the_gain_too(window):
    """isSliderDown() is false for these, so a release never arrives."""
    view, controller = window()
    view._auto_gain.setChecked(False)
    controller.calls.clear()

    view._gain_slider.triggerAction(
        view._gain_slider.SliderAction.SliderPageStepAdd)

    applied = [c for c in controller.calls if c[0] == "set_gain"]
    assert applied, "a page step never reached the receiver"
    assert applied[-1][1] == view._gain_slider.value() / 10.0


def test_a_drag_sends_one_gain_change_not_one_per_pixel(window):
    view, controller = window()
    view._auto_gain.setChecked(False)
    controller.calls.clear()

    view._gain_slider.setSliderDown(True)
    for value in (100, 150, 200, 250):
        view._gain_slider.setValue(value)
    assert not [c for c in controller.calls if c[0] == "set_gain"]

    # setSliderDown(False) is what ends a drag; Qt emits sliderReleased
    # itself, so emitting it here too would count the release twice.
    view._gain_slider.setSliderDown(False)
    assert [c for c in controller.calls if c[0] == "set_gain"] == [
        ("set_gain", 25.0)]


def test_following_the_receiver_still_does_not_echo_the_gain_back(window):
    """valueChanged is live now, so the refresh has to keep blocking it."""
    view, controller = window()
    view._auto_gain.setChecked(False)
    controller.calls.clear()

    controller.status = snapshot(gain_db=44.5, auto_gain=False)
    view.refresh()

    assert view._gain_slider.value() == 445
    assert not [c for c in controller.calls if c[0] == "set_gain"]


# ----------------------------------------------------------------------
# Naming the recording can fail before the receiver is asked anything
# ----------------------------------------------------------------------

@pytest.mark.parametrize("button,call", [
    ("_record_audio", "start_recording"),
    ("_record_iq", "start_iq_recording"),
])
def test_a_recordings_folder_that_cannot_be_made_is_reported(
        window, monkeypatch, button, call):
    """Naming the file creates recordings/, and that can fail.

    It happens on the worker now, with everything else about starting a
    recording, so it arrives as a failed request rather than out of the
    slot.
    """
    controller = FakeController(snapshot())
    controller.record_error = PermissionError(13, "permission denied")
    view, _ = window(controller)

    getattr(view, button).click()       # must not raise out of the slot
    controller.settled()
    view.refresh()

    assert not getattr(view, button).isChecked()
    assert "permission denied" in view._health.text(), view._health.text()


# ----------------------------------------------------------------------
# Stepping while a write is still in the air
# ----------------------------------------------------------------------

def held(controller):
    """Stop the worker mid-write, so nothing the window asks for lands."""
    running = threading.Event()
    release = threading.Event()
    controller.device_worker.submit(
        TUNE, "the one in flight",
        lambda: (running.set(), release.wait(10)))
    assert running.wait(5), "the worker never started the first write"
    return release


def test_two_steps_during_one_write_move_two_steps(window):
    """A step starts from where the tuner is going, not where it is.

    The receiver keeps reporting the frequency it is on until the write
    lands - 60 ms - so stepping from that asks for the same place twice
    and ends up half as far as the user asked to go.
    """
    view, controller = window()
    release = held(controller)
    try:
        view._up.click()
        view._up.click()

        assert [c for c in controller.calls if c[0] == "tune"] == [
            ("tune", 80.1e6), ("tune", 80.2e6)]
    finally:
        release.set()


def test_a_preset_moves_the_place_the_steps_start_from(window):
    """Choosing a station and then stepping goes one step from there."""
    view, controller = window()
    release = held(controller)
    try:
        view._presets.setCurrentIndex(1)
        view._presets.activated.emit(1)
        view._up.click()

        tunes = [c[1] for c in controller.calls if c[0] == "tune"]
        assert tunes[-1] == pytest.approx(tunes[-2] + 0.1e6), tunes
    finally:
        release.set()


def test_the_step_goes_back_to_the_receiver_once_the_tune_lands(window):
    """Nothing on its way means the receiver is the truth again."""
    view, controller = window()
    view._up.click()
    controller.tuned()
    view.refresh()

    assert view._tuning_to is None, "still thinks a tune is on its way"


# ----------------------------------------------------------------------
# Watching what this window asked for, not what happened last
# ----------------------------------------------------------------------

def test_a_gain_landing_does_not_answer_for_a_tune_that_has_not(window):
    """The worker's last finished request may be somebody else's.

    The AGC writes gains of its own accord.  One of those landing between
    two refreshes must not take the "tuning to..." line down, nor stand in
    for an answer the tune has not given yet.
    """
    view, controller = window()
    release = held(controller)
    try:
        view._up.click()
        # Something else finishes in the meantime, as the AGC does.
        done = controller.device_worker.submit(GAIN, "gain 20", lambda: None)
        view.refresh()

        assert not done.finished, "the worker is supposed to be busy"
        assert "tuning to 80.1 MHz" in view._health.text(), view._health.text()
    finally:
        release.set()
    controller.tuned()
    view.refresh()
    assert "tuning to" not in view._health.text()


def test_a_gain_that_fails_is_reported_as_well_as_a_tune(window):
    """A failure of any write the window asked for is worth saying."""
    view, controller = window()
    controller.gain_error = OSError("LIBUSB_ERROR_TIMEOUT")

    view._auto_gain.setChecked(False)
    view._gain_slider.setValue(int(22.0 * 10))
    view._gain_slider.sliderReleased.emit()
    controller.settled()
    view.refresh()

    assert "failed" in view._health.text(), view._health.text()
    assert "LIBUSB_ERROR_TIMEOUT" in view._health.text()


def test_a_tune_replaced_by_a_later_one_is_not_reported_as_a_failure(window):
    """Superseded is what the window asked for happening once, not failing."""
    view, controller = window()
    release = held(controller)
    try:
        view._up.click()
        view._up.click()                # replaces the first
    finally:
        release.set()
    controller.tuned()
    view.refresh()

    assert "failed" not in view._health.text(), view._health.text()


def test_a_failure_stays_readable_while_a_tune_is_still_going(window):
    """Both want the one status line, and the failure needs it more.

    A tune that has not landed says so again on the next refresh, and
    the one after that.  A gain that would not write gets one chance to
    be read, so it is not the one that gives way.
    """
    view, controller = window()
    view._auto_gain.setChecked(False)
    controller.settled()
    controller.gain_error = OSError("LIBUSB_ERROR_TIMEOUT")
    view._gain_slider.setValue(220)          # 22.0 dB, tenths
    view._gain_slider.sliderReleased.emit()
    controller.settled()

    release = held(controller)
    try:
        # The click refreshes, and that one refresh has both to report:
        # the gain that failed and the tune that has not landed.
        view._up.click()

        assert "LIBUSB_ERROR_TIMEOUT" in view._health.text(),             view._health.text()

        # Once it has had its few seconds the tune is still going, and
        # saying so is the best thing left to say.
        message, _expires, sort = view._notice
        view._notice = (message, time.monotonic() - 1.0, sort)
        view.refresh()

        assert "tuning to 80.1 MHz" in view._health.text(),             view._health.text()
    finally:
        release.set()


def test_turning_auto_off_is_true_before_the_write_lands(window):
    """The controller flips the mode under its own lock, not on the worker.

    AutoGainController.disable sets ``_enabled`` and comes back; what it
    hands to the worker is the gain that pins the device where the AGC
    left it, and that is a gain write, not a mode one.  A refresh
    between the two has to find Auto already off.
    """
    view, controller = window()
    assert controller.auto_gain, "the stand-in is meant to start with Auto on"
    release = held(controller)
    try:
        view._auto_gain.setChecked(False)
        view.refresh()

        assert not controller.auto_gain, "the mode waited for the USB write"
        asked = view._asked_for[-1]
        assert not asked.finished, "the worker was supposed to be busy"
        assert asked.kind == GAIN,             f"turning Auto off pins the gain; that is a {asked.kind} write"
    finally:
        release.set()




def test_a_recording_let_go_of_before_it_starts_does_not_start(window):
    """Pressed and pressed again while the receiver was still busy.

    The button is back up, so nothing should be recording - and the
    start still sitting on the worker must not land behind it.
    """
    view, controller = window()
    release = held(controller)
    try:
        view._record_audio.click()
        asked = controller.last_write
        view.refresh()
        assert view._record_audio.isChecked(), "the first press did nothing"

        view._record_audio.click()          # let go before it started
        view.refresh()

        assert not view._record_audio.isChecked()
        assert view._recording_status.text() == ""
    finally:
        release.set()

    assert asked.wait(5), "nobody answered for the start"
    assert asked.cancelled, f"it was not taken back: {asked!r}"
    view.refresh()

    assert not controller.is_recording(), "it started after being let go"
    assert view._recording_status.text() == ""


def test_an_iq_recording_let_go_of_before_it_starts_does_not_start(window):
    view, controller = window()
    release = held(controller)
    try:
        view._record_iq.click()
        asked = controller.last_write
        view._record_iq.click()
        view.refresh()
    finally:
        release.set()

    assert asked.wait(5)
    assert asked.cancelled, f"it was not taken back: {asked!r}"
    assert not controller.is_iq_recording()


def test_a_second_recording_over_the_first_is_reported_not_claimed(window):
    """The stand-in refuses it, as the receiver does, and the window says so."""
    view, controller = window()
    view._record_audio.click()
    controller.settled()
    view.refresh()
    assert controller.is_recording()

    # A second start without a stop, which only something other than
    # the button can ask for.
    asked = controller.start_recording()
    assert asked.wait(5)

    assert asked.failed, "it claimed a recording it did not make"
    assert controller.recording_path is not None
    assert "second" not in str(controller.recording_path)


def test_a_recording_that_is_being_finished_says_so(window):
    """The button is up, and the file is still being written.

    Saying nothing at all would be the same silence the window used to
    keep while a recording was being closed - and the user would have
    no way to know the file was not ready yet.
    """
    view, controller = window()
    view._record_audio.click()
    controller.settled()
    view.refresh()
    assert "recording audio" in view._recording_status.text()

    view._record_audio.click()          # stop
    view.refresh()

    assert not view._record_audio.isChecked(), "it is not recording"
    assert "finishing audio" in view._recording_status.text(), \
        view._recording_status.text()

    controller.finished_the_recordings()
    view.refresh()

    assert view._recording_status.text() == "", \
        view._recording_status.text()


def test_a_recording_can_be_started_again_while_the_last_one_finishes(window):
    """The line says both, and the button is down for the new one."""
    view, controller = window()
    view._record_audio.click()
    controller.settled()
    view._record_audio.click()          # stop; still finishing
    view.refresh()

    release = held(controller)
    try:
        view._record_audio.click()      # start another
        view.refresh()

        text = view._recording_status.text()
        assert "starting audio" in text and "finishing audio" in text, text
        assert view._record_audio.isChecked()
    finally:
        release.set()


# ----------------------------------------------------------------------
# The window is built before the radio is switched on
# ----------------------------------------------------------------------

def test_the_window_is_up_before_the_receiver_starts(qt_app, monkeypatch):
    """Building a window is a gap in the audio, if there is audio.

    Fonts, a graphics context, a few hundred widgets' worth of layout:
    long enough to be heard.  Done first it costs nothing, because
    there is nothing to interrupt yet.
    """
    from fm_radio.gui import main_window

    order = []
    controller = FakeController(snapshot())

    real_show = main_window.ReceiverWindow.show

    def watched_show(self):
        order.append("window shown")
        return real_show(self)

    def switch_on():
        order.append("receiver started")

    monkeypatch.setattr(main_window.ReceiverWindow, "show", watched_show)
    monkeypatch.setattr(main_window.QApplication, "exec",
                        lambda self: (qt_app.processEvents(), 0)[1])

    assert main_window.run_window(controller, switch_on) == 0

    assert order == ["window shown", "receiver started"], order


def test_the_receiver_is_left_alone_when_nobody_hands_over_a_switch(
        qt_app, monkeypatch):
    """A caller that started it already gets the old behaviour."""
    from fm_radio.gui import main_window

    controller = FakeController(snapshot())
    monkeypatch.setattr(main_window.QApplication, "exec",
                        lambda self: (qt_app.processEvents(), 0)[1])

    assert main_window.run_window(controller) == 0


def test_a_receiver_that_will_not_start_is_shown_in_the_window(qt_app,
                                                               monkeypatch):
    """There is a window by then, so there is somewhere to say it.

    The whole way through, not just as far as the controller: the
    reason is recorded, the notice reaches the GUI thread, and the
    window shows it and goes quiet.  Checking the controller alone
    passes even when nothing is ever shown, because run_window joins
    the starting thread before it returns.

    The window's own refresh timer is stopped for this, so that the
    only thing that can have shown the failure is the signal.  With
    it running, a start that takes longer than one interval gets the
    window refreshed on the timer, and a notice that does nothing at
    all passes.
    """
    from fm_radio.gui import main_window

    from fm_radio.exceptions import SDRDeviceError

    controller = FakeController(snapshot())
    controller.device_failure = None
    window = []
    refreshed_on = []

    real_refresh = main_window.ReceiverWindow.refresh
    real_init = main_window.ReceiverWindow.__init__

    def quiet_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._timer.stop()

    def watched_refresh(self):
        refreshed_on.append(threading.current_thread())
        window.append(self)
        return real_refresh(self)

    def refuse():
        raise SDRDeviceError("no radio here")

    def exec_(self):
        # The notice is queued to this thread from the starting
        # thread, so there has to be something to queue before the
        # events are run.  Bounded, and the assertions are out here
        # rather than in the switch, where the start-failure handler
        # would swallow them.
        told = shown.wait(5)
        qt_app.processEvents()
        ran.append(told)
        return 0

    shown = threading.Event()
    ran = []

    monkeypatch.setattr(main_window.ReceiverWindow, "__init__", quiet_init)
    monkeypatch.setattr(main_window.ReceiverWindow, "refresh", watched_refresh)
    monkeypatch.setattr(main_window.QApplication, "exec", exec_)

    original_emit = main_window.ReceiverSwitch._run

    def watched_run(self):
        try:
            original_emit(self)
        finally:
            shown.set()

    monkeypatch.setattr(main_window.ReceiverSwitch, "_run", watched_run)

    assert main_window.run_window(controller, refuse) == 0

    assert ran == [True], "the switch never reported the failure"
    assert controller.device_failure == "no radio here"
    assert refreshed_on, "the window was never told"
    assert all(t is threading.main_thread() for t in refreshed_on), (
        "the widgets were touched from %s" % refreshed_on)
    view = window[0]
    assert view._health.text() == (
        "SDR disconnected - the receiver has stopped"), view._health.text()
    assert view._health.toolTip() == "no radio here", view._health.toolTip()
    assert not view._record_audio.isEnabled(),         "the controls still offer to reach a radio that is not there"
    assert not view._presets.isEnabled()


def test_the_receiver_is_not_started_on_the_gui_thread(qt_app, monkeypatch):
    """Starting takes about a second and a quarter, nearly all of it
    the JIT pre-warm.  On the GUI thread that is a second and a
    quarter of a window that is up and has never painted: a white
    rectangle that looks like a program that has hung.
    """
    from fm_radio.gui import main_window

    where = []
    controller = FakeController(snapshot())

    def switch_on():
        where.append(threading.current_thread())

    monkeypatch.setattr(main_window.QApplication, "exec",
                        lambda self: (qt_app.processEvents(), 0)[1])

    main_window.run_window(controller, switch_on)

    assert where, "the receiver was never started"
    assert where[0] is not threading.main_thread(),         "the receiver was started on the thread that has to paint"


def test_the_window_gets_on_with_things_while_the_receiver_starts(
        qt_app, monkeypatch):
    """The point of the other thread: the GUI thread is free meanwhile.

    The start blocks until the window has serviced its events, which
    it can only do if the start is not what it is doing.
    """
    from fm_radio.gui import main_window

    starting = threading.Event()
    serviced = threading.Event()
    got_a_turn = []
    controller = FakeController(snapshot())

    def switch_on():
        # Reported back rather than asserted here: an assert on this
        # thread is caught by the same handler that catches a radio
        # that will not start, and the test would never see it.
        starting.set()
        got_a_turn.append(serviced.wait(5))

    def exec_(self):
        assert starting.wait(5), "the receiver never began starting"
        qt_app.processEvents()
        serviced.set()
        return 0

    monkeypatch.setattr(main_window.QApplication, "exec", exec_)

    assert main_window.run_window(controller, switch_on) == 0
    assert got_a_turn == [True],         "the window never got a turn while the receiver was starting"


def test_the_window_does_not_return_while_the_receiver_is_starting(
        qt_app, monkeypatch):
    """Whoever runs cleanup() next would be tearing down a receiver
    that is still being built.
    """
    from fm_radio.gui import main_window

    still_starting = threading.Lock()
    entered = threading.Event()
    never = threading.Event()
    controller = FakeController(snapshot())

    def switch_on():
        still_starting.acquire()
        entered.set()
        try:
            never.wait(0.2)         # a start that has not finished yet
        finally:
            still_starting.release()

    def exec_(self):
        assert entered.wait(5), "the receiver never began starting"
        return 0

    monkeypatch.setattr(main_window.QApplication, "exec", exec_)

    main_window.run_window(controller, switch_on)

    assert still_starting.acquire(blocking=False),         "the window returned while the receiver was still starting"
