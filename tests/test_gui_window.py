"""The status window: what it shows, and what it asks the receiver to do.

Driven against a stand-in controller rather than a real receiver, because
what is being checked is the window — that it reads state only through
``get_status()``, changes it only through the facade, and does not fight the
user for a control it is also updating.

Qt runs offscreen (see the ``qt_app`` fixture), so these need no display.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import traceback
from dataclasses import replace

import pytest

from fm_radio.dsp_settings import DspSettings
from fm_radio.multipath import CLEAN_AM_DEPTH, NOISE_AM_DEPTH
from fm_radio.telemetry import SILENCE_DBFS, StatusSnapshot

#: What a receiver would hand the settings tab as its defaults.  A
#: real DspSettings, so the tab's conversions meet the real thing,
#: with the standard chain's subcarrier phase.
DSP_DEFAULTS = DspSettings(
    force_blend_factor=None,
    subcarrier_phase_offset_rad=math.radians(85.0),
    mono_delay_samples=0,
    iq_phase_correction_enabled=True,
    lr_high_max_gain=1.0,
    lr_super_high_max_gain=1.0,
    side_nr_enabled=True,
    side_nr_alpha_floor=0.30,
    side_nr_beta=1.0,
)

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
        self.spectrum = None
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
        # The DSP facade the settings tab uses.  update_ goes
        # through the real dataclass, so a value the settings would
        # refuse is refused here too.
        self.dsp_defaults = DSP_DEFAULTS
        self.dsp_settings = DSP_DEFAULTS
        self.dsp_updates: list[dict] = []
        self.dsp_sets: list[DspSettings] = []
        self.calls: list[tuple] = []
        self.cleanups: list[str] = []
        self.quit_event = _Event()
        self.station = _Station("TOKYO FM")
        self.presets = [("TOKYO FM", 80.0e6), ("J-WAVE", 81.3e6)]
        # Where the radio is, and what it knows is out there.  Empty
        # is the honest default: a receiver with no catalogue cannot
        # work out where it is, and nothing should ask.
        self.catalogue: list = []
        self.area: str | None = None
        self.saved: list[str] = []
        self.save_error: Exception | None = None
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

    def get_spectrum(self):
        """The picture of the band, or None when there is not one.

        The real one publishes these at 10 Hz from the processing
        thread and hides any from before a retune, so None is an
        ordinary answer and the window has to cope with it.
        """
        return self.spectrum

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

    def stations_at(self, freqs):
        """What the dial would call each of *freqs*; the recordings tab
        asks."""
        freqs = list(freqs)
        self.calls.append(("stations_at", freqs))
        names = {80.0e6: "TOKYO FM", 81.3e6: "J-WAVE"}
        return {f: _Station(names[f]) if f in names else None
                for f in freqs}

    def get_dsp_defaults(self) -> DspSettings:
        return self.dsp_defaults

    def get_dsp_settings(self) -> DspSettings:
        return self.dsp_settings

    def update_dsp_settings(self, **changes) -> DspSettings:
        self.dsp_updates.append(dict(changes))
        self.dsp_settings = replace(self.dsp_settings, **changes)
        return self.dsp_settings

    def set_dsp_settings(self, settings: DspSettings) -> None:
        self.dsp_sets.append(settings)
        self.dsp_settings = settings

    def get_stations_list(self):
        return list(self.presets)

    def get_catalogue(self):
        return list(self.catalogue)

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

    def remember_where_this_is(self, area):
        """Write it and use it, as the real one does - or neither.

        The real one names by the new area only once the file has
        taken it, so a failure here leaves the area as it was.
        """
        self.calls.append(("remember_where_this_is", area))
        if self.save_error is not None:
            raise self.save_error
        self.area = area
        self.saved.append(area)
        return "/home/someone/.config/fm_radio/stations.toml"

    def cleanup(self) -> None:
        """Give the device back, as the window asks when one goes.

        The window calls this off its own thread once it has shown
        why nothing is coming; the real one closes the recording and
        the audio stream.
        """
        self.cleanups.append("cleanup")

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
        am_depth=0.06,
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
    """A window over a stand-in controller, closed afterwards.

    Anything a slot raises is caught on the way out and fails the
    test.  Qt has nowhere to send an exception raised in a slot, so
    it prints it and carries on - which meant a window that broke
    halfway through handling a signal left every assertion after it
    still passing, because what came before had already been done.
    """
    built = []
    swallowed = []
    was = sys.excepthook
    sys.excepthook = lambda *trouble: swallowed.append(trouble)

    def _build(controller=None):
        controller = controller or FakeController(snapshot())
        instance = ReceiverWindow(controller)
        built.append(instance)
        return instance, controller

    try:
        yield _build
        for instance in built:
            instance._timer.stop()
            instance.close()
    finally:
        sys.excepthook = was
    assert not swallowed, "a slot raised: %s" % "".join(
        traceback.format_exception(*swallowed[0]))


# ----------------------------------------------------------------------
# The tabs
# ----------------------------------------------------------------------

def test_the_controls_are_on_the_radio_tab(window):
    """The same controls, one page down.

    Named rather than counted: the next change adds the DSP tab's
    contents, and a test that says "two tabs" would pass while they
    were in the wrong one.
    """
    view, _ = window()

    assert [view._tabs.tabText(i) for i in range(view._tabs.count())] == [
        "Radio", "DSP", "Recordings"]
    radio = view._tabs.widget(0)
    for control in (view._frequency, view._down, view._up, view._presets,
                    view._found, view._scan_button, view._band,
                    view._blend, view._auto_gain, view._gain_slider,
                    view._record_audio, view._record_iq):
        assert radio.isAncestorOf(control), (
            "%r is not on the Radio tab" % (control,))


def test_the_window_still_updates_the_tab_nobody_is_looking_at(window):
    """A control that stopped being updated out of sight would be
    wrong the moment its tab came back.
    """
    view, controller = window(FakeController(snapshot(freq_hz=80.0e6)))
    view._tabs.setCurrentIndex(1)               # the DSP tab
    assert view._tabs.currentIndex() == 1

    controller.status = snapshot(freq_hz=81.3e6, station="J-WAVE")
    view.refresh()

    assert view._frequency.text() == "81.3 MHz"
    assert view._station.text() == "J-WAVE"


def test_the_dsp_tab_holds_the_settings(window):
    """The nine of them, and they are usable to begin with."""
    view, _ = window()
    page = view._tabs.widget(1)

    assert page is view._dsp
    assert len(view._dsp._rows) == 9
    assert view._dsp._reset_all.isEnabled()


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
    ({"stereo": True, "blend_factor": 0.3}, "0.30"),
])
def test_the_mode_distinguishes_mono_from_a_half_open_blend(
        window, overrides, expected):
    view, _ = window(FakeController(snapshot(**overrides)))
    assert view._mode.text() == expected


@pytest.mark.parametrize("blend", [0.51, 0.76, 0.92, 0.994])
def test_a_blend_the_receiver_calls_stereo_is_still_shown_as_a_figure(
        window, blend):
    """The receiver calls anything over half stereo; the bar does not.

    STEREO beside a bar half way along reads as a contradiction, and
    the bar is the reason this exists: it says how much there is.
    So the word only claims the whole of it when the bar is full,
    and says the figure otherwise - which is also what the bar shows.
    """
    view, _ = window(FakeController(snapshot(stereo=True,
                                             blend_factor=blend)))

    assert view._mode.text() == f"{blend:.2f}", "the word overstates it"
    assert view._blend.value() == round(blend * 100)


@pytest.mark.parametrize("blend,word", [
    (1.0, "STEREO"),
    (0.995, "STEREO"),
])
def test_the_word_claims_the_whole_of_it_only_when_the_bar_is_full(
        window, blend, word):
    view, _ = window(FakeController(snapshot(stereo=True,
                                             blend_factor=blend)))

    assert view._blend.value() == 100
    assert view._mode.text() == word


@pytest.mark.parametrize("blend", [-0.2, 1.4, float("nan")])
def test_the_word_never_says_something_the_bar_cannot_show(window, blend):
    """The bar clamps, so the figure beside it has to clamp with it.

    It printed the raw blend, so a blend of -0.2 was an empty bar
    labelled -0.20 and a NaN was an empty bar labelled nan.
    """
    view, _ = window(FakeController(snapshot(stereo=True,
                                             blend_factor=blend)))

    shown = view._mode.text()
    if shown == "STEREO":
        assert view._blend.value() == 100
    else:
        assert float(shown) == pytest.approx(view._blend.value() / 100.0)


# ----------------------------------------------------------------------
# How much stereo there is
# ----------------------------------------------------------------------

@pytest.mark.parametrize("blend,expected", [
    (1.0, 100),
    (0.75, 75),
    (0.3, 30),
    (0.0, 0),
])
def test_the_bar_follows_the_blend(window, blend, expected):
    """The reading the word cannot give: how far along it is."""
    view, _ = window(FakeController(snapshot(stereo=True,
                                             blend_factor=blend)))

    assert view._blend.value() == expected


def test_mono_shows_an_empty_bar_whatever_the_blend_says(window):
    """The blend factor starts at 1.0 and the mono path never moves it.

    So a receiver asked for mono reports full blend, and a bar that
    believed it would sit at the top saying the stereo image is all
    the way through when there is no stereo image at all.
    """
    view, _ = window(FakeController(snapshot(stereo=False,
                                             blend_factor=1.0)))

    assert view._blend.value() == 0
    assert view._mode.text() == "MONO"


@pytest.mark.parametrize("blend,expected", [
    (1.4, 100),
    (-0.2, 0),
])
def test_a_blend_outside_its_range_stays_on_the_bar(window, blend,
                                                    expected):
    """Qt clamps it anyway; this says what it should clamp to."""
    view, _ = window(FakeController(snapshot(stereo=True,
                                             blend_factor=blend)))

    assert view._blend.value() == expected


def test_the_blend_line_does_not_change_width_as_it_settles(window,
                                                              qt_app):
    """Or the bar shrinks and grows under a receiver finding its feet.

    A minimum width picked by eye is not a fixed width: the column
    grows the moment the text is wider than it.

    In silence, because the level labels share that column and
    "-12.3 dBFS" is wider than anything the blend line says - so
    with a signal in them they size the column and the blend line
    could be any width at all without it showing.
    """
    def quiet(**kw):
        return snapshot(level_left_dbfs=SILENCE_DBFS,
                        level_right_dbfs=SILENCE_DBFS, **kw)

    controller = FakeController(quiet(stereo=True, blend_factor=0.30))
    view, _ = window(controller)
    view.resize(800, 600)
    view.show()
    qt_app.processEvents()
    assert view._left_db.text() == "--", "the meters were meant to be silent"
    while_blending = view._blend.width()

    for stereo, blend in ((True, 1.0), (True, 0.0), (False, 1.0)):
        controller.status = quiet(stereo=stereo, blend_factor=blend)
        view.refresh()
        qt_app.processEvents()

        assert view._blend.width() == while_blending, (
            "the bar went from %d to %d wide showing %r"
            % (while_blending, view._blend.width(), view._mode.text()))


def test_every_figure_the_line_can_print_was_measured(window):
    """The column is sized from _BLEND_WORDS and nothing else.

    So anything the blend line can put in it has to be in there, or
    the width was measured against a string the label never shows
    and the column grows for the one it does.  One figure taken as a
    stand-in for the rest only holds in a font whose digits are all
    the same width.
    """
    from fm_radio.gui import main_window

    controller = FakeController(snapshot())
    view, _ = window(controller)
    said = set()

    for hundredths in range(101):
        for stereo in (True, False):
            controller.status = snapshot(stereo=stereo,
                                         blend_factor=hundredths / 100.0)
            view.refresh()
            said.add(view._mode.text())
    controller.status = None
    view.refresh()
    said.add(view._mode.text())

    unmeasured = said - set(main_window._BLEND_WORDS)
    assert not unmeasured, (
        "the line can say %s, which nothing measured"
        % sorted(unmeasured))


def test_the_bar_empties_when_there_is_no_snapshot(window):
    """A stale bar is the window saying the radio is still playing."""
    controller = FakeController(snapshot(stereo=True, blend_factor=1.0))
    view, _ = window(controller)
    assert view._blend.value() == 100

    controller.status = None
    view.refresh()

    assert view._blend.value() == 0
    assert view._mode.text() == "--"


# ----------------------------------------------------------------------
# How clean the channel is
# ----------------------------------------------------------------------

@pytest.mark.parametrize("depth,figure", [
    (0.0, "0.000"),
    (0.016, "0.016"),
    (0.067, "0.067"),
    (0.592, "0.592"),
])
def test_the_am_depth_is_shown_as_it_was_measured(window, depth, figure):
    view, _ = window(FakeController(snapshot(am_depth=depth)))

    assert view._am_depth_value.text() == figure


@pytest.mark.parametrize("depth,expected", [
    (0.0, 0),
    (0.016, 3),                     # 82.5 MHz, the cleanest measured
    (0.067, 13),                    # 80.0 MHz
    (NOISE_AM_DEPTH, 100),          # what noise reads
    (0.9, 100),                     # past it, and the bar stops
])
def test_the_bar_fills_towards_what_noise_reads(window, depth, expected):
    """Everything worth telling apart is between clean and noise.

    Scaling to 1.0 - a figure nothing reaches - would put every real
    station in the bottom tenth of the bar and the difference
    between a good one and a bad one inside a pixel.
    """
    view, _ = window(FakeController(snapshot(am_depth=depth)))

    assert view._am_depth.value() == expected


def test_a_clean_station_barely_moves_the_bar(window):
    """The reading a listener wants at a glance: empty is good."""
    view, _ = window(FakeController(snapshot(am_depth=CLEAN_AM_DEPTH)))

    assert view._am_depth.value() < 20


def test_an_unmeasurable_am_depth_is_not_shown_as_clean(window):
    """None is "nothing to measure", and 0.000 is the best signal
    there is.  Showing the second for the first is the mistake the
    measurement itself made before it was fixed.
    """
    view, _ = window(FakeController(snapshot(am_depth=None)))

    assert view._am_depth_value.text() == "--"
    assert view._am_depth.value() == 0


def test_the_am_depth_empties_when_there_is_no_snapshot(window):
    controller = FakeController(snapshot(am_depth=0.5))
    view, _ = window(controller)
    assert view._am_depth.value() > 0

    controller.status = None
    view.refresh()

    assert view._am_depth.value() == 0
    assert view._am_depth_value.text() == "--"


def test_every_reading_that_can_be_missing_says_so_the_same_way(window):
    """One mark, one place: see NOTHING_MEASURED.

    Two of these can come back unmeasurable and more are coming, and
    the mistake to avoid is a display that prints a number for one
    of them - on both of these zero is the best possible signal.
    """
    from fm_radio.gui import main_window

    view, _ = window(FakeController(snapshot(pilot_snr_db=None,
                                             am_depth=None)))

    assert view._pilot.text() == main_window.NOTHING_MEASURED
    assert view._am_depth_value.text() == main_window.NOTHING_MEASURED


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
# Sweeping the band from the window
# ----------------------------------------------------------------------

def a_find(mhz: float, power: float = -10.0, sort=None):
    """One of the things a scan hands back."""
    from fm_radio import band_scan

    return band_scan.Signal(
        mhz * 1e6, power, 52.0,
        sort if sort is not None else band_scan.CONFIRMED)


def a_scan(window, monkeypatch, found=None, fails=None, hold=None):
    """A window whose Scan button runs a stand-in sweep.

    The sweep still goes on its own thread - that is the thing worth
    keeping - but it sweeps a list instead of a band.
    """
    from fm_radio.gui import main_window

    ran = {}

    class StandIn:
        def __init__(self, controller, on_progress=None) -> None:
            ran["controller"] = controller
            self._say = on_progress
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def run(self, listen_sec=0.25):
            if self._say is not None:
                self._say("looking at 76.4 MHz (1 of 24)")
            if hold is not None:
                assert hold.wait(5), "the test never let the sweep finish"
            if fails is not None:
                raise fails
            return list(found or [])

    monkeypatch.setattr(main_window, "BandScan", StandIn)
    view, controller = window(FakeController(snapshot()))
    return view, controller, ran


def finish(view, qt_app, timeout: float = 5.0) -> None:
    """Let the sweeping thread end and its signals be delivered."""
    sweep = view._sweep
    if sweep is not None:
        sweep.wait()
    deadline = time.monotonic() + timeout
    while view._sweep is not None and time.monotonic() < deadline:
        qt_app.processEvents()


def test_the_recordings_are_read_when_first_looked_at_and_not_before(
        window, monkeypatch, qt_app):
    """Not when the window is built, not on a refresh, once on first look.

    A refresh is twenty times a second, and building the window is
    already the slowest thing it does; a directory of a few hundred
    sidecars, each part opened, belongs to neither.
    """
    from fm_radio.gui import recordings_tab

    reads = []
    monkeypatch.setattr(recordings_tab, "scan_recordings",
                        lambda directory: reads.append(directory) or [])
    view, _ = window(FakeController(snapshot()))
    for _ in range(3):
        view.refresh()
    qt_app.processEvents()
    assert reads == [], "the directory was read before anyone looked"

    view._tabs.setCurrentWidget(view._recordings)
    view._recordings._scan.wait()
    deadline = time.monotonic() + 5.0
    while view._recordings._scan is not None and time.monotonic() < deadline:
        qt_app.processEvents()

    view._tabs.setCurrentIndex(0)
    view._tabs.setCurrentWidget(view._recordings)
    view.refresh()
    qt_app.processEvents()

    assert len(reads) == 1
    assert view._tabs.tabText(view._tabs.indexOf(view._recordings)) == (
        "Recordings")


def test_the_sweep_does_not_run_on_the_thread_that_draws(window,
                                                          monkeypatch,
                                                          qt_app):
    """Five seconds of retuning on the GUI thread is a white window.

    The same fault #47 was about, and the same answer.
    """
    where = []

    from fm_radio.gui import main_window

    class WatchingStandIn:
        def __init__(self, controller, on_progress=None):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def run(self, listen_sec=0.25):
            where.append(threading.current_thread())
            return []

    monkeypatch.setattr(main_window, "BandScan", WatchingStandIn)
    view, _ = window(FakeController(snapshot()))

    view._scan_button.click()
    finish(view, qt_app)

    assert where, "the sweep never ran"
    assert where[0] is not threading.main_thread()


def test_what_the_sweep_found_can_be_tuned_to(window, monkeypatch, qt_app):
    """The point of scanning is to pick from what it found."""
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(89.7), a_find(81.3)])

    view._scan_button.click()
    finish(view, qt_app)

    assert view._found.count() == 3, "a heading and two finds"
    view._found.setCurrentIndex(1)
    view._found_chosen(1)

    assert [freq for what, freq in controller.calls if what == "tune"] == [
        pytest.approx(89.7e6)]


def test_each_find_says_where_it_is_and_what_it_is(window, monkeypatch,
                                                    qt_app):
    from fm_radio import band_scan

    view, _controller, _ = a_scan(window, monkeypatch, found=[
        a_find(89.7, -7.0, band_scan.CONFIRMED),
        a_find(82.1, -25.0, band_scan.LIKELY_SKIRT),
        a_find(90.5, -33.0, band_scan.UNCONFIRMED),
    ])

    view._scan_button.click()
    finish(view, qt_app)

    lines = [view._found.itemText(i) for i in range(1, view._found.count())]
    assert "89.7 MHz" in lines[0] and "stereo" in lines[0]
    assert "spill?" in lines[1]
    assert "no pilot" in lines[2]


def test_the_tuner_is_the_sweeps_while_it_runs(window, monkeypatch, qt_app):
    """Anything else that tunes makes the sweep fail, so nothing else
    is offered the chance.
    """
    let_it_go = threading.Event()
    view, _controller, _ = a_scan(window, monkeypatch, hold=let_it_go)

    view._scan_button.click()
    qt_app.processEvents()

    try:
        for widget in (view._down, view._up, view._presets, view._found,
                       view._auto_gain, view._gain_slider):
            assert not widget.isEnabled(), "%s was left live" % widget
        assert view._scan_button.text() == "Stop"
    finally:
        let_it_go.set()
        finish(view, qt_app)

    for widget in (view._down, view._up, view._presets, view._found):
        assert widget.isEnabled(), "%s was not given back" % widget
    assert view._scan_button.text() == "Scan"


def test_the_button_stops_the_sweep_it_started(window, monkeypatch, qt_app):
    let_it_go = threading.Event()
    view, _controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()

    view._scan_button.click()           # now it says Stop

    assert view._sweep._scan.cancelled
    let_it_go.set()
    finish(view, qt_app)


def test_a_sweep_that_failed_says_so_and_gives_the_tuner_back(
        window, monkeypatch, qt_app):
    """A sweep fails when something else tunes, which is a thing
    that happens; the window has to be usable afterwards.
    """
    from fm_radio.band_scan import ScanFailed

    view, _controller, _ = a_scan(
        window, monkeypatch, fails=ScanFailed("the tuner moved"))

    view._scan_button.click()
    finish(view, qt_app)

    assert "the tuner moved" in view._health.text()
    assert view._down.isEnabled(), "the tuner was not given back"
    assert view._scan_button.text() == "Scan"


def test_a_sweep_that_found_nothing_says_that_too(window, monkeypatch,
                                                   qt_app):
    view, _controller, _ = a_scan(window, monkeypatch, found=[])

    view._scan_button.click()
    finish(view, qt_app)

    assert "nothing" in view._health.text()
    assert view._found.count() == 1, "a heading and no finds"


def test_the_refresh_does_not_hand_the_gain_back_mid_sweep(
        window, monkeypatch, qt_app):
    """The sweep turns the AGC off, and the window used to believe it.

    A refresh saw manual gain, handed the slider back, and a gain
    moved then puts the sweep's hops in different units - two
    frequencies measured at two gains, compared as though they were
    not.  Nothing in the sweep can notice that; it only watches the
    tuning.
    """
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()
    assert not view._gain_slider.isEnabled()

    # What BandScan does to the receiver, arriving in a snapshot.
    controller.status = snapshot(auto_gain=False)
    view.refresh()
    qt_app.processEvents()

    try:
        assert not view._gain_slider.isEnabled(), \
            "the slider was handed back in the middle of a sweep"
        assert not view._auto_gain.isEnabled()
    finally:
        let_it_go.set()
        finish(view, qt_app)


def test_the_gain_comes_back_on_manual_when_the_sweep_is_over(
        window, monkeypatch, qt_app):
    """And the slider follows the Auto box again, as it did before."""
    view, controller, _ = a_scan(window, monkeypatch, found=[])
    view._scan_button.click()
    finish(view, qt_app)

    controller.status = snapshot(auto_gain=False)
    view.refresh()

    assert view._gain_slider.isEnabled()

    controller.status = snapshot(auto_gain=True)
    view.refresh()

    assert not view._gain_slider.isEnabled()


def test_what_the_sweep_is_doing_survives_a_refresh(window, monkeypatch,
                                                     qt_app):
    """The window refreshes every 50 ms and the sweep takes seconds.

    The progress line shared a sort with a tune's, and the refresh
    clears that whenever the window has no tune of its own
    outstanding - which is every refresh during a scan, because the
    scan's tunes are the scan's.  The line lasted under 50 ms.
    """
    let_it_go = threading.Event()
    view, _controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()
    said = view._health.text()
    assert "76.4 MHz" in said, said

    view.refresh()

    try:
        assert view._health.text() == said, "the refresh wiped it"
    finally:
        let_it_go.set()
        finish(view, qt_app)


def test_the_dsp_settings_go_with_the_rest_of_the_controls(
        window, monkeypatch, qt_app):
    """A sweep has the demodulator; a receiver that has gone has
    nothing.  The DSP tab is decided in the same one place as
    everything else, so it cannot disagree with the rest.
    """
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    a_control = view._dsp._rows[1].widgets[0]
    assert a_control.isEnabled(), "this test needs a live receiver first"

    view._scan_button.click()
    qt_app.processEvents()
    try:
        assert not a_control.isEnabled(), "the sweep owns the demodulator"
        assert not view._dsp._reset_all.isEnabled()
    finally:
        let_it_go.set()
        finish(view, qt_app)

    assert a_control.isEnabled(), "the sweep did not hand them back"

    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    qt_app.processEvents()
    assert not a_control.isEnabled(), "there is no receiver to ask"
    assert not view._dsp._reset_all.isEnabled()


def test_a_device_that_goes_mid_sweep_keeps_the_controls_shut(
        window, monkeypatch, qt_app):
    """A sweep that ends afterwards must not hand them back.

    The refresh timer stops when the device goes, so anything
    wrongly re-enabled then stays that way - offering to tune a
    receiver that has been given back.
    """
    from fm_radio.band_scan import ScanFailed

    view, controller, _ = a_scan(
        window, monkeypatch, fails=ScanFailed("the SDR went"))
    view._scan_button.click()
    finish(view, qt_app)
    assert view._down.isEnabled(), "this test needs a live receiver first"

    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    qt_app.processEvents()

    for widget in (view._down, view._up, view._presets, view._found,
                   view._auto_gain, view._gain_slider, view._scan_button):
        assert not widget.isEnabled(), "%s still offers to work" % widget


def test_a_sweep_ending_after_the_device_went_changes_nothing(
        window, monkeypatch, qt_app):
    """The order that really happens: the device goes, then the
    sweep notices and says it failed.
    """
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()

    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    let_it_go.set()
    finish(view, qt_app)

    for widget in (view._down, view._up, view._presets, view._found,
                   view._auto_gain, view._gain_slider, view._scan_button):
        assert not widget.isEnabled(), "%s was handed back" % widget


def test_a_device_that_goes_mid_sweep_cancels_it(window, monkeypatch,
                                                  qt_app):
    """There is no band to sweep any more."""
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()
    sweep = view._sweep

    controller.device_failure = "the SDR was unplugged"
    view.refresh()

    assert sweep._scan.cancelled
    let_it_go.set()
    finish(view, qt_app)


def test_finished_sweeps_do_not_pile_up_under_the_window(window,
                                                          monkeypatch,
                                                          qt_app):
    """Each one holds a demodulator and a spectrum maker.

    Their parent is the window, so without being deleted every
    sweep ever run stays a child of it.
    """
    from fm_radio.gui.main_window import Sweep

    view, _controller, _ = a_scan(window, monkeypatch, found=[])

    for _ in range(4):
        view._scan_button.click()
        finish(view, qt_app)
    qt_app.processEvents()              # deleteLater happens here

    assert view.findChildren(Sweep) == []


def test_recording_is_not_offered_during_a_sweep(window, monkeypatch,
                                                  qt_app):
    """The button could not keep its promise.

    A tune shuts any recording that is running, and a sweep is two
    dozen tunes: a recording started mid-sweep is a file less than
    one hop long, ended by the next hop.
    """
    let_it_go = threading.Event()
    view, _controller, _ = a_scan(window, monkeypatch, hold=let_it_go)

    view._scan_button.click()
    qt_app.processEvents()

    try:
        assert not view._record_audio.isEnabled()
        assert not view._record_iq.isEnabled()
    finally:
        let_it_go.set()
        finish(view, qt_app)

    assert view._record_audio.isEnabled(), "not given back afterwards"
    assert view._record_iq.isEnabled()


def test_the_scan_button_says_a_recording_will_be_ended(window):
    """It is a surprise worth not having."""
    view, _ = window(FakeController(snapshot()))

    assert "recording" in view._scan_button.toolTip()


def test_closing_the_window_stops_a_sweep(window, monkeypatch, qt_app):
    """It would go on retuning a receiver being taken apart."""
    let_it_go = threading.Event()
    view, _controller, _ = a_scan(window, monkeypatch, hold=let_it_go)
    view._scan_button.click()
    qt_app.processEvents()
    sweep = view._sweep

    let_it_go.set()                     # it would end on its own too
    view.close()

    assert sweep._scan.cancelled
    assert view._sweep is None


def test_the_window_says_where_the_sweep_has_got_to(window, monkeypatch,
                                                     qt_app):
    """Five seconds is long enough to wonder whether it is working."""
    view, _controller, _ = a_scan(window, monkeypatch, found=[a_find(89.7)])

    view._scan_button.click()
    finish(view, qt_app)

    # The progress line is overwritten by the result; what matters is
    # that the sweep's own words reached the window at all.
    assert view._health.text()


# ----------------------------------------------------------------------
# What the sweep makes of where the radio is
# ----------------------------------------------------------------------

def a_transmitter(mhz: float, site: str = "東京",
                  area: str = "関東"):
    """One catalogue entry, of the shape where_this_is scores."""
    from fm_radio.stations import Station

    return Station(name="%s %.1f" % (site, mhz), freq_mhz=mhz, site=site,
                   area=area)


def one_place(mhz=(80.0, 81.3)):
    """A catalogue that puts those frequencies in one place.

    Two of them from one site is what where_this_is wants: two
    finds it explains, and nothing else explaining either.
    """
    return [a_transmitter(each) for each in mhz]


def watch_the_boxes(monkeypatch, answer=None, meanwhile=None):
    """Catch the modal boxes instead of putting one on the screen.

    ``exec`` is what is patched, not the window's own asking, so
    what these tests read is the box the window really built - its
    words, its buttons, and what it made of the answer.  A box that
    reached a screen here would hold the test up until it timed out.
    """
    from PySide6.QtWidgets import QMessageBox

    shown = []

    def instead(box):
        shown.append(box)
        if meanwhile is not None and len(shown) == 1:
            # What a timer firing inside the modal loop does.  Qt
            # goes on delivering them while a box is up, so the
            # window behind it keeps refreshing - and can find out
            # that the receiver has gone.
            meanwhile()
        return QMessageBox.Yes if answer is None else answer

    monkeypatch.setattr(QMessageBox, "exec", instead)
    return shown


def test_a_sweep_that_places_the_radio_offers_to_keep_it(window,
                                                          monkeypatch,
                                                          qt_app):
    """The sweep is the one moment the evidence is in hand."""
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert len(shown) == 1, "it asked %d times" % len(shown)
    said = shown[0].text() + shown[0].informativeText()
    assert "関東" in said, said
    assert controller.saved == ["関東"]
    assert "関東" in view._health.text()


def test_it_asks_before_it_writes(window, monkeypatch, qt_app):
    """The file is the user's, and an area they are not in takes
    the name off every station they can hear.
    """
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    from PySide6.QtWidgets import QMessageBox
    shown = watch_the_boxes(monkeypatch, answer=QMessageBox.No)

    view._scan_button.click()
    finish(view, qt_app)

    assert len(shown) == 1, "it did not ask"
    assert controller.saved == [], "it wrote anyway"
    assert ("remember_where_this_is", "関東") not in controller.calls


def test_a_question_with_two_answers_offers_two_buttons(window,
                                                         monkeypatch,
                                                         qt_app):
    """A box with only Ok on it is not a question, whatever it says."""
    from PySide6.QtWidgets import QMessageBox

    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    buttons = shown[0].standardButtons()
    assert buttons & QMessageBox.Yes and buttons & QMessageBox.No


def test_it_says_nothing_when_the_area_is_the_one_already_set(window,
                                                               monkeypatch,
                                                               qt_app):
    """A box that only ever says "no change" is one the user learns
    to dismiss unread, and the next one will be dismissed with it.
    """
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    controller.area = "関東"
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert shown == [], "it asked about the area it is already in"
    assert controller.saved == []


@pytest.mark.parametrize("already", [None, "関東"])
def test_it_says_nothing_when_the_sweep_points_nowhere(window, monkeypatch,
                                                        qt_app, already):
    """Two sites that explain the same finds are not an answer, and
    a guess put in the file names every station wrongly.

    Both ways round: with an area already set, "nowhere" is not the
    same answer as the one set, and a window comparing the two
    without looking at what nowhere is would ask about it.
    """
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.area = already
    controller.catalogue = one_place() + [
        a_transmitter(80.0, "札幌", "北海道"),
        a_transmitter(81.3, "札幌", "北海道"),
    ]
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert shown == [], "it asked about a place it had not worked out"
    assert controller.saved == []


def test_a_sweep_the_user_stopped_is_not_asked_about(window, monkeypatch,
                                                      qt_app):
    """It saw part of the band, and they were on their way somewhere.

    What it found before it stopped still comes back, so what it
    found is not what says this: the sweep having been cancelled is.
    """
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    qt_app.processEvents()
    view._scan_button.click()           # Stop
    let_it_go.set()
    finish(view, qt_app)

    assert shown == [], "it asked after a sweep the user stopped"
    assert controller.saved == []


def test_a_sweep_the_device_went_out_from_under_is_not_asked_about(
        window, monkeypatch, qt_app):
    """The window is already showing why the radio stopped."""
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    qt_app.processEvents()
    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    let_it_go.set()
    finish(view, qt_app)

    assert shown == [], "it asked over a window showing a dead radio"
    assert controller.saved == []


def test_a_sweep_the_window_no_longer_holds_is_not_asked_about(window,
                                                                monkeypatch,
                                                                qt_app):
    """Closing takes the sweep, and its last signal can be behind it.

    stop_any_sweep waits for the sweeping thread, but the signal it
    already emitted is a queued event: it is delivered by whatever
    event loop runs next, which is the one taking the window apart.
    """
    view, controller = window(FakeController(snapshot()))
    controller.catalogue = one_place()
    shown = watch_the_boxes(monkeypatch)

    view.stop_any_sweep()               # as closing does
    view._scan_ended([a_find(80.0), a_find(81.3)], "")

    assert shown == [], "it asked about a sweep it had let go of"
    assert controller.saved == []


def test_a_radio_that_goes_while_the_question_is_up_still_says_so(
        window, monkeypatch, qt_app):
    """The line about the device is the last one the window writes.

    The refresh stops with it, so a notice put up after it stays up:
    the window would sit there saying the stations are named from
    関東 now, about a radio that is not there.
    """
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()

    def the_cable_comes_out():
        controller.device_failure = "the SDR was unplugged"
        view.refresh()                  # the timer, inside the modal loop

    watch_the_boxes(monkeypatch, meanwhile=the_cable_comes_out)

    view._scan_button.click()
    finish(view, qt_app)

    assert "SDR disconnected" in view._health.text(), view._health.text()
    # Still written: the file is about the next start, and the user
    # answered the question.
    assert controller.saved == ["関東"]


def test_a_sweep_that_ends_after_the_device_went_does_not_talk_over_it(
        window, monkeypatch, qt_app):
    """The same line, and the same reason.  A sweep ends a few
    hundred milliseconds after the device goes - it stops at the
    next hop - and it used to report what it found over the top of
    the explanation the user was reading.
    """
    let_it_go = threading.Event()
    view, controller, _ = a_scan(window, monkeypatch, hold=let_it_go,
                                 found=[a_find(89.7)])

    view._scan_button.click()
    qt_app.processEvents()
    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    let_it_go.set()
    finish(view, qt_app)

    assert "SDR disconnected" in view._health.text(), view._health.text()


def test_nothing_is_written_over_the_line_about_the_device(window):
    """One place decides it, for every notice there is."""
    view, controller = window(FakeController(snapshot()))
    controller.device_failure = "the SDR was unplugged"
    view.refresh()
    was = view._health.text()

    view._set_notice("something happened")

    assert view._health.text() == was


def test_a_file_that_cannot_be_changed_is_said_in_full(window, monkeypatch,
                                                        qt_app):
    """What it refuses to do it says how to do, and the words are
    the only thing the user has left to go on - the file, and the
    line to type into it.  A status line would take the end off.
    """
    from fm_radio.stations import WillNotEdit
    from PySide6.QtWidgets import QMessageBox

    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    controller.save_error = WillNotEdit(
        '/home/someone/stations.toml cannot be read as TOML, so it will '
        'not be written to; add: area = "関東"')
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert len(shown) == 2, "asked, and then said nothing about it"
    told = shown[1]
    assert str(controller.save_error) in told.text()
    assert told.icon() == QMessageBox.Warning
    assert "not saved" in view._health.text()


def test_the_window_still_works_after_a_file_it_could_not_change(
        window, monkeypatch, qt_app):
    """It is one setting, not the end of the session."""
    from fm_radio.stations import WillNotEdit

    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    controller.save_error = WillNotEdit("no")
    watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert view._scan_button.isEnabled()
    assert view._up.isEnabled()
    assert view._found.count() == 3, "the finds went with the failure"


def test_a_file_that_will_not_open_is_said_in_full_too(window, monkeypatch,
                                                        qt_app):
    """A read-only directory is the same story as a file that will
    not parse: nothing was written, and the user has to do it.
    """
    view, controller, _ = a_scan(window, monkeypatch,
                                 found=[a_find(80.0), a_find(81.3)])
    controller.catalogue = one_place()
    controller.save_error = PermissionError(13, "Permission denied")
    shown = watch_the_boxes(monkeypatch)

    view._scan_button.click()
    finish(view, qt_app)

    assert len(shown) == 2
    assert "Permission denied" in shown[1].text()


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
