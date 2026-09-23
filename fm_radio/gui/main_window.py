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
"""The receiver's status window.

The Radio tab is redrawn on a timer.  Each tick starts by asking whether
the device has gone (``device_failure``).  While it has not, most of what
the tab shows comes from one call to ``controller.get_status()``, but not
all of it: on the same tick the window asks the facade for the band
picture (``get_spectrum()``) and whether each recorder is running or
finishing; it looks at the requests the facade handed back for the
writes it asked the SDR for, to see how they went; and whenever there is
no snapshot it asks for the frequency, the station and the gain itself.

Once the device has gone, the snapshot, the band picture and the
requests are not read again, and the recorders are asked about once.
Each tick asks for the frequency, the station, the gain and whether it
is manual, as it does when there is no snapshot, then puts "no device"
where the station was; the timer runs on only until a recording that
was running has been closed - see ``_show_the_device_has_gone``.

The DSP tab reads its settings once, when it is built; the Recordings
tab reads the disk the first time it is chosen, and again when its
Reload button is pressed.

The window and its tabs touch the receiver only through the
controller's facade, with one exception: closing the window sets the
controller's ``quit_event`` directly - the event the receiver's threads
watch, and the one the CLI sets to quit.  When the window's loop has
ended, ``__main__`` calls ``cleanup()``, which sets it as well.  The band
scan the window starts is not held to the same rule: ``band_scan``
tunes and switches the AGC through the facade, but reads the SDR's
sample rate and tuning, watches its blocks and holds the audio output
through the controller's ``sdr_receiver`` and ``audio_output``
directly.

What does not concern the receiver does not change it - which tab is
in front, and the Recordings tab's filter, Reload, Open folder,
Re-decode and Save CSV among them: those change what the window shows,
read the disk (asking the facade only for the station names), open the
file manager, decode a recorded file in a process of its own, or write
the file the user named.

The window holds none of the running receiver's state, so a refresh that
arrives while the user is mid-gesture cannot fight them for a widget —
except for the two controls that would, which say so where they are
handled.  What it does keep is its own: the requests it has made - where
it asked the tuner to go, the recordings it asked to start, the writes
it is waiting to hear about.  Of the receiver's state it keeps one thing,
and only once the device has gone: whether a recording was running at
that moment, asked once before the release starts, because the
receiver's own answer changes while the file is still being written.

The spectrum and waterfall are in ``band_view``; the blend bar is here,
because it reads off the same snapshot as the rest of the Signal group.
The DSP settings and the recordings are tabs of their own, in ``dsp_tab``
and ``recordings_tab``: this module builds them, and tells the recordings
tab when it is first chosen.
"""

from __future__ import annotations

import threading

import logging
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from fm_radio.band_scan import (
    BandScan, CONFIRMED, LIKELY_SKIRT, where_this_is,
)
from fm_radio.constants import RECORDINGS_DIR
from fm_radio.exceptions import SDRDeviceError
from fm_radio.gui.band_view import BandView
from fm_radio.gui.dsp_tab import DspTab
from fm_radio.gui.recordings_tab import RecordingsTab
from fm_radio.multipath import NOISE_AM_DEPTH
from fm_radio.stations import WillNotEdit
from fm_radio.device_worker import TUNE
from fm_radio.telemetry import SILENCE_DBFS, StatusSnapshot

logger = logging.getLogger('fm_receiver.gui')

#: How often the window reads the published state.  Matches the rate the
#: processing thread publishes at; asking faster only re-reads the same
#: snapshot.
REFRESH_INTERVAL_MS = 50

#: Tuning step for the arrow buttons, in Hz.  Japanese FM allocations sit on
#: a 0.1 MHz grid, which is also why the readout carries one decimal.
TUNING_STEP_HZ = 100e3

#: Range the level meters cover.  Below this is silence as far as a meter is
#: concerned, and the bars sit at the bottom.
METER_FLOOR_DBFS = -60.0

#: How long a failure stays in the status bar.  The health line is rewritten
#: every refresh, so without this a message about something the user just
#: tried would be gone in 50 ms - faster than they can read it.
NOTICE_SECONDS = 5.0

#: What the window shows for a reading there is not.  Every
#: diagnostic here can come back unmeasurable - a pilot SNR before
#: the first pilot, an AM depth of a block that was not finite - and
#: they all say it the same way, through _reading.  One place,
#: because the mistake to avoid is a display that prints 0.0 for
#: "nothing to measure": on these two readings zero is the best
#: possible signal, which is the opposite of what happened.
NOTHING_MEASURED = "--"

#: Where the AM depth bar is full.  Noise measures about this - the
#: envelope of complex Gaussian noise has a spread over its mean of
#: 0.52 - so a bar that fills at the noise figure reads as "how far
#: along the way to no signal at all", and a well-received station
#: sits near the bottom of it at 0.06.
AM_DEPTH_FULL_SCALE = NOISE_AM_DEPTH

#: Everything the blend line can say: the three words, and every
#: figure between them - 1.00 is never printed, because a full bar
#: says STEREO.  The widest of them is measured and kept, so that
#: the bar beside it does not change width as the receiver settles.
#: All hundred of the figures, rather than one as a stand-in: the
#: digits are only the same width in a font that says they are, and
#: "0.88" is wider than "0.00" in one that does not.
_BLEND_WORDS = ("--", "MONO", "STEREO") + tuple(
    "%.2f" % (hundredths / 100.0) for hundredths in range(100))

#: Room for the longest thing the scan list says, so that a sweep
#: that finds a lot does not push the tuner's own controls about.
_FOUND_WIDTH = 230

#: Gain slider resolution: the widget is integral, the tuner is in dB.
_GAIN_SCALE = 10.0
_GAIN_MAX_DB = 49.6


def _room_for_the_widest(label: QLabel) -> int:
    """How wide *label* has to be to hold any of _BLEND_WORDS.

    Measured rather than guessed.  A minimum width only holds until
    the text is wider than it, so a number picked by eye is not a
    fixed width at all: the column grows for the longest word and
    the bar beside it shrinks, which is the flicker this is here to
    prevent.  Asking the label itself covers the font it is really
    going to use and whatever margins it has.
    """
    was = label.text()
    try:
        widest = 0
        for word in _BLEND_WORDS:
            label.setText(word)
            widest = max(widest, label.sizeHint().width())
        return widest
    finally:
        label.setText(was)


def _blend_percent(blend: float, stereo: bool) -> int:
    """Map the stereo blend onto a meter's 0-100.

    Mono is empty whatever the blend says.  The blend factor is only
    meaningful while stereo is being attempted: it starts at 1.0 and
    the mono path never moves it, so a receiver asked for mono would
    otherwise show a full bar.
    """
    if not stereo:
        return 0
    return int(round(min(100.0, max(0.0, float(blend) * 100.0))))


def _what_was_found(signal) -> str:
    """One line for a thing the scan found.

    The frequency, how loud, and what the scan made of it.  A pilot
    is the only evidence a sweep has that a peak is a broadcast, and
    it is not proof that the peak is a station of its own - a strong
    station's skirt carries its pilot - so the word is the scan's
    own reading rather than a verdict.
    """
    said = {CONFIRMED: "stereo", LIKELY_SKIRT: "spill?"}.get(
        signal.sort, "no pilot")
    return "%.1f MHz  %+.0f dB  %s" % (
        signal.freq_mhz, signal.power_dbfs, said)


def _reading(value: "float | None", pattern: str) -> str:
    """A number for the window, or the mark for one there is not."""
    return NOTHING_MEASURED if value is None else pattern % value


def _am_depth_percent(depth: "float | None") -> int:
    """Map the AM depth onto a meter's 0-100.

    Full at the noise figure rather than at 1.0: everything worth
    telling apart happens between a clean station and an empty
    channel, and scaling to a number nothing reaches would put all
    of it in the bottom tenth of the bar.
    """
    if depth is None:
        return 0
    return int(round(min(100.0, max(
        0.0, float(depth) / AM_DEPTH_FULL_SCALE * 100.0))))


def _level_percent(dbfs: float) -> int:
    """Map a level in dBFS onto a meter's 0-100."""
    if dbfs <= METER_FLOOR_DBFS:
        return 0
    return int(round(min(100.0, (dbfs - METER_FLOOR_DBFS)
                         / -METER_FLOOR_DBFS * 100.0)))


#: What a notice is about.  A failure is news the user has to read and
#: gets its few seconds; progress is a line that will be repeated on the
#: next refresh if it is still true, so anything may take its place.
_FAILURE = "failure"
_PROGRESS = "progress"
#: A band scan saying where it has got to.  Its own sort because
#: _show_the_device_worker clears _PROGRESS whenever the window has
#: no tune of its own outstanding - which is every refresh during a
#: scan, since the scan's tunes are not the window's.  The line
#: lasted under 50 ms.
_SCANNING = "scanning"


def _still_going(request):
    """The request if it has not finished, else None.

    A finished one has nothing left to say here: what became of it is
    the recorder's state, or a notice from _show_the_device_worker.
    """
    return request if request is not None and not request.finished else None


class Sweep(QObject):
    """Runs a band scan off the GUI thread and reports back on it.

    A sweep is five seconds of retuning, so it cannot happen on the
    thread that draws: the window would be a white rectangle for the
    whole of it, which is the fault #47 was about.  It happens on a
    thread of its own and says what it is doing through signals,
    which Qt delivers on the GUI thread.

    It owns the tuner while it runs.  Anything else that tunes makes
    the sweep fail - it notices now, and says so rather than
    reporting a station at the wrong frequency - so the window puts
    its own tuning controls out of reach until it is finished.
    """

    #: A line about what the sweep is doing now.
    progress = Signal(str)
    #: The sweep has ended: (signals found, why it stopped or "").
    finished = Signal(object, str)

    def __init__(self, parent: QObject, controller) -> None:
        super().__init__(parent)
        self._scan = BandScan(controller, on_progress=self.progress.emit)
        self._thread = threading.Thread(
            target=self._run, name="BandScan", daemon=True)

    def go(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """Ask it to stop at the next hop."""
        self._scan.cancel()

    @property
    def cancelled(self) -> bool:
        """True if it was asked to stop, however it then ended.

        A cancelled sweep still hands back what it found before it
        was stopped, so what it found is not the question: everything
        that cancels one - the user pressing Stop, the device going,
        the window closing - is a reason not to put a question in
        front of them about it.
        """
        return self._scan.cancelled

    def wait(self) -> None:
        """Block until the sweep has finished, one way or the other."""
        if self._thread.ident is not None:
            self._thread.join()

    def _run(self) -> None:
        """The sweeping thread: sweep, and say how it went either way."""
        try:
            found = self._scan.run()
        except Exception as e:
            logger.error("The band scan stopped: %s", e, exc_info=True)
            self.finished.emit([], str(e))
        else:
            self.finished.emit(found, "")


class ReceiverWindow(QMainWindow):
    """Status and control for a running receiver."""

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("SDR FM Receiver")
        # (message, expiry, sort); see NOTICE_SECONDS and _set_notice.
        self._notice: tuple[str, float, str] | None = None
        #: The band scan now running, or None.
        self._sweep: "Sweep | None" = None
        #: True between asking a sweep to stop and its saying it has.
        self._stopping: bool = False

        # Everything there has ever been is the first tab; the
        # second is where the DSP settings are going.  The window
        # updates every control every refresh whether or not its tab
        # is the one showing - a control that stopped being updated
        # while it was out of sight would be wrong the moment it
        # came back, and Qt keeps hidden widgets alive and willing.
        # The third is the exception: it is a list of files, and a
        # refresh twenty times a second is no reason to read a
        # directory.  It reads when it is first chosen and when asked.
        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._build_radio_tab(), "Radio")
        self._tabs.addTab(self._build_dsp_tab(), "DSP")
        self._tabs.addTab(self._build_recordings_tab(), "Recordings")
        self._tabs.currentChanged.connect(self._tab_shown)
        self.setCentralWidget(self._tabs)

        self.setStatusBar(QStatusBar(self))
        self._health = QLabel("waiting for the first block")
        # A driver message can run to a couple of hundred characters, and a
        # status label is allowed to ask the window to be that wide.  It is
        # not: the window is the size the controls need, and a line that
        # does not fit is elided rather than allowed to push the edge out.
        self._health.setSizePolicy(QSizePolicy.Policy.Ignored,
                                   QSizePolicy.Policy.Preferred)
        self.statusBar().addWidget(self._health, 1)

        # Before the first refresh, which may already find that the device
        # has gone - it stops the timer, and cannot stop one that does not
        # exist yet.  Starting it here is safe: a Qt timer only fires once
        # there is an event loop to fire it in, and there is not one until
        # this window has been built and shown.
        # Started once when the device goes; see _free_the_device.
        self._releasing: threading.Thread | None = None
        # Where the tuner is going, which is not where it is: a write
        # takes 60 ms and a step asked for during one has to start from
        # the frequency the last step asked for, not from the one the
        # receiver is still on.  None when nothing is on its way.
        self._tuning_to: float | None = None
        # A recording that has been asked for and has not started yet.
        # The receiver decides when, because the tuner may be in front
        # of it, and until then the button has to look pressed or it
        # reads as a button that did nothing.
        self._starting_audio = None
        self._starting_iq = None
        # The writes this window has asked for and not yet reported on.
        # Watching these rather than the worker's last finished request:
        # that one can be somebody else's, and a gain landing between two
        # refreshes would otherwise hide a tune that is still going.
        self._asked_for: list = []
        # Whether a recording was running when the device went.  Asked
        # once, before the release starts, because the release answers it
        # differently long before it is finished; see
        # _show_the_recording_ending.
        self._was_recording: bool | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self._load_presets()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_radio_tab(self) -> QWidget:
        """The tuner, the band, and what to do about what is there.

        The same controls in the same order as before there were
        tabs; only their parent has changed.
        """
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_tuner())
        # Directly under the tuner: it is a picture of where the tuner
        # is, and the controls that follow are about what to do there.
        self._band = BandView(page)
        # With the stretch, so a taller window is a taller picture
        # rather than a taller gap under the controls.
        layout.addWidget(self._band, 1)
        layout.addWidget(self._build_signal())
        layout.addWidget(self._build_gain())
        layout.addWidget(self._build_recording())
        return page

    def _build_dsp_tab(self) -> QWidget:
        """The nine settings the demodulator will take while it runs.

        Its own module: this window is long enough, and the tab
        talks to the same facade through its own controller
        reference.  See fm_radio.gui.dsp_tab.
        """
        self._dsp = DspTab(self.controller, self)
        return self._dsp

    def _build_recordings_tab(self) -> QWidget:
        """The recordings on disk, as their sidecars describe them.

        The directory is the one the CLI records into.  See
        fm_radio.gui.recordings_tab.
        """
        self._recordings = RecordingsTab(self.controller, RECORDINGS_DIR,
                                         self)
        return self._recordings

    def _tab_shown(self, index: int) -> None:
        """Read the recordings the first time anyone looks at them."""
        if self._tabs.widget(index) is self._recordings:
            self._recordings.first_look()

    def _build_tuner(self) -> QGroupBox:
        box = QGroupBox("Tuner", self)
        row = QHBoxLayout(box)

        self._down = QPushButton("◀", box)
        self._down.setToolTip("Down 0.1 MHz")
        self._down.clicked.connect(lambda: self._step(-TUNING_STEP_HZ))
        row.addWidget(self._down)

        self._frequency = QLabel("--.- MHz", box)
        font = self._frequency.font()
        font.setPointSize(font.pointSize() + 10)
        self._frequency.setFont(font)
        self._frequency.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frequency.setMinimumWidth(160)
        row.addWidget(self._frequency)

        self._up = QPushButton("▶", box)
        self._up.setToolTip("Up 0.1 MHz")
        self._up.clicked.connect(lambda: self._step(TUNING_STEP_HZ))
        row.addWidget(self._up)

        self._station = QLabel("", box)
        row.addWidget(self._station, 1)

        self._presets = QComboBox(box)
        self._presets.setToolTip("Preset stations")
        self._presets.activated.connect(self._preset_chosen)
        row.addWidget(self._presets)

        # What is really on the band, as against what the catalogue
        # says should be.  Empty until a sweep has run, and kept
        # afterwards: the point of scanning is to pick from it.
        self._found = QComboBox(box)
        self._found.setToolTip("What the last scan found")
        self._found.setMinimumWidth(_FOUND_WIDTH)
        self._found.activated.connect(self._found_chosen)
        self._show_what_was_found([])
        row.addWidget(self._found)

        self._scan_button = QPushButton("Scan", box)
        self._scan_button.setToolTip(
            "Sweep 76-95 MHz for whatever is transmitting. "
            "Takes a few seconds, and the radio goes with it. "
            "A recording that is running will be ended: the first "
            "hop is a tune, and a tune closes the file.")
        self._scan_button.clicked.connect(self._scan_or_stop)
        row.addWidget(self._scan_button)
        return box

    def _build_signal(self) -> QGroupBox:
        box = QGroupBox("Signal", self)
        grid = QGridLayout(box)

        # How much stereo there is, rather than whether there is any:
        # the blend moves continuously with the pilot, and a receiver
        # that is halfway is the interesting case - a word for it can
        # only say STEREO or MONO, both of which would be wrong.
        self._blend = QProgressBar(box)
        self._blend.setRange(0, 100)
        self._blend.setTextVisible(False)
        self._blend.setToolTip(
            "How much of the stereo image is being let through: "
            "empty is mono, full is the whole of it")
        self._mode = QLabel("--", box)
        self._mode.setMinimumWidth(_room_for_the_widest(self._mode))
        grid.addWidget(QLabel("Blend", box), 0, 0)
        grid.addWidget(self._blend, 0, 1)
        grid.addWidget(self._mode, 0, 2)

        self._pilot = QLabel(NOTHING_MEASURED, box)
        grid.addWidget(QLabel("Pilot SNR", box), 1, 0)
        grid.addWidget(self._pilot, 1, 1)

        # How far the channel is from the one thing FM promises.  A
        # bar that fills as things get worse: empty is a carrier
        # holding its amplitude, full is an envelope moving as much
        # as noise does.  It does not say what moved it - an empty
        # channel looks like the worst multipath - so it is here
        # beside the pilot SNR, which says whether there is a
        # station there to be spoiled.
        self._am_depth = QProgressBar(box)
        self._am_depth.setRange(0, 100)
        self._am_depth.setTextVisible(False)
        self._am_depth.setToolTip(
            "How much the envelope moves, against how much it moves on "
            "noise. FM holds its amplitude, so empty is a clean signal. "
            "Read with the pilot SNR: this cannot tell multipath from "
            "an empty channel.")
        self._am_depth_value = QLabel(NOTHING_MEASURED, box)
        grid.addWidget(QLabel("AM depth", box), 2, 0)
        grid.addWidget(self._am_depth, 2, 1)
        grid.addWidget(self._am_depth_value, 2, 2)

        self._left = QProgressBar(box)
        self._right = QProgressBar(box)
        for meter in (self._left, self._right):
            meter.setRange(0, 100)
            meter.setTextVisible(False)
        self._left_db = QLabel("--", box)
        self._right_db = QLabel("--", box)
        grid.addWidget(QLabel("L", box), 3, 0)
        grid.addWidget(self._left, 3, 1)
        grid.addWidget(self._left_db, 3, 2)
        grid.addWidget(QLabel("R", box), 4, 0)
        grid.addWidget(self._right, 4, 1)
        grid.addWidget(self._right_db, 4, 2)
        grid.setColumnStretch(1, 1)
        return box

    def _build_gain(self) -> QGroupBox:
        box = QGroupBox("Gain", self)
        row = QHBoxLayout(box)

        self._auto_gain = QCheckBox("Auto", box)
        self._auto_gain.toggled.connect(self._auto_gain_toggled)
        row.addWidget(self._auto_gain)

        self._gain_slider = QSlider(Qt.Orientation.Horizontal, box)
        self._gain_slider.setRange(0, int(_GAIN_MAX_DB * _GAIN_SCALE))
        # valueChanged catches the wheel, the arrow keys and a click on the
        # groove; sliderReleased catches the end of a drag, which
        # valueChanged deliberately ignores so that one gesture does not
        # send a gain change per pixel.
        self._gain_slider.valueChanged.connect(self._gain_moved)
        self._gain_slider.sliderReleased.connect(self._gain_chosen)
        row.addWidget(self._gain_slider, 1)

        self._gain_value = QLabel("-- dB", box)
        self._gain_value.setMinimumWidth(70)
        row.addWidget(self._gain_value)

        self._iq_peak = QLabel("peak --", box)
        row.addWidget(self._iq_peak)
        return box

    def _build_recording(self) -> QGroupBox:
        box = QGroupBox("Recording", self)
        row = QHBoxLayout(box)

        self._record_audio = QPushButton("Record audio", box)
        self._record_audio.setCheckable(True)
        self._record_audio.clicked.connect(self._audio_recording_toggled)
        row.addWidget(self._record_audio)

        self._record_iq = QPushButton("Record IQ", box)
        self._record_iq.setCheckable(True)
        self._record_iq.clicked.connect(self._iq_recording_toggled)
        row.addWidget(self._record_iq)

        self._recording_status = QLabel("", box)
        row.addWidget(self._recording_status, 1)
        return box

    def _load_presets(self) -> None:
        """Fill the preset list from the catalogue's favourites."""
        self._presets.clear()
        self._presets.addItem("Presets", None)
        for name, freq_hz in self.controller.get_stations_list():
            self._presets.addItem(f"{freq_hz / 1e6:.1f}  {name}", freq_hz)

    # ------------------------------------------------------------------
    # Controls — every one of these goes through the facade
    # ------------------------------------------------------------------

    def _tune(self, freq_hz: float) -> None:
        """Ask for a new frequency and carry on drawing.

        The write is 60 ms of USB on a device that is answering, so this
        does not wait for it: what went wrong, if anything did, arrives
        on a later refresh, through the request this keeps hold of.
        Until then the reading on screen is the station the receiver is
        still on, which is the truth.
        """
        self._tuning_to = freq_hz
        self._watch(self.controller.tune(freq_hz))
        self.refresh()

    def _watch(self, request):
        """Keep a request until there is something to say about it."""
        if request is not None:
            self._asked_for.append(request)
        return request

    def _step(self, delta_hz: float) -> None:
        """Move by one step from wherever the tuner is heading.

        Two clicks in the time one write takes are two steps, not one.
        The receiver still reports the frequency it is on until the first
        write lands, so stepping from that would ask for the same place
        twice and end up half as far as the user asked to go.
        """
        from_hz = (self._tuning_to if self._tuning_to is not None
                   else self.controller.get_frequency())
        self._tune(from_hz + delta_hz)

    def _preset_chosen(self, index: int) -> None:
        freq_hz = self._presets.itemData(index)
        if freq_hz is not None:
            self._tune(float(freq_hz))
        self._presets.setCurrentIndex(0)

    def _found_chosen(self, index: int) -> None:
        freq_hz = self._found.itemData(index)
        if freq_hz is not None:
            self._tune(float(freq_hz))

    def _scan_or_stop(self) -> None:
        """The one button: start a sweep, or stop the one running."""
        if self._sweep is not None:
            self._sweep.cancel()
            self._stopping = True
            self._say_what_can_be_used()
            self._set_notice("stopping the scan...", _SCANNING)
            return
        self._sweep = Sweep(self, self.controller)
        self._sweep.progress.connect(self._scanning)
        self._sweep.finished.connect(self._scan_ended)
        self._say_what_can_be_used()
        self._sweep.go()

    def _scanning(self, line: str) -> None:
        """On the GUI thread: where the sweep has got to."""
        self._set_notice(line, _SCANNING)

    def _scan_ended(self, found, why: str) -> None:
        """On the GUI thread: the sweep is over, for whatever reason."""
        sweep, self._sweep = self._sweep, None
        self._stopping = False
        # Whether it ran to the end of the band, asked before the
        # sweep is handed to Qt to delete.  A sweep this window no
        # longer holds is one stop_any_sweep took, which is the
        # window closing.
        the_whole_band = sweep is not None and not sweep.cancelled
        if sweep is not None:
            # Its parent is the window, so without this every sweep
            # ever run stays a child of it - and each one holds a
            # demodulator and a spectrum maker.
            sweep.deleteLater()
        self._say_what_can_be_used()
        if why:
            self._set_notice("the scan stopped: %s" % why)
            return
        self._show_what_was_found(found)
        self._set_notice(
            "scan found %d" % len(found) if found
            else "scan found nothing", _SCANNING)
        if the_whole_band:
            self._offer_to_keep_the_area(found)

    def _offer_to_keep_the_area(self, found) -> None:
        """Ask whether to keep the area the sweep points at.

        A frequency is not unique in Japan, so with no area set the
        catalogue will name almost anything the tuner sits on, as
        readily after a transmitter a thousand kilometres away.  The
        end of a sweep is the one moment the radio has the evidence
        to work out which of them it is hearing, and the file it
        goes in is the user's: they are asked, and they are told
        what it will do.

        Nothing is said when there is nothing to say - the evidence
        points nowhere clearly enough, or it points where the
        receiver already is.  A dialog that only ever says "no
        change" is one the user learns to dismiss unread.
        """
        area = where_this_is(found, self.controller.get_catalogue())
        if area is None or area == self.controller.area:
            return
        if self._ask(
                "Where is this radio?",
                "The scan matches transmitters in %s." % area,
                "Save it in your station file?  Stations will be named "
                "from %s from now on, and at every start - which is what "
                "stops a frequency being named after a transmitter on "
                "the other side of the country." % area) != QMessageBox.Yes:
            return
        try:
            path = self.controller.remember_where_this_is(area)
        except (WillNotEdit, OSError) as trouble:
            # Said in full and in a box, not summarised onto the
            # status line: it names the file and the line to type,
            # which is all the user has left to go on, and it is the
            # answer to a question they were just asked.
            self._tell_them(
                "The area was not saved", str(trouble),
                QMessageBox.Warning)
            self._set_notice("the area was not saved")
            return
        self._set_notice("stations are named from %s now (saved in %s)"
                         % (area, path))

    def _ask(self, title: str, text: str, detail: str) -> int:
        """Put a yes-or-no question up and return the button pressed."""
        return self._a_box(title, text, detail, QMessageBox.Question,
                           QMessageBox.Yes | QMessageBox.No,
                           QMessageBox.Yes).exec()

    def _tell_them(self, title: str, text: str, icon) -> None:
        """Put something the user has to read up, and wait for them."""
        self._a_box(title, text, "", icon, QMessageBox.Ok,
                    QMessageBox.Ok).exec()

    def _a_box(self, title: str, text: str, detail: str, icon,
               buttons, default) -> QMessageBox:
        """One place that builds the modal boxes, so they match.

        Parented on the window, so a box outlives neither it nor the
        receiver behind it, and so the user cannot start a second
        sweep behind one.
        """
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(text)
        if detail:
            box.setInformativeText(detail)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default)
        return box

    def _say_what_can_be_used(self) -> None:
        """Work out what is live, from everything that decides it.

        Three things do: whether the device is still there, whether
        a sweep owns the tuner, and whether the gain is on auto.
        They were decided in three places and undid each other - a
        sweep turns the AGC off, the next refresh saw manual gain
        and handed the slider back mid-sweep, and a gain moved then
        puts the sweep's hops in different units.  A sweep that has
        ended cannot hand the controls back to a receiver that has
        gone, either.
        """
        alive = getattr(self.controller, "device_failure", None) is None
        sweeping = self._sweep is not None
        usable = alive and not sweeping
        for widget in (self._down, self._up, self._presets, self._found,
                       self._auto_gain):
            widget.setEnabled(usable)
        # The slider is the one control with a third say in it.
        self._gain_slider.setEnabled(usable
                                     and not self._auto_gain.isChecked())
        self._scan_button.setText("Stop" if sweeping else "Scan")
        # Stopping is the one thing still worth offering mid-sweep,
        # and nothing is once the device has gone.
        self._scan_button.setEnabled(alive and not self._stopping)
        # Recording goes with the rest.  A tune shuts any recording
        # that is running - that is the receiver's rule, and a sweep
        # is two dozen tunes - so a button pressed mid-sweep makes a
        # file less than one hop long and then stops.  It offers
        # something it cannot do.
        for widget in (self._record_audio, self._record_iq):
            widget.setEnabled(usable)
        # The DSP settings go with them: a sweep is using the
        # demodulator for its own purposes, and a receiver that has
        # gone cannot be asked for anything.
        self._dsp.set_usable(usable)

    def _show_what_was_found(self, found) -> None:
        """Fill the list of what is on the band, newest sweep only."""
        self._found.clear()
        self._found.addItem(
            "%d found" % len(found) if found else "nothing found yet", None)
        for signal in found:
            self._found.addItem(_what_was_found(signal), signal.freq_hz)
        self._found.setCurrentIndex(0)

    def _auto_gain_toggled(self, checked: bool) -> None:
        self._watch(self.controller.set_agc_mode(checked))
        self._say_what_can_be_used()

    def _gain_moved(self, value: int) -> None:
        """Apply a move that is not part of a drag.

        The refresh blocks this signal while it follows the receiver, so
        anything arriving here came from the user.
        """
        if self._gain_slider.isSliderDown():
            return                      # _gain_chosen applies it on release
        self._apply_gain(value)

    def _gain_chosen(self) -> None:
        """Apply the value a drag finished on."""
        self._apply_gain(self._gain_slider.value())

    def _apply_gain(self, value: int) -> None:
        if self._auto_gain.isChecked():
            return
        self._watch(self.controller.set_gain(value / _GAIN_SCALE))

    def _audio_recording_toggled(self, checked: bool) -> None:
        """Ask for a recording, or end one.  Neither waits here.

        The receiver names the file: the name says which station this
        is, and which station that will be is only settled once the
        tuner has finished whatever it is doing.  What comes back is a
        request, watched like any other.
        """
        if not checked:
            # The receiver takes back a start it has not carried out,
            # so a recording asked for a moment ago does not begin
            # after the button has been let go.
            self.controller.stop_recording()
            self._starting_audio = None
        else:
            self._starting_audio = self._watch(
                self.controller.start_recording())
        # Show the new state now rather than at the next refresh: a control
        # that answers 50 ms late reads as a control that did not work.
        self._show_recording()

    def _iq_recording_toggled(self, checked: bool) -> None:
        if not checked:
            self.controller.stop_iq_recording()
            self._starting_iq = None
        else:
            self._starting_iq = self._watch(
                self.controller.start_iq_recording())
        self._show_recording()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _set_notice(self, message: str, sort: str = _FAILURE) -> None:
        """Put *message* in the status bar and keep it there to be read.

        Nothing goes over the line that says the receiver has
        stopped.  That line is the last thing the window will say -
        the refresh stops with it - so anything written after it
        stays written: a sweep that ends after the cable comes out
        would leave the window saying "scan found 3" about a radio
        that is not there, and a question answered after it the
        same.  _show_the_device_has_gone writes that line itself and
        does not come through here.
        """
        if getattr(self.controller, "device_failure", None) is not None:
            return
        self._notice = (message, time.monotonic() + NOTICE_SECONDS, sort)
        self._health.setText(message)

    def _current_notice(self) -> str | None:
        """The notice still worth showing, or None once it has had its time."""
        if self._notice is None:
            return None
        message, expires, _sort = self._notice
        if time.monotonic() >= expires:
            self._notice = None
            return None
        return message

    def _showing(self, sort: str) -> bool:
        """True when the notice still worth showing is of this sort."""
        if self._current_notice() is None:
            return False
        return self._notice is not None and self._notice[2] == sort

    def refresh(self) -> None:
        """Read the published state and show it.

        Called on a timer and after anything that changes the receiver.  A
        snapshot is a plain value, so this neither blocks the processing
        thread nor sees a half-updated one; None means there is no current
        snapshot, which happens at startup, just after tuning, and while
        snapshots cannot be built.
        """
        failure = getattr(self.controller, "device_failure", None)
        if failure is not None:
            self._show_the_device_has_gone(failure)
            return
        status = self.controller.get_status()
        if status is None:
            self._show_without_status()
        else:
            self._show_status(status)
        self._show_recording()
        self._band.show_the_frame(self.controller.get_spectrum())
        # A failure the user just caused outranks the health line until it
        # has been up long enough to read.
        self._show_the_device_worker()
        notice = self._current_notice()
        if notice is not None:
            self._health.setText(notice)

    def _show_the_device_has_gone(self, why: str) -> None:
        """Say why the receiver stopped, and stop pretending otherwise.

        The window stays: closing it here would take the explanation with
        it, and the person who just pulled a cable is the one who needs to
        read it.  Nothing in it will reach the device again - the receiver
        is already shutting down - so the controls go quiet and the timer
        stops rather than redrawing the same dead reading fifty times a
        second.  Closing the window runs the usual cleanup.

        The readings go with the controls.  Leaving the last ones up -
        STEREO, a pilot SNR, two meters near the top of their range - is
        the window saying the radio is playing, about a radio that is not
        there.  The recording line is the same, and is the last thing
        still moving: see :meth:`_show_the_recording_ending`.
        """
        self._show_without_status()
        self._station.setText("no device")
        self._health.setText("SDR disconnected - the receiver has stopped")
        # The driver's own words, for whoever wants them.  Not on the
        # status line, where they would be most of a paragraph.
        self._health.setToolTip(why)
        # A sweep of a band the receiver can no longer hear is over,
        # whatever it thinks; and it must not hand the controls back
        # to a device that has gone when it notices.
        if self._sweep is not None:
            self._sweep.cancel()
        self._say_what_can_be_used()
        # Asked before the release starts, not after: stop_recording
        # clears the flag and then flushes the queue, closes the wave file
        # and writes the sidecar, so a release already under way would
        # answer "nothing is recording" about a file still being written.
        if self._was_recording is None:
            self._was_recording = (self.controller.is_recording()
                                   or self.controller.is_iq_recording())
        self._free_the_device()
        self._show_the_recording_ending()

    def _show_the_recording_ending(self) -> None:
        """Follow the recording out, and stop refreshing once it has gone.

        Freeing the device takes a moment and happens on another thread,
        so for that moment there really is still a recording open.  Saying
        "recording audio" after it has been closed would be the same lie
        the meters were telling, and saying nothing while it is still
        being written would be another - so the window keeps refreshing
        until the file is closed, and then goes quiet for good.

        "Until the file is closed" means until the release thread is
        finished, not until the receiver says it is no longer recording.
        Those are a long way apart: stop_recording clears its flag first
        and then flushes the queue, waits for the worker, closes the wave
        file and writes the sidecar.  Asking the receiver would clear this
        line while the file was still being written.

        The record buttons are left unchecked as well as disabled: a
        disabled button still shows that it is pressed in, which reads as
        a recording that is running.
        """
        if (self._was_recording and self._releasing is not None
                and self._releasing.is_alive()):
            self._recording_status.setText("closing the recording")
            return                      # the timer brings us back
        self._timer.stop()
        self._recording_status.setText("")
        for button in (self._record_audio, self._record_iq):
            button.setChecked(False)

    def _free_the_device(self) -> None:
        """Close the recording and the audio stream, once, off this thread.

        The window stays up so the reason can be read, but a recording the
        user had running should not sit half-written until they get round
        to closing it, and the audio stream has nothing left to play.  The
        record buttons are disabled by now, so this is the only thing that
        will end it.

        On a thread of its own because cleanup() has several bounded waits
        in it, and a window frozen for a few seconds is a poor way to
        explain what happened.  cleanup() is idempotent and serialised, so
        the one that runs when the window closes is the same call arriving
        second.
        """
        if self._releasing is not None:
            return
        self._releasing = threading.Thread(
            target=self.controller.cleanup,
            name="DeviceLossCleanup", daemon=True)
        self._releasing.start()

    def _show_the_device_worker(self) -> None:
        """Say what became of the writes this window asked the SDR for.

        Its own, not the worker's last finished request: that one may
        have been asked for by the AGC, and a gain landing between two
        refreshes would hide a tune that is still on its way - or worse,
        answer for one that has not been made yet.

        Nothing waits for a device write any more, so this is where the
        answer turns up: a failure becomes a notice, and a tune that has
        been asked for and not yet made says so rather than leaving the
        window looking as though the button did nothing.
        """
        still_going = []
        for request in self._asked_for:
            if not request.finished:
                still_going.append(request)
                continue
            if request.failed:
                logger.error("%s failed: %s", request.what, request.error)
                self._set_notice(f"{request.what} failed: {request.error}")
        self._asked_for = still_going

        tuning = [r for r in still_going if r.kind == TUNE]
        if tuning and self._tuning_to is not None:
            # Unless there is a failure up that the user has not had
            # time to read.  A gain that would not write is worth more
            # than the news that a tune is still going: the tune says so
            # again on the next refresh, and the failure will not.
            if not self._showing(_FAILURE):
                self._set_notice(
                    f"tuning to {self._tuning_to / 1e6:.1f} MHz...", _PROGRESS)
        else:
            self._tuning_to = None
            if self._showing(_PROGRESS):
                # "tuning to 80.1 MHz..." was true while it was; a notice
                # normally sits for a few seconds so it can be read, and
                # this one has nothing left to say the moment it lands.
                self._notice = None

    def _show_without_status(self) -> None:
        """Show what can be known without a snapshot: the tuner's own state."""
        freq_hz = self.controller.get_frequency()
        self._frequency.setText(f"{freq_hz / 1e6:.1f} MHz")
        station = self.controller.current_station()
        self._station.setText(station.name if station else "")
        self._blend.setValue(0)
        self._mode.setText(NOTHING_MEASURED)
        self._pilot.setText(NOTHING_MEASURED)
        self._am_depth.setValue(0)
        self._am_depth_value.setText(NOTHING_MEASURED)
        for meter, label in ((self._left, self._left_db),
                             (self._right, self._right_db)):
            meter.setValue(0)
            label.setText(NOTHING_MEASURED)
        self._show_gain(self.controller.get_gain(),
                        not self.controller.is_manual_gain())
        self._iq_peak.setText("peak --")
        self._health.setText("waiting for a block")

    def _show_status(self, status: StatusSnapshot) -> None:
        self._frequency.setText(f"{status.freq_hz / 1e6:.1f} MHz")
        self._station.setText(status.station)

        self._show_blend(status)

        self._pilot.setText(_reading(status.pilot_snr_db, "%.1f dB"))
        self._am_depth.setValue(_am_depth_percent(status.am_depth))
        self._am_depth_value.setText(_reading(status.am_depth, "%.3f"))

        for meter, label, level in (
                (self._left, self._left_db, status.level_left_dbfs),
                (self._right, self._right_db, status.level_right_dbfs)):
            meter.setValue(_level_percent(level))
            label.setText(NOTHING_MEASURED if level <= SILENCE_DBFS
                          else f"{level:.1f} dBFS")

        self._show_gain(status.gain_db, status.auto_gain)
        self._iq_peak.setText(f"peak {status.iq_peak:.2f}")
        self._health.setText(self._health_text(status))

    def _show_blend(self, status: StatusSnapshot) -> None:
        """The bar, and a word for what the bar amounts to.

        Both off the same number, so that they cannot disagree.  The
        word said STEREO from a blend of 0.5 up, which is what the
        receiver calls stereo, and beside a bar half way along it
        read as a contradiction; and it printed the raw blend while
        the bar clamped it, so a blend of -0.2 was an empty bar
        labelled -0.20.

        The word is still worth having.  MONO says something the bar
        cannot - that nobody asked for stereo, rather than that the
        pilot is too poor for it - and a number is worth more than a
        bar to read a figure off.
        """
        percent = _blend_percent(status.blend_factor, status.stereo)
        self._blend.setValue(percent)
        if not status.stereo:
            self._mode.setText("MONO")
        elif percent >= 100:
            self._mode.setText("STEREO")
        else:
            self._mode.setText(f"{percent / 100.0:.2f}")

    def _show_gain(self, gain_db: float, auto: bool) -> None:
        """Show the gain without fighting the user for the slider.

        Auto gain moves the value several times a second, so the slider has
        to follow it; while the user is dragging, it is theirs.
        """
        self._gain_value.setText(f"{gain_db:.1f} dB")
        if self._auto_gain.isChecked() != auto:
            self._auto_gain.blockSignals(True)
            self._auto_gain.setChecked(auto)
            self._auto_gain.blockSignals(False)
            # Not setEnabled here: a sweep turns the AGC off for its
            # own reasons, and this ran every refresh and handed the
            # slider back in the middle of one.
            self._say_what_can_be_used()
        if not self._gain_slider.isSliderDown():
            self._gain_slider.blockSignals(True)
            self._gain_slider.setValue(int(round(gain_db * _GAIN_SCALE)))
            self._gain_slider.blockSignals(False)

    @staticmethod
    def _health_text(status: StatusSnapshot) -> str:
        state = "healthy" if status.healthy else "loaded"
        return (f"{state}   block {status.block_ms:.1f}/"
                f"{status.block_budget_ms:.0f} ms   "
                f"sdr q {status.sdr_queue}/{status.sdr_queue_max}   "
                f"drops {status.iq_drops}/{status.audio_drops}   "
                f"underruns {status.audio_underruns}   "
                f"up {status.uptime_sec / 60:.0f} min")

    def _show_recording(self) -> None:
        """Follow the receiver's recording state, including rotation stops.

        The buttons are checkable, but the receiver is what decides whether
        it is recording — a recording can stop without the button being
        pressed.

        One that has been asked for and not started yet counts as
        pressed.  The receiver decides when it starts, because a tune
        may be in front of it on the worker, and a button that springs
        back up in the meantime reads as a button that did nothing.

        One that has been stopped and is still being written does not
        count as pressed - it is taking nothing in - but it is still a
        file being written, and saying nothing about it would be the
        same silence the window used to keep while a recording was
        being closed.
        """
        self._starting_audio = _still_going(self._starting_audio)
        self._starting_iq = _still_going(self._starting_iq)
        audio = self.controller.is_recording()
        iq = self.controller.is_iq_recording()
        for button, active, starting in (
                (self._record_audio, audio, self._starting_audio),
                (self._record_iq, iq, self._starting_iq)):
            # No blockSignals: these are wired to clicked, which setChecked
            # does not emit.  The Auto checkbox is wired to toggled, which it
            # does, and blocks them for that reason.
            button.setChecked(active or starting is not None)
        self._recording_status.setText(self._what_the_recorders_are_doing(
            audio, iq))

    def _what_the_recorders_are_doing(self, audio: bool, iq: bool) -> str:
        """The recording line: what is running, starting and finishing."""
        parts = [name for name, on in (("audio", audio), ("IQ", iq)) if on]
        starting = [name for name, asked in
                    (("audio", self._starting_audio),
                     ("IQ", self._starting_iq)) if asked is not None]
        finishing = [
            name for name, still in
            (("audio", self.controller.is_finishing_a_recording()),
             ("IQ", self.controller.is_finishing_an_iq_recording()))
            if still]
        said = ["recording " + " + ".join(parts)] if parts else []
        if starting:
            said.append("starting " + " + ".join(starting) + "...")
        if finishing:
            said.append("finishing " + " + ".join(finishing) + "...")
        return ", ".join(said)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the timer, stop any sweep or re-decode, and shut the
        receiver down.

        A sweep left running would go on retuning a receiver that is
        being taken apart, and would put it back afterwards to a
        frequency nobody is listening to.  A re-decode left running
        would go on using a core for a result nobody will see.
        """
        self._timer.stop()
        self.stop_any_sweep()
        self._recordings.shutdown()
        self.controller.quit_event.set()
        super().closeEvent(event)

    def stop_any_sweep(self) -> None:
        """Cancel a sweep and wait for it, if one is running.

        The wait is not bounded by a hop.  A sweep stops at the next
        one, but the hop it is in may be waiting on the device - a
        request has five seconds - and it puts the receiver back
        afterwards, which is two more writes.  On a device that has
        stopped answering, closing the window can take that long.
        Letting it go instead would leave something retuning a
        receiver being taken apart, which is worse.
        """
        sweep, self._sweep = self._sweep, None
        if sweep is not None:
            sweep.cancel()
            sweep.wait()


class ReceiverSwitch(QObject):
    """Switches the receiver on without holding up the window.

    Starting takes about a second and a quarter, nearly all of it the
    JIT pre-warm, and on the GUI thread that is a second and a quarter
    of a window that is up but has never painted: a white rectangle
    that looks like a program that has hung.  It happens off the GUI
    thread instead, so the window paints while the compiler works.

    A start that fails records why on the controller, the same as a
    device that goes mid-listen does, and asks for the window through
    a signal: Qt queues that onto the GUI thread rather than letting
    this one touch widgets.
    """

    #: Emitted from the starting thread when the receiver would not go.
    failed = Signal(str)

    def __init__(self, window: "ReceiverWindow", controller, start) -> None:
        super().__init__(window)
        self._window = window
        self._controller = controller
        self._start = start
        self.failed.connect(self._say_so)
        self._thread = threading.Thread(
            target=self._run, name="ReceiverStart", daemon=True)

    def go(self) -> None:
        """Begin starting the receiver."""
        self._thread.start()

    def wait(self) -> None:
        """Block until the start has finished, one way or the other.

        The window can be closed while the receiver is still coming up,
        and whoever runs cleanup() next would otherwise be tearing down
        a receiver that is still being built.
        """
        if self._thread.ident is not None:
            self._thread.join()

    def _run(self) -> None:
        """The starting thread: throw the switch, record a failure.

        The reason is written down here rather than in the slot below,
        because a window that is closing may never run the slot and
        the exit status is taken from the controller either way.  A
        receiver that cannot start is the same state as one whose
        device has gone, and it is recorded in the same place.
        """
        try:
            self._start()
        except Exception as e:
            logger.critical("The receiver would not start: %s", e,
                            exc_info=True)
            if getattr(self._controller, "device_failure", None) is None:
                self._controller.device_failure = str(e)
            self.failed.emit(str(e))

    def _say_so(self, why: str) -> None:
        """On the GUI thread: show it now, not at the next refresh."""
        self._window.refresh()


def run_window(controller, start=None) -> int:
    """Create the application, show the window, and run the event loop.

    Args:
        controller: The receiver the window is of.
        start: What switches the receiver on, called on a thread of its
            own once the window is up.  Building a window is expensive
            enough to be a gap in the audio - fonts, a graphics
            context, several hundred widgets' worth of layout - so it
            is done first, while there is no audio to interrupt.  None
            leaves the receiver alone, for a caller that has already
            started it.

    Returns:
        The exit code for the process.
    """
    app = QApplication.instance() or QApplication([])
    window = ReceiverWindow(controller)
    window.show()
    switch = None
    if start is not None:
        switch = ReceiverSwitch(window, controller, start)
        switch.go()
    try:
        return app.exec()
    finally:
        if switch is not None:
            switch.wait()
