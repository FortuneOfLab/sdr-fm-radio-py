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

Everything shown here comes from one call to ``controller.get_status()`` on a
timer; everything the user changes goes back through the controller's facade.
The window holds no receiver state of its own, so a refresh that arrives
while the user is mid-gesture cannot fight them for a widget — except for the
two controls that would, which say so where they are handled.

Spectrum, waterfall, the blend bar and the DSP settings are the next step and
are deliberately not here.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QStatusBar,
    QVBoxLayout, QWidget,
)

from fm_radio.cli import build_recording_path
from fm_radio.exceptions import RecordingError, SDRDeviceError
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

#: Gain slider resolution: the widget is integral, the tuner is in dB.
_GAIN_SCALE = 10.0
_GAIN_MAX_DB = 49.6


def _level_percent(dbfs: float) -> int:
    """Map a level in dBFS onto a meter's 0-100."""
    if dbfs <= METER_FLOOR_DBFS:
        return 0
    return int(round(min(100.0, (dbfs - METER_FLOOR_DBFS)
                         / -METER_FLOOR_DBFS * 100.0)))


class ReceiverWindow(QMainWindow):
    """Status and control for a running receiver."""

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("SDR FM Receiver")
        # (message, expiry); see NOTICE_SECONDS.
        self._notice: tuple[str, float] | None = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_tuner())
        layout.addWidget(self._build_signal())
        layout.addWidget(self._build_gain())
        layout.addWidget(self._build_recording())
        layout.addStretch(1)
        self.setCentralWidget(central)

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
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self._load_presets()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        return box

    def _build_signal(self) -> QGroupBox:
        box = QGroupBox("Signal", self)
        grid = QGridLayout(box)

        self._mode = QLabel("--", box)
        grid.addWidget(QLabel("Mode", box), 0, 0)
        grid.addWidget(self._mode, 0, 1)

        self._pilot = QLabel("--", box)
        grid.addWidget(QLabel("Pilot SNR", box), 1, 0)
        grid.addWidget(self._pilot, 1, 1)

        self._left = QProgressBar(box)
        self._right = QProgressBar(box)
        for meter in (self._left, self._right):
            meter.setRange(0, 100)
            meter.setTextVisible(False)
        self._left_db = QLabel("--", box)
        self._right_db = QLabel("--", box)
        grid.addWidget(QLabel("L", box), 2, 0)
        grid.addWidget(self._left, 2, 1)
        grid.addWidget(self._left_db, 2, 2)
        grid.addWidget(QLabel("R", box), 3, 0)
        grid.addWidget(self._right, 3, 1)
        grid.addWidget(self._right_db, 3, 2)
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
        try:
            self.controller.tune(freq_hz)
        except SDRDeviceError as exc:
            # A tuner that refused is worth saying out loud, but not worth
            # taking the window down over.
            logger.error("Could not tune to %.1f MHz: %s", freq_hz / 1e6, exc)
            self._set_notice(f"tuning failed: {exc}")
            return
        self.refresh()

    def _step(self, delta_hz: float) -> None:
        self._tune(self.controller.get_frequency() + delta_hz)

    def _preset_chosen(self, index: int) -> None:
        freq_hz = self._presets.itemData(index)
        if freq_hz is not None:
            self._tune(float(freq_hz))
        self._presets.setCurrentIndex(0)

    def _auto_gain_toggled(self, checked: bool) -> None:
        self.controller.set_agc_mode(checked)
        self._gain_slider.setEnabled(not checked)

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
        self.controller.set_gain(value / _GAIN_SCALE)

    def _audio_recording_toggled(self, checked: bool) -> None:
        if not checked:
            self.controller.stop_recording()
        else:
            try:
                # Inside the try: naming the file creates recordings/, which
                # fails with an OSError of its own before the receiver has
                # been asked for anything.
                path = build_recording_path(
                    self.controller.get_frequency() / 1e6)
                self.controller.start_recording(path)
            except (RecordingError, OSError) as exc:
                logger.error("Could not start recording: %s", exc)
                self._set_notice(f"recording failed: {exc}")
                self._record_audio.setChecked(False)
        # Show the new state now rather than at the next refresh: a control
        # that answers 50 ms late reads as a control that did not work.
        self._show_recording()

    def _iq_recording_toggled(self, checked: bool) -> None:
        if not checked:
            self.controller.stop_iq_recording()
        else:
            try:
                path = build_recording_path(
                    self.controller.get_frequency() / 1e6, iq=True)
                self.controller.start_iq_recording(path)
            except (RecordingError, OSError) as exc:
                logger.error("Could not start IQ recording: %s", exc)
                self._set_notice(f"IQ recording failed: {exc}")
                self._record_iq.setChecked(False)
        self._show_recording()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _set_notice(self, message: str) -> None:
        """Put *message* in the status bar and keep it there to be read."""
        self._notice = (message, time.monotonic() + NOTICE_SECONDS)
        self._health.setText(message)

    def _current_notice(self) -> str | None:
        """The notice still worth showing, or None once it has had its time."""
        if self._notice is None:
            return None
        message, expires = self._notice
        if time.monotonic() >= expires:
            self._notice = None
            return None
        return message

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
        # A failure the user just caused outranks the health line until it
        # has been up long enough to read.
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
        there.
        """
        self._timer.stop()
        self._show_without_status()
        self._station.setText("no device")
        self._health.setText("SDR disconnected - the receiver has stopped")
        # The driver's own words, for whoever wants them.  Not on the
        # status line, where they would be most of a paragraph.
        self._health.setToolTip(why)
        for widget in (self._down, self._up, self._presets,
                       self._auto_gain, self._gain_slider,
                       self._record_audio, self._record_iq):
            widget.setEnabled(False)

    def _show_without_status(self) -> None:
        """Show what can be known without a snapshot: the tuner's own state."""
        freq_hz = self.controller.get_frequency()
        self._frequency.setText(f"{freq_hz / 1e6:.1f} MHz")
        station = self.controller.current_station()
        self._station.setText(station.name if station else "")
        self._mode.setText("--")
        self._pilot.setText("--")
        for meter, label in ((self._left, self._left_db),
                             (self._right, self._right_db)):
            meter.setValue(0)
            label.setText("--")
        self._show_gain(self.controller.get_gain(),
                        not self.controller.is_manual_gain())
        self._iq_peak.setText("peak --")
        self._health.setText("waiting for a block")

    def _show_status(self, status: StatusSnapshot) -> None:
        self._frequency.setText(f"{status.freq_hz / 1e6:.1f} MHz")
        self._station.setText(status.station)

        if not status.stereo:
            self._mode.setText("MONO")
        elif status.stereo_locked:
            self._mode.setText("STEREO")
        else:
            self._mode.setText(f"BLENDING ({status.blend_factor:.2f})")

        self._pilot.setText(
            "--" if status.pilot_snr_db is None
            else f"{status.pilot_snr_db:.1f} dB")

        for meter, label, level in (
                (self._left, self._left_db, status.level_left_dbfs),
                (self._right, self._right_db, status.level_right_dbfs)):
            meter.setValue(_level_percent(level))
            label.setText("--" if level <= SILENCE_DBFS
                          else f"{level:.1f} dBFS")

        self._show_gain(status.gain_db, status.auto_gain)
        self._iq_peak.setText(f"peak {status.iq_peak:.2f}")
        self._health.setText(self._health_text(status))

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
            self._gain_slider.setEnabled(not auto)
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
        """
        audio = self.controller.is_recording()
        iq = self.controller.is_iq_recording()
        for button, active in ((self._record_audio, audio),
                               (self._record_iq, iq)):
            # No blockSignals: these are wired to clicked, which setChecked
            # does not emit.  The Auto checkbox is wired to toggled, which it
            # does, and blocks them for that reason.
            button.setChecked(active)
        parts = [name for name, on in (("audio", audio), ("IQ", iq)) if on]
        self._recording_status.setText(
            "recording " + " + ".join(parts) if parts else "")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the timer and ask the receiver to shut down."""
        self._timer.stop()
        self.controller.quit_event.set()
        super().closeEvent(event)


def run_window(controller) -> int:
    """Create the application, show the window, and run the event loop."""
    app = QApplication.instance() or QApplication([])
    window = ReceiverWindow(controller)
    window.show()
    return app.exec()
