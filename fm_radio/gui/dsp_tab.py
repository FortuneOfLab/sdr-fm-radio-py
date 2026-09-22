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
"""The nine DSP settings, on a tab, while the radio plays.

Every one of them is a parameter ``quality_selftest`` already
overrides to run an experiment offline; this is the same set with a
handle on it, so that something found by ear can be measured
afterwards with the same knobs.  See :mod:`fm_radio.dsp_settings`.

Three rules the rest of this follows.

**The tab owns what is on screen; the receiver owns what is running.**
The values are read from the facade once, when the tab is built, and
after that the tab writes and never reads: the window refreshes
twenty times a second, and a control that is re-read at that rate is
a control that fights the hand moving it.  Nothing else changes these
settings while the radio runs.

**One parameter at a time goes through update_dsp_settings**, which
does the read-modify-write under the receiver's lock.  Resetting the
lot is one set_dsp_settings, because that is the whole set at once
and last-writer-wins is what it is for.

**The defaults come from the receiver**, not from constants.py: the
standard and light chains do not share them (85.0 degrees of
subcarrier phase against 84.3), so "reset" means the running
variant's own default.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from fm_radio.dsp_settings import MAX_MONO_DELAY_SAMPLES

#: Slider positions for a fraction 0..1, so one step is 0.01.
_FRACTION_STEPS = 100
#: And for the over-subtraction factor 0..3, so one step is 0.1.
#: Positions, not steps per unit: ten of them would put the default
#: of 1.0 between two positions and show it as 0.9.
_BETA_MAX = 3.0
_BETA_STEPS = 30
#: What the phase spin box will go to.  Wider than anything useful -
#: the two variants sit at 85.0 and 84.3 degrees - because the
#: setting itself takes any angle and a box that would not let the
#: user type 0 to see what happens is a box in the way of the
#: experiment this tab exists for.
_PHASE_LIMIT_DEG = 180.0


class _Row:
    """One setting: the control that changes it and the button that
    puts it back.

    ``read`` turns what the widget says into what the setting wants,
    ``show`` does the other direction, and both are given by whoever
    builds the row - a slider counts in hundredths, the phase box in
    degrees, and the setting itself knows nothing about either.
    """

    def __init__(self, field: str, widgets, read, show,
                 readout: "QLabel | None" = None) -> None:
        self.field = field
        #: Everything that is enabled and disabled together.
        self.widgets = tuple(widgets)
        self.read = read
        self.show = show
        self.readout = readout
        #: The button that puts this one back; _finish makes it.
        self.reset: "QPushButton | None" = None

    def put(self, value) -> None:
        """Show *value* without asking for it to be applied.

        Signals are blocked: setting a slider from code raises
        valueChanged exactly as a hand would, and a reset would then
        write the value it had just been given back to the receiver
        - harmless, but it makes the log and any A/B record say
        things happened that did not.
        """
        for widget in self.widgets:
            widget.blockSignals(True)
        try:
            self.show(value)
        finally:
            for widget in self.widgets:
                widget.blockSignals(False)


class DspTab(QWidget):
    """The DSP settings page.

    Talks to the receiver through the same facade the rest of the
    window uses: ``get_dsp_defaults``, ``get_dsp_settings``,
    ``update_dsp_settings`` and ``set_dsp_settings``.
    """

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._defaults = controller.get_dsp_defaults()
        self._rows: list[_Row] = []
        #: Set while the tab is writing to its own widgets, so that
        #: the signals it cannot block do not come back as changes.
        self._settling = False

        outer = QVBoxLayout(self)
        box = QGroupBox("DSP", self)
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        outer.addWidget(box)

        settings = controller.get_dsp_settings()
        self._add_blend(grid, 0, settings)
        self._add_flag(grid, 1, "Side noise reduction",
                       "side_nr_enabled", settings)
        self._add_fraction(grid, 2, "NR floor", "side_nr_alpha_floor",
                           settings, _FRACTION_STEPS, 1.0)
        self._add_fraction(grid, 3, "NR strength", "side_nr_beta",
                           settings, _BETA_STEPS, _BETA_MAX)
        self._add_fraction(grid, 4, "L-R 7-12k ceiling",
                           "lr_high_max_gain", settings, _FRACTION_STEPS, 1.0)
        self._add_fraction(grid, 5, "L-R 12-15k ceiling",
                           "lr_super_high_max_gain", settings,
                           _FRACTION_STEPS, 1.0)
        self._add_phase(grid, 6, settings)
        self._add_delay(grid, 7, settings)
        self._add_flag(grid, 8, "I/Q phase correction",
                       "iq_phase_correction_enabled", settings)

        self._reset_all = QPushButton("Reset all", box)
        self._reset_all.clicked.connect(self._put_everything_back)
        grid.addWidget(self._reset_all, 9, 3)

        outer.addStretch(1)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _finish(self, grid, at: int, label: str, row: _Row) -> None:
        """Put a row's label, readout and reset button around it."""
        grid.addWidget(QLabel(label, self), at, 0)
        if row.readout is not None:
            row.readout.setMinimumWidth(56)
            grid.addWidget(row.readout, at, 2)
        back = QPushButton("Reset", self)
        back.clicked.connect(lambda _=False, r=row: self._put_back(r))
        grid.addWidget(back, at, 3)
        row.reset = back
        self._rows.append(row)

    def _a_slider(self, steps: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(0, steps)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
        return slider

    def _add_fraction(self, grid, at: int, label: str, field: str,
                      settings, steps: int, top: float) -> None:
        """A number from 0 to *top*, on a slider counting in steps."""
        slider = self._a_slider(steps)
        readout = QLabel("--", self)

        def read():
            return slider.value() * top / steps

        def show(value):
            slider.setValue(int(round(float(value) * steps / top)))
            readout.setText("%.2f" % float(value))

        row = _Row(field, (slider,), read, show, readout)
        slider.valueChanged.connect(lambda _=0, r=row: self._changed(r))
        grid.addWidget(slider, at, 1)
        self._finish(grid, at, label, row)
        row.put(getattr(settings, field))

    def _add_flag(self, grid, at: int, label: str, field: str,
                  settings) -> None:
        box = QCheckBox(self)

        def read():
            return box.isChecked()

        def show(value):
            box.setChecked(bool(value))

        row = _Row(field, (box,), read, show)
        box.toggled.connect(lambda _=False, r=row: self._changed(r))
        grid.addWidget(box, at, 1)
        self._finish(grid, at, label, row)
        row.put(getattr(settings, field))

    def _add_blend(self, grid, at: int, settings) -> None:
        """The forced blend: a switch, because its off is None.

        None is not a number on the slider - it is the adaptive
        blend left to itself - so the switch says whether the slider
        is being listened to at all.
        """
        holder = QWidget(self)
        row_layout = QHBoxLayout(holder)
        row_layout.setContentsMargins(0, 0, 0, 0)
        forced = QCheckBox("Force", holder)
        slider = QSlider(Qt.Orientation.Horizontal, holder)
        slider.setRange(0, _FRACTION_STEPS)
        row_layout.addWidget(forced)
        row_layout.addWidget(slider, 1)
        readout = QLabel("--", self)

        def read():
            if not forced.isChecked():
                return None
            return slider.value() / _FRACTION_STEPS

        def show(value):
            forced.setChecked(value is not None)
            slider.setEnabled(self.isEnabled() and value is not None)
            if value is None:
                # The slider is left where it was: turning the force
                # off is not a reason to lose the value the user had
                # found, and they may well turn it back on.
                readout.setText("auto")
            else:
                slider.setValue(int(round(float(value) * _FRACTION_STEPS)))
                readout.setText("%.2f" % float(value))

        self._blend_forced = forced
        self._blend_slider = slider
        row = _Row("force_blend_factor", (forced, slider), read, show,
                   readout)
        forced.toggled.connect(lambda _=False, r=row: self._changed(r))
        slider.valueChanged.connect(lambda _=0, r=row: self._changed(r))
        grid.addWidget(holder, at, 1)
        self._finish(grid, at, "Blend", row)
        row.put(settings.force_blend_factor)

    def _add_phase(self, grid, at: int, settings) -> None:
        spin = QDoubleSpinBox(self)
        spin.setRange(-_PHASE_LIMIT_DEG, _PHASE_LIMIT_DEG)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setSuffix(" deg")

        def read():
            return math.radians(spin.value())

        def show(value):
            spin.setValue(math.degrees(float(value)))

        row = _Row("subcarrier_phase_offset_rad", (spin,), read, show)
        spin.valueChanged.connect(lambda _=0.0, r=row: self._changed(r))
        grid.addWidget(spin, at, 1)
        self._finish(grid, at, "Subcarrier phase", row)
        row.put(settings.subcarrier_phase_offset_rad)

    def _add_delay(self, grid, at: int, settings) -> None:
        spin = QSpinBox(self)
        spin.setRange(0, MAX_MONO_DELAY_SAMPLES)
        spin.setSuffix(" samples")

        def read():
            return spin.value()

        def show(value):
            spin.setValue(int(value))

        row = _Row("mono_delay_samples", (spin,), read, show)
        spin.valueChanged.connect(lambda _=0, r=row: self._changed(r))
        grid.addWidget(spin, at, 1)
        self._finish(grid, at, "Mono delay", row)
        row.put(settings.mono_delay_samples)

    # ------------------------------------------------------------------
    # What the user does
    # ------------------------------------------------------------------

    def _changed(self, row: _Row) -> None:
        """One control moved: ask the receiver for that one change."""
        if self._settling:
            return
        value = row.read()
        self.controller.update_dsp_settings(**{row.field: value})
        row.put(value)                  # the readout, and nothing else

    def _put_back(self, row: _Row) -> None:
        """This parameter, back to what the receiver started under."""
        value = getattr(self._defaults, row.field)
        row.put(value)
        self.controller.update_dsp_settings(**{row.field: value})

    def _put_everything_back(self) -> None:
        """All nine at once: one set, one write, one block boundary.

        Not nine updates - the whole set is what set_dsp_settings is
        for, and nine of them would be nine chances for a block to
        be demodulated under a half-reset configuration.
        """
        self._settling = True
        try:
            for row in self._rows:
                row.put(getattr(self._defaults, row.field))
        finally:
            self._settling = False
        self.controller.set_dsp_settings(self._defaults)

    # ------------------------------------------------------------------
    # What the window does
    # ------------------------------------------------------------------

    def set_usable(self, usable: bool) -> None:
        """Let the controls be used, or do not.

        Called from the window's one place for this decision; a sweep
        owns the receiver and a device that has gone cannot be asked
        for anything.
        """
        for row in self._rows:
            for widget in row.widgets:
                widget.setEnabled(usable)
            if row.reset is not None:
                row.reset.setEnabled(usable)
        self._reset_all.setEnabled(usable)
        # The blend slider has a second say in it, the way the gain
        # slider does in the window: it is only live while the blend
        # is being forced at all.
        self._blend_slider.setEnabled(usable and
                                      self._blend_forced.isChecked())

    def showing(self) -> "dict[str, object]":
        """What the controls say, by field name.  For tests."""
        return {row.field: row.read() for row in self._rows}
