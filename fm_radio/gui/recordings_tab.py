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
"""The recordings on disk, as their sidecars describe them.

What each recording was, whether its audio is still there, and how
long it is; and, for an IQ capture, what the demodulator makes of the
start of it now - see :mod:`fm_radio.redecode`.

Three rules the list follows.

**The directory is read off the thread that draws.**  Reading it opens
every part that is still there, and an open on a filesystem that has
stopped answering does not come back - see
:func:`fm_radio.recording_meta.scan_recordings`.  On the window's own
thread that would be a window that never draws again.  The read runs
on a thread of its own and hands its rows back through a signal, the
way the band scan does.

**Nothing reads it until someone looks.**  Not when the window is
built - that is already slow enough - and not on the window's refresh,
which is twenty times a second: the directory is read the first time
the tab is chosen, and again when the Reload button is pressed.

**Every recording is counted, whether or not it is shown.**  Most
sidecars in a directory that has been cleared of audio describe
recordings that are gone, so by default only the complete ones - every
part there and a file (see ``Recording.complete``) - are listed.  The
line above the list says how many there are of each kind, so that a
filter never makes a recording disappear without saying so.

A re-decode runs in a process of its own (see :mod:`fm_radio.redecode`
for why), one at a time, and adds a row to the table under the list.
Those rows last as long as the window does; Save CSV writes them in the
command line's ``--noise-csv`` format, so that a file of them can be
read alongside one the command line wrote.
"""

from __future__ import annotations

import logging
import os
import threading

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from fm_radio.constants import AUDIO_OUTPUT_RATE
from fm_radio.quality_selftest import (
    IQ_CSV_HEADER, IqMeasurement, iq_csv_row, iq_report_lines,
)
from fm_radio.recording_meta import Recording, scan_recordings
from fm_radio.redecode import (
    CANCELLED, WINDOW_DEFAULT_S, WINDOW_MAX_S, WINDOW_MIN_S, Job,
    first_part_path, why_not,
)

logger = logging.getLogger("fm_receiver.gui")

#: The columns, left to right.
COLUMNS = ("Started", "Kind", "MHz", "Station", "Length", "Rate", "Gain",
           "Dropped", "Audio", "File")

#: The re-decode table's columns, left to right.
RESULT_COLUMNS = ("File", "MHz", "Station", "Window", "Measured",
                  "Blend avg", "Pilot SNR p10", "Pilot SNR p50",
                  "L-R / L+R", "L/R corr", "Mid HF p10", "Side HF p10",
                  "HF penalty")

#: Shown for a length worked out from the sidecar's timestamps rather
#: than measured from the WAV headers: wall-clock time, to the second,
#: including any stretch the recorder spent dropping blocks.
CLOCK_MARK = "≈"


# ----------------------------------------------------------------------
# What each cell says.  Plain functions of a Recording, so what the
# list shows can be checked without a window.
# ----------------------------------------------------------------------

def started_text(rec: Recording) -> str:
    """The start time as the recorder wrote it, to the minute."""
    if rec.started_at is None:
        return ""
    return rec.started_at.strftime("%Y-%m-%d %H:%M")


def kind_text(rec: Recording) -> str:
    return {"iq": "IQ", "audio": "Audio"}.get(rec.kind, rec.kind)


def frequency_text(rec: Recording) -> str:
    """MHz to one decimal, the way the tuner shows a frequency."""
    if rec.center_freq_hz is None:
        return ""
    return f"{rec.center_freq_hz / 1e6:.1f}"


def length_text(rec: Recording) -> str:
    """h:mm:ss or m:ss, marked when it comes from the clock."""
    seconds = rec.duration_s
    if seconds is None or seconds < 0:
        return ""
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = (f"{hours}:{minutes:02d}:{secs:02d}" if hours
            else f"{minutes}:{secs:02d}")
    return text if rec.duration_is_measured else f"{CLOCK_MARK} {text}"


def rate_text(rec: Recording) -> str:
    if rec.sample_rate_hz is None:
        return ""
    return f"{rec.sample_rate_hz / 1e3:g} kHz"


def gain_text(rec: Recording) -> str:
    return "" if rec.gain_db is None else f"{rec.gain_db:.1f} dB"


def dropped_text(rec: Recording) -> str:
    return "" if rec.dropped is None else str(rec.dropped)


def audio_text(rec: Recording) -> str:
    """Whether the recording itself is still there, in one word.

    In the order a question about it would be settled: a sidecar with a
    problem says nothing reliable about its parts; a part that could not
    be confirmed is not a part anyone has; then what is missing.

    Of the complete ones, only a recording whose headers measured some
    audio is "yes".  "empty" is one whose headers add up to none - a
    44-byte WAV, header and no frames.  "unknown" is one whose length
    could not be measured at all: a part that is not a WAV, or is
    truncated, or would not open, or several parts on a filesystem that
    cannot tell them apart.  Complete is true of all three and all three
    are listed; only the first has been shown to hold something to
    listen to.
    """
    if rec.problem:
        return "problem"
    if rec.unconfirmed:
        return "unconfirmed"
    if not rec.parts:
        return "none named"
    if not rec.missing:
        if rec.audio_seconds is None:
            return "unknown"
        return "empty" if _is_empty(rec) else "yes"
    have = len(rec.parts) - len(rec.missing)
    if have == 0:
        return "gone"
    return f"{have} of {len(rec.parts)} parts"


def _is_empty(rec: Recording) -> bool:
    """Complete, and the headers measure no audio at all."""
    return rec.complete and rec.audio_seconds == 0


def _is_unknown(rec: Recording) -> bool:
    """Complete, and the headers could not be measured."""
    return rec.complete and rec.audio_seconds is None


def notes_for(rec: Recording) -> str:
    """The longer story, for a tooltip."""
    lines = []
    if rec.problem:
        lines.append(f"{os.path.basename(rec.sidecar)} {rec.problem}")
    if rec.missing:
        lines.append("Missing: " + ", ".join(rec.missing))
    if rec.unconfirmed:
        lines.append("Could not be confirmed as files: "
                     + ", ".join(rec.unconfirmed))
    if rec.duration_s is not None and not rec.duration_is_measured:
        lines.append("Length from the start and stop times, not the audio.")
    if _is_unknown(rec):
        lines.append("The parts are there, but how much audio they hold "
                     "could not be measured.")
    return "\n".join(lines)


def cells_for(rec: Recording, station: str) -> tuple[str, ...]:
    """Every column's text for one recording, in COLUMNS order."""
    return (
        started_text(rec),
        kind_text(rec),
        frequency_text(rec),
        station,
        length_text(rec),
        rate_text(rec),
        gain_text(rec),
        dropped_text(rec),
        audio_text(rec),
        os.path.basename(rec.sidecar),
    )


# ----------------------------------------------------------------------
# What a re-decode says
# ----------------------------------------------------------------------

class Result:
    """One re-decode: what was asked, of which recording, and the answer."""

    def __init__(self, wav_path: str, frequency: str, station: str,
                 window_s: int, measured: IqMeasurement) -> None:
        self.wav_path = wav_path
        self.frequency = frequency
        self.station = station
        self.window_s = window_s
        self.measured = measured

    @property
    def tag(self) -> str:
        """The CSV's tag: the file's name, the command line's default."""
        return os.path.basename(self.wav_path)

    def csv_row(self) -> str:
        """The row the command line writes for the same file and window."""
        return iq_csv_row(self.measured, self.tag, self.window_s)


def result_cells(result: Result) -> tuple[str, ...]:
    """Every column's text for one re-decode, in RESULT_COLUMNS order.

    The numbers in the command line's own formats.  "Measured" is the
    audio the numbers are over: about the window less the warmup, or
    less when the recording is shorter than the window.  The noise
    floor columns are blank when the floor was not measured.
    """
    m = result.measured
    floor = m.noise_band_hz is not None
    return (
        result.tag,
        result.frequency,
        result.station,
        f"{result.window_s} s",
        f"{m.samples / AUDIO_OUTPUT_RATE:.1f} s",
        f"{m.blend_mean:.3f}",
        f"{m.pilot_snr_p10_db:.2f} dB",
        f"{m.pilot_snr_median_db:.2f} dB",
        f"{m.side_over_mono:.4f}",
        f"{m.correlation:.4f}",
        f"{m.mid_hf_p10_db:.2f} dB" if floor else "",
        f"{m.side_hf_p10_db:.2f} dB" if floor else "",
        f"{m.listen_penalty_db:+.2f} dB" if floor else "",
    )


# ----------------------------------------------------------------------
# Reading the directory, off the thread that draws
# ----------------------------------------------------------------------

#: Every Scan that has started and not yet been retired.  Held here,
#: on the GUI thread's side, so that the last reference to a Scan is
#: never the one its own thread lets go of - see Scan._retire.
_RUNNING: set = set()


class Scan(QObject):
    """Reads the recordings directory on a thread of its own.

    Not given a Qt parent, deliberately.  The thread may outlive the
    window - a read stuck on a dead mount does not end - and a parent
    would delete this object from under it when the window closed.

    Nor is it left to its thread to keep alive.  The thread holds its
    target, a bound method of this object, and lets go of it on its way
    out; were that the last reference, this QObject - which belongs to
    the GUI thread - would be destroyed on the reading thread.  So a
    running Scan is also held in :data:`_RUNNING`, and let go of there,
    on the GUI thread, once its thread has finished.
    """

    #: The read has ended: (list of (Recording, station name), why it
    #: failed or "").
    finished = Signal(object, str)

    def __init__(self, controller, directory: str) -> None:
        super().__init__()
        self._controller = controller
        self._directory = directory
        self._thread = threading.Thread(
            target=self._run, name="RecordingsScan", daemon=True)

    def go(self) -> None:
        # After whoever built this has connected to finished, so that
        # they hear about the result before it is retired.
        self.finished.connect(self._retire)
        _RUNNING.add(self)
        self._thread.start()

    def _retire(self, rows, why: str) -> None:
        """Let go of this Scan, on the GUI thread, after its thread.

        Delivered here, on the thread this object lives on, after the
        read has emitted.  The join is short: the emit is the last
        thing the reading thread does, and all that is left of it is
        returning.  Once it has returned it no longer holds this
        object, and dropping it from _RUNNING leaves the last reference
        with this thread.
        """
        self._thread.join()
        _RUNNING.discard(self)

    def wait(self) -> None:
        """Block until the read has finished.  For tests."""
        if self._thread.ident is not None:
            self._thread.join()

    def _run(self) -> None:
        try:
            recordings = scan_recordings(self._directory)
            # Every frequency named in one call, from one naming view: a
            # call per row would name the first rows by the area as it
            # was and the rest by an area chosen half way through.
            freqs = sorted({rec.center_freq_hz for rec in recordings
                            if rec.center_freq_hz is not None})
            stations = self._controller.stations_at(freqs)
            rows = []
            for rec in recordings:
                station = stations.get(rec.center_freq_hz)
                rows.append((rec, station.name if station else ""))
        except Exception as e:
            logger.error("Reading the recordings failed: %s", e,
                         exc_info=True)
            self.finished.emit([], str(e))
        else:
            self.finished.emit(rows, "")


class Redecode(QObject):
    """A :class:`~fm_radio.redecode.Job`, reporting through a signal.

    Kept alive the way :class:`Scan` is, and for the same reason: the
    thread that waits for the child calls back into this object, and
    must not hold the last reference to it.
    """

    #: The re-decode has ended: (IqMeasurement or None, why or "").
    finished = Signal(object, str)

    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job

    def go(self) -> None:
        """Start the job; raises if it cannot be started.

        Held only once it has started: one that could not start has no
        report coming to let go of it.  Its report cannot arrive before
        this returns - the signal is delivered on this thread.
        """
        self.finished.connect(self._retire)
        self.job.start(self.finished.emit)
        _RUNNING.add(self)

    def cancel(self) -> None:
        self.job.cancel()

    def _retire(self, measured, why: str) -> None:
        self.job.wait()
        _RUNNING.discard(self)

    def wait(self) -> None:
        """Block until the job has reported.  For tests."""
        self.job.wait()


# ----------------------------------------------------------------------
# The tab
# ----------------------------------------------------------------------

class RecordingsTab(QWidget):
    """The recordings page.

    Uses one facade call, ``stations_at``, to name every recording's
    frequency the way the dial names it, from one naming view.
    """

    def __init__(self, controller, directory: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.directory = os.path.abspath(directory)
        #: The read under way, or None.
        self._scan: Scan | None = None
        #: Every recording the last read found, shown or not.
        self._rows: list[tuple[Recording, str]] = []
        #: True once the directory has been read or is being read.
        self._looked = False
        #: The rows the table shows, in its order.
        self._listed: list[tuple[Recording, str]] = []
        #: The re-decode under way, what it is of, and whether it has
        #: been asked to stop.
        self._job: Redecode | None = None
        self._decoding: tuple[str, str, str, int] | None = None
        self._cancelling = False
        #: Every re-decode that has finished, oldest first.
        self._results: list[Result] = []

        outer = QVBoxLayout(self)

        where = QLabel(f"Recordings in {self.directory}", self)
        where.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(where)

        controls = QHBoxLayout()
        self.show_all = QCheckBox("Show every recording", self)
        self.show_all.setToolTip(
            "Include the ones with a part missing or unconfirmed, "
            "and sidecars with a problem.")
        self.show_all.toggled.connect(self._fill)
        controls.addWidget(self.show_all)
        controls.addStretch(1)
        self.reload_button = QPushButton("Reload", self)
        self.reload_button.clicked.connect(self.reload)
        controls.addWidget(self.reload_button)
        self.open_button = QPushButton("Open folder", self)
        self.open_button.clicked.connect(self._open_folder)
        controls.addWidget(self.open_button)
        outer.addLayout(controls)

        self.counts = QLabel("Not read yet.", self)
        outer.addWidget(self.counts)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_redecode)
        outer.addWidget(self.table, 2)

        decode = QHBoxLayout()
        decode.addWidget(QLabel("Re-decode the first", self))
        self.window_box = QSpinBox(self)
        self.window_box.setRange(WINDOW_MIN_S, WINDOW_MAX_S)
        self.window_box.setValue(WINDOW_DEFAULT_S)
        self.window_box.setSuffix(" s")
        decode.addWidget(self.window_box)
        decode.addWidget(QLabel(
            "of the chosen IQ recording, with the default DSP settings",
            self))
        decode.addStretch(1)
        self.redecode_button = QPushButton("Re-decode", self)
        self.redecode_button.clicked.connect(self._redecode_pressed)
        decode.addWidget(self.redecode_button)
        outer.addLayout(decode)

        self.decode_status = QLabel("", self)
        outer.addWidget(self.decode_status)

        self.results = QTableWidget(0, len(RESULT_COLUMNS), self)
        self.results.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.results.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.results, 1)

        saving = QHBoxLayout()
        saving.addStretch(1)
        self.save_button = QPushButton("Save CSV...", self)
        self.save_button.setToolTip(
            "Write every re-decode above to a CSV file, in the format "
            "quality_selftest --noise-csv appends.")
        self.save_button.clicked.connect(self._save_csv)
        saving.addWidget(self.save_button)
        outer.addLayout(saving)

        self._update_redecode()
        self._update_save()

    # --- reading --------------------------------------------------------

    def first_look(self) -> None:
        """Read the directory, unless it has been read already."""
        if not self._looked:
            self.reload()

    def reload(self) -> None:
        """Read the directory again, off this thread.

        A second press while a read is under way is ignored rather
        than queued: the read in progress will show what is there.
        """
        if self._scan is not None:
            return
        self._looked = True
        self.reload_button.setEnabled(False)
        self.counts.setText("Reading the recordings...")
        self._scan = Scan(self.controller, self.directory)
        self._scan.finished.connect(self._scan_ended)
        self._scan.go()

    def _scan_ended(self, rows, why: str) -> None:
        self._scan = None
        self.reload_button.setEnabled(True)
        if why:
            self._rows = []
            self._fill()
            self.counts.setText(f"Could not read the recordings: {why}")
            return
        self._rows = list(rows)
        self._fill()

    # --- showing --------------------------------------------------------

    def _shown(self) -> list[tuple[Recording, str]]:
        if self.show_all.isChecked():
            return self._rows
        return [(rec, name) for rec, name in self._rows if rec.complete]

    def _fill(self) -> None:
        shown = self._shown()
        self._listed = list(shown)
        self.table.clearSelection()
        self.table.setRowCount(len(shown))
        for row, (rec, name) in enumerate(shown):
            tip = notes_for(rec)
            for column, text in enumerate(cells_for(rec, name)):
                item = QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                self.table.setItem(row, column, item)
        if self._looked and self._scan is None:
            self.counts.setText(self._count_line(len(shown)))
        self._update_redecode()

    def _count_line(self, shown: int) -> str:
        """How many there are of each kind; the three add up to the total.

        A row with a problem is never complete, so the three are
        disjoint: complete, incomplete without a problem, and with one.
        """
        total = len(self._rows)
        with_problem = sum(1 for rec, _ in self._rows if rec.problem)
        complete = sum(1 for rec, _ in self._rows if rec.complete)
        empty = sum(1 for rec, _ in self._rows if _is_empty(rec))
        unknown = sum(1 for rec, _ in self._rows if _is_unknown(rec))
        incomplete = total - complete - with_problem
        # The complete ones that are not "yes", said only when there are
        # any: a line that always read "(0 empty, 0 unknown)" would
        # teach the eye to skip the brackets.
        not_yes = [f"{n} {word}" for n, word in
                   ((empty, "empty"), (unknown, "unknown")) if n]
        complete_text = (f"{complete} complete ({', '.join(not_yes)})"
                         if not_yes else f"{complete} complete")
        return (f"{shown} of {total} shown: {complete_text}, "
                f"{incomplete} incomplete, {with_problem} with a problem")

    # --- re-decoding ----------------------------------------------------

    def _chosen(self) -> tuple[Recording, str] | None:
        """The recording chosen in the list, if one is."""
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            return None
        row = rows[0].row()
        if not 0 <= row < len(self._listed):
            return None
        return self._listed[row]

    def _update_redecode(self) -> None:
        """What the button does now, and whether it can."""
        button = self.redecode_button
        if self._job is not None:
            button.setText("Cancel")
            button.setEnabled(not self._cancelling)
            button.setToolTip("Stop the re-decode under way.")
            return
        button.setText("Re-decode")
        chosen = self._chosen()
        if chosen is None:
            button.setEnabled(False)
            button.setToolTip("Choose a recording in the list.")
            return
        why = why_not(chosen[0])
        button.setEnabled(not why)
        button.setToolTip(why or "Decode the start of it again and measure "
                                 "what the demodulator makes of it.")

    def _redecode_pressed(self) -> None:
        if self._job is not None:
            self._cancelling = True
            self._job.cancel()
            self.decode_status.setText("Stopping the re-decode...")
            self._update_redecode()
            return
        chosen = self._chosen()
        if chosen is None or why_not(chosen[0]):
            return
        rec, station = chosen
        window = self.window_box.value()
        path = first_part_path(rec)
        job = Redecode(Job(path, window))
        job.finished.connect(self._redecode_ended)
        try:
            job.go()
        except Exception as e:
            # A process that cannot be started - out of memory, or of
            # handles - is said where the answer would have been.
            logger.error("Could not start a re-decode: %s", e,
                         exc_info=True)
            self.decode_status.setText(
                f"Could not start the re-decode of "
                f"{os.path.basename(path)}: {e}")
            return
        self._job = job
        self._decoding = (path, frequency_text(rec), station, window)
        self.window_box.setEnabled(False)
        self.decode_status.setText(
            f"Re-decoding the first {window} s of "
            f"{os.path.basename(path)}...")
        self._update_redecode()

    def _redecode_ended(self, measured, why: str) -> None:
        path, frequency, station, window = self._decoding
        name = os.path.basename(path)
        self._job = None
        self._decoding = None
        self._cancelling = False
        self.window_box.setEnabled(True)
        if measured is None:
            self.decode_status.setText(
                f"Re-decode of {name} stopped." if why == CANCELLED
                else f"Re-decode of {name} failed: {why}")
        else:
            result = Result(path, frequency, station, window, measured)
            self._results.append(result)
            self._add_result(result)
            self.decode_status.setText(
                f"Re-decoded the first {window} s of {name}.")
        self._update_redecode()
        self._update_save()

    def _add_result(self, result: Result) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        # The whole report, as the command line prints it, on every cell.
        tip = "\n".join(iq_report_lines(result.measured))
        for column, text in enumerate(result_cells(result)):
            item = QTableWidgetItem(text)
            item.setToolTip(tip)
            self.results.setItem(row, column, item)

    def shutdown(self) -> None:
        """Stop a re-decode, if one is running.  For the window closing."""
        if self._job is not None:
            self._cancelling = True
            self._job.cancel()

    # --- saving ---------------------------------------------------------

    def _update_save(self) -> None:
        self.save_button.setEnabled(bool(self._results))

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the re-decodes",
            os.path.join(self.directory, "redecode.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        try:
            # Text mode, as the command line writes it, so that the two
            # files end their lines the same way.
            with open(path, "w", encoding="utf-8") as f:
                f.write(IQ_CSV_HEADER)
                for result in self._results:
                    f.write(result.csv_row())
        except OSError as e:
            self.decode_status.setText(f"Could not save {path}: {e}")
            return
        self.decode_status.setText(
            f"Saved {len(self._results)} re-decodes to {path}")

    # --- the folder -----------------------------------------------------

    def _open_folder(self) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(self.directory)):
            self.counts.setText(f"Could not open {self.directory}")
