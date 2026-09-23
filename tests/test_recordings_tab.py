"""The Recordings tab (P5 PR-B): what the list says, and when and where
the directory is read.

Qt runs offscreen (see the ``qt_app`` fixture in conftest), so these need
no display.  The sidecars and WAVs are real files in ``tmp_path``: the tab
is read through ``recording_meta.scan_recordings``, and a list checked
against stand-in rows would not be checking what the tab shows.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import wave
import weakref

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="the GUI is optional")

from fm_radio.gui import recordings_tab                      # noqa: E402
from fm_radio.gui.recordings_tab import (                     # noqa: E402
    CLOCK_MARK, COLUMNS, RecordingsTab, audio_text, length_text,
)
from fm_radio.recording_meta import Recording                # noqa: E402

_STARTED = "2026-09-20T01:29:48+09:00"
_STOPPED = "2026-09-20T01:31:48+09:00"     # two minutes on the clock


def _write_wav(path, frames, rate=48000, channels=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (frames * channels * 2))


def _sidecar(directory, name, parts, *, kind="iq", freq=81.3e6,
             started=_STARTED, stopped=_STOPPED, **extra):
    meta = {
        "type": kind,
        "file": parts[0] if parts else "",
        "sample_rate_hz": 1024000 if kind == "iq" else 48000,
        "center_freq_hz": freq,
        "gain_db": 8.7,
        "started_at": started,
        "stopped_at": stopped,
        "parts": list(parts),
        ("dropped_blocks" if kind == "iq" else "dropped_chunks"): 2,
    }
    meta.update(extra)
    with open(str(directory / (name + ".json")), "w", encoding="utf-8") as f:
        json.dump(meta, f)


class Namer:
    """The one facade call the tab makes - and which thread made it."""

    def __init__(self):
        self.calls: list[list[float]] = []
        self.threads: list[threading.Thread] = []

    def stations_at(self, freqs):
        freqs = list(freqs)
        self.calls.append(freqs)
        self.threads.append(threading.current_thread())
        names = {80.0e6: "TOKYO FM", 81.3e6: "J-WAVE"}
        return {f: types.SimpleNamespace(name=names[f]) if f in names
                else None for f in freqs}


@pytest.fixture
def make_tab(qt_app, tmp_path):
    """A tab over *tmp_path*; anything a slot raises fails the test.

    Qt has nowhere to send an exception raised in a slot, so it prints
    it and carries on, and every assertion after it would still pass.
    """
    swallowed = []
    was = sys.excepthook
    sys.excepthook = lambda *trouble: swallowed.append(trouble)
    built = []

    def _make(namer=None, directory=None):
        tab = RecordingsTab(namer or Namer(), str(directory or tmp_path))
        built.append(tab)
        return tab

    try:
        yield _make
        for tab in built:
            if tab._scan is not None:
                tab._scan.wait()
            if tab._job is not None:
                tab._job.cancel()
                tab._job.wait()
            tab.deleteLater()
        qt_app.processEvents()
    finally:
        sys.excepthook = was
    assert not swallowed, f"a slot raised: {swallowed[0][1]!r}"


def finish(tab, qt_app, timeout: float = 5.0) -> None:
    """Let the reading thread end and its signal be delivered."""
    scan = tab._scan
    if scan is not None:
        scan.wait()
    deadline = time.monotonic() + timeout
    while tab._scan is not None and time.monotonic() < deadline:
        qt_app.processEvents()
    assert tab._scan is None, "the read never reported back"


def shown(tab) -> list[dict[str, str]]:
    """The table as it is on screen: one dict per row, by column name."""
    rows = []
    for r in range(tab.table.rowCount()):
        rows.append({name: tab.table.item(r, c).text()
                     for c, name in enumerate(COLUMNS)})
    return rows


def a_mixed_directory(tmp_path):
    """One of each: complete, gone, partial, and broken."""
    _sidecar(tmp_path, "complete", ["complete.wav"],
             started="2026-09-21T10:00:00+09:00",
             stopped="2026-09-21T10:02:00+09:00")
    _write_wav(tmp_path / "complete.wav", 96000)           # 2.0 s
    _sidecar(tmp_path, "gone", ["gone.wav"], kind="audio", freq=80.0e6)
    _sidecar(tmp_path, "partial", ["p.wav", "p.part001.wav"],
             started="2026-09-19T10:00:00+09:00")
    _write_wav(tmp_path / "p.wav", 48000)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")


# ----------------------------------------------------------------------
# When and where the directory is read
# ----------------------------------------------------------------------

def test_nothing_is_read_until_someone_looks(make_tab, qt_app, monkeypatch):
    reads = []
    monkeypatch.setattr(recordings_tab, "scan_recordings",
                        lambda d: reads.append(d) or [])
    tab = make_tab()
    qt_app.processEvents()
    assert reads == []
    assert tab.counts.text() == "Not read yet."

    tab.first_look()
    finish(tab, qt_app)
    tab.first_look()                         # a second look reads nothing
    finish(tab, qt_app)

    assert reads == [tab.directory]


def test_the_directory_is_read_off_the_thread_that_draws(make_tab, qt_app,
                                                         tmp_path,
                                                         monkeypatch):
    """Every part is opened, and an open on a dead mount never returns.

    The station names are worked out on the same thread: a few hundred
    lookups are not something to do on the one that draws either.
    """
    _sidecar(tmp_path, "one", ["one.wav"])
    where = []
    real = recordings_tab.scan_recordings

    def watching(directory):
        where.append(threading.current_thread())
        return real(directory)

    monkeypatch.setattr(recordings_tab, "scan_recordings", watching)
    namer = Namer()
    tab = make_tab(namer)

    tab.reload()
    finish(tab, qt_app)

    assert where and where[0] is not threading.main_thread()
    assert namer.threads and namer.threads[0] is not threading.main_thread()


def test_a_second_reload_while_reading_is_ignored(make_tab, qt_app,
                                                  monkeypatch):
    """The read in progress will show what is there; a queue of them
    would read the same directory again for nothing."""
    hold = threading.Event()
    reads = []

    def slow(directory):
        reads.append(directory)
        assert hold.wait(5), "the test never let the read finish"
        return []

    monkeypatch.setattr(recordings_tab, "scan_recordings", slow)
    tab = make_tab()

    tab.reload()
    first = tab._scan
    assert tab.reload_button.isEnabled() is False
    tab.reload()
    assert tab._scan is first

    hold.set()
    finish(tab, qt_app)
    assert len(reads) == 1
    assert tab.reload_button.isEnabled() is True


def test_a_read_that_outlives_the_tab_is_let_go_of_where_it_was_made(
        qt_app, tmp_path, monkeypatch):
    """A read stuck on a dead mount can outlast the window.

    Two ways that goes wrong.  If the tab owned the reader, closing the
    window would delete it under its own thread.  If nothing but the
    thread held it, the thread would drop the last reference on its way
    out, and a QObject that belongs to the GUI thread would be destroyed
    on the reading one.  So the tab and every local reference to the
    reader are let go of here before the read finishes, and the test
    watches which thread frees it.
    """
    import gc

    import shiboken6

    hold = threading.Event()

    def slow(directory):
        assert hold.wait(5), "the test never let the read finish"
        return []

    monkeypatch.setattr(recordings_tab, "scan_recordings", slow)
    raised = []
    monkeypatch.setattr(threading, "excepthook",
                        lambda args: raised.append(args))
    freed_on = []
    tab = RecordingsTab(Namer(), str(tmp_path))
    tab.reload()
    watch = weakref.ref(
        tab._scan,
        lambda _: freed_on.append(threading.current_thread()))
    thread = tab._scan._thread

    shiboken6.delete(tab)                 # the window has gone ...
    del tab                               # ... and nothing here holds it
    gc.collect()
    hold.set()
    thread.join()
    deadline = time.monotonic() + 5.0
    while watch() is not None and time.monotonic() < deadline:
        qt_app.processEvents()

    assert watch() is None, "the reader was never let go of"
    assert freed_on == [threading.main_thread()], (
        f"freed on {freed_on[0].name if freed_on else 'no thread'}")
    assert raised == [], f"the reader raised: {raised[0].exc_value!r}"
    assert recordings_tab._RUNNING == set()


def test_a_read_that_fails_says_so(make_tab, qt_app, monkeypatch):
    def fails(directory):
        raise RuntimeError("the share went away")

    monkeypatch.setattr(recordings_tab, "scan_recordings", fails)
    tab = make_tab()

    tab.reload()
    finish(tab, qt_app)

    assert tab.table.rowCount() == 0
    assert tab.counts.text() == (
        "Could not read the recordings: the share went away")
    assert tab.reload_button.isEnabled() is True


def test_reload_sees_what_was_recorded_since(make_tab, qt_app, tmp_path):
    _sidecar(tmp_path, "first", ["first.wav"])
    _write_wav(tmp_path / "first.wav", 48000)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    assert [r["File"] for r in shown(tab)] == ["first.json"]

    _sidecar(tmp_path, "second", ["second.wav"],
             started="2026-09-22T10:00:00+09:00",
             stopped="2026-09-22T10:01:00+09:00")
    _write_wav(tmp_path / "second.wav", 48000)
    tab.reload()
    finish(tab, qt_app)

    assert [r["File"] for r in shown(tab)] == ["second.json", "first.json"]


# ----------------------------------------------------------------------
# What is shown, and what is counted
# ----------------------------------------------------------------------

def test_by_default_only_complete_recordings_are_listed(make_tab, qt_app,
                                                        tmp_path):
    a_mixed_directory(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)

    assert [r["File"] for r in shown(tab)] == ["complete.json"]
    assert tab.counts.text() == (
        "1 of 4 shown: 1 complete, 2 incomplete, 1 with a problem")


def test_every_recording_is_shown_when_asked_for(make_tab, qt_app,
                                                 tmp_path):
    a_mixed_directory(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)

    tab.show_all.setChecked(True)

    rows = shown(tab)
    # Newest first, and the one with no start time last.
    assert [r["File"] for r in rows] == [
        "complete.json", "gone.json", "partial.json", "broken.json"]
    assert [r["Audio"] for r in rows] == [
        "yes", "gone", "1 of 2 parts", "problem"]
    assert tab.counts.text() == (
        "4 of 4 shown: 1 complete, 2 incomplete, 1 with a problem")

    tab.show_all.setChecked(False)
    assert [r["File"] for r in shown(tab)] == ["complete.json"]


def test_what_a_row_says(make_tab, qt_app, tmp_path):
    a_mixed_directory(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    tab.show_all.setChecked(True)

    complete, gone = shown(tab)[:2]
    assert complete == {
        "Started": "2026-09-21 10:00",
        "Kind": "IQ",
        "MHz": "81.3",
        "Station": "J-WAVE",
        "Length": "0:02",           # the header's 2 s, not the clock's 120
        "Rate": "1024 kHz",
        "Gain": "8.7 dB",
        "Dropped": "2",
        "Audio": "yes",
        "File": "complete.json",
    }
    # No audio to measure: the clock's two minutes, marked as the clock.
    assert gone["Length"] == f"{CLOCK_MARK} 2:00"
    assert gone["Kind"] == "Audio"
    assert gone["Rate"] == "48 kHz"
    assert gone["Station"] == "TOKYO FM"


def test_a_row_says_why_in_its_tooltip(make_tab, qt_app, tmp_path):
    a_mixed_directory(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    tab.show_all.setChecked(True)

    by_file = {tab.table.item(r, COLUMNS.index("File")).text(): r
               for r in range(tab.table.rowCount())}
    partial_tip = tab.table.item(by_file["partial.json"], 0).toolTip()
    broken_tip = tab.table.item(by_file["broken.json"], 0).toolTip()
    complete_tip = tab.table.item(by_file["complete.json"], 0).toolTip()

    assert "Missing: p.part001.wav" in partial_tip
    assert broken_tip.startswith("broken.json is not valid JSON")
    assert complete_tip == ""


def test_station_names_are_asked_for_once_per_read(make_tab, qt_app,
                                                        tmp_path):
    for i in range(3):
        _sidecar(tmp_path, f"j{i}", [f"j{i}.wav"], freq=81.3e6)
    _sidecar(tmp_path, "t", ["t.wav"], freq=80.0e6)
    namer = Namer()
    tab = make_tab(namer)
    tab.reload()
    finish(tab, qt_app)
    tab.show_all.setChecked(True)

    # One call for the whole read, every frequency once: one naming
    # view names every row.
    assert namer.calls == [[80.0e6, 81.3e6]]
    assert sorted(r["Station"] for r in shown(tab)) == [
        "J-WAVE", "J-WAVE", "J-WAVE", "TOKYO FM"]


def test_a_read_with_no_frequencies_still_asks_once(make_tab, qt_app,
                                                    tmp_path):
    """Once per read, including a read with nothing to name.

    An empty directory, or sidecars that give no frequency: the one call
    is made with no frequencies in it, rather than skipped - "once per
    read" is then true of every read, not of the ones with something in
    them.
    """
    _sidecar(tmp_path, "nofreq", ["n.wav"], freq=None)
    namer = Namer()
    tab = make_tab(namer)
    tab.reload()
    finish(tab, qt_app)
    tab.show_all.setChecked(True)

    assert namer.calls == [[]]
    assert [r["Station"] for r in shown(tab)] == [""]


def test_open_folder_opens_the_directory(make_tab, monkeypatch):
    opened = []
    monkeypatch.setattr(recordings_tab.QDesktopServices, "openUrl",
                        lambda url: opened.append(url) or True)
    tab = make_tab()
    tab.open_button.click()
    assert [u.toLocalFile() for u in opened] == [
        tab.directory.replace("\\", "/")]


def test_open_folder_says_so_when_it_cannot(make_tab, monkeypatch):
    monkeypatch.setattr(recordings_tab.QDesktopServices, "openUrl",
                        lambda url: False)
    tab = make_tab()
    tab.open_button.click()
    assert tab.counts.text() == f"Could not open {tab.directory}"


# ----------------------------------------------------------------------
# The words, without a window
# ----------------------------------------------------------------------

def _rec(**fields) -> Recording:
    base = dict(
        sidecar="x.json", kind="iq", parts=("a.wav",), missing=(),
        unconfirmed=(), sample_rate_hz=None, center_freq_hz=None,
        gain_db=None, channels=None, started_at=None, stopped_at=None,
        dropped=None, audio_seconds=None, wall_seconds=None, problem="",
    )
    base.update(fields)
    return Recording(**base)


def test_length_is_minutes_and_seconds_then_hours():
    assert length_text(_rec(audio_seconds=59.6)) == "1:00"
    assert length_text(_rec(audio_seconds=2234.0)) == "37:14"
    assert length_text(_rec(audio_seconds=3 * 3600 + 5.0)) == "3:00:05"
    assert length_text(_rec(wall_seconds=120.0)) == f"{CLOCK_MARK} 2:00"
    assert length_text(_rec()) == ""
    # Stopped before it started: a sidecar edited by hand, or a clock
    # that jumped.  Nothing is shown rather than a negative length.
    assert length_text(_rec(wall_seconds=-5.0)) == ""


def test_the_audio_word_settles_the_question_in_order():
    # A problem says nothing reliable about the parts, whatever else.
    assert audio_text(_rec(problem="x", missing=("a.wav",))) == "problem"
    # An unconfirmed part is not a part anyone has.
    assert audio_text(_rec(parts=("a.wav", "b.wav"), missing=("b.wav",),
                           unconfirmed=("a.wav",))) == "unconfirmed"
    assert audio_text(_rec(parts=())) == "none named"
    assert audio_text(_rec(audio_seconds=1.5)) == "yes"
    # Complete, and nothing in it: a header with no frames.
    assert audio_text(_rec(audio_seconds=0.0)) == "empty"
    # Complete, and nothing measured: not a WAV, truncated, shut, or
    # parts that cannot be told apart.  Not "yes".
    assert audio_text(_rec()) == "unknown"
    assert audio_text(_rec(missing=("a.wav",))) == "gone"
    assert audio_text(_rec(parts=("a.wav", "b.wav", "c.wav"),
                           missing=("c.wav",))) == "2 of 3 parts"


def test_an_empty_recording_is_listed_and_called_empty(make_tab, qt_app,
                                                        tmp_path):
    """Five of the thirty complete recordings in the real directory are
    44-byte WAVs.  Complete is true of them; "yes" is not."""
    _sidecar(tmp_path, "empty", ["empty.wav"])
    _write_wav(tmp_path / "empty.wav", 0)
    _sidecar(tmp_path, "full", ["full.wav"],
             started="2026-09-21T10:00:00+09:00",
             stopped="2026-09-21T10:00:01+09:00")
    _write_wav(tmp_path / "full.wav", 48000)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)

    assert [(r["File"], r["Audio"], r["Length"]) for r in shown(tab)] == [
        ("full.json", "yes", "0:01"), ("empty.json", "empty", "0:00")]
    assert tab.counts.text() == (
        "2 of 2 shown: 2 complete (1 empty), 0 incomplete, 0 with a problem")


def test_a_recording_whose_audio_cannot_be_measured_is_not_yes(
        make_tab, qt_app, tmp_path):
    """Every part there, and nothing to measure: a WAV of 0 bytes.

    Complete is true of it; "yes" - there is something to listen to -
    is not shown by anything, and the length on the clock says nothing
    about what the file holds.
    """
    _sidecar(tmp_path, "bad", ["bad.wav"])
    (tmp_path / "bad.wav").write_bytes(b"")
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)

    (row,) = shown(tab)
    assert (row["Audio"], row["Length"]) == ("unknown", f"{CLOCK_MARK} 2:00")
    assert tab.counts.text() == (
        "1 of 1 shown: 1 complete (1 unknown), 0 incomplete, "
        "0 with a problem")
    tip = tab.table.item(0, 0).toolTip()
    assert "could not be measured" in tip


# ----------------------------------------------------------------------
# Re-decoding (P5 PR-D)
#
# Most of these put a stand-in where the job would be: what is under
# test is what the tab asks for and what it does with the answer.  The
# job itself is fm_radio.redecode's, tested in test_redecode; the last
# tests here run a real one, from the button to the saved file.
# ----------------------------------------------------------------------

from fm_radio.quality_selftest import (                       # noqa: E402
    IQ_CSV_HEADER, IqMeasurement, iq_csv_row, iq_report_lines,
)
from fm_radio.gui.recordings_tab import (                     # noqa: E402
    RESULT_COLUMNS, Result, result_cells,
)
from fm_radio.redecode import CANCELLED, Job                  # noqa: E402

_MEASURED = IqMeasurement(
    samples=1_401_600, rms_left=0.1, rms_right=0.05,
    side_over_mono=0.3333, correlation=0.9012,
    blend_mean=0.8766, blend_min=0.2, blend_max=1.0,
    pilot_snr_p10_db=24.126, pilot_snr_median_db=28.5,
    pilot_snr_mean_db=27.994, pilot_snr_max_db=31.0,
    noise_band_hz=(10000.0, 14500.0),
    mid_hf_p10_db=-80.0, side_hf_p10_db=-72.5, listen_penalty_db=0.154,
)


class StandInJob:
    """What the tab asked for, and a way to answer it as the job would.

    The answer comes from a thread of its own, as a real job's does.
    """

    made: list = []

    def __init__(self, wav_path, window_s):
        self.wav_path = wav_path
        self.window_s = window_s
        self.cancelled = False
        self.on_done = None
        self._thread = None
        StandInJob.made.append(self)

    def start(self, on_done):
        self.on_done = on_done

    def cancel(self):
        self.cancelled = True

    def wait(self):
        if self._thread is not None:
            self._thread.join()

    def answer(self, measured, why=""):
        self._thread = threading.Thread(target=self.on_done,
                                        args=(measured, why))
        self._thread.start()
        self._thread.join()


@pytest.fixture
def stand_in(monkeypatch, qt_app):
    StandInJob.made = []
    monkeypatch.setattr(recordings_tab, "Job", StandInJob)
    yield StandInJob.made
    # A job a test left unanswered is answered here, so that it is let
    # go of rather than left in _RUNNING for a later test to find.
    for job in StandInJob.made:
        if job._thread is None and job.on_done is not None:
            job.answer(None, CANCELLED)
    qt_app.processEvents()


def decoded(tab, qt_app, timeout: float = 5.0) -> None:
    """Let the job's answer be delivered."""
    deadline = time.monotonic() + timeout
    while tab._job is not None and time.monotonic() < deadline:
        qt_app.processEvents()
    assert tab._job is None, "the re-decode never reported back"


def three_kinds(tmp_path):
    """A standard-rate IQ capture, a light one and an audio recording."""
    _sidecar(tmp_path, "iq", ["iq.wav", "iq.part001.wav"], freq=80.0e6,
             started="2026-09-21T10:00:00+09:00")
    _write_wav(tmp_path / "iq.wav", 1024, rate=1024000)
    _write_wav(tmp_path / "iq.part001.wav", 1024, rate=1024000)
    _sidecar(tmp_path, "light", ["light.wav"], sample_rate_hz=250000,
             started="2026-09-20T10:00:00+09:00")
    _write_wav(tmp_path / "light.wav", 1024, rate=250000)
    _sidecar(tmp_path, "audio", ["audio.wav"], kind="audio",
             started="2026-09-19T10:00:00+09:00")
    _write_wav(tmp_path / "audio.wav", 1024)


def choose(tab, file_name: str) -> None:
    """Choose the row whose File column is *file_name*."""
    for r, row in enumerate(shown(tab)):
        if row["File"] == file_name:
            tab.table.selectRow(r)
            return
    raise AssertionError(f"{file_name} is not listed")


def results(tab) -> list[dict[str, str]]:
    return [{name: tab.results.item(r, c).text()
             for c, name in enumerate(RESULT_COLUMNS)}
            for r in range(tab.results.rowCount())]


def test_nothing_chosen_nothing_to_re_decode(make_tab, qt_app, tmp_path):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    assert tab.redecode_button.isEnabled() is False
    assert "Choose" in tab.redecode_button.toolTip()
    assert tab.save_button.isEnabled() is False


def test_the_button_says_whether_the_chosen_one_can_be(make_tab, qt_app,
                                                        tmp_path):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)

    choose(tab, "iq.json")
    assert tab.redecode_button.isEnabled() is True
    choose(tab, "light.json")
    assert tab.redecode_button.isEnabled() is False
    assert "250 kHz" in tab.redecode_button.toolTip()
    choose(tab, "audio.json")
    assert tab.redecode_button.isEnabled() is False
    assert "Only IQ" in tab.redecode_button.toolTip()


def test_a_reload_forgets_the_choice(make_tab, qt_app, tmp_path):
    """The rows are rebuilt, so a row number no longer names the same
    recording; nothing stays chosen that was not chosen again."""
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.reload()
    finish(tab, qt_app)
    assert tab.redecode_button.isEnabled() is False


def test_the_first_part_is_decoded_for_the_window_asked(
        make_tab, qt_app, tmp_path, stand_in):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.window_box.setValue(45)

    tab.redecode_button.click()

    (job,) = stand_in
    assert (job.wav_path, job.window_s) == (str(tmp_path / "iq.wav"), 45)
    assert tab.redecode_button.text() == "Cancel"
    assert tab.redecode_button.isEnabled() is True
    assert tab.window_box.isEnabled() is False
    assert tab.decode_status.text() == "Re-decoding the first 45 s of iq.wav..."


def test_a_finished_re_decode_is_a_row_with_the_report_behind_it(
        make_tab, qt_app, tmp_path, stand_in):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()

    stand_in[0].answer(_MEASURED)
    decoded(tab, qt_app)

    assert results(tab) == [{
        "File": "iq.wav", "MHz": "80.0", "Station": "TOKYO FM",
        "Window": "30 s", "Measured": "29.2 s",
        "Blend avg": "0.877", "Pilot SNR p10": "24.13 dB",
        "Pilot SNR p50": "28.50 dB", "L-R / L+R": "0.3333",
        "L/R corr": "0.9012", "Mid HF p10": "-80.00 dB",
        "Side HF p10": "-72.50 dB", "HF penalty": "+0.15 dB",
    }]
    assert tab.results.item(0, 4).toolTip() == "\n".join(
        iq_report_lines(_MEASURED))
    assert tab.decode_status.text() == "Re-decoded the first 30 s of iq.wav."
    assert tab.redecode_button.text() == "Re-decode"
    assert tab.redecode_button.isEnabled() is True     # still chosen
    assert tab.window_box.isEnabled() is True
    assert tab.save_button.isEnabled() is True


def test_without_a_noise_floor_its_columns_are_blank():
    from dataclasses import replace
    nan = float("nan")
    cells = dict(zip(RESULT_COLUMNS, result_cells(Result(
        "x.wav", "80.0", "", 30,
        replace(_MEASURED, noise_band_hz=None, mid_hf_p10_db=nan,
                side_hf_p10_db=nan, listen_penalty_db=nan)))))
    assert (cells["Mid HF p10"], cells["Side HF p10"],
            cells["HF penalty"]) == ("", "", "")


def test_cancel_asks_the_job_to_stop_and_says_so(make_tab, qt_app, tmp_path,
                                                 stand_in):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()

    tab.redecode_button.click()
    assert stand_in[0].cancelled is True
    assert len(stand_in) == 1                  # and nothing new started
    assert tab.redecode_button.isEnabled() is False
    assert tab.decode_status.text() == "Stopping the re-decode..."

    stand_in[0].answer(None, CANCELLED)
    decoded(tab, qt_app)
    assert tab.decode_status.text() == "Re-decode of iq.wav stopped."
    assert results(tab) == []
    assert tab.redecode_button.text() == "Re-decode"
    assert tab.save_button.isEnabled() is False


def test_a_failed_re_decode_says_why(make_tab, qt_app, tmp_path, stand_in):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()

    stand_in[0].answer(None, "ValueError: no")
    decoded(tab, qt_app)
    assert tab.decode_status.text() == (
        "Re-decode of iq.wav failed: ValueError: no")
    assert results(tab) == []


def test_a_reload_while_decoding_keeps_the_answer_with_its_recording(
        make_tab, qt_app, tmp_path, stand_in):
    """What was asked is kept with the job, not read back from the list:
    by the time the answer comes, the row it was asked from may be
    another recording or none."""
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()

    (tmp_path / "iq.json").unlink()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "light.json")
    stand_in[0].answer(_MEASURED)
    decoded(tab, qt_app)

    (row,) = results(tab)
    assert (row["File"], row["MHz"], row["Station"]) == (
        "iq.wav", "80.0", "TOKYO FM")


def test_save_csv_writes_every_re_decode_in_the_command_lines_format(
        make_tab, qt_app, tmp_path, stand_in, monkeypatch):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    for window, measured in ((30, _MEASURED),
                             (60, IqMeasurement(**{
                                 **_MEASURED.__dict__, "blend_mean": 0.25}))):
        tab.window_box.setValue(window)
        tab.redecode_button.click()
        stand_in[-1].answer(measured)
        decoded(tab, qt_app)

    target = tmp_path / "out" / "saved.csv"
    target.parent.mkdir()
    asked = []
    monkeypatch.setattr(
        recordings_tab.QFileDialog, "getSaveFileName",
        lambda *a: asked.append(a) or (str(target), "CSV files (*.csv)"))
    tab.save_button.click()

    assert asked[0][2] == str(tmp_path / "redecode.csv")
    assert target.read_text(encoding="utf-8") == (
        IQ_CSV_HEADER
        + iq_csv_row(_MEASURED, "iq.wav", 30)
        + iq_csv_row(IqMeasurement(**{**_MEASURED.__dict__,
                                      "blend_mean": 0.25}), "iq.wav", 60))
    assert tab.decode_status.text() == f"Saved 2 re-decodes to {target}"


def test_save_csv_that_is_not_chosen_writes_nothing(make_tab, qt_app,
                                                   tmp_path, stand_in,
                                                   monkeypatch):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()
    stand_in[0].answer(_MEASURED)
    decoded(tab, qt_app)
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(recordings_tab.QFileDialog, "getSaveFileName",
                        lambda *a: ("", ""))
    tab.save_button.click()
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert tab.decode_status.text() == "Re-decoded the first 30 s of iq.wav."


def test_save_csv_that_cannot_be_written_says_so(make_tab, qt_app, tmp_path,
                                                stand_in, monkeypatch):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()
    stand_in[0].answer(_MEASURED)
    decoded(tab, qt_app)
    nowhere = tmp_path / "no-such-folder" / "x.csv"
    monkeypatch.setattr(recordings_tab.QFileDialog, "getSaveFileName",
                        lambda *a: (str(nowhere), ""))
    tab.save_button.click()
    assert tab.decode_status.text().startswith(f"Could not save {nowhere}")


def test_a_re_decode_that_cannot_start_says_so(make_tab, qt_app, tmp_path,
                                               monkeypatch):
    """A process that cannot be started leaves nothing running: the
    button, the window and the registry are as they were."""

    class WillNotStart(StandInJob):
        def start(self, on_done):
            raise OSError("no handles left")

    monkeypatch.setattr(recordings_tab, "Job", WillNotStart)
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "iq.json")
    tab.redecode_button.click()

    assert tab.decode_status.text() == (
        "Could not start the re-decode of iq.wav: no handles left")
    assert tab._job is None
    assert tab.redecode_button.text() == "Re-decode"
    assert tab.redecode_button.isEnabled() is True
    assert tab.window_box.isEnabled() is True
    assert recordings_tab._RUNNING == set()


def test_shutdown_stops_a_running_re_decode(make_tab, qt_app, tmp_path,
                                            stand_in):
    three_kinds(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    tab.shutdown()                                   # nothing running
    choose(tab, "iq.json")
    tab.redecode_button.click()
    tab.shutdown()
    assert stand_in[0].cancelled is True


# ----------------------------------------------------------------------
# With a real job
# ----------------------------------------------------------------------

def _a_real_capture(tmp_path, seconds=3.0):
    from iq_capture import write_stereo_iq_wav
    write_stereo_iq_wav(tmp_path / "real.wav", seconds)
    _sidecar(tmp_path, "real", ["real.wav"], freq=80.0e6)


def test_the_tab_saves_what_the_command_line_writes(make_tab, qt_app,
                                                    tmp_path, monkeypatch,
                                                    capsys):
    """From the button to the file, against the command line's own
    --noise-csv for the same capture and window: the same bytes."""
    import fm_radio.quality_selftest as qs

    _a_real_capture(tmp_path)
    tab = make_tab()
    tab.reload()
    finish(tab, qt_app)
    choose(tab, "real.json")
    tab.window_box.setValue(2)
    tab.redecode_button.click()
    tab._job.wait()
    decoded(tab, qt_app)
    assert results(tab), tab.decode_status.text()

    saved = tmp_path / "gui.csv"
    monkeypatch.setattr(recordings_tab.QFileDialog, "getSaveFileName",
                        lambda *a: (str(saved), ""))
    tab.save_button.click()

    written = tmp_path / "cli.csv"
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--iq-wav", str(tmp_path / "real.wav"),
        "--duration", "2", "--noise-csv", str(written)])
    qs.main()
    capsys.readouterr()
    assert saved.read_bytes() == written.read_bytes()


def test_a_re_decode_that_outlives_the_tab_is_let_go_of_where_it_was_made(
        qt_app, tmp_path, monkeypatch):
    """As for the read: the job's thread calls back into the object
    that reports for it, and must not be what frees it."""
    import gc
    from multiprocessing.connection import Listener

    import shiboken6

    import redecode_children

    _a_real_capture(tmp_path, 0.2)
    raised = []
    monkeypatch.setattr(threading, "excepthook",
                        lambda args: raised.append(args))
    freed_on = []
    with Listener() as listener:
        monkeypatch.setattr(recordings_tab, "Job", lambda path, window: Job(
            listener.address, window,
            target=redecode_children.answer_when_told))
        tab = RecordingsTab(Namer(), str(tmp_path))
        tab.reload()
        finish(tab, qt_app)
        choose(tab, "real.json")
        tab.redecode_button.click()
        job = tab._job.job
        watch = weakref.ref(
            tab._job, lambda _: freed_on.append(threading.current_thread()))
        with listener.accept() as child:
            child.recv()                  # it is running
            shiboken6.delete(tab)         # the window has gone ...
            del tab                       # ... and nothing here holds it
            gc.collect()
            child.send("answer")
        job.wait()
    deadline = time.monotonic() + 5.0
    while watch() is not None and time.monotonic() < deadline:
        qt_app.processEvents()
        gc.collect()

    assert watch() is None, "the re-decode was never let go of"
    assert freed_on == [threading.main_thread()], (
        f"freed on {freed_on[0].name if freed_on else 'no thread'}")
    assert raised == [], f"a thread raised: {raised[0].exc_value!r}"
    assert recordings_tab._RUNNING == set()
