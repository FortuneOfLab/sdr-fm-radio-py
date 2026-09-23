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


def test_the_redecode_button_is_there_and_does_nothing_yet(make_tab):
    tab = make_tab()
    assert tab.redecode_button.isEnabled() is False
    assert tab.redecode_button.toolTip()


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
