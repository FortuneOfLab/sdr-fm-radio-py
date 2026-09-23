"""Metadata sidecar tests (backlog item B5), recording-path tests (B4),
and tests for reading the sidecars back (P5 PR-A)."""

from __future__ import annotations

import builtins
import json
import os
import re
import stat
import threading
import wave

import numpy as np
import pytest

import fm_radio.audio_output as ao_mod
import fm_radio.sdr_receiver as sr_mod
from fm_radio.recording_meta import (
    Recording, read_sidecar, scan_recordings, sidecar_path,
)


CHUNK = np.zeros(768 * 2, dtype=np.float32) + 0.25
IQ_BLOCK = (np.zeros(16384) + 0.1 + 0.05j).astype(np.complex64)


def _load_sidecar(base_wav):
    with open(sidecar_path(str(base_wav)), encoding="utf-8") as f:
        return json.load(f)


def test_audio_sidecar_written_and_finalised(audio_output, tmp_path):
    ao = audio_output
    base = tmp_path / "a.wav"
    ao.start_recording(
        str(base), metadata={"center_freq_hz": 91.6e6, "gain_db": 20.7},
    )
    meta = _load_sidecar(base)
    assert meta["type"] == "audio"
    assert meta["sample_rate_hz"] == 48000
    assert meta["center_freq_hz"] == 91.6e6
    assert meta["gain_db"] == 20.7
    assert "started_at" in meta
    assert "stopped_at" not in meta  # not finalised yet

    for _ in range(3):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    meta = _load_sidecar(base)
    assert "stopped_at" in meta
    assert meta["parts"] == ["a.wav"]
    assert meta["dropped_chunks"] == 0


def test_audio_sidecar_lists_rotated_parts(audio_output, tmp_path, monkeypatch):
    monkeypatch.setattr(ao_mod, "AUDIO_RECORD_ROTATE_THRESHOLD_BYTES", 50_000)
    ao = audio_output
    base = tmp_path / "rot.wav"
    ao.start_recording(str(base))
    for _ in range(50):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    meta = _load_sidecar(base)
    assert meta["parts"][0] == "rot.wav"
    assert len(meta["parts"]) >= 2
    assert meta["parts"][1] == "rot.part001.wav"
    # Every listed part must actually exist on disk.
    for name in meta["parts"]:
        assert (tmp_path / name).exists()


def test_iq_sidecar_written_and_finalised(sdr_receiver, tmp_path):
    recv = sdr_receiver
    base = tmp_path / "iq.wav"
    recv.start_iq_recording(str(base))
    meta = _load_sidecar(base)
    assert meta["type"] == "iq"
    assert meta["sample_rate_hz"] == 1024000
    assert meta["center_freq_hz"] == pytest.approx(80e6)
    assert "gain_db" in meta
    assert "started_at" in meta

    for _ in range(2):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
    recv.stop_iq_recording()

    meta = _load_sidecar(base)
    assert "stopped_at" in meta
    assert meta["parts"] == ["iq.wav"]
    assert meta["dropped_blocks"] == 0


def test_iq_sidecar_lists_rotated_parts(sdr_receiver, tmp_path, monkeypatch):
    monkeypatch.setattr(sr_mod, "IQ_RECORD_ROTATE_THRESHOLD_BYTES", 200_000)
    recv = sdr_receiver
    base = tmp_path / "rot.wav"
    recv.start_iq_recording(str(base))
    import time
    for _ in range(10):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
        time.sleep(0.01)
    recv.stop_iq_recording()

    meta = _load_sidecar(base)
    assert len(meta["parts"]) >= 2
    for name in meta["parts"]:
        assert (tmp_path / name).exists()


def test_numpy_metadata_does_not_break_recording(audio_output, tmp_path):
    # Codex repro (PR #13 review): np.float32 in metadata raised
    # TypeError from json.dump AFTER the WAV was opened and
    # recording=True, leaving a half-written sidecar.
    ao = audio_output
    base = tmp_path / "np.wav"
    ao.start_recording(
        str(base), metadata={"gain_db": np.float32(20.7)},
    )  # must not raise
    assert ao.recording
    meta = _load_sidecar(base)  # sidecar must be complete, valid JSON
    assert meta["gain_db"] == pytest.approx(20.7)
    ao.record(CHUNK.copy())
    ao.stop_recording()
    meta = _load_sidecar(base)
    assert "stopped_at" in meta


def test_unserialisable_metadata_skips_sidecar_but_records(audio_output,
                                                          tmp_path):
    class Hostile:
        def __str__(self):
            raise RuntimeError("no string for you")

    ao = audio_output
    base = tmp_path / "hostile.wav"
    ao.start_recording(str(base), metadata={"bad": Hostile()})  # must not raise
    assert ao.recording
    ao.record(CHUNK.copy())
    ao.stop_recording()  # must not raise either
    import wave as wave_mod
    with wave_mod.open(str(base), "rb") as r:
        assert r.getnframes() == 768  # the recording itself survived


def test_build_recording_path_uses_recordings_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from fm_radio.cli import build_recording_path
    from fm_radio.constants import RECORDINGS_DIR

    p_audio = build_recording_path(91.6, iq=False)
    p_iq = build_recording_path(80.0, iq=True)
    assert (tmp_path / RECORDINGS_DIR).is_dir()
    assert re.fullmatch(
        re.escape(RECORDINGS_DIR) + r"[\\/]\d{8}_\d{6}_91\.6MHz\.wav", p_audio,
    )
    assert re.fullmatch(
        re.escape(RECORDINGS_DIR) + r"[\\/]\d{8}_\d{6}_80\.0MHz_IQ\.wav", p_iq,
    )


# --------------------------------------------------------------------
# Reading the sidecars back (P5 PR-A)
# --------------------------------------------------------------------
#
# The numbers here are chosen so that the two sources of a duration
# cannot be confused for one another: every fixture's WAV headers add
# up to a fraction of a second while its timestamps are two minutes
# apart.  A reader that quietly used the clock where it promised the
# headers reads 120.0 where the test wants 0.3.

#: Two minutes apart, so a clock-derived duration is 120.0 s.
_STARTED = "2026-09-20T01:29:48+09:00"
_STOPPED = "2026-09-20T01:31:48+09:00"


def _write_wav(path, frames, rate=48000, channels=2):
    """A real WAV of *frames* frames, so its header can be read."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (frames * channels * 2))


def _write_json(path, meta):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _iq_meta(parts, **extra):
    meta = {
        "type": "iq",
        "file": parts[0],
        "sample_rate_hz": 1024000,
        "center_freq_hz": 91.6e6,
        "gain_db": 8.7,
        "started_at": _STARTED,
        "stopped_at": _STOPPED,
        "parts": list(parts),
        "dropped_blocks": 2,
    }
    meta.update(extra)
    return meta


def test_reader_gets_back_what_the_writer_put_in(audio_output, tmp_path):
    # Read what the recorder itself wrote, not a hand-made fixture:
    # the two halves of this module have to agree on the spelling of
    # every key, and only a round trip can say that they do.
    ao = audio_output
    base = tmp_path / "rt.wav"
    ao.start_recording(
        str(base), metadata={"center_freq_hz": 91.6e6, "gain_db": 20.7},
    )
    for _ in range(3):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    rec = read_sidecar(sidecar_path(str(base)))
    assert rec.problem == ""
    assert rec.kind == "audio"
    assert rec.sample_rate_hz == 48000
    assert rec.channels == 2
    assert rec.center_freq_hz == pytest.approx(91.6e6)
    assert rec.gain_db == pytest.approx(20.7)
    assert rec.dropped == 0
    assert rec.parts == ("rt.wav",)
    assert rec.missing == ()
    assert rec.complete is True
    # Three chunks of 768 frames went in at 48 kHz.
    assert rec.audio_seconds == pytest.approx(3 * 768 / 48000.0)
    assert rec.started_at is not None and rec.stopped_at is not None


def test_length_comes_from_the_wav_headers(tmp_path):
    parts = ["m.wav", "m.part001.wav"]
    _write_json(tmp_path / "m.json", _iq_meta(parts))
    _write_wav(tmp_path / parts[0], 4800)    # 0.1 s
    _write_wav(tmp_path / parts[1], 9600)    # 0.2 s

    rec = read_sidecar(str(tmp_path / "m.json"))
    assert rec.missing == ()
    assert rec.audio_seconds == pytest.approx(0.3)
    assert rec.wall_seconds == pytest.approx(120.0)
    # The headers win over the clock, and say so.
    assert rec.duration_s == pytest.approx(0.3)
    assert rec.duration_is_measured is True


def test_one_missing_part_falls_back_to_the_clock(tmp_path):
    parts = ["m.wav", "m.part001.wav"]
    _write_json(tmp_path / "m.json", _iq_meta(parts))
    _write_wav(tmp_path / parts[0], 4800)    # the second part is gone

    rec = read_sidecar(str(tmp_path / "m.json"))
    assert rec.missing == ("m.part001.wav",)
    assert rec.complete is False
    # 0.1 s of the session is still there, but a tenth of a second is
    # not the length of it: report nothing measured rather than a
    # fraction dressed up as the whole.
    assert rec.audio_seconds is None
    assert rec.duration_s == pytest.approx(120.0)
    assert rec.duration_is_measured is False


def test_no_audio_left_at_all(tmp_path):
    # The common case in the user's directory: 339 of 369 sidecars
    # outlived their WAVs.
    _write_json(tmp_path / "gone.json", _iq_meta(["gone.wav"]))

    rec = read_sidecar(str(tmp_path / "gone.json"))
    assert rec.problem == ""
    assert rec.parts == ("gone.wav",)
    assert rec.missing == ("gone.wav",)
    assert rec.complete is False
    assert rec.duration_s == pytest.approx(120.0)
    assert rec.duration_is_measured is False
    # Everything the sidecar knows still comes back.
    assert rec.center_freq_hz == pytest.approx(91.6e6)
    assert rec.sample_rate_hz == 1024000
    assert rec.dropped == 2


def test_a_part_that_is_not_a_readable_wav_is_present_but_unmeasured(tmp_path):
    _write_json(tmp_path / "junk.json", _iq_meta(["junk.wav"]))
    (tmp_path / "junk.wav").write_bytes(b"not a RIFF header at all")

    rec = read_sidecar(str(tmp_path / "junk.json"))
    assert rec.problem == ""
    assert rec.missing == ()      # the file is there
    assert rec.complete is True
    assert rec.audio_seconds is None   # but it cannot be measured
    assert rec.duration_s == pytest.approx(120.0)


def test_a_session_that_never_stopped(tmp_path):
    # Written at the start and never finalised: no parts, no stop.
    _write_json(tmp_path / "half.json", {
        "type": "iq",
        "file": "half.wav",
        "sample_rate_hz": 1024000,
        "center_freq_hz": 80.0e6,
        "gain_db": 36.4,
        "started_at": _STARTED,
    })
    _write_wav(tmp_path / "half.wav", 24000, rate=48000)   # 0.5 s

    rec = read_sidecar(str(tmp_path / "half.json"))
    assert rec.problem == ""
    assert rec.parts == ("half.wav",)     # taken from "file"
    assert rec.stopped_at is None
    assert rec.wall_seconds is None
    assert rec.dropped is None
    # The clock cannot answer, but the header can.
    assert rec.duration_s == pytest.approx(0.5)
    assert rec.duration_is_measured is True


def test_parts_are_looked_for_beside_the_sidecar(tmp_path):
    # A sidecar naming a path must not send the reader off to it.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_json(tmp_path / "p.json", _iq_meta(["elsewhere/p.wav"]))
    _write_wav(elsewhere / "p.wav", 48000)     # 1.0 s, but not beside
    rec = read_sidecar(str(tmp_path / "p.json"))
    assert rec.missing == ("elsewhere/p.wav",)
    assert rec.audio_seconds is None

    _write_wav(tmp_path / "p.wav", 4800)       # 0.1 s, beside it
    rec = read_sidecar(str(tmp_path / "p.json"))
    assert rec.missing == ()
    assert rec.audio_seconds == pytest.approx(0.1)


def test_the_two_spellings_of_the_drop_count(tmp_path):
    _write_json(tmp_path / "i.json", _iq_meta(["i.wav"]))
    _write_json(tmp_path / "a.json", {
        "type": "audio", "file": "a.wav", "parts": ["a.wav"],
        "sample_rate_hz": 48000, "channels": 2,
        "started_at": _STARTED, "stopped_at": _STOPPED,
        "dropped_chunks": 7,
    })
    assert read_sidecar(str(tmp_path / "i.json")).dropped == 2
    assert read_sidecar(str(tmp_path / "a.json")).dropped == 7


def test_a_sidecar_that_is_not_json_says_so_and_keeps_its_row(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    _write_json(tmp_path / "ok.json", _iq_meta(["ok.wav"]))

    broken = read_sidecar(str(tmp_path / "broken.json"))
    assert "JSON" in broken.problem
    assert broken.parts == () and broken.duration_s is None

    a_list = read_sidecar(str(tmp_path / "list.json"))
    assert "object" in a_list.problem

    missing = read_sidecar(str(tmp_path / "nothing-here.json"))
    assert missing.problem != ""

    # The good one is still in the scan, and the bad ones are rows
    # rather than an exception that loses the directory.
    rows = scan_recordings(str(tmp_path))
    assert len(rows) == 3
    assert sum(1 for r in rows if r.problem) == 2
    good = [r for r in rows if not r.problem]
    assert len(good) == 1 and good[0].center_freq_hz == pytest.approx(91.6e6)


def test_keys_that_are_missing_or_the_wrong_type_are_not_an_error(tmp_path):
    _write_json(tmp_path / "odd.json", {
        "type": 17,                      # not a string
        "sample_rate_hz": "forty-eight",  # not a number
        "center_freq_hz": None,
        "started_at": "not a timestamp",
        "stopped_at": _STOPPED,
        "parts": ["a.wav", "b.wav"],
        "something_new": {"nested": True},
    })
    rec = read_sidecar(str(tmp_path / "odd.json"))
    assert rec.problem == ""
    assert rec.kind == ""
    assert rec.sample_rate_hz is None
    assert rec.center_freq_hz is None
    assert rec.started_at is None
    assert rec.wall_seconds is None
    assert rec.parts == ("a.wav", "b.wav")


def test_numeric_fields_survive_the_shapes_json_allows(tmp_path):
    _write_json(tmp_path / "n.json", _iq_meta(
        ["n.wav"], sample_rate_hz=1024000.0, gain_db="8.7", channels=True,
    ))
    rec = read_sidecar(str(tmp_path / "n.json"))
    assert rec.sample_rate_hz == 1024000
    assert rec.gain_db == pytest.approx(8.7)
    assert rec.channels is None      # a bool is not a channel count


def test_scan_is_newest_first_and_undated_last(tmp_path):
    def at(name, started):
        meta = _iq_meta([name + ".wav"])
        if started is None:
            del meta["started_at"]
        else:
            meta["started_at"] = started
        _write_json(tmp_path / (name + ".json"), meta)

    at("old", "2026-09-19T10:00:00+09:00")
    at("new", "2026-09-21T10:00:00+09:00")
    at("mid", "2026-09-20T10:00:00+09:00")
    # A naive timestamp next to the aware ones: ordering must not
    # raise, and it must not push the dated ones around.
    at("naive", "2026-09-20T12:00:00")
    at("undated", None)

    rows = scan_recordings(str(tmp_path))
    names = [os.path.basename(r.sidecar) for r in rows]
    assert names[0] == "new.json"
    assert names[-1] == "undated.json"
    assert names.index("mid.json") < names.index("old.json")
    assert set(names) == {
        "new.json", "mid.json", "naive.json", "old.json", "undated.json",
    }


def test_scan_ignores_wavs_that_no_sidecar_names(tmp_path):
    _write_json(tmp_path / "kept.json", _iq_meta(["kept.wav"]))
    _write_wav(tmp_path / "kept.wav", 4800)
    _write_wav(tmp_path / "orphan.wav", 48000)
    _write_wav(tmp_path / "orphan.part001.wav", 48000)

    rows = scan_recordings(str(tmp_path))
    assert len(rows) == 1
    assert rows[0].parts == ("kept.wav",)


def test_scan_of_a_directory_that_is_not_there(tmp_path):
    assert scan_recordings(str(tmp_path / "no-such-dir")) == []


def test_recordings_are_hashable_and_frozen(tmp_path):
    _write_json(tmp_path / "f.json", _iq_meta(["f.wav"]))
    rec = read_sidecar(str(tmp_path / "f.json"))
    assert isinstance(rec, Recording)
    assert len({rec, read_sidecar(str(tmp_path / "f.json"))}) == 1
    with pytest.raises(Exception):
        rec.kind = "audio"


def test_the_numbers_json_allows_that_are_not_numbers(tmp_path):
    # json.loads accepts NaN and Infinity, so a sidecar can hand over
    # either.  int(inf) raises OverflowError, which a reader that
    # promises never to raise cannot afford to let out.
    (tmp_path / "wild.json").write_text(
        '{"type": "iq", "file": "w.wav",'
        ' "sample_rate_hz": Infinity, "channels": -Infinity,'
        ' "center_freq_hz": NaN, "gain_db": Infinity}',
        encoding="utf-8",
    )
    rec = read_sidecar(str(tmp_path / "wild.json"))
    assert rec.problem == ""
    assert rec.kind == "iq" and rec.parts == ("w.wav",)
    assert rec.sample_rate_hz is None
    assert rec.channels is None
    assert rec.center_freq_hz is None
    assert rec.gain_db is None
    assert len(scan_recordings(str(tmp_path))) == 1


def test_start_times_the_platform_cannot_turn_into_posix_seconds(tmp_path):
    # datetime.timestamp() raises OSError on Windows for a NAIVE
    # timestamp outside the local clock's range.  One hand-edited
    # sidecar dated 1960 must not cost the scan its directory.
    for name, started in (
        ("ancient", "1960-01-01T00:00:00"),
        ("distant", "9999-12-31T23:59:59"),
        ("normal", "2026-09-20T01:29:48+09:00"),
    ):
        meta = _iq_meta([name + ".wav"])
        meta["started_at"] = started
        _write_json(tmp_path / (name + ".json"), meta)

    rows = scan_recordings(str(tmp_path))
    assert [os.path.basename(r.sidecar) for r in rows] == [
        "distant.json", "normal.json", "ancient.json",
    ]
    assert all(r.problem == "" for r in rows)
    assert rows[-1].started_at.year == 1960


# --- Round 2: what a hand-edited sidecar can still do (PR #70 review) ---


def test_numbers_too_large_for_a_float(tmp_path):
    # Valid JSON, and float() raises OverflowError on it: an integer
    # of 400 digits is larger than a float can be.
    huge = "9" * 400
    (tmp_path / "huge.json").write_text(
        '{"type": "iq", "file": "h.wav", "sample_rate_hz": %s,'
        ' "center_freq_hz": %s, "gain_db": -%s, "channels": %s,'
        ' "dropped_blocks": %s}' % (huge, huge, huge, huge, huge),
        encoding="utf-8",
    )
    rec = read_sidecar(str(tmp_path / "huge.json"))
    assert rec.problem == ""
    assert rec.parts == ("h.wav",)
    assert rec.sample_rate_hz is None
    assert rec.center_freq_hz is None
    assert rec.gain_db is None
    assert rec.channels is None
    assert rec.dropped is None


def test_json_that_parses_but_not_without_raising(tmp_path):
    # Two ways valid JSON costs an exception that is not a
    # JSONDecodeError: recursion, and CPython's 4300-digit limit on
    # turning a number into an int.
    (tmp_path / "deep.json").write_text(
        '{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}", encoding="utf-8")
    (tmp_path / "digits.json").write_text(
        '{"gain_db":' + "1" * 5000 + "}", encoding="utf-8")
    _write_json(tmp_path / "ok.json", _iq_meta(["ok.wav"]))

    deep = read_sidecar(str(tmp_path / "deep.json"))
    assert "recursion" in deep.problem
    digits = read_sidecar(str(tmp_path / "digits.json"))
    assert "4300" in digits.problem

    rows = scan_recordings(str(tmp_path))
    assert len(rows) == 3
    good = [r for r in rows if not r.problem]
    assert len(good) == 1
    assert good[0].center_freq_hz == pytest.approx(91.6e6)


def test_a_parts_list_holding_something_that_is_not_a_name(tmp_path):
    # Dropping the odd entry and keeping the rest is the trap: the
    # names that remain are all present, so the recording would be
    # called complete and measured at a second, when the sidecar says
    # there was another part.
    _write_json(tmp_path / "mixed.json", _iq_meta(["a.wav", 5]))
    _write_wav(tmp_path / "a.wav", 48000)    # 1.0 s, and present

    rec = read_sidecar(str(tmp_path / "mixed.json"))
    assert rec.problem == "lists a part that is not a string"
    assert rec.parts == ()
    assert rec.complete is False
    assert rec.audio_seconds is None
    # The row keeps what it could read on its own, so it can still be
    # found by when it was made and what it was tuned to.
    assert rec.center_freq_hz == pytest.approx(91.6e6)
    assert rec.started_at is not None
    assert rec.duration_s == pytest.approx(120.0)
    assert rec.duration_is_measured is False


def test_two_part_names_that_are_one_file(tmp_path):
    # Both resolve to a.wav beside the sidecar, so a one-second
    # recording would measure two seconds long.
    _write_json(tmp_path / "dup.json", _iq_meta(["x/a.wav", "y/a.wav"]))
    _write_wav(tmp_path / "a.wav", 48000)    # 1.0 s

    rec = read_sidecar(str(tmp_path / "dup.json"))
    assert rec.problem == "names a.wav as more than one part"
    assert rec.audio_seconds is None
    assert rec.complete is False


# --- Round 3: parts that cannot be believed (PR #70, second review) ---


def test_a_parts_that_is_not_a_list_does_not_fall_back_to_file(tmp_path):
    # "parts" present and wrong is not "parts" absent.  Falling back
    # to "file" would answer with the base of a rotated session as
    # though it were all of it: a.wav is there, so the recording
    # would come back complete and one second long.
    _write_json(tmp_path / "bad.json", {
        "type": "iq", "file": "a.wav", "parts": 5,
        "center_freq_hz": 91.6e6, "gain_db": 8.7,
        "started_at": _STARTED, "stopped_at": _STOPPED,
    })
    _write_wav(tmp_path / "a.wav", 48000)      # 1.0 s, and present

    rec = read_sidecar(str(tmp_path / "bad.json"))
    assert rec.problem == "its parts are not a list"
    assert rec.parts == ()
    assert rec.complete is False
    assert rec.audio_seconds is None
    # ...and everything read on its own still stands, so the row can
    # be found by when it was made and what it was tuned to.
    assert rec.center_freq_hz == pytest.approx(91.6e6)
    assert rec.gain_db == pytest.approx(8.7)
    assert rec.kind == "iq"
    assert rec.started_at is not None
    assert rec.duration_s == pytest.approx(120.0)
    assert rec.duration_is_measured is False


def test_the_same_part_named_twice(tmp_path):
    _write_json(tmp_path / "twice.json", _iq_meta(["a.wav", "a.wav"]))
    _write_wav(tmp_path / "a.wav", 48000)      # 1.0 s

    rec = read_sidecar(str(tmp_path / "twice.json"))
    assert rec.problem == "names a.wav as more than one part"
    assert rec.audio_seconds is None
    # Both names are present, so only the problem keeps this from
    # calling itself complete.
    assert rec.missing == ()
    assert rec.complete is False


def test_two_spellings_of_one_name(tmp_path):
    # Whether a.wav and A.wav are one file is the filesystem's
    # answer, not normcase's: Windows folds the case and says so
    # through normcase, a case-insensitive APFS folds it and does
    # not, and ext4 does not fold it at all.  So ask the filesystem
    # what it did, and only then say which of the two checks should
    # have caught it.
    _write_json(tmp_path / "case.json", _iq_meta(["a.wav", "A.wav"]))
    _write_wav(tmp_path / "a.wav", 48000)      # 1.0 s

    rec = read_sidecar(str(tmp_path / "case.json"))
    upper = tmp_path / "A.wav"
    if not upper.exists():
        # Two names, and nothing behind the second one.
        assert rec.problem == ""
        assert rec.missing == ("A.wav",)
        assert rec.audio_seconds is None       # one part is missing
        return

    assert os.path.samefile(str(tmp_path / "a.wav"), str(upper))
    assert rec.audio_seconds is None           # NOT 2.0
    if os.path.normcase("A.wav") == os.path.normcase("a.wav"):
        assert rec.problem == "names A.wav as more than one part"
    else:
        # The spelling check could not know; identity did.
        assert rec.problem == "names one file as more than one part"


def test_two_names_for_one_file(tmp_path):
    # Spelled differently, folded differently, and still one file:
    # a hard link has its own name and the same inode.  Caught by
    # identity, which is what a trailing dot on Windows needs too.
    _write_wav(tmp_path / "a.wav", 48000)      # 1.0 s
    try:
        os.link(str(tmp_path / "a.wav"), str(tmp_path / "b.wav"))
    except (OSError, NotImplementedError, AttributeError) as e:
        pytest.skip(f"this filesystem has no hard links: {e}")
    _write_json(tmp_path / "linked.json", _iq_meta(["a.wav", "b.wav"]))

    rec = read_sidecar(str(tmp_path / "linked.json"))
    assert rec.problem == "names one file as more than one part"
    assert rec.audio_seconds is None
    assert rec.complete is False


def test_a_part_name_that_is_a_string_but_not_a_file_name(tmp_path):
    # The test is "a list of strings", not "a list of file names":
    # these get as far as being looked for and not found, which is a
    # missing part, not a broken sidecar.
    for name, part in (("empty", ""), ("nul", "bad\x00.wav")):
        _write_json(tmp_path / (name + ".json"), _iq_meta([part]))
        rec = read_sidecar(str(tmp_path / (name + ".json")))
        assert rec.problem == ""
        assert rec.parts == (part,)
        assert rec.missing == (part,)
        assert rec.complete is False


def test_scan_of_a_directory_name_the_platform_will_not_take(tmp_path):
    # glob reaches os.scandir, which raises ValueError on a NUL
    # rather than returning no matches.
    assert scan_recordings("\x00") == []
    assert scan_recordings(str(tmp_path) + "\x00") == []


# --- Round 4: what the filesystem, not the sidecar, can do ------------


def _read_with_unnumbered_files(path):
    """read_sidecar as a filesystem that reports inode 0 would see it.

    os.fstat, not os.stat: the identity of a part is taken from the
    handle it is measured through, so zeroing what os.stat says would
    leave the real inode arriving by the other route and the test
    passing for no reason.  Patched around the call alone, so that
    nothing else in the run has to live without fstat telling the
    truth.
    """
    real_fstat = os.fstat

    def without_inodes(fd, **kw):
        fields = list(real_fstat(fd, **kw))
        fields[1] = 0                     # st_ino
        return os.stat_result(fields)

    os.fstat = without_inodes
    try:
        return read_sidecar(path)
    finally:
        os.fstat = real_fstat


def test_a_filesystem_that_does_not_number_its_files(tmp_path):
    # Two names that got as far as being opened cannot be told from
    # one name twice without inodes, and the spelling check cannot
    # stand in: it does not know which names such a filesystem folds
    # together.  Here a.wav and b.wav really are two files - the
    # point is that nothing available can prove it, so the length is
    # withheld rather than guessed.
    _write_wav(tmp_path / "a.wav", 48000)          # 1.0 s
    _write_wav(tmp_path / "b.wav", 96000)          # 2.0 s
    _write_json(tmp_path / "two.json", _iq_meta(["a.wav", "b.wav"]))

    rec = _read_with_unnumbered_files(str(tmp_path / "two.json"))
    assert rec.problem == ""                       # the files are there
    assert rec.missing == ()
    assert rec.complete is True
    assert rec.audio_seconds is None               # NOT 3.0
    assert rec.duration_s == pytest.approx(120.0)  # the clock instead
    assert rec.duration_is_measured is False

    # One part cannot be itself twice, so it is still measured.
    _write_json(tmp_path / "one.json", _iq_meta(["a.wav"]))
    rec = _read_with_unnumbered_files(str(tmp_path / "one.json"))
    assert rec.audio_seconds == pytest.approx(1.0)
    assert rec.duration_is_measured is True


def _hook_os_open(after_open=None, instead=None):
    """Put a hook on os.open for the duration of a with-block.

    *after_open(path, fd)* runs after a real open; *instead(path)*
    runs in its place and its result or exception is the open's.
    Either returns None to leave a given path alone.
    """
    class _Hooked:
        def __enter__(self):
            self.real = os.open

            def hooked(path, flags, *a, **kw):
                if instead is not None:
                    got = instead(str(path))
                    if got is not None:
                        return got
                fd = self.real(path, flags, *a, **kw)
                if after_open is not None:
                    after_open(str(path), fd)
                return fd

            os.open = hooked
            return self

        def __exit__(self, *exc):
            os.open = self.real
            return False

    return _Hooked()


def test_a_part_identified_and_measured_through_one_handle(tmp_path):
    # Open a.wav, and then have the NAME a.wav become another name
    # for the 10-second b.wav.  The descriptor still holds the
    # 1-second file it opened, so that is what is identified and what
    # is measured: 1 + 10 = 11 s.  A reader that went back to the name
    # for either answer would read b.wav twice - 20 s - or call the
    # two parts one file.
    _write_wav(tmp_path / "a.wav", 48000)          # 1.0 s
    _write_wav(tmp_path / "b.wav", 480000)         # 10.0 s
    _write_json(tmp_path / "swap.json", _iq_meta(["a.wav", "b.wav"]))

    swapped = []

    def swap_after_opening_a(path, fd):
        if not swapped and path.endswith("a.wav"):
            swapped.append(True)
            try:
                os.replace(str(tmp_path / "a.wav"), str(tmp_path / "old.wav"))
                os.link(str(tmp_path / "b.wav"), str(tmp_path / "a.wav"))
            except (OSError, NotImplementedError, AttributeError):
                swapped.append("no")

    with _hook_os_open(after_open=swap_after_opening_a):
        rec = read_sidecar(str(tmp_path / "swap.json"))
    if "no" in swapped:
        # Windows will not rename a file another handle holds open, so
        # the swap this test stages cannot happen there at all: the
        # race it guards against is one only POSIX can run.
        pytest.skip("an open file cannot be renamed here")

    assert swapped, "the open hook never fired"
    assert rec.problem == ""
    assert rec.audio_seconds == pytest.approx(11.0)    # NOT 20.0


def test_a_part_that_is_a_symlink_out_of_the_directory(tmp_path):
    # Followed on purpose: the "beside the sidecar" rule is about
    # what the sidecar says, not about what the owner of the disk has
    # arranged.  Moving a 4 GB capture off this drive and leaving a
    # symlink must not lose the recording.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_wav(elsewhere / "far.wav", 48000)       # 1.0 s, not beside
    try:
        os.symlink(str(elsewhere / "far.wav"), str(tmp_path / "here.wav"))
    except (OSError, NotImplementedError, AttributeError) as e:
        pytest.skip(f"symlinks are not available here: {e}")
    _write_json(tmp_path / "link.json", _iq_meta(["here.wav"]))

    rec = read_sidecar(str(tmp_path / "link.json"))
    assert rec.problem == ""
    assert rec.missing == ()
    assert rec.complete is True
    assert rec.audio_seconds == pytest.approx(1.0)


def test_missing_parts_are_collected_even_when_there_is_a_problem(tmp_path):
    # The walk used to stop at the problem, and missing came back
    # empty - saying nothing was absent when gone.wav was.
    _write_wav(tmp_path / "a.wav", 48000)
    try:
        os.link(str(tmp_path / "a.wav"), str(tmp_path / "b.wav"))
    except (OSError, NotImplementedError, AttributeError) as e:
        pytest.skip(f"this filesystem has no hard links: {e}")
    _write_json(tmp_path / "both.json",
                _iq_meta(["gone.wav", "a.wav", "b.wav"]))

    rec = read_sidecar(str(tmp_path / "both.json"))
    assert rec.problem == "names one file as more than one part"
    assert rec.missing == ("gone.wav",)
    assert rec.audio_seconds is None
    assert rec.complete is False


def test_a_part_that_is_gone_by_the_time_it_is_opened(tmp_path):
    # Two ways for a part to vanish around the open.  Gone before it:
    # the open says FileNotFoundError.  Refused, and gone by the time
    # the path is asked what is there: the second look finds nothing.
    # Both are missing, not "there, unidentified".
    for name, refuse in (("gone", False), ("refused", True)):
        _write_wav(tmp_path / (name + ".wav"), 48000)
        _write_json(tmp_path / (name + ".json"), _iq_meta([name + ".wav"]))
        target = str(tmp_path / (name + ".wav"))

        def vanish(path, target=target, refuse=refuse):
            if path != target:
                return None
            os.remove(target)
            if refuse:
                raise PermissionError(13, "refused, then removed")
            return None                  # and the real open finds nothing

        with _hook_os_open(instead=vanish):
            rec = read_sidecar(str(tmp_path / (name + ".json")))
        assert rec.missing == (name + ".wav",), name
        assert rec.complete is False, name
        assert rec.audio_seconds is None, name


def test_a_close_that_fails_is_not_an_error(tmp_path):
    # Some network filesystems fail a close.  The descriptor was only
    # read from, so there is nothing to lose - and the answer was
    # already worked out, so letting the exception out would throw a
    # good measurement away and break the promise not to raise.
    _write_wav(tmp_path / "b.wav", 48000)      # 1.0 s
    _write_json(tmp_path / "b.json", _iq_meta(["b.wav"]))

    ours = []

    def remember(path, fd):
        if path.endswith("b.wav"):
            ours.append(fd)

    real_close = os.close
    refused = []

    def close_badly(fd):
        real_close(fd)
        if fd in ours:
            refused.append(fd)
            raise OSError(5, "the share went away")

    os.close = close_badly
    try:
        with _hook_os_open(after_open=remember):
            rec = read_sidecar(str(tmp_path / "b.json"))   # must not raise
    finally:
        os.close = real_close

    # The failure has to have happened for this to mean anything.
    assert refused == ours and len(ours) == 1
    assert rec.problem == ""
    assert rec.missing == ()
    assert rec.audio_seconds == pytest.approx(1.0)


def test_a_part_that_is_a_directory(tmp_path):
    # Refused on one platform, opened on the other: Windows's os.open
    # raises PermissionError on a directory and Linux's opens it, and
    # fstat then says what it opened.  Two different routes to one
    # answer, and this test does not care which one was taken.
    (tmp_path / "a.wav").mkdir()
    _write_json(tmp_path / "one.json", _iq_meta(["a.wav"]))

    rec = read_sidecar(str(tmp_path / "one.json"))
    assert rec.problem == ""
    assert rec.missing == ("a.wav",)
    assert rec.complete is False
    assert rec.audio_seconds is None


def test_a_part_that_is_there_and_will_not_open(tmp_path):
    # The other side of the same fork: still a plain file, just shut.
    # Present, so not missing - and unidentified, so not measured.
    _write_wav(tmp_path / "shut.wav", 48000)
    _write_json(tmp_path / "shut.json", _iq_meta(["shut.wav"]))
    target = str(tmp_path / "shut.wav")

    def refuse(path):
        if path == target:
            raise PermissionError(13, "the file is shut")
        return None

    with _hook_os_open(instead=refuse):
        rec = read_sidecar(str(tmp_path / "shut.json"))

    assert rec.problem == ""
    assert rec.missing == ()
    assert rec.complete is True
    assert rec.audio_seconds is None
    assert rec.duration_s == pytest.approx(120.0)   # the clock
    assert rec.duration_is_measured is False


def test_a_handle_that_turns_out_not_to_be_a_plain_file(tmp_path):
    # The handle is what gets measured, so the handle is what has to
    # be a plain file - asked of the open file, not of the name.
    _write_wav(tmp_path / "a.wav", 48000)
    _write_json(tmp_path / "one.json", _iq_meta(["a.wav"]))

    real_fstat = os.fstat

    def fstat_says_directory(fd, **kw):
        fields = list(real_fstat(fd, **kw))
        fields[0] = stat.S_IFDIR | 0o755          # st_mode
        return os.stat_result(fields)

    os.fstat = fstat_says_directory
    try:
        rec = read_sidecar(str(tmp_path / "one.json"))
    finally:
        os.fstat = real_fstat

    assert rec.missing == ("a.wav",)
    assert rec.complete is False
    assert rec.audio_seconds is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
def test_a_named_pipe_does_not_stop_the_scan(tmp_path):
    # A named pipe opened for reading waits for a writer.  Without
    # O_NONBLOCK the scan would never come back, so the read runs on
    # a thread and the join is a watchdog, not a synchronisation: on
    # the passing path the thread has finished before the join is
    # reached, and the timeout is only how long a regression is given
    # before it is called one.  A thread still stuck is released by
    # opening the write end, so that it does not outlive the test.
    fifo = str(tmp_path / "pipe.wav")
    os.mkfifo(fifo)
    _write_json(tmp_path / "pipe.json", _iq_meta(["pipe.wav"]))

    got = []
    t = threading.Thread(
        target=lambda: got.append(read_sidecar(str(tmp_path / "pipe.json"))),
        daemon=True)
    t.start()
    t.join(10.0)
    if t.is_alive():
        release = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        t.join(10.0)
        os.close(release)
        pytest.fail("read_sidecar blocked on a named pipe")

    rec = got[0]
    assert rec.problem == ""
    assert rec.missing == ("pipe.wav",)
    assert rec.complete is False
