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
"""Metadata sidecar files for recordings.

Every recording session (audio or IQ) gets a ``<base>.json`` sidecar
next to the WAV, written when the session starts and finalised when it
stops.  Since 4-GiB rotation can split one session across several
``.partNNN.wav`` files, the sidecar is the single place that ties a
session together: capture parameters (frequency, gain, rate), the full
part list, drop counts, and start/stop timestamps.

Sidecar writes happen on the CLI thread (start/stop), never on the
realtime path, and failures are logged but never abort a recording.

Reading them back is the other half: :func:`scan_recordings` turns a
directory of sidecars into :class:`Recording` rows for a browser to
list, saying for each one what the capture was and how much of its
audio is still on disk.  That half never raises - see the comment
above :class:`Recording`.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import wave
from dataclasses import dataclass
from datetime import datetime


def sidecar_path(base_wav_path: str) -> str:
    """``recordings/foo.wav`` -> ``recordings/foo.json``."""
    root, _ext = os.path.splitext(base_wav_path)
    return root + ".json"


def _json_default(obj):
    """Best-effort conversion for non-JSON-native metadata values.

    NumPy scalars (np.float32 gain values etc.) expose ``.item()``;
    anything else falls back to ``str`` so a caller-supplied metadata
    value can never make serialisation fail.
    """
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(obj)


def write_sidecar(
    base_wav_path: str,
    meta: dict,
    logger: logging.Logger | None = None,
) -> None:
    """Write (or overwrite) the sidecar for *base_wav_path*.

    Never raises: a metadata failure must not break a recording.  The
    JSON text is fully serialised *before* the file is opened, so a
    serialisation error can never leave a half-written sidecar behind.
    """
    path = sidecar_path(base_wav_path)
    try:
        text = json.dumps(
            meta, indent=2, ensure_ascii=False, default=_json_default,
        )
    except Exception as e:
        if logger is not None:
            logger.warning(
                "Could not serialise recording metadata for %s: %s", path, e,
            )
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    except OSError as e:
        if logger is not None:
            logger.warning("Could not write recording sidecar %s: %s", path, e)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def part_list(base_wav_path: str, part_index: int, make_part_path) -> list[str]:
    """Names of all files of a session: the base plus rotated parts."""
    parts = [os.path.basename(base_wav_path)]
    for i in range(1, part_index + 1):
        parts.append(os.path.basename(make_part_path(base_wav_path, i)))
    return parts


# --------------------------------------------------------------------
# Reading the sidecars back
# --------------------------------------------------------------------
#
# The writer above runs while a recording is being made; everything
# below runs long afterwards, over a directory of them.  A sidecar is
# the only place a recording's capture parameters survive - the WAV
# carries a rate and a channel count and nothing else - so browsing
# recordings means reading these files, not the audio.
#
# Nothing here raises for a sidecar it cannot make sense of.  A
# directory of them is read to put a list in front of someone, and one
# hand-edited or half-written file must not take the rest of them with
# it; what could not be read is carried in ``Recording.problem`` and
# can be shown as its own row.


#: The two spellings of the drop count: the IQ path writes the first,
#: the audio path the second, and neither writes both.
_DROPPED_KEYS = ("dropped_blocks", "dropped_chunks")


@dataclass(frozen=True)
class Recording:
    """One sidecar, and how much of what it names is still on disk.

    The fields up to :attr:`dropped` are what the sidecar says, or
    None where it does not say it: every key is optional here, because
    a sidecar written at the start of a session and never finalised
    has no ``stopped_at`` and no ``parts``, and an audio one carries a
    frequency and a gain only because the controller passes them in.

    The last three are worked out at read time.  :attr:`missing` is
    the parts with no file beside the sidecar - the common case in a
    directory that has been cleared of audio - and
    :attr:`audio_seconds` comes from the WAV headers, which is why it
    is filled in only when every part is there to be measured.
    """

    #: Path of the ``.json`` itself.
    sidecar: str
    #: ``"iq"``, ``"audio"``, or ``""`` when the sidecar does not say.
    kind: str
    #: Part file names, base first, as the sidecar spells them.
    parts: tuple[str, ...]
    #: Those of :attr:`parts` with no file beside the sidecar.
    missing: tuple[str, ...]
    sample_rate_hz: int | None
    center_freq_hz: float | None
    gain_db: float | None
    #: Audio recordings only; IQ sidecars do not carry it.
    channels: int | None
    started_at: datetime | None
    #: None for a session that never stopped cleanly.
    stopped_at: datetime | None
    #: Blocks (IQ) or chunks (audio) the recorder had to drop.
    dropped: int | None
    #: Summed WAV-header length of the parts, or None unless every one
    #: of them is present and readable.
    audio_seconds: float | None
    #: ``stopped_at - started_at``, or None if either is missing.
    wall_seconds: float | None
    #: Why the sidecar could not be read, or ``""`` when it was.
    problem: str

    @property
    def complete(self) -> bool:
        """True when every part the sidecar names is on disk."""
        return bool(self.parts) and not self.missing

    @property
    def duration_is_measured(self) -> bool:
        """True when :attr:`duration_s` came from the WAV headers."""
        return self.audio_seconds is not None

    @property
    def duration_s(self) -> float | None:
        """How long the recording is: the headers, else the clock.

        The headers are the better answer and the only exact one -
        the timestamps are written to the second, and a session that
        dropped blocks holds less audio than it spent wall-clock time
        - but they exist only while the WAV does.  For a sidecar
        whose audio is gone, the clock is all there is.
        """
        if self.audio_seconds is not None:
            return self.audio_seconds
        return self.wall_seconds


def _a_whole_number(value) -> int | None:
    """*value* as an int, or None if it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _a_number(value) -> float | None:
    """*value* as a float, or None if it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _a_time(value) -> datetime | None:
    """An ISO-8601 timestamp as :func:`now_iso` writes them, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _named_parts(meta: dict) -> tuple[str, ...]:
    """The part file names *meta* lists, base first.

    ``parts`` is written when the session stops and is the whole
    story; ``file`` is written when it starts and names the base
    alone.  A session killed between the two has only ``file``, and
    one part of it is better than none.
    """
    parts = meta.get("parts")
    if isinstance(parts, list):
        return tuple(p for p in parts if isinstance(p, str))
    base = meta.get("file")
    if isinstance(base, str):
        return (base,)
    return ()


def _seconds_of_wav(path: str) -> float | None:
    """The WAV header's length in seconds, or None if it cannot say.

    Only the header is read, so this costs one open per part however
    large the file is: a 4 GB IQ capture answers as fast as a short
    one.  A file that is not a WAV at all, or whose header is
    truncated, is not an error here - the part exists, its length is
    simply unknown.
    """
    try:
        with wave.open(path, "rb") as r:
            rate = r.getframerate()
            frames = r.getnframes()
    except Exception:
        return None
    if rate <= 0:
        return None
    return frames / float(rate)


def _unreadable(path: str, problem: str) -> Recording:
    """A row for a sidecar that could not be read, saying so."""
    return Recording(
        sidecar=path, kind="", parts=(), missing=(),
        sample_rate_hz=None, center_freq_hz=None, gain_db=None,
        channels=None, started_at=None, stopped_at=None, dropped=None,
        audio_seconds=None, wall_seconds=None, problem=problem,
    )


def read_sidecar(path: str) -> Recording:
    """Read one ``.json`` sidecar and look for the audio it names.

    Never raises.  A file that is not there, is not JSON, or is JSON
    that is not an object comes back as a :class:`Recording` with
    :attr:`~Recording.problem` set and everything else empty.  A
    sidecar that parses always comes back with ``problem == ""``,
    however few of its keys are present: a key this does not
    recognise is ignored, and a key it wants and does not find is
    None.

    Parts are looked for beside the sidecar, by base name.  The
    writer only ever puts base names in, and taking the base name
    rather than joining what is written means a sidecar carrying a
    path cannot send this looking somewhere else on the disk.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except OSError as e:
        return _unreadable(path, f"could not be opened: {e}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return _unreadable(path, f"is not valid JSON: {e}")
    if not isinstance(meta, dict):
        return _unreadable(
            path, f"is not a JSON object but a {type(meta).__name__}",
        )

    here = os.path.dirname(path)
    parts = _named_parts(meta)
    missing: list[str] = []
    lengths: list[float | None] = []
    for name in parts:
        beside = os.path.join(here, os.path.basename(name))
        if os.path.isfile(beside):
            lengths.append(_seconds_of_wav(beside))
        else:
            missing.append(name)

    # Only a complete set of readable headers adds up to the length of
    # the recording.  Summing what is left of a session whose other
    # parts have been deleted would report a fraction as though it
    # were the whole, which is worse than saying nothing and falling
    # back to the clock.
    if parts and not missing and all(s is not None for s in lengths):
        audio_seconds = float(sum(lengths))
    else:
        audio_seconds = None

    started = _a_time(meta.get("started_at"))
    stopped = _a_time(meta.get("stopped_at"))
    wall_seconds = None
    if started is not None and stopped is not None:
        try:
            wall_seconds = (stopped - started).total_seconds()
        except TypeError:
            # One of them carries a UTC offset and the other does not.
            wall_seconds = None

    kind = meta.get("type")
    dropped = None
    for key in _DROPPED_KEYS:
        if key in meta:
            dropped = _a_whole_number(meta[key])
            break

    return Recording(
        sidecar=path,
        kind=kind if isinstance(kind, str) else "",
        parts=parts,
        missing=tuple(missing),
        sample_rate_hz=_a_whole_number(meta.get("sample_rate_hz")),
        center_freq_hz=_a_number(meta.get("center_freq_hz")),
        gain_db=_a_number(meta.get("gain_db")),
        channels=_a_whole_number(meta.get("channels")),
        started_at=started,
        stopped_at=stopped,
        dropped=dropped,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        problem="",
    )


def _newest_first(recording: Recording) -> tuple:
    """Sort key: by start time, latest first, undated ones last.

    Compared as POSIX timestamps rather than as datetimes, because a
    hand-edited sidecar can carry a naive timestamp next to an aware
    one and Python refuses to order those against each other.
    """
    started = recording.started_at
    if started is None:
        return (1, 0.0, recording.sidecar)
    return (0, -started.timestamp(), recording.sidecar)


def scan_recordings(directory: str) -> list[Recording]:
    """Every ``.json`` sidecar in *directory*, newest recording first.

    Sidecars only: a WAV that no sidecar names is not a recording
    this knows anything about - not its frequency, not its gain, not
    when it was made - and is left where it is.

    A directory that does not exist is not an error; glob finds
    nothing in it and the answer is an empty list, which is what a
    fresh checkout should get.
    """
    found = glob.glob(os.path.join(directory, "*.json"))
    return sorted((read_sidecar(p) for p in found), key=_newest_first)
