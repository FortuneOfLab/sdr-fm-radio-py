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
import math
import os
import stat as stat_flags
import wave
from dataclasses import dataclass
from datetime import datetime, timezone


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
# it; what could not be used is carried in ``Recording.problem`` and
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
    #: Why the sidecar could not be used, or ``""`` when it can be.
    problem: str

    @property
    def complete(self) -> bool:
        """True when every part the sidecar names is on disk.

        Never true of a row with a :attr:`problem`: a sidecar whose
        part list could not be believed is not a recording anything
        has all of.
        """
        return bool(self.parts) and not self.missing and not self.problem

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


def _a_number(value) -> float | None:
    """*value* as a finite float, or None if it is not one.

    JSON is allowed to carry ``NaN`` and ``Infinity`` and Python's
    decoder accepts both, so a sidecar can hand over either.  Neither
    is a frequency or a gain, and an infinity is also what ``int()``
    raises OverflowError on - so what would have been an exception out
    of a function that promises not to raise is simply "not a number".
    A bool is not one either: ``"channels": true`` is not two channels.

    ``float()`` raises an OverflowError of its own on a JSON integer
    too large to be one - four hundred digits of 9 is still valid
    JSON - which is the same answer under a different name.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _a_whole_number(value) -> int | None:
    """*value* as an int, or None if it is not one."""
    number = _a_number(value)
    return None if number is None else int(number)


def _a_time(value) -> datetime | None:
    """An ISO-8601 timestamp as :func:`now_iso` writes them, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _named_parts(meta: dict) -> tuple[tuple[str, ...], str]:
    """The part names *meta* lists, base first, and why they cannot be.

    ``parts`` is written when the session stops and is the whole
    story; ``file`` is written when it starts and names the base
    alone.  A session killed between the two has only ``file``, and
    one part of it is better than none - but only when there is no
    ``parts`` at all.  A ``parts`` that is present and wrong is not
    an absent one: falling back to ``file`` would answer with the
    first part of a rotated session as though it were all of it.

    The same reasoning covers an entry that is not a string.
    Dropping that entry and keeping the rest is the worst answer
    available: the names that remain are all present, so the
    recording would be called complete and its length measured, when
    what the sidecar says is that there was another part and nothing
    here can find it.

    The test is "a list of strings", not "a list of file names":
    ``""`` and a name with a NUL in it are strings, and they get as
    far as being looked for and not found.
    """
    if "parts" in meta:
        parts = meta["parts"]
        if not isinstance(parts, list):
            return (), "its parts are not a list"
        if not all(isinstance(p, str) for p in parts):
            return (), "lists a part that is not a string"
        return tuple(parts), ""
    base = meta.get("file")
    if isinstance(base, str):
        return (base,), ""
    return (), ""


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


def _which_file(path: str) -> tuple[bool, tuple | None]:
    """Whether a plain file is at *path*, and which file it is.

    Identity, not spelling.  Windows matches file names without
    regard to case and ignores a trailing dot, so ``a.wav``,
    ``A.wav`` and ``a.wav.`` are three names for one file - and
    measuring that file once per name would make a one-second
    recording three seconds long.  Device and inode are the same for
    all three, and for a hard link or a symlink to it as well.

    ``os.stat`` follows symlinks, so a part that is a symlink is
    measured as what it points at, wherever that is.  Deliberately:
    the rule that parts are looked for beside the sidecar is there so
    that *what the sidecar says* cannot send the reader elsewhere,
    not to overrule what the person whose disk it is has arranged.
    Moving a 4 GB capture to another drive and leaving a symlink
    behind should not lose the recording.

    Three answers.  ``(False, None)`` - nothing to measure: no file,
    or something that is not a plain file, such as a directory called
    ``a.wav`` or a path with a NUL in it.  ``(True, None)`` - a file
    is there but the filesystem does not number its files (inode 0),
    so it cannot be told from any other; the caller decides what that
    is worth.  ``(True, identity)`` otherwise.
    """
    try:
        found = os.stat(path)
    except (OSError, ValueError):
        return False, None
    if not stat_flags.S_ISREG(found.st_mode):
        return False, None
    if not found.st_ino:
        return True, None
    return True, (found.st_dev, found.st_ino)


def _look_for_the_parts(
    here: str, parts: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[float | None, ...], str]:
    """Find *parts* beside *here*: (missing, lengths, problem).

    Parts are looked for beside the sidecar, by base name.  The
    writer only ever puts base names in, and taking the base name
    rather than joining what is written means a sidecar carrying a
    path cannot send this looking somewhere else on the disk.

    Two names for one file is a problem rather than a measurement:
    the writer never repeats a part, so a sidecar that does cannot be
    believed about its parts at all.

    The walk finishes whatever it finds.  Returning at the first
    problem would leave ``missing`` saying that nothing is absent
    when a part before the problem was, and ``missing`` is read by
    whoever has to decide what can still be played.  The length is
    the only thing a problem costs, and the caller withholds that.
    """
    missing: list[str] = []
    lengths: list[float | None] = []
    spellings: set[str] = set()
    files: set[tuple] = set()
    unnumbered = 0
    problem = ""
    for name in parts:
        base = os.path.basename(name)
        spelling = os.path.normcase(base)
        if spelling in spellings and not problem:
            problem = f"names {base} as more than one part"
        spellings.add(spelling)
        beside = os.path.join(here, base)
        there, which = _which_file(beside)
        if not there:
            missing.append(name)
            continue
        if which is None:
            unnumbered += 1
        elif which in files:
            if not problem:
                problem = "names one file as more than one part"
        else:
            files.add(which)
        lengths.append(_seconds_of_wav(beside))

    # A filesystem that does not number its files cannot say whether
    # two names that got this far are two files or one, and the
    # spelling check above cannot either - it does not know which
    # names that filesystem folds together.  One part is safe; more
    # than one might be the same file counted twice, and a length
    # that might be double is worse than no length at all.
    if unnumbered and len(lengths) > 1:
        lengths = [None] * len(lengths)
    return tuple(missing), tuple(lengths), problem


def _unusable(path: str, problem: str) -> Recording:
    """A row for a sidecar that had nothing to say, saying why.

    Only for the ones that could not be read as a JSON object at
    all: there is no frequency in a file that would not parse.  A
    sidecar that parses and then says something unusable about its
    *parts* keeps the rest of what it says and carries the problem
    alongside it - :func:`read_sidecar` builds that row itself.
    """
    return Recording(
        sidecar=path, kind="", parts=(), missing=(),
        sample_rate_hz=None, center_freq_hz=None, gain_db=None,
        channels=None, started_at=None, stopped_at=None, dropped=None,
        audio_seconds=None, wall_seconds=None, problem=problem,
    )


def read_sidecar(path: str) -> Recording:
    """Read one ``.json`` sidecar and look for the audio it names.

    Never raises.  A file that is not there, is not JSON, or is JSON
    that is not an object has nothing to say and comes back as a
    :class:`Recording` with :attr:`~Recording.problem` set and
    everything else empty.

    A sidecar that parses as an object comes back with everything it
    does say, however few of its keys are present: a key this does
    not recognise is ignored, and an optional one it does not find
    reads as None.  If what that object says about its *parts*
    cannot be believed, the problem is on the row and the rest of the
    row stands - the frequency, the gain and the timestamps were read
    on their own and are no less true for it, and a row that lost its
    start time would fall out of the ordering and could not be found
    by when it was made.  Such a row is never
    :attr:`~Recording.complete` and never carries a measured length.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except OSError as e:
        return _unusable(path, f"could not be opened: {e}")
    except (ValueError, RecursionError) as e:
        # ValueError covers JSONDecodeError and UnicodeDecodeError,
        # and also the integer-digit limit CPython raises on a number
        # of more than 4300 digits.  RecursionError is what deeply
        # nested but otherwise valid JSON costs.  BaseException is
        # left alone on purpose: a KeyboardInterrupt is not something
        # to turn into a row.
        return _unusable(path, f"is not valid JSON: {e}")
    if not isinstance(meta, dict):
        return _unusable(
            path, f"is not a JSON object but a {type(meta).__name__}",
        )

    parts, problem = _named_parts(meta)
    missing: tuple[str, ...] = ()
    lengths: tuple[float | None, ...] = ()
    if not problem:
        missing, lengths, problem = _look_for_the_parts(
            os.path.dirname(path), parts,
        )

    # Only a complete set of readable headers adds up to the length of
    # the recording.  Summing what is left of a session whose other
    # parts have been deleted would report a fraction as though it
    # were the whole, which is worse than saying nothing and falling
    # back to the clock.
    if (not problem and parts and not missing
            and all(s is not None for s in lengths)):
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
        missing=missing,
        sample_rate_hz=_a_whole_number(meta.get("sample_rate_hz")),
        center_freq_hz=_a_number(meta.get("center_freq_hz")),
        gain_db=_a_number(meta.get("gain_db")),
        channels=_a_whole_number(meta.get("channels")),
        started_at=started,
        stopped_at=stopped,
        dropped=dropped,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        problem=problem,
    )


#: Ordering measures start times from here.  Only their order
#: matters, so any fixed point would do.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _seconds_from_epoch(when: datetime) -> float:
    """*when* as seconds from 1970, by arithmetic alone.

    Not ``datetime.timestamp()``: that one asks the platform, and on
    Windows it raises OSError for a naive timestamp outside the local
    clock's range - 1960, or 9999 - which would cost a whole scan its
    directory over one hand-edited sidecar.  Subtraction has no such
    range.

    A naive timestamp is read as UTC, which can put it up to a day
    from where whoever typed it meant.  For ordering rows that is
    nothing, and the recorder itself always writes the offset.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - _EPOCH).total_seconds()


def _newest_first(recording: Recording) -> tuple:
    """Sort key: by start time, latest first, undated ones last.

    Ordered by seconds rather than by the datetimes themselves,
    because a hand-edited sidecar can carry a naive timestamp next to
    an aware one and Python refuses to order those against each other.
    """
    started = recording.started_at
    if started is None:
        return (1, 0.0, recording.sidecar)
    return (0, -_seconds_from_epoch(started), recording.sidecar)


def scan_recordings(directory: str) -> list[Recording]:
    """Every ``.json`` sidecar in *directory*, newest recording first.

    Sidecars only: a WAV that no sidecar names is not a recording
    this knows anything about - not its frequency, not its gain, not
    when it was made - and is left where it is.

    A directory that does not exist is not an error; glob finds
    nothing in it and the answer is an empty list, which is what a
    fresh checkout should get.

    Glob itself can still raise, which is why it is guarded: a
    directory name with a NUL in it reaches os.scandir and comes back
    as ValueError, not as no matches.  That guard was here, taken out
    in review as unreachable because a directory that is merely
    absent does not raise, and put back when this one turned up.
    """
    try:
        found = glob.glob(os.path.join(directory, "*.json"))
    except (OSError, ValueError):
        return []
    return sorted((read_sidecar(p) for p in found), key=_newest_first)
