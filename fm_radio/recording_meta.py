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
"""

from __future__ import annotations

import json
import logging
import os
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
