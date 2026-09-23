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
"""Decoding a recorded IQ capture again, away from the live receiver.

What ``quality_selftest --iq-wav FILE --duration SECONDS`` measures, with
every other option at its default, worked out in a process of its own
and handed back as an :class:`~fm_radio.quality_selftest.IqMeasurement`.

**A process, not a thread.**  The demodulator is Python between its
numpy calls, and a second one running in the receiver's process takes
the interpreter lock from the live one.  Measured on a 30 s window of
``20260722_014211_82.5MHz_IQ.wav`` against a stand-in live loop (one
16384-sample block every 16 ms): the median block took 7.3-7.4 ms with
nothing else running, 11.0-11.1 ms with the re-decode on a thread and
7.6-7.7 ms with it in a child process; on the thread 98-99 % of blocks
finished after they were due and the backlog reached 68-188 ms, against
14-15 ms with nothing running (two runs of each).  A child process costs
about 5 s before it can answer - 4.8-5.6 s for one that does nothing,
22-23 s in all for that 30 s window - and can be stopped by ending it.

**The window is the start of the first part.**  A recording is split
into parts only when one reaches ``IQ_RECORD_ROTATE_THRESHOLD_BYTES``,
about 16 minutes at 1.024 Msps, so a window of at most
:data:`WINDOW_MAX_S` from the start never reaches a second part.  The
window is normalised to its own peak, as the command line does: the
same seconds of a recording measure the same whichever tool asks, but a
longer window of the same recording can be scaled differently.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import os
import threading
from typing import Callable

from scipy.io import wavfile

from fm_radio import quality_selftest as qs
from fm_radio.constants import SDR_SAMPLE_RATE
from fm_radio.recording_meta import Recording

#: The window the tab offers first, in seconds.
WINDOW_DEFAULT_S = 30
#: The longest window it accepts: about a minute of decoding.  Whole
#: files are out of reach - 1.024 Msps of complex64 is 8.2 MB a second.
WINDOW_MAX_S = 120
#: The shortest.  Above the command line's warmup (0.8 s), which it
#: refuses to be at or under.
WINDOW_MIN_S = 1

#: What :meth:`Job.cancel` reports.
CANCELLED = "Cancelled."


def why_not(rec: Recording) -> str:
    """Why *rec* cannot be re-decoded, or ``""`` when it can.

    In the order the questions settle it: only an IQ capture has
    anything to demodulate; only a complete one has all its parts; only
    one whose headers measured some audio has a window to take; and
    only the standard rate is what the demodulator was measured at.
    """
    if rec.kind != "iq":
        return "Only IQ recordings can be re-decoded."
    if not rec.complete:
        return "Part of this recording is not there."
    if rec.audio_seconds is None:
        return "How much this recording holds could not be measured."
    if rec.audio_seconds <= 0:
        return "This recording holds no samples."
    if rec.sample_rate_hz is None:
        return "The sidecar does not say what rate it was recorded at."
    if rec.sample_rate_hz != int(SDR_SAMPLE_RATE):
        return (f"Recorded at {rec.sample_rate_hz / 1e3:g} kHz; only "
                f"{SDR_SAMPLE_RATE / 1e3:g} kHz IQ can be re-decoded yet.")
    return ""


def first_part_path(rec: Recording) -> str:
    """Where the first part of *rec* is.

    Beside the sidecar, under the last component of the name it gives,
    which is where :func:`fm_radio.recording_meta.scan_recordings`
    looked for it when it called the recording complete.
    """
    return os.path.join(os.path.dirname(rec.sidecar),
                        os.path.basename(rec.parts[0]))


def decode_file(wav_path: str, window_s: float):
    """Measure the first *window_s* seconds of an IQ WAV.

    The command line's ``--iq-wav`` path with nothing but ``--duration``
    given: the same loader, the demodulator with none of its settings
    overridden, and the warmup and noise band the command line defaults
    to, read from its own parser so that the two cannot drift apart.

    The file's own rate is checked first.  The command line resamples
    whatever it is given to the standard rate; a file whose sidecar
    says 1.024 Msps and whose header says otherwise is not what the row
    in the list claims it is, and a measurement of it would be filed
    under the wrong recording.
    """
    fs, _ = wavfile.read(wav_path, mmap=True)
    if int(fs) != int(SDR_SAMPLE_RATE):
        raise ValueError(
            f"{os.path.basename(wav_path)} is {fs / 1e3:g} kHz, "
            f"not {SDR_SAMPLE_RATE / 1e3:g} kHz")
    defaults = qs._parser()
    iq = qs._load_iq_wav(wav_path, int(SDR_SAMPLE_RATE), float(window_s))
    demodulated = qs._run_demod_diag_iq(iq)
    return qs.measure_demodulated(
        demodulated,
        warmup_s=float(defaults.get_default("warmup_s")),
        noise_diag=bool(defaults.get_default("noise_diag")),
        noise_hf_lo_hz=float(defaults.get_default("noise_hf_lo_hz")),
        noise_hf_hi_hz=float(defaults.get_default("noise_hf_hi_hz")),
    )


def _decode_in_child(send, wav_path: str, window_s: float) -> None:
    """The child process: decode, and send back one answer either way."""
    try:
        measured = decode_file(wav_path, window_s)
    except Exception as e:
        send.send(("failed", f"{type(e).__name__}: {e}"))
    else:
        send.send(("measured", measured))
    finally:
        send.close()


class Job:
    """One re-decode, in a child process, waited for on a thread.

    :meth:`start` starts the child and a thread that waits for its one
    answer; the thread calls *on_done* with ``(measurement, "")`` or
    ``(None, why)`` - why it failed, :data:`CANCELLED`, or that the
    child ended without answering.  *on_done* runs on that thread.

    The child is started with ``spawn`` on every platform: a fork of a
    process with Qt and audio threads running is not a process anyone
    can reason about.  It is a daemon, so a receiver that exits
    normally ends it on the way out even if nobody cancelled it; one
    that is killed does not.

    *target* is the child's entry point, for tests; it is called as
    ``target(send, wav_path, window_s)`` and must send one message.
    """

    def __init__(self, wav_path: str, window_s: float, *,
                 target: Callable | None = None) -> None:
        self.wav_path = wav_path
        self.window_s = float(window_s)
        self._target = target or _decode_in_child
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._receive = None
        self._cancelled = False
        self._thread = threading.Thread(
            target=self._wait_for_it, name="Redecode", daemon=True)
        self._on_done: Callable | None = None

    @property
    def process(self):
        """The child, once started.  For tests."""
        return self._process

    def start(self, on_done: Callable) -> None:
        self._on_done = on_done
        receive, send = self._context.Pipe(duplex=False)
        self._receive = receive
        self._process = self._context.Process(
            target=self._target, args=(send, self.wav_path, self.window_s),
            name="Redecode", daemon=True)
        try:
            self._process.start()
        except BaseException:
            receive.close()
            raise
        finally:
            # The child has its own copy of the sending end; this one is
            # of no use here.  (Its going is not how the waiting thread
            # learns the child has gone - see _wait_for_it.)
            send.close()
        try:
            self._thread.start()
        except BaseException:
            # With nothing to wait for it, a child left running would
            # decode for nobody and hold the pipe open until it ended.
            self._process.terminate()
            self._process.join()
            receive.close()
            raise

    def cancel(self) -> None:
        """End the child, and report :data:`CANCELLED` instead of an answer.

        Unless the waiting thread has already reported: a cancel that
        comes after that is too late to change what was said, and a
        caller that must not act on an answer after cancelling has to
        set it aside itself (the Recordings tab does).
        """
        self._cancelled = True
        if self._process is not None:
            self._process.terminate()

    def wait(self) -> None:
        """Block until the waiting thread has reported."""
        if self._thread.ident is not None:
            self._thread.join()

    def _wait_for_it(self) -> None:
        # For the answer or for the child's end, whichever comes first.
        # The end of the pipe alone is not enough: on Windows the child's
        # copy of the sending end is held by this process until the
        # child takes it, which it does only once it has started, so a
        # child ended while it is still importing leaves the pipe open
        # here for good, and a recv() would never return.
        multiprocessing.connection.wait(
            [self._receive, self._process.sentinel])
        answer = None
        try:
            if self._receive.poll():
                answer = self._receive.recv()
        except (EOFError, OSError):
            # Ended without sending: cancelled, or killed, or crashed
            # somewhere a Python exception could not catch.
            pass
        self._process.join()
        self._receive.close()
        if self._cancelled:
            outcome = (None, CANCELLED)
        elif answer is None:
            outcome = (None, "The decoding process ended without an "
                             f"answer (exit code {self._process.exitcode}).")
        elif answer[0] == "measured":
            outcome = (answer[1], "")
        else:
            outcome = (None, answer[1])
        self._on_done(*outcome)
