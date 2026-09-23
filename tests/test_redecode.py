"""Re-decoding a recorded IQ capture in a process of its own (P5 PR-D).

What it measures is held to what the command line measures for the
same file and window: the report's lines and the CSV row are compared
whole.  How the child ends - an answer, an exception, a cancel before
or after it is running, a crash - is exercised with real spawned
processes, since that is where the ways of ending differ.
"""

from __future__ import annotations

import os
import sys
import threading
from multiprocessing.connection import Listener

import pytest

import fm_radio.quality_selftest as qs
from fm_radio.quality_selftest import (
    IQ_CSV_HEADER, iq_csv_row, iq_report_lines,
)
from fm_radio.recording_meta import Recording
from fm_radio.redecode import (
    CANCELLED, Job, decode_file, first_part_path, why_not,
)

import redecode_children
from iq_capture import write_stereo_iq_wav

#: Long enough for a child to start, decode three seconds and answer on
#: a slow runner; a deadline for a test that would otherwise hang, not
#: a wait for anything to happen.
_DEADLINE_S = 120.0


def _rec(**fields) -> Recording:
    base = dict(
        sidecar=os.path.join("somewhere", "x.json"), kind="iq",
        parts=("x.wav",), missing=(), unconfirmed=(),
        sample_rate_hz=1024000, center_freq_hz=80.0e6, gain_db=8.7,
        channels=None, started_at=None, stopped_at=None, dropped=0,
        audio_seconds=60.0, wall_seconds=60.0, problem="",
    )
    base.update(fields)
    return Recording(**base)


# ----------------------------------------------------------------------
# Which recordings can be
# ----------------------------------------------------------------------

def test_a_complete_standard_rate_iq_capture_can_be_re_decoded():
    assert why_not(_rec()) == ""


@pytest.mark.parametrize("fields, word", [
    (dict(kind="audio", sample_rate_hz=48000), "Only IQ"),
    (dict(missing=("x.wav",)), "not there"),
    (dict(unconfirmed=("x.wav",)), "not there"),
    (dict(problem="its parts are not a list"), "not there"),
    (dict(audio_seconds=None), "could not be measured"),
    (dict(audio_seconds=0.0), "no samples"),
    (dict(sample_rate_hz=None), "does not say"),
    (dict(sample_rate_hz=250000), "250 kHz"),
])
def test_what_cannot_be_re_decoded_says_why(fields, word):
    why = why_not(_rec(**fields))
    assert word in why


def test_the_first_part_is_looked_for_beside_the_sidecar():
    rec = _rec(sidecar=os.path.join("d", "r.json"),
               parts=(os.path.join("elsewhere", "r.wav"), "r.part001.wav"))
    assert first_part_path(rec) == os.path.join("d", "r.wav")


# ----------------------------------------------------------------------
# What it measures: what the command line measures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    return write_stereo_iq_wav(tmp_path_factory.mktemp("iq") / "c.wav")


def _command_line(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["quality_selftest", *args])
    qs.main()
    return capsys.readouterr().out


def test_the_report_is_the_command_lines(monkeypatch, capsys, capture):
    printed = _command_line(monkeypatch, capsys,
                            "--iq-wav", capture, "--duration", "2")
    measured = decode_file(capture, 2)
    assert printed.splitlines() == iq_report_lines(measured)
    # Something to compare: a measured blend, pilot and noise floor.
    assert measured.blend_mean > 0.5
    assert measured.pilot_snr_median_db > 20
    assert measured.noise_band_hz is not None


def test_the_csv_row_is_the_command_lines(monkeypatch, capsys, capture,
                                          tmp_path):
    csv = tmp_path / "cli.csv"
    _command_line(monkeypatch, capsys, "--iq-wav", capture,
                  "--duration", "2", "--noise-csv", str(csv))
    measured = decode_file(capture, 2)
    assert csv.read_text(encoding="utf-8") == (
        IQ_CSV_HEADER + iq_csv_row(measured, "c.wav", 2))


def test_a_file_at_another_rate_is_refused(tmp_path):
    wav = write_stereo_iq_wav(tmp_path / "light.wav", 0.5, rate=250000)
    with pytest.raises(ValueError, match="250 kHz"):
        decode_file(wav, 2)


# ----------------------------------------------------------------------
# The child, and the ways it ends
# ----------------------------------------------------------------------

def _run(job: Job) -> tuple:
    """Start *job* and return what it reported, failing if it never does."""
    told = []
    job.start(lambda measured, why: told.append((measured, why)))
    job._thread.join(_DEADLINE_S)
    assert not job._thread.is_alive(), "the job never reported"
    assert len(told) == 1
    return told[0]


def test_the_child_answers_with_the_measurement(capture):
    measured, why = _run(Job(capture, 2))
    assert why == ""
    assert iq_report_lines(measured) == iq_report_lines(
        decode_file(capture, 2))


def test_the_child_says_what_went_wrong(tmp_path):
    measured, why = _run(Job(str(tmp_path / "not-there.wav"), 2))
    assert measured is None
    assert why.startswith("FileNotFoundError")


def test_a_child_that_dies_without_answering_is_reported(tmp_path):
    measured, why = _run(Job(str(tmp_path / "x.wav"), 2,
                             target=redecode_children.die_without_answering))
    assert measured is None
    assert "without an answer" in why
    assert "exit code 3" in why


def test_a_waiting_thread_that_cannot_start_takes_the_child_with_it():
    """The child is started first.  If the thread that would wait for it
    cannot be, nothing would ever hear from it: it is ended, and the
    pipe closed, before the failure is passed on."""
    with Listener() as listener:
        job = Job(listener.address, 2,
                  target=redecode_children.wait_forever)

        def no_threads_left():
            raise RuntimeError("can't start new thread")

        job._thread.start = no_threads_left
        try:
            with pytest.raises(RuntimeError, match="can't start"):
                job.start(lambda measured, why: None)
            assert not job.process.is_alive()
            assert job.process.exitcode is not None
            assert job._receive.closed
        finally:
            if job.process.is_alive():
                job.process.terminate()
                job.process.join()


def test_cancelling_a_running_child_ends_it():
    with Listener() as listener:
        job = Job(listener.address, 2,
                  target=redecode_children.wait_forever)
        told = []
        job.start(lambda measured, why: told.append((measured, why)))
        # Accepting blocks until the child has connected: it is running,
        # and past anything the start of a process does.
        accepted = []
        accepting = threading.Thread(
            target=lambda: accepted.append(listener.accept()), daemon=True)
        accepting.start()
        accepting.join(_DEADLINE_S)
        assert accepted, "the child never said it was running"
        with accepted[0] as conn:
            assert conn.recv() == job.process.pid

        job.cancel()
        job._thread.join(_DEADLINE_S)
    assert not job._thread.is_alive(), "cancelling did not end the wait"
    assert told == [(None, CANCELLED)]
    assert not job.process.is_alive()
    assert job.process.exitcode != 0


def test_cancelling_before_the_child_has_started_ends_the_wait(tmp_path):
    """A cancel that lands while the child is still starting.

    On Windows the sending end of the pipe is held by the parent until
    the child takes it, which it does once it has started; ended before
    then, it never does, and the pipe never closes.  A wait on the pipe
    alone would never return.  Nothing here holds the child back, so
    this relies on a cancel issued straight after start() landing
    before an interpreter has started - which is what makes it the
    case above.
    """
    job = Job(str(tmp_path / "x.wav"), 2,
              target=redecode_children.wait_forever)
    told = []
    job.start(lambda measured, why: told.append((measured, why)))
    job.cancel()
    job._thread.join(_DEADLINE_S)
    assert not job._thread.is_alive(), "cancelling did not end the wait"
    assert told == [(None, CANCELLED)]
