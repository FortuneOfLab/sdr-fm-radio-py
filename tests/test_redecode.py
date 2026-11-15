"""Re-decoding a recorded IQ capture in a process of its own (P5 PR-D).

What it measures is held to what the command line measures for the
same file and window: the report's lines and the CSV row are compared
whole.  How the child ends - an answer, an exception, a cancel before
or after it is running, a crash - is exercised with real spawned
processes, since that is where the ways of ending differ.
"""

from __future__ import annotations

import math
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


# ----------------------------------------------------------------------
# Under DSP settings: what the command line asks under the same flags
# ----------------------------------------------------------------------
#
# The Recordings tab turns a DSP-tab slot into the command line's flags
# (dsp_arguments) and the child decodes under them (decode_file).  What
# is compared here is what the demodulator is set up with - captured
# from it the moment the overrides are on, through the same capture()
# the DSP tab's defaults come from - and the CSV row, against the
# command line run in this process with the same flags.  A capture of a
# twentieth of a second is enough for that: the overrides are applied
# before the first block, and the row is compared byte for byte
# whatever it holds.

from dataclasses import fields, replace                         # noqa: E402

from fm_radio.demodulator import FMDemodulator                  # noqa: E402
from fm_radio.dsp_settings import (                             # noqa: E402
    DspSettings, MAX_MONO_DELAY_SAMPLES, capture as dsp_capture,
)
from fm_radio.redecode import DSP_FLAGS, dsp_arguments          # noqa: E402

#: What a case expects when the command line turns its flags down.
REFUSED = "refused"
CASES = ("no-override sentinel", "default", "non-default", "out of range")
FIELDS = [f.name for f in fields(DspSettings)]


@pytest.fixture(scope="module")
def defaults() -> DspSettings:
    """The standard chain's own defaults: what a re-decode starts from."""
    return dsp_capture(FMDemodulator(stereo=True))


#: The short capture holds less than the warmup, so nothing is left to
#: measure after it: the row's averages are of nothing, and say nan.
_EMPTY_AFTER_WARMUP = (
    "ignore:Mean of empty slice:RuntimeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
)
pytestmark = [pytest.mark.filterwarnings(w) for w in _EMPTY_AFTER_WARMUP]


@pytest.fixture(scope="module")
def short_capture(tmp_path_factory):
    return write_stereo_iq_wav(
        tmp_path_factory.mktemp("short") / "s.wav", 0.05)


@pytest.fixture
def asked(monkeypatch):
    """Every demodulator set up for a decode, as capture() reads it
    once the overrides are on - one entry per demodulator built."""
    seen = []
    real = qs._set_up_the_demod

    def watching(demod, **overrides):
        real(demod, **overrides)
        seen.append(dsp_capture(demod))

    monkeypatch.setattr(qs, "_set_up_the_demod", watching)
    return seen


def _same(got: DspSettings, wanted: DspSettings) -> None:
    """Equal, the phase to a rounding error (degrees and back)."""
    assert replace(got, subcarrier_phase_offset_rad=0.0) == replace(
        wanted, subcarrier_phase_offset_rad=0.0)
    assert got.subcarrier_phase_offset_rad == pytest.approx(
        wanted.subcarrier_phase_offset_rad, abs=1e-12)


def _cases(d: DspSettings) -> dict:
    """For each of the nine, the four cases: (flags, what is expected).

    What is expected is the settings the demodulator should end up
    with, or REFUSED.  Notes on the ones that are not plain numbers:

    - The blend's default is the adaptive blend, so its "default" is
      the explicit spelling of that, a negative number; and an
      out-of-range blend is clipped, not refused - the command line
      has always done that, and a re-decode does what it does.
    - The command line has no flag that turns I/Q correction *on*, so
      its "default" is the flag left out, as its sentinel is.  Its
      out of range, and side NR's, is a value given to a switch.
    """
    deg = repr(math.degrees(d.subcarrier_phase_offset_rad))
    nr_on, nr_off = (("--side-nr", "--no-side-nr") if d.side_nr_enabled
                     else ("--no-side-nr", "--side-nr"))
    a = replace

    def number(flag, field, value, other, wrong):
        return {
            "no-override sentinel": ([f"{flag}=nan"], d),
            "default": ([f"{flag}={value!r}"], d),
            "non-default": ([f"{flag}={other!r}"], a(d, **{field: other})),
            "out of range": ([f"{flag}={wrong!r}"], REFUSED),
        }

    return {
        "force_blend_factor": {
            "no-override sentinel": (["--fixed-blend=-1"], d),
            "default": (["--fixed-blend=-0.5"], d),
            "non-default": (["--fixed-blend=0.3"],
                            a(d, force_blend_factor=0.3)),
            "out of range": (["--fixed-blend=1.5"],
                             a(d, force_blend_factor=1.0)),
        },
        "subcarrier_phase_offset_rad": {
            "no-override sentinel": (
                ["--subcarrier-phase-offset-deg=nan"], d),
            "default": ([f"--subcarrier-phase-offset-deg={deg}"], d),
            "non-default": (
                ["--subcarrier-phase-offset-deg=40.0"],
                a(d, subcarrier_phase_offset_rad=math.radians(40.0))),
            "out of range": (
                ["--subcarrier-phase-offset-deg=inf"], REFUSED),
        },
        "mono_delay_samples": {
            "no-override sentinel": (["--mono-delay-samples=-1"], d),
            "default": ([f"--mono-delay-samples={d.mono_delay_samples}"],
                        d),
            "non-default": (["--mono-delay-samples=100"],
                            a(d, mono_delay_samples=100)),
            "out of range": (
                [f"--mono-delay-samples={MAX_MONO_DELAY_SAMPLES + 1}"],
                REFUSED),
        },
        "iq_phase_correction_enabled": {
            "no-override sentinel": ([], d),
            "default": ([], d),
            "non-default": (["--disable-iq-phase-correction"],
                            a(d, iq_phase_correction_enabled=False)),
            "out of range": (["--disable-iq-phase-correction=yes"],
                             REFUSED),
        },
        "lr_high_max_gain": number(
            "--lr-high-max-gain", "lr_high_max_gain",
            d.lr_high_max_gain, 0.5, 1.5),
        "lr_super_high_max_gain": number(
            "--lr-super-high-max-gain", "lr_super_high_max_gain",
            d.lr_super_high_max_gain, 0.25, 1.5),
        "side_nr_enabled": {
            "no-override sentinel": ([], d),
            "default": ([nr_on], d),
            "non-default": ([nr_off],
                            a(d, side_nr_enabled=not d.side_nr_enabled)),
            "out of range": (["--side-nr=maybe"], REFUSED),
        },
        "side_nr_alpha_floor": number(
            "--side-nr-alpha-floor", "side_nr_alpha_floor",
            d.side_nr_alpha_floor, 0.5, 1.5),
        "side_nr_beta": number(
            "--side-nr-beta", "side_nr_beta", d.side_nr_beta, 2.0, -0.1),
    }


def test_every_setting_has_its_four_cases(defaults):
    cases = _cases(defaults)
    assert cases.keys() == set(FIELDS)
    for field, four in cases.items():
        assert tuple(four) == CASES, field
        # Each non-default case really is one, or it tests nothing.
        assert four["non-default"][1] != defaults, field


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("field", FIELDS)
def test_a_re_decode_asks_what_the_command_line_asks(
        field, case, defaults, short_capture, asked, monkeypatch, capsys,
        tmp_path):
    """The roadmap's comparison: each of the nine, in each of the four
    cases, through the re-decode and through the command line."""
    flags, expected = _cases(defaults)[field][case]
    csv = tmp_path / "cli.csv"
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--iq-wav", short_capture, "--duration", "1",
        *flags, "--noise-csv", str(csv), "--noise-tag", "T"])
    try:
        qs.main()
    except SystemExit as e:
        cli_refused = e
    else:
        cli_refused = None
    capsys.readouterr()
    cli_asked = list(asked)
    asked.clear()

    try:
        measured = decode_file(short_capture, 1, flags)
    except ValueError as e:
        gui_refused = e
    else:
        gui_refused = None
    capsys.readouterr()

    if expected is REFUSED:
        assert cli_refused is not None, "the command line took it"
        assert gui_refused is not None, "the re-decode took it"
        assert cli_asked == asked == [], "a demodulator was built"
        if isinstance(cli_refused.code, str):
            # The range check's own sentence, word for word.
            assert str(gui_refused) == cli_refused.code
        else:
            # argparse's: it printed its complaint and gave a status.
            assert cli_refused.code == 2
        return
    assert cli_refused is None, cli_refused.code
    assert gui_refused is None, str(gui_refused)
    assert len(asked) == len(cli_asked) == 1
    assert asked == cli_asked
    _same(asked[0], expected)
    assert csv.read_text(encoding="utf-8") == (
        IQ_CSV_HEADER + iq_csv_row(measured, "T", 1))


def _elsewhere(d: DspSettings) -> dict:
    """A value for each setting that is not the standard default."""
    return dict(
        force_blend_factor=0.25,
        subcarrier_phase_offset_rad=math.radians(40.0),
        mono_delay_samples=100,
        iq_phase_correction_enabled=not d.iq_phase_correction_enabled,
        lr_high_max_gain=0.5,
        lr_super_high_max_gain=0.25,
        side_nr_enabled=not d.side_nr_enabled,
        side_nr_alpha_floor=0.5,
        side_nr_beta=2.0,
    )


def test_a_slot_at_the_defaults_says_nothing(defaults, short_capture,
                                             asked):
    assert dsp_arguments(defaults, defaults) == ()
    decode_file(short_capture, 1, ())
    assert asked == [defaults]


@pytest.mark.parametrize("field", FIELDS)
def test_a_slot_is_decoded_under_what_it_holds(field, defaults,
                                               short_capture, asked):
    """From a DSP-tab slot to the demodulator: the one setting
    changed, and nothing else."""
    slot = replace(defaults, **{field: _elsewhere(defaults)[field]})
    flags = dsp_arguments(slot, defaults)
    assert len(flags) == 1
    assert flags[0].split("=", 1)[0] in DSP_FLAGS
    decode_file(short_capture, 1, flags)
    (got,) = asked
    _same(got, slot)


def test_a_slot_with_everything_changed(defaults, short_capture, asked):
    slot = replace(defaults, **_elsewhere(defaults))
    flags = dsp_arguments(slot, defaults)
    assert len(flags) == 9
    decode_file(short_capture, 1, flags)
    (got,) = asked
    _same(got, slot)


@pytest.mark.parametrize("field, wrong", [
    ("force_blend_factor", 1.5),
    ("subcarrier_phase_offset_rad", math.inf),
    ("mono_delay_samples", MAX_MONO_DELAY_SAMPLES + 1),
    ("iq_phase_correction_enabled", "yes"),
    ("lr_high_max_gain", 1.5),
    ("lr_super_high_max_gain", 1.5),
    ("side_nr_enabled", "yes"),
    ("side_nr_alpha_floor", 1.5),
    ("side_nr_beta", -0.1),
])
def test_a_slot_cannot_hold_what_is_out_of_range(field, wrong, defaults):
    """The GUI's out of range: there is no slot to make flags from.
    (A switch given a word is a TypeError; a number out of its range a
    ValueError.)"""
    with pytest.raises((ValueError, TypeError)):
        replace(defaults, **{field: wrong})


def test_a_setting_at_the_receivers_default_is_left_to_the_decoder(
        defaults, short_capture, asked):
    """A receiver running the light chain has another phase default.
    Left alone, it is not said, and the standard chain decodes with
    its own."""
    light = replace(defaults, subcarrier_phase_offset_rad=(
        defaults.subcarrier_phase_offset_rad - math.radians(0.7)))
    assert dsp_arguments(light, light) == ()
    decode_file(short_capture, 1, dsp_arguments(light, light))
    assert asked == [defaults]


def test_what_the_command_line_cannot_say_is_not_said(defaults):
    with pytest.raises(ValueError, match="I/Q phase correction"):
        dsp_arguments(
            replace(defaults, iq_phase_correction_enabled=True),
            replace(defaults, iq_phase_correction_enabled=False))
    with pytest.raises(ValueError, match="adaptive blend"):
        dsp_arguments(defaults, replace(defaults, force_blend_factor=0.5))


@pytest.mark.parametrize("flag", ["--warmup-s=0.1", "--noise-hf-lo-hz=1",
                                  "--duration=5", "--side-nrx"])
def test_only_dsp_flags_are_taken(flag, short_capture, asked):
    with pytest.raises(ValueError, match="not one of the DSP"):
        decode_file(short_capture, 1, [flag])
    assert asked == []


def test_the_child_decodes_under_the_flags_it_is_given(capture, monkeypatch,
                                                       capsys, tmp_path):
    """Through the pipe and a spawned child, against the command line:
    the same row, and not the defaults' row."""
    flags = ("--fixed-blend=0.0",)
    measured, why = _run(Job(capture, 2, flags))
    assert why == ""
    assert measured.blend_mean == 0.0
    assert decode_file(capture, 2).blend_mean > 0.5

    csv = tmp_path / "cli.csv"
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--iq-wav", capture, "--duration", "2",
        *flags, "--noise-csv", str(csv), "--noise-tag", "T"])
    qs.main()
    capsys.readouterr()
    assert csv.read_text(encoding="utf-8") == (
        IQ_CSV_HEADER + iq_csv_row(measured, "T", 2))
