"""Choosing the sound device, and moving the radio to another one mid-play.

The machine is the fake in conftest: one headphone amplifier under MME
(the system default), WASAPI and WDM-KS, a second card under WASAPI, and
a headset WASAPI mixes at 32 kHz.  What matters is which device a stream
is opened on, and that after a switch the audio comes out of the new one.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fm_radio.audio_output import AudioOutput
from fm_radio.constants import AUDIO_FRAMES_PER_BUFFER
from fm_radio.exceptions import AudioOutputError

FPB = AUDIO_FRAMES_PER_BUFFER
HP_MME, HP_WASAPI, REALTEK, HEADSET, HP_WDMKS = range(5)


@pytest.fixture
def make_output(fake_sound_devices):
    made = []

    def make(**kwargs):
        out = AudioOutput(**kwargs)
        made.append(out)
        return out

    yield make
    for out in made:
        out.cleanup()


def playing(out, level=0.25):
    """Feed it until it plays, then take a buffer; returns what played."""
    block = np.full(768, level, dtype=np.float32)
    while not out._playing:
        out.enqueue_audio(block, block)
    for _ in range(3):
        out.enqueue_audio(block, block)
    played, _ = out.callback(None, FPB, {}, 0)
    return np.frombuffer(played, dtype=np.float32)


# ----------------------------------------------------------------------
# What is offered, and what is chosen
# ----------------------------------------------------------------------

def test_only_devices_that_can_play_48_khz_stereo_are_offered(make_output):
    out = make_output()
    assert out.output_devices == [
        "MME: Speakers (USB Audio HP-A8)",
        "WASAPI: Speakers (USB Audio HP-A8)",
        "WASAPI: Realtek Digital Output",
    ]


def test_the_default_is_the_system_default_under_wasapi(make_output,
                                                         fake_sound_devices):
    """MME is PortAudio's default, and 100 ms later than it need be."""
    out = make_output()
    assert out.device == "WASAPI: Speakers (USB Audio HP-A8)"
    assert fake_sound_devices.opened[-1]["output_device_index"] == HP_WASAPI


def test_a_named_device_is_opened(make_output, fake_sound_devices):
    out = make_output(device="WASAPI: Realtek Digital Output")
    assert out.device == "WASAPI: Realtek Digital Output"
    assert fake_sound_devices.opened[-1]["output_device_index"] == REALTEK


def test_part_of_a_name_will_do_when_it_names_one(make_output,
                                                  fake_sound_devices):
    out = make_output(device="realtek")
    assert fake_sound_devices.opened[-1]["output_device_index"] == REALTEK
    assert out.device == "WASAPI: Realtek Digital Output"


@pytest.mark.parametrize("wanted", ["no such card", "USB Audio"])
def test_a_name_that_is_not_one_device_gets_the_default(make_output, caplog,
                                                        wanted):
    """Unplugged headphones, or a name two APIs share: still a radio."""
    with caplog.at_level(logging.WARNING, logger="fm_receiver.AudioOutput"):
        out = make_output(device=wanted)
    assert out.device == "WASAPI: Speakers (USB Audio HP-A8)"
    assert any(wanted in r.getMessage() for r in caplog.records)


def test_a_device_that_will_not_open_falls_back_to_the_wasapi_default(
        make_output, fake_sound_devices, caplog):
    """Not to PortAudio's, which is MME and 100 ms further behind."""
    fake_sound_devices.refuse_to_open.add(REALTEK)
    with caplog.at_level(logging.WARNING, logger="fm_receiver.AudioOutput"):
        out = make_output(device="WASAPI: Realtek Digital Output")
    assert out.device == "WASAPI: Speakers (USB Audio HP-A8)"
    assert fake_sound_devices.opened[-1]["output_device_index"] == HP_WASAPI
    assert any("Could not open" in r.getMessage() for r in caplog.records)
    assert np.max(playing(out)) > 0.2


def test_with_the_default_refusing_too_it_falls_back_to_portaudio(
        make_output, fake_sound_devices):
    fake_sound_devices.refuse_to_open.update({REALTEK, HP_WASAPI})
    out = make_output(device="WASAPI: Realtek Digital Output")
    assert out.device is None
    assert fake_sound_devices.opened[-1]["output_device_index"] is None
    assert np.max(playing(out)) > 0.2


def test_the_default_refusing_is_tried_once(make_output,
                                            fake_sound_devices):
    """No device asked for and the default will not open: PortAudio's."""
    fake_sound_devices.refuse_to_open.add(HP_WASAPI)
    out = make_output()
    assert out.device is None


def test_with_no_devices_listed_it_plays_on_portaudios_default(
        audio_output):
    """A machine PortAudio lists nothing on - and every other test."""
    assert audio_output.device is None
    assert audio_output.output_devices == []
    assert np.max(playing(audio_output)) > 0.2


# ----------------------------------------------------------------------
# Switching mid-play
# ----------------------------------------------------------------------

def test_a_switch_moves_the_audio_to_the_new_device(make_output,
                                                    fake_sound_devices):
    out = make_output()
    assert np.max(playing(out)) > 0.2
    old = out.stream

    assert out.switch_device("realtek") == "WASAPI: Realtek Digital Output"

    # The old card was stopped and closed; the new one waits to start.
    assert old.stopped and getattr(old, "closed", False)
    assert out.stream is not old
    assert out.stream.device_index == REALTEK
    assert not out.stream.started
    assert out.device == "WASAPI: Realtek Digital Output"
    # Audio from before the switch was dropped, and the new card starts
    # from behind a cushion, as at startup: what plays is the new audio.
    # (Not the old 0.25: the resampler starts clean after a hold and
    # rings a little over the step up from nothing.)
    played = playing(out, level=0.5)
    assert out.stream.started
    assert np.max(played) > 0.4
    assert out.underruns == 0


def test_a_device_that_will_not_open_leaves_the_radio_where_it_was(
        make_output, fake_sound_devices):
    out = make_output()
    playing(out)
    old = out.stream
    fake_sound_devices.refuse_to_open.add(REALTEK)

    with pytest.raises(AudioOutputError):
        out.switch_device("realtek")

    assert out.stream is old and not getattr(old, "closed", False)
    assert out.device == "WASAPI: Speakers (USB Audio HP-A8)"
    assert not out.held
    # Let go: the next blocks start the old card again.
    assert np.max(playing(out)) > 0.2
    assert old.started


def test_a_name_that_is_no_device_changes_nothing(make_output):
    out = make_output()
    playing(out)
    old = out.stream
    with pytest.raises(ValueError):
        out.switch_device("no such card")
    assert out.stream is old and old.started and not out.held


def test_switching_to_the_device_already_playing_does_not_stop_it(
        make_output):
    out = make_output()
    playing(out)
    old = out.stream
    out.switch_device("WASAPI: Speakers (USB Audio HP-A8)")
    assert out.stream is old and old.started


def test_a_switch_after_cleanup_is_refused(make_output):
    out = make_output()
    out.cleanup()
    with pytest.raises(AudioOutputError):
        out.switch_device("realtek")


def test_a_switch_takes_the_new_cards_buffer_into_the_preroll(
        make_output, monkeypatch):
    """The preroll is the cushion plus what the card takes at once."""
    out = make_output()
    before = out._preroll_frames
    monkeypatch.setattr(type(out.stream), "output_latency", 0.02)
    out.switch_device("realtek")
    # WASAPI, so twice what it reports, on top of the matcher's target;
    # see _what_the_card_takes_at_once and _size_up_the_card.
    assert out._preroll_frames == round(out._matcher.target
                                        + 2 * 0.02 * 48000)
    assert out._preroll_frames < before


# ----------------------------------------------------------------------
# Remembered in the user's file
# ----------------------------------------------------------------------

from fm_radio import stations as st  # noqa: E402

needs_tomllib = pytest.mark.skipif(st.tomllib is None,
                                   reason="needs tomllib to check the file")


def test_the_device_is_written_and_read_back(tmp_path):
    path = tmp_path / "stations.toml"
    st.remember_audio_device("WASAPI: Speakers (USB Audio HP-A8)", path)
    assert path.read_text(encoding="utf-8") == (
        'audio_device = "WASAPI: Speakers (USB Audio HP-A8)"\n')
    assert st.configured_audio_device(path) == (
        "WASAPI: Speakers (USB Audio HP-A8)")


@needs_tomllib
def test_a_second_choice_changes_the_line_and_nothing_else(tmp_path):
    path = tmp_path / "stations.toml"
    path.write_text('area = "関東"\r\naudio_device = "MME: x"  # mine\r\n\r\n'
                    '[[station]]\r\nname = "y"\r\nfreq_mhz = 80.0\r\n',
                    encoding="utf-8", newline="")
    st.remember_audio_device("WASAPI: Realtek Digital Output", path)
    with path.open(encoding="utf-8", newline="") as handle:
        assert handle.read() == (
            'area = "関東"\r\n'
            'audio_device = "WASAPI: Realtek Digital Output"  # mine\r\n\r\n'
            '[[station]]\r\nname = "y"\r\nfreq_mhz = 80.0\r\n')
    assert st.home_area(path) == "関東"


@needs_tomllib
def test_the_area_and_the_device_do_not_write_over_each_other(tmp_path):
    path = tmp_path / "stations.toml"
    st.remember_area("関東", path)
    st.remember_audio_device("WASAPI: Realtek Digital Output", path)
    st.remember_area("近畿", path)
    assert st.home_area(path) == "近畿"
    assert st.configured_audio_device(path) == "WASAPI: Realtek Digital Output"


@needs_tomllib
def test_a_name_with_double_quotes_is_written_as_a_literal_string(tmp_path):
    path = tmp_path / "stations.toml"
    st.remember_audio_device('MME: "Studio" Out', path)
    assert st.configured_audio_device(path) == 'MME: "Studio" Out'
    st.remember_audio_device("MME: plain", path)
    assert st.configured_audio_device(path) == "MME: plain"


def test_a_name_with_both_quotes_is_not_written(tmp_path):
    path = tmp_path / "stations.toml"
    with pytest.raises(st.WillNotEdit):
        st.remember_audio_device('MME: "it' + "'" + 's"', path)
    assert not path.exists()


@needs_tomllib
def test_a_device_that_is_not_a_string_is_reported_and_ignored(tmp_path):
    path = tmp_path / "stations.toml"
    path.write_text("audio_device = 3\n", encoding="utf-8")
    said = []
    assert st.configured_audio_device(path, warn=said.append) is None
    assert said and "audio_device" in said[0]


# ----------------------------------------------------------------------
# The receiver: which device it starts on, and remembering a switch
# ----------------------------------------------------------------------

@pytest.fixture
def receiver_on(fake_sound_devices, no_user_config):
    """A receiver over the fake devices; ``stations.toml`` in tmp."""
    from fm_radio.controller import FMReceiverController
    built = []

    def build(**kwargs):
        instance = FMReceiverController(
            light=True, stations_path=str(no_user_config), **kwargs)
        built.append(instance)
        return instance

    yield build
    for instance in built:
        instance.quit_event.set()
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def test_a_switch_is_remembered_for_next_time(receiver_on, no_user_config):
    first = receiver_on()
    assert first.get_audio_device() == "WASAPI: Speakers (USB Audio HP-A8)"
    assert first.set_audio_device("realtek") == (
        "WASAPI: Realtek Digital Output")
    assert first.get_audio_device() == "WASAPI: Realtek Digital Output"
    assert st.configured_audio_device(no_user_config) == (
        "WASAPI: Realtek Digital Output")

    second = receiver_on()
    assert second.get_audio_device() == "WASAPI: Realtek Digital Output"


def test_the_command_line_device_wins_for_the_run_and_is_not_written(
        receiver_on, no_user_config):
    st.remember_audio_device("WASAPI: Realtek Digital Output",
                             no_user_config)
    receiver = receiver_on(audio_device="MME: Speakers (USB Audio HP-A8)")
    assert receiver.get_audio_device() == "MME: Speakers (USB Audio HP-A8)"
    assert st.configured_audio_device(no_user_config) == (
        "WASAPI: Realtek Digital Output")


def test_a_remembered_device_that_is_gone_gets_the_default(
        receiver_on, no_user_config):
    st.remember_audio_device("WASAPI: Headphones I lent out",
                             no_user_config)
    receiver = receiver_on()
    assert receiver.get_audio_device() == "WASAPI: Speakers (USB Audio HP-A8)"


def test_a_file_that_cannot_be_written_does_not_undo_the_switch(
        receiver_on, no_user_config, capsys):
    no_user_config.write_text("this is = = not toml\n", encoding="utf-8")
    receiver = receiver_on()
    assert receiver.set_audio_device("realtek") == (
        "WASAPI: Realtek Digital Output")
    assert receiver.get_audio_device() == "WASAPI: Realtek Digital Output"
    assert "could not remember it" in capsys.readouterr().err
    assert no_user_config.read_text(encoding="utf-8") == (
        "this is = = not toml\n")


def test_a_failed_switch_is_not_remembered(receiver_on, no_user_config,
                                           fake_sound_devices):
    receiver = receiver_on()
    fake_sound_devices.refuse_to_open.add(REALTEK)
    with pytest.raises(AudioOutputError):
        receiver.set_audio_device("realtek")
    assert st.configured_audio_device(no_user_config) is None


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------

@pytest.fixture
def say(receiver_on, capsys):
    from fm_radio.cli import CommandLineInterface
    receiver = receiver_on()
    interface = CommandLineInterface(receiver)

    def _say(line: str) -> str:
        capsys.readouterr()
        assert interface._dispatch(line) is True
        return capsys.readouterr().out

    _say.receiver = receiver
    return _say


def test_out_lists_the_devices_and_marks_the_one_playing(say):
    lines = say("out").splitlines()
    assert lines == [
        "   1. MME: Speakers (USB Audio HP-A8)",
        " * 2. WASAPI: Speakers (USB Audio HP-A8)",
        "   3. WASAPI: Realtek Digital Output",
    ]


def test_out_with_a_number_switches_to_it(say):
    assert "Playing through WASAPI: Realtek Digital Output." in say("out 3")
    assert say.receiver.get_audio_device() == "WASAPI: Realtek Digital Output"


def test_out_with_a_name_switches_to_it_whatever_its_case(say):
    say("out mme: speakers")
    assert say.receiver.get_audio_device() == "MME: Speakers (USB Audio HP-A8)"


@pytest.mark.parametrize("line", ["out 9", "out 0", "out nothing like it",
                                  "out speakers"])
def test_out_with_no_one_device_says_so_and_stays(say, line):
    said = say(line)
    assert "'out' lists them" in said
    assert say.receiver.get_audio_device() == (
        "WASAPI: Speakers (USB Audio HP-A8)")


def test_out_says_why_when_the_device_will_not_open(say, fake_sound_devices):
    fake_sound_devices.refuse_to_open.add(REALTEK)
    assert "Could not switch" in say("out 3")


# ----------------------------------------------------------------------
# The flags
# ----------------------------------------------------------------------

@pytest.fixture
def run_main(monkeypatch):
    import sys
    import fm_radio.__main__ as entry
    built = []

    class Entry:
        def __init__(self, **kwargs):
            import threading
            self.kwargs = kwargs
            self.quit_event = threading.Event()
            self.device_failure = None
            built.append(self)

        def start(self):
            pass

    monkeypatch.setattr(entry, "FMReceiverController", Entry)

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["fm_receiver.py", *argv])
        try:
            entry.main()
        except SystemExit as exc:
            return exc.code
        return None

    _run.built = built
    return _run


def test_the_device_flag_reaches_the_receiver(run_main):
    run_main("--light", "--audio-device", "WASAPI: Realtek")
    (controller,) = run_main.built
    assert controller.kwargs["audio_device"] == "WASAPI: Realtek"


def test_without_the_flag_the_receiver_is_left_to_choose(run_main):
    run_main("--light")
    (controller,) = run_main.built
    assert controller.kwargs["audio_device"] is None


def test_the_device_flag_needs_a_name(run_main, capsys):
    assert run_main("--audio-device") == 2
    assert run_main.built == []
    assert "--list-audio-devices" in capsys.readouterr().err


def test_listing_the_devices_starts_no_receiver(run_main, capsys,
                                                fake_sound_devices,
                                                monkeypatch, tmp_path):
    monkeypatch.setattr(st, "user_config_path",
                        lambda: tmp_path / "stations.toml")
    assert run_main("--list-audio-devices") is None
    assert run_main.built == []
    said = capsys.readouterr().out
    assert "  WASAPI: Speakers (USB Audio HP-A8) (default)" in said
    assert "  MME: Speakers (USB Audio HP-A8)\n" in said
    assert "Headset" not in said


def test_a_name_the_console_cannot_show_is_printed_anyway():
    """cp932 has no (R) sign, and a game controller's headset has one."""
    import io
    from fm_radio.audio_devices import printable
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    shown = printable("MME: Headset (DUALSHOCK®4)", console)
    console.write(shown)            # must not raise
    assert shown == "MME: Headset (DUALSHOCK?4)"


# ----------------------------------------------------------------------
# What a WASAPI card takes when it starts
# ----------------------------------------------------------------------

def _start_like_wasapi(out, block_frames=768, seconds=3.0, stall=None):
    """Start the card the way WASAPI does; count the underruns.

    Measured on the hardware (FOSTEX HP-A8, shared mode, 1024-frame
    buffers): it reports 42.7 ms of output latency, then calls back
    three times at once and twice more 22 ms later - 5120 frames in
    the first 22 ms, twice what it says it holds plus the buffer the
    clock asks for.  After that one call every 21.3 ms.  Blocks keep
    arriving every 16 ms throughout, except for *stall* = (start,
    length) in seconds, when none arrive and then all of them do -
    the first second on the hardware has auto gain writing to the
    dongle five times, and the blocks come in bunches.
    """
    block = np.full(block_frames, 0.25, dtype=np.float32)
    while not out._playing:
        out.enqueue_audio(block, block)
    interval = block_frames / 48000.0
    callbacks = [0.0, 0.0, 0.001, 0.0217, 0.0217]
    t = 0.0423
    while t < seconds:
        callbacks.append(t)
        t += FPB / 48000.0
    next_block = interval
    for when in callbacks:
        held_up = stall is not None and stall[0] <= when < sum(stall)
        while next_block <= when and not held_up:
            out.enqueue_audio(block, block)
            next_block += interval
        out.callback(None, FPB, {}, 0)
    return out.underruns


def test_a_wasapi_card_starting_does_not_run_the_cushion_dry(
        make_output, monkeypatch):
    """What switching to WASAPI did on the hardware: underruns at the start."""
    from conftest import _FakeStream
    monkeypatch.setattr(_FakeStream, "output_latency", 0.0427)
    out = make_output()
    assert out.device.startswith("WASAPI:")
    assert _start_like_wasapi(out) == 0


def test_a_wasapi_start_is_not_left_thin(make_output, monkeypatch):
    """What it is left with after its start is the matcher's target.

    The hardware ran dry a second after a WASAPI start: the card's
    first 22 ms had left 1800 frames against a target of 5900, and the
    matcher fills a gap that size over a minute.  A 60 ms stall in the
    blocks then is the cushion's to cover, and with 1800 it cannot.
    """
    from conftest import _FakeStream
    monkeypatch.setattr(_FakeStream, "output_latency", 0.0427)
    out = make_output()
    assert _start_like_wasapi(out, stall=(0.1, 0.06)) == 0
    # ...and the target is the least one and its margin, not the
    # preroll: the queue that stays is how late the radio is.
    assert out._matcher.target == pytest.approx(
        2 * out._cushion_frames + 0.03 * 48000 + 0.02 * 48000)


def test_after_a_switch_to_wasapi_the_start_does_not_run_dry(
        make_output, monkeypatch):
    from conftest import _FakeStream
    monkeypatch.setattr(_FakeStream, "output_latency", 0.0427)
    out = make_output(device="MME: Speakers")
    playing(out)
    out.switch_device("WASAPI: Realtek Digital Output")
    assert _start_like_wasapi(out) == 0


def test_an_mme_card_is_not_given_wasapis_allowance(make_output,
                                                     monkeypatch):
    """MME takes what it says, at most; doubling it is 100 ms for nothing."""
    from conftest import _FakeStream
    monkeypatch.setattr(_FakeStream, "output_latency", 0.1067)
    out = make_output(device="MME: Speakers")
    assert out._what_the_card_takes_at_once() == round(0.1067 * 48000)


def test_a_wasapi_stream_rides_out_the_dip_the_hardware_had(make_output,
                                                            monkeypatch):
    """Forty seconds into a WASAPI run the level fell by 4500 frames.

    The blocks stopped and then came all at once - the SDR's doing, as
    the dips on MME are - and from a target of 5024 that left 838
    frames and an underrun.  The blocks here come more evenly than the
    hardware's, so the stall is made 110 ms to take as much.  An MME stream settles 1900
    frames higher, at its preroll, and has ridden out dips like it
    for as long as there have been logs.
    """
    from conftest import _FakeStream
    monkeypatch.setattr(_FakeStream, "output_latency", 0.0427)
    out = make_output()
    assert _start_like_wasapi(out, seconds=6.0, stall=(4.0, 0.11)) == 0


# ----------------------------------------------------------------------
# Review fixes: two writers, and a remembered name
# ----------------------------------------------------------------------

@needs_tomllib
def test_the_area_and_the_device_written_at_once_both_stay(tmp_path,
                                                           monkeypatch):
    """The window writes the area while the switch thread writes the device.

    Each read the old file and put back its own edit of it, and the one
    that finished last wiped out the other - or found the shared .new
    file already moved away.  The first writer is held at the point of
    putting its file in place, and the second started then.
    """
    import threading
    import time
    path = tmp_path / "stations.toml"
    real_replace = st.os.replace
    at_the_door = threading.Event()

    def slow_replace(source, destination):
        if not at_the_door.is_set():
            at_the_door.set()
            time.sleep(0.3)
        real_replace(source, destination)

    monkeypatch.setattr(st.os, "replace", slow_replace)
    trouble = []

    def write(what):
        try:
            what()
        except Exception as e:          # pragma: no cover - the failure
            trouble.append(e)

    first = threading.Thread(target=write, args=(
        lambda: st.remember_area("関東", path),))
    second = threading.Thread(target=write, args=(
        lambda: st.remember_audio_device("WASAPI: Realtek Digital Output",
                                         path),))
    first.start()
    assert at_the_door.wait(5.0)
    second.start()
    first.join(5.0)
    second.join(5.0)

    assert not trouble, trouble
    assert st.home_area(path) == "関東"
    assert st.configured_audio_device(path) == "WASAPI: Realtek Digital Output"


def test_a_remembered_name_is_not_taken_for_part_of_another(
        receiver_on, no_user_config):
    """Remembered "WASAPI: Realtek", and only "...Realtek Digital Output" is here.

    That is another device that happens to start the same way; the
    remembered one is gone, and the default plays instead.
    """
    st.remember_audio_device("WASAPI: Realtek", no_user_config)
    receiver = receiver_on()
    assert receiver.get_audio_device() == "WASAPI: Speakers (USB Audio HP-A8)"


def test_a_typed_name_may_still_be_part_of_one(receiver_on):
    receiver = receiver_on(audio_device="WASAPI: Realtek")
    assert receiver.get_audio_device() == "WASAPI: Realtek Digital Output"


# ----------------------------------------------------------------------
# Saying so when the radio is not playing where it was asked to
# ----------------------------------------------------------------------

def test_a_device_not_found_is_given_as_the_reason(make_output):
    out = make_output(device="WASAPI: Headphones I lent out")
    why = out.fallback_reason
    assert "WASAPI: Headphones I lent out" in why
    assert "WASAPI: Speakers (USB Audio HP-A8)" in why


def test_a_device_that_will_not_open_is_given_as_the_reason(
        make_output, fake_sound_devices):
    fake_sound_devices.refuse_to_open.add(REALTEK)
    out = make_output(device="WASAPI: Realtek Digital Output")
    why = out.fallback_reason
    assert "Could not open WASAPI: Realtek Digital Output" in why
    assert "WASAPI: Speakers (USB Audio HP-A8)" in why


def test_playing_where_asked_has_no_reason(make_output):
    assert make_output(device="realtek").fallback_reason is None
    assert make_output().fallback_reason is None


def test_the_default_refusing_says_where_it_plays_instead(
        make_output, fake_sound_devices):
    fake_sound_devices.refuse_to_open.add(HP_WASAPI)
    why = make_output().fallback_reason
    assert "Could not open WASAPI: Speakers (USB Audio HP-A8)" in why
    assert "the system's default device" in why


def test_the_console_is_told_when_the_remembered_device_is_gone(
        receiver_on, no_user_config, capsys):
    """Logging is off in an ordinary run: a warning in it reaches nobody."""
    st.remember_audio_device("WASAPI: Headphones I lent out",
                             no_user_config)
    receiver = receiver_on()
    said = capsys.readouterr().err
    assert "WASAPI: Headphones I lent out" in said
    assert "WASAPI: Speakers (USB Audio HP-A8)" in said
    assert receiver.get_audio_notice() == receiver.audio_output.fallback_reason


def test_nothing_is_said_when_it_plays_where_it_was_asked(receiver_on,
                                                          capsys):
    receiver = receiver_on()
    assert "Sound:" not in capsys.readouterr().err
    assert receiver.get_audio_notice() is None


def test_a_name_several_devices_share_is_given_as_that(make_output):
    """Not "no device is called": the user would go looking for a cable."""
    why = make_output(device="hp-a8").fallback_reason
    assert "No sound device is called" not in why
    assert "'hp-a8' names 2 sound devices, not one" in why
    assert "MME: Speakers (USB Audio HP-A8)" in why
    assert "playing through WASAPI: Speakers (USB Audio HP-A8)" in why


def test_a_remembered_name_is_never_called_ambiguous(make_output):
    """Taken whole, a name names one device or none."""
    why = make_output(device="hp-a8", exact_device=True).fallback_reason
    assert why.startswith("No sound device is called 'hp-a8'")


def test_a_switch_to_a_shared_name_says_which_share_it(make_output):
    out = make_output()
    with pytest.raises(ValueError, match="names 2 sound devices"):
        out.switch_device("hp-a8")
    with pytest.raises(ValueError, match="No sound device is called"):
        out.switch_device("no such card")


def test_out_with_a_shared_name_says_so(say):
    said = say("out hp-a8")
    assert "names 2 sound devices" in said
    assert "No sound device is called" not in said
