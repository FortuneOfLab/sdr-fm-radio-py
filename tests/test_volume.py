"""The listening volume: what the card is given, and what is not touched.

Every test here measures the audio itself - what comes out of the
PortAudio callback, or what lands in the WAV - rather than asking the
output what it thinks its volume is.  A volume that is stored and
reported but never applied would pass a test of the flag.
"""

from __future__ import annotations

import sys
import wave

import numpy as np
import pytest

from fm_radio.audio_output import clamp_volume, volume_gain
from fm_radio.cli import CommandLineInterface
from fm_radio.constants import (
    AUDIO_FRAMES_PER_BUFFER, AUDIO_OUTPUT_RATE, RECORD_MAX_INT16,
    VOLUME_STEP_PERCENT,
)

FRAMES = AUDIO_FRAMES_PER_BUFFER
#: A 1 kHz tone at -6 dBFS: loud enough to measure 60 dB down, far
#: enough from full scale that the recording's clip is not in play.
AMPLITUDE = 0.5


def _tone(frames: int = FRAMES, amplitude: float = AMPLITUDE) -> np.ndarray:
    t = np.arange(frames) / AUDIO_OUTPUT_RATE
    return (amplitude * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)


def _played(audio_output, frames: int = FRAMES) -> np.ndarray:
    """Queue one block of tone and return what the card is handed for it.

    Shape (frames, 2): left and right de-interleaved.
    """
    tone = _tone(frames)
    audio_output.enqueue_audio(tone, tone)
    out, _ = audio_output.callback(None, frames, {}, 0)
    return np.frombuffer(out, dtype=np.float32).reshape(frames, 2)


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def _settled(audio_output) -> np.ndarray:
    """What the card is handed once a volume change has been ramped in."""
    _played(audio_output)               # the buffer the change ramps over
    return _played(audio_output)


REFERENCE_DB = _rms_db(_tone())         # -9.03 dBFS


# ----------------------------------------------------------------------
# The curve
# ----------------------------------------------------------------------

@pytest.mark.parametrize("percent, gain_db", [
    (100, 0.0), (80, -5.81), (50, -18.06), (20, -41.94), (10, -60.0),
])
def test_the_curve_is_the_cube(percent, gain_db):
    assert 20 * np.log10(volume_gain(percent)) == pytest.approx(gain_db,
                                                                abs=0.01)


def test_zero_is_silence_not_just_quiet():
    assert volume_gain(0) == 0.0


@pytest.mark.parametrize("asked, got", [
    (-5, 0), (0, 0), (49.6, 50), (100, 100), (250, 100)])
def test_a_volume_is_clamped_to_a_whole_percent(asked, got):
    assert clamp_volume(asked) == got


def test_not_a_number_is_refused():
    with pytest.raises(ValueError):
        clamp_volume(float("nan"))


# ----------------------------------------------------------------------
# What the card is given
# ----------------------------------------------------------------------

def test_full_volume_plays_the_audio_untouched(audio_output):
    """The default is what the receiver played before there was a knob."""
    out = _played(audio_output)
    assert audio_output.volume == 100
    np.testing.assert_array_equal(out[:, 0], _tone())
    np.testing.assert_array_equal(out[:, 1], _tone())


@pytest.mark.parametrize("percent", [80, 50, 20, 10])
def test_the_card_gets_the_level_the_volume_says(audio_output, percent):
    audio_output.set_volume(percent)
    out = _settled(audio_output)
    expected = REFERENCE_DB + 20 * np.log10(volume_gain(percent))
    assert _rms_db(out[:, 0]) == pytest.approx(expected, abs=0.01)
    assert _rms_db(out[:, 1]) == pytest.approx(expected, abs=0.01)


def test_zero_plays_silence(audio_output):
    audio_output.set_volume(0)
    out = _settled(audio_output)
    assert not np.any(out)


def test_turning_it_back_up_restores_the_audio_exactly(audio_output):
    audio_output.set_volume(30)
    _settled(audio_output)
    audio_output.set_volume(100)
    out = _settled(audio_output)
    np.testing.assert_array_equal(out[:, 0], _tone())


def test_a_change_is_ramped_over_one_buffer_not_stepped(audio_output):
    """A step in the gain is a step in the waveform, which is a click.

    Played on DC rather than a tone so the gain can be read straight
    off the samples: it has to move frame by frame from where it was
    to where it is going, arriving on the last frame and not before.
    """
    dc = np.full(FRAMES, AMPLITUDE, dtype=np.float32)
    audio_output.enqueue_audio(dc, dc)
    audio_output.callback(None, FRAMES, {}, 0)          # at 100 %
    audio_output.set_volume(0)
    audio_output.enqueue_audio(dc, dc)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    gain = np.frombuffer(out, dtype=np.float32).reshape(FRAMES, 2) / AMPLITUDE

    np.testing.assert_array_equal(gain[:, 0], gain[:, 1])
    steps = np.diff(np.concatenate([[1.0], gain[:, 0]]))
    assert np.all(steps < 0), "the gain should fall on every frame"
    # One equal share of the whole fall each frame, so no step is
    # bigger than 1/1024 of full scale: -60 dB, inaudible.
    assert np.max(np.abs(steps)) == pytest.approx(1.0 / FRAMES, rel=1e-3)
    assert gain[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert gain[FRAMES // 2, 0] == pytest.approx(0.5, abs=2.0 / FRAMES)


def test_an_underrun_is_still_counted_and_still_silent_at_any_volume(
        audio_output):
    audio_output.set_volume(40)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert not np.any(np.frombuffer(out, dtype=np.float32))
    assert audio_output.underruns == 1


# ----------------------------------------------------------------------
# Mute
# ----------------------------------------------------------------------

def test_mute_plays_silence(audio_output):
    audio_output.set_volume(70)
    assert audio_output.set_muted(True) is True
    out = _settled(audio_output)
    assert not np.any(out)
    assert audio_output.muted
    assert audio_output.volume == 70, "the volume is kept to come back to"


def test_unmute_comes_back_at_the_volume_it_had(audio_output):
    audio_output.set_volume(50)
    audio_output.set_muted(True)
    _settled(audio_output)
    assert audio_output.set_muted(False) is False
    out = _settled(audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 18.06,
                                               abs=0.01)


def test_muting_is_ramped_not_stepped(audio_output):
    """Muting is a fall to zero, and a step to zero is the loudest click."""
    dc = np.full(FRAMES, AMPLITUDE, dtype=np.float32)
    audio_output.enqueue_audio(dc, dc)
    audio_output.callback(None, FRAMES, {}, 0)
    audio_output.set_muted(True)
    audio_output.enqueue_audio(dc, dc)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    gain = np.frombuffer(out, dtype=np.float32)[0::2] / AMPLITUDE
    steps = np.diff(np.concatenate([[1.0], gain]))
    assert np.max(np.abs(steps)) == pytest.approx(1.0 / FRAMES, rel=1e-3)
    assert gain[-1] == pytest.approx(0.0, abs=1e-6)


def test_setting_a_volume_while_muted_unmutes_at_it(audio_output):
    audio_output.set_muted(True)
    _settled(audio_output)
    audio_output.set_volume(20)
    assert not audio_output.muted
    out = _settled(audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 41.94,
                                               abs=0.01)


def test_a_recording_stays_at_full_level_while_muted(audio_output, tmp_path):
    path = tmp_path / "muted.wav"
    audio_output.set_muted(True)
    _settled(audio_output)
    audio_output.start_recording(str(path))
    tone = _tone()
    stereo = np.empty(FRAMES * 2, dtype=np.float32)
    stereo[0::2] = tone
    stereo[1::2] = tone
    audio_output.record(stereo)
    out = _played(audio_output)
    audio_output.stop_recording()

    with wave.open(str(path), "rb") as r:
        frames = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
    written = frames.astype(np.float64).reshape(-1, 2) / RECORD_MAX_INT16
    assert _rms_db(written[:, 0]) == pytest.approx(REFERENCE_DB, abs=0.01)
    assert not np.any(out)


# ----------------------------------------------------------------------
# What is not touched
# ----------------------------------------------------------------------

def test_a_recording_stays_at_full_level_whatever_is_played(audio_output,
                                                            tmp_path):
    """The recording is of the broadcast, not of the listener's knob.

    The same block goes to both, as the processing thread sends it:
    the card gets it 42 dB down and the file gets it as it was.
    """
    path = tmp_path / "full.wav"
    audio_output.set_volume(20)
    _settled(audio_output)
    audio_output.start_recording(str(path))
    tone = _tone()
    stereo = np.empty(FRAMES * 2, dtype=np.float32)
    stereo[0::2] = tone
    stereo[1::2] = tone
    audio_output.record(stereo)
    out = _played(audio_output)
    audio_output.stop_recording()

    with wave.open(str(path), "rb") as r:
        frames = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
    written = frames.astype(np.float64).reshape(-1, 2) / RECORD_MAX_INT16

    assert _rms_db(written[:, 0]) == pytest.approx(REFERENCE_DB, abs=0.01)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 41.94,
                                               abs=0.01)


# ----------------------------------------------------------------------
# Through the controller, the command line and the entry point
# ----------------------------------------------------------------------

@pytest.fixture
def receiver(no_user_config):
    """A controller on fake hardware whose threads are never started."""
    from fm_radio.controller import FMReceiverController
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def test_the_controller_sets_what_the_card_plays(receiver):
    assert receiver.get_volume() == 100
    assert receiver.set_volume(50) == 50
    assert receiver.get_volume() == 50
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 18.06,
                                               abs=0.01)


def test_the_controller_mutes_what_the_card_plays(receiver):
    receiver.set_volume(50)
    assert receiver.is_muted() is False
    assert receiver.set_muted(True) is True
    assert receiver.is_muted() is True
    assert not np.any(_settled(receiver.audio_output))
    receiver.set_muted(False)
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 18.06,
                                               abs=0.01)


def test_the_controller_clamps_rather_than_raising(receiver):
    assert receiver.set_volume(400) == 100
    assert receiver.set_volume(-1) == 0


@pytest.fixture
def cli(receiver, capsys):
    """A command line over the real receiver; run() is never called."""
    interface = CommandLineInterface(receiver)

    def say(line: str) -> str:
        capsys.readouterr()
        assert interface._dispatch(line) is True
        return capsys.readouterr().out

    return interface, say


def test_vol_on_the_command_line_sets_the_level_played(cli, receiver):
    _, say = cli
    assert "Volume: 20%" in say("vol 20")
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 41.94,
                                               abs=0.01)


def test_vol_plus_and_minus_step_it(cli, receiver):
    _, say = cli
    say("vol 50")
    assert f"Volume: {50 + VOLUME_STEP_PERCENT}%" in say("vol +")
    say("vol -")
    assert f"Volume: {50 - VOLUME_STEP_PERCENT}%" in say("VOL -")
    assert receiver.get_volume() == 50 - VOLUME_STEP_PERCENT


def test_vol_plus_stops_at_the_top(cli, receiver):
    _, say = cli
    assert "Volume: 100%" in say("vol +")
    assert receiver.get_volume() == 100


def test_bare_vol_says_where_it_is(cli, receiver):
    _, say = cli
    receiver.set_volume(35)
    assert "Volume: 35%" in say("volume")


@pytest.mark.parametrize("line", ["vol 500", "vol -3", "vol loud",
                                  "vol 1 2", "vol nan"])
def test_a_bad_vol_changes_nothing(cli, receiver, line):
    _, say = cli
    receiver.set_volume(60)
    said = say(line)
    assert "Volume:" not in said
    assert receiver.get_volume() == 60
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(
        REFERENCE_DB + 20 * np.log10(volume_gain(60)), abs=0.01)


def test_mute_on_the_command_line_silences_and_unmute_restores(cli,
                                                               receiver):
    _, say = cli
    say("vol 50")
    assert "Muted (volume 50% when unmuted)." in say("mute")
    assert not np.any(_settled(receiver.audio_output))
    assert "Volume: 50%" in say("unmute")
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 18.06,
                                               abs=0.01)


def test_bare_mute_toggles(cli, receiver):
    _, say = cli
    say("mute")
    assert receiver.is_muted()
    say("mute")
    assert not receiver.is_muted()


def test_mute_on_and_off_say_which(cli, receiver):
    _, say = cli
    say("mute on")
    say("mute on")
    assert receiver.is_muted(), "mute on twice is still muted"
    say("mute off")
    say("mute off")
    assert not receiver.is_muted()


def test_bare_vol_says_it_is_muted(cli, receiver):
    _, say = cli
    say("vol 35")
    say("mute")
    assert "Muted (volume 35% when unmuted)." in say("vol")


def test_vol_while_muted_unmutes_at_the_new_level(cli, receiver):
    _, say = cli
    say("mute")
    assert "Volume: 20%" in say("vol 20")
    out = _settled(receiver.audio_output)
    assert _rms_db(out[:, 0]) == pytest.approx(REFERENCE_DB - 41.94,
                                               abs=0.01)


@pytest.mark.parametrize("line", ["mute please", "unmute now", "mute 1"])
def test_a_bad_mute_changes_nothing(cli, receiver, line):
    _, say = cli
    assert "Usage: mute" in say(line)
    assert not receiver.is_muted()


def test_vol_is_a_word_not_a_prefix(cli, receiver):
    """'volt' is not 'vol t' - the same rule as 'searchlight'."""
    _, say = cli
    say("volt")
    assert receiver.get_volume() == 100


class _EntryController:
    """Stands in for the receiver main() builds; records the volume."""

    built: list["_EntryController"] = []

    def __init__(self, **kwargs) -> None:
        import threading
        self.quit_event = threading.Event()
        self.device_failure = None
        self.volumes: list[float] = []
        self.muted: list[bool] = []
        self.calls: list[str] = []
        _EntryController.built.append(self)

    def set_volume(self, percent):
        self.calls.append("set_volume")
        self.volumes.append(percent)
        return int(percent)

    def set_muted(self, muted):
        self.calls.append("set_muted")
        self.muted.append(muted)
        return muted

    def start(self) -> None:
        self.calls.append("start")


@pytest.fixture
def run_main(monkeypatch):
    import fm_radio.__main__ as entry
    _EntryController.built = []
    monkeypatch.setattr(entry, "FMReceiverController", _EntryController)

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["fm_receiver.py", *argv])
        try:
            entry.main()
        except SystemExit as exc:
            return exc.code
        return None

    return _run


def test_volume_flag_reaches_the_receiver_before_it_starts(run_main):
    assert run_main("--light", "--volume", "40") is None
    (controller,) = _EntryController.built
    assert controller.volumes == [40.0]
    assert controller.calls == ["set_volume", "start"]


def test_without_the_flag_the_volume_is_left_alone(run_main):
    run_main("--light")
    (controller,) = _EntryController.built
    assert controller.volumes == []
    assert controller.muted == []


@pytest.mark.parametrize("argv", [("--mute", "--volume", "40"),
                                  ("--volume", "40", "--mute")])
def test_mute_flag_wins_over_the_volume_whatever_the_order(run_main, argv):
    """The volume unmutes, so it has to go first to leave the radio muted."""
    run_main(*argv)
    (controller,) = _EntryController.built
    assert controller.calls == ["set_volume", "set_muted", "start"]
    assert controller.muted == [True]


@pytest.mark.parametrize("argv", [
    ("--volume", "loud"), ("--volume", "101"), ("--volume", "-1"),
    ("--volume",), ("--volume", "nan")])
def test_a_bad_volume_flag_stops_before_anything_is_opened(run_main, capsys,
                                                           argv):
    assert run_main(*argv) == 2
    assert _EntryController.built == []
    assert "--volume needs a number from 0 to 100" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Coming out of a silence the volume never saw
# ----------------------------------------------------------------------

def _dc(frames: int = 4096) -> np.ndarray:
    return np.full(frames, AMPLITUDE, dtype=np.float32)


def _start_playing(audio_output) -> None:
    """Queue the preroll in DC so the stream starts, as at startup."""
    while not audio_output._playing:
        audio_output.enqueue_audio(_dc(), _dc())


def _first_audible(audio_output) -> np.ndarray:
    """Left channel of the first callback that is not all silence."""
    for _ in range(200):
        out, _ = audio_output.callback(None, FRAMES, {}, 0)
        left = np.frombuffer(out, dtype=np.float32)[0::2]
        if np.any(left) or not audio_output._refilling:
            return left
    raise AssertionError("the refill never ended")


def _into_a_refill(audio_output) -> None:
    """Playing at full volume, then a retune's silence starts."""
    _start_playing(audio_output)
    while audio_output.audio_buffer_queue.qsize() or audio_output._buffer_len:
        audio_output.callback(None, FRAMES, {}, 0)       # heard at 100 %
    audio_output.rebuild_cushion()
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert audio_output._refilling, "the refill was meant to be under way"
    assert not np.any(np.frombuffer(out, dtype=np.float32))


def _out_of_the_refill(audio_output) -> np.ndarray:
    for _ in range(8):
        audio_output.enqueue_audio(_dc(), _dc())
    return _first_audible(audio_output)


def test_muting_during_a_refill_stays_silent_after_it(audio_output):
    """The refill's silence did not pass through the volume, so the gain
    it left behind is the one from before - and a ramp from there to
    zero is half a buffer of the station the listener just muted."""
    _into_a_refill(audio_output)
    audio_output.set_muted(True)
    left = _out_of_the_refill(audio_output)
    assert not audio_output._refilling
    assert not np.any(left), (
        "muted, yet the first frame after the refill was %.4f" % left[0])


def test_a_volume_set_during_a_refill_is_where_the_audio_comes_back(
        audio_output):
    _into_a_refill(audio_output)
    audio_output.set_volume(20)
    left = _out_of_the_refill(audio_output)
    np.testing.assert_allclose(left, AMPLITUDE * volume_gain(20), rtol=1e-5)


def test_a_volume_set_before_the_stream_starts_is_its_first_frame(
        audio_output):
    """Nothing has been played to ramp from: the first frame is the volume."""
    audio_output.set_volume(20)
    _start_playing(audio_output)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    left = np.frombuffer(out, dtype=np.float32)[0::2]
    np.testing.assert_allclose(left, AMPLITUDE * volume_gain(20), rtol=1e-5)


def test_a_mute_set_while_held_is_silent_when_the_stream_restarts(
        audio_output):
    """A hold stops the card; the restart is a start like the first."""
    _start_playing(audio_output)
    audio_output.callback(None, FRAMES, {}, 0)           # heard at 100 %
    audio_output.hold()
    audio_output.set_muted(True)
    audio_output.resume()
    _start_playing(audio_output)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert not np.any(np.frombuffer(out, dtype=np.float32))


# ----------------------------------------------------------------------
# Two setters at once
# ----------------------------------------------------------------------

def test_a_mute_landing_inside_a_volume_change_is_not_lost(audio_output,
                                                          monkeypatch):
    """set_volume clears the mute and then works out the gain.  A mute
    landing between the two used to be overwritten by that gain: muted
    by every account and playing all the same.

    The interleaving is forced, not hoped for: set_volume is stopped
    inside its gain lookup until set_muted has been given every chance
    to run.
    """
    import threading
    import fm_radio.audio_output as ao_mod

    inside, go = threading.Event(), threading.Event()
    real = ao_mod.volume_gain

    def held(percent):
        if threading.current_thread().name == "volume":
            inside.set()
            assert go.wait(5.0)
        return real(percent)

    monkeypatch.setattr(ao_mod, "volume_gain", held)
    volume = threading.Thread(target=audio_output.set_volume, args=(30,),
                              name="volume")
    volume.start()
    assert inside.wait(5.0)
    mute = threading.Thread(target=audio_output.set_muted, args=(True,),
                            name="mute")
    mute.start()
    mute.join(0.2)              # done by now unless it is made to wait
    go.set()
    volume.join(5.0)
    mute.join(5.0)

    assert audio_output.muted
    assert audio_output.volume == 30
    assert not np.any(_settled(audio_output)), (
        "muted, and still playing at %.3f" % audio_output._gain_wanted)



def test_muting_during_a_refill_stays_silent_on_the_settle_path(audio_output):
    _into_a_refill(audio_output)
    audio_output.set_muted(True)
    frames = audio_output._cushion_frames + FRAMES + 16
    audio_output.enqueue_audio(_dc(frames), _dc(frames))
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert audio_output._refilling, (
        "the refill was meant to stay armed: this is the settle path")
    left = np.frombuffer(out, dtype=np.float32)[0::2]
    assert not np.any(left), (
        "muted, yet the first frame after the silence was %.4f" % left[0])


def test_a_volume_set_during_a_refill_holds_on_the_settle_path(audio_output):
    _into_a_refill(audio_output)
    audio_output.set_volume(20)
    frames = audio_output._cushion_frames + FRAMES + 16
    audio_output.enqueue_audio(_dc(frames), _dc(frames))
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert audio_output._refilling
    left = np.frombuffer(out, dtype=np.float32)[0::2]
    np.testing.assert_allclose(left, AMPLITUDE * volume_gain(20), rtol=1e-5)


def test_a_refill_that_played_no_silence_still_ramps(audio_output):
    """Heard audio before, heard audio after: the ramp is the right one.

    The refill is armed, but the cushion is already there, so the next
    buffer is audio through the settle branch with no silence before
    it.  Snapping the gain there would be a step from what was just
    heard - the click the ramp exists to prevent.
    """
    _start_playing(audio_output)
    audio_output.callback(None, FRAMES, {}, 0)           # heard at 100 %
    audio_output.rebuild_cushion()
    audio_output.set_volume(0)
    out, _ = audio_output.callback(None, FRAMES, {}, 0)
    assert audio_output._refilling, "meant to be the settle branch"
    gain = np.frombuffer(out, dtype=np.float32)[0::2] / AMPLITUDE
    steps = np.diff(np.concatenate([[1.0], gain]))
    assert np.max(np.abs(steps)) == pytest.approx(1.0 / FRAMES, rel=1e-3)
