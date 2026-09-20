"""AudioOutput async recording behaviour.

Covers the audio-dropout fixes and their codex-review follow-ups:
realtime record() must never block on disk, stop_recording must flush
the queued tail, duplicate/concurrent starts must not truncate existing
files, WAV rotation at the 4-GiB threshold, and worker survival on
unexpected write errors.
"""

from __future__ import annotations

import os
import queue
import struct
import threading
import time
import wave as wave_mod

import numpy as np
import pytest

from fm_radio.exceptions import RecordingError

import fm_radio.audio_output as ao_mod
from fm_radio.constants import (
    AUDIO_CARD_BUFFER_MAX_SEC, AUDIO_CHANNELS, AUDIO_FRAMES_PER_BUFFER,
    AUDIO_OUTPUT_RATE,
)


CHUNK = np.zeros(768 * 2, dtype=np.float32) + 0.25
CHUNK_DATA_BYTES = CHUNK.size * 2  # int16


def _wav_frames(path):
    with wave_mod.open(str(path), "rb") as r:
        return r.getnframes()


def test_record_not_blocked_by_slow_disk(audio_output, tmp_path):
    ao = audio_output
    ao.start_recording(str(tmp_path / "a.wav"))
    orig = ao.record_wave.writeframes

    def slow_write(data):
        time.sleep(0.5)
        return orig(data)

    ao.record_wave.writeframes = slow_write
    worst = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        ao.record(CHUNK.copy())
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 0.2
    ao.stop_recording()


def test_stop_recording_flushes_queued_tail(audio_output, tmp_path):
    ao = audio_output
    path = tmp_path / "flush.wav"
    ao.start_recording(str(path))
    orig = ao.record_wave.writeframes
    n_writes = [0]

    def slow_write(data):
        n_writes[0] += 1
        time.sleep(0.1)
        return orig(data)

    ao.record_wave.writeframes = slow_write
    for _ in range(5):
        ao.record(CHUNK.copy())
    ao.stop_recording()  # must block until all 5 writes have happened
    assert n_writes[0] == 5
    assert _wav_frames(path) == 5 * 768


def test_single_chunk_then_immediate_stop_is_not_lost(audio_output, tmp_path):
    # Codex repro from PR #2 review: 1 chunk queued -> stop -> 0 writes.
    ao = audio_output
    path = tmp_path / "one.wav"
    ao.start_recording(str(path))
    orig = ao.record_wave.writeframes
    n_writes = [0]

    def slow_first(data):
        n_writes[0] += 1
        time.sleep(0.3)
        return orig(data)

    ao.record_wave.writeframes = slow_first
    ao.record(CHUNK.copy())
    time.sleep(0.01)  # let the worker pop mid-write
    ao.stop_recording()
    assert n_writes[0] == 1
    assert _wav_frames(path) == 768


def test_duplicate_start_does_not_truncate_target(audio_output, tmp_path):
    ao = audio_output
    victim = tmp_path / "victim.wav"
    victim.write_bytes(b"X" * 1200)
    ao.start_recording(str(tmp_path / "first.wav"))
    ao.start_recording(str(victim))  # duplicate: must be a strict no-op
    assert victim.stat().st_size == 1200
    ao.stop_recording()


def test_concurrent_starts_leave_exactly_one_recording(audio_output,
                                                       tmp_path):
    """Two at once: one records, and the other leaves nothing behind.

    Neither may touch a file that was already there - a start makes
    its file rather than taking one over - so the loser's path is the
    one it never created.
    """
    ao = audio_output
    f1 = tmp_path / "c1.wav"
    f2 = tmp_path / "c2.wav"

    barrier = threading.Barrier(2)
    refused: list = []

    def starter(path):
        barrier.wait()
        try:
            ao.start_recording(str(path))
        except RecordingError as e:
            refused.append(e)

    threads = [threading.Thread(target=starter, args=(f,)) for f in (f1, f2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    made = [f for f in (f1, f2) if f.exists()]
    assert len(made) == 1, f"{len(made)} files were made: {made}"
    assert ao._record_base_path == str(made[0])
    ao.stop_recording()
    assert made[0].stat().st_size > 0


def test_a_start_will_not_take_over_a_file_that_is_already_there(
        audio_output, tmp_path):
    """The file it would truncate can be one somebody is recording to."""
    ao = audio_output
    path = tmp_path / "theirs.wav"
    path.write_bytes(b"X" * 1200)

    with pytest.raises(RecordingError):
        ao.start_recording(str(path))

    assert path.stat().st_size == 1200, "it truncated a file it did not make"
    assert not ao.recording


def test_rotation_preserves_every_sample(audio_output, tmp_path, monkeypatch):
    # ~16 chunks per part at a 50 kB threshold.
    monkeypatch.setattr(ao_mod, "AUDIO_RECORD_ROTATE_THRESHOLD_BYTES", 50_000)
    ao = audio_output
    base = tmp_path / "rot.wav"
    ao.start_recording(str(base))
    n_chunks = 50
    for _ in range(n_chunks):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    files = sorted(
        p for p in tmp_path.iterdir()
        if p.name.startswith("rot") and p.suffix == ".wav"
    )
    assert len(files) >= 2, "rotation must have produced part files"
    total_frames = sum(_wav_frames(p) for p in files)
    assert total_frames == n_chunks * 768


def test_worker_survives_unexpected_write_error(audio_output, tmp_path):
    ao = audio_output
    ao.start_recording(str(tmp_path / "err.wav"))
    orig = ao.record_wave.writeframes
    n_calls = [0]

    def failing_write(data):
        n_calls[0] += 1
        if n_calls[0] == 2:
            raise struct.error("argument out of range")
        return orig(data)

    ao.record_wave.writeframes = failing_write
    for _ in range(4):
        ao.record(CHUNK.copy())
        time.sleep(0.02)
    time.sleep(0.3)
    assert ao._record_worker.is_alive()
    assert n_calls[0] == 4  # the failing call did not kill the loop
    ao.stop_recording()


# ----------------------------------------------------------------------
# Output-side health counters
# ----------------------------------------------------------------------

def drain_into_buffer(audio_output, frames: int) -> None:
    """Hand the output *frames* frames and let its queue settle."""
    block = np.zeros(frames, dtype=np.float32)
    audio_output.enqueue_audio(block, block)
    deadline = time.monotonic() + 2.0
    while (audio_output.audio_buffer_queue.qsize()
           and time.monotonic() < deadline):
        time.sleep(0.005)


def test_a_full_callback_is_not_an_underrun(audio_output):
    drain_into_buffer(audio_output, 384)
    out, _ = audio_output.callback(None, 384, {}, 0)

    assert np.frombuffer(out, dtype=np.float32).size == 384 * 2
    assert audio_output.underruns == 0


def test_a_partly_filled_callback_counts_as_an_underrun(audio_output):
    """Half a buffer of silence is an audible gap just like a whole one."""
    drain_into_buffer(audio_output, 192)
    out, _ = audio_output.callback(None, 384, {}, 0)

    samples = np.frombuffer(out, dtype=np.float32)
    assert samples.size == 384 * 2
    assert np.all(samples[192 * 2:] == 0.0), "the tail should be silence"
    assert audio_output.underruns == 1


def test_an_empty_callback_counts_as_an_underrun(audio_output):
    out, _ = audio_output.callback(None, 384, {}, 0)

    samples = np.frombuffer(out, dtype=np.float32)
    assert samples.size == 384 * 2
    assert np.all(samples == 0.0)
    assert audio_output.underruns == 1


def test_underruns_accumulate(audio_output):
    audio_output.callback(None, 384, {}, 0)
    drain_into_buffer(audio_output, 192)
    audio_output.callback(None, 384, {}, 0)
    assert audio_output.underruns == 2


def test_a_dropped_block_is_counted(audio_output, monkeypatch):
    """The output queue being full was only ever a debug log."""
    def full(*args, **kwargs):
        raise queue.Full

    monkeypatch.setattr(audio_output.audio_buffer_queue, "put", full)
    block = np.zeros(192, dtype=np.float32)
    audio_output.enqueue_audio(block, block)
    audio_output.enqueue_audio(block, block)

    assert audio_output.dropped_blocks == 2


# ----------------------------------------------------------------------
# Nothing plays until there is something to play
# ----------------------------------------------------------------------

def test_the_stream_does_not_run_before_there_is_audio(audio_output):
    """A running stream with nothing behind it is an underrun a buffer.

    On this machine there were forty-three of them between the
    receiver being built and its first block arriving, most of them
    during the JIT pre-warm.
    """
    assert audio_output.stream.started is False
    assert audio_output._playing is False


#: A block the size the receiver really produces: one SDR block of
#: 16384 samples at 1.024 MHz is 16 ms, which is 768 frames at
#: 48 kHz, and the card asks for 1024 at a time.
BLOCK_FRAMES = 768

#: The two modes, as (name, seconds between blocks).  Light mode
#: reads the same 16384 samples at 250 kHz, so its blocks are four
#: times as far apart and four times as big.
MODES = [("standard", 16384 / 1.024e6), ("light", 16384 / 0.25e6)]


def _feed(audio_output, frames, block=BLOCK_FRAMES):
    """Enqueue at least ``frames`` frames, a receiver block at a time."""
    sent = 0
    while sent < frames:
        one = np.zeros(block, dtype=np.float32)
        audio_output.enqueue_audio(one, one)
        sent += block
    return sent


def _play_out(audio_output, interval_sec, seconds=2.0, card_sec=0.1067):
    """Run the producer and the card against each other on the clock.

    No sleeping and no jitter.  Blocks go in every ``interval_sec``
    of simulated time; the card fills its own buffer the instant the
    stream starts, calling back as fast as it can until it holds
    ``card_sec``, and asks every 1024 frames' worth after that.  That
    is what a real one does - five callbacks in the first 39 ms on
    the USB DAC these numbers come from - and it is the part that was
    missed: that burst empties any cushion smaller than the card's
    buffer, at the one moment there is nothing else to draw on.
    """
    rate = float(AUDIO_OUTPUT_RATE)
    callback_sec = AUDIO_FRAMES_PER_BUFFER / rate
    block = 0                       # next block's index
    produced = 0                    # frames handed over so far
    t_block = 0.0
    t_callback = None               # set when the card starts asking
    while min(t_block, t_callback if t_callback is not None else t_block)             <= seconds:
        # The card goes first when the two fall together: a block due
        # at the same instant may be a hair late, and that is the
        # case that costs a gap.
        if t_callback is not None and t_callback <= t_block:
            if t_callback > seconds:
                break
            audio_output.callback(None, AUDIO_FRAMES_PER_BUFFER, {}, 0)
            t_callback += callback_sec
            continue
        if t_block > seconds:
            break
        # The receiver produces at exactly the audio rate; the block
        # boundaries only decide how it is parcelled up, so each one
        # carries whatever is needed to keep the running total right.
        block += 1
        want = int(round(block * interval_sec * rate))
        one = np.zeros(want - produced, dtype=np.float32)
        produced = want
        audio_output.enqueue_audio(one, one)
        if t_callback is None and audio_output._playing:
            # The card fills itself here, back to back, and only
            # settles into asking on the clock afterwards.
            for _ in range(int(round(card_sec * rate))
                           // AUDIO_FRAMES_PER_BUFFER):
                audio_output.callback(None, AUDIO_FRAMES_PER_BUFFER, {}, 0)
            t_callback = t_block
        t_block += interval_sec


def test_one_block_is_not_enough_to_start_on(audio_output):
    """The receiver produces at exactly the rate the card consumes.

    768 frames every 16 ms against 1024 every 21.3: the same rate, so
    a stream started on the first block never gets ahead, and the
    first thing that runs late is a gap.
    """
    left = np.zeros(BLOCK_FRAMES, dtype=np.float32)

    audio_output.enqueue_audio(left, left)

    assert audio_output.stream.started is False
    assert audio_output._playing is False


def test_the_stream_starts_once_there_is_a_cushion(audio_output):
    _feed(audio_output, audio_output._preroll_frames)

    assert audio_output.stream.started is True
    assert audio_output._playing is True


@pytest.mark.parametrize("name, interval", MODES)
def test_the_card_is_never_left_short_in_either_mode(name, interval):
    """The whole point, measured the way it goes wrong: on the clock.

    Blocks arrive every interval and the card asks every 21.3 ms
    from the moment the stream starts.  Nothing here is late - there
    is no jitter in this at all - and a cushion that does not cover
    a block interval still runs dry, because the two schedules do
    not line up.  In light mode blocks are 65.5 ms apart, so a
    stream started on one block is empty by the fourth callback.
    """
    ao = ao_mod.AudioOutput(block_interval_sec=interval)
    try:
        _play_out(ao, interval)

        assert ao._playing, "the stream never started"
        assert ao.underruns == 0, (
            "%s mode: %d gaps with nothing late, starting on %d frames"
            % (name, ao.underruns, ao._preroll_frames))
    finally:
        ao.cleanup()


def _the_stream_class():
    """The stream class the output will really be handed.

    Taken from the fake pyaudio rather than imported from conftest:
    tests/ is not a package, so importing it by name would make a
    second copy of the module and patch a class nobody is given.
    """
    return type(ao_mod.pyaudio.PyAudio().open())


def _with_a_card_that_holds(monkeypatch, held):
    """An output whose stream says the device holds ``held`` seconds."""
    monkeypatch.setattr(_the_stream_class(), "output_latency", held)
    return ao_mod.AudioOutput()


def test_the_cushion_sits_on_top_of_what_the_card_holds(monkeypatch):
    """Or the card swallows the cushion filling itself.

    PortAudio fills the device's buffer the moment the stream is
    started - five callbacks back to back on the DAC these numbers
    come from - and none of that is playing time.
    """
    ao = _with_a_card_that_holds(monkeypatch, 0.1067)
    try:
        held = int(round(0.1067 * AUDIO_OUTPUT_RATE))

        assert ao._preroll_frames == ao._cushion_frames + held, (
            "starts on %d; the cushion is %d and the card takes %d"
            % (ao._preroll_frames, ao._cushion_frames, held))
    finally:
        ao.cleanup()


@pytest.mark.parametrize("held", [0.0, -1.0])
def test_a_card_that_says_nothing_useful_is_taken_as_holding_nothing(
        monkeypatch, held):
    """Not every host API answers, and none of them have to."""
    ao = _with_a_card_that_holds(monkeypatch, held)
    try:
        assert ao._preroll_frames == ao._cushion_frames
    finally:
        ao.cleanup()


def test_a_card_that_will_not_say_is_taken_as_holding_nothing(monkeypatch):
    """A stream that raises is a host API that does not keep the figure."""

    def refuse(self):
        raise OSError("no such thing here")

    monkeypatch.setattr(_the_stream_class(), "get_output_latency", refuse)
    ao = ao_mod.AudioOutput()
    try:
        assert ao._preroll_frames == ao._cushion_frames
    finally:
        ao.cleanup()


def test_a_card_that_claims_far_too_much_is_not_believed(monkeypatch):
    """Waiting out the claim would be worse than the gaps it saves.

    A device that says it holds a second and a half would mean a
    second and a half of silence before the radio started.
    """
    ao = _with_a_card_that_holds(monkeypatch, 1.5)
    try:
        ceiling = int(round(AUDIO_CARD_BUFFER_MAX_SEC * AUDIO_OUTPUT_RATE))

        assert ao._preroll_frames == ao._cushion_frames + ceiling
    finally:
        ao.cleanup()


def test_an_empty_block_takes_no_place_in_the_queue(audio_output):
    """The first block out of the demodulator after a reset has none.

    Every retune resets it, so these arrive in runs, and a queue of
    fifty holds fifty of them exactly as well as it holds fifty
    blocks of audio: see the test below for what that costs.
    """
    empty = np.zeros(0, dtype=np.float32)

    for _ in range(80):
        audio_output.enqueue_audio(empty, empty)

    assert audio_output.audio_buffer_queue.qsize() == 0
    assert audio_output.dropped_blocks == 0, "they were not dropped, either"


def test_a_queue_that_cannot_take_more_is_enough_to_start_on(audio_output):
    """Or the radio waits for a cushion that can never arrive.

    The threshold is frames and the queue is blocks, so a run of
    short blocks can fill the queue without ever reaching it.  Every
    block after that is dropped, which means the count stops moving,
    which means the stream never starts: silent until it is
    restarted, with audio arriving the whole time.
    """
    one = np.zeros(1, dtype=np.float32)
    room = audio_output.audio_buffer_queue.maxsize

    for _ in range(room + 5):
        audio_output.enqueue_audio(one, one)

    assert audio_output._frames_ready < audio_output._preroll_frames, (
        "this test is meant to fill the queue before the cushion")
    assert audio_output._playing, "the radio would never have played"
    assert audio_output.stream.started is True


def test_the_radio_plays_again_after_a_run_of_retunes(audio_output):
    """The way the empty blocks really arrive, end to end.

    A retune resets the demodulator and its next block carries no
    audio.  Enough retunes before the stream has started - each one
    the only block of its generation - and the queue is full of
    nothing.  What has to happen is that ordinary audio afterwards
    still gets the radio playing.
    """
    from fm_radio.demodulator import FMDemodulator
    from fm_radio.constants import SDR_BLOCK_SIZE, SDR_SAMPLE_RATE

    demod = FMDemodulator(iq_sample_rate=SDR_SAMPLE_RATE,
                          final_audio_rate=AUDIO_OUTPUT_RATE, stereo=True)
    iq = np.zeros(SDR_BLOCK_SIZE, dtype=np.complex64)
    empties = 0
    for _ in range(60):
        demod.reset()                       # a retune
        left, right = demod.demodulate(demod.process_iq_samples(iq))
        empties += left.size == 0
        audio_output.enqueue_audio(left, right)

    assert empties == 60, "the demodulator no longer starts empty"
    assert not audio_output._playing, "nothing had any audio in it"

    for _ in range(60):                     # the station settles
        left, right = demod.demodulate(demod.process_iq_samples(iq))
        audio_output.enqueue_audio(left, right)

    assert audio_output._playing, "the audio came back and the radio did not"


@pytest.mark.parametrize("name, interval", MODES)
def test_the_cushion_covers_a_whole_block_interval(name, interval):
    """Where the number comes from, said as arithmetic.

    The deficit at the worst moment is everything the card takes
    during one block interval - the rates are equal, so that is one
    block - and the callbacks do not line up with the blocks, so one
    of them can fall entirely inside that moment.
    """
    ao = ao_mod.AudioOutput(block_interval_sec=interval)
    try:
        one_interval = interval * AUDIO_OUTPUT_RATE

        assert ao._preroll_frames >= one_interval + AUDIO_FRAMES_PER_BUFFER, (
            "%s mode starts on %d frames; a block interval is %.0f and a "
            "callback %d" % (name, ao._preroll_frames, one_interval,
                             AUDIO_FRAMES_PER_BUFFER))
    finally:
        ao.cleanup()


def test_a_whole_callback_is_still_in_hand_after_the_first_one(audio_output):
    """What the card asks for first is there, and so is the next one.

    The second one is the point.  The receiver never catches up from
    behind - it produces at exactly the rate the card consumes - so
    whatever is in hand when the stream starts is all there will ever
    be to absorb a late block, and one callback of it is used up
    immediately.

    Asked through the real callback rather than by counting what is in
    the queue: a cushion the callback cannot actually draw on is not a
    cushion.
    """
    _feed(audio_output, audio_output._preroll_frames)
    before = audio_output.underruns

    audio_output.callback(None, AUDIO_FRAMES_PER_BUFFER, {}, 0)

    assert audio_output.underruns == before, "the first callback went short"
    left_over = (audio_output._buffer_len
                 + sum(a.size for a, _ in
                       list(audio_output.audio_buffer_queue.queue)) * 2)
    assert left_over >= AUDIO_FRAMES_PER_BUFFER * AUDIO_CHANNELS, (
        "only %d samples behind the first buffer, less than the %d the "
        "next callback will ask for"
        % (left_over, AUDIO_FRAMES_PER_BUFFER * AUDIO_CHANNELS))


def test_the_blocks_are_queued_before_the_stream_is_started(audio_output):
    """Or the card asks before there is anything to give it."""
    when = []
    real_start = audio_output.stream.start_stream
    real_put = audio_output.audio_buffer_queue.put

    def watched_start():
        when.append("start")
        return real_start()

    def watched_put(*args, **kwargs):
        when.append("queue")
        return real_put(*args, **kwargs)

    audio_output.stream.start_stream = watched_start
    audio_output.audio_buffer_queue.put = watched_put

    _feed(audio_output, audio_output._preroll_frames)

    assert when[-1] == "start", when
    assert when.count("start") == 1, when
    assert (when.count("queue") * BLOCK_FRAMES
            >= audio_output._preroll_frames), when


def test_the_stream_is_only_started_once(audio_output):
    starts = []
    real_start = audio_output.stream.start_stream
    audio_output.stream.start_stream = lambda: (starts.append(1),
                                                real_start())

    _feed(audio_output, audio_output._preroll_frames * 3)

    assert len(starts) == 1, f"started {len(starts)} times"


def test_a_closed_output_does_not_start_the_stream(audio_output):
    """A block arriving after shutdown has nowhere to go."""
    audio_output.cleanup()

    _feed(audio_output, audio_output._preroll_frames)

    assert audio_output.stream.started is False


def test_cleanup_does_not_stop_a_stream_that_never_started(audio_output):
    """PortAudio is entitled to object to being asked."""
    audio_output.cleanup()

    assert audio_output.stream.stopped is False


def test_cleanup_stops_a_stream_that_did_start(audio_output):
    _feed(audio_output, audio_output._preroll_frames)

    audio_output.cleanup()

    assert audio_output.stream.stopped is True
