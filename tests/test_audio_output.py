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


def test_the_first_block_starts_the_stream(audio_output):
    left = np.zeros(256, dtype=np.float32)

    audio_output.enqueue_audio(left, left)

    assert audio_output.stream.started is True
    assert audio_output._playing is True


def test_the_block_is_queued_before_the_stream_is_started(audio_output):
    """Or the card asks once before there is anything to give it."""
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

    left = np.zeros(256, dtype=np.float32)
    audio_output.enqueue_audio(left, left)

    assert when == ["queue", "start"], when


def test_the_stream_is_only_started_once(audio_output):
    starts = []
    real_start = audio_output.stream.start_stream
    audio_output.stream.start_stream = lambda: (starts.append(1),
                                                real_start())

    left = np.zeros(256, dtype=np.float32)
    for _ in range(5):
        audio_output.enqueue_audio(left, left)

    assert len(starts) == 1, f"started {len(starts)} times"


def test_a_closed_output_does_not_start_the_stream(audio_output):
    """A block arriving after shutdown has nowhere to go."""
    audio_output.cleanup()
    left = np.zeros(256, dtype=np.float32)

    audio_output.enqueue_audio(left, left)

    assert audio_output.stream.started is False


def test_cleanup_does_not_stop_a_stream_that_never_started(audio_output):
    """PortAudio is entitled to object to being asked."""
    audio_output.cleanup()

    assert audio_output.stream.stopped is False


def test_cleanup_stops_a_stream_that_did_start(audio_output):
    left = np.zeros(256, dtype=np.float32)
    audio_output.enqueue_audio(left, left)

    audio_output.cleanup()

    assert audio_output.stream.stopped is True
