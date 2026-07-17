"""AudioOutput async recording behaviour.

Covers the audio-dropout fixes and their codex-review follow-ups:
realtime record() must never block on disk, stop_recording must flush
the queued tail, duplicate/concurrent starts must not truncate existing
files, WAV rotation at the 4-GiB threshold, and worker survival on
unexpected write errors.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import wave as wave_mod

import numpy as np
import pytest

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


def test_concurrent_start_truncates_exactly_one_file(audio_output, tmp_path):
    ao = audio_output
    f1 = tmp_path / "c1.wav"
    f2 = tmp_path / "c2.wav"
    for f in (f1, f2):
        f.write_bytes(b"X" * 1200)

    barrier = threading.Barrier(2)

    def starter(path):
        barrier.wait()
        ao.start_recording(str(path))

    threads = [threading.Thread(target=starter, args=(f,)) for f in (f1, f2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    ao.stop_recording()  # winner's file gets a valid header on close

    sizes = sorted([f1.stat().st_size, f2.stat().st_size])
    assert sizes[1] == 1200, "loser's file must be untouched"
    assert sizes[0] != 1200, "winner's file must have been recreated"


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

    files = sorted(p for p in tmp_path.iterdir() if p.name.startswith("rot"))
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
