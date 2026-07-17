"""SDRReceiver async IQ-recording behaviour (mirror of the audio path).

The SDR callback runs on the rtlsdr library thread and must never block
on disk I/O; stop_iq_recording must flush the queued tail; rotation must
kick in before the WAV 4-GiB limit; a failing write must not kill the
worker thread.
"""

from __future__ import annotations

import struct
import threading
import time
import wave as wave_mod

import numpy as np
import pytest

import fm_radio.sdr_receiver as sr_mod


IQ_BLOCK = (np.zeros(16384) + 0.1 + 0.05j).astype(np.complex64)
BLOCK_FRAMES = 16384


def _wav_frames(path):
    with wave_mod.open(str(path), "rb") as r:
        return r.getnframes()


def test_callback_not_blocked_by_slow_disk(sdr_receiver, tmp_path):
    recv = sdr_receiver
    recv.start_iq_recording(str(tmp_path / "iq.wav"))
    orig = recv.iq_record_wave.writeframes

    def slow_write(data):
        time.sleep(0.5)
        return orig(data)

    recv.iq_record_wave.writeframes = slow_write
    worst = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        recv.callback(IQ_BLOCK.copy(), None)
        worst = max(worst, time.perf_counter() - t0)
        try:
            recv.data_queue.get_nowait()  # keep the demod queue drained
        except Exception:
            pass
    assert worst < 0.2
    recv.stop_iq_recording()


def test_stop_flushes_queued_tail(sdr_receiver, tmp_path):
    recv = sdr_receiver
    path = tmp_path / "flush.wav"
    recv.start_iq_recording(str(path))
    orig = recv.iq_record_wave.writeframes
    n_writes = [0]

    def slow_write(data):
        n_writes[0] += 1
        time.sleep(0.1)
        return orig(data)

    recv.iq_record_wave.writeframes = slow_write
    for _ in range(5):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
    recv.stop_iq_recording()
    assert n_writes[0] == 5
    assert _wav_frames(path) == 5 * BLOCK_FRAMES


def test_callback_not_blocked_by_slow_start(sdr_receiver, tmp_path,
                                            monkeypatch):
    # PR #2 codex repro: wave.open stall must not propagate to callback.
    recv = sdr_receiver
    orig_open = sr_mod.wave.open

    def slow_open(*args, **kwargs):
        time.sleep(0.4)
        return orig_open(*args, **kwargs)

    monkeypatch.setattr(sr_mod.wave, "open", slow_open)
    t = threading.Thread(
        target=recv.start_iq_recording, args=(str(tmp_path / "s.wav"),),
    )
    t.start()
    time.sleep(0.05)  # let start reach wave.open
    t0 = time.perf_counter()
    recv.callback(IQ_BLOCK.copy(), None)
    dt = time.perf_counter() - t0
    t.join(timeout=5)
    assert dt < 0.2
    recv.stop_iq_recording()


def test_duplicate_start_does_not_truncate_target(sdr_receiver, tmp_path):
    recv = sdr_receiver
    victim = tmp_path / "victim.wav"
    victim.write_bytes(b"Y" * 1200)
    recv.start_iq_recording(str(tmp_path / "first.wav"))
    recv.start_iq_recording(str(victim))
    assert victim.stat().st_size == 1200
    recv.stop_iq_recording()


def test_concurrent_start_truncates_exactly_one_file(sdr_receiver, tmp_path):
    recv = sdr_receiver
    f1 = tmp_path / "c1.wav"
    f2 = tmp_path / "c2.wav"
    for f in (f1, f2):
        f.write_bytes(b"Y" * 1200)

    barrier = threading.Barrier(2)

    def starter(path):
        barrier.wait()
        recv.start_iq_recording(str(path))

    threads = [threading.Thread(target=starter, args=(f,)) for f in (f1, f2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    recv.stop_iq_recording()

    sizes = sorted([f1.stat().st_size, f2.stat().st_size])
    assert sizes[1] == 1200
    assert sizes[0] != 1200


def test_rotation_preserves_every_sample(sdr_receiver, tmp_path, monkeypatch):
    # Each block is 16384*2ch*2B = 64 kB; ~3 blocks per part at 200 kB.
    monkeypatch.setattr(sr_mod, "IQ_RECORD_ROTATE_THRESHOLD_BYTES", 200_000)
    recv = sdr_receiver
    recv.start_iq_recording(str(tmp_path / "rot.wav"))
    n_blocks = 10
    for _ in range(n_blocks):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
    recv.stop_iq_recording()

    files = sorted(
        p for p in tmp_path.iterdir()
        if p.name.startswith("rot") and p.suffix == ".wav"
    )
    assert len(files) >= 2
    total_frames = sum(_wav_frames(p) for p in files)
    assert total_frames == n_blocks * BLOCK_FRAMES


def test_worker_survives_unexpected_write_error(sdr_receiver, tmp_path):
    recv = sdr_receiver
    recv.start_iq_recording(str(tmp_path / "err.wav"))
    orig = recv.iq_record_wave.writeframes
    n_calls = [0]

    def failing_write(data):
        n_calls[0] += 1
        if n_calls[0] == 2:
            raise struct.error("argument out of range")
        return orig(data)

    recv.iq_record_wave.writeframes = failing_write
    for _ in range(4):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
        time.sleep(0.02)
    time.sleep(0.3)
    assert recv._iq_record_worker.is_alive()
    assert n_calls[0] == 4
    recv.stop_iq_recording()


def test_stop_shuts_down_worker(sdr_receiver):
    recv = sdr_receiver
    recv.stop()
    assert not recv._iq_record_worker.is_alive()
