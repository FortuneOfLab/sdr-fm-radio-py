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

from fm_radio.exceptions import RecordingError

import fm_radio.sdr_receiver as sr_mod


IQ_BLOCK = (np.zeros(16384) + 0.1 + 0.05j).astype(np.complex64)
BLOCK_FRAMES = 16384


def _wav_frames(path):
    with wave_mod.open(str(path), "rb") as r:
        return r.getnframes()


# ----------------------------------------------------------------------
# Being handed the same blocks without taking them
# ----------------------------------------------------------------------

def a_block(n: int = 256) -> np.ndarray:
    return (np.arange(n) + 1j * np.arange(n)).astype(np.complex64)


def test_a_watcher_is_handed_the_same_block_the_demodulator_gets(
        sdr_receiver):
    """Not a copy, and not instead of: both get it.

    A band scan runs while the receiver is still playing.  Reading
    the demodulator's own queue would take blocks out of its stream,
    and its filters carry straight across the gap.
    """
    tap = sdr_receiver.watch_the_blocks()
    block = a_block()

    sdr_receiver.callback(block, None)

    _generation, theirs = sdr_receiver.data_queue.get_nowait()
    _generation, ours = tap.get_nowait()
    assert ours is theirs, "the watcher got a different array"
    assert np.array_equal(ours, block)


def test_nothing_is_handed_over_when_nobody_is_watching(sdr_receiver):
    sdr_receiver.callback(a_block(), None)

    assert sdr_receiver.data_queue.qsize() == 1
    assert sdr_receiver._tap is None


def test_a_watcher_that_is_behind_loses_blocks_rather_than_holding_up(
        sdr_receiver):
    """This is the realtime callback; it cannot wait for a reader."""
    tap = sdr_receiver.watch_the_blocks(depth=1)

    for _ in range(5):
        sdr_receiver.callback(a_block(), None)

    assert tap.qsize() == 1
    assert sdr_receiver.data_queue.qsize() == 5, "the receiver lost blocks"


def test_a_second_watcher_is_refused(sdr_receiver):
    """Quietly replacing the first leaves it wondering why the
    blocks stopped: a hop that timed out, with nothing to say why.
    """
    sdr_receiver.watch_the_blocks()

    with pytest.raises(RuntimeError, match="already watching"):
        sdr_receiver.watch_the_blocks()


def test_watching_again_after_stopping_is_fine(sdr_receiver):
    first = sdr_receiver.watch_the_blocks()
    sdr_receiver.stop_watching(first)

    second = sdr_receiver.watch_the_blocks()

    sdr_receiver.callback(a_block(), None)
    assert second.qsize() == 1
    assert first.qsize() == 0


def test_stopping_somebody_elses_watch_does_nothing(sdr_receiver):
    """Whoever is watching now keeps watching."""
    mine = sdr_receiver.watch_the_blocks()
    sdr_receiver.stop_watching(mine)
    theirs = sdr_receiver.watch_the_blocks()

    sdr_receiver.stop_watching(mine)         # late, and not ours to stop

    sdr_receiver.callback(a_block(), None)
    assert theirs.qsize() == 1


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


def test_concurrent_starts_leave_exactly_one_recording(sdr_receiver,
                                                       tmp_path):
    """Two at once: one records, and the other leaves nothing behind."""
    recv = sdr_receiver
    f1 = tmp_path / "c1.wav"
    f2 = tmp_path / "c2.wav"

    barrier = threading.Barrier(2)

    def starter(path):
        barrier.wait()
        try:
            recv.start_iq_recording(str(path))
        except RecordingError:
            pass

    threads = [threading.Thread(target=starter, args=(f,)) for f in (f1, f2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    made = [f for f in (f1, f2) if f.exists()]
    assert len(made) == 1, f"{len(made)} files were made: {made}"
    assert recv._iq_record_base_path == str(made[0])
    recv.stop_iq_recording()


def test_a_start_will_not_take_over_a_file_that_is_already_there(
        sdr_receiver, tmp_path):
    recv = sdr_receiver
    path = tmp_path / "theirs.wav"
    path.write_bytes(b"Y" * 1200)

    with pytest.raises(RecordingError):
        recv.start_iq_recording(str(path))

    assert path.stat().st_size == 1200
    assert not recv.iq_recording


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


# ----------------------------------------------------------------------
# Tuning generation
# ----------------------------------------------------------------------

def test_a_block_is_stamped_with_the_current_tuning(sdr_receiver):
    generation = sdr_receiver.tuning_generation
    sdr_receiver.callback(np.zeros(8, dtype=np.complex128), None)

    stamped, _ = sdr_receiver.data_queue.get_nowait()
    assert stamped == generation


def test_retuning_advances_the_generation(sdr_receiver):
    first = sdr_receiver.tuning_generation
    sdr_receiver.set_center_frequency(81.3e6)
    assert sdr_receiver.tuning_generation != first


def test_a_retune_during_the_conversion_does_not_restamp_the_block(
        sdr_receiver, stalling_samples):
    """The generation is read on entry, not after the copy.

    pyrtlsdr hands over complex128, so np.asarray copies; a retune landing
    inside that copy used to stamp the old station's samples with the new
    tuning, which put them past the flush tune() had just done.
    """
    samples = stalling_samples(np.zeros(8, dtype=np.complex128))
    before = sdr_receiver.tuning_generation

    thread = threading.Thread(target=sdr_receiver.callback,
                              args=(samples, None), daemon=True)
    thread.start()
    try:
        assert samples.converting.wait(5), "the conversion never started"
        sdr_receiver.set_center_frequency(81.3e6)
        samples.retuned.set()
    finally:
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert samples.retuned_in_time, (
        "the retune did not complete while the samples were being converted, "
        "so this run never built the ordering it is meant to test")
    assert sdr_receiver.tuning_generation != before, "the retune did not land"
    stamped, _ = sdr_receiver.data_queue.get_nowait()
    assert stamped == before, "the pre-retune block was stamped as current"
