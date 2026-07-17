"""Metadata sidecar tests (backlog item B5) and recording-path tests (B4)."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

import fm_radio.audio_output as ao_mod
import fm_radio.sdr_receiver as sr_mod
from fm_radio.recording_meta import sidecar_path


CHUNK = np.zeros(768 * 2, dtype=np.float32) + 0.25
IQ_BLOCK = (np.zeros(16384) + 0.1 + 0.05j).astype(np.complex64)


def _load_sidecar(base_wav):
    with open(sidecar_path(str(base_wav)), encoding="utf-8") as f:
        return json.load(f)


def test_audio_sidecar_written_and_finalised(audio_output, tmp_path):
    ao = audio_output
    base = tmp_path / "a.wav"
    ao.start_recording(
        str(base), metadata={"center_freq_hz": 91.6e6, "gain_db": 20.7},
    )
    meta = _load_sidecar(base)
    assert meta["type"] == "audio"
    assert meta["sample_rate_hz"] == 48000
    assert meta["center_freq_hz"] == 91.6e6
    assert meta["gain_db"] == 20.7
    assert "started_at" in meta
    assert "stopped_at" not in meta  # not finalised yet

    for _ in range(3):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    meta = _load_sidecar(base)
    assert "stopped_at" in meta
    assert meta["parts"] == ["a.wav"]
    assert meta["dropped_chunks"] == 0


def test_audio_sidecar_lists_rotated_parts(audio_output, tmp_path, monkeypatch):
    monkeypatch.setattr(ao_mod, "AUDIO_RECORD_ROTATE_THRESHOLD_BYTES", 50_000)
    ao = audio_output
    base = tmp_path / "rot.wav"
    ao.start_recording(str(base))
    for _ in range(50):
        ao.record(CHUNK.copy())
    ao.stop_recording()

    meta = _load_sidecar(base)
    assert meta["parts"][0] == "rot.wav"
    assert len(meta["parts"]) >= 2
    assert meta["parts"][1] == "rot.part001.wav"
    # Every listed part must actually exist on disk.
    for name in meta["parts"]:
        assert (tmp_path / name).exists()


def test_iq_sidecar_written_and_finalised(sdr_receiver, tmp_path):
    recv = sdr_receiver
    base = tmp_path / "iq.wav"
    recv.start_iq_recording(str(base))
    meta = _load_sidecar(base)
    assert meta["type"] == "iq"
    assert meta["sample_rate_hz"] == 1024000
    assert meta["center_freq_hz"] == pytest.approx(80e6)
    assert "gain_db" in meta
    assert "started_at" in meta

    for _ in range(2):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
    recv.stop_iq_recording()

    meta = _load_sidecar(base)
    assert "stopped_at" in meta
    assert meta["parts"] == ["iq.wav"]
    assert meta["dropped_blocks"] == 0


def test_iq_sidecar_lists_rotated_parts(sdr_receiver, tmp_path, monkeypatch):
    monkeypatch.setattr(sr_mod, "IQ_RECORD_ROTATE_THRESHOLD_BYTES", 200_000)
    recv = sdr_receiver
    base = tmp_path / "rot.wav"
    recv.start_iq_recording(str(base))
    import time
    for _ in range(10):
        recv.callback(IQ_BLOCK.copy(), None)
        try:
            recv.data_queue.get_nowait()
        except Exception:
            pass
        time.sleep(0.01)
    recv.stop_iq_recording()

    meta = _load_sidecar(base)
    assert len(meta["parts"]) >= 2
    for name in meta["parts"]:
        assert (tmp_path / name).exists()


def test_build_recording_path_uses_recordings_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from fm_radio.cli import build_recording_path
    from fm_radio.constants import RECORDINGS_DIR

    p_audio = build_recording_path(91.6, iq=False)
    p_iq = build_recording_path(80.0, iq=True)
    assert (tmp_path / RECORDINGS_DIR).is_dir()
    assert re.fullmatch(
        re.escape(RECORDINGS_DIR) + r"[\\/]\d{8}_\d{6}_91\.6MHz\.wav", p_audio,
    )
    assert re.fullmatch(
        re.escape(RECORDINGS_DIR) + r"[\\/]\d{8}_\d{6}_80\.0MHz_IQ\.wav", p_iq,
    )
