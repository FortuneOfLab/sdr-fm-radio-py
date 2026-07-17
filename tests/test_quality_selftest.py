"""quality_selftest loader and CLI-validation tests (PR #5 and PR #1)."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.io import wavfile

from fm_radio.quality_selftest import (
    EPS, _load_iq_wav, _load_stereo_wav, _read_wav_window, _resample,
    _to_float_audio,
)


FS = 48000


def _write_wav(path, data, fs=FS):
    wavfile.write(str(path), fs, data)
    return str(path)


def _load_iq_reference(path, target_fs, duration_s):
    """The pre-mmap eager algorithm, kept verbatim as the oracle."""
    fs, raw = wavfile.read(path)
    x = _to_float_audio(np.asarray(raw))
    i = x[:, 0]
    q = x[:, 1]
    if duration_s is not None and duration_s > 0:
        n = int(duration_s * fs)
        i = i[:n]
        q = q[:n]
    i = _resample(i, fs, target_fs)
    q = _resample(q, fs, target_fs)
    n = min(i.size, q.size)
    iq = i[:n].astype(np.float32) + 1j * q[:n].astype(np.float32)
    peak = float(np.max(np.abs(iq)) + EPS)
    return (0.95 * iq / peak).astype(np.complex64)


@pytest.fixture
def int16_stereo_wav(tmp_path, rng):
    data = (rng.standard_normal((FS, 2)) * 8000).astype(np.int16)
    return _write_wav(tmp_path / "s16.wav", data)


@pytest.mark.parametrize("duration", [None, 0.25, 10.0])
def test_iq_loader_matches_eager_reference(int16_stereo_wav, duration):
    new = _load_iq_wav(int16_stereo_wav, FS, duration)
    ref = _load_iq_reference(int16_stereo_wav, FS, duration)
    assert np.array_equal(new, ref)


def test_iq_loader_rejects_mono(tmp_path, rng):
    mono = (rng.standard_normal(FS) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "mono.wav", mono)
    with pytest.raises(ValueError):
        _load_iq_wav(path, FS, 1.0)


def test_stereo_loader_handles_mono_and_float(tmp_path, rng):
    mono = (rng.standard_normal(FS) * 0.5).astype(np.float32)
    path = _write_wav(tmp_path / "monof.wav", mono)
    left, right = _load_stereo_wav(path, FS, 0.5)
    assert np.array_equal(left, right)
    assert left.size > 0


def test_read_window_bounds_memory_to_the_window(tmp_path, rng):
    data = (rng.standard_normal((FS, 2)) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "w.wav", data)
    fs, raw = _read_wav_window(path, 0.1)
    assert fs == FS
    assert raw.shape[0] == int(0.1 * FS)


def test_read_window_duration_beyond_file_is_clamped(tmp_path, rng):
    data = (rng.standard_normal((100, 2)) * 8000).astype(np.int16)
    path = _write_wav(tmp_path / "short.wav", data)
    _fs, raw = _read_wav_window(path, 100.0)
    assert raw.shape[0] == 100


def test_main_rejects_duration_not_exceeding_warmup(monkeypatch):
    import fm_radio.quality_selftest as qs
    monkeypatch.setattr(sys, "argv", [
        "quality_selftest", "--duration", "0.5", "--warmup-s", "1.0",
        "--cnr-db", "35",
    ])
    with pytest.raises(SystemExit):
        qs.main()
