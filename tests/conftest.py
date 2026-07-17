"""Shared test fixtures and hardware-module fakes.

Installs lightweight fakes for the hardware-facing third-party modules
(``pyaudio``, ``rtlsdr``) BEFORE any ``fm_radio`` import, so the test
suite runs on machines and CI runners without a sound device, PortAudio,
or the librtlsdr driver.  The fakes are installed unconditionally: tests
must never touch real hardware even on a developer machine that has it.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Fake pyaudio
# ----------------------------------------------------------------------

class _FakeStream:
    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


class _FakePyAudio:
    def open(self, **kwargs):
        return _FakeStream()

    def terminate(self) -> None: ...


def _install_fake_pyaudio() -> None:
    mod = types.ModuleType("pyaudio")
    mod.PyAudio = _FakePyAudio
    mod.paFloat32 = 1
    mod.paContinue = 0
    mod.paComplete = 1
    sys.modules["pyaudio"] = mod


# ----------------------------------------------------------------------
# Fake rtlsdr
# ----------------------------------------------------------------------

class FakeRtlSdr:
    """Stands in for rtlsdr.RtlSdr; records gain calls for assertions."""

    def __init__(self) -> None:
        self.sample_rate = 1.024e6
        self.center_freq = 80e6
        self.direct_sampling = 0
        self.gain_calls: list[float] = []

    def set_manual_gain_enabled(self, manual: bool) -> None: ...

    def set_gain(self, gain: float) -> None:
        self.gain_calls.append(gain)

    def get_gain(self) -> float:
        return 0.0

    def read_samples_async(self, cb, num_samples=None) -> None: ...

    def cancel_read_async(self) -> None: ...

    def close(self) -> None: ...


def _install_fake_rtlsdr() -> None:
    mod = types.ModuleType("rtlsdr")
    mod.RtlSdr = FakeRtlSdr
    sys.modules["rtlsdr"] = mod


_install_fake_pyaudio()
_install_fake_rtlsdr()


# ----------------------------------------------------------------------
# Common fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def audio_output():
    """AudioOutput instance backed by the fake pyaudio; always cleaned up."""
    from fm_radio.audio_output import AudioOutput
    ao = AudioOutput()
    yield ao
    ao.cleanup()


@pytest.fixture
def sdr_receiver():
    """SDRReceiver instance backed by the fake rtlsdr; always stopped."""
    from fm_radio.sdr_receiver import SDRReceiver
    recv = SDRReceiver()
    yield recv
    recv.stop()
