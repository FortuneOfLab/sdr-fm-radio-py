"""Shared test fixtures and hardware-module fakes.

Installs lightweight fakes for the hardware-facing third-party modules
(``pyaudio``, ``rtlsdr``) BEFORE any ``fm_radio`` import, so the test
suite runs on machines and CI runners without a sound device, PortAudio,
or the librtlsdr driver.  The fakes are installed unconditionally: tests
must never touch real hardware even on a developer machine that has it.
"""

from __future__ import annotations

import sys
import threading
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
    """Stands in for rtlsdr.RtlSdr, including the parts that bite.

    The async read blocks until it is cancelled, as the real one does, and
    the cancel copies pyrtlsdr's behaviour rather than the C library's: the
    C call returns an error whenever no read is running, and the wrapper
    answers that by closing the device and raising (rtlsdr.py:699-706).
    Tests that never start a read therefore see the same trap the real
    driver sets.
    """

    def __init__(self) -> None:
        self.sample_rate = 1.024e6
        self.center_freq = 80e6
        self.direct_sampling = 0
        self.gain_calls: list[float] = []
        self.device_opened = True
        self.read_async_canceling = False
        self.calls: list[str] = []
        self.reading: threading.Event = threading.Event()
        self.cancelled: threading.Event = threading.Event()

    def set_manual_gain_enabled(self, manual: bool) -> None: ...

    def set_gain(self, gain: float) -> None:
        self.gain_calls.append(gain)

    def get_gain(self) -> float:
        return 0.0

    def read_samples_async(self, cb, num_samples=None) -> None:
        # read_bytes_async clears the wrapper's flag on its way in.
        self.read_async_canceling = False
        self.calls.append("read")
        self.cancelled.clear()
        self.reading.set()
        try:
            self.cancelled.wait(30)
        finally:
            self.reading.clear()

    def cancel_read_async(self) -> None:
        if self.reading.is_set():
            self.calls.append("cancel")
            self.read_async_canceling = True
            self.cancelled.set()
            return
        # rtlsdr_cancel_async returned -2: no read is running.
        if not self.read_async_canceling:
            self.calls.append("cancel failed")
            self.close()
            raise OSError(
                "LIBUSB_ERROR_INVALID_PARAM: Could not cancel async read")
        self.read_async_canceling = True

    def close(self) -> None:
        self.calls.append("close")
        self.device_opened = False


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
def stalling_samples():
    """Factory for IQ samples that pause while NumPy is converting them.

    The pause lives in the object's own ``__array__`` rather than in a
    patched ``np.asarray``, so only this block is held up and nothing else
    the code converts meanwhile is affected.

    The returned object carries two events - ``converting``, set when the
    conversion starts, and ``retuned``, which the test sets to release it -
    and records in ``retuned_in_time`` whether that release actually
    arrived.  A test that could not build the ordering it needs has to fail
    for that reason rather than quietly go on to check something else.
    """
    class _StallingSamples:
        def __init__(self, data, timeout: float = 5.0) -> None:
            self._data = data
            self._timeout = timeout
            self.converting = threading.Event()
            self.retuned = threading.Event()
            self.retuned_in_time = False

        def __array__(self, dtype=None, copy=None):
            self.converting.set()
            self.retuned_in_time = self.retuned.wait(self._timeout)
            return np.asarray(self._data, dtype=dtype)

    return _StallingSamples


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


# ----------------------------------------------------------------------
# Station catalogue fixtures
#
# Shared through fixtures rather than through imports between test modules:
# a test module is not importable under every pytest import mode, and the
# catalogue tests must never read the developer's own stations.toml.
# ----------------------------------------------------------------------

#: The presets the receiver shipped with before the catalogue existed.  The
#: names moved from legal names to brand names, but these ten frequencies are
#: what `list` and numeric tuning must keep producing.
LEGACY_PRESET_MHZ = [78.0, 79.5, 80.0, 81.3, 82.5, 84.7, 89.7, 90.5, 91.6, 93.0]


@pytest.fixture
def legacy_preset_mhz() -> list[float]:
    return list(LEGACY_PRESET_MHZ)


@pytest.fixture
def no_user_config(tmp_path):
    """A stations.toml path that does not exist, inside this test's tmp dir.

    Every catalogue test has to pass one of these: with no explicit path the
    loader reads the real user configuration, and a developer with their own
    favourites would see unrelated failures.
    """
    return tmp_path / "absent-stations.toml"


@pytest.fixture(scope="session")
def _catalogue_data(tmp_path_factory):
    """The bundled catalogue, loaded once for the whole session."""
    from fm_radio import stations as st
    absent = tmp_path_factory.mktemp("catalogue") / "absent-stations.toml"
    return st.load_stations(user_path=absent)


@pytest.fixture
def catalogue(_catalogue_data):
    """The bundled catalogue with no user layer applied.

    A fresh list each time so a test that sorts or filters in place cannot
    reach the next one.  Station is frozen, so the entries can be shared.
    """
    return list(_catalogue_data)


@pytest.fixture
def write_toml(tmp_path):
    """Write *body* as a stations.toml and return its path."""
    def _write(body: str):
        path = tmp_path / "stations.toml"
        path.write_text(body, encoding="utf-8")
        return path
    return _write


@pytest.fixture
def load_config(write_toml):
    """Load the catalogue with *body* as the user's stations.toml.

    Returns (stations, warnings), so a test can assert both on the merged
    catalogue and on what the user would have been told about their file.
    """
    from fm_radio import stations as st

    def _load(body: str, **kwargs):
        warnings: list[str] = []
        stations = st.load_stations(user_path=write_toml(body),
                                    warn=warnings.append, **kwargs)
        return stations, warnings
    return _load
