"""Shared test fixtures and hardware-module fakes.

Installs lightweight fakes for the hardware-facing third-party modules
(``pyaudio``, ``rtlsdr``) BEFORE any ``fm_radio`` import, so the test
suite runs on machines and CI runners without a sound device, PortAudio,
or the librtlsdr driver.  The fakes are installed unconditionally: tests
must never touch real hardware even on a developer machine that has it.
"""

from __future__ import annotations

import os
import sys
import threading
import types

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Fake pyaudio
# ----------------------------------------------------------------------

class _FakeStream:
    """As much of a PortAudio stream as the receiver asks for.

    Including whether it has been started: pyaudio opens a stream
    running unless told otherwise, and the difference matters - a
    running stream asks for a buffer every few milliseconds whether
    or not anybody has made one.
    """

    #: What a card says it holds.  PortAudio reports the device's
    #: own buffer and fills all of it the moment the stream starts;
    #: the USB DAC these numbers come from holds 107 ms, which is
    #: five callbacks pulled back to back.  A fake that says nothing
    #: would let the output believe the card takes nothing.
    output_latency: float = 0.1067

    def __init__(self, start: bool = True) -> None:
        self.started = bool(start)
        self.stopped = False

    def start_stream(self) -> None:
        self.started = True

    def stop_stream(self) -> None:
        self.started = False
        self.stopped = True

    def is_active(self) -> bool:
        return self.started

    def get_output_latency(self) -> float:
        return self.output_latency

    def close(self) -> None: ...


class _FakePyAudio:
    def open(self, **kwargs):
        return _FakeStream(start=kwargs.get("start", True))

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

# librtlsdr's async states, as rtlsdr_cancel_async sees them.
RTLSDR_INACTIVE = 0
RTLSDR_RUNNING = 1
RTLSDR_CANCELING = 2


class FakeLibRtlSdr:
    """The C layer: rtlsdr_cancel_async and nothing else.

    Copies librtlsdr's own logic - two field writes, no USB traffic, and
    -2 for every state but RUNNING, including CANCELING (the branch that
    would forgive that one is compiled out of librtlsdr).  Crucially it
    never closes the device: that is the wrapper's doing, and keeping the
    two apart is what lets a test tell which route the receiver took.
    """

    @staticmethod
    def rtlsdr_cancel_async(dev_p):
        if dev_p is None:
            return -1
        if not dev_p.device_opened:
            # rtlsdr_close has freed the struct; the real thing reads
            # whatever is now at that address.  Recorded rather than
            # simulated, so a test can see it happened at all.
            dev_p.calls.append("cancel after close")
            return -1
        if dev_p.async_status == RTLSDR_RUNNING:
            dev_p.calls.append("cancel")
            dev_p.async_status = RTLSDR_CANCELING
            dev_p.cancelled.set()
            return 0
        # librtlsdr only honours RUNNING -> CANCELING.  The branch that
        # would forgive any other state is compiled out (#if 0), so a
        # second ask while the first is still unwinding gets -2 as well,
        # and the state is left alone.
        dev_p.calls.append("cancel (no read running)")
        return -2


class FakeRtlSdr:
    """Stands in for rtlsdr.RtlSdr, including the parts that bite.

    The async read blocks until it is cancelled, as the real one does, and
    ``cancel_read_async`` copies pyrtlsdr's wrapper rather than the C
    library: on a failed call it closes the device and raises
    (rtlsdr.py:699-706).  It also clears ``read_async_canceling`` on the
    way into a read (rtlsdr.py:599), which is what makes that flag
    unusable as a safety measure.  Any use of the wrapper shows up in
    ``calls`` as "wrapper cancel".
    """

    def __init__(self) -> None:
        self.sample_rate = 1.024e6
        self._center_freq = 80e6
        self.direct_sampling = 0
        self.gain_calls: list[float] = []
        self.device_opened = True
        self.read_async_canceling = False
        self.async_status = RTLSDR_INACTIVE
        self.calls: list[str] = []
        self.reading: threading.Event = threading.Event()
        self.cancelled: threading.Event = threading.Event()
        # librtlsdr's rtlsdr_dev_t *, which the C cancel takes.
        self.dev_p = self

    @property
    def center_freq(self):
        self._note_use("read center_freq")
        return self._center_freq

    @center_freq.setter
    def center_freq(self, value) -> None:
        self._note_use("write center_freq")
        self._center_freq = value

    def _note_use(self, what: str) -> None:
        """Record any use of a handle that has been freed."""
        if not self.device_opened:
            self.calls.append(f"{what} after close")

    def set_manual_gain_enabled(self, manual: bool) -> None:
        self._note_use("set_manual_gain_enabled")

    def set_gain(self, gain: float) -> None:
        self._note_use("set_gain")
        self.gain_calls.append(gain)

    def get_gain(self) -> float:
        self._note_use("get_gain")
        return 0.0

    def read_samples_async(self, cb, num_samples=None) -> None:
        # read_bytes_async clears the wrapper's flag on its way in
        # (rtlsdr.py:599), before librtlsdr knows a read exists.
        self.read_async_canceling = False
        self.calls.append("read")
        self.cancelled.clear()
        self.reading.set()
        self.async_status = RTLSDR_RUNNING      # the C call takes over here
        try:
            self.cancelled.wait(30)
        finally:
            self.async_status = RTLSDR_INACTIVE
            self.reading.clear()

    def cancel_read_async(self) -> None:
        """pyrtlsdr's wrapper, close-and-raise included."""
        self.calls.append("wrapper cancel")
        result = FakeLibRtlSdr.rtlsdr_cancel_async(self.dev_p)
        if result < 0 and not self.read_async_canceling:
            self.calls.append("cancel failed")
            self.close()
            raise OSError(
                "LIBUSB_ERROR_INVALID_PARAM: Could not cancel async read")
        self.read_async_canceling = True

    def close(self) -> None:
        if self.reading.is_set():
            # librtlsdr is inside rtlsdr_read_async, using this handle.
            self.calls.append("close during read")
        self.calls.append("close")
        self.device_opened = False


def _install_fake_rtlsdr() -> None:
    mod = types.ModuleType("rtlsdr")
    mod.RtlSdr = FakeRtlSdr
    # fm_radio reaches the C cancel through rtlsdr.librtlsdr.librtlsdr,
    # so the fake has to be shaped the same way round.
    lib_mod = types.ModuleType("rtlsdr.librtlsdr")
    lib_mod.librtlsdr = FakeLibRtlSdr
    mod.librtlsdr = lib_mod
    sys.modules["rtlsdr"] = mod
    sys.modules["rtlsdr.librtlsdr"] = lib_mod


_install_fake_pyaudio()
_install_fake_rtlsdr()


# ----------------------------------------------------------------------
# Common fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="session")
def qt_app():
    """One offscreen QApplication for the session.

    Qt allows only one, and the platform has to be chosen before it is
    built: a runner with no display aborts the process rather than
    failing a test, and the abort takes the whole run with it.  Shared
    from here so that whichever test file reaches Qt first sets it up the
    same way - a second file building its own found the platform already
    chosen, and on CI that was the wrong one.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication
    yield QApplication.instance() or QApplication([])


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
