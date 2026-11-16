"""How many USB transfers the async read keeps in flight.

pyrtlsdr has no argument for it: it reads ``DEFAULT_ASYNC_BUF_NUMBER``
off the instance when it calls ``rtlsdr_read_async``.  So two things
have to hold, and each is checked here: the receiver sets it before it
starts reading, and pyrtlsdr really does read it from the instance and
hand it to librtlsdr - the second against the installed pyrtlsdr, not
the stand-in, because a rename on its side would silently put the read
back on librtlsdr's 15.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from fm_radio.constants import SDR_ASYNC_BUF_NUMBER, SDR_BLOCK_SIZE

#: librtlsdr's own default, what pyrtlsdr's 0 means.
LIBRTLSDR_DEFAULT_BUFFERS = 15


def test_the_receiver_asks_for_more_transfers_than_the_default(
        sdr_receiver, monkeypatch):
    seen = {}

    def read(self, cb, num_samples=None):
        seen["buffers"] = getattr(self, "DEFAULT_ASYNC_BUF_NUMBER", None)
        seen["samples"] = num_samples

    monkeypatch.setattr(type(sdr_receiver.sdr), "read_samples_async", read)
    sdr_receiver.start()
    assert seen["buffers"] == SDR_ASYNC_BUF_NUMBER
    assert SDR_ASYNC_BUF_NUMBER > LIBRTLSDR_DEFAULT_BUFFERS
    assert seen["samples"] == SDR_BLOCK_SIZE


def _the_installed_pyrtlsdr():
    """The real pyrtlsdr, under a name of its own.

    The test session has a stand-in installed as ``rtlsdr``, so the
    real package is loaded beside it.  Skipped where it is not
    installed or its C library will not load.
    """
    for entry in sys.path:
        init = os.path.join(entry, "rtlsdr", "__init__.py")
        if os.path.isfile(init):
            break
    else:
        pytest.skip("pyrtlsdr is not installed")
    name = "_real_pyrtlsdr"
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[os.path.dirname(init)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as trouble:            # no librtlsdr on this machine
        sys.modules.pop(name, None)
        pytest.skip(f"pyrtlsdr will not load here: {trouble}")
    return sys.modules[f"{name}.rtlsdr"]


def test_pyrtlsdr_hands_the_instance_count_to_librtlsdr(monkeypatch):
    real = _the_installed_pyrtlsdr()
    seen = {}

    def read_async(dev_p, callback, context, buf_num, buf_len):
        seen["buf_num"], seen["buf_len"] = buf_num, buf_len
        return 0

    monkeypatch.setattr(real.librtlsdr, "rtlsdr_read_async", read_async)
    device = object.__new__(real.RtlSdr)    # no USB device opened
    device.dev_p = None
    device.DEFAULT_ASYNC_BUF_NUMBER = SDR_ASYNC_BUF_NUMBER
    device.read_samples_async(lambda samples, ctx: None,
                              num_samples=SDR_BLOCK_SIZE)
    assert seen == {"buf_num": SDR_ASYNC_BUF_NUMBER,
                    "buf_len": 2 * SDR_BLOCK_SIZE}
