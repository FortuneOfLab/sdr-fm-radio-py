"""Which sound devices the radio can play through, and which one by default.

PortAudio lists each device once per host API - on Windows the same
headphone amplifier is there as MME, DirectSound, WASAPI and WDM-KS -
and the API matters more than the device for how late the audio is:
MME holds about 107 ms of it on the card this was written on, WASAPI
a few milliseconds.  So a device here is the pair, named the way a
person can pick it out of a list and type it back: ``"WASAPI:
Speakers (USB Audio)"``.

Only what can actually play the receiver's output is offered: two
channels of float32 at the output rate.  WASAPI in shared mode plays
at the rate Windows mixes at and nothing else, and a device that
mixes at 32 kHz is not one to offer at 48.  WDM-KS is left out on
Windows: it lists internal endpoints under names nobody would know,
and it is the API PortAudio's own notes warn about.

Kept apart from :mod:`fm_radio.audio_output` because it is asked
about without an output - by ``--list-audio-devices``, before any
receiver exists.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger("fm_receiver.AudioDevices")

#: Whether the Windows list of APIs applies; a name for tests to set.
_ON_WINDOWS: bool = sys.platform == "win32"

#: The host APIs offered on Windows, by PortAudio's name for them, and
#: what they are called here.  In order of preference for the default:
#: WASAPI first because it is the one that is not late.
_WINDOWS_APIS: dict[str, str] = {
    "Windows WASAPI": "WASAPI",
    "MME": "MME",
    "Windows DirectSound": "DirectSound",
}


@dataclass(frozen=True)
class OutputDevice:
    """One device under one host API, as the user picks it."""

    #: What the user sees and types back: ``"<api>: <device name>"``.
    key: str
    #: PortAudio's device index; good for this run only.
    index: int
    api: str
    name: str


def printable(text: str, stream=None) -> str:
    """*text* as the console can show it: what it cannot, becomes ``?``.

    Device names are the system's, not ours.  The Windows console in
    Japan is cp932, which has no ``®`` - and a game controller's
    headset is called ``DUALSHOCK®4`` - so printing the list raised
    UnicodeEncodeError on that line and lost the rest of it.
    """
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return text


def _api_label(api_name: str) -> "str | None":
    """What a host API is called here, or None if it is not offered."""
    if _ON_WINDOWS:
        return _WINDOWS_APIS.get(api_name)
    return api_name


def _plays_our_output(pa, index: int, rate: int, channels: int,
                      sample_format) -> bool:
    """Whether the device would open for what the receiver plays."""
    try:
        return bool(pa.is_format_supported(
            rate, output_device=index, output_channels=channels,
            output_format=sample_format))
    except Exception:
        # PortAudio says no by raising ValueError; anything else is a
        # device that cannot be asked, which is a device not to offer.
        return False


def list_output_devices(pa, rate: int, channels: int,
                        sample_format) -> list[OutputDevice]:
    """Every device the receiver could play through, in PortAudio's order.

    Args:
        pa: A ``pyaudio.PyAudio``.
        rate, channels, sample_format: What will be played.

    Returns:
        The devices, with keys made unique: two devices of one name
        under one API get `` #2`` and so on after the second.
    """
    found: list[OutputDevice] = []
    seen: dict[str, int] = {}
    try:
        count = int(pa.get_device_count())
    except Exception as e:
        logger.warning("Could not list the sound devices: %s", e)
        return found
    for index in range(count):
        try:
            info = pa.get_device_info_by_index(index)
            if int(info.get("maxOutputChannels", 0)) < channels:
                continue
            api = _api_label(str(
                pa.get_host_api_info_by_index(info["hostApi"])["name"]))
        except Exception:
            continue
        if api is None:
            continue
        name = " ".join(str(info.get("name", "")).split())
        if not name or not _plays_our_output(pa, index, rate, channels,
                                             sample_format):
            continue
        key = f"{api}: {name}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = f"{key} #{seen[key]}"
        found.append(OutputDevice(key=key, index=index, api=api, name=name))
    return found


def default_device(pa, devices: list[OutputDevice]) -> "OutputDevice | None":
    """The device to use when none is asked for.

    The system's default output, under the first API in order of
    preference that has it: WASAPI where there is one, so that the
    radio is not 100 ms later than it need be for want of a setting.
    None when nothing on the list is the default, which leaves the
    choice to PortAudio.
    """
    defaults: dict[str, int] = {}
    try:
        for a in range(int(pa.get_host_api_count())):
            host = pa.get_host_api_info_by_index(a)
            label = _api_label(str(host["name"]))
            if label is not None and int(host["defaultOutputDevice"]) >= 0:
                defaults[label] = int(host["defaultOutputDevice"])
    except Exception as e:
        logger.debug("Could not ask for the default devices: %s", e)
    order = (list(_WINDOWS_APIS.values()) if _ON_WINDOWS
             else list(defaults))
    for api in order:
        index = defaults.get(api)
        for device in devices:
            if device.index == index:
                return device
    try:
        index = int(pa.get_default_output_device_info()["index"])
    except Exception:
        return None
    return next((d for d in devices if d.index == index), None)


def find_device(devices: list[OutputDevice], wanted: str,
                exact: bool = False) -> "OutputDevice | None":
    """The device *wanted* names, or None.

    The key itself, or failing that the one device whose key contains
    it, ignoring case - so ``--audio-device "wasapi: speakers"`` is
    enough, while ``fostex`` alone names that device under three APIs
    and is no answer.  Ambiguous is None: picking one of two for the
    user is how a radio ends up in the wrong headphones.

    *exact* takes the key and nothing else.  For a name that was a
    whole key when it was written down - the remembered device - a
    part of it is not a short way of saying it: ``WASAPI: Speakers``
    gone and ``WASAPI: Speakers (TV)`` there is a different device.
    """
    wanted = " ".join(wanted.split())
    for device in devices:
        if device.key == wanted:
            return device
    if exact:
        return None
    folded = wanted.casefold()
    if not folded:
        return None
    matches = [d for d in devices if folded in d.key.casefold()]
    return matches[0] if len(matches) == 1 else None
