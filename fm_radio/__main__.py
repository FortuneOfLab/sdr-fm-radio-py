#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) [2025] FortuneOfLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
FM Receiver System

This program receives FM broadcast signals using RTL-SDR,
performs FM demodulation via PLL and various filter processes,
and outputs/records audio using PyAudio.

Usage Examples:
  - Standard mode (no logging):
      $ python fm_receiver.py
  - Light mode:
      $ python fm_receiver.py --light
  - With logging enabled:
      $ python fm_receiver.py --log
      $ python fm_receiver.py --verbose  (or -v)
  - With debug logging:
      $ python fm_receiver.py --debug
  - Save logs to file:
      $ python fm_receiver.py --log-file fm_receiver.log
  - Start at a lower listening volume (0-100, default 100):
      $ python fm_receiver.py --volume 40
  - Start muted (unmute with 'unmute', or from the window):
      $ python fm_receiver.py --mute
  - See the sound devices, then play through one for this run:
      $ python fm_receiver.py --list-audio-devices
      $ python fm_receiver.py --audio-device "WASAPI: Speakers"
  - Combine options:
      $ python fm_receiver.py --light --debug --log-file debug.log

Command Examples (during execution):
  'list'              : Show available stations
  'stereo on'         : Enable stereo demodulation
  'stereo off' / 'mono': Enable mono demodulation
  'record start'      : Start recording (file name auto-generated)
  'record stop'       : Stop recording
  'iqrec start'       : Start raw IQ recording (I/Q 2ch WAV)
  'iqrec stop'        : Stop raw IQ recording
  'agc on'            : Enable auto gain control
  'agc off'           : Disable auto gain (manual mode)
  'gain <value>'      : Set manual gain in dB (when auto gain is off)
  'vol <0-100>'       : Set the listening volume ('vol +' / 'vol -' step it)
  'mute' / 'unmute'   : Silence the listening output and bring it back
  'out' / 'out <n>'   : List the sound devices / play through another
  '<station_num>' or '<freq_MHz>' : Tune to the specified station
  'q'                 : Quit the program
"""

from __future__ import annotations

import sys
import gc
import time
import logging

from fm_radio.logging_config import setup_logging, logger
from fm_radio.controller import FMReceiverController


def _install_gc_monitor() -> None:
    """Install a GC callback that logs Gen2 collections.

    Generation-2 collections can pause the interpreter for tens of
    milliseconds, long enough to back up the SDR data_queue and cause
    audio dropouts.  Logging the timing of Gen2 events lets us
    correlate them with observed audio glitches.
    """
    state: dict[str, float] = {
        "last_t": time.perf_counter(),
        "gc_start": 0.0,
    }
    gc_logger = logging.getLogger("fm_receiver.GCMonitor")

    def _gc_cb(phase: str, info: dict) -> None:
        gen = info.get("generation", 0)
        if gen < 2:
            return
        if phase == "start":
            state["gc_start"] = time.perf_counter()
            return
        if phase != "stop":
            return
        now = time.perf_counter()
        gc_dt_ms = (now - state["gc_start"]) * 1000.0
        elapsed_since = now - state["last_t"]
        gc_logger.warning(
            "GC Gen%d: pause=%.1fms collected=%d uncollectable=%d "
            "since_last_gen2=%.1fs",
            gen, gc_dt_ms, info.get("collected", -1),
            info.get("uncollectable", -1), elapsed_since,
        )
        state["last_t"] = now

    gc.callbacks.append(_gc_cb)


def _volume_argument(text: str | None) -> float:
    """Read --volume's value, or stop here saying what it should be.

    Checked before the receiver is built, so a typo costs nothing: no
    device is opened and no sound card is touched.  Out of range is
    refused rather than clamped - '--volume 400' is not a wish for 100.
    """
    try:
        value = float(text.rstrip('%')) if text is not None else None
    except ValueError:
        value = None
    if value is None or not 0.0 <= value <= 100.0:
        print("--volume needs a number from 0 to 100", file=sys.stderr)
        sys.exit(2)
    return value


def _list_audio_devices() -> None:
    """Print the sound devices the receiver could play through, and stop.

    Without building a receiver: the dongle need not even be plugged in
    to find out which name to give --audio-device.
    """
    import pyaudio
    from fm_radio.audio_devices import (
        default_device, list_output_devices, printable,
    )
    from fm_radio.constants import AUDIO_CHANNELS, AUDIO_OUTPUT_RATE
    from fm_radio.stations import configured_audio_device
    pa = pyaudio.PyAudio()
    try:
        devices = list_output_devices(pa, AUDIO_OUTPUT_RATE, AUDIO_CHANNELS,
                                      pyaudio.paFloat32)
        default = default_device(pa, devices)
    finally:
        pa.terminate()
    if not devices:
        print("No sound device listed; the receiver will use PortAudio's "
              "default.")
        return
    print("Sound devices that can play the receiver's output:")
    for device in devices:
        mark = " (default)" if device == default else ""
        print(printable(f"  {device.key}{mark}"))
    remembered = configured_audio_device()
    if remembered:
        print(printable(f"Last chosen, used unless --audio-device says "
                        f"otherwise: {remembered}"))


def main() -> None:
    """Entry point for the FM receiver application."""
    # Parse command line arguments
    light_mode: bool = False
    enable_logging: bool = False
    log_level: int = logging.INFO
    log_file: str | None = None
    stations_path: str | None = None
    use_gui: bool = False
    volume: float | None = None
    muted: bool = False
    audio_device: str | None = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--light':
            light_mode = True
        elif arg == '--gui':
            use_gui = True
        elif arg == '--stations' and i + 1 < len(sys.argv):
            stations_path = sys.argv[i + 1]
        elif arg == '--mute':
            muted = True
        elif arg == '--list-audio-devices':
            _list_audio_devices()
            return
        elif arg == '--audio-device':
            if i + 1 >= len(sys.argv) or not sys.argv[i + 1].strip():
                print("--audio-device needs a device name; "
                      "--list-audio-devices shows them", file=sys.stderr)
                sys.exit(2)
            audio_device = sys.argv[i + 1]
        elif arg == '--volume':
            volume = _volume_argument(sys.argv[i + 1]
                                      if i + 1 < len(sys.argv) else None)
        elif arg in ('--log', '--verbose', '-v'):
            enable_logging = True
        elif arg == '--debug':
            enable_logging = True
            log_level = logging.DEBUG
        elif arg == '--log-file' and i + 1 < len(sys.argv):
            log_file = sys.argv[i + 1]
            enable_logging = True

    # Setup logging only if requested
    if enable_logging:
        setup_logging(log_level=log_level, log_file=log_file)
        logger.info("=" * 60)
        logger.info("FM Receiver System Starting")
        logger.info("=" * 60)
        # Install GC monitor for diagnostics (audio-dropout investigation).
        _install_gc_monitor()
        logger.info("GC monitor installed (Gen2 collections will be logged)")
    else:
        # Disable all logging by setting to CRITICAL+1
        logging.disable(logging.CRITICAL)

    try:
        controller = FMReceiverController(light=light_mode,
                                          stations_path=stations_path,
                                          audio_device=audio_device)
        if volume is not None:
            # Before anything starts, so the first block out is already
            # at this level: nothing has been played yet to jump from.
            controller.set_volume(volume)
        if muted:
            # After the volume, which unmutes: '--volume 40 --mute' is
            # muted at 40, whichever order they were typed in.
            controller.set_muted(True)
        if use_gui:
            # The window owns the main loop, so the receiver is started in
            # the background and shut down when the window closes.
            from fm_radio import gui
            try:
                # The window first and the receiver second: building a
                # window takes long enough to be a gap in the audio if
                # there is audio to interrupt, and there is none yet.
                # start_background is handed over rather than called
                # here because it also has to be inside the try - it
                # starts the SDR thread before the processing thread,
                # so a failure between the two leaves threads running
                # that only cleanup() stops.
                exit_code = gui.run(controller, controller.start_background)
            finally:
                # The window sets quit_event when it closes, but it may
                # never have opened; cleanup() sets it either way and waits
                # for the threads before closing what they are using.
                controller.cleanup()
            if controller.device_failure and exit_code == 0:
                # The window closed normally, but only because the device
                # went.  A script that started this should be able to tell
                # that apart from a person closing the window, the same way
                # it can on the command line.
                exit_code = 1
            sys.exit(exit_code)
        controller.start()
        if controller.device_failure:
            # The command line came back because the device went, not
            # because anybody asked it to.  Said here as well as in
            # _leave_past_the_blocked_reader, which only runs when the
            # command thread is still sitting in input() - it often is
            # not, and the status must not depend on that.
            sys.exit(1)
    except Exception as e:
        if enable_logging:
            logger.critical(f"Failed to start FM Receiver: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        if enable_logging:
            print("Check the log for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()
