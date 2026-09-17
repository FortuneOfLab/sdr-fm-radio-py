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


def main() -> None:
    """Entry point for the FM receiver application."""
    # Parse command line arguments
    light_mode: bool = False
    enable_logging: bool = False
    log_level: int = logging.INFO
    log_file: str | None = None
    stations_path: str | None = None
    use_gui: bool = False

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--light':
            light_mode = True
        elif arg == '--gui':
            use_gui = True
        elif arg == '--stations' and i + 1 < len(sys.argv):
            stations_path = sys.argv[i + 1]
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
                                          stations_path=stations_path)
        if use_gui:
            # The window owns the main loop, so the receiver is started in
            # the background and shut down when the window closes.
            from fm_radio import gui
            try:
                # start_background() is inside the try: it starts the SDR
                # thread before the processing thread, so a failure between
                # the two leaves threads running that only cleanup() stops.
                controller.start_background()
                exit_code = gui.run(controller)
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
