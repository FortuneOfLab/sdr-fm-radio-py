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
"""Command line interface for the FM receiver.

Uses a dispatch-table pattern: exact-match commands are looked up in a
dictionary, while prefix-match commands (``agc``, ``gain``) and numeric
input (station number / frequency) are handled via fallback logic.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Callable

from fm_radio.exceptions import RecordingError

if TYPE_CHECKING:
    from fm_radio.controller import FMReceiverController


class CommandLineInterface(threading.Thread):
    """Thread for handling command line input.

    Receives user commands and controls FMReceiverController via its
    public facade API.  Each command is handled by a dedicated
    ``_cmd_*`` method, keeping the main loop minimal.
    """

    def __init__(self, controller: FMReceiverController) -> None:
        super().__init__(daemon=True)
        self.controller: FMReceiverController = controller

        # Dispatch table: exact command string -> handler method.
        # Each handler receives the raw command string and returns
        # True to continue or False to quit.
        self._commands: dict[str, Callable[[str], bool]] = {
            'q':            self._cmd_quit,
            'list':         self._cmd_list,
            'stereo on':    self._cmd_stereo_on,
            'stereo':       self._cmd_stereo_on,
            'stereo off':   self._cmd_stereo_off,
            'mono':         self._cmd_stereo_off,
            'record start': self._cmd_record_start,
            'record stop':  self._cmd_record_stop,
            'iqrec start':  self._cmd_iq_record_start,
            'iqrec stop':   self._cmd_iq_record_stop,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self.controller.quit_event.is_set():
            self._print_help()
            cmd = input().strip().lower()
            if not self._dispatch(cmd):
                break

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str) -> bool:
        """Route *cmd* to the appropriate handler.

        Resolution order:
          1. Exact match in the dispatch table.
          2. Prefix match (``agc``, ``gain``).
          3. Numeric input interpreted as station number or frequency.

        Returns:
            bool: True to continue the command loop, False to quit.
        """
        # 1. Exact match
        handler = self._commands.get(cmd)
        if handler:
            return handler(cmd)

        # 2. Prefix match
        if cmd.startswith('agc'):
            return self._cmd_agc(cmd)
        if cmd.startswith('gain'):
            return self._cmd_gain(cmd)

        # 3. Numeric / frequency input -> tune
        return self._cmd_tune(cmd)

    # ------------------------------------------------------------------
    # Command handlers  (each returns True=continue, False=quit)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_help() -> None:
        """Display available commands."""
        print("\nEnter command:")
        print("  'list' -> show station list")
        print("  'stereo on/off' or 'mono' -> toggle stereo demodulation")
        print("  'record start' -> start recording with auto-generated filename")
        print("  'record stop' -> stop recording")
        print("  'iqrec start' -> start raw IQ recording (I/Q 2ch WAV)")
        print("  'iqrec stop' -> stop raw IQ recording")
        print("  'agc on' -> enable auto gain control")
        print("  'agc off' -> disable auto gain (manual mode)")
        print("  'gain <value>' -> set manual gain in dB (when auto gain is off)")
        print("  <station_num> or <freq_MHz> -> tune")
        print("  'q' -> quit")

    def _cmd_quit(self, cmd: str) -> bool:
        """Handle 'q' — request shutdown."""
        print("Exiting command input...")
        self.controller.quit_event.set()
        return False

    def _cmd_list(self, cmd: str) -> bool:
        """Handle 'list' — display station list."""
        print("Available stations:")
        for i, (name, freq) in enumerate(self.controller.get_stations_list(), start=1):
            print(f"{i}: {name} ({freq/1e6:.1f} MHz)")
        return True

    def _cmd_stereo_on(self, cmd: str) -> bool:
        """Handle 'stereo on' / 'stereo' — enable stereo demodulation."""
        if self.controller.set_stereo(True):
            print("Stereo demodulation enabled.")
        else:
            print("Stereo demodulation not supported.")
        return True

    def _cmd_stereo_off(self, cmd: str) -> bool:
        """Handle 'stereo off' / 'mono' — enable mono demodulation."""
        if self.controller.set_stereo(False):
            print("Mono demodulation enabled.")
        else:
            print("Stereo demodulation not supported.")
        return True

    def _cmd_record_start(self, cmd: str) -> bool:
        """Handle 'record start' — begin recording with auto-generated filename."""
        current_time = time.strftime("%Y%m%d_%H%M%S")
        freq = self.controller.get_frequency() / 1e6
        filename = f"{current_time}_{freq:.1f}MHz.wav"
        try:
            self.controller.start_recording(filename)
            print(f"Recording started: {filename}")
        except RecordingError as e:
            print(f"Recording start failed: {e}")
        return True

    def _cmd_record_stop(self, cmd: str) -> bool:
        """Handle 'record stop' — stop recording."""
        if self.controller.is_recording():
            self.controller.stop_recording()
            print("Recording stopped.")
        else:
            print("Not currently recording.")
        return True

    def _cmd_iq_record_start(self, cmd: str) -> bool:
        """Handle 'iqrec start' - begin IQ recording with auto-generated filename."""
        current_time = time.strftime("%Y%m%d_%H%M%S")
        freq = self.controller.get_frequency() / 1e6
        filename = f"{current_time}_{freq:.1f}MHz_IQ.wav"
        try:
            self.controller.start_iq_recording(filename)
            print(f"IQ recording started: {filename}")
        except RecordingError as e:
            print(f"IQ recording start failed: {e}")
        return True

    def _cmd_iq_record_stop(self, cmd: str) -> bool:
        """Handle 'iqrec stop' - stop IQ recording."""
        if self.controller.is_iq_recording():
            self.controller.stop_iq_recording()
            print("IQ recording stopped.")
        else:
            print("IQ recording is not active.")
        return True

    def _cmd_agc(self, cmd: str) -> bool:
        """Handle 'agc on', 'agc off', 'agc off <gain>' — auto gain control."""
        tokens = cmd.split()
        if len(tokens) == 2:
            if tokens[1] == "on":
                self.controller.set_agc_mode(True)
                print("Auto gain control enabled.")
            elif tokens[1] == "off":
                self.controller.set_agc_mode(False)
                print(f"Manual gain mode. Current gain: {self.controller.get_gain():.1f} dB")
            else:
                print("Invalid agc command format.")
        elif len(tokens) == 3 and tokens[1] == "off":
            try:
                gain_value = float(tokens[2])
                self.controller.set_agc_mode(False)
                self.controller.set_gain(gain_value)
                print(f"Manual gain mode. Gain set to {self.controller.get_gain():.1f} dB")
            except ValueError:
                print("Invalid gain value.")
        else:
            print("Invalid agc command format.")
        return True

    def _cmd_gain(self, cmd: str) -> bool:
        """Handle 'gain <value>' — set manual gain."""
        tokens = cmd.split()
        if len(tokens) == 2:
            try:
                gain_value = float(tokens[1])
                if self.controller.is_manual_gain():
                    self.controller.set_gain(gain_value)
                    print(f"Manual gain set to {self.controller.get_gain():.1f} dB")
                else:
                    print("Auto gain control is active. Use 'agc off' first.")
            except ValueError:
                print("Invalid gain command.")
        else:
            print("Invalid gain command format.")
        return True

    def _cmd_tune(self, cmd: str) -> bool:
        """Handle station number or frequency input — tune to station."""
        try:
            if cmd.isdigit():
                idx = int(cmd) - 1
                stations = self.controller.get_stations_list()
                if 0 <= idx < len(stations):
                    name, new_freq = stations[idx]
                    self.controller.tune(new_freq)
                    print(f"Tuned to {name} ({new_freq/1e6:.1f} MHz).")
                else:
                    print("Invalid station number.")
            else:
                freq_val = float(cmd)
                new_freq = freq_val * 1e6
                self.controller.tune(new_freq)
                print(f"Tuned to {new_freq/1e6:.1f} MHz.")
        except ValueError:
            print("Unknown command.")
        return True
