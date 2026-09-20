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

import logging
import os
import time
import threading
from typing import TYPE_CHECKING, Callable

from fm_radio.console_input import ConsoleReader, console_reader
from fm_radio.constants import RECORDINGS_DIR

if TYPE_CHECKING:
    from fm_radio.controller import FMReceiverController


def build_recording_path(freq_mhz: float, iq: bool = False) -> str:
    """Build an auto-generated recording path under RECORDINGS_DIR.

    Creates the directory on first use so recordings never land in the
    repository root (which used to accumulate dozens of WAVs).
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_IQ" if iq else ""
    return os.path.join(RECORDINGS_DIR, f"{stamp}_{freq_mhz:.1f}MHz{suffix}.wav")


#: How long a typed "record start" waits before the prompt comes back
#: without a filename.  Making the file is quick; the tuner that may be
#: in front of it is not always, and the prompt should not be.
RECORD_REPORT_TIMEOUT_SEC: float = 2.0

#: How long a typed tune waits for the SDR before the prompt comes back
#: without an answer.  A tune is 60 ms of USB on a device that is
#: answering; this is long enough to look instant and short enough that a
#: device which has stopped answering does not take the prompt with it.
TUNE_REPORT_TIMEOUT_SEC: float = 2.0

#: Upper bound on catalogue rows printed at once.  The command prompt
#: shares the terminal with this output, so a full 983-line dump would
#: scroll the prompt away.
_MAX_LISTED_STATIONS = 40


class CommandLineInterface(threading.Thread):
    """Thread for handling command line input.

    Receives user commands and controls FMReceiverController via its
    public facade API.  Each command is handled by a dedicated
    ``_cmd_*`` method, keeping the main loop minimal.
    """

    def __init__(self, controller: FMReceiverController,
                 reader: "ConsoleReader | None" = None) -> None:
        super().__init__(daemon=True)
        self.controller: FMReceiverController = controller
        self.logger = logging.getLogger(__name__)
        # How the next line is waited for.  Not plain input(): a thread
        # blocked in that cannot be woken, and shutdown then has nothing
        # to do but take the process down under it.  See
        # fm_radio.console_input.
        #
        # None until run() needs one, because making one takes a pipe
        # and the window never runs this thread at all - a controller
        # built for the GUI would otherwise hold two file descriptors
        # for a command line nobody was going to type at.
        self.reader: ConsoleReader | None = reader
        # Between run() installing a reader and shutdown asking it to
        # stop, on two different threads.
        self._reader_lock: threading.Lock = threading.Lock()

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
        """Take commands until one of them - or shutdown - says stop.

        A line of None means there is not one coming: stdin ended, or
        :meth:`stop_reading` was called.  Either way this thread is
        finished, and shutdown no longer has to leave it behind.
        """
        reader = self._the_reader()
        try:
            while not self.controller.quit_event.is_set():
                self._print_help()
                cmd = reader.read_line()
                if cmd is None:
                    break
                if self.controller.quit_event.is_set():
                    # Asked to quit while this line was being typed.  A
                    # command that arrives after that is not one to run.
                    break
                if not self._dispatch(cmd.strip()):
                    break
        finally:
            reader.close()

    def _the_reader(self) -> ConsoleReader:
        """The reader this thread will use, made if there is not one yet."""
        with self._reader_lock:
            if self.reader is None:
                self.reader = console_reader(self.logger)
            return self.reader

    def stop_reading(self) -> None:
        """End the wait for a command that is not coming.

        Called from shutdown, on another thread.  Whether it can
        actually end a read that is already under way is
        ``reader.can_be_stopped``; where it cannot, the caller needs
        something blunter.  A reader that was never made is a thread
        that never ran, and there is nothing to stop.
        """
        with self._reader_lock:
            reader = self.reader
        if reader is not None:
            reader.stop()

    def close_reader(self) -> None:
        """Let go of a reader this thread is not going to use.

        A reader holds a pipe.  ``run`` closes its own on the way out,
        so this is for the times run never happened - the window, which
        builds a controller and never starts this thread, and a startup
        that failed before it could.
        """
        if self.is_alive():
            return                      # run() closes its own
        with self._reader_lock:
            reader, self.reader = self.reader, None
        if reader is not None:
            reader.close()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str) -> bool:
        """Route *cmd* to the appropriate handler.

        Resolution order:
          1. Exact match in the dispatch table.
          2. First whole word (``agc``, ``gain``, ``list``, ``search``).
          3. Numeric input interpreted as station number or frequency.

        Commands are matched case-insensitively, but the argument keeps the
        case it was typed in: a user-defined area or a station name is not
        ours to fold.  Matching on the first *word* rather than on a prefix
        keeps ``searchlight`` from being read as ``search light``.

        Returns:
            bool: True to continue the command loop, False to quit.
        """
        cmd = cmd.strip()

        # 1. Exact match
        handler = self._commands.get(cmd.lower())
        if handler:
            return handler(cmd)

        # 2. First-word match
        verb = cmd.split(maxsplit=1)[0].lower() if cmd.split() else ""
        if verb == 'agc':
            return self._cmd_agc(cmd.lower())
        if verb == 'gain':
            return self._cmd_gain(cmd.lower())
        if verb == 'list':
            return self._cmd_list(cmd)
        if verb == 'search':
            return self._cmd_search(cmd)

        # 3. Numeric / frequency input -> tune
        return self._cmd_tune(cmd)

    # ------------------------------------------------------------------
    # Command handlers  (each returns True=continue, False=quit)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_help() -> None:
        """Display available commands."""
        print("\nEnter command:")
        print("  'list' -> show preset stations")
        print("  'list all' or 'list <area>' -> browse the full catalogue")
        print("  'search <text>' -> find a station by name, site or frequency")
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
        """Handle 'list', 'list all' and 'list <area>'.

        Bare 'list' shows the presets, which are the entries the numeric
        tune command indexes into.  With an argument it browses the full
        catalogue, which is far too long to tune by number.
        """
        argument = cmd.split(maxsplit=1)[1].strip() if len(cmd.split()) > 1 else ''
        if not argument:
            print("Preset stations:")
            for i, (name, freq) in enumerate(
                    self.controller.get_stations_list(), start=1):
                print(f"{i}: {name} ({freq/1e6:.1f} MHz)")
            print("('list all' or 'list <area>' for every known station)")
            return True

        catalogue = self.controller.get_catalogue()
        if argument.lower() == 'all':
            self._print_stations(catalogue, "All stations")
        else:
            matched = self.controller.stations_in_area(argument)
            if matched:
                # Echo the area as the catalogue spells it, not as it was typed.
                self._print_stations(matched, f"Stations in {matched[0].area}")
            else:
                areas = sorted({s.area for s in catalogue if s.area})
                print(f"Unknown area: {argument}")
                print("Areas: " + " / ".join(areas))
        return True

    def _cmd_search(self, cmd: str) -> bool:
        """Handle 'search <text>' — find stations anywhere in the catalogue."""
        query = cmd.split(maxsplit=1)[1].strip() if len(cmd.split()) > 1 else ''
        if not query:
            print("Usage: search <name, transmitter site or frequency>")
            return True
        self._print_stations(self.controller.search_stations(query),
                             f"Search: {query}")
        return True

    @staticmethod
    def _print_stations(stations: list, title: str) -> None:
        """Print a catalogue slice, truncated so it cannot flood the prompt."""
        if not stations:
            print(f"{title}: no match")
            return
        print(f"{title} ({len(stations)}):")
        for station in stations[:_MAX_LISTED_STATIONS]:
            print(f"  {station.describe()}")
        hidden = len(stations) - _MAX_LISTED_STATIONS
        if hidden > 0:
            print(f"  ... and {hidden} more - narrow the search to see them")
        print("Tune by typing the frequency in MHz.")

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
        self._report_recording(self.controller.start_recording(), "Recording")
        return True

    def _report_recording(self, request, what: str) -> None:
        """Say how a recording that was asked for went, if it has yet.

        The receiver names the file, because the name says which
        station this is and only the worker knows which station that
        will be by the time the file is made.  Somebody is looking at
        the prompt, so this waits a moment for an answer - but only a
        moment, the same as a typed tune.
        """
        if not request.wait(RECORD_REPORT_TIMEOUT_SEC):
            print(f"{what} asked for; the receiver has not answered yet.")
        elif request.cancelled:
            print(f"{what} was stopped before it started.")
        elif request.superseded:
            print(f"{what} was dropped: the receiver is stopping.")
        elif request.failed:
            print(f"{what} start failed: {request.error}")
        else:
            print(f"{what} started: {request.result}")

    def _cmd_record_stop(self, cmd: str) -> bool:
        """Handle 'record stop' — stop recording, started or not.

        Asked for unconditionally: a recording that was asked for and
        has not started yet is still a recording to stop, and asking
        the flag first would leave it to start afterwards.  That is
        what happens when "record start" printed "has not answered
        yet" and the user changed their mind.
        """
        if self.controller.stop_recording():
            print("Recording stopped.")
        else:
            print("Not currently recording.")
        return True

    def _cmd_iq_record_start(self, cmd: str) -> bool:
        """Handle 'iqrec start' - begin IQ recording with auto-generated filename."""
        self._report_recording(self.controller.start_iq_recording(),
                               "IQ recording")
        return True

    def _cmd_iq_record_stop(self, cmd: str) -> bool:
        """Handle 'iqrec stop' - stop IQ recording, started or not."""
        if self.controller.stop_iq_recording():
            print("IQ recording stopped.")
        else:
            print("Not currently IQ recording.")
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
                    self._report_tuning(self.controller.tune(new_freq),
                                        f"{name} ({new_freq/1e6:.1f} MHz)")
                else:
                    print("Invalid station number.")
            else:
                freq_val = float(cmd)
                new_freq = freq_val * 1e6
                self._report_tuning(self.controller.tune(new_freq),
                                    f"{new_freq/1e6:.1f} MHz")
        except ValueError:
            print("Unknown command.")
        return True

    def _report_tuning(self, request, where: str) -> None:
        """Say how the tune went, having waited a moment for it to go.

        Somebody typed a command and is looking at the prompt, so this
        waits - unlike the window, which has a whole interface to keep
        answering.  Bounded, because the wait is the thing being avoided
        everywhere else: a device that is not answering gets a line
        saying so rather than a prompt that never comes back.
        """
        if not request.wait(TUNE_REPORT_TIMEOUT_SEC):
            print(f"Tuning to {where}... (the SDR has not answered yet)")
        elif request.superseded:
            print(f"Tuning to {where} was replaced by a later one.")
        elif request.failed:
            print(f"Could not tune to {where}: {request.error}")
        else:
            print(f"Tuned to {where}.")
