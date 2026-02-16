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
"""Command line interface for the FM receiver."""

import time
import threading


class CommandLineInterface(threading.Thread):
    """
    Thread for handling command line input

    Receives user commands and controls FMReceiverController functions.
    """
    def __init__(self, controller):
        super().__init__(daemon=True)
        self.controller = controller

    def run(self):
        while not self.controller.quit_event.is_set():
            print("\nEnter command:")
            print("  'list' -> show station list")
            print("  'stereo on/off' or 'mono' -> toggle stereo demodulation")
            print("  'record start' -> start recording with auto-generated filename")
            print("  'record stop' -> stop recording")
            print("  'agc on' -> enable AGC")
            print("  'agc off' -> disable AGC (manual mode)")
            print("  'gain <value>' -> set manual gain")
            print("  <station_num> or <freq_MHz> -> tune")
            print("  'q' -> quit")
            cmd = input().strip().lower()

            if cmd == 'q':
                print("Exiting command input...")
                self.controller.quit_event.set()
                break
            elif cmd == 'list':
                print("Available stations:")
                for i, (name, freq) in enumerate(self.controller.stations_list, start=1):
                    print(f"{i}: {name} ({freq/1e6:.1f} MHz)")
            elif cmd in ("stereo on", "stereo"):
                if hasattr(self.controller.fm_demodulator, 'stereo'):
                    self.controller.fm_demodulator.stereo = True
                    print("Stereo demodulation enabled.")
                else:
                    print("Stereo demodulation not supported.")
            elif cmd in ("stereo off", "mono"):
                if hasattr(self.controller.fm_demodulator, 'stereo'):
                    self.controller.fm_demodulator.stereo = False
                    print("Mono demodulation enabled.")
                else:
                    print("Stereo demodulation not supported.")
            elif cmd == "record start":
                current_time = time.strftime("%Y%m%d_%H%M%S")
                freq = self.controller.sdr_receiver.get_center_frequency() / 1e6
                filename = f"{current_time}_{freq:.1f}MHz.wav"
                self.controller.audio_output.start_recording(filename)
            elif cmd == "record stop":
                self.controller.audio_output.stop_recording()
            elif cmd.startswith("agc"):
                tokens = cmd.split()
                if len(tokens) == 2:
                    if tokens[1] in ["on"]:
                        self.controller.sdr_receiver.set_manual_gain_mode(False)
                        print("Automatic gain control enabled.")
                    elif tokens[1] in ["off"]:
                        self.controller.sdr_receiver.set_manual_gain_mode(True)
                        print(f"Manual gain control enabled. Current gain: {self.controller.sdr_receiver.get_gain():.1f}")
                    else:
                        print("Invalid agc command format.")
                elif len(tokens) == 3:
                    if tokens[1] in ["off"]:
                        try:
                            gain_value = float(tokens[2])
                            self.controller.sdr_receiver.set_manual_gain_mode(True)
                            self.controller.sdr_receiver.set_gain(gain_value)
                            print(f"Manual gain control enabled. Gain set to {gain_value:.1f}")
                        except ValueError:
                            print("Invalid gain value.")
                    else:
                        print("Invalid gain command format.")
                else:
                    print("Invalid agc command format.")
            elif cmd.startswith("gain"):
                tokens = cmd.split()
                if len(tokens) == 2:
                    try:
                        gain_value = float(tokens[1])
                        if self.controller.sdr_receiver.manual_gain:
                            self.controller.sdr_receiver.set_gain(gain_value)
                            print(f"Manual gain set to {gain_value:.1f}")
                        else:
                            print("Automatic gain control is enabled. Please disable it to set manual gain.")
                    except ValueError:
                        print("Invalid gain command.")
                else:
                    print("Invalid gain command format.")
            else:
                try:
                    if cmd.isdigit():
                        idx = int(cmd) - 1
                        if 0 <= idx < len(self.controller.stations_list):
                            new_freq = self.controller.stations_list[idx][1]
                            self.controller.sdr_receiver.set_center_frequency(new_freq)
                            print(f"Tuned to {self.controller.stations_list[idx][0]} ({new_freq/1e6:.1f} MHz).")
                            self.controller.flush_data_queue()
                            self.controller.fm_demodulator.reset()
                            if self.controller.audio_output.recording:
                                self.controller.audio_output.stop_recording()
                        else:
                            print("Invalid station number.")
                    else:
                        freq_val = float(cmd)
                        new_freq = freq_val * 1e6
                        self.controller.sdr_receiver.set_center_frequency(new_freq)
                        print(f"Tuned to {new_freq/1e6:.1f} MHz.")
                        self.controller.flush_data_queue()
                        self.controller.fm_demodulator.reset()
                        if self.controller.audio_output.recording:
                            self.controller.audio_output.stop_recording()
                except ValueError:
                    print("Unknown command.")
