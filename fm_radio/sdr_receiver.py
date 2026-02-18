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
"""SDR receiver class using RTL-SDR."""

from __future__ import annotations

import queue
import logging

import numpy as np
from rtlsdr import RtlSdr

from fm_radio.interfaces import SDRReceiverInterface
from fm_radio.constants import SDR_SAMPLE_RATE, SDR_CENTER_FREQ_DEFAULT, SDR_BLOCK_SIZE, SDR_QUEUE_MAXSIZE


class SDRReceiver(SDRReceiverInterface):
    """
    Receiver class using RTL-SDR

    Retrieves samples from the RTL-SDR device and asynchronously stores them in a queue.
    """
    def __init__(
        self,
        sample_rate: float = SDR_SAMPLE_RATE,
        center_freq: float = SDR_CENTER_FREQ_DEFAULT,
        block_size: int = SDR_BLOCK_SIZE,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger('fm_receiver.SDRReceiver')
        self.sample_rate: float = sample_rate
        self.center_freq: float = center_freq
        self.block_size: int = block_size
        self.data_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=SDR_QUEUE_MAXSIZE)

        try:
            self.sdr: RtlSdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.set_manual_gain_enabled(False)
            self.manual_gain: bool = False
            self.sdr.set_gain(0)
            self.logger.info(f"SDR initialized: sample_rate={sample_rate/1e6:.3f}MHz, center_freq={center_freq/1e6:.1f}MHz")
        except Exception as e:
            self.logger.error(f"Failed to initialize RTL-SDR device: {e}")
            raise

        try:
            # Disable direct_sampling if available
            self.sdr.direct_sampling = 0
            self.logger.debug("Direct sampling disabled")
        except Exception as e:
            self.logger.warning(f"Failed to disable direct_sampling (may not be supported): {e}")

    def set_center_frequency(self, freq: float) -> None:
        """Change the center frequency."""
        try:
            self.center_freq = freq
            self.sdr.center_freq = freq
            self.logger.info(f"Center frequency set to {freq/1e6:.1f} MHz")
        except Exception as e:
            self.logger.error(f"Failed to set center frequency to {freq/1e6:.1f} MHz: {e}")
            raise

    def get_center_frequency(self) -> float:
        """Retrieve the current center frequency."""
        return self.sdr.center_freq

    def set_gain(self, gain: float) -> None:
        """Set gain value (for manual mode)."""
        try:
            self.sdr.set_gain(gain)
            self.logger.info(f"Gain set to {gain:.1f} dB")
        except Exception as e:
            self.logger.error(f"Failed to set gain to {gain:.1f} dB: {e}")
            raise

    def get_gain(self) -> float:
        """Retrieve the current gain value."""
        return self.sdr.get_gain()

    def set_manual_gain_mode(self, manual: bool) -> None:
        """
        Set manual gain mode.

        Args:
            manual (bool): True for manual mode, False for AGC.
        """
        try:
            self.manual_gain = manual
            self.sdr.set_manual_gain_enabled(manual)
            mode = "manual" if manual else "AGC"
            self.logger.info(f"Gain mode set to {mode}")
        except Exception as e:
            self.logger.error(f"Failed to set gain mode: {e}")
            raise

    def callback(self, iq_samples: np.ndarray, sdr_obj: RtlSdr) -> None:
        """Callback to store received IQ samples in the data queue.

        Args:
            iq_samples: Received IQ samples.
            sdr_obj: SDR object (unused).
        """
        try:
            # Convert to numpy array allowing a copy if necessary (NumPy 2.x compatibility).
            iq = np.asarray(iq_samples, dtype=np.complex64)
            self.data_queue.put(iq, block=False)
        except queue.Full:
            # Discard sample if queue is full.
            self.logger.debug("SDR data queue full, dropping samples")
        except Exception as e:
            # Drop this buffer if conversion fails to avoid crashing the SDR ctypes callback.
            self.logger.error(f"Error in SDR callback: {e}", exc_info=True)

    def start(self) -> None:
        """Start asynchronous sample retrieval."""
        try:
            self.logger.info("Starting SDR async read")
            self.sdr.read_samples_async(self.callback, num_samples=self.block_size)
        except Exception as e:
            self.logger.error(f"Failed to start SDR async read: {e}")
            raise

    def stop(self) -> None:
        """Stop asynchronous sample retrieval and close SDR."""
        try:
            self.logger.info("Stopping SDR async read")
            self.sdr.cancel_read_async()
        except Exception as e:
            self.logger.warning(f"Error canceling async read: {e}")

        try:
            self.sdr.close()
            self.logger.info("SDR closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing SDR: {e}")
