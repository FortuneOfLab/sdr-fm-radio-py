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
"""How far the signal is from the one thing FM promises: constant amplitude.

A frequency-modulated carrier carries everything in its phase, so its
envelope should not move at all.  Anything that moves it - noise, a
neighbour leaking in, or the same transmission arriving twice by
different paths and the two adding and cancelling across the band - is
something the demodulator has to work around.

This measures the moving, and says nothing about the cause: a channel
with nothing in it looks as bad as the worst multipath, because noise
has no constant envelope either.  Read with the pilot SNR, which is
high exactly when there is a signal there to be spoiled, the two
separate: a strong pilot and a moving envelope is multipath, a weak
pilot and a moving envelope is an empty channel.
"""

from __future__ import annotations

import numpy as np

#: What a well-received station measures.  Four Tokyo stations at
#: 30-41 dB pilot SNR came out at 0.060-0.063; empty channels at
#: 0.65-0.68, which is roughly what uniform noise gives.  There is
#: nothing magic about the number - it is here so that the scale
#: means something to whoever reads one.
CLEAN_AM_DEPTH: float = 0.06

#: What noise alone measures.  The envelope of complex Gaussian noise
#: is Rayleigh, whose standard deviation over its mean is
#: sqrt(4/pi - 1) = 0.5227; measured 0.65-0.68 on an empty channel,
#: which is that plus the receiver's own gain riding on it.
NOISE_AM_DEPTH: float = 0.52


def am_depth(channel_iq: np.ndarray) -> float:
    """How much the envelope moves, as a fraction of its mean.

    ``std|z| / mean|z|``: zero for a carrier that holds its amplitude,
    rising as anything else is added to it.  Being a ratio, it does
    not care what the gain is doing, which matters on a receiver
    whose AGC moves while it measures.

    Args:
        channel_iq: The IQ of the channel being received, after the
            filter that keeps the neighbours out.  It has to be
            filtered.  Measured against the raw sample rate, a station
            at 90.1 MHz with nothing on it read 0.124 - a clean-looking
            figure produced entirely by a strong neighbour 0.4 MHz
            away - and 0.646 once the channel was all that was left.

    Returns:
        Zero or more; 0.0 for an empty array, which is what a block
        too short to say anything about should read as.
    """
    if channel_iq.size == 0:
        return 0.0
    envelope = np.abs(channel_iq)
    mean = float(np.mean(envelope))
    if not mean > 0.0:
        # Nothing at all arriving is not an unsteady envelope; it is
        # no envelope.  Dividing here would be a NaN on the display.
        return 0.0
    return float(np.std(envelope) / mean)
