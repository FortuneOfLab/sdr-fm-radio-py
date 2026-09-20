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
"""What is actually on the band, found by looking rather than by listing.

The station catalogue says what should be there.  This says what is: the
receiver hops across the band, and every peak that stands clear of the
floor is reported with how loud it was and whether it carries a stereo
pilot.  A pilot is the strongest evidence that a peak is a broadcast
rather than a spur - nothing else at 19 kHz above a demodulated carrier
is a coincidence.

It is not evidence that the peak is a station of its own.  Tuned to
the edge of a strong signal the receiver still hears that signal, and
demodulates its pilot along with it: 82.1 MHz, which is NHK-FM at
82.5 leaking through, came back with 15 dB of pilot.  A pilot says
there is a broadcast in the channel, not that the channel is where it
is transmitted from.

Two things this had to learn from the band it was written against.

The gain is held for the whole sweep.  With the AGC running, each hop
finds its own gain and the powers from different hops are in different
units; a quiet channel after a loud one reads as loud as the loud one.

Peaks are picked rather than channels thresholded.  A strong station
spills over its neighbours - measuring every 0.1 MHz slot around a
station at 82.5 MHz reported five stations, at 82.3 through 82.8 - so a
peak has to beat everything within a station's width of it before it is
one.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import numpy as np

from fm_radio.constants import AUDIO_OUTPUT_RATE, COMPOSITE_RATE
from fm_radio.demodulator import FMDemodulator
from fm_radio.spectrum import SpectrumMaker

logger = logging.getLogger('fm_receiver.band_scan')

#: The Japanese FM band, in Hz.  76-90 MHz is FM broadcasting proper and
#: 90-95 MHz the complementary band the AM stations moved into.
BAND_START_HZ: float = 76.0e6
BAND_END_HZ: float = 95.0e6

#: Japanese allocations sit on a 0.1 MHz grid, which is also how finely
#: the tuner is asked to move.
CHANNEL_STEP_HZ: float = 100e3

#: How much of each capture to trust.  The channel filter and the
#: tuner's own skirts make the edges of the sample rate unreliable, so
#: a hop reports only the middle of what it captured and the hops
#: overlap by the rest.
USABLE_FRACTION: float = 0.8

#: How far apart two peaks have to be to be two stations.  An FM signal
#: is about 200 kHz wide, so anything within that of a louder peak is
#: the same station's skirt.
STATION_WIDTH_HZ: float = 200e3

#: How far above the floor a peak has to stand.  The floor is the
#: median of the whole sweep, which on a real band is receiver noise;
#: 10 dB over it was comfortably below the quietest real station
#: measured (93.0 MHz at -42 dBFS against a -52 dBFS floor) and well
#: above the loudest thing that was not one.
PEAK_OVER_FLOOR_DB: float = 10.0

#: The stereo pilot, and the bands either side of it that say what the
#: noise at 19 kHz would be if there were no pilot.  Same idea as the
#: demodulator's own pilot SNR, measured here on a block of composite
#: rather than tracked over time.
PILOT_HZ: float = 19000.0
PILOT_HALF_WIDTH_HZ: float = 400.0
PILOT_NOISE_OFFSET_HZ: float = 2000.0

#: Blocks thrown away after tuning before the pilot is measured.
#: The first one out of a demodulator reset carries no audio at all,
#: and the ones after it are its filters filling.
SETTLING_BLOCKS: int = 2

#: How far the pilot has to stand above that noise to count as one.
#: Measured on this radio: real stereo stations came out at 20-40 dB
#: and channels with nothing on them below 3 dB.
PILOT_OVER_NOISE_DB: float = 10.0


#: A stereo pilot was heard here: this is a broadcast.
CONFIRMED = "confirmed"
#: Something is here and nothing proved what.  A mono station reads
#: this way, and so does a strong station's skirt, and so does a
#: spur - the pilot is the only evidence this scan has, and its
#: absence is not evidence of absence.
UNCONFIRMED = "unconfirmed"
#: Unconfirmed, and there is a confirmed station close enough and
#: far enough above it to be what this is the edge of.
LIKELY_SKIRT = "likely skirt"

#: How far a skirt reaches, and how far below its station it is by
#: then.  82.1 MHz is NHK-FM at 82.5 leaking through; the peak test
#: only rejects skirts within 200 kHz, and widening that would lose
#: real neighbours instead.
#:
#: Power is the only test there is for this.  A skirt carries its
#: station's pilot - 82.1 read 15 dB of one - so being a broadcast
#: does not make a peak a station, and 82.1 comes back as a skirt
#: only when it is far enough below 82.5.  On a run where the two
#: were 1.6 dB apart it did not, and was reported as a station.
SKIRT_REACH_HZ: float = 500e3
SKIRT_BELOW_DB: float = 15.0


@dataclasses.dataclass(frozen=True)
class Signal:
    """One thing found on the band."""

    freq_hz: float
    #: Peak power in the channel, dBFS, comparable across the sweep
    #: because the gain was held for the whole of it.
    power_dbfs: float
    #: How far the 19 kHz pilot stood above the noise beside it, or
    #: None if the frequency was never listened to for one.
    pilot_over_noise_db: float | None = None

    #: CONFIRMED, UNCONFIRMED or LIKELY_SKIRT.  Set once the whole
    #: sweep is in, because what a peak is depends on its neighbours.
    sort: str = UNCONFIRMED

    @property
    def stereo(self) -> bool:
        """True when a stereo pilot was heard here.

        Not the same question as whether this is a station: a mono
        broadcast has no pilot either.  See ``sort``.
        """
        return (self.pilot_over_noise_db is not None
                and self.pilot_over_noise_db >= PILOT_OVER_NOISE_DB)

    @property
    def freq_mhz(self) -> float:
        return self.freq_hz / 1e6


def classify(signals: "list[Signal]") -> "list[Signal]":
    """Say of each peak how sure we are that it is a broadcast.

    Three answers, because there are three cases and only two of
    them look alike from here.

    A pilot and nothing louder beside it is a broadcast.  No pilot
    is not an answer at all: a mono station has none, and neither
    does a spur, and this cannot tell them apart - so they are both
    unconfirmed rather than both "not stereo", because the next
    thing to use this writes unknown stations to a file and must
    not treat a spur as a quiet station.

    A peak near something much louder is the louder one's skirt,
    whether or not it has a pilot of its own.  It has one either
    way: tuned to the edge of a strong signal the receiver still
    hears that signal, pilot and all.  So the test is on where the
    peak is and how far below its neighbour, and not on the pilot.
    """
    confirmed = [s for s in signals if s.stereo]
    out = []
    for signal in signals:
        louder_beside = [c for c in confirmed
                         if c is not signal
                         and abs(c.freq_hz - signal.freq_hz) <= SKIRT_REACH_HZ
                         and c.power_dbfs - signal.power_dbfs
                         >= SKIRT_BELOW_DB]
        if louder_beside:
            sort = LIKELY_SKIRT
        elif signal.stereo:
            sort = CONFIRMED
        else:
            sort = UNCONFIRMED
        out.append(dataclasses.replace(signal, sort=sort))
    return out


def hop_centres(sample_rate_hz: float, start_hz: float = BAND_START_HZ,
                end_hz: float = BAND_END_HZ) -> list[float]:
    """Where to tune so that the whole band is seen once.

    One capture is a sample rate wide, of which the middle
    ``USABLE_FRACTION`` is trusted, so the hops are that far apart and
    the first and last reach past the band's edges rather than stopping
    short of them.
    """
    usable = float(sample_rate_hz) * USABLE_FRACTION
    if not usable > 0.0:
        raise ValueError("a sample rate of %r covers nothing"
                         % (sample_rate_hz,))
    centre = float(start_hz) + usable / 2.0
    centres = []
    while centre - usable / 2.0 < float(end_hz):
        centres.append(centre)
        centre += usable
    return centres


def channel_powers(frame, start_hz: float = BAND_START_HZ,
                   end_hz: float = BAND_END_HZ,
                   step_hz: float = CHANNEL_STEP_HZ) -> dict[float, float]:
    """The loudest bin in each allocation slot this frame covers.

    Only the middle of the frame: see USABLE_FRACTION.  Returns dBFS by
    frequency, with the frequencies on the allocation grid so that two
    hops that saw the same channel can be compared.
    """
    if not frame.dbfs:
        return {}
    freqs = frame.frequencies_hz()
    dbfs = np.asarray(frame.dbfs, dtype=np.float64)
    reach = frame.span_hz * USABLE_FRACTION / 2.0
    found: dict[float, float] = {}
    first = int(np.ceil((max(frame.center_hz - reach, start_hz) - start_hz)
                        / step_hz))
    last = int(np.floor((min(frame.center_hz + reach, end_hz) - start_hz)
                        / step_hz))
    for index in range(first, last + 1):
        channel = start_hz + index * step_hz
        near = np.abs(freqs - channel) <= step_hz / 2.0
        if near.any():
            found[round(channel, 1)] = float(dbfs[near].max())
    return found


def peaks(powers: dict[float, float],
          width_hz: float = STATION_WIDTH_HZ,
          over_floor_db: float = PEAK_OVER_FLOOR_DB) -> list[Signal]:
    """The channels that are stations rather than a station's skirts.

    A peak has to be the loudest thing within ``width_hz`` of itself
    and stand ``over_floor_db`` above the median of everything looked
    at.  The median is the floor because most of the band is empty.

    On this band a mean would do as well - seven stations in a
    hundred and ninety-one channels move it about 3 dB - and the
    median is here for the band where that is not true: something
    loud across half of it, or a sweep asked to look at a narrow
    range that is mostly stations, walks a mean past the quiet ones.
    """
    if not powers:
        return []
    floor = float(np.median(list(powers.values())))
    threshold = floor + over_floor_db
    found = []
    for freq, power in sorted(powers.items()):
        if power < threshold:
            continue
        neighbours = [p for f, p in powers.items()
                      if f != freq and abs(f - freq) <= width_hz]
        if neighbours and max(neighbours) > power:
            continue                    # somebody louder owns this bump
        if any(p == power and f < freq for f, p in powers.items()
               if abs(f - freq) <= width_hz):
            continue                    # a tie; the lower one keeps it
        found.append(Signal(freq_hz=freq, power_dbfs=power))
    return found


def pilot_and_noise_power(composite: np.ndarray,
                          rate_hz: float = COMPOSITE_RATE
                          ) -> "tuple[float, float] | None":
    """Power at 19 kHz, and power in the bands either side of it.

    The two halves of the pilot measurement, unreduced, so that a
    listen of several blocks can add them up before dividing once.

    None when the block is too short to resolve the bands, which is
    what a hop that captured nothing gives.
    """
    if composite.size < int(rate_hz / PILOT_HALF_WIDTH_HZ):
        return None
    windowed = composite * np.hanning(composite.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(composite.size, 1.0 / rate_hz)

    def power_in(centre: float) -> float:
        band = np.abs(freqs - centre) <= PILOT_HALF_WIDTH_HZ
        return float(spectrum[band].mean()) if band.any() else 0.0

    pilot = power_in(PILOT_HZ)
    below = power_in(PILOT_HZ - PILOT_NOISE_OFFSET_HZ)
    above = power_in(PILOT_HZ + PILOT_NOISE_OFFSET_HZ)
    return pilot, 0.5 * (below + above)


def over_noise_db(pilot_power: float, noise_power: float) -> float:
    """The pilot reading the two powers amount to."""
    return 10.0 * np.log10((pilot_power + 1e-30) / (noise_power + 1e-30))


def pilot_over_noise_db(composite: np.ndarray,
                        rate_hz: float = COMPOSITE_RATE) -> float | None:
    """How far 19 kHz stands above the noise either side of it.

    The stereo pilot is the one thing that says a carrier is a
    broadcast and not a spur: a transmitter puts it there on purpose,
    and nothing else does.

    One block of it.  A scan listens for longer than that and adds
    the powers up before dividing - see BandScan._listen_for_a_pilot,
    and why taking the best of several readings was wrong.
    """
    both = pilot_and_noise_power(composite, rate_hz)
    return None if both is None else over_noise_db(*both)


class ScanFailed(Exception):
    """The receiver would not do what the sweep asked of it."""


class BandScan:
    """Sweeps the band and says what is on it.

    Takes the receiver over for the length of the sweep: it retunes,
    holds the gain, and puts both back afterwards - including when it
    is cancelled or something goes wrong halfway.  The audio is of
    wherever the sweep happens to be while it runs, which is what a
    scan sounds like on any radio.

    The blocks it looks at are copies the receiver hands over, not
    blocks taken from the demodulator's queue: see
    SDRReceiver.watch_the_blocks.  Taking them left the demodulator
    with gaps that its filters carried straight across, which is a
    worse noise than the sweep's own.

    A pilot is listened for over a quarter of a second, and the
    powers are added up before being divided once.  Taking the best
    reading of several blocks, which is what this did first, gets
    better at finding noise the longer it listens and no better at
    finding a weak pilot.
    """

    def __init__(self, controller, on_progress=None) -> None:
        self.controller = controller
        self._on_progress = on_progress
        self._stop = threading.Event()
        #: Where the receiver copies its blocks while a sweep runs.
        self._blocks: "object | None" = None
        # Its own, not the receiver's.  A scan runs on whatever thread
        # started it while the processing thread is still going, and
        # both the spectrum maker and the demodulator carry state from
        # one block to the next - filter histories, cached windows,
        # the pilot's phase.  Sharing them would have two threads
        # stepping on one set of it, and the audio would be the one
        # that suffered.
        rate = controller.sdr_receiver.sample_rate
        self._spectrum = SpectrumMaker(rate)
        self._demodulator = FMDemodulator(
            iq_sample_rate=rate, final_audio_rate=AUDIO_OUTPUT_RATE,
            stereo=True)

    def cancel(self) -> None:
        """Ask the sweep to stop at the next hop."""
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def run(self, listen_sec: float = 0.25) -> list[Signal]:
        """Sweep the band and return what was found, loudest first.

        Args:
            listen_sec: How long to stay on each candidate listening
                for a pilot.  A quarter of a second is fifteen blocks,
                which is more than the demodulator needs to settle.

        Returns:
            The signals found, or as many as were found before the
            sweep was cancelled.
        """
        was_at = self.controller.get_frequency()
        was_auto = not self.controller.is_manual_gain()
        sdr = self.controller.sdr_receiver
        self._blocks = sdr.watch_the_blocks()
        try:
            found = self._sweep(listen_sec)
        except BaseException as went_wrong:
            # The sweep's own failure is the one worth raising; a
            # failure to tidy up after it is a second line on the
            # same story.
            try:
                self._put_the_receiver_back(was_at, was_auto)
            except Exception as and_then:
                went_wrong.add_note(
                    "the receiver was not put back: %s" % and_then)
            raise
        finally:
            sdr.stop_watching(self._blocks)
            self._blocks = None
        # Not in the finally: a sweep that worked and left the
        # receiver somewhere else has not worked.  The person who
        # started it was listening to something.
        self._put_the_receiver_back(was_at, was_auto)
        return found

    # ------------------------------------------------------------------

    def _sweep(self, listen_sec: float) -> list[Signal]:
        self._hold_the_gain()
        powers = self._look_at_the_band()
        found = peaks(powers)
        self._say("found %d signals; listening for pilots" % len(found))
        heard = []
        for signal in found:
            if self.cancelled:
                break
            heard.append(self._listen_for_a_pilot(signal, listen_sec))
        heard = classify(heard)
        heard.sort(key=lambda s: s.power_dbfs, reverse=True)
        return heard

    def _look_at_the_band(self) -> dict[float, float]:
        """One capture per hop, turned into power per allocation."""
        powers: dict[float, float] = {}
        centres = hop_centres(self.controller.sdr_receiver.sample_rate)
        for number, centre in enumerate(centres, start=1):
            if self.cancelled:
                break
            self._say("looking at %.1f MHz (%d of %d)"
                      % (centre / 1e6, number, len(centres)))
            frame = self._frame_at(centre)
            if frame is None:
                continue
            for freq, power in channel_powers(frame).items():
                # The hops overlap at the edges; the louder reading is
                # the one taken nearer the middle of a capture.
                powers[freq] = max(powers.get(freq, -np.inf), power)
        return powers

    def _frame_at(self, centre_hz: float):
        """Tune there and make one picture of what is around it."""
        self._tune_and_settle(centre_hz)
        block = self._a_fresh_block()
        if block is None:
            logger.debug("no block came back at %.1f MHz", centre_hz / 1e6)
            return None
        return self._spectrum.frame(block, centre_hz, time.perf_counter())

    def _listen_for_a_pilot(self, signal: "Signal",
                            listen_sec: float) -> "Signal":
        """Stay on a candidate long enough to hear whether it is stereo."""
        self._say("listening at %.1f MHz" % signal.freq_mhz)
        self._tune_and_settle(signal.freq_hz)
        # Nothing of the last frequency in the filters: the first
        # block here is of this station and must be demodulated as
        # though it were the first of a session.
        self._demodulator.reset()
        deadline = time.monotonic() + listen_sec
        pilot = noise = 0.0
        blocks = 0
        while time.monotonic() < deadline and not self.cancelled:
            block = self._a_fresh_block()
            if block is None:
                break
            composite = self._demodulator.process_iq_samples(block)
            both = pilot_and_noise_power(composite)
            if both is None:
                continue
            blocks += 1
            if blocks <= SETTLING_BLOCKS:
                # The filters are still filling, and the first block
                # out of a reset carries no audio at all.
                continue
            pilot += both[0]
            noise += both[1]
        heard = over_noise_db(pilot, noise) if noise > 0.0 else None
        return dataclasses.replace(signal, pilot_over_noise_db=heard)

    # ------------------------------------------------------------------

    def _hold_the_gain(self) -> None:
        """Fix the gain, so that two hops can be compared.

        With the AGC running every hop finds its own gain and the
        powers come back in different units: a quiet channel after a
        loud one reads as loud as the loud one did.
        """
        if not self.controller.is_manual_gain():
            self._wait_for(self.controller.set_agc_mode(False),
                           "holding the gain")

    def _tune_and_settle(self, freq_hz: float) -> None:
        self._wait_for(self.controller.tune(freq_hz),
                       "tuning to %.1f MHz" % (freq_hz / 1e6))

    def _a_fresh_block(self, timeout_sec: float = 1.0):
        """The next block that belongs to where the tuner is now.

        From the watcher's queue, not from the one the demodulator
        reads: that is a work queue, so a block taken from it is a
        block the receiver never sees.  Taking them during a sweep
        left the demodulator with gaps its filters carried straight
        across, which is a worse noise than the sweep's own.
        """
        if self._blocks is None:                # pragma: no cover - guard
            return None
        sdr = self.controller.sdr_receiver
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.cancelled:
                return None
            wanted = sdr.tuning_generation
            try:
                generation, block = self._blocks.get(timeout=0.1)
            except Exception:
                continue
            if generation == wanted:
                return block
        return None

    def _put_the_receiver_back(self, freq_hz: float, auto_gain: bool) -> None:
        """Where it was, however the sweep ended.

        Both halves are tried even if the first one fails - a tuner
        that would not move is no reason to leave the AGC off as well
        - and what did not work is raised afterwards rather than
        logged and forgotten.  A receiver left on a frequency nobody
        asked for, with the gain pinned, is not a finished scan.
        """
        trouble = []
        try:
            self._wait_for(self.controller.tune(freq_hz),
                           "tuning back to %.1f MHz" % (freq_hz / 1e6))
        except Exception as e:
            logger.error("%s", e)
            trouble.append(str(e))
        if auto_gain:
            try:
                self._wait_for(self.controller.set_agc_mode(True),
                               "putting the gain back on auto")
            except Exception as e:
                logger.error("%s", e)
                trouble.append(str(e))
        if trouble:
            raise ScanFailed("; and ".join(trouble))

    @staticmethod
    def _wait_for(request, what: str, timeout_sec: float = 5.0) -> None:
        """Let a device request finish, and find out whether it did.

        A sweep cannot run ahead of the tuner, and it cannot carry on
        as though a write happened when it did not: the next thing it
        does is label a block with the frequency it asked for.  A
        None request means there was nothing to do - asking for a
        mode the receiver is already in - which is success.
        """
        if request is None:
            return
        if not request.wait(timeout_sec):
            raise ScanFailed("%s did not finish in %.0f s"
                             % (what, timeout_sec))
        if request.failed:
            raise ScanFailed("%s failed: %s" % (what, request.error))
        if request.superseded:
            raise ScanFailed("%s was replaced by a later one" % what)
        if request.cancelled:
            raise ScanFailed("%s was cancelled" % what)

    def _say(self, what: str) -> None:
        logger.debug("%s", what)
        if self._on_progress is not None:
            try:
                self._on_progress(what)
            except Exception as e:              # pragma: no cover - guard
                logger.debug("progress report failed: %s", e)
