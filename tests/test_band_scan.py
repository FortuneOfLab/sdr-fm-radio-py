"""Sweeping the band to find out what is on it.

The arithmetic is tested against built spectra, and the sweep against a
stand-in receiver: what it tunes to, what it holds, what it puts back.
The figures from the real band are in the pull request.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from fm_radio.band_scan import (
    BAND_END_HZ, BAND_START_HZ, PILOT_HZ, PILOT_OVER_NOISE_DB, BandScan,
    Signal, channel_powers, hop_centres, peaks, pilot_over_noise_db,
)
from fm_radio.constants import COMPOSITE_RATE
from fm_radio.spectrum import SpectrumFrame


# ----------------------------------------------------------------------
# Where to tune
# ----------------------------------------------------------------------

def test_the_hops_cover_the_whole_band():
    centres = hop_centres(1.024e6)

    assert centres[0] - 1.024e6 * 0.4 <= BAND_START_HZ
    assert centres[-1] + 1.024e6 * 0.4 >= BAND_END_HZ


def test_the_hops_do_not_leave_gaps_between_them():
    """Each one picks up where the last one stopped being trusted."""
    rate = 1.024e6
    reach = rate * 0.4
    centres = hop_centres(rate)

    for before, after in zip(centres, centres[1:]):
        assert after - reach <= before + reach + 1.0, (
            "nothing looks at %.3f-%.3f MHz"
            % ((before + reach) / 1e6, (after - reach) / 1e6))


def test_a_wider_receiver_takes_fewer_hops():
    assert len(hop_centres(2.048e6)) < len(hop_centres(1.024e6))


def test_a_receiver_that_sees_nothing_is_an_error():
    with pytest.raises(ValueError):
        hop_centres(0.0)


# ----------------------------------------------------------------------
# What each hop sees
# ----------------------------------------------------------------------

def a_frame(center_hz: float, span_hz: float = 1.024e6, bins: int = 512,
            loud_at: dict[float, float] | None = None) -> SpectrumFrame:
    """A flat band with whatever was asked for standing out of it."""
    dbfs = np.full(bins, -60.0)
    frame = SpectrumFrame(center_hz, span_hz, tuple(dbfs), 1.0)
    freqs = frame.frequencies_hz()
    for hz, power in (loud_at or {}).items():
        dbfs[np.argmin(np.abs(freqs - hz))] = power
    return SpectrumFrame(center_hz, span_hz, tuple(dbfs), 1.0)


def test_only_the_middle_of_a_capture_is_believed():
    """The edges are the tuner's skirts, and the next hop covers them."""
    frame = a_frame(80.0e6)

    looked_at = sorted(channel_powers(frame))

    assert min(looked_at) >= 80.0e6 - 1.024e6 * 0.4 - 1.0
    assert max(looked_at) <= 80.0e6 + 1.024e6 * 0.4 + 1.0


def test_channels_are_reported_on_the_allocation_grid():
    """Two hops that saw the same channel have to agree what to call it."""
    one = channel_powers(a_frame(80.0e6))
    two = channel_powers(a_frame(80.8e6))
    shared = set(one) & set(two)

    assert shared, "the hops did not overlap at all"
    for hz in shared:
        assert abs(round(hz / 1e5) * 1e5 - hz) < 1.0


def test_nothing_outside_the_band_is_looked_at():
    below = channel_powers(a_frame(BAND_START_HZ))
    above = channel_powers(a_frame(BAND_END_HZ))

    assert min(below) >= BAND_START_HZ
    assert max(above) <= BAND_END_HZ


def test_a_frame_with_nothing_in_it_says_nothing():
    assert channel_powers(SpectrumFrame(80.0e6, 1.024e6, (), 1.0)) == {}


# ----------------------------------------------------------------------
# Which of them are stations
# ----------------------------------------------------------------------

def a_band(loud: dict[float, float], floor: float = -52.0) -> dict:
    """The whole band at the floor, with some channels standing out."""
    powers = {round(BAND_START_HZ + i * 1e5, 1): floor for i in range(191)}
    for mhz, db in loud.items():
        powers[round(mhz * 1e6, 1)] = db
    return powers


def test_a_station_is_reported_once_and_not_as_its_own_skirts():
    """The band this was written against.

    Reading every 0.1 MHz slot around NHK-FM at 82.5 reported five
    stations, 82.3 through 82.8, because a strong signal spills over
    its neighbours.  Only the top of the hill is a station.
    """
    powers = a_band({82.3: -37.2, 82.4: -24.3, 82.5: -9.3,
                     82.6: -24.8, 82.7: -35.3})

    found = peaks(powers)

    assert [round(s.freq_mhz, 1) for s in found] == [82.5]


def test_stations_further_apart_than_one_is_wide_are_both_kept():
    powers = a_band({81.3: 0.3, 82.5: -9.3, 90.5: -34.5})

    assert [round(s.freq_mhz, 1) for s in peaks(powers)] == [81.3, 82.5, 90.5]


def test_the_floor_is_not_a_station():
    assert peaks(a_band({})) == []


def test_a_bump_too_small_to_be_a_station_is_not_one():
    powers = a_band({85.0: -52.0 + 5.0})      # under PEAK_OVER_FLOOR_DB

    assert peaks(powers) == []


def test_a_band_that_is_mostly_signal_still_has_a_floor():
    """Where the median earns its place over a mean.

    Not on the band this was written against: seven stations in a
    hundred and ninety-one channels move a mean by about 3 dB, and
    either would do.  It is the case where something is loud across
    half the band - a transmitter splattering, or a sweep asked to
    look at a narrow range that is mostly stations - that a mean
    walks the threshold up past the quiet ones.
    """
    powers = a_band({76.0 + 0.1 * i: -10.0 for i in range(90)})
    powers[round(93.0e6, 1)] = -40.0         # 12 dB over the real floor

    assert 93.0 in [round(s.freq_mhz, 1) for s in peaks(powers)]


def test_nothing_at_all_is_not_an_error():
    assert peaks({}) == []


# ----------------------------------------------------------------------
# Whether it is a broadcast
# ----------------------------------------------------------------------

def composite(pilot_amplitude: float, noise: float = 0.0,
              seconds: float = 0.05, seed: int = 5) -> np.ndarray:
    """A block of composite with a 19 kHz pilot of a given size."""
    n = int(COMPOSITE_RATE * seconds)
    t = np.arange(n) / COMPOSITE_RATE
    rng = np.random.default_rng(seed)
    return (pilot_amplitude * np.sin(2.0 * np.pi * PILOT_HZ * t)
            + noise * rng.standard_normal(n)).astype(np.float32)


def test_a_pilot_stands_out_of_the_noise_beside_it():
    heard = pilot_over_noise_db(composite(0.1, noise=0.05))

    assert heard > PILOT_OVER_NOISE_DB


def test_noise_with_no_pilot_in_it_is_not_a_pilot():
    heard = pilot_over_noise_db(composite(0.0, noise=0.05))

    assert heard < PILOT_OVER_NOISE_DB


def test_a_louder_pilot_reads_higher():
    quieter = pilot_over_noise_db(composite(0.05, noise=0.05))

    assert pilot_over_noise_db(composite(0.2, noise=0.05)) > quieter


def test_a_tone_that_is_not_the_pilot_is_not_one():
    """Programme material goes up to 15 kHz, and the pilot is at 19."""
    n = int(COMPOSITE_RATE * 0.05)
    t = np.arange(n) / COMPOSITE_RATE
    music = (0.3 * np.sin(2.0 * np.pi * 14000.0 * t)).astype(np.float32)

    assert pilot_over_noise_db(music) < PILOT_OVER_NOISE_DB


def test_a_block_too_short_to_resolve_the_bands_says_nothing():
    assert pilot_over_noise_db(np.zeros(8, dtype=np.float32)) is None


@pytest.mark.parametrize("reading,stereo", [
    (54.6, True),
    (PILOT_OVER_NOISE_DB, True),
    (PILOT_OVER_NOISE_DB - 0.1, False),
    (3.1, False),
    (None, False),
])
def test_what_counts_as_stereo(reading, stereo):
    assert Signal(80.0e6, -10.0, reading).stereo is stereo


# ----------------------------------------------------------------------
# The sweep itself
# ----------------------------------------------------------------------

class FakeRequest:
    def wait(self, timeout=None):
        return True


class FakeSDR:
    sample_rate = 1.024e6

    def __init__(self) -> None:
        self.tuning_generation = 0
        self.data_queue = _AlwaysABlock(self)


class _AlwaysABlock:
    """A queue that hands out a block of the frequency now tuned."""

    def __init__(self, sdr) -> None:
        self._sdr = sdr

    def get(self, timeout=None):
        rng = np.random.default_rng(1)
        block = (rng.standard_normal(16384)
                 + 1j * rng.standard_normal(16384)).astype(np.complex64)
        return self._sdr.tuning_generation, block


class FakeController:
    """Enough receiver to be swept, and a record of what was asked."""

    def __init__(self, auto_gain: bool = True, at_hz: float = 80.0e6) -> None:
        self.sdr_receiver = FakeSDR()
        self._freq = at_hz
        self._auto = auto_gain
        self.tuned_to: list[float] = []
        self.agc_calls: list[bool] = []
        self.fm_demodulator = None          # a scan must not reach for this
        self._spectrum_maker = None         # nor this

    def get_frequency(self) -> float:
        return self._freq

    def is_manual_gain(self) -> bool:
        return not self._auto

    def tune(self, freq_hz):
        self.tuned_to.append(freq_hz)
        self._freq = freq_hz
        self.sdr_receiver.tuning_generation += 1
        return FakeRequest()

    def set_agc_mode(self, enabled):
        self.agc_calls.append(bool(enabled))
        self._auto = bool(enabled)
        return FakeRequest()


def test_the_gain_is_held_for_the_whole_sweep():
    """Or the hops come back in different units.

    With the AGC running each hop finds its own gain, and a quiet
    channel after a loud one reads as loud as the loud one did.
    """
    controller = FakeController(auto_gain=True)

    BandScan(controller).run(listen_sec=0.0)

    assert controller.agc_calls[0] is False, "the AGC was left running"


def test_the_receiver_is_put_back_where_it_was():
    controller = FakeController(auto_gain=True, at_hz=81.3e6)

    BandScan(controller).run(listen_sec=0.0)

    assert controller.get_frequency() == 81.3e6
    assert controller.agc_calls[-1] is True, "the AGC was left off"


def test_a_receiver_on_manual_gain_is_left_on_manual():
    """Putting it back means back, not back to automatic."""
    controller = FakeController(auto_gain=False)

    BandScan(controller).run(listen_sec=0.0)

    assert True not in controller.agc_calls
    assert controller.is_manual_gain()


def test_the_receiver_is_put_back_even_when_the_sweep_fails(monkeypatch):
    """A sweep that broke halfway still moved the tuner and the gain.

    The person who started it was listening to something.
    """
    controller = FakeController(auto_gain=True, at_hz=82.5e6)
    scan = BandScan(controller)

    def explode(*args, **kwargs):
        raise RuntimeError("the sweep broke")

    monkeypatch.setattr(scan, "_look_at_the_band", explode)

    with pytest.raises(RuntimeError):
        scan.run(listen_sec=0.0)

    assert controller.get_frequency() == 82.5e6
    assert controller.agc_calls[-1] is True


def test_a_cancelled_sweep_stops_and_puts_things_back():
    controller = FakeController(auto_gain=True, at_hz=80.0e6)
    scan = BandScan(controller)
    scan.cancel()

    found = scan.run(listen_sec=0.0)

    assert found == []
    assert controller.get_frequency() == 80.0e6
    assert controller.agc_calls[-1] is True


def test_cancelling_partway_through_stops_the_hops():
    controller = FakeController()
    scan = BandScan(controller)
    hops = []
    real = scan._frame_at

    def watched(centre_hz):
        hops.append(centre_hz)
        if len(hops) == 3:
            scan.cancel()
        return real(centre_hz)

    scan._frame_at = watched
    scan.run(listen_sec=0.0)

    assert len(hops) == 3, "it went on hopping after being told to stop"


def test_the_sweep_says_what_it_is_doing():
    said = []
    BandScan(FakeController(), on_progress=said.append).run(listen_sec=0.0)

    assert said, "a sweep of a whole band said nothing about itself"
    assert any("MHz" in line for line in said)


def test_a_progress_report_that_fails_does_not_stop_the_sweep():
    """It is somebody else's window, and this is halfway through a band."""
    def unhappy(_line):
        raise RuntimeError("the window has gone")

    controller = FakeController(at_hz=80.0e6)
    BandScan(controller, on_progress=unhappy).run(listen_sec=0.0)

    assert controller.get_frequency() == 80.0e6


def test_the_sweep_brings_its_own_dsp():
    """The receiver's belongs to the processing thread, which is running.

    Both the spectrum maker and the demodulator carry state from one
    block to the next, and two threads stepping on one set of it would
    come out of the audio.
    """
    controller = FakeController()
    scan = BandScan(controller)

    assert scan._spectrum is not None
    assert scan._demodulator is not None
    assert scan._spectrum is not controller._spectrum_maker
    assert scan._demodulator is not controller.fm_demodulator

    scan.run(listen_sec=0.0)        # must not reach for the None ones
