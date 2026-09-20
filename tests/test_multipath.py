"""How steady the envelope is, and what moves it.

The numbers here are built rather than captured: a frequency-modulated
carrier with a known amount of something else added to it, so that what
the measurement says can be checked against what was put in.  The
figures it produces on the real radio are in the pull request.
"""

from __future__ import annotations

import numpy as np
import pytest

from fm_radio.multipath import CLEAN_AM_DEPTH, NOISE_AM_DEPTH, am_depth


def fm_carrier(samples: int = 16384, rate: float = 1.024e6,
               tone_hz: float = 1000.0, deviation_hz: float = 75e3,
               amplitude: float = 0.5) -> np.ndarray:
    """A carrier with everything in its phase and nothing in its size."""
    t = np.arange(samples) / rate
    phase = (2.0 * np.pi * deviation_hz / (2.0 * np.pi * tone_hz)
             * np.sin(2.0 * np.pi * tone_hz * t))
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def echoed(iq: np.ndarray, delay_samples: int, ratio: float) -> np.ndarray:
    """The same signal arriving a second time, later and quieter."""
    delayed = np.roll(iq, delay_samples)
    delayed[:delay_samples] = 0.0
    return (iq + ratio * delayed).astype(np.complex64)


# ----------------------------------------------------------------------
# What FM promises
# ----------------------------------------------------------------------

def test_a_carrier_that_holds_its_amplitude_reads_as_nothing():
    """The whole of the measurement: FM does not move the envelope."""
    assert am_depth(fm_carrier()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("amplitude", [0.02, 0.5, 2.0])
def test_the_gain_it_arrives_at_makes_no_difference(amplitude):
    """It is a ratio, and the AGC moves while this is being measured."""
    quiet = am_depth(fm_carrier(amplitude=0.01))
    assert am_depth(fm_carrier(amplitude=amplitude)) == pytest.approx(
        quiet, abs=1e-6)


def test_nothing_at_all_is_not_an_unsteady_envelope():
    """Dividing by a mean of zero would be a NaN on the display."""
    assert am_depth(np.zeros(1024, dtype=np.complex64)) == 0.0


def test_a_block_with_nothing_in_it_reads_as_nothing():
    assert am_depth(np.zeros(0, dtype=np.complex64)) == 0.0


# ----------------------------------------------------------------------
# What moves it
# ----------------------------------------------------------------------

def test_the_same_signal_arriving_twice_moves_the_envelope():
    """Multipath, which is what this is for.

    Two paths of the same transmission add and cancel across the band
    as their phase difference walks with frequency, and an FM carrier
    walks its frequency constantly - so the sum breathes.
    """
    clean = fm_carrier()

    assert am_depth(echoed(clean, 8, 0.5)) > 10.0 * am_depth(clean)


@pytest.mark.parametrize("ratio", [0.05, 0.1, 0.2, 0.4, 0.8])
def test_a_stronger_echo_reads_higher(ratio):
    """Monotonic in the thing it is measuring, or it is not a measure."""
    weaker = am_depth(echoed(fm_carrier(), 8, ratio / 2.0))

    assert am_depth(echoed(fm_carrier(), 8, ratio)) > weaker


def test_noise_reads_about_what_noise_reads():
    """An empty channel is Rayleigh, whose spread over its mean is
    sqrt(4/pi - 1).  The constant is documentation, so it has to be
    right: the real radio measured 0.65-0.68 on empty channels, which
    is this plus the receiver's own gain riding on it.
    """
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(65536)
             + 1j * rng.standard_normal(65536)).astype(np.complex64)

    assert am_depth(noise) == pytest.approx(NOISE_AM_DEPTH, abs=0.01)


def test_a_well_received_station_is_far_below_noise():
    """The two ends of the scale the constants describe are apart.

    A tenth of the carrier arriving again is a bad echo, and it is
    still nothing like an empty channel: the point of reading this
    against the pilot SNR rather than on its own.
    """
    bad_echo = am_depth(echoed(fm_carrier(), 8, 0.1))

    assert CLEAN_AM_DEPTH < NOISE_AM_DEPTH
    assert bad_echo < NOISE_AM_DEPTH
