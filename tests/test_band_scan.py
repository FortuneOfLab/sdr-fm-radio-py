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
    BAND_END_HZ, BAND_START_HZ, CONFIRMED, LIKELY_SKIRT, PILOT_HZ,
    PILOT_OVER_NOISE_DB, SETTLING_BLOCKS, UNCONFIRMED, BandScan,
    ScanFailed, Signal,
    channel_powers, classify, hop_centres, over_noise_db, peaks,
    pilot_and_noise_power, pilot_over_noise_db,
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
    """A device request that went however it was told to go."""

    def __init__(self, done: bool = True, error=None,
                 superseded: bool = False, cancelled: bool = False) -> None:
        self._done = done
        self.error = error
        self.superseded = superseded
        self.cancelled = cancelled

    @property
    def failed(self) -> bool:
        return self.error is not None

    def wait(self, timeout=None):
        return self._done


class FakeSDR:
    sample_rate = 1.024e6

    def __init__(self, at_hz: float = 80.0e6) -> None:
        self.tuning_generation = 0
        #: The frequency the current generation stands for, kept with
        #: it the way the real receiver keeps it.
        self.generation_freq_hz = at_hz
        # The demodulator's own queue.  Reading from it is the defect
        # this fake is here to catch: a block taken out of it is a
        # block the receiver never sees.
        self.data_queue = _NotYours()
        self._tap = None
        self.watched = 0
        self.stopped_watching = 0

    def the_tuning_and_its_frequency(self):
        return self.tuning_generation, self.generation_freq_hz

    def moved_to(self, freq_hz: float) -> None:
        """What a tune that landed does to the tuner."""
        self.tuning_generation += 1
        self.generation_freq_hz = freq_hz

    def watch_the_blocks(self, depth: int = 1):
        if self._tap is not None:
            raise RuntimeError("something is already watching the blocks")
        self.watched += 1
        self._tap = _AlwaysABlock(self)
        return self._tap

    def stop_watching(self, tap) -> None:
        if self._tap is tap:
            self._tap = None
            self.stopped_watching += 1


class _NotYours:
    """The work queue, which a sweep must not read.

    It counts as well as complaining: the sweep retries a block that
    would not come, so raising on its own only makes a sweep that
    reads this slow rather than failed.
    """

    def __init__(self) -> None:
        self.reads = 0

    def get(self, timeout=None):
        self.reads += 1
        raise AssertionError(
            "the sweep took a block out of the demodulator's queue")


class _AlwaysABlock:
    """The watcher's queue: a block of the frequency now tuned."""

    def __init__(self, sdr) -> None:
        self._sdr = sdr

    def get(self, timeout=None):
        rng = np.random.default_rng(1)
        block = (rng.standard_normal(16384)
                 + 1j * rng.standard_normal(16384)).astype(np.complex64)
        return self._sdr.tuning_generation, block

    def put_nowait(self, item):             # pragma: no cover - unused
        raise AssertionError("the fake feeds itself")


class FakeController:
    """Enough receiver to be swept, and a record of what was asked."""

    def __init__(self, auto_gain: bool = True, at_hz: float = 80.0e6,
                 tune_answers=None, agc_answers=None) -> None:
        self.sdr_receiver = FakeSDR(at_hz)
        self._freq = at_hz
        self._auto = auto_gain
        self.tuned_to: list[float] = []
        self.agc_calls: list[bool] = []
        self.fm_demodulator = None          # a scan must not reach for this
        self._spectrum_maker = None         # nor this
        # What the device says to each request, for the sweeps that
        # have to cope with one that did not work.
        self._tune_answers = list(tune_answers or [])
        self._agc_answers = list(agc_answers or [])
        #: A frequency the tuner will not go to, whenever it is asked.
        self.will_not_tune_to = None

    def get_frequency(self) -> float:
        return self._freq

    def is_manual_gain(self) -> bool:
        return not self._auto

    def tune(self, freq_hz):
        if (self.will_not_tune_to is not None
                and abs(freq_hz - self.will_not_tune_to) < 1.0):
            self.tuned_to.append(freq_hz)
            return FakeRequest(error=OSError("the tuner is stuck"))
        answer = (self._tune_answers.pop(0) if self._tune_answers
                  else FakeRequest())
        self.tuned_to.append(freq_hz)
        if not answer.failed and not answer.superseded and answer.wait(0):
            self._freq = freq_hz
            self.sdr_receiver.moved_to(freq_hz)
        return answer

    def set_agc_mode(self, enabled):
        answer = (self._agc_answers.pop(0) if self._agc_answers
                  else FakeRequest())
        self.agc_calls.append(bool(enabled))
        if not answer.failed:
            self._auto = bool(enabled)
        return answer


# ----------------------------------------------------------------------
# What the device said
# ----------------------------------------------------------------------

@pytest.mark.parametrize("answer,because", [
    (FakeRequest(done=False), "did not finish"),
    (FakeRequest(error=OSError("the dongle went")), "failed"),
    (FakeRequest(superseded=True), "replaced"),
    (FakeRequest(cancelled=True), "cancelled"),
])
def test_a_tune_that_did_not_happen_stops_the_sweep(answer, because):
    """Carrying on would label the next block with a frequency the
    receiver is not on.

    _a_fresh_block reads the generation as it finds it, so a hop
    whose write never landed measures the last frequency and files
    it under this one.
    """
    controller = FakeController(tune_answers=[answer])

    with pytest.raises(ScanFailed) as complaint:
        BandScan(controller).run(listen_sec=0.0)

    assert because in str(complaint.value)


def test_a_gain_that_would_not_hold_stops_the_sweep():
    """Or the hops come back in units that cannot be compared."""
    controller = FakeController(
        auto_gain=True, agc_answers=[FakeRequest(error=OSError("no"))])

    with pytest.raises(ScanFailed):
        BandScan(controller).run(listen_sec=0.0)


def test_a_receiver_that_could_not_be_put_back_is_not_a_finished_sweep():
    """The person who started it was listening to something.

    81.3 MHz is not one of the hops, so the only time the sweep
    asks for it is at the end.
    """
    controller = FakeController(auto_gain=False, at_hz=81.3e6)
    controller.will_not_tune_to = 81.3e6

    with pytest.raises(ScanFailed) as complaint:
        BandScan(controller).run(listen_sec=0.0)

    assert "81.3" in str(complaint.value)


def test_both_halves_of_putting_it_back_are_tried():
    """A tuner that will not move is no reason to leave the gain pinned."""
    controller = FakeController(auto_gain=True, at_hz=81.3e6)
    controller.will_not_tune_to = 81.3e6

    with pytest.raises(ScanFailed):
        BandScan(controller).run(listen_sec=0.0)

    assert controller.agc_calls[-1] is True, "the gain was left held"


def test_the_sweeps_own_failure_is_the_one_that_is_raised(monkeypatch):
    """A failure to tidy up is a second line on the same story."""
    controller = FakeController(auto_gain=True, at_hz=81.3e6)
    controller.will_not_tune_to = 81.3e6
    scan = BandScan(controller)

    def explode(*args, **kwargs):
        raise RuntimeError("the sweep broke")

    monkeypatch.setattr(scan, "_look_at_the_band", explode)

    with pytest.raises(RuntimeError, match="the sweep broke"):
        scan.run(listen_sec=0.0)


# ----------------------------------------------------------------------
# Which tuning a block belongs to
# ----------------------------------------------------------------------

def test_a_block_is_of_the_tuning_the_sweep_asked_for():
    """Not of whatever the tuner happens to be on when it arrives.

    The window can tune too.  If it does between the sweep's own
    tune finishing and the block turning up, reading the generation
    then takes the new station's samples for this hop - and nothing
    complains, because the sweep's request finished long before it
    was overtaken.
    """
    controller = FakeController()
    scan = BandScan(controller)
    sdr = controller.sdr_receiver
    scan._blocks = sdr.watch_the_blocks()
    ours = scan._tune_and_settle(78.0e6)

    sdr.moved_to(90.5e6)            # somebody else, after ours landed

    with pytest.raises(ScanFailed, match="90.5"):
        scan._a_fresh_block(ours, timeout_sec=0.2)


def test_a_tune_that_landed_somewhere_else_is_a_failure():
    """The request says it worked; the tuner says otherwise."""
    controller = FakeController()
    scan = BandScan(controller)

    def sideways(freq_hz):
        controller.tuned_to.append(freq_hz)
        controller.sdr_receiver.moved_to(freq_hz + 1e6)
        return FakeRequest()

    controller.tune = sideways

    with pytest.raises(ScanFailed, match="the tuner is on"):
        scan._tune_and_settle(78.0e6)


def test_a_block_of_the_right_tuning_is_taken():
    """The other half: nothing moved, so the block is this hop's."""
    controller = FakeController()
    scan = BandScan(controller)
    scan._blocks = controller.sdr_receiver.watch_the_blocks()
    ours = scan._tune_and_settle(78.0e6)

    assert scan._a_fresh_block(ours, timeout_sec=0.5) is not None


# ----------------------------------------------------------------------
# Whose blocks are whose
# ----------------------------------------------------------------------

def test_the_sweep_watches_the_blocks_rather_than_taking_them():
    """data_queue is a work queue: a block taken from it is one the
    demodulator never sees, and its filters carry straight across
    the gap.
    """
    controller = FakeController()

    BandScan(controller).run(listen_sec=0.0)

    assert controller.sdr_receiver.watched == 1
    assert controller.sdr_receiver.data_queue.reads == 0, (
        "the sweep read the demodulator's queue %d times"
        % controller.sdr_receiver.data_queue.reads)


def test_the_sweep_stops_watching_when_it_is_done():
    controller = FakeController()

    BandScan(controller).run(listen_sec=0.0)

    assert controller.sdr_receiver.stopped_watching == 1
    assert controller.sdr_receiver._tap is None


def test_the_sweep_stops_watching_even_when_it_fails(monkeypatch):
    controller = FakeController()
    scan = BandScan(controller)
    monkeypatch.setattr(scan, "_look_at_the_band",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))

    with pytest.raises(RuntimeError):
        scan.run(listen_sec=0.0)

    assert controller.sdr_receiver._tap is None


# ----------------------------------------------------------------------
# How sure we are that a peak is a broadcast
# ----------------------------------------------------------------------

def test_a_second_watcher_is_an_error_rather_than_a_takeover():
    """The one already there would just stop being fed.

    It would find out as a hop that timed out with nothing to say
    why, which is the kind of fault that takes an afternoon.
    """
    sdr = FakeSDR()
    sdr.watch_the_blocks()

    with pytest.raises(RuntimeError, match="already watching"):
        sdr.watch_the_blocks()


def test_a_queue_that_goes_wrong_is_not_a_hop_that_timed_out():
    """Only an empty queue is a reason to go round again.

    Anything else is something wrong with the queue, and turning it
    into a slow, quiet, incomplete sweep loses the reason and the
    hop together.
    """
    controller = FakeController()
    scan = BandScan(controller)
    ours = scan._tune_and_settle(78.0e6)

    class Broken:
        def get(self, timeout=None):
            raise ValueError("the tap is wrong")

    scan._blocks = Broken()

    with pytest.raises(ValueError, match="the tap is wrong"):
        scan._a_fresh_block(ours, timeout_sec=0.5)


def test_an_empty_queue_is_waited_out():
    """The one exception that is ordinary."""
    import queue as queue_module

    controller = FakeController()
    scan = BandScan(controller)
    ours = scan._tune_and_settle(78.0e6)

    class Empty:
        def get(self, timeout=None):
            raise queue_module.Empty()

    scan._blocks = Empty()

    assert scan._a_fresh_block(ours, timeout_sec=0.2) is None


def test_a_pilot_settles_it():
    found = classify([Signal(81.3e6, -2.0, 52.4)])

    assert found[0].sort == CONFIRMED


def test_a_skirt_of_a_confirmed_station_is_named_as_one():
    """82.1 MHz came back 25 dB under NHK-FM at 82.5 and is the same
    transmission.  The peak test only rejects skirts within 200 kHz.
    """
    found = classify([Signal(82.5e6, 0.0, 54.6), Signal(82.1e6, -25.1, 9.9)])

    assert {round(s.freq_mhz, 1): s.sort for s in found} == {
        82.5: CONFIRMED, 82.1: LIKELY_SKIRT}


def test_a_station_with_no_pilot_and_nothing_beside_it_is_not_a_skirt():
    """A mono broadcast has no pilot either, and this cannot tell.

    Saying "not stereo" of a mono station and of a skirt alike would
    let the next thing to use this treat them the same, and the next
    thing writes unknown stations to a file.
    """
    found = classify([Signal(81.3e6, -2.0, 52.4), Signal(90.5e6, -32.3, 5.3)])

    assert {round(s.freq_mhz, 1): s.sort for s in found} == {
        81.3: CONFIRMED, 90.5: UNCONFIRMED}


def test_a_skirt_with_a_pilot_of_its_own_is_still_a_skirt():
    """It has one because it is hearing the station beside it.

    Tuned to the edge of a strong signal the receiver still
    demodulates that signal: 82.1 MHz came back with 15 dB of
    pilot, which is NHK-FM's.  So a pilot cannot be what rescues a
    peak from being a skirt.
    """
    found = classify([Signal(82.5e6, 0.0, 54.6), Signal(82.1e6, -20.0, 15.0)])

    assert {round(s.freq_mhz, 1): s.sort for s in found} == {
        82.5: CONFIRMED, 82.1: LIKELY_SKIRT}


def test_a_peak_beside_a_confirmed_one_but_nearly_as_loud_is_not_a_skirt():
    """Two real stations can be neighbours; a skirt is well below."""
    found = classify([Signal(82.5e6, 0.0, 54.6), Signal(82.1e6, -2.0, 4.0)])

    assert [s.sort for s in found if round(s.freq_mhz, 1) == 82.1] == [
        UNCONFIRMED]


def test_nothing_confirmed_means_nothing_is_a_skirt():
    found = classify([Signal(82.5e6, 0.0, 4.0), Signal(82.1e6, -25.0, 3.0)])

    assert {s.sort for s in found} == {UNCONFIRMED}


# ----------------------------------------------------------------------
# Listening for longer
# ----------------------------------------------------------------------

def added_up(blocks) -> float:
    """The reading a listen of several blocks amounts to."""
    pilot = noise = 0.0
    for block in blocks:
        p, n = pilot_and_noise_power(block)
        pilot += p
        noise += n
    return over_noise_db(pilot, noise)


def test_listening_longer_does_not_walk_the_reading_up_on_its_own():
    """Why the powers are added up rather than the best one taken.

    Not that adding up reads higher - on the same noise it reads
    lower, and that is the point: the best of several readings is
    the luckiest one, and the more blocks are listened to the
    luckier the luckiest gets.  A quarter of a second of nothing was
    reading further above the noise than a sixteenth of a second of
    nothing, which is a scan that finds more stations the longer it
    listens to an empty channel.

    Adding the powers up is the same measurement however long it
    runs.
    """
    blocks = [composite(0.0, noise=0.05, seed=n) for n in range(40)]

    best_drift = abs(max(pilot_over_noise_db(b) for b in blocks)
                     - max(pilot_over_noise_db(b) for b in blocks[:4]))
    added_drift = abs(added_up(blocks) - added_up(blocks[:4]))

    assert added_drift < best_drift, (
        "adding up moved %.2f dB with the listening, the best of them "
        "%.2f dB" % (added_drift, best_drift))


def test_adding_the_powers_up_does_not_manufacture_a_pilot():
    """Forty blocks of noise are still not a station."""
    blocks = [composite(0.0, noise=0.05, seed=n) for n in range(40)]

    assert added_up(blocks) < PILOT_OVER_NOISE_DB


def test_a_pilot_that_is_really_there_survives_being_added_up():
    heard = added_up([composite(0.1, noise=0.05, seed=n) for n in range(15)])

    assert heard > PILOT_OVER_NOISE_DB


def a_listen(scan, blocks, listen_sec: float = 5.0):
    """Run the listening loop over a fixed set of blocks.

    The sweep tests hand it listen_sec=0.0, which skips this loop
    altogether - so nothing there says how the readings are put
    together.  This hands it blocks that are already composite and
    lets it do the rest.
    """
    handed = iter(blocks)
    scan._a_fresh_block = lambda timeout_sec=1.0: next(handed, None)
    scan._demodulator.process_iq_samples = lambda block: block
    return scan._listen_for_a_pilot(Signal(81.3e6, -2.0), listen_sec)


def test_the_listen_adds_the_powers_up_rather_than_taking_the_best():
    """The aggregation, pinned to the number it should come to."""
    blocks = [composite(0.0, noise=0.05, seed=n) for n in range(12)]
    scan = BandScan(FakeController())

    heard = a_listen(scan, blocks)

    assert heard.pilot_over_noise_db == pytest.approx(
        added_up(blocks[SETTLING_BLOCKS:]), abs=0.01)
    assert heard.pilot_over_noise_db != pytest.approx(
        max(pilot_over_noise_db(b) for b in blocks), abs=0.01)


def test_the_first_blocks_after_tuning_are_thrown_away():
    """The first one out of a demodulator reset carries no audio at
    all, and the ones after it are its filters filling.
    """
    quiet = [composite(0.0, noise=0.05, seed=n) for n in range(2)]
    loud = [composite(0.2, noise=0.05, seed=n) for n in range(10)]
    scan = BandScan(FakeController())

    heard = a_listen(scan, quiet + loud)

    assert heard.pilot_over_noise_db == pytest.approx(
        added_up(loud), abs=0.01), "the settling blocks were counted"


def test_a_listen_that_gets_no_blocks_says_nothing():
    scan = BandScan(FakeController())

    heard = a_listen(scan, [])

    assert heard.pilot_over_noise_db is None


def test_the_best_of_several_blocks_of_noise_rises_with_the_listening():
    """The defect in taking the maximum, stated on its own.

    More tries at pure noise finds a higher noise, so listening
    longer made a spur more likely to pass and a weak station no
    more likely to.
    """
    blocks = [composite(0.0, noise=0.05, seed=n) for n in range(40)]

    few = max(pilot_over_noise_db(b) for b in blocks[:4])
    many = max(pilot_over_noise_db(b) for b in blocks)

    assert many > few


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
