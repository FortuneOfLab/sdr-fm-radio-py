"""The audio cushion, as the log reports it.

Diagnostic, but a diagnostic that reads wrong sends the next
investigation the wrong way, so the counted level is checked against
the audio that is actually waiting - the queue's own contents plus
what the callback holds - through every path that moves audio in or
out: plain playback, a retune's refill, a hold, and a full queue.
"""

from __future__ import annotations

import logging
import queue

import numpy as np
import pytest

from fm_radio.constants import AUDIO_CHANNELS, AUDIO_FRAMES_PER_BUFFER

FRAMES = AUDIO_FRAMES_PER_BUFFER


def _block(frames: int) -> np.ndarray:
    return np.full(frames, 0.25, dtype=np.float32)


def _really_waiting(audio_output) -> int:
    """Frames waiting, counted from the audio itself, not the counters."""
    with audio_output.audio_buffer_queue.mutex:
        queued = sum(left.size for left, _ in audio_output.audio_buffer_queue.queue)
    return queued + audio_output._buffer_len // AUDIO_CHANNELS


def _call(audio_output) -> int:
    """One callback; returns what it noted as on hand at its entry."""
    expected = _really_waiting(audio_output)
    audio_output.callback(None, FRAMES, {}, 0)
    assert audio_output._level_last == expected, (
        "noted %d on hand, %d were waiting" % (audio_output._level_last,
                                               expected))
    return expected


def test_the_level_is_what_is_queued_plus_what_the_callback_holds(
        audio_output):
    audio_output.enqueue_audio(_block(4096), _block(4096))
    assert _call(audio_output) == 4096
    assert _call(audio_output) == 3072
    audio_output.enqueue_audio(_block(768), _block(768))
    assert _call(audio_output) == 2048 + 768
    for _ in range(4):
        _call(audio_output)                 # down through an underrun
    assert audio_output._level_last == 0


def test_the_level_holds_through_a_refill(audio_output):
    """The refill pulls from the queue by a path of its own."""
    while not audio_output._playing:
        audio_output.enqueue_audio(_block(4096), _block(4096))
    while _really_waiting(audio_output):
        _call(audio_output)
    audio_output.rebuild_cushion()
    _call(audio_output)
    assert audio_output._refilling
    for size in (768, 512, 4096, 4096):
        audio_output.enqueue_audio(_block(size), _block(size))
        _call(audio_output)
    for _ in range(6):
        _call(audio_output)


def test_the_level_holds_through_a_hold(audio_output):
    """What a hold throws away is no longer waiting."""
    while not audio_output._playing:
        audio_output.enqueue_audio(_block(4096), _block(4096))
    _call(audio_output)
    audio_output.hold()
    assert _really_waiting(audio_output) == 0
    audio_output.resume()
    while not audio_output._playing:
        audio_output.enqueue_audio(_block(4096), _block(4096))
    _call(audio_output)
    _call(audio_output)


def test_a_block_the_full_queue_turned_away_is_not_counted(audio_output,
                                                           monkeypatch):
    audio_output.enqueue_audio(_block(1000), _block(1000))

    def full(*args, **kwargs):
        raise queue.Full

    monkeypatch.setattr(audio_output.audio_buffer_queue, "put", full)
    audio_output.enqueue_audio(_block(4096), _block(4096))
    assert audio_output.dropped_blocks == 1
    monkeypatch.undo()
    assert _call(audio_output) == 1000


def test_the_stats_are_per_window(audio_output):
    assert audio_output.take_cushion_stats() is None, (
        "no callback, no cushion to report")
    audio_output.enqueue_audio(_block(4096), _block(4096))
    for _ in range(3):
        _call(audio_output)                 # 4096, 3072, 2048
    least, mean, last = audio_output.take_cushion_stats()
    assert (least, last) == (2048, 2048)
    assert mean == pytest.approx((4096 + 3072 + 2048) / 3)

    _call(audio_output)                     # 1024
    least, mean, last = audio_output.take_cushion_stats()
    assert (least, mean, last) == (1024, 1024, 1024), (
        "the second window must not carry the first one's readings")
    assert audio_output.take_cushion_stats() is None


def test_the_window_minimum_starts_again_rather_than_sticking_low(
        audio_output):
    audio_output.callback(None, FRAMES, {}, 0)          # 0 on hand
    audio_output.take_cushion_stats()
    audio_output.enqueue_audio(_block(4096), _block(4096))
    _call(audio_output)
    least, _, _ = audio_output.take_cushion_stats()
    assert least == 4096


def test_an_underrun_line_says_how_much_was_on_hand(audio_output, caplog):
    audio_output.enqueue_audio(_block(300), _block(300))
    with caplog.at_level(logging.INFO, logger="fm_receiver.AudioOutput"):
        audio_output.callback(None, FRAMES, {}, 0)
    (line,) = [r.getMessage() for r in caplog.records
               if "underrun" in r.getMessage()]
    assert f"{FRAMES - 300} of {FRAMES} frames short" in line
    assert "300 frames on hand" in line


# ----------------------------------------------------------------------
# The summary line
# ----------------------------------------------------------------------

def _profiler(cushion, clock):
    from fm_radio.controller import _BlockProfiler
    return _BlockProfiler(logging.getLogger("test.cushion"), 80,
                          clock=lambda: clock[0], cushion=cushion,
                          output_rate=48000)


def _a_minute_of_blocks(profiler, clock):
    from fm_radio.controller import _PROFILE_SUMMARY_INTERVAL_SEC
    for _ in range(3):
        clock[0] += _PROFILE_SUMMARY_INTERVAL_SEC / 2
        profiler.record(0.008, q_depth=0)


def test_the_minute_summary_carries_the_cushion(caplog):
    clock = [100.0]
    profiler = _profiler(lambda: (480, 1200.0, 960), clock)
    with caplog.at_level(logging.INFO, logger="test.cushion"):
        _a_minute_of_blocks(profiler, clock)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("AudioCushion")]
    assert lines, "no cushion line in the summary"
    assert "min=480 avg=1200 last=960 frames (10.0/25.0/20.0 ms)" in lines[0]


def test_the_cushion_line_carries_the_rate_correction(caplog):
    from fm_radio.controller import _BlockProfiler
    clock = [100.0]
    profiler = _BlockProfiler(logging.getLogger("test.cushion"), 80,
                              clock=lambda: clock[0],
                              cushion=lambda: (480, 1200.0, 960),
                              output_rate=48000,
                              rate_correction=lambda: (523.4, 480.2, 26880))
    with caplog.at_level(logging.INFO, logger="test.cushion"):
        _a_minute_of_blocks(profiler, clock)
    (line,) = [r.getMessage() for r in caplog.records
               if r.getMessage().startswith("AudioCushion")]
    assert line.endswith("rate +523 ppm (learned +480) trimmed 26880 frames")


@pytest.mark.parametrize("cushion", [None, lambda: None],
                         ids=["no output", "nothing played"])
def test_no_cushion_line_without_a_cushion(caplog, cushion):
    clock = [100.0]
    profiler = _profiler(cushion, clock)
    with caplog.at_level(logging.INFO, logger="test.cushion"):
        _a_minute_of_blocks(profiler, clock)
    assert not [r for r in caplog.records
                if r.getMessage().startswith("AudioCushion")]


def test_the_receiver_hands_its_output_to_the_profiler(no_user_config,
                                                       monkeypatch):
    from fm_radio import controller as controller_mod
    seen = {}
    real = controller_mod._BlockProfiler

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(controller_mod, "_BlockProfiler", capture)
    receiver = controller_mod.FMReceiverController(
        light=True, stations_path=str(no_user_config))
    try:
        receiver.quit_event.set()
        receiver.processing_thread()
        assert seen["cushion"] == receiver.audio_output.take_cushion_stats
        assert seen["output_rate"] == receiver.audio_output.output_rate
        rates = seen["rates"]
        # Marks of our own, so that None on both sides - which is
        # what an unwired reader and a receiver with no callbacks yet
        # would both say - cannot pass for the real thing.
        receiver.sdr_receiver._arrival = sdr_mark = (16384, 1.5)
        receiver.audio_output._card_mark = card_mark = (1, 1024, 2.5)
        assert rates._sdr_mark() is sdr_mark
        assert rates._card_mark() is card_mark
        assert rates._sdr_nominal == receiver.sdr_receiver.sample_rate
        assert rates._card_nominal == receiver.audio_output.output_rate
        receiver.audio_output._matcher.ppm = 321.0
        assert seen["rate_correction"]()[0] == 321.0
    finally:
        receiver.auto_gain.stop()
        receiver.audio_output.cleanup()


# ----------------------------------------------------------------------
# The two rates
# ----------------------------------------------------------------------

def test_the_sdr_marks_every_sample_that_arrives(sdr_receiver, monkeypatch):
    """Counted even when the queue turns the block away: it arrived."""
    import fm_radio.sdr_receiver as sdr_mod
    monkeypatch.setattr(sdr_mod.time, "perf_counter", lambda: 12.5)
    assert sdr_receiver.arrival_mark is None
    block = np.zeros(16384, dtype=np.complex64)
    sdr_receiver.callback(block, None)
    assert sdr_receiver.arrival_mark == (16384, 12.5)
    while not sdr_receiver.data_queue.full():
        sdr_receiver.data_queue.put((0, block))
    sdr_receiver.callback(block, None)
    assert sdr_receiver.dropped_blocks == 1
    assert sdr_receiver.arrival_mark == (2 * 16384, 12.5)


def test_the_card_count_starts_again_with_the_stream(audio_output):
    audio_output.callback(None, FRAMES, {}, 0)
    audio_output.callback(None, FRAMES, {}, 0)
    start, frames, _ = audio_output.card_mark
    assert frames == 2 * FRAMES
    while not audio_output._playing:
        audio_output.enqueue_audio(_block(4096), _block(4096))
    audio_output.callback(None, FRAMES, {}, 0)
    again, frames, _ = audio_output.card_mark
    assert again == start + 1
    assert frames == FRAMES, "the count before the start is not this stream's"


class _Clocks:
    """Both callbacks on a clock of our own, each on its own schedule.

    The schedule carries over from one run to the next, so splitting a
    run in two does not squeeze an extra block in at the join.
    """

    def __init__(self, sdr_receiver, audio_output, monkeypatch,
                 sdr_ppm, card_ppm):
        import fm_radio.audio_output as ao_mod
        import fm_radio.sdr_receiver as sdr_mod
        self.now = 0.0
        monkeypatch.setattr(sdr_mod.time, "perf_counter", lambda: self.now)
        monkeypatch.setattr(ao_mod.time, "perf_counter", lambda: self.now)
        self.sdr, self.card = sdr_receiver, audio_output
        self.sdr_every = 16384 / (1.024e6 * (1 + sdr_ppm * 1e-6))
        self.card_every = FRAMES / (48000.0 * (1 + card_ppm * 1e-6))
        self.t_sdr = self.t_card = 0.0
        self.block = np.zeros(16384, dtype=np.complex64)

    def run_until(self, end):
        while min(self.t_sdr, self.t_card) < end:
            if self.t_sdr <= self.t_card:
                self.now = self.t_sdr
                self.sdr.callback(self.block, None)
                self.sdr.data_queue.queue.clear()
                self.t_sdr += self.sdr_every
            else:
                self.now = self.t_card
                self.card.callback(None, FRAMES, {}, 0)
                self.t_card += self.card_every


def _watch(sdr_receiver, audio_output):
    from fm_radio.controller import RateWatch
    return RateWatch(lambda: sdr_receiver.arrival_mark,
                     lambda: audio_output.card_mark, 1.024e6, 48000.0)


@pytest.mark.parametrize("sdr_ppm, card_ppm", [(-300, 0), (0, 0), (40, -25)])
def test_the_rates_come_out_in_ppm(sdr_receiver, audio_output, monkeypatch,
                                   sdr_ppm, card_ppm):
    watch = _watch(sdr_receiver, audio_output)
    clocks = _Clocks(sdr_receiver, audio_output, monkeypatch, sdr_ppm,
                     card_ppm)
    clocks.run_until(5.0)
    assert watch.line(5.0) is None, "the first call only takes the marks"
    clocks.run_until(125.0)
    sdr_hz, card_hz, span = watch.measure()
    assert watch.ppm(sdr_hz, 1.024e6) == pytest.approx(sdr_ppm, abs=1.0)
    assert watch.ppm(card_hz, 48000.0) == pytest.approx(card_ppm, abs=1.0)
    assert span == pytest.approx(120.0, abs=0.1)


def test_the_line_says_what_the_gap_does_to_the_cushion(sdr_receiver,
                                                        audio_output,
                                                        monkeypatch):
    watch = _watch(sdr_receiver, audio_output)
    clocks = _Clocks(sdr_receiver, audio_output, monkeypatch, -500, 0)
    clocks.run_until(5.0)
    watch.line(5.0)
    clocks.run_until(125.0)
    line = watch.line(125.0)
    assert "(-500 ppm)" in line and "(+0 ppm)" in line
    # 500 ppm of 48 kHz for a minute is 1440 frames the SDR does not
    # pay for: the rate the log showed the cushion falling at.
    assert "sdr-card=-500 ppm (cushion -1440 frames/min)" in line


def test_a_restarted_card_is_measured_afresh(sdr_receiver, audio_output,
                                             monkeypatch):
    """The time the card spent stopped is not time it was playing."""
    watch = _watch(sdr_receiver, audio_output)
    clocks = _Clocks(sdr_receiver, audio_output, monkeypatch, 0, 0)
    clocks.run_until(5.0)
    watch.measure()
    clocks.run_until(65.0)
    audio_output._starts += 1               # what a hold and restart do
    audio_output._card_frames = 0
    clocks.t_card += 30.0                   # thirty seconds stopped
    clocks.run_until(96.0)
    _, card_hz, _ = watch.measure()
    assert card_hz is None, "a new stream is a new reference"
    clocks.run_until(156.0)
    _, card_hz, _ = watch.measure()
    assert watch.ppm(card_hz, 48000.0) == pytest.approx(0.0, abs=1.0)


def test_the_summary_carries_the_rates(caplog):
    from fm_radio.controller import _BlockProfiler

    class Said:
        def line(self, elapsed):
            return f"Rates: t={elapsed:.0f}s stand-in"

    clock = [100.0]
    profiler = _BlockProfiler(logging.getLogger("test.cushion"), 80,
                              clock=lambda: clock[0], rates=Said())
    with caplog.at_level(logging.INFO, logger="test.cushion"):
        _a_minute_of_blocks(profiler, clock)
    assert [r for r in caplog.records
            if r.getMessage().startswith("Rates: ")]
