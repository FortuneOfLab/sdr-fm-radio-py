"""Rate matching: the resampler's accuracy, and the loop that holds the cushion.

The resampler is held to an ideal resampled tone rather than to its
own idea of itself.  The loop is held to the thing it is for: a
receiver whose samples arrive slower than the card plays them - as
measured on the hardware, 270 to 500 ppm short - plays for many
simulated minutes without an underrun, where the same receiver without
it runs dry.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fm_radio.constants import AUDIO_FRAMES_PER_BUFFER
from fm_radio.rate_match import FractionalResampler, RateMatcher

FS = 48000.0


# ----------------------------------------------------------------------
# The resampler
# ----------------------------------------------------------------------

def _through(resampler, x, ratio, block=768):
    out = []
    for i in range(0, x.size, block):
        left, right = resampler.process(x[i:i + block], x[i:i + block], ratio)
        np.testing.assert_array_equal(left, right)
        out.append(left)
    return np.concatenate(out)


def _against_an_ideal_tone(freq, ppm, seconds=2.0):
    """(gain dB, residual dB) of a tone resampled by *ppm*."""
    ratio = 1.0 + ppm * 1e-6
    n = int(seconds * FS)
    x = (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / FS)).astype(np.float32)
    y = _through(FractionalResampler(), x, ratio).astype(np.float64)
    y = y[2000:-200]                            # past the start-up
    k = np.arange(y.size) + 2000
    w = 2 * np.pi * freq / FS / ratio           # the tone, played back
    basis = np.column_stack([np.sin(w * k), np.cos(w * k)])
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    amplitude = float(np.hypot(*coeffs))
    residual = np.sqrt(np.mean((y - basis @ coeffs) ** 2))
    return (20 * np.log10(amplitude / 0.5),
            20 * np.log10(residual / (amplitude / np.sqrt(2))))


@pytest.mark.parametrize("ppm", [-2000, -600, 600, 2000])
@pytest.mark.parametrize("freq", [100, 1000, 5000, 10000, 15000])
def test_a_tone_comes_out_at_its_level_and_nothing_else(freq, ppm):
    """Measured -89 dB or better and 0.0005 dB, across the audio band."""
    gain_db, residual_db = _against_an_ideal_tone(freq, ppm)
    assert gain_db == pytest.approx(0.0, abs=0.005)
    assert residual_db < -85.0


@pytest.mark.parametrize("ppm", [-1500, 0, 437, 2000])
def test_the_frames_out_are_the_frames_in_times_the_ratio(ppm):
    ratio = 1.0 + ppm * 1e-6
    rng = np.random.default_rng(3)
    x = rng.standard_normal(768 * 1500).astype(np.float32)
    y = _through(FractionalResampler(), x, ratio)
    assert abs(y.size - x.size * ratio) <= 1.0


def test_how_the_blocks_fall_does_not_change_the_audio():
    """The joins carry their state: split anywhere, the same output."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(30000).astype(np.float32)
    whole = _through(FractionalResampler(), x, 1.0007, block=x.size)
    pieces, at, res = [], 0, FractionalResampler()
    while at < x.size:
        size = int(rng.integers(1, 2000))
        left, _ = res.process(x[at:at + size], x[at:at + size], 1.0007)
        pieces.append(left)
        at += size
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=2e-6)


def test_at_a_ratio_of_one_it_is_a_delay():
    """On audio the receiver makes - nothing above 15 kHz - it is a delay.

    Not on white noise: the kernel's cutoff is 21.6 kHz, so content
    between that and 24 kHz, which the receiver never makes, is cut.
    """
    from scipy.signal import firwin, lfilter
    noise = np.random.default_rng(5).standard_normal(12000)
    x = lfilter(firwin(255, 15000, fs=FS), 1.0, noise)[1000:].astype(np.float32)
    res = FractionalResampler()
    y = _through(res, x, 1.0)
    err = y[res.half:] - x[:x.size - res.half]
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    # Measured -79.9 dB: what is left is the polynomial's fit to the
    # kernel (7e-5) on the noise up near the 15 kHz edge.
    assert 20 * np.log10(np.sqrt(np.mean(err[100:] ** 2)) / rms) < -75.0


# ----------------------------------------------------------------------
# The loop on its own
# ----------------------------------------------------------------------

def test_too_few_waiting_asks_for_more_frames_and_too_many_for_fewer():
    low = RateMatcher(FS, target_frames=5000)
    assert low.update(2000, 768) > 1.0
    high = RateMatcher(FS, target_frames=5000)
    assert high.update(9000, 768) < 1.0


def test_the_correction_is_bounded():
    matcher = RateMatcher(FS, target_frames=5000, max_ppm=2000)
    for _ in range(100000):                     # 27 minutes of blocks
        matcher.update(0, 768)
    assert matcher.ppm == pytest.approx(2000)
    assert matcher.integral_ppm == pytest.approx(2000)


@pytest.mark.parametrize("drift_ppm", [-500, -270, 150])
def test_the_integral_learns_the_mismatch(drift_ppm):
    """A plant that drains at *drift_ppm*: the loop ends up cancelling it."""
    matcher = RateMatcher(FS, target_frames=5000)
    level = 5000.0
    for _ in range(int(900 / (768 / FS))):          # 15 minutes of blocks
        matcher.update(level, 768)
        level += 768 * (matcher.ppm + drift_ppm) * 1e-6
    assert matcher.integral_ppm == pytest.approx(-drift_ppm, abs=15)
    # The level given is the trough just before a block goes in, and
    # what is held at the target is the mean: half a block above it.
    assert level == pytest.approx(5000 - 768 / 2, abs=150)


# ----------------------------------------------------------------------
# The receiver's output, drifting
# ----------------------------------------------------------------------

def _play(audio_output, minutes, drift_ppm, block_sizes=(768,),
          card_sec=0.1067):
    """Feed blocks *drift_ppm* slow against the card's clock; play them.

    Blocks come at the rate the SDR delivers samples (the block
    interval stretched by the drift) and callbacks at the card's own
    rate.  When the stream starts, the card takes what it holds in one
    burst, as a real one does.  Returns the cushion level, in frames,
    at the entry to every callback in the last two minutes.
    """
    rate = FS
    fpb = AUDIO_FRAMES_PER_BUFFER
    callback_sec = fpb / rate
    sizes = list(block_sizes)
    t_block, t_callback, i = 0.0, None, 0
    end = minutes * 60.0
    levels = []
    while True:
        due = t_block if t_callback is None else min(t_block, t_callback)
        if due > end:
            break
        if t_callback is not None and t_callback <= t_block:
            if t_callback > end - 120.0:
                levels.append(audio_output._frames_on_hand())
            audio_output.callback(None, fpb, {}, 0)
            t_callback += callback_sec
            continue
        size = sizes[i % len(sizes)]
        i += 1
        block = np.full(size, 0.1, dtype=np.float32)
        audio_output.enqueue_audio(block, block)
        t_block += size / rate / (1.0 + drift_ppm * 1e-6)
        if t_callback is None and audio_output._playing:
            for _ in range(int(round(card_sec * rate / fpb))):
                audio_output.callback(None, fpb, {}, 0)
            t_callback = t_block
    return np.asarray(levels)


@pytest.mark.parametrize("drift_ppm", [-500, -1000, 300])
def test_a_drifting_receiver_plays_on_without_a_gap(audio_output, drift_ppm):
    levels = _play(audio_output, 12, drift_ppm)
    assert audio_output.underruns == 0
    target = audio_output._matcher.target
    assert np.mean(levels) == pytest.approx(target, abs=600)
    assert audio_output._matcher.integral_ppm == pytest.approx(
        -drift_ppm, abs=60)


def test_without_matching_the_same_receiver_runs_dry(audio_output,
                                                     monkeypatch):
    """The control: the drift in the test above is enough to matter."""
    monkeypatch.setattr(audio_output, "_match_the_rate",
                        lambda left, right: (left, right))
    _play(audio_output, 12, -500)
    assert audio_output.underruns > 0


def test_light_mode_blocks_drift_and_play_on(monkeypatch):
    """3072- and 3328-frame blocks 65 ms apart, as the light chain makes."""
    from fm_radio.audio_output import AudioOutput
    from fm_radio.constants import SDR_BLOCK_SIZE, SDR_SAMPLE_RATE_LIGHT
    out = AudioOutput(block_interval_sec=SDR_BLOCK_SIZE / SDR_SAMPLE_RATE_LIGHT)
    try:
        levels = _play(out, 12, -500, block_sizes=(3072, 3328, 3072, 3072))
        assert out.underruns == 0
        assert np.mean(levels) == pytest.approx(out._matcher.target, abs=1200)
    finally:
        out.cleanup()


def test_a_startup_burst_is_trimmed_to_the_target(audio_output, caplog):
    """A second of audio delivered at once is dropped, not bled off."""
    block = np.full(768, 0.1, dtype=np.float32)
    while not audio_output._playing:
        audio_output.enqueue_audio(block, block)
    for _ in range(5):
        audio_output.callback(None, AUDIO_FRAMES_PER_BUFFER, {}, 0)
    with caplog.at_level(logging.INFO, logger="fm_receiver.AudioOutput"):
        for _ in range(45):                     # 0.7 s more, all at once
            audio_output.enqueue_audio(block, block)
    # Dropped back to the target whenever it passed the trim line, so
    # never more than the line plus the block that crossed it.
    on_hand = audio_output._frames_on_hand()
    assert on_hand <= audio_output._trim_above + 768
    # ...and never below the target: the trim takes the excess, not
    # the cushion.
    assert on_hand >= audio_output._matcher.target - 768
    assert audio_output.rate_correction[2] >= 768 * 45 - 17100
    assert any("Audio trimmed" in r.getMessage() for r in caplog.records)


def test_ordinary_playing_trims_nothing(audio_output):
    _play(audio_output, 3, -300)
    assert audio_output.rate_correction[2] == 0


def test_a_recording_is_not_resampled(audio_output, tmp_path):
    """The recording takes the receiver's audio, on the SDR's timeline."""
    import wave
    path = tmp_path / "plain.wav"
    audio_output.start_recording(str(path))
    x = np.random.default_rng(6).uniform(-0.5, 0.5, 768 * 2).astype(np.float32)
    audio_output._matcher.integral_ppm = 1500.0     # matching hard
    audio_output.record(x)
    audio_output.stop_recording()
    with wave.open(str(path), "rb") as r:
        written = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
    np.testing.assert_array_equal(written, np.int16(x * 32767))
