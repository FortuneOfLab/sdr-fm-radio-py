"""Streaming-filter state management and reconstruction tests.

Covers the defects fixed in PR #1 (filter reset on re-tune), PR #1's
StatefulResampler continuity, and the SideNoiseReducer overlap-add
reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sg

from fm_radio.filters import (
    LowpassFilter, BandpassFilter, NotchFilter, DeemphasisIIRFilter,
    StatefulResampler, SideNoiseReducer, StreamAligner,
)


def _warm(filt_apply, rng, n_blocks=5, size=4096):
    for _ in range(n_blocks):
        filt_apply(rng.standard_normal(size).astype(np.float32))


def test_lowpass_reset_gives_zero_output_for_zero_input(rng):
    lp = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    _warm(lp.apply, rng)
    assert np.linalg.norm(lp.zi) > 0
    lp.reset()
    out = lp.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_bandpass_reset_gives_zero_output_for_zero_input(rng):
    bp = BandpassFilter(order=9, lowcut=23000, highcut=53000, sample_rate=192000)
    _warm(bp.apply, rng)
    bp.reset()
    out = bp.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_notch_reset_gives_zero_output_for_zero_input(rng):
    nf = NotchFilter(freq=19000, Q=30, sample_rate=192000)
    _warm(nf.apply, rng)
    nf.reset()
    out = nf.apply(np.zeros(1024, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_deemphasis_reset_gives_zero_output_for_zero_input():
    de = DeemphasisIIRFilter(sample_rate=48000, tau=50e-6)
    de.process(np.ones(1024, dtype=np.float64))
    assert de.prev_output != 0.0
    de.reset()
    out = de.process(np.zeros(1024, dtype=np.float64))
    assert np.allclose(out, 0.0)


def test_lowpass_streaming_matches_oneshot(rng):
    lp_stream = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    lp_ref = LowpassFilter(order=5, cutoff=15000, sample_rate=192000)
    x = rng.standard_normal(8192)
    y_stream = np.concatenate(
        [lp_stream.apply(x[i:i + 1024]) for i in range(0, x.size, 1024)]
    )
    y_ref = lp_ref.apply(x)
    assert np.allclose(y_stream, y_ref, atol=1e-10)


def test_stateful_resampler_matches_oneshot_exactly(rng):
    """Streamed output must equal one-shot resample_poly sample-for-sample.

    The resampler holds back the trailing half-filter-length outputs of
    each block until their FIR support is fully received, so the
    streamed output is a truncated prefix of the one-shot result — but
    every emitted sample must match exactly (no block-edge transients;
    the pre-fix version failed this with errors up to ~0.17 at every
    block end).
    """
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0))
    n_block = 16384
    x = rng.standard_normal(n_block * 4)
    y_stream = np.concatenate(
        [r.process(x[i:i + n_block]) for i in range(0, x.size, n_block)]
    )
    y_ref = sg.resample_poly(x, up, down, window=("kaiser", 10.0))
    # Held-back tail: half_len inputs => half_len*up/down = 30 outputs.
    assert y_ref.size - y_stream.size == 30
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_emit_align_keeps_blocks_decimatable(rng):
    """With emit_align=N every emitted block size is a multiple of N.

    The composite->audio stage decimates each block with a stateless
    resample_poly(1, 4); non-multiple-of-4 composite blocks would shift
    its per-block output grid and add a fractional-sample phase jump at
    every boundary (measured as a THD+N regression from -25.6 to
    -19.1 dB before this alignment existed).
    """
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0), emit_align=4)
    n_block = 16384
    x = rng.standard_normal(n_block * 4)
    segs = [r.process(x[i:i + n_block]) for i in range(0, x.size, n_block)]
    assert all(seg.size % 4 == 0 for seg in segs)
    # Alignment must not break exactness: still a prefix of one-shot.
    y_stream = np.concatenate(segs)
    y_ref = sg.resample_poly(x, up, down, window=("kaiser", 10.0))
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_light_ratio_is_exact(rng):
    """96/125 (light chain) with 16384-sample blocks must be exact.

    16384 is not a multiple of the 96/125 polyphase grid period (125),
    so the fixed-length-tail implementation emitted every block with a
    fractional-phase offset from the global grid - order-of-signal
    errors (max ~4.6 on unit-variance wideband input).  The variable-
    length tail keeps ext grid-aligned for arbitrary block sizes.
    """
    r = StatefulResampler(96, 125)
    x = rng.standard_normal(16384 * 4)
    y_stream = np.concatenate(
        [r.process(x[i:i + 16384]) for i in range(0, x.size, 16384)]
    )
    y_ref = sg.resample_poly(x, 96, 125)
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_arbitrary_block_sizes_are_exact(rng):
    """Mixed, grid-unaligned block sizes must still be an exact prefix."""
    r = StatefulResampler(3, 16, window=("kaiser", 10.0))
    sizes = [1000, 16384, 7, 12345, 500, 9000]
    x = rng.standard_normal(sum(sizes))
    pos, segs = 0, []
    for s in sizes:
        segs.append(r.process(x[pos:pos + s]))
        pos += s
    y_stream = np.concatenate(segs)
    y_ref = sg.resample_poly(x, 3, 16, window=("kaiser", 10.0))
    assert np.allclose(y_stream, y_ref[:y_stream.size], atol=1e-9)


def test_stateful_resampler_reset_gives_repeatable_stream(rng):
    up, down = 3, 16
    r = StatefulResampler(up, down, window=("kaiser", 10.0))
    x = rng.standard_normal(16384)
    y1 = r.process(x)
    r.reset()
    y2 = r.process(x)
    assert np.array_equal(y1, y2)


def test_side_nr_passthrough_is_exact(rng):
    # alpha_floor=1.0 clamps the DD-Wiener gain (always <= 1) to exactly 1
    # in every bin: pure analysis/synthesis reconstruction.
    nr = SideNoiseReducer(
        sample_rate=48000, frame=1024, hop=256, alpha_floor=1.0, beta=1.0,
    )
    n = 48000
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    out = []
    i = 0
    while i < n:
        step = int(rng.integers(50, 500))
        y = nr.process(x[i:i + step])
        if y.size:
            out.append(y)
        i += step
    y = np.concatenate(out)
    lat = nr.latency_samples
    m = y.size - 2 * lat
    assert m > 0
    assert np.allclose(y[lat:lat + m], x[lat:lat + m], atol=1e-5)


def test_side_nr_reduces_stationary_noise(rng):
    fs = 48000
    nr = SideNoiseReducer(
        sample_rate=fs, frame=1024, hop=256, alpha_floor=0.15, beta=1.0,
    )
    n = 20 * fs
    x = (rng.standard_normal(n) * 0.02).astype(np.float32)
    out = []
    for i in range(0, n, 480):
        y = nr.process(x[i:i + 480])
        if y.size:
            out.append(y)
    y = np.concatenate(out)
    # Compare in-band (NR acts on 1.5-15 kHz) after settling.
    sos = sg.butter(8, [1500 / (fs / 2), 15000 / (fs / 2)], btype="band",
                    output="sos")
    y_in = sg.sosfilt(sos, y[-5 * fs:])
    x_in = sg.sosfilt(sos, x[-5 * fs:])
    red_db = 20 * np.log10(
        np.sqrt(np.mean(y_in[fs:] ** 2)) / np.sqrt(np.mean(x_in[fs:] ** 2))
    )
    assert red_db < -8.0, f"in-band reduction only {red_db:.1f} dB"


def test_side_nr_beta_zero_produces_finite_output(rng):
    # --side-nr-beta 0 is reachable from the CLI; xi=0 with beta=0 used
    # to produce 0/0 = NaN gains that silenced the side channel.
    nr = SideNoiseReducer(
        sample_rate=48000, frame=1024, hop=256, alpha_floor=0.3, beta=0.0,
    )
    x = (rng.standard_normal(48000) * 0.05).astype(np.float32)
    out = []
    for i in range(0, x.size, 480):
        y = nr.process(x[i:i + 480])
        if y.size:
            out.append(y)
    y = np.concatenate(out)
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y ** 2)) > 0  # not silenced


def test_side_nr_reset_restores_initial_state(rng):
    nr = SideNoiseReducer(sample_rate=48000, frame=1024, hop=256)
    nr.process(rng.standard_normal(4096).astype(np.float32))
    assert nr.noise_floor is not None
    nr.reset()
    assert nr.noise_floor is None
    assert nr.in_buf.size == 0


def test_stream_aligner_preserves_order_and_counts():
    al = StreamAligner()
    a = np.arange(10, dtype=np.float32)
    b = np.arange(10, 20, dtype=np.float32)
    out1 = al.feed_and_take(a, 4)
    out2 = al.feed_and_take(b, 10)
    assert np.array_equal(out1, a[:4])
    assert np.array_equal(out2, np.arange(4, 14, dtype=np.float32))
