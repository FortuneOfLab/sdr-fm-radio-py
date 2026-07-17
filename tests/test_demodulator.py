"""Demodulator DSP correctness tests.

Covers block-continuity of the IQ lowpass (PR #4), the analytic
heterodyne pilot path (PR #6), the discriminator main demod (PR #7),
mode-coupled subcarrier offsets, and state reset on re-tune.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sg

import fm_radio.demodulator as dm
from fm_radio.constants import COMPOSITE_RATE, SDR_BLOCK_SIZE
from fm_radio.demodulator import FMDemodulator, FMDemodulatorLight


def _random_iq(rng, n):
    return (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64) * 0.3


def test_iq_lowpass_blockwise_matches_oneshot(rng):
    demod = FMDemodulator(stereo=True)
    n = SDR_BLOCK_SIZE * 4
    x = _random_iq(rng, n)
    zi = np.zeros_like(demod._iq_zi)
    y_blocks = []
    for i in range(0, n, SDR_BLOCK_SIZE):
        yb, zi = sg.sosfilt(demod.iq_sos, x[i:i + SDR_BLOCK_SIZE], zi=zi)
        y_blocks.append(yb)
    y_stream = np.concatenate(y_blocks)
    y_ref, _ = sg.sosfilt(demod.iq_sos, x, zi=np.zeros_like(demod._iq_zi))
    assert np.max(np.abs(y_stream - y_ref)) < 1e-9


def _fm_iq(n):
    """Deterministic FM-modulated IQ with realistic bounded deviation.

    Random complex noise is unsuitable for discriminator equivalence
    tests: its instantaneous frequency rides the +-pi branch cut, where
    1e-9 numeric differences flip into 2*pi jumps.  A real FM signal
    keeps |freq| well inside (-pi, pi).
    """
    fs = 1.024e6
    t = np.arange(n) / fs
    mpx = (
        0.30 * np.sin(2 * np.pi * 3_000.0 * t)
        + 0.15 * np.sin(2 * np.pi * 38_000.0 * t)
        + 0.05 * np.sin(2 * np.pi * 53_000.0 * t)
    )
    phase = np.cumsum(mpx)  # amplitudes are directly rad/sample
    return np.exp(1j * phase).astype(np.complex64)


def test_composite_is_block_size_invariant():
    """The IQ->composite chain must be stateful end to end.

    IQ lowpass state, discriminator carry-over and the resampler's
    held-back emission window together must make block-wise processing
    match one-shot processing sample-for-sample.
    """
    n = SDR_BLOCK_SIZE * 4
    x = _fm_iq(n)

    # Disable the DC-offset EMA for the comparison: it updates once per
    # process_iq_samples call by design, so one-shot and block-wise runs
    # legitimately compute different DC corrections.
    d_one = FMDemodulator(stereo=True)
    d_one.dc_alpha = 0.0
    comp_one = d_one.process_iq_samples(x)

    d_blk = FMDemodulator(stereo=True)
    d_blk.dc_alpha = 0.0
    comp_blk = np.concatenate([
        d_blk.process_iq_samples(x[i:i + SDR_BLOCK_SIZE])
        for i in range(0, n, SDR_BLOCK_SIZE)
    ])

    assert comp_blk.size == comp_one.size
    assert np.allclose(comp_blk, comp_one, atol=1e-5)


def test_discriminator_is_default_and_pll_selectable(monkeypatch):
    d = FMDemodulator(stereo=True)
    assert d.use_pll_demod is False
    assert abs(np.rad2deg(d.subcarrier_phase_offset_rad) - 316.0) < 0.01

    monkeypatch.setattr(dm, "MAIN_DEMOD_USE_PLL", True)
    d_pll = FMDemodulator(stereo=True)
    assert d_pll.use_pll_demod is True
    assert abs(np.rad2deg(d_pll.subcarrier_phase_offset_rad) - 285.0) < 0.01


def test_light_demodulator_keeps_its_operating_point():
    d = FMDemodulatorLight(stereo=True)
    assert abs(np.rad2deg(d.subcarrier_phase_offset_rad) - 297.4) < 0.01


def test_pll_mode_produces_finite_composite(rng, monkeypatch):
    monkeypatch.setattr(dm, "MAIN_DEMOD_USE_PLL", True)
    d = FMDemodulator(stereo=True)
    comp = d.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
    assert np.all(np.isfinite(comp))


def test_pilot_heterodyne_tracks_offset_pilot_exactly():
    """Noise-free pilot with carrier offset: phase error must be ~zero.

    The pre-PR#6 FFT-Hilbert path had up to ~12 deg block-edge error in
    this exact scenario.
    """
    fs = COMPOSITE_RATE
    n_block = 3072
    n_blocks = 40
    n = n_block * n_blocks
    f_pilot = 19003.7
    t = np.arange(n) / fs
    true_phase = 2 * np.pi * f_pilot * t
    comp = (0.1 * np.cos(true_phase)).astype(np.float32)

    demod = FMDemodulator(stereo=True)
    est = []
    for i in range(0, n, n_block):
        phase, _resid = demod._estimate_pilot_phase(comp[i:i + n_block])
        est.append(phase)
    est = np.concatenate(est)

    settle = n_block * 20  # let the pilot PLL settle
    err = np.angle(np.exp(1j * (est[settle:] - true_phase[settle:])))
    err = err - np.median(err)
    assert np.rad2deg(np.max(np.abs(err))) < 0.05


def test_pilot_power_scaling_matches_real_bandpass_convention():
    """2*mean(|residual|^2) must equal A^2/2 for a pilot of amplitude A."""
    fs = COMPOSITE_RATE
    n_block = 3072
    amp = 0.1
    t = np.arange(n_block * 30) / fs
    comp = (amp * np.cos(2 * np.pi * 19000.0 * t)).astype(np.float32)

    demod = FMDemodulator(stereo=True)
    powers = []
    for i in range(0, comp.size, n_block):
        _phase, resid = demod._estimate_pilot_phase(comp[i:i + n_block])
        powers.append(2.0 * float(np.mean(np.abs(resid) ** 2)))
    measured = np.mean(powers[10:])
    expected = amp ** 2 / 2.0
    assert abs(measured - expected) / expected < 0.05


def test_reset_clears_all_streaming_state(rng):
    demod = FMDemodulator(stereo=True)
    # Warm every path with random IQ.
    for _ in range(5):
        comp = demod.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
        demod.demodulate(comp)
    demod.reset()
    assert np.all(demod._iq_zi == 0)
    assert np.all(demod._pilot_lp_zi == 0)
    assert demod._disc_last is None
    # Zero composite in -> zero audio out (no leakage from the warm state).
    left, right = demod.demodulate(np.zeros(3072, dtype=np.float32))
    if left.size:
        assert np.allclose(left, 0.0, atol=1e-12)
        assert np.allclose(right, 0.0, atol=1e-12)


def test_light_demodulator_end_to_end_is_finite(rng):
    d = FMDemodulatorLight(stereo=True)
    x = _random_iq(rng, 4096)
    for _ in range(5):
        comp = d.process_iq_samples(x)
        left, right = d.demodulate(comp)
    assert comp.size % 4 == 0  # emit_align invariant holds for light too
    assert np.all(np.isfinite(comp))
    assert np.all(np.isfinite(left))


def test_demodulate_returns_matched_stereo_pair(rng):
    demod = FMDemodulator(stereo=True)
    comp = demod.process_iq_samples(_random_iq(rng, SDR_BLOCK_SIZE))
    left, right = demod.demodulate(comp)
    assert left.shape == right.shape
    assert left.dtype == np.float32
    assert np.all(np.isfinite(left))
