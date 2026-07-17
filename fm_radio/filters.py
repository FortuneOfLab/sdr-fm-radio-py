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
"""Filter classes for FM signal processing."""

from __future__ import annotations

import math

import numpy as np
import scipy.signal as signal
from numba import njit


# --------------------------------------------------
# Deemphasis IIR Filter Class (Numba optimized)
# --------------------------------------------------
@njit
def deemphasis_iir_process_numba(
    x: np.ndarray, alpha: float, prev_output: float,
) -> tuple[np.ndarray, float]:
    """Numba-optimized de-emphasis IIR filter processing.

    Args:
        x: Input audio signal array.
        alpha: Filter coefficient.
        prev_output: Previous output sample for filter state.

    Returns:
        Tuple of (filtered output array, last output value).
    """
    y = np.empty_like(x)
    prev = prev_output
    for i in range(x.shape[0]):
        y[i] = (1.0 - alpha) * x[i] + alpha * prev
        prev = y[i]
    return y, prev


class DeemphasisIIRFilter:
    """IIR filter for de-emphasis processing in FM broadcasting.

    Args:
        sample_rate: Audio signal sample rate.
        tau: Time constant (e.g., 50e-6 seconds).
    """

    def __init__(self, sample_rate: float, tau: float) -> None:
        self.sample_rate: float = sample_rate
        self.tau: float = tau
        self.alpha: float = math.exp(-1.0 / (sample_rate * tau))
        self.prev_output: float = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply de-emphasis processing to the input signal.

        Args:
            x: Input audio signal array.

        Returns:
            Audio signal after filtering.
        """
        y, self.prev_output = deemphasis_iir_process_numba(x, self.alpha, self.prev_output)
        return y

    def reset(self) -> None:
        """Clear the filter's running state.

        Called by ``BaseFMDemodulator.reset()`` so a re-tune does not
        leak the previous station's audio into the new one.
        """
        self.prev_output = 0.0


# --------------------------------------------------
# Filter Classes (LowpassFilter & BandpassFilter)
# --------------------------------------------------
class LowpassFilter:
    """Butterworth lowpass filter (streaming using lfilter with state)."""

    def __init__(self, order: int, cutoff: float, sample_rate: float) -> None:
        nyquist = sample_rate / 2.0
        self.sos: np.ndarray = signal.butter(
            order, cutoff / nyquist, btype="low", analog=False, output="sos",
        )
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y

    def reset(self) -> None:
        """Clear the filter's running state."""
        self.zi = signal.sosfilt_zi(self.sos) * 0.0


class BandpassFilter:
    """Butterworth bandpass filter (streaming using lfilter with state)."""

    def __init__(self, order: int, lowcut: float, highcut: float, sample_rate: float) -> None:
        nyquist = sample_rate / 2.0
        self.sos: np.ndarray = signal.butter(
            order, [lowcut / nyquist, highcut / nyquist], btype="band",
            analog=False, output="sos",
        )
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the bandpass filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y

    def reset(self) -> None:
        """Clear the filter's running state."""
        self.zi = signal.sosfilt_zi(self.sos) * 0.0


class NotchFilter:
    """IIR notch (band-reject) filter (streaming using lfilter with state)."""

    def __init__(self, freq: float, Q: float, sample_rate: float) -> None:
        b, a = signal.iirnotch(freq, Q, sample_rate)
        self.sos: np.ndarray = signal.tf2sos(b, a)
        self.zi: np.ndarray = signal.sosfilt_zi(self.sos) * 0.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the notch filter to streaming chunk, preserving state."""
        y, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return y

    def reset(self) -> None:
        """Clear the filter's running state."""
        self.zi = signal.sosfilt_zi(self.sos) * 0.0


# --------------------------------------------------
# Stateful polyphase resampler (overlap-save)
# --------------------------------------------------
class StatefulResampler:
    """Overlap-save wrapper for ``scipy.signal.resample_poly``.

    ``resample_poly`` is stateless: each call zero-pads the block edges,
    producing boundary transients that manifest as impulse noise in
    downstream stereo L-R processing.  This class keeps the tail of the
    previous input block and prepends it to the current block so that
    the polyphase filter sees a continuous stream.

    Emission policy: an output sample is only released once its entire
    FIR support lies within samples actually received, i.e. the last
    ``half_len`` input samples' worth of outputs are held back until the
    next block arrives.  (An earlier revision emitted them immediately,
    computed against a zero-padded future, and never re-emitted the
    corrected values — leaving a block-rate transient of up to ~0.17
    absolute at the end of every block.)  The held-back tail introduces
    a constant latency of ``half_len`` input samples (~156 us at
    1.024 Msps) and means a continuous stream's final tail is only
    produced once more input arrives, which is the correct behaviour
    for an endless radio stream.

    Global input/output counters make the emission window bookkeeping
    exact when block sizes keep the polyphase grid aligned (all block
    sizes multiples of ``down / gcd(up, down)``, as the fixed
    SDR_BLOCK_SIZE does); for other sizes the counters still guarantee
    the long-run output count never drifts.
    """

    def __init__(self, up: int, down: int, window: object = None,
                 emit_align: int = 1) -> None:
        self.up: int = int(up)
        self.down: int = int(down)
        self.window: object = window
        # Emission boundaries are rounded down to a multiple of this.
        # Downstream stages that decimate the output block-wise with a
        # stateless polyphase filter (composite -> audio uses
        # resample_poly(1, 4) per block) silently require every block
        # they receive to be a multiple of their decimation factor, or
        # their per-block output grids stop tiling the global grid and
        # each block boundary picks up a fractional-sample phase jump.
        self.emit_align: int = max(1, int(emit_align))
        # Half-length of the internal polyphase FIR (scipy default)
        self._half_len: int = 10 * max(self.up, self.down)
        self._overlap: int = self._half_len * 2
        self._prev_tail: np.ndarray | None = None
        self._in_total: int = 0     # input samples consumed so far
        self._out_emitted: int = 0  # output samples emitted so far

    # ----- public API -----

    def process(self, x: np.ndarray) -> np.ndarray:
        """Resample *x* while maintaining block-to-block continuity."""
        if self._prev_tail is None:
            # Zero seed: makes the first emitted outputs identical to the
            # leading edge of a one-shot resample_poly (which zero-pads).
            self._prev_tail = np.zeros(self._overlap, dtype=x.dtype)

        ext_start = self._in_total - self._overlap
        ext = np.concatenate([self._prev_tail, x])
        y = self._resample(ext)
        self._in_total += x.size

        # Local output j of ``y`` corresponds to global output j + off.
        off = (ext_start * self.up) // self.down
        # Only emit outputs whose FIR support is fully inside received
        # input: global outputs strictly below
        # (in_total - half_len) * up / down, rounded down to the
        # emission alignment.
        out_max_global = ((self._in_total - self._half_len) * self.up) // self.down
        out_max_global = (out_max_global // self.emit_align) * self.emit_align
        a = max(self._out_emitted - off, 0)
        b = min(max(out_max_global - off, a), y.size)
        seg = y[a:b]
        self._out_emitted += seg.size

        if x.size >= self._overlap:
            self._prev_tail = x[-self._overlap:].copy()
        else:
            self._prev_tail = ext[-self._overlap:].copy()
        return seg

    def reset(self) -> None:
        """Discard saved tail so the next call starts fresh."""
        self._prev_tail = None
        self._in_total = 0
        self._out_emitted = 0

    # ----- internal -----

    def _resample(self, x: np.ndarray) -> np.ndarray:
        if self.window is not None:
            return signal.resample_poly(x, self.up, self.down, window=self.window)
        return signal.resample_poly(x, self.up, self.down)


# --------------------------------------------------
# Side-channel spectral noise reducer (STFT / Wiener)
# --------------------------------------------------
class SideNoiseReducer:
    """STFT-based spectral noise reduction for the stereo side (L-R)/2 channel.

    Streaming overlap-add with Hann analysis + synthesis windowing.  Estimates
    a per-frequency-bin noise floor via running minimum with leakage and
    applies a Wiener gain bounded by ``alpha_floor`` to limit musical-noise
    artefacts.  Output is delayed by ``frame - hop`` samples.
    """

    def __init__(
        self, sample_rate: float, frame: int = 1024, hop: int = 256,
        alpha_floor: float = 0.15, beta: float = 1.0,
        noise_decay_db_per_sec: float = 1.5,
        noise_bias: float = 3.0,
        power_smooth_ms: float = 120.0,
        dd_alpha: float = 0.98,
        gain_freq_smooth_bins: int = 3,
        lo_hz: float = 1500.0, hi_hz: float = 15000.0,
    ) -> None:
        if frame <= 0 or hop <= 0:
            raise ValueError("frame and hop must be positive")
        if frame % hop != 0 or frame // hop < 2:
            raise ValueError(
                "frame must be an integer multiple of hop with >= 50% overlap"
            )
        self.sample_rate: float = float(sample_rate)
        self.frame: int = int(frame)
        self.hop: int = int(hop)
        self.alpha_floor: float = float(alpha_floor)
        # Over-subtraction factor; clamped non-negative because a
        # negative beta would produce gains above 1 / sign flips.
        self.beta: float = max(float(beta), 0.0)
        self.window: np.ndarray = signal.windows.hann(
            self.frame, sym=False,
        ).astype(np.float32)
        self.cola_norm: float = self._compute_cola_norm()
        freqs = np.fft.rfftfreq(self.frame, d=1.0 / self.sample_rate)
        self.band_mask: np.ndarray = (
            (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
        ).astype(np.float32)
        # Per-hop time constants
        hop_dt_ms = 1000.0 * self.hop / self.sample_rate
        # Smooth raw power per bin (variance reduction)
        self.power_smooth_alpha: float = float(
            np.exp(-hop_dt_ms / max(power_smooth_ms, 1e-3))
        )
        # Slow upward leak of the noise-floor minimum tracker.
        self.noise_decay: float = float(
            10.0 ** (
                float(noise_decay_db_per_sec) / (10.0 * 1000.0 / hop_dt_ms)
            )
        )
        # Bias correction: minimum-of-smoothed-power systematically
        # underestimates noise level due to sampling variance of the
        # smoothed estimator; multiply by this factor when using as the
        # noise estimate.
        self.noise_bias: float = float(noise_bias)
        # Decision-directed smoothing factor for the a priori SNR; 0.98 is
        # the value standard in Ephraim-Malah, suppresses musical noise.
        self.dd_alpha: float = float(np.clip(dd_alpha, 0.0, 0.999))
        # Number of FFT bins over which to smooth the gain (median filter).
        # Helps avoid isolated tonal artefacts.  Set <=1 to disable.
        self.gain_freq_smooth_bins: int = int(max(1, gain_freq_smooth_bins))
        self.in_buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.synth_overlap: np.ndarray = np.zeros(
            self.frame - self.hop, dtype=np.float32,
        )
        self.power_smooth: np.ndarray | None = None
        self.noise_floor: np.ndarray | None = None
        self.prev_gain: np.ndarray | None = None
        self.prev_gamma: np.ndarray | None = None

    def _compute_cola_norm(self) -> float:
        n, h = self.frame, self.hop
        t = max(4 * n, 8 * h)
        s = np.zeros(t, dtype=np.float64)
        k = 0
        w_sq = (self.window.astype(np.float64)) ** 2
        while k * h + n <= t:
            s[k * h:k * h + n] += w_sq
            k += 1
        return float(s[t // 2])

    @property
    def latency_samples(self) -> int:
        return self.frame - self.hop

    def reset(self) -> None:
        self.in_buf = np.zeros(0, dtype=np.float32)
        self.synth_overlap = np.zeros(
            self.frame - self.hop, dtype=np.float32,
        )
        self.power_smooth = None
        self.noise_floor = None
        self.prev_gain = None
        self.prev_gamma = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        self.in_buf = np.concatenate([self.in_buf, x])
        n, h = self.frame, self.hop
        out_chunks: list[np.ndarray] = []
        while self.in_buf.size >= n:
            frame = self.in_buf[:n]
            self.in_buf = self.in_buf[h:]

            windowed = frame * self.window
            spec = np.fft.rfft(windowed)
            power = spec.real * spec.real + spec.imag * spec.imag

            # Stage 1: smooth raw power per bin (variance reduction)
            if self.power_smooth is None:
                self.power_smooth = power.copy()
            else:
                ap = self.power_smooth_alpha
                self.power_smooth = ap * self.power_smooth + (1.0 - ap) * power

            # Stage 2: minimum-statistics noise tracker on smoothed power.
            # Initialised from the first frame so initial gain stays near 1
            # before the floor settles; subsequent frames track minimum
            # with a slow upward leak so silent-bin dips lock the estimate.
            if self.noise_floor is None:
                self.noise_floor = self.power_smooth.copy()
            else:
                self.noise_floor = np.minimum(
                    self.noise_floor * self.noise_decay,
                    self.power_smooth,
                )
            noise_est = self.noise_bias * self.noise_floor + 1e-18

            # Stage 3: Ephraim-Malah Decision-Directed Wiener.
            # ξ̂(k,n) = α·G(k,n-1)²·γ(k,n-1) + (1-α)·max(γ(k,n)-1, 0)
            # G = ξ̂ / (ξ̂ + 1)
            gamma = self.power_smooth / noise_est
            posterior = np.maximum(0.0, gamma - 1.0)
            if self.prev_gain is None or self.prev_gamma is None:
                xi = posterior
            else:
                xi = (
                    self.dd_alpha * (self.prev_gain ** 2) * self.prev_gamma
                    + (1.0 - self.dd_alpha) * posterior
                )
            # The epsilon keeps the division defined for beta=0 (which is
            # reachable via --side-nr-beta 0): xi=0, beta=0 would
            # otherwise produce 0/0 = NaN and propagate silence-killing
            # NaNs into the overlap-add output.
            gain_w = xi / (xi + self.beta + 1e-12)
            gain = np.maximum(self.alpha_floor, gain_w)

            # Optional smoothing across nearby bins to break up isolated
            # gain spikes that produce musical-noise tonal artefacts.
            if self.gain_freq_smooth_bins > 1:
                k = self.gain_freq_smooth_bins
                kernel = np.ones(k, dtype=np.float32) / float(k)
                gain = np.convolve(gain, kernel, mode="same").astype(np.float32)
                # Re-clip lower bound after smoothing.
                gain = np.maximum(self.alpha_floor, gain)

            # Save state for the next frame's DD smoothing.
            self.prev_gain = gain.astype(np.float32)
            self.prev_gamma = gamma.astype(np.float32)

            gain = self.band_mask * gain + (1.0 - self.band_mask)

            spec_out = spec * gain
            out_frame = (
                np.fft.irfft(spec_out, n=n).astype(np.float32)
                * self.window
            )

            out_hop = (
                self.synth_overlap[:h] + out_frame[:h]
            ) / self.cola_norm

            new_overlap = np.zeros(n - h, dtype=np.float32)
            n_carry = max(0, n - 2 * h)
            if n_carry > 0:
                new_overlap[:n_carry] = self.synth_overlap[h:h + n_carry]
            new_overlap += out_frame[h:n]
            self.synth_overlap = new_overlap

            out_chunks.append(out_hop)

        if not out_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out_chunks)


class StreamAligner:
    """Holds samples and releases them in chunks matched to another stream.

    Used to align the un-processed mid (L+R)/2 path with the latency of
    :class:`SideNoiseReducer` so that ``mid + side_clean`` reconstructs the
    output with sample-accurate alignment.
    """

    def __init__(self) -> None:
        self.buf: np.ndarray = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)

    def feed_and_take(self, x: np.ndarray, n_take: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size:
            self.buf = np.concatenate([self.buf, x])
        if n_take <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.buf.size < n_take:
            n_take = self.buf.size
        out = self.buf[:n_take]
        self.buf = self.buf[n_take:]
        return out
