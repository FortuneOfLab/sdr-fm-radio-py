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
"""Constants for the FM receiver system."""

# --------------------------------------------------
# SDR
# --------------------------------------------------
SDR_SAMPLE_RATE = 1.024e6           # Standard mode sample rate (Hz)
SDR_SAMPLE_RATE_LIGHT = 0.25e6      # Light mode sample rate (Hz)
SDR_CENTER_FREQ_DEFAULT = 80e6      # Default center frequency (Hz)
SDR_BLOCK_SIZE = 16384              # Samples per SDR read
SDR_QUEUE_MAXSIZE = 80              # Max queued SDR sample blocks (~1.28 sec
                                    # of headroom for transient processing
                                    # stalls; samples are dropped beyond this)

# --------------------------------------------------
# Main FM demodulation
# --------------------------------------------------
# The standard demodulator can recover the composite either with a PLL
# or with an arctan discriminator (angle(x[n]*conj(x[n-1]))).  Measured
# closed-loop response of the PLL (Kp/Ki below) over the MPX band:
#   +3.9 dB peaking at 19-23 kHz, -4.9 dB at 53 kHz (9 dB tilt across
#   the L-R band) and a -31 deg phase inconsistency between 19 kHz and
#   38 kHz.  The discriminator is exactly flat (0.00 dB) with pure-delay
#   phase (38k - 2x19k consistency: -0.2 deg), so it is the default.
# The PLL path is kept for A/B listening comparison.
MAIN_DEMOD_USE_PLL = False          # True: legacy PLL demod, False: discriminator

# --------------------------------------------------
# PLL gains
# --------------------------------------------------
MAIN_PLL_KP = 0.12926              # Main PLL proportional gain
MAIN_PLL_KI = 0.0208844            # Main PLL integral gain
PILOT_PLL_KP = 0.032               # Pilot PLL proportional gain (reduced jitter)
PILOT_PLL_KI = 0.00008             # Pilot PLL integral gain (reduced jitter)

# --------------------------------------------------
# Filter parameters
# --------------------------------------------------
IQ_LOWPASS_ORDER = 5                # IQ lowpass filter order
IQ_LOWPASS_CUTOFF = 200e3           # IQ lowpass cutoff frequency (Hz)

MONO_LOWPASS_ORDER = 15             # Mono lowpass filter order (standard)
MONO_LOWPASS_ORDER_LIGHT = 1        # Mono lowpass filter order (light)
MONO_LOWPASS_CUTOFF = 15000.0       # Mono/baseband lowpass cutoff (Hz)
LR_BASE_LOWPASS_CUTOFF = 15000.0    # L-R baseband lowpass cutoff (Hz)
LR_HIGH_SPLIT_CUTOFF = 7000.0       # L-R split frequency for high-band damping (Hz)
LR_HIGH_SUPER_SPLIT_CUTOFF = 12000.0  # L-R split frequency between mid-high and super-high (Hz)
LR_HIGH_MIN_GAIN = 0.40             # Minimum mid-high (7-12k) gain at low stereo blend
LR_HIGH_MAX_GAIN = 1.00             # Maximum mid-high gain at low pilot SNR (1.0 at
                                    # HF_BLEND HI threshold).  1.0 = the SNR-adaptive HF
                                    # damping ceiling is neutral.  History: 0.85 was
                                    # chosen by listening test BEFORE the side-channel
                                    # NR existed; with side NR covering HF side noise
                                    # directly, ablation showed the damping cost 7.8 dB
                                    # of separation at weak signal for 0.7 dB of SNR,
                                    # and a fresh listening test confirmed the neutral
                                    # setting sounds fine.  The mechanism stays in place
                                    # (set <1.0 to re-enable damping for a noisier
                                    # station).
LR_SUPER_HIGH_MIN_GAIN = 0.20       # Minimum super-high (12-15k) gain at low stereo blend
LR_SUPER_HIGH_MAX_GAIN = 1.00       # Maximum super-high gain at low pilot SNR (1.0 at
                                    # HF_BLEND HI threshold).  See LR_HIGH_MAX_GAIN
                                    # above: neutral by default since the side-channel
                                    # NR supersedes broadband HF damping (was 0.50).

PILOT_BANDPASS_ORDER = 9            # Pilot bandpass filter order (standard)
PILOT_BANDPASS_ORDER_LIGHT = 1      # Pilot bandpass filter order (light)
PILOT_BANDPASS_LOW = 18000.0        # Pilot bandpass lower edge (Hz)
PILOT_BANDPASS_HIGH = 20000.0       # Pilot bandpass upper edge (Hz)
PILOT_NOISE_BAND1_LOW = 16000.0     # Pilot SNR noise band 1 lower edge (Hz)
PILOT_NOISE_BAND1_HIGH = 17500.0    # Pilot SNR noise band 1 upper edge (Hz)
PILOT_NOISE_BAND2_LOW = 20500.0     # Pilot SNR noise band 2 lower edge (Hz)
PILOT_NOISE_BAND2_HIGH = 22000.0    # Pilot SNR noise band 2 upper edge (Hz)
STEREO_PILOT_RESIDUAL_CENTER_HZ = 19000.0  # Center frequency used by residual pilot tracking
STEREO_SUBCARRIER_PHASE_OFFSET_DEG = 316.0  # Fixed phase offset for 38k subcarrier generation
                                    # (standard demodulator).  History of the value:
                                    #   300.0  original tuning (PLL demod + real order-9
                                    #          pilot bandpass + FFT Hilbert; the bandpass
                                    #          hid -15 deg at the subcarrier)
                                    #   285.0  analytic heterodyne pilot path (0 deg
                                    #          static phase; 300 - 15)
                                    #   316.0  discriminator main demod: the PLL's
                                    #          closed loop had a -30.7 deg phase
                                    #          inconsistency between 19 kHz and 38 kHz
                                    #          which the discriminator does not
                                    #          (285 + 30.7 = 315.7).  Synthetic sweep
                                    #          confirms a broad optimum at 315-320 with
                                    #          separation improving to ~26/33 dB.
HARDWARE_SUBCARRIER_PHASE_TRIM_DEG = 84.0  # Front-end (tuner) phase trim added to every
                                    # variant's DSP-intrinsic subcarrier offset for real
                                    # hardware.  Discovery: ALL real captures - antenna
                                    # 91.6 (-83 deg), antenna 80.0 (-80), and an optical-
                                    # fibre feed 80.0 with no multipath (+92 = axis -88) -
                                    # showed the same ~+-85-90 deg corrector demand that
                                    # synthetic IQ (no tuner) does not, identifying it as
                                    # the R820T IF filter's 19k/38k phase characteristic,
                                    # not multipath.  Sitting at the +-90 deg boundary
                                    # also made the acquisition branch flip between
                                    # sessions (the optical capture decoded L/R-swapped
                                    # vs the antenna ones).  With this trim (316+84 =
                                    # 40 deg effective) the tracker settles at 0 +- 4.3
                                    # deg on all three captures, on the same branch as
                                    # every historical antenna session (decode total
                                    # ~39 deg, matching the listening-approved state).
STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL = 285.0  # Operating point when the legacy PLL main
                                    # demod is selected (MAIN_DEMOD_USE_PLL = True): the
                                    # PLL chain includes the -30.7 deg 19k/38k phase
                                    # inconsistency, so it keeps the pre-discriminator
                                    # value.  FMDemodulator picks the matching offset
                                    # automatically based on MAIN_DEMOD_USE_PLL.
STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT = 297.4  # Same operating-point preservation for
                                    # the light demodulator: its old pilot bandpass was
                                    # order 1 with only -1.31 deg phase at 19 kHz
                                    # (-2.62 deg at the subcarrier), so the old effective
                                    # offset was 300 - 2.62 = 297.4 deg.
STEREO_MONO_DELAY_SAMPLES = 18      # Delay mono path to match LR path group delay (at COMPOSITE_RATE)
STEREO_LR_SIDE_RATIO_CAP_ENABLE = False     # Enable limiting of |L-R|/|L+R| ratio for stability
STEREO_LR_SIDE_RATIO_CAP_TARGET = 0.35     # Target upper bound of |L-R|/|L+R| before limiting
STEREO_LR_SIDE_RATIO_CAP_MIN_GAIN = 0.35   # Lower bound of side-cap gain to avoid mono-collapse
STEREO_LR_SIDE_RATIO_CAP_ATTACK = 0.25     # Gain attack speed when limiting engages
STEREO_LR_SIDE_RATIO_CAP_RELEASE = 0.45    # Gain release speed when limiting disengages
STEREO_PHASE_ERR_SMOOTHING = 0.15   # EMA smoothing for LR demod phase correction
STEREO_PHASE_ANISO_GATE = 0.2       # Minimum covariance anisotropy
                                    # sqrt((varI-varQ)^2+4cov^2)/(varI+varQ) required to
                                    # update the phase tracker.  The principal-axis
                                    # estimate is only meaningful when the demodulated
                                    # (I,Q) pair is strongly 1-D; on mono programme the
                                    # side channel is noise (isotropic) and unclamped
                                    # updates would random-walk the tracker across the
                                    # 180-deg branch boundary (L/R swap).  Measured on
                                    # the reference station: music blocks p5 = 0.55,
                                    # isotropic-noise blocks p99 = 0.05, so 0.2 has an
                                    # order of magnitude of margin on both sides.  The
                                    # tracker FREEZES (holds its last angle) while the
                                    # gate is closed.
                                    # History: this replaces STEREO_PHASE_ERR_LIMIT_DEG
                                    # (45, then 75).  A clamp guards the +-90 deg
                                    # principal-axis ambiguity but also truncates
                                    # legitimate large corrections (the reference
                                    # station's raw estimates sit around -83 deg and
                                    # wrap past the boundary).  The tracker now resolves
                                    # the 180-deg branch by CONTINUITY (nearest candidate
                                    # in the pi-periodic family to the tracked state) and
                                    # needs no clamp; only the initial acquisition
                                    # assumes the true rotation lies within +-90 deg,
                                    # which is the FM standard's pilot phase convention.
STEREO_PHASE_SIDE_GATE_DB = -18.0   # Minimum demodulated side power relative to mono
                                    # power (dB) for a phase-tracker update.  Anisotropy
                                    # alone is scale-invariant: on a MONO broadcast the
                                    # tiny deterministic residue in the side band can be
                                    # strongly 1-D at ~-32 dB below mono, which would
                                    # otherwise acquire a random angle.  Measured
                                    # denom/mono: real stereo music p5 = -11 dB,
                                    # noiseless-mono residue median = -32 dB, mono at
                                    # CNR 20 = -22 dB (also blocked by the anisotropy
                                    # gate), so -18 leaves ~7 dB of margin both ways.
STEREO_PHASE_ACQUIRE_BLOCKS = 6     # Consecutive informative blocks (~100 ms) required
                                    # before cold-start acquisition; the initial angle is
                                    # the doubled-angle circular mean over the streak,
                                    # which is invariant to +-90 deg wrapping of the raw
                                    # estimates (a single-block init on a station near
                                    # the boundary would lock the wrong 180-deg branch -
                                    # a permanent L/R swap - with the probability of one
                                    # raw estimate wrapping, ~20% on the reference
                                    # station).
STEREO_PHASE_SIDE_OVER_NOISE_DB = 24.0  # Minimum demodulated side power above the
                                    # pilot-band noise estimate (dB) for a tracker
                                    # update.  FM discriminator noise rises as f^2, so
                                    # the side band's own noise sits ABOVE the mono band
                                    # (mono-relative gating inverts during silence) and
                                    # is anisotropic (band asymmetry about 38 kHz plus
                                    # deterministic FM products), forming a stable
                                    # pseudo-axis at aniso ~0.5 that overlaps genuine
                                    # content.  Measured through the real demod path:
                                    # noise-only sits at med 22.1 (synthetic silence,
                                    # CNR 35; p95 23.4), 16.5 (CNR 20), 4-11 dB
                                    # (hardware); genuine stereo content p5 = 24.6
                                    # (91.6 music).  24.0 blocks all noise regimes to
                                    # the ~1% level - the residual trickle is mopped up
                                    # by the confidence weighting and the gate-closed
                                    # leak - while music tracks on its strong majority
                                    # of blocks.
STEREO_PHASE_NOISE_CONF_RAMP_DB = 6.0  # Confidence ramp above the side-over-noise
                                    # gate: an update's weight scales linearly from 0
                                    # at the gate to 1 at gate + this, multiplied with
                                    # the anisotropy weight.  Noise pseudo-axis blocks
                                    # that barely clear the gate are nearly weightless.
STEREO_PHASE_CONF_ANISO = 0.6       # Anisotropy at which a tracker update gets FULL
                                    # weight; between the gate (0.2) and this the
                                    # innovation is scaled linearly.  Field capture
                                    # (2026-07-20 antenna, near-mono nighttime
                                    # programme) showed gate-passing marginal blocks
                                    # (aniso 0.2-0.5) walking the tracker ~74 deg and
                                    # across the branch boundary; genuine stereo blocks
                                    # measure 0.75-0.89 and keep full-speed tracking.
STEREO_PHASE_BRANCH_CONF = 0.7      # Minimum recent-confidence EMA required to let the
                                    # tracked angle cross +-90 deg (a nearest-branch
                                    # L/R polarity flip).  With the hardware trim all
                                    # legitimate operating points sit near 0, so only
                                    # sustained confident tracking (e.g. a real channel
                                    # drift, as in the e2e drift test) may cross;
                                    # low-confidence wander halts at the boundary.
STEREO_PHASE_LEAK_DEG_PER_SEC = 0.5 # Decay rate of the tracked angle toward 0 (the
                                    # hardware-trim prior) while the gates are CLOSED
                                    # after acquisition.  With no side information the
                                    # prior beats holding a possibly wandered value; a
                                    # genuine static offset re-converges within ~1 s of
                                    # confident content returning.
STEREO_IQ_PHASE_CORRECTION_ENABLE = True   # Enable I/Q rotation correction in LR demod

LR_BANDPASS_ORDER = 15              # L-R bandpass filter order (standard)
LR_BANDPASS_ORDER_LIGHT = 1         # L-R bandpass filter order (light)
LR_BANDPASS_LOW = 23000.0           # L-R bandpass lower edge (Hz)
LR_BANDPASS_HIGH = 53000.0          # L-R bandpass upper edge (Hz)
STEREO_LR_DEMOD_GAIN = 2.0          # Gain compensation for DSB-SC synchronous demod
STEREO_DIAG_ENABLE = False                  # Enable stereo demod diagnostics logging
STEREO_DIAG_LOG_INTERVAL_BLOCKS = 120       # Log interval (composite blocks) for diagnostics

DEEMPHASIS_TAU = 50e-6              # De-emphasis time constant (seconds)
DC_OFFSET_ALPHA = 0.01              # DC offset smoothing coefficient

# --------------------------------------------------
# Audio output
# --------------------------------------------------
AUDIO_OUTPUT_RATE = 48000           # Audio output sample rate (Hz)
AUDIO_FRAMES_PER_BUFFER = 1024     # Frames per audio callback
AUDIO_QUEUE_MAXSIZE = 50            # Max queued audio blocks
AUDIO_CHANNELS = 2                  # Stereo output channels
AUDIO_ENQUEUE_TIMEOUT = 0.01       # Timeout for audio queue put (seconds)

# --------------------------------------------------
# Recording
# --------------------------------------------------
RECORDINGS_DIR = "recordings"       # Directory (relative to CWD) where the
                                    # CLI places auto-named recordings and
                                    # their .json metadata sidecars
RECORD_SAMPLE_WIDTH = 2             # 16-bit PCM sample width (bytes)
RECORD_MAX_INT16 = 32767            # Max value for int16 conversion
RECORD_QUEUE_MAXSIZE = 200          # Max queued recording chunks (~3.2 s at
                                    # 48 kHz / 768-sample chunks); absorbs
                                    # disk-write stalls so the realtime path
                                    # is not blocked by file I/O
AUDIO_RECORD_ROTATE_THRESHOLD_BYTES = 4_000_000_000
                                    # Same WAV 4-GiB limit as the IQ path
                                    # (see IQ_RECORD_ROTATE_THRESHOLD_BYTES
                                    # below).  At 48 kHz / 16-bit / 2 ch =
                                    # 192 KB/s a single file fills in ~6.2
                                    # hours; rotate to a new file before
                                    # ``wave.writeframes`` overflows its
                                    # 32-bit data-size header field.
IQ_RECORD_QUEUE_MAXSIZE = 200       # Max queued IQ blocks for async IQ-WAV
                                    # recording (~3.2 s at 1.024 Msps /
                                    # 16384-sample blocks).  Each entry is a
                                    # complex64 array (~128 kB) so the cap
                                    # bounds peak memory at ~26 MB.
IQ_RECORD_ROTATE_THRESHOLD_BYTES = 4_000_000_000
                                    # WAV format caps the data chunk at
                                    # 2^32 - 1 bytes (4 GiB) and Python's
                                    # wave module raises struct.error past
                                    # that. At 1.024 Msps / 16-bit IQ the
                                    # rate is ~4 MB/s so a single file fills
                                    # in ~16 min; rotate to a new file once
                                    # the next chunk would push us above
                                    # this threshold.  Leaves ~290 MB of
                                    # headroom under the hard limit for
                                    # the header patch.

# --------------------------------------------------
# Demodulator
# --------------------------------------------------
COMPOSITE_RATE = 192000             # Composite signal sample rate (Hz)
LIGHT_COMPOSITE_SCALE = 0.35       # Scaling factor for light mode composite
STANDARD_RESAMPLE_KAISER_BETA = 10.0  # Kaiser beta for standard IQ->composite resample
# --------------------------------------------------
# Adaptive stereo blend (pilot SNR based)
# --------------------------------------------------
STEREO_BLEND_PILOT_SNR_DB_HI = 16.5        # Pilot SNR above this -> full stereo
STEREO_BLEND_PILOT_SNR_DB_LO = 7.0         # Pilot SNR below this -> full mono
STEREO_BLEND_PILOT_SNR_EMA_ALPHA = 0.10    # EMA alpha for pilot SNR tracking
STEREO_BLEND_PILOT_JITTER_EMA_ALPHA = 0.12  # EMA alpha for pilot SNR jitter tracking
STEREO_BLEND_PILOT_JITTER_REF_DB = 2.5     # Jitter reference in dB (higher -> less sensitive)
STEREO_BLEND_STABILITY_MIN_FACTOR = 0.85   # Minimum stereo factor when pilot is unstable
STEREO_BLEND_SMOOTHING = 0.08              # EMA smoothing for blend factor (0-1)

# --------------------------------------------------
# Adaptive HF stereo blend (frequency-axis blend, pilot SNR based)
# --------------------------------------------------
# Independently shapes the LR_*_MAX_GAIN ceilings as a function of pilot SNR:
# above HI -> ceilings ramp to 1.0 (no HF damping);
# below LO -> ceilings stay at LR_*_MAX_GAIN (aggressive HF damping).
# When LR_*_MAX_GAIN are 1.0, this has no effect.
STEREO_HF_BLEND_PILOT_SNR_DB_HI = 35.0     # Above this -> full HF stereo width
STEREO_HF_BLEND_PILOT_SNR_DB_LO = 15.0     # Below this -> configured MAX_GAIN damping

# --------------------------------------------------
# Side-channel STFT noise reducer (mid/side spectral suppression)
# --------------------------------------------------
# Operates on the (L-R)/2 path at the audio rate (post de-emphasis), leaving
# the mid (L+R)/2 path untouched. Estimates the noise floor per FFT bin via
# running minimum with leakage and applies a Wiener gain bounded by
# SIDE_NR_ALPHA_FLOOR to limit musical-noise artefacts.
SIDE_NR_ENABLE = True
SIDE_NR_FRAME = 1024            # STFT frame size (samples at AUDIO_OUTPUT_RATE)
SIDE_NR_HOP = 256               # STFT hop size (75% overlap)
SIDE_NR_ALPHA_FLOOR = 0.30      # Minimum Wiener gain (linear). 0.30 ≈ -10 dB max attenuation
SIDE_NR_BETA = 1.0              # Over-subtraction factor (1.0 = pure Wiener)
SIDE_NR_NOISE_DECAY_DB_PER_SEC = 6.0  # Noise floor leakage rate (dB/sec)
SIDE_NR_TONE_PROTECT_DB = 8.0   # Clamp noise-tracker input to local median + this (dB);
                                # protects stationary tones from being absorbed into the
                                # noise floor (measured -10 dB tone loss without it).
                                # <= 0 disables the protection.
SIDE_NR_TONE_PROTECT_MED_BINS = 33  # Median window (bins) for the tonal-protection clamp
SIDE_NR_LO_HZ = 1500.0          # Lower edge of NR band (preserve low-frequency stereo)
SIDE_NR_HI_HZ = 15000.0         # Upper edge of NR band

# --------------------------------------------------
# Pilot tone notch filter (19 kHz removal)
# --------------------------------------------------
PILOT_NOTCH_FREQ = 19000.0          # Notch centre frequency (Hz)
PILOT_NOTCH_Q = 30.0                # Quality factor (narrow notch)

# --------------------------------------------------
# Auto gain control (hardware gain adjustment)
# --------------------------------------------------
# RTL-SDR valid gain values in tenths of dB (R820T tuner)
AGC_GAIN_TABLE: tuple[int, ...] = (
    0, 9, 14, 27, 37, 77, 87, 125, 144, 157,
    166, 197, 207, 229, 254, 280, 297, 328, 338, 364,
    372, 386, 402, 421, 434, 439, 445, 480, 496,
)
AGC_DEFAULT_GAIN_INDEX = 19         # 36.4 dB (upper-mid range)
AGC_CLIP_THRESHOLD = 0.95           # Peak |IQ| above this -> clipping
AGC_WEAK_THRESHOLD = 0.3            # Peak |IQ| below this -> weak signal
AGC_CLIP_COUNT = 3                  # Consecutive clipping blocks to step down
AGC_WEAK_COUNT = 15                 # Consecutive weak blocks to step up
AGC_HOLDOFF_BLOCKS = 10             # Blocks to skip after a gain change
AGC_WARMUP_SEC = 2.0                # Suppress AGC for this long after startup
                                    # (Numba JIT compile + filter settling)
