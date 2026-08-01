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
PILOT_NOISE_BAND_ORDER = 9          # Pilot SNR noise-band filter order, BOTH variants.
                                    # Historically the light variant reused its order-1
                                    # pilot order here, and the order-1 skirts leaked the
                                    # 19 kHz pilot itself into the noise reference
                                    # (measured -9.6/-10.4 dB at 19 kHz, and only
                                    # -7.9 dB for 23 kHz DSB content in band 2), locking
                                    # light's pilot SNR at 9.975 dB and its blend at
                                    # 0.313 on a PURE pilot of any amplitude - light
                                    # never reached full stereo and its side NR could
                                    # barely train.  Order 9 (the standard variant's
                                    # value, unchanged there) puts the pilot at
                                    # -82/-89.5 dB in the bands, giving light a real
                                    # noise measurement and SNR-scale parity with the
                                    # standard chain, so the shared blend/tracker/NR
                                    # thresholds mean the same thing in both variants.
                                    # The light PILOT LP stays order 1: it sets the
                                    # subcarrier phase operating point (offset 0.3 deg
                                    # was calibrated against it) and does not touch the
                                    # noise reference.
PILOT_NOISE_BAND1_LOW = 16000.0     # Pilot SNR noise band 1 lower edge (Hz)
PILOT_NOISE_BAND1_HIGH = 17500.0    # Pilot SNR noise band 1 upper edge (Hz)
PILOT_NOISE_BAND2_LOW = 20500.0     # Pilot SNR noise band 2 lower edge (Hz)
PILOT_NOISE_BAND2_HIGH = 22000.0    # Pilot SNR noise band 2 upper edge (Hz)
STEREO_PILOT_RESIDUAL_CENTER_HZ = 19000.0  # Center frequency used by residual pilot tracking
STEREO_SUBCARRIER_PHASE_OFFSET_DEG = 1.0  # Fixed phase offset for 38k subcarrier generation
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
                                    #          (285 + 30.7 = 315.7)
                                    #   1.0    linear-phase FIR bank + raw-composite
                                    #          demod: the old value was almost entirely
                                    #          compensating the removed 23-53k Butterworth
                                    #          bandpass's group delay at 38 kHz.  With the
                                    #          matched FIR bank the chain is intrinsically
                                    #          phase-true; the synthetic sweep (hifi TX,
                                    #          noiseless, corrector off, 0.1 deg steps)
                                    #          peaks at 1.0 deg with 47.6 dB separation
                                    #          at 1 kHz, flat within 0.1 dB of 0 deg.
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
                                    # vs the antenna ones).  With the FIR bank's DSP
                                    # offset of 1.0 deg the total applied is 1+84 =
                                    # 85 deg; re-validated on the FIR chain: the
                                    # tracker settles at med -1.1 (antenna 91.6),
                                    # -3.0 (CATV 83.7) and -3.6 deg (optical 82.5),
                                    # on the same branch as every historical antenna
                                    # session.  The trim itself is a front-end
                                    # property and is NOT retuned with DSP changes.
STEREO_SUBCARRIER_PHASE_OFFSET_DEG_PLL = 331.1  # Operating point when the legacy PLL main
                                    # demod is selected (MAIN_DEMOD_USE_PLL = True): the
                                    # PLL chain carries its own 19k/38k phase
                                    # inconsistency, so its optimum stays far from the
                                    # discriminator's.  Re-swept for the FIR bank
                                    # (was 285.0 with the IIR bank + 23-53k bandpass);
                                    # the PLL chain itself caps separation at ~26 dB.
                                    # FMDemodulator picks the matching offset
                                    # automatically based on MAIN_DEMOD_USE_PLL.
STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT = 0.3  # Light demodulator operating point,
                                    # re-swept for the FIR bank (was 297.4 with the
                                    # order-1 IIR bank): like the standard variant it
                                    # lands near 0 deg once the bandpass group delay is
                                    # out of the chain; the light pilot path (order-1
                                    # lowpass) caps separation at ~24 dB.
STEREO_MONO_DELAY_SAMPLES = 0       # Mono-path delay compensation; 0 because the FIR bank's shared tap count matches mono/side group delays by construction
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
STEREO_PHASE_SIDE_OVER_NOISE_DB = 26.0  # Minimum demodulated side power above the
                                    # pilot-band noise estimate (dB) for a tracker
                                    # update.  FM discriminator noise rises as f^2, so
                                    # the side band's own noise sits ABOVE the mono band
                                    # (mono-relative gating inverts during silence) and
                                    # is anisotropic (band asymmetry about 38 kHz plus
                                    # deterministic FM products), forming a stable
                                    # pseudo-axis at aniso ~0.5 that overlaps genuine
                                    # content.  Recalibrated for the FIR bank (its flat
                                    # passband passes ~0.5 dB more side-band noise than
                                    # the old droopy Butterworth cascade, and without a
                                    # gate margin the 20 s silence leak test wandered to
                                    # the pseudo-axis instead of decaying home).
                                    # Measured through the FIR demod path: noise-only
                                    # med 22.6 / p95 23.9 / MAX 25.2 (synthetic silence,
                                    # CNR 35), max 24.4 (CNR 20), max 23.5 (CNR 10);
                                    # genuine stereo content (aniso > 0.6 blocks) p5 /
                                    # med: 23.9 / 31.4 (antenna 91.6 music, pilot SNR
                                    # 24.7), 26.0 / 36.5 (CATV 83.7 music), 27.2 med
                                    # (optical 82.5).  26.0 sits 0.8 dB above the worst
                                    # observed noise block - silence now decays purely -
                                    # while music still updates on its strong majority
                                    # of blocks; the blocked weak-station tail falls
                                    # back to the hardware-trim prior of 0, which the
                                    # real-capture check confirms (tracker med -1.1 to
                                    # -3.6 deg on all three reference captures).
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

# Linear-phase FIR mono/side filter bank (see BaseFMDemodulator).  One
# shared tap count keeps every path's group delay identical; the L-R
# path demodulates the raw composite (no pre-demod bandpass), so the
# post-demod 15 kHz lowpass IS the side channel's band limit.
STEREO_FIR_NTAPS = 321              # Bank FIR length at 192 kHz composite (group delay 160 samples / 0.83 ms)
STEREO_FIR_TRANSITION_HZ = 3500.0   # Bank FIR transition width (15k passband edge -> ~-100 dB by 18.5k; pilot at 19k)
STEREO_LR_DEMOD_GAIN = 2.0          # Gain compensation for DSB-SC synchronous demod
STEREO_DIAG_ENABLE = False                  # Enable stereo demod diagnostics logging
STEREO_DIAG_LOG_INTERVAL_BLOCKS = 120       # Log interval (composite blocks) for diagnostics

DEEMPHASIS_TAU = 50e-6              # De-emphasis time constant (seconds)
DC_BLOCK_CUTOFF_HZ = 0.1            # IQ DC-blocker highpass cutoff (Hz).  An LTI
                                    # one-pole complex highpass replaced the old
                                    # block-mean EMA subtraction (DC_OFFSET_ALPHA=0.01).
                                    # Root cause of the "8-12 kHz separation dip":
                                    # to first order, removing an offset c from an FM
                                    # signal y = exp(j*phi) - c adds a phase error
                                    # -Im(c*e^{-j*phi}); the removed component is an
                                    # ADDITIVE error whose phase-direction part the
                                    # discriminator's nonlinearity turns into
                                    # modulation-correlated products across the
                                    # composite.  On FM the block mean is
                                    # modulation-dominated, so the old EMA estimate
                                    # wandered (-22 dB products at a 12 kHz side
                                    # tone; separation capped at ~32 dB at zero
                                    # carrier offset).  Widening the removal
                                    # bandwidth increases the input power of that
                                    # error (measured on the optical capture's
                                    # composite noise bands: none -73.8 / 0.1 Hz
                                    # -72.8 / 1 Hz -71.3 / 20 Hz -64.4 dB).  0.1 Hz
                                    # equals the old EMA's EFFECTIVE bandwidth
                                    # (alpha 0.01 per 16 ms block -> tau 1.6 s),
                                    # giving measured pilot-SNR parity to 0.01 dB on
                                    # the reference captures with the same ~1.6 s
                                    # settle, while adding what the EMA lacked:
                                    # exact block-size invariance (stateful LTI), an
                                    # exact null at 0 Hz, and no wandering estimate.
                                    # The residual pathology is CONTINUOUS across
                                    # the notch transition - measured separation at
                                    # a 12 kHz side tone: 32.8 dB at 0 Hz offset /
                                    # 35.8 at 0.1 Hz / 47.4 at 0.3 Hz / 48.5 at
                                    # 1 Hz and flat beyond - so it matters when the
                                    # residual carrier offset falls within roughly
                                    # 0.1-0.3 Hz of the notch.  Exact zero tuning is
                                    # not physically impossible, but this hardware's
                                    # measured residual offset (~60 Hz) sits far
                                    # outside the notch; synthetic periodic tones
                                    # show the pathology strongly because their IQ
                                    # spectrum has DISCRETE carrier lines that can
                                    # land exactly in the notch.  The sweep tooling
                                    # prints a note near the notch and offers
                                    # --carrier-offset-hz for realistic charts.
# --------------------------------------------------
# Audio output
# --------------------------------------------------
AUDIO_OUTPUT_RATE = 48000           # Audio output sample rate (Hz)
# Final audio band limit: one linear-phase FIR applied IDENTICALLY to L
# and R after the stereo matrix (identical filters cannot degrade
# channel separation).  Needed because the raw-composite demod's
# equivalence to an ideal bandpass holds in the 0-15 kHz target band
# only: through the bank FIR's 15-18.5 kHz transition, out-of-band
# composite (20.5-22 kHz and 54-56.5 kHz noise) maps to 16-18.5 kHz
# near-audible side content (measured +22-25 dB vs the old IIR chain
# in the 16-17.5 kHz band; side NR does not reach above 15 kHz).  The
# sharp 15->16.5 kHz transition at the 48 kHz audio rate costs 1/4 the
# taps of doing it at the composite rate and crushes the leak below
# the old chain's floor.
AUDIO_FINAL_LP_NTAPS = 183          # At 48 kHz (Kaiser beta 9, ~-90 dB); group delay 91 samples = 1.9 ms
AUDIO_FINAL_LP_CUTOFF_HZ = 15000.0  # Passband edge (Hz)
AUDIO_FINAL_LP_STOP_HZ = 16500.0    # Stopband edge (Hz)
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
STEREO_BLEND_STABILITY_MIN_FACTOR = 1.00  # Floor of the pilot-"jitter" stability factor.
                                    # 1.0 = the stability term is NEUTRAL (blend follows
                                    # pilot SNR alone).  History: 0.85 penalised blend by
                                    # up to 15% via the EMA of |snr_db - snr_ema| with
                                    # REF 2.5 dB - calibrated on synthetic tones (jitter
                                    # 0.06-0.4 dB).  Field measurement showed real
                                    # broadcasts sit at 1.4-4.1 dB REGARDLESS of quality
                                    # (cleanest feed, optical SNR 42.6: jitter 4.1 =
                                    # worst; antenna SNR 24.8: 1.9), because the pilot-
                                    # SNR noise reference bands (16-17.5k / 20.5-22k)
                                    # catch fluctuating PROGRAMME spill from the
                                    # station's 15 kHz and 23 kHz band edges - the term
                                    # measures programme dynamics, not reception.  The
                                    # mechanism stays in place (set <1.0 to re-enable).
STEREO_BLEND_SMOOTHING = 0.08       # EMA rate for the blend factor while it CLOSES
                                    # (per 16 ms reference block; time-normalised to the
                                    # actual block, see _demodulate_stereo).  This is the
                                    # GRADUAL closing path only - a real dropout is
                                    # handled by the latched fast-close and is unaffected
                                    # by this constant.
STEREO_BLEND_SMOOTHING_OPEN = 0.04  # EMA rate while the blend OPENS.  Deliberately
                                    # slower than the closing rate: widening the stereo
                                    # image is the direction a listener notices, and with
                                    # a symmetric EMA an intermittently degraded stream
                                    # pumped the image - each good block pulled the blend
                                    # back up before the next bad one pushed it down
                                    # (measured +-0.14 per block on a 3-bad/1-good
                                    # pattern at light's real block size).  Slowing BOTH
                                    # directions also fixes that, but costs the same
                                    # again on every legitimate recovery (blend > 0.9
                                    # went 0.52 s -> 0.98 s at half the rate) and lets
                                    # more false side through a pilot-less tune-in's
                                    # settle window (side/mid peak 0.54 -> 0.65).  That
                                    # last one is worth being precise about: during the
                                    # fast-close settle guard the blend comes down on
                                    # the ORDINARY closing EMA, so a pilot-less tune-in
                                    # is not structurally independent of these constants
                                    # - it is unchanged here only because the CLOSING
                                    # rate is left at its old value.  What IS independent
                                    # of both is a real dropout, which the latched
                                    # fast-close handles on its own path.
STEREO_BLEND_DROPOUT_POWER_DROP_DB = 15.0  # Pilot-power collapse (dB below its own EMA)
                                    # that identifies a genuine pilot DROPOUT for the
                                    # blend fast-close.  A real dropout collapses the
                                    # measured pilot power by tens of dB within one
                                    # block; programme spill into the NOISE bands
                                    # (which dips the per-block SNR for several
                                    # consecutive blocks on real music) leaves the
                                    # pilot power stable.  Measured over 20 s of real
                                    # programme, the drop never exceeds +1.2 dB on
                                    # either reference capture and at either block
                                    # size (CATV max +0.19 dB / p99 +0.11, antenna
                                    # max +1.13 / p99 +0.85), so 15 dB clears the
                                    # worst real-programme excursion by ~14 dB.
STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF = 16.0  # Reference (16 ms) blocks of CONTINUOUS
                                    # sub-STEREO_BLEND_PILOT_SNR_DB_LO instantaneous
                                    # pilot SNR that identifies a SUSTAINED degradation
                                    # (~256 ms).  Third fast-close trigger, for the case
                                    # the other two miss: a noise floor that rises while
                                    # the pilot itself stays intact leaves the pilot
                                    # power flat (so the collapse test cannot fire) and
                                    # only crosses the slow SNR EMA after ~0.65 s, which
                                    # left side/mid ~0.7 for the first ~0.2 s.  Real
                                    # programme dips the instantaneous SNR below LO too
                                    # (noise-band spill), but only in bursts: measured
                                    # over 60 s of each reference capture, the longest
                                    # CONTINUOUS sub-LO run is 48 ms (CATV) / 80 ms
                                    # (optical 82.5) at the 16 ms block, and both
                                    # quantise up to 131 ms (2 blocks) at light's
                                    # 65.5 ms block; the antenna and optical-80 captures
                                    # never dip at all.  256 ms therefore clears the
                                    # worst measured burst by ~1.95x (3.2x against the
                                    # true, unquantised 80 ms) and fires on none of
                                    # them (blend floor 0.90-1.00 across all four, from
                                    # the ordinary EMA - the fast-close stays out).
                                    # Raised 12 -> 16 on codex round 6: at light's
                                    # 65.5 ms block the counter steps 4.096 at a time,
                                    # so 12 fired on the 3rd block against a measured
                                    # worst of 2 - one block of margin.  16 fires on
                                    # the 4th and costs 66 ms of closing time.
                                    # Accumulated in reference-block TIME, so it means
                                    # the same duration at either block size.
STEREO_BLEND_DROPOUT_RELEASE_REF = 8.0  # Reference (16 ms) blocks of CONTINUOUS healthy
                                    # instantaneous SNR before the fast-close latch
                                    # releases (~130 ms).  The latch and this hold cover
                                    # ALL THREE triggers - whichever fired sets it, and
                                    # the hold runs from the last block on which any of
                                    # them held (only STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF
                                    # belongs to the sustained-degradation trigger alone).
                                    # Debouncing only the
                                    # ATTACK let a single good block release the
                                    # trigger, so an intermittently degraded stream
                                    # (3 bad blocks, 1 good, at light's real block size)
                                    # fast-closed and re-opened every cycle: blend
                                    # swinging 0.01 <-> 0.26 with 37 sign flips in 5 s -
                                    # audible stereo-width pumping.  The hold makes the
                                    # mechanism behave like a normal receiver blend:
                                    # fast to mono, deliberate back to stereo (it costs
                                    # ~130 ms on a clean recovery).  Deliberately a HOLD
                                    # at the same LO threshold, not a higher release
                                    # level: a level hysteresis would strand a
                                    # legitimately mid-SNR signal (7-16 dB, where the
                                    # blend is meant to sit partially open) in mono.
STEREO_BLEND_FAST_CLOSE_SETTLE_REF = 12.0  # Reference (16 ms) blocks after a pilot-chain
                                    # (re)start before the fast-close may trigger
                                    # (~190 ms): the resampler's priming blocks read
                                    # instantaneous SNR ~ 0 even on a strong capture,
                                    # and tripping there broke bit-identity with main.
STEREO_BLEND_FAST_CLOSE_FACTOR = 0.5  # Per-16 ms-reference-block blend decay while the
                                    # three-part dropout detector holds (see
                                    # _demodulate_stereo): the pilot POWER has
                                    # collapsed >= STEREO_BLEND_DROPOUT_POWER_DROP_DB
                                    # below its own EMA, OR the instantaneous AND the
                                    # EMA pilot SNR are both under
                                    # STEREO_BLEND_PILOT_SNR_DB_LO (steady pilot-less
                                    # content), OR the instantaneous SNR has sat below
                                    # LO continuously for
                                    # STEREO_BLEND_DROPOUT_SNR_DEBOUNCE_REF reference
                                    # blocks (noise floor up, pilot intact - the case
                                    # the first two miss).  An instantaneous-SNR-only
                                    # trigger, with no debounce, was
                                    # measured and rejected: programme spill into the
                                    # NOISE bands dips the per-block SNR under LO for
                                    # several consecutive blocks on real music, which
                                    # walked the blend to 0.12 on the CATV reference.
                                    # Why a fast path at all: the regular smoothed EMA
                                    # lagged for seconds on the light variant's 65.5 ms
                                    # blocks (blend 0.716 at 0.26 s of a pilot-less
                                    # cold start, 2.4-3.6 s to close), leaving audible
                                    # false side driven by programme leakage.  Halving
                                    # per reference block closes 1.0 -> <0.05 in
                                    # ~80 ms; a healthy pilot trips none of the three, so
                                    # valid streams are untouched.  All blend/SNR EMAs
                                    # are additionally time-normalised to the 16 ms
                                    # reference block, so light's real block size
                                    # gets identical time constants to standard.

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
# Side NR adaptation gate on the stereo blend.  The NR input is the
# POST-blend side, so at low blend it is an attenuated (or, at
# blend = 0, exactly ZERO) copy of the genuine side: TRAINING there
# poisons the learned floor - zero is an ABSORBING state for the
# minimum tracker (floor = min(floor*decay, 0) = 0 forever; measured:
# 4 s of blend-0 stereo left the floor at exactly 0 with gain pinned
# at 1.0 permanently), and a small-blend initialisation sits
# ~20*log10(blend) dB low.  Below the gate the NR runs in FREEZE mode
# (adapt=False, bypass=False): the floor stops updating but the gain
# computation keeps running against it, so a trained reducer keeps
# suppressing CONTINUOUSLY through blend dips (a unity bypass here
# measured a +6.5 dB side-noise step exactly when reception
# degrades); an untrained reducer outputs unity until the gate opens.
#
# Threshold choice (PR #32: light's pilot SNR is fixed, so the
# PR #31-era rationale - light's blend saturating at 0.313 and
# staying there - no longer exists and the original thresholds are
# restored).  An untrained initialisation at forced blend 0.5
# measures -9.1 dB vs the blend-1 steady floor (-34.1 vs -25.0 dB;
# healed by the 6 dB/s upward leak in ~1.5 s); in the blend-step
# case (0 -> 1) the gate
# opens at full blend and the floor initialises at parity
# immediately.  The OFF threshold's 0.15 hysteresis is far wider
# than the blend EMA's block-to-block jitter (flap-tested across the
# band).  With FREEZE mode below the gate the absorbing-zero failure
# is structurally impossible at any threshold; the thresholds only
# decide where the model may LEARN.
SIDE_NR_ADAPT_BLEND_ON = 0.5    # Blend at/above which NR adaptation (re)opens
SIDE_NR_ADAPT_BLEND_OFF = 0.35  # Blend at/below which NR adaptation freezes
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
