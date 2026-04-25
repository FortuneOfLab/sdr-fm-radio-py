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
SDR_QUEUE_MAXSIZE = 20              # Max queued SDR sample blocks

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
LR_HIGH_MIN_GAIN = 0.40             # Minimum mid-high gain at low stereo blend
LR_SUPER_HIGH_MIN_GAIN = 0.20       # Minimum super-high gain at low stereo blend
LR_HIGH_GATE_THRESHOLD = 0.0028     # RMS threshold for opening L-R high-band gate
LR_HIGH_GATE_KNEE_MULT = 2.2        # Full-open level as multiple of threshold
LR_HIGH_GATE_MIN_GAIN = 0.75        # Minimum gate gain when below threshold
LR_HIGH_GATE_SMOOTHING = 0.20       # EMA smoothing for high-band gate gain
STEREO_HIGH_GATE_SNR_ASSIST_ENABLE = True  # Boost high-gate opening only when pilot SNR is good
STEREO_HIGH_GATE_SNR_ASSIST_DB_LO = 12.0   # Assist starts above this pilot SNR (dB)
STEREO_HIGH_GATE_SNR_ASSIST_DB_HI = 20.0   # Assist reaches max above this pilot SNR (dB)
STEREO_HIGH_GATE_SNR_ASSIST_MAX = 0.35     # Max pull of gate_target toward 1.0 (0-1)
STEREO_HIGH_GATE_SNR_FLOOR_BOOST_MAX = 0.20  # Max extra floor for gate_target at high pilot SNR

PILOT_BANDPASS_ORDER = 9            # Pilot bandpass filter order (standard)
PILOT_BANDPASS_ORDER_LIGHT = 1      # Pilot bandpass filter order (light)
PILOT_BANDPASS_LOW = 18000.0        # Pilot bandpass lower edge (Hz)
PILOT_BANDPASS_HIGH = 20000.0       # Pilot bandpass upper edge (Hz)
PILOT_NOISE_BAND1_LOW = 16000.0     # Pilot SNR noise band 1 lower edge (Hz)
PILOT_NOISE_BAND1_HIGH = 17500.0    # Pilot SNR noise band 1 upper edge (Hz)
PILOT_NOISE_BAND2_LOW = 20500.0     # Pilot SNR noise band 2 lower edge (Hz)
PILOT_NOISE_BAND2_HIGH = 22000.0    # Pilot SNR noise band 2 upper edge (Hz)
STEREO_PILOT_RESIDUAL_CENTER_HZ = 19000.0  # Center frequency used by residual pilot tracking
STEREO_SUBCARRIER_PHASE_OFFSET_DEG = 300.0  # Fixed phase offset for 38k subcarrier generation
STEREO_MONO_DELAY_SAMPLES = 18      # Delay mono path to match LR path group delay (at COMPOSITE_RATE)
STEREO_LR_SIDE_RATIO_CAP_ENABLE = False     # Enable limiting of |L-R|/|L+R| ratio for stability
STEREO_LR_SIDE_RATIO_CAP_TARGET = 0.35     # Target upper bound of |L-R|/|L+R| before limiting
STEREO_LR_SIDE_RATIO_CAP_MIN_GAIN = 0.35   # Lower bound of side-cap gain to avoid mono-collapse
STEREO_LR_SIDE_RATIO_CAP_ATTACK = 0.25     # Gain attack speed when limiting engages
STEREO_LR_SIDE_RATIO_CAP_RELEASE = 0.45    # Gain release speed when limiting disengages
STEREO_PHASE_ERR_SMOOTHING = 0.15   # EMA smoothing for LR demod phase correction
STEREO_PHASE_ERR_LIMIT_DEG = 45.0   # Clamp limit for LR demod phase correction (deg)
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
RECORD_SAMPLE_WIDTH = 2             # 16-bit PCM sample width (bytes)
RECORD_MAX_INT16 = 32767            # Max value for int16 conversion

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
