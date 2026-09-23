"""A stereo FM capture written the way the receiver writes one, for tests.

16-bit I/Q in a two-channel WAV at 1.024 Msps: a 1 kHz tone, louder on
the left, with a pilot, 1237 Hz off the tuned frequency (off the DC
blocker - see test_e2e_quality) and noise at a CNR the pilot survives,
so that the blend, the pilot SNR and the noise floor all have something
to measure.
"""

import wave

import numpy as np

from fm_radio.quality_selftest import _apply_channel, _synthesize_iq_tone

RATE = 1_024_000


def write_stereo_iq_wav(path, seconds: float = 3.0, *, rate: int = RATE,
                        seed: int = 7) -> str:
    iq = _synthesize_iq_tone(
        seconds, RATE, 1000.0, left_amp=0.6, right_amp=0.15,
        pilot_amp=0.09, freq_dev_hz=75_000.0)
    np.random.seed(seed)
    iq = _apply_channel(iq, RATE, 35.0, carrier_offset_hz=1237.0)
    frames = np.empty((iq.size, 2), dtype=np.int16)
    frames[:, 0] = np.clip(iq.real * 16000, -32768, 32767)
    frames[:, 1] = np.clip(iq.imag * 16000, -32768, 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        # The header's rate is what is being written, not what was
        # synthesised: a test of a mislabelled file says a different one.
        w.setframerate(rate)
        w.writeframes(frames.tobytes())
    return str(path)
