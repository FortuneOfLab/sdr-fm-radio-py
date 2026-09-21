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
"""The DSP parameters a front end is allowed to change while it runs.

Nine of them.  They are not a new idea: every one is already a
per-instance attribute that ``quality_selftest._run_demod_diag_iq``
overrides to run an experiment offline, which is how each of the
numbers in ``constants.py`` was arrived at in the first place.  What is
new is doing it to the demodulator that is playing, from a window,
without stopping the radio.

Two rules hold the rest of this module together.

**The defaults come from the demodulator, not from this module.**  A
:class:`DspSettings` has no default values of its own and cannot be
constructed without all nine.  ``capture(demod)`` is the only way to
learn what "unchanged" means, because it is not the same in both
variants: the standard and the light chain have their own subcarrier
phase offsets (``STEREO_SUBCARRIER_PHASE_OFFSET_DEG`` against
``..._DEG_LIGHT``, plus the hardware trim), so a reset that went to a
single constant would put the light chain 0.7 degrees off and call it
the default.

**A settings object is immutable, so changing one is one rebinding.**
The processing thread reads the wanted object once per block and
applies it whole; it never sees a set half written.  See
``FMReceiverController._take_any_dsp_change``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(name: str, value: float) -> float:
    """Return *value* as a float, or say which field was not a number."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, not {value!r}")
    return number


def _within(name: str, value: float, low: float, high: float) -> float:
    """Return *value* as a float, having checked it is in range."""
    number = _finite(name, value)
    if not low <= number <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}, not {number}")
    return number


@dataclass(frozen=True)
class DspSettings:
    """What nine demodulator attributes are to be set to.

    Frozen, and with no field defaults on purpose: see the module
    docstring.  Build one from ``capture(demod)`` and
    ``dataclasses.replace`` it.

    The phase offset is kept in radians, the unit the demodulator
    holds it in, so that ``capture`` after ``apply`` returns exactly
    what was applied.  :attr:`subcarrier_phase_offset_deg` is the same
    number in the unit ``constants.py`` states it in.
    """

    #: Blend to hold the stereo/mono crossfade at, or None to let the
    #: adaptive blend decide.  Overrides what the blend bar displays.
    force_blend_factor: float | None
    #: Stereo subcarrier phase, radians, hardware trim included.
    subcarrier_phase_offset_rad: float
    #: Mono-path delay compensation, in composite samples.
    mono_delay_samples: int
    #: I/Q rotation correction in the L-R demodulator.
    iq_phase_correction_enabled: bool
    #: Ceiling on the 7-12 kHz L-R band gain.
    lr_high_max_gain: float
    #: Ceiling on the 12-15 kHz L-R band gain.
    lr_super_high_max_gain: float
    #: Side-channel STFT noise reduction, on or off.
    side_nr_enabled: bool
    #: Its minimum Wiener gain (linear): 1.0 attenuates nothing.
    side_nr_alpha_floor: float
    #: Its over-subtraction factor (1.0 = pure Wiener).
    side_nr_beta: float

    def __post_init__(self) -> None:
        """Refuse a value the demodulator would take and misbehave on.

        The demodulator defends itself against some of these at the
        point of use - it clips the forced blend into 0..1 and lifts a
        band ceiling that has fallen below its floor - but not against
        all of them: a fractional ``mono_delay_samples`` reaches a
        slice, and a negative ``side_nr_beta`` a gain that changes
        sign.  ``SideNoiseReducer`` clamps beta in its constructor and
        the constructor is not on this path.
        """
        blend = self.force_blend_factor
        if blend is not None:
            _within("force_blend_factor", blend, 0.0, 1.0)
        _finite("subcarrier_phase_offset_rad", self.subcarrier_phase_offset_rad)
        delay = self.mono_delay_samples
        if int(delay) != delay or delay < 0:
            raise ValueError(
                f"mono_delay_samples must be a whole number of samples, "
                f"not {delay!r}")
        _within("lr_high_max_gain", self.lr_high_max_gain, 0.0, 1.0)
        _within("lr_super_high_max_gain", self.lr_super_high_max_gain, 0.0, 1.0)
        _within("side_nr_alpha_floor", self.side_nr_alpha_floor, 0.0, 1.0)
        if _finite("side_nr_beta", self.side_nr_beta) < 0.0:
            raise ValueError(
                f"side_nr_beta must not be negative, not {self.side_nr_beta}")

    @property
    def subcarrier_phase_offset_deg(self) -> float:
        """The subcarrier phase offset in degrees, for a front end."""
        return math.degrees(self.subcarrier_phase_offset_rad)


def capture(demod) -> DspSettings:
    """Read the nine settings a demodulator is running under.

    Called on a freshly built demodulator, this is what its variant's
    defaults are - which is what a reset goes back to.
    """
    return DspSettings(
        force_blend_factor=(
            None if demod.force_blend_factor is None
            else float(demod.force_blend_factor)),
        subcarrier_phase_offset_rad=float(demod.subcarrier_phase_offset_rad),
        mono_delay_samples=int(demod.mono_delay_samples),
        iq_phase_correction_enabled=bool(demod.iq_phase_correction_enabled),
        lr_high_max_gain=float(demod.lr_high_max_gain),
        lr_super_high_max_gain=float(demod.lr_super_high_max_gain),
        side_nr_enabled=bool(demod.side_nr_enabled),
        side_nr_alpha_floor=float(demod.side_nr.alpha_floor),
        side_nr_beta=float(demod.side_nr.beta),
    )


def apply(settings: DspSettings, demod) -> None:
    """Write the nine settings onto a demodulator.

    Only public attributes are written.  The one that owns streaming
    state, ``mono_delay_samples``, needs nothing done to it here:
    ``_apply_mono_delay`` notices its delay line is the wrong length
    and starts a new one, which is the same thing this could do and is
    done at the point the state is used rather than from another
    thread.

    Call it between blocks.  Written attribute by attribute, it is not
    atomic, and a demodulator that read half of it would be running a
    chain nobody configured.
    """
    demod.force_blend_factor = settings.force_blend_factor
    demod.subcarrier_phase_offset_rad = settings.subcarrier_phase_offset_rad
    demod.mono_delay_samples = settings.mono_delay_samples
    demod.iq_phase_correction_enabled = settings.iq_phase_correction_enabled
    demod.lr_high_max_gain = settings.lr_high_max_gain
    demod.lr_super_high_max_gain = settings.lr_super_high_max_gain
    demod.side_nr_enabled = settings.side_nr_enabled
    demod.side_nr.alpha_floor = settings.side_nr_alpha_floor
    demod.side_nr.beta = settings.side_nr_beta
