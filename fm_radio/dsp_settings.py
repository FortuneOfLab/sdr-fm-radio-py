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
import numbers
from dataclasses import dataclass, fields

#: The longest mono-path delay a front end may ask for, in composite
#: samples.  The setting exists to compensate the L-R FIR bank group
#: delay, which is (STEREO_FIR_NTAPS - 1) / 2 = 160 samples, 0.83 ms
#: at the 192 kHz composite rate; this is six times that and holds a
#: 4 KiB delay line.  There has to be a limit, because the next block
#: allocates whatever was asked for: 100_000_000 samples asks numpy
#: for 400 MB, and 10**100 raises "Maximum allowed dimension
#: exceeded" on the processing thread, where an exception costs the
#: block.
MAX_MONO_DELAY_SAMPLES = 1024


def _a_number(name: str, value) -> float:
    """Return *value* as a float, or refuse what is not a number.

    float() alone is not a check: it accepts a string, and then the
    demodulator gets the string, because checking a value is not the
    same as storing it.  Everything here is stored back onto the
    frozen instance by __post_init__.  bool is a Real in Python, and
    True as a gain is a mistake worth naming rather than reading as
    1.0.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {value!r}")
    try:
        number = float(value)
    except OverflowError:
        # An int with no float to convert to: out of range, not a
        # type error, and not an ArithmeticError for a caller that
        # is catching what a bad value raises.
        raise ValueError(f"{name} is too large: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, not {value!r}")
    return number


def _within(name: str, value, low: float, high: float) -> float:
    """Return *value* as a float, having checked it is in range."""
    number = _a_number(name, value)
    if not low <= number <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}, not {number}")
    return number


def _a_flag(name: str, value) -> bool:
    """Return *value* as a bool, or refuse what is not one.

    Strictly bool, because the near misses are all truthy: the string
    "false", the string "off", a "0.0" read out of a text field.  A
    flag that silently reads as its opposite is worse than one that
    raises.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, not {value!r}")
    return value


def _whole(name: str, value, low: int, high: int) -> int:
    """Return *value* as a whole number in range, or refuse it."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a whole number, not {value!r}")
    if not isinstance(value, numbers.Integral):
        # Anything real that is not an integer type - float, numpy
        # float, Fraction - goes through float() first, because
        # int() on a nan or an infinity raises something else
        # entirely (ValueError "cannot convert", OverflowError) and
        # both of those are values to refuse, not to crash on.  An
        # integer type skips it: float() is what overflows on an int
        # of a few hundred digits, and the range check below has an
        # answer for that one.
        as_float = float(value)
        if not math.isfinite(as_float) or int(as_float) != as_float:
            raise ValueError(
                f"{name} must be a whole number of samples, not {value!r}")
        value = int(as_float)
    number = int(value)
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
        slice, an enormous one an allocation, and a negative
        ``side_nr_beta`` a gain that changes sign.
        ``SideNoiseReducer`` clamps beta in its constructor, and the
        constructor is not on this path.

        Each value is checked AND STORED BACK, so what reaches the
        demodulator is the checked number and not whatever was handed
        in: ``"1.0"`` passes float() and is still a string, and
        ``iq_phase_correction_enabled="false"`` is not False.
        """
        checked = {
            "force_blend_factor": (
                None if self.force_blend_factor is None
                else _within("force_blend_factor",
                             self.force_blend_factor, 0.0, 1.0)),
            "subcarrier_phase_offset_rad": _a_number(
                "subcarrier_phase_offset_rad",
                self.subcarrier_phase_offset_rad),
            "mono_delay_samples": _whole(
                "mono_delay_samples", self.mono_delay_samples,
                0, MAX_MONO_DELAY_SAMPLES),
            "iq_phase_correction_enabled": _a_flag(
                "iq_phase_correction_enabled",
                self.iq_phase_correction_enabled),
            "lr_high_max_gain": _within(
                "lr_high_max_gain", self.lr_high_max_gain, 0.0, 1.0),
            "lr_super_high_max_gain": _within(
                "lr_super_high_max_gain",
                self.lr_super_high_max_gain, 0.0, 1.0),
            "side_nr_enabled": _a_flag(
                "side_nr_enabled", self.side_nr_enabled),
            "side_nr_alpha_floor": _within(
                "side_nr_alpha_floor", self.side_nr_alpha_floor, 0.0, 1.0),
            "side_nr_beta": _within(
                "side_nr_beta", self.side_nr_beta, 0.0, math.inf),
        }
        # All nine, so that a field added later cannot quietly go
        # unchecked - and unnormalised.
        assert checked.keys() == {f.name for f in fields(self)}
        for name, value in checked.items():
            object.__setattr__(self, name, value)

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
