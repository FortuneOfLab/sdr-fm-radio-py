"""Changing the DSP while the radio is playing.

Two things are being checked here.  That a settings object says what
the demodulator is running under - all nine parameters, including the
two that live on the noise reducer rather than on the demodulator -
and that a change asked for from another thread reaches it between
two blocks and never inside one.

The processing loop is the real one: IQ blocks go on the SDR queue and
``FMReceiverController.processing_thread`` takes them off.  The
demodulator's two entry points are wrapped so that each block is
asked, twice, what it is running under.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace

import numpy as np
import pytest

from fm_radio.constants import (
    HARDWARE_SUBCARRIER_PHASE_TRIM_DEG,
    LR_HIGH_MAX_GAIN,
    LR_SUPER_HIGH_MAX_GAIN,
    SIDE_NR_ALPHA_FLOOR,
    SIDE_NR_BETA,
    SIDE_NR_ENABLE,
    STEREO_IQ_PHASE_CORRECTION_ENABLE,
    STEREO_MONO_DELAY_SAMPLES,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG,
    STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT,
)
from fm_radio.controller import FMReceiverController
from fm_radio.demodulator import FMDemodulator, FMDemodulatorLight
from fm_radio.dsp_settings import DspSettings, apply, capture


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

@pytest.fixture
def receiver(no_user_config):
    """A controller on fake hardware whose SDR thread is never started."""
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.quit_event.set()
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def a_settings() -> DspSettings:
    """One of everything, none of it a default."""
    return DspSettings(
        force_blend_factor=0.25,
        subcarrier_phase_offset_rad=math.radians(83.5),
        mono_delay_samples=3,
        iq_phase_correction_enabled=False,
        lr_high_max_gain=0.85,
        lr_super_high_max_gain=0.50,
        side_nr_enabled=False,
        side_nr_alpha_floor=0.12,
        side_nr_beta=1.70,
    )


def feed(controller, blocks: int) -> None:
    """Put *blocks* IQ blocks on the queue, stamped as the SDR would."""
    size = controller.sdr_receiver.block_size
    rng = np.random.default_rng(0)
    block = (0.3 * (rng.standard_normal(size)
                    + 1j * rng.standard_normal(size))).astype(np.complex64)
    for _ in range(blocks):
        controller.sdr_receiver.data_queue.put(
            (controller.sdr_receiver.tuning_generation, block))


def watch(controller, monkeypatch, blocks_wanted: int, during_a_block=None):
    """Record what the demodulator is running under, twice per block.

    The reading is taken by :func:`capture` from the live demodulator
    at the top of each of its two entry points, so what is recorded is
    the DSP's own state and not a copy of what was asked for.

    *during_a_block* is called from INSIDE the first block, between
    the reading and the work: that is the interleaving under test, and
    a call made around the block instead would leave the change
    landing between blocks, which is where it is allowed to land.

    Returns the list of ``(stage, settings)`` readings and an event set
    once *blocks_wanted* blocks have been through.
    """
    demod = controller.fm_demodulator
    seen: list[tuple[str, DspSettings]] = []
    done = threading.Event()
    real_process = demod.process_iq_samples
    real_demodulate = demod.demodulate

    def process(iq_samples):
        seen.append(("process", capture(demod)))
        if len(seen) == 1 and during_a_block is not None:
            during_a_block()
        return real_process(iq_samples)

    def demodulate(composite):
        seen.append(("demodulate", capture(demod)))
        try:
            return real_demodulate(composite)
        finally:
            if len(seen) >= 2 * blocks_wanted:
                done.set()

    monkeypatch.setattr(demod, "process_iq_samples", process)
    monkeypatch.setattr(demod, "demodulate", demodulate)
    return seen, done


def run_until(controller, done: threading.Event, timeout: float = 10.0) -> None:
    """Run the real processing loop until *done*, then stop it."""
    controller.quit_event.clear()
    thread = threading.Thread(target=controller.processing_thread, daemon=True)
    thread.start()
    try:
        assert done.wait(timeout), "the blocks never went through"
    finally:
        controller.quit_event.set()
        thread.join(timeout=timeout)
    assert not thread.is_alive(), "processing thread did not stop"


# ----------------------------------------------------------------------
# What the defaults are
# ----------------------------------------------------------------------

def test_the_standard_defaults_are_the_constants():
    settings = capture(FMDemodulator(stereo=True))
    assert settings.force_blend_factor is None
    assert settings.subcarrier_phase_offset_deg == pytest.approx(
        STEREO_SUBCARRIER_PHASE_OFFSET_DEG + HARDWARE_SUBCARRIER_PHASE_TRIM_DEG)
    assert settings.mono_delay_samples == STEREO_MONO_DELAY_SAMPLES
    assert settings.iq_phase_correction_enabled == STEREO_IQ_PHASE_CORRECTION_ENABLE
    assert settings.lr_high_max_gain == LR_HIGH_MAX_GAIN
    assert settings.lr_super_high_max_gain == LR_SUPER_HIGH_MAX_GAIN
    assert settings.side_nr_enabled == SIDE_NR_ENABLE
    assert settings.side_nr_alpha_floor == SIDE_NR_ALPHA_FLOOR
    assert settings.side_nr_beta == SIDE_NR_BETA


def test_the_light_chain_has_a_phase_default_of_its_own():
    """The reason a reset cannot go to one constant.

    Everything else the two variants share; the subcarrier phase they
    do not, and the difference is the 0.7 degrees between the two
    calibrations.
    """
    light = capture(FMDemodulatorLight(stereo=True))
    standard = capture(FMDemodulator(stereo=True))
    assert light.subcarrier_phase_offset_deg == pytest.approx(
        STEREO_SUBCARRIER_PHASE_OFFSET_DEG_LIGHT
        + HARDWARE_SUBCARRIER_PHASE_TRIM_DEG)
    assert light.subcarrier_phase_offset_rad != standard.subcarrier_phase_offset_rad
    assert replace(
        light, subcarrier_phase_offset_rad=standard.subcarrier_phase_offset_rad,
    ) == standard


def test_the_defaults_are_read_from_the_variant_that_is_running(no_user_config):
    defaults = {}
    for light in (True, False):
        controller = FMReceiverController(light=light,
                                          stations_path=str(no_user_config))
        try:
            assert controller.get_dsp_defaults() == capture(
                controller.fm_demodulator)
            assert controller.get_dsp_settings() == controller.get_dsp_defaults()
            defaults[light] = controller.get_dsp_defaults()
        finally:
            controller.quit_event.set()
            controller.auto_gain.stop()
            controller.audio_output.cleanup()
    assert (defaults[True].subcarrier_phase_offset_rad
            != defaults[False].subcarrier_phase_offset_rad), (
        "both controllers reported the same phase default")


# ----------------------------------------------------------------------
# Reading and writing the nine
# ----------------------------------------------------------------------

def test_apply_then_capture_is_what_was_applied():
    demod = FMDemodulatorLight(stereo=True)
    wanted = a_settings()
    apply(wanted, demod)
    assert capture(demod) == wanted
    # The two that are not attributes of the demodulator at all.
    assert demod.side_nr.alpha_floor == wanted.side_nr_alpha_floor
    assert demod.side_nr.beta == wanted.side_nr_beta


def test_a_new_mono_delay_takes_effect_in_the_delay_line():
    """apply() leaves the delay line to the demodulator, so check it.

    Nothing resizes ``_mono_delay_state`` when the setting changes;
    the claim is that the delay line notices for itself at the point
    it is used.
    """
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    delayed = demod._apply_mono_delay(np.arange(1, 9, dtype=np.float32))
    assert list(delayed) == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_the_phase_offset_is_offered_in_the_unit_the_constants_use():
    settings = replace(a_settings(),
                       subcarrier_phase_offset_rad=math.radians(91.0))
    assert settings.subcarrier_phase_offset_deg == pytest.approx(91.0)


@pytest.mark.parametrize("field, value", [
    ("force_blend_factor", 1.5),
    ("force_blend_factor", -0.1),
    ("force_blend_factor", float("nan")),
    ("subcarrier_phase_offset_rad", float("inf")),
    ("mono_delay_samples", -1),
    ("mono_delay_samples", 2.5),
    ("lr_high_max_gain", 1.2),
    ("lr_super_high_max_gain", -0.1),
    ("side_nr_alpha_floor", 1.1),
    ("side_nr_beta", -0.5),
])
def test_a_value_the_dsp_would_misbehave_on_is_refused(field, value):
    with pytest.raises(ValueError) as refused:
        replace(a_settings(), **{field: value})
    assert field in str(refused.value)


def test_a_forced_blend_may_be_taken_off_again():
    assert replace(a_settings(), force_blend_factor=None).force_blend_factor is None


def test_set_dsp_settings_refuses_anything_else(receiver):
    with pytest.raises(TypeError):
        receiver.set_dsp_settings({"side_nr_enabled": False})
    assert receiver.get_dsp_settings() == receiver.get_dsp_defaults()


# ----------------------------------------------------------------------
# When a change lands
# ----------------------------------------------------------------------

def test_nothing_reaches_the_demodulator_until_a_block_arrives(receiver):
    defaults = receiver.get_dsp_defaults()
    wanted = replace(defaults, side_nr_alpha_floor=0.77)
    receiver.set_dsp_settings(wanted)
    assert receiver.get_dsp_settings() == wanted, "the asking is not remembered"
    assert capture(receiver.fm_demodulator) == defaults, (
        "the DSP was written to from the calling thread")


def test_a_block_is_never_demodulated_half_under_each_set(receiver, monkeypatch):
    """The change is asked for from inside a block; the block finishes.

    Both readings of the block that asks have to be the old set, and
    both readings of the block after it the new one.  A reading that
    differs across one block is a demodulator reconfigured under its
    own feet.
    """
    defaults = receiver.get_dsp_defaults()
    wanted = replace(
        defaults,
        side_nr_alpha_floor=0.77,
        lr_high_max_gain=0.61,
        iq_phase_correction_enabled=not defaults.iq_phase_correction_enabled,
    )
    seen, done = watch(receiver, monkeypatch, blocks_wanted=2,
                       during_a_block=lambda: receiver.set_dsp_settings(wanted))
    feed(receiver, 2)
    run_until(receiver, done)

    assert [stage for stage, _ in seen] == [
        "process", "demodulate", "process", "demodulate"]
    assert seen[0][1] == defaults
    assert seen[1][1] == defaults, (
        "the second half of the block ran under the new settings")
    assert seen[2][1] == wanted, "the next block did not pick the change up"
    assert seen[3][1] == wanted


def test_the_last_of_several_changes_is_the_one_applied(receiver, monkeypatch):
    defaults = receiver.get_dsp_defaults()
    first = replace(defaults, side_nr_alpha_floor=0.40)
    last = replace(defaults, side_nr_alpha_floor=0.55, side_nr_enabled=False)
    receiver.set_dsp_settings(first)
    receiver.set_dsp_settings(last)
    seen, done = watch(receiver, monkeypatch, blocks_wanted=1)
    feed(receiver, 1)
    run_until(receiver, done)
    assert seen[0][1] == last


def test_a_forced_blend_reaches_the_demodulation(receiver, monkeypatch):
    """Not the attribute: the blend the stereo chain actually used.

    0.37 is a number the adaptive blend cannot arrive at on its own,
    and the first two blocks are run without the setting to show that
    it does not.
    """
    assert receiver.set_stereo(True)
    _, done = watch(receiver, monkeypatch, blocks_wanted=2)
    feed(receiver, 2)
    run_until(receiver, done)
    assert receiver.fm_demodulator.blend_factor != 0.37

    receiver.set_dsp_settings(
        replace(receiver.get_dsp_defaults(), force_blend_factor=0.37))
    _, done = watch(receiver, monkeypatch, blocks_wanted=2)
    feed(receiver, 2)
    run_until(receiver, done)
    assert receiver.fm_demodulator.blend_factor == 0.37
