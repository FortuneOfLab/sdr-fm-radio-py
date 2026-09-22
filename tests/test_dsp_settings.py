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

import logging
import math
import threading
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

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
from fm_radio.dsp_settings import (
    MAX_MONO_DELAY_SAMPLES, DspSettings, apply, capture,
)


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
    """The delay reaches the line, and the line is ready at once.

    ``mono_delay_samples`` is a property whose setter starts a new
    delay line, so the line is the right length as soon as the
    settings are applied - before anything reads it.
    """
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    assert demod._mono_delay_state.size == 4
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


@pytest.mark.parametrize("field, value", [
    ("force_blend_factor", "0.5"),
    ("subcarrier_phase_offset_rad", "1.0"),
    ("mono_delay_samples", True),
    ("mono_delay_samples", "4"),
    ("iq_phase_correction_enabled", "false"),
    ("iq_phase_correction_enabled", 1),
    ("side_nr_enabled", "off"),
    ("lr_high_max_gain", True),
    ("side_nr_beta", None),
])
def test_a_value_of_the_wrong_type_is_refused(field, value):
    """float() is not a check, and every near miss for a flag is truthy.

    "1.0" passes float() and is still a string when the demodulator
    multiplies by it; "false" and "off" are both True.
    """
    with pytest.raises(TypeError) as refused:
        replace(a_settings(), **{field: value})
    assert field in str(refused.value)


def test_a_checked_value_is_the_value_the_dsp_gets():
    """The check stores what it checked, so nothing else can arrive."""
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4.0,
                  lr_high_max_gain=1), demod)
    assert type(demod.mono_delay_samples) is int
    assert type(demod.lr_high_max_gain) is float
    # And the delay line is built from it: a float length raises.
    assert list(demod._apply_mono_delay(np.ones(2, dtype=np.float32))) == [0.0, 0.0]


@pytest.mark.parametrize("value, trouble", [
    (np.int64(4), None),
    (np.float32(4.0), None),
    (np.float16(4.0), None),
    (np.longdouble(4.0), None),
    (Fraction(4, 1), None),
    (np.float32(2.5), ValueError),
    (np.float32(np.inf), ValueError),
    (np.float64(np.nan), ValueError),
    # Not 1, however much float() insists that it is.
    (Fraction(2 ** 60 + 1, 2 ** 60), ValueError),
    # Whole, and far outside the range: the range must answer it,
    # not an overflow on the way there.
    (Fraction(10 ** 400, 1), ValueError),
    (np.bool_(True), TypeError),
    (Decimal(4), TypeError),
])
def test_a_real_number_is_taken_or_refused_but_never_crashes(value, trouble):
    """Whatever a caller hands in comes back as TypeError or ValueError.

    int() on an infinity raises OverflowError and float() on a large
    enough Fraction raises it too; update_dsp_settings promises
    callers the other two.
    """
    if trouble is None:
        assert replace(a_settings(),
                       mono_delay_samples=value).mono_delay_samples == 4
        return
    with pytest.raises(trouble) as refused:
        replace(a_settings(), mono_delay_samples=value)
    assert "mono_delay_samples" in str(refused.value)


def test_the_longest_mono_delay_is_bounded():
    """The next block allocates whatever was asked for."""
    at_the_limit = replace(a_settings(),
                           mono_delay_samples=MAX_MONO_DELAY_SAMPLES)
    assert at_the_limit.mono_delay_samples == MAX_MONO_DELAY_SAMPLES
    for too_much in (MAX_MONO_DELAY_SAMPLES + 1, 100_000_000, 10 ** 100):
        with pytest.raises(ValueError) as refused:
            replace(a_settings(), mono_delay_samples=too_much)
        assert "mono_delay_samples" in str(refused.value)


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


# ----------------------------------------------------------------------
# Switching a setting must not damage the stream
# ----------------------------------------------------------------------

def transparent_nr(demod):
    """Make the noise reducer pass the side channel through unchanged.

    alpha_floor 1.0 floors the Wiener gain at unity, beta 0 asks for
    no over-subtraction, so the whole tail becomes an identity - to
    within the FFT round trip, which is parts in 10^7 here - that
    holds frame - hop samples.  Anything bigger than that between
    the output and the input is the tail mishandling the stream, not
    denoising.
    """
    apply(replace(capture(demod), side_nr_alpha_floor=1.0,
                  side_nr_beta=0.0), demod)


def side_through_the_tail(demod, side_in):
    """Push one block through the tail as pure side (mid = 0)."""
    left, right = side_in, (-side_in).astype(np.float32)
    out_l, out_r = demod._apply_side_nr(left, right)
    return (0.5 * (out_l - out_r)).astype(np.float32)


def test_switching_the_noise_reducer_does_not_disturb_the_stream():
    """Off and on again, with the tail transparent: nothing moves.

    Skipping the tail while the NR is off leaves what it is holding
    inside it and takes 16 ms out of the timeline; the held samples
    then replay when it is switched back on.  Keeping the tail in
    the chain - computing, and only withholding the gain - keeps
    the latency and the sample accounting the same in both states,
    which is the fix issue #29 made for the mono/stereo switch: the
    same tail, the same reason.
    """
    rng = np.random.default_rng(3)
    blocks = [rng.standard_normal(1024).astype(np.float32) for _ in range(12)]

    def run(off_at=None, on_at=None):
        demod = FMDemodulatorLight(stereo=True)
        transparent_nr(demod)
        out = []
        for i, side_in in enumerate(blocks):
            if i == off_at or i == on_at:
                apply(replace(capture(demod), side_nr_enabled=(i == on_at),
                              side_nr_alpha_floor=1.0, side_nr_beta=0.0),
                      demod)
            out.append(side_through_the_tail(demod, side_in))
        return np.concatenate(out), demod

    fed = np.concatenate(blocks)
    steady, demod = run()
    toggled, _ = run(off_at=4, on_at=8)
    hold = demod.side_nr.frame - demod.side_nr.hop
    ramp_in = demod.side_nr.frame

    # Same accounting as a run that was never touched, and the same
    # samples in the same places.
    assert steady.size == fed.size - hold
    assert toggled.size == steady.size
    assert np.max(np.abs(steady[ramp_in:] - fed[ramp_in:steady.size])) < 1e-5, (
        "the tail is not transparent even without a switch")
    assert np.max(np.abs(toggled[ramp_in:] - fed[ramp_in:toggled.size])) < 1e-5, (
        "switching the noise reducer moved the stream")


def test_the_noise_reducer_does_not_replay_what_it_held():
    """Switch off over silence, switch on, and hear nothing.

    The reviewer's case: three blocks of hard-panned signal, then
    silence with the NR off, then the NR back on.  What it was
    holding when it was switched off must have come out while it was
    off, not on the way back in.
    """
    demod = FMDemodulatorLight(stereo=True)
    transparent_nr(demod)
    loud = np.ones(4096, dtype=np.float32)
    quiet = np.zeros(4096, dtype=np.float32)

    for _ in range(3):
        assert np.max(np.abs(side_through_the_tail(demod, loud))) > 0.5

    apply(replace(capture(demod), side_nr_enabled=False,
                  side_nr_alpha_floor=1.0, side_nr_beta=0.0), demod)
    flushed = [side_through_the_tail(demod, quiet) for _ in range(3)]
    assert np.max(np.abs(flushed[1])) == 0.0, (
        "the tail is still emptying a block after it was switched off")

    apply(replace(capture(demod), side_nr_enabled=True,
                  side_nr_alpha_floor=1.0, side_nr_beta=0.0), demod)
    back_on = side_through_the_tail(demod, quiet)
    assert np.max(np.abs(back_on)) == 0.0, (
        "audio from before the noise reducer was switched off was replayed")


def nr_ratio(off_seconds: float, keep_learning: bool) -> float:
    """Output over input RMS with the NR on, after the noise changed.

    Two seconds of quiet side noise to learn a floor, then a stretch
    twenty times louder either with the NR switched off or with it
    left on, then half a second measured with it on.  A reducer whose
    model went stale while it was off passes nearly all of the input.
    """
    demod = FMDemodulatorLight(stereo=True)
    rng = np.random.default_rng(11)
    block = 768
    rate = 48000

    def push(amplitude, seconds):
        fed, got = [], []
        for _ in range(int(seconds * rate / block)):
            side = (amplitude * rng.standard_normal(block)).astype(np.float32)
            fed.append(side)
            got.append(side_through_the_tail(demod, side))
        return np.concatenate(fed), np.concatenate(got)

    push(0.01, 2.0)
    demod.side_nr_enabled = keep_learning
    push(0.2, off_seconds)
    demod.side_nr_enabled = True
    fed, got = push(0.2, 0.5)
    n = min(fed.size, got.size)
    return float(np.sqrt(np.mean(got[:n] ** 2))
                 / np.sqrt(np.mean(fed[:n] ** 2)))


def test_a_switched_off_noise_reducer_leaves_the_audio_alone():
    """The gain is computed either way; off means it is not applied.

    The other tests here run the reducer at unity (alpha_floor 1.0)
    or throw the switched-off output away, so none of them would
    notice an NR that went on suppressing while it said it was off.
    This one watches the same stream through both states with the
    reducer at its normal settings, block by block, and compares
    each output block with the input block it answers rather than
    with an average: a switched-off reducer passes the input through
    its unity-gain FFT and overlap-add, which preserves it to within
    their numerical error (measured 1.5e-08 peak on a signal of
    amplitude 0.05), so there is nothing to average over.
    """
    demod = FMDemodulatorLight(stereo=True)
    rng = np.random.default_rng(7)
    rate, block = 48000, 768
    half_a_second = int(0.5 * rate / block)
    # The tail holds frame - hop = 768 samples, one block exactly,
    # so the block that comes out answers the one fed before it.
    held = [np.zeros(block, dtype=np.float32)]

    def run(blocks, amplitude=0.05):
        """Feed *blocks* blocks; return (output over input, worst change).

        The second number is what an exact passthrough makes zero.
        """
        fed, got = [], []
        for _ in range(blocks):
            side = (amplitude * rng.standard_normal(block)).astype(np.float32)
            out = side_through_the_tail(demod, side)
            answered, held[0] = held[0], side
            if out.size == answered.size:
                fed.append(answered)
                got.append(out)
        a, b = np.concatenate(fed), np.concatenate(got)
        return (float(np.sqrt(np.mean(b ** 2)) / np.sqrt(np.mean(a ** 2))),
                float(np.max(np.abs(b - a))))

    run(6 * half_a_second)                       # learn a floor
    suppressing, changed = run(half_a_second)
    assert suppressing < 0.9, "the reducer was not suppressing to begin with"
    assert changed > 0.01

    # Block by block across the switch.  The first block out is a
    # mixture - it carries three hops the reducer had already
    # processed - and every block after it is the input again, to
    # within the round trip.  A switch that took two or three blocks
    # to arrive would leave one of these first two reading like the
    # old state, which an average over the next half second would
    # hide.
    apply(replace(capture(demod), side_nr_enabled=False), demod)
    first, first_changed = run(1)
    assert first > suppressing + 0.05, (
        "the switch had not reached the first block after it")
    assert first_changed > 0.01, "nothing of the old setting is left in it"
    second, second_changed = run(1)
    assert second_changed < 1e-6, (
        "the second block after the switch is still being changed")
    assert second == pytest.approx(1.0, abs=1e-3)
    steady, steady_changed = run(half_a_second)
    assert steady_changed < 1e-6, (
        "a switched-off noise reducer is still changing the audio")

    # And back: again the first block is a mixture and the second is
    # already suppressing as hard as it was before.
    apply(replace(capture(demod), side_nr_enabled=True), demod)
    back_first, _ = run(1)
    assert back_first < 0.95
    back_second, back_changed = run(1)
    assert back_second < 0.8, "it did not start suppressing again"
    assert back_changed > 0.01


def test_the_noise_reducer_keeps_its_model_current_while_it_is_off():
    """Off is "computed, not applied", not "frozen".

    A bypass freezes the learned floor, so an NR switched back on
    after the noise around it changed suppresses nothing for
    seconds: 0.954 of the input passed, against 0.697 for one that
    stayed on, recovering over about 5 s.  That is no use for an A/B
    where the point is to hear the difference the moment it is
    switched.
    """
    stayed_on = nr_ratio(5.0, keep_learning=True)
    switched = nr_ratio(5.0, keep_learning=False)
    assert stayed_on < 0.8, "the reducer is not suppressing at all"
    assert switched < 0.8, (
        "an NR switched back on after 5 s off passed %.3f of the input "
        "against %.3f for one that stayed on" % (switched, stayed_on))
    assert abs(switched - stayed_on) < 0.05


def test_a_mono_delay_that_goes_through_zero_forgets_what_it_held():
    """Zero is not a pause: the line has to drop what it is carrying.

    Otherwise a return to the SAME delay finds a state array of the
    right length, keeps it, and plays samples from before the
    setting was changed.
    """
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    demod._apply_mono_delay(np.arange(1, 9, dtype=np.float32))
    assert list(demod._mono_delay_state) == [5.0, 6.0, 7.0, 8.0]

    apply(replace(capture(demod), mono_delay_samples=0), demod)
    straight = demod._apply_mono_delay(np.arange(20, 24, dtype=np.float32))
    assert list(straight) == [20.0, 21.0, 22.0, 23.0]

    apply(replace(capture(demod), mono_delay_samples=4), demod)
    after = demod._apply_mono_delay(np.arange(30, 34, dtype=np.float32))
    assert list(after) == [0.0, 0.0, 0.0, 0.0], (
        "samples from before the delay was taken off came back out")


def test_a_mono_delay_changed_during_mono_clears_the_line_then():
    """The delay line is only READ by the stereo path.

    So the clearing cannot wait until it is next read: a delay
    changed while the receiver is in mono, and changed back before
    it returns to stereo, would find its old line intact.
    """
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    demod._apply_mono_delay(np.arange(1, 9, dtype=np.float32))
    assert list(demod._mono_delay_state) == [5.0, 6.0, 7.0, 8.0]

    demod.stereo = False
    apply(replace(capture(demod), mono_delay_samples=0), demod)
    assert demod._mono_delay_state.size == 0, (
        "the line still holds the last stereo block's samples")
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    demod.stereo = True
    back = demod._apply_mono_delay(np.arange(30, 34, dtype=np.float32))
    assert list(back) == [0.0, 0.0, 0.0, 0.0], (
        "samples from before the receiver went mono came back out")


def test_changing_something_else_leaves_the_delay_line_alone():
    """Clearing on every apply would click on every unrelated change."""
    demod = FMDemodulatorLight(stereo=True)
    apply(replace(capture(demod), mono_delay_samples=4), demod)
    demod._apply_mono_delay(np.arange(1, 9, dtype=np.float32))
    held = list(demod._mono_delay_state)

    apply(replace(capture(demod), side_nr_alpha_floor=0.5), demod)
    assert list(demod._mono_delay_state) == held
    assert list(demod._apply_mono_delay(np.zeros(4, dtype=np.float32))) == held


# ----------------------------------------------------------------------
# Changing one parameter while somebody else changes another
# ----------------------------------------------------------------------

def test_update_changes_one_parameter_and_leaves_the_rest(receiver):
    before = receiver.get_dsp_settings()
    after = receiver.update_dsp_settings(side_nr_enabled=False)
    assert after.side_nr_enabled is False
    assert after == replace(before, side_nr_enabled=False)
    assert receiver.get_dsp_settings() == after


def test_update_with_a_value_the_dsp_would_misbehave_on_changes_nothing(receiver):
    before = receiver.get_dsp_settings()
    for bad in ({"side_nr_alpha_floor": 1.5}, {"side_nr_enabled": "off"},
                {"no_such_parameter": 1}):
        with pytest.raises((TypeError, ValueError)):
            receiver.update_dsp_settings(**bad)
    assert receiver.get_dsp_settings() == before


def test_two_writers_changing_different_parameters_do_not_undo_each_other(
        receiver, monkeypatch):
    """Both changes survive a read-modify-write that overlaps another.

    The overlap is forced, not hoped for: the first writer is held
    inside its own read-modify-write until the second has reached the
    lock.  That is exactly the window in which a lock-free version
    loses a change - the second writer would build its nine values
    from the set the first one is in the middle of replacing.
    """
    import fm_radio.controller as controller_module

    may_start = threading.Event()
    at_the_lock = threading.Event()
    second = {}

    real_lock = receiver._dsp_lock

    class Watched:
        """The lock, saying when the second writer arrives at it."""

        def acquire(self, *args, **kwargs):
            if threading.get_ident() == second.get("id"):
                at_the_lock.set()
            return real_lock.acquire(*args, **kwargs)

        def release(self):
            real_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *exc):
            self.release()

    monkeypatch.setattr(receiver, "_dsp_lock", Watched())

    real_replace = controller_module.dsp_replace
    first = []

    def replace_while_the_other_one_tries(settings, **changes):
        if not first:
            first.append(True)
            may_start.set()
            assert at_the_lock.wait(10), (
                "the second writer never reached the lock - it did its own "
                "read-modify-write inside the first one")
        return real_replace(settings, **changes)

    monkeypatch.setattr(controller_module, "dsp_replace",
                        replace_while_the_other_one_tries)

    def second_writer():
        second["id"] = threading.get_ident()
        may_start.wait(10)
        receiver.update_dsp_settings(side_nr_beta=2.0)

    other = threading.Thread(target=second_writer)
    other.start()
    try:
        receiver.update_dsp_settings(side_nr_alpha_floor=0.4)
    finally:
        other.join(timeout=10)
    assert not other.is_alive(), "the second writer never finished"

    wanted = receiver.get_dsp_settings()
    assert wanted.side_nr_alpha_floor == 0.4, "the first change was undone"
    assert wanted.side_nr_beta == 2.0, "the second change was undone"


def test_the_line_about_a_change_says_only_what_changed(receiver, monkeypatch,
                                                        caplog):
    """It is written on the processing thread, into a file handler.

    The whole set came to 346 characters, and a dragged slider put
    forty of those a second through the realtime path until the SDR
    queue was 42 blocks deep and the audio ran dry.
    """
    wanted = replace(receiver.get_dsp_defaults(), side_nr_alpha_floor=0.4)
    receiver.set_dsp_settings(wanted)

    with caplog.at_level(logging.INFO, logger=receiver.logger.name):
        seen, done = watch(receiver, monkeypatch, blocks_wanted=1)
        feed(receiver, 1)
        run_until(receiver, done)

    said = [r.getMessage() for r in caplog.records
            if "DSP settings changed" in r.getMessage()]
    assert len(said) == 1
    assert said[0] == "DSP settings changed: side_nr_alpha_floor=0.4", said[0]
    assert len(said[0]) < 100, "the line is long enough to hurt again"
