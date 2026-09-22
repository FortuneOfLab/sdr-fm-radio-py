"""The DSP settings tab: what it sends, and what it shows.

Driven against the same stand-in controller the window tests use, so
the tab meets a real ``DspSettings`` - a value the settings would
refuse is refused here too - without a receiver behind it.
"""

from __future__ import annotations

import logging
import math

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="the GUI is optional")

from dataclasses import replace                                # noqa: E402

from fm_radio.dsp_settings import MAX_MONO_DELAY_SAMPLES       # noqa: E402
from fm_radio.gui.dsp_tab import (                             # noqa: E402
    SLOTS, _BETA_MAX, _BETA_STEPS, _FRACTION_STEPS, DspTab,
)
from test_gui_window import DSP_DEFAULTS, FakeController       # noqa: E402


@pytest.fixture
def tab(qt_app):
    """A settings tab over a stand-in controller, closed afterwards."""
    built = []

    def _build(controller=None):
        controller = controller or FakeController()
        page = DspTab(controller)
        built.append(page)
        return page, controller

    yield _build
    for page in built:
        page.close()


def row(page: DspTab, field: str):
    """The row for a field, by the name the settings know it by."""
    for one in page._rows:
        if one.field == field:
            return one
    raise AssertionError("no row for %s" % field)


# ----------------------------------------------------------------------
# What it starts as
# ----------------------------------------------------------------------

def test_it_starts_from_what_the_receiver_is_running(tab):
    """Not from the constants: the two variants do not share them."""
    controller = FakeController()
    controller.dsp_settings = replace(
        DSP_DEFAULTS, side_nr_alpha_floor=0.45, side_nr_enabled=False,
        mono_delay_samples=7)
    page, _ = tab(controller)

    showing = page.showing()
    assert showing["side_nr_alpha_floor"] == pytest.approx(0.45)
    assert showing["side_nr_enabled"] is False
    assert showing["mono_delay_samples"] == 7
    assert controller.dsp_updates == [], "it wrote while it was reading"


def test_every_setting_has_a_row(tab):
    page, _ = tab()
    assert set(page.showing()) == {
        "force_blend_factor", "subcarrier_phase_offset_rad",
        "mono_delay_samples", "iq_phase_correction_enabled",
        "lr_high_max_gain", "lr_super_high_max_gain", "side_nr_enabled",
        "side_nr_alpha_floor", "side_nr_beta",
    }


def test_the_phase_is_shown_in_degrees(tab):
    """The settings keep radians; constants.py talks in degrees."""
    page, _ = tab()
    spin = row(page, "subcarrier_phase_offset_rad").widgets[0]

    assert spin.value() == pytest.approx(85.0, abs=0.05)
    assert "deg" in spin.suffix()


def test_the_delay_cannot_be_asked_for_beyond_the_limit(tab):
    """DspSettings would refuse it; the box does not offer it."""
    page, _ = tab()
    spin = row(page, "mono_delay_samples").widgets[0]

    assert spin.maximum() == MAX_MONO_DELAY_SAMPLES
    assert spin.minimum() == 0


# ----------------------------------------------------------------------
# What it sends
# ----------------------------------------------------------------------

def test_moving_a_slider_asks_for_that_one_change(tab):
    """One parameter through update_dsp_settings, not the whole set.

    The whole set from a stale read is how two writers undo each
    other; the receiver's update_ does the read-modify-write itself.
    """
    page, controller = tab()
    slider = row(page, "side_nr_alpha_floor").widgets[0]

    slider.setValue(int(0.55 * _FRACTION_STEPS))

    assert controller.dsp_updates == [{"side_nr_alpha_floor": 0.55}]
    assert controller.dsp_sets == [], "it wrote the whole set for one slider"
    assert controller.dsp_settings.side_nr_alpha_floor == pytest.approx(0.55)


def test_the_strength_slider_counts_in_tenths(tab):
    """0 to 3 in steps of 0.1, and the default of 1.0 lands on one.

    With ten positions instead of thirty it did not: the default
    came back as 0.9, and no reset could reach it.
    """
    page, controller = tab()
    slider = row(page, "side_nr_beta").widgets[0]

    assert slider.maximum() == _BETA_STEPS
    assert _BETA_MAX / _BETA_STEPS == pytest.approx(0.1)

    slider.setValue(15)
    assert controller.dsp_updates[-1]["side_nr_beta"] == pytest.approx(1.5)

    row(page, "side_nr_beta").reset.click()
    assert page.showing()["side_nr_beta"] == pytest.approx(
        DSP_DEFAULTS.side_nr_beta)


def test_a_flag_asks_for_a_bool(tab):
    """Not 0 or 1: DspSettings refuses anything but a bool."""
    page, controller = tab()
    box = row(page, "side_nr_enabled").widgets[0]

    box.setChecked(False)

    assert controller.dsp_updates == [{"side_nr_enabled": False}]
    assert controller.dsp_settings.side_nr_enabled is False


def test_the_phase_box_asks_in_radians(tab):
    page, controller = tab()
    spin = row(page, "subcarrier_phase_offset_rad").widgets[0]

    spin.setValue(90.0)

    asked = controller.dsp_updates[-1]["subcarrier_phase_offset_rad"]
    assert asked == pytest.approx(math.radians(90.0))


def test_the_blend_is_none_until_it_is_forced(tab):
    """Off is not a number on the slider: it is the adaptive blend."""
    page, controller = tab()
    forced, slider = row(page, "force_blend_factor").widgets

    assert page.showing()["force_blend_factor"] is None
    assert not slider.isEnabled(), "the slider is not being listened to"

    forced.setChecked(True)
    slider.setValue(int(0.40 * _FRACTION_STEPS))

    assert slider.isEnabled()
    assert controller.dsp_settings.force_blend_factor == pytest.approx(0.40)

    forced.setChecked(False)
    assert controller.dsp_settings.force_blend_factor is None
    assert controller.dsp_updates[-1] == {"force_blend_factor": None}


def test_turning_the_force_off_does_not_lose_the_value(tab):
    """They may well turn it back on."""
    page, controller = tab()
    forced, slider = row(page, "force_blend_factor").widgets

    forced.setChecked(True)
    slider.setValue(int(0.40 * _FRACTION_STEPS))
    forced.setChecked(False)

    assert slider.value() == int(0.40 * _FRACTION_STEPS)
    forced.setChecked(True)
    assert controller.dsp_settings.force_blend_factor == pytest.approx(0.40)


# ----------------------------------------------------------------------
# Putting things back
# ----------------------------------------------------------------------

def test_one_reset_puts_one_parameter_back(tab):
    page, controller = tab()
    slider = row(page, "side_nr_alpha_floor").widgets[0]
    other = row(page, "lr_high_max_gain").widgets[0]
    slider.setValue(int(0.55 * _FRACTION_STEPS))
    other.setValue(int(0.60 * _FRACTION_STEPS))

    row(page, "side_nr_alpha_floor").reset.click()

    assert controller.dsp_settings.side_nr_alpha_floor == pytest.approx(
        DSP_DEFAULTS.side_nr_alpha_floor)
    assert controller.dsp_settings.lr_high_max_gain == pytest.approx(0.60), (
        "the reset took the other one with it")
    assert page.showing()["side_nr_alpha_floor"] == pytest.approx(
        DSP_DEFAULTS.side_nr_alpha_floor)


def test_reset_all_is_one_write_of_the_whole_set(tab):
    """Nine updates would be nine chances to be half reset.

    A block demodulated between the third and the fourth of them is
    a block under a configuration nobody asked for.
    """
    page, controller = tab()
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.55 * _FRACTION_STEPS))
    row(page, "side_nr_enabled").widgets[0].setChecked(False)
    row(page, "mono_delay_samples").widgets[0].setValue(12)
    before = len(controller.dsp_updates)

    page._reset_all.click()

    assert controller.dsp_sets == [DSP_DEFAULTS]
    assert len(controller.dsp_updates) == before, (
        "it reset them one at a time as well")
    assert controller.dsp_settings == DSP_DEFAULTS
    assert page.showing() == {
        field: getattr(DSP_DEFAULTS, field) for field in page.showing()
    }


def test_putting_a_control_back_does_not_write_twice(tab):
    """Setting a widget from code raises the same signal a hand does."""
    page, controller = tab()
    slider = row(page, "side_nr_alpha_floor").widgets[0]
    slider.setValue(int(0.55 * _FRACTION_STEPS))
    before = len(controller.dsp_updates)

    row(page, "side_nr_alpha_floor").reset.click()

    assert len(controller.dsp_updates) == before + 1, (
        "the widget's own signal asked for it again")


# ----------------------------------------------------------------------
# When it may be used
# ----------------------------------------------------------------------

def test_nothing_can_be_touched_while_the_receiver_is_busy(tab):
    page, _ = tab()

    page.set_usable(False)

    for one in page._rows:
        for widget in one.widgets:
            assert not widget.isEnabled(), one.field
        assert not one.reset.isEnabled(), one.field
    assert not page._reset_all.isEnabled()


def test_the_blend_slider_stays_out_of_reach_until_it_is_forced(tab):
    """Usable again does not mean every control is: the slider has a
    second say in it, the way the gain slider does in the window.
    """
    page, _ = tab()
    forced, slider = row(page, "force_blend_factor").widgets

    page.set_usable(False)
    page.set_usable(True)

    assert forced.isEnabled()
    assert not slider.isEnabled(), "handed back without being forced"

    forced.setChecked(True)
    page.set_usable(False)
    page.set_usable(True)
    assert slider.isEnabled()


# ----------------------------------------------------------------------
# A and B
# ----------------------------------------------------------------------

def a_slot_button(page: DspTab, which: str):
    """The radio button for a slot."""
    for button in page._slot_buttons.buttons():
        if button.text() == which:
            return button
    raise AssertionError("no button for slot %s" % which)


def test_it_starts_on_a_with_b_at_the_defaults(tab):
    """The first comparison anyone wants needs nothing set up."""
    controller = FakeController()
    controller.dsp_settings = replace(DSP_DEFAULTS, side_nr_alpha_floor=0.45)
    page, _ = tab(controller)

    assert page.slot() == "A"
    assert page.showing()["side_nr_alpha_floor"] == pytest.approx(0.45)
    assert page._slots["B"] == DSP_DEFAULTS
    assert set(SLOTS) == {"A", "B"}


def test_switching_slots_writes_the_whole_set_once(tab):
    """A switch that arrived in pieces would have a block of neither."""
    page, controller = tab()
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.55 * _FRACTION_STEPS))
    updates_before = len(controller.dsp_updates)

    a_slot_button(page, "B").setChecked(True)

    assert page.slot() == "B"
    assert controller.dsp_sets == [DSP_DEFAULTS]
    assert len(controller.dsp_updates) == updates_before, (
        "it switched one parameter at a time")
    assert page.showing()["side_nr_alpha_floor"] == pytest.approx(
        DSP_DEFAULTS.side_nr_alpha_floor)


def test_the_slots_keep_their_own_settings(tab):
    """Editing B does not touch A, and A comes back as it was."""
    page, controller = tab()
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.55 * _FRACTION_STEPS))          # in A

    a_slot_button(page, "B").setChecked(True)
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.20 * _FRACTION_STEPS))          # in B

    assert page._slots["A"].side_nr_alpha_floor == pytest.approx(0.55)
    assert page._slots["B"].side_nr_alpha_floor == pytest.approx(0.20)

    a_slot_button(page, "A").setChecked(True)

    assert page.showing()["side_nr_alpha_floor"] == pytest.approx(0.55)
    assert controller.dsp_settings.side_nr_alpha_floor == pytest.approx(0.55)
    assert controller.dsp_sets[-1] == page._slots["A"]


def test_switching_does_not_write_the_controls_back(tab):
    """Showing a slot raises every widget's signal; none is a change."""
    page, controller = tab()
    before = len(controller.dsp_updates)

    a_slot_button(page, "B").setChecked(True)
    a_slot_button(page, "A").setChecked(True)

    assert len(controller.dsp_updates) == before, (
        "the controls wrote themselves back as the slot was shown")
    assert len(controller.dsp_sets) == 2


def test_a_switch_says_in_the_log_which_slot_and_what_is_in_it(tab, caplog):
    """The question a listening test ends on is "which one was that?"."""
    page, _ = tab()
    row(page, "side_nr_beta").widgets[0].setValue(15)      # 1.5 in A

    with caplog.at_level(logging.INFO, logger="fm_receiver.gui"):
        a_slot_button(page, "B").setChecked(True)
        a_slot_button(page, "A").setChecked(True)

    said = [r.getMessage() for r in caplog.records if "DSP slot" in r.getMessage()]
    assert len(said) == 2
    assert said[0].startswith("DSP slot B:")
    assert "side_nr_beta=1.0" in said[0], said[0]
    assert said[1].startswith("DSP slot A:")
    assert "side_nr_beta=1.5" in said[1], said[1]
    assert "subcarrier_phase_offset_rad=" in said[1], (
        "the line has to say the whole set, not only what changed")


def test_reset_all_leaves_the_other_slot_alone(tab):
    """It is the other half of the comparison being made."""
    page, _ = tab()
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.55 * _FRACTION_STEPS))                      # A
    a_slot_button(page, "B").setChecked(True)
    row(page, "side_nr_alpha_floor").widgets[0].setValue(
        int(0.20 * _FRACTION_STEPS))                      # B

    page._reset_all.click()

    assert page._slots["B"] == DSP_DEFAULTS
    assert page._slots["A"].side_nr_alpha_floor == pytest.approx(0.55), (
        "Reset all emptied the slot it was not on")


def test_the_switch_goes_dark_with_everything_else(tab):
    page, _ = tab()

    page.set_usable(False)

    for button in page._slot_buttons.buttons():
        assert not button.isEnabled()


# ----------------------------------------------------------------------
# A drag is one change
# ----------------------------------------------------------------------

class _Dragging:
    """A slider with its handle held down, as Qt reports one."""

    def __init__(self, slider) -> None:
        self.slider = slider

    def __enter__(self):
        self.slider.setSliderDown(True)
        return self.slider

    def __exit__(self, *exc):
        # Qt emits sliderReleased itself when the handle goes up.
        self.slider.setSliderDown(False)


def test_a_drag_asks_for_one_change_at_the_end_of_it(tab):
    """Every step used to be a write and a log line on the
    processing thread: one drag measured forty a second, the SDR
    queue 42 blocks deep and the audio running dry.
    """
    page, controller = tab()
    slider = row(page, "side_nr_alpha_floor").widgets[0]

    with _Dragging(slider):
        for step in range(20, 61, 5):
            slider.setValue(step)
            assert controller.dsp_updates == [], (
                "a step of the drag reached the receiver")
            assert row(page, "side_nr_alpha_floor").readout.text() == (
                "%.2f" % (step / _FRACTION_STEPS)), "the readout stopped"

    assert controller.dsp_updates == [{"side_nr_alpha_floor": 0.60}]
    assert controller.dsp_settings.side_nr_alpha_floor == pytest.approx(0.60)
    assert page._slots["A"].side_nr_alpha_floor == pytest.approx(0.60)


def test_a_click_on_the_groove_still_arrives(tab):
    """Not every move is a drag: a click moves the handle at once."""
    page, controller = tab()
    slider = row(page, "side_nr_beta").widgets[0]

    slider.setValue(15)                  # no handle held down

    assert controller.dsp_updates == [{"side_nr_beta": 1.5}]


def test_a_drag_of_the_blend_is_one_change_too(tab):
    page, controller = tab()
    forced, slider = row(page, "force_blend_factor").widgets
    forced.setChecked(True)
    before = len(controller.dsp_updates)

    with _Dragging(slider):
        for step in (10, 20, 30):
            slider.setValue(step)

    assert len(controller.dsp_updates) == before + 1
    assert controller.dsp_settings.force_blend_factor == pytest.approx(0.30)
