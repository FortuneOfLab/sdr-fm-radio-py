"""The controller's use of the station catalogue.

The point of these tests is the wiring rather than the catalogue itself: a
station file that was silently ignored has to reach the user, and the
application disables logging entirely unless ``--log`` was passed.
"""

from __future__ import annotations

import logging

import pytest

from fm_radio import stations
from fm_radio.controller import FMReceiverController


@pytest.fixture
def build_controller():
    """Build a controller on fake hardware, cleaning up whatever was built.

    A factory rather than a plain fixture: the test that checks the
    controller ignores the real user configuration has to redirect that
    configuration *before* the controller reads it, which rules out building
    one during fixture setup.
    """
    built = []

    def _build(stations_path):
        instance = FMReceiverController(light=True,
                                        stations_path=str(stations_path))
        built.append(instance)
        return instance

    yield _build
    for instance in built:
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


@pytest.fixture
def controller(build_controller, no_user_config):
    """A controller pointed at a stations.toml that does not exist."""
    return build_controller(no_user_config)


def test_catalogue_is_loaded_and_presets_are_the_shipped_ten(
        controller, legacy_preset_mhz):
    assert len(controller.get_catalogue()) > 900
    assert ([freq / 1e6 for _, freq in controller.get_stations_list()]
            == legacy_preset_mhz)


def test_current_station_names_the_default_frequency(controller):
    station = controller.current_station()
    assert station is not None and station.freq_hz == controller.get_frequency()


def test_search_reaches_the_whole_catalogue(controller):
    assert controller.search_stations("エフエム東京")      # legal name
    assert controller.stations_in_area("関東")


def test_an_explicit_path_is_used_instead_of_the_user_config(
        build_controller, monkeypatch, tmp_path, legacy_preset_mhz):
    """The catalogue tests must not depend on the developer's own presets.

    The redirect has to happen before the controller is built, and the first
    assertion has to prove the redirect would otherwise take effect — without
    it, this test passes whether or not stations_path is honoured.
    """
    theirs = tmp_path / "real-stations.toml"
    theirs.write_text("""
[[station]]
name = "自宅の局"
freq_mhz = 79.2
favorite = true
""", encoding="utf-8")
    monkeypatch.setattr(stations, "user_config_path", lambda: theirs)

    # The redirect is live: left to itself the loader picks that file up.
    assert [s.name for s in
            stations.favorites(stations.load_stations())] == ["自宅の局"]

    # The controller was given a path, so it must not consult the other one.
    instance = build_controller(tmp_path / "absent.toml")
    assert ([freq / 1e6 for _, freq in instance.get_stations_list()]
            == legacy_preset_mhz)


def test_broken_stations_file_is_reported_with_logging_disabled(
        build_controller, tmp_path, capsys):
    """--log is off by default, so a print is the only channel left."""
    path = tmp_path / "stations.toml"
    path.write_text("[[station]\nname = broken", encoding="utf-8")

    logging.disable(logging.CRITICAL)
    try:
        instance = build_controller(path)
    finally:
        logging.disable(logging.NOTSET)
    message = capsys.readouterr().err

    assert "Station list" in message
    assert str(path) in message
    # The receiver still came up with the bundled catalogue.
    assert len(instance.get_catalogue()) > 900
