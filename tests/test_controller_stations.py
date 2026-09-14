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
def controller(no_user_config):
    """A controller on fake hardware, cleaned up afterwards.

    Pointed at a stations.toml that does not exist, so the test never reads
    the developer's own favourites.
    """
    instance = FMReceiverController(light=True,
                                    stations_path=str(no_user_config))
    try:
        yield instance
    finally:
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


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


def test_broken_stations_file_is_reported_with_logging_disabled(
        tmp_path, capsys):
    """--log is off by default, so a print is the only channel left."""
    path = tmp_path / "stations.toml"
    path.write_text("[[station]\nname = broken", encoding="utf-8")

    logging.disable(logging.CRITICAL)
    try:
        instance = FMReceiverController(light=True, stations_path=str(path))
    finally:
        logging.disable(logging.NOTSET)
    try:
        message = capsys.readouterr().err
    finally:
        instance.auto_gain.stop()
        instance.audio_output.cleanup()

    assert "Station list" in message
    assert str(path) in message
    # The receiver still came up with the bundled catalogue.
    assert len(instance.get_catalogue()) > 900


def test_the_fixture_does_not_read_the_real_user_config(monkeypatch, tmp_path,
                                                        controller,
                                                        legacy_preset_mhz):
    """The controller fixture passes an explicit absent path, so a developer
    with their own favourites still sees the shipped presets here."""
    theirs = tmp_path / "real-stations.toml"
    theirs.write_text("[[station]]\nname='自宅'\nfreq_mhz=79.2\nfavorite=true\n",
                      encoding="utf-8")
    monkeypatch.setattr(stations, "user_config_path", lambda: theirs)

    assert ([freq / 1e6 for _, freq in controller.get_stations_list()]
            == legacy_preset_mhz)
