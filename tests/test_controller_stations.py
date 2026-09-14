"""The controller's use of the station catalogue.

The point of these tests is the wiring rather than the catalogue itself: a
station file that was silently ignored has to reach the user, and the
application disables logging entirely unless ``--log`` was passed.
"""

from __future__ import annotations

import logging

import pytest

from fm_radio.controller import FMReceiverController


@pytest.fixture
def controller(request):
    """A controller on fake hardware, cleaned up afterwards."""
    stations_path = getattr(request, "param", None)
    instance = FMReceiverController(light=True, stations_path=stations_path)
    try:
        yield instance
    finally:
        instance.auto_gain.stop()
        instance.audio_output.cleanup()


def test_catalogue_is_loaded_and_presets_are_the_shipped_ten(controller):
    assert len(controller.get_catalogue()) > 900
    assert [freq / 1e6 for _, freq in controller.get_stations_list()] == [
        78.0, 79.5, 80.0, 81.3, 82.5, 84.7, 89.7, 90.5, 91.6, 93.0]


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
