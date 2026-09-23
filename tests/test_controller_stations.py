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


@pytest.mark.skipif(stations.tomllib is None,
                    reason="tomllib requires Python 3.11+")
def test_an_explicit_path_is_used_instead_of_the_user_config(
        build_controller, monkeypatch, tmp_path, legacy_preset_mhz):
    """The catalogue tests must not depend on the developer's own presets.

    The redirect has to happen before the controller is built, and the first
    assertion has to prove the redirect would otherwise take effect — without
    it, this test passes whether or not stations_path is honoured.

    Skipped without tomllib: the decoy configuration is a TOML file, so on
    3.9/3.10 the loader correctly falls back to the bundled catalogue and
    there is nothing for the redirect to prove.  The rest of this module does
    not depend on the user layer and still runs there.
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


needs_tomllib = pytest.mark.skipif(
    stations.tomllib is None, reason="tomllib requires Python 3.11+")


@needs_tomllib
def test_the_area_it_names_by_is_the_one_in_the_file(build_controller,
                                                     tmp_path):
    path = tmp_path / "stations.toml"
    path.write_text('area = "北海道"\n', encoding="utf-8")

    instance = build_controller(path)

    assert instance.area == "北海道"


@needs_tomllib
def test_settling_on_an_area_writes_it_and_names_by_it_now(build_controller,
                                                            no_user_config):
    """Either alone is half an answer: a setting only in the file
    does nothing until the next start, and one only in the receiver
    is gone by then.
    """
    instance = build_controller(no_user_config)
    assert instance.area is None
    # 80.0 MHz is TOKYO FM to a receiver that does not know where it
    # is, and nothing at all to one in Hokkaido: no transmitter there
    # is on that channel.  Which is the point of the setting.
    was = instance.current_station()
    assert was is not None and was.area == "関東"

    written = instance.remember_where_this_is("北海道")

    assert instance.area == "北海道"
    assert stations.home_area(written) == "北海道"
    assert instance.current_station() is None, "it is still naming Tokyo"


@needs_tomllib
def test_a_recording_is_named_as_the_dial_would_name_it(build_controller,
                                                        no_user_config):
    """stations_at is current_station for frequencies the tuner is not on.

    The recordings tab names each sidecar's frequency with it.  A name
    worked out from anything but the dial's own list - the whole
    catalogue, say - would call one frequency two things, and would
    name a Tokyo station in Hokkaido.
    """
    instance = build_controller(no_user_config)
    at = instance.get_frequency()
    assert at != 81.3e6

    names = instance.stations_at([81.3e6, at])

    assert names[81.3e6].name == "J-WAVE"
    assert names[at] == instance.current_station()

    instance.remember_where_this_is("\u5317\u6d77\u9053")

    assert instance.stations_at([81.3e6]) == {81.3e6: None}, (
        "it is still naming Tokyo")


@needs_tomllib
def test_one_call_names_every_frequency_from_one_view(build_controller,
                                                      no_user_config,
                                                      monkeypatch):
    """An area settled on half way through does not split the answer.

    The area is changed from inside the first lookup, which is the
    worst moment for it: a call that went back to the receiver's list
    for each frequency would name 80.0 by Kanto and 81.3 by Hokkaido,
    where nothing is on 81.3.
    """
    from fm_radio import controller as module

    instance = build_controller(no_user_config)
    hokkaido = module.here(instance.catalogue, "\u5317\u6d77\u9053")
    real = module.nearest
    moved = []

    def nearest_then_the_area_changes(stations, freq_hz):
        found = real(stations, freq_hz)
        if not moved:
            moved.append(True)
            instance._here = hokkaido
        return found

    monkeypatch.setattr(module, "nearest", nearest_then_the_area_changes)

    names = instance.stations_at([80.0e6, 81.3e6])

    assert moved, "the area never changed"
    assert names[80.0e6].name == "TOKYO FM"
    assert names[81.3e6] is not None and names[81.3e6].name == "J-WAVE"


@needs_tomllib
def test_a_name_worked_out_before_does_not_outlive_the_area(
        build_controller, no_user_config):
    """The snapshot does not name through current_station().

    It goes through _station_name_for, which keeps the last answer
    because walking 983 transmitters is most of what a snapshot
    costs.  Keyed on the frequency alone, that answer outlived the
    list it was worked out from - and a sweep puts the receiver back
    on the frequency it started on, which is exactly the frequency
    the cache has an answer for.  The window would go on showing
    TOKYO FM in Hokkaido until the user tuned away.

    The private call is the point: it is the processing thread's
    path, and the public one is a snapshot built on that thread.
    """
    instance = build_controller(no_user_config)
    at = instance.get_frequency()
    assert instance._station_name_for(at) == "TOKYO FM"

    instance.remember_where_this_is("\u5317\u6d77\u9053")

    assert instance._station_name_for(at) == "", \
        "the name outlived the area it was worked out under"


@needs_tomllib
def test_the_name_is_still_only_worked_out_once(build_controller,
                                                 no_user_config,
                                                 monkeypatch):
    """The generation is part of the key, not a way round the cache.

    Naming is ~270 us over the whole catalogue and happens on the
    processing thread; a cache that misses every block would put
    that in the audio path 60 times a second.
    """
    from fm_radio import controller as module

    instance = build_controller(no_user_config)
    at = instance.get_frequency()
    walks = []
    real = module.nearest
    monkeypatch.setattr(module, "nearest",
                        lambda stations, freq: walks.append(freq) or real(
                            stations, freq))

    for _ in range(5):
        instance._station_name_for(at)
    assert len(walks) == 1, "it walked the catalogue %d times" % len(walks)

    instance.remember_where_this_is("\u95a2\u6771")
    for _ in range(5):
        instance._station_name_for(at)

    assert len(walks) == 2, "it walked the catalogue %d times" % len(walks)


@needs_tomllib
def test_it_writes_to_the_file_it_was_given(build_controller, monkeypatch,
                                            tmp_path):
    """The receiver was pointed at a file; so is this."""
    theirs = tmp_path / "somebody-elses.toml"
    monkeypatch.setattr(stations, "user_config_path", lambda: theirs)
    mine = tmp_path / "mine.toml"

    instance = build_controller(mine)
    written = instance.remember_where_this_is("関東")

    assert written == mine
    assert not theirs.exists(), "it wrote to the user's own file"


@needs_tomllib
def test_a_file_it_cannot_change_leaves_the_naming_where_it_was(
        build_controller, tmp_path, capsys):
    """A receiver naming stations by a setting the file does not
    have would be telling the user their next start keeps this.
    """
    path = tmp_path / "stations.toml"
    had = "[[station\nname = broken\n"
    path.write_text(had, encoding="utf-8")
    instance = build_controller(path)
    capsys.readouterr()                 # the parse failure, already said

    was = instance.current_station()

    with pytest.raises(stations.WillNotEdit):
        # Hokkaido, because narrowing to it would show: no
        # transmitter there is on the frequency this starts at.
        instance.remember_where_this_is("北海道")

    assert instance.area is None
    assert path.read_text(encoding="utf-8") == had
    assert instance.current_station() is was, "it narrowed anyway"


@needs_tomllib
def test_an_area_that_is_not_one_is_refused_before_anything_is_written(
        build_controller, no_user_config):
    """The loader would refuse it at the next start, which is a
    receiver that names nothing and a file the user has to find.
    """
    instance = build_controller(no_user_config)

    with pytest.raises(ValueError):
        instance.remember_where_this_is("Kanto")

    assert instance.area is None
    assert not no_user_config.exists()


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
