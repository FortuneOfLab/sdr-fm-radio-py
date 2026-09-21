"""Tests for the station catalogue and its user-editable override layer."""

from __future__ import annotations

import json
import sys

import pytest

from fm_radio import stations as st
from fm_radio.stations import Station


#: Skip marker for the user layer, which needs tomllib.  Defined per module
#: rather than shared through an import: importing one test module from
#: another breaks under pytest's importlib import mode.
# ----------------------------------------------------------------------
# Bundled snapshot
# ----------------------------------------------------------------------

def test_bundled_catalogue_loads(catalogue):
    assert len(catalogue) > 900
    assert all(isinstance(s, Station) for s in catalogue)


def test_every_entry_is_in_the_broadcast_band(catalogue):
    # Japan's FM band, wide-FM included, is 76.1-94.9 MHz.
    assert all(76.0 <= s.freq_mhz <= 95.0 for s in catalogue)


def test_entries_have_a_name_area_and_known_kind(catalogue):
    assert all(s.name for s in catalogue)
    assert all(s.area in st.AREAS for s in catalogue)
    assert {s.kind for s in catalogue} <= {"fm", "widefm", "nhk"}


def test_transmitters_are_unique(catalogue):
    keys = [s.key for s in catalogue]
    assert len(keys) == len(set(keys))


def test_freq_hz_matches_freq_mhz(catalogue):
    station = catalogue[0]
    assert station.freq_hz == pytest.approx(station.freq_mhz * 1e6)


def test_brand_names_replace_legal_names(catalogue):
    """The MIC list says エフエム東京; listeners say TOKYO FM."""
    tokyo = [s for s in catalogue if s.freq_mhz == 80.0 and s.site == "東京"]
    assert len(tokyo) == 1
    assert tokyo[0].name == "TOKYO FM"
    assert tokyo[0].legal_name == "エフエム東京"


def test_nhk_is_present(catalogue):
    nhk = [s for s in catalogue if s.kind == "nhk"]
    assert len(nhk) > 500
    tokyo = [s for s in nhk if s.site == "東京"]
    assert tokyo and tokyo[0].freq_mhz == 82.5


def test_relay_transmitters_are_included(catalogue):
    """A broadcaster appears once per transmitter site, not once overall."""
    sites = {s.site for s in catalogue if s.name == "TOKYO FM"}
    assert {"東京", "八王子"} <= sites


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

def test_default_presets_match_the_shipped_station_list(catalogue,
                                                       legacy_preset_mhz):
    preset_freqs = [s.freq_mhz for s in st.favorites(catalogue)]
    assert preset_freqs == legacy_preset_mhz


def test_presets_are_named(catalogue):
    names = {s.name for s in st.favorites(catalogue)}
    assert {"TOKYO FM", "J-WAVE", "NACK5", "BAYFM78", "NHK-FM 東京"} <= names


# ----------------------------------------------------------------------
# Lookup helpers
# ----------------------------------------------------------------------

def test_search_matches_brand_legal_name_and_site(catalogue):
    assert st.search(catalogue, "tokyo fm")
    assert st.search(catalogue, "エフエム東京")          # legal name
    assert st.search(catalogue, "八王子")                # transmitter site
    assert st.search(catalogue, "80.0")                  # frequency


def test_search_is_case_insensitive(catalogue):
    assert st.search(catalogue, "j-wave") == st.search(catalogue, "J-WAVE")


def test_in_area_filters(catalogue):
    kanto = st.in_area(catalogue, "関東")
    assert kanto and all(s.area == "関東" for s in kanto)


def test_nearest_names_the_tuned_frequency(catalogue):
    # 80.0 MHz is allocated in several areas; the preset wins the tie.
    station = st.nearest(catalogue, 80.0e6)
    assert station is not None and station.name == "TOKYO FM"


def test_nearest_tolerates_small_offsets(catalogue):
    assert st.nearest(catalogue, 80.02e6) is not None


def test_nearest_returns_none_off_station():
    near = Station(name="near", freq_mhz=80.0)
    assert st.nearest([near], 80.2e6) is None
    assert st.nearest([near], 80.04e6) is near


def test_nearest_prefers_the_closer_entry_over_a_preset():
    preset = Station(name="preset", freq_mhz=80.0, favorite=True)
    closer = Station(name="closer", freq_mhz=80.02)
    assert st.nearest([preset, closer], 80.02e6) is closer


# ----------------------------------------------------------------------
# User layer
# ----------------------------------------------------------------------

needs_tomllib = pytest.mark.skipif(
    st.tomllib is None, reason="tomllib requires Python 3.11+")


def test_missing_user_file_is_not_an_error(no_user_config):
    assert st.load_stations(user_path=no_user_config)


@needs_tomllib
def test_user_station_is_added_and_wins_the_frequency(write_toml):
    path = write_toml("""
[[station]]
name = "レインボータウンFM"
freq_mhz = 79.2
site = "江東"
area = "関東"
favorite = true
""")
    loaded = st.load_stations(user_path=path)
    added = [s for s in loaded if s.name == "レインボータウンFM"]
    assert len(added) == 1
    assert added[0].favorite and added[0].source == "user"
    # An explicit favourite replaces the shipped preset list entirely.
    assert [s.name for s in st.favorites(loaded)] == ["レインボータウンFM"]


# ----------------------------------------------------------------------
# Where the radio is
# ----------------------------------------------------------------------

@needs_tomllib
def test_an_area_does_not_narrow_the_catalogue_itself(write_toml):
    """Looking a station up is not the same question as naming one.

    ``list 北海道`` is about Japan; the dial is about this receiver.
    A radio that would not look up a station a thousand kilometres
    away is less useful, not more.
    """
    whole = st.load_stations(user_path=None, warn=lambda _m: None)

    loaded = st.load_stations(user_path=write_toml('area = "関東"\n'))

    assert len(loaded) == len(whole)
    assert {x.area for x in loaded} > {"関東"}, "other areas went missing"


@needs_tomllib
def test_the_area_is_what_may_put_a_name_on_the_dial(write_toml):
    path = write_toml('area = "関東"\n')
    catalogue = st.load_stations(user_path=path)

    mine = st.here(catalogue, st.home_area(path))

    assert mine, "nothing was left at all"
    assert {x.area for x in mine} == {"関東"}
    assert len(mine) < len(catalogue)


@needs_tomllib
def test_no_area_names_from_everything(write_toml):
    """Most people never write one, and nothing changes for them."""
    path = write_toml("# nothing here\n")
    catalogue = st.load_stations(user_path=path)

    assert st.home_area(path) is None
    assert len(st.here(catalogue, None)) == len(catalogue)


def test_an_area_that_holds_nothing_names_from_everything():
    """A line in a file must not leave the receiver unable to name."""
    only_kanto = [Station(name="J-WAVE", freq_mhz=81.3, site="東京",
                          area="関東")]

    assert st.here(only_kanto, "北海道") == only_kanto


@needs_tomllib
def test_an_area_stops_a_distant_transmitter_naming_a_local_signal():
    """The reading this exists for.

    A band scan in Tokyo found a signal at 82.1 MHz - it is NHK-FM
    at 82.5 spilling over - and the catalogue named it FM NORTH WAVE
    at 稚内, because that is the nearest 82.1 anywhere.  Saying
    nothing is the right answer.
    """
    everywhere = st.load_stations(user_path=None, warn=lambda _m: None)
    heard_at = 82.1e6

    named_anywhere = st.nearest(everywhere, heard_at)
    kanto = [x for x in everywhere if x.area == "関東"]
    named_here = st.nearest(kanto, heard_at)

    assert named_anywhere is not None, "the catalogue used to name this"
    assert named_here is None, "82.1 is not allocated in 関東"


@needs_tomllib
def test_a_station_the_user_added_can_still_name_the_dial(write_toml):
    """They put it there; it is not for this to second-guess them."""
    path = write_toml("""
area = "関東"

[[station]]
name = "どこかの局"
freq_mhz = 88.1
site = "うち"
area = "北海道"
""")
    mine = st.here(st.load_stations(user_path=path), st.home_area(path))

    assert [x.name for x in mine if x.name == "どこかの局"] == ["どこかの局"]


@needs_tomllib
def test_an_entry_with_no_area_at_all_is_kept(write_toml):
    """The bundled data always has one, so one without is the user's."""
    path = write_toml("""
area = "関東"

[[station]]
name = "名無しの地域"
freq_mhz = 88.3
site = "うち"
""")
    mine = st.here(st.load_stations(user_path=path), st.home_area(path))

    assert [x.name for x in mine if x.name == "名無しの地域"]


@needs_tomllib
@pytest.mark.parametrize("line,complaint", [
    ('area = "かんとう"\n', "not an area"),
    ('area = 42\n', "must be the name of one"),
    ('area = ""\n', "must be the name of one"),
])
def test_an_area_that_is_not_one_is_reported_and_ignored(write_toml, line,
                                                          complaint):
    """A line in a file must not leave the receiver unable to name
    anything at all.
    """
    said = []
    path = write_toml(line)

    assert st.home_area(path, warn=said.append) is None
    assert any(complaint in m for m in said), said

    catalogue = st.load_stations(user_path=path, warn=lambda _m: None)
    assert len(st.here(catalogue, None)) > 100


def test_asking_where_the_radio_is_without_a_file_says_nothing(tmp_path):
    assert st.home_area(tmp_path / "there is none.toml") is None


@needs_tomllib
def test_user_station_replaces_a_colliding_bundled_entry(write_toml):
    path = write_toml("""
[[station]]
name = "自宅の80.0"
freq_mhz = 80.0
site = "東京"
""")
    loaded = st.load_stations(user_path=path)
    at_key = [s for s in loaded if s.key == (80.0, "東京")]
    assert len(at_key) == 1
    assert at_key[0].name == "自宅の80.0"


@needs_tomllib
def test_override_hides_a_transmitter(write_toml):
    path = write_toml("""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
hidden = true
""")
    loaded = st.load_stations(user_path=path)
    assert not [s for s in loaded if s.key == (80.0, "東京")]


@needs_tomllib
def test_override_corrects_a_frequency(write_toml):
    path = write_toml("""
[[override]]
match_name = "TOKYO FM"
match_site = "八王子"
freq_mhz = 80.6
""")
    loaded = st.load_stations(user_path=path)
    corrected = [s for s in loaded if s.name == "TOKYO FM" and s.site == "八王子"]
    assert corrected and corrected[0].freq_mhz == 80.6


@needs_tomllib
def test_override_matches_on_the_legal_name(write_toml):
    path = write_toml("""
[[override]]
match_name = "エフエム東京"
match_site = "東京"
name = "TFM"
""")
    loaded = st.load_stations(user_path=path)
    assert [s for s in loaded if s.name == "TFM"]


@needs_tomllib
def test_override_without_a_match_key_changes_nothing(write_toml, catalogue):
    """A rule that selects everything would silently rewrite the catalogue."""
    path = write_toml("""
[[override]]
hidden = true
""")
    assert len(st.load_stations(user_path=path)) == len(catalogue)


@needs_tomllib
def test_override_can_mark_a_favorite(write_toml):
    path = write_toml("""
[[override]]
match_name = "FM COCOLO"
favorite = true
""")
    loaded = st.load_stations(user_path=path)
    assert {s.name for s in st.favorites(loaded)} == {"FM COCOLO"}


@needs_tomllib
def test_broken_toml_falls_back_to_the_bundled_list(write_toml, catalogue):
    path = write_toml("[[station]\nname = broken")
    assert len(st.load_stations(user_path=path)) == len(catalogue)


@needs_tomllib
def test_station_without_frequency_is_skipped(write_toml, catalogue):
    path = write_toml("""
[[station]]
name = "周波数なし"
""")
    loaded = st.load_stations(user_path=path)
    assert len(loaded) == len(catalogue)
    assert not [s for s in loaded if s.name == "周波数なし"]


def test_malformed_bundled_entries_are_skipped(tmp_path, no_user_config):
    data = tmp_path / "stations.json"
    data.write_text(json.dumps({"stations": [
        {"name": "良", "freq_mhz": 80.0, "area": "関東"},
        {"name": "周波数なし"},
        {"name": "文字列", "freq_mhz": "abc"},
    ]}), encoding="utf-8")
    loaded = st.load_stations(user_path=no_user_config, data_path=data)
    assert [s.name for s in loaded] == ["良"]


def test_unreadable_bundled_file_yields_an_empty_catalogue(tmp_path,
                                                           no_user_config):
    assert st.load_stations(user_path=no_user_config,
                            data_path=tmp_path / "absent.json") == []


# ----------------------------------------------------------------------
# Config location
# ----------------------------------------------------------------------

def test_user_config_path_ends_in_the_expected_file():
    path = st.user_config_path()
    assert path.name == "stations.toml"
    assert path.parent.name == "fm_radio"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX config layout")
def test_user_config_path_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert st.user_config_path() == tmp_path / "fm_radio" / "stations.toml"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows config layout")
def test_user_config_path_follows_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert st.user_config_path() == tmp_path / "fm_radio" / "stations.toml"
