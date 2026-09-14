"""Regression tests for malformed station data and stations.toml semantics.

Split from ``test_stations.py``, which covers the catalogue itself: this
module is about what happens when the input is wrong, and about the merge
and favourite rules a user's ``stations.toml`` relies on.
"""

from __future__ import annotations

import json
import logging

import pytest

from fm_radio import stations as st
from fm_radio.stations import Station

#: See test_stations.py: defined per module so that no test module has to
#: import another one.
needs_tomllib = pytest.mark.skipif(
    st.tomllib is None, reason="tomllib requires Python 3.11+")


# ----------------------------------------------------------------------
# Malformed configuration must not stop the receiver starting
# ----------------------------------------------------------------------

@needs_tomllib
@pytest.mark.parametrize("body", ["override = 1", "station = 1",
                                  'override = "x"', "station = [1, 2]"])
def test_wrong_type_for_a_table_array_is_reported_not_raised(
        body, load_config, catalogue):
    loaded, warnings = load_config(body + "\n")
    assert len(loaded) == len(catalogue)
    assert warnings


@needs_tomllib
def test_a_bad_entry_does_not_discard_the_good_ones(load_config):
    loaded, warnings = load_config("""
[[station]]
name = "良"
freq_mhz = 79.2

[[station]]
name = "周波数なし"
""")
    assert [s for s in loaded if s.name == "良"]
    assert not [s for s in loaded if s.name == "周波数なし"]
    assert warnings


@needs_tomllib
def test_a_non_table_element_is_skipped_but_the_array_still_applies(load_config):
    loaded, warnings = load_config("""
station = [ 1, { name = "良", freq_mhz = 79.2 } ]
""")
    assert [s for s in loaded if s.name == "良"]
    assert warnings


@pytest.mark.parametrize("body", ["[]", '{"stations": null}',
                                  '{"stations": 5}', '"nonsense"'])
def test_unexpected_bundled_json_shape_is_reported_not_raised(
        body, tmp_path, no_user_config):
    data = tmp_path / "stations.json"
    data.write_text(body, encoding="utf-8")
    warnings: list[str] = []
    loaded = st.load_stations(user_path=no_user_config,
                              data_path=data, warn=warnings.append)
    assert loaded == []
    assert warnings


def test_non_dict_bundled_entries_are_skipped(tmp_path, no_user_config):
    data = tmp_path / "stations.json"
    data.write_text(json.dumps({"stations": [
        "not a station", None, {"name": "良", "freq_mhz": 80.0, "area": "関東"},
    ]}), encoding="utf-8")
    loaded = st.load_stations(user_path=no_user_config, data_path=data)
    assert [s.name for s in loaded] == ["良"]


# ----------------------------------------------------------------------
# Non-finite frequencies
# ----------------------------------------------------------------------

@needs_tomllib
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_match_freq_matches_nothing(value, load_config, catalogue):
    """abs(x - nan) > tol is false, so a NaN rule used to select everything."""
    loaded, warnings = load_config(f"""
[[override]]
match_freq_mhz = {value}
hidden = true
""")
    assert len(loaded) == len(catalogue)
    assert warnings


@needs_tomllib
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_station_frequency_is_skipped(value, load_config, catalogue):
    loaded, warnings = load_config(f"""
[[station]]
name = "無限"
freq_mhz = {value}
""")
    assert len(loaded) == len(catalogue)
    assert not [s for s in loaded if s.name == "無限"]
    assert warnings


@needs_tomllib
def test_non_finite_override_frequency_leaves_the_station_alone(load_config):
    loaded, warnings = load_config("""
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
freq_mhz = nan
""")
    tokyo = [s for s in loaded if s.name == "TOKYO FM" and s.site == "東京"]
    assert tokyo and tokyo[0].freq_mhz == 80.0
    assert warnings


def test_nearest_ignores_a_non_finite_entry():
    broken = Station(name="broken", freq_mhz=float("nan"))
    assert st.nearest([broken], 80.0e6) is None
    good = Station(name="good", freq_mhz=80.0)
    assert st.nearest([broken, good], 80.0e6) is good


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
                                   "abc", None, True, [1]])
def test_normalize_freq_rejects_unusable_values(value):
    assert st.normalize_freq(value) is None


def test_normalize_freq_rounds_to_the_catalogue_grid():
    assert st.normalize_freq("80.0") == 80.0
    assert st.normalize_freq(80.00049) == 80.0
    assert st.normalize_freq(80.0006) == 80.001


# ----------------------------------------------------------------------
# Favourites: "unset" and "explicitly false" are different
# ----------------------------------------------------------------------

@needs_tomllib
def test_favorite_false_removes_one_shipped_preset(load_config,
                                                   legacy_preset_mhz):
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
favorite = false
""")
    assert [s.freq_mhz for s in st.favorites(loaded)] == [
        f for f in legacy_preset_mhz if f != 80.0]


@needs_tomllib
def test_all_presets_can_be_switched_off(load_config):
    body = "".join(f"""
[[override]]
match_freq_mhz = {freq}
match_site = "{site}"
favorite = false
""" for freq, site in st.DEFAULT_FAVORITES)
    loaded, _ = load_config(body)
    assert st.favorites(loaded) == []


@needs_tomllib
def test_one_true_replaces_the_shipped_presets(load_config):
    loaded, _ = load_config("""
[[override]]
match_name = "FM COCOLO"
favorite = true

[[override]]
match_freq_mhz = 80.0
match_site = "東京"
favorite = false
""")
    assert {s.name for s in st.favorites(loaded)} == {"FM COCOLO"}


@needs_tomllib
def test_hidden_only_keeps_the_remaining_shipped_presets(load_config,
                                                         legacy_preset_mhz):
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
hidden = true
""")
    assert [s.freq_mhz for s in st.favorites(loaded)] == [
        f for f in legacy_preset_mhz if f != 80.0]


def test_favorite_is_always_a_bool_after_loading(catalogue):
    assert all(isinstance(s.favorite, bool) for s in catalogue)


# ----------------------------------------------------------------------
# Override ordering
# ----------------------------------------------------------------------

RENAME_THEN_HIDE = """
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
name = "改名後"

[[override]]
match_name = "TOKYO FM"
match_site = "東京"
hidden = true
"""

HIDE_THEN_RENAME = """
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
hidden = true

[[override]]
match_name = "TOKYO FM"
match_site = "東京"
name = "改名後"
"""


@needs_tomllib
@pytest.mark.parametrize("body", [RENAME_THEN_HIDE, HIDE_THEN_RENAME])
def test_hidden_wins_whatever_the_rule_order(body, load_config):
    loaded, _ = load_config(body)
    assert not [s for s in loaded if s.key == (80.0, "東京")]


@needs_tomllib
def test_rules_match_the_bundled_entry_not_the_edited_one(load_config):
    """Renaming must not change which later rules apply."""
    loaded, _ = load_config("""
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
name = "改名後"

[[override]]
match_name = "TOKYO FM"
match_site = "東京"
area = "テスト"
""")
    tokyo = [s for s in loaded if s.key == (80.0, "東京")]
    assert tokyo and tokyo[0].name == "改名後" and tokyo[0].area == "テスト"


@needs_tomllib
def test_later_edits_win_over_earlier_ones(load_config):
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
name = "一番目"

[[override]]
match_freq_mhz = 80.0
match_site = "東京"
name = "二番目"
""")
    tokyo = [s for s in loaded if s.key == (80.0, "東京")]
    assert tokyo and tokyo[0].name == "二番目"


@needs_tomllib
def test_a_hidden_key_can_be_re_added(load_config):
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
hidden = true

[[station]]
name = "自前の80.0"
freq_mhz = 80.0
site = "東京"
""")
    assert [s.name for s in loaded if s.key == (80.0, "東京")] == ["自前の80.0"]


# ----------------------------------------------------------------------
# One definition of "the same frequency"
# ----------------------------------------------------------------------

@needs_tomllib
def test_override_frequency_match_is_exact_on_the_catalogue_grid(load_config):
    """80.004 is not 80.0: identity and matching use the same rounding."""
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.004
match_site = "東京"
hidden = true
""")
    assert [s for s in loaded if s.key == (80.0, "東京")]


@needs_tomllib
def test_override_frequency_match_tolerates_sub_khz_rounding(load_config):
    loaded, _ = load_config("""
[[override]]
match_freq_mhz = 80.0004
match_site = "東京"
hidden = true
""")
    assert not [s for s in loaded if s.key == (80.0, "東京")]


@needs_tomllib
def test_a_user_entry_on_the_same_grid_point_replaces_the_bundled_one(load_config):
    loaded, _ = load_config("""
[[station]]
name = "端数"
freq_mhz = 80.0004
site = "東京"
""")
    assert [s.name for s in loaded if s.key == (80.0, "東京")] == ["端数"]


# ----------------------------------------------------------------------
# Areas the user named themselves
# ----------------------------------------------------------------------

@needs_tomllib
def test_in_area_ignores_case(load_config):
    loaded, _ = load_config("""
[[station]]
name = "自宅"
freq_mhz = 79.2
area = "Home"
""")
    assert [s.name for s in st.in_area(loaded, "home")] == ["自宅"]
    assert [s.name for s in st.in_area(loaded, "HOME")] == ["自宅"]


# ----------------------------------------------------------------------
# Problems reach the user even with logging switched off
# ----------------------------------------------------------------------

@needs_tomllib
def test_warn_callback_fires_while_logging_is_disabled(load_config):
    logging.disable(logging.CRITICAL)
    try:
        _, warnings = load_config("[[station]\nname = broken")
    finally:
        logging.disable(logging.NOTSET)
    assert warnings and "TOML" in warnings[0]
