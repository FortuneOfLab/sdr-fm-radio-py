"""Regressions for input that reaches the loader from outside a TOML table.

Everything here is a case where a value parsed fine as TOML or JSON but was
still unusable: a number too large to be a float, a path that cannot be
looked at, a string where a boolean was meant, or a rule that quietly matched
nothing at all.
"""

from __future__ import annotations

import pathlib

import pytest

from fm_radio import stations as st

needs_tomllib = pytest.mark.skipif(
    st.tomllib is None, reason="tomllib requires Python 3.11+")


# ----------------------------------------------------------------------
# Numbers TOML accepts but float() cannot take
# ----------------------------------------------------------------------

@needs_tomllib
def test_an_enormous_integer_frequency_is_skipped(load_config, catalogue):
    """TOML integers are unbounded; float() raises OverflowError on this."""
    loaded, warnings = load_config("""
[[station]]
name = "巨大"
freq_mhz = %s
""" % ("9" * 400))
    assert len(loaded) == len(catalogue)
    assert not [s for s in loaded if s.name == "巨大"]
    assert warnings


@needs_tomllib
def test_an_enormous_integer_in_an_override_is_ignored(load_config):
    loaded, warnings = load_config("""
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
freq_mhz = %s
""" % ("9" * 400))
    tokyo = [s for s in loaded if s.name == "TOKYO FM" and s.site == "東京"]
    assert tokyo and tokyo[0].freq_mhz == 80.0
    assert warnings


def test_normalize_freq_survives_an_unconvertible_integer():
    assert st.normalize_freq(10 ** 400) is None


# ----------------------------------------------------------------------
# The config path itself may be unreadable
# ----------------------------------------------------------------------

def test_an_unreadable_config_path_does_not_stop_the_catalogue(monkeypatch):
    """Path.exists() can raise on a locked or unreachable location."""
    def explode(self):
        raise PermissionError(13, "permission denied")

    monkeypatch.setattr(pathlib.Path, "exists", explode)
    warnings: list[str] = []
    # The bundled JSON is read through the same Path type, so point the
    # loader at data it can still open: only the user path is consulted
    # through exists().
    loaded = st.load_stations(user_path="/locked/stations.toml",
                              warn=warnings.append)
    assert loaded
    assert any("Could not check for" in w for w in warnings)


# ----------------------------------------------------------------------
# Flags must be real booleans
# ----------------------------------------------------------------------

@needs_tomllib
@pytest.mark.parametrize("value", ['"false"', '"no"', "0", '""'])
def test_a_non_boolean_hidden_does_not_hide(value, load_config, catalogue):
    """A non-empty string is truthy, which would hide the opposite of intent."""
    loaded, warnings = load_config(f"""
[[override]]
match_name = "TOKYO FM"
hidden = {value}
""")
    assert [s for s in loaded if s.name == "TOKYO FM"]
    assert len(loaded) == len(catalogue)
    assert any("hidden must be true or false" in w for w in warnings)


@needs_tomllib
@pytest.mark.parametrize("value", ['"false"', "1", '"true"'])
def test_a_non_boolean_favorite_in_an_override_is_ignored(
        value, load_config, legacy_preset_mhz):
    loaded, warnings = load_config(f"""
[[override]]
match_freq_mhz = 80.0
match_site = "東京"
favorite = {value}
""")
    assert [s.freq_mhz for s in st.favorites(loaded)] == legacy_preset_mhz
    assert any("favorite must be true or false" in w for w in warnings)


@needs_tomllib
@pytest.mark.parametrize("value", ['"true"', "1"])
def test_a_non_boolean_favorite_on_an_added_station_is_ignored(
        value, load_config, legacy_preset_mhz):
    loaded, warnings = load_config(f"""
[[station]]
name = "自宅"
freq_mhz = 79.2
favorite = {value}
""")
    assert [s for s in loaded if s.name == "自宅"]
    # The shipped presets stand: the string never became a favourite.
    assert [s.freq_mhz for s in st.favorites(loaded)] == legacy_preset_mhz
    assert any("favorite must be true or false" in w for w in warnings)


@needs_tomllib
def test_real_booleans_still_work(load_config):
    loaded, warnings = load_config("""
[[override]]
match_name = "TOKYO FM"
hidden = true
""")
    assert not [s for s in loaded if s.name == "TOKYO FM"]
    assert not warnings


# ----------------------------------------------------------------------
# A rule that matches nothing is a silent no-op otherwise
# ----------------------------------------------------------------------

@needs_tomllib
def test_a_rule_matching_no_station_is_reported(load_config):
    loaded, warnings = load_config("""
[[override]]
match_name = "存在しない局名"
favorite = true
""")
    assert any("#1 matched no station" in w for w in warnings)
    # Nothing was marked, so the shipped presets still apply.
    assert st.favorites(loaded)


@needs_tomllib
def test_the_reported_index_points_at_the_rule(load_config):
    _, warnings = load_config("""
[[override]]
match_name = "TOKYO FM"
favorite = true

[[override]]
match_site = "存在しない送信所"
hidden = true
""")
    assert any("#2 matched no station" in w for w in warnings)
    assert not any("#1 matched no station" in w for w in warnings)


@needs_tomllib
def test_a_rule_that_matches_is_not_reported(load_config):
    _, warnings = load_config("""
[[override]]
match_name = "TOKYO FM"
match_site = "東京"
name = "改名後"
""")
    assert not warnings


@needs_tomllib
def test_a_rule_that_cannot_match_is_only_reported_once(load_config):
    """No match_* key: already reported as unusable, not again as unmatched."""
    _, warnings = load_config("""
[[override]]
hidden = true
""")
    assert len(warnings) == 1
    assert "matches nothing" in warnings[0]


# ----------------------------------------------------------------------
# The tests themselves must not read the developer's own configuration
# ----------------------------------------------------------------------

@needs_tomllib
def test_an_explicit_path_ignores_the_real_user_config(monkeypatch, tmp_path,
                                                       legacy_preset_mhz):
    """A developer with their own favourites must not fail the suite."""
    theirs = tmp_path / "real-stations.toml"
    theirs.write_text("""
[[station]]
name = "自宅の局"
freq_mhz = 79.2
favorite = true
""", encoding="utf-8")
    monkeypatch.setattr(st, "user_config_path", lambda: theirs)

    # Left to itself the loader would pick that file up ...
    assert [s.name for s in st.favorites(st.load_stations())] == ["自宅の局"]
    # ... but an explicit path, as every test passes, does not.
    absent = tmp_path / "absent.toml"
    assert ([s.freq_mhz for s in st.favorites(st.load_stations(user_path=absent))]
            == legacy_preset_mhz)
