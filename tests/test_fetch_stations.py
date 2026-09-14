"""Offline tests for the station-list generator.

``tools/fetch_stations.py`` scrapes HTML and XML with regular expressions, so
the failure that matters is not a crash but a quiet one: an upstream markup
change that drops a block and leaves a plausible-looking, shorter file behind.
These tests drive the parsers and the validation from fixtures — no network,
and never touching the committed snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "fetch_stations.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("fetch_stations", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fetch_stations", module)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# Fixtures shaped like the real sources
# ----------------------------------------------------------------------

def mic_page(*, wide_first: bool = True, housou_attrs: str = "",
             banner_attrs: str = "", spacing: str = "") -> str:
    """A one-area MIC page with one commercial and one wide-FM broadcaster."""
    fm = f'''<li>{spacing}FM放送局{spacing}</li>
<li><img{banner_attrs} src="img/bnr_tokyoFM.png" alt=""/>
<ul{housou_attrs} class="housou">
<li>エフエム東京</li>
<li>（東京）80.0MHz</li>
<li>（八王子）80.5MHz</li>
</ul>
</li>'''
    wide = f'''<li>{spacing}FM補完放送局(ワイドFM){spacing}</li>
<li><img{banner_attrs} src="img/bnr_tbs.png" alt=""/>
<ul{housou_attrs} class="housou">
<li>TBSラジオ</li>
<li>（墨田）90.5MHz</li>
</ul>
</li>'''
    groups = [wide, fm] if wide_first else [fm, wide]
    return ('<h2 class="area_list_title kanto">関東エリアのFM局一覧</h2>\n'
            + "\n".join(f"<ul>\n{g}\n</ul>" for g in groups)
            + '\n<a href="#select_area">back</a>')


NHK_JSON = {"fm": {"koshinetsu": {"東京都": {"東京": "82.5", "八丈": "82.9"}}}}

RADIKO_XML = """<region>
<station><id>FMT</id>
    <name>TOKYO FM</name>
    <ascii_name>TOKYO FM</ascii_name>
</station>
<station><id>TBS</id>
    <name>ＴＢＳラジオ</name>
    <ascii_name>TBS RADIO</ascii_name>
</station>
</region>"""


def build(tool, **kwargs):
    return tool.build_payload(mic_page(**kwargs), NHK_JSON, RADIKO_XML)


# ----------------------------------------------------------------------
# MIC parsing
# ----------------------------------------------------------------------

def test_parses_both_groups(tool):
    payload = build(tool)
    kinds = {(r["name"], r["site"]): r["kind"] for r in payload["stations"]}
    assert kinds[("TOKYO FM", "東京")] == "fm"
    assert kinds[("TBSラジオ", "墨田")] == "widefm"
    assert payload["counts"]["commercial"] == 3


def test_group_order_does_not_change_the_kind(tool):
    """The group labels end each other; both orders must classify correctly."""
    for wide_first in (True, False):
        payload = build(tool, wide_first=wide_first)
        kinds = {(r["name"], r["site"]): r["kind"] for r in payload["stations"]}
        assert kinds[("TBSラジオ", "墨田")] == "widefm", wide_first
        assert kinds[("TOKYO FM", "東京")] == "fm", wide_first


@pytest.mark.parametrize("kwargs", [
    {"housou_attrs": ' id="hokkaido"'},
    {"banner_attrs": ' width="120"'},
    {"spacing": " "},
])
def test_extra_attributes_and_whitespace_do_not_drop_blocks(tool, kwargs):
    payload = build(tool, **kwargs)
    assert payload["counts"]["commercial"] == 3
    assert not payload["_unresolved_brands"]


def test_an_unparsable_line_is_reported_and_skipped(tool, capsys):
    page = mic_page().replace("<li>（八王子）80.5MHz</li>", "<li>八王子 80.5</li>")
    payload = tool.build_payload(page, NHK_JSON, RADIKO_XML)
    assert payload["counts"]["commercial"] == 2
    assert "unparsed MIC entry" in capsys.readouterr().err


def test_a_renamed_class_is_still_matched(tool):
    """'housou-v2' still says housou; a suffix must not drop the block."""
    page = mic_page().replace('class="housou"', 'class="housou-v2"')
    assert tool.build_payload(page, NHK_JSON,
                              RADIKO_XML)["counts"]["commercial"] == 3


def test_a_missing_block_shows_up_as_a_count_not_an_exception(tool):
    page = mic_page().replace('class="housou"', 'class="bangumi"')
    payload = tool.build_payload(page, NHK_JSON, RADIKO_XML)
    assert payload["counts"]["commercial"] == 0


# ----------------------------------------------------------------------
# NHK and radiko
# ----------------------------------------------------------------------

def test_nhk_rows_are_remapped_onto_the_mic_areas(tool):
    payload = build(tool)
    nhk = [r for r in payload["stations"] if r["kind"] == "nhk"]
    assert {r["area"] for r in nhk} == {"関東"}
    assert {r["name"] for r in nhk} == {"NHK-FM 東京"}


def test_nhk_payload_without_fm_is_an_error(tool):
    with pytest.raises(tool.BuildError):
        tool.build_payload(mic_page(), {"r1": {}}, RADIKO_XML)


def test_unparsable_nhk_frequency_is_skipped(tool, capsys):
    payload = tool.build_payload(
        mic_page(), {"fm": {"x": {"東京都": {"東京": "82.5", "変": "-"}}}},
        RADIKO_XML)
    assert payload["counts"]["nhk"] == 1
    assert "unparsed NHK frequency" in capsys.readouterr().err


def test_brand_names_come_from_radiko(tool):
    names = {r["name"] for r in build(tool)["stations"]}
    assert "TOKYO FM" in names          # MIC says エフエム東京
    assert "TBSラジオ" in names          # full-width in radiko, normalised here


def test_unresolved_brands_are_listed_from_the_records(tool):
    """A broadcaster with no radiko match must be named, banner or not."""
    without_tokyo_fm = "\n".join(
        line for line in RADIKO_XML.splitlines() if "FMT" not in line
        and "TOKYO FM" not in line)
    payload = tool.build_payload(mic_page(), NHK_JSON, without_tokyo_fm)
    assert "エフエム東京" in payload["_unresolved_brands"]
    assert "no brand name for エフエム東京" in "\n".join(
        tool._check_structure(payload))


def test_a_renamed_radiko_station_behind_an_override_is_reported(tool, capsys):
    xml = RADIKO_XML.replace(
        "<station><id>FMT</id>\n    <name>TOKYO FM</name>",
        "<station><id>INT</id>\n    <name>interfm897</name>")
    tool.brand_names({}, tool.parse_radiko(xml))
    err = capsys.readouterr().err
    assert "radiko renamed INT" in err


def test_a_missing_manual_mapping_target_is_reported(tool, capsys):
    tool.brand_names({}, tool.parse_radiko(RADIKO_XML))
    err = capsys.readouterr().err
    assert "RADIKO_ID_BY_LEGAL maps" in err


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def good_payload(tool):
    """A payload that passes every structural check.

    Both sources cover every area, as the real ones do: the area check is
    per source, because a gap in one is invisible in the union.
    """
    stations = []
    for i, area in enumerate(tool.AREAS):
        stations.append({"name": f"民放{i}", "legal_name": "",
                         "freq_mhz": 80.0 + i * 0.1, "site": f"minpo{i}",
                         "area": area, "kind": "fm", "source": "soumu"})
        stations.append({"name": f"NHK-FM {i}", "legal_name": "",
                         "freq_mhz": 85.0 + i * 0.1, "site": f"nhk{i}",
                         "area": area, "kind": "nhk", "source": "nhk"})
    return {
        "counts": {"total": 20, "commercial": 10, "nhk": 10, "broadcasters": 20},
        "stations": stations,
        "_unresolved_brands": [],
    }


def test_a_complete_payload_passes(tool):
    assert tool._check_structure(good_payload(tool)) == []


def test_empty_output_is_rejected(tool):
    payload = good_payload(tool)
    payload["stations"] = []
    assert tool._check_structure(payload) == ["no transmitters parsed at all"]


def test_a_missing_area_is_rejected(tool):
    payload = good_payload(tool)
    payload["stations"] = payload["stations"][:-1]
    assert any("nhk has no transmitters in" in p
               for p in tool._check_structure(payload))


def test_one_source_missing_an_area_is_rejected(tool):
    """The union still covers every area, so only a per-source check sees it."""
    payload = good_payload(tool)
    payload["stations"] = [r for r in payload["stations"]
                           if not (r["source"] == "soumu"
                                   and r["area"] == tool.AREAS[3])]
    problems = tool._check_structure(payload)
    assert any("soumu has no transmitters in" in p for p in problems)


def test_a_missing_source_is_rejected(tool):
    payload = good_payload(tool)
    for record in payload["stations"]:
        record["source"] = "soumu"
        record["kind"] = "fm"
    assert any("no transmitters from nhk" in p
               for p in tool._check_structure(payload))


@pytest.mark.parametrize("field", ["name", "site", "area", "kind", "source"])
def test_an_empty_required_field_is_rejected(tool, field):
    payload = good_payload(tool)
    payload["stations"][0][field] = ""
    assert any(f"empty {field}" in p or "unknown" in p
               for p in tool._check_structure(payload))


@pytest.mark.parametrize("freq", [float("nan"), float("inf"), 42.0, 150.0])
def test_an_impossible_frequency_is_rejected(tool, freq):
    payload = good_payload(tool)
    payload["stations"][0]["freq_mhz"] = freq
    assert tool._check_structure(payload)


def test_a_duplicate_transmitter_is_rejected(tool):
    payload = good_payload(tool)
    payload["stations"].append(dict(payload["stations"][0]))
    assert any("same frequency and site" in p
               for p in tool._check_structure(payload))


def test_the_same_transmitter_under_two_names_is_rejected(tool):
    """The receiver keys on (frequency, site); the name is not part of it."""
    payload = good_payload(tool)
    clash = dict(payload["stations"][0])
    clash["name"] = "別名"
    payload["stations"].append(clash)
    assert any("same frequency and site" in p
               for p in tool._check_structure(payload))


def test_frequencies_that_collide_after_rounding_are_rejected(tool):
    payload = good_payload(tool)
    clash = dict(payload["stations"][0])
    clash["freq_mhz"] += 0.0001
    payload["stations"].append(clash)
    assert any("same frequency and site" in p
               for p in tool._check_structure(payload))


def test_drift_is_allowed_within_the_limit(tool):
    payload = good_payload(tool)
    previous = {"counts": {"commercial": 10, "nhk": 10}}
    assert tool._check_drift(payload, previous) == []


def test_a_collapsed_source_count_is_caught(tool):
    payload = good_payload(tool)
    previous = {"counts": {"commercial": 451, "nhk": 532}}
    problems = tool._check_drift(payload, previous)
    assert len(problems) == 2
    assert "commercial transmitters went 451 -> 10" in problems[0]


def test_drift_needs_no_previous_snapshot(tool):
    assert tool._check_drift(good_payload(tool), None) == []


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def run_main(tool, monkeypatch, output, *extra):
    """Invoke main() with a baseline that does not exist unless asked for."""
    argv = ["fetch_stations", "-o", str(output)]
    if not any(a == "--baseline" for a in extra):
        argv += ["--baseline", str(output.parent / "no-baseline.json")]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    monkeypatch.setattr(tool, "fetch", lambda url: b"{}")
    return tool.main()


def test_failing_checks_leave_the_existing_file_alone(tool, tmp_path, monkeypatch):
    output = tmp_path / "stations.json"
    output.write_text('{"stations": ["existing"]}', encoding="utf-8")

    broken = tool.build_payload(
        mic_page().replace('class="housou"', 'class="bangumi"'),
        NHK_JSON, RADIKO_XML)
    monkeypatch.setattr(tool, "build_payload", lambda *a, **k: broken)

    assert run_main(tool, monkeypatch, output) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "stations": ["existing"]}


def test_a_valid_payload_is_written_without_internal_keys(tool, tmp_path,
                                                          monkeypatch):
    output = tmp_path / "stations.json"
    monkeypatch.setattr(tool, "build_payload",
                        lambda *a, **k: good_payload(tool))

    assert run_main(tool, monkeypatch, output) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert len(written["stations"]) == 2 * len(tool.AREAS)
    assert not [k for k in written if k.startswith("_")]


def test_force_waives_drift_but_not_structure(tool, tmp_path, monkeypatch):
    output = tmp_path / "stations.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"counts": {"commercial": 451, "nhk": 532},
                                    "stations": []}), encoding="utf-8")
    monkeypatch.setattr(tool, "build_payload",
                        lambda *a, **k: good_payload(tool))

    assert run_main(tool, monkeypatch, output,
                    "--baseline", str(baseline)) == 1       # drift stops it
    assert run_main(tool, monkeypatch, output,
                    "--baseline", str(baseline), "--force") == 0

    broken = good_payload(tool)
    broken["stations"][0]["freq_mhz"] = float("nan")
    monkeypatch.setattr(tool, "build_payload", lambda *a, **k: broken)
    assert run_main(tool, monkeypatch, output,
                    "--baseline", str(baseline), "--force") == 1


def test_a_new_output_path_is_still_compared_against_the_baseline(
        tool, tmp_path, monkeypatch):
    """Writing a review copy elsewhere must not switch the check off."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"counts": {"commercial": 451, "nhk": 532},
                                    "stations": []}), encoding="utf-8")
    fresh = tmp_path / "somewhere-new.json"
    monkeypatch.setattr(tool, "build_payload",
                        lambda *a, **k: good_payload(tool))

    assert run_main(tool, monkeypatch, fresh, "--baseline", str(baseline)) == 1
    assert not fresh.exists()


def test_an_absent_baseline_skips_the_drift_check(tool, tmp_path, monkeypatch,
                                                  capsys):
    output = tmp_path / "stations.json"
    monkeypatch.setattr(tool, "build_payload",
                        lambda *a, **k: good_payload(tool))
    assert run_main(tool, monkeypatch, output) == 0
    assert "no baseline" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["not json", '{"counts": null}', "[]"])
def test_an_unusable_baseline_stops_the_build(tool, tmp_path, monkeypatch,
                                              content):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(content, encoding="utf-8")
    output = tmp_path / "stations.json"
    monkeypatch.setattr(tool, "build_payload",
                        lambda *a, **k: good_payload(tool))

    assert run_main(tool, monkeypatch, output, "--baseline", str(baseline)) == 1
    assert not output.exists()
    # ... and --force is the documented way past it.
    assert run_main(tool, monkeypatch, output,
                    "--baseline", str(baseline), "--force") == 0


def test_a_fetch_failure_leaves_the_file_alone(tool, tmp_path, monkeypatch):
    output = tmp_path / "stations.json"
    output.write_text("original", encoding="utf-8")

    def explode(url):
        raise tool.BuildError("no network")

    monkeypatch.setattr(tool, "fetch", explode)
    monkeypatch.setattr(sys, "argv", ["fetch_stations", "-o", str(output)])
    assert tool.main() == 1
    monkeypatch.undo()
    assert output.read_text(encoding="utf-8") == "original"
