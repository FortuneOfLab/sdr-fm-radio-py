#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) [2025] FortuneOfLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Regenerate ``fm_radio/data/stations.json`` from its upstream sources.

Three sources, each a primary one:

``MIC_URL``
    総務省「全国民放FM局・ワイドFM局一覧」 — every commercial FM and wide-FM
    transmitter in Japan, grouped by area, as HTML.  Carries legal entity
    names (エフエム東京), not the names listeners know.

``NHK_URL``
    The JSON that NHK's own frequency page renders from.  NHK-FM is absent
    from the MIC list, so it has to come from here.

``RADIKO_URL``
    radiko's station list, used only for the brand name (愛称) of each
    commercial broadcaster.  Joined onto the MIC rows through the banner
    image slug the MIC page uses per broadcaster; ``RADIKO_ID_BY_LEGAL``
    covers the broadcasters that slug does not resolve.

Run it whenever the upstream lists change::

    python tools/fetch_stations.py

Two of the sources are HTML and XML scraped with regular expressions, so the
real risk is not a crash but a quiet one: an upstream markup change that drops
a block and leaves a plausible-looking, shorter file behind.  Everything is
therefore validated before anything is written — structure, required fields,
area coverage, key uniqueness, brand resolution, and the size of the change
against the snapshot already committed — and a failing check exits non-zero
with the existing file untouched.  ``--force`` overrides only the checks about
how much the data changed, never the structural ones.

The script is deliberately dependency-free (urllib + re) so it runs anywhere
the receiver does, and it writes nothing except the output JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

MIC_URL = ("https://www.soumu.go.jp/menu_seisaku/ictseisaku/"
           "housou_suishin/fm-list.html")
NHK_URL = "https://www.nhk.or.jp/radio/include/freq.json"
RADIKO_URL = "https://radiko.jp/v3/station/region/full.xml"

USER_AGENT = "sdr-fm-radio-py station-list builder"

OUT_PATH = Path(__file__).resolve().parent.parent / "fm_radio" / "data" / "stations.json"

# The ten areas the MIC list is grouped by.  NHK groups its own regions
# differently (関東・甲信越 / 東海・北陸), so NHK rows are remapped onto these
# through the prefecture they are listed under.
AREAS = ("北海道", "東北", "関東", "信越", "北陸",
         "東海", "近畿", "中国", "四国", "九州・沖縄")

BAND_MIN_MHZ = 76.0
BAND_MAX_MHZ = 95.0

#: Must match fm_radio.stations.FREQ_DECIMALS: the receiver identifies a
#: transmitter by its frequency rounded to this many places plus its site, so
#: the generator has to reject collisions on exactly that key.
FREQ_DECIMALS = 3

#: The per-source transmitter counts the drift check compares.  A baseline
#: without a usable value for each of these cannot be compared against, and
#: is treated as a problem rather than as "nothing to compare".
COUNTED_SOURCES = ("commercial", "nhk")

#: How far each source's transmitter count may move from the committed
#: snapshot before the build stops and asks for ``--force``.  Real edits to
#: these lists are a handful of transmitters at a time; a markup change that
#: drops a block takes out far more than that.
MAX_COUNT_DRIFT = 0.10

AREA_BY_PREFECTURE = {}
for _area, _prefectures in {
    "北海道": "北海道",
    "東北": "青森県 岩手県 宮城県 秋田県 山形県 福島県",
    "関東": "茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 神奈川県 山梨県",
    "信越": "新潟県 長野県",
    "北陸": "富山県 石川県 福井県",
    "東海": "岐阜県 静岡県 愛知県 三重県",
    "近畿": "滋賀県 京都府 大阪府 兵庫県 奈良県 和歌山県",
    "中国": "鳥取県 島根県 岡山県 広島県 山口県",
    "四国": "徳島県 香川県 愛媛県 高知県",
    "九州・沖縄": "福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県",
}.items():
    for _prefecture in _prefectures.split():
        AREA_BY_PREFECTURE[_prefecture] = _area

# Broadcasters whose MIC banner slug does not resolve to a radiko id.
RADIKO_ID_BY_LEGAL = {
    "エフエム岩手": "FMI", "エフエム仙台": "DATEFM", "エフエム福島": "FMF",
    "文化放送": "QRR", "ニッポン放送": "LFR", "アール・エフ・ラジオ日本": "JORF",
    "InterFM897": "INT", "エフエム栃木": "RADIOBERRY", "ベイエフエム": "BAYFM78",
    "横浜エフエム放送": "YFM", "東海ラジオ": "TOKAIRADIO",
    "静岡エフエム放送": "K-MIX", "エフエム滋賀": "E-RADIO",
    "エフエム京都": "ALPHA-STATION", "エフエム大阪": "FMO", "FM COCOLO": "CCL",
    "兵庫エフエム放送": "KISSFMKOBE", "広島エフエム放送": "HFM",
    "エフエム山口": "FMY", "エフエム徳島": "FM807", "エフエム愛媛": "JOEU-FM",
    "エフエム高知": "HI-SIX", "琉球放送": "RBC", "エフエム熊本": "FMK",
    "エフエム鹿児島": "MYUFM",
}

# radiko's display name is not always the name to put in the list: some are
# compounds that carry the legal name along with the brand.  Each entry
# records the radiko name it was written against, so that a later change to
# that name is reported instead of being hidden by the override.
BRAND_OVERRIDES = {
    "AIR-G":         ("AIR-G'(FM北海道)", "AIR-G'"),
    "RFM":           ("Rhythm Station エフエム山形", "Rhythm Station"),
    "DATEFM":        ("Date fm エフエム仙台", "Date fm"),
    "FMK":           ("FMKエフエム熊本", "FMK"),
    "ALPHA-STATION": ("α-STATION FM KYOTO", "α-STATION"),
    "E-RADIO":       ("e-radio FM滋賀", "e-radio"),
    "INT":           ("interfm", "InterFM897"),
    "TOKAIRADIO":    ("TOKAI RADIO", "東海ラジオ"),
    "HI-SIX":        ("エフエム高知", "Hi-Six"),
    "MYUFM":         ("μFM", "μFM"),
}


class BuildError(Exception):
    """A check failed; the existing snapshot must be left alone."""


def fetch(url: str) -> bytes:
    """GET *url* with certificate verification and return the raw body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "CERTIFICATE" in str(reason).upper():
            raise BuildError(
                f"TLS verification failed for {url}: {reason}\n"
                f"The certificates Python trusts are out of date or "
                f"incomplete. Fix the trust store rather than skipping "
                f"verification: update 'certifi' (pip install -U certifi), "
                f"run 'Install Certificates.command' on a python.org macOS "
                f"install, or point SSL_CERT_FILE at a current CA bundle."
            ) from exc
        raise BuildError(f"Could not fetch {url}: {reason}") from exc


def strip_tags(fragment: str) -> str:
    return re.sub(r"<[^>]+>", "", fragment).strip()


# ----------------------------------------------------------------------
# 総務省 — commercial FM and wide-FM transmitters
# ----------------------------------------------------------------------

# The MIC page labels the two groups with a bare <li>.  Both labels have to
# appear in the lookahead that ends a group, or a group runs on into the next
# one and its transmitters are filed under the wrong kind.
_KIND_LABEL = r"FM補完放送局\(ワイドFM\)|FM放送局"

_AREA_BLOCK = re.compile(
    r'<h2[^>]*class="[^"]*area_list_title[^"]*"[^>]*>(.*?)エリア.*?</h2>(.*?)'
    r'(?=<h2[^>]*class="[^"]*area_list_title|<a[^>]*href="#select_area")', re.S)
_KIND_BLOCK = re.compile(
    rf"<li>\s*({_KIND_LABEL})\s*</li>(.*?)"
    rf"(?=<li>\s*(?:{_KIND_LABEL})\s*</li>|\Z)", re.S)
_BROADCASTER = re.compile(
    r'<ul[^>]*class="[^"]*housou[^"]*"[^>]*>(.*?)</ul>', re.S)
_ITEM = re.compile(r"<li>(.*?)</li>", re.S)
_SITE_FREQ = re.compile(r"[（(](.+?)[）)]\s*([\d.]+)\s*MHz")
_BANNER = re.compile(
    r'<img[^>]*src="img/(bnr_[^"]+)\.(?:png|gif|jpg)"[^>]*>\s*'
    r'<ul[^>]*class="[^"]*housou[^"]*"[^>]*>\s*<li>(.*?)</li>', re.S)


def parse_mic(html: str) -> tuple[list[dict], dict[str, str]]:
    """Return (transmitter records, {legal name: banner slug})."""
    records: list[dict] = []
    for area, body in _AREA_BLOCK.findall(html):
        for kind_label, kind_body in _KIND_BLOCK.findall(body):
            kind = "widefm" if "補完" in kind_label else "fm"
            for block in _BROADCASTER.findall(kind_body):
                items = [strip_tags(x) for x in _ITEM.findall(block)]
                items = [x for x in items if x]
                if not items:
                    continue
                legal_name = items[0]
                for item in items[1:]:
                    match = _SITE_FREQ.match(item)
                    if not match:
                        print("warning: unparsed MIC entry %r (%s)"
                              % (item, legal_name), file=sys.stderr)
                        continue
                    records.append({
                        "name": legal_name,
                        "legal_name": legal_name,
                        "freq_mhz": float(match.group(2)),
                        "site": match.group(1),
                        "area": area.strip(),
                        "kind": kind,
                        "source": "soumu",
                    })

    slug_by_legal: dict[str, str] = {}
    for slug, name in _BANNER.findall(html):
        slug_by_legal.setdefault(strip_tags(name), slug[len("bnr_"):])
    return records, slug_by_legal


# ----------------------------------------------------------------------
# NHK — NHK-FM transmitters
# ----------------------------------------------------------------------

def parse_nhk(payload: dict) -> list[dict]:
    records: list[dict] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("fm"), dict):
        raise BuildError("NHK freq.json has no 'fm' object")
    for region in payload["fm"].values():
        if not isinstance(region, dict):
            continue
        for prefecture, sites in region.items():
            area = AREA_BY_PREFECTURE.get(prefecture)
            if area is None:
                print("warning: unknown NHK prefecture %r" % prefecture,
                      file=sys.stderr)
                continue
            if not isinstance(sites, dict):
                continue
            label = prefecture if prefecture == "北海道" else prefecture[:-1]
            for site, freq in sites.items():
                if not freq:
                    continue
                try:
                    freq_mhz = float(freq)
                except (TypeError, ValueError):
                    print("warning: unparsed NHK frequency %r (%s %s)"
                          % (freq, prefecture, site), file=sys.stderr)
                    continue
                records.append({
                    # The brand already says NHK, so there is no legal name
                    # worth repeating on all ~530 rows.
                    "name": "NHK-FM %s" % label,
                    "legal_name": "",
                    "freq_mhz": freq_mhz,
                    "site": site,
                    "area": area,
                    "kind": "nhk",
                    "source": "nhk",
                })
    return records


# ----------------------------------------------------------------------
# radiko — brand names
# ----------------------------------------------------------------------

_RADIKO_STATION = re.compile(
    r"<station><id>(.*?)</id>\s*<name>(.*?)</name>\s*"
    r"<ascii_name>(.*?)</ascii_name>", re.S)


def parse_radiko(xml: str) -> dict[str, tuple[str, str]]:
    """Return {station id: (display name, ascii name)}."""
    stations: dict[str, tuple[str, str]] = {}
    for station_id, name, ascii_name in _RADIKO_STATION.findall(xml):
        name = unicodedata.normalize("NFKC", name.strip())
        name = name.replace("&apos;", "'").replace("&amp;", "&")
        stations[station_id.strip()] = (name, ascii_name.strip())
    return stations


def _key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def brand_names(slug_by_legal: dict[str, str],
                radiko: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Return {legal name: brand name} for every broadcaster we can resolve.

    Also reports a ``BRAND_OVERRIDES`` entry whose upstream display name has
    changed, since the override would otherwise hide the change.
    """
    for station_id, (expected, _) in BRAND_OVERRIDES.items():
        if station_id not in radiko:
            print("warning: BRAND_OVERRIDES has %s, which radiko no longer "
                  "lists" % station_id, file=sys.stderr)
        elif radiko[station_id][0] != expected:
            print("warning: radiko renamed %s from %r to %r; check whether "
                  "BRAND_OVERRIDES still says the right thing"
                  % (station_id, expected, radiko[station_id][0]),
                  file=sys.stderr)

    for legal_name, station_id in RADIKO_ID_BY_LEGAL.items():
        if station_id not in radiko:
            print("warning: RADIKO_ID_BY_LEGAL maps %s to %s, which radiko no "
                  "longer lists" % (legal_name, station_id), file=sys.stderr)

    brands: dict[str, str] = {}
    for legal_name, slug in slug_by_legal.items():
        station_id = RADIKO_ID_BY_LEGAL.get(legal_name)
        if station_id is None:
            slug_key = _key(slug)
            for candidate, (_, ascii_name) in radiko.items():
                if (_key(candidate) in (slug_key, slug_key.replace("FM", ""))
                        or _key(ascii_name) == slug_key):
                    station_id = candidate
                    break
        if station_id is None or station_id not in radiko:
            continue
        override = BRAND_OVERRIDES.get(station_id)
        brands[legal_name] = override[1] if override else radiko[station_id][0]
    return brands


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------

def build_payload(mic_html: str, nhk_payload: dict, radiko_xml: str) -> dict:
    """Assemble the snapshot from already-fetched source documents."""
    records, slug_by_legal = parse_mic(mic_html)
    commercial = len(records)

    records += parse_nhk(nhk_payload)

    brands = brand_names(slug_by_legal, parse_radiko(radiko_xml))
    # Brand resolution is checked against the parsed records rather than
    # against the banner list: a broadcaster whose banner went missing would
    # otherwise pass unnoticed.  It has to be collected here, before an
    # unresolved record has its legal name blanked below.
    unresolved: set[str] = set()
    for record in records:
        legal_name = record["legal_name"]
        brand = brands.get(legal_name)
        if brand:
            record["name"] = brand
        elif legal_name and record["kind"] in ("fm", "widefm"):
            unresolved.add(legal_name)
        if record["name"] == record["legal_name"]:
            # Nothing gained by repeating the same string twice.
            record["legal_name"] = ""

    records.sort(key=lambda r: (
        AREAS.index(r["area"]) if r["area"] in AREAS else len(AREAS),
        r["freq_mhz"], r["name"]))
    return {
        "sources": {"soumu": MIC_URL, "nhk": NHK_URL, "radiko": RADIKO_URL},
        "counts": {
            "total": len(records),
            "commercial": commercial,
            "nhk": len(records) - commercial,
            "broadcasters": len({r["name"] for r in records}),
        },
        "stations": records,
        "_unresolved_brands": sorted(unresolved),
    }


def _check_structure(payload: dict) -> list[str]:
    """Structural checks. These can never be waived: the file would be wrong."""
    problems: list[str] = []
    records = payload["stations"]

    if not records:
        return ["no transmitters parsed at all"]

    seen: dict[tuple, str] = {}
    for record in records:
        label = "%s %s %s MHz" % (record.get("name"), record.get("site"),
                                  record.get("freq_mhz"))
        for field in ("name", "site", "area", "kind", "source"):
            if not record.get(field):
                problems.append(f"{label}: empty {field}")
        freq = record.get("freq_mhz")
        if not isinstance(freq, float) or not math.isfinite(freq):
            problems.append(f"{label}: frequency is not a finite number")
        elif not (BAND_MIN_MHZ <= freq <= BAND_MAX_MHZ):
            problems.append(f"{label}: frequency outside "
                            f"{BAND_MIN_MHZ}-{BAND_MAX_MHZ} MHz")
        if record.get("area") not in AREAS:
            problems.append(f"{label}: unknown area {record.get('area')!r}")
        if record.get("kind") not in ("fm", "widefm", "nhk"):
            problems.append(f"{label}: unknown kind {record.get('kind')!r}")

        # The receiver identifies a transmitter by (rounded frequency, site)
        # and a user entry on that key replaces whatever is there, so two
        # records sharing it - even under different names - would collapse.
        if isinstance(freq, float) and math.isfinite(freq):
            key = (round(freq, FREQ_DECIMALS), record.get("site"))
            if key in seen:
                problems.append(f"{label}: same frequency and site as "
                                f"{seen[key]}")
            else:
                seen[key] = label

    # Both upstream lists cover all ten areas, so a gap in either one means
    # a block went missing - which an overall area check cannot see, because
    # the other source still covers that area.
    for source in ("soumu", "nhk"):
        from_source = [r for r in records if r.get("source") == source]
        if not from_source:
            problems.append(f"no transmitters from {source}")
            continue
        missing = [a for a in AREAS
                   if not any(r.get("area") == a for r in from_source)]
        if missing:
            problems.append(f"{source} has no transmitters in "
                            + ", ".join(missing))

    unresolved = payload.get("_unresolved_brands") or []
    if unresolved:
        problems.append(
            "no brand name for " + ", ".join(unresolved)
            + " - add them to RADIKO_ID_BY_LEGAL")

    # Report at most a screenful; the first few say what went wrong.
    return problems[:20] + (
        [f"... and {len(problems) - 20} more"] if len(problems) > 20 else [])


def _check_drift(payload: dict, previous: dict | None) -> list[str]:
    """Compare against the baseline snapshot. Waivable with --force.

    ``previous`` has been through :func:`_read_baseline`, so every count in
    :data:`COUNTED_SOURCES` is known to be a positive integer.
    """
    if not previous:
        return []
    old_counts = previous["counts"]
    problems = []
    for label in COUNTED_SOURCES:
        old = old_counts[label]
        new = payload["counts"][label]
        drift = abs(new - old) / old
        if drift > MAX_COUNT_DRIFT:
            problems.append(
                f"{label} transmitters went {old} -> {new} "
                f"({drift:+.0%}, limit {MAX_COUNT_DRIFT:.0%})")
    return problems


def _read_baseline(path: Path) -> tuple[dict | None, str | None]:
    """Return (baseline payload, problem) for the snapshot at *path*.

    An absent baseline is the first-ever generation and simply skips the
    drift checks.  Anything else that stops the comparison from happening —
    unreadable, not JSON, or counts that cannot be compared against — is a
    problem: a checkout in that state is exactly when a truncated list gets
    committed unnoticed.  The file is opened directly rather than tested with
    exists() first, so a permission error on the path is reported like any
    other read failure instead of propagating.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot read the baseline {path} ({exc})"
    except ValueError as exc:
        return None, f"the baseline {path} is not valid JSON ({exc})"

    if not isinstance(payload, dict):
        return None, (f"the baseline {path} is a {type(payload).__name__}, "
                      f"not a JSON object")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        return None, f"the baseline {path} has no 'counts' object"
    for label in COUNTED_SOURCES:
        value = counts.get(label)
        # bool is an int, and a count of zero or less cannot be a ratio.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None, (f"the baseline {path} has no usable "
                          f"counts['{label}'] (found {value!r})")
    return payload, None


def write_snapshot(payload: dict, path: Path) -> None:
    """Write *payload* to *path* atomically, via a temporary file."""
    payload = {k: v for k, v in payload.items() if not k.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=OUT_PATH,
                        help="output path (default: %(default)s)")
    parser.add_argument("--baseline", type=Path, default=OUT_PATH,
                        help="snapshot the transmitter counts are compared "
                             "against (default: %(default)s). Keep the "
                             "default when writing elsewhere with -o, so the "
                             "comparison still happens.")
    parser.add_argument("--force", action="store_true",
                        help="write even when the transmitter counts moved "
                             "further than expected (structural checks still "
                             "apply)")
    args = parser.parse_args()

    try:
        payload = build_payload(
            fetch(MIC_URL).decode("cp932", errors="replace"),
            json.loads(fetch(NHK_URL).decode("utf-8")),
            fetch(RADIKO_URL).decode("utf-8"),
        )
    except BuildError as exc:
        print("error: %s" % exc, file=sys.stderr)
        print("%s left unchanged" % args.output, file=sys.stderr)
        return 1

    problems = _check_structure(payload)

    # Compare against the committed snapshot rather than against whatever is
    # at --output: writing a review copy elsewhere must not disable the check.
    baseline, baseline_problem = _read_baseline(args.baseline)
    drift = _check_drift(payload, baseline)
    if baseline_problem:
        drift.append(baseline_problem)
    elif baseline is None:
        print("note: no baseline at %s; transmitter counts not compared"
              % args.baseline, file=sys.stderr)
    if drift and not args.force:
        problems += drift + ["re-run with --force if this change is expected"]

    if problems:
        print("error: the generated list did not pass its checks:",
              file=sys.stderr)
        for problem in problems:
            print("  - %s" % problem, file=sys.stderr)
        print("%s left unchanged" % args.output, file=sys.stderr)
        return 1

    write_snapshot(payload, args.output)
    counts = payload["counts"]
    print("wrote %s: %d transmitters (%d commercial + %d NHK), %d broadcasters"
          % (args.output, counts["total"], counts["commercial"],
             counts["nhk"], counts["broadcasters"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
