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

Network access is required.  The script is deliberately dependency-free
(urllib + re) so it runs anywhere the receiver does; it never writes
anything except the output JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import unicodedata
import urllib.parse
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
# compounds that carry the legal name along with the brand.
BRAND_OVERRIDES = {
    "AIR-G": "AIR-G'",                # radiko: AIR-G'(FM北海道)
    "RFM": "Rhythm Station",          # radiko: Rhythm Station エフエム山形
    "DATEFM": "Date fm",              # radiko: Date fm エフエム仙台
    "FMK": "FMK",                     # radiko: FMKエフエム熊本
    "ALPHA-STATION": "α-STATION",     # radiko: α-STATION FM KYOTO
    "E-RADIO": "e-radio",             # radiko: e-radio FM滋賀
    "INT": "InterFM897",              # radiko: interfm
    "TOKAIRADIO": "東海ラジオ",         # radiko: TOKAI RADIO
    "HI-SIX": "Hi-Six",               # radiko: エフエム高知
    "MYUFM": "μFM",
}


def fetch(url: str) -> bytes:
    """GET *url* and return the raw body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            return response.read()
    except ssl.SSLError:
        # Some Python installs ship an expired CA bundle.  The three sources
        # are public read-only lists, so falling back is acceptable here; the
        # data is reviewed in the diff before it is committed.
        print("warning: TLS verification failed for %s, retrying unverified"
              % url, file=sys.stderr)
        unverified = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=60, context=unverified) as response:
            return response.read()


def strip_tags(fragment: str) -> str:
    return re.sub(r"<[^>]+>", "", fragment).strip()


# ----------------------------------------------------------------------
# 総務省 — commercial FM and wide-FM transmitters
# ----------------------------------------------------------------------

_AREA_BLOCK = re.compile(
    r'<h2 class="area_list_title [^"]*">(.*?)エリア.*?</h2>(.*?)'
    r'(?=<h2 class="area_list_title|<a href="#select_area")', re.S)
_KIND_BLOCK = re.compile(
    r"<li>(FM補完放送局\(ワイドFM\)|FM放送局)</li>(.*?)"
    r"(?=<li>(?:FM補完放送局|FM放送局)</li>|\Z)", re.S)
_BROADCASTER = re.compile(r'<ul class="housou">(.*?)</ul>', re.S)
_ITEM = re.compile(r"<li>(.*?)</li>", re.S)
_SITE_FREQ = re.compile(r"[（(](.+?)[）)]\s*([\d.]+)\s*MHz")
_BANNER = re.compile(
    r'<img src="img/(bnr_[^"]+)\.(?:png|gif|jpg)"[^>]*/?>\s*'
    r'<ul class="housou">\s*<li>(.*?)</li>', re.S)


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
                        "area": area,
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
    for region in payload["fm"].values():
        for prefecture, sites in region.items():
            area = AREA_BY_PREFECTURE.get(prefecture)
            if area is None:
                print("warning: unknown NHK prefecture %r" % prefecture,
                      file=sys.stderr)
                continue
            label = prefecture if prefecture == "北海道" else prefecture[:-1]
            for site, freq in sites.items():
                if not freq:
                    continue
                records.append({
                    # The brand already says NHK, so there is no legal name
                    # worth repeating on all ~530 rows.
                    "name": "NHK-FM %s" % label,
                    "legal_name": "",
                    "freq_mhz": float(freq),
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
    """Return {legal name: brand name} for every broadcaster we can resolve."""
    brands: dict[str, str] = {}
    unresolved: list[str] = []
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
            unresolved.append(legal_name)
            continue
        brands[legal_name] = BRAND_OVERRIDES.get(station_id, radiko[station_id][0])
    if unresolved:
        print("warning: no brand name for %s" % ", ".join(unresolved),
              file=sys.stderr)
    return brands


# ----------------------------------------------------------------------

def build() -> dict:
    mic_html = fetch(MIC_URL).decode("cp932", errors="replace")
    records, slug_by_legal = parse_mic(mic_html)
    commercial = len(records)

    records += parse_nhk(json.loads(fetch(NHK_URL).decode("utf-8")))

    brands = brand_names(slug_by_legal,
                         parse_radiko(fetch(RADIKO_URL).decode("utf-8")))
    for record in records:
        brand = brands.get(record["legal_name"])
        if brand:
            record["name"] = brand
        if record["name"] == record["legal_name"]:
            # Nothing gained by repeating the same string twice.
            record["legal_name"] = ""

    records.sort(key=lambda r: (AREAS.index(r["area"]), r["freq_mhz"], r["name"]))
    return {
        "sources": {
            "soumu": MIC_URL,
            "nhk": NHK_URL,
            "radiko": RADIKO_URL,
        },
        "counts": {
            "total": len(records),
            "commercial": commercial,
            "nhk": len(records) - commercial,
            "broadcasters": len({r["name"] for r in records}),
        },
        "stations": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=OUT_PATH,
                        help="output path (default: %(default)s)")
    args = parser.parse_args()

    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    counts = payload["counts"]
    print("wrote %s: %d transmitters (%d commercial + %d NHK), %d broadcasters"
          % (args.output, counts["total"], counts["commercial"],
             counts["nhk"], counts["broadcasters"]))


if __name__ == "__main__":
    main()
