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
"""Station catalogue: the bundled nationwide list plus the user's own edits.

Two layers:

* ``fm_radio/data/stations.json`` — every FM, wide-FM and NHK-FM transmitter
  in Japan, regenerated from its upstream sources by
  ``tools/fetch_stations.py``.  Treated as a build artefact: never edited by
  hand, replaced wholesale when it is refreshed.
* ``stations.toml`` in the user's config directory — additions, corrections
  and favourites.  Survives a refresh of the bundled snapshot.

The user layer always wins.  ``[[station]]`` adds an entry, ``[[override]]``
edits or hides a bundled one, and ``favorite = true`` puts an entry in the
short preset list the CLI tunes by number.

Reading the user file needs ``tomllib`` (Python 3.11+).  On older
interpreters the bundled catalogue still loads and a warning is logged, so
the receiver keeps working without the user layer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

try:                                                    # Python 3.11+
    import tomllib
except ImportError:                                     # pragma: no cover
    tomllib = None                                      # type: ignore[assignment]

logger = logging.getLogger('fm_receiver.stations')

DATA_PATH: Path = Path(__file__).resolve().parent / "data" / "stations.json"

USER_CONFIG_FILENAME = "stations.toml"

#: Areas in the order the bundled catalogue groups them.
AREAS: tuple[str, ...] = (
    "北海道", "東北", "関東", "信越", "北陸",
    "東海", "近畿", "中国", "四国", "九州・沖縄",
)

#: Preset list used when the user has not marked any favourites — the ten
#: stations the receiver shipped with, identified by (MHz, transmitter site)
#: so they keep pointing at the same transmitters as the catalogue is
#: refreshed and broadcasters are renamed.
DEFAULT_FAVORITES: tuple[tuple[float, str], ...] = (
    (78.0, "千葉"),      # BAYFM78
    (79.5, "さいたま"),   # NACK5
    (80.0, "東京"),      # TOKYO FM
    (81.3, "東京"),      # J-WAVE
    (82.5, "東京"),      # NHK-FM 東京
    (84.7, "横浜"),      # FMヨコハマ
    (89.7, "東京"),      # InterFM897
    (90.5, "墨田"),      # TBSラジオ
    (91.6, "墨田"),      # 文化放送
    (93.0, "墨田"),      # ニッポン放送
)

_FREQ_TOLERANCE_MHZ = 0.005


@dataclass(frozen=True)
class Station:
    """One transmitter.

    A broadcaster appears once per transmitter site, so ``name`` is not
    unique; ``(freq_mhz, site)`` is what identifies an entry.
    """

    name: str
    freq_mhz: float
    site: str = ""
    area: str = ""
    kind: str = "fm"            # fm | widefm | nhk | user
    legal_name: str = ""
    source: str = ""
    favorite: bool = False

    @property
    def freq_hz(self) -> float:
        """Frequency in Hz, as the tuner wants it."""
        return self.freq_mhz * 1e6

    @property
    def key(self) -> tuple[float, str]:
        """Identity of this entry within the catalogue."""
        return (round(self.freq_mhz, 3), self.site)

    def matches(self, query: str) -> bool:
        """True if *query* appears in the name, legal name, site or frequency."""
        query = query.strip().casefold()
        if not query:
            return True
        haystack = " ".join((
            self.name, self.legal_name, self.site, self.area,
            f"{self.freq_mhz:.1f}",
        )).casefold()
        return query in haystack

    def describe(self) -> str:
        """One-line description for the CLI."""
        parts = [f"{self.name} ({self.freq_mhz:.1f} MHz)"]
        if self.site:
            parts.append(f"[{self.site}]")
        if self.legal_name:
            parts.append(f"- {self.legal_name}")
        return " ".join(parts)


def user_config_path() -> Path:
    """Return the per-user ``stations.toml`` path for this platform."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return root / "fm_radio" / USER_CONFIG_FILENAME


def _load_bundled(path: Path) -> list[Station]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.error("Could not read bundled station list %s: %s", path, exc)
        return []

    stations: list[Station] = []
    for entry in payload.get("stations", ()):
        try:
            stations.append(Station(
                name=str(entry["name"]),
                freq_mhz=float(entry["freq_mhz"]),
                site=str(entry.get("site", "")),
                area=str(entry.get("area", "")),
                kind=str(entry.get("kind", "fm")),
                legal_name=str(entry.get("legal_name", "")),
                source=str(entry.get("source", "")),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed station entry %r: %s", entry, exc)
    return stations


def _read_user_toml(path: Path) -> dict:
    if tomllib is None:
        logger.warning(
            "%s exists but tomllib is unavailable (Python 3.11+ required); "
            "using the bundled station list only", path)
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
    except Exception as exc:                    # tomllib.TOMLDecodeError
        logger.error("%s is not valid TOML (%s); ignoring it", path, exc)
    return {}


def _station_from_toml(entry: dict) -> Station | None:
    try:
        freq_mhz = float(entry["freq_mhz"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping [[station]] without a usable freq_mhz: %r", entry)
        return None
    name = str(entry.get("name") or f"{freq_mhz:.1f} MHz")
    return Station(
        name=name,
        freq_mhz=freq_mhz,
        site=str(entry.get("site", "")),
        area=str(entry.get("area", "")),
        kind=str(entry.get("kind", "user")),
        legal_name=str(entry.get("legal_name", "")),
        source="user",
        favorite=bool(entry.get("favorite", False)),
    )


def _override_matches(station: Station, rule: dict) -> bool:
    """True if *rule* selects *station*.

    A rule with no ``match_*`` key at all matches nothing: silently editing
    every station in the catalogue is never what was meant.
    """
    checked = False
    name = rule.get("match_name")
    if name is not None:
        checked = True
        if station.name != name and station.legal_name != name:
            return False
    site = rule.get("match_site")
    if site is not None:
        checked = True
        if station.site != site:
            return False
    freq = rule.get("match_freq_mhz")
    if freq is not None:
        checked = True
        try:
            if abs(station.freq_mhz - float(freq)) > _FREQ_TOLERANCE_MHZ:
                return False
        except (TypeError, ValueError):
            return False
    return checked


def _apply_override(station: Station, rule: dict) -> Station:
    changes: dict[str, object] = {}
    for field in ("name", "site", "area", "kind", "legal_name"):
        if field in rule:
            changes[field] = str(rule[field])
    if "freq_mhz" in rule:
        try:
            changes["freq_mhz"] = float(rule["freq_mhz"])
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric freq_mhz in [[override]] %r", rule)
    if "favorite" in rule:
        changes["favorite"] = bool(rule["favorite"])
    return replace(station, **changes) if changes else station


def _merge_user_layer(stations: list[Station], config: dict) -> list[Station]:
    """Apply ``[[override]]`` rules, then append ``[[station]]`` entries."""
    rules = [r for r in config.get("override", ()) if isinstance(r, dict)]
    if rules:
        merged: list[Station] = []
        for station in stations:
            hidden = False
            for rule in rules:
                if not _override_matches(station, rule):
                    continue
                if rule.get("hidden"):
                    hidden = True
                    break
                station = _apply_override(station, rule)
            if not hidden:
                merged.append(station)
        stations = merged

    for entry in config.get("station", ()):
        if not isinstance(entry, dict):
            continue
        added = _station_from_toml(entry)
        if added is None:
            continue
        # A user entry replaces the bundled one it collides with, so that
        # correcting a transmitter does not leave a duplicate behind.
        stations = [s for s in stations if s.key != added.key]
        stations.append(added)
    return stations


def _apply_default_favorites(stations: list[Station]) -> list[Station]:
    """Mark the shipped preset stations when the user marked none."""
    wanted = {(round(freq, 3), site) for freq, site in DEFAULT_FAVORITES}
    return [replace(s, favorite=True) if s.key in wanted else s for s in stations]


def load_stations(user_path: Path | str | None = None,
                  data_path: Path | str | None = None) -> list[Station]:
    """Load the catalogue: bundled snapshot merged with the user's TOML.

    Args:
        user_path: Explicit ``stations.toml`` location.  ``None`` uses
            :func:`user_config_path`; a path that does not exist is not an
            error (most users never create one).
        data_path: Override for the bundled JSON, for tests.

    Returns:
        Stations sorted by area then frequency.  Never raises: a broken
        catalogue degrades to an empty list rather than stopping the
        receiver from starting.
    """
    stations = _load_bundled(Path(data_path) if data_path else DATA_PATH)

    path = Path(user_path) if user_path is not None else user_config_path()
    if path.exists():
        config = _read_user_toml(path)
        if config:
            before = len(stations)
            stations = _merge_user_layer(stations, config)
            logger.info("Applied %s (%d -> %d stations)", path, before, len(stations))

    if not any(s.favorite for s in stations):
        stations = _apply_default_favorites(stations)

    stations.sort(key=lambda s: (
        AREAS.index(s.area) if s.area in AREAS else len(AREAS),
        s.freq_mhz, s.name,
    ))
    return stations


def favorites(stations: list[Station]) -> list[Station]:
    """Return the preset stations, ordered by frequency."""
    return sorted((s for s in stations if s.favorite), key=lambda s: s.freq_mhz)


def search(stations: list[Station], query: str) -> list[Station]:
    """Return the stations matching *query* (name, legal name, site, area, MHz)."""
    return [s for s in stations if s.matches(query)]


def in_area(stations: list[Station], area: str) -> list[Station]:
    """Return the stations in *area*."""
    return [s for s in stations if s.area == area]


def nearest(stations: list[Station], freq_hz: float,
            tolerance_hz: float = 50e3) -> Station | None:
    """Return the catalogue entry closest to *freq_hz*, or None if none is near.

    Used to put a name on whatever the tuner is currently sitting on.  A
    frequency is not unique nationwide — 80.0 MHz is TOKYO FM in 関東 and
    RKKラジオ in 熊本 — and the receiver cannot know where it is, so ties
    are broken in favour of a preset: the user's own favourites are the
    only evidence available about which transmitter they can actually
    hear.  Remaining ties keep catalogue order, so the result is stable.
    """
    best: Station | None = None
    best_delta = tolerance_hz
    for station in stations:
        delta = abs(station.freq_hz - freq_hz)
        if delta > tolerance_hz:
            continue
        if (best is None or delta < best_delta
                or (delta == best_delta and station.favorite and not best.favorite)):
            best, best_delta = station, delta
    return best
