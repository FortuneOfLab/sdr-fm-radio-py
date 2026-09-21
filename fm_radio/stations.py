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

Merge order, which is what makes a ``stations.toml`` predictable:

1. Every ``[[override]]`` rule is matched against the **bundled** entry, not
   against the result of an earlier rule.  Renaming a station therefore never
   changes which later rules apply to it.
2. Non-hidden edits are applied in the order they appear in the file, so the
   last rule to set a field wins.
3. A matching ``hidden = true`` drops the entry regardless of where it sits in
   the file: hiding always wins over editing.
4. ``[[station]]`` entries are appended last and replace any bundled entry with
   the same ``(frequency, transmitter site)`` — including one an override just
   hid, so hide-then-re-add works.

A top-level ``area`` says where the receiver is.  A frequency is not unique
nationwide — 983 transmitters cover 173 of the 191 channels between 76 and
95 MHz — so without it the catalogue will put a name on almost anything the
tuner sits on, and it is as likely to be a transmitter a thousand kilometres
away as the one being received.  Measured on this radio: of twelve signals a
band scan found in Tokyo, four were named after transmitters in 北海道, 東北
and 四国.  With ``area = "関東"`` those four are named nothing, which is the
right answer, and the eight real ones keep their names.

It narrows naming and nothing else.  :func:`load_stations` returns every
transmitter whatever the setting says, because ``list all``, ``list 北海道``
and searching are about the catalogue rather than about this receiver — you
can look up a station you are nowhere near.  :func:`here` is the view that
may put a name on the dial, and :func:`home_area` reads the setting.

Frequencies are normalised to :data:`FREQ_DECIMALS` decimal places (1 kHz)
everywhere: in the bundled data, in user entries, and in ``match_freq_mhz``.
Two entries are the same transmitter when that normalised frequency and the
site match exactly.  :func:`nearest` is the one place that works with a
tolerance, and that is a question about where the tuner is sitting, not about
which catalogue entries are the same.

Reading the user file needs ``tomllib`` (Python 3.11+).  On older
interpreters the bundled catalogue still loads and a warning is reported, so
the receiver keeps working without the user layer.

Nothing in here raises on bad input.  A broken catalogue, a malformed TOML
file or a nonsensical rule is reported and skipped so the receiver still
starts; pass ``warn`` to :func:`load_stations` to put those reports in front
of the user, since the application disables logging unless asked for it.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

try:                                                    # Python 3.11+
    import tomllib
except ImportError:                                     # pragma: no cover
    tomllib = None                                      # type: ignore[assignment]

logger = logging.getLogger('fm_receiver.stations')

DATA_PATH: Path = Path(__file__).resolve().parent / "data" / "stations.json"

USER_CONFIG_FILENAME = "stations.toml"

#: Decimal places every frequency is rounded to, in MHz.  Japanese FM
#: allocations sit on a 0.1 MHz grid; 1 kHz leaves room for a typo without
#: letting two distinct allocations collide.
FREQ_DECIMALS = 3

#: The Japanese FM broadcast band, wide-FM included.  Outside it is not
#: rejected — the tuner will go wherever the SDR can — but it is reported.
BAND_MIN_MHZ = 76.0
BAND_MAX_MHZ = 95.0

#: Areas in the order the bundled catalogue groups them.
AREAS: tuple[str, ...] = (
    "北海道", "東北", "関東", "信越", "北陸",
    "東海", "近畿", "中国", "四国", "九州・沖縄",
)

#: Preset list used when the user has not marked any favourite — the ten
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

Reporter = Callable[[str], None]


def normalize_freq(value: object) -> float | None:
    """Return *value* as a usable frequency in MHz, or None.

    Rejects anything that is not a finite number: TOML accepts ``nan`` and
    ``inf`` as numbers, and a NaN frequency compares equal to nothing and
    unequal to nothing, which lets one rule silently match every station.
    """
    if isinstance(value, bool):         # bool is an int; never a frequency
        return None
    try:
        freq = float(value)             # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        # OverflowError: TOML integers are unbounded, so a 400-digit literal
        # parses fine and only fails on the conversion to float.
        return None
    if not math.isfinite(freq):
        return None
    return round(freq, FREQ_DECIMALS)


@dataclass(frozen=True)
class Station:
    """One transmitter.

    A broadcaster appears once per transmitter site, so ``name`` is not
    unique; ``(freq_mhz, site)`` is what identifies an entry.

    ``favorite`` is tri-state while the catalogue is being merged: ``None``
    means the user said nothing, which is what lets an explicit
    ``favorite = false`` differ from silence.  :func:`load_stations` always
    resolves it to a plain bool before returning.
    """

    name: str
    freq_mhz: float
    site: str = ""
    area: str = ""
    kind: str = "fm"            # fm | widefm | nhk | user
    legal_name: str = ""
    source: str = ""
    favorite: bool | None = None

    @property
    def freq_hz(self) -> float:
        """Frequency in Hz, as the tuner wants it."""
        return self.freq_mhz * 1e6

    @property
    def key(self) -> tuple[float, str]:
        """Identity of this entry within the catalogue."""
        return (round(self.freq_mhz, FREQ_DECIMALS), self.site)

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


def _make_reporter(warn: Reporter | None) -> Reporter:
    """Return a function that logs a problem and, optionally, shows it."""
    def report(message: str) -> None:
        logger.warning("%s", message)
        if warn is not None:
            warn(message)
    return report


def _dict_items(value: object, table: str,
                report: Reporter) -> list[tuple[int, dict]]:
    """Return the ``[[table]]`` entries as (position, entry) pairs.

    The position is where the entry sits in the file, counting the ones that
    were dropped: renumbering what survives would point every later
    diagnostic at the wrong rule.

    TOML lets ``station = 1`` parse happily, and iterating that raises.  A
    wrong type here means one mistyped line, so that entry is skipped and the
    rest of the file still applies.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        report(f"stations.toml: [[{table}]] must be a table array, "
               f"found {type(value).__name__}; ignoring it")
        return []
    entries: list[tuple[int, dict]] = []
    for position, entry in enumerate(value, start=1):
        if isinstance(entry, dict):
            entries.append((position, entry))
        else:
            report(f"stations.toml: [[{table}]] #{position} is a "
                   f"{type(entry).__name__}, not a table; ignoring it")
    return entries


# ----------------------------------------------------------------------
# Bundled snapshot
# ----------------------------------------------------------------------

def _load_bundled(path: Path, report: Reporter) -> list[Station]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        report(f"Could not read the bundled station list {path}: {exc}")
        return []

    if not isinstance(payload, dict):
        report(f"{path}: expected a JSON object, found "
               f"{type(payload).__name__}; no stations loaded")
        return []
    entries = payload.get("stations")
    if entries is None:
        report(f"{path}: no 'stations' key; no stations loaded")
        return []
    if not isinstance(entries, list):
        report(f"{path}: 'stations' must be a list, found "
               f"{type(entries).__name__}; no stations loaded")
        return []

    stations: list[Station] = []
    skipped = 0
    for entry in entries:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        freq_mhz = normalize_freq(entry.get("freq_mhz"))
        name = entry.get("name")
        if freq_mhz is None or not isinstance(name, str) or not name:
            skipped += 1
            continue
        stations.append(Station(
            name=name,
            freq_mhz=freq_mhz,
            site=str(entry.get("site", "")),
            area=str(entry.get("area", "")),
            kind=str(entry.get("kind", "fm")),
            legal_name=str(entry.get("legal_name", "")),
            source=str(entry.get("source", "")),
        ))
    if skipped:
        report(f"{path}: skipped {skipped} malformed station entries")
    return stations


# ----------------------------------------------------------------------
# User layer
# ----------------------------------------------------------------------

def _read_user_toml(path: Path, report: Reporter) -> dict:
    if tomllib is None:
        report(f"{path} exists but reading it needs Python 3.11+ (tomllib); "
               f"using the bundled station list only")
        return {}
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except OSError as exc:
        report(f"Could not read {path}: {exc}")
        return {}
    except Exception as exc:                    # tomllib.TOMLDecodeError
        report(f"{path} is not valid TOML ({exc}); ignoring it")
        return {}
    if not isinstance(payload, dict):           # pragma: no cover - tomllib
        report(f"{path}: expected a table at the top level; ignoring it")
        return {}
    return payload


def _as_bool(value: object, where: str, report: Reporter) -> bool | None:
    """Return *value* if it is a real bool, else None after reporting it.

    ``hidden = "false"`` is a non-empty string, which is truthy: taken at face
    value it hides every station the rule matches, which is the opposite of
    what was written.  Frequencies deliberately do accept strings; these two
    flags do not.
    """
    if isinstance(value, bool):
        return value
    report(f"stations.toml: {where} must be true or false, found "
           f"{value!r}; ignoring that setting")
    return None


def _station_from_toml(entry: dict, position: int,
                       report: Reporter) -> Station | None:
    freq_mhz = normalize_freq(entry.get("freq_mhz"))
    if freq_mhz is None:
        report(f"stations.toml: [[station]] #{position} needs a finite "
               f"freq_mhz, found {entry.get('freq_mhz')!r}; skipping it")
        return None
    if not (BAND_MIN_MHZ <= freq_mhz <= BAND_MAX_MHZ):
        report(f"stations.toml: [[station]] #{position} is at {freq_mhz} MHz, "
               f"outside the FM band ({BAND_MIN_MHZ}-{BAND_MAX_MHZ} MHz); "
               f"keeping it anyway")
    name = entry.get("name")
    name = str(name) if name else f"{freq_mhz:.1f} MHz"
    favorite = None
    if "favorite" in entry:
        favorite = _as_bool(entry["favorite"],
                            f"[[station]] #{position} favorite", report)
    return Station(
        name=name,
        freq_mhz=freq_mhz,
        site=str(entry.get("site", "")),
        area=str(entry.get("area", "")),
        kind=str(entry.get("kind", "user")),
        legal_name=str(entry.get("legal_name", "")),
        source="user",
        favorite=favorite,
    )


def _can_match(rule: dict) -> bool:
    """True if *rule* could match something, i.e. it is worth counting."""
    if "match_freq_mhz" in rule and normalize_freq(rule["match_freq_mhz"]) is None:
        return False
    return any(k in rule for k in ("match_name", "match_site", "match_freq_mhz"))


def _override_matches(station: Station, rule: dict) -> bool:
    """True if *rule* selects *station*.

    Every ``match_*`` key given must match.  A rule with no ``match_*`` key
    matches nothing: silently editing every station in the catalogue is never
    what was meant.
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
    if "match_freq_mhz" in rule:
        wanted = normalize_freq(rule["match_freq_mhz"])
        if wanted is None:
            return False                # reported once by _check_override
        checked = True
        if station.key[0] != wanted:
            return False
    return checked


def _clean_rule(rule: dict, position: int, report: Reporter) -> dict:
    """Return *rule* with unusable settings dropped, reporting each once.

    Done up front rather than per station: the same rule is tested against
    every entry in the catalogue, and a problem with the rule is a property
    of the rule, not of the 983 stations it was compared against.
    """
    cleaned = dict(rule)

    for field in ("hidden", "favorite"):
        if field in cleaned and _as_bool(
                cleaned[field], f"[[override]] #{position} {field}", report) is None:
            del cleaned[field]

    if "freq_mhz" in cleaned and normalize_freq(cleaned["freq_mhz"]) is None:
        report(f"stations.toml: [[override]] #{position} has a non-finite "
               f"freq_mhz ({cleaned['freq_mhz']!r}); the frequency is left alone")
        del cleaned["freq_mhz"]

    if "match_freq_mhz" in cleaned and normalize_freq(cleaned["match_freq_mhz"]) is None:
        report(f"stations.toml: [[override]] #{position} has a non-finite "
               f"match_freq_mhz ({cleaned['match_freq_mhz']!r}); it matches nothing")
    elif not any(k in cleaned for k in
                 ("match_name", "match_site", "match_freq_mhz")):
        report(f"stations.toml: [[override]] #{position} has no match_name / "
               f"match_site / match_freq_mhz; it matches nothing")
    return cleaned


def _apply_override(station: Station, rule: dict) -> Station:
    """Return *station* with the edits in *rule* applied.

    ``rule`` has been through :func:`_clean_rule`, so any setting still
    present here is of a usable type.
    """
    changes: dict[str, object] = {}
    for field in ("name", "site", "area", "kind", "legal_name"):
        if field in rule:
            changes[field] = str(rule[field])
    if "freq_mhz" in rule:
        freq_mhz = normalize_freq(rule["freq_mhz"])
        if freq_mhz is not None:
            changes["freq_mhz"] = freq_mhz
    if "favorite" in rule:
        changes["favorite"] = rule["favorite"]
    return replace(station, **changes) if changes else station


def _merge_user_layer(stations: list[Station], config: dict,
                      report: Reporter) -> list[Station]:
    """Apply ``[[override]]`` rules, then append ``[[station]]`` entries."""
    rules = [(position, _clean_rule(rule, position, report)) for position, rule
             in _dict_items(config.get("override"), "override", report)]

    if rules:
        match_counts = [0] * len(rules)
        merged: list[Station] = []
        for station in stations:
            # Match every rule against the bundled entry, so renaming a
            # station cannot change which later rules apply to it.
            matching = []
            hidden = False
            for slot, (_, rule) in enumerate(rules):
                if not _override_matches(station, rule):
                    continue
                match_counts[slot] += 1
                if rule.get("hidden"):
                    hidden = True              # hiding wins over editing
                else:
                    matching.append(rule)
            if hidden:
                continue
            for rule in matching:
                station = _apply_override(station, rule)
            merged.append(station)
        stations = merged

        # A rule that matches nothing is almost always a typo, or a station
        # the upstream list renamed; either way its effect is simply missing.
        for slot, (position, rule) in enumerate(rules):
            if match_counts[slot] == 0 and _can_match(rule):
                report(f"stations.toml: [[override]] #{position} matched no "
                       f"station in the bundled catalogue; check the name, "
                       f"site or frequency")

    for position, entry in _dict_items(config.get("station"), "station", report):
        added = _station_from_toml(entry, position, report)
        if added is None:
            continue
        # A user entry replaces the bundled one it collides with, so that
        # correcting a transmitter does not leave a duplicate behind.
        stations = [s for s in stations if s.key != added.key]
        stations.append(added)
    return stations


def _resolve_favorites(stations: list[Station]) -> list[Station]:
    """Turn the tri-state ``favorite`` into a plain bool.

    ``favorite = true`` anywhere means the user is listing their own presets,
    so the shipped list steps aside entirely.  Otherwise the shipped list
    applies, minus anything the user switched off with ``favorite = false`` —
    which is how all ten can be dropped without naming a replacement.
    """
    if any(s.favorite is True for s in stations):
        return [replace(s, favorite=s.favorite is True) for s in stations]
    wanted = {(round(freq, FREQ_DECIMALS), site) for freq, site in DEFAULT_FAVORITES}
    return [replace(s, favorite=(s.key in wanted and s.favorite is not False))
            for s in stations]


def load_stations(user_path: Path | str | None = None,
                  data_path: Path | str | None = None,
                  warn: Reporter | None = None) -> list[Station]:
    """Load the catalogue: bundled snapshot merged with the user's TOML.

    Args:
        user_path: Explicit ``stations.toml`` location.  ``None`` uses
            :func:`user_config_path`; a path that does not exist is not an
            error (most users never create one).
        data_path: Override for the bundled JSON, for tests.
        warn: Called once per problem found while loading, with a message
            meant for the user.  Problems are always logged as well; this is
            how they reach someone running without ``--log``.

    Returns:
        Every transmitter, sorted by area then frequency, each with
        ``favorite`` resolved to a bool - including the ones the
        receiver cannot possibly hear, because looking a station up
        is not the same question as naming the one on the dial; see
        :func:`here`.  Never raises: bad input is reported and
        skipped rather than stopping the receiver from starting.
    """
    report = _make_reporter(warn)
    stations = _load_bundled(Path(data_path) if data_path else DATA_PATH, report)

    path = Path(user_path) if user_path is not None else user_config_path()
    try:
        present = path.exists()
    except OSError as exc:
        # A permission error or an unreachable network path must not be the
        # thing that stops the receiver from starting.
        report(f"Could not check for {path}: {exc}; "
               f"using the bundled station list only")
        present = False
    if present:
        config = _read_user_toml(path, report)
        if config:
            before = len(stations)
            stations = _merge_user_layer(stations, config, report)
            logger.info("Applied %s (%d -> %d stations)", path, before, len(stations))

    stations = _resolve_favorites(stations)
    stations.sort(key=lambda s: (
        AREAS.index(s.area) if s.area in AREAS else len(AREAS),
        s.freq_mhz, s.name,
    ))
    return stations


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------

def favorites(stations: Iterable[Station]) -> list[Station]:
    """Return the preset stations, ordered by frequency."""
    return sorted((s for s in stations if s.favorite), key=lambda s: s.freq_mhz)


def search(stations: Iterable[Station], query: str) -> list[Station]:
    """Return the stations matching *query* (name, legal name, site, area, MHz)."""
    return [s for s in stations if s.matches(query)]


def in_area(stations: Iterable[Station], area: str) -> list[Station]:
    """Return the stations in *area*, compared case-insensitively.

    Areas in the bundled data are Japanese and unaffected by case, but a user
    entry may name its own area in Latin script, and the CLI lower-cases what
    it reads.
    """
    folded = area.strip().casefold()
    return [s for s in stations if s.area.casefold() == folded]


def home_area(user_path: Path | str | None = None,
              warn: Reporter | None = None) -> str | None:
    """The area the user's file says the receiver is in, or None.

    Its own reader rather than a second return value from
    :func:`load_stations`, because it answers a different question:
    that one is "what stations exist", this one is "where is this
    radio".  Only the naming path needs the second.

    An area that is not one of :data:`AREAS` is reported and treated
    as unset.  A line in a file must not leave the receiver unable
    to name anything at all.
    """
    report = _make_reporter(warn)
    path = Path(user_path) if user_path is not None else user_config_path()
    try:
        if not path.exists():
            return None
    except OSError as exc:
        report(f"Could not check for {path}: {exc}")
        return None
    config = _read_user_toml(path, report)
    wanted = config.get("area") if config else None
    if wanted is None:
        return None
    if not isinstance(wanted, str) or not wanted.strip():
        report("stations.toml: area must be the name of one, found %r"
               % (wanted,))
        return None
    wanted = wanted.strip()
    if wanted not in AREAS:
        report("stations.toml: %r is not an area; the areas are %s"
               % (wanted, ", ".join(AREAS)))
        return None
    return wanted


def here(stations: Iterable[Station], area: str | None) -> list[Station]:
    """The entries that may put a name on this receiver's dial.

    Everything when no area is set, and otherwise that area plus two
    kinds of entry that are the user's own: a ``[[station]]`` they
    added, whatever area it carries, because they put it there; and
    anything with no area at all, because the bundled data always
    has one.

    Not a filter on the catalogue itself.  ``list 北海道`` is a
    question about Japan and this is a question about the dial, and
    a receiver that would not look up a station a thousand
    kilometres away is less useful, not more.

    Falls back to everything if the area turns out to hold nothing,
    which would otherwise leave the receiver unable to name any
    station over a line in a file.
    """
    stations = list(stations)
    if area is None:
        return stations
    mine = [s for s in stations
            if s.area == area or not s.area or s.source == "user"]
    if not mine:
        logger.warning("Area %s holds no stations; naming from all of them",
                       area)
        return stations
    return mine


def nearest(stations: Iterable[Station], freq_hz: float,
            tolerance_hz: float = 50e3) -> Station | None:
    """Return the catalogue entry closest to *freq_hz*, or None if none is near.

    Used to put a name on whatever the tuner is currently sitting on.  A
    frequency is not unique nationwide — 80.0 MHz is TOKYO FM in 関東 and
    RKKラジオ in 熊本 — and the receiver cannot know where it is, so ties are
    broken in favour of a preset: the user's own favourites are the only
    evidence available about which transmitter they can actually hear.
    Remaining ties keep catalogue order, so the result is stable.
    """
    best: Station | None = None
    best_delta = tolerance_hz
    for station in stations:
        delta = abs(station.freq_hz - freq_hz)
        if not math.isfinite(delta) or delta > tolerance_hz:
            continue
        if (best is None or delta < best_delta
                or (delta == best_delta and station.favorite and not best.favorite)):
            best, best_delta = station, delta
    return best
