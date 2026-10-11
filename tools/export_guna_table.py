"""Export the 36-guna table getRishtas scores with, and the TS parity fixture (matching §2/§3).

    python tools/export_guna_table.py --out-dir <MauiApp2>/firebase-functions/src
                                      [--fixture <MauiApp2>/firebase-tests/fixtures/guna_pairs_fixture.json]

Writes <out-dir>/gunaTable.json and the fixture (default: tests/guna_pairs_fixture.json in this repo). Both are
generated, never edited by hand, and byte-identical on every run (fixed seed, fixed order), so TypeScript and
Python can never drift apart.

gunaTable.json
    {"v": 1, "keys": ["r_n_h", ...], "total": [[...], ...]}
    keys: every Moon key [rashi, nak, half] a chart can produce (fates_astro.moon_keys): every nakshatra in every
    sign it touches, half 0; Dhanu (8) and Makara (9) also with half 1 (the Moon past 15 deg: the Vashya split).
    total[i][j]: the Ashtakoota points of the groom keys[i] with the bride keys[j] (fates_astro.koota_points).

How getRishtas reproduces fates_astro.match_charts(a, b, aGender, bGender) with it (guna.ts):
    roles = gender_roles(aGender, bGender) (one direction, or both averaged for the same or unknown genders);
    points(x, y) = mean over roles of (total[x][y] if a is the groom else total[y][x]);
    total = round_half_even(2 * points(a.main, b.main)) / 2      (Python's round is half-to-even)
    totalMin / totalMax = min / max of points over a.options x b.options, keys [r, n, 0], NOT rounded.

guna_pairs_fixture.json
    {"v": 1, "seed": ..., "pairs": [{"a": {"main", "options"}, "b": {...}, "aGender", "bGender", "total",
    "totalMin", "totalMax"}, ...]}: 300 pairs with match_charts' own answers, including same and unknown
    genders (both directions averaged, x.5 means), Dhanu/Makara halves and Moon options. A .25 mean can't occur
    with today's tables (see fixture() step 2); guna.ts still rounds half-to-even like Python.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fates_astro as fa  # noqa: E402

SEED = 20261011
PAIRS = 300
NAK_SPAN = 360.0 / 27.0
GENDERS = ["male", "female"]
ODD_GENDERS = [("male", "male"), ("female", "female"), ("", ""), ("male", ""), ("", "female"), ("other", "male"),
               ("female", "other")]


def valid_keys() -> List[Tuple[int, int, int]]:
    keys = []
    for r in range(12):
        for n in range(27):
            if any((n * 4 + p) // 9 == r for p in range(4)):    # the nakshatra has a pada in this sign
                keys.append((r, n, 0))
                if r in (8, 9):
                    keys.append((r, n, 1))
    return keys


def _moon(key: Tuple[int, int, int]) -> Dict[str, Any]:
    r, n, h = key
    return {"rashi": r, "nak": n, "lon": None if not h else r * 30.0 + 22.5}


def _num(x: float) -> Any:
    return int(x) if float(x).is_integer() else x


def table() -> Dict[str, Any]:
    keys = valid_keys()
    total = [[_num(sum(fa.koota_points(_moon(b), _moon(g)).values())) for g in keys] for b in keys]
    return {"v": 1, "keys": [f"{r}_{n}_{h}" for r, n, h in keys], "total": total}


def _chart(rng: random.Random, lon: float, no_time: bool) -> Dict[str, Any]:
    lon = round(lon % 360.0, 4)
    r, n = int(lon // 30.0) % 12, int(lon // NAK_SPAN) % 27
    options = [[r, n]]
    if no_time:   # the day's other Moons, in the order fates_astro_chart lists them (the main one first)
        for d in (-6.4, -3.1, 3.3, 6.6):
            if rng.random() < 0.7:
                x = (lon + d) % 360.0
                o = [int(x // 30.0) % 12, int(x // NAK_SPAN) % 27]
                if o not in options:
                    options.append(o)
    return {"rashi": r, "nak": n, "moonLon": lon, "options": options, "precision": "date" if no_time else "exact"}


def _pair(a: Dict[str, Any], b: Dict[str, Any], ga: str, gb: str) -> Dict[str, Any]:
    m = fa.match_charts(a, b, ga, gb)
    ka, kb = fa.moon_keys(a), fa.moon_keys(b)
    return {"a": {"main": ka["main"], "options": ka["options"]}, "b": {"main": kb["main"], "options": kb["options"]},
            "aGender": ga, "bGender": gb, "total": _num(m["total"]), "totalMin": _num(m["totalMin"]),
            "totalMax": _num(m["totalMax"])}


def raw_runs(a: Dict[str, Any], b: Dict[str, Any], ga: str, gb: str) -> List[float]:
    """match_charts' main points per gender role, before the mean and the rounding."""
    ma = {"rashi": a["rashi"], "nak": a["nak"], "lon": a.get("moonLon")}
    mb = {"rashi": b["rashi"], "nak": b["nak"], "lon": b.get("moonLon")}
    return [sum((fa.koota_points(ma, mb) if boy else fa.koota_points(mb, ma)).values()) for boy in fa.gender_roles(ga, gb)]


def fixture() -> Dict[str, Any]:
    rng = random.Random(SEED)
    pairs: List[Dict[str, Any]] = []

    def any_chart(no_time_share: float) -> Dict[str, Any]:
        return _chart(rng, rng.uniform(0, 360), rng.random() < no_time_share)

    # 1) Dhanu and Makara Moons on both sides of 15 deg (the Vashya split), as the first or the second person.
    for i in range(40):
        r, half = 8 + i % 2, (i // 2) % 2
        a = _chart(rng, r * 30.0 + (rng.uniform(0.2, 14.8) if half == 0 else rng.uniform(15.2, 29.8)), rng.random() < 0.3)
        b = any_chart(0.3)
        ga, gb = ("male", "female") if i % 3 else ("female", "male")
        pairs.append(_pair(a, b, ga, gb) if (i // 4) % 2 else _pair(b, a, gb, ga))
    # 2) Same or unknown genders: 30 where both directions are averaged and differ (Gana, Varna and some Yoni
    #    cells are one-sided), 40 others. With today's tables the mean of two directions is always a multiple of
    #    0.5 (Vashya and Graha Maitri are symmetric): a .25 average can't happen, and test_astro checks that over
    #    every pair of keys, so the half-to-even rounding never changes a total today.
    averaged = others = 0
    while averaged < 30 or others < 40:
        ga, gb = ODD_GENDERS[rng.randrange(len(ODD_GENDERS))]
        a, b = any_chart(0.5), any_chart(0.5)
        runs = raw_runs(a, b, ga, gb)
        if len(runs) == 2 and runs[0] != runs[1]:
            if averaged < 30:
                pairs.append(_pair(a, b, ga, gb))
                averaged += 1
        elif others < 40:
            pairs.append(_pair(a, b, ga, gb))
            others += 1
    # 3) Everyone else at random, mostly a groom and a bride.
    while len(pairs) < PAIRS:
        ga, gb = (GENDERS[rng.randrange(2)], GENDERS[rng.randrange(2)]) if rng.random() < 0.85 else \
            ODD_GENDERS[rng.randrange(len(ODD_GENDERS))]
        pairs.append(_pair(any_chart(0.4), any_chart(0.4), ga, gb))
    return {"v": 1, "seed": SEED, "note": "Generated by shaadiparrot-face-verification/tools/export_guna_table.py. "
            "Never edit by hand.", "pairs": pairs}


def dump_table(t: Dict[str, Any]) -> str:
    rows = ",\n".join("  " + json.dumps(row, separators=(",", ":")) for row in t["total"])
    return ('{"v":1,"note":"Generated by shaadiparrot-face-verification/tools/export_guna_table.py. Never edit by hand.",\n'
            f' "keys":{json.dumps(t["keys"], separators=(",", ":"))},\n "total":[\n{rows}\n ]}}\n')


def dump_fixture(f: Dict[str, Any]) -> str:
    rows = ",\n".join("  " + json.dumps(p, separators=(",", ":")) for p in f["pairs"])
    return (f'{{"v":{f["v"]},"seed":{f["seed"]},"note":{json.dumps(f["note"])},\n "pairs":[\n{rows}\n ]}}\n')


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="where gunaTable.json goes (MauiApp2/firebase-functions/src)")
    ap.add_argument("--fixture", default=str(ROOT / "tests" / "guna_pairs_fixture.json"),
                    help="the parity fixture path (default: tests/guna_pairs_fixture.json in this repo)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path, fixture_path = out_dir / "gunaTable.json", Path(args.fixture)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_bytes(dump_table(table()).encode("utf-8"))
    fixture_path.write_bytes(dump_fixture(fixture()).encode("utf-8"))
    print(f"wrote {table_path} ({table_path.stat().st_size} bytes) and {fixture_path} ({PAIRS} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
