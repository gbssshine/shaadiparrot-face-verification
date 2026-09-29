"""Build places.sqlite3 (birth place -> lat/lon/timezone) from the app's bundled location DBs.

The app stores birthCityName / birthStateName / birthCountryIso2 picked from these same DBs,
so the server can resolve a birth place without a geocoding API.

Usage: python tools/build_places_db.py [path/to/MauiApp2/Resources/Raw] [out.sqlite3]
"""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.normpath(os.path.join(HERE, "..", "..", "MauiApp2", "MauiApp2", "Resources", "Raw"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "places.sqlite3"))


def norm(s):
    return " ".join(str(s or "").casefold().split())


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    def ro(name):
        return sqlite3.connect(f"file:{os.path.join(raw, name)}?mode=ro", uri=True)

    src_countries, src_states, src_cities = ro("countries.sqlite3"), ro("states.sqlite3"), ro("cities.sqlite3")

    if os.path.exists(out):
        os.remove(out)
    dst = sqlite3.connect(out)
    dst.executescript("""
        CREATE TABLE countries(cc TEXT PRIMARY KEY, name TEXT NOT NULL, lat REAL, lon REAL, tz TEXT);
        CREATE TABLE states(id INTEGER PRIMARY KEY, cc TEXT NOT NULL, state TEXT NOT NULL, lat REAL, lon REAL, tz TEXT);
        CREATE TABLE cities(cc TEXT NOT NULL, city TEXT NOT NULL, state_id INTEGER, lat REAL NOT NULL, lon REAL NOT NULL, tz TEXT, pop INTEGER);
    """)

    for iso2, name, lat, lon, tzs in src_countries.execute("SELECT iso2, name, latitude, longitude, timezones FROM countries"):
        if not iso2:
            continue
        tz = None
        try:
            zones = json.loads(tzs or "[]")
            tz = zones[0].get("zoneName") if zones else None
        except ValueError:
            pass
        dst.execute("INSERT OR REPLACE INTO countries VALUES (?,?,?,?,?)",
                    (iso2.upper(), norm(name), lat, lon, tz))

    for sid, cc, name, lat, lon, tz in src_states.execute(
            "SELECT id, country_code, name, latitude, longitude, timezone FROM states"):
        dst.execute("INSERT INTO states VALUES (?,?,?,?,?,?)",
                    (sid, (cc or "").upper(), norm(name), lat, lon, tz))

    rows = src_cities.execute(
        "SELECT country_code, name, state_id, latitude, longitude, timezone, population FROM cities")
    dst.executemany("INSERT INTO cities VALUES (?,?,?,?,?,?,?)", (
        ((cc or "").upper(), norm(name), sid, round(float(lat), 4), round(float(lon), 4), tz, pop)
        for cc, name, sid, lat, lon, tz, pop in rows
    ))

    dst.executescript("""
        CREATE INDEX ix_cities ON cities(cc, city);
        CREATE INDEX ix_states ON states(cc, state);
    """)
    dst.commit()
    counts = [dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("countries", "states", "cities")]
    dst.execute("VACUUM")
    dst.close()
    print(f"wrote {out}: countries={counts[0]} states={counts[1]} cities={counts[2]} "
          f"size={os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
