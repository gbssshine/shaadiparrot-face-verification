"""content-visuals Part E (good dates: Roka, nakshatra/tithi chips, the season note, the combust wording), the
question-wording table (engagement N3), the chart's transits, moon.options / moon.vh (viral §5.6a) and the
approximate birth time (engagement N1)."""
from datetime import date, datetime, timedelta, timezone

import astro_content
import astro_core as ac
import fates_astro
import fates_astro_chart as fc

DELHI = (28.6139, 77.209, "Asia/Kolkata")
NOW = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ Part E: good dates
def test_engagement_days_sit_inside_its_sets():
    ev = ac.EVENTS["engagement"]
    assert ev["title"] == "Roka / engagement" and ev["season"] and not ev["planets"]
    assert ev["naks"] == ac.EVENTS["wedding"]["naks"] and ev["tithis"] == ac.EVENTS["wedding"]["tithis"]
    assert ev["days"] == {0, 2, 3, 4}
    g = ac.good_days("engagement", date(2026, 11, 21), 90, *DELHI)
    assert g["event"] == "engagement" and g["title"] == "Roka / engagement"
    assert g["days"], g
    for d in g["days"]:
        assert d["naks"] and set(d["naks"]) <= ev["naks"], d
        assert d["tithis"] and set(d["tithis"]) <= ev["tithis"], d
        assert d["naks"] == sorted(d["naks"]) and d["tithis"] == sorted(d["tithis"])


def test_every_found_day_carries_its_nakshatras_and_tithis():
    g = ac.good_days("wedding", date(2026, 11, 25), 1, *DELHI)
    day = g["days"][0]
    # Rohini (3) then Mrigashira (4); Krishna Pratipada (16) then Dwitiya (17): the reasons say the same in words.
    assert day["naks"] == [3, 4] and day["tithis"] == [16, 17]
    assert day["reasons"][0] == "Rohini and Mrigashira nakshatras"


def test_season_note_even_when_days_are_found():
    g = ac.good_days("wedding", date(2026, 10, 11), 90, *DELHI)
    assert g["days"] and all(d["date"] > "2026-11-20" for d in g["days"])
    assert "Chaturmas" in (g["seasonNote"] or "")
    assert g["blockedNote"] is None   # unchanged: only when nothing was found
    # The Roka search sees the same pause.
    assert "Chaturmas" in (ac.good_days("engagement", date(2026, 10, 11), 90, *DELHI)["seasonNote"] or "")


def test_no_season_note_without_a_pause():
    # Naming has no season rule; a wedding search inside the open season meets no pause.
    assert ac.good_days("naming", date(2026, 10, 11), 60, *DELHI)["seasonNote"] is None
    assert ac.good_days("wedding", date(2026, 11, 21), 20, *DELHI)["seasonNote"] is None


def test_blocked_note_is_unchanged_when_nothing_is_found():
    g = ac.good_days("wedding", date(2026, 10, 11), 30, *DELHI)
    assert g["days"] == []
    assert g["blockedNote"].startswith("Chaturmas") and g["seasonNote"] == g["blockedNote"]


def test_unknown_event_still_falls_back():
    g = ac.good_days("space_launch", date(2026, 11, 21), 30, *DELHI)
    assert g["event"] == "griha_pravesh" and g["title"] == ac.EVENTS["griha_pravesh"]["title"]
    assert "seasonNote" in g


def test_combust_reason_wording():
    ev = dict(ac.EVENTS["wedding"], season=False)
    start = ac.jd_of(datetime(2026, 10, 11, tzinfo=timezone.utc))
    reasons = {ac._blocked(start + i, ev) for i in range(0, 700, 3)} - {None}
    assert reasons, "no combust spell in two years"
    for r in reasons:
        assert r.endswith("is too close to the Sun (combust): families traditionally skip these weeks"), r
        assert r.split(" ")[0] in ("Venus", "Jupiter")
    assert "Venus is too close to the Sun (combust): families traditionally skip these weeks" in reasons


def test_koota_names_are_the_eight_server_names():
    assert list(fates_astro.KOOTA_MAX) == ["Varna", "Vashya", "Tara", "Yoni", "Graha Maitri", "Gana", "Bhakoot", "Nadi"]


# ------------------------------------------------------------------ N3: question wording
def test_question_wording_for_every_faith():
    q = {x["id"]: x for x in astro_content.QUESTIONS}
    assert q["morning_first"]["options"] == ["Check my phone", "Pray or meditate", "Exercise", "Wake the kids"]
    assert q["anniversary"]["options"] == ["Dinner out", "A gift", "A prayer or puja together", "We forget"]
    assert q["holiday_spot"]["options"] == ["Hills", "Beach", "The village home", "A pilgrimage"]
    assert q["vrat"]["text"] == "Do you keep fasts (vrat, roza, upvas)?"
    # Option indexes don't move, so the stored vote counts stay valid.
    assert q["vrat"]["options"] == ["Regularly", "On festivals", "Never"]
    assert len(astro_content.QUESTIONS) == 42 and len(q) == 42


# ------------------------------------------------------------------ transits
def test_transits_sample():
    # 21 Aug 1994 without a time: Moon in Kumbha. Saturn is in Meena (the 2nd from Kumbha): Sade Sati phase 3,
    # until Saturn leaves Meena for good (after its 2027 return).
    c = ac.birth_chart((1994, 8, 21), None, None, now=NOW)
    assert c["moon"]["rashiName"] == "Kumbha"
    sade, jup = c["transits"]
    assert sade == {"kind": "sade_sati", "title": "Sade Sati · phase 3 of 3", "phase": 3,
                    "text": "Slow, steady years: effort pays.", "sign": 11, "signName": "Meena",
                    "until": "2028-02-23", "approximate": True}
    assert jup["kind"] == "jupiter" and jup["date"] == "2026-10-31" and jup["signName"] == "Simha"
    assert jup["house"] == 7 and jup["title"] == "Jupiter moves into Simha · 31 Oct 2026"
    assert jup["text"] == "Your 7th house from the Moon: a warm year for marriage and partnerships begins."
    assert "phase" not in jup and "until" not in jup   # never null: the app's ints stay safe


def test_sade_sati_phases_by_moon_sign():
    phases = {r: next((t["phase"] for t in ac.transits(r, NOW) if t["kind"] == "sade_sati"), None) for r in range(12)}
    # Saturn in Meena (11): Moon in Mesha = 12th (phase 1), Meena = 1st (phase 2), Kumbha = 2nd (phase 3).
    assert phases == {0: 1, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None,
                      10: 3, 11: 2}
    for r in range(12):
        for t in ac.transits(r, NOW):
            words = (t["title"] + " " + t["text"]).lower()
            assert "dosha" not in words and "bad" not in words and "danger" not in words


def test_next_jupiter_after_the_ingress():
    later = ac.transits(4, datetime(2026, 11, 2, tzinfo=timezone.utc))   # Moon in Simha
    jup = next(t for t in later if t["kind"] == "jupiter")
    assert jup["date"] == "2027-01-25" and jup["signName"] == "Karka" and jup["house"] == 12


def test_exact_chart_has_sure_transits():
    place = {"lat": DELHI[0], "lon": DELHI[1], "tz": DELHI[2], "precision": "city"}
    c = ac.birth_chart((1990, 5, 15), (10, 30), place, now=NOW)
    assert c["transits"] and not any(t.get("approximate") for t in c["transits"])


# ------------------------------------------------------------------ moon.options and moon.vh
def _moon_day(rashi):
    d = date(2000, 1, 1)
    while True:
        c = ac.birth_chart((d.year, d.month, d.day), (12, 0), None, now=NOW)
        if c["moon"]["rashi"] == rashi:
            return d, c
        d += timedelta(days=1)


def test_moon_options_without_a_time():
    c = ac.birth_chart((1994, 8, 21), None, None, now=NOW)
    opts = c["moon"]["options"]
    assert opts[0] == [c["moon"]["rashi"], c["moon"]["nak"]]
    assert len(opts) == len({tuple(o) for o in opts})
    # The same scan as the match and the site.
    same = fc.chart_from_birth((1994, 8, 21), None, None)["options"]
    assert sorted(map(tuple, opts)) == sorted(map(tuple, same))


def test_no_options_with_a_time():
    c = ac.birth_chart((1994, 8, 21), (7, 40), None, now=NOW)
    assert "options" not in c["moon"]


def test_moon_half_only_for_dhanu_and_makara():
    for rashi in (8, 9):
        d, c = _moon_day(rashi)
        lon = c["planets"][1]["lon"]
        assert c["moon"]["vh"] == (0 if lon % 30 < 15 else 1), (d, lon)
    _, other = _moon_day(4)
    assert other["moon"]["vh"] is None
    assert ac.moon_half(8, 240.0) == 0 and ac.moon_half(8, 255.0) == 1 and ac.moon_half(9, 284.9) == 0
    assert ac.moon_half(9, 285.0) == 1 and ac.moon_half(3, 100.0) is None


# ------------------------------------------------------------------ N1: approximate birth time
def test_approximate_birth_time_draws_the_moon_only():
    place = {"lat": DELHI[0], "lon": DELHI[1], "tz": DELHI[2], "precision": "city"}
    for key, hm in (("morning", (9, 0)), ("afternoon", (14, 30)), ("evening", (19, 0)), ("night", (3, 0))):
        c = ac.birth_chart((1994, 8, 21), None, place, now=NOW, approx=key)
        timed = ac.birth_chart((1994, 8, 21), hm, place, now=NOW)
        assert c["precision"] == "approximate" and c["timeApprox"] == key
        assert c["lagna"] is None and c["basis"] == "moon" and not c["timeKnown"] and c["dashaApproximate"]
        assert c["planets"][1]["lon"] == timed["planets"][1]["lon"]   # the Moon is taken at that clock time
        assert timed["lagna"] is not None                             # a real time would give a Lagna
        assert c["moon"]["options"][0] == [c["moon"]["rashi"], c["moon"]["nak"]]
        assert c["lines"][0]["title"] == "Rising sign: add your birth time"


def test_approximate_time_never_beats_a_real_time_or_counts_when_unknown():
    real = ac.birth_chart((1994, 8, 21), (7, 40), None, now=NOW, approx="night")
    assert real["precision"] == "time" and "timeApprox" not in real
    assert real == ac.birth_chart((1994, 8, 21), (7, 40), None, now=NOW)
    for junk in (None, "", "dawn", 5):
        c = ac.birth_chart((1994, 8, 21), None, None, now=NOW, approx=junk)
        assert c["precision"] == "date" and "timeApprox" not in c
    noon = ac.birth_chart((1994, 8, 21), None, None, now=NOW)
    assert noon["planets"][1]["lon"] == ac.birth_chart((1994, 8, 21), (12, 0), None, now=NOW)["planets"][1]["lon"]
    assert ac.approx_time(" Evening ") == (19, 0) and ac.approx_time("noonish") is None
