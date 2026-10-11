"""Manglik status (audit 9, plan C3): partial from the Moon or Venus, the common cancellations, 'check' without a
birth time, and the match row's status."""
import json
import os
import random
import re
from collections import Counter

import pytest

import astro_api
import astro_core as ac
import fates_astro as fa

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "manglik_fixtures.json"), encoding="utf-8"))
SIGN = {r[0]: i for i, r in enumerate(fa.RASHIS)}
BLR = {"lat": 12.97, "lon": 77.59, "tz": "Asia/Kolkata", "precision": "city"}


def chart(mars, lagna=None, moon=None, venus=None, jupiter=None):
    """Mars in a sign; the references default to signs that put Mars in the 3rd house (never Manglik)."""
    third = (mars - 2) % 12
    c = {"marsRashi": mars, "lagna": lagna, "rashi": third if moon is None else moon,
         "venusRashi": third if venus is None else venus}
    if jupiter is not None:
        c["jupiterRashi"] = jupiter
    return c


def ref_for(mars, house):
    """The reference sign from which Mars sits in that house."""
    return (mars - house + 1) % 12


# ---------------------------------------------------------------- one chart
def test_status_per_case():
    mars = SIGN["Mithuna"]   # not own, not exalted
    # Lagna known: 7th from the Lagna is a full Manglik.
    s = fa.manglik_status(chart(mars, lagna=ref_for(mars, 7)))
    assert s["status"] == "yes" and s["sure"] is True and s["manglik"] and s["from"] == ["Lagna"]
    assert s["cancelReasons"] == [] and "cancelledBy" not in s
    for h in (1, 4, 8, 12):
        assert fa.manglik_status(chart(mars, lagna=ref_for(mars, h)))["status"] == "yes", h
    # Only the 2nd house from the Lagna: partial.
    s2 = fa.manglik_status(chart(SIGN["Simha"], lagna=ref_for(SIGN["Simha"], 2)))
    assert s2["status"] == "partial" and s2["mild"]
    # Lagna clear, the Moon or Venus Manglik: partial.
    assert fa.manglik_status(chart(mars, lagna=ref_for(mars, 3), moon=ref_for(mars, 4)))["status"] == "partial"
    assert fa.manglik_status(chart(mars, lagna=ref_for(mars, 3), venus=ref_for(mars, 8)))["status"] == "partial"
    # The Lagna wins over a 2nd house elsewhere: 7th from the Lagna and 2nd from the Moon is a full yes.
    assert fa.manglik_status(chart(mars, lagna=ref_for(mars, 7), moon=ref_for(mars, 2)))["status"] == "yes"
    # Nothing anywhere with the Lagna known: no, and nothing was cancelled.
    none = fa.manglik_status(chart(mars, lagna=ref_for(mars, 3)))
    assert none["status"] == "no" and none["cancelReasons"] == [] and "cancelledBy" not in none and not none["manglik"]


def test_check_without_a_birth_time():
    mars = SIGN["Mithuna"]
    moon_only = fa.manglik_status(chart(mars, moon=ref_for(mars, 7)))       # no Lagna
    assert moon_only["status"] == "check" and moon_only["sure"] is False
    assert moon_only["manglik"] is True and moon_only["from"] == ["Moon"]  # 1.0.7 apps keep the old reading
    nothing = fa.manglik_status(chart(mars))
    assert nothing["status"] == "check" and not nothing["manglik"]
    # Without a time even a cancellation that would hold whatever the Lagna is stays a check (C3 row 1): its
    # reason is listed, but there is no verdict, so no cancelledBy.
    own = fa.manglik_status(chart(SIGN["Vrischika"], moon=ref_for(SIGN["Vrischika"], 7)))
    assert own["status"] == "check" and own["sure"] is False and "cancelledBy" not in own
    assert own["cancelReasons"] == ["Mars in its own or exalted sign"] and not own["manglik"]
    jup = fa.manglik_status(chart(mars, jupiter=mars))
    assert jup["status"] == "check" and jup["cancelReasons"] == ["Jupiter with or aspecting Mars"]
    assert "cancelledBy" not in jup
    # A house-sign exception from the Moon without a time: listed, still a check.
    exc = fa.manglik_status(chart(SIGN["Karka"], moon=ref_for(SIGN["Karka"], 7)))
    assert exc["status"] == "check" and "cancelledBy" not in exc
    assert exc["cancelReasons"] == ["Mars's sign is an accepted exception for its house from the Moon"]
    # No Mars at all (an old cached chart): check, never a verdict.
    assert fa.manglik_status({"rashi": 3})["status"] == "check"


def test_own_or_exalted_sign_cancels():
    for mars in (SIGN["Mesha"], SIGN["Vrischika"], SIGN["Makara"]):
        s = fa.manglik_status(chart(mars, lagna=ref_for(mars, 7), moon=ref_for(mars, 8)))
        assert s["status"] == "no" and not s["manglik"], mars
        assert s["cancelledBy"] == "Mars in its own or exalted sign"
        assert s["cancelReasons"][0] == "Mars in its own or exalted sign"


@pytest.mark.parametrize("house,sign", [(1, "Mesha"), (2, "Mithuna"), (2, "Kanya"), (4, "Mesha"), (4, "Vrischika"),
                                        (7, "Karka"), (7, "Makara"), (8, "Dhanu"), (8, "Meena"), (12, "Vrishabha"),
                                        (12, "Tula")])
def test_house_sign_exceptions(house, sign):
    mars = SIGN[sign]
    s = fa.manglik_status(chart(mars, lagna=ref_for(mars, house)))
    assert s["status"] == "no" and not s["manglik"], (house, sign)
    assert "Mars's sign is an accepted exception for its house from the Lagna" in s["cancelReasons"]
    assert s["cancelledBy"]


def test_an_exception_drops_only_its_own_house():
    mars = SIGN["Karka"]
    # 7th from the Lagna in Karka is excepted; 1st from the Moon in Karka is not: Moon-only, so partial.
    s = fa.manglik_status(chart(mars, lagna=ref_for(mars, 7), moon=ref_for(mars, 1)))
    assert s["status"] == "partial" and s["manglik"] and s["from"] == ["Moon"]
    assert s["cancelReasons"] == ["Mars's sign is an accepted exception for its house from the Lagna"]
    # Not an exception: Karka in the 8th from the Lagna.
    assert fa.manglik_status(chart(mars, lagna=ref_for(mars, 8)))["status"] == "yes"


def test_jupiter_with_or_aspecting_mars_cancels():
    mars = SIGN["Mithuna"]
    for house in (1, 5, 7, 9):   # Mars in these houses from Jupiter: together, or Jupiter's 5th/7th/9th aspect
        s = fa.manglik_status(chart(mars, lagna=ref_for(mars, 7), jupiter=ref_for(mars, house)))
        assert s["status"] == "no" and s["cancelledBy"] == "Jupiter with or aspecting Mars", house
    for house in (2, 3, 4, 6, 8, 10, 11, 12):
        s = fa.manglik_status(chart(mars, lagna=ref_for(mars, 7), jupiter=ref_for(mars, house)))
        assert s["status"] == "yes", house
    assert fa.manglik_status(chart(mars, lagna=ref_for(mars, 7)))["status"] == "yes"   # Jupiter unknown: no cancellation


def test_reasons_never_name_a_sign_or_a_house():
    rng = random.Random(9)
    names = {r[0] for r in fa.RASHIS} | {r[1] for r in fa.RASHIS}
    for _ in range(3000):
        c = chart(rng.randrange(12), lagna=rng.choice([None, rng.randrange(12)]), moon=rng.randrange(12),
                  venus=rng.randrange(12), jupiter=rng.randrange(12))
        s = fa.manglik_status(c)
        assert s["status"] in ("yes", "partial", "no", "check")
        for r in s["cancelReasons"]:
            assert not re.search(r"\d", r) and not any(n in r for n in names), r
        assert (s["status"] == "check") == (c["lagna"] is None)    # no Lagna: always a check, never a verdict
        if s["status"] == "check":
            assert "cancelledBy" not in s and s["sure"] is False
        if s["status"] in ("yes", "partial"):
            assert s["manglik"] and s["sure"]


# ---------------------------------------------------------------- the match row
def test_dosha_statuses():
    mars = SIGN["Mithuna"]
    yes = chart(mars, lagna=ref_for(mars, 7))
    partial = chart(mars, lagna=ref_for(mars, 3), moon=ref_for(mars, 4))
    no = chart(mars, lagna=ref_for(mars, 3))
    check = chart(mars, moon=ref_for(mars, 7))
    assert fa.manglik_dosha(no, no) == {"name": "Manglik", "present": False, "cancelled": False,
                                        "why": "Neither of you is Manglik.", "status": "clear"}
    both = fa.manglik_dosha(yes, partial)
    assert both["status"] == "cancelled" and both["present"] and both["cancelled"]
    assert both["why"] == "You are both Manglik, which cancels it out."
    one_partial = fa.manglik_dosha(no, partial)
    assert one_partial["status"] == "cancelled" and one_partial["cancelled"]
    present = fa.manglik_dosha(yes, no)
    assert present["status"] == "present" and present["present"] and not present["cancelled"]
    assert present["why"] == "You are Manglik and the other isn’t. Worth knowing: many families talk it over with a pandit."
    assert fa.manglik_dosha(no, yes)["why"].startswith("They are Manglik")
    cancelled_no_time = chart(SIGN["Mesha"], moon=ref_for(SIGN["Mesha"], 7))   # own sign, no Lagna
    for a, b in ((check, yes), (yes, check), (check, no), (check, check), (cancelled_no_time, no),
                 (no, cancelled_no_time), (cancelled_no_time, yes)):
        d = fa.manglik_dosha(a, b)
        assert d["status"] == "check" and d["why"] == fa.MANGLIK_CHECK_WHY   # never a verdict without both Lagnas
    # The names go in for a match between two other people.
    assert astro_api._named(present["why"], "Rahul", "Neha").startswith("Rahul is Manglik and the other isn")
    assert astro_api._named(fa.manglik_dosha(no, yes)["why"], "Rahul", "Neha").startswith("Neha is Manglik")
    assert astro_api._named(both["why"], "Rahul", "Neha") == "Rahul and Neha are both Manglik, which cancels it out."


def test_no_fear_words():
    banned = re.compile(r"\b(danger|death|die|widow|curse|remed|puja|bad luck|harm)", re.I)
    for text in [fa.MANGLIK_CHECK_WHY, fa.CANCEL_OWN_SIGN, fa.CANCEL_JUPITER, fa._sign_exception("Moon")]:
        assert not banned.search(text)
    mars = SIGN["Mithuna"]
    for a in (chart(mars, lagna=ref_for(mars, 7)), chart(mars, lagna=ref_for(mars, 3)), chart(mars)):
        for b in (chart(mars, lagna=ref_for(mars, 7)), chart(mars, lagna=ref_for(mars, 3))):
            assert not banned.search(fa.manglik_dosha(a, b)["why"])


def test_charts_carry_jupiter():
    c = fa.chart_from_birth((1990, 5, 15), (10, 30), {"lat": 28.61, "lon": 77.21, "tz": "Asia/Kolkata", "precision": "city"})
    full = ac.birth_chart((1990, 5, 15), (10, 30), {"lat": 28.61, "lon": 77.21, "tz": "Asia/Kolkata", "precision": "city"})
    jup = next(p for p in full["planets"] if p["code"] == "Ju")
    assert c["jupiterRashi"] == jup["rashi"]
    assert c["marsRashi"] == next(p for p in full["planets"] if p["code"] == "Ma")["rashi"]
    no_time = fa.chart_from_birth((1990, 5, 15), None, None)
    assert 0 <= no_time["jupiterRashi"] < 12 and no_time["lagna"] is None
    assert fa.manglik_status(no_time)["status"] == "check"
    assert fa.chart_for_profile({"birthDate": "1990-05-15"}, lambda p: None)["jupiterRashi"] == no_time["jupiterRashi"]
    assert fa.chart_for_profile({"birthDate": "1990-02-30"}, lambda p: None) is None


# ---------------------------------------------------------------- AstroSage cross-check
def test_fixture_has_ten_charts():
    assert len(FIX["charts"]) >= 10 and all(c["url"].startswith("https://www.astrosage.com/") for c in FIX["charts"])


@pytest.mark.parametrize("ex", FIX["charts"], ids=[c["id"] for c in FIX["charts"]])
def test_astrosage_charts(ex):
    """Mars sits in the same houses from the Lagna and the Moon as AstroSage computes; our status follows C3."""
    y, m, d = (int(x) for x in ex["birthDate"].split("-"))
    hh, mm = (int(x) for x in ex["birthTime"].split(":"))
    c = fa.chart_from_birth((y, m, d), (hh, mm), {"lat": ex["lat"], "lon": ex["lon"], "tz": ex["tz"], "precision": "city"})
    assert fa.house_from(c["marsRashi"], c["lagna"]) == ex["astrosage"]["marsHouseFromLagna"]
    assert fa.house_from(c["marsRashi"], c["rashi"]) == ex["astrosage"]["marsHouseFromMoon"]
    s = fa.manglik_status(c)
    assert s["status"] == ex["expected"]["status"] and s["cancelReasons"] == ex["expected"]["cancelReasons"]
    # AstroSage leaves the 2nd house out and applies no cancellation: we never say "yes" where its Lagna chart is clear.
    if "not present in Lagna" in ex["astrosage"]["verdict"] or "neither" in ex["astrosage"]["verdict"]:
        assert s["status"] != "yes"


# ---------------------------------------------------------------- the 69 % problem
def _random_charts(n, with_time, seed):
    rng = random.Random(seed)
    places = [BLR, {"lat": 28.61, "lon": 77.21, "tz": "Asia/Kolkata", "precision": "city"},
              {"lat": 19.08, "lon": 72.88, "tz": "Asia/Kolkata", "precision": "city"},
              {"lat": 13.08, "lon": 80.27, "tz": "Asia/Kolkata", "precision": "city"}]
    out = []
    for _ in range(n):
        date = (rng.randint(1970, 2007), rng.randint(1, 12), rng.randint(1, 28))
        time_ = (rng.randint(0, 23), rng.randint(0, 59)) if with_time else None
        out.append(fa.chart_from_birth(date, time_, rng.choice(places)))
    return out


def manglik_distribution(n=2000, seed=20261011):
    known = Counter(fa.manglik_status(c)["status"] for c in _random_charts(n, True, seed))
    unknown = Counter(fa.manglik_status(c)["status"] for c in _random_charts(n // 4, False, seed + 1))
    legacy = Counter(fa.manglik_status(c)["manglik"] for c in _random_charts(n // 4, True, seed + 2))
    return known, unknown, legacy


def test_full_manglik_is_at_most_40_percent_with_a_known_time():
    known, unknown, legacy = manglik_distribution()
    total = sum(known.values())
    assert total == 2000
    share = {k: round(100.0 * v / total, 1) for k, v in sorted(known.items())}
    print("known time:", share, "| no time:", dict(unknown), "| old flag manglik:", dict(legacy))
    assert known["yes"] / total <= 0.40, share
    assert known["check"] == 0
    assert set(unknown) == {"check"}                       # without a time: never a verdict, not even "no"
