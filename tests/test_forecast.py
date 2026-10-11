"""The week ahead (engagement N4): week_days, best days, sky events, the AI prompt, the validator and the cache."""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pytest

import astro_core as ac
import astro_forecast as af

DELHI = (28.6139, 77.209, "Asia/Kolkata")
PLACE = {"name": "New Delhi", "lat": DELHI[0], "lon": DELHI[1], "tz": DELHI[2]}
KANYA, HASTA = 5, 12   # the fixture chart: Moon in Kanya, Hasta nakshatra

GOOD_TEXT = (
    "Love & family\n"
    "The Moon moves through Tula early in the week, so family talk and good food bring you closer on Monday and "
    "Tuesday. Keep Friday evening calm at home.\n\n"
    "Work & money\n"
    "Wednesday 14 October and Thursday 15 October give you extra courage: make the call, send the message and take "
    "the first step on a plan you have been holding back.\n\n"
    "Health\n"
    "Friday is a quiet day for rest and patience. An early night and a short walk will keep your energy steady "
    "into the weekend.\n\n"
    "Best day for…\n"
    "Thursday is your best day for work and money. Saturday 17 October suits a quiet family day, and Sunday "
    "18 October brings Durga Puja (Saptami) with sweets, prayers and time together."
)


def _week(today=date(2026, 10, 14), **kw):
    kw.setdefault("festivals", [{"date": "2026-10-18", "name": "Durga Puja (Saptami)", "id": "durga_puja"}])
    return af.build_week(KANYA, HASTA, today, PLACE, **kw)


def _facts(**kw):
    return af.week_facts(_week(**kw), natal_rashi=KANYA, natal_nak=HASTA)


# ------------------------------------------------------------------ week_days and best days
def test_week_days_is_deterministic_for_a_fixture_chart():
    a = ac.week_days(KANYA, HASTA, date(2026, 10, 12), *DELHI)
    assert a == ac.week_days(KANYA, HASTA, date(2026, 10, 12), *DELHI)
    assert [d["date"] for d in a] == [f"2026-10-{12 + i}" for i in range(7)]
    assert [d["weekday"] for d in a] == ac.WEEKDAYS
    assert [d["house"] for d in a] == [2, 2, 3, 3, 4, 4, 4]
    assert [d["moonSign"] for d in a] == ["Tula", "Tula", "Vrischika", "Vrischika", "Dhanu", "Dhanu", "Dhanu"]
    assert [d["tone"] for d in a] == ["steady", "steady", "steady", "good", "gentle", "steady", "steady"]
    assert a[0]["tithi"] == "Shukla Dwitiya" and a[4]["tithi"] == "Shukla Shashthi"
    assert all(d["line"] == ac.SELF_LINES[d["house"]] for d in a)


def test_house_8_is_gentle_and_never_bad():
    seen8 = 0
    for rashi in range(12):
        for nak in (0, 9, 18):
            mine = ac.week_days(rashi, nak, date(2026, 10, 12), *DELHI)
            theirs = ac.week_days(rashi, nak, date(2026, 10, 12), *DELHI, name="Riya", relation="daughter")
            for d in mine + theirs:
                assert "bad" not in d["line"].lower()
                if d["house"] == 8:
                    seen8 += 1
                    assert d["tone"] == "gentle", d
    assert seen8 > 0
    assert all("bad" not in t.lower() for t in ac.SELF_LINES.values())
    assert ac.week_tone(8, 2) == "gentle" and ac.week_tone(10, 2) == "good" and ac.week_tone(12, 3) == "gentle"
    assert ac.week_tone(12, 2) == "steady" and ac.week_tone(10, 3) == "steady"


def test_managed_week_uses_the_family_lines():
    d = ac.week_days(KANYA, HASTA, date(2026, 10, 12), *DELHI, name="Ananya", relation="daughter")
    assert d[0]["line"] == "Family talk and good food go well for Ananya today."
    assert all("Ananya" in x["line"] for x in d)


def test_managed_week_never_reads_like_a_child():
    # The week is the account's own profile, and a managed rishta profile is an adult (18+/21+): profileFor 'son'
    # or 'daughter' must never bring "homework", "school" or "bedtime" lines.
    child = {h: t["child"].format(n="Ananya") for h, t in ac.LINES.items() if "child" in t}
    assert {5, 6, 8, 10, 12} <= set(child)
    houses = set()
    for relation in ("son", "daughter", "Daughter", "brother", "relative", ""):
        for rashi in range(12):
            for d in ac.week_days(rashi, HASTA, date(2026, 10, 12), *DELHI, name="Ananya", relation=relation):
                houses.add(d["house"])
                assert d["line"] == ac.LINES[d["house"]]["any"].format(n="Ananya"), (relation, d)
                assert d["line"] not in child.values()
                for word in ("homework", "school", "bedtime", "studies", "a hug"):
                    assert word not in d["line"]
    assert houses == set(range(1, 13))
    w = af.build_week(KANYA, HASTA, date(2026, 10, 28), PLACE, name="Ananya", relation="daughter")
    assert all(x["line"] not in child.values() for x in w["days"])


def test_best_days_rules():
    def day(i, house, tone):
        return {"date": f"2026-10-{12 + i}", "house": house, "tone": tone}
    days = [day(0, 5, "gentle"), day(1, 7, "steady"), day(2, 9, "good"), day(3, 11, "good"),
            day(4, 8, "gentle"), day(5, 4, "steady"), day(6, 12, "gentle")]
    best = ac.best_days(days)
    # Love: 5/7/9/11 and not gentle, good days first, two at most, in date order.
    assert best["love"] == ["2026-10-14", "2026-10-15"]
    # Work: 3/6/10/11 not gentle.
    assert best["work"] == ["2026-10-15"]
    # Quiet: 4/8/12, the gentlest first.
    assert best["quiet"] == ["2026-10-16", "2026-10-18"]
    assert ac.best_days([]) == {"love": [], "work": [], "quiet": []}


# ------------------------------------------------------------------ sky events
def test_sky_events_match_the_verified_ingresses():
    ev = ac.sky_events(date(2026, 10, 1), 450)
    got = [(e["planet"], e["signName"], e["date"]) for e in ev]
    assert got == [("Jupiter", "Simha", "2026-10-31"), ("Rahu", "Makara", "2026-12-05"),
                   ("Jupiter", "Karka", "2027-01-25"), ("Saturn", "Mesha", "2027-06-03"),
                   ("Jupiter", "Simha", "2027-06-26"), ("Saturn", "Meena", "2027-10-20"),
                   ("Jupiter", "Kanya", "2027-11-26")]
    jup = datetime.fromisoformat(ev[0]["atIso"])
    assert abs((jup - datetime.fromisoformat("2026-10-31T12:02+05:30")).total_seconds()) <= 3600
    sat = next(e for e in ev if e["planet"] == "Saturn")
    assert sat["date"] == "2027-06-03" and not sat["retro"]
    assert next(e for e in ev if e["signName"] == "Karka")["retro"]
    assert [e["atIso"] for e in ev] == sorted(e["atIso"] for e in ev)


def test_sky_events_are_scanned_once_at_import():
    assert ac.SKY_EVENTS == ac.sky_events(ac.SKY_EVENTS_START, ac.SKY_DAYS)
    t = time.perf_counter()
    for _ in range(50):
        ac.sky_events_between(ac.SKY_EVENTS_START, ac.SKY_EVENTS_START + timedelta(days=30))
    assert time.perf_counter() - t < 0.2
    inside = ac.sky_events_between(date(2026, 10, 26), date(2026, 11, 1))
    assert [(e["planet"], e["date"]) for e in inside] == [("Jupiter", "2026-10-31")]


# ------------------------------------------------------------------ the week payload
def test_week_period_and_titles():
    assert af.week_period(date(2026, 10, 14)) == (date(2026, 10, 12), False)
    assert af.week_period(date(2026, 10, 12)) == (date(2026, 10, 12), False)
    assert af.week_period(date(2026, 10, 17)) == (date(2026, 10, 19), True)    # Saturday: next week
    assert af.week_period(date(2026, 10, 18)) == (date(2026, 10, 19), True)    # Sunday: next week
    assert af.week_title(date(2026, 10, 12), False) == "Your week · 12–18 Oct"
    assert af.week_title(date(2026, 10, 19), True) == "Next week · 19–25 Oct"
    assert af.week_title(date(2026, 10, 26), False) == "Your week · 26 Oct – 1 Nov"
    assert af.week_title(date(2026, 10, 12), False, "Ananya") == "Ananya's week · 12–18 Oct"
    assert af.week_title(date(2026, 10, 19), True, "Ananya") == "Ananya's next week · 19–25 Oct"


def test_build_week_shape():
    w = _week(festivals=[{"date": "2026-10-18", "name": "Durga Puja (Saptami)", "id": "durga_puja"},
                         {"date": "2026-10-20", "name": "Dussehra", "id": "dussehra"}])
    assert set(w) == {"span", "periodKey", "title", "precision", "days", "best", "events"}
    assert w["span"] == "week" and w["periodKey"] == "2026-10-12" and w["title"] == "Your week · 12–18 Oct"
    assert w["precision"] == "exact" and len(w["days"]) == 7
    assert set(w["best"]) == {"love", "work", "quiet"}
    assert w["events"] == [{"date": "2026-10-18", "kind": "festival", "id": "durga_puja", "text": "Durga Puja (Saptami)"}]
    assert _week(precision="date")["precision"] == "approximate"
    json.dumps(w)   # plain JSON


def test_week_events_carry_the_sky_and_the_person():
    w = af.build_week(KANYA, HASTA, date(2026, 10, 28), PLACE, name="Ananya", relation="daughter")
    assert w["title"] == "Ananya's week · 26 Oct – 1 Nov"
    assert w["events"] == [{"date": "2026-10-31", "kind": "sky", "id": "jupiter",
                            "text": "Jupiter moves into Simha, Ananya's 12th house from the Moon: a warm year for "
                                    "rest, travel and letting go begins."}]
    mine = af.build_week(KANYA, HASTA, date(2026, 10, 28), PLACE)
    assert "your 12th house" in mine["events"][0]["text"]


def test_festivals_in_the_list_never_rename_the_sky_reader():
    # Karva Chauth (29 Oct) and Jupiter into Simha (31 Oct) share a week; Diwali (8 Nov) is listed but falls outside.
    # Each festival name once leaked into the sky line ("Diwali's 12th house", "Karva Chauth's 12th house").
    karva = {"date": "2026-10-29", "name": "Karva Chauth", "id": "karva_chauth"}
    diwali = {"date": "2026-11-08", "name": "Diwali", "id": "diwali"}
    line = "Jupiter moves into Simha, {} 12th house from the Moon: a warm year for rest, travel and letting go begins."
    for name, whose in ((None, "your"), ("Ananya", "Ananya's")):
        for fests in ([karva, diwali], [diwali, karva]):
            w = af.build_week(KANYA, HASTA, date(2026, 10, 28), PLACE, name=name,
                              relation="daughter" if name else "", festivals=fests)
            assert w["events"] == [
                {"date": "2026-10-29", "kind": "festival", "id": "karva_chauth", "text": "Karva Chauth"},
                {"date": "2026-10-31", "kind": "sky", "id": "jupiter", "text": line.format(whose)}]
            facts = af.week_facts(w, natal_rashi=KANYA, natal_nak=HASTA, name=name)
            assert facts["events"][1]["text"] == line.format(whose)
            txt = af.facts_text(facts)
            assert "Diwali" not in txt and "Karva Chauth's" not in txt


def test_retrograde_return_and_re_entry_name_no_year():
    ev = {e["date"]: e for e in ac.sky_events(date(2026, 10, 1), 450)}
    first, back, again, sat_back = ev["2026-10-31"], ev["2027-01-25"], ev["2027-06-26"], ev["2027-10-20"]
    assert not first["retro"] and not first["again"]
    assert back["retro"] and not back["again"] and again["again"] and not again["retro"]
    assert sat_back["retro"] and not ev["2027-06-03"]["again"] and not ev["2026-12-05"]["again"]
    # Jupiter goes back from Simha into Karka on 25 Jan 2027: no new "warm year" begins.
    assert ac.sky_line(back, KANYA) == ("Jupiter moves back into Karka, your 11th house from the Moon: a gentle second "
                                        "look at gains, friends and wishes.")
    assert ac.sky_line(back, KANYA, "Ananya").startswith("Jupiter moves back into Karka, Ananya's 11th house")
    assert ac.sky_line(back, None) == "Jupiter moves back into Karka (Cancer)."
    assert ac.sky_line(again, KANYA) == ("Jupiter moves into Simha again, your 12th house from the Moon: warmth "
                                         "returns to rest, travel and letting go.")
    assert ac.sky_line(sat_back, KANYA) == ("Saturn moves back into Meena, your 7th house from the Moon: a patient "
                                            "second look at marriage and partnerships.")
    for e in (back, again, sat_back):
        for r in range(12):
            assert "year" not in ac.sky_line(e, r) and "begins" not in ac.sky_line(e, r)
    # The week of the return carries the same line, and so do the AI facts.
    w = af.build_week(KANYA, HASTA, date(2027, 1, 27), PLACE, name="Ananya", relation="daughter")
    assert w["events"] == [{"date": "2027-01-25", "kind": "sky", "id": "jupiter",
                            "text": ac.sky_line(back, KANYA, "Ananya")}]
    assert "a gentle second look" in af.facts_text(af.week_facts(w, natal_rashi=KANYA, natal_nak=HASTA, name="Ananya"))


# ------------------------------------------------------------------ facts and prompt
def test_facts_and_prompt():
    f = _facts()
    assert f["reader"] == "self" and f["monday"] == "2026-10-12" and f["sunday"] == "2026-10-18"
    txt = af.facts_text(f)
    assert txt.startswith("READER: self\nASTRO_MODE: full\nWEEK: Monday 12 October to Sunday 18 October 2026")
    assert "NATAL: Moon sign Kanya (Virgo), Hasta nakshatra" in txt
    assert "EVENTS: Sunday 18 October: Durga Puja (Saptami)" in txt
    msgs = af.week_prompt(f)
    assert msgs[0]["role"] == "system" and msgs[1]["content"].startswith("FACTS:\n" + txt)
    for section in ("Love & family", "Work & money", "Health", "Best day for…"):
        assert section in msgs[0]["content"]
    assert "120-180 words" in msgs[0]["content"]
    again = af.week_prompt(f, stricter="markdown")
    assert "refused (markdown)" in again[1]["content"]


def test_managed_facts_name_the_reader():
    w = _week(name="Ananya", relation="daughter")
    f = af.week_facts(w, natal_rashi=KANYA, natal_nak=HASTA, name="Ananya", astro_mode="fun")
    txt = af.facts_text(f)
    assert "READER: family of Ananya (the profile was made by a parent or relative)" in txt
    assert "ASTRO_MODE: fun (keep it light, entertainment framing)" in txt


# ------------------------------------------------------------------ the validator
def test_validator_accepts_a_good_text():
    assert af.problems(GOOD_TEXT, _facts()) is None and af.validate(GOOD_TEXT, _facts())


@pytest.mark.parametrize("change, why", [
    (("Tula", "Leo"), "'Leo' is not in the facts"),                 # an invented sign
    (("The Moon moves", "Venus moves"), "'Venus' is not in the facts"),   # an invented planet
    (("Tula early", "Simha early"), "'Simha' is not in the facts"),
    (("Thursday 15 October", "Tuesday 20 October"), "a date outside the week"),
    (("Thursday 15 October", "October 25"), "a date outside the week"),
    (("Thursday 15 October", "2026-10-25"), "a date outside the week"),
    (("Thursday 15 October", "31 February"), "a date outside the week"),
    (("the weekend.", "the weekend of 2027."), "the year 2027"),
    (("Keep Friday", "Avoid danger and keep Friday"), "banned word 'danger'"),
    (("Friday is a quiet day", "Friday is a bad day"), "banned word 'bad day'"),
    (("Keep Friday", "No doshas trouble you. Keep Friday"), "banned word 'doshas'"),
    (("An early night", "Illness stays away. An early night"), "banned word 'Illness'"),
    (("Love & family", "**Love & family**"), "markdown"),
    (("Love & family", "## Love & family"), "markdown"),
    (("Health\n", "Health\n- "), "markdown"),
])
def test_validator_rejects(change, why):
    text = GOOD_TEXT.replace(*change)
    assert text != GOOD_TEXT
    assert af.problems(text, _facts()) == why


def test_validator_rejects_length_and_empty():
    f = _facts()
    assert af.problems("Love & family\nA calm week.", f).endswith("words")
    assert af.problems((GOOD_TEXT + " ") * 2, f).endswith("words")
    assert af.problems("", f) == "empty" and af.problems(None, f) == "empty"


def test_signs_and_names_from_the_facts_are_fine():
    # A sky event in the facts makes its planet and sign usable; a word like Sunday is no Sun; a week across two
    # months takes dates from both.
    w = af.build_week(KANYA, HASTA, date(2026, 10, 28), PLACE)
    f = af.week_facts(w, natal_rashi=KANYA, natal_nak=HASTA)
    text = (
        "Love & family\n"
        "The Moon moves through Vrishabha in midweek, so Wednesday and Thursday bring warm family talk and "
        "blessings from elders. Keep Monday and Tuesday calm and gentle at home.\n\n"
        "Work & money\n"
        "Friday 30 October and Saturday 31 October are good for being seen at work. On Saturday Jupiter moves "
        "into Simha: a warm year for rest and travel begins.\n\n"
        "Health\n"
        "Monday 26 October and Tuesday 27 October are quiet days for rest and patience. Sleep early and drink "
        "plenty of water.\n\n"
        "Best day for…\n"
        "Wednesday is your best day for love and family, Saturday for work and money, and Sunday 1 November "
        "brings good news from friends."
    )
    assert af.problems(text, f) is None, af.problems(text, f)
    assert af.problems(text.replace("Sunday 1 November", "Sunday 8 November"), f) == "a date outside the week"
    assert af.problems(text, _facts()) is not None   # the same text in another week names the wrong things


# ------------------------------------------------------------------ writing with one retry
def test_write_week_retries_once_with_a_stricter_note():
    calls = []

    def complete(msgs, max_tokens, temperature, kind):
        calls.append(msgs)
        return ("**bad**" if len(calls) == 1 else GOOD_TEXT), "stop"

    assert af.write_week(complete, _facts()) == GOOD_TEXT
    assert len(calls) == 2 and "refused" in calls[1][1]["content"] and "refused" not in calls[0][1]["content"]


def test_write_week_gives_up_after_two_refusals_or_an_error():
    calls = []

    def bad(msgs, *a):
        calls.append(1)
        return GOOD_TEXT.replace("Tula", "Leo"), "stop"

    assert af.write_week(bad, _facts()) is None and len(calls) == 2

    def boom(*a):
        raise RuntimeError("down")

    assert af.write_week(boom, _facts()) is None


# ------------------------------------------------------------------ cache and lease
class Snap:
    def __init__(self, data):
        self.exists, self._d = data is not None, data

    def to_dict(self):
        return None if self._d is None else dict(self._d)


class Ref:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def get(self):
        with self.db.lock:
            return Snap(self.db.docs.get(self.path))

    def set(self, data, merge=False):
        with self.db.lock:
            self.db.docs[self.path] = {**(self.db.docs.get(self.path) or {}), **data} if merge else dict(data)


class Col:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def document(self, id_):
        return Ref(self.db, f"{self.path}/{id_}")


class Db:
    def __init__(self, docs=None):
        self.docs, self.lock = dict(docs or {}), threading.Lock()

    def collection(self, name):
        return Col(self, name)


class Leases:
    def __init__(self):
        self.held, self.lock, self.taken, self.released = set(), threading.Lock(), [], []

    def take(self, name, ttl):
        with self.lock:
            if name in self.held:
                return False
            self.held.add(name)
            self.taken.append(name)
            return True

    def release(self, name):
        with self.lock:
            self.held.discard(name)
            self.released.append(name)


def _cached(db, leases, complete, sig="sig1", rate=True, **kw):
    return af.cached_week_text(db, "u1", "2026-10-12", _facts(), sig, complete, take_lease=leases.take,
                               release_lease=leases.release, take_rate=lambda *a: rate, poll_sec=0.02, **kw)


def test_one_ai_call_under_parallel_requests():
    db, leases, calls = Db(), Leases(), []

    def slow(msgs, *a):
        calls.append(1)
        time.sleep(0.3)
        return GOOD_TEXT, "stop"

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: _cached(db, leases, slow, wait_sec=5.0), range(6)))
    assert len(calls) == 1
    assert all(text == GOOD_TEXT for text, _ in results)
    assert sorted(cached for _, cached in results) == [False] + [True] * 5
    assert leases.taken == ["forecast_u1_2026-10-12"]
    doc = db.docs["forecasts/u1__self__week__2026-10-12"]
    assert doc["text"] == GOOD_TEXT and doc["birthSig"] == "sig1" and doc["periodKey"] == "2026-10-12"
    assert doc["factsHash"] == af.facts_hash(_facts()) and doc["createdAtIso"]
    # Later calls read the cache.
    assert _cached(db, leases, slow) == (GOOD_TEXT, True) and len(calls) == 1


def test_birth_change_regenerates_within_the_daily_cap():
    path = "forecasts/u1__self__week__2026-10-12"
    db = Db({path: {"text": "old text", "birthSig": "sig0"}})
    calls = []

    def complete(*a):
        calls.append(1)
        return GOOD_TEXT, "stop"

    assert _cached(db, Leases(), complete, sig="sig0") == ("old text", True) and not calls
    assert _cached(db, Leases(), complete, sig="sig1", rate=False) == ("old text", True) and not calls
    leases = Leases()
    assert _cached(db, leases, complete, sig="sig1") == (GOOD_TEXT, False) and len(calls) == 1
    assert leases.taken == ["forecast_u1_2026-10-12_sig1"]
    assert db.docs[path]["birthSig"] == "sig1"


def test_failed_text_releases_the_lease_and_stores_nothing():
    db, leases = Db(), Leases()
    text, cached = _cached(db, leases, lambda *a: ("", "stop"))
    assert (text, cached) == (None, False)
    assert leases.released == ["forecast_u1_2026-10-12"] and not db.docs


def test_without_a_db_the_text_is_written_but_not_kept():
    assert af.cached_week_text(None, "u1", "2026-10-12", _facts(), "s", lambda *a: (GOOD_TEXT, "stop"),
                               take_lease=None, release_lease=None, take_rate=None) == (GOOD_TEXT, False)


def test_birth_sig_formula():
    raw = "1994-08-21|07:40|Pune|"
    assert af.birth_sig("1994-08-21", "07:40", "Pune", None) == hashlib.sha1(raw.encode()).hexdigest()[:12]
    assert af.birth_sig("1994-08-21", "", "", "night") != af.birth_sig("1994-08-21", "", "", "morning")
    assert af.forecast_doc_id("u1", "2026-10-12") == "u1__self__week__2026-10-12"
