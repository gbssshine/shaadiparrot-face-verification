import json
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.api_core import exceptions as gexc

import astro_api
import astro_content
import astro_core as ac
import fates_astro_chart as fc
import main

DELHI = (28.6139, 77.209, "Asia/Kolkata")
BLR = (12.9719, 77.5937, "Asia/Kolkata")


def test_panchang_known_days():
    # Mahalaya Amavasya 10 Oct 2026; Navratri starts the next day (Ashwin Shukla Pratipada).
    p = ac.panchang(date(2026, 10, 10), *BLR)
    assert p["tithi"]["name"] == "Amavasya" and p["weekday"] == "Saturday"
    assert p["sunrise"] == "06:09" and p["sunset"] == "18:03"
    n = ac.panchang(date(2026, 10, 11), *BLR)
    assert n["tithi"]["name"] == "Pratipada" and n["tithi"]["paksha"] == "Shukla"
    # Karva Chauth 29 Oct 2026: Krishna Chaturthi, moonrise in Delhi ~20:12 (published).
    k = ac.panchang(date(2026, 10, 29), *DELHI)
    assert k["tithi"]["number"] == 19
    h, m = map(int, k["moonrise"].split(":"))
    assert abs(h * 60 + m - (20 * 60 + 12)) <= 3


def test_rahu_kaal_is_an_eighth_of_daylight():
    p = ac.panchang(date(2026, 10, 10), *BLR)   # Saturday: 3rd part
    rise = datetime.fromisoformat(p["gulika"]["startIso"])   # Saturday Gulika is the 1st part
    assert p["gulika"]["start"] == p["sunrise"]
    assert p["rahuKaal"]["start"] == p["yamaganda"]["start"] or p["rahuKaal"]["start"] > p["sunrise"]
    span = datetime.fromisoformat(p["rahuKaal"]["endIso"]) - datetime.fromisoformat(p["rahuKaal"]["startIso"])
    assert abs(span.total_seconds() / 60 - p["dayLengthMinutes"] / 8) < 1.5
    assert rise.tzinfo is not None


def test_good_times_never_sit_on_rahu_yamaganda_or_gulika():
    for i in range(7):   # Monday 12 Oct to Sunday 18 Oct 2026
        p = ac.panchang(date(2026, 10, 12) + timedelta(days=i), *DELHI)
        bad = {p["rahuKaal"]["startIso"], p["yamaganda"]["startIso"], p["gulika"]["startIso"]}
        rahu = (p["rahuKaal"]["startIso"], p["rahuKaal"]["endIso"])
        for g in p["goodTimes"]:
            assert g["startIso"] not in bad, (p["weekday"], g)
            if g["name"].startswith("Abhijit"):
                assert g["startIso"] >= rahu[1] or g["endIso"] <= rahu[0], (p["weekday"], g, rahu)
        abhijit = [g for g in p["goodTimes"] if g["name"].startswith("Abhijit")]
        if p["weekday"] == "Wednesday":
            assert not abhijit
        if p["weekday"] == "Friday":   # Rahu kaal covers the first half: what's left starts as it ends
            assert abhijit and abhijit[0]["start"] == p["rahuKaal"]["end"]


def test_adhik_month_keeps_its_name_in_the_dark_half():
    # 2026 has Adhik Jyeshtha: its dark half is still Jyeshtha in the purnimanta reckoning.
    assert ac.panchang(date(2026, 6, 10), *DELHI)["masa"] == {"amanta": "Jyeshtha", "purnimanta": "Jyeshtha", "adhik": True}
    assert ac.panchang(date(2026, 5, 5), *DELHI)["masa"] == {"amanta": "Vaishakha", "purnimanta": "Jyeshtha", "adhik": False}
    assert ac.panchang(date(2026, 7, 5), *DELHI)["masa"] == {"amanta": "Jyeshtha", "purnimanta": "Ashadha", "adhik": False}


def test_upcoming_has_festivals_and_no_duplicate_vrats():
    items = ac.upcoming(date(2026, 10, 10), 45, *DELHI, limit=30)
    names = [i["name"] for i in items]
    assert "Sharad Navratri begins" in names and "Karva Chauth" in names and "Diwali" in names
    assert "Sankashti Chaturthi" not in names            # Karva Chauth already is that day
    eka = [i["date"] for i in items if i["kind"] == "ekadashi"]
    assert all(abs((date.fromisoformat(a) - date.fromisoformat(b)).days) > 3 for a, b in zip(eka, eka[1:]))


def test_kshaya_tithis_are_not_lost():
    # A tithi that begins after one sunrise and ends before the next belongs to that day.
    vr = ac.vrats_between(date(2026, 10, 10), 400, *DELHI)
    on = {(v["date"], v["kind"]) for v in vr}
    assert ("2026-11-20", "ekadashi") in on             # Dev Uthani: Dashami at sunrise, Dwadashi at the next
    assert ("2027-07-29", "ekadashi") in on
    assert ("2027-02-24", "sankashti") in on            # Chaturthi was over before that night's moonrise
    assert any(("2026-12-%02d" % d, "pradosh") in on for d in range(20, 24))
    year = [v for v in vr if v["date"] < "2027-10-10"]
    count = {k: sum(1 for v in year if v["kind"] == k) for k in ("ekadashi", "pradosh", "sankashti", "purnima", "amavasya")}
    assert 24 <= count["ekadashi"] <= 25 and 23 <= count["pradosh"] <= 25, count
    assert 12 <= count["sankashti"] <= 13 and 12 <= count["purnima"] <= 13 and 12 <= count["amavasya"] <= 13, count
    for kind, longest in (("ekadashi", 17), ("pradosh", 17), ("sankashti", 31), ("purnima", 31), ("amavasya", 31)):
        days = [date.fromisoformat(v["date"]) for v in vr if v["kind"] == kind]
        assert max((b - a).days for a, b in zip(days, days[1:])) <= longest, kind
    # On Today the festival stands for the vrat.
    ahead = ac.upcoming(date(2026, 11, 15), 10, *DELHI, limit=20)
    assert any(i["date"] == "2026-11-20" and i["name"] == "Dev Uthani Ekadashi" for i in ahead)
    assert not any(i["kind"] == "ekadashi" for i in ahead)


def test_festival_list_runs_ahead():
    days = [f["date"] for f in ac.FESTIVALS]
    assert days == sorted(days)
    # Today's festival feed goes quiet when the list runs out: extend astro_core.FESTIVALS before this fails.
    assert date.fromisoformat(days[-1]) - date.today() >= timedelta(days=60)
    ahead = ac.upcoming(date(2027, 8, 1), 40, *DELHI, limit=20)
    assert [i["name"] for i in ahead if i["type"] == "festival"] == [
        "Hariyali Teej", "Independence Day", "Raksha Bandhan", "Janmashtami", "Ganesh Chaturthi"]
    assert not any(i["kind"] == "purnima" for i in ahead)   # Raksha Bandhan stands for it


def test_chart_and_dashas():
    place = {"lat": DELHI[0], "lon": DELHI[1], "tz": DELHI[2], "precision": "city"}
    c = ac.birth_chart((1990, 5, 15), (10, 30), place, now=datetime(2026, 10, 10, tzinfo=timezone.utc))
    assert c["lagnaName"] == "Karka" and c["moon"]["nakshatra"] == "Uttara Ashadha"
    assert [p["code"] for p in c["planets"]] == ["Su", "Mo", "Ma", "Me", "Ju", "Ve", "Sa", "Ra", "Ke"]
    assert c["planets"][-1]["rashi"] == (c["planets"][-2]["rashi"] + 6) % 12
    # Uttara Ashadha is ruled by the Sun: life starts in the Sun mahadasha.
    assert c["dasha"]["periods"][0]["lord"] == "Sun"
    assert c["dasha"]["current"]["lord"] == "Rahu" and c["dasha"]["antar"]["lord"] == "Sun"
    total = sum(p["years"] for p in c["dasha"]["periods"][1:10])
    assert total == 120
    no_time = ac.birth_chart((1990, 5, 15), None, None)
    assert no_time["basis"] == "moon" and no_time["lagna"] is None and no_time["dashaApproximate"]


def test_good_days_respect_season():
    g = ac.good_days("wedding", date(2026, 10, 10), 90, *DELHI)
    assert g["days"], g
    assert all(d["date"] >= "2026-11-21" for d in g["days"])            # Chaturmas ends with Dev Uthani
    assert not any("2026-12-17" <= d["date"] <= "2027-01-13" for d in g["days"])   # Kharmas
    # A window after midnight is printed on the next calendar day, still the same Hindu day (sunrise to sunrise).
    assert all(d["weekday"] in ("Monday", "Wednesday", "Thursday", "Friday") or d["window"]["night"] for d in g["days"])


def test_good_windows_join_across_a_change():
    # 25 Nov 2026 in Delhi: Rohini then Mrigashira, Pratipada then Dwitiya, all four good for a wedding.
    # The whole Hindu day is one window, not three pieces with only the night one shown.
    g = ac.good_days("wedding", date(2026, 11, 25), 1, *DELHI)
    day = g["days"][0]
    assert day["date"] == "2026-11-25"
    assert day["window"] == {"start": "06:51", "end": "06:52", "endsNextDay": True, "night": False}
    assert day["score"] == 4
    assert day["reasons"][0] == "Rohini and Mrigashira nakshatras"
    assert day["reasons"][1] == "Krishna Pratipada and Dwitiya"


def test_good_days_fast_after_warm_up():
    start = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    for ev in ac.EVENTS:   # warm-up: today's tithi and nakshatra table and the city's day frames
        ac.good_days(ev, start, 120, *DELHI)
    for ev in ac.EVENTS:
        t = time.perf_counter()
        ac.good_days(ev, start, 120, *DELHI, people=[{"name": "Asha", "rashi": 4, "nak": 9}])
        assert time.perf_counter() - t < 0.3, ev


def test_day_line_tones():
    line = ac.day_line("Riya", "daughter", 0, 1, 4, 3)   # Moon 5th from hers
    assert "Riya" in line["text"] and "studies" in line["text"]
    assert line["tone"] in ("good", "steady", "gentle")
    nameless = ac.day_line("", "", 0, 1, 0, 1)
    assert nameless["text"].startswith("Your loved one feels")
    assert "for your loved one" in ac.day_line("", "", 0, 1, 1, 1)["text"]


def test_zone_survives_bad_names():
    assert ac.zone("Europe/London").key == "Europe/London"
    for bad in ("Asia", "x" * 300, "../etc/passwd", "", None, "Foo/Bar"):
        assert ac.zone(bad).key == "Asia/Kolkata", bad


def test_question_rotates():
    a = astro_content.question_for(date(2026, 10, 10))
    b = astro_content.question_for(date(2026, 10, 11))
    assert a["id"] != b["id"] and 2 <= len(a["options"]) <= 4


def test_impossible_dates_are_no_dates():
    assert fc._parse_date("2023-02-31") is None and fc._parse_date("2000-04-31") is None
    assert fc._parse_date("2023-02-29") is None and fc._parse_date("2024-02-29") == (2024, 2, 29)
    assert fc._parse_date("1999-12-31T00:00:00Z") == (1999, 12, 31)


# ------------------------------------------------------------------ a small Firestore stand-in
class FakeSnap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d, self.exists = ref, ref.id, data, data is not None

    def to_dict(self):
        return None if self._d is None else json.loads(json.dumps(self._d))


class FakeRef:
    def __init__(self, db, path):
        self._db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def collection(self, name):
        return FakeCol(self._db, f"{self.path}/{name}")

    def get(self):
        if self.path in self._db.hide_once:   # another device writes right after this read
            self._db.hide_once.discard(self.path)
            return FakeSnap(self, None)
        return FakeSnap(self, self._db.docs.get(self.path))

    def set(self, data, merge=False):
        self._db.write(self.path, data, merge)


class FakeCol:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return FakeRef(self._db, f"{self.path}/{id_}")

    def limit(self, n):
        return self

    def stream(self):
        prefix = self.path + "/"
        return [FakeSnap(FakeRef(self._db, p), d) for p, d in list(self._db.docs.items())
                if p.startswith(prefix) and "/" not in p[len(prefix):]]


class FakeBatch:
    def __init__(self, db):
        self._db, self._ops = db, []

    def create(self, ref, data):
        self._ops.append(("create", ref, data, False))

    def set(self, ref, data, merge=False):
        self._ops.append(("set", ref, data, merge))

    def commit(self):
        if any(op == "create" and ref.path in self._db.docs for op, ref, _, _ in self._ops):
            raise gexc.AlreadyExists("Document already exists")
        for op, ref, data, merge in self._ops:
            self._db.write(ref.path, data, merge)


class FakeDb:
    def __init__(self, docs=None):
        self.docs = {k: dict(v) for k, v in (docs or {}).items()}
        self.reads = 0
        self.hide_once = set()

    def collection(self, name):
        return FakeCol(self, name)

    def get_all(self, refs):
        refs = list(refs)
        self.reads += len(refs)
        return [FakeSnap(r, self.docs.get(r.path)) for r in reversed(refs)]   # any order, like Firestore

    def batch(self):
        return FakeBatch(self)

    def write(self, path, data, merge):
        def apply(dst, src):
            for k, v in src.items():
                if isinstance(v, dict):
                    apply(dst.setdefault(k, {}), v)
                elif type(v).__name__ == "Increment":
                    dst[k] = (dst.get(k) or 0) + v.value
                else:
                    dst[k] = v
        cur = dict(self.docs.get(path) or {}) if merge else {}
        apply(cur, data)
        self.docs[path] = cur


# ------------------------------------------------------------------ HTTP
ME = "me123456"
H = {"Authorization": "Bearer ok"}
# The way the app stores it: coordinates and birth details on the private profile, a rounded copy of the
# coordinates on the public card, the users doc without any place.
PROFILE = {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21", "birthTime": "07:40",
           "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN",
           "cityName": "Bengaluru", "stateName": "Karnataka", "countryName": "India", "countryIso2": "IN",
           "lat": 12.9719, "lon": 77.5937}
PUBLIC = {"firstName": "Asha", "gender": "Female", "cityName": "Bengaluru", "lat": 12.97, "lon": 77.59,
          "isDiscoverable": True}
USER = {"gender": "Female", "birthDate": "1994-08-21", "lifeStage": "single"}


def make_client(profile=PROFILE, public=PUBLIC, user=USER, db=None, premium=False, complete=None):
    astro_api._hits.clear()
    astro_api._report_hits.clear()
    astro_api._scores_cache.clear()
    astro_api._moon_written.clear()
    astro_api.configure(astro_api.Deps(
        db=lambda: db,
        verify_uid=lambda h: ME if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        resolve_place=main._resolve_birth_place,
        places_query=main._places_query,
        load_user_docs=lambda uid: (dict(profile), dict(public), dict(user)) if uid == ME else ({}, {}, {}),
        is_premium=lambda u, p, now: premium,
        complete=complete or (lambda *a, **k: ("## The short answer\nA warm match.", "stop")),
        take_rate=lambda *a: True,
    ))
    app = FastAPI()
    app.include_router(astro_api.router)
    return TestClient(app)


@pytest.fixture()
def client():
    return make_client()


def test_today_endpoint(client):
    r = client.post("/astro/today", json={"place": {"lat": 12.97, "lon": 77.59}, "date": "2026-10-10",
                                         "members": [{"id": "k1", "name": "Riya", "relation": "daughter", "birthDate": "2018-03-02"}]},
                    headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["place"]["name"] == "Bengaluru"
    assert body["panchang"]["tithi"]["name"] == "Amavasya"
    assert body["family"][0]["id"] == "k1" and "Riya" in body["family"][0]["text"]
    assert body["upcoming"][0]["date"] == "2026-10-10"


def test_today_uses_profile_place(client):
    r = client.post("/astro/today", json={}, headers=H)
    assert r.status_code == 200 and r.json()["place"]["name"] == "Bengaluru"


def test_today_falls_back_to_the_profile_city():
    # No coordinates anywhere (an unfinished account): the profile's city name still finds the place.
    prof = {k: v for k, v in PROFILE.items() if k not in ("lat", "lon")}
    prof.update({"cityName": "Mumbai", "stateName": "Maharashtra"})
    c = make_client(profile=prof, public={"firstName": "Asha"}, user={})
    body = c.post("/astro/today", json={"place": None}, headers=H).json()
    assert body["place"] == {"name": "Mumbai", "tz": "Asia/Kolkata"}
    nothing = make_client(profile={}, public={}, user={}).post("/astro/today", json={}, headers=H).json()
    assert nothing["place"]["name"] == "New Delhi"


def test_today_takes_odd_input(client):
    # Mid-ocean (no city within 3 degrees) with a zone name that is a folder: India time, no error.
    r = client.post("/astro/today", json={"place": {"lat": 0, "lon": -140, "tz": "Asia"}}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["place"]["tz"] == "Asia/Kolkata"
    # A very long typed name or city is cut, not refused: Today still loads for everyone else.
    r = client.post("/astro/today", json={"members": [
        {"id": "k1", "name": "R" * 100, "relation": "daughter", "birthDate": "2018-03-02", "birthCity": "x" * 120},
        {"id": "k2", "name": "Bad", "birthDate": "2018-02-30"}]}, headers=H)
    assert r.status_code == 200, r.text
    assert [f["id"] for f in r.json()["family"]] == ["k1"]


def test_chart_me_and_other(client):
    r = client.post("/astro/chart", json={}, headers=H)
    assert r.status_code == 200, r.text
    c = r.json()["chart"]
    assert c["name"] == "Asha" and c["basis"] == "lagna" and len(c["planets"]) == 9
    r2 = client.post("/astro/chart", json={"person": {"name": "Riya", "birthDate": "2018-03-02"}}, headers=H)
    assert r2.json()["chart"]["basis"] == "moon"
    bad = client.post("/astro/chart", json={"person": {"name": "X", "birthDate": "02/03/2018"}}, headers=H)
    assert bad.status_code == 422
    impossible = client.post("/astro/chart", json={"person": {"name": "X", "birthDate": "2023-02-31"}}, headers=H)
    assert impossible.status_code == 422 and impossible.json()["detail"] == "bad_birth_date"


def test_someone_without_a_date_is_never_me(client):
    # A typed person with no birth date used to turn into the caller under their name.
    r = client.post("/astro/match", json={"b": {"name": "Vikram"}}, headers=H)
    assert r.status_code == 422 and r.json()["detail"] == "bad_birth_date"
    assert client.post("/astro/chart", json={"person": {"name": "Vikram"}}, headers=H).status_code == 422


def test_match_any_pair(client):
    r = client.post("/astro/match", json={
        "a": {"name": "Rahul", "gender": "male", "birthDate": "1995-01-10", "birthTime": "06:30", "birthCity": "Delhi"},
        "b": {"name": "Neha", "gender": "female", "birthDate": "1997-07-22"}}, headers=H)
    assert r.status_code == 200, r.text
    m = r.json()["match"]
    assert 0 <= m["total"] <= 36 and len(m["kootas"]) == 8
    assert m["a"]["name"] == "Rahul" and m["verdict"]["word"]
    assert not any("You are" in d["why"] for d in m["doshas"])
    # The 10 South Indian poruthams come with every match.
    p = m["porutham"]
    assert [i["key"] for i in p["items"]] == ["dina", "gana", "mahendra", "stree_deergha", "yoni", "rasi",
                                              "rasi_adhipathi", "vasya", "rajju", "vedha"]
    assert 0 <= p["countMin"] <= p["count"] <= p["countMax"] <= 10 and p["verdict"]["word"]
    assert all(i["level"] in ("good", "fair", "no") for i in p["items"])
    # Manglik (C3): without Neha's birth time her side is a check, even when a cancellation would hold whatever
    # the Lagna is (here one does: it is listed, but no verdict); the row is a check while a side is a check.
    mang = next(d for d in m["doshas"] if d["name"] == "Manglik")
    sa, sb = m["a"]["manglik"]["status"], m["b"]["manglik"]["status"]
    assert m["b"]["manglik"]["sure"] is False and m["a"]["manglik"]["sure"] is True
    assert (sa, sb, mang["status"]) == ("no", "check", "check")   # 22 Jul 1997: Jupiter aspects her Mars
    assert "cancelledBy" not in m["b"]["manglik"]
    assert m["b"]["manglik"]["cancelReasons"] == ["Jupiter with or aspecting Mars",
                                                  "Mars's sign is an accepted exception for its house from Venus"]
    assert "stays a check, not a verdict" in mang["why"]
    # 5 Dec 1997, no time: Mars is Manglik from her Moon and Venus. 1.0.7 used to show a red "present" here.
    r2 = client.post("/astro/match", json={
        "a": {"name": "Rahul", "gender": "male", "birthDate": "1995-01-10", "birthTime": "06:30", "birthCity": "Delhi"},
        "b": {"name": "Neha", "gender": "female", "birthDate": "1997-12-05"}}, headers=H).json()["match"]
    assert r2["b"]["manglik"]["status"] == "check" and r2["b"]["manglik"]["manglik"] is True
    row = next(d for d in r2["doshas"] if d["name"] == "Manglik")
    assert row["status"] == "check" and "stays a check, not a verdict" in row["why"]


def test_match_me_with_someone(client):
    r = client.post("/astro/match", json={"b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"}}, headers=H)
    assert r.status_code == 200 and r.json()["match"]["a"]["name"] == "Asha"


def _chart_and_match_manglik(c, person):
    astro_api._hits.clear()
    chart = c.post("/astro/chart", json={"person": person}, headers=H)
    match = c.post("/astro/match", json={"a": person, "b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"}},
                   headers=H)
    assert chart.status_code == 200 and match.status_code == 200, (chart.text, match.text)
    return chart.json()["chart"], match.json()["match"]["a"]["manglik"]


def test_chart_and_match_give_one_manglik_status(client):
    """C3: one status per chart. /chart used to read Manglik without Jupiter, so a chart page could say 'yes'
    where the match (and Fates, and the biodata hint) said 'no'."""
    fixtures = json.load(open(os.path.join(os.path.dirname(__file__), "manglik_fixtures.json"), encoding="utf-8"))
    ex = next(x for x in fixtures["charts"] if x["id"] == "shraddha-kapoor")   # 1989-03-03 12:00 Mumbai
    person = {"name": "Shraddha", "gender": "female", "birthDate": ex["birthDate"], "birthTime": ex["birthTime"],
              "birthCity": "Mumbai", "birthState": "Maharashtra"}
    chart, from_match = _chart_and_match_manglik(client, person)
    assert chart["placeKnown"] and chart["manglik"] == from_match
    assert chart["manglik"]["status"] == ex["expected"]["status"] == "no"
    assert chart["manglik"]["cancelReasons"] == ex["expected"]["cancelReasons"] == ["Jupiter with or aspecting Mars"]
    # Mars sits in the Lagna, Jupiter aspects it: the Mangal line follows the cancelled status.
    assert next(p for p in chart["planets"] if p["code"] == "Ma")["house"] == 1
    assert [l["title"] for l in chart["lines"] if l["title"].startswith("Mangal:")] == ["Mangal: not Manglik"]
    # Any chart, with or without a time: the two routes agree on the whole Manglik reading.
    rng = random.Random(1011)
    for _ in range(40):
        p = {"name": "P", "gender": "female",
             "birthDate": "%04d-%02d-%02d" % (rng.randint(1970, 2005), rng.randint(1, 12), rng.randint(1, 28)),
             "birthTime": "%02d:%02d" % (rng.randint(0, 23), rng.randint(0, 59)) if rng.random() < 0.75 else "",
             "birthCity": rng.choice(["Mumbai", "Delhi", "Chennai", "Bengaluru"])}
        chart, from_match = _chart_and_match_manglik(client, p)
        assert chart["manglik"] == from_match, p
        assert (chart["manglik"]["status"] == "check") == (not p["birthTime"]), p


def test_report_needs_parrot_plus(client):
    r = client.post("/astro/match/report", json={"b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"}}, headers=H)
    assert r.status_code == 402
    r = client.post("/astro/match/report", json={"b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"},
                                                 "system": "porutham"}, headers=H)
    assert r.status_code == 402


def test_good_days_endpoint(client):
    r = client.post("/astro/good-days", json={"event": "naming", "place": {"lat": 28.61, "lon": 77.21}, "days": 30,
                                             "people": [{"me": True}, {"name": "Baby", "birthDate": "2026-09-01"},
                                                        {"name": "Typo", "birthDate": "2026-02-30"}, {"me": True}]}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"].startswith("Naming") and body["place"]["name"]
    assert all(len(d["people"]) == 2 for d in body["days"])   # the impossible date and the second "me" are left out


def test_question_without_db(client):
    r = client.post("/astro/question", json={"vote": 1}, headers=H)
    assert r.status_code == 200 and r.json()["question"]["options"]


def test_auth_required(client):
    assert client.post("/astro/today", json={}).status_code == 401


def test_places_search(client):
    r = client.post("/astro/places", json={"q": "benga"}, headers=H)
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["places"]]
    assert names[0] == "Bengaluru"
    assert client.post("/astro/places", json={"q": "b"}, headers=H).json()["places"] == []
    assert client.post("/astro/places", json={"q": "%%"}, headers=H).json()["places"] == []   # no wildcards
    assert client.post("/astro/places", json={"q": "m" * 90}, headers=H).status_code == 200


def test_scores_without_db_is_empty(client):
    r = client.post("/astro/scores", json={"uids": ["abcdefgh", "../bad"]}, headers=H)
    assert r.status_code == 200 and r.json()["scores"] == {}


SOON = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()


def _member_docs(uid, birth="1996-04-11", public=None, profile=None):
    prof = {"firstName": "Neha", "gender": "Female", "birthDate": birth, "birthTime": "05:20",
            "birthCityName": "Jaipur", "birthCountryIso2": "IN", **(profile or {})}
    if birth is None:
        prof.pop("birthDate")
    return {f"profiles/{uid}": prof,
            f"publicProfiles/{uid}": {"firstName": "Neha", "gender": "Female", "isDiscoverable": True, **(public or {})}}


def test_scores_only_for_people_i_could_see():
    docs = {}
    docs.update(_member_docs("bride1234"))
    docs.update(_member_docs("nobirth12", birth=None))
    docs.update(_member_docs("hidden123", public={"isDiscoverable": False}))
    docs.update(_member_docs("paused123", public={"isProfileActive": False}))
    docs.update(_member_docs("banned123", profile={"moderationStatus": "banned"}))
    docs.update(_member_docs("suspend12", profile={"moderationStatus": "suspended", "moderationUntil": SOON}))
    docs.update(_member_docs("wasoff123", profile={"moderationStatus": "suspended", "moderationUntil": PAST}))
    docs.update(_member_docs("iblocked1"))
    docs.update(_member_docs("blockme12"))
    docs.update(_member_docs("kundli123", profile={"profileFor": "none"}))
    docs.update(_member_docs("typo12345", birth="2001-02-30"))
    docs.update(_member_docs("after1234", birth="1997-07-22"))
    docs["profiles/nocard123"] = {"birthDate": "1995-05-05", "gender": "Female"}   # no public card
    docs[f"users/{ME}/blocks/iblocked1"] = {"at": 1}
    docs[f"users/{ME}/blockedBy/blockme12"] = {"at": 1}
    c = make_client(profile={"firstName": "Rahul", "gender": "Male", "birthDate": "1993-09-02"}, public={}, user={},
                    db=FakeDb(docs))
    ids = ["bride1234", "nobirth12", "hidden123", "paused123", "banned123", "suspend12", "wasoff123", "iblocked1",
           "blockme12", "kundli123", "typo12345", "after1234", "nocard123", "ghost1234", ME, "../bad"]
    sc = c.post("/astro/scores", json={"uids": ids}, headers=H).json()["scores"]
    assert list(sc) == ["bride1234", "wasoff123", "after1234"]
    assert all(0 <= s["total"] <= 36 and 0 <= s["min"] <= s["max"] <= 36 for s in sc.values())


def test_scores_follow_my_own_birth_details():
    db = FakeDb(_member_docs("bride1234"))
    me = {"firstName": "Rahul", "gender": "Male", "birthDate": "1993-09-02"}
    c = make_client(profile=me, public={}, user={}, db=db)
    first = c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H).json()["scores"]
    reads = db.reads
    again = c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H).json()["scores"]
    assert again == first and db.reads == reads           # from the cache
    me["birthDate"] = "1993-09-20"                         # I fixed my birth date
    c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H)
    assert db.reads > reads                                 # worked out again, not the old score


def test_member_match_reads_their_profile_on_the_server():
    db = FakeDb(_member_docs("bride1234"))
    c = make_client(profile={**PROFILE, "gender": "Male", "firstName": "Rahul"}, db=db)
    r = c.post("/astro/match", json={"b": {"uid": "bride1234", "name": "N."}}, headers=H)
    assert r.status_code == 200, r.text
    m = r.json()["match"]
    assert m["a"]["name"] == "Rahul" and m["b"]["name"] == "Neha" and 0 <= m["total"] <= 36
    assert m["bMoon"]["pada"] == 0 and m["bMoon"]["nakshatra"]
    for secret in ("1996", "04-11", "05:20", "Jaipur"):   # nothing of their birth details comes back
        assert secret not in r.text
    # The porutham part is there and carries no longitude, pada, options or birth field.
    p = m["porutham"]
    assert len(p["items"]) == 10 and p["bRajju"]

    def keys_of(v):
        if isinstance(v, dict):
            return set(v) | {k for x in v.values() for k in keys_of(x)}
        if isinstance(v, list):
            return {k for x in v for k in keys_of(x)}
        return set()
    assert not keys_of(p) & {"lon", "moonLon", "pada", "options", "birthDate", "birthTime", "birthCityName",
                             "lagna", "marsRashi", "jupiterRashi"}
    assert "jupiterRashi" not in r.text and "options" not in r.text
    # Their chart would give the birth moment away; good dates may count them in.
    assert c.post("/astro/chart", json={"person": {"uid": "bride1234"}}, headers=H).status_code == 403
    g = c.post("/astro/good-days", json={"event": "wedding", "days": 120, "people": [{"me": True}, {"uid": "bride1234"}]},
               headers=H).json()
    assert all([p["name"] for p in d["people"]] == ["Rahul", "Neha"] for d in g["days"])
    assert "1996" not in json.dumps(g)


def test_member_match_refused_when_hidden_or_blocked():
    docs = {}
    docs.update(_member_docs("bride1234"))
    docs.update(_member_docs("hidden123", public={"isDiscoverable": "false"}))
    docs.update(_member_docs("banned123", public={"moderationStatus": "banned"}))
    docs.update(_member_docs("nobirth12", birth=None))
    docs.update(_member_docs("blockme12"))
    docs.update(_member_docs("iblocked1"))
    docs[f"users/blockme12/blocks/{ME}"] = {"at": 1}
    docs[f"users/{ME}/blocks/iblocked1"] = {"at": 1}
    c = make_client(db=FakeDb(docs))

    def status(uid):
        r = c.post("/astro/match", json={"b": {"uid": uid}}, headers=H)
        return r.status_code, r.json().get("detail")

    assert status("bride1234")[0] == 200
    for uid in ("hidden123", "banned123", "blockme12", "iblocked1", "ghost1234", ME, "../../x"):
        assert status(uid) == (404, "member_unavailable"), uid
    assert status("nobirth12") == (422, "member_no_birth_date")
    assert make_client(db=None).post("/astro/match", json={"b": {"uid": "bride1234"}}, headers=H).status_code == 503


def test_question_counts_only_the_question_shown():
    db = FakeDb()
    c = make_client(db=db)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    q = astro_content.question_for(today)
    vote_path = f"dailyQuestionVotes/{today.isoformat()}__{ME}"
    # Yesterday's question still on screen just after midnight: nothing is counted, today's comes back.
    r = c.post("/astro/question", json={"vote": 1, "qid": "yesterdays_question"}, headers=H).json()
    assert r["voted"] is None and r["question"]["id"] == q["id"] and vote_path not in db.docs
    # The right question; the place is the phone's own name for it, the count goes under the database's city.
    place = {"lat": 12.97, "lon": 77.59, "city": "Bangalore"}
    r = c.post("/astro/question", json={"vote": 1, "qid": q["id"], "place": place}, headers=H).json()
    assert r["voted"] == 1 and r["results"]["total"] == 1 and r["results"]["city"]["name"] == "Bangalore"
    counts = db.docs[f"dailyQuestions/{today.isoformat()}"]
    assert counts["counts"] == {"1": 1} and counts["cities"] == {"bengaluru": {"1": 1}}
    # Once only, even from another phone at the same moment.
    r = c.post("/astro/question", json={"vote": 0, "qid": q["id"], "place": place}, headers=H).json()
    assert r["voted"] == 1
    db.hide_once.add(vote_path)
    r = c.post("/astro/question", json={"vote": 0, "qid": q["id"], "place": place}, headers=H).json()
    assert r["voted"] == 1 and db.docs[f"dailyQuestions/{today.isoformat()}"]["counts"] == {"1": 1}


def test_family_context_sends_ages_not_birth_dates(monkeypatch):
    born = (datetime.now(timezone.utc) - timedelta(days=365 * 7 + 30)).date().isoformat()
    db = FakeDb({"users/u1/family/k1": {"name": "Riya", "relation": "Daughter", "birthDate": born},
                 "users/u1/family/k2": {"name": "Papa", "relation": "Father"},
                 "users/u1/family/k3": {"name": "Odd", "relation": "Son", "birthDate": "2020-02-30"}})
    monkeypatch.setattr(main, "firestore_client", db)
    ctx = main._family_context("u1")
    assert "Riya (Daughter, age 7, Moon " in ctx and "Papa (Father)" in ctx and "Odd (Son)" in ctx
    assert born not in ctx and "born" not in ctx


# ------------------------------------------------------------------ porutham readings, scores, Moon keys (W1-S-ASTRO)
VIKRAM = {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14", "birthTime": "09:10", "birthCity": "Chennai",
          "birthState": "Tamil Nadu"}


def _reports(db):
    return {p: d for p, d in db.docs.items() if p.startswith("astroReports/")}


def test_report_facts_and_keys_per_system():
    calls = []

    def complete(msgs, *a, **k):
        calls.append(msgs)
        return "## The short answer\nA warm match.", "stop"

    db = FakeDb()
    c = make_client(db=db, premium=True, complete=complete)
    north = c.post("/astro/match/report", json={"b": VIKRAM}, headers=H).json()
    assert north["cached"] is False and len(calls) == 1
    # The same 36-guna facts with or without a system: the same cache key, so a stored North reading still hits.
    again = c.post("/astro/match/report", json={"b": VIKRAM, "system": "gunas"}, headers=H).json()
    assert again["cached"] is True and len(calls) == 1
    assert c.post("/astro/match/report", json={"b": VIKRAM, "system": " North "}, headers=H).json()["cached"] is True
    north_docs = _reports(db)
    assert len(north_docs) == 1
    (north_path, north_doc), = north_docs.items()
    assert north_doc["system"] == "gunas" and north_doc["uid"] == ME
    north_facts = calls[0][1]["content"]
    assert north_facts == "FACTS: " + str(astro_api._report_facts(north["match"], False))
    assert "'kootas'" in north_facts and calls[0][0]["content"] == astro_api.REPORT_SYSTEM

    south = c.post("/astro/match/report", json={"b": VIKRAM, "system": "Porutham"}, headers=H).json()
    assert south["cached"] is False and len(calls) == 2
    docs = _reports(db)
    assert len(docs) == 2
    (south_path, south_doc), = [(p, d) for p, d in docs.items() if p != north_path]
    assert south_doc["system"] == "porutham" and south_doc["uid"] == ME
    facts = calls[1][1]["content"]
    assert "'kootas'" not in facts and "'poruthams'" in facts and "'rajjuOk'" in facts and "'Nadi dosha'" not in facts
    assert "'system': 'porutham'" in facts and "Manglik" in facts
    system = calls[1][0]["content"]
    assert "Speak in poruthams, never in gunas or points out of 36." in system
    assert "Ashtakoota points" not in system
    for heading in ("## The short answer", "## Where they click", "## Where they need patience", "## Doshas, honestly",
                    "## Mithu's advice", "Under 330 words"):
        assert heading in system
    # Asked again in poruthams: the stored South reading.
    assert c.post("/astro/match/report", json={"b": VIKRAM, "system": "porutham"}, headers=H).json()["cached"] is True


def test_report_system_swaps_one_sentence():
    assert astro_api._FACTS_GUNAS in astro_api.REPORT_SYSTEM
    south = astro_api._report_system(True)
    assert south != astro_api.REPORT_SYSTEM and astro_api._report_system(False) == astro_api.REPORT_SYSTEM
    assert south.replace(astro_api._FACTS_PORUTHAM, astro_api._FACTS_GUNAS) == astro_api.REPORT_SYSTEM


def test_report_without_porutham_falls_back_to_gunas(monkeypatch):
    calls = []

    def complete(msgs, *a, **k):
        calls.append(msgs)
        return "## The short answer\nA warm match.", "stop"

    monkeypatch.setattr(astro_api.porutham, "dasa_porutham", lambda *a, **k: 1 / 0)
    db = FakeDb()
    c = make_client(db=db, premium=True, complete=complete)
    r = c.post("/astro/match/report", json={"b": VIKRAM, "system": "porutham"}, headers=H).json()
    assert "porutham" not in r["match"] and "'kootas'" in calls[0][1]["content"]
    assert next(iter(_reports(db).values()))["system"] == "gunas"


def test_a_porutham_failure_never_takes_the_match_down(monkeypatch):
    monkeypatch.setattr(astro_api.porutham, "dasa_porutham", lambda *a, **k: 1 / 0)
    c = make_client(db=FakeDb(_member_docs("bride1234")), profile={"firstName": "Rahul", "gender": "Male",
                                                                   "birthDate": "1993-09-02"}, public={}, user={})
    r = c.post("/astro/match", json={"b": VIKRAM}, headers=H)
    assert r.status_code == 200, r.text
    m = r.json()["match"]
    assert "porutham" not in m and 0 <= m["total"] <= 36 and len(m["kootas"]) == 8
    sc = c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H).json()["scores"]
    assert set(sc["bride1234"]) == {"total", "min", "max"}


def test_scores_carry_the_porutham_count():
    db = FakeDb({**_member_docs("bride1234"), **_member_docs("after1234", birth="1997-07-22",
                                                               profile={"birthTime": ""})})
    c = make_client(profile={"firstName": "Rahul", "gender": "Male", "birthDate": "1993-09-02"}, public={}, user={}, db=db)
    sc = c.post("/astro/scores", json={"uids": ["bride1234", "after1234"]}, headers=H).json()["scores"]
    assert set(sc) == {"bride1234", "after1234"}
    for s in sc.values():
        assert set(s) == {"total", "min", "max", "p", "pMin", "pMax", "rajju"}   # only scores, nothing else new
        assert 0 <= s["pMin"] <= s["p"] <= s["pMax"] <= 10 and isinstance(s["rajju"], bool)
        assert all(isinstance(s[k], int) for k in ("p", "pMin", "pMax"))


# ------------------------------------------------------------------ Moon keys
class CountingDb(FakeDb):
    def __init__(self, docs=None):
        super().__init__(docs)
        self.moon_writes = []

    def write(self, path, data, merge):
        if path.endswith("/feedState/moon"):
            self.moon_writes.append(path)
        super().write(path, data, merge)


RAHUL_PROFILE = {"firstName": "Rahul", "gender": "Male", "birthDate": "1993-09-02", "birthTime": "06:15",
                 "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN"}
RAHUL_PUBLIC = {"firstName": "Rahul", "gender": "Male", "cityName": "Pune"}


def test_scores_remember_moon_keys_once_per_sig():
    docs = {**_member_docs("bride1234"), **_member_docs("after1234", birth="1997-07-22")}
    db = CountingDb(docs)
    profile = dict(RAHUL_PROFILE)
    c = make_client(profile=profile, public=RAHUL_PUBLIC, user={"birthDate": "1990-01-01"}, db=db)
    c.post("/astro/scores", json={"uids": ["bride1234", "after1234"]}, headers=H)
    assert sorted(db.moon_writes) == sorted([f"users/{ME}/feedState/moon", "users/bride1234/feedState/moon",
                                             "users/after1234/feedState/moon"])
    mine = db.docs[f"users/{ME}/feedState/moon"]
    # Signed over my card under my profile, never the users doc (its birthDate differs on purpose).
    assert mine["sig"] == astro_api._moon_sig({**RAHUL_PUBLIC, **RAHUL_PROFILE})
    assert set(mine) == {"main", "options", "exact", "manglik", "sig", "atIso"}
    assert mine["exact"] is True and mine["manglik"] in ("yes", "mild", "no", "unknown")
    assert len(mine["main"]) == 3 and all(len(o) == 2 for o in mine["options"])
    theirs = db.docs["users/bride1234/feedState/moon"]
    assert theirs["sig"] == astro_api._moon_sig({**docs["publicProfiles/bride1234"], **docs["profiles/bride1234"]})
    # Fresh scores again (cache cleared) with the same details: nothing is written twice.
    astro_api._scores_cache.clear()
    c.post("/astro/scores", json={"uids": ["bride1234", "after1234"]}, headers=H)
    assert len(db.moon_writes) == 3
    # I fix my birth time: my keys are written again, theirs are not.
    profile["birthTime"] = "06:45"
    astro_api._scores_cache.clear()
    c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H)
    assert db.moon_writes[3:] == [f"users/{ME}/feedState/moon"]
    assert db.docs[f"users/{ME}/feedState/moon"]["sig"] == astro_api._moon_sig({**RAHUL_PUBLIC, **profile})


def test_moon_keys_only_for_me_and_rishta_cards():
    db = CountingDb(_member_docs("bride1234"))
    c = make_client(profile=dict(RAHUL_PROFILE), public=RAHUL_PUBLIC, user={}, db=db)
    # Typed-in people, family members and a member matched by uid: no Moon keys.
    c.post("/astro/match", json={"b": VIKRAM}, headers=H)
    c.post("/astro/match", json={"a": VIKRAM, "b": {"name": "Neha", "gender": "female", "birthDate": "1997-07-22"}},
           headers=H)
    c.post("/astro/match", json={"b": {"uid": "bride1234"}}, headers=H)
    c.post("/astro/chart", json={"person": {"name": "Riya", "birthDate": "2018-03-02"}}, headers=H)
    c.post("/astro/today", json={"members": [{"id": "k1", "name": "Riya", "birthDate": "2018-03-02"}]}, headers=H)
    c.post("/astro/good-days", json={"event": "naming", "days": 30, "people": [{"name": "Baby", "birthDate": "2026-09-01"}]},
           headers=H)
    assert db.moon_writes == []
    # My own chart: mine, once.
    assert c.post("/astro/chart", json={}, headers=H).status_code == 200
    assert c.post("/astro/chart", json={"person": {"me": True}}, headers=H).status_code == 200
    assert db.moon_writes == [f"users/{ME}/feedState/moon"]


def test_no_moon_keys_for_a_kundli_only_account():
    db = CountingDb(_member_docs("bride1234"))
    c = make_client(profile={**RAHUL_PROFILE, "profileFor": "none"}, public={}, user={}, db=db)
    c.post("/astro/chart", json={}, headers=H)
    c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H)
    assert db.moon_writes == ["users/bride1234/feedState/moon"]
    # A kundli-only card is never scored, so never written either (the _visible gate).
    db2 = CountingDb(_member_docs("kundli123", profile={"profileFor": "none"}))
    make_client(profile=dict(RAHUL_PROFILE), public=RAHUL_PUBLIC, user={}, db=db2).post(
        "/astro/scores", json={"uids": ["kundli123"]}, headers=H)
    assert "users/kundli123/feedState/moon" not in db2.docs


def test_a_failed_moon_write_never_breaks_scores():
    class Broken(FakeDb):
        def batch(self):
            raise RuntimeError("firestore down")

    c = make_client(profile=dict(RAHUL_PROFILE), public=RAHUL_PUBLIC, user={}, db=Broken(_member_docs("bride1234")))
    r = c.post("/astro/scores", json={"uids": ["bride1234"]}, headers=H)
    assert r.status_code == 200 and "bride1234" in r.json()["scores"]
    assert astro_api._moon_written == {}   # not marked written: the next call tries again
    assert c.post("/astro/chart", json={}, headers=H).status_code == 200


def test_moon_sig_rules():
    base = {"birthDate": "1993-09-02", "birthTime": " 06:15 ", "birthCityName": "Pune", "gender": "Male"}
    same = {**base, "birthTime": "06:15", "birthStateName": None, "firstName": "Rahul", "birthCountryIso2": 91}
    assert astro_api._moon_sig(base) == astro_api._moon_sig(same)   # trimmed; non-strings count as empty
    raw = "1993-09-02|06:15|Pune||||Male"
    import hashlib
    assert astro_api._moon_sig(base) == hashlib.sha1(raw.encode()).hexdigest()[:12]
    assert astro_api._moon_sig({**base, "birthTime": "06:16"}) != astro_api._moon_sig(base)


def test_moon_dedupe_dict_is_capped(monkeypatch):
    monkeypatch.setattr(astro_api, "_MOON_MAX", 10)
    astro_api._moon_written.clear()
    db = FakeDb()
    chart = fc.chart_from_birth((1990, 1, 1), None, None)
    for i in range(25):
        astro_api._remember_moon(db, f"user{i:04d}", {"birthDate": "1990-01-01"}, chart)
    assert len(astro_api._moon_written) <= 10 and "user0024" in astro_api._moon_written
    astro_api._moon_written.clear()


def test_moon_keys_shape():
    import fates_astro as fa
    dhanu_late = {"rashi": 8, "nak": 19, "moonLon": 259.0, "options": [[8, 19], [8, 20]], "precision": "date",
                  "marsRashi": 2, "venusRashi": 5, "lagna": None}
    k = fa.moon_keys(dhanu_late)
    assert k == {"main": [8, 19, 1], "options": [[8, 19], [8, 20]], "exact": False, "manglik": "unknown"}
    assert fa.moon_keys({**dhanu_late, "moonLon": 248.0})["main"] == [8, 19, 0]          # first half of Dhanu
    assert fa.moon_keys({**dhanu_late, "rashi": 9, "nak": 21, "moonLon": 287.5})["main"] == [9, 21, 1]   # Makara
    assert fa.moon_keys({**dhanu_late, "rashi": 4, "nak": 9, "moonLon": 129.0})["main"] == [4, 9, 0]     # other signs: 0
    assert fa.moon_keys({**dhanu_late, "moonLon": None})["main"] == [8, 19, 0]           # as Python reads options
    # Mars in Simha; the Moon and Venus in Mithuna (Mars 3rd from them, never Manglik).
    exact = {"rashi": 2, "nak": 5, "moonLon": 70.0, "precision": "exact", "marsRashi": 4, "venusRashi": 2,
             "lagna": (4 - 7 + 1) % 12, "options": [[2, 5]]}                            # Mars 7th from the Lagna
    assert fa.moon_keys(exact)["exact"] is True and fa.moon_keys(exact)["manglik"] == "yes"
    assert fa.moon_keys({**exact, "lagna": (4 - 2 + 1) % 12})["manglik"] == "mild"        # 2nd only: partial -> mild
    assert fa.moon_keys({**exact, "marsRashi": 0, "lagna": 0})["manglik"] == "no"          # Mars in its own sign
    # The same cancellation without a birth time: unknown, never a confirmed "no" for pref_manglik.
    assert fa.moon_keys({**dhanu_late, "marsRashi": 0})["manglik"] == "unknown"
    assert fa.moon_keys({**dhanu_late, "marsRashi": 0, "jupiterRashi": 0})["manglik"] == "unknown"
    assert fa.moon_keys({"rashi": 1, "nak": 3})["options"] == [[1, 3]]


# ------------------------------------------------------------------ tools: the guna table and the Moon-key backfill
import importlib.util as _ilu  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_TOOLS = _Path(__file__).resolve().parents[1] / "tools"


def _tool(name):
    spec = _ilu.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _guna_between(table, a, b, ga, gb):
    """What guna.ts does with gunaTable.json (the tool's docstring is the spec)."""
    import fates_astro as fa
    idx = {k: i for i, k in enumerate(table["keys"])}

    def points(x, y):
        runs = [table["total"][idx[x]][idx[y]] if boy else table["total"][idx[y]][idx[x]] for boy in fa.gender_roles(ga, gb)]
        return sum(runs) / len(runs)

    key = lambda m: f"{m[0]}_{m[1]}_{m[2] if len(m) > 2 else 0}"   # noqa: E731
    total = round(points(key(a["main"]), key(b["main"])) * 2) / 2    # Python rounds half to even, as match_charts
    opts = [points(key(oa), key(ob)) for oa in a["options"] for ob in b["options"]]
    return total, min(opts), max(opts)


def test_guna_export_is_deterministic_and_matches_match_charts(tmp_path):
    import fates_astro as fa
    tool = _tool("export_guna_table")
    for run in ("one", "two"):
        assert tool.main(["--out-dir", str(tmp_path / run), "--fixture", str(tmp_path / run / "fixture.json")]) == 0
    for name in ("gunaTable.json", "fixture.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name
    # The committed fixture is today's output: regenerate it with the tool after any table change.
    committed = _Path(__file__).resolve().parent / "guna_pairs_fixture.json"
    assert committed.read_bytes().replace(b"\r\n", b"\n") == (tmp_path / "one" / "fixture.json").read_bytes()   # autocrlf
    table = json.loads((tmp_path / "one" / "gunaTable.json").read_text(encoding="utf-8"))
    assert (tmp_path / "one" / "gunaTable.json").stat().st_size < 15000
    assert len(table["keys"]) == 42 and len(table["total"]) == 42 and all(len(r) == 42 for r in table["total"])
    assert {"8_18_1", "8_19_0", "8_19_1", "9_21_0", "9_21_1"} <= set(table["keys"])
    assert all(0 <= v <= 36 for row in table["total"] for v in row)
    pairs = json.loads(committed.read_text(encoding="utf-8"))["pairs"]
    assert len(pairs) == 300
    for p in pairs:
        # The table alone gives match_charts' answer for every pair...
        assert _guna_between(table, p["a"], p["b"], p["aGender"], p["bGender"]) == (p["total"], p["totalMin"], p["totalMax"]), p
        # ...and the fixture is match_charts' answer for charts with those Moons.
        charts = [{"rashi": s["main"][0], "nak": s["main"][1], "moonLon": s["main"][0] * 30.0 + (22.5 if s["main"][2] else 7.5),
                   "options": s["options"]} for s in (p["a"], p["b"])]
        m = fa.match_charts(charts[0], charts[1], p["aGender"], p["bGender"])
        assert (m["total"], m["totalMin"], m["totalMax"]) == (p["total"], p["totalMin"], p["totalMax"])
    # The cases the TS port has to get right are all there.
    genders = {(p["aGender"], p["bGender"]) for p in pairs}
    assert {("male", "male"), ("female", "female"), ("", "")} <= genders
    assert any(p[s]["main"][0] in (8, 9) and p[s]["main"][2] == 1 for p in pairs for s in ("a", "b"))
    assert any(p[s]["main"][0] in (8, 9) and p[s]["main"][2] == 0 for p in pairs for s in ("a", "b"))
    assert any(len(p["a"]["options"]) > 1 and len(p["b"]["options"]) > 1 for p in pairs)
    assert any(float(p["total"]) % 1 == 0.5 and len(fa.gender_roles(p["aGender"], p["bGender"])) == 2 for p in pairs)


def test_no_quarter_averages_with_todays_tables():
    """Both directions averaged always land on a half point (Vashya and Graha Maitri are symmetric), so Python's
    half-to-even rounding never changes a total today. If a table change breaks this, guna.ts must round half to
    even exactly like Python, and the fixture gains .25 cases."""
    tool = _tool("export_guna_table")
    t = tool.table()
    n = len(t["keys"])
    assert all(((t["total"][i][j] + t["total"][j][i]) / 2 * 4) % 2 == 0 for i in range(n) for j in range(n))


def test_backfill_moon_keys(capsys):
    tool = _tool("backfill_moon_keys")
    docs = {}
    docs.update(_member_docs("bride1234"))
    docs.update(_member_docs("notime123", profile={"birthTime": ""}))
    docs.update(_member_docs("kundli123", profile={"profileFor": "none"}))
    docs.update(_member_docs("nobirth12", birth=None))
    docs["publicProfiles/noprof123"] = {"firstName": "Ghost"}
    db = CountingDb(docs)
    dry = tool.run(db, main._resolve_birth_place)
    assert db.moon_writes == [] and "DRY RUN" in capsys.readouterr().out
    assert (dry["cards"], dry["due"], dry["kundli_only"], dry["no_birth_date"], dry["no_profile"]) == (5, 2, 1, 1, 1)
    done = tool.run(db, main._resolve_birth_place, apply=True)
    assert done["written"] == 2 and sorted(db.moon_writes) == ["users/bride1234/feedState/moon", "users/notime123/feedState/moon"]
    # Exactly what /scores writes for the same card.
    astro_api._moon_written.clear()
    db2 = CountingDb(_member_docs("bride1234"))
    make_client(profile=dict(RAHUL_PROFILE), public=RAHUL_PUBLIC, user={}, db=db2).post(
        "/astro/scores", json={"uids": ["bride1234"]}, headers=H)
    a, b = dict(db.docs["users/bride1234/feedState/moon"]), dict(db2.docs["users/bride1234/feedState/moon"])
    a.pop("atIso"), b.pop("atIso")
    assert a == b
    assert len(db.docs["users/notime123/feedState/moon"]["options"]) >= 1
    again = tool.run(db, main._resolve_birth_place, apply=True)
    assert again["up_to_date"] == 2 and again["written"] == 0 and len(db.moon_writes) == 2
    assert "--apply" in tool.__doc__ and "dry run" in tool.__doc__.lower()
