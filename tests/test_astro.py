from datetime import date, datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import astro_api
import astro_content
import astro_core as ac
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


def test_upcoming_has_festivals_and_no_duplicate_vrats():
    items = ac.upcoming(date(2026, 10, 10), 45, *DELHI, limit=30)
    names = [i["name"] for i in items]
    assert "Sharad Navratri begins" in names and "Karva Chauth" in names and "Diwali" in names
    assert "Sankashti Chaturthi" not in names            # Karva Chauth already is that day
    eka = [i["date"] for i in items if i["kind"] == "ekadashi"]
    assert all(abs((date.fromisoformat(a) - date.fromisoformat(b)).days) > 3 for a, b in zip(eka, eka[1:]))


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
    assert all(d["weekday"] in ("Monday", "Wednesday", "Thursday", "Friday") or d["window"]["night"] for d in g["days"])


def test_day_line_tones():
    line = ac.day_line("Riya", "daughter", 0, 1, 4, 3)   # Moon 5th from hers
    assert "Riya" in line["text"] and "studies" in line["text"]
    assert line["tone"] in ("good", "steady", "gentle")


def test_question_rotates():
    a = astro_content.question_for(date(2026, 10, 10))
    b = astro_content.question_for(date(2026, 10, 11))
    assert a["id"] != b["id"] and 2 <= len(a["options"]) <= 4


# ------------------------------------------------------------------ HTTP
@pytest.fixture()
def client():
    me = {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21", "birthTime": "07:40",
          "birthCityName": "Pune", "birthCountryIso2": "IN", "lat": 12.97, "lon": 77.59}
    astro_api.configure(astro_api.Deps(
        db=lambda: None,
        verify_uid=lambda h: "u1" if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        resolve_place=main._resolve_birth_place,
        places_query=main._places_query,
        load_user_docs=lambda uid: (me, {}, me),
        is_premium=lambda u, p, now: False,
        complete=lambda *a, **k: ("text", "stop"),
        take_rate=lambda *a: True,
    ))
    app = FastAPI()
    app.include_router(astro_api.router)
    return TestClient(app)


H = {"Authorization": "Bearer ok"}


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


def test_chart_me_and_other(client):
    r = client.post("/astro/chart", json={}, headers=H)
    assert r.status_code == 200, r.text
    c = r.json()["chart"]
    assert c["name"] == "Asha" and c["basis"] == "lagna" and len(c["planets"]) == 9
    r2 = client.post("/astro/chart", json={"person": {"name": "Riya", "birthDate": "2018-03-02"}}, headers=H)
    assert r2.json()["chart"]["basis"] == "moon"
    bad = client.post("/astro/chart", json={"person": {"name": "X", "birthDate": "02/03/2018"}}, headers=H)
    assert bad.status_code == 422


def test_match_any_pair(client):
    r = client.post("/astro/match", json={
        "a": {"name": "Rahul", "gender": "male", "birthDate": "1995-01-10", "birthTime": "06:30", "birthCity": "Delhi"},
        "b": {"name": "Neha", "gender": "female", "birthDate": "1997-07-22"}}, headers=H)
    assert r.status_code == 200, r.text
    m = r.json()["match"]
    assert 0 <= m["total"] <= 36 and len(m["kootas"]) == 8
    assert m["a"]["name"] == "Rahul" and m["verdict"]["word"]
    assert not any("You are" in d["why"] for d in m["doshas"])


def test_match_me_with_someone(client):
    r = client.post("/astro/match", json={"b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"}}, headers=H)
    assert r.status_code == 200 and r.json()["match"]["a"]["name"] == "Asha"


def test_report_needs_parrot_plus(client):
    r = client.post("/astro/match/report", json={"b": {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14"}}, headers=H)
    assert r.status_code == 402


def test_good_days_endpoint(client):
    r = client.post("/astro/good-days", json={"event": "naming", "place": {"lat": 28.61, "lon": 77.21}, "days": 30,
                                             "people": [{"me": True}, {"name": "Baby", "birthDate": "2026-09-01"}]}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"].startswith("Naming") and body["place"]["name"]
    assert all(len(d["people"]) == 2 for d in body["days"])


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
