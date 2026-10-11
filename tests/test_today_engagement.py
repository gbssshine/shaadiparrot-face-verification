"""The R1 engagement data wired into the astro routes (W1b-S-API; engagement N3-N6, N8 route cases):

- /astro/today: the reader's own festivals and fasts, the next big day, other faiths' big days as a greeting, what
  the lists are picked by, where the place came from, and Mithu's note of the day; the old merged list unchanged;
- /astro/forecast: the week ahead, astro_off, one AI call under parallel requests, the 'forecast' daily rate;
- /astro/question: yesterday's answers only with enough votes.

Firestore is an in-memory fake; the place database and the birth-place resolver are the real ones (main).
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.api_core import exceptions as gexc

import astro_api
import astro_content
import astro_core as ac
import astro_festivals as afest
import astro_forecast
import main

ME = "eng123456"
H = {"Authorization": "Bearer ok"}
DELHI_IN = {"lat": 28.61, "lon": 77.21}
KOLKATA_IN = {"lat": 22.57, "lon": 88.36}
# A Hindu reader in New Delhi, birth details exact (time and city).
PROFILE = {"firstName": "Asha", "gender": "Female", "religion": "Hindu", "birthDate": "1994-08-21",
           "birthTime": "07:40", "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN",
           "cityName": "New Delhi", "stateName": "Delhi", "countryIso2": "IN", "lat": 28.6139, "lon": 77.209,
           "lifeStage": "single", "profileFor": "self"}
PUBLIC = {"firstName": "Asha", "gender": "Female", "cityName": "New Delhi", "lat": 28.61, "lon": 77.21}

GOOD_WEEK = (
    "Love & family\n"
    "The week starts softly at home. Family talk over a shared meal brings you closer, and a small kind word "
    "goes a long way with the people you love most.\n\n"
    "Work & money\n"
    "Plans you started earlier move forward now. Keep notes, answer messages on time and take one practical step "
    "each day; steady effort is noticed by the right people.\n\n"
    "Health\n"
    "Sleep a little earlier and take a short walk in the evening. Plenty of water and simple food keep your "
    "energy calm and even through the busy days.\n\n"
    "Best day for…\n"
    "Pick the calmer days for family time and the brighter ones for work calls. Keep one quiet evening just for "
    "rest and your own thoughts."
)


# ------------------------------------------------------------------ a small Firestore stand-in
class Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d, self.exists = ref, ref.id, data, data is not None

    def to_dict(self):
        return None if self._d is None else json.loads(json.dumps(self._d))


class Ref:
    def __init__(self, db, path):
        self._db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def collection(self, name):
        return Col(self._db, f"{self.path}/{name}")

    def get(self, transaction=None):
        with self._db.lock:
            return Snap(self, self._db.docs.get(self.path))

    def set(self, data, merge=False):
        self._db.write(self.path, data, merge)


class Col:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")


class Batch:
    def __init__(self, db):
        self._db, self._ops = db, []

    def create(self, ref, data):
        self._ops.append(("create", ref, data, False))

    def set(self, ref, data, merge=False):
        self._ops.append(("set", ref, data, merge))

    def commit(self):
        with self._db.lock:
            if any(op == "create" and ref.path in self._db.docs for op, ref, _, _ in self._ops):
                raise gexc.AlreadyExists("Document already exists")
        for op, ref, data, merge in self._ops:
            self._db.write(ref.path, data, merge)


class Db:
    def __init__(self, docs=None):
        self.docs = {k: dict(v) for k, v in (docs or {}).items()}
        self.lock = threading.RLock()

    def collection(self, name):
        return Col(self, name)

    def get_all(self, refs):
        refs = list(refs)
        with self.lock:
            return [Snap(r, self.docs.get(r.path)) for r in reversed(refs)]

    def batch(self):
        return Batch(self)

    def write(self, path, data, merge):
        with self.lock:
            cur = dict(self.docs.get(path) or {}) if merge else {}
            for k, v in data.items():
                cur[k] = (cur.get(k) or 0) + v.value if type(v).__name__ == "Increment" else v
            self.docs[path] = cur


class Leases:
    def __init__(self):
        self.held, self.lock, self.taken = set(), threading.Lock(), []

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


def make_client(profile=PROFILE, public=PUBLIC, user=None, db=None, complete=None, rate=None, leases=None,
                today=None, monkeypatch=None):
    """rate: (uid, key, limit, per) -> bool; leases: a Leases or None (the old wiring without leases)."""
    astro_api._hits.clear()
    astro_api._report_hits.clear()
    astro_api._scores_cache.clear()
    astro_api._moon_written.clear()
    if today is not None:
        monkeypatch.setattr(astro_api, "_real_today", lambda tz: today)
    deps = dict(
        db=lambda: db,
        verify_uid=lambda h: ME if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        resolve_place=main._resolve_birth_place,
        places_query=main._places_query,
        load_user_docs=lambda uid: (dict(profile), dict(public), dict(user or {})) if uid == ME else ({}, {}, {}),
        is_premium=lambda u, p, now: False,
        complete=complete or (lambda *a, **k: (GOOD_WEEK, "stop")),
        take_rate=rate or (lambda *a: True),
    )
    if leases is not None:
        deps.update(take_lease=leases.take, release_lease=leases.release)
    astro_api.configure(astro_api.Deps(**deps))
    app = FastAPI()
    app.include_router(astro_api.router)
    return TestClient(app)


def today_of(c, day, place=DELHI_IN, **extra):
    r = c.post("/astro/today", json={"place": place, "date": day.isoformat(), **extra}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def ids(rows):
    return [r["id"] for r in rows]


# ------------------------------------------------------------------ the Deps contract (shared with W1b-S-PDF)
def test_deps_keep_working_without_the_lease_fields():
    """main.py from before this wave builds Deps without take_lease / release_lease: both default to None."""
    d = astro_api.Deps(db=lambda: None, verify_uid=lambda h: ME, resolve_place=lambda p: None,
                       places_query=lambda q, a: [], load_user_docs=lambda u: ({}, {}, {}),
                       is_premium=lambda u, p, n: False, complete=lambda *a: ("", ""), take_rate=lambda *a: True)
    assert d.take_lease is None and d.release_lease is None
    fields = list(astro_api.Deps.__dataclass_fields__)
    assert fields[-2:] == ["take_lease", "release_lease"]


# ------------------------------------------------------------------ /today: the old payload
def test_today_keeps_the_old_payload(monkeypatch):
    c = make_client(today=date(2026, 10, 11), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 10, 11))
    for key in ("ok", "place", "dayKey", "panchang", "upcoming", "family"):
        assert key in body
    # The old merged list: festivals and every vrat, the old shape, from the all-India table (1.0.7 apps).
    assert body["upcoming"] == afest.legacy_merged(date(2026, 10, 11), 40, 28.61, 77.21, "Asia/Kolkata", limit=8,
                                                   vrats=ac.vrats_between)
    assert all(set(i) == {"date", "id", "name", "note", "kind", "type", "inDays"}
               for i in body["upcoming"] if i["type"] == "festival")
    assert ("2026-10-18", "durga_puja") in {(i["date"], i.get("id")) for i in body["upcoming"]}   # the corrected day
    assert body["place"] == {"name": "New Delhi", "tz": "Asia/Kolkata", "source": "app"}


def test_place_source(monkeypatch):
    c = make_client(today=date(2026, 10, 11), monkeypatch=monkeypatch)
    assert c.post("/astro/today", json={"place": {"city": "Jaipur", "state": "Rajasthan"}},
                  headers=H).json()["place"]["source"] == "app"
    assert c.post("/astro/today", json={}, headers=H).json()["place"]["source"] == "profile"
    nowhere = make_client(profile={}, public={}, user={}, today=date(2026, 10, 11), monkeypatch=monkeypatch)
    assert nowhere.post("/astro/today", json={}, headers=H).json()["place"] == {
        "name": "New Delhi", "tz": "Asia/Kolkata", "source": "default"}


# ------------------------------------------------------------------ /today: the reader's own lists
@pytest.mark.parametrize("day", [date(2026, 10, 11), date(2026, 10, 20)])
def test_diwali_is_never_crowded_out_for_new_delhi(day, monkeypatch):
    c = make_client(today=day, monkeypatch=monkeypatch)
    body = today_of(c, day)
    seen = {(r["date"], r["id"]) for r in body["festivals"]} | (
        {(body["bigDay"]["date"], body["bigDay"]["id"])} if body["bigDay"] else set())
    assert ("2026-11-08", "diwali") in seen
    assert len(body["festivals"]) <= afest.FESTIVAL_LIMIT and len(body["fasts"]) <= afest.FAST_LIMIT
    assert body["fasts"] and all(f["type"] == "vrat" for f in body["fasts"])
    assert body["showing"] == {"faiths": ["hindu"], "community": "", "state": "Delhi", "source": "religion"}


def test_a_muslim_reader_gets_eid_and_no_vrats(monkeypatch):
    prof = {**PROFILE, "religion": "Muslim", "firstName": "Sana"}
    c = make_client(profile=prof, today=date(2027, 2, 20), monkeypatch=monkeypatch)
    body = today_of(c, date(2027, 2, 20))
    assert "eid_ul_fitr" in ids(body["festivals"]) and body["fasts"] == []
    assert not {"holi", "shivratri", "holika_dahan"} & set(ids(body["festivals"]))
    assert body["showing"]["faiths"] == ["muslim"]
    # On Diwali the Hindu big day is only a greeting for them.
    diwali = make_client(profile=prof, today=date(2026, 11, 8), monkeypatch=monkeypatch)
    d = today_of(diwali, date(2026, 11, 8))
    assert "diwali" not in ids(d["festivals"])
    assert {"name": "Diwali", "greeting": "Shubh Deepavali"}.items() <= next(
        a for a in d["alsoInIndia"] if a["id"] == "diwali").items()


def test_a_bengali_hindu_family_in_kolkata(monkeypatch):
    prof = {**PROFILE, "community": "Bengali", "stateName": "West Bengal", "cityName": "Kolkata"}
    c = make_client(profile=prof, today=date(2026, 10, 15), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 10, 15), place=KOLKATA_IN)
    got = {(r["date"], r["id"]) for r in body["festivals"]}
    assert ("2026-10-18", "durga_puja") in got
    assert body["showing"] == {"faiths": ["hindu"], "community": "Bengali", "state": "West Bengal",
                               "source": "religion"}


def test_the_users_doc_choices_win(monkeypatch):
    """celebrates and community live on users/{uid} only (never on the card); the merged docs carry them."""
    prof = {k: v for k, v in PROFILE.items() if k != "religion"}
    c = make_client(profile=prof, user={"celebrates": ["christian", "hindu"], "astroMode": "fun"},
                    today=date(2026, 12, 20), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 12, 20))
    assert body["showing"]["source"] == "chosen" and body["showing"]["faiths"] == ["hindu", "christian"]
    assert "christmas" in ids(body["festivals"])
    unknown = make_client(profile=prof, user={}, today=date(2026, 12, 20), monkeypatch=monkeypatch)
    u = today_of(unknown, date(2026, 12, 20))
    assert u["showing"]["source"] == "unknown" and u["fasts"] == []
    assert "christmas" in ids(u["festivals"])        # an unknown faith sees every faith's big days


def test_festival_lists_failing_never_take_today_down(monkeypatch):
    c = make_client(today=date(2026, 10, 11), monkeypatch=monkeypatch)
    monkeypatch.setattr(astro_api.afest, "upcoming", lambda *a, **k: 1 / 0)
    body = today_of(c, date(2026, 10, 11))
    assert body["festivals"] == [] and body["fasts"] == [] and body["bigDay"] is None and body["mithu"] is None
    assert body["panchang"] and body["upcoming"]


# ------------------------------------------------------------------ /today: Mithu's note
def test_mithu_note_on_diwali_is_stored_once(monkeypatch):
    db = Db()
    c = make_client(db=db, today=date(2026, 11, 8), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 11, 8))
    note = body["mithu"]
    assert note["topic"] == "festival" and note["title"] == "Diwali is today"
    assert note["text"].startswith("Shubh Deepavali! Diwali is today. Pradosh time in New Delhi:")
    assert note["id"] == f"{ME}__2026-11-08" and note["day"] == "2026-11-08"
    assert [a["type"] for a in note["actions"]] == ["open_panchang", "share_greeting"]
    again = today_of(c, date(2026, 11, 8))
    assert again["mithu"] == note
    assert len([k for k in db.docs if k.startswith("mithuNotes/")]) == 1
    chats = [v for k, v in db.docs.items() if "/messages/" in k and v.get("kind") == "mithu_note"]
    assert len(chats) == 1 and chats[0]["noteId"] == note["id"]


def test_mithu_note_birthday_from_the_family_members(monkeypatch):
    c = make_client(today=date(2026, 10, 15), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 10, 15), members=[
        {"id": "k1", "name": "Aarav Sharma", "relation": "son", "birthDate": "2019-10-16"}])
    assert body["mithu"]["title"] == "Aarav's birthday is tomorrow"
    assert body["mithu"]["actions"][-1] == {"type": "open_family_member", "label": "See Aarav's day", "arg": "k1"}


@pytest.mark.parametrize("religion", ["Prefer not to say", "No religion / Spiritual"])
@pytest.mark.parametrize("day", [date(2026, 12, 25), date(2027, 3, 10)])
def test_an_unknown_faith_sees_the_day_but_mithu_never_wishes_it(religion, day, monkeypatch):
    prof = {**PROFILE, "religion": religion, "birthDate": "1994-08-21"}
    c = make_client(profile=prof, user={"astroMode": "fun"}, today=day, monkeypatch=monkeypatch)
    body = today_of(c, day)
    big = {"christmas", "eid_ul_fitr"} & set(ids(body["festivals"]))
    assert big                                            # the list shows it
    assert body["mithu"] is None                          # no "Merry Christmas" / "Eid Mubarak" from Mithu


def test_astro_off_gets_only_festival_and_birthday_notes(monkeypatch):
    c = make_client(user={"astroMode": "off"}, today=date(2026, 11, 8), monkeypatch=monkeypatch)
    note = today_of(c, date(2026, 11, 8))["mithu"]
    assert note["text"] == "Diwali is today. Shubh Deepavali to you and your family!"
    quiet = make_client(user={"astroMode": "off"}, today=date(2026, 10, 30), monkeypatch=monkeypatch)
    assert today_of(quiet, date(2026, 10, 30))["mithu"] is None   # no sky note for 'off'


def test_no_note_for_a_day_far_from_today(monkeypatch):
    db = Db()
    c = make_client(db=db, today=date(2026, 10, 1), monkeypatch=monkeypatch)
    body = today_of(c, date(2026, 11, 8))
    assert body["dayKey"] == "2026-11-08" and body["mithu"] is None
    assert not [k for k in db.docs if k.startswith("mithuNotes/")]


def test_a_failing_note_never_takes_today_down(monkeypatch):
    c = make_client(today=date(2026, 11, 8), monkeypatch=monkeypatch)
    monkeypatch.setattr(astro_api.astro_notes, "note_for", lambda *a, **k: 1 / 0)
    body = today_of(c, date(2026, 11, 8))
    assert body["mithu"] is None and "diwali" in ids(body["festivals"])


# ------------------------------------------------------------------ /forecast
def forecast_of(c, **body):
    return c.post("/astro/forecast", json=body, headers=H)


def test_forecast_shape(monkeypatch):
    c = make_client(today=date(2026, 10, 14), monkeypatch=monkeypatch)
    r = forecast_of(c, span="week", place=DELHI_IN)
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["ok"] is True and f["span"] == "week" and f["periodKey"] == "2026-10-12"
    assert f["title"] == "Your week · 12–18 Oct" and f["precision"] == "exact"
    assert [d["date"] for d in f["days"]] == [(date(2026, 10, 12) + timedelta(days=i)).isoformat() for i in range(7)]
    for d in f["days"]:
        assert {"date", "weekday", "tone", "house", "line", "tithi"} <= set(d)
        assert d["tone"] in ("good", "steady", "gentle") and "bad" not in d["line"].lower()
    assert set(f["best"]) == {"love", "work", "quiet"}
    events = {(e["date"], e.get("id")) for e in f["events"] if e["kind"] == "festival"}
    assert ("2026-10-18", "durga_puja") in events                   # a major day of her own faith, in the week
    assert all("2026-10-12" <= d <= "2026-10-18" for d, _ in events)
    assert f["text"] == GOOD_WEEK
    # Saturday: next week.
    sat = make_client(today=date(2026, 10, 17), monkeypatch=monkeypatch)
    nxt = forecast_of(sat, place=DELHI_IN).json()
    assert nxt["periodKey"] == "2026-10-19" and nxt["title"].startswith("Next week · ")


def test_forecast_events_are_the_readers_own_days(monkeypatch):
    hindu = make_client(today=date(2026, 11, 3), monkeypatch=monkeypatch)
    ev = {e["id"] for e in forecast_of(hindu, place=DELHI_IN).json()["events"] if e["kind"] == "festival"}
    assert "diwali" in ev                                           # 2–8 Nov
    unknown = make_client(profile={**PROFILE, "religion": "Prefer not to say"}, today=date(2026, 12, 22),
                          monkeypatch=monkeypatch)
    ev = {e["id"] for e in forecast_of(unknown, place=DELHI_IN).json()["events"] if e["kind"] == "festival"}
    assert "christmas" not in ev and "christmas_eve" not in ev


def test_forecast_astro_off_and_bad_input(monkeypatch):
    off = make_client(user={"astroMode": "off"}, today=date(2026, 10, 14), monkeypatch=monkeypatch)
    r = forecast_of(off, span="week")
    assert r.status_code == 200 and r.json() == {"ok": False, "error": "astro_off", "span": "week"}
    c = make_client(today=date(2026, 10, 14), monkeypatch=monkeypatch)
    assert forecast_of(c, span="month").status_code == 400
    assert forecast_of(c).status_code == 200                        # span defaults to week
    nobirth = make_client(profile={**PROFILE, "birthDate": ""}, today=date(2026, 10, 14), monkeypatch=monkeypatch)
    r = forecast_of(nobirth)
    assert r.status_code == 422 and r.json()["detail"] == "missing_birth_data"
    assert c.post("/astro/forecast", json={}).status_code == 401


def test_forecast_for_a_managed_profile_and_an_approximate_time(monkeypatch):
    calls = []

    def complete(msgs, *a, **k):
        calls.append(msgs[1]["content"])
        return GOOD_WEEK, "stop"

    prof = {**PROFILE, "profileFor": "daughter", "firstName": "Ananya Rao", "birthTime": "",
            "birthTimeApprox": "evening"}
    c = make_client(profile=prof, user={"birthTimeApprox": "evening"}, complete=complete, today=date(2026, 10, 14),
                    monkeypatch=monkeypatch)
    f = forecast_of(c, place=DELHI_IN).json()
    assert f["title"] == "Ananya's week · 12–18 Oct" and f["precision"] == "approximate"
    assert all("Ananya" in d["line"] for d in f["days"])
    assert "READER: family of Ananya" in calls[0] and "PRECISION: approximate" in calls[0]
    # The Moon is read at the approximate time (19:00), as the chart is.
    rashi, nak, precision = astro_api._natal_moon(prof)
    chart = ac.birth_chart((1994, 8, 21), None, main._resolve_birth_place(prof), approx="evening")
    assert (rashi, nak, precision) == (chart["moon"]["rashi"], chart["moon"]["nak"], "approximate")


def test_forecast_ask_first_faith_reads_for_fun(monkeypatch):
    calls = []
    c = make_client(profile={**PROFILE, "religion": "Sikh"}, today=date(2026, 10, 14), monkeypatch=monkeypatch,
                    complete=lambda msgs, *a: (calls.append(msgs[1]["content"]) or (GOOD_WEEK, "stop")))
    assert forecast_of(c).json()["ok"] is True
    assert "ASTRO_MODE: fun" in calls[0]


def test_forecast_one_ai_call_under_parallel_requests(monkeypatch):
    db, leases, calls = Db(), Leases(), []

    def slow(*a, **k):
        calls.append(1)
        time.sleep(0.3)
        return GOOD_WEEK, "stop"

    make_client(db=db, complete=slow, leases=leases, today=date(2026, 10, 14), monkeypatch=monkeypatch)
    monkeypatch.setattr(astro_forecast, "cached_week_text",
                        lambda *a, **k: _orig_cached(*a, **{**k, "poll_sec": 0.02, "wait_sec": 5.0}))
    req = astro_api.ForecastReq(span="week")
    with ThreadPoolExecutor(max_workers=5) as pool:
        out = list(pool.map(lambda _: astro_api.forecast(req, "Bearer ok"), range(5)))
    assert len(calls) == 1 and all(o["text"] == GOOD_WEEK for o in out)
    assert leases.taken == [f"forecast_{ME}_2026-10-12"]
    doc = db.docs[f"forecasts/{ME}__self__week__2026-10-12"]
    assert doc["text"] == GOOD_WEEK and doc["birthSig"] == astro_forecast.birth_sig("1994-08-21", "07:40", "Pune", None)


_orig_cached = astro_forecast.cached_week_text


def test_forecast_daily_rate_counts_only_new_text(monkeypatch):
    db, rates, calls = Db(), [], []

    def rate(uid, key, limit, per):
        rates.append((key, limit, per))
        return key != "forecast" or sum(1 for k, *_ in rates if k == "forecast") <= 1

    def bad(*a, **k):   # a text the validator refuses: no text is kept, so every call needs the AI again
        calls.append(1)
        return "Too short.", "stop"

    c = make_client(db=db, complete=bad, rate=rate, today=date(2026, 10, 14), monkeypatch=monkeypatch)
    first = forecast_of(c).json()
    assert first["ok"] and first["text"] is None and len(first["days"]) == 7   # the strip shows alone
    assert len(calls) == 2                                                     # the answer and one stricter retry
    second = forecast_of(c).json()
    assert second["ok"] and second["text"] is None and len(calls) == 2        # over today's cap: no AI call
    assert ("forecast", 6, "day") in rates
    # A kept text is served whatever the cap says.
    db.docs[f"forecasts/{ME}__self__week__2026-10-12"] = {
        "text": GOOD_WEEK, "birthSig": astro_forecast.birth_sig("1994-08-21", "07:40", "Pune", None)}
    assert forecast_of(c).json()["text"] == GOOD_WEEK and len(calls) == 2


def test_forecast_without_lease_wiring_still_works(monkeypatch):
    db = Db()
    c = make_client(db=db, leases=None, today=date(2026, 10, 14), monkeypatch=monkeypatch)
    assert astro_api._d().take_lease is None
    assert forecast_of(c).json()["text"] == GOOD_WEEK
    assert f"forecasts/{ME}__self__week__2026-10-12" in db.docs


def test_forecast_birth_change_regenerates(monkeypatch):
    path = f"forecasts/{ME}__self__week__2026-10-12"
    db = Db({path: {"text": "old text", "birthSig": "something else"}})
    keys = []

    def rate(uid, key, limit, per):
        keys.append((key, limit))
        return True

    c = make_client(db=db, rate=rate, leases=Leases(), today=date(2026, 10, 14), monkeypatch=monkeypatch)
    assert forecast_of(c).json()["text"] == GOOD_WEEK
    assert ("horoscope_regen", 2) in keys and ("forecast", 6) in keys
    assert db.docs[path]["birthSig"] == astro_forecast.birth_sig("1994-08-21", "07:40", "Pune", None)


# ------------------------------------------------------------------ /question: yesterday
def _seed_yesterday(db, day, counts, cities=None, my_vote=None, qid=None):
    yday = day - timedelta(days=1)
    q = astro_content.question_for(yday)
    db.docs[f"dailyQuestions/{yday.isoformat()}"] = {
        "qid": qid or q["id"], "counts": {str(i): n for i, n in enumerate(counts)}, "cities": cities or {}}
    if my_vote is not None:
        db.docs[f"dailyQuestionVotes/{yday.isoformat()}__{ME}"] = {"option": my_vote, "qid": q["id"]}
    return q


def test_yesterday_needs_twenty_votes_in_india(monkeypatch):
    day = date(2026, 10, 12)
    db = Db()
    q = _seed_yesterday(db, day, [10, 9], my_vote=1)
    c = make_client(db=db, today=day, monkeypatch=monkeypatch)
    assert c.post("/astro/question", json={}, headers=H).json()["yesterday"] is None      # 19 votes
    q = _seed_yesterday(db, day, [12, 8] + [0] * (len(q["options"]) - 2), my_vote=1)
    y = c.post("/astro/question", json={}, headers=H).json()["yesterday"]
    assert y["total"] == 20 and y["counts"][:2] == [12, 8] and len(y["counts"]) == len(q["options"])
    assert y["text"] == q["text"] and y["options"] == q["options"] and y["voted"] == 1
    assert y["day"] == "2026-10-11" and y["city"] is None            # no votes from the reader's city


def test_yesterday_city_view_needs_fifteen(monkeypatch):
    day = date(2026, 10, 12)
    db = Db()
    n = len(astro_content.question_for(day - timedelta(days=1))["options"])
    counts = [20] + [5] * (n - 1)
    _seed_yesterday(db, day, counts, cities={"bengaluru": {"0": 10, "1": 4}})
    c = make_client(db=db, today=day, monkeypatch=monkeypatch)
    place = {"lat": 12.97, "lon": 77.59}
    y = c.post("/astro/question", json={"place": place}, headers=H).json()["yesterday"]
    assert y["total"] == sum(counts) and y["city"] is None and y["voted"] is None   # 14 in the city
    _seed_yesterday(db, day, counts, cities={"bengaluru": {"0": 10, "1": 5}})
    y = c.post("/astro/question", json={"place": place}, headers=H).json()["yesterday"]
    assert y["city"]["total"] == 15 and y["city"]["name"] == "Bengaluru" and y["city"]["counts"][:2] == [10, 5]


def test_yesterday_for_another_question_is_not_shown(monkeypatch):
    day = date(2026, 10, 12)
    db = Db()
    _seed_yesterday(db, day, [30, 30], qid="some_other_question")
    c = make_client(db=db, today=day, monkeypatch=monkeypatch)
    assert c.post("/astro/question", json={}, headers=H).json()["yesterday"] is None
    nodb = make_client(db=None, today=day, monkeypatch=monkeypatch)
    assert nodb.post("/astro/question", json={}, headers=H).json()["yesterday"] is None
