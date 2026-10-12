"""Rate caps under contention: when the counter can't be written, paid or AI work is refused, never let through.

Parallel requests from one account contend on serverRate/{uid}; Firestore's transactional() then retries and gives
up (ValueError). The old counter answered "yes" then (fail open), so a burst of parallel requests skipped every cap.
Now:
- the AI calls (chat, horoscope rewrite, week forecast), the biodata PDF, /astro/match/report and the face checks
  refuse: 503 busy_try_again (the chat answers with its "send that again in a moment" reply, a rewrite or the week
  text is skipped and the kept text served);
- the open counter (photo checks, kundli scores) stays open only for a Firestore outage, its documented reason, and
  refuses under contention too;
- a lease under contention is someone else's (the parallel request waits for that one's reading).

The fake Firestore's transactions never commit (every commit is aborted), so google.cloud.firestore's own
transactional() runs its retries and gives up exactly as it does in production.
"""
import dataclasses
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.api_core import exceptions as gexc

import astro_api
import biodata_route
import main

ME = "busy123456"
H = {"Authorization": "Bearer ok"}
PROFILE = {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21", "birthTime": "07:40",
           "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN",
           "cityName": "Bengaluru", "stateName": "Karnataka", "countryIso2": "IN", "lat": 12.9719, "lon": 77.5937,
           "religion": "Hindu", "community": "Marathi", "lifeStage": "single", "profileFor": "self"}
PUBLIC = {"firstName": "Asha", "gender": "Female", "cityName": "Bengaluru", "isDiscoverable": True}
VIKRAM = {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14", "birthTime": "09:10", "birthCity": "Chennai",
          "birthState": "Tamil Nadu"}
WEEK = ("Theme: A steady week.\nMonday is good for family talks and Tuesday for work. Midweek is calm, and the "
        "weekend brings time with people you love.")


# ------------------------------------------------------------------ a Firestore whose transactions always give up
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
        return Snap(self, self._db.docs.get(self.path))

    def set(self, data, merge=False):
        self._db.docs[self.path] = {**(self._db.docs.get(self.path) or {}), **data} if merge else dict(data)


class Col:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")


class GiveUpTx:
    """Every commit is aborted (another request holds the docs): transactional() retries max_attempts times, then
    raises ValueError, as Firestore does under contention."""
    _read_only = False
    _max_attempts = 5

    def __init__(self, db):
        self._db, self._id = db, None

    def _clean_up(self):
        self._id = None

    def _begin(self, retry_id=None):
        self._id = b"tx"

    def _commit(self):
        self._db.commits += 1
        raise gexc.Aborted("too much contention on these documents")

    def _rollback(self):
        self._id = None

    def set(self, ref, data, merge=False):
        pass

    def update(self, ref, data):
        pass


class GiveUpDb:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.commits = 0

    def collection(self, name):
        return Col(self, name)

    def document(self, path):
        return Ref(self, path)

    def transaction(self):
        return GiveUpTx(self)


@pytest.fixture()
def busy(monkeypatch):
    db = GiveUpDb({f"profiles/{ME}": dict(PROFILE), f"publicProfiles/{ME}": dict(PUBLIC), f"users/{ME}": {}})
    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_verify_firebase_token_or_401", lambda a: ME)
    calls = []

    def complete(messages, max_tokens, temperature=None, kind="chat", timeout=None):
        calls.append(kind)
        return (WEEK if kind == "astro_week" else "Theme: A calm day.\n✅ Do\n• Rest.\n• Call home.\n⛔ Avoid\n• Rush.\n• Fights."), "stop"

    monkeypatch.setattr(main, "_deepseek_complete", complete)
    db.ai_calls = calls
    return db


@pytest.fixture()
def astro(busy, monkeypatch):
    """/astro routes on main's own Deps (its real rate counters), with this test's docs and model."""
    saved = astro_api._deps
    astro_api._hits.clear()
    astro_api._report_hits.clear()
    deps = dataclasses.replace(
        main._astro_deps(), db=lambda: busy,
        verify_uid=lambda h: ME if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        load_user_docs=lambda uid: (dict(PROFILE), dict(PUBLIC), {}) if uid == ME else ({}, {}, {}),
        is_premium=lambda u, p, now: True,
        complete=lambda msgs, max_tokens, temperature, kind: main._deepseek_complete(msgs, max_tokens, temperature, kind))
    astro_api.configure(deps)
    app = FastAPI()
    app.include_router(astro_api.router)
    app.include_router(biodata_route.router)
    yield TestClient(app)
    if saved is not None:
        astro_api.configure(saved)


# ------------------------------------------------------------------ the counter itself
def test_the_counter_is_busy_when_the_transaction_gives_up(busy):
    assert main._rate_state(ME, "astroReport", 12) == "busy"
    assert busy.commits == GiveUpTx._max_attempts                  # Firestore's own retries ran, then it gave up
    assert main._take_rate_strict(ME, "astroReport", 12) is False
    with pytest.raises(main.HTTPException) as e:
        main._take_rate_or_busy(ME, "astroReport", 12)
    assert e.value.status_code == 503 and e.value.detail == "busy_try_again"
    # The open counter (photo checks, scores) stays open only for an outage, never under contention.
    assert main._take_rate(ME, "photoChecks", 60) is False


def test_the_open_counter_stays_open_for_an_outage_only(monkeypatch):
    class Down:
        def collection(self, name):
            raise RuntimeError("firestore is down")

    monkeypatch.setattr(main, "firestore_client", Down())
    assert main._rate_state(ME, "photoChecks", 60) == "down"
    assert main._take_rate(ME, "photoChecks", 60) is True             # documented: a hiccup must not stop uploads
    assert main._take_rate_strict(ME, "aiChatMinute", 8) is False
    with pytest.raises(main.HTTPException) as e:
        main._take_rate_or_busy(ME, "biodataPdf", 30)
    assert e.value.status_code == 503


# ------------------------------------------------------------------ the AI calls
def test_chat_is_refused_not_answered(busy, monkeypatch):
    monkeypatch.setattr(main, "AI_CHAT_QUOTA_ENFORCED", False)       # the per-minute cap is then the only one
    r = TestClient(main.app).post("/ai-chat", json={"text": "What does my kundli say about marriage this year?",
                                                    "thread_id": f"mobile_{ME}"}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["blocked"] is True and body["reason"] == "busy" and busy.ai_calls == []


def test_the_horoscope_rewrite_is_skipped(busy, monkeypatch):
    monkeypatch.setattr(main, "_take_lease", lambda name, ttl: True)
    monkeypatch.setattr(main, "_release_lease", lambda name: None)
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz)
    cached = {"text": "Yesterday's birth details", "birthSig": "old", "regenCount": 0}
    out = main._regenerate_daily_horoscope(ME, f"mobile_{ME}", now.date().isoformat(), tz, now, dict(PROFILE),
                                           cached, main._birth_sig(PROFILE))
    assert out is None and busy.ai_calls == []                      # the cached reading is served


def test_parallel_first_readings_wait_for_the_one_holding_the_lease(busy):
    assert main._take_lease(f"horoscope_{ME}_2026-10-11", 45) is False


def test_the_week_text_is_skipped_and_the_kept_one_served(astro, busy):
    # The lease is granted here (a busy lease only makes the request wait for another one's text): the counter is
    # what is under test.
    astro_api.configure(dataclasses.replace(astro_api._d(), take_lease=lambda name, ttl: True,
                                            release_lease=lambda name: None))
    r = astro.post("/astro/forecast", json={"span": "week"}, headers=H)
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["ok"] is True and f["text"] is None and len(f["days"]) == 7 and busy.ai_calls == []
    # New birth details: the rewrite (horoscope_regen counter) is refused too, the kept text stays.
    busy.docs[f"forecasts/{ME}__self__week__{f['periodKey']}"] = {"text": WEEK, "birthSig": "old"}
    f = astro.post("/astro/forecast", json={"span": "week"}, headers=H).json()
    assert f["text"] == WEEK and busy.ai_calls == []


# ------------------------------------------------------------------ the paid and heavy routes
def test_the_match_report_answers_busy(astro, busy):
    r = astro.post("/astro/match/report", json={"b": VIKRAM}, headers=H)
    assert r.status_code == 503 and r.json()["detail"] == "busy_try_again"
    assert busy.ai_calls == []


def test_the_biodata_pdf_answers_busy(astro):
    r = astro.post("/astro/biodata", json={"photo": False}, headers=H)
    assert r.status_code == 503 and r.json()["detail"] == "busy_try_again"


def test_face_check_attempts_answer_busy(busy):
    for take in (main._take_face_attempt, lambda uid: main._take_counter(uid, "faceRecheckAttempts", 20)):
        with pytest.raises(main.HTTPException) as e:
            take(ME)
        assert e.value.status_code == 503 and e.value.detail == "busy_try_again"


def test_photo_checks_answer_busy_under_contention(busy, monkeypatch):
    monkeypatch.setattr(main, "vision_client", object())             # Vision would be called next
    r = TestClient(main.app).post("/verify-photo", json={
        "gcs_uri": f"gs://shaadiparrot.firebasestorage.app/users/{ME}/photos/pending/a.jpg"}, headers=H)
    assert r.status_code == 503 and r.json()["detail"] == "busy_try_again"
