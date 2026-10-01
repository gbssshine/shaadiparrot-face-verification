"""End-to-end Daily Fates API test against the Firestore + Auth emulators.

Run:  firebase emulators:exec --only auth,firestore --project demo-fates "python -m pytest tests/test_fates_api_it.py"
Skipped unless FIRESTORE_EMULATOR_HOST and FIREBASE_AUTH_EMULATOR_HOST are set.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not (os.getenv("FIRESTORE_EMULATOR_HOST") and os.getenv("FIREBASE_AUTH_EMULATOR_HOST")),
    reason="needs the Firestore and Auth emulators")

PROJECT = os.getenv("GCLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "demo-fates"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT)
NOW = datetime.now(timezone.utc)


def _token(uid: str) -> str:
    host = os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signUp?key=fake",
                      json={"email": f"{uid}@test.dev", "password": "secret1", "returnSecureToken": True})
    if r.status_code != 200:
        r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=fake",
                          json={"email": f"{uid}@test.dev", "password": "secret1", "returnSecureToken": True})
    body = r.json()
    return body["idToken"], body["localId"]


@pytest.fixture(scope="module")
def env():
    import main  # noqa: F401  (configures fates_api)
    import fates_api
    from fastapi.testclient import TestClient
    from google.cloud import firestore

    requests.delete(f"http://{os.environ['FIRESTORE_EMULATOR_HOST']}/emulator/v1/projects/{PROJECT}/databases/(default)/documents")
    fates_api.blur_thumbs = lambda url: ("data:image/jpeg;base64,QkxVUg==", "data:image/jpeg;base64,UEVFSw==")

    calls = []

    def fake_complete(messages, max_tokens, temperature, kind):
        calls.append(kind)
        payload = json.loads(messages[-1]["content"])
        if kind == "fates_pick":
            paths = [{"path": p, "pick": len(opts) - 1, "hook": "You both want the same kind of home.", "why": "A calm, kind match for you."}
                     for p, opts in payload["paths"].items()]
            return json.dumps({"paths": paths}), "stop"
        name = payload.get("their_first_name")
        return json.dumps({
            "verdict": f"{name} is a steady match for you. Start with what you both love.",
            "strengths": ["Same plans", "Calm together", "Shared interests"],
            "talk_about": [], "openers": ["What made you smile today?", "Coffee or chai?", "Favourite weekend plan?"],
            "date_idea": "A morning walk in Cubbon Park, then filter coffee.",
        }), "stop"

    fates_api._d().complete = fake_complete
    client = TestClient(main.app)
    client.__enter__()
    db = firestore.Client(project=PROJECT)

    viewer_token, viewer = _token("arjun")
    active = (NOW - timedelta(hours=3)).isoformat()

    def person(uid, **pub):
        base = {"uid": uid, "firstName": uid.title(), "age": 26, "gender": "Female", "lookingForGender": "Male",
                "cityName": "Bengaluru", "lat": 12.97, "lon": 77.59, "isDiscoverable": True, "isFaceVerified": True,
                "photos": [f"https://example.com/{uid}.jpg"], "relationshipIntent": "Serious long-term partner",
                "religion": "Hindu", "community": "Kannada", "languages": ["Hindi", "English"],
                "interests": ["Trekking", "Coffee"], "smoking": "Never", "drinking": "Socially", "bio": f"I am {uid}."}
        base.update(pub)
        db.document(f"publicProfiles/{uid}").set(base)
        db.document(f"profiles/{uid}").set({"birthDate": "1999-05-02", "birthTime": "10:15", "birthCityName": "Bengaluru",
                                             "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "faceVerified": True,
                                             "faceVerifiedPhotos": [f"https://example.com/{uid}.jpg"]})
        db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})

    db.document(f"publicProfiles/{viewer}").set({
        "uid": viewer, "firstName": "Arjun", "age": 28, "gender": "Male", "lookingForGender": "Female", "cityName": "Bengaluru",
        "lat": 12.96, "lon": 77.64, "isDiscoverable": True, "photos": ["https://example.com/arjun.jpg"],
        "relationshipIntent": "Serious long-term partner", "religion": "Hindu", "community": "Kannada",
        "languages": ["Hindi", "English"], "interests": ["Trekking", "Cricket"], "smoking": "Never", "drinking": "Socially"})
    db.document(f"profiles/{viewer}").set({"birthDate": "1996-11-20", "birthTime": "06:30", "birthCityName": "Bengaluru",
                                            "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "crownsBalance": 1, "faceVerified": True,
                                            "faceVerifiedPhotos": ["https://example.com/arjun.jpg"]})
    db.document(f"users/{viewer}").set({"lastAppOpenAtUtcIso": active})

    for u in ("anaya", "meera", "tanisha", "riya"):
        person(u)
    person("friendsonly", relationshipIntent="Friends / social circle")      # hard filter: intent
    person("sleepy")
    db.document("users/sleepy").set({"lastAppOpenAtUtcIso": (NOW - timedelta(days=40)).isoformat()})  # inactive
    person("man", gender="Male", lookingForGender="Female")                    # wrong gender
    db.document(f"users/{viewer}/blocks/riya").set({"x": 1})                   # blocked by the viewer

    yield {"client": client, "db": db, "uid": viewer, "h": {"Authorization": f"Bearer {viewer_token}"}, "calls": calls}
    client.__exit__(None, None, None)


def test_today_generates_three_distinct_anonymous_paths(env):
    r = env["client"].post("/fates/today", headers=env["h"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and len(body["paths"]) >= 1
    txt = json.dumps(body)
    for hidden in ("friendsonly", "sleepy", "\"man\"", "riya", "example.com", "Anaya", "Meera", "Tanisha"):
        assert hidden not in txt, hidden                      # closed paths stay anonymous
    assert all(p["blurThumb"] and not p["opened"] for p in body["paths"])
    assert body["opensLeft"] == 1
    assert "fates_pick" in env["calls"]
    doc = env["db"].document(f"dailyFates/{env['uid']}__{body['dayKey']}").get().to_dict()
    targets = [p["targetUid"] for p in doc["paths"].values()]
    assert len(targets) == len(set(targets))
    assert set(targets) <= {"anaya", "meera", "tanisha"}
    # Same answer on a second call (cached for the day), and no second pick call.
    again = env["client"].post("/fates/today", headers=env["h"]).json()
    assert [p["path"] for p in again["paths"]] == [p["path"] for p in body["paths"]]
    assert env["calls"].count("fates_pick") == 1


def test_open_reveals_report_then_verdict(env):
    today = env["client"].post("/fates/today", headers=env["h"]).json()
    path = today["paths"][0]["path"]
    r = env["client"].post("/fates/open", json={"path": path}, headers=env["h"])
    assert r.status_code == 200, r.text
    opened = next(p for p in r.json()["paths"] if p["path"] == path)
    rep = opened["report"]
    assert opened["opened"] and rep["person"]["firstName"] in ("Anaya", "Meera", "Tanisha")
    assert rep["stars"] and 0 <= rep["stars"]["total"] <= 36 and len(rep["stars"]["kootas"]) == 8
    assert rep["person"]["pronoun"]["subj"] == "she"
    assert r.json()["opensLeft"] == 0
    v = env["client"].post("/fates/verdict", json={"path": path}, headers=env["h"]).json()
    assert v["ok"] and v["verdict"]["by"] == "ai" and len(v["verdict"]["openers"]) == 3
    mem = env["db"].document(f"fatesMemory/{env['uid']}").get().to_dict()
    assert mem["seen"][rep["person"]["uid"]]["opened"] is True
    assert mem["streak"] == 1 and mem["learned"]["pathOpens"][path] == 1


def test_second_open_needs_an_unlock(env):
    today = env["client"].post("/fates/today", headers=env["h"]).json()
    closed = [p["path"] for p in today["paths"] if not p["opened"]]
    if not closed:
        pytest.skip("only one path today")
    r = env["client"].post("/fates/open", json={"path": closed[0]}, headers=env["h"])
    assert r.status_code == 402
    assert env["client"].post("/fates/unlock", json={"method": "ad"}, headers=env["h"]).status_code == 200
    assert env["client"].post("/fates/unlock", json={"method": "ad"}, headers=env["h"]).status_code == 429   # once a day
    ok = env["client"].post("/fates/open", json={"path": closed[0]}, headers=env["h"])
    assert ok.status_code == 200, ok.text
    if len(closed) > 1:
        assert env["client"].post("/fates/unlock", json={"method": "crown"}, headers=env["h"]).status_code == 200
        prof = env["db"].document(f"profiles/{env['uid']}").get().to_dict()
        assert prof["crownsBalance"] == 0
        assert env["client"].post("/fates/unlock", json={"method": "crown"}, headers=env["h"]).status_code == 402


def test_decision_learning_and_journal(env):
    today = env["client"].post("/fates/today", headers=env["h"]).json()
    path = next(p["path"] for p in today["paths"] if p["opened"])
    r = env["client"].post("/fates/decision", json={"path": path, "decision": "skipped", "reason": "too_far"}, headers=env["h"])
    assert r.status_code == 200
    mem = env["db"].document(f"fatesMemory/{env['uid']}").get().to_dict()
    assert mem["learned"]["skipReasons"]["too_far"] == 1 and mem["learned"]["farPenaltyPerKm"] > 0
    j = env["client"].post("/fates/journal", headers=env["h"]).json()
    assert j["ok"] and any(i.get("person") for i in j["items"])
    closed = [i for i in j["items"] if not i["opened"]]
    assert all("person" not in i for i in closed)


def test_requires_auth(env):
    assert env["client"].post("/fates/today").status_code == 401
    assert env["client"].post("/fates/today", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_stats_are_counted(env):
    today = env["client"].post("/fates/today", headers=env["h"]).json()
    stats: dict = {}
    for shard in env["db"].collection(f"fatesStats/{today['dayKey']}/shards").stream():
        for k, v in (shard.to_dict() or {}).items():
            stats[k] = stats.get(k, 0) + v
    assert stats["generated"] == 1 and stats["opens"] >= 1 and stats["aiPickOk"] == 1



def test_a_removed_verification_stops_opening(env):
    c, db, h, uid = env["client"], env["db"], env["h"], env["uid"]
    db.document(f"profiles/{uid}").set({"faceVerified": False, "isFaceVerified": False, "faceVerifiedReason": "review_rejected"}, merge=True)
    try:
        r = c.post("/fates/open", headers=h, json={"path": "stars"})
        assert r.status_code == 403 and r.json()["detail"] == "not_verified"
        today = c.post("/fates/today", headers=h).json()
        assert today["needsVerification"] is True and today["verifyReason"] == "review_rejected"
    finally:
        db.document(f"profiles/{uid}").set({"faceVerified": True, "isFaceVerified": True, "faceVerifiedReason": "ok"}, merge=True)
