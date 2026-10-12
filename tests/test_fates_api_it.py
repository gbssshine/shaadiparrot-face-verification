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


def own_photo(uid: str, name: str = "p1.jpg") -> str:
    """An own Storage photo link (face_live.own_photo_url): the only kind the face check verifies and Daily Fates
    checks (fates_engine.own_private_photos), on a public card as on a private profile."""
    return f"https://firebasestorage.googleapis.com/v0/b/{PROJECT}.appspot.com/o/users%2F{uid}%2Fphotos%2F{name}?alt=media"


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
    fates_api.blur_thumbs = lambda url, uid: ("data:image/jpeg;base64,QkxVUg==", "data:image/jpeg;base64,UEVFSw==")

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
                "photos": [own_photo(uid)], "relationshipIntent": "Serious long-term partner",
                "religion": "Hindu", "community": "Kannada", "languages": ["Hindi", "English"],
                "interests": ["Trekking", "Coffee"], "smoking": "Never", "drinking": "Socially", "bio": f"I am {uid}."}
        base.update(pub)
        db.document(f"publicProfiles/{uid}").set(base)
        db.document(f"profiles/{uid}").set({"birthDate": "1999-05-02", "birthTime": "10:15", "birthCityName": "Bengaluru",
                                             "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "faceVerified": True,
                                             "faceVerifiedPhotos": [own_photo(uid)]})
        db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})

    db.document(f"publicProfiles/{viewer}").set({
        "uid": viewer, "firstName": "Arjun", "age": 28, "gender": "Male", "lookingForGender": "Female", "cityName": "Bengaluru",
        "lat": 12.96, "lon": 77.64, "isDiscoverable": True, "photos": [own_photo(viewer)],
        "relationshipIntent": "Serious long-term partner", "religion": "Hindu", "community": "Kannada",
        "languages": ["Hindi", "English"], "interests": ["Trekking", "Cricket"], "smoking": "Never", "drinking": "Socially"})
    db.document(f"profiles/{viewer}").set({"birthDate": "1996-11-20", "birthTime": "06:30", "birthCityName": "Bengaluru",
                                            "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "crownsBalance": 1, "faceVerified": True,
                                            "faceVerifiedPhotos": [own_photo(viewer)]})
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
    for hidden in ("friendsonly", "sleepy", "\"man\"", "riya", "firebasestorage", "Anaya", "Meera", "Tanisha"):
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


# ======================================================================================================================
# 1.0.8: rishta mode ({mode: 'rishta'}) and the pools (plan §2 C1). Seeded after the dating tests above (their day is
# already made), with their own uids.
# ======================================================================================================================
RISHTA_INDEX = {"feedRishta": True, "feedRishtaFromDay": "2017-01-01", "feedQuick": False, "feedQuickFromDay": "2017-01-01"}
CONSENT = {"marriageIntent": True, "intentConsentAtIso": "2026-10-01T00:00:00+00:00", "intentConsentVersion": "intent-2026-10"}
RISHTA_ONLY = {"priya", "kavya", "lowguna", "jainonly", "notyet"}
# Women a dating viewer without a stated intent may get: the 1.0.7 cards above ("friendsonly" clashes only with a
# "serious" viewer) and the ones in the quick pool.
QUICK_POOL = {"anaya", "meera", "tanisha", "riya", "friendsonly", "quickonly", "bothok"}


@pytest.fixture(scope="module")
def rishta_env(env):
    db = env["db"]
    active = (NOW - timedelta(hours=2)).isoformat()

    def seed(uid, birth, pub=None, priv=None):
        photo = own_photo(uid)
        card = {"uid": uid, "firstName": uid.title(), "age": 27, "gender": "Female", "lookingForGender": "Male",
                "cityName": "Bengaluru", "lat": 12.97, "lon": 77.59, "isDiscoverable": True, "isFaceVerified": True,
                "photos": [photo], "religion": "Hindu", "community": "Kannada", "maritalStatus": "Never married",
                "diet": "Vegetarian", "familyValues": "Moderate", "education": "Master's",
                "languages": ["Kannada", "English"], "interests": ["Music", "Travel"], "bio": f"I am {uid}."}
        card.update(pub or {})
        db.document(f"publicProfiles/{uid}").set(card)
        db.document(f"profiles/{uid}").set({"birthDate": birth, "birthTime": "10:15", "birthCityName": "Bengaluru",
                                             "birthCountryIso2": "IN", "faceVerified": True, "faceVerifiedPhotos": [photo],
                                             "gender": "female", **(priv or {})})
        db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})

    # Candidates; the gunas are with the viewers' birth (1996-11-20 06:30, Bengaluru).
    seed("priya", "1999-08-02", RISHTA_INDEX, {**CONSENT, "profileFor": "self"})                      # 33 gunas
    seed("kavya", "1999-10-23", {**RISHTA_INDEX, "profileFor": "daughter"},                            # 33, made by a parent
         {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00+00:00", "profileFor": "daughter",
          "creatorRelation": "Mother", "lifeStage": "MarriedKids"})
    seed("bothok", "1999-06-02", {**RISHTA_INDEX, "feedQuick": True},                                  # 31.5, both pools
         {**CONSENT, "profileFor": "self", "quickSwipeOk": True})
    seed("lowguna", "1999-03-16", RISHTA_INDEX, {**CONSENT, "profileFor": "self"})                    # 8.5 gunas
    seed("jainonly", "1999-04-09", RISHTA_INDEX,                                                       # her family's must-have
         {**CONSENT, "profileFor": "self", "pref_religions": ["Jain"], "pref_strict": ["religions"]})
    seed("quickonly", "1999-01-16")                                                                    # a 1.0.7 card
    seed("notyet", "1999-08-23", {**RISHTA_INDEX, "feedRishtaFromDay": "2099-01-01"}, {**CONSENT, "profileFor": "self"})
    seed("hidden", "1999-01-23", {**RISHTA_INDEX, "photoPrivacy": "accepted",
                                   "photos": ["https://example.com/users/hidden/blur/a.jpg"]}, {**CONSENT, "profileFor": "self"})

    viewers = {}

    def viewer(name, priv):
        token, uid = _token(name)
        photo = own_photo(uid)
        db.document(f"publicProfiles/{uid}").set({
            "uid": uid, "firstName": name.title(), "age": 29, "gender": "Male", "lookingForGender": "Female",
            "cityName": "Bengaluru", "lat": 12.96, "lon": 77.64, "isDiscoverable": True, "isFaceVerified": True,
            "photos": [photo], "religion": "Hindu", "community": "Kannada", "maritalStatus": "Never married",
            "diet": "Vegetarian", "familyValues": "Traditional", "education": "Bachelor's",
            "languages": ["Kannada", "English"], "interests": ["Music", "Cricket"]})
        db.document(f"profiles/{uid}").set({"birthDate": "1996-11-20", "birthTime": "06:30", "birthCityName": "Bengaluru",
                                             "birthCountryIso2": "IN", "gender": "male", "faceVerified": True,
                                             "faceVerifiedPhotos": [photo], **priv})
        db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})
        viewers[name] = {"uid": uid, "h": {"Authorization": f"Bearer {token}"}}

    viewer("rohan", {**CONSENT, "profileFor": "self"})                          # wants marriage: rishta mode
    viewer("vikram", {**CONSENT, "profileFor": "self"})                         # the same, asks in both modes
    viewer("arun", {**CONSENT, "profileFor": "self"})                           # flips between the modes all day
    viewer("newbie", {**CONSENT, "profileFor": "self", "faceVerified": False})  # not verified yet, flips too
    viewer("dev", {"profileFor": "self"})                                        # a dating account
    viewer("sam", {"profileFor": "self"})                                        # asks for rishta without intent
    viewer("kid", {**CONSENT, "profileFor": "self", "birthDate": "2006-01-01"})  # a groom of 20
    viewer("mom", {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00+00:00", "profileFor": "son"})
    yield {**env, "v": viewers}


def _targets(env, uid, day):
    doc = env["db"].document(f"dailyFates/{uid}__{day}").get().to_dict() or {}
    return doc, {p["targetUid"] for p in (doc.get("paths") or {}).values()}


def test_rishta_mode_brings_only_rishta_people_who_fit_both_ways(rishta_env):
    c, v = rishta_env["client"], rishta_env["v"]["rohan"]
    r = c.post("/fates/today", json={"mode": "rishta"}, headers=v["h"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "rishta" and body["paths"] and "reason" not in body
    doc, targets = _targets(rishta_env, v["uid"], body["dayKey"])
    assert doc["mode"] == "rishta"
    assert targets and targets <= {"priya", "kavya", "bothok"}
    assert all((p["teaser"].get("gunas") or 0) >= 18 for p in body["paths"])    # 18+ gunas on every path
    again = c.post("/fates/today", json={"mode": "rishta"}, headers=v["h"]).json()
    assert [p["path"] for p in again["paths"]] == [p["path"] for p in body["paths"]]


def test_a_rishta_report_compares_what_families_compare(rishta_env):
    c, v = rishta_env["client"], rishta_env["v"]["rohan"]
    today = c.post("/fates/today", json={"mode": "rishta"}, headers=v["h"]).json()
    path = today["paths"][0]["path"]
    r = c.post("/fates/open", json={"path": path}, headers=v["h"])
    assert r.status_code == 200, r.text
    rep = next(p for p in r.json()["paths"] if p["path"] == path)["report"]
    rows = {row["key"]: row["status"] for row in rep["home"]["rows"]}
    assert {"marital", "food", "family_values", "education"} <= set(rows)
    assert rows["marital"] == "aligns" and rows["food"] == "aligns" and rows["education"] == "aligns"
    assert "lifeStage" not in str(rep) and "pref_" not in str(rep)


def test_dating_fates_never_bring_a_family_made_or_marriage_only_profile(rishta_env):
    c, v = rishta_env["client"], rishta_env["v"]["dev"]
    for kwargs in ({}, {"json": {}}):                                            # no body, and the {} that 1.0.7 sends
        r = c.post("/fates/today", headers=v["h"], **kwargs)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "quick" and "reason" not in body and body["paths"]
        _, targets = _targets(rishta_env, v["uid"], body["dayKey"])
        assert not targets & (RISHTA_ONLY | {"hidden"})
        assert targets <= QUICK_POOL


def test_rishta_mode_viewer_rules(rishta_env):
    import fates_api
    c, vs, db = rishta_env["client"], rishta_env["v"], rishta_env["db"]
    day = fates_api.day_key()
    # Plan §2 C1: rishta-mode Fates refuse like getRishtas and swipeAction, a 403 with the reason as the detail.
    for name, reason in (("sam", "needs_intent"), ("kid", "groom_age")):
        r = c.post("/fates/today", json={"mode": "rishta"}, headers=vs[name]["h"])
        assert r.status_code == 403, r.text
        assert r.json() == {"detail": reason}
        assert not db.document(f"dailyFates/{vs[name]['uid']}__{day}").get().exists        # nothing stored
    # Dating mode for a family-made profile: an empty day (1.0.7 reads any other 403 as a server error).
    r = c.post("/fates/today", json={}, headers=vs["mom"]["h"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["paths"] == [] and body["reason"] == "no_quick" and body["opensLeft"] == 0
    assert not db.document(f"dailyFates/{vs['mom']['uid']}__{body['dayKey']}").get().exists


def test_a_mode_switch_never_shows_the_other_modes_unopened_people(rishta_env):
    c, v = rishta_env["client"], rishta_env["v"]["vikram"]
    dating = c.post("/fates/today", json={}, headers=v["h"]).json()                 # an old app first
    assert dating["mode"] == "quick" and dating["paths"]
    _, first = _targets(rishta_env, v["uid"], dating["dayKey"])
    assert first <= QUICK_POOL
    rishta = c.post("/fates/today", json={"mode": "rishta"}, headers=v["h"]).json()  # then 1.0.8, nothing opened yet
    assert rishta["mode"] == "rishta"
    doc, targets = _targets(rishta_env, v["uid"], rishta["dayKey"])
    assert doc["mode"] == "rishta" and targets <= {"priya", "kavya", "bothok"}


def test_alternating_modes_remake_the_day_at_most_once(rishta_env, monkeypatch):
    import fates_api
    c, v, db = rishta_env["client"], rishta_env["v"]["arun"], rishta_env["db"]
    made, real = [], fates_api.generate

    def counting(db_, viewer, memory, key, now_ms):
        made.append("rishta" if viewer.rishta else "quick")
        return real(db_, viewer, memory, key, now_ms)

    monkeypatch.setattr(fates_api, "generate", counting)
    views = [c.post("/fates/today", json={"mode": "rishta"} if i % 2 == 0 else {}, headers=v["h"]).json() for i in range(5)]
    assert made == ["rishta", "quick"]                                # the day, then one remake for the switch
    assert [x["mode"] for x in views] == ["rishta", "quick", "rishta", "quick", "rishta"]
    assert all(x["paths"] == [] for x in views[2:])                   # after that nobody unopened from the other mode
    doc, targets = _targets(rishta_env, v["uid"], views[0]["dayKey"])
    assert doc["mode"] == "rishta" and not targets
    mem = db.document(f"fatesMemory/{v['uid']}").get().to_dict()
    assert mem["modeSwitch"] == {"day": views[0]["dayKey"], "n": 1}


def test_the_verify_teaser_counts_each_mode_once_a_day(rishta_env, monkeypatch):
    import fates_api
    c, v, db = rishta_env["client"], rishta_env["v"]["newbie"], rishta_env["db"]
    day = fates_api.day_key()
    mem_ref = db.document(f"fatesMemory/{v['uid']}")
    mem_ref.set({"verifyTeaser": {"day": "2000-01-01", "n": {"quick": 99, "rishta": 99}}})    # another day's counts
    counted, real = [], fates_api._candidates

    def counting(db_, viewer, now_ms, memory):
        counted.append("rishta" if viewer.rishta else "quick")
        return real(db_, viewer, now_ms, memory)

    monkeypatch.setattr(fates_api, "_candidates", counting)
    views = [c.post("/fates/today", json={"mode": "rishta"} if i % 2 == 0 else {}, headers=v["h"]).json() for i in range(5)]
    assert counted == ["rishta", "quick"]
    assert all(x["needsVerification"] and x["paths"] == [] for x in views)
    rishta_n, quick_n = views[0]["verifiedNearby"], views[1]["verifiedNearby"]
    assert [x["verifiedNearby"] for x in views] == [rishta_n, quick_n, rishta_n, quick_n, rishta_n] and rishta_n != 99
    assert mem_ref.get().to_dict()["verifyTeaser"] == {"day": day, "n": {"rishta": rishta_n, "quick": quick_n}}
    assert not db.document(f"dailyFates/{v['uid']}__{day}").get().exists
