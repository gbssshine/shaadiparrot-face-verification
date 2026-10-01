"""People who chose the viewer ("Chose you"): brought as a path, listed blurred, revealed for a crown.

Run with the Firestore + Auth emulators (see test_fates_api_it.py). Skipped without them.
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


def _token(email_user: str):
    host = os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    body = {"email": f"{email_user}@test.dev", "password": "secret1", "returnSecureToken": True}
    r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signUp?key=fake", json=body)
    if r.status_code != 200:
        r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=fake", json=body)
    j = r.json()
    return j["idToken"], j["localId"]


@pytest.fixture(scope="module")
def env():
    import main  # noqa: F401
    import fates_api
    from fastapi.testclient import TestClient
    from google.cloud import firestore

    requests.delete(f"http://{os.environ['FIRESTORE_EMULATOR_HOST']}/emulator/v1/projects/{PROJECT}/databases/(default)/documents")
    fates_api.blur_thumbs = lambda url: ("data:image/jpeg;base64,QkxVUg==", "data:image/jpeg;base64,UEVFSw==")
    fates_api._d().complete = lambda *a, **k: (json.dumps({"paths": []}), "stop")   # the AI gives nothing usable
    client = TestClient(main.app)
    client.__enter__()
    db = firestore.Client(project=PROJECT)
    token, viewer = _token("dev_chosen")
    active = (NOW - timedelta(hours=2)).isoformat()

    def person(uid, **pub):
        base = {"uid": uid, "firstName": uid.title(), "age": 26, "gender": "Female", "lookingForGender": "Male",
                "cityName": "Bengaluru", "lat": 12.97, "lon": 77.59, "isDiscoverable": True,
                "photos": [f"https://example.com/{uid}.jpg"], "relationshipIntent": "Serious long-term partner",
                "religion": "Hindu", "community": "Kannada", "languages": ["Hindi", "English"],
                "interests": ["Trekking", "Coffee"], "smoking": "Never", "drinking": "Socially"}
        base.update(pub)
        db.document(f"publicProfiles/{uid}").set(base)
        db.document(f"profiles/{uid}").set({"birthDate": "1999-05-02", "birthTime": "10:15", "birthCityName": "Bengaluru",
                                             "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "faceVerified": True,
                                             "faceVerifiedPhotos": [f"https://example.com/{uid}.jpg"]})
        db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})

    def chose(uid, minutes_ago, note):
        at = (NOW - timedelta(minutes=minutes_ago)).isoformat()
        db.document(f"users/{viewer}/incoming/{uid}").set({"type": "like", "action": "like", "fromUid": uid, "viaFate": True,
                                                            "note": note, "fatePath": "heart", "createdAtIso": at})

    db.document(f"publicProfiles/{viewer}").set({
        "uid": viewer, "firstName": "Dev", "age": 28, "gender": "Male", "lookingForGender": "Female", "cityName": "Bengaluru",
        "lat": 12.96, "lon": 77.64, "isDiscoverable": True, "photos": ["https://example.com/dev.jpg"],
        "relationshipIntent": "Serious long-term partner", "religion": "Hindu", "community": "Kannada",
        "languages": ["Hindi", "English"], "interests": ["Trekking"], "smoking": "Never", "drinking": "Socially"})
    db.document(f"profiles/{viewer}").set({"birthDate": "1996-11-20", "birthTime": "06:30", "birthCityName": "Bengaluru",
                                            "birthCountryIso2": "IN", "settings_wantChildren": "Yes", "crownsBalance": 1, "faceVerified": True,
                                            "faceVerifiedPhotos": ["https://example.com/dev.jpg"]})
    db.document(f"users/{viewer}").set({"lastAppOpenAtUtcIso": active})

    for u in ("asha", "bela", "kavya", "tara"):
        person(u)
    person("neha", relationshipIntent="Friends / social circle")     # outside the viewer's filters
    chose("kavya", 300, "Your trek photos made me smile.")            # oldest: brought today
    chose("tara", 120, "Filter coffee first?")                         # next in line
    chose("neha", 60, "Hi!")                                           # Mithu can't bring her
    yield {"client": client, "db": db, "uid": viewer, "h": {"Authorization": f"Bearer {token}"}}
    client.__exit__(None, None, None)


def test_the_oldest_chooser_is_brought_today_and_marked(env):
    body = env["client"].post("/fates/today", headers=env["h"]).json()
    doc = env["db"].document(f"dailyFates/{env['uid']}__{body['dayKey']}").get().to_dict()
    targets = {path: p["targetUid"] for path, p in doc["paths"].items()}
    assert "kavya" in targets.values() and "tara" not in targets.values()
    path = next(k for k, v in targets.items() if v == "kavya")
    teaser = next(p["teaser"] for p in body["paths"] if p["path"] == path)
    assert teaser["choseYou"] is True
    assert "Kavya" not in json.dumps(body)                            # still anonymous until opened
    mem = env["db"].document(f"fatesMemory/{env['uid']}").get().to_dict()
    assert mem["chooserBrought"]["kavya"] == 1
    # Tara also chose the viewer; if she came in on her own merits, her card is marked too.
    for path, uid in targets.items():
        t = next(p["teaser"] for p in body["paths"] if p["path"] == path)
        assert t["choseYou"] is (uid in ("kavya", "tara", "neha"))


def test_chose_you_lists_them_blurred_with_when_they_come(env):
    r = env["client"].post("/fates/chosen", headers=env["h"])
    assert r.status_code == 200, r.text
    listed = r.json()["items"]
    items = {it["uid"]: it for it in listed}
    assert items["kavya"]["status"] == "today" and items["kavya"]["note"] == "Your trek photos made me smile."
    assert items["tara"]["status"] == "next" and items["tara"]["position"] == 1
    assert items["neha"]["status"] == "later"
    for it in listed:
        assert it["revealed"] is False and "person" not in it and it["blurThumb"]
    assert [it["uid"] for it in listed] == ["neha", "tara", "kavya"]    # newest first


def test_reveal_costs_one_crown_then_402(env):
    c = env["client"]
    r = c.post("/fates/chosen/reveal", headers=env["h"], json={"uid": "tara"})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["item"]["person"]["firstName"] == "Tara" and got["charged"] is True and got["crownsLeft"] == 0
    again = c.post("/fates/chosen/reveal", headers=env["h"], json={"uid": "tara"})
    assert again.status_code == 200 and again.json()["charged"] is False          # already shown: free
    assert c.post("/fates/chosen/reveal", headers=env["h"], json={"uid": "neha"}).status_code == 402
    assert c.post("/fates/chosen/reveal", headers=env["h"], json={"uid": "asha"}).status_code == 404   # never chose
    mem = env["db"].document(f"fatesMemory/{env['uid']}").get().to_dict()
    assert mem["seen"]["tara"]["opened"] is True and "tara" in mem["revealed"]
    items = {it["uid"]: it for it in c.post("/fates/chosen", headers=env["h"]).json()["items"]}
    assert items["tara"]["status"] == "revealed" and items["tara"]["person"]["firstName"] == "Tara"



def test_an_unverified_viewer_is_asked_to_verify_first(env):
    import fates_api
    db = env["db"]
    token, uid = _token("raj_unverified")
    active = (NOW - timedelta(hours=1)).isoformat()
    db.document(f"publicProfiles/{uid}").set({
        "uid": uid, "firstName": "Raj", "age": 27, "gender": "Male", "lookingForGender": "Female", "cityName": "Bengaluru",
        "lat": 12.96, "lon": 77.64, "isDiscoverable": True, "photos": ["https://example.com/raj.jpg"],
        "relationshipIntent": "Serious long-term partner", "religion": "Hindu", "community": "Kannada"})
    db.document(f"profiles/{uid}").set({"birthDate": "1997-01-01", "faceVerificationPassed": True})   # the app's copy: not trusted
    db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": active})
    h = {"Authorization": f"Bearer {token}"}
    body = env["client"].post("/fates/today", headers=h).json()
    assert body["needsVerification"] is True and body["paths"] == [] and body["verifiedNearby"] >= 1
    assert body["verifyReason"] == ""
    assert not db.document(f"dailyFates/{uid}__{body['dayKey']}").get().exists
    opened = env["client"].post("/fates/open", headers=h, json={"path": "heart"})
    assert opened.status_code == 403 and opened.json()["detail"] == "not_verified"
    db.document(f"profiles/{uid}").set({"faceVerified": True, "faceVerifiedPhotos": ["https://example.com/raj.jpg"]}, merge=True)
    after = env["client"].post("/fates/today", headers=h).json()                                     # the server verified them
    assert not after.get("needsVerification") and len(after["paths"]) >= 1

    # A new photo the server hasn't checked pauses it: the app is told to re-check.
    db.document(f"publicProfiles/{uid}").set({"photos": ["https://example.com/raj.jpg", "https://example.com/new.jpg"]}, merge=True)
    db.document(f"dailyFates/{uid}__{after['dayKey']}").delete()
    paused = env["client"].post("/fates/today", headers=h).json()
    assert paused["needsVerification"] is True and paused["verifyReason"] == "photos_changed"
