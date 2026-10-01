"""Face verification end to end: challenge, liveness, face match, review, duplicates, retention, withdraw.

Runs against the Firestore + Auth + Storage emulators (see test_fates_api_it.py; the Storage one on 9199 unless
FIREBASE_STORAGE_EMULATOR_HOST says otherwise). Cloud Vision and the face model are replaced by fakes keyed by
the selfie's name, so the server's own logic is what's tested.
"""
import os
import urllib.parse
from types import SimpleNamespace as NS

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not (os.getenv("FIRESTORE_EMULATOR_HOST") and os.getenv("FIREBASE_AUTH_EMULATOR_HOST")),
    reason="needs the Firestore, Auth and Storage emulators")

PROJECT = os.getenv("GCLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "demo-fates"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT)
os.environ.setdefault("FIREBASE_STORAGE_EMULATOR_HOST", "127.0.0.1:9199")
BUCKET = f"{PROJECT}.appspot.com"
STORAGE = os.environ["FIREBASE_STORAGE_EMULATOR_HOST"]

# Faces as vectors: every test user has their own face ("<name>"), "<name>~almost" is just under the main
# photo's bar, "stranger" is nobody's.
_people = {}


def vec(key: str):
    base, _, variant = key.partition("~")
    v = [0.0] * 32
    if base == "stranger":
        v[30] = 1.0
        return v
    i = _people.setdefault(base, len(_people))
    if variant == "almost":
        v[i], v[31] = 0.42, 0.907
    else:
        v[i] = 1.0
    return v

PAN = {"front": 1.0, "right": 24.0, "left": -22.0, "smile": 3.0}


def _token(name: str):
    host = os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    body = {"email": f"{name}@face.dev", "password": "secret1", "returnSecureToken": True}
    r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signUp?key=fake", json=body)
    if r.status_code != 200:
        r = requests.post(f"http://{host}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=fake", json=body)
    j = r.json()
    return j["idToken"], j["localId"]


def _url(path: str) -> str:
    return f"http://{STORAGE}/v0/b/{BUCKET}/o/{urllib.parse.quote(path, safe='')}?alt=media"


@pytest.fixture(scope="module")
def env():
    import main
    import face_match
    from fastapi.testclient import TestClient
    from google.cloud import firestore

    requests.delete(f"http://{os.environ['FIRESTORE_EMULATOR_HOST']}/emulator/v1/projects/{PROJECT}/databases/(default)/documents")

    # A selfie's bytes say what it shows: "<face>|<pose>|<n>" (n keeps every upload unique).
    blobs = {}
    main._download_image_bytes = lambda url, max_mb=10: blobs[url]

    def vision(data):
        face, pose = data.decode().split("|")[:2]
        joy = 5 if pose == "smile" else 1
        verts = [NS(x=0, y=0), NS(x=320, y=0), NS(x=320, y=320), NS(x=0, y=320)]
        f = NS(pan_angle=PAN[pose], roll_angle=0.0, tilt_angle=0.0, detection_confidence=0.95, joy_likelihood=joy,
               bounding_poly=NS(vertices=verts))
        return "VERY_UNLIKELY", "VERY_UNLIKELY", "VERY_UNLIKELY", [f]

    main._detect_faces_and_safety_from_bytes = vision
    face_match.available = lambda: True
    face_match.decode = lambda data: data.decode().split("|")[0]
    face_match.mirror = lambda key: key                      # the fake faces look the same either way
    face_match.face_features = lambda key: [vec(key)] if key else []
    photos = {}
    main._own_photos = lambda uid: photos.get(uid, [])
    main._photo_faces = lambda urls: [[vec(urllib.parse.unquote(u.rsplit("/", 1)[-1]).rsplit("/", 1)[-1].rsplit(".", 1)[0])]
                                      for u in urls]
    main._cohort_scores = lambda uid, front: []
    main._templates.update(at=0.0, full=0.0, since="", v={})

    client = TestClient(main.app)
    client.__enter__()
    db = firestore.Client(project=PROJECT)
    gcs = main._gcs().bucket(BUCKET)
    counter = {"n": 0}

    def user(name, photo_face=None):
        token, uid = _token(name)
        photos[uid] = [f"https://firebasestorage.googleapis.com/v0/b/{BUCKET}/o/users%2F{uid}%2Fphotos%2F{photo_face or name}.jpg"]
        db.document(f"publicProfiles/{uid}").set({"uid": uid, "gender": "Female", "photos": photos[uid]})
        db.document(f"profiles/{uid}").set({"firstName": name})
        return uid, {"Authorization": f"Bearer {token}"}

    def selfies(uid, steps, face, poses=None):
        urls = []
        for step, pose in zip(steps, poses or steps):
            counter["n"] += 1
            path = f"users/{uid}/verification/{counter['n']}.jpg"
            data = f"{face}|{pose}|{counter['n']}|".encode() * 200
            gcs.blob(path).upload_from_string(data, content_type="image/jpeg")
            blobs[_url(path)] = data
            urls.append(_url(path))
        return urls

    def attempt(uid, h, face, poses=None):
        start = client.post("/verify-face-start", headers=h, json={"consent": "face-test"})
        assert start.status_code == 200, start.text
        ch = start.json()
        urls = selfies(uid, ch["steps"], face, poses(ch["steps"]) if poses else None)
        r = client.post("/verify-face-live", headers=h, json={"challengeId": ch["challengeId"], "selfies": urls})
        return r, ch, urls

    def left(uid):
        return sorted(_url(b.name) for b in gcs.list_blobs(prefix=f"users/{uid}/verification/"))

    yield NS(client=client, db=db, user=user, attempt=attempt, selfies=selfies, left=left, photos=photos, main=main)
    client.__exit__(None, None, None)


def test_a_challenge_is_random_short_lived_and_used_once(env):
    uid, h = env.user("chal")
    assert env.client.post("/verify-face-start", headers=h, json={}).json()["detail"] == "consent_needed"
    ch = env.client.post("/verify-face-start", headers=h, json={"consent": "face-2026-10"}).json()
    assert env.db.document(f"faceVerifications/{uid}").get().to_dict()["consent"]["version"] == "face-2026-10"
    assert ch["steps"][0] == "front" and len(ch["steps"]) == 3 and ch["expiresInSec"] == 600
    urls = env.selfies(uid, ch["steps"], "chal")
    bad = env.client.post("/verify-face-live", headers=h, json={"challengeId": "made-up", "selfies": urls})
    assert bad.status_code == 400 and bad.json()["detail"] == "challenge_expired"
    ok = env.client.post("/verify-face-live", headers=h, json={"challengeId": ch["challengeId"], "selfies": urls})
    assert ok.json()["status"] == "ok"
    again = env.client.post("/verify-face-live", headers=h, json={"challengeId": ch["challengeId"], "selfies": urls})
    assert again.status_code == 400                                     # the same challenge can't be replayed


def test_passing_verifies_keeps_only_the_straight_selfie(env):
    uid, h = env.user("pass")
    r, ch, urls = env.attempt(uid, h, "pass")
    assert r.json() == {"ok": True, "status": "ok", "reason": "ok"}
    prof = env.db.document(f"profiles/{uid}").get().to_dict()
    assert prof["faceVerified"] and prof["isFaceVerified"] and prof["faceVerifiedPhotos"] == env.photos[uid]
    assert prof["faceVerifiedMethod"] == "challenge+photos"
    assert env.db.document(f"publicProfiles/{uid}").get().to_dict()["isFaceVerified"] is True
    rec = env.db.document(f"faceVerifications/{uid}").get().to_dict()
    assert rec["status"] == "ok" and rec["frontSelfie"] == urls[0] and rec["selfies"] == [urls[0]] and "review" not in rec
    assert env.left(uid) == [urls[0]]                                   # the two other selfies are deleted
    assert env.db.document(f"faceTemplates/{uid}").get().to_dict()["status"] == "verified"


def test_a_failed_pose_keeps_nothing(env):
    uid, h = env.user("pose")
    r, _, _ = env.attempt(uid, h, "pose", poses=lambda steps: ["front"] * len(steps))
    assert r.json()["status"] == "failed" and r.json()["reason"] in ("turn_your_head_more", "smile_please")
    rec = env.db.document(f"faceVerifications/{uid}").get().to_dict()
    assert rec["frontSelfie"] == "" and rec["posesOk"] is False
    assert env.left(uid) == []
    assert not env.db.document(f"faceTemplates/{uid}").get().exists


def test_photos_that_are_someone_else_fail_but_keep_the_selfie_for_a_recheck(env):
    uid, h = env.user("wrongphoto", photo_face="stranger")
    r, _, urls = env.attempt(uid, h, "wrongphoto")
    assert r.json()["reason"] == "main_photo_not_you" and r.json()["status"] == "failed"
    assert env.left(uid) == [urls[0]]
    env.photos[uid] = [env.photos[uid][0].replace("stranger", "wrongphoto")]  # the user puts in a photo of themselves
    rc = env.client.post("/verify-face-recheck", headers=h).json()
    assert rc["ok"] and env.db.document(f"profiles/{uid}").get().to_dict()["faceVerified"] is True


def test_just_under_the_bar_goes_to_a_person(env):
    uid, h = env.user("almost", photo_face="almost~almost")
    r, _, urls = env.attempt(uid, h, "almost")
    assert r.json() == {"ok": False, "status": "in_review", "reason": "review_main_photo"}
    rec = env.db.document(f"faceVerifications/{uid}").get().to_dict()
    assert rec["review"]["kind"] == "photos" and rec["review"]["frontSelfie"] == urls[0]
    assert env.db.document(f"profiles/{uid}").get().to_dict()["faceVerified"] is False
    assert env.db.document(f"faceTemplates/{uid}").get().to_dict()["status"] == "pending"


def test_the_same_face_on_a_second_account_goes_to_a_person(env):
    first, h1 = env.user("original")
    assert env.attempt(first, h1, "original")[0].json()["status"] == "ok"
    second, h2 = env.user("copycat", photo_face="original")           # the same person, a second account
    r, _, _ = env.attempt(second, h2, "original")
    assert r.json()["reason"] == "review_duplicate"
    rec = env.db.document(f"faceVerifications/{second}").get().to_dict()
    assert rec["review"]["kind"] == "duplicate" and rec["review"]["duplicateOf"][0]["uid"] == first


def test_withdraw_removes_everything(env):
    uid, h = env.user("withdraw")
    assert env.attempt(uid, h, "withdraw")[0].json()["status"] == "ok"
    assert env.client.post("/verify-face-withdraw", headers=h).json() == {"ok": True}
    assert env.left(uid) == []
    rec = env.db.document(f"faceVerifications/{uid}").get().to_dict()
    assert set(rec) == {"status", "reason", "updatedAtIso", "consent"} and rec["status"] == "withdrawn"
    assert rec["consent"]["version"] == "face-test" and rec["consent"]["withdrawnAtIso"]
    assert not env.db.document(f"faceTemplates/{uid}").get().exists
    prof = env.db.document(f"profiles/{uid}").get().to_dict()
    assert prof["faceVerified"] is False and prof["faceVerifiedPhotos"] == [] and prof["faceVerifiedReason"] == "withdrawn"
    assert uid not in env.main._templates["v"]


def test_tries_per_day_are_limited(env):
    uid, h = env.user("tries")
    for _ in range(env.main.FACE_VERIFY_DAILY_ATTEMPTS):
        env.attempt(uid, h, "tries", poses=lambda steps: ["front"] * 3)
    assert env.client.post("/verify-face-start", headers=h, json={"consent": "face-test"}).status_code == 429


def test_the_duplicate_cache_reads_only_what_changed(env):
    m = env.main
    m._templates.update(at=0.0)
    before = dict(m._verified_templates())
    assert before and m._templates["full"] > 0
    # an admin approves someone (functions/faceReview.ts sets status + updatedAtIso)
    env.db.document("faceTemplates/approved_later").set({"v": [0.0] * 31 + [1.0], "status": "verified", "updatedAtIso": m._now_iso()})
    env.db.document("faceTemplates/still_pending").set({"v": [1.0] + [0.0] * 31, "status": "pending", "updatedAtIso": m._now_iso()})
    m._templates.update(at=0.0)
    after = m._verified_templates()
    assert "approved_later" in after and "still_pending" not in after
    assert set(before) <= set(after)
