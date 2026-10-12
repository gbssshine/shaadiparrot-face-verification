"""Face verification, part 1: a live person doing a random challenge (like Bumble's random pose).

The server picks the challenge: a straight selfie, then two of "turn right", "turn left", "smile" in a random
order, valid for 10 minutes and once. Cloud Vision reads each shot (one clear face, not explicit, head angle,
smile). A set of photos made in advance can't know the order. Part 2 (face_match.py) checks it's the person in
the profile photos.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

LIKELIHOOD = {"UNKNOWN": 0, "VERY_UNLIKELY": 1, "UNLIKELY": 2, "POSSIBLE": 3, "LIKELY": 4, "VERY_LIKELY": 5}
MIN_FACE_AREA = 18_000
FRONT_MAX_PAN = 15.0     # degrees: looking at the camera
SIDE_MIN_PAN = 10.0      # a real turn
SIDE_MAX_PAN = 50.0      # still a face Vision can read
MAX_ROLL_TILT = 30.0


GOOGLE_STORAGE_HOSTS = ("firebasestorage.googleapis.com", "storage.googleapis.com")
# The production project's own buckets (the app uploads to the first). Bucket names ending in .appspot.com or
# .firebasestorage.app can only be made by the project they name, so no one else can hold a bucket called this.
PROJECT_BUCKETS = ("shaadiparrot.firebasestorage.app", "shaadiparrot.appspot.com")
MAX_OBJECT_PATH = 1024


def storage_hosts() -> Tuple[str, ...]:
    hosts = list(GOOGLE_STORAGE_HOSTS)
    emu = (os.getenv("FIREBASE_STORAGE_EMULATOR_HOST") or "").strip()
    if emu:
        hosts.append(emu.split("//")[-1].split("/")[0].lower())
        hosts.append("10.0.2.2:" + emu.rsplit(":", 1)[-1])   # the Android emulator's name for the host
    return tuple(hosts)


def photo_buckets() -> Tuple[str, ...]:
    """The buckets a member's own upload (photo or selfie) may live in: the project's own (PROJECT_BUCKETS), the
    ones FIREBASE_STORAGE_BUCKET names (comma-separated, gs:// allowed), and the default buckets of the project the
    server runs as (GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT: the emulator tests' demo project). Never any other bucket,
    so a link to users/{uid}/photos/ in someone else's bucket is not that user's photo."""
    out = list(PROJECT_BUCKETS)
    for b in (os.getenv("FIREBASE_STORAGE_BUCKET") or "").split(","):
        b = b.strip().lower()
        b = b[len("gs://"):] if b.startswith("gs://") else b
        b = b.strip("/")
        if b:
            out.append(b)
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        project = (os.getenv(var) or "").strip().lower()
        if project:
            out += [f"{project}.firebasestorage.app", f"{project}.appspot.com"]
    return tuple(dict.fromkeys(out))


def _safe_object(path: str) -> bool:
    """media.ts safeObjectPath: 1..1024 characters, no control characters, no empty, '.' or '..' segment."""
    return (0 < len(path) <= MAX_OBJECT_PATH and not any(ord(c) < 32 or ord(c) == 127 for c in path)
            and not any(seg in ("", ".", "..") for seg in path.split("/")))


def _storage_ref(url: str, hosts: Optional[Iterable[str]] = None) -> Optional[Tuple[str, str]]:
    """(bucket, object path) of an http(s) link to an object in one of the project's buckets (photo_buckets), on a
    Storage host (https only on Google's; http only for the local emulator). Two forms:
      {host}/v0/b/{bucket}/o/{object, URL-encoded as one segment}    Firebase download links (media.ts storageRef)
      storage.googleapis.com/{bucket}/{object}                       plain Cloud Storage links
    None for anything else (another host, another bucket, an unsafe object path)."""
    if not isinstance(url, str):
        return None
    try:
        u = urlparse(url)
    except ValueError:
        return None
    host = u.netloc.lower()
    allowed = tuple(h.lower() for h in (hosts or storage_hosts()))
    if host not in allowed or u.scheme not in ("https", "http"):
        return None
    if u.scheme == "http" and host in GOOGLE_STORAGE_HOSTS:
        return None
    parts = u.path.split("/")
    if len(parts) >= 2 and parts[1] == "v0":
        if len(parts) != 6 or parts[0] or parts[2] != "b" or parts[4] != "o":
            return None
        bucket, obj = unquote(parts[3]), unquote(parts[5])
    elif host == "storage.googleapis.com" and len(parts) >= 3 and not parts[0]:
        bucket, obj = unquote(parts[1]), unquote("/".join(parts[2:]))
    else:
        return None
    bucket = bucket.lower()
    if not bucket or bucket not in photo_buckets() or not _safe_object(obj):
        return None
    return bucket, obj


def selfie_url_ok(url: str, uid: str, hosts: Optional[Iterable[str]] = None) -> bool:
    """Only the user's own verification uploads: a Storage link (https, or the local emulator) to an object in the
    project's own bucket under users/{uid}/verification/."""
    if not uid or "/" in uid:
        return False
    ref = _storage_ref(url, hosts)
    return ref is not None and ref[1].startswith(f"users/{uid}/verification/")


def own_photo_object(url: str, uid: str, hosts: Optional[Iterable[str]] = None) -> Optional[str]:
    """The object path of a profile photo the server may download, or None: a Storage link (https, or the local
    emulator) to an object in the project's own bucket (photo_buckets) whose path starts with users/{uid}/photos/.
    The rule of media.ts parsePhoto's 'own' kind, so the server-made blurs (users/{uid}/blur/), the resized variants
    (users/{uid}/variants/), a file nested in someone else's folder and any other bucket or host never count."""
    if not uid or "/" in uid:
        return None
    ref = _storage_ref(url, hosts)
    return ref[1] if ref is not None and ref[1].startswith(f"users/{uid}/photos/") else None


def own_photo_url(url: str, uid: str, hosts: Optional[Iterable[str]] = None) -> bool:
    """A profile photo the server may download (own_photo_object): on a Storage host, in the project's own bucket and
    in the user's own photos folder, users/{uid}/photos/ (no other host or bucket, ever)."""
    return own_photo_object(url, uid, hosts) is not None


def storage_object(url: str) -> Optional[Tuple[str, str]]:
    """(bucket, object path) of a Firebase Storage download URL (.../v0/b/{bucket}/o/{path}), else None."""
    try:
        parts = urlparse(url).path.split("/")
    except ValueError:
        return None
    if len(parts) >= 6 and parts[1] == "v0" and parts[2] == "b" and parts[4] == "o" and parts[3] and parts[5]:
        path = unquote("/".join(parts[5:]))
        return (parts[3], path) if ".." not in path else None
    return None


def _f(face: Any, name: str) -> float:
    try:
        return float(getattr(face, name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def area(face: Any) -> float:
    pts = getattr(getattr(face, "bounding_poly", None), "vertices", None) or []
    xs = [p.x for p in pts if getattr(p, "x", None) is not None]
    ys = [p.y for p in pts if getattr(p, "y", None) is not None]
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def unsafe(adult: str, racy: str, violence: str) -> Optional[str]:
    if LIKELIHOOD.get(adult, 0) >= LIKELIHOOD["LIKELY"]:
        return "adult_content"
    if LIKELIHOOD.get(racy, 0) >= LIKELIHOOD["VERY_LIKELY"]:
        return "highly_racy"
    if LIKELIHOOD.get(violence, 0) >= LIKELIHOOD["VERY_LIKELY"]:
        return "high_violence"
    return None


def one_face(faces: Sequence[Any]) -> Tuple[Optional[Any], Optional[str]]:
    """The face of the shot: exactly one clear face (a second big face means someone else is in it)."""
    if not faces:
        return None, "no_face_detected"
    big = sorted(faces, key=area, reverse=True)
    if len(big) > 1 and area(big[1]) > 0.35 * area(big[0]):
        return None, "more_than_one_face"
    face = big[0]
    if area(face) < MIN_FACE_AREA:
        return None, "face_too_small"
    if _f(face, "detection_confidence") < 0.45:
        return None, "low_confidence"
    if abs(_f(face, "roll_angle")) > MAX_ROLL_TILT or abs(_f(face, "tilt_angle")) > MAX_ROLL_TILT:
        return None, "face_too_angled"
    return face, None


STEP_POOL = ("right", "left", "smile")
SMILE_MIN = LIKELIHOOD["LIKELY"]
SMILE_MAX_PAN = 25.0


def new_challenge(rng=None) -> List[str]:
    """["front", x, y]: x and y two different steps from STEP_POOL, in a random order."""
    import random
    r = rng or random.SystemRandom()
    return ["front"] + r.sample(list(STEP_POOL), 2)


def _joy(face: Any) -> int:
    v = getattr(face, "joy_likelihood", 0)
    try:
        return int(getattr(v, "value", v))
    except (TypeError, ValueError):
        return LIKELIHOOD.get(str(v), 0)


def judge_challenge(steps: Sequence[str], shots: Sequence[Tuple[str, str, str, Sequence[Any]]]) -> Tuple[bool, str, List[float]]:
    """Each shot against its step. Returns (ok, reason, pan angles)."""
    if len(shots) != len(steps) or not steps or steps[0] != "front":
        return False, "challenge_mismatch", []
    pans: List[float] = []
    faces_by_step = []
    for i, (adult, racy, violence, faces) in enumerate(shots):
        bad = unsafe(adult, racy, violence)
        if bad:
            return False, bad, pans
        face, why = one_face(faces)
        if face is None:
            return False, f"{why}_{i + 1}", pans
        pans.append(_f(face, "pan_angle"))
        faces_by_step.append(face)
    if abs(pans[0]) > FRONT_MAX_PAN:
        return False, "look_straight_first", pans
    turns = []
    for step, face, pan in zip(steps[1:], faces_by_step[1:], pans[1:]):
        if step in ("right", "left"):
            if not (SIDE_MIN_PAN <= abs(pan) <= SIDE_MAX_PAN):
                return False, "turn_your_head_more", pans
            turns.append(pan)
        elif step == "smile":
            if _joy(face) < SMILE_MIN:
                return False, "smile_please", pans
            if abs(pan) > SMILE_MAX_PAN:
                return False, "look_at_the_camera_to_smile", pans
        else:
            return False, "challenge_mismatch", pans
    if len(turns) == 2 and (turns[0] > 0) == (turns[1] > 0):
        return False, "turn_to_both_sides", pans          # both shots turned the same way: one photo reused
    return True, "ok", pans


def judge(shots: Sequence[Tuple[str, str, str, Sequence[Any]]]) -> Tuple[bool, str, List[float]]:
    """shots: (adult, racy, violence, faces) for the straight, right and left selfies, in that order.
    Returns (ok, reason, pan angles)."""
    if len(shots) != 3:
        return False, "three_selfies_needed", []
    pans: List[float] = []
    for i, (adult, racy, violence, faces) in enumerate(shots):
        bad = unsafe(adult, racy, violence)
        if bad:
            return False, bad, pans
        face, why = one_face(faces)
        if face is None:
            return False, f"{why}_{i + 1}", pans
        pans.append(_f(face, "pan_angle"))
    front, a, b = pans
    if abs(front) > FRONT_MAX_PAN:
        return False, "look_straight_first", pans
    for side in (a, b):
        if not (SIDE_MIN_PAN <= abs(side) <= SIDE_MAX_PAN):
            return False, "turn_your_head_more", pans
    if (a > 0) == (b > 0):
        return False, "turn_to_both_sides", pans      # both shots turned the same way: likely one photo reused
    return True, "ok", pans
