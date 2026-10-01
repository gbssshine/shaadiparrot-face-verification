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


def storage_hosts() -> Tuple[str, ...]:
    hosts = ["firebasestorage.googleapis.com", "storage.googleapis.com"]
    emu = (os.getenv("FIREBASE_STORAGE_EMULATOR_HOST") or "").strip()
    if emu:
        hosts.append(emu.split("//")[-1].split("/")[0].lower())
        hosts.append("10.0.2.2:" + emu.rsplit(":", 1)[-1])   # the Android emulator's name for the host
    return tuple(hosts)


def selfie_url_ok(url: str, uid: str, hosts: Optional[Iterable[str]] = None) -> bool:
    """Only the user's own verification uploads: https (or the local emulator) on a Storage host, and the
    object under users/{uid}/verification/."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    allowed = tuple(h.lower() for h in (hosts or storage_hosts()))
    if u.netloc.lower() not in allowed:
        return False
    if u.scheme != "https" and not (u.scheme == "http" and u.netloc.lower() not in ("firebasestorage.googleapis.com", "storage.googleapis.com")):
        return False
    path = unquote(u.path)
    return f"users/{uid}/verification/" in path and ".." not in path


def own_photo_url(url: str, uid: str, hosts: Optional[Iterable[str]] = None) -> bool:
    """A profile photo the server may download: on a Storage host and under users/{uid}/ (no other hosts, ever)."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    allowed = tuple(h.lower() for h in (hosts or storage_hosts()))
    if u.netloc.lower() not in allowed or u.scheme not in ("https", "http"):
        return False
    if u.scheme == "http" and u.netloc.lower() in ("firebasestorage.googleapis.com", "storage.googleapis.com"):
        return False
    path = unquote(u.path)
    return f"users/{uid}/" in path and ".." not in path


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
