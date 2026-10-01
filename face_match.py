"""Is the person in the selfies the person in the profile photos? (OpenCV YuNet + SFace, runs on Cloud Run.)

Models (opencv_zoo, commit 25f423d, checked by sha256 when the image is built, see Dockerfile):
  face_detection_yunet_2023mar.onnx   MIT         finds faces and their five landmarks
  face_recognition_sface_2021dec.onnx Apache-2.0  turns an aligned face into 128 numbers
Two faces are compared by cosine similarity of those numbers (1 = identical, around 0 = unrelated).
SFace's published threshold for "same person" is 0.363. The main photo must clear a stricter bar,
because a wrong "verified" badge is worse than asking someone to try again.

Rules (all tunable by environment, see the constants):
  - the three selfies show one person (front vs each side);
  - the main photo shows that person (strong match); a main photo without a face lets any photo carry it;
  - every other photo with one face shows that person too; in a group photo one of the faces must;
  - photos without a face (a landscape, a pet) don't count either way.

Look-alikes: some faces are simply close (family, similar features, the same pose and light). So the selfie is
also compared with a "cohort": the main photos of other people of the same gender. A photo only counts as you
when it is clearly closer to your selfie than 90% of those strangers are (cohort score normalisation). With
fewer than COHORT_MIN strangers to compare with, the fixed thresholds apply alone.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

SAME_PERSON = float(os.getenv("FACE_SAME_PERSON") or "0.363")       # SFace's published threshold
MAIN_PHOTO = float(os.getenv("FACE_MAIN_PHOTO") or "0.45")          # the main photo must be clearly you
SELFIES_SAME = float(os.getenv("FACE_SELFIES_SAME") or "0.30")      # a turned head scores lower than a straight one
# Just under a bar is not a "no": a person on the team looks (like Bumble's human review).
REVIEW_BAND = float(os.getenv("FACE_REVIEW_BAND") or "0.08")
# The same face already verified on another account (Tinder's duplicate check): a person on the team looks.
DUPLICATE = float(os.getenv("FACE_DUPLICATE") or "0.65")
COHORT_MIN = int(os.getenv("FACE_COHORT_MIN") or "5")
COHORT_MARGIN = float(os.getenv("FACE_COHORT_MARGIN") or "0.10")   # the main photo: this much above the 90th percentile
COHORT_CAP = float(os.getenv("FACE_COHORT_CAP") or "0.75")         # never ask for more than this (a cohort of twins)
DETECT_SCORE = 0.85
MAX_SIDE = 1024
MODEL_DIR = os.getenv("FACE_MODEL_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"

_lock = threading.Lock()
_models: Optional[Tuple[Any, Any]] = None


def _load():
    global _models
    with _lock:
        if _models is None:
            import cv2
            det = cv2.FaceDetectorYN.create(os.path.join(MODEL_DIR, YUNET), "", (320, 320), DETECT_SCORE, 0.3, 5000)
            rec = cv2.FaceRecognizerSF.create(os.path.join(MODEL_DIR, SFACE), "")
            _models = (det, rec)
        return _models


def available() -> bool:
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in (YUNET, SFACE))


def decode(data: bytes):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def face_features(img) -> List[Any]:
    """One 128-number vector per face found, biggest face first. The detector runs on one image at a time
    (it keeps the input size), so calls are serialised."""
    if img is None:
        return []
    det, rec = _load()
    h, w = img.shape[:2]
    with _lock:
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None:
            return []
        faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)
        return [rec.feature(rec.alignCrop(img, f)) for f in faces]


def similarity(a: Any, b: Any) -> float:
    """Cosine similarity of two feature vectors (works for SFace output and for plain lists in tests)."""
    import numpy as np
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / n) if n > 0 else 0.0


def bars(cohort: Sequence[float]) -> Tuple[float, float, Optional[float]]:
    """(main photo bar, other photos bar, the cohort's 90th percentile or None)."""
    if len(cohort) < COHORT_MIN:
        return MAIN_PHOTO, SAME_PERSON, None
    import numpy as np
    p90 = float(np.percentile(np.asarray(cohort, dtype=np.float64), 90))
    return (max(MAIN_PHOTO, min(p90 + COHORT_MARGIN, COHORT_CAP)),
            max(SAME_PERSON, min(p90, COHORT_CAP)), p90)


def judge(front: Any, sides: Sequence[Any], photos: Sequence[Sequence[Any]],
          cohort: Sequence[float] = ()) -> Tuple[bool, str, Dict[str, Any]]:
    """front: the straight selfie's face; sides: the turned ones (may be empty for a re-check);
    photos: the faces found in each profile photo, in profile order; cohort: the front selfie's similarity to
    strangers' faces. Returns (ok, reason, scores)."""
    main_bar, other_bar, p90 = bars(cohort)
    meta: Dict[str, Any] = {"sides": [], "photos": [], "bars": [round(main_bar, 3), round(other_bar, 3)],
                            "cohort": {"n": len(cohort), "p90": None if p90 is None else round(p90, 3)}}
    for s in sides:
        sim = similarity(front, s)
        meta["sides"].append(round(sim, 3))
        if sim < SELFIES_SAME:
            return False, "selfies_not_same_person", meta

    best: List[Optional[float]] = []
    for faces in photos:
        best.append(max((similarity(front, f) for f in faces), default=None))
    meta["photos"] = [None if b is None else round(b, 3) for b in best]

    with_face = [i for i, b in enumerate(best) if b is not None]
    if not with_face:
        return False, "no_face_in_photos", meta

    if best[0] is not None:
        if best[0] < main_bar:
            return False, ("review_main_photo" if best[0] >= main_bar - REVIEW_BAND else "main_photo_not_you"), meta
    else:
        top = max(best[i] for i in with_face)
        if top < main_bar:
            return False, ("review_main_photo" if top >= main_bar - REVIEW_BAND else "photos_not_you"), meta

    for i in with_face[1:] if best[0] is not None else with_face:
        if best[i] < other_bar:
            return False, (f"review_photo_{i + 1}" if best[i] >= other_bar - REVIEW_BAND else f"photo_{i + 1}_not_you"), meta
    return True, "ok", meta


def mirror(img):
    """The image flipped left to right. Front cameras show (and some phones save) a mirror image, profile
    photos may be either way, and SFace scores the same face up to ~0.06 apart between the two."""
    if img is None:
        return None
    import cv2
    return cv2.flip(img, 1)


def outcome_rank(ok: bool, reason: str, meta: Dict[str, Any]) -> Tuple[int, float]:
    """Which of two judgements (the frames as taken, or mirrored) to keep: a pass, then a "person looks",
    then a fail; within those, the better main-photo score."""
    best = [s for s in (meta.get("photos") or []) if s is not None]
    return (2 if ok else 1 if needs_review(reason) else 0, max(best, default=-1.0))


def needs_review(reason: str) -> bool:
    return reason.startswith("review_")


def duplicates(front: Any, templates: Dict[str, Any], exclude: str = "") -> List[Tuple[str, float]]:
    """Other accounts whose verified face looks like this one, most alike first."""
    hits = [(uid, similarity(front, v)) for uid, v in templates.items() if uid != exclude]
    return sorted([h for h in hits if h[1] >= DUPLICATE], key=lambda h: -h[1])
