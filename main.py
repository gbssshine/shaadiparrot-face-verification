import os
import json
import math
import hashlib
from typing import List, Optional, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse

# InsightFace (embeddings + detection)
from insightface.app import FaceAnalysis


APP_NAME = "shaadiparrot-face-verification"

# ========== CONFIG ==========
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "0") == "1"
# Minimum similarity score (cosine). Typical good threshold: 0.30-0.45 depending on model/data.
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.35"))
# Allow service to download image URLs (Firebase Storage downloadURL)
ALLOW_URLS = os.getenv("ALLOW_URLS", "1") == "1"
# Max photos to compare in one request (protect cost)
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "12"))

# ========== APP ==========
app = FastAPI(title=APP_NAME)

# Initialize face model once (global)
# InsightFace downloads/caches models. On Cloud Run cache goes to /root/.insightface by default.
face_app = FaceAnalysis(
    name="buffalo_l",  # good quality model
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _read_image_bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file (cannot decode).")
    return img


def _download_url_to_bytes(url: str) -> bytes:
    # Avoid adding heavy dependencies; use stdlib urllib
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Failed to download image_url: {url}")


def _detect_single_face_embedding(bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Returns (embedding, meta)
    """
    faces = face_app.get(bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Need exactly 1 face.")

    f = faces[0]
    emb = f.embedding
    if emb is None or len(emb) == 0:
        raise HTTPException(status_code=400, detail="Face embedding extraction failed.")

    # Basic quality hints (optional)
    bbox = f.bbox.astype(int).tolist() if getattr(f, "bbox", None) is not None else None
    det_score = float(getattr(f, "det_score", 0.0))

    return emb.astype(np.float32), {
        "bbox": bbox,
        "det_score": det_score,
    }


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _auth_check(authorization: Optional[str]) -> dict:
    """
    Optional Firebase idToken verification.
    If REQUIRE_AUTH=1 -> must pass valid token.
    """
    if not REQUIRE_AUTH:
        return {"auth": "skipped"}

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <Firebase idToken>")

    id_token = authorization.split(" ", 1)[1].strip()
    try:
        import firebase_admin
        from firebase_admin import auth as fb_auth

        if not firebase_admin._apps:
            # Uses ADC on Cloud Run automatically
            firebase_admin.initialize_app()

        decoded = fb_auth.verify_id_token(id_token)
        return {"auth": "ok", "uid": decoded.get("uid")}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase idToken")


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "require_auth": REQUIRE_AUTH,
        "threshold": VERIFY_THRESHOLD,
    }


@app.post("/v1/verify/compare")
async def verify_compare(
    # AUTH
    authorization: Optional[str] = Header(default=None),

    # FORM DATA
    user_id: str = Form(...),
    selfie: UploadFile = File(...),
    photo_urls: str = Form(...),  # JSON string array
):
    """
    Compare selfie face with multiple profile photos.

    Request: multipart/form-data
      - user_id: string
      - selfie: file
      - photo_urls: JSON array string ["https://...", "https://..."]

    Response:
      - verified: bool
      - best_score: float
      - per_photo: [{url_hash, score, ok, reason?}]
    """
    _auth_check(authorization)

    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")

    # Parse urls
    try:
        urls = json.loads(photo_urls)
        if not isinstance(urls, list) or not all(isinstance(x, str) for x in urls):
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="photo_urls must be a JSON array of strings.")

    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="photo_urls is empty.")

    if len(urls) > MAX_PHOTOS:
        raise HTTPException(status_code=400, detail=f"Too many photo_urls. Max is {MAX_PHOTOS}.")

    if not ALLOW_URLS:
        raise HTTPException(status_code=400, detail="URL downloading disabled on this service.")

    # Read selfie
    selfie_bytes = await selfie.read()
    if not selfie_bytes:
        raise HTTPException(status_code=400, detail="Selfie file is empty.")

    selfie_bgr = _read_image_bytes_to_bgr(selfie_bytes)
    selfie_emb, selfie_meta = _detect_single_face_embedding(selfie_bgr)

    per_photo = []
    best = -1.0

    for url in urls:
        try:
            img_bytes = _download_url_to_bytes(url)
            bgr = _read_image_bytes_to_bgr(img_bytes)
            emb, meta = _detect_single_face_embedding(bgr)
            score = _cosine_similarity(selfie_emb, emb)
            best = max(best, score)
            per_photo.append({
                "url_hash": _short_hash(url),
                "score": round(score, 4),
                "ok": score >= VERIFY_THRESHOLD,
                "det_score": round(float(meta.get("det_score", 0.0)), 4),
            })
        except HTTPException as e:
            per_photo.append({
                "url_hash": _short_hash(url),
                "score": None,
                "ok": False,
                "reason": e.detail,
            })

    verified = best >= VERIFY_THRESHOLD

    return JSONResponse({
        "status": "ok",
        "verified": verified,
        "threshold": VERIFY_THRESHOLD,
        "best_score": round(float(best), 4) if best >= 0 else None,
        "user_id": user_id,
        "selfie": {
            "det_score": round(float(selfie_meta.get("det_score", 0.0)), 4),
            "has_face": True,
        },
        "per_photo": per_photo,
    })
