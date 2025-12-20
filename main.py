from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import logging
import os
import requests
from google.cloud import vision

app = FastAPI(title="ShaadiParrot Face Verification")

# ===== logger =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-verification")

# ===== Vision client =====
vision_client: vision.ImageAnnotatorClient | None = None


class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: HttpUrl  # Firebase Storage signed URL


@app.on_event("startup")
def startup_event():
    """
    Cloud Run best-practice:
    heavy init only here.
    """
    global vision_client
    try:
        vision_client = vision.ImageAnnotatorClient()
        logger.info("Google Vision client initialized")
    except Exception:
        vision_client = None
        logger.exception("Failed to init Google Vision client")


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "shaadiparrot-face-verification",
        "vision_ready": vision_client is not None
    }


def _download_image_bytes(url: str, max_mb: int = 10) -> bytes:
    # Firebase signed URL often uses redirects; allow them.
    headers = {
        "User-Agent": "shaadiparrot-face-verification/1.0"
    }
    r = requests.get(url, headers=headers, timeout=25, stream=True, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {r.status_code}")

    content_type = (r.headers.get("Content-Type") or "").lower()
    if not any(x in content_type for x in ["image/jpeg", "image/jpg", "image/png", "image/webp"]):
        # Some storages return octet-stream. We'll still allow if bytes look ok, but keep check мягким.
        logger.warning(f"Suspicious Content-Type: {content_type}")

    max_bytes = max_mb * 1024 * 1024
    data = b""
    for chunk in r.iter_content(chunk_size=1024 * 256):
        if not chunk:
            continue
        data += chunk
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Image too large (>{max_mb}MB)")

    if len(data) < 2000:
        raise HTTPException(status_code=400, detail="Downloaded file is too small / invalid")

    return data


def _face_area_ratio(face: vision.FaceAnnotation) -> float:
    # Bounding poly has 4 points.
    pts = face.bounding_poly.vertices
    xs = [p.x for p in pts if p.x is not None]
    ys = [p.y for p in pts if p.y is not None]
    if not xs or not ys:
        return 0.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(0, max_x - min_x)
    h = max(0, max_y - min_y)
    face_area = float(w * h)

    # Image size can be inferred from bounding box only poorly; Vision doesn't return image dims here.
    # We'll use a heuristic: require face box to be "not tiny" by raw size in px.
    # For typical phone photos, < 80x80 is almost always trash.
    if w < 80 or h < 80:
        return 0.0

    # Return proxy metric (not true ratio), used only as “big enough” check.
    return face_area


@app.post("/verify-face")
def verify_face(data: VerifyFaceRequest):
    if vision_client is None:
        raise HTTPException(status_code=503, detail="Vision client not available")

    # 1) download image bytes (signed URL)
    img_bytes = _download_image_bytes(str(data.image_url), max_mb=10)

    # 2) call Google Vision face detection
    image = vision.Image(content=img_bytes)

    try:
        response = vision_client.face_detection(image=image)
    except Exception as e:
        logger.exception("Vision API call failed")
        raise HTTPException(status_code=502, detail=f"Vision API error: {type(e).__name__}")

    if response.error and response.error.message:
        raise HTTPException(status_code=502, detail=f"Vision API error: {response.error.message}")

    faces = response.face_annotations or []
    face_count = len(faces)

    # ===== Rules (anti-fake / quality gate) =====
    # "Top-level dating" минимум:
    # - must be exactly 1 face
    # - decent detection confidence
    # - face box not tiny (likely too far/blurred)
    if face_count == 0:
        return {
            "status": "rejected",
            "reason": "no_face_detected",
            "user_id": data.user_id,
            "image_url": str(data.image_url),
            "faces": 0
        }

    if face_count > 1:
        return {
            "status": "rejected",
            "reason": "multiple_faces_detected",
            "user_id": data.user_id,
            "image_url": str(data.image_url),
            "faces": face_count
        }

    face = faces[0]
    det_conf = float(getattr(face, "detection_confidence", 0.0) or 0.0)
    lm_conf = float(getattr(face, "landmarking_confidence", 0.0) or 0.0)
    area_proxy = _face_area_ratio(face)

    # thresholds (можем потом подкрутить)
    if det_conf < 0.65:
        return {
            "status": "rejected",
            "reason": "low_detection_confidence",
            "user_id": data.user_id,
            "image_url": str(data.image_url),
            "faces": 1,
            "detection_confidence": det_conf
        }

    if lm_conf < 0.30:
        return {
            "status": "rejected",
            "reason": "low_landmark_confidence",
            "user_id": data.user_id,
            "image_url": str(data.image_url),
            "faces": 1,
            "landmarking_confidence": lm_conf
        }

    if area_proxy <= 0.0:
        return {
            "status": "rejected",
            "reason": "face_too_small_or_far",
            "user_id": data.user_id,
            "image_url": str(data.image_url),
            "faces": 1
        }

    # If passed:
    return {
        "status": "verified",
        "user_id": data.user_id,
        "image_url": str(data.image_url),
        "faces": 1,
        "detection_confidence": det_conf,
        "landmarking_confidence": lm_conf
    }
