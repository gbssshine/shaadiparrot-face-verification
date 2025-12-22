from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
import logging
import os
import requests
import re
import time
from typing import Optional, List, Literal, Dict, Any, Tuple

from google.cloud import vision

# ✅ verify Firebase idToken from MAUI
import firebase_admin
from firebase_admin import auth

# ✅ Firestore Admin client (Cloud Run service account)
from google.cloud import firestore

app = FastAPI(title="ShaadiParrot Cloud Run")

# ===== logger =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shaadiparrot-cloudrun")

# ===== Vision client =====
vision_client: vision.ImageAnnotatorClient | None = None

# ===== Firestore client =====
firestore_client: firestore.Client | None = None

# ===== DeepSeek config =====
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ===== Firebase Admin init =====
if not firebase_admin._apps:
    firebase_admin.initialize_app()

# =========================
# SIMPLE IN-MEMORY CACHES
# =========================
# uid -> (expires_at_epoch, profile_summary_text)
_profile_summary_cache: Dict[str, Tuple[float, str]] = {}

PROFILE_CACHE_TTL_SEC = 600  # 10 min


def _cache_get_profile_summary(uid: str) -> str:
    item = _profile_summary_cache.get(uid)
    if not item:
        return ""
    exp, txt = item
    if time.time() > exp:
        try:
            del _profile_summary_cache[uid]
        except Exception:
            pass
        return ""
    return txt


def _cache_set_profile_summary(uid: str, txt: str):
    _profile_summary_cache[uid] = (time.time() + PROFILE_CACHE_TTL_SEC, txt)


# =========================
# FACE VERIFICATION MODELS
# =========================
class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


# =========================
# AI CHAT MODELS
# =========================
Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


class AiChatRequest(BaseModel):
    text: str
    locale: Optional[str] = "en"
    mode: Optional[str] = "shaadi_parrot"
    history: Optional[List[ChatTurn]] = None


class AiChatResponse(BaseModel):
    reply_text: str
    blocked: bool = False
    reason: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global vision_client, firestore_client

    try:
        vision_client = vision.ImageAnnotatorClient()
        logger.info("Google Vision client initialized")
    except Exception:
        vision_client = None
        logger.exception("Failed to init Google Vision client")

    try:
        firestore_client = firestore.Client()
        logger.info("Firestore client initialized")
    except Exception:
        firestore_client = None
        logger.exception("Failed to init Firestore client")


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "shaadiparrot-cloudrun",
        "vision_ready": vision_client is not None,
        "firestore_ready": firestore_client is not None,
        "deepseek_ready": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL,
    }


# =========================
# Helpers: Face Verification
# =========================
def _download_image_bytes(url: str, max_mb: int = 10) -> bytes:
    headers = {"User-Agent": "shaadiparrot-face-verification/1.0"}
    r = requests.get(url, headers=headers, timeout=25, stream=True, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {r.status_code}")

    content_type = (r.headers.get("Content-Type") or "").lower()
    if not any(x in content_type for x in ["image/jpeg", "image/jpg", "image/png", "image/webp"]):
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


def _face_area_proxy(face: vision.FaceAnnotation) -> float:
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

    if w < 80 or h < 80:
        return 0.0

    return face_area


@app.post("/verify-face")
def verify_face(data: VerifyFaceRequest):
    if vision_client is None:
        raise HTTPException(status_code=503, detail="Vision client not available")

    img_bytes = _download_image_bytes(str(data.image_url), max_mb=10)
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

    if face_count == 0:
        return {"status": "rejected", "reason": "no_face_detected", "user_id": data.user_id, "faces": 0}

    if face_count > 1:
        return {"status": "rejected", "reason": "multiple_faces_detected", "user_id": data.user_id, "faces": face_count}

    face = faces[0]
    det_conf = float(getattr(face, "detection_confidence", 0.0) or 0.0)
    lm_conf = float(getattr(face, "landmarking_confidence", 0.0) or 0.0)
    area_proxy = _face_area_proxy(face)

    if det_conf < 0.65:
        return {"status": "rejected", "reason": "low_detection_confidence", "faces": 1, "detection_confidence": det_conf}

    if lm_conf < 0.30:
        return {"status": "rejected", "reason": "low_landmark_confidence", "faces": 1, "landmarking_confidence": lm_conf}

    if area_proxy <= 0.0:
        return {"status": "rejected", "reason": "face_too_small_or_far", "faces": 1}

    return {"status": "verified", "faces": 1, "detection_confidence": det_conf, "landmarking_confidence": lm_conf}


# =========================
# Helpers: Auth
# =========================
def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    authorization = authorization.strip()
    if not authorization.lower().startswith("bearer "):
        return ""
    return authorization[7:].strip()


def _verify_firebase_token_or_401(authorization: Optional[str]) -> str:
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    try:
        decoded = auth.verify_id_token(token)
        uid = (decoded.get("uid") or "").strip()
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token (no uid)")
        return uid
    except Exception:
        logger.exception("Firebase token verify failed")
        raise HTTPException(status_code=401, detail="Invalid/expired token")


def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# Topic gate
# =========================
def _is_allowed_topic(user_text: str) -> bool:
    t = (user_text or "").lower()
    allowed_keywords = [
        "bio", "profile", "photos", "photo", "pictures", "rewrite", "about me",
        "match", "matches", "message", "reply", "text her", "text him", "chat",
        "fate", "daily", "astrology", "kundli", "compatibility",
        "shaadi", "parrot", "app", "how it works", "premium", "subscription",
    ]
    return any(k in t for k in allowed_keywords)


def _build_system_prompt(locale: str) -> str:
    # ✅ shorter system prompt = fewer tokens
    return (
        "You are Shaadi Parrot inside a dating+fates app.\n"
        "Only help with: profile/bio/photos, matches & messaging, Daily Fates meaning, app usage.\n"
        "If off-topic: refuse briefly and redirect.\n"
        "Be concise and practical. 1 emoji max.\n"
        f"Reply in {locale or 'en'}.\n"
    )


# =========================
# Profile -> compact summary
# =========================
def _safe_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}

    deny_prefixes = ["geo", "lat", "lng", "location", "idtoken", "refreshtoken", "token", "__"]
    deny_exact = {"updatedAt", "createdAtIso", "geoCapturedAtUtc", "geoSource", "deviceId", "pushToken"}

    clean: Dict[str, Any] = {}
    for k, v in raw.items():
        key = (k or "").strip()
        if not key:
            continue
        lk = key.lower()

        if lk in (x.lower() for x in deny_exact):
            continue

        if any(lk.startswith(p) for p in deny_prefixes):
            continue

        clean[key] = v

    return clean


def _flatten_value(v: Any, max_len: int = 140) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return (s[:max_len] + "…") if len(s) > max_len else s
    if isinstance(v, list):
        parts = []
        for item in v[:12]:
            s = _flatten_value(item, max_len=40)
            if s:
                parts.append(s)
        out = ", ".join(parts)
        if len(v) > 12:
            out += "…"
        return (out[:max_len] + "…") if len(out) > max_len else out
    if isinstance(v, dict):
        # не раздуваемся
        parts = []
        for i, (kk, vv) in enumerate(v.items()):
            if i >= 8:
                parts.append("…")
                break
            s = _flatten_value(vv, max_len=40)
            if s:
                parts.append(f"{kk}:{s}")
        out = "; ".join(parts)
        return (out[:max_len] + "…") if len(out) > max_len else out

    s = str(v).strip()
    return (s[:max_len] + "…") if len(s) > max_len else s


def _profile_summary_text(profile_doc: Dict[str, Any]) -> str:
    """
    ✅ Супер-компактный summary = мало токенов.
    Тут только то, что реально помогает переписке/био/советам.
    """
    if not profile_doc:
        return ""

    preferred_keys = [
        "firstName", "lastName", "gender",
        "seekerType", "lookingFor", "seeking",
        "birthDate", "age",
        "cityName", "countryName", "community",
        "bio", "aboutMe",
        "interests", "languages",
        "drinking", "smoking", "workout", "diet",
        "education", "jobTitle", "occupation",
        "relationshipGoal", "relationshipIntent", "intent",
    ]

    parts: List[str] = []
    for k in preferred_keys:
        if k in profile_doc:
            val = _flatten_value(profile_doc.get(k))
            if val:
                parts.append(f"{k}={val}")

        if len(parts) >= 18:
            break

    if not parts:
        return ""

    # одна строка, коротко
    return "USER_PROFILE: " + " | ".join(parts)


async def _load_user_profile_summary(uid: str) -> str:
    """
    Берем profiles/{uid}, чистим, делаем summary и кладем в кэш.
    """
    cached = _cache_get_profile_summary(uid)
    if cached:
        return cached

    if firestore_client is None:
        return ""

    try:
        snap = firestore_client.collection("profiles").document(uid).get()
        if not snap.exists:
            return ""
        raw = snap.to_dict() or {}
        safe = _safe_profile_dict(raw)
        summary = _profile_summary_text(safe)
        if summary:
            _cache_set_profile_summary(uid, summary)
        return summary
    except Exception:
        logger.exception("Failed to load profiles/{uid}")
        return ""


# =========================
# DeepSeek call
# =========================
async def _call_deepseek(messages: List[Dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set in Cloud Run env vars")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 280,  # ✅ меньше output токенов
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    import json
    try:
        import httpx
        async with httpx.AsyncClient(timeout=35) as client:
            r = await client.post(DEEPSEEK_URL, headers=headers, content=json.dumps(payload))
            text = r.text
            if r.status_code != 200:
                logger.error(f"DeepSeek error {r.status_code}: {text}")
                raise HTTPException(status_code=502, detail=f"DeepSeek API error: HTTP {r.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("DeepSeek call failed")
        raise HTTPException(status_code=502, detail=f"DeepSeek call failed: {type(e).__name__}")

    try:
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return "I couldn't generate a reply. Try again."
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        return content or "I couldn't generate a reply. Try again."
    except Exception:
        logger.exception("DeepSeek response parse failed")
        return "I couldn't parse the reply. Try again."


# =========================
# Endpoint: AI Chat
# =========================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text)
    if not user_text:
        return AiChatResponse(reply_text="Ask me something 🙂")

    if not _is_allowed_topic(user_text):
        return AiChatResponse(
            reply_text=(
                "I can help only with:\n"
                "• improving your profile/bio/photos\n"
                "• match & messaging advice\n"
                "• Daily Fates meaning\n"
                "• how the app works 🦜\n\n"
                "Ask me about one of these."
            ),
            blocked=True,
            reason="topic_not_allowed",
        )

    locale = (body.locale or "en").strip() or "en"
    system_prompt = _build_system_prompt(locale)

    # ✅ profile summary (cached)
    profile_summary = await _load_user_profile_summary(uid)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # ✅ Add short profile summary (NOT the whole doc)
    if profile_summary:
        messages.append({"role": "system", "content": profile_summary})

    # ✅ include small history (already reduced on client, but keep guard here too)
    if body.history:
        turns = body.history[-6:]  # ✅ server guard
        for t in turns:
            role = t.role
            txt = _normalize_text(t.text)
            if not txt:
                continue
            if role not in ["user", "assistant"]:
                continue
            # extra trim for safety
            if len(txt) > 320:
                txt = txt[:320] + "…"
            messages.append({"role": role, "content": txt})

    messages.append({"role": "user", "content": user_text})

    reply = await _call_deepseek(messages)

    logger.info(f"[ai_chat] uid={uid} user_len={len(user_text)} has_profile={bool(profile_summary)}")
    return AiChatResponse(reply_text=reply, blocked=False)
