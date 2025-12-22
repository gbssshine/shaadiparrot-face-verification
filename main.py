from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
import logging
import os
import requests
import re
import time
from datetime import datetime, timezone
from typing import Optional, List, Literal, Dict, Any, Tuple

from google.cloud import vision

# Firebase Admin (verify idToken)
import firebase_admin
from firebase_admin import auth

# Firestore Admin client (Cloud Run service account)
from google.cloud import firestore

# Swiss Ephemeris (Vedic sidereal)
import swisseph as swe


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
# CONFIG
# =========================
CHAT_SUBCOLLECTION = "parrotChats"     # users/{uid}/parrotChats/{threadId}
DEFAULT_THREAD_ID = "default"

# how many turns we keep in memory
MAX_STORED_TURNS = 40  # 40 turns = 20 user+assistant pairs
# how many turns we send to LLM (token-friendly)
MAX_LLM_TURNS = 12     # last 12 turns is usually enough

PROFILE_CACHE_TTL_SEC = 600  # 10 min
_profile_summary_cache: Dict[str, Tuple[float, str]] = {}


def _utc_now():
    return datetime.now(timezone.utc)


def _utc_iso():
    return _utc_now().isoformat()


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
# MODELS
# =========================
class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


class AiChatRequest(BaseModel):
    text: str
    locale: Optional[str] = "en"
    mode: Optional[str] = "shaadi_parrot"
    thread_id: Optional[str] = DEFAULT_THREAD_ID
    history: Optional[List[ChatTurn]] = None  # client can still send, but server will persist


class AiChatResponse(BaseModel):
    reply_text: str
    blocked: bool = False
    reason: Optional[str] = None
    thread_id: Optional[str] = DEFAULT_THREAD_ID


class HistoryResponse(BaseModel):
    thread_id: str
    messages: List[ChatTurn]


class ResetResponse(BaseModel):
    thread_id: str
    ok: bool = True


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup_event():
    global vision_client, firestore_client

    try:
        swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)
        logger.info("Swiss Ephemeris sidereal mode set: Lahiri")
    except Exception:
        logger.exception("Failed to set Swiss Ephemeris sidereal mode")

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
        "sidereal": "Lahiri",
    }


# =========================
# FACE VERIFICATION
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

    if w < 80 or h < 80:
        return 0.0

    return float(w * h)


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
# AUTH HELPERS
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
# TOPIC ROUTER (human-friendly)
# =========================
_OFFTOPIC_BLOCK_PATTERNS = [
    r"\b(recipe|cook|cooking|pancake|pancakes|omelet|cake|bake|ingredients)\b",
    r"\b(hack|ddos|phish|malware|exploit|crack|keylogger)\b",
    r"\b(code|python|c\+\+|java|javascript|sql|api|compile|bug fix|debug)\b",
    r"\b(election|president|politics|war|propaganda)\b",
    r"\b(heroin|cocaine|meth|weed|drug deal|how to buy)\b",
]

_ALLOWED_HINTS_PATTERNS = [
    r"\b(girl|girls|guy|guys|dating|date|relationship|love|crush|romance)\b",
    r"\b(text|message|dm|reply|respond|what should i say|what to say|how to respond)\b",
    r"\b(ghost(ed|ing)|ignored|left on read|seen)\b",
    r"\b(match|matches|tinder|bumble|hinge)\b",
    r"\b(bio|profile|about me|photos?|pictures?)\b",
    r"\b(astrology|vedic|kundli|horoscope|zodiac|nakshatra|moon sign|sun sign|compatib)\b",
    r"\b(app|shaadi|parrot|premium|subscription|how it works)\b",
]


def _looks_offtopic(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(re.search(pat, t, re.IGNORECASE) for pat in _OFFTOPIC_BLOCK_PATTERNS)


def _looks_allowed(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(re.search(pat, t, re.IGNORECASE) for pat in _ALLOWED_HINTS_PATTERNS)


def _build_soft_redirect(locale: str) -> str:
    # nicer + still focused + allows emojis
    return (
        "I’m Shaadi Parrot 🦜✨\n"
        "I’m your guide to love, texting, profiles, and Vedic astrology.\n\n"
        "Try one of these:\n"
        "• “What should I text her next?” 💬\n"
        "• “Rewrite my bio from my profile” 📝\n"
        "• “Explain my Daily Fate” 🔮\n"
        "• “What’s my Moon sign / Nakshatra?” 🌙"
    )


# =========================
# SYSTEM PROMPT (PERSONA)
# =========================
def _build_system_prompt(locale: str) -> str:
    return (
        "You are Shaadi Parrot — a cheerful Indian astrologer + dating coach inside a dating & Daily Fates app.\n"
        "Mission: guide the user through love, relationships, mutual understanding, and astrology.\n"
        "You help with: texting/replies; dating strategy; interpreting signals; profile/bio/photo tips; Vedic astrology; compatibility; Daily Fates; app usage.\n"
        "Style: charming, clear, emotionally intelligent, confident. Use 1–3 emojis naturally.\n"
        "If user is vague ('help me with girls'): ask ONE clarifying question and give 2–3 ready options.\n"
        "You may receive USER_PROFILE and ASTRO_COMPUTED. Use them automatically.\n"
        "If exact chart needs birthplace/timezone: say so briefly, still give best approximate answer.\n"
        "Do NOT help with cooking/recipes, coding, politics, hacking, drugs, or illegal activity.\n"
        f"Reply in {locale or 'en'}.\n"
    )


# =========================
# PROFILE SANITIZE + SUMMARY
# =========================
def _safe_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}

    deny_prefixes = ["geo", "lat", "lng", "location", "idtoken", "refreshtoken", "token", "__"]
    deny_exact = {"updatedAt", "deviceId", "pushToken"}

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
    if not profile_doc:
        return ""

    preferred_keys = [
        "firstName", "lastName", "name", "gender",
        "seekerType", "lookingFor", "seeking", "orientation", "lookingForGender",
        "birthDate", "birthTime", "age",
        "cityName", "countryName", "stateName", "community", "religion",
        "relationshipIntent", "relationshipGoal", "intent",
        "bio", "aboutMe",
        "interests", "languages", "tags",
        "drinking", "smoking", "workout", "diet", "pets", "social",
        "education", "jobTitle", "occupation",
        "height", "heightDisplayText",
    ]

    parts: List[str] = []
    used = set()

    for k in preferred_keys:
        if k in profile_doc:
            val = _flatten_value(profile_doc.get(k))
            if val:
                parts.append(f"{k}={val}")
                used.add(k)
        if len(parts) >= 22:
            break

    extra_keys = [k for k in profile_doc.keys() if k not in used]
    extra_keys = extra_keys[:30]

    out = ""
    if parts:
        out += "USER_PROFILE: " + " | ".join(parts)
    if extra_keys:
        out += "\nPROFILE_KEYS_AVAILABLE: " + ", ".join(extra_keys)
    return out.strip()


async def _load_user_profile(uid: str) -> Dict[str, Any]:
    if firestore_client is None:
        return {}
    try:
        snap = firestore_client.collection("profiles").document(uid).get()
        if not snap.exists:
            return {}
        raw = snap.to_dict() or {}
        return _safe_profile_dict(raw)
    except Exception:
        logger.exception("Failed to load profiles/{uid}")
        return {}


async def _load_user_profile_summary(uid: str) -> str:
    cached = _cache_get_profile_summary(uid)
    if cached:
        return cached

    profile = await _load_user_profile(uid)
    summary = _profile_summary_text(profile)
    if summary:
        _cache_set_profile_summary(uid, summary)
    return summary


# =========================
# ASTROLOGY (VEDIC SIDEREAL LAHIRI)
# =========================
_ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

_NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashirsha","Ardra","Punarvasu","Pushya","Ashlesha",
    "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha","Purva Bhadrapada","Uttara Bhadrapada","Revati"
]


def _parse_birth_date(profile: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    raw = profile.get("birthDate")
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", s)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _parse_birth_time(profile: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    raw = profile.get("birthTime")
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        m = re.match(r"^\s*(\d{1,2}):(\d{2})", s)
        if not m:
            return None
        hh, mm = int(m.group(1)), int(m.group(2))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return (hh, mm)
    return None


def _sign_from_sidereal_longitude(lon_deg: float) -> str:
    idx = int((lon_deg % 360.0) / 30.0)
    idx = max(0, min(11, idx))
    return _ZODIAC[idx]


def _nakshatra_from_sidereal_longitude(lon_deg: float) -> str:
    seg = 360.0 / 27.0
    idx = int((lon_deg % 360.0) / seg)
    idx = max(0, min(26, idx))
    return _NAKSHATRAS[idx]


def _calc_sidereal_lon_ut(jd_ut: float, planet: int) -> float:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    res, _ = swe.calc_ut(jd_ut, planet, flags)
    return float(res[0]) % 360.0


def _compute_astro(profile: Dict[str, Any]) -> str:
    bd = _parse_birth_date(profile)
    if not bd:
        return ""

    y, mo, d = bd
    bt = _parse_birth_time(profile)

    if bt:
        hh, mm = bt
        hour = hh + (mm / 60.0)
        time_mode = "birthTime_provided_timezone_unknown"
        note = "Exact ascendant/houses need birthPlace + timezone."
    else:
        hour = 12.0
        time_mode = "date_only"
        note = "Moon/Nakshatra accuracy improves with birthTime + birthPlace/timezone."

    jd_ut = swe.julday(y, mo, d, hour)

    try:
        sun_lon = _calc_sidereal_lon_ut(jd_ut, swe.SUN)
        moon_lon = _calc_sidereal_lon_ut(jd_ut, swe.MOON)

        sun_sign = _sign_from_sidereal_longitude(sun_lon)
        moon_sign = _sign_from_sidereal_longitude(moon_lon)
        nak = _nakshatra_from_sidereal_longitude(moon_lon)

        return (
            "ASTRO_COMPUTED (Vedic sidereal Lahiri): "
            f"SunSign={sun_sign}; MoonSign={moon_sign}; Nakshatra={nak}; "
            f"TimeMode={time_mode}. {note}"
        )
    except Exception:
        logger.exception("Astro compute failed")
        return ""


# =========================
# CHAT STORAGE (Firestore)
# =========================
def _thread_id_clean(thread_id: Optional[str]) -> str:
    x = (thread_id or DEFAULT_THREAD_ID).strip()
    if not x:
        x = DEFAULT_THREAD_ID
    # Firestore doc id safe-ish
    x = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", x)
    return x[:80]


def _chat_doc_ref(uid: str, thread_id: str):
    if firestore_client is None:
        return None
    tid = _thread_id_clean(thread_id)
    return firestore_client.collection("users").document(uid).collection(CHAT_SUBCOLLECTION).document(tid)


def _to_turns(raw_messages: Any) -> List[ChatTurn]:
    out: List[ChatTurn] = []
    if not isinstance(raw_messages, list):
        return out
    for m in raw_messages:
        try:
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            if role not in ["user", "assistant"]:
                continue
            if not text:
                continue
            out.append(ChatTurn(role=role, text=text))
        except Exception:
            continue
    return out


def _from_turns(turns: List[ChatTurn]) -> List[Dict[str, Any]]:
    now_iso = _utc_iso()
    result: List[Dict[str, Any]] = []
    for t in turns:
        txt = _normalize_text(t.text)
        if not txt:
            continue
        result.append({"role": t.role, "text": txt, "ts": now_iso})
    return result


async def _load_thread_history(uid: str, thread_id: str) -> List[ChatTurn]:
    ref = _chat_doc_ref(uid, thread_id)
    if ref is None:
        return []
    try:
        snap = ref.get()
        if not snap.exists:
            return []
        data = snap.to_dict() or {}
        return _to_turns(data.get("messages"))
    except Exception:
        logger.exception("Failed to load chat history")
        return []


async def _save_thread_history(uid: str, thread_id: str, turns: List[ChatTurn]) -> None:
    ref = _chat_doc_ref(uid, thread_id)
    if ref is None:
        return
    try:
        # keep only last MAX_STORED_TURNS
        trimmed = turns[-MAX_STORED_TURNS:]
        payload = {
            "uid": uid,
            "threadId": _thread_id_clean(thread_id),
            "messages": _from_turns(trimmed),
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "updatedAtIso": _utc_iso(),
        }
        ref.set(payload, merge=True)
    except Exception:
        logger.exception("Failed to save chat history")


# =========================
# DEEPSEEK CALL
# =========================
async def _call_deepseek(messages: List[Dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set in Cloud Run env vars")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 420,
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
# API: HISTORY + RESET
# =========================
@app.get("/ai/history", response_model=HistoryResponse)
async def get_history(thread_id: Optional[str] = DEFAULT_THREAD_ID, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = _thread_id_clean(thread_id)
    turns = await _load_thread_history(uid, tid)
    return HistoryResponse(thread_id=tid, messages=turns)


@app.post("/ai/reset", response_model=ResetResponse)
async def reset_history(thread_id: Optional[str] = DEFAULT_THREAD_ID, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = _thread_id_clean(thread_id)
    ref = _chat_doc_ref(uid, tid)
    if ref is not None:
        try:
            ref.set(
                {
                    "uid": uid,
                    "threadId": tid,
                    "messages": [],
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "updatedAtIso": _utc_iso(),
                },
                merge=True,
            )
        except Exception:
            logger.exception("Failed to reset chat")
    return ResetResponse(thread_id=tid, ok=True)


# =========================
# AI CHAT ENDPOINT (with persistence)
# =========================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text)
    if not user_text:
        return AiChatResponse(reply_text="Say hi, tell me what’s happening in your love story 🦜💛", blocked=False)

    locale = (body.locale or "en").strip() or "en"
    thread_id = _thread_id_clean(body.thread_id)

    # hard off-topic redirect
    if _looks_offtopic(user_text):
        return AiChatResponse(reply_text=_build_soft_redirect(locale), blocked=False, reason="off_topic_redirect", thread_id=thread_id)

    # soft redirect if not clearly allowed
    if not _looks_allowed(user_text):
        return AiChatResponse(reply_text=_build_soft_redirect(locale), blocked=False, reason="needs_redirect", thread_id=thread_id)

    # ===== Load server history (source of truth) =====
    stored_turns = await _load_thread_history(uid, thread_id)

    # Optional: if client sends history, we can merge it lightly to avoid “lost” messages in edge cases.
    # We will append any client messages that are not already at the end (simple heuristic).
    if body.history:
        client_turns = [ChatTurn(role=t.role, text=_normalize_text(t.text)) for t in body.history if t and _normalize_text(t.text)]
        if client_turns:
            # merge by last 6 items string match to reduce duplicates
            stored_tail = [(t.role, _normalize_text(t.text)) for t in stored_turns[-12:]]
            for ct in client_turns[-12:]:
                key = (ct.role, _normalize_text(ct.text))
                if key not in stored_tail:
                    stored_turns.append(ct)

    # add current user message
    stored_turns.append(ChatTurn(role="user", text=user_text))

    # ===== Profile + Astro =====
    profile_summary = await _load_user_profile_summary(uid)
    profile = await _load_user_profile(uid)
    astro_computed = _compute_astro(profile)

    # ===== Build LLM messages =====
    system_prompt = _build_system_prompt(locale)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if profile_summary:
        messages.append({"role": "system", "content": profile_summary})
    if astro_computed:
        messages.append({"role": "system", "content": astro_computed})

    # include last MAX_LLM_TURNS from stored history
    llm_turns = stored_turns[-MAX_LLM_TURNS:]
    for t in llm_turns:
        txt = _normalize_text(t.text)
        if not txt:
            continue
        messages.append({"role": t.role, "content": txt})

    # ===== Call model =====
    reply = await _call_deepseek(messages)

    # save assistant response
    stored_turns.append(ChatTurn(role="assistant", text=_normalize_text(reply)))

    # persist
    await _save_thread_history(uid, thread_id, stored_turns)

    logger.info(
        f"[ai_chat] uid={uid} thread={thread_id} user_len={len(user_text)} stored_turns={len(stored_turns)} "
        f"astro={'yes' if bool(astro_computed) else 'no'}"
    )

    return AiChatResponse(reply_text=reply, blocked=False, thread_id=thread_id)
