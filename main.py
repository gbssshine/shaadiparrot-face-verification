from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import logging
import os
import re
import time
import hashlib
from typing import Optional, List, Literal, Dict, Any, Tuple

import requests
import swisseph as swe

from google.cloud import vision
import firebase_admin
from firebase_admin import auth
from google.cloud import firestore

# =========================
# APP
# =========================
app = FastAPI(title="ShaadiParrot Cloud Run")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shaadiparrot-cloudrun")

vision_client: vision.ImageAnnotatorClient | None = None
firestore_client: firestore.Client | None = None

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# token economy knobs
DS_TEMPERATURE = float(os.getenv("DS_TEMPERATURE") or "0.65")
DS_MAX_TOKENS_DEFAULT = int(os.getenv("DS_MAX_TOKENS_DEFAULT") or "260")
DS_TIMEOUT_SEC = int(os.getenv("DS_TIMEOUT_SEC") or "45")

# prompt economy knobs
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT") or "8")              # last N msgs only
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS") or "240")    # truncate each msg
USER_TEXT_MAX_CHARS = int(os.getenv("USER_TEXT_MAX_CHARS") or "700")

# summary memory knobs
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS") or "650")
SUMMARY_UPDATE_EVERY_TURNS = int(os.getenv("SUMMARY_UPDATE_EVERY_TURNS") or "6")

if not firebase_admin._apps:
    firebase_admin.initialize_app()

# =========================
# MODELS
# =========================
class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


class VerifyPhotoRequest(BaseModel):
    gcs_uri: str
    require_face: bool = True


Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str


class AiChatRequest(BaseModel):
    text: str
    locale: Optional[str] = "en"
    mode: Optional[str] = "shaadi_parrot"
    thread_id: Optional[str] = "default"
    history: Optional[List[ChatTurn]] = None  # IGNORED for token economy


class AiChatResponse(BaseModel):
    reply_text: str
    blocked: bool = False
    reason: Optional[str] = None
    thread_id: Optional[str] = None


class HistoryResponse(BaseModel):
    thread_id: str = "default"
    messages: List[ChatTurn] = []


class ResetResponse(BaseModel):
    thread_id: str = "default"
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
        "economy": {
            "history_limit": HISTORY_LIMIT,
            "history_max_chars": HISTORY_MAX_CHARS,
            "max_tokens_default": DS_MAX_TOKENS_DEFAULT,
            "temperature": DS_TEMPERATURE,
        }
    }


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


def _stable_pick(variants: List[str], key: str) -> str:
    if not variants:
        return ""
    h = hashlib.sha256((key or "").encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(variants)
    return variants[idx]


# =========================
# TOPIC GATE (SOFT)
# =========================
def _is_allowed_topic(user_text: str) -> bool:
    t = (user_text or "").lower()
    blocked_keywords = [
        "recipe", "cook", "cooking", "pancake", "omelet", "baking", "cake",
        "how to fry", "ingredients", "gram", "ml", "kefir", "flour", "sugar",
        "рецепт", "готовить", "омлет", "блин", "мука", "ингредиент",
        "готовка", "кулинар", "сколько яиц", "жарить"
    ]
    if any(k in t for k in blocked_keywords):
        return False
    return True


def _topic_block_reply(user_text: str, locale: str) -> str:
    variants_en = [
        "I can’t help with recipes 🦜 But I can help with: dating, texting, profile glow-up, Vedic daily fates. Ask me one of those 🙂",
        "Not a cooking parrot 😅 I’m best at: love, texting, profile, Vedic astrology. What do you want today? 🦜",
    ]
    variants_ru = [
        "С рецептами не помогу 🦜 Но могу: свидания/отношения, тексты сообщений, улучшение профиля, ведический daily fate. Спроси 🙂",
        "Я не кулинарный попугай 😅 Зато топ в: отношениях, сообщениях, профиле, астрологии. Что делаем? 🦜",
    ]
    lang = (locale or "en").strip().lower()
    pool = variants_ru if lang.startswith("ru") else variants_en
    return _stable_pick(pool, user_text)


# =========================
# INTENT (cheap)
# =========================
class Intent:
    ASTRO = "astro"
    TEXTING = "texting"
    PROFILE = "profile"
    RELATION = "relation"
    GENERAL = "general"


def _infer_intent(user_text: str) -> str:
    t = (user_text or "").lower()

    astro_words = [
        "horoscope", "forecast", "today", "daily fate", "vedic", "kundli", "nakshatra", "rashi",
        "гороскоп", "прогноз", "сегодня", "ведичес", "накшатр", "раши", "кундли", "совместим"
    ]
    if any(w in t for w in astro_words):
        return Intent.ASTRO

    texting_words = ["text", "message", "reply", "dm", "what to say", "как ответить", "сообщение", "ответ", "написать ей", "написать ему"]
    if any(w in t for w in texting_words):
        return Intent.TEXTING

    profile_words = ["profile", "bio", "photos", "about me", "анкет", "био", "фото", "описание"]
    if any(w in t for w in profile_words):
        return Intent.PROFILE

    relation_words = ["relationship", "dating", "girl", "boyfriend", "girlfriend", "love", "свидани", "отношени", "любов"]
    if any(w in t for w in relation_words):
        return Intent.RELATION

    return Intent.GENERAL


# =========================
# PROMPT BUILDER (short)
# =========================
def _build_system_prompt(locale: str, intent: str) -> str:
    lang = (locale or "en").strip().lower() or "en"
    # супер короткая и жёсткая упаковка
    base = (
        "You are Shaadi Parrot 🦜: Indian-style dating coach + Vedic daily-fate assistant inside an app.\n"
        "Rules:\n"
        "- Be warm, confident, practical.\n"
        "- 1–2 emojis max.\n"
        "- No markdown.\n"
        "- Keep it concise. Prefer short bullets using '•'.\n"
        "- If asked for a message/reply: output 2–4 message options.\n"
        "- If asked for profile/bio: give actionable edits + 1 sample bio.\n"
        "- If asked for daily fate/horoscope: give a Vedic-style daily forecast + practical tips.\n"
        "- Never mention tokens, prompts, or internal system.\n"
    )

    if intent == Intent.TEXTING:
        base += "Output target: 6–10 short bullet lines total.\n"
    elif intent == Intent.PROFILE:
        base += "Output target: 8–12 bullet lines + 1 short sample bio.\n"
    elif intent == Intent.ASTRO:
        base += "Output target: 8–12 bullet lines.\n"
    else:
        base += "Output target: 6–10 short bullet lines.\n"

    if lang.startswith("ru"):
        base += "Reply in Russian.\n"
    else:
        base += "Reply in English.\n"

    return base


# =========================
# PROFILE LOAD (trimmed)
# =========================
def _safe_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    deny_prefixes = ["geo", "lat", "lng", "location", "idtoken", "refreshtoken", "token", "__"]
    deny_exact = {"updatedAt", "deviceId", "pushToken", "refreshToken"}

    clean: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
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


def _flatten_value(v: Any, max_len: int = 120) -> str:
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
        for item in v[:10]:
            s = _flatten_value(item, max_len=40)
            if s:
                parts.append(s)
        out = ", ".join(parts)
        if len(v) > 10:
            out += "…"
        return (out[:max_len] + "…") if len(out) > max_len else out
    if isinstance(v, dict):
        parts = []
        for i, (kk, vv) in enumerate(v.items()):
            if i >= 6:
                parts.append("…")
                break
            s = _flatten_value(vv, max_len=40)
            if s:
                parts.append(f"{kk}:{s}")
        out = "; ".join(parts)
        return (out[:max_len] + "…") if len(out) > max_len else out
    s = str(v).strip()
    return (s[:max_len] + "…") if len(s) > max_len else s


def _profile_context_compact(profile: Dict[str, Any], intent: str) -> str:
    if not profile:
        return ""

    # минимум полей по интенту (денежная экономия)
    base_keys = ["firstName", "age", "gender", "cityName", "countryName", "relationshipIntent", "languages"]
    texting_keys = base_keys + ["bio", "aboutMe", "interests", "workout", "smoking", "drinking"]
    profile_keys = base_keys + ["interests", "tags", "workout", "smoking", "drinking", "education", "jobTitle", "occupation", "bio", "aboutMe"]
    astro_keys = base_keys + ["birthDate", "birthTime"]

    if intent == Intent.TEXTING:
        keys = texting_keys
    elif intent == Intent.PROFILE:
        keys = profile_keys
    elif intent == Intent.ASTRO:
        keys = astro_keys
    else:
        keys = base_keys + ["interests"]

    parts: List[str] = []
    for k in keys:
        if k in profile:
            val = _flatten_value(profile.get(k))
            if val:
                parts.append(f"{k}={val}")
        if len(parts) >= 14:
            break

    if not parts:
        return ""
    return "USER_CONTEXT: " + " | ".join(parts)


# =========================
# ASTRO (ULTRA SHORT)
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


def _sign_from_lon(lon_deg: float) -> str:
    idx = int((lon_deg % 360.0) / 30.0)
    idx = max(0, min(11, idx))
    return _ZODIAC[idx]


def _nakshatra_from_lon(lon_deg: float) -> str:
    seg = 360.0 / 27.0
    idx = int((lon_deg % 360.0) / seg)
    idx = max(0, min(26, idx))
    return _NAKSHATRAS[idx]


def _calc_sidereal_lon_ut(jd_ut: float, planet: int) -> float:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    res, _ = swe.calc_ut(jd_ut, planet, flags)
    return float(res[0]) % 360.0


def _compute_astro_short(profile: Dict[str, Any]) -> str:
    bd = _parse_birth_date(profile)
    if not bd:
        return ""

    y, mo, d = bd
    # без таймзоны/места — берём noon UTC (стабильно и дешево)
    jd_ut = swe.julday(y, mo, d, 12.0)

    try:
        sun_lon = _calc_sidereal_lon_ut(jd_ut, swe.SUN)
        moon_lon = _calc_sidereal_lon_ut(jd_ut, swe.MOON)
        sun_sign = _sign_from_lon(sun_lon)
        moon_sign = _sign_from_lon(moon_lon)
        nak = _nakshatra_from_lon(moon_lon)
        return f"ASTRO: Sun={sun_sign}; Moon(Rashi)={moon_sign}; Nakshatra={nak}."
    except Exception:
        logger.exception("Astro compute failed")
        return ""


# =========================
# FIRESTORE CHAT STORAGE
# =========================
def _safe_thread_id(thread_id: str) -> str:
    tid = (thread_id or "default").strip()
    if not tid:
        tid = "default"
    if tid == "default":
        return "default"
    h = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:16]
    return f"t_{h}"


def _chat_doc_id(uid: str, thread_id: str) -> str:
    tid = _safe_thread_id(thread_id)
    if tid == "default":
        return uid
    return f"{uid}__{tid}"


def _chat_doc_ref(uid: str, thread_id: str):
    if firestore_client is None:
        return None
    return firestore_client.collection("parrotChats").document(_chat_doc_id(uid, thread_id))


def _chat_msgs_col_ref(uid: str, thread_id: str):
    if firestore_client is None:
        return None
    return _chat_doc_ref(uid, thread_id).collection("messages")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _load_chat_state(uid: str, thread_id: str) -> Dict[str, Any]:
    ref = _chat_doc_ref(uid, thread_id)
    if ref is None:
        return {}
    try:
        snap = ref.get()
        if not snap.exists:
            return {}
        return snap.to_dict() or {}
    except Exception:
        logger.exception("Failed to load parrotChats state")
        return {}


def _save_chat_state(uid: str, thread_id: str, patch: Dict[str, Any]) -> None:
    ref = _chat_doc_ref(uid, thread_id)
    if ref is None:
        return
    try:
        ref.set(patch, merge=True)
    except Exception:
        logger.exception("Failed to save parrotChats state")


def _save_chat_message_batch(
    uid: str,
    thread_id: str,
    user_text: str,
    assistant_text: str,
    created_at_iso: str,
    created_at_ms: int,
) -> None:
    col = _chat_msgs_col_ref(uid, thread_id)
    docref = _chat_doc_ref(uid, thread_id)
    if col is None or docref is None or firestore_client is None:
        return

    try:
        batch = firestore_client.batch()

        user_hash = hashlib.md5((user_text or "").encode("utf-8")).hexdigest()[:8]
        asst_hash = hashlib.md5((assistant_text or "").encode("utf-8")).hexdigest()[:8]

        user_id = f"{created_at_ms}_u_{user_hash}"
        asst_id = f"{created_at_ms + 2}_a_{asst_hash}"

        batch.set(col.document(user_id), {
            "role": "user",
            "text": user_text,
            "createdAtIso": created_at_iso,
            "createdAtMs": created_at_ms,
        })

        batch.set(col.document(asst_id), {
            "role": "assistant",
            "text": assistant_text,
            "createdAtIso": created_at_iso,
            "createdAtMs": created_at_ms + 2,
        })

        batch.set(docref, {
            "uid": uid,
            "threadId": (thread_id or "default"),
            "updatedAtIso": created_at_iso,
            "updatedAtMs": created_at_ms,
        }, merge=True)

        batch.commit()
    except Exception:
        logger.exception("Failed to batch save chat messages")


def _load_chat_history(uid: str, thread_id: str, limit: int = 24) -> List[Dict[str, str]]:
    col = _chat_msgs_col_ref(uid, thread_id)
    if col is None:
        return []

    state = _load_chat_state(uid, thread_id)
    cleared_ms = state.get("clearedAtMs")
    try:
        cleared_ms = int(cleared_ms) if cleared_ms is not None else 0
    except Exception:
        cleared_ms = 0

    try:
        q = col.order_by("createdAtMs", direction=firestore.Query.DESCENDING).limit(limit)
        snaps = list(q.stream())
        rows = []
        for s in snaps:
            d = s.to_dict() or {}
            role = (d.get("role") or "").strip()
            text = (d.get("text") or "").strip()
            ms = d.get("createdAtMs")
            try:
                ms = int(ms) if ms is not None else 0
            except Exception:
                ms = 0

            if not role or not text:
                continue
            if cleared_ms and ms and ms <= cleared_ms:
                continue

            rows.append({"role": role, "content": text})

        rows.reverse()
        return rows
    except Exception:
        logger.exception("Failed to load chat history")
        return []


# =========================
# SUMMARY MEMORY (cheap, no LLM)
# =========================
def _get_summary(state: Dict[str, Any]) -> str:
    s = state.get("summary")
    if isinstance(s, str):
        return s.strip()
    return ""


def _append_summary(summary: str, key: str, value: str) -> str:
    summary = (summary or "").strip()
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return summary

    # replace if exists
    pattern = re.compile(rf"(?i)\b{re.escape(key)}\s*=\s*[^|]+")
    if pattern.search(summary):
        summary = pattern.sub(f"{key}={value}", summary)
    else:
        if summary:
            summary += " | "
        summary += f"{key}={value}"

    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[-SUMMARY_MAX_CHARS:]
    return summary


def _update_summary_from_turn(summary: str, user_text: str, assistant_text: str, intent: str) -> str:
    t = (user_text or "").strip()

    # lightweight “facts” extraction
    # partner name
    m = re.search(r"\b(?:her name is|his name is|my gf is|my bf is)\s+([A-Z][a-z]+)\b", t, re.IGNORECASE)
    if m:
        summary = _append_summary(summary, "partnerName", m.group(1))

    # zodiac claims
    mz = re.search(r"\b(i'?m|i am)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz:
        summary = _append_summary(summary, "userZodiac", mz.group(2).title())

    mz2 = re.search(r"\b(she'?s|she is|he'?s|he is)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz2:
        summary = _append_summary(summary, "partnerZodiac", mz2.group(2).title())

    # last topic
    summary = _append_summary(summary, "lastTopic", intent)

    return summary


# =========================
# PHOTO MODERATION
# =========================
_LIKELIHOOD = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 1,
    "UNLIKELY": 2,
    "POSSIBLE": 3,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}

@app.post("/verify-photo")
def verify_photo(body: VerifyPhotoRequest, authorization: Optional[str] = Header(default=None)):
    _ = _verify_firebase_token_or_401(authorization)

    if vision_client is None:
        raise HTTPException(status_code=503, detail="Vision client not available")

    gcs_uri = (body.gcs_uri or "").strip()
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="gcs_uri must start with gs://")

    try:
        image = vision.Image(source=vision.ImageSource(gcs_image_uri=gcs_uri))
        resp = vision_client.annotate_image({
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {"type_": vision.Feature.Type.FACE_DETECTION},
            ],
        })
    except Exception as e:
        logger.exception("Vision annotate_image failed")
        raise HTTPException(status_code=502, detail=f"Vision API error: {type(e).__name__}")

    if resp.error and resp.error.message:
        raise HTTPException(status_code=502, detail=f"Vision API error: {resp.error.message}")

    ss = resp.safe_search_annotation
    adult = (ss.adult.name if ss and ss.adult else "UNKNOWN")
    racy = (ss.racy.name if ss and ss.racy else "UNKNOWN")
    violence = (ss.violence.name if ss and ss.violence else "UNKNOWN")

    faces = len(resp.face_annotations or [])

    if _LIKELIHOOD.get(adult, 0) >= _LIKELIHOOD["LIKELY"]:
        return {"ok": False, "reason": "adult_content", "adult": adult, "racy": racy, "violence": violence, "faces": faces}

    if _LIKELIHOOD.get(racy, 0) >= _LIKELIHOOD["VERY_LIKELY"]:
        return {"ok": False, "reason": "highly_racy", "adult": adult, "racy": racy, "violence": violence, "faces": faces}

    if _LIKELIHOOD.get(violence, 0) >= _LIKELIHOOD["VERY_LIKELY"]:
        return {"ok": False, "reason": "high_violence", "adult": adult, "racy": racy, "violence": violence, "faces": faces}

    if bool(body.require_face) and faces == 0:
        return {"ok": False, "reason": "no_face_detected", "adult": adult, "racy": racy, "violence": violence, "faces": faces}

    return {"ok": True, "reason": "ok", "adult": adult, "racy": racy, "violence": violence, "faces": faces}


# =========================
# FACE VERIFICATION
# =========================
def _download_image_bytes(url: str, max_mb: int = 10) -> bytes:
    headers = {"User-Agent": "shaadiparrot-face-verification/1.0"}
    r = requests.get(url, headers=headers, timeout=25, stream=True, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {r.status_code}")

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
# HISTORY/RESET
# =========================
@app.get("/ai/history", response_model=HistoryResponse)
def ai_history(thread_id: str = "default", limit: int = 24, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    limit = max(1, min(80, int(limit)))
    rows = _load_chat_history(uid, thread_id, limit=limit)
    msgs: List[ChatTurn] = []
    for r in rows:
        role = r.get("role") or ""
        txt = r.get("content") or ""
        if role in ["user", "assistant"] and txt:
            msgs.append(ChatTurn(role=role, text=txt))
    return HistoryResponse(thread_id=(thread_id or "default"), messages=msgs)


@app.post("/ai/reset", response_model=ResetResponse)
def ai_reset(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_state(uid, thread_id, {
        "uid": uid,
        "threadId": (thread_id or "default"),
        "clearedAtIso": now_iso,
        "clearedAtMs": now_ms,
        "updatedAtIso": now_iso,
        "updatedAtMs": now_ms,
        "turnCount": 0,
        "summary": "",
    })
    return ResetResponse(thread_id=(thread_id or "default"), ok=True)


# Backward-compatible old endpoints
@app.get("/parrot/history")
def parrot_history(limit: int = 24, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    limit = max(1, min(80, int(limit)))
    items = _load_chat_history(uid, "default", limit=limit)
    return {"uid": uid, "count": len(items), "items": items}


@app.post("/parrot/reset")
def parrot_reset(authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_state(uid, "default", {
        "uid": uid,
        "threadId": "default",
        "clearedAtIso": now_iso,
        "clearedAtMs": now_ms,
        "updatedAtIso": now_iso,
        "updatedAtMs": now_ms,
        "turnCount": 0,
        "summary": "",
    })
    return {"status": "ok", "uid": uid, "clearedAtIso": now_iso}


# =========================
# DEEPSEEK CALL
# =========================
async def _call_deepseek(messages: List[Dict[str, str]], max_tokens: int) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set in Cloud Run env vars")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": DS_TEMPERATURE,
        "max_tokens": int(max(120, min(420, max_tokens))),
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    import json
    try:
        import httpx
        async with httpx.AsyncClient(timeout=DS_TIMEOUT_SEC) as client:
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
            return "I couldn't generate a reply. Please try again."
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        return content or "I couldn't generate a reply. Please try again."
    except Exception:
        logger.exception("DeepSeek response parse failed")
        return "I couldn't parse the reply. Please try again."


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _build_messages_for_llm(
    locale: str,
    intent: str,
    user_context: str,
    astro_short: str,
    summary: str,
    history_rows: List[Dict[str, str]],
    user_text: str,
) -> List[Dict[str, str]]:
    system_prompt = _build_system_prompt(locale, intent)

    # 1 system + 1 context (максимум 2 system)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # компактный контекст одной строкой
    ctx_bits: List[str] = []
    if user_context:
        ctx_bits.append(user_context)
    if intent == Intent.ASTRO and astro_short:
        ctx_bits.append(astro_short)
    if summary:
        ctx_bits.append("THREAD_SUMMARY: " + _truncate(summary, SUMMARY_MAX_CHARS))

    if ctx_bits:
        messages.append({"role": "system", "content": " | ".join(ctx_bits)})

    # история — только последние N сообщений, короткие
    if history_rows:
        trimmed = history_rows[-HISTORY_LIMIT:]
        for m in trimmed:
            role = m.get("role") or ""
            content = _truncate(m.get("content") or "", HISTORY_MAX_CHARS)
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    # текущее сообщение
    messages.append({"role": "user", "content": _truncate(user_text, USER_TEXT_MAX_CHARS)})

    return messages


def _pick_max_tokens(intent: str) -> int:
    # можно дальше резать, но так сохраняется качество
    if intent == Intent.TEXTING:
        return 240
    if intent == Intent.PROFILE:
        return 300
    if intent == Intent.ASTRO:
        return 300
    return DS_MAX_TOKENS_DEFAULT


# =========================
# AI CHAT
# =========================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text)
    thread_id = (body.thread_id or "default").strip() or "default"
    locale = (body.locale or "en").strip() or "en"

    if not user_text:
        return AiChatResponse(reply_text="Ask me something 🦜", thread_id=thread_id)

    if len(user_text) > USER_TEXT_MAX_CHARS:
        user_text = _truncate(user_text, USER_TEXT_MAX_CHARS)

    if not _is_allowed_topic(user_text):
        return AiChatResponse(
            reply_text=_topic_block_reply(user_text, locale),
            blocked=True,
            reason="topic_not_allowed",
            thread_id=thread_id,
        )

    intent = _infer_intent(user_text)

    # load state + short summary
    state = _load_chat_state(uid, thread_id)
    summary = _get_summary(state)

    # history from firestore only (IGNORE client body.history for safety + tokens)
    persisted_history = _load_chat_history(uid, thread_id, limit=max(18, HISTORY_LIMIT))

    # profile (trim)
    profile = await _load_user_profile(uid)
    user_context = _profile_context_compact(profile, intent)

    # astro short only if needed
    astro_short = _compute_astro_short(profile) if intent == Intent.ASTRO else ""

    # build messages
    messages = _build_messages_for_llm(
        locale=locale,
        intent=intent,
        user_context=user_context,
        astro_short=astro_short,
        summary=summary,
        history_rows=persisted_history,
        user_text=user_text,
    )

    max_tokens = _pick_max_tokens(intent)

    reply = await _call_deepseek(messages, max_tokens=max_tokens)

    # persist
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_message_batch(uid, thread_id, user_text, reply, now_iso, now_ms)

    # update turn count + summary every N turns (cheap)
    turn_count = state.get("turnCount")
    try:
        turn_count = int(turn_count) if turn_count is not None else 0
    except Exception:
        turn_count = 0
    turn_count += 1

    if (turn_count % max(1, SUMMARY_UPDATE_EVERY_TURNS)) == 0:
        summary = _update_summary_from_turn(summary, user_text, reply, intent)

    _save_chat_state(uid, thread_id, {
        "uid": uid,
        "threadId": thread_id,
        "updatedAtIso": now_iso,
        "updatedAtMs": now_ms,
        "turnCount": turn_count,
        "summary": _truncate(summary, SUMMARY_MAX_CHARS),
    })

    logger.info(
        f"[ai_chat] uid={uid} thread={thread_id} intent={intent} "
        f"user_len={len(user_text)} hist={len(persisted_history)} "
        f"ctx={'yes' if bool(user_context) else 'no'} astro={'yes' if bool(astro_short) else 'no'} "
        f"max_tokens={max_tokens}"
    )

    return AiChatResponse(reply_text=reply, blocked=False, thread_id=thread_id)
