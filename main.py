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
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT") or "8")
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS") or "240")
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
    MATCH = "match"        # ✅ NEW
    ASTRO = "astro"
    TEXTING = "texting"
    PROFILE = "profile"
    RELATION = "relation"
    GENERAL = "general"


def _parse_match_command(user_text: str) -> Optional[str]:
    """
    Accepts: "/match <uid>"
    """
    t = (user_text or "").strip()
    m = re.match(r"^/match\s+([A-Za-z0-9_\-:]{6,})\s*$", t)
    if not m:
        return None
    return (m.group(1) or "").strip()


def _infer_intent(user_text: str) -> str:
    # ✅ command has priority
    if _parse_match_command(user_text):
        return Intent.MATCH

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
# PROMPT BUILDER
# =========================
def _build_system_prompt(locale: str, intent: str) -> str:
    lang = (locale or "en").strip().lower() or "en"
    base = (
        "You are Shaadi Parrot 🦜: Indian-style dating coach + Vedic daily-fate assistant inside an app.\n"
        "Rules:\n"
        "- Be warm, confident, practical.\n"
        "- 1–2 emojis max.\n"
        "- No markdown.\n"
        "- Keep it concise. Prefer short bullets using '•'.\n"
        "- Never mention tokens, prompts, or internal system.\n"
    )

    if intent == Intent.MATCH:
        base += (
            "Task: produce a FULL 'match breakdown' for two users (USER + MATCH).\n"
            "Structure (use short headings + bullets):\n"
            "1) Quick vibe summary\n"
            "2) Strengths (why it can work)\n"
            "3) Friction points / red flags\n"
            "4) Indian-style compatibility (fun but respectful): family vibe, lifestyle, values, routines\n"
            "5) Vedic-style compatibility notes (based on provided ASTRO lines)\n"
            "6) Distance & logistics (based on distance_km)\n"
            "7) Best conversation starters (3–6)\n"
            "8) A 3-step first date plan\n"
            "Output target: 35–70 short bullet lines total.\n"
        )
    else:
        base += (
            "- If asked for a message/reply: output 2–4 message options.\n"
            "- If asked for profile/bio: give actionable edits + 1 sample bio.\n"
            "- If asked for daily fate/horoscope: give a Vedic-style daily forecast + practical tips.\n"
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
# PROFILE LOAD
# =========================
def _safe_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    deny_prefixes = ["geo", "location", "idtoken", "refreshtoken", "token", "__"]
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


def _load_user_profile_raw(uid: str) -> Dict[str, Any]:
    """
    ✅ raw profile for internal computations (lat/lon), NOT directly sent to LLM
    """
    if firestore_client is None:
        return {}
    try:
        snap = firestore_client.collection("profiles").document(uid).get()
        if not snap.exists:
            return {}
        return snap.to_dict() or {}
    except Exception:
        logger.exception("Failed to load profiles/{uid} raw")
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


def _profile_context_compact(profile: Dict[str, Any], intent: str, prefix: str = "USER") -> str:
    if not profile:
        return ""

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
    elif intent == Intent.MATCH:
        keys = list(dict.fromkeys(base_keys + ["bio", "aboutMe", "interests", "tags", "workout", "smoking", "drinking", "education", "jobTitle", "occupation", "birthDate", "birthTime"]))
    else:
        keys = base_keys + ["interests"]

    parts: List[str] = []
    for k in keys:
        if k in profile:
            val = _flatten_value(profile.get(k))
            if val:
                parts.append(f"{k}={val}")
        if len(parts) >= 18:
            break

    if not parts:
        return ""
    return f"{prefix}_CONTEXT: " + " | ".join(parts)


# =========================
# ASTRO (SHORT)
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


def _compute_astro_short(profile: Dict[str, Any], label: str = "ASTRO") -> str:
    bd = _parse_birth_date(profile)
    if not bd:
        return ""

    y, mo, d = bd
    jd_ut = swe.julday(y, mo, d, 12.0)

    try:
        sun_lon = _calc_sidereal_lon_ut(jd_ut, swe.SUN)
        moon_lon = _calc_sidereal_lon_ut(jd_ut, swe.MOON)
        sun_sign = _sign_from_lon(sun_lon)
        moon_sign = _sign_from_lon(moon_lon)
        nak = _nakshatra_from_lon(moon_lon)
        return f"{label}: Sun={sun_sign}; Moon(Rashi)={moon_sign}; Nakshatra={nak}."
    except Exception:
        logger.exception("Astro compute failed")
        return ""


# =========================
# DISTANCE
# =========================
def _try_get_float(d: Dict[str, Any], key: str) -> float:
    v = d.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _distance_km_from_profiles(raw_a: Dict[str, Any], raw_b: Dict[str, Any]) -> Optional[int]:
    lat1 = _try_get_float(raw_a, "lat")
    lon1 = _try_get_float(raw_a, "lon")
    lat2 = _try_get_float(raw_b, "lat")
    lon2 = _try_get_float(raw_b, "lon")
    if not lat1 or not lon1 or not lat2 or not lon2:
        return None
    try:
        km = _haversine_km(lat1, lon1, lat2, lon2)
        if km < 0:
            return None
        return int(round(km))
    except Exception:
        return None


# =========================
# FIRESTORE CHAT STORAGE (unchanged)
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

    m = re.search(r"\b(?:her name is|his name is|my gf is|my bf is)\s+([A-Z][a-z]+)\b", t, re.IGNORECASE)
    if m:
        summary = _append_summary(summary, "partnerName", m.group(1))

    mz = re.search(r"\b(i'?m|i am)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz:
        summary = _append_summary(summary, "userZodiac", mz.group(2).title())

    mz2 = re.search(r"\b(she'?s|she is|he'?s|he is)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz2:
        summary = _append_summary(summary, "partnerZodiac", mz2.group(2).title())

    summary = _append_summary(summary, "lastTopic", intent)
    return summary


# =========================
# PHOTO MODERATION + FACE VERIFICATION (unchanged)
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
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return 0.0
    return float(w * h)


def _pick_largest_face(faces: List[vision.FaceAnnotation]) -> Optional[vision.FaceAnnotation]:
    if not faces:
        return None
    best = None
    best_area = 0.0
    for f in faces:
        a = _face_area_proxy(f)
        if a > best_area:
            best_area = a
            best = f
    return best


def _likelihood_name_to_int(name: str) -> int:
    return int(_LIKELIHOOD.get((name or "UNKNOWN").strip().upper(), 0))


def _face_quality_checks(face: vision.FaceAnnotation) -> Tuple[bool, str]:
    """
    Lightweight heuristics to reduce obvious junk:
    - too tilted (roll/pan/tilt)
    - too tiny face (area proxy threshold handled elsewhere)
    - very low detection confidence
    """
    try:
        conf = float(getattr(face, "detection_confidence", 0.0) or 0.0)
        if conf < 0.45:
            return False, "low_confidence"

        # Vision angles are degrees
        roll = abs(float(getattr(face, "roll_angle", 0.0) or 0.0))
        pan = abs(float(getattr(face, "pan_angle", 0.0) or 0.0))
        tilt = abs(float(getattr(face, "tilt_angle", 0.0) or 0.0))

        if roll > 30 or pan > 30 or tilt > 30:
            return False, "face_too_angled"

        # eyes open check (soft)
        left_open = getattr(face, "left_eye_open_probability", None)
        right_open = getattr(face, "right_eye_open_probability", None)
        if left_open is not None and right_open is not None:
            try:
                if float(left_open) < 0.10 and float(right_open) < 0.10:
                    return False, "eyes_closed"
            except Exception:
                pass

        return True, "ok"
    except Exception:
        return False, "face_quality_check_failed"


def _detect_faces_and_safety_from_bytes(img_bytes: bytes):
    if vision_client is None:
        raise HTTPException(status_code=503, detail="Vision client not available")

    try:
        image = vision.Image(content=img_bytes)
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
    faces = list(resp.face_annotations or [])

    return adult, racy, violence, faces


def _store_face_verified(uid: str, ok: bool, reason: str, meta: Dict[str, Any]) -> None:
    """
    Stores a minimal verification outcome.
    """
    if firestore_client is None:
        return
    try:
        patch = {
            "faceVerified": bool(ok),
            "faceVerifiedReason": (reason or "").strip(),
            "faceVerifiedAtIso": _now_iso(),
            "faceVerifiedMeta": meta or {},
        }
        firestore_client.collection("profiles").document(uid).set(patch, merge=True)
    except Exception:
        logger.exception("Failed to store face verification outcome")


@app.post("/verify-face")
def verify_face(body: VerifyFaceRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    # Only allow self-verify
    target_uid = (body.user_id or "").strip()
    if not target_uid or target_uid != uid:
        raise HTTPException(status_code=403, detail="user_id must match auth uid")

    url = (str(body.image_url) or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="image_url required")

    img_bytes = _download_image_bytes(url, max_mb=10)
    adult, racy, violence, faces = _detect_faces_and_safety_from_bytes(img_bytes)

    # Safety gates
    if _LIKELIHOOD.get(adult, 0) >= _LIKELIHOOD["LIKELY"]:
        _store_face_verified(uid, False, "adult_content", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces)})
        return {"ok": False, "reason": "adult_content", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    if _LIKELIHOOD.get(racy, 0) >= _LIKELIHOOD["VERY_LIKELY"]:
        _store_face_verified(uid, False, "highly_racy", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces)})
        return {"ok": False, "reason": "highly_racy", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    if _LIKELIHOOD.get(violence, 0) >= _LIKELIHOOD["VERY_LIKELY"]:
        _store_face_verified(uid, False, "high_violence", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces)})
        return {"ok": False, "reason": "high_violence", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    if len(faces) == 0:
        _store_face_verified(uid, False, "no_face_detected", {"adult": adult, "racy": racy, "violence": violence, "faces": 0})
        return {"ok": False, "reason": "no_face_detected", "adult": adult, "racy": racy, "violence": violence, "faces": 0}

    # Pick largest face and validate quality
    best = _pick_largest_face(faces)
    if best is None:
        _store_face_verified(uid, False, "no_face_detected", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces)})
        return {"ok": False, "reason": "no_face_detected", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    # Make sure the face isn't tiny
    area = _face_area_proxy(best)
    # heuristic threshold (works as a proxy across common resolutions)
    if area < 18_000:
        _store_face_verified(uid, False, "face_too_small", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces), "area": int(area)})
        return {"ok": False, "reason": "face_too_small", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    ok_quality, q_reason = _face_quality_checks(best)
    if not ok_quality:
        _store_face_verified(uid, False, q_reason, {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces), "area": int(area)})
        return {"ok": False, "reason": q_reason, "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    _store_face_verified(uid, True, "ok", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces), "area": int(area)})
    return {"ok": True, "reason": "ok", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}


# =========================
# DEEPSEEK CALL
# =========================
def _deepseek_chat(messages: List[Dict[str, str]], max_tokens: int) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail="DeepSeek API key not configured")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": DS_TEMPERATURE,
        "max_tokens": int(max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=DS_TIMEOUT_SEC)
    except Exception as e:
        logger.exception("DeepSeek request failed")
        raise HTTPException(status_code=502, detail=f"DeepSeek request error: {type(e).__name__}")

    if r.status_code != 200:
        logger.error("DeepSeek non-200: %s %s", r.status_code, (r.text or "")[:400])
        raise HTTPException(status_code=502, detail=f"DeepSeek error HTTP {r.status_code}")

    try:
        data = r.json()
    except Exception:
        logger.exception("DeepSeek JSON parse failed")
        raise HTTPException(status_code=502, detail="DeepSeek invalid JSON")

    try:
        txt = (data["choices"][0]["message"]["content"] or "").strip()
        return txt
    except Exception:
        logger.exception("DeepSeek response shape unexpected: %s", str(data)[:400])
        raise HTTPException(status_code=502, detail="DeepSeek response invalid")


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _build_llm_messages(
    locale: str,
    intent: str,
    summary: str,
    user_profile: Dict[str, Any],
    match_profile: Optional[Dict[str, Any]],
    distance_km: Optional[int],
    user_text: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    system = _build_system_prompt(locale, intent)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

    # summary memory (cheap)
    if summary:
        msgs.append({"role": "system", "content": f"SUMMARY_MEMORY: {_trim_text(summary, SUMMARY_MAX_CHARS)}"})

    # compact profile context
    uctx = _profile_context_compact(user_profile, intent, prefix="USER")
    if uctx:
        msgs.append({"role": "system", "content": uctx})

    if intent == Intent.MATCH and match_profile:
        mctx = _profile_context_compact(match_profile, intent, prefix="MATCH")
        if mctx:
            msgs.append({"role": "system", "content": mctx})

        # extra signals for match mode
        ua = _compute_astro_short(user_profile, label="USER_ASTRO")
        ma = _compute_astro_short(match_profile, label="MATCH_ASTRO")
        if ua:
            msgs.append({"role": "system", "content": ua})
        if ma:
            msgs.append({"role": "system", "content": ma})
        if distance_km is not None:
            msgs.append({"role": "system", "content": f"DISTANCE_KM: {int(distance_km)}"})

    # history (already trimmed by loader and knobs)
    for h in (history or [])[-HISTORY_LIMIT:]:
        role = (h.get("role") or "").strip()
        content = _trim_text(h.get("content") or "", HISTORY_MAX_CHARS)
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})

    # current turn
    msgs.append({"role": "user", "content": _trim_text(user_text, USER_TEXT_MAX_CHARS)})
    return msgs


def _extract_reply_safe(txt: str) -> str:
    txt = (txt or "").strip()
    if not txt:
        return "…"
    # No markdown requested
    txt = txt.replace("**", "").replace("```", "")
    # Hard cap to keep mobile UI sane
    if len(txt) > 1800:
        txt = txt[:1800].rstrip() + "…"
    return txt


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/ai-chat", response_model=AiChatResponse)
def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text or "")
    locale = (body.locale or "en").strip()
    thread_id = (body.thread_id or "default").strip() or "default"

    if not user_text:
        raise HTTPException(status_code=400, detail="text required")

    # Topic gate
    if not _is_allowed_topic(user_text):
        reply = _topic_block_reply(user_text, locale)
        return AiChatResponse(reply_text=reply, blocked=True, reason="topic_blocked", thread_id=thread_id)

    intent = _infer_intent(user_text)

    # Load state + summary
    state = _load_chat_state(uid, thread_id)
    summary = _get_summary(state)

    # Load history from Firestore (ignored body.history for token economy)
    history = _load_chat_history(uid, thread_id, limit=24)

    # Load user profile
    user_profile = {}
    user_profile_raw = {}
    try:
        user_profile = asyncio.run(_load_user_profile(uid))  # type: ignore
    except Exception:
        # if event loop already exists (rare in some deployments), fallback to sync raw only
        user_profile = _safe_profile_dict(_load_user_profile_raw(uid))

    user_profile_raw = _load_user_profile_raw(uid)

    # MATCH command: /match <uid>
    match_uid = None
    match_profile = None
    distance_km = None

    if intent == Intent.MATCH:
        match_uid = _parse_match_command(user_text)
        if not match_uid:
            reply = "Usage: /match <uid>"
            return AiChatResponse(reply_text=reply, blocked=False, reason=None, thread_id=thread_id)

        # Load match profile
        match_profile = _safe_profile_dict(_load_user_profile_raw(match_uid))
        match_profile_raw = _load_user_profile_raw(match_uid)

        # Distance
        distance_km = _distance_km_from_profiles(user_profile_raw, match_profile_raw)

    # Build messages and call DeepSeek
    msgs = _build_llm_messages(
        locale=locale,
        intent=intent,
        summary=summary,
        user_profile=user_profile,
        match_profile=match_profile,
        distance_km=distance_km,
        user_text=user_text,
        history=history,
    )

    max_tokens = DS_MAX_TOKENS_DEFAULT
    if intent == Intent.MATCH:
        max_tokens = max(420, DS_MAX_TOKENS_DEFAULT)  # match breakdown needs more room

    assistant_text = _deepseek_chat(msgs, max_tokens=max_tokens)
    assistant_text = _extract_reply_safe(assistant_text)

    # Save messages
    created_at_ms = _now_ms()
    created_at_iso = _now_iso()
    _save_chat_message_batch(uid, thread_id, user_text, assistant_text, created_at_iso, created_at_ms)

    # Summary update every N turns (cheap)
    try:
        turns = int(state.get("turns") or 0)
    except Exception:
        turns = 0
    turns += 1

    if turns % max(1, SUMMARY_UPDATE_EVERY_TURNS) == 0:
        new_summary = _update_summary_from_turn(summary, user_text, assistant_text, intent)
        _save_chat_state(uid, thread_id, {"summary": new_summary, "turns": turns})
    else:
        _save_chat_state(uid, thread_id, {"turns": turns})

    return AiChatResponse(reply_text=assistant_text, blocked=False, reason=None, thread_id=thread_id)


# =========================
# HISTORY + RESET
# =========================
@app.get("/history", response_model=HistoryResponse)
def history(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = (thread_id or "default").strip() or "default"

    rows = _load_chat_history(uid, tid, limit=40)
    out: List[ChatTurn] = []
    for r in rows:
        role = (r.get("role") or "").strip()
        txt = (r.get("content") or "").strip()
        if role in ("user", "assistant") and txt:
            out.append(ChatTurn(role=role, text=txt))

    return HistoryResponse(thread_id=tid, messages=out)


@app.post("/reset", response_model=ResetResponse)
def reset(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = (thread_id or "default").strip() or "default"

    # Soft reset: set clearedAtMs
    ms = _now_ms()
    _save_chat_state(uid, tid, {"clearedAtMs": ms, "updatedAtMs": ms, "updatedAtIso": _now_iso()})
    return ResetResponse(thread_id=tid, ok=True)

