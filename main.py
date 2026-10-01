from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import logging
import os
import re
import time
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Literal, Dict, Any, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
import swisseph as swe

from google.api_core import exceptions as google_exceptions

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

# =========================
# ECONOMY KNOBS
# =========================
DS_TEMPERATURE = float(os.getenv("DS_TEMPERATURE") or "0.78")
DS_TIMEOUT_SEC = int(os.getenv("DS_TIMEOUT_SEC") or "45")

# Output caps per intent, sized to the answer length each intent's prompt asks for.
DS_MAX_TOKENS_DEFAULT = int(os.getenv("DS_MAX_TOKENS_DEFAULT") or "320")
DS_MAX_TOKENS_TEXTING = int(os.getenv("DS_MAX_TOKENS_TEXTING") or "280")
DS_MAX_TOKENS_PROFILE = int(os.getenv("DS_MAX_TOKENS_PROFILE") or "360")
DS_MAX_TOKENS_ASTRO = int(os.getenv("DS_MAX_TOKENS_ASTRO") or "380")
DS_MAX_TOKENS_MATCH = int(os.getenv("DS_MAX_TOKENS_MATCH") or "1100")

# Prompt window: the last N stored messages. Older turns live in the rolling memory.
HISTORY_LIMIT = max(4, int(os.getenv("HISTORY_LIMIT") or "6"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS") or "240")  # user messages
HISTORY_ASSISTANT_MAX_CHARS = int(os.getenv("HISTORY_ASSISTANT_MAX_CHARS") or "120")
# The latest reply is what follow-ups ("the 2nd option", "shorter") refer to, so it keeps more.
HISTORY_LAST_ASSISTANT_MAX_CHARS = int(os.getenv("HISTORY_LAST_ASSISTANT_MAX_CHARS") or "400")
# Extra rows beyond the window so the memory can catch up on messages it has not summarized.
HISTORY_FETCH_LIMIT = HISTORY_LIMIT + 10
USER_TEXT_MAX_CHARS = int(os.getenv("USER_TEXT_MAX_CHARS") or "900")

# Rolling memory (~120 tokens), persisted per uid/thread in parrotChats.
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS") or "520")
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS") or "160")

HOROSCOPE_MAX_TOKENS = int(os.getenv("HOROSCOPE_MAX_TOKENS") or "480")
HOROSCOPE_DEFAULT_TZ = "Asia/Kolkata"

PLACES_DB_PATH = os.getenv("PLACES_DB_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "places.sqlite3")

# Server-side Parrot quota: an LLM reply is served only if the app already spent
# a request via Cloud Functions `consumeParrotQuota` (premium users skip that call).
AI_CHAT_QUOTA_ENFORCED = (os.getenv("AI_CHAT_QUOTA_ENFORCED") or "1").strip() != "0"
# Hard cost guard for everyone, premium included.
AI_CHAT_HARD_DAILY_CAP = max(1, int(os.getenv("AI_CHAT_HARD_DAILY_CAP") or "150"))

if not firebase_admin._apps:
    firebase_admin.initialize_app()

# =========================
# MODELS
# =========================
class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


class VerifyFaceStartRequest(BaseModel):
    consent: str = ""           # the version of the notice the person agreed to in the app (e.g. "face-2026-10")


class VerifyFaceLiveRequest(BaseModel):
    challengeId: str            # from /verify-face-start
    selfies: List[str]          # one per challenge step, in order (the user's own Storage uploads)


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
    history: Optional[List[ChatTurn]] = None  # ignored on server


class AiChatResponse(BaseModel):
    reply_text: str
    blocked: bool = False
    reason: Optional[str] = None
    thread_id: Optional[str] = None


class HistoryResponse(BaseModel):
    thread_id: str = "default"
    messages: List[ChatTurn] = Field(default_factory=list)


class DailyHoroscopeRequest(BaseModel):
    tz: Optional[str] = None
    thread_id: Optional[str] = None


class DailyHoroscopeResponse(BaseModel):
    ok: bool
    text: Optional[str] = None
    dayKey: Optional[str] = None
    cached: Optional[bool] = None
    error: Optional[str] = None


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
            "history_assistant_max_chars": HISTORY_ASSISTANT_MAX_CHARS,
            "summary_max_chars": SUMMARY_MAX_CHARS,
            "max_tokens_default": DS_MAX_TOKENS_DEFAULT,
            "max_tokens_texting": DS_MAX_TOKENS_TEXTING,
            "max_tokens_profile": DS_MAX_TOKENS_PROFILE,
            "max_tokens_astro": DS_MAX_TOKENS_ASTRO,
            "max_tokens_match": DS_MAX_TOKENS_MATCH,
            "places_db": os.path.exists(PLACES_DB_PATH),
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
# INTENT
# =========================
class Intent:
    MATCH = "match"
    ASTRO = "astro"
    TEXTING = "texting"
    PROFILE = "profile"
    RELATION = "relation"
    GENERAL = "general"


def _parse_match_command(user_text: str) -> Optional[str]:
    t = (user_text or "").strip()
    m = re.match(r"^/match\s+([A-Za-z0-9_\-:]{6,})\s*$", t)
    if not m:
        return None
    return (m.group(1) or "").strip()


def _infer_intent(user_text: str) -> str:
    if _parse_match_command(user_text):
        return Intent.MATCH

    t = (user_text or "").lower()

    astro_words = [
        "horoscope", "forecast", "daily fate", "vedic", "kundli", "nakshatra", "rashi",
        "tomorrow", "tmr", "zodiac", "moon sign",
        "гороскоп", "прогноз", "сегодня", "ведичес", "накшатр", "раши", "кундли", "совместим"
    ]
    if any(w in t for w in astro_words):
        return Intent.ASTRO

    texting_words = [
        "text", "message", "reply", "dm", "what to say",
        "как ответить", "сообщение", "ответ", "написать ей", "написать ему"
    ]
    if any(w in t for w in texting_words):
        return Intent.TEXTING

    profile_words = ["profile", "bio", "photos", "about me", "анкет", "био", "фото", "описание"]
    if any(w in t for w in profile_words):
        return Intent.PROFILE

    relation_words = ["relationship", "dating", "girl", "boyfriend", "girlfriend", "love", "свидани", "отношени", "любов"]
    if any(w in t for w in relation_words):
        return Intent.RELATION

    return Intent.GENERAL


def _recent_history_has_astro(history: List[Dict[str, str]], lookback: int = 5) -> bool:
    if not history:
        return False

    astro_markers = [
        "horoscope", "forecast", "daily fate", "vedic", "nakshatra", "rashi", "zodiac",
        "гороскоп", "прогноз", "накшат", "раши", "астро"
    ]

    tail = (history or [])[-max(1, lookback):]
    for item in tail:
        content = (item.get("content") or "").lower()
        if any(m in content for m in astro_markers):
            return True

    return False


def _infer_intent_with_history(user_text: str, history: List[Dict[str, str]]) -> str:
    inferred = _infer_intent(user_text)
    if inferred != Intent.GENERAL:
        return inferred

    t = (user_text or "").strip().lower()
    if not t:
        return inferred

    # Follow-up short asks like "what about tomorrow" should keep astro context
    follow_up_time_markers = ["tomorrow", "tmr", "next day", "завтра", "на завтра"]
    if any(m in t for m in follow_up_time_markers) and _recent_history_has_astro(history, lookback=5):
        return Intent.ASTRO

    return inferred


def _is_identity_question(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False

    markers = [
        "who are you", "what are you", "who r u", "what can you do",
        "кто ты", "ты кто", "что ты умеешь", "чем ты можешь помочь"
    ]
    return any(m in t for m in markers)


def _identity_reply(locale: str) -> str:
    lang = (locale or "en").strip().lower()
    if lang.startswith("ru"):
        return (
            "Я Shaadi Parrot 🦜\n"
            "• Помогаю с общением в дейтинге и отношениях\n"
            "• Делаю разборы мэтчей и идеи для первого шага\n"
            "• Даю ведические daily-fate подсказки\n"
            "Напиши, что сейчас происходит — разберём по шагам ✨"
        )

    return (
        "I’m Shaadi Parrot 🦜\n"
        "• I help with dating chats and relationship advice\n"
        "• I do match breakdowns and first-message ideas\n"
        "• I give Vedic-style daily fate guidance\n"
        "Tell me what’s happening, and I’ll break it down step by step ✨"
    )


# =========================
# PROMPT BUILDER
# =========================
# Sent byte-identical as the first message of every chat request, so DeepSeek serves it
# from its prefix cache. Anything per-intent or per-user goes in later messages.
_SYSTEM_PROMPT_BASE = (
    "You are Shaadi Parrot 🦜: Indian-style dating coach + Vedic daily-fate assistant inside an app.\n"
    "Rules:\n"
    "- Be warm, confident, practical, a bit playful.\n"
    "- Use 4–10 emojis TOTAL across the whole answer (not every line).\n"
    "- No markdown.\n"
    "- Use short headings + bullets using '•'.\n"
    "- Make it fun to read: vivid phrasing, mini-hooks, short punchy lines.\n"
    "- Never mention tokens, prompts, or internal system.\n"
    "- Default to a detailed, useful answer unless user explicitly asks for short.\n"
)


def _max_tokens_for_intent(intent: str) -> int:
    return {
        Intent.MATCH: DS_MAX_TOKENS_MATCH,
        Intent.TEXTING: DS_MAX_TOKENS_TEXTING,
        Intent.PROFILE: DS_MAX_TOKENS_PROFILE,
        Intent.ASTRO: DS_MAX_TOKENS_ASTRO,
    }.get(intent, DS_MAX_TOKENS_DEFAULT)


def _build_intent_prompt(locale: str, intent: str) -> str:
    lang = (locale or "en").strip().lower() or "en"
    base = ""

    if intent == Intent.MATCH:
        base += (
            "Task: produce a FULL 'match breakdown' for two users (USER + MATCH).\n"
            "Structure (short headings + bullets):\n"
            "1) Quick vibe summary\n"
            "2) Strengths (why it can work)\n"
            "3) Friction points / red flags\n"
            "4) Indian-style compatibility (fun but respectful): family vibe, lifestyle, values, routines\n"
            "5) Vedic-style compatibility notes (based on provided ASTRO lines)\n"
            "6) Distance & logistics (based on distance_km)\n"
            "7) Best conversation starters (3–8)\n"
            "8) A 3-step first date plan\n"
            "Finish with 1 verdict line: 'Worth it' or 'Proceed with caution' + 1 emoji.\n"
            "Output target: 45–90 short bullet lines total.\n"
        )
    else:
        base += (
            "- If asked for a message/reply: output 2–4 message options.\n"
            "- If asked for profile/bio: give actionable edits + 1 sample bio.\n"
            "- If asked for daily fate/horoscope: give a Vedic-style daily forecast + practical tips.\n"
        )

        if intent == Intent.TEXTING:
            base += "Output target: 8–14 short bullet lines total.\n"
        elif intent == Intent.PROFILE:
            base += "Output target: 10–16 bullet lines + 1 short sample bio.\n"
        elif intent == Intent.ASTRO:
            base += "Output target: 12–18 bullet lines with sections: Energy, Love, Work, Practical tips, Mantra.\n"
        else:
            base += "Output target: 9–15 short bullet lines.\n"

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


def _merge_dicts_prefer_first(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k not in out and v is not None:
                out[k] = v
    return out


def _load_user_docs(uid: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """profiles/{uid}, publicProfiles/{uid}, users/{uid}: read once per request, in one round trip."""
    if firestore_client is None:
        return {}, {}, {}
    refs = [firestore_client.collection(name).document(uid) for name in ("profiles", "publicProfiles", "users")]
    try:
        by_path = {snap.reference.path: (snap.to_dict() or {}) for snap in firestore_client.get_all(refs) if snap.exists}
    except Exception:
        logger.exception("Failed to load user docs")
        return {}, {}, {}
    profile_raw, public_raw, user_raw = (by_path.get(ref.path, {}) for ref in refs)
    return profile_raw, public_raw, user_raw


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
    birth_keys = ["birthDate", "birthTime", "birthCityName", "birthStateName", "birthCountryName"]
    astro_keys = base_keys + birth_keys + ["interests"]

    if intent == Intent.TEXTING:
        keys = texting_keys
    elif intent == Intent.PROFILE:
        keys = profile_keys
    elif intent == Intent.ASTRO:
        keys = astro_keys
    elif intent == Intent.MATCH:
        keys = list(dict.fromkeys(base_keys + [
            "bio", "aboutMe", "interests", "tags", "workout", "smoking", "drinking",
            "education", "jobTitle", "occupation"
        ] + birth_keys))
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
# ASTRO
# =========================
_ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

_NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
    "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
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


def _parse_birth_time(profile: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(profile.get("birthTime") or ""))
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _zone(name: Any, fallback: str = HOROSCOPE_DEFAULT_TZ) -> ZoneInfo:
    try:
        return ZoneInfo(str(name or "").strip() or fallback)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(fallback)


def _jd_ut(moment: datetime) -> float:
    utc = moment.astimezone(timezone.utc)
    return swe.julday(utc.year, utc.month, utc.day, utc.hour + utc.minute / 60.0 + utc.second / 3600.0)


# places.sqlite3 is built by tools/build_places_db.py from the app's own location DBs, so the
# birthCityName / birthStateName / birthCountryIso2 the app stores resolve without a geocoder.
_places_conn: Optional[sqlite3.Connection] = None
_places_lock = threading.Lock()


def _places_query(sql: str, params: Tuple[Any, ...]) -> List[Tuple[Any, ...]]:
    global _places_conn
    with _places_lock:
        if _places_conn is None:
            if not os.path.exists(PLACES_DB_PATH):
                return []
            _places_conn = sqlite3.connect(f"file:{PLACES_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        return _places_conn.execute(sql, params).fetchall()


def _norm_place(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _resolve_birth_place(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """lat/lon/tz of the birth place; precision is city, state or country."""
    city = _norm_place(profile.get("birthCityName"))
    state = _norm_place(profile.get("birthStateName"))
    cc = str(profile.get("birthCountryIso2") or "").strip().upper()
    try:
        if not cc:
            rows = _places_query("SELECT cc FROM countries WHERE name = ?", (_norm_place(profile.get("birthCountryName")),))
            cc = rows[0][0] if rows else ""
        if not cc:
            return None
        if city:
            rows = _places_query(
                "SELECT c.lat, c.lon, c.tz, s.state FROM cities c LEFT JOIN states s ON s.id = c.state_id "
                "WHERE c.cc = ? AND c.city = ? ORDER BY COALESCE(c.pop, 0) DESC", (cc, city))
            if rows:
                lat, lon, tz, _ = next((r for r in rows if state and r[3] == state), rows[0])
                return {"lat": lat, "lon": lon, "tz": tz, "precision": "city"}
        if state:
            rows = _places_query("SELECT lat, lon, tz FROM states WHERE cc = ? AND state = ?", (cc, state))
            if rows and rows[0][0] is not None:
                return {"lat": rows[0][0], "lon": rows[0][1], "tz": rows[0][2], "precision": "state"}
        rows = _places_query("SELECT lat, lon, tz FROM countries WHERE cc = ?", (cc,))
        if rows and rows[0][0] is not None:
            return {"lat": rows[0][0], "lon": rows[0][1], "tz": rows[0][2], "precision": "country"}
    except sqlite3.Error:
        logger.exception("Birth place lookup failed")
    return None


def _natal_chart(profile: Dict[str, Any], fallback_tz: str = HOROSCOPE_DEFAULT_TZ) -> Optional[Dict[str, Any]]:
    """Sidereal (Lahiri) natal basics. Unknown birth time -> local noon and no lagna."""
    bd = _parse_birth_date(profile)
    if not bd:
        return None
    bt = _parse_birth_time(profile)
    place = _resolve_birth_place(profile)
    tz = _zone(place.get("tz") if place else None, fallback=_zone(fallback_tz).key)
    try:
        jd = _jd_ut(datetime(bd[0], bd[1], bd[2], bt[0] if bt else 12, bt[1] if bt else 0, tzinfo=tz))
        sun_lon = _calc_sidereal_lon_ut(jd, swe.SUN)
        moon_lon = _calc_sidereal_lon_ut(jd, swe.MOON)
        lagna = None
        # The ascendant moves a sign every ~2h: only trust it with a real time and a city.
        if bt and place and place["precision"] == "city":
            _, ascmc = swe.houses_ex(jd, float(place["lat"]), float(place["lon"]), b"W", swe.FLG_SIDEREAL)
            lagna = _sign_from_lon(float(ascmc[0]))
    except Exception:
        logger.exception("Natal chart compute failed")
        return None
    return {
        "sun_sign": _sign_from_lon(sun_lon),
        "moon_sign": _sign_from_lon(moon_lon),
        "moon_sign_idx": int(moon_lon // 30.0) % 12,
        "nakshatra": _nakshatra_from_lon(moon_lon),
        "nakshatra_idx": int(moon_lon // (360.0 / 27.0)) % 27,
        "lagna": lagna,
        "time_known": bt is not None,
    }


def _compute_astro_short(profile: Dict[str, Any], label: str = "ASTRO") -> str:
    chart = _natal_chart(profile)
    if not chart:
        return ""
    line = f"{label}: Sun={chart['sun_sign']}; Moon(Rashi)={chart['moon_sign']}; Nakshatra={chart['nakshatra']}"
    if chart["lagna"]:
        line += f"; Lagna={chart['lagna']}"
    return line + "."


# =========================
# DISTANCE
# =========================
def _try_get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _distance_km_from_profiles(raw_a: Dict[str, Any], raw_b: Dict[str, Any]) -> Optional[int]:
    lat1 = _try_get_float(raw_a, "lat")
    lon1 = _try_get_float(raw_a, "lon")
    lat2 = _try_get_float(raw_b, "lat")
    lon2 = _try_get_float(raw_b, "lon")
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    if not (-90.0 <= lat1 <= 90.0 and -180.0 <= lon1 <= 180.0 and -90.0 <= lat2 <= 90.0 and -180.0 <= lon2 <= 180.0):
        return None

    try:
        km = _haversine_km(lat1, lon1, lat2, lon2)
        if km < 0:
            return None
        return int(round(km))
    except Exception:
        return None


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
    docref = _chat_doc_ref(uid, thread_id)
    if docref is None:
        return None
    return docref.collection("messages")


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


def _looks_like_legacy_prompt_dump(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    markers = [
        "USER_CONTEXT:",
        "MATCH_CONTEXT:",
        "SUMMARY_MEMORY:",
        "USER_ASTRO:",
        "MATCH_ASTRO:",
        "DISTANCE_KM:",
        "[USER_MESSAGE]",
        "photoPaths=",
        "photoUrls=",
        "updatedAtIso=",
        "firebase",
        "firebasestorage.googleapis.com",
    ]

    marker_hits = sum(1 for m in markers if m.lower() in t.lower())
    pipe_count = t.count("|")
    eq_count = t.count("=")

    if marker_hits >= 2:
        return True
    if pipe_count >= 6 and eq_count >= 8:
        return True
    return False


def _sanitize_legacy_history_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    if "[USER_MESSAGE]" in t:
        tail = t.split("[USER_MESSAGE]", 1)[1].strip()
        if tail:
            return _trim_text(_normalize_text(tail), 600)

    if _looks_like_legacy_prompt_dump(t):
        return ""

    # Keep line breaks and full length: replies are multi-line (headings + bullets) and /history
    # must show them as they were sent. The prompt window trims separately.
    t = re.sub(r"[ \t]+", " ", t.replace("\r\n", "\n"))
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return _trim_text(t, 7000)


def _read_message_text(doc: Dict[str, Any]) -> str:
    if not isinstance(doc, dict):
        return ""

    for key in ("text", "content", "message", "reply_text"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def _load_chat_history(uid: str, thread_id: str, limit: int = 24, state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Oldest-first rows {role, content, ms}; pass `state` when the caller already loaded it."""
    col = _chat_msgs_col_ref(uid, thread_id)
    if col is None:
        return []

    if state is None:
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
            text = _sanitize_legacy_history_text(_read_message_text(d))
            ms = d.get("createdAtMs")
            try:
                ms = int(ms) if ms is not None else 0
            except Exception:
                ms = 0

            if not role or not text:
                continue
            if cleared_ms and ms and ms <= cleared_ms:
                continue
            if role not in ("user", "assistant"):
                continue

            rows.append({"role": role, "content": text, "ms": ms})

        rows.reverse()
        return rows
    except Exception:
        logger.exception("Failed to load chat history")
        return []


# =========================
# ROLLING MEMORY
# =========================
# parrotChats state keeps `summary` (<= SUMMARY_MAX_CHARS) covering every stored message up to
# `summaryThroughMs`. The prompt carries the summary plus the last HISTORY_LIMIT messages. Once
# HISTORY_LIMIT messages are unsummarized, the oldest is about to leave the window, so they are
# folded into the summary by a small parallel LLM call (no added latency for the user).
_MEMORY_SYSTEM_PROMPT = (
    "You maintain the long-term memory that Shaadi Parrot, a dating coach and Vedic astrology assistant, "
    "keeps about ONE app user.\n"
    "Merge PREVIOUS MEMORY with NEW MESSAGES into one updated memory.\n"
    "Keep only durable, useful facts: the user's situation and goals, people they mention (names, relationship "
    "stage, zodiac), preferences and boundaries, decisions made, advice already given, open questions to follow up.\n"
    "Drop greetings, small talk, and anything outdated.\n"
    "Write in English, third person ('User ...'), at most 80 words, short phrases separated by '; ', no markdown.\n"
    "Output only the memory text."
)

_memory_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memory")


def _get_summary(state: Dict[str, Any]) -> str:
    s = state.get("summary")
    if isinstance(s, str):
        return s.strip()
    return ""


def _unsummarized(history: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    through_ms = _to_int(state.get("summaryThroughMs"))
    return [h for h in history if _to_int(h.get("ms")) > through_ms]


def _summarize_memory(previous: str, messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = _normalize_text(m.get("content") or "")
        if m.get("role") == "user":
            lines.append("User: " + _trim_text(content, 300))
        else:
            lines.append("Parrot: " + _trim_text(content, 200))
    msgs = [
        {"role": "system", "content": _MEMORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"PREVIOUS MEMORY: {previous or '(empty)'}\nNEW MESSAGES:\n" + "\n".join(lines)},
    ]
    text, _ = _deepseek_complete(msgs, max_tokens=MEMORY_MAX_TOKENS, temperature=0.2, kind="memory")
    return _trim_text(_normalize_text(text.replace("**", "")), SUMMARY_MAX_CHARS)


# =========================
# PHOTO MODERATION + FACE VERIFICATION
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


def _face_quality_checks(face: vision.FaceAnnotation) -> Tuple[bool, str]:
    try:
        conf = float(getattr(face, "detection_confidence", 0.0) or 0.0)
        if conf < 0.45:
            return False, "low_confidence"

        roll = abs(float(getattr(face, "roll_angle", 0.0) or 0.0))
        pan = abs(float(getattr(face, "pan_angle", 0.0) or 0.0))
        tilt = abs(float(getattr(face, "tilt_angle", 0.0) or 0.0))

        if roll > 30 or pan > 30 or tilt > 30:
            return False, "face_too_angled"

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

    target_uid = (body.user_id or "").strip()
    if not target_uid or target_uid != uid:
        raise HTTPException(status_code=403, detail="user_id must match auth uid")

    url = (str(body.image_url) or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="image_url required")

    img_bytes = _download_image_bytes(url, max_mb=10)
    adult, racy, violence, faces = _detect_faces_and_safety_from_bytes(img_bytes)

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

    best = _pick_largest_face(faces)
    if best is None:
        _store_face_verified(uid, False, "no_face_detected", {"adult": adult, "racy": racy, "violence": violence, "faces": len(faces)})
        return {"ok": False, "reason": "no_face_detected", "adult": adult, "racy": racy, "violence": violence, "faces": len(faces)}

    area = _face_area_proxy(best)
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
def _deepseek_complete(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: Optional[float] = None,
    kind: str = "chat",
    timeout: Optional[int] = None,
) -> Tuple[str, str]:
    """Returns (text, finish_reason) and logs token usage, including prefix-cache hits."""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail="DeepSeek API key not configured")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": DS_TEMPERATURE if temperature is None else temperature,
        "max_tokens": int(max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=timeout or DS_TIMEOUT_SEC)
    except Exception as e:
        logger.exception("DeepSeek request failed")
        raise HTTPException(status_code=502, detail=f"DeepSeek request error: {type(e).__name__}")

    if r.status_code != 200:
        logger.error("DeepSeek non-200: %s %s", r.status_code, (r.text or "")[:500])
        raise HTTPException(status_code=502, detail=f"DeepSeek error HTTP {r.status_code}")

    try:
        data = r.json()
    except Exception:
        logger.exception("DeepSeek JSON parse failed")
        raise HTTPException(status_code=502, detail="DeepSeek invalid JSON")

    try:
        choice = data["choices"][0]
        txt = (choice["message"]["content"] or "").strip()
        finish_reason = str(choice.get("finish_reason") or "")
    except Exception:
        logger.exception("DeepSeek response shape unexpected: %s", str(data)[:500])
        raise HTTPException(status_code=502, detail="DeepSeek response invalid")

    usage = data.get("usage") or {}
    logger.info(
        "deepseek_usage kind=%s prompt=%s cache_hit=%s completion=%s max_tokens=%s finish=%s",
        kind, usage.get("prompt_tokens"), usage.get("prompt_cache_hit_tokens"),
        usage.get("completion_tokens"), max_tokens, finish_reason,
    )
    return txt, finish_reason


def _drop_cut_off_tail(text: str) -> str:
    """For replies stopped by max_tokens: drop the unfinished last line instead of showing half a sentence."""
    lines = (text or "").rstrip().splitlines()
    if len(lines) >= 3:
        return "\n".join(lines[:-1]).rstrip()
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if cut >= int(len(text) * 0.6):
        return text[:cut + 1].rstrip()
    return text


def _normalize_for_duplicate_check(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]+", "", t)
    return t


def _last_assistant_text(history: List[Dict[str, str]]) -> str:
    for item in reversed(history or []):
        role = (item.get("role") or "").strip()
        if role != "assistant":
            continue
        content = (item.get("content") or "").strip()
        if content:
            return content
    return ""


def _looks_repetitive_reply(candidate: str, history: List[Dict[str, str]]) -> bool:
    prev = _last_assistant_text(history)
    if not prev:
        return False

    a = _normalize_for_duplicate_check(candidate)
    b = _normalize_for_duplicate_check(prev)
    if not a or not b:
        return False

    if a == b:
        return True

    min_len = min(len(a), len(b))
    if min_len >= 120 and (a in b or b in a):
        return True

    return False


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _history_window(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    window = [h for h in (history or [])[-HISTORY_LIMIT:] if h.get("role") in ("user", "assistant")]
    last_assistant = max((i for i, h in enumerate(window) if h["role"] == "assistant"), default=-1)
    out: List[Dict[str, str]] = []
    for i, h in enumerate(window):
        if h["role"] == "user":
            limit = HISTORY_MAX_CHARS
        elif i == last_assistant:
            limit = HISTORY_LAST_ASSISTANT_MAX_CHARS
        else:
            limit = HISTORY_ASSISTANT_MAX_CHARS
        content = _trim_text(h.get("content") or "", limit)
        if content:
            out.append({"role": h["role"], "content": content})
    return out


def _is_repeated_question(user_text: str, history: List[Dict[str, Any]]) -> bool:
    prev = next((h.get("content") or "" for h in reversed(history or []) if h.get("role") == "user"), "")
    a = _normalize_for_duplicate_check(user_text)
    return bool(a) and a == _normalize_for_duplicate_check(prev)


def _build_llm_messages(
    locale: str,
    intent: str,
    summary: str,
    user_profile: Dict[str, Any],
    match_profile: Optional[Dict[str, Any]],
    distance_km: Optional[int],
    user_text: str,
    history: List[Dict[str, Any]],
    repeated_question: bool = False,
) -> List[Dict[str, str]]:
    # Most stable first, so consecutive requests share a cached prefix: global rules,
    # then this user's memory, then the per-intent instructions and facts.
    msgs: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT_BASE}]

    if summary:
        msgs.append({"role": "system", "content": f"SUMMARY_MEMORY: {_trim_text(summary, SUMMARY_MAX_CHARS)}"})

    msgs.append({"role": "system", "content": _build_intent_prompt(locale, intent)})

    uctx = _profile_context_compact(user_profile, intent, prefix="USER")
    if uctx:
        msgs.append({"role": "system", "content": uctx})

    if intent == Intent.ASTRO:
        ua = _compute_astro_short(user_profile, label="USER_ASTRO")
        if ua:
            msgs.append({"role": "system", "content": ua})

    if intent == Intent.MATCH and match_profile:
        mctx = _profile_context_compact(match_profile, intent, prefix="MATCH")
        if mctx:
            msgs.append({"role": "system", "content": mctx})

        ua = _compute_astro_short(user_profile, label="USER_ASTRO")
        ma = _compute_astro_short(match_profile, label="MATCH_ASTRO")
        if ua:
            msgs.append({"role": "system", "content": ua})
        if ma:
            msgs.append({"role": "system", "content": ma})
        if distance_km is not None:
            msgs.append({"role": "system", "content": f"DISTANCE_KM: {int(distance_km)}"})

    msgs.extend(_history_window(history))

    if repeated_question:
        msgs.append({
            "role": "system",
            "content": "The user sent the same message again. Answer it with fresh wording and a new angle; do not repeat your previous reply."
        })

    msgs.append({"role": "user", "content": _trim_text(user_text, USER_TEXT_MAX_CHARS)})
    return msgs


def _fix_trailing_garbage(txt: str) -> str:
    t = (txt or "").strip()
    if not t:
        return t

    lines = [x.rstrip() for x in t.splitlines()]
    while lines:
        last = lines[-1].strip()

        if last in ("•", "-", "—", "\"", "“", "”", "''", "'"):
            lines.pop()
            continue

        if last.startswith("•") and len(last.replace("•", "").strip()) == 0:
            lines.pop()
            continue

        if len(last) <= 1:
            lines.pop()
            continue

        break

    return "\n".join(lines).strip()


def _extract_reply_safe(txt: str) -> str:
    txt = (txt or "").strip()
    if not txt:
        return "…"

    txt = txt.replace("**", "").replace("```", "")

    if len(txt) > 6500:
        txt = txt[:6500].rstrip() + "…"

    return txt


def _append_profile_hint_if_needed(reply_text: str, user_profile: Dict[str, Any], intent: str, history: List[Dict[str, str]]) -> str:
    text = (reply_text or "").strip()
    if not text:
        return text

    if intent != Intent.ASTRO:
        return text

    birth_date = (user_profile.get("birthDate") or "").strip() if isinstance(user_profile.get("birthDate"), str) else ""
    birth_time = (user_profile.get("birthTime") or "").strip() if isinstance(user_profile.get("birthTime"), str) else ""
    birth_place = str(user_profile.get("birthCityName") or "").strip()

    has_birth_date = bool(birth_date)
    missing_time_or_place = not birth_time or not birth_place

    lower = text.lower()
    astroish = any(k in lower for k in [
        "horoscope", "vedic", "nakshatra", "rashi", "moon", "venus", "cosmic",
        "астро", "гороскоп", "накшатра", "раши", "венера", "луна"
    ])

    if astroish and has_birth_date and missing_time_or_place:
        hint = (
            "\n\nRemember, for a more precise reading next time, add your birth time and place in your profile 🪐"
        )
        prev_assistant = (_last_assistant_text(history) or "").lower()
        if len(text) + len(hint) <= 6900 and "birth time and place" not in lower and "birth time and place" not in prev_assistant:
            text += hint

    return text


# =========================
# PARROT QUOTA (SERVER-SIDE)
# =========================
# Cloud Functions `consumeParrotQuota` counts spent requests in
# users/{uid}.parrotQuotaUsedCount for the UTC day users/{uid}.parrotQuotaDayKey.
# The app calls it before every /ai-chat unless it considers the user premium
# (SubscriptionEntitlementService.IsPremiumActiveAsync). Here we count served
# LLM replies per UTC day and refuse to serve more than were paid for.
PARROT_USAGE_COLLECTION = "parrotServerUsage"


def _utc_day_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _to_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_premium_like_client(users_data: Dict[str, Any], profile_data: Dict[str, Any], now: datetime) -> bool:
    # Must match the app's IsPremiumActiveAsync: the app skips consumeParrotQuota for these users.
    if users_data.get("premiumActive") is True:
        return True
    promo_until = _to_utc_datetime(profile_data.get("promoPremiumUntilUtcIso"))
    return promo_until is not None and promo_until > now


def _paid_requests_today(users_data: Dict[str, Any], day_key: str) -> int:
    if str(users_data.get("parrotQuotaDayKey") or "").strip() != day_key:
        return 0
    return _to_int(users_data.get("parrotQuotaUsedCount"))


def _parrot_quota_refusal(served_today: int, paid_today: int, is_premium: bool, hard_cap: int) -> Optional[str]:
    """None when one more LLM reply may be served, otherwise the refusal reason."""
    if served_today >= hard_cap:
        return "daily_cap"
    if is_premium:
        return None
    if served_today >= paid_today:
        return "quota_exhausted"
    return None


def _reserve_parrot_reply(uid: str, users_data: Dict[str, Any], profile_data: Dict[str, Any]) -> Optional[str]:
    """Atomically counts one served reply. Returns the refusal reason, or None if allowed.
    users_data / profile_data are this request's users/{uid} and profiles/{uid} docs."""
    if not AI_CHAT_QUOTA_ENFORCED or firestore_client is None:
        return None

    try:
        day_key = _utc_day_key()
        is_premium = _is_premium_like_client(users_data, profile_data, datetime.now(timezone.utc))
        paid_today = _paid_requests_today(users_data, day_key)

        usage_ref = firestore_client.collection(PARROT_USAGE_COLLECTION).document(uid)

        @firestore.transactional
        def _txn(tx) -> Optional[str]:
            usage = usage_ref.get(transaction=tx).to_dict() or {}
            served = _to_int(usage.get("servedCount")) if usage.get("dayKey") == day_key else 0
            refusal = _parrot_quota_refusal(served, paid_today, is_premium, AI_CHAT_HARD_DAILY_CAP)
            if refusal is None:
                tx.set(usage_ref, {
                    "uid": uid,
                    "dayKey": day_key,
                    "servedCount": served + 1,
                    "updatedAtIso": _now_iso(),
                })
            return refusal

        try:
            return _txn(firestore_client.transaction())
        except (ValueError, google_exceptions.Aborted):
            # Contention here means parallel requests from one user (the app sends one at a time),
            # so refuse instead of failing open: otherwise a burst of requests skips the quota.
            logger.warning("Parrot quota transaction contention uid=%s", uid)
            return "busy"
    except Exception:
        # Fail open: a Firestore hiccup must not take the bot down.
        logger.exception("Parrot quota reservation failed uid=%s", uid)
        return None


def _parrot_quota_reply(reason: str) -> str:
    if reason == "busy":
        return "Parrot is catching its breath 🦜 Please send that again in a moment."
    if reason == "daily_cap":
        return "That’s a lot of chatting for one day 🦜 Let’s continue tomorrow!"
    return "You’re out of Parrot requests for today 🦜 Watch a short ad to unlock more, or get Parrot Plus for unlimited chats."


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

    if not _is_allowed_topic(user_text):
        reply = _topic_block_reply(user_text, locale)
        return AiChatResponse(reply_text=reply, blocked=True, reason="topic_blocked", thread_id=thread_id)

    match_uid = _parse_match_command(user_text)
    if match_uid and not _is_mutual_match(uid, match_uid):
        # /match puts the other person's profile and birth data into the prompt: matches only.
        return AiChatResponse(
            reply_text="I can only break down people you’ve matched with 🦜 Pick one of your matches above.",
            blocked=True, reason="not_a_match", thread_id=thread_id,
        )

    profile_raw, public_raw, user_raw = _load_user_docs(uid)
    is_identity = _is_identity_question(user_text)

    if not is_identity:
        refusal = _reserve_parrot_reply(uid, user_raw, profile_raw)
        if refusal:
            logger.info("Parrot reply refused uid=%s reason=%s", uid, refusal)
            return AiChatResponse(reply_text=_parrot_quota_reply(refusal), blocked=True, reason=refusal, thread_id=thread_id)

    user_profile_raw = _merge_dicts_prefer_first(profile_raw, public_raw, user_raw)
    user_profile = _safe_profile_dict(user_profile_raw)

    state = _load_chat_state(uid, thread_id)
    summary = _get_summary(state)
    history = _load_chat_history(uid, thread_id, limit=HISTORY_FETCH_LIMIT, state=state)

    intent = _infer_intent_with_history(user_text, history)

    match_profile = None
    distance_km = None

    if intent == Intent.MATCH:
        if not match_uid:
            reply = "Use this format: /match <uid> 🦜"
            return AiChatResponse(reply_text=reply, blocked=False, reason=None, thread_id=thread_id)

        match_profile_raw = _merge_dicts_prefer_first(*_load_user_docs(match_uid))
        match_profile = _safe_profile_dict(match_profile_raw)
        distance_km = _distance_km_from_profiles(user_profile_raw, match_profile_raw)

    repeated_question = _is_repeated_question(user_text, history)

    msgs = _build_llm_messages(
        locale=locale,
        intent=intent,
        summary=summary,
        user_profile=user_profile,
        match_profile=match_profile,
        distance_km=distance_km,
        user_text=user_text,
        history=history,
        repeated_question=repeated_question,
    )

    # Fold messages about to leave the prompt window into the memory, in parallel with the reply.
    unsummarized = _unsummarized(history, state)
    memory_job = None
    if not is_identity and len(unsummarized) >= HISTORY_LIMIT:
        memory_job = _memory_executor.submit(_summarize_memory, summary, unsummarized)

    if is_identity:
        assistant_text = _identity_reply(locale)
    else:
        max_tokens = _max_tokens_for_intent(intent)
        assistant_text = _complete_reply(msgs, max_tokens, kind=f"chat:{intent}")

        # A repeated question already asked for fresh wording above; otherwise retry once if
        # the model still echoed its previous answer.
        if not repeated_question and _looks_repetitive_reply(assistant_text, history):
            retry_msgs = list(msgs)
            retry_msgs.append({
                "role": "system",
                "content": "Previous draft repeats older assistant text. Rewrite with fresh wording and directly answer the latest user message. Do not copy previous paragraphs."
            })
            retry_text = _complete_reply(retry_msgs, max_tokens, kind="retry")
            if retry_text and not _looks_repetitive_reply(retry_text, history):
                assistant_text = retry_text

    assistant_text = _append_profile_hint_if_needed(assistant_text, user_profile, intent, history)

    created_at_ms = _now_ms()
    created_at_iso = _now_iso()
    _save_chat_message_batch(uid, thread_id, user_text, assistant_text, created_at_iso, created_at_ms)

    state_patch: Dict[str, Any] = {"turns": _to_int(state.get("turns")) + 1}
    if memory_job is not None:
        try:
            new_summary = memory_job.result(timeout=DS_TIMEOUT_SEC)
        except Exception:
            logger.exception("Memory update failed uid=%s", uid)
            new_summary = ""
        if new_summary:
            state_patch.update({
                "summary": new_summary,
                "summaryThroughMs": max(_to_int(m.get("ms")) for m in unsummarized),
                "summaryUpdatedAtIso": _now_iso(),
            })
    _save_chat_state(uid, thread_id, state_patch)

    return AiChatResponse(reply_text=assistant_text, blocked=False, reason=None, thread_id=thread_id)


def _complete_reply(msgs: List[Dict[str, str]], max_tokens: int, kind: str) -> str:
    text, finish_reason = _deepseek_complete(msgs, max_tokens=max_tokens, kind=kind)
    if finish_reason == "length":
        text = _drop_cut_off_tail(text)
    return _fix_trailing_garbage(_extract_reply_safe(text))


def _is_mutual_match(uid: str, other_uid: str) -> bool:
    if firestore_client is None:
        return False
    try:
        return firestore_client.collection("users").document(uid).collection("matches").document(other_uid).get().exists
    except Exception:
        logger.exception("Match lookup failed uid=%s", uid)
        return False


# =========================
# HISTORY + RESET
# =========================
@app.get("/history", response_model=HistoryResponse)
def history_endpoint(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = (thread_id or "default").strip() or "default"

    rows = _load_chat_history(uid, tid, limit=40)
    out: List[ChatTurn] = []
    for r in rows:
        role = (r.get("role") or "").strip()
        txt = _sanitize_legacy_history_text((r.get("content") or "").strip())
        if role in ("user", "assistant") and txt:
            out.append(ChatTurn(role=role, text=txt))

    return HistoryResponse(thread_id=tid, messages=out)


@app.post("/reset", response_model=ResetResponse)
def reset_endpoint(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = (thread_id or "default").strip() or "default"

    ms = _now_ms()
    _save_chat_state(uid, tid, {
        "clearedAtMs": ms, "updatedAtMs": ms, "updatedAtIso": _now_iso(),
        # A fresh start forgets the rolling memory too.
        "summary": "", "summaryThroughMs": ms,
    })
    return ResetResponse(thread_id=tid, ok=True)


# =========================
# DAILY HOROSCOPE
# =========================
# One personal Jyotish reading per uid per local day (tz from the app, IST by default), cached in
# dailyHoroscopes/{uid}__{dayKey} and added once to the user's Parrot chat. It does not use the Parrot quota:
# it is a once-a-day engagement feature, and repeat requests the same day cost no tokens.
HOROSCOPE_COLLECTION = "dailyHoroscopes"

_WEEKDAY_LORDS = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun"]  # datetime.weekday(): Monday=0
_TITHIS = [
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami",
    "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi",
]
# Tara bala: nakshatra count from the birth star to today's Moon star, mod 9.
_TARAS = [
    ("Janma", "mixed: a sensitive day, go gently"),
    ("Sampat", "supportive: good for gains and progress"),
    ("Vipat", "tricky: avoid risky moves"),
    ("Kshema", "supportive: steady wellbeing"),
    ("Pratyak", "tricky: expect some friction"),
    ("Sadhana", "supportive: effort pays off"),
    ("Naidhana", "tricky: slow down and keep it simple"),
    ("Mitra", "supportive: friendly, social energy"),
    ("Parama Mitra", "very supportive: a warm, lucky day"),
]
# Chandra gochara: today's Moon counted from the natal Moon sign.
_MOON_HOUSE_THEMES = {
    1: "mood, self-care and fresh starts",
    2: "money, family and how you speak",
    3: "courage, messages and initiative",
    4: "home, comfort and inner peace",
    5: "romance, creativity and fun",
    6: "work routines, health habits and small hurdles",
    7: "partnerships and one-to-one connections",
    8: "a sensitive day (Chandrashtama): patience and caution",
    9: "luck, blessings and learning",
    10: "career, visibility and duty",
    11: "gains, friends and wishes coming true",
    12: "rest, spending and letting go",
}
# Numerology planets 1-9 and their traditional colors.
_NUMBER_PLANETS = ["Sun", "Moon", "Jupiter", "Rahu", "Mercury", "Venus", "Ketu", "Saturn", "Mars"]
_PLANET_COLORS = {
    "Sun": "saffron orange", "Moon": "pearl white", "Jupiter": "golden yellow", "Rahu": "smoky grey",
    "Mercury": "emerald green", "Venus": "rose pink", "Ketu": "earthy brown", "Saturn": "deep blue",
    "Mars": "coral red",
}

_HOROSCOPE_SYSTEM_PROMPT = (
    "You are Shaadi Parrot 🦜, the warm Vedic astrology companion inside an Indian dating app.\n"
    "Write today's personal daily horoscope (Jyotish) for ONE user, using only the facts provided.\n"
    "Format: plain text, no markdown, no asterisks.\n"
    "Line 1: a short friendly greeting with the user's first name (if given) and today's weekday.\n"
    "Then six sections. Each is a heading line followed by 1–2 lines that start with '• ':\n"
    "🌙 Overall mood\n"
    "💞 Love & relationships (dating-app angle: chats, matches, first dates, honesty and kindness with a partner)\n"
    "💼 Career & money\n"
    "🌿 Health & energy\n"
    "🎨 Lucky color & number (use exactly LUCKY_COLOR and LUCKY_NUMBER)\n"
    "✨ Today's tip (one practical, specific action for today)\n"
    "Last line: one gentle sentence that astrology is guidance to apply with discernment.\n"
    "Rules:\n"
    "- 180–250 words in total. Warm, encouraging, specific, easy to read.\n"
    "- Ground the reading in the facts: name the Moon's transit house or sign, the tara, the tithi or the "
    "weekday lord at least twice, each briefly explained.\n"
    "- Emojis only at the start of the section headings, plus at most two elsewhere.\n"
    "- Never predict death, illness, accidents, breakups or guaranteed outcomes; no medical, legal or "
    "financial advice beyond everyday common sense.\n"
    "- Never mention being an AI, prompts, or data fields.\n"
    "- English only.\n"
)


def _tithi_name(sun_lon: float, moon_lon: float) -> str:
    idx = int(((moon_lon - sun_lon) % 360.0) // 12.0)  # 0..29
    if idx == 14:
        return "Purnima (full moon)"
    if idx == 29:
        return "Amavasya (new moon)"
    return ("Shukla " if idx < 15 else "Krishna ") + _TITHIS[idx % 15]


def _daily_sky(now_local: datetime) -> Dict[str, Any]:
    # Panchang takes the day's tithi and Moon at sunrise; 06:00 local is close enough for a daily reading.
    jd = _jd_ut(now_local.replace(hour=6, minute=0, second=0, microsecond=0))
    sun_lon = _calc_sidereal_lon_ut(jd, swe.SUN)
    moon_lon = _calc_sidereal_lon_ut(jd, swe.MOON)
    night_moon = _calc_sidereal_lon_ut(_jd_ut(now_local.replace(hour=23, minute=59, second=0, microsecond=0)), swe.MOON)
    moon_sign = _sign_from_lon(moon_lon)
    later_sign = _sign_from_lon(night_moon)
    return {
        "weekday": now_local.strftime("%A"),
        "date": f"{now_local.day} {now_local:%B %Y}",
        "lord": _WEEKDAY_LORDS[now_local.weekday()],
        "tithi": _tithi_name(sun_lon, moon_lon),
        "moon_sign": moon_sign,
        "moon_sign_idx": int(moon_lon // 30.0) % 12,
        "moon_nakshatra": _nakshatra_from_lon(moon_lon),
        "moon_nak_idx": int(moon_lon // (360.0 / 27.0)) % 27,
        "moon_sign_later": later_sign if later_sign != moon_sign else None,
    }


def _horoscope_facts(profile: Dict[str, Any], natal: Dict[str, Any], sky: Dict[str, Any]) -> str:
    house = (sky["moon_sign_idx"] - natal["moon_sign_idx"]) % 12 + 1
    tara_idx = ((sky["moon_nak_idx"] - natal["nakshatra_idx"]) % 27) % 9
    tara_name, tara_quality = _TARAS[tara_idx]
    lucky_planet = _NUMBER_PLANETS[tara_idx]

    user_bits = []
    for label, value in (
        ("first_name", profile.get("firstName") or profile.get("name")),
        ("gender", profile.get("gender")),
        ("looking_for", profile.get("relationshipIntent")),
    ):
        val = _flatten_value(value, max_len=60)
        if val:
            user_bits.append(f"{label}={val}")

    moon_line = f"MOON_TRANSIT: {sky['moon_sign']}, {sky['moon_nakshatra']} nakshatra"
    if sky["moon_sign_later"]:
        moon_line += f" (moves into {sky['moon_sign_later']} later today)"
    natal_line = f"NATAL: Moon sign (Rashi)={natal['moon_sign']}; Nakshatra={natal['nakshatra']}; Sun sign={natal['sun_sign']}; "
    natal_line += f"Lagna={natal['lagna']}" if natal["lagna"] else "Lagna unknown (no birth time or place)"

    lines = [
        "USER: " + ("; ".join(user_bits) or "unknown"),
        f"TODAY: {sky['weekday']}, {sky['date']}",
        f"WEEKDAY_LORD: {sky['lord']}",
        f"TITHI: {sky['tithi']}",
        moon_line,
        natal_line,
        f"MOON_FROM_NATAL_MOON: house {house}, theme: {_MOON_HOUSE_THEMES[house]}",
        f"TARA: {tara_name}, {tara_quality}",
        f"LUCKY_COLOR: {_PLANET_COLORS[lucky_planet]}",
        f"LUCKY_NUMBER: {tara_idx + 1}",
    ]
    return "\n".join(lines)


def _horoscope_ref(uid: str, day_key: str):
    return firestore_client.collection(HOROSCOPE_COLLECTION).document(f"{uid}__{day_key}")


def _store_daily_horoscope(uid: str, thread_id: str, day_key: str, tz_name: str, text: str) -> Tuple[str, bool]:
    """Caches today's reading and adds it to the chat once. Returns (text, already_cached)."""
    cache_ref = _horoscope_ref(uid, day_key)
    ms = _now_ms()
    iso = _now_iso()
    msg_id = f"{ms}_a_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}"

    # create() makes the whole batch fail if today's reading already exists, so parallel taps
    # end up with one reading and one chat message, without transaction lock contention.
    batch = firestore_client.batch()
    batch.create(cache_ref, {
        "uid": uid, "dayKey": day_key, "text": text, "tz": tz_name, "threadId": thread_id, "createdAtIso": iso,
    })
    batch.set(_chat_msgs_col_ref(uid, thread_id).document(msg_id), {
        "role": "assistant", "text": text, "createdAtIso": iso, "createdAtMs": ms,
        "kind": "daily_horoscope", "dayKey": day_key,
    })
    batch.set(_chat_doc_ref(uid, thread_id), {"uid": uid, "threadId": thread_id, "updatedAtIso": iso, "updatedAtMs": ms}, merge=True)
    try:
        batch.commit()
        return text, False
    except google_exceptions.AlreadyExists:
        existing = cache_ref.get().to_dict() or {}
        return existing.get("text") or text, True


@app.post("/daily-horoscope", response_model=DailyHoroscopeResponse, response_model_exclude_none=True)
def daily_horoscope(body: Optional[DailyHoroscopeRequest] = None, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    req = body or DailyHoroscopeRequest()
    tz = _zone(req.tz)
    now_local = datetime.now(tz)
    day_key = now_local.date().isoformat()
    thread_id = (req.thread_id or "").strip() or f"mobile_{uid}"
    unavailable = DailyHoroscopeResponse(ok=False, error="temporarily_unavailable")

    if firestore_client is None:
        return unavailable

    try:
        cached = _horoscope_ref(uid, day_key).get().to_dict() or {}
    except Exception:
        logger.exception("Daily horoscope cache read failed uid=%s", uid)
        return unavailable
    if cached.get("text"):
        return DailyHoroscopeResponse(ok=True, text=cached["text"], dayKey=day_key, cached=True)

    profile = _merge_dicts_prefer_first(*_load_user_docs(uid))
    if not _parse_birth_date(profile):
        return DailyHoroscopeResponse(ok=False, error="missing_birth_data")

    natal = _natal_chart(profile, fallback_tz=tz.key)
    if not natal:
        return unavailable

    try:
        msgs = [
            {"role": "system", "content": _HOROSCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": _horoscope_facts(profile, natal, _daily_sky(now_local))},
        ]
        text = _complete_reply(msgs, HOROSCOPE_MAX_TOKENS, kind="horoscope")
    except Exception:
        logger.exception("Daily horoscope generation failed uid=%s", uid)
        return unavailable
    if not text or text == "…":
        return unavailable

    try:
        text, was_cached = _store_daily_horoscope(uid, thread_id, day_key, tz.key, text)
    except Exception:
        logger.exception("Daily horoscope store failed uid=%s", uid)
        was_cached = False
    return DailyHoroscopeResponse(ok=True, text=text, dayKey=day_key, cached=was_cached)


# =========================
# DAILY FATES
# =========================
# Three paths a day (Stars, Heart, Home): picking, kundli matching and the AI verdict live in fates_*.py.
import face_live  # noqa: E402
import face_match  # noqa: E402

FACE_VERIFY_DAILY_ATTEMPTS = int(os.getenv("FACE_VERIFY_DAILY_ATTEMPTS") or "6")


def _take_face_attempt(uid: str) -> bool:
    """Up to FACE_VERIFY_DAILY_ATTEMPTS tries a day (each costs three Vision calls)."""
    return _take_counter(uid, "faceVerifyAttempts", FACE_VERIFY_DAILY_ATTEMPTS)


FACE_RECHECKS_PER_DAY = 20
FACE_CHALLENGE_TTL_SEC = 600
FACE_METHOD = "challenge+photos"
FACE_MODEL = "sface_2021dec"


def _face_ref(uid: str):
    return firestore_client.collection("faceVerifications").document(uid)


def _counter_used(uid: str, field: str) -> int:
    cur = ((firestore_client.collection("users").document(uid).get().to_dict() or {}).get(field) or {})
    return int(cur.get("n") or 0) if cur.get("day") == datetime.now(timezone.utc).date().isoformat() else 0


def _take_challenge(uid: str, challenge_id: str) -> Optional[Dict[str, Any]]:
    """The challenge issued by /verify-face-start, if it's this one, unused and not expired. Used up here."""
    from google.cloud import firestore as fs
    ref = _face_ref(uid)

    @fs.transactional
    def take(tx) -> Optional[Dict[str, Any]]:
        ch = (ref.get(transaction=tx).to_dict() or {}).get("challenge") or {}
        if not challenge_id or ch.get("id") != challenge_id or ch.get("used"):
            return None
        try:
            if datetime.fromisoformat(str(ch.get("expiresAtIso"))) < datetime.now(timezone.utc):
                return None
        except ValueError:
            return None
        tx.set(ref, {"challenge": {"used": True}}, merge=True)
        return ch

    return take(firestore_client.transaction())


_gcs_client = None


def _gcs():
    """Cloud Storage (the local Storage emulator when FIREBASE_STORAGE_EMULATOR_HOST is set)."""
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage as gcs
        emu = (os.getenv("FIREBASE_STORAGE_EMULATOR_HOST") or "").strip()
        if emu:
            os.environ.setdefault("STORAGE_EMULATOR_HOST", emu if "://" in emu else "http://" + emu)
            from google.auth.credentials import AnonymousCredentials
            _gcs_client = gcs.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or "demo", credentials=AnonymousCredentials())
        else:
            _gcs_client = gcs.Client()
    return _gcs_client


def _verification_bucket(uid: str) -> Optional[str]:
    rec = (_face_ref(uid).get().to_dict() or {}) if firestore_client else {}
    for u in [rec.get("frontSelfie")] + list(rec.get("selfies") or []) + list(rec.get("photos") or []):
        obj = face_live.storage_object(str(u or ""))
        if obj:
            return obj[0]
    return os.getenv("FIREBASE_STORAGE_BUCKET") or None


def _clean_selfies(uid: str, keep: Optional[str] = None, bucket: Optional[str] = None) -> int:
    """Deletes everything in users/{uid}/verification/ except the selfie kept for re-checks. Returns how many."""
    kept = face_live.storage_object(keep or "")
    name = bucket or (kept[0] if kept else None) or _verification_bucket(uid)
    if not name:
        return 0
    n = 0
    try:
        for blob in _gcs().bucket(name).list_blobs(prefix=f"users/{uid}/verification/"):
            if kept and blob.name == kept[1]:
                continue
            blob.delete()
            n += 1
    except Exception:
        logger.exception("selfie cleanup failed")
    return n


def _fresh_uploads(urls: List[str], issued_iso: str) -> bool:
    """The selfies were uploaded after the challenge was issued (photos made before can't be replayed).
    If Storage can't be asked, the check is skipped (logged)."""
    try:
        issued = datetime.fromisoformat(issued_iso) - timedelta(seconds=60)      # clock skew
        for u in urls:
            obj = face_live.storage_object(u)
            blob = _gcs().bucket(obj[0]).get_blob(obj[1]) if obj else None
            if blob is None or blob.time_created is None:
                raise ValueError("no upload time")
            if blob.time_created < issued:
                return False
        return True
    except Exception:
        logger.warning("selfie upload time not checked", exc_info=True)
        return True


_TEMPLATES_TTL_SEC = 600            # look for new or changed templates this often
_TEMPLATES_FULL_SEC = 24 * 3600     # and read them all once a day (picks up deletions made on other instances)
_templates: Dict[str, Any] = {"at": 0.0, "full": 0.0, "since": "", "v": {}}


def _verified_templates() -> Dict[str, Any]:
    """uid -> face template of every verified account, for the duplicate check. The whole set is read once a
    day per instance; in between only templates changed since the last look (a few reads instead of all)."""
    import numpy as np
    now = time.time()
    if now - _templates["at"] <= _TEMPLATES_TTL_SEC:
        return _templates["v"]
    full = now - _templates["full"] > _TEMPLATES_FULL_SEC
    started = _now_iso()
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        col = firestore_client.collection("faceTemplates")
        # A range on one field only (no composite index needed); the status is checked here.
        q = col if full else col.where(filter=FieldFilter("updatedAtIso", ">=", _templates["since"]))
        found: Dict[str, Any] = {} if full else dict(_templates["v"])
        for d in q.select(["v", "status"]).stream():
            x = d.to_dict() or {}
            v = x.get("v")
            if x.get("status") == "verified" and isinstance(v, list) and v:
                found[d.id] = np.asarray(v, dtype=np.float32)
            else:
                found.pop(d.id, None)
        _templates.update(at=now, since=started, v=found, **({"full": now} if full else {}))
    except Exception:
        logger.exception("face templates load failed")
    return _templates["v"]


def _store_template(uid: str, vec: Any, status: str) -> None:
    import numpy as np
    v = np.asarray(vec, dtype=np.float32).ravel()
    try:
        firestore_client.collection("faceTemplates").document(uid).set({
            "v": [round(float(x), 6) for x in v], "status": status, "model": FACE_MODEL, "updatedAtIso": _now_iso()})
        if status == "verified":
            _templates["v"][uid] = v
        else:
            _templates["v"].pop(uid, None)
    except Exception:
        logger.exception("face template write failed")


def _own_photos(uid: str) -> List[str]:
    pub = (firestore_client.collection("publicProfiles").document(uid).get().to_dict() or {}) if firestore_client else {}
    photos = [str(p) for p in (pub.get("photos") or []) if isinstance(p, str) and p.strip()][:6]
    return [p for p in photos if face_live.own_photo_url(p, uid)]


def _photo_faces(urls: List[str]) -> List[List[Any]]:
    out = []
    for u in urls:
        try:
            out.append(face_match.face_features(face_match.decode(_download_image_bytes(u, max_mb=12))))
        except Exception:
            logger.warning("face match: photo unreadable", exc_info=True)
            out.append([])
    return out


_COHORT_TTL_SEC = 6 * 3600
_COHORT_SIZE = 40
_cohort_cache: Dict[str, Tuple[float, List[Tuple[str, Any]]]] = {}


def _cohort(uid: str) -> List[Any]:
    """Faces from the main photos of up to 40 other people of the same gender, for face_match's look-alike check.
    Kept in this instance's memory only (never stored), refreshed every few hours."""
    import time as _time
    if firestore_client is None:
        return []
    me = firestore_client.collection("publicProfiles").document(uid).get().to_dict() or {}
    gender = str(me.get("gender") or "").strip().lower()
    hit = _cohort_cache.get(gender)
    if not hit or _time.time() - hit[0] > _COHORT_TTL_SEC:
        faces: List[Tuple[str, Any]] = []
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            q = firestore_client.collection("publicProfiles").where(filter=FieldFilter("isDiscoverable", "==", True))
            if gender:
                q = q.where(filter=FieldFilter("gender", "==", me.get("gender")))
            for d in q.limit(_COHORT_SIZE * 2).stream():
                if len(faces) >= _COHORT_SIZE:
                    break
                photos = [p for p in ((d.to_dict() or {}).get("photos") or []) if isinstance(p, str)]
                if not photos or not face_live.own_photo_url(photos[0], d.id):
                    continue
                try:
                    found = face_match.face_features(face_match.decode(_download_image_bytes(photos[0], max_mb=12)))
                except Exception:
                    continue
                if found:
                    faces.append((d.id, found[0]))
        except Exception:
            logger.exception("face cohort build failed")
        hit = (_time.time(), faces)
        _cohort_cache[gender] = hit
    return [f for owner, f in hit[1] if owner != uid]


def _cohort_scores(uid: str, front: Any) -> List[float]:
    return [face_match.similarity(front, f) for f in _cohort(uid)]


def _save_face_result(uid: str, status: str, reason: str, photos: List[str], record: Dict[str, Any],
                      review: Optional[Dict[str, Any]] = None) -> None:
    """status: "ok", "failed" or "in_review" (a person on the team decides; not verified meanwhile).
    profiles: the flags + which photos were checked (the app can't write these); publicProfiles: the badge;
    faceVerifications/{uid} (server only): the kept selfie, the scores and what a reviewer needs."""
    if firestore_client is None:
        return
    from google.cloud import firestore as fs
    ok = status == "ok"
    now = _now_iso()
    try:
        firestore_client.collection("profiles").document(uid).set({
            "faceVerified": ok, "isFaceVerified": ok, "faceVerifiedReason": reason,
            "faceVerifiedAtIso": now, "faceVerifiedMethod": FACE_METHOD,
            "faceVerifiedPhotos": photos if ok else [],
        }, merge=True)
        firestore_client.collection("publicProfiles").document(uid).set({"isFaceVerified": ok}, merge=True)
        _face_ref(uid).set({**record, "status": status, "reason": reason, "updatedAtIso": now,
                            "review": ({**review, "requestedAtIso": now} if review else fs.DELETE_FIELD)}, merge=True)
    except Exception:
        logger.exception("face verification result write failed")


def _take_counter(uid: str, field: str, limit: int) -> bool:
    if firestore_client is None:
        return True
    from google.cloud import firestore as fs
    ref = firestore_client.collection("users").document(uid)
    day = datetime.now(timezone.utc).date().isoformat()

    @fs.transactional
    def take(tx) -> bool:
        cur = (ref.get(transaction=tx).to_dict() or {}).get(field) or {}
        n = int(cur.get("n") or 0) if cur.get("day") == day else 0
        if n >= limit:
            return False
        tx.set(ref, {field: {"day": day, "n": n + 1}}, merge=True)
        return True

    try:
        return take(firestore_client.transaction())
    except Exception:
        logger.exception("face counter failed")
        return True


@app.post("/verify-face-start")
def verify_face_start(body: Optional[VerifyFaceStartRequest] = None, authorization: Optional[str] = Header(default=None)):
    """A new random challenge: a straight selfie, then two of "turn right", "turn left", "smile" in a random
    order. Valid for 10 minutes and for one try. Needs the person's consent to the notice shown in the app
    (DPDP: the selfies and face templates are personal data); which notice and when is kept as the record."""
    uid = _verify_firebase_token_or_401(authorization)
    consent = re.sub(r"[^A-Za-z0-9_.-]", "", str((body.consent if body else "") or ""))[:40]
    if not consent:
        raise HTTPException(status_code=400, detail="consent_needed")
    if firestore_client is None or not face_match.available():
        raise HTTPException(status_code=503, detail="face_match_unavailable")
    if _counter_used(uid, "faceVerifyAttempts") >= FACE_VERIFY_DAILY_ATTEMPTS:
        raise HTTPException(status_code=429, detail="too_many_attempts")
    now = datetime.now(timezone.utc)
    ch = {"id": secrets.token_urlsafe(12), "steps": face_live.new_challenge(), "issuedAtIso": now.isoformat(),
          "expiresAtIso": (now + timedelta(seconds=FACE_CHALLENGE_TTL_SEC)).isoformat(), "used": False}
    _face_ref(uid).set({"challenge": ch, "consent": {"version": consent, "atIso": now.isoformat()}}, merge=True)
    return {"challengeId": ch["id"], "steps": ch["steps"], "expiresInSec": FACE_CHALLENGE_TTL_SEC}


def _judge_live(uid: str, steps: List[str], urls: List[str], issued_iso: str):
    """-> (status, reason, record, review, template). Liveness first (Cloud Vision, the challenge), then the face
    (OpenCV): the selfies are one person, that person is in the profile photos, and not on another account."""
    data = [_download_image_bytes(u, max_mb=10) for u in urls]
    record: Dict[str, Any] = {"steps": steps, "posesOk": False, "attemptAtIso": _now_iso()}
    if len({hashlib.sha256(b).hexdigest() for b in data}) != len(data):
        return "failed", "three_selfies_needed", record, None, None
    if not _fresh_uploads(urls, issued_iso):
        return "failed", "take_new_selfies", record, None, None
    shots = [(*_detect_faces_and_safety_from_bytes(b),) for b in data]
    ok, reason, pans = face_live.judge_challenge(steps, shots)
    record["pans"] = [round(p, 1) for p in pans]
    if not ok:
        return "failed", reason, record, None, None
    images = [face_match.decode(b) for b in data]
    photos = _own_photos(uid)
    photo_faces = _photo_faces(photos)
    # The frames as the camera gave them and mirrored (front cameras mirror, profile photos may not): the
    # better of the two judgements counts, all three frames always the same way round.
    tries = []
    for flip in (False, True):
        faces = [face_match.face_features(face_match.mirror(im) if flip else im) for im in images]
        if not faces[0]:
            continue
        front = faces[0][0]
        ok, reason, scores = face_match.judge(front, [f[0] for f in faces[1:] if f], photo_faces, _cohort_scores(uid, front))
        tries.append((face_match.outcome_rank(ok, reason, scores), flip, front, ok, reason, scores))
    if not tries:
        return "failed", "no_face_detected_1", record, None, None
    _, flipped, front, ok, reason, scores = max(tries, key=lambda t: t[0])
    record.update({"posesOk": True, "photos": photos, "mirrored": flipped})
    record["scores"] = scores
    if reason == "selfies_not_same_person":
        record["posesOk"] = False                       # nothing worth keeping for a re-check
    if face_match.needs_review(reason):
        return "in_review", reason, record, {"kind": "photos"}, front
    if not ok:
        return "failed", reason, record, None, None
    dups = face_match.duplicates(front, _verified_templates(), exclude=uid)
    if dups:
        record["duplicates"] = [{"uid": u, "sim": round(s, 3)} for u, s in dups[:3]]
        return "in_review", "review_duplicate", record, {"kind": "duplicate", "duplicateOf": record["duplicates"]}, front
    return "ok", "ok", record, None, front


@app.post("/verify-face-live")
def verify_face_live(body: VerifyFaceLiveRequest, authorization: Optional[str] = Header(default=None)):
    """The challenge's selfies. Passing marks the profile verified (the server writes it, never the app); just
    under a bar, or the same face on another account, goes to a person on the team ("in_review").
    Afterwards only the straight selfie is kept (for re-checks after a photo change); the rest is deleted."""
    uid = _verify_firebase_token_or_401(authorization)
    urls = [str(u or "").strip() for u in (body.selfies or [])]
    if len(urls) != 3 or len(set(urls)) != 3 or not all(face_live.selfie_url_ok(u, uid) for u in urls):
        raise HTTPException(status_code=400, detail="bad_selfies")
    if firestore_client is None or not face_match.available():
        raise HTTPException(status_code=503, detail="face_match_unavailable")
    ch = _take_challenge(uid, str(body.challengeId or ""))
    if ch is None:
        raise HTTPException(status_code=400, detail="challenge_expired")
    if not _take_face_attempt(uid):
        raise HTTPException(status_code=429, detail="too_many_attempts")
    steps = [str(s) for s in (ch.get("steps") or [])]
    status, reason, record, review, front = _judge_live(uid, steps, urls, str(ch.get("issuedAtIso") or ""))
    keep = urls[0] if record.get("posesOk") else None
    record.update({"frontSelfie": keep or "", "selfies": [keep] if keep else []})
    if front is not None:
        _store_template(uid, front, "verified" if status == "ok" else "pending")
    _save_face_result(uid, status, reason, record.get("photos") or [], record,
                      {**review, "frontSelfie": keep, "photos": record.get("photos") or []} if review else None)
    _clean_selfies(uid, keep, bucket=(face_live.storage_object(urls[0]) or (None,))[0])
    return {"ok": status == "ok", "status": status, "reason": reason}


@app.post("/verify-face-recheck")
def verify_face_recheck(authorization: Optional[str] = Header(default=None)):
    """After the photos changed: the new photos against the selfie kept from the last verification, no new
    selfies needed. Restores (or removes) the verified flags."""
    uid = _verify_firebase_token_or_401(authorization)
    if firestore_client is None or not face_match.available():
        raise HTTPException(status_code=503, detail="face_match_unavailable")
    rec = firestore_client.collection("faceVerifications").document(uid).get().to_dict() or {}
    front_url = str(rec.get("frontSelfie") or "")
    if not rec.get("posesOk") or not face_live.selfie_url_ok(front_url, uid):
        return {"ok": False, "reason": "selfies_needed"}
    if not _take_counter(uid, "faceRecheckAttempts", FACE_RECHECKS_PER_DAY):
        raise HTTPException(status_code=429, detail="too_many_attempts")
    img = face_match.decode(_download_image_bytes(front_url, max_mb=10))
    photos = _own_photos(uid)
    photo_faces = _photo_faces(photos)
    tries = []
    for flip in (False, True):          # as taken and mirrored, like the first check
        f = face_match.face_features(face_match.mirror(img) if flip else img)
        if f:
            ok, reason, scores = face_match.judge(f[0], [], photo_faces, _cohort_scores(uid, f[0]))
            tries.append((face_match.outcome_rank(ok, reason, scores), f, ok, reason, scores))
    if not tries:
        return {"ok": False, "reason": "selfies_needed"}
    _, front, ok, reason, scores = max(tries, key=lambda t: t[0])
    status = "ok" if ok else ("in_review" if face_match.needs_review(reason) else "failed")
    _save_face_result(uid, status, reason, photos, {"scores": scores, "photos": photos, "recheckedAtIso": _now_iso()},
                      {"kind": "photos", "frontSelfie": front_url, "photos": photos} if status == "in_review" else None)
    if ok:
        tpl = firestore_client.collection("faceTemplates").document(uid).get().to_dict() or {}
        if tpl.get("status") != "verified":
            _store_template(uid, front[0], "verified")
    return {"ok": ok, "status": status, "reason": reason}


@app.post("/verify-face-withdraw")
def verify_face_withdraw(authorization: Optional[str] = Header(default=None)):
    """"Remove my verification": deletes the selfies, the face template and the scores; the badge goes. Only
    the consent record stays (when it was given and withdrawn, no images), until the account is deleted."""
    uid = _verify_firebase_token_or_401(authorization)
    if firestore_client is None:
        raise HTTPException(status_code=503, detail="unavailable")
    bucket = _verification_bucket(uid)
    _clean_selfies(uid, None, bucket=bucket)
    firestore_client.collection("faceTemplates").document(uid).delete()
    _templates["v"].pop(uid, None)
    consent = (_face_ref(uid).get().to_dict() or {}).get("consent") or {}
    _face_ref(uid).set({"status": "withdrawn", "reason": "withdrawn", "updatedAtIso": _now_iso(),
                        "consent": {**consent, "withdrawnAtIso": _now_iso()}})
    firestore_client.collection("profiles").document(uid).set({
        "faceVerified": False, "isFaceVerified": False, "faceVerifiedReason": "withdrawn",
        "faceVerifiedAtIso": _now_iso(), "faceVerifiedPhotos": []}, merge=True)
    firestore_client.collection("publicProfiles").document(uid).set({"isFaceVerified": False}, merge=True)
    return {"ok": True}


import fates_api  # noqa: E402

FATES_AI_TIMEOUT_SEC = int(os.getenv("FATES_AI_TIMEOUT_SEC") or "14")

fates_api.configure(fates_api.Deps(
    db=lambda: firestore_client,
    verify_uid=_verify_firebase_token_or_401,
    complete=lambda messages, max_tokens, temperature, kind: _deepseek_complete(
        messages, max_tokens, temperature=temperature, kind=kind, timeout=FATES_AI_TIMEOUT_SEC),
    resolve_place=_resolve_birth_place,
    is_premium=_is_premium_like_client,
))
app.include_router(fates_api.router)
