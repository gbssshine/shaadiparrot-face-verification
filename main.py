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

import astro_core
import astro_festivals
import astro_notes
import photo_fetch

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
class VerifyFaceStartRequest(BaseModel):
    consent: str = ""           # the version of the notice the person agreed to in the app (e.g. "face-2026-10")


class VerifyFaceLiveRequest(BaseModel):
    challengeId: str            # from /verify-face-start
    selfies: List[str]          # one per challenge step, in order (the user's own Storage uploads)
    # A profile made by a parent or relative: the person themself does the face check and agrees to the profile
    # on the same screen ("I'm Arjun: I agree"). Recorded as their consent only when the check passes.
    personConsent: bool = False
    consentTextVersion: str = ""


class VerifyPhotoRequest(BaseModel):
    gcs_uri: str
    require_face: bool = True


Role = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: Role
    text: str
    # What a server-written message is: 'daily_horoscope', 'mithu_note'. None for a normal reply.
    kind: Optional[str] = None


class AiChatRequest(BaseModel):
    text: str
    locale: Optional[str] = "en"
    mode: Optional[str] = "shaadi_parrot"
    thread_id: Optional[str] = "default"
    history: Optional[List[ChatTurn]] = None  # ignored on server
    # What the app can show (1.0.8+: ["actions"]). Without "actions" the birth hint stays a line of text.
    caps: Optional[List[str]] = None


class AiChatResponse(BaseModel):
    reply_text: str
    blocked: bool = False
    reason: Optional[str] = None
    thread_id: Optional[str] = None
    # For apps that sent caps ["actions"]: buttons under the reply, e.g. [{"type": "add_birth_time", "label": ...}].
    actions: Optional[List[Dict[str, str]]] = None
    # 'birth_details' whenever the reply asks for the birth time or place (as a button or as the text line).
    hint: Optional[str] = None


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
# One HTTP connection pool for DeepSeek and image downloads (a new TLS handshake per call was slow under load).
from requests.adapters import HTTPAdapter  # noqa: E402

_HTTP = requests.Session()
_HTTP.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=64))
_HTTP.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=16))

# Every handler is sync and runs in AnyIO's thread pool (40 by default). Calls that wait on DeepSeek or Vision
# hold a thread, so with a small pool one slow upstream made every other endpoint wait. More threads (waiting is
# cheap: the GIL is released) plus a cap on parallel DeepSeek calls, so a slow DeepSeek can only take part of them.
THREADPOOL_SIZE = int(os.getenv("THREADPOOL_SIZE") or "96")
DEEPSEEK_MAX_PARALLEL = int(os.getenv("DEEPSEEK_MAX_PARALLEL") or "32")
_deepseek_slots = threading.BoundedSemaphore(DEEPSEEK_MAX_PARALLEL)


@app.on_event("startup")
async def _bigger_thread_pool():
    import anyio
    anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_SIZE


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
        "I can’t help with recipes 🦜 But I can help with: rishtas and marriage, your kundli, family life, your biodata, today's panchang. Ask me one of those 🙂",
        "Not a cooking parrot 😅 I’m best at: rishtas, kundli, family and Vedic astrology. What do you want today? 🦜",
    ]
    variants_ru = [
        "С рецептами не помогу 🦜 Но могу: ришта и брак, твоя кундли, семья, биодата, панчанг на сегодня. Спроси 🙂",
        "Я не кулинарный попугай 😅 Зато топ в: ришта, кундли, семье и астрологии. Что делаем? 🦜",
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


# Hindi astro words, in Devanagari and in the Roman spellings families type ("Hinglish"). Only words an English
# question wouldn't use, so English replies stay as they were.
_HINDI_ASTRO_WORDS = [
    "राशिफल", "राशि", "कुंडली", "कुण्डली", "नक्षत्र", "ज्योतिष", "ग्रह", "भविष्य", "पंचांग", "मुहूर्त",
    "मांगलिक", "मंगल दोष", "गुण मिलान", "आज का दिन", "कल का दिन",
    "rashifal", "kundali", "janam kundli", "jyotish", "bhavishya", "gun milan", "guna milan",
    "aaj ka din", "kal ka din",
]


def _infer_intent(user_text: str) -> str:
    if _parse_match_command(user_text):
        return Intent.MATCH

    t = (user_text or "").lower()

    astro_words = [
        "horoscope", "forecast", "daily fate", "vedic", "kundli", "nakshatra", "rashi",
        "tomorrow", "tmr", "zodiac", "moon sign",
        "гороскоп", "прогноз", "сегодня", "ведичес", "накшатр", "раши", "кундли", "совместим"
    ] + _HINDI_ASTRO_WORDS
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
        "гороскоп", "прогноз", "накшат", "раши", "астро",
        "राशि", "नक्षत्र", "कुंडली", "कुण्डली", "ज्योतिष", "rashifal", "kundali", "jyotish",
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
            "Я Митху, попугай-астролог Shaadi Parrot 🦜\n"
            "• Читаю кундли и сверяю гуны пары\n"
            "• Помогаю с риштой, семьёй, работой и деньгами\n"
            "• Каждый день подсказываю по панчангу и звёздам\n"
            "Напиши, что сейчас происходит — разберём по шагам ✨"
        )

    return (
        "I’m Mithu, the Shaadi Parrot astrologer 🦜\n"
        "• I read kundlis and check the gunas of a pair\n"
        "• I help with rishtas, family, work and money\n"
        "• I give daily guidance from the panchang and your stars\n"
        "Tell me what’s happening, and I’ll break it down step by step ✨"
    )


# =========================
# PROMPT BUILDER
# =========================
# Sent byte-identical as the first message of every chat request, so DeepSeek serves it
# from its prefix cache. Anything per-intent or per-user goes in later messages.
_SYSTEM_PROMPT_BASE = (
    "You are Mithu 🦜, the warm, witty Vedic astrologer parrot of Shaadi Parrot, an Indian matrimony and family "
    "astrology app: kundli matching, rishtas, panchang and family life.\n"
    "You help with marriage and rishta questions (with the family involved), kundli and gunas, married life, in-laws, "
    "children, work, money and everyday life.\n"
    "Rules:\n"
    "- Be warm, confident, practical, a bit playful; respect Indian family values.\n"
    "- Follow the user's life_stage and profileFor in USER_CONTEXT: single = finding a life partner (a rishta, families "
    "involved); profileFor son/daughter/brother/sister/relative/friend = a parent or relative searching for that person, "
    "so speak to them about that person; newly_married / married = harmony at home and with the in-laws; "
    "married_kids = the family and the children. Never suggest casual dating or hook-ups.\n"
    "- No fear: never predict death, illness, accidents, divorce or guaranteed outcomes; no paid pujas or remedies; "
    "doshas are explained calmly with their cancellations.\n"
    "- Respect every faith. Use the user's own festival names. If ASTRO_MODE is off, don't bring up astrology "
    "unless asked.\n"
    "- Give a festival or fast time only when TODAY_OBSERVANCE has it; never guess one.\n"
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


# Families often write in Hindi: Mithu answers in the language and script of the question (English stays English).
CHAT_LANGUAGE_RULE = (
    "Reply in the language and script the user wrote in (Hindi in Devanagari if they wrote Hindi; never Roman "
    "Hinglish unless they did). Mithu speaks of himself in the masculine."
)


def _build_intent_prompt(locale: str, intent: str) -> str:
    lang = (locale or "en").strip().lower() or "en"
    base = ""

    if intent == Intent.MATCH:
        base += (
            "Task: produce a FULL rishta breakdown for two people (USER + MATCH) who accepted each other's interest.\n"
            "Structure (short headings + bullets):\n"
            "1) The short answer\n"
            "2) Where they click (values, family, lifestyle, goals)\n"
            "3) What to talk about early (expectations, family, career, city, children)\n"
            "4) Families: how the two families may see this rishta, respectfully\n"
            "5) Vedic notes (only from the provided ASTRO lines)\n"
            "6) Distance & logistics (based on distance_km)\n"
            "7) Good first questions to ask each other (3–6)\n"
            "8) A respectful next step (a call, then the families meet)\n"
            "Finish with 1 verdict line: 'Worth taking forward' or 'Go slowly' + 1 emoji.\n"
            "Output target: 45–90 short bullet lines total.\n"
        )
    else:
        base += (
            "- If asked for a message/reply: output 2–4 respectful message options (matrimony, families involved).\n"
            "- If asked for profile/biodata: give actionable edits + 1 sample 'about me' for a matrimony biodata.\n"
            "- If asked for daily fate/horoscope: give a Vedic-style daily forecast + practical tips.\n"
        )

        if intent == Intent.TEXTING:
            base += "Output target: 8–14 short bullet lines total.\n"
        elif intent == Intent.PROFILE:
            base += "Output target: 10–16 bullet lines + 1 short sample bio.\n"
        elif intent == Intent.ASTRO:
            base += "Output target: 12–18 bullet lines with sections: Energy, Love & family, Work, Practical tips, Mantra.\n"
        else:
            base += "Output target: 9–15 short bullet lines.\n"

    if lang.startswith("ru"):
        base += "Reply in Russian.\n"
    else:
        base += CHAT_LANGUAGE_RULE + "\n"

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

    base_keys = ["firstName", "age", "gender", "lifeStage", "profileFor", "maritalStatus", "cityName", "countryName", "relationshipIntent", "languages"]
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
        # No birth details here: they are the other person's private data. The MATCH_ASTRO line (signs only,
        # computed on the server) carries what the breakdown needs. birth_keys stay only in astro_keys (my own).
        keys = list(dict.fromkeys(base_keys + [
            "bio", "aboutMe", "interests", "tags", "workout", "smoking", "drinking",
            "education", "jobTitle", "occupation"
        ]))
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
# READER, ASTRO MODE, TODAY'S OBSERVANCE (facts for the chat and the horoscope)
# =========================
# profileFor values of a profile made by a parent or relative for someone else.
MANAGED_PROFILE_FOR = frozenset({"son", "daughter", "brother", "sister", "relative", "friend"})
ASTRO_MODES = ("full", "fun", "off")
# Faiths whose people are asked first ("Kundli, your way") and get the light 'fun' mode until they answer.
_ASK_FIRST_FAITHS = frozenset({"muslim", "christian", "sikh", "parsi", "jewish", "buddhist", "spiritual"})


def _faith_key(religion: Any) -> str:
    """'Parsi / Zoroastrian' -> 'parsi', 'No religion / Spiritual' -> 'spiritual', 'Hindu' -> 'hindu'."""
    r = str(religion or "").strip().lower()
    if not r:
        return ""
    if "spiritual" in r or r.startswith("no religion"):
        return "spiritual"
    head = r.split("/")[0].split()
    return head[0] if head else ""


def _astro_mode(profile: Dict[str, Any]) -> str:
    """users.astroMode when the user chose one; else 'fun' for ask-first faiths (by religion or the festivals they
    celebrate) and 'full' for everyone else."""
    mode = str(profile.get("astroMode") or "").strip().lower()
    if mode in ASTRO_MODES:
        return mode
    celebrates = profile.get("celebrates")
    faiths = {_faith_key(profile.get("religion"))}
    if isinstance(celebrates, list):
        faiths |= {str(c or "").strip().lower() for c in celebrates}
    return "fun" if faiths & _ASK_FIRST_FAITHS else "full"


def _is_managed(profile: Dict[str, Any]) -> bool:
    return str(profile.get("profileFor") or "").strip().lower() in MANAGED_PROFILE_FOR


def _first_name(profile: Dict[str, Any]) -> str:
    return _flatten_value(profile.get("firstName") or profile.get("name"), max_len=40)


def _reader_line(profile: Dict[str, Any]) -> str:
    """Who reads the horoscope: the person themself, or the family that made the profile for them."""
    if not _is_managed(profile):
        return "READER: self"
    return f"READER: family of {_first_name(profile) or 'this person'} (the profile was made by a parent or relative)"


OBSERVANCE_MAX_CHARS = 520   # astro_festivals.observance_for gives at most 400, plus the city times added here


def _observance_text(observance: Optional[str]) -> str:
    """Today's festival or fast for the user's own faith (the caller looks it up), or 'none'."""
    return _trim_text(_normalize_text(observance or ""), OBSERVANCE_MAX_CHARS) or "none"


_ASTRO_MODE_NOTES = {
    "full": "full",
    "fun": "fun (keep it light, entertainment framing)",
    "off": "off (no astrology unless the user asks)",
}
# In the horoscope the reader asked for the reading, so 'off' needs no note there.
_HOROSCOPE_MODE_NOTES = {**_ASTRO_MODE_NOTES, "off": "off"}


def _day_facts(profile: Dict[str, Any], observance: Optional[str] = None) -> str:
    """The chat's per-user facts line: ASTRO_MODE and TODAY_OBSERVANCE."""
    return f"ASTRO_MODE={_ASTRO_MODE_NOTES[_astro_mode(profile)]} | TODAY_OBSERVANCE={_observance_text(observance)}"


# =========================
# TODAY_OBSERVANCE (engagement N3): today's own festival or fast for the chat and the daily horoscope
# =========================
# A fast whose time is known for a city: the Pradosh window. (A Sankashti row already carries its moonrise.)
_FAST_TIME_KINDS = {"pradosh": "pradosh"}


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _reader_place(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Where the reader lives, for today's vrats and festival times: {name, lat, lon, tz}, or None when unknown
    (then there are no fasts and no times: nothing is guessed). The profile's city from the places database (its
    coordinates when the profile has them), else the city nearest to the profile's coordinates."""
    city = " ".join(str(doc.get("cityName") or doc.get("city") or "").split())[:60]
    lat, lon = doc.get("lat"), doc.get("lon")
    point = _finite(lat) and _finite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
    try:
        if city:
            iso2 = str(doc.get("countryIso2") or "").strip().upper()[:2]
            country = str(doc.get("countryName") or "")
            found = _resolve_birth_place({"birthCityName": city, "birthStateName": doc.get("stateName") or "",
                                          "birthCountryIso2": iso2 or ("" if country.strip() else "IN"),
                                          "birthCountryName": country})
            if found and found.get("precision") in ("city", "state") and found.get("tz"):
                if point:
                    return {"name": city, "lat": round(float(lat), 3), "lon": round(float(lon), 3), "tz": found["tz"]}
                return {"name": city, "lat": round(float(found["lat"]), 3), "lon": round(float(found["lon"]), 3),
                        "tz": found["tz"]}
        if point:
            la, lo = float(lat), float(lon)
            rows = _places_query("SELECT city, lat, lon, tz, pop FROM cities WHERE lat BETWEEN ? AND ? "
                                 "AND lon BETWEEN ? AND ?", (la - 0.3, la + 0.3, lo - 0.3, lo + 0.3))
            if rows:
                # the closest place, a big city winning over a village a little nearer (as astro_api does)
                best = min(rows, key=lambda r: ((r[1] - la) ** 2 + (r[2] - lo) ** 2)
                           / (1.0 + min(int(r[4] or 0), 5_000_000) / 50_000.0))
                return {"name": " ".join(w.capitalize() for w in str(best[0]).split()), "lat": round(la, 3),
                        "lon": round(lo, 3), "tz": str(best[3] or HOROSCOPE_DEFAULT_TZ)}
    except Exception:
        logger.warning("reader place lookup failed", exc_info=True)
    return None


def _first_text(doc: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _with_city_times(line: str, day: Any, place: Optional[Dict[str, Any]],
                     fasts: Optional[List[Dict[str, Any]]] = None) -> str:
    """Adds the city time to each of today's festivals (and Pradosh fasts) in the line that has one: "Karva Chauth:
    The fast ends at moonrise... Moonrise in Delhi: 20:12." The time comes from the panchang for the reader's own
    place; without a place, or for a day without a time kind, the line stays as observance_for wrote it."""
    if not place or not line or line == "none":
        return line
    iso = day.isoformat()
    timed = [(str(e["name"]), e) for e in astro_festivals.FESTIVALS if e.get("date") == iso and e.get("timeKind")]
    for v in fasts or []:
        kind = _FAST_TIME_KINDS.get(str(v.get("kind") or ""))
        if kind and v.get("date") == iso and v.get("name"):
            timed.append((str(v["name"]), {"timeKind": kind}))
    if not timed:
        return line
    parts = line.split(" | ")
    for i, part in enumerate(parts):
        for name, row in timed:
            if part == name or part.startswith(name + ":") or part.startswith(name + "."):
                when = astro_notes._city_time(row, day, place)
                if when:
                    parts[i] = part.rstrip() + " " + when
                break
    return " | ".join(parts)


def _today_observance(doc: Dict[str, Any], day: Any = None) -> str:
    """TODAY_OBSERVANCE for one reader: astro_festivals.observance_for(day, users.celebrates, community,
    religion=..., state=..., fasts=...) with today's vrats only when the reader's place is known, plus the city
    time of a festival that has one. `doc` is the merged profiles + publicProfiles + users doc; `day` is the
    reader's local date (the reader's own place's, else India's). Never raises: "none" on any problem."""
    try:
        place = _reader_place(doc)
        if day is None:
            day = datetime.now(_zone(place["tz"] if place else None)).date()
        fasts = astro_core.vrats_between(day, 1, place["lat"], place["lon"], place["tz"]) if place else None
        line = astro_festivals.observance_for(
            day, doc.get("celebrates"), _first_text(doc, "community", "settings_community"),
            religion=_first_text(doc, "religion", "settings_religion"),
            state=_first_text(doc, "stateName", "birthStateName"), fasts=fasts)
        return _with_city_times(line, day, place, fasts)
    except Exception:
        logger.warning("today's observance failed", exc_info=True)
        return "none"


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


# "I don't know my birth time" answered with a part of the day (users.birthTimeApprox): the Moon is taken at that
# hour instead of noon, the same hour as the charts (astro_core.approx_time). It only sharpens the Moon; a Lagna
# still needs a real time.
def _natal_chart(profile: Dict[str, Any], fallback_tz: str = HOROSCOPE_DEFAULT_TZ) -> Optional[Dict[str, Any]]:
    """Sidereal (Lahiri) natal basics. Unknown birth time -> local noon (or the approximate part of the day the
    user picked) and no lagna."""
    bd = _parse_birth_date(profile)
    if not bd:
        return None
    bt = _parse_birth_time(profile)
    hour, minute = bt or astro_core.approx_time(profile.get("birthTimeApprox")) or (12, 0)
    place = _resolve_birth_place(profile)
    tz = _zone(place.get("tz") if place else None, fallback=_zone(fallback_tz).key)
    try:
        jd = _jd_ut(datetime(bd[0], bd[1], bd[2], hour, minute, tzinfo=tz))
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
    """Oldest-first rows {role, content, ms, kind?}; pass `state` when the caller already loaded it."""
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

            row = {"role": role, "content": text, "ms": ms}
            kind = d.get("kind")
            if isinstance(kind, str) and kind.strip():
                row["kind"] = kind.strip()[:40]     # 'daily_horoscope', 'mithu_note': /history passes it on
            rows.append(row)

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
    "You maintain the long-term memory that Mithu, the matrimony and family Vedic astrology assistant of Shaadi Parrot, "
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


PHOTO_CHECKS_PER_DAY = int(os.getenv("PHOTO_CHECKS_PER_DAY") or "60")


def _own_gcs_photo(gcs_uri: str, uid: str) -> bool:
    """gs://{bucket}/users/{uid}/... only, in one of the project's own buckets (face_live.photo_buckets): a caller
    can't have someone else's (or any other bucket's) object read."""
    m = re.match(r"^gs://([a-z0-9][a-z0-9._-]{1,220})/(.+)$", gcs_uri)
    return (bool(m) and m.group(1) in face_live.photo_buckets() and m.group(2).startswith(f"users/{uid}/")
            and ".." not in m.group(2))


def _take_rate(uid: str, key: str, limit: int, per: str = "day") -> bool:
    """A counter in serverRate/{uid} (no client can read or write that collection), per day or per minute.
    Fails open: a Firestore hiccup must not take a feature down. For AI cost caps use _take_rate_strict."""
    return _rate_counter(uid, key, limit, per, fail_open=True)


def _take_rate_strict(uid: str, key: str, limit: int, per: str = "day") -> bool:
    """The same counter, but it fails closed: no Firestore or any error means "no". For every AI cost cap
    (a Firestore outage must not turn into unlimited model calls)."""
    return _rate_counter(uid, key, limit, per, fail_open=False)


def _rate_counter(uid: str, key: str, limit: int, per: str, fail_open: bool) -> bool:
    if firestore_client is None:
        return fail_open
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%dT%H:%M") if per == "minute" else now.date().isoformat()

    def take(tx) -> bool:
        cur = (ref.get(transaction=tx).to_dict() or {}).get(key) or {}
        n = int(cur.get("n") or 0) if cur.get("day") == day else 0
        if n >= limit:
            return False
        tx.set(ref, {key: {"day": day, "n": n + 1}}, merge=True)
        return True

    try:
        ref = firestore_client.collection("serverRate").document(uid)
        return _run_transaction(take)
    except Exception:
        logger.exception("rate counter failed (%s)", "open" if fail_open else "closed")
        return fail_open


def _run_transaction(work):
    """Runs work(transaction) in a Firestore transaction (retried on contention) and returns its result."""
    from google.cloud import firestore as fs
    return fs.transactional(work)(firestore_client.transaction())


@app.post("/verify-photo")
def verify_photo(body: VerifyPhotoRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    if vision_client is None:
        raise HTTPException(status_code=503, detail="Vision client not available")

    gcs_uri = (body.gcs_uri or "").strip()
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="gcs_uri must start with gs://")
    if not _own_gcs_photo(gcs_uri, uid):
        raise HTTPException(status_code=403, detail="not_your_photo")
    if not _take_rate(uid, "photoChecks", PHOTO_CHECKS_PER_DAY):
        raise HTTPException(status_code=429, detail="too_many_photo_checks")

    try:
        image = vision.Image(source=vision.ImageSource(gcs_image_uri=gcs_uri))
        resp = vision_client.annotate_image({
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {"type_": vision.Feature.Type.FACE_DETECTION},
            ],
        }, timeout=20)
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


FACE_FETCH_SEC = 20.0      # one selfie or profile photo download for the face checks, in all


def _download_image_bytes(url: str, max_mb: int = 10) -> bytes:
    """A selfie or profile photo for the face checks, bounded (photo_fetch): no redirects (the links are the
    user's own Storage objects, which never redirect), at most max_mb (12 at most) MB within FACE_FETCH_SEC in all,
    and refused when its header says more than 40 megapixels, before anything decodes it."""
    headers = {"User-Agent": "shaadiparrot-face-verification/1.0"}
    try:
        data = photo_fetch.fetch(url, max_bytes=max_mb * 1024 * 1024, total_sec=FACE_FETCH_SEC, timeout=(5, 10),
                                 session=_HTTP, headers=headers)
    except photo_fetch.FetchError as e:
        if e.reason == "too_large":
            raise HTTPException(status_code=413, detail=f"Image too large (>{max_mb}MB)")
        if e.reason == "status":
            raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {e.status}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e.reason}")
    if len(data) < 2000:
        raise HTTPException(status_code=400, detail="Downloaded file is too small / invalid")
    if photo_fetch.too_many_pixels(data):
        raise HTTPException(status_code=413, detail="Image too large (pixels)")
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
        }, timeout=20)
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

    # A slot first: when DeepSeek is slow, at most DEEPSEEK_MAX_PARALLEL calls wait on it; the rest give up
    # quickly (callers fall back to templates or "try again") instead of piling up threads.
    if not _deepseek_slots.acquire(timeout=8):
        logger.warning("DeepSeek busy: no free slot kind=%s", kind)
        raise HTTPException(status_code=503, detail="DeepSeek busy")
    try:
        r = _HTTP.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=(5, timeout or DS_TIMEOUT_SEC))
    except Exception as e:
        logger.exception("DeepSeek request failed")
        raise HTTPException(status_code=502, detail=f"DeepSeek request error: {type(e).__name__}")
    finally:
        _deepseek_slots.release()

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


def _family_context(uid: str) -> str:
    """The family the user added in the app (users/{uid}/family): who they are, their age and Moon sign, so
    Mithu can talk about 'your daughter' by name. Ages, not birth dates: the exact dates (children's too) don't
    need to reach the LLM provider. Empty when there is none."""
    if firestore_client is None:
        return ""
    try:
        docs = list(firestore_client.collection("users").document(uid).collection("family").limit(8).stream())
    except Exception:
        logger.warning("family context read failed", exc_info=True)
        return ""
    today = datetime.now(timezone.utc).date()
    parts = []
    for d in docs:
        m = d.to_dict() or {}
        name = _trim_text(str(m.get("name") or ""), 30)
        rel = _trim_text(str(m.get("relation") or ""), 20)
        if not name:
            continue
        bits = [rel] if rel else []
        bd = fates_astro_chart._parse_date(m.get("birthDate"))
        if bd:
            y, mo, dd = bd
            age = today.year - y - ((today.month, today.day) < (mo, dd))
            if 0 <= age <= 120:
                bits.append("a baby" if age == 0 else f"age {age}")
            try:
                c = fates_astro_chart.chart_from_birth(bd, None, None)
                bits.append("Moon " + _ZODIAC[c["rashi"]])
            except Exception:
                pass
        parts.append(f"{name} ({', '.join(bits)})" if bits else name)
    return ("FAMILY (people the user cares about; use their names when it helps): " + "; ".join(parts)) if parts else ""


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
    family_ctx: str = "",
    observance: Optional[str] = None,
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
    if family_ctx:
        msgs.append({"role": "system", "content": family_ctx})
    msgs.append({"role": "system", "content": _day_facts(user_profile, observance)})

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


# Words that make a reply an astrology reading (English, Russian, Hindi in Devanagari and in Roman letters).
_ASTROISH_REPLY_WORDS = [
    "horoscope", "vedic", "nakshatra", "rashi", "moon", "venus", "cosmic",
    "астро", "гороскоп", "накшатра", "раши", "венера", "луна",
    "राशि", "नक्षत्र", "कुंडली", "कुण्डली", "ग्रह", "राशिफल", "चंद्र", "चन्द्र", "शुक्र", "ज्योतिष",
    "rashifal", "kundali", "jyotish", "chandra", "shukra",
]

BIRTH_HINT = "birth_details"
BIRTH_ACTION_EVERY_DAYS = 14        # the "Add my birth time" button: at most once per 14 days per chat


def _missing_birth_detail(reply_text: str, user_profile: Dict[str, Any], intent: str) -> Optional[str]:
    """'time' or 'place' when an astro reply would be more exact with that birth detail (the birth date is
    known, the time or the place isn't); None when no hint applies."""
    text = (reply_text or "").strip()
    if not text or intent != Intent.ASTRO:
        return None

    birth_date = (user_profile.get("birthDate") or "").strip() if isinstance(user_profile.get("birthDate"), str) else ""
    birth_time = (user_profile.get("birthTime") or "").strip() if isinstance(user_profile.get("birthTime"), str) else ""
    birth_place = str(user_profile.get("birthCityName") or "").strip()
    if not birth_date or (birth_time and birth_place):
        return None

    lower = text.lower()
    if not any(k in lower for k in _ASTROISH_REPLY_WORDS):
        return None
    return "time" if not birth_time else "place"


def _append_profile_hint_if_needed(reply_text: str, user_profile: Dict[str, Any], intent: str, history: List[Dict[str, str]]) -> str:
    """Apps without caps ["actions"] (1.0.7): the birth hint as a line of text, which the app turns into a button
    by its marker 'birth time and place'. Not twice in a row."""
    text = (reply_text or "").strip()
    if not _missing_birth_detail(text, user_profile, intent):
        return text

    hint = (
        "\n\nRemember, for a more precise reading next time, add your birth time and place in your profile 🪐"
    )
    lower = text.lower()
    prev_assistant = (_last_assistant_text(history) or "").lower()
    if len(text) + len(hint) <= 6900 and "birth time and place" not in lower and "birth time and place" not in prev_assistant:
        text += hint

    return text


def _birth_actions(missing: str, state: Dict[str, Any], now: datetime) -> Optional[List[Dict[str, str]]]:
    """Apps with caps ["actions"]: a button instead of the text line, at most once per BIRTH_ACTION_EVERY_DAYS per
    chat (state asked.birth_time); None while that time hasn't passed."""
    asked = _to_utc_datetime(((state or {}).get("asked") or {}).get("birth_time"))
    if asked is not None and now - asked < timedelta(days=BIRTH_ACTION_EVERY_DAYS):
        return None
    label = "Add my birth time" if missing == "time" else "Add my birth place"
    return [{"type": "add_birth_time", "label": label}]


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


AI_CHAT_PER_MINUTE = int(os.getenv("AI_CHAT_PER_MINUTE") or "8")


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

    # A burst limit on top of the daily quota: a script (or a stuck retry loop) can't fire DeepSeek calls.
    if not _take_rate(uid, "aiChatMinute", AI_CHAT_PER_MINUTE, per="minute"):
        return AiChatResponse(reply_text=_parrot_quota_reply("busy"), blocked=True, reason="slow_down", thread_id=thread_id)

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
        # The "who are you" reply is canned: no family read and no charts for it.
        family_ctx="" if is_identity else _family_context(uid),
        observance=None if is_identity else _today_observance(user_profile_raw),
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

    state_patch: Dict[str, Any] = {"turns": _to_int(state.get("turns")) + 1}

    # The birth hint: a button for apps that can show one (caps "actions"), the old text line for 1.0.7.
    actions: Optional[List[Dict[str, str]]] = None
    hint: Optional[str] = None
    missing = None if is_identity else _missing_birth_detail(assistant_text, user_profile, intent)
    if missing and _wants_actions(body.caps):
        actions = _birth_actions(missing, state, datetime.now(timezone.utc))
        if actions:
            hint = BIRTH_HINT
            state_patch["asked"] = {"birth_time": _now_iso()}
    elif missing:
        with_hint = _append_profile_hint_if_needed(assistant_text, user_profile, intent, history)
        if with_hint != assistant_text.strip():
            hint = BIRTH_HINT
        assistant_text = with_hint

    created_at_ms = _now_ms()
    created_at_iso = _now_iso()
    _save_chat_message_batch(uid, thread_id, user_text, assistant_text, created_at_iso, created_at_ms)

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

    return AiChatResponse(reply_text=assistant_text, blocked=False, reason=None, thread_id=thread_id,
                          actions=actions, hint=hint)


def _wants_actions(caps: Optional[List[str]]) -> bool:
    return any(isinstance(c, str) and c.strip().lower() == "actions" for c in (caps or [])[:20])


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
# exclude_none: a message without a kind looks exactly as before ({role, text}).
@app.get("/history", response_model=HistoryResponse, response_model_exclude_none=True)
def history_endpoint(thread_id: str = "default", authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    tid = (thread_id or "default").strip() or "default"

    rows = _load_chat_history(uid, tid, limit=40)
    out: List[ChatTurn] = []
    for r in rows:
        role = (r.get("role") or "").strip()
        txt = _sanitize_legacy_history_text((r.get("content") or "").strip())
        if role in ("user", "assistant") and txt:
            out.append(ChatTurn(role=role, text=txt, kind=r.get("kind") or None))

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
# Chandra gochara: today's Moon counted from the natal Moon sign. One table with the week forecast.
_MOON_HOUSE_THEMES = astro_core.HOUSE_THEMES
# Numerology planets 1-9 and their traditional colors.
_NUMBER_PLANETS = ["Sun", "Moon", "Jupiter", "Rahu", "Mercury", "Venus", "Ketu", "Saturn", "Mars"]
_PLANET_COLORS = {
    "Sun": "saffron orange", "Moon": "pearl white", "Jupiter": "golden yellow", "Rahu": "smoky grey",
    "Mercury": "emerald green", "Venus": "rose pink", "Ketu": "earthy brown", "Saturn": "deep blue",
    "Mars": "coral red",
}

_HOROSCOPE_SYSTEM_PROMPT = (
    "You are Mithu, the Shaadi Parrot 🦜, the warm Vedic astrology companion of an Indian app for kundli, family and rishta.\n"
    "Write today's personal daily horoscope (Jyotish) for ONE user, using only the facts provided.\n"
    "Format: plain text, no markdown, no asterisks.\n"
    "Line 1: a short friendly greeting and today's weekday; with the user's first name (if given) when READER is "
    "self, without any name when READER is family.\n"
    "Line 2: 'Theme: ' followed by today's theme in at most 7 words.\n"
    "Then five sections. Each is a heading line followed by 1–2 lines that start with '• ':\n"
    "🌙 Overall mood\n"
    "💞 Love & family (follow life_stage: single = meeting someone, rishta talks, honesty; relationship = the "
    "partner; married = the spouse and the in-laws; married_kids = the spouse and the children; none given = love "
    "and the people close to them)\n"
    "💼 Career & money\n"
    "🌿 Health & energy\n"
    "🎨 Lucky color & number (use exactly LUCKY_COLOR and LUCKY_NUMBER)\n"
    "Then two short lists. Each is a heading line followed by exactly 2 lines that start with '• ', each line at "
    "most 8 words:\n"
    "✅ Do\n"
    "⛔ Avoid (gentle everyday cautions, never fear)\n"
    "Last line: one gentle sentence that astrology is guidance to apply with discernment.\n"
    "Rules:\n"
    "- 180–250 words in total. Warm, encouraging, specific, easy to read.\n"
    "- Ground the reading in the facts: name the Moon's transit house or sign, the tara, the tithi or the "
    "weekday lord at least twice, each briefly explained.\n"
    "- Emojis only at the start of the section headings, plus at most two elsewhere.\n"
    "- Never predict death, illness, accidents, breakups or guaranteed outcomes; no medical, legal or "
    "financial advice beyond everyday common sense.\n"
    "- If READER is family, write about the person named in READER in the third person ('Ananya's day…'). "
    "Greet the reader warmly without a name. Give tips the family can act on.\n"
    "- If ASTRO_MODE is fun, keep it light: entertainment framing.\n"
    "- Respect every faith. If TODAY_OBSERVANCE names a festival or fast, wish the reader for it once, by its own "
    "name; if it is none, don't bring up festivals. Give a clock time for it only when TODAY_OBSERVANCE has one.\n"
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


def _horoscope_facts(profile: Dict[str, Any], natal: Dict[str, Any], sky: Dict[str, Any],
                     observance: Optional[str] = None) -> str:
    """The facts the reading is written from. `observance`: today's festival or fast for the reader's own faith
    (looked up by the caller), else "none"."""
    house = (sky["moon_sign_idx"] - natal["moon_sign_idx"]) % 12 + 1
    tara_idx = ((sky["moon_nak_idx"] - natal["nakshatra_idx"]) % 27) % 9
    tara_name, tara_quality = _TARAS[tara_idx]
    lucky_planet = _NUMBER_PLANETS[tara_idx]

    user_bits = []
    for label, value in (
        ("first_name", profile.get("firstName") or profile.get("name")),
        ("gender", profile.get("gender")),
        ("looking_for", profile.get("relationshipIntent")),
        ("life_stage", profile.get("lifeStage")),
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
        _reader_line(profile),
        f"ASTRO_MODE: {_HOROSCOPE_MODE_NOTES[_astro_mode(profile)]}",
        f"TODAY_OBSERVANCE: {_observance_text(observance)}",
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


HOROSCOPE_LEASE_SEC = 45


def _take_lease(name: str, ttl_sec: float) -> bool:
    """serverLeases/{name}: True for the one caller that may do the work now. A lease older than ttl_sec (its
    holder died or gave up) can be taken over. Without Firestore everyone may (local tests)."""
    if firestore_client is None:
        return True
    from google.cloud import firestore as fs
    ref = firestore_client.collection("serverLeases").document(name)
    now = time.time()

    @fs.transactional
    def take(tx) -> bool:
        snap = ref.get(transaction=tx)
        at = float((snap.to_dict() or {}).get("at") or 0)
        if snap.exists and now - at < ttl_sec:
            return False
        tx.set(ref, {"at": now, "atIso": _now_iso()})
        return True

    try:
        return take(firestore_client.transaction())
    except Exception:
        logger.exception("lease failed: %s", name)
        return True


def _release_lease(name: str) -> None:
    """The work failed: the next caller may try at once."""
    if firestore_client is None:
        return
    try:
        firestore_client.collection("serverLeases").document(name).delete()
    except Exception:
        logger.warning("lease release failed: %s", name, exc_info=True)


HOROSCOPE_REGENS_PER_DAY = 2


def _birth_sig(profile: Dict[str, Any]) -> str:
    """A short signature of the birth details a reading is built on. When it changes the same day (the user
    added their birth time, fixed the date, picked an approximate time), today's reading is written again."""
    raw = "|".join(str(profile.get(k) or "") for k in ("birthDate", "birthTime", "birthCityName", "birthTimeApprox"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _horoscope_messages(profile: Dict[str, Any], natal: Dict[str, Any], sky: Dict[str, Any],
                        observance: Optional[str] = None) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _HOROSCOPE_SYSTEM_PROMPT},
        {"role": "user", "content": _horoscope_facts(profile, natal, sky, observance)},
    ]


def _store_daily_horoscope(uid: str, thread_id: str, day_key: str, tz_name: str, text: str,
                           birth_sig: str = "") -> Tuple[str, bool]:
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
        "birthSig": birth_sig, "regenCount": 0,
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


def _store_regenerated_horoscope(uid: str, thread_id: str, day_key: str, sig: str, text: str) -> Tuple[str, bool]:
    """One transaction: today's reading becomes the new text (a set, not create(): the doc exists) with the new
    birth signature and regenCount + 1, plus one chat message flagged updated. Returns (text, written_now).
    A reading another request already wrote for these birth details wins; past the daily limit nothing is written."""
    ref = _horoscope_ref(uid, day_key)
    ms = _now_ms()
    iso = _now_iso()
    msg_ref = _chat_msgs_col_ref(uid, thread_id).document(f"{ms}_a_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}")
    chat_ref = _chat_doc_ref(uid, thread_id)

    def work(tx) -> Tuple[str, bool]:
        cur = ref.get(transaction=tx).to_dict() or {}
        if cur.get("text") and cur.get("birthSig") == sig:
            return cur["text"], False
        n = _to_int(cur.get("regenCount"))
        if n >= HOROSCOPE_REGENS_PER_DAY:
            return cur.get("text") or text, False
        tx.set(ref, {"uid": uid, "dayKey": day_key, "text": text, "birthSig": sig, "regenCount": n + 1,
                     "updatedAtIso": iso}, merge=True)
        tx.set(msg_ref, {
            "role": "assistant", "text": text, "createdAtIso": iso, "createdAtMs": ms,
            "kind": "daily_horoscope", "dayKey": day_key, "updated": True,
        })
        tx.set(chat_ref, {"uid": uid, "threadId": thread_id, "updatedAtIso": iso, "updatedAtMs": ms}, merge=True)
        return text, True

    return _run_transaction(work)


def _regenerate_daily_horoscope(uid: str, thread_id: str, day_key: str, tz: ZoneInfo, now_local: datetime,
                                profile: Dict[str, Any], cached: Dict[str, Any], sig: str,
                                observance: Optional[str] = None) -> Optional[Tuple[str, bool]]:
    """Today's reading was written for other birth details: write it again, at most HOROSCOPE_REGENS_PER_DAY
    times a day (regenCount on the doc and the horoscope_regen counter). Returns (text, written_now), or None
    to keep serving the cached reading."""
    if _to_int(cached.get("regenCount")) >= HOROSCOPE_REGENS_PER_DAY or not _parse_birth_date(profile):
        return None
    # The lease first: parallel taps wait for one rewrite instead of each spending the daily counter.
    lease = f"horoscope_{uid}_{day_key}_{sig}"
    if not _take_lease(lease, HOROSCOPE_LEASE_SEC):
        for _ in range(24):
            time.sleep(0.5)
            try:
                cur = _horoscope_ref(uid, day_key).get().to_dict() or {}
            except Exception:
                cur = {}
            if cur.get("text") and cur.get("birthSig") == sig:
                return cur["text"], False
        return None
    if not _take_rate(uid, "horoscope_regen", HOROSCOPE_REGENS_PER_DAY, "day"):
        _release_lease(lease)
        return None

    natal = _natal_chart(profile, fallback_tz=tz.key)
    text = ""
    if natal:
        try:
            text = _complete_reply(_horoscope_messages(profile, natal, _daily_sky(now_local), observance),
                                   HOROSCOPE_MAX_TOKENS, kind="horoscope_regen")
        except Exception:
            logger.exception("Daily horoscope regen failed uid=%s", uid)
            text = ""
    if not text or text == "…":
        _release_lease(lease)
        return None
    try:
        return _store_regenerated_horoscope(uid, thread_id, day_key, sig, text)
    except Exception:
        logger.exception("Daily horoscope regen store failed uid=%s", uid)
        return text, True


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
        served = DailyHoroscopeResponse(ok=True, text=cached["text"], dayKey=day_key, cached=True)
        if not cached.get("birthSig"):
            return served       # written before birth signatures existed: kept as it is for the rest of that day
        profile = _merge_dicts_prefer_first(*_load_user_docs(uid))
        sig = _birth_sig(profile)
        if cached["birthSig"] == sig:
            return served
        fresh = _regenerate_daily_horoscope(uid, thread_id, day_key, tz, now_local, profile, cached, sig,
                                            observance=_today_observance(profile, now_local.date()))
        if not fresh:
            return served
        return DailyHoroscopeResponse(ok=True, text=fresh[0], dayKey=day_key, cached=not fresh[1])

    profile = _merge_dicts_prefer_first(*_load_user_docs(uid))
    if not _parse_birth_date(profile):
        return DailyHoroscopeResponse(ok=False, error="missing_birth_data")

    # Parallel calls (two taps, a retry) wrote one reading but each paid for a DeepSeek call. Now one request
    # takes a lease and writes it; the others wait a little for that reading.
    if not _take_lease(f"horoscope_{uid}_{day_key}", HOROSCOPE_LEASE_SEC):
        for _ in range(24):
            time.sleep(0.5)
            try:
                cached = _horoscope_ref(uid, day_key).get().to_dict() or {}
            except Exception:
                cached = {}
            if cached.get("text"):
                return DailyHoroscopeResponse(ok=True, text=cached["text"], dayKey=day_key, cached=True)
        return unavailable

    lease = f"horoscope_{uid}_{day_key}"
    natal = _natal_chart(profile, fallback_tz=tz.key)
    if not natal:
        _release_lease(lease)
        return unavailable

    try:
        observance = _today_observance(profile, now_local.date())
        text = _complete_reply(_horoscope_messages(profile, natal, _daily_sky(now_local), observance),
                               HOROSCOPE_MAX_TOKENS, kind="horoscope")
    except Exception:
        logger.exception("Daily horoscope generation failed uid=%s", uid)
        _release_lease(lease)
        return unavailable
    if not text or text == "…":
        _release_lease(lease)
        return unavailable

    try:
        text, was_cached = _store_daily_horoscope(uid, thread_id, day_key, tz.key, text, birth_sig=_birth_sig(profile))
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
import fates_engine  # noqa: E402  (own_private_photos: the face check's photo list, one rule for both)

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


PHOTOS_PRIVATE = "accepted"     # profiles.photoPrivacy: real photos only after acceptance, blurs on the card


def _own_photos(uid: str) -> List[str]:
    """The user's real profile photos, which the face check compares with. While the photos are private the card
    holds only blurs, so the links come from the profile; otherwise from the card. The first 6 http links, THEN
    only the user's own Storage photos (face_live.own_photo_url: never a blur, a variant or another host): the
    one rule fates_engine.own_private_photos applies too, so the faceVerifiedPhotos saved from this list always
    cover the photos Daily Fates checks against them (a 7th own photo never stands in for a non-own link)."""
    if firestore_client is None:
        return []
    refs = [firestore_client.collection(name).document(uid) for name in ("profiles", "publicProfiles")]
    by_path = {snap.reference.path: (snap.to_dict() or {}) for snap in firestore_client.get_all(refs) if snap.exists}
    profile, card = (by_path.get(ref.path, {}) for ref in refs)
    source = profile if str(profile.get("photoPrivacy") or "").strip() == PHOTOS_PRIVATE else card
    return fates_engine.own_private_photos(uid, source)


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


def _person_consent_fields(profile_for: Any, person_consent: bool, text_version: Any, status: str,
                           now_iso: str) -> Optional[Dict[str, str]]:
    """The person's own consent to a profile a parent or relative made for them, given on the face-check screen.
    Recorded only for a passed check on a managed profile; None otherwise."""
    if status != "ok" or not person_consent:
        return None
    if str(profile_for or "").strip().lower() not in MANAGED_PROFILE_FOR:
        return None
    return {
        "personConsentAtIso": now_iso,
        "personConsentVia": "face_check",
        "personConsentTextVersion": re.sub(r"[^A-Za-z0-9_.-]", "", str(text_version or ""))[:40],
    }


def _save_face_result(uid: str, status: str, reason: str, photos: List[str], record: Dict[str, Any],
                      review: Optional[Dict[str, Any]] = None,
                      person_consent: Optional[Dict[str, str]] = None) -> None:
    """status: "ok" (verified), "failed" or "in_review" (on hold: only the same face as a banned account).
    Verified first, looked at later: a pass that the server wasn't fully sure about is verified at once and
    carries `review` + needsLook, so the team can look in the web admin and remove it if it's wrong.
    profiles: the flags + which photos were checked (the app can't write these); publicProfiles: the badge;
    faceVerifications/{uid} (server only): the kept frame, the scores and what a reviewer needs.
    One batch, so the badge and the private flags never disagree.
    person_consent (a passed check on a managed profile, _person_consent_fields): the person's own consent goes on
    profiles (server-only fields, never on the card) and is copied into faceVerifications/{uid}, in the same batch."""
    if firestore_client is None:
        return
    from google.cloud import firestore as fs
    ok = status == "ok"
    now = _now_iso()
    consent = dict(person_consent or {}) if ok else {}
    try:
        batch = firestore_client.batch()
        batch.set(firestore_client.collection("profiles").document(uid), {
            "faceVerified": ok, "isFaceVerified": ok, "faceVerifiedReason": reason,
            "faceVerifiedAtIso": now, "faceVerifiedMethod": FACE_METHOD,
            "faceVerifiedPhotos": photos if ok else [],
            **consent,
        }, merge=True)
        batch.set(firestore_client.collection("publicProfiles").document(uid), {"isFaceVerified": ok}, merge=True)
        batch.set(_face_ref(uid), {**record, **consent, "status": status, "reason": reason, "updatedAtIso": now,
                                   "needsLook": bool(review),
                                   "review": ({"requestedAtIso": now, **review} if review else fs.DELETE_FIELD)}, merge=True)
        batch.commit()
    except Exception:
        logger.exception("face verification result write failed")


def _restricted(uids: List[str]) -> List[str]:
    """Which of these accounts are banned or suspended right now (moderationStates, written by the web admin)."""
    out = []
    for u in uids:
        try:
            st = (firestore_client.collection("moderationStates").document(u).get().to_dict() or {}).get("status")
        except Exception:
            st = None
        if st in ("banned", "suspended"):
            out.append(u)
    return out


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
    review: Optional[Dict[str, Any]] = None
    if face_match.needs_review(reason):
        review = {"kind": "photos", "reason": reason}   # a live face just under a bar: verified, the team looks
    elif not ok:
        return "failed", reason, record, None, None
    dups = face_match.duplicates(front, _verified_templates(), exclude=uid)
    if dups:
        record["duplicates"] = [{"uid": u, "sim": round(s, 3)} for u, s in dups[:3]]
        held = _restricted([d["uid"] for d in record["duplicates"]])
        review = {"kind": "duplicate", "duplicateOf": record["duplicates"], **({"restricted": held} if held else {})}
        if held:
            # The same face as a banned or suspended account: no badge until a person looks.
            return "in_review", "review_duplicate", record, review, front
    return "ok", "ok", record, review, front


@app.post("/verify-face-live")
def verify_face_live(body: VerifyFaceLiveRequest, authorization: Optional[str] = Header(default=None)):
    """The challenge's frames. A live face that matches the profile photos is verified at once (the server writes
    it, never the app); if the server wasn't fully sure (just under a bar, the same face on another account, or
    verified again after the team removed it) the team also gets it to look at and can remove it. Only the same
    face as a banned account waits ("in_review"). Afterwards only the straight frame is kept (for re-checks
    after a photo change); the rest is deleted."""
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
    if status == "ok" and review is None and (_face_ref(uid).get().to_dict() or {}).get("revokedBefore"):
        review = {"kind": "after_removal"}
    keep = urls[0] if record.get("posesOk") else None
    record.update({"frontSelfie": keep or "", "selfies": [keep] if keep else []})
    if front is not None:
        _store_template(uid, front, "verified" if status == "ok" else "pending")
    person_consent = None
    if status == "ok" and body.personConsent:
        try:
            profile_for = (firestore_client.collection("profiles").document(uid).get().to_dict() or {}).get("profileFor")
            person_consent = _person_consent_fields(profile_for, True, body.consentTextVersion, status, _now_iso())
        except Exception:
            logger.exception("person consent: profile read failed")     # the check's result is still saved
    _save_face_result(uid, status, reason, record.get("photos") or [], record,
                      {**review, "frontSelfie": keep, "photos": record.get("photos") or []} if review else None,
                      person_consent=person_consent)
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
    if rec.get("status") == "in_review":
        # On hold for a person (the same face as a banned account): new photos don't change that.
        return {"ok": False, "status": "in_review", "reason": str(rec.get("reason") or "review_duplicate")}
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
    review: Optional[Dict[str, Any]] = None
    if face_match.needs_review(reason):
        ok, review = True, {"kind": "photos", "reason": reason}     # verified, the team looks
        reason = "ok"
    elif ok and (rec.get("review") or {}).get("kind") not in (None, "photos"):
        review = rec["review"]          # new photos don't settle "the same face on another account"
    status = "ok" if ok else "failed"
    _save_face_result(uid, status, reason, photos, {"scores": scores, "photos": photos, "recheckedAtIso": _now_iso()},
                      {**review, "frontSelfie": front_url, "photos": photos} if review else None)
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
import fates_astro_chart  # noqa: E402

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

import astro_api  # noqa: E402


def _astro_deps_extra(deps_cls: Any) -> Dict[str, Any]:
    """The serverLeases helpers for astro_api (the week forecast's cache), passed only when astro_api.Deps has the
    take_lease / release_lease fields, so either merge order of W1b works."""
    fields = getattr(deps_cls, "__dataclass_fields__", {}) or {}
    return {k: v for k, v in (("take_lease", _take_lease), ("release_lease", _release_lease)) if k in fields}


astro_api.configure(astro_api.Deps(
    db=lambda: firestore_client,
    verify_uid=_verify_firebase_token_or_401,
    resolve_place=_resolve_birth_place,
    places_query=_places_query,
    load_user_docs=_load_user_docs,
    is_premium=_is_premium_like_client,
    complete=lambda messages, max_tokens, temperature, kind: _deepseek_complete(
        messages, max_tokens, temperature=temperature, kind=kind, timeout=FATES_AI_TIMEOUT_SEC + 6),
    take_rate=_take_rate,
    **_astro_deps_extra(astro_api.Deps),
))
app.include_router(astro_api.router)

import biodata_route  # noqa: E402  (POST /astro/biodata, on astro_api's sign-in check and Deps)

app.include_router(biodata_route.router)
