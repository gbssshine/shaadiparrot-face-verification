from fastapi import FastAPI, HTTPException, Header
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shaadiparrot-cloudrun")

vision_client: vision.ImageAnnotatorClient | None = None
firestore_client: firestore.Client | None = None

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not firebase_admin._apps:
    firebase_admin.initialize_app()

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
    thread_id: Optional[str] = "default"
    history: Optional[List[ChatTurn]] = None  # optional client-provided history


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

    # Swiss Ephemeris sidereal: Lahiri (Vedic)
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
# TOPIC GATE (NO PANCAKES)
# =========================
def _is_allowed_topic(user_text: str) -> bool:
    t = (user_text or "").lower()

    # Block obvious cooking/recipes
    blocked_keywords = [
        "recipe", "cook", "cooking", "pancake", "omelet", "baking", "cake",
        "how to fry", "ingredients", "gram", "ml", "kefir", "flour", "sugar",
        "рецепт", "готовить", "омлет", "блин", "мука", "ингредиент",
        "готовка", "кулинар", "сколько яиц", "жарить"
    ]
    if any(k in t for k in blocked_keywords):
        return False

    allowed_keywords = [
        # relationships / texting
        "dating", "relationship", "love", "crush", "girlfriend", "boyfriend",
        "girls", "guys", "her", "him", "she", "he",
        "message", "reply", "text", "what should i say", "how to respond",
        "pick up", "flirt", "date idea", "first date", "apology", "breakup", "jealous",

        # profile help
        "profile", "bio", "about me", "photos", "photo", "pictures", "rewrite",
        "describe my profile", "improve my profile",

        # astrology / fates (✅ include common misspellings)
        "astrology", "vedic", "kundli", "horoscope", "horoskop", "horoskope", "horoskopе",
        "гороскоп", "зодиак", "zodiac", "sign",
        "birth date", "birthdate", "born", "nakshatra", "moon", "sun",
        "compatibility", "match", "fate", "daily fates", "planets",

        # app context
        "shaadi", "parrot", "app", "premium", "subscription",
    ]
    return any(k in t for k in allowed_keywords)


def _topic_block_reply(user_text: str, locale: str) -> str:
    # Multiple variations so it doesn't repeat the same "I can help with..."
    variants_en = [
        "That’s outside my nest 🦜\nI’m here for love + dating + Vedic astrology.\nTell me:\n• what happened\n• what you want to text\n• your birth date (or your sign)\n• or share your profile/bio for improvements 🙂",
        "I’m not the best for that topic 😅\nBut I can be amazing at:\n• relationships & dating\n• writing messages\n• profile glow-up\n• Vedic astrology & compatibility\nPick one and I’ll jump in 🦜✨",
        "I can’t help with that request.\nBut if you want, I can:\n• write the exact message you should send\n• decode mixed signals\n• do a quick compatibility read\n• improve your profile/bio\nWhat’s your situation? 🦜",
        "Not my specialty 🦜\nI’m your dating + astrology guide.\nTell me what’s going on in your love life, or ask for a daily horoscope using your sign/birth date 🙂"
    ]

    variants_ru = [
        "Это вне моего гнезда 🦜\nЯ про любовь, отношения и ведическую астрологию.\nСкажи:\n• что случилось\n• что хочешь написать\n• дату рождения (или знак)\n• или скинь био/профиль — улучшим 🙂",
        "С этим не помогу 😅\nНо я супер в:\n• отношениях и свиданиях\n• текстах/ответах (напишу сообщение)\n• улучшении профиля\n• ведической астрологии и совместимости\nВыбирай, и поехали 🦜✨",
        "Не мой запрос.\nЗато могу:\n• написать сообщение слово-в-слово\n• разобрать сигналы\n• прикинуть совместимость\n• прокачать профиль\nЧто у тебя за ситуация? 🦜",
        "Не моя тема 🦜\nЯ твой гид по свиданиям и астрологии.\nРасскажи про ситуацию, или попроси гороскоп, указав знак/дату рождения 🙂"
    ]

    lang = (locale or "en").strip().lower()
    pool = variants_ru if lang.startswith("ru") else variants_en
    return _stable_pick(pool, user_text)


# =========================
# SYSTEM PROMPT (PERSONA)
# =========================
def _build_system_prompt(locale: str) -> str:
    lang = (locale or "en").strip() or "en"
    return (
        "You are Shaadi Parrot 🦜 — a cheerful Indian astrologer + dating coach inside a dating & Daily Fates app.\n"
        "Your mission: help users with love, relationships, texting, profiles, and Vedic astrology.\n"
        "You can do:\n"
        "• daily horoscope (short, practical, uplifting)\n"
        "• what to text/reply (write the exact message)\n"
        "• dating strategy and relationship advice\n"
        "• profile/bio/photo improvements\n"
        "• Vedic astrology & compatibility (sidereal Lahiri)\n"
        "Style:\n"
        "- Warm, confident, practical.\n"
        "- Emojis naturally (1–3 max).\n"
        "- No markdown formatting (no **bold**, no headings, no backticks).\n"
        "- Prefer clean bullets with '•'.\n"
        "Accuracy:\n"
        "- If birth place + timezone are missing, do NOT claim Ascendant/houses.\n"
        "- You can still do: daily horoscope + Sun/Moon sign insights from available data.\n"
        "- If you need the user's sign, ask: 'What’s your Sun sign or your birth date (YYYY-MM-DD)?'\n"
        f"Reply in {lang}.\n"
    )


# =========================
# PROFILE LOAD (FROM profiles/{uid})
# =========================
def _safe_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    deny_prefixes = ["geo", "lat", "lng", "location", "idtoken", "refreshtoken", "token", "__"]
    deny_exact = {"updatedAt", "deviceId", "pushToken", "refreshToken"}

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
        "firstName", "lastName", "gender",
        "seekerType", "orientation", "lookingForGender",
        "birthDate", "birthTime", "age",
        "cityName", "countryName", "stateName",
        "community", "religion",
        "relationshipIntent",
        "bio", "aboutMe",
        "interests", "languages", "tags",
        "drinking", "smoking", "workout", "pets", "social",
        "education", "jobTitle", "occupation",
        "height", "heightDisplayText",
    ]

    parts: List[str] = []
    for k in preferred_keys:
        if k in profile_doc:
            val = _flatten_value(profile_doc.get(k))
            if val:
                parts.append(f"{k}={val}")
        if len(parts) >= 22:
            break

    if not parts:
        return ""

    return "USER_PROFILE: " + " | ".join(parts)


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

_PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,
}

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
    else:
        hour = 12.0
        time_mode = "date_only"

    jd_ut = swe.julday(y, mo, d, hour)

    try:
        sun_lon = _calc_sidereal_lon_ut(jd_ut, swe.SUN)
        moon_lon = _calc_sidereal_lon_ut(jd_ut, swe.MOON)

        sun_sign = _sign_from_lon(sun_lon)
        moon_sign = _sign_from_lon(moon_lon)
        nak = _nakshatra_from_lon(moon_lon)

        planet_bits: List[str] = []
        for name, pid in _PLANETS.items():
            lon = _calc_sidereal_lon_ut(jd_ut, pid)
            planet_bits.append(f"{name}:{_sign_from_lon(lon)}")

        rahu_lon = _calc_sidereal_lon_ut(jd_ut, swe.MEAN_NODE)
        ketu_lon = (rahu_lon + 180.0) % 360.0
        planet_bits.append(f"Ketu:{_sign_from_lon(ketu_lon)}")

        note = "Ascendant/houses need birthPlace + timezone."
        if time_mode == "date_only":
            note = "Moon/Nakshatra accuracy improves with birthTime + birthPlace/timezone."

        return (
            "ASTRO_COMPUTED (Vedic sidereal Lahiri): "
            f"SunSign={sun_sign}; MoonSign={moon_sign}; Nakshatra={nak}; "
            f"Planets={', '.join(planet_bits)}; "
            f"TimeMode={time_mode}. {note}"
        )
    except Exception:
        logger.exception("Astro compute failed")
        return ""


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
# PERSISTENT CHAT STORAGE (Firestore)
# =========================
def _chat_doc_ref(uid: str):
    if firestore_client is None:
        return None
    return firestore_client.collection("parrotChats").document(uid)

def _chat_msgs_col_ref(uid: str):
    if firestore_client is None:
        return None
    return firestore_client.collection("parrotChats").document(uid).collection("messages")

def _now_ms() -> int:
    return int(time.time() * 1000)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

def _load_chat_state(uid: str) -> Dict[str, Any]:
    ref = _chat_doc_ref(uid)
    if ref is None:
        return {}
    try:
        snap = ref.get()
        if not snap.exists:
            return {}
        return snap.to_dict() or {}
    except Exception:
        logger.exception("Failed to load parrotChats/{uid}")
        return {}

def _save_chat_state(uid: str, patch: Dict[str, Any]) -> None:
    ref = _chat_doc_ref(uid)
    if ref is None:
        return
    try:
        ref.set(patch, merge=True)
    except Exception:
        logger.exception("Failed to save parrotChats/{uid}")

def _save_chat_message(uid: str, role: str, text: str, created_at_iso: str, created_at_ms: int) -> None:
    col = _chat_msgs_col_ref(uid)
    if col is None:
        return
    try:
        # unique id even if multiple messages in same ms
        msg_id = f"{created_at_ms}_{role}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:6]}"
        col.document(msg_id).set({
            "role": role,
            "text": text,
            "createdAtIso": created_at_iso,
            "createdAtMs": created_at_ms
        })
    except Exception:
        logger.exception("Failed to save chat message")

def _load_chat_history(uid: str, limit: int = 24) -> List[Dict[str, str]]:
    col = _chat_msgs_col_ref(uid)
    if col is None:
        return []

    state = _load_chat_state(uid)
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

def _memory_text_from_state(state: Dict[str, Any]) -> str:
    mem = state.get("memory") if isinstance(state.get("memory"), dict) else {}
    if not mem:
        return ""
    bits = []
    for k in ["lastTopic", "partnerName", "partnerBirthDate", "partnerZodiac", "userZodiac"]:
        v = (mem.get(k) or "").strip()
        if v:
            bits.append(f"{k}={v}")
    if not bits:
        return ""
    return "PARROT_MEMORY: " + " | ".join(bits)

def _update_memory_from_text(state: Dict[str, Any], user_text: str, assistant_text: str) -> Dict[str, Any]:
    mem = state.get("memory") if isinstance(state.get("memory"), dict) else {}
    t = (user_text or "").strip()

    m = re.search(r"\b(?:born|birthday)\b.*?\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b", t, re.IGNORECASE)
    if m:
        day = m.group(1)
        mon = m.group(2)
        mem["partnerBirthDate"] = f"{day} {mon.title()}"

    m2 = re.search(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b", t)
    if m2:
        d = m2.group(1)
        mo = m2.group(2)
        yr = m2.group(3) or ""
        mem["partnerBirthDate"] = f"{d}/{mo}" + (f"/{yr}" if yr else "")

    mz = re.search(r"\b(i'?m|i am)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz:
        mem["userZodiac"] = mz.group(2).title()

    mz2 = re.search(r"\b(she'?s|she is|he'?s|he is)\s+a?\s*(aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", t, re.IGNORECASE)
    if mz2:
        mem["partnerZodiac"] = mz2.group(2).title()

    tl = t.lower()
    if "text" in tl or "message" in tl or "reply" in tl:
        mem["lastTopic"] = "texting"
    elif "compat" in tl or "match" in tl or "kundli" in tl:
        mem["lastTopic"] = "compatibility"
    elif "profile" in tl or "bio" in tl or "photos" in tl:
        mem["lastTopic"] = "profile"
    elif "relationship" in tl or "girls" in tl or "love" in tl:
        mem["lastTopic"] = "relationships"
    elif "horoscope" in tl or "horoskop" in tl or "гороскоп" in tl or "vedic" in tl or "astrology" in tl:
        mem["lastTopic"] = "astrology"

    state["memory"] = mem
    return state


# =========================
# NEW: AI HISTORY/RESET endpoints (match your MAUI client)
# =========================
@app.get("/ai/history", response_model=HistoryResponse)
def ai_history(thread_id: str = "default", limit: int = 24, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    limit = max(1, min(80, int(limit)))

    rows = _load_chat_history(uid, limit=limit)  # role/content
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
    _save_chat_state(uid, {"uid": uid, "clearedAtIso": now_iso, "clearedAtMs": now_ms, "updatedAtIso": now_iso})
    return ResetResponse(thread_id=(thread_id or "default"), ok=True)


# (keep your old endpoints if you want)
@app.get("/parrot/history")
def parrot_history(limit: int = 24, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    limit = max(1, min(80, int(limit)))
    items = _load_chat_history(uid, limit=limit)
    return {"uid": uid, "count": len(items), "items": items}


@app.post("/parrot/reset")
def parrot_reset(authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_state(uid, {"uid": uid, "clearedAtIso": now_iso, "clearedAtMs": now_ms, "updatedAtIso": now_iso})
    return {"status": "ok", "uid": uid, "clearedAtIso": now_iso}


# =========================
# DEEPSEEK CALL
# =========================
async def _call_deepseek(messages: List[Dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set in Cloud Run env vars")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.85,
        "max_tokens": 520,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    import json
    try:
        import httpx
        async with httpx.AsyncClient(timeout=45) as client:
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


# =========================
# AI CHAT ENDPOINT (WITH MEMORY)
# =========================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text)
    if not user_text:
        return AiChatResponse(reply_text="Ask me something 🦜", thread_id=(body.thread_id or "default"))

    locale = (body.locale or "en").strip() or "en"

    if not _is_allowed_topic(user_text):
        return AiChatResponse(
            reply_text=_topic_block_reply(user_text, locale),
            blocked=True,
            reason="topic_not_allowed",
            thread_id=(body.thread_id or "default"),
        )

    system_prompt = _build_system_prompt(locale)

    # Load user profile (for personalization)
    profile = await _load_user_profile(uid)
    profile_summary = _profile_summary_text(profile)

    # Astro computed (vedic sidereal)
    astro_computed = _compute_astro(profile)

    # Load persistent chat memory + last messages
    state = _load_chat_state(uid)
    memory_text = _memory_text_from_state(state)
    persisted_history = _load_chat_history(uid, limit=30)

    # Optional: merge client history
    client_history: List[Dict[str, str]] = []
    if body.history:
        for t in body.history[-10:]:
            role = t.role
            txt = _normalize_text(t.text)
            if role in ["user", "assistant"] and txt:
                if len(txt) > 360:
                    txt = txt[:360] + "…"
                client_history.append({"role": role, "content": txt})

    # Build final LLM messages
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if profile_summary:
        messages.append({"role": "system", "content": profile_summary})

    if astro_computed:
        messages.append({"role": "system", "content": astro_computed})

    if memory_text:
        messages.append({"role": "system", "content": memory_text})

    combined = persisted_history[-16:]
    if client_history and len(combined) < 8:
        combined = (combined + client_history)[-16:]

    for m in combined:
        if m.get("role") in ["user", "assistant"] and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    reply = await _call_deepseek(messages)

    # Persist messages
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_message(uid, "user", user_text, now_iso, now_ms)
    _save_chat_message(uid, "assistant", reply, now_iso, now_ms + 1)  # ensure assistant is after user in sorting

    state = _update_memory_from_text(state, user_text, reply)
    _save_chat_state(uid, {"uid": uid, "updatedAtIso": now_iso, "memory": state.get("memory", {})})

    logger.info(
        f"[ai_chat] uid={uid} user_len={len(user_text)} profile_fields={len(profile) if profile else 0} "
        f"persisted_hist={len(persisted_history)} astro={'yes' if bool(astro_computed) else 'no'}"
    )
    return AiChatResponse(reply_text=reply, blocked=False, thread_id=(body.thread_id or "default"))
