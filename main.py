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


# ✅ NEW: verify uploaded photo in Firebase Storage via gs://... URI
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
# TOPIC GATE (SOFT)
# - Block only obvious cooking/recipes. Everything else is allowed.
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
    # If blocked (recipes), gently redirect without being rude.
    variants_en = [
        "I can’t help with recipes 🦜\nBut I can help with:\n• daily horoscope (Vedic)\n• love, dating, relationships\n• what to text/reply (I’ll write it)\n• profile/bio glow-up\nAsk me one of those 🙂",
        "Not a cooking parrot 😅\nBut I’m great at:\n• Vedic astrology & Daily Fates\n• relationships + dating\n• writing messages\n• improving your profile\nWhat do you want today? 🦜",
    ]
    variants_ru = [
        "С рецептами не помогу 🦜\nНо я могу:\n• гороскоп (ведический)\n• любовь/свидания/отношения\n• сообщения (напишу текст)\n• улучшение профиля/био\nСпроси что-то из этого 🙂",
        "Я не кулинарный попугай 😅\nЗато я топ в:\n• ведической астрологии и Daily Fates\n• отношениях и свиданиях\n• сообщениях\n• улучшении профиля\nЧто разбираем? 🦜",
    ]
    lang = (locale or "en").strip().lower()
    pool = variants_ru if lang.startswith("ru") else variants_en
    return _stable_pick(pool, user_text)


# =========================
# INTENT HINTS (cheap, deterministic)
# - We add a tiny hint to the model to interpret vague requests correctly.
# =========================
def _infer_intent_hint(user_text: str, locale: str) -> str:
    t = (user_text or "").lower()

    # Forecast / today / vibe style questions should be treated as horoscope/daily fate automatically.
    forecast_words = [
        "forecast", "today", "my day", "how will my day", "what about today", "daily", "vibe",
        "гороскоп", "сегодня", "прогноз", "как пройдет день", "мой день", "вайб"
    ]
    if any(w in t for w in forecast_words):
        return (
            "INTENT_HINT: The user is asking for a Daily Fate / daily horoscope. "
            "Answer as a Vedic-style daily horoscope and personalize using USER_PROFILE + ASTRO_COMPUTED."
        )

    # If user asks about interests/profile, it is still in-scope: personalize.
    if "interest" in t or "interests" in t or "profile" in t or "bio" in t or "about me" in t or "анкет" in t:
        return (
            "INTENT_HINT: Use USER_PROFILE (interests/lifestyle) to personalize your answer."
        )

    return ""


# =========================
# SYSTEM PROMPT (UPGRADED: PERSONAL, INDIAN STYLE, USE PROFILE)
# =========================
def _build_system_prompt(locale: str) -> str:
    lang = (locale or "en").strip() or "en"
    return (
        "You are Shaadi Parrot 🦜 — a charming, friendly Indian astrologer + dating coach inside a dating & Daily Fates app.\n"
        "Your goal: make the user feel seen and understood using their profile data, while staying practical and warm.\n"
        "\n"
        "IMPORTANT BEHAVIOR:\n"
        "1) Be personable. If USER_PROFILE includes firstName, greet them by name.\n"
        "2) If USER_PROFILE includes birthDate, reference it briefly to make it feel personal (do not over-repeat).\n"
        "3) If ASTRO_COMPUTED exists, use Indian/Vedic framing:\n"
        "   - Use terms: Rashi (Moon sign), Surya (Sun sign), Nakshatra.\n"
        "   - Do NOT mention 'sidereal Lahiri' unless the user asks how you know.\n"
        "4) If user asks for today's horoscope / daily forecast (even vaguely), ALWAYS answer as a Daily Fate.\n"
        "5) Personalize using at least 2 anchors when available: name, interests, lifestyle (workout/smoking/drinking), relationshipIntent, city/country.\n"
        "   Example: if interests include gym/fitness, tie energy/discipline tips to training.\n"
        "6) Never be strict or dismissive. Do NOT say 'outside my nest' unless the user asks for cooking/recipes.\n"
        "7) Keep it practical and uplifting. Avoid random 'lucky colors/numbers' unless user asks.\n"
        "\n"
        "WHAT YOU CAN DO:\n"
        "• daily horoscope (uplifting + practical)\n"
        "• Vedic astrology + compatibility\n"
        "• write exact messages (texting/DMs)\n"
        "• dating strategy + relationship advice\n"
        "• profile/bio/photos improvements\n"
        "\n"
        "OUTPUT STYLE:\n"
        "- Warm, confident, supportive.\n"
        "- 1–3 emojis max.\n"
        "- No markdown, no **bold**, no headings.\n"
        "- Prefer bullets with '•'.\n"
        "- Keep it concise but not shallow: usually 6–10 bullets max.\n"
        "\n"
        "ACCURACY RULES:\n"
        "- If birth place + timezone are missing: do NOT claim Ascendant/houses.\n"
        "- If ASTRO_COMPUTED is missing: ask ONE question (Sun sign or birthDate) BUT still give a gentle general daily horoscope.\n"
        "\n"
        f"Reply in {lang}.\n"
    )


# =========================
# PROFILE LOAD
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


def _flatten_value(v: Any, max_len: int = 180) -> str:
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
        for item in v[:16]:
            s = _flatten_value(item, max_len=50)
            if s:
                parts.append(s)
        out = ", ".join(parts)
        if len(v) > 16:
            out += "…"
        return (out[:max_len] + "…") if len(out) > max_len else out
    if isinstance(v, dict):
        parts = []
        for i, (kk, vv) in enumerate(v.items()):
            if i >= 10:
                parts.append("…")
                break
            s = _flatten_value(vv, max_len=60)
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
        "birthDate", "birthTime", "age",
        "cityName", "stateName", "countryName",
        "languages", "religion", "community",
        "relationshipIntent", "seekerType", "orientation", "lookingForGender",
        "interests", "tags",
        "workout", "smoking", "drinking", "social", "pets",
        "education", "jobTitle", "occupation",
        "bio", "aboutMe",
    ]

    parts: List[str] = []
    for k in preferred_keys:
        if k in profile_doc:
            val = _flatten_value(profile_doc.get(k))
            if val:
                parts.append(f"{k}={val}")
        if len(parts) >= 28:
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

    # Without timezone, we assume UTC. If time missing -> noon UTC.
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
# PHOTO MODERATION (SafeSearch + Face detection) ✅ NEW
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
    # protect endpoint from abuse
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

    # thresholds (simple & safe)
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
# PERSISTENT CHAT STORAGE (Firestore)
# - Supports thread_id without breaking existing "default" history.
# =========================
def _safe_thread_id(thread_id: str) -> str:
    tid = (thread_id or "default").strip()
    if not tid:
        tid = "default"
    if tid == "default":
        return "default"
    # Make thread doc id safe and short
    h = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:16]
    return f"t_{h}"


def _chat_doc_id(uid: str, thread_id: str) -> str:
    tid = _safe_thread_id(thread_id)
    # keep backward compatibility for default
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
    """
    Writes user+assistant messages in ONE batch commit.
    This helps prevent "second message appears only after restart" issues when the UI refetches quickly.
    """
    col = _chat_msgs_col_ref(uid, thread_id)
    docref = _chat_doc_ref(uid, thread_id)
    if col is None or docref is None or firestore_client is None:
        return

    try:
        batch = firestore_client.batch()

        user_hash = hashlib.md5((user_text or "").encode("utf-8")).hexdigest()[:8]
        asst_hash = hashlib.md5((assistant_text or "").encode("utf-8")).hexdigest()[:8]

        user_id = f"{created_at_ms}_user_{user_hash}"
        asst_id = f"{created_at_ms + 2}_assistant_{asst_hash}"

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

        # also touch parent doc for "updatedAtMs" so lists can refresh fast
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

    m = re.search(
        r"\b(?:born|birthday)\b.*?\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b",
        t, re.IGNORECASE
    )
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
    elif "forecast" in tl or "today" in tl or "прогноз" in tl or "сегодня" in tl:
        mem["lastTopic"] = "astrology"

    state["memory"] = mem
    return state


# =========================
# HISTORY/RESET endpoints
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
    })
    return ResetResponse(thread_id=(thread_id or "default"), ok=True)


# keep old endpoints (optional)
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
    })
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
        "max_tokens": 700,
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
# AI CHAT ENDPOINT
# =========================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat(body: AiChatRequest, authorization: Optional[str] = Header(default=None)):
    uid = _verify_firebase_token_or_401(authorization)

    user_text = _normalize_text(body.text)
    thread_id = (body.thread_id or "default").strip() or "default"
    locale = (body.locale or "en").strip() or "en"

    if not user_text:
        return AiChatResponse(reply_text="Ask me something 🦜", thread_id=thread_id)

    if not _is_allowed_topic(user_text):
        return AiChatResponse(
            reply_text=_topic_block_reply(user_text, locale),
            blocked=True,
            reason="topic_not_allowed",
            thread_id=thread_id,
        )

    system_prompt = _build_system_prompt(locale)

    # Load user profile (for personalization)
    profile = await _load_user_profile(uid)
    profile_summary = _profile_summary_text(profile)

    # Astro computed (vedic sidereal)
    astro_computed = _compute_astro(profile)

    # Load persistent chat memory + last messages
    state = _load_chat_state(uid, thread_id)
    memory_text = _memory_text_from_state(state)
    persisted_history = _load_chat_history(uid, thread_id, limit=30)

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

    intent_hint = _infer_intent_hint(user_text, locale)

    # Build final LLM messages
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if profile_summary:
        messages.append({"role": "system", "content": profile_summary})

    if astro_computed:
        messages.append({"role": "system", "content": astro_computed})

    if memory_text:
        messages.append({"role": "system", "content": memory_text})

    if intent_hint:
        messages.append({"role": "system", "content": intent_hint})

    combined = persisted_history[-16:]
    if client_history and len(combined) < 8:
        combined = (combined + client_history)[-16:]

    for m in combined:
        if m.get("role") in ["user", "assistant"] and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    # Current user message
    messages.append({"role": "user", "content": user_text})

    reply = await _call_deepseek(messages)

    # Persist messages (batch commit to reduce "missing until restart" effects)
    now_iso = _now_iso()
    now_ms = _now_ms()
    _save_chat_message_batch(uid, thread_id, user_text, reply, now_iso, now_ms)

    # Update memory/state
    state = _update_memory_from_text(state, user_text, reply)
    _save_chat_state(uid, thread_id, {
        "uid": uid,
        "threadId": thread_id,
        "updatedAtIso": now_iso,
        "updatedAtMs": now_ms,
        "memory": state.get("memory", {}),
    })

    logger.info(
        f"[ai_chat] uid={uid} thread={thread_id} user_len={len(user_text)} "
        f"profile_fields={len(profile) if profile else 0} persisted_hist={len(persisted_history)} "
        f"astro={'yes' if bool(astro_computed) else 'no'}"
    )

    return AiChatResponse(reply_text=reply, blocked=False, thread_id=thread_id)
