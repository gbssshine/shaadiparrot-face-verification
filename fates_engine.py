"""Daily Fates: pure compatibility and selection logic (no Firestore, no network).

Numbers here are deterministic so a reading can be explained and reproduced. The AI layer
(fates_ai.py) only chooses among the best candidates this module ranks and writes the words.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote

import face_live

PATHS = ("stars", "heart", "home")
PATH_LABELS = {"stars": "Path of the Stars", "heart": "Path of the Heart", "home": "Path of the Home"}

# A fate is only brought when the pair clears this overall score. Mithu shows fewer paths rather
# than filling a slot with a weak match.
MIN_FATE_SCORE = int(os.getenv("FATES_MIN_SCORE") or "55")
# Someone who already chose the viewer as their fate is brought back with a softer bar: they said yes.
CHOOSER_MIN_FATE = int(os.getenv("FATES_CHOOSER_MIN") or "40")
CHOOSER_BOOST = 30
# Only face-verified people are brought, and only to face-verified people (owner, 2026-09-30).
VERIFIED_ONLY = (os.getenv("FATES_VERIFIED_ONLY") or "1").strip() != "0"
# Paid reach, never paid fit: a boost (or Parrot+) moves a good match up among good matches and lets
# them be brought to more people. The fate number never changes, the bar (MIN_FATE_SCORE) never drops,
# and a viewer gets at most one promoted person a day, so most of what free people see stays free.
BOOST_RANK = 5
PREMIUM_RANK = 2
MAX_PROMOTED_PER_DAY = 1
MIN_GUNAS_FOR_STARS_PATH = 18
ACTIVE_DAYS = int(os.getenv("FATES_ACTIVE_DAYS") or "14")
DEFAULT_RADIUS_KM = 50

# How much each part weighs in the overall fate score, by the viewer's "How much do the stars
# matter to you?" answer. Missing parts drop out and the rest are renormalised.
WEIGHTS = {
    "none":   {"stars": 0.00, "heart": 0.45, "home": 0.40, "everyday": 0.15},
    "little": {"stars": 0.20, "heart": 0.35, "home": 0.33, "everyday": 0.12},
    "lot":    {"stars": 0.35, "heart": 0.28, "home": 0.27, "everyday": 0.10},
    "must":   {"stars": 0.35, "heart": 0.28, "home": 0.27, "everyday": 0.10},
}

_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fates_tests_catalog.json")
_catalog: Optional[Dict[str, Any]] = None


def tests_catalog() -> Dict[str, Any]:
    global _catalog
    if _catalog is None:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            _catalog = json.load(f)
    return _catalog


# =========================
# PERSON
# =========================
@dataclass
class Person:
    uid: str
    first_name: str = ""
    age: Optional[int] = None
    gender: str = ""            # "male" | "female" | ""
    looking_for: str = "any"    # "male" | "female" | "any"
    city: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    bio: str = ""
    interests: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    religion: str = ""
    community: str = ""
    intent: str = ""            # serious | both | dating | friends | unsure | ""
    smoking: str = ""
    drinking: str = ""
    pets: str = ""
    workout: str = ""
    social: str = ""
    have_children: str = ""
    want_children: str = ""
    relocate: str = ""
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    radius_km: Optional[float] = None
    require_religion: bool = False
    require_community: bool = False
    tests: Dict[str, int] = field(default_factory=dict)   # testId -> scorePercent 0..100
    photos: List[str] = field(default_factory=list)
    discoverable: bool = False
    face_verified: bool = False
    boosted: bool = False       # publicProfiles.boostActiveUntilUtcIso in the future (server-written)
    geo_precise: bool = False   # a real position (GPS / last known), not a city picked by hand
    units: str = "km"           # the viewer's settings_units: km | mi
    show_distance: bool = True  # the viewer's settings_showDistance
    premium: bool = False       # publicProfiles.isPremium (server-written)
    last_active_ms: Optional[int] = None
    stars_pref: str = "little"  # none | little | lot | must
    use_tests: bool = True
    chart: Optional[Dict[str, Any]] = None  # from fates_astro.chart_*()
    # Matrimony (plan §2 C1 / C11). The facts are what families see on the card (public first); income is the public
    # card value only, never a matches-only one. prefs are the private pref_* answers (prefs_from_doc).
    profile_for: str = ""       # "" | self | son | daughter | ... | none (kundli only)
    marriage_intent: bool = False   # profiles.marriageIntent (the private, trusted fact)
    marital_status: str = ""
    children: str = ""          # "" | No children | Yes, living together | Yes, living separately
    diet: str = ""
    education: str = ""
    profession: str = ""
    caste: str = ""
    manglik: str = ""           # Yes | No | Partial (Anshik) | Don't know | ""
    income: str = ""            # an income band code ("10-15L"), public card only
    family_values: str = ""
    prefs: Dict[str, Any] = field(default_factory=dict)
    rishta: bool = False        # the viewer's Fates today are in rishta mode ({mode: 'rishta'})


def _future_iso(v: Any) -> bool:
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def verified_now(private: Dict[str, Any], user: Dict[str, Any], photos: List[str]) -> bool:
    """Verified, and still the photos that were checked against the selfies. Only server-written fields count
    (the rules stop the app from setting them); a new or replaced photo pauses it until the server re-checks."""
    flagged = any(_to_bool(private.get(k)) or _to_bool(user.get(k)) for k in ("isFaceVerified", "faceVerified"))
    checked = {str(u) for u in (private.get("faceVerifiedPhotos") or []) if u}
    return flagged and bool(photos) and set(photos) <= checked


def precise_geo_source(v: Any) -> bool:
    """Same rule as the app's DistanceDisplay.IsPreciseGeoSource: GPS / last known (and seeded test people)."""
    src = _norm(v)
    return src.startswith("gps") or src.startswith("lastknown") or src == "seed"


def approx_distance(km: Optional[float], precise: bool, units: str = "km") -> str:
    """Same text as the app's DistanceDisplay.ApproxText: "<2 km", "~4 km", "~15 km", "~850 km" (or mi).
    Never exact, and nothing under 20 km when a side only picked a city."""
    if km is None or km < 0:
        return ""
    if not precise and km < 20:
        return ""
    miles = units == "mi"
    value = km * 0.621371 if miles else float(km)
    unit = "mi" if miles else "km"
    if value < 2:
        return f"<2 {unit}"
    step = 1 if value < 10 else 5 if value < 50 else 10 if value < 200 else 50 if value < 1000 else 100
    rounded = int(math.floor(value / step + 0.5) * step)
    return f"~{rounded:,} {unit}"


def distance_text(viewer: Person, p: "Pair") -> str:
    if not viewer.show_distance:
        return ""
    return approx_distance(p.distance_km, viewer.geo_precise and p.cand.geo_precise, viewer.units)


def _s(v: Any) -> str:
    return " ".join(str(v or "").split()).strip()


def _norm(v: Any) -> str:
    return _s(v).casefold()


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return _norm(v) in ("true", "1", "yes", "on")


def _str_list(v: Any) -> List[str]:
    if isinstance(v, (list, tuple)):
        return [s for s in (_s(x) for x in v) if s]
    if isinstance(v, str) and v.strip():
        return [s for s in (_s(x) for x in v.split(",")) if s]
    return []


def norm_gender(v: Any) -> str:
    g = _norm(v)
    if g in ("male", "man", "men", "m", "boy"):
        return "male"
    if g in ("female", "woman", "women", "f", "girl"):
        return "female"
    return ""


def norm_looking_for(v: Any) -> str:
    g = _norm(v)
    if g in ("any", "everyone", "both", "all", ""):
        return "any"
    if "women" in g or g in ("female", "woman", "f"):
        return "female"
    if "men" in g or g in ("male", "man", "m"):
        return "male"
    return "any"


def norm_intent(v: Any) -> str:
    s = _norm(v)
    if not s:
        return ""
    if s.startswith("serious") or "long-term partner" in s or "marri" in s:
        return "serious"
    if "open to both" in s or s.startswith("long-term or short-term") or s == "both":
        return "both"
    if s.startswith("dating") or "see where it goes" in s or "casual" in s:
        return "dating"
    if s.startswith("friend"):
        return "friends"
    if "not sure" in s:
        return "unsure"
    return ""


def _iso_to_ms(v: Any) -> Optional[int]:
    from datetime import datetime, timezone
    if v is None:
        return None
    if hasattr(v, "timestamp"):
        try:
            return int(v.timestamp() * 1000)
        except Exception:
            return None
    s = _s(v)
    if len(s) < 10:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def person_from_docs(uid: str, public: Dict[str, Any], private: Optional[Dict[str, Any]] = None,
                     user: Optional[Dict[str, Any]] = None) -> Person:
    """Builds a Person from publicProfiles/{uid}, profiles/{uid} (private) and users/{uid}."""
    private = private or {}
    user = user or {}

    def pick(*keys: str) -> Any:
        for src in (private, public):
            for k in keys:
                v = src.get(k)
                if v not in (None, "", [], {}):
                    return v
        return None

    tests: Dict[str, int] = {}
    for src in (public, private):
        for k, v in src.items():
            if k.startswith("tests_") and k.endswith("_scorePercent"):
                tid = k[len("tests_"):-len("_scorePercent")]
                pct = _to_int(v)
                if tid and pct is not None:
                    tests[tid] = max(0, min(100, pct))

    last_active = None
    for k in ("lastAppOpenAtUtcIso", "lastAppOpenAt", "lastSeenAtIso"):
        last_active = _iso_to_ms(user.get(k))
        if last_active:
            break

    lat = _to_float(pick("lat", "latitude"))
    lon = _to_float(pick("lon", "lng", "longitude"))
    if lat == 0 and lon == 0:
        lat = lon = None

    card_photos = [p for p in _str_list(public.get("photos")) if p.startswith("http")]
    if card_blurred(uid, public):
        # Private photos: the card holds server-made blurs. They are never a candidate's photos ('no_photos' then
        # keeps them out of everyone's Fates, so nothing is revealed), and the face check is held against the real
        # links on the private doc, so a private viewer keeps Daily Fates.
        photos: List[str] = []
        verify_photos = own_private_photos(uid, private)
    else:
        photos = verify_photos = card_photos
    fates_prefs = private.get("fatesPrefs") if isinstance(private.get("fatesPrefs"), dict) else {}

    def fact(*keys: str) -> str:
        """A biodata fact as families see it: the public card first (it is what the app shows), then the profile."""
        for src in (public, private):
            for k in keys:
                v = _s(src.get(k))
                if v:
                    return v
        return ""

    return Person(
        uid=uid,
        first_name=_s(pick("firstName", "name")).split(" ")[0] if pick("firstName", "name") else "",
        age=_to_int(pick("age")),
        gender=norm_gender(pick("gender")),
        looking_for=norm_looking_for(pick("settings_lookingForGender", "lookingForGender")),
        city=_s(pick("cityName", "city")),
        lat=lat, lon=lon,
        bio=_s(pick("bio", "aboutMe"))[:600],
        interests=_str_list(pick("interests")),
        languages=_str_list(pick("languages")),
        religion=_s(pick("religion", "settings_religion")),
        community=_s(pick("community", "settings_community")),
        intent=norm_intent(pick("relationshipIntent", "settings_goal", "relationshipGoal")),
        smoking=_s(pick("smoking")), drinking=_s(pick("drinking")), pets=_s(pick("pets")),
        workout=_s(pick("workout")), social=_s(pick("social")),
        have_children=_s(pick("settings_haveChildren", "haveChildren")),
        want_children=_s(pick("settings_wantChildren", "wantChildren")),
        relocate=_s(pick("settings_relocate", "relocate")),
        age_min=_to_int(pick("settings_ageMin", "preferredAgeMin")),
        age_max=_to_int(pick("settings_ageMax", "preferredAgeMax")),
        radius_km=_to_float(pick("settings_radiusKm", "maxDistanceKm")),
        require_religion=_to_bool(pick("settings_requireReligion", "requireReligionMatch")),
        require_community=_to_bool(pick("settings_requireCommunity", "requireCommunityMatch")),
        tests=tests,
        photos=photos,
        discoverable=public.get("isDiscoverable") is True,
        face_verified=verified_now(private, user, verify_photos),
        boosted=_future_iso(public.get("boostActiveUntilUtcIso")),
        geo_precise=precise_geo_source(pick("geoSource")),
        units="mi" if _norm(pick("settings_units")) == "mi" else "km",
        show_distance=pick("settings_showDistance") is not False,
        premium=_to_bool(public.get("isPremium")),
        last_active_ms=last_active,
        stars_pref=_s(fates_prefs.get("stars") or "little") if _s(fates_prefs.get("stars")) in WEIGHTS else "little",
        use_tests=fates_prefs.get("useTests") is not False,
        profile_for=_norm(pick("profileFor")),
        marriage_intent=private.get("marriageIntent") is True,
        marital_status=fact("maritalStatus"),
        children=fact("children"),
        diet=fact("diet"),
        education=fact("education"),
        profession=fact("profession"),
        caste=fact("caste"),
        manglik=fact("manglik"),
        income=_s(public.get("annualIncome")),
        family_values=fact("familyValues"),
        prefs=prefs_from_doc(private),
    )


def card_blurred(uid: str, public: Dict[str, Any]) -> bool:
    """The card shows blurs instead of the real photos: photoPrivacy is 'accepted', or a card photo lives in the
    server's blur folder (users/{uid}/blur/, either URL kind)."""
    if _s(public.get("photoPrivacy")) == "accepted":
        return True
    marker = f"users/{uid}/blur/"
    return any(marker in unquote(p) for p in _str_list(public.get("photos")))


def own_private_photos(uid: str, private: Dict[str, Any]) -> List[str]:
    """The real photo links of a private profile, as the face check uses them: profiles.photos, first 6, only the
    user's own Storage photos (face_live.own_photo_url)."""
    links = [p for p in _str_list(private.get("photos")) if p.startswith("http")][:6]
    return [p for p in links if face_live.own_photo_url(p, uid)]


# =========================
# MATRIMONY: POOLS (plan §2 C1) AND PARTNER PREFERENCES (§2 C11)
# =========================
QUICK_MALE_MIN_AGE = 21           # default of serverState/pools.quickMaleMinAge (the owner may lower it to 18)
GROOM_MIN_AGE = 21                # a man looking for a bride through Rishtas
SELF_PROFILE = ("", "self")       # profileFor of a profile someone made for themselves
CONSENT_KEYS = ("intentConsentAtIso", "consentDeclaredAtIso")   # self / managed (family-made) consent
KUNDLI_MODES = ("show", "rank", "18", "24")
KUNDLI_RANK_FAITHS = ("hindu", "jain", "sikh", "buddhist")
# pref_strict tokens a family can mark "Must have". Never saved: religion and marital status are must-haves.
STRICT_TOKENS = ("religions", "communities", "maritalStatuses", "diets", "education")
DEFAULT_STRICT = ("religions", "maritalStatuses")
# Preferences that are must-haves by what they say ("Caste no bar" / chosen castes, Manglik "Must match", an income
# minimum, "Prefer no children"). Work (professions) and height only ever rank, never hide anyone.
ALWAYS_MUST = ("castes", "manglik", "children", "income")
PREF_LABELS = {"religions": "Religion", "communities": "Mother tongue", "castes": "Caste", "manglik": "Manglik",
               "maritalStatuses": "Marital status", "children": "Children", "diets": "Food", "education": "Education",
               "professions": "Work", "height": "Height", "income": "Income"}
MIXED_CASTE = "inter-caste / mixed"
PARTIAL_MANGLIK = "partial (anshik)"
# Biodata.IncomeRanges codes -> lakh minimum; anything else (prefer_not, "", junk) is unknown and never excludes.
INCOME_RANKS = {"0-2l": 0, "2-4l": 2, "4-7l": 4, "7-10l": 7, "10-15l": 10, "15-20l": 15, "20-30l": 20,
                "30-50l": 30, "50l-1cr": 50, "1cr+": 100}
INCOME_MINIMUMS = (4, 7, 10, 15, 20, 30, 50, 100)
# RishtaMatch.EducationRank over the exact Biodata.Educations strings; anything else is 0 (unknown).
EDUCATION_RANKS = {"high school": 1, "diploma": 2, "bachelor's": 3, "law (llb)": 3, "master's": 4, "mba": 4,
                   "ca / cs": 4, "mbbs / medical": 4, "doctorate": 5}
VEG_DIETS = ("vegetarian", "jain food", "vegan")
FAMILY_VALUES = {"traditional": 0, "moderate": 1, "liberal": 2}

_CASTE_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caste_list.json")
_caste_state: Dict[str, Any] = {"loaded": False, "umbrellas": None}


def load_caste_list(data: Optional[Dict[str, Any]]) -> None:
    """Sets the caste umbrellas from a CasteList export ({entries: [{name, umbrella, religions, tongues}], blocked});
    None means "no list" (exact caste matching only)."""
    umbrellas: Optional[Dict[str, str]] = None
    if isinstance(data, dict):
        umbrellas = {}
        for e in data.get("entries") or []:
            if isinstance(e, dict) and _norm(e.get("name")):
                umbrellas[_norm(e.get("name"))] = _norm(e.get("umbrella"))
    _caste_state.update(loaded=True, umbrellas=umbrellas)


def caste_umbrellas() -> Optional[Dict[str, str]]:
    """name -> umbrella (both case-folded) from caste_list.json, the export the app and the functions share.
    None while the file isn't there."""
    if not _caste_state["loaded"]:
        try:
            with open(_CASTE_LIST_PATH, encoding="utf-8") as f:
                load_caste_list(json.load(f))
        except (OSError, ValueError):
            load_caste_list(None)
    return _caste_state["umbrellas"]


def unstated(v: Any) -> bool:
    """Biodata.IsUnstated: empty, "Prefer not to say" or "Don't know" (unknown never hides anyone)."""
    s = _norm(v)
    return not s or s.startswith("prefer not") or s in ("don't know", "don’t know")


def pref_fits(wanted: Any, value: Any) -> bool:
    """Biodata.PrefFits: nothing wanted, an unstated value, or one of the wanted values (ignoring case)."""
    want = [_norm(w) for w in _str_list(wanted)]
    return not want or unstated(value) or _norm(value) in want


def caste_fits(wanted: Any, theirs: Any) -> bool:
    """CasteList.Fits: "Caste no bar" (nothing wanted), an unstated or "Inter-caste / Mixed" caste, or a wanted one.
    With caste_list.json an umbrella fits its communities ("Brahmin" fits "Brahmin - Iyer", never the other way);
    without it only the exact caste fits."""
    want = [_norm(w) for w in _str_list(wanted)]
    t = _norm(theirs)
    if not want or unstated(theirs) or t == MIXED_CASTE or t in want:
        return True
    umbrellas = caste_umbrellas()
    if umbrellas is None:
        return False
    if umbrellas.get(t) in want:
        return True
    return any(t.startswith(w + " - ") for w in want)


def manglik_fits(mine: Any, theirs: Any) -> bool:
    """Biodata.ManglikFits: fits when either side is unstated or Partial (Anshik), otherwise only the same answer."""
    a, b = _norm(mine), _norm(theirs)
    if unstated(mine) or unstated(theirs) or PARTIAL_MANGLIK in (a, b):
        return True
    return a == b


def income_rank(code: Any) -> int:
    r = INCOME_RANKS.get(_norm(code))
    return -1 if r is None else r


def income_at_least(code: Any, min_lakh: int) -> bool:
    """Biodata.IncomeAtLeast: no minimum, an unknown or private income, or a band from the minimum up."""
    rank = income_rank(code)
    return min_lakh <= 0 or rank == -1 or rank >= min_lakh


def normal_min_lakh(v: int) -> int:
    """Biodata.NormalMinLakh: the largest allowed minimum not above v, or 0 below 4."""
    return max([m for m in INCOME_MINIMUMS if m <= (v or 0)], default=0)


def education_rank(v: Any) -> int:
    return EDUCATION_RANKS.get(_norm(v), 0)


def prefs_from_doc(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The partner preferences saved on profiles/{uid} (private). strict is None when pref_strict was never saved."""
    d = d or {}
    raw_strict = d.get("pref_strict")
    strict = [t for t in (_s(x) for x in raw_strict) if t] if isinstance(raw_strict, (list, tuple)) else None
    kundli = _norm(d.get("pref_kundli"))
    return {
        "religions": _str_list(d.get("pref_religions")),
        "communities": _str_list(d.get("pref_communities")),
        "castes": _str_list(d.get("pref_castes")),
        "manglik": "match" if _norm(d.get("pref_manglik")) == "match" else "",
        "maritalStatuses": _str_list(d.get("pref_maritalStatuses")),
        "childrenOk": "none" if _norm(d.get("pref_childrenOk")) == "none" else "any",
        "diets": _str_list(d.get("pref_diets")),
        "educationMin": max(0, _to_int(d.get("pref_educationMin")) or 0),
        "professions": _str_list(d.get("pref_professions")),
        "incomeMinLakh": normal_min_lakh(_to_int(d.get("pref_incomeMinLakh")) or 0),
        "kundli": kundli if kundli in KUNDLI_MODES else "",
        "strict": strict,
        "ageMin": max(0, _to_int(d.get("pref_ageMin")) or 0),
        "ageMax": max(0, _to_int(d.get("pref_ageMax")) or 0),
        "scope": _norm(d.get("pref_scope")),
    }


def pref_misses(prefs: Dict[str, Any], other: Person, own_manglik: str) -> List[str]:
    """The preferences (tokens, PREF_LABELS order) `other` clearly misses. A value they didn't give never misses."""
    out: List[str] = []
    if not pref_fits(prefs.get("religions"), other.religion):
        out.append("religions")
    if not pref_fits(prefs.get("communities"), other.community):
        out.append("communities")
    if not caste_fits(prefs.get("castes"), other.caste):
        out.append("castes")
    if prefs.get("manglik") == "match" and not manglik_fits(own_manglik, other.manglik):
        out.append("manglik")
    if not pref_fits(prefs.get("maritalStatuses"), other.marital_status):
        out.append("maritalStatuses")
    if prefs.get("childrenOk") == "none" and _norm(other.children).startswith("yes"):
        out.append("children")
    if not pref_fits(prefs.get("diets"), other.diet):
        out.append("diets")
    need, rank = int(prefs.get("educationMin") or 0), education_rank(other.education)
    if need > 0 and rank > 0 and rank < need:
        out.append("education")
    if not pref_fits(prefs.get("professions"), other.profession):
        out.append("professions")
    if not income_at_least(other.income, int(prefs.get("incomeMinLakh") or 0)):
        out.append("income")
    return out


def must_haves(prefs: Dict[str, Any]) -> set:
    """Which preferences hide people: the "Must have" marks (religion and marital status when never saved) plus the
    ones that are must-haves by what they say. Work and height never hide anyone."""
    strict = prefs.get("strict")
    marked = DEFAULT_STRICT if strict is None else [t for t in strict if t in STRICT_TOKENS]
    return set(marked) | set(ALWAYS_MUST)


def _pref_gate(viewer: Person, cand: Person) -> Optional[str]:
    """Partner preferences both ways (rishta mode). My must-haves follow the default rule; theirs count only when
    their family saved pref_strict (the copy that says must-haves hide both ways). Unknown values pass.
    Returns None, or why the candidate is left out."""
    mine = must_haves(viewer.prefs)
    for t in pref_misses(viewer.prefs, cand, viewer.manglik):
        if t in mine:
            return f"pref_{t}"
    if cand.prefs.get("strict") is not None:
        theirs = must_haves(cand.prefs)
        for t in pref_misses(cand.prefs, viewer, cand.manglik):
            if t in theirs:
                return f"their_pref_{t}"
    return None


def effective_kundli(p: Person) -> str:
    """pref_kundli, else 'rank' for Hindu, Jain, Sikh, Buddhist, unstated or no religion, and 'show' for others."""
    k = p.prefs.get("kundli") or ""
    if k in KUNDLI_MODES:
        return k
    r = _norm(p.religion)
    return "rank" if (not r or r.startswith("prefer not") or r in KUNDLI_RANK_FAITHS) else "show"


def rishta_min_gunas(viewer: Person) -> Optional[int]:
    """Gunas every rishta-mode fate needs: none in 'show' mode, 24 for '24+ gunas only', else 18."""
    mode = effective_kundli(viewer)
    if mode == "show":
        return None
    return 24 if mode == "24" else MIN_GUNAS_FOR_STARS_PATH


def rishta_kundli_ok(viewer: Person, p: "Pair") -> bool:
    """Rishta mode: both charts and enough gunas on every path (unless my kundli mode is 'show'), and never below a
    minimum the other family chose ('18' / '24') when the gunas are known."""
    need = rishta_min_gunas(viewer)
    total = None if not p.stars else float(p.stars.get("total") or 0)
    if need is not None and (total is None or total < need):
        return False
    theirs = {"18": 18, "24": 24}.get(p.cand.prefs.get("kundli") or "")
    return not (theirs is not None and total is not None and total < theirs)


def rishta_ready(p: Person, viewer: bool = False) -> Person:
    """Rishta mode reads the partner preferences page: its age range (pref_ageMin/Max, else the settings ones) and, for
    the viewer, its area (pref_scope near = 50 km, all India = no limit, my state = 300 km, also when never saved)."""
    if p.prefs.get("ageMin"):
        p.age_min = int(p.prefs["ageMin"])
    if p.prefs.get("ageMax"):
        p.age_max = int(p.prefs["ageMax"])
    if viewer:
        scope = p.prefs.get("scope") or ""
        p.radius_km = 50.0 if scope == "near" else (20_000.0 if scope == "india" else 300.0)
        p.rishta = True
    return p


def ist_day(now: Optional[Any] = None) -> str:
    """Today's date in India ('YYYY-MM-DD'), the day the pool index fields are written in."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    return (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()


def _birth_date(v: Any) -> Optional[Any]:
    from datetime import date, datetime
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if hasattr(v, "timestamp"):
        try:
            from datetime import timezone
            return datetime.fromtimestamp(v.timestamp(), tz=timezone.utc).date()
        except Exception:
            return None
    s = _s(v)
    if len(s) < 10:
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


def years_on(birth: Any, today_day: str) -> Optional[int]:
    """Full years from a birth date to today_day ('YYYY-MM-DD'); None when the date is unusable."""
    from datetime import date
    b = _birth_date(birth)
    try:
        t = date.fromisoformat(str(today_day)[:10])
    except ValueError:
        return None
    if b is None or b.year <= 1900 or b >= t:
        return None
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))


def card_age(card: Dict[str, Any], today_day: str) -> int:
    """feed.ts cardAge: from a birth date on the card, else its stored age (missing = 18); under 18 counts as 0."""
    raw = next((card.get(k) for k in ("birthDate", "birthDateIso", "dob") if card.get(k) not in (None, "")), None)
    age = years_on(raw, today_day)
    if age is None:
        stored = _to_int(card.get("age"))
        age = 18 if stored is None else stored
    return 0 if age < 18 else age


def _pool_gender(v: Any) -> str:
    """feed.ts normGender for the pool rule (only 'male' matters there)."""
    s = _norm(v)
    if s in ("male", "man", "m", "men", "boy", "guy"):
        return "male"
    if s in ("female", "woman", "f", "women", "girl", "lady") or any(w in s for w in ("female", "woman", "women")):
        return "female"
    return "male" if ("male" in s or "man" in s) else ""


def pools_of(card: Optional[Dict[str, Any]], today_day: str, quick_male_min_age: int = QUICK_MALE_MIN_AGE) -> Dict[str, bool]:
    """The Python mirror of feed.ts poolsOf (plan §2 C1), read from the server-written card index:
    - inRishta: feedRishta is true and today has reached feedRishtaFromDay (21 for men, 18 for women). A card
      without feedRishta is never in rishta.
    - inQuick: feedQuick is true and today has reached feedQuickFromDay. A card indexed before 1.0.8 (no feedQuick)
      uses the legacy rule from its own facts: made for oneself, not marked for marriage, and old enough
      (men quick_male_min_age, everyone else 18)."""
    card = card or {}
    day = str(today_day or "")
    rishta_from = card.get("feedRishtaFromDay")
    in_rishta = card.get("feedRishta") is True and isinstance(rishta_from, str) and bool(rishta_from) and day >= rishta_from
    if card.get("feedQuick") is None:
        min_age = max(18, int(quick_male_min_age)) if _pool_gender(card.get("gender")) == "male" else 18
        in_quick = (_norm(card.get("profileFor")) in SELF_PROFILE and card.get("marriageIntent") is not True
                    and card_age(card, day) >= min_age)
    else:
        quick_from = card.get("feedQuickFromDay")
        in_quick = card.get("feedQuick") is True and day >= (quick_from if isinstance(quick_from, str) else "")
    return {"inRishta": bool(in_rishta), "inQuick": bool(in_quick)}


def rishta_viewer_block(private: Dict[str, Any], public: Optional[Dict[str, Any]], today_day: str) -> Optional[str]:
    """Rishta-mode Fates are for a profile that wants marriage (profiles.marriageIntent, never the card copy), and a
    man making his own profile must be 21 by profiles.birthDate. None, or 'needs_intent' / 'groom_age'."""
    private, public = private or {}, public or {}
    pf = _norm(private.get("profileFor"))
    if pf == "none" or private.get("marriageIntent") is not True:
        return "needs_intent"
    gender = norm_gender(private.get("gender") or public.get("gender"))
    if pf in SELF_PROFILE and gender == "male":
        age = years_on(private.get("birthDate"), today_day)
        if age is None or age < GROOM_MIN_AGE:
            return "groom_age"
    return None


def quick_viewer_block(private: Dict[str, Any]) -> Optional[str]:
    """Dating Fates (no mode) are only for someone's own profile: a family-made or kundli-only profile gets none."""
    return None if _norm((private or {}).get("profileFor")) in SELF_PROFILE else "no_quick"


def candidate_gate(viewer: Person, cand: Person, card: Dict[str, Any], private: Optional[Dict[str, Any]],
                   today_day: str, quick_male_min_age: int = QUICK_MALE_MIN_AGE) -> Optional[str]:
    """Mode rules for one candidate, on top of hard_filter. None when they may be brought, else why not.
    - Dating (no mode): the quick pool only, by the card index and again by the private, trusted facts, so a dating
      viewer never gets a family-made profile or one shown for marriage only.
    - Rishta mode: the rishta pool only, the private facts still say marriage with consent, never a kundli-only
      profile, then partner preferences both ways. The life stage is never read: a parent's account that carries a
      son's or daughter's profile keeps the parent's own life stage."""
    private = private or {}
    pools = pools_of(card, today_day, quick_male_min_age)
    pf = _norm(private.get("profileFor") if "profileFor" in private else card.get("profileFor"))
    if viewer.rishta:
        if not pools["inRishta"]:
            return "pool"
        if pf == "none":
            return "kundli_only"
        if private.get("marriageIntent") is not True or not any(_s(private.get(k)) for k in CONSENT_KEYS):
            return "pool"
        return _pref_gate(viewer, cand)
    if not pools["inQuick"]:
        return "pool"
    if pf not in SELF_PROFILE or (private.get("marriageIntent") is True and private.get("quickSwipeOk") is not True):
        return "pool"
    return None


# =========================
# HARD FILTERS
# =========================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def distance_km(a: Person, b: Person) -> Optional[float]:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return None
    return haversine_km(a.lat, a.lon, b.lat, b.lon)  # type: ignore[arg-type]


def _wants(seeker: Person, other: Person) -> bool:
    if not other.gender:
        return seeker.looking_for == "any"
    return seeker.looking_for in ("any", other.gender)


def _age_ok(seeker: Person, other: Person) -> bool:
    if other.age is None:
        return True
    lo = seeker.age_min if seeker.age_min else 18
    hi = seeker.age_max if seeker.age_max else 99
    return lo <= other.age <= hi


def intent_status(a: str, b: str) -> Optional[str]:
    """aligns | talk | differs | None (unknown)."""
    if not a or not b:
        return None
    if a == b:
        return "aligns"
    if "friends" in (a, b):
        return "differs"
    if "unsure" in (a, b) or "both" in (a, b):
        return "talk"
    return "talk"  # serious vs dating


def hard_filter(viewer: Person, cand: Person, now_ms: int, radius_km: Optional[float] = None) -> Optional[str]:
    """Returns None when the candidate may be shown, else the reason they are excluded."""
    if cand.uid == viewer.uid:
        return "self"
    if not cand.discoverable:
        return "not_discoverable"
    if not cand.photos:
        return "no_photos"
    if VERIFIED_ONLY and not cand.face_verified:
        return "not_verified"
    if not (_wants(viewer, cand) and _wants(cand, viewer)):
        return "gender"
    if not (_age_ok(viewer, cand) and _age_ok(cand, viewer)):
        return "age"
    if cand.last_active_ms is None or now_ms - cand.last_active_ms > ACTIVE_DAYS * 86_400_000:
        return "inactive"
    if intent_status(viewer.intent, cand.intent) == "differs":
        return "intent"
    for p, q in ((viewer, cand), (cand, viewer)):
        if p.require_religion and p.religion and q.religion and _norm(p.religion) != _norm(q.religion):
            return "religion"
        if p.require_community and p.community and q.community and _norm(p.community) != _norm(q.community):
            return "community"
    if _children_clash(viewer, cand):
        return "children"
    d = distance_km(viewer, cand)
    limit = radius_km if radius_km else (viewer.radius_km or DEFAULT_RADIUS_KM)
    if d is not None and d > max(limit, 1):
        return "distance"
    return None


def _yes(v: str) -> Optional[bool]:
    s = _norm(v)
    if not s or "prefer not" in s or "not sure" in s or "open" in s or "maybe" in s or "someday" in s:
        return None
    if s.startswith(("yes", "want", "have")) or s in ("true",):
        return True
    if s.startswith(("no", "don't", "dont", "never")) or s in ("false",):
        return False
    return None


def _children_clash(a: Person, b: Person) -> bool:
    wa, wb = _yes(a.want_children), _yes(b.want_children)
    return wa is not None and wb is not None and wa != wb


# =========================
# COMPONENTS
# =========================
def heart_component(viewer: Person, cand: Person) -> Dict[str, Any]:
    """Compares tests both people took, one row per category."""
    if not (viewer.use_tests and cand.use_tests):
        return {"score": None, "rows": [], "missing": [], "reason": "tests_off"}
    cat = tests_catalog()
    by_cat: Dict[str, List[Tuple[str, int, int]]] = {}
    for tid, pa in viewer.tests.items():
        pb = cand.tests.get(tid)
        meta = cat["tests"].get(tid)
        if pb is None or not meta:
            continue
        by_cat.setdefault(meta["category"], []).append((tid, pa, pb))

    rows = []
    sims = []
    complementary = {"love_style", "personality"}
    for c, items in by_cat.items():
        tid, pa, pb = max(items, key=lambda x: abs(x[1] - x[2]))  # the most telling difference
        diff = abs(pa - pb)
        if diff < 20:
            verdict = "alike"
            sim = 1.0 - diff / 100.0
        elif c in complementary:
            verdict = "complete"
            sim = 0.8
        elif diff < 45:
            verdict = "talk"
            sim = 0.62
        else:
            verdict = "differs"
            sim = 0.4
        sims.append(sim)
        meta = cat["tests"][tid]
        rows.append({
            "category": c, "categoryTitle": cat["categories"].get(c, c), "testId": tid, "testTitle": meta["title"],
            "you": pa, "them": pb, "lowPole": meta["lowPole"], "highPole": meta["highPole"], "verdict": verdict,
        })
    order = ["attachment", "love_style", "communication", "conflict", "values", "jealousy", "family_focus", "long_term", "personality"]
    rows.sort(key=lambda r: order.index(r["category"]) if r["category"] in order else 99)
    missing = [c for c in order if c not in by_cat]
    score = round(100 * sum(sims) / len(sims)) if sims else None
    return {"score": score, "rows": rows, "missing": missing}


def _cmp_row(key: str, label: str, a: str, b: str, status: Optional[str]) -> Dict[str, Any]:
    return {"key": key, "label": label, "you": a, "them": b, "status": status}


def _eq_status(a: str, b: str, soft: bool = False) -> Optional[str]:
    if not a or not b or "prefer not" in _norm(a) or "prefer not" in _norm(b):
        return None
    if _norm(a) == _norm(b):
        return "aligns"
    return "talk" if soft else "differs"


def _food_status(a: str, b: str) -> Optional[str]:
    """Food at home: the same aligns, vegetarian (or Jain, vegan) against non-vegetarian differs, the rest is a talk."""
    if unstated(a) or unstated(b):
        return None
    x, y = _norm(a), _norm(b)
    if x == y:
        return "aligns"
    if (x == "non-vegetarian" and y in VEG_DIETS) or (y == "non-vegetarian" and x in VEG_DIETS):
        return "differs"
    return "talk"


def _habit_level(v: str) -> Optional[int]:
    s = _norm(v)
    if not s or "prefer not" in s:
        return None
    if s.startswith(("never", "no", "non", "don")):
        return 0
    if any(w in s for w in ("rare", "social", "sometimes", "occasion")):
        return 1
    if any(w in s for w in ("regular", "often", "daily", "yes")):
        return 2
    return None


def home_component(viewer: Person, cand: Person) -> Dict[str, Any]:
    rows = []
    rows.append(_cmp_row("marriage", "Marriage and plans", viewer.intent, cand.intent, intent_status(viewer.intent, cand.intent)))
    wa, wb = _yes(viewer.want_children), _yes(cand.want_children)
    rows.append(_cmp_row("children", "Children", viewer.want_children, cand.want_children,
                         None if wa is None or wb is None else ("aligns" if wa == wb else "differs")))
    rows.append(_cmp_row("faith", "Faith", viewer.religion, cand.religion, _eq_status(viewer.religion, cand.religion)))
    rows.append(_cmp_row("community", "Community", viewer.community, cand.community, _eq_status(viewer.community, cand.community, soft=True)))
    ra, rb = _yes(viewer.relocate), _yes(cand.relocate)
    rows.append(_cmp_row("moving", "Moving cities", viewer.relocate, cand.relocate,
                         None if ra is None and rb is None else ("aligns" if (ra or rb) else "talk")))
    ha = [_habit_level(viewer.smoking), _habit_level(cand.smoking)]
    rows.append(_cmp_row("smoking", "Smoking", viewer.smoking, cand.smoking,
                         None if None in ha else ("aligns" if abs(ha[0] - ha[1]) == 0 else ("talk" if abs(ha[0] - ha[1]) == 1 else "differs"))))
    da = [_habit_level(viewer.drinking), _habit_level(cand.drinking)]
    rows.append(_cmp_row("drinking", "Drinking", viewer.drinking, cand.drinking,
                         None if None in da else ("aligns" if abs(da[0] - da[1]) <= 1 else "talk")))
    shared_langs = sorted({_s(x) for x in viewer.languages} & {_s(x) for x in cand.languages}, key=str.casefold)
    rows.append(_cmp_row("languages", "Languages", ", ".join(viewer.languages), ", ".join(cand.languages),
                         None if not viewer.languages or not cand.languages else ("aligns" if shared_langs else "talk")))
    if viewer.rishta:
        # What families compare first (rishta mode only, so dating picks stay as they were).
        rows.append(_cmp_row("marital", "Marital status", viewer.marital_status, cand.marital_status,
                             None if unstated(viewer.marital_status) or unstated(cand.marital_status)
                             else _eq_status(viewer.marital_status, cand.marital_status, soft=True)))
        rows.append(_cmp_row("food", "Food", viewer.diet, cand.diet, _food_status(viewer.diet, cand.diet)))
        va, vb = FAMILY_VALUES.get(_norm(viewer.family_values)), FAMILY_VALUES.get(_norm(cand.family_values))
        rows.append(_cmp_row("family_values", "Family values", viewer.family_values, cand.family_values,
                             None if va is None or vb is None else ("aligns", "talk", "differs")[abs(va - vb)]))
        ea, eb = education_rank(viewer.education), education_rank(cand.education)
        rows.append(_cmp_row("education", "Education", viewer.education, cand.education,
                             None if not ea or not eb else ("aligns" if abs(ea - eb) <= 1 else "talk")))
    known = [r for r in rows if r["status"]]
    points = {"aligns": 1.0, "talk": 0.5, "differs": 0.0}
    score = round(100 * sum(points[r["status"]] for r in known) / len(known)) if len(known) >= 3 else None
    counts = {k: sum(1 for r in known if r["status"] == k) for k in points}
    return {"score": score, "rows": rows, "counts": counts, "sharedLanguages": shared_langs}


def everyday_component(viewer: Person, cand: Person) -> Dict[str, Any]:
    ia = {_norm(x): x for x in viewer.interests}
    ib = {_norm(x): x for x in cand.interests}
    shared = [ia[k] for k in ia if k in ib]
    only_you = [ia[k] for k in ia if k not in ib]
    only_them = [ib[k] for k in ib if k not in ia]
    dice = (2 * len(shared) / (len(ia) + len(ib))) if (ia and ib) else None
    lifestyle = []
    for label, a, b in (("Pets", viewer.pets, cand.pets), ("Workout", viewer.workout, cand.workout), ("Social", viewer.social, cand.social)):
        st = _eq_status(a, b, soft=True)
        if st:
            lifestyle.append({"label": label, "you": a, "them": b, "status": st})
    parts = []
    if dice is not None:
        parts.append(min(1.0, 0.35 + dice))
    if lifestyle:
        parts.append(sum(1.0 if r["status"] == "aligns" else 0.5 for r in lifestyle) / len(lifestyle))
    score = round(100 * sum(parts) / len(parts)) if parts else None
    return {"score": score, "shared": shared, "onlyYou": only_you, "onlyThem": only_them, "lifestyle": lifestyle}


def stars_score(match: Optional[Dict[str, Any]]) -> Optional[int]:
    """Guna total -> 0..100, with uncancelled doshas pulling it down."""
    if not match:
        return None
    total = float(match["total"])
    score = total / 36.0 * 100.0
    for d in match.get("doshas", []):
        if d.get("present") and not d.get("cancelled"):
            score -= 12
    return max(0, min(100, round(score)))


def fate_score(parts: Dict[str, Optional[int]], stars_pref: str) -> Optional[int]:
    w = WEIGHTS.get(stars_pref, WEIGHTS["little"])
    num = den = 0.0
    for k, weight in w.items():
        v = parts.get(k)
        if v is None or weight <= 0:
            continue
        num += weight * v
        den += weight
    if den <= 0:
        return None
    return round(num / den)


# =========================
# PAIR EVALUATION
# =========================
@dataclass
class Pair:
    cand: Person
    distance_km: Optional[float]
    stars: Optional[Dict[str, Any]]      # ashtakoota result (or None)
    stars_score: Optional[int]
    heart: Dict[str, Any]
    home: Dict[str, Any]
    everyday: Dict[str, Any]
    fate: Optional[int]
    recency: float
    mutual: bool = False
    chose_you: bool = False      # they accepted the viewer as their fate (a like with a note, waiting)

    def part(self, path: str) -> Optional[int]:
        return {"stars": self.stars_score, "heart": self.heart["score"], "home": self.home["score"]}[path]


def evaluate_pair(viewer: Person, cand: Person, now_ms: int,
                  match_fn: Optional[Callable[[Person, Person], Optional[Dict[str, Any]]]] = None) -> Pair:
    stars = match_fn(viewer, cand) if match_fn and viewer.chart and cand.chart else None
    s_score = stars_score(stars)
    heart = heart_component(viewer, cand)
    home = home_component(viewer, cand)
    everyday = everyday_component(viewer, cand)
    parts = {"stars": s_score if viewer.stars_pref != "none" else None, "heart": heart["score"],
             "home": home["score"], "everyday": everyday["score"]}
    fate = fate_score(parts, viewer.stars_pref)
    days = (now_ms - (cand.last_active_ms or 0)) / 86_400_000
    recency = 1.0 if days <= 1 else (0.95 if days <= 3 else 0.88)
    return Pair(cand=cand, distance_km=distance_km(viewer, cand), stars=stars, stars_score=s_score,
                heart=heart, home=home, everyday=everyday, fate=fate, recency=recency)


def _eligible_for_path(viewer: Person, p: Pair, path: str) -> bool:
    if viewer.rishta and not rishta_kundli_ok(viewer, p):
        return False
    if p.fate is None or p.fate < (CHOOSER_MIN_FATE if p.chose_you else MIN_FATE_SCORE):
        return False
    if path == "stars":
        if viewer.stars_pref == "none" or not p.stars:
            return False
        return p.stars["total"] >= MIN_GUNAS_FOR_STARS_PATH
    if path == "heart":
        return p.heart["score"] is not None
    return p.home["score"] is not None


def rank_key(p: Pair, path: str, learned: Optional[Dict[str, Any]] = None) -> float:
    base = p.part(path) or 0
    boost = 6 if p.mutual else 0
    boost += CHOOSER_BOOST if p.chose_you else 0
    boost += 2 if p.cand.face_verified else 0
    boost += BOOST_RANK if p.cand.boosted else (PREMIUM_RANK if p.cand.premium else 0)
    penalty = 0.0
    if learned and p.distance_km is not None:
        # "too far" skips teach Mithu that distance matters more to this person.
        far = float(learned.get("farPenaltyPerKm") or 0.0)
        penalty += far * max(0.0, p.distance_km - 5.0)
    return (0.7 * base + 0.3 * (p.fate or 0) + boost - penalty) * p.recency


# =========================
# LEARNING FROM THE VIEWER (bounded, per person)
# =========================
SKIP_REASONS = ("too_far", "faith", "age", "not_my_type", "other")


def learn(learned: Optional[Dict[str, Any]], event: str, path: Optional[str] = None,
          reason: Optional[str] = None) -> Dict[str, Any]:
    """Updates the per-person tuning stored in fatesMemory.learned. Every value is clamped so a few
    odd days can never make the picks strange."""
    l = dict(learned or {})
    opens = dict(l.get("pathOpens") or {})
    if event == "open" and path in PATHS:
        opens[path] = min(1000, int(opens.get(path) or 0) + 1)
    l["pathOpens"] = opens
    if event == "skip":
        skips = dict(l.get("skipReasons") or {})
        r = reason if reason in SKIP_REASONS else "other"
        skips[r] = min(1000, int(skips.get(r) or 0) + 1)
        l["skipReasons"] = skips
        if r == "too_far":
            l["farPenaltyPerKm"] = round(min(0.6, float(l.get("farPenaltyPerKm") or 0.0) + 0.08), 3)
    if event == "accept":
        # Accepting a far fate relaxes the distance penalty again.
        l["farPenaltyPerKm"] = round(max(0.0, float(l.get("farPenaltyPerKm") or 0.0) - 0.04), 3)
    return l


def preferred_path_order(learned: Optional[Dict[str, Any]]) -> List[str]:
    """Paths the person opens most come first on the Today screen (ties keep Stars, Heart, Home)."""
    opens = (learned or {}).get("pathOpens") or {}
    return sorted(PATHS, key=lambda p: (-int(opens.get(p) or 0), PATHS.index(p)))


def shortlist(viewer: Person, pairs: List[Pair], per_path: int = 3,
              learned: Optional[Dict[str, Any]] = None) -> Dict[str, List[Pair]]:
    """Top candidates for each path; a person can appear under more than one path here."""
    out: Dict[str, List[Pair]] = {}
    for path in PATHS:
        pool = [p for p in pairs if _eligible_for_path(viewer, p, path)]
        pool.sort(key=lambda p: rank_key(p, path, learned), reverse=True)
        out[path] = pool[:per_path]
    return out


def chooser_path(viewer: Person, p: Pair) -> Optional[str]:
    """The path a chooser is brought on: where the pair is strongest, among the paths they qualify for."""
    fits = [path for path in PATHS if _eligible_for_path(viewer, p, path)]
    return max(fits, key=lambda path: p.part(path) or 0) if fits else None


def place_chooser(viewer: Person, short: Dict[str, List[Pair]], chooser: Pair) -> Optional[str]:
    """Puts today's chooser first on their best path (and nowhere else). Returns that path, or None when
    they fit no path. The caller must keep them first there even if the AI prefers someone else."""
    path = chooser_path(viewer, chooser)
    if path is None:
        return None
    for k in short:
        short[k] = [x for x in short[k] if x.cand.uid != chooser.cand.uid]
    short.setdefault(path, [])
    short[path].insert(0, chooser)
    return path


def promoted(p: Pair) -> bool:
    """Brought partly because they paid (a boost). Someone who chose the viewer never counts."""
    return p.cand.boosted and not p.chose_you


def resolve_paths(short: Dict[str, List[Pair]], choice: Optional[Dict[str, int]] = None,
                  max_promoted: int = MAX_PROMOTED_PER_DAY) -> Dict[str, Pair]:
    """One distinct person per path, at most `max_promoted` boosted people. `choice` is the AI's
    preferred index per path (optional)."""
    used: set = set()
    result: Dict[str, Pair] = {}
    promoted_left = max_promoted
    # Scarcer paths pick first so a person who fits only one path is not taken by another.
    for path in sorted(PATHS, key=lambda k: len(short.get(k, []))):
        options = short.get(path, [])
        if not options:
            continue
        preferred = (choice or {}).get(path)
        ordered = list(options)
        if preferred is not None and 0 <= preferred < len(options):
            ordered.insert(0, ordered.pop(preferred))
        for p in ordered:
            if p.cand.uid in used:
                continue
            if promoted(p):
                if promoted_left <= 0:
                    continue
                promoted_left -= 1
            used.add(p.cand.uid)
            result[path] = p
            break
    return result


def exposure_cap(p: Person, base: int) -> int:
    """How many people may get this person as a fate today: a boost doubles it."""
    return base * 2 if p.boosted else base


# =========================
# CARD + REPORT PAYLOADS
# =========================
def pronoun(p: Person) -> Dict[str, str]:
    if p.gender == "female":
        return {"subj": "she", "obj": "her", "poss": "her"}
    if p.gender == "male":
        return {"subj": "he", "obj": "him", "poss": "his"}
    return {"subj": "they", "obj": "them", "poss": "their"}


def teaser(path: str, p: Pair, viewer: Optional[Person] = None) -> Dict[str, Any]:
    c = p.cand
    guna = None if not p.stars else p.stars["total"]
    return {
        "path": path, "label": PATH_LABELS[path], "age": c.age, "city": c.city,
        "distance": distance_text(viewer, p) if viewer else "",
        "fate": p.fate, "part": p.part(path), "gunas": guna, "mutual": p.mutual, "choseYou": p.chose_you,
        "verified": c.face_verified,
    }


# One plain line per koota at full points, heaviest first (Nadi 8 ... Vashya 2; Varna's single point says little).
KOOTA_HOOKS = {
    "Nadi": "your energies balance each other",
    "Bhakoot": "your Moon signs bless a shared home",
    "Gana": "your temperaments match",
    "Graha Maitri": "your Moon lords are friends",
    "Yoni": "there’s a natural ease between you",
    "Tara": "your birth stars favour each other",
    "Vashya": "you draw each other in",
}
HEART_ALIKE_HOOKS = {
    "attachment": "You feel safe in love the same way.",
    "love_style": "You show love the same way.",
    "communication": "You talk things through the same way.",
    "conflict": "You handle a fight the same way.",
    "values": "You care about the same things.",
    "jealousy": "You trust the same way.",
    "family_focus": "Family means the same to you both.",
    "long_term": "You picture the long run the same way.",
    "personality": "Your personalities are alike.",
}
HEART_COMPLETE_HOOKS = {
    "love_style": "You show love in ways that complete each other.",
    "personality": "Your personalities complete each other.",
}
# Home rows as nouns for "You agree on ...", most important first; the two habits share one noun.
HOME_NOUNS = [("marriage", "marriage plans"), ("children", "children"), ("faith", "faith"),
              ("family_values", "family values"), ("food", "food at home"), ("moving", "where to live"),
              ("community", "community"), ("smoking", "habits"), ("drinking", "habits")]


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def template_hook(path: str, p: Pair) -> str:
    """A short, name-free line for the card when the AI didn't write one."""
    if path == "stars" and p.stars:
        lo, hi = p.stars.get("totalMin", p.stars["total"]), p.stars.get("totalMax", p.stars["total"])
        total = f"{lo:g}–{hi:g} of 36 gunas" if lo != hi else f"{p.stars['total']:g} of 36 gunas"
        full = {k["name"] for k in p.stars["kootas"] if k["points"] >= k["max"]}
        best = next((line for name, line in KOOTA_HOOKS.items() if name in full), None)
        return f"{total}, and {best}." if best else f"{total} between you."
    if path == "heart":
        rows = p.heart["rows"]
        for verdict, lines in (("alike", HEART_ALIKE_HOOKS), ("complete", HEART_COMPLETE_HOOKS)):
            hit = next((r for r in rows if r["verdict"] == verdict and r["category"] in lines), None)
            if hit:
                return lines[hit["category"]]
        return "Your hearts work in ways that fit."
    status = {r["key"]: r["status"] for r in p.home["rows"]}
    nouns: List[str] = []
    for key, noun in HOME_NOUNS:
        if status.get(key) == "aligns" and noun not in nouns:
            nouns.append(noun)
    nouns = nouns[:2]
    if status.get("languages") == "aligns" and len(nouns) < 2:
        return f"You agree on {nouns[0]} and share a language." if nouns else "You share a language and want a similar life."
    if nouns:
        return f"You agree on {_and(nouns)}."
    return "You want a similar life."


# Kept as written inside a sentence; other interests go lower case (same list as the app's BioWriter).
PROPER_INTERESTS = {w.casefold() for w in (
    "Bollywood", "Hollywood", "YouTube", "TikTok", "Instagram", "Netflix", "K-pop", "K-drama", "IPL", "Formula 1", "F1",
    "Diwali", "Holi", "Garba", "Bhangra", "Carnatic", "Sufi", "Ghazal", "Kathak", "Bharatanatyam", "Marvel", "Disney", "Anime", "Yoga")}


def _interest_in_sentence(s: str) -> str:
    s = s.strip()
    return s if s.casefold() in PROPER_INTERESTS else s.lower()


def template_why(path: str, p: Pair) -> str:
    """The reason shown once a path is open, when the AI didn't write one: the whole picture, this path first."""
    bits: Dict[str, str] = {}
    # Gunas are a reason only from the usual minimum up; below it they'd read as praise for a weak match.
    if p.stars and p.stars["total"] >= MIN_GUNAS_FOR_STARS_PATH:
        bits["stars"] = f"{p.stars['total']:g} of 36 gunas"
    if p.heart.get("score") is not None:
        bits["heart"] = f"hearts {p.heart['score']}% in tune"
    counts = p.home.get("counts") or {}
    known = sum(counts.values())
    if p.home.get("score") is not None and known:
        bits["home"] = f"{counts.get('aligns', 0)} of {known} life answers alike"
    order = [path] + [k for k in PATHS if k != path]
    parts = [bits[k] for k in order if k in bits]
    line = (_and(parts)[:1].upper() + _and(parts)[1:] + ".") if parts else ""
    shared = [_interest_in_sentence(s) for s in p.everyday.get("shared", [])[:2] if s.strip()]
    if shared:
        line = f"{line} You both love {_and(shared)}.".strip()
    return line or template_hook(path, p)


def facts_for_ai(viewer: Person, path: str, p: Pair) -> Dict[str, Any]:
    """The only facts the AI may use for this pair."""
    c = p.cand
    facts: Dict[str, Any] = {
        "path": path, "fate": p.fate, "their_first_name": c.first_name, "their_age": c.age,
        "their_city": c.city, "pronoun": pronoun(c)["subj"],
        "distance": distance_text(viewer, p) or None,
        "their_bio": c.bio[:280], "your_bio": viewer.bio[:200],
        "shared_interests": p.everyday["shared"][:6],
        "home": [{"topic": r["label"], "status": r["status"]} for r in p.home["rows"] if r["status"]],
        "heart": [{"area": r["categoryTitle"], "verdict": r["verdict"], "you_lean": _lean(r["you"], r), "they_lean": _lean(r["them"], r)}
                  for r in p.heart["rows"]],
        "mutual_fate": p.mutual,
        "they_chose_you": p.chose_you,
    }
    if p.stars:
        facts["stars"] = {
            "gunas": p.stars["total"], "of": 36,
            "kootas": [{"name": k["name"], "points": k["points"], "max": k["max"]} for k in p.stars["kootas"]],
            "doshas": [{"name": d["name"], "present": d["present"], "cancelled": d.get("cancelled", False), "why": d.get("why", "")}
                       for d in p.stars["doshas"]],
            "your_moon": p.stars.get("aMoon"), "their_moon": p.stars.get("bMoon"),
        }
    return facts


def _lean(pct: int, row: Dict[str, Any]) -> str:
    if pct <= 35:
        return row["lowPole"]
    if pct >= 65:
        return row["highPole"]
    return "in between"


def report(viewer: Person, path: str, p: Pair) -> Dict[str, Any]:
    c = p.cand
    return {
        "path": path, "label": PATH_LABELS[path], "fate": p.fate,
        "parts": {"stars": p.stars_score, "heart": p.heart["score"], "home": p.home["score"], "everyday": p.everyday["score"]},
        "person": {
            "uid": c.uid, "firstName": c.first_name, "age": c.age, "city": c.city, "photos": c.photos[:6],
            "bio": c.bio, "verified": c.face_verified, "pronoun": pronoun(c),
            "distance": distance_text(viewer, p),
        },
        "stars": p.stars, "heart": p.heart, "home": p.home, "everyday": p.everyday, "mutual": p.mutual,
    }
