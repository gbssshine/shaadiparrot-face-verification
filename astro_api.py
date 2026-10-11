"""Today / Kundli / Family HTTP API (Cloud Run). main.py wires the shared helpers in with configure().

  POST /astro/today       city panchang, festivals and vrats ahead, a day line for each family member
  POST /astro/chart       a birth chart (planets by house, Lagna or Moon based), Vimshottari dashas
  POST /astro/match       36-guna match of any two people (me, someone typed in, or a rishta member), doshas
  POST /astro/match/report Mithu's full written match reading (Parrot+), kept so it is never charged twice
  POST /astro/good-days   dates that suit an event (wedding, griha pravesh, naming...) by the panchang
  POST /astro/question    Mithu's question of the day: vote once, then see your city and India
  POST /astro/places      city search for the panchang place and for birth places (India first)
  POST /astro/scores      36-guna score of me with each rishta card in my list (only the score leaves the server)

A rishta member (PersonIn.uid) is read from their own profile here, so their birth date, time and place never
reach the phone. Only people the caller could see in the list are matched: a public card that isn't hidden,
banned or suspended, and no block either way.

Data (server-only): dailyQuestions/{dayKey} vote counts, dailyQuestionVotes/{dayKey}__{uid},
astroReports/{uid}__{hash} written readings. Birth details sent for other people are used for the
answer and not stored.
"""
from __future__ import annotations

import hashlib
import math
import re
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date as Date, datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, BeforeValidator, Field

import astro_content
import astro_core as ac
import fates_astro
import fates_astro_chart as fc

logger = logging.getLogger("shaadiparrot-cloudrun.astro")
router = APIRouter(prefix="/astro")

DEFAULT_PLACE = {"name": "New Delhi", "lat": 28.6139, "lon": 77.209, "tz": "Asia/Kolkata", "key": "india"}
PER_MINUTE = 40
REPORTS_PER_DAY = 12
SCORES_PER_DAY = 300   # /scores calls a day that need fresh work (cached scores don't count)


@dataclass
class Deps:
    db: Callable[[], Any]
    verify_uid: Callable[[Optional[str]], str]
    resolve_place: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    places_query: Callable[[str, tuple], list]
    load_user_docs: Callable[[str], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]
    is_premium: Callable[[Dict[str, Any], Dict[str, Any], datetime], bool]
    complete: Callable[..., Tuple[str, str]]
    take_rate: Callable[[str, str, int, str], bool]


_deps: Optional[Deps] = None


def configure(deps: Deps) -> None:
    global _deps
    _deps = deps
    _nearest_row.cache_clear()


def _d() -> Deps:
    if _deps is None:
        raise HTTPException(status_code=503, detail="not configured")
    return _deps


# ------------------------------------------------------------------ small per-instance limiter
_hits: Dict[str, List[float]] = {}
_hits_lock = threading.Lock()


def _limit(uid: str) -> None:
    now = time.monotonic()
    with _hits_lock:
        recent = [t for t in _hits.get(uid, []) if now - t < 60.0]
        if len(recent) >= PER_MINUTE:
            raise HTTPException(status_code=429, detail="slow_down")
        recent.append(now)
        _hits[uid] = recent
        if len(_hits) > 5000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60.0]:
                _hits.pop(k, None)


def _auth(authorization: Optional[str]) -> str:
    uid = _d().verify_uid(authorization)
    _limit(uid)
    return uid


# ------------------------------------------------------------------ inputs
def _cut(n: int) -> BeforeValidator:
    return BeforeValidator(lambda v: v.strip()[:n] if isinstance(v, str) else v)


def _first(n: int) -> BeforeValidator:
    return BeforeValidator(lambda v: v[:n] if isinstance(v, list) else v)


# Over-long text is cut, not refused: one long name typed for a family member must not fail all of Today.
Text2 = Annotated[Optional[str], _cut(2)]
Text20 = Annotated[Optional[str], _cut(20)]
Text30 = Annotated[Optional[str], _cut(30)]
Text60 = Annotated[Optional[str], _cut(60)]
Text64 = Annotated[Optional[str], _cut(64)]
Text80 = Annotated[Optional[str], _cut(80)]
Text128 = Annotated[Optional[str], _cut(128)]


class PlaceIn(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    city: Text80 = None
    state: Text80 = None
    countryIso2: Text2 = None
    tz: Text64 = None


class PersonIn(BaseModel):
    id: Text128 = None
    name: Text60 = None
    relation: Text30 = None
    gender: Text20 = None
    birthDate: Text20 = None
    birthTime: Text20 = None
    birthCity: Text80 = None
    birthState: Text80 = None
    birthCountryIso2: Text2 = None
    me: bool = False
    # Another member (a rishta card): their birth details are read here and never sent to the phone.
    uid: Optional[str] = Field(default=None, max_length=128)


class TodayReq(BaseModel):
    place: Optional[PlaceIn] = None
    date: Text20 = None
    members: Annotated[List[PersonIn], _first(8)] = Field(default_factory=list)


class ChartReq(BaseModel):
    person: Optional[PersonIn] = None


class MatchReq(BaseModel):
    a: Optional[PersonIn] = None
    b: PersonIn


class GoodDaysReq(BaseModel):
    event: Annotated[str, _cut(30)] = "griha_pravesh"
    place: Optional[PlaceIn] = None
    days: int = 60
    people: Annotated[List[PersonIn], _first(6)] = Field(default_factory=list)


class QuestionReq(BaseModel):
    vote: Optional[int] = None
    # The question the app showed: a vote for yesterday's question (sent just after midnight) isn't counted.
    qid: Text64 = None
    place: Optional[PlaceIn] = None


class ScoresReq(BaseModel):
    uids: Annotated[List[str], _first(40)] = Field(default_factory=list)


class PlacesReq(BaseModel):
    q: Annotated[str, _cut(60)] = ""
    countryIso2: Text2 = "IN"


# ------------------------------------------------------------------ the caller
def _own_profile(uid: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    profile, public, user = _d().load_user_docs(uid)
    merged = {**public, **user, **profile}
    return merged, profile, user


class _Me:
    """The caller's own docs (the private profile wins, then the user doc, then the public card), read at most
    once per request however many times the request needs them."""

    def __init__(self, uid: str):
        self.uid = uid
        self._docs: Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None

    def docs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if self._docs is None:
            self._docs = _own_profile(self.uid)
        return self._docs

    def merged(self) -> Dict[str, Any]:
        return self.docs()[0]


# ------------------------------------------------------------------ places
def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in str(name or "").split())


def _city_key(name: str) -> str:
    k = "".join(ch for ch in str(name or "").lower() if ch.isalnum())[:40]
    return k or "india"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@lru_cache(maxsize=4096)
def _nearest_row(lat: float, lon: float) -> Optional[Tuple[str, str]]:
    """(city, tz) of the city nearest to a point rounded to ~1 km: remembered, since the lookup scans the table."""
    for r in (0.3, 1.0, 3.0):
        rows = _d().places_query(
            "SELECT city, lat, lon, tz, pop FROM cities WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (lat - r, lat + r, lon - r, lon + r))
        if rows:
            # The closest place, a big city winning over a village a little nearer.
            def cost(row):
                d2 = (row[1] - lat) ** 2 + (row[2] - lon) ** 2
                return d2 / (1.0 + min(int(row[4] or 0), 5_000_000) / 50_000.0)
            best = min(rows, key=cost)
            return str(best[0]), str(best[3] or "Asia/Kolkata")
    return None


def _nearest_city(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    hit = _nearest_row(round(lat, 2), round(lon, 2))
    if hit is None:
        return None
    city, tz = hit
    # "key" groups the question-of-the-day votes by the database's own city name, whatever the phone calls it.
    return {"name": _title(city), "lat": float(lat), "lon": float(lon), "tz": ac.zone(tz).key, "key": _city_key(city)}


def _city_place(city: str, state: str, iso2: str, country_name: str) -> Optional[Dict[str, Any]]:
    """A city (or at least its state) by name from the places database."""
    city = city.strip()
    iso2 = iso2.strip().upper()[:2] or ("" if country_name.strip() else "IN")
    found = _d().resolve_place({"birthCityName": city, "birthStateName": state, "birthCountryIso2": iso2,
                                "birthCountryName": country_name})
    if not found or not _num(found.get("lat")) or not _num(found.get("lon")):
        return None
    if found.get("precision") == "country" and iso2 in ("IN", ""):
        return None   # the middle of India says less than New Delhi does
    return {"name": city[:60], "lat": round(float(found["lat"]), 2), "lon": round(float(found["lon"]), 2),
            "tz": ac.zone(found.get("tz")).key, "key": _city_key(city)}


def _place(p: Optional[PlaceIn], own: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Where the panchang is for: the place the app sent (coordinates, else a city name), else the caller's own
    profile (its coordinates, else its city), else New Delhi. Coordinates are rounded to ~1 km: plenty for
    sunrise, and people nearby share the cached day frames. own() is only called when it is needed."""
    try:
        if p and _num(p.lat) and _num(p.lon) and -90 <= p.lat <= 90 and -180 <= p.lon <= 180:
            lat, lon = round(p.lat, 2), round(p.lon, 2)
            near = _nearest_city(lat, lon)
            name = (p.city or "").strip()[:60]
            if near:
                return {**near, "name": name or near["name"]}
            return {"name": name or "Your city", "lat": lat, "lon": lon, "tz": ac.zone(p.tz).key,
                    "key": _city_key(name) if name else "india"}
        if p and p.city:
            found = _city_place(p.city, p.state or "", p.countryIso2 or "IN", "")
            if found:
                return found
        doc = own() if own else None
        if doc:
            lat, lon = doc.get("lat"), doc.get("lon")
            if _num(lat) and _num(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                near = _nearest_city(round(float(lat), 2), round(float(lon), 2))
                if near:
                    return near
            city = str(doc.get("cityName") or doc.get("city") or "").strip()
            if city:
                found = _city_place(city, str(doc.get("stateName") or ""), str(doc.get("countryIso2") or ""),
                                    str(doc.get("countryName") or ""))
                if found:
                    return found
    except HTTPException:
        raise
    except Exception:
        logger.warning("place resolve failed", exc_info=True)
    return dict(DEFAULT_PLACE)


def _local_day(tz: str, raw: Optional[str] = None) -> Date:
    if raw:
        try:
            d = Date.fromisoformat(raw[:10])
            today = datetime.now(ac.zone(tz)).date()
            if abs((d - today).days) <= 400:
                return d
        except ValueError:
            pass
    return datetime.now(ac.zone(tz)).date()


# ------------------------------------------------------------------ members (rishta cards)
_UID_RE = re.compile(r"^[A-Za-z0-9_-]{6,128}$")


def _is_false(v: Any) -> bool:
    return v is False or (isinstance(v, str) and v.strip().lower() == "false")


def _as_utc(v: Any) -> Optional[datetime]:
    """A Firestore timestamp (a datetime here), an ISO string or epoch milliseconds."""
    try:
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if _num(v):
            return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc)
        if isinstance(v, str) and v.strip():
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _moderated(doc: Dict[str, Any]) -> bool:
    """Banned, or suspended and the suspension isn't over (the same test as hiddenByModeration in the rules)."""
    status = str(doc.get("moderationStatus") or "").strip().lower()
    if status == "banned":
        return True
    if status == "suspended":
        until = _as_utc(doc.get("moderationUntil"))
        return until is None or until > datetime.now(timezone.utc)
    return False


def _visible(public: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    """Someone the caller could meet in the rishta list."""
    if _is_false(public.get("isDiscoverable")) or _is_false(public.get("isProfileActive")):
        return False
    if _moderated(profile) or _moderated(public):
        return False
    # A kundli-only account (Biodata.KundliOnly) is not looking for a rishta.
    return not any(str(d.get("profileFor") or "").strip().lower() == "none" for d in (profile, public))


def _get_docs(db: Any, refs: List[Any]) -> Dict[str, Dict[str, Any]]:
    """path -> data for the refs that exist, in one round trip (get_all returns them in any order)."""
    return {s.reference.path: (s.to_dict() or {}) for s in db.get_all(refs) if s.exists}


def _member(uid: str, other: str, fallback_name: str) -> Tuple[Dict[str, Any], str, str]:
    """A rishta member's own profile (name, gender, birth details) for the calculation only."""
    other = other.strip()
    if other == uid or not _UID_RE.match(other):
        raise HTTPException(status_code=404, detail="member_unavailable")
    db = _d().db()
    if db is None:
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    users = db.collection("users")
    refs = [db.collection("profiles").document(other), db.collection("publicProfiles").document(other),
            users.document(uid).collection("blocks").document(other),
            users.document(other).collection("blocks").document(uid)]
    try:
        got = _get_docs(db, refs)
    except Exception:
        logger.warning("member read failed", exc_info=True)
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    profile, public = got.get(refs[0].path), got.get(refs[1].path)
    blocked = refs[2].path in got or refs[3].path in got
    # Hidden, blocked or gone all look the same from outside.
    if profile is None or public is None or blocked or not _visible(public, profile):
        raise HTTPException(status_code=404, detail="member_unavailable")
    doc = {**public, **profile}
    if not fc._parse_date(doc.get("birthDate")):
        raise HTTPException(status_code=422, detail="member_no_birth_date")
    return doc, _name_of(public) or _name_of(profile) or fallback_name or "Them", _gender(doc.get("gender"))


def _block_ids(db: Any, uid: str) -> Optional[Set[str]]:
    """Everyone I blocked and everyone who blocked me (blockUser keeps both lists in my tree)."""
    try:
        me = db.collection("users").document(uid)
        return {d.id for sub in ("blocks", "blockedBy") for d in me.collection(sub).limit(2000).stream()}
    except Exception:
        logger.warning("block lists read failed", exc_info=True)
        return None


# ------------------------------------------------------------------ people
def _person_profile(p: PersonIn) -> Dict[str, Any]:
    return {"birthDate": p.birthDate or "", "birthTime": p.birthTime or "", "birthCityName": p.birthCity or "",
            "birthStateName": p.birthState or "", "birthCountryIso2": (p.birthCountryIso2 or "").upper(),
            "birthCountryName": "India" if not p.birthCountryIso2 else ""}


def _gender(v: Any) -> str:
    s = str(v or "").strip().lower()
    return "male" if s in ("male", "man", "m", "boy") else ("female" if s in ("female", "woman", "f", "girl") else "")


def _name_of(doc: Dict[str, Any]) -> str:
    return str(doc.get("firstName") or doc.get("name") or doc.get("displayName") or "").strip()[:40]


def _is_member(p: Optional[PersonIn]) -> bool:
    return p is not None and not p.me and bool(p.uid)


def _resolve_person(me: _Me, p: Optional[PersonIn]) -> Tuple[Dict[str, Any], str, str]:
    """(profile-like dict, name, gender): the caller when p is empty or 'me', a rishta member by uid, or someone
    typed in. A typed person without a valid birth date is refused, never swapped for the caller."""
    if p is None or p.me:
        merged = me.merged()
        if not fc._parse_date(merged.get("birthDate")):
            raise HTTPException(status_code=422, detail="missing_birth_data")
        return merged, (p.name if p and p.name else _name_of(merged)) or "You", _gender(merged.get("gender"))
    if p.uid:
        return _member(me.uid, p.uid, (p.name or "").strip())
    prof = _person_profile(p)
    if not fc._parse_date(prof["birthDate"]):
        raise HTTPException(status_code=422, detail="bad_birth_date")
    return prof, (p.name or "").strip() or "Them", _gender(p.gender)


def _chart_for(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    place = None
    try:
        place = _d().resolve_place(profile)
    except Exception:
        place = None
    date = fc._parse_date(profile.get("birthDate"))
    time_ = fc._parse_time(profile.get("birthTime"))
    return fc.chart_from_birth(date, time_, place), place


# ------------------------------------------------------------------ /today
@router.post("/today")
def today(body: TodayReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    place = _place(body.place, _Me(uid).merged)
    day = _local_day(place["tz"], body.date)
    try:
        p = ac.panchang(day, place["lat"], place["lon"], place["tz"])
        ahead = ac.upcoming(day, 40, place["lat"], place["lon"], place["tz"], limit=8)
    except Exception:
        logger.exception("panchang failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    jd_rise = p.pop("_jd")["sunrise"]
    p.pop("choghadiya", None)
    today_moon = ac.moon_lon(jd_rise + 0.25)   # the Moon around mid-morning
    today_rashi, today_nak = int(today_moon // 30.0), int(today_moon // ac.NAK_SPAN) % 27

    family = []
    for m in body.members[:8]:
        if not fc._parse_date(m.birthDate):
            continue
        try:
            chart, _ = _chart_for(_person_profile(m))
            line = ac.day_line((m.name or "").strip()[:30], m.relation or "", chart["rashi"], chart["nak"], today_rashi, today_nak)
            family.append({"id": m.id, "name": m.name, "relation": m.relation, **line,
                           "moonSign": fates_astro.RASHIS[chart["rashi"]][0]})
        except Exception:
            logger.warning("family line failed", exc_info=True)
    return {"ok": True, "place": {"name": place["name"], "tz": place["tz"]}, "dayKey": day.isoformat(),
            "panchang": p, "upcoming": ahead, "family": family}


# ------------------------------------------------------------------ /chart
@router.post("/chart")
def chart(body: ChartReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    if _is_member(body.person):
        # Planets to the degree would give a member's birth date and time away: their kundli is only matched.
        raise HTTPException(status_code=403, detail="not_allowed")
    profile, name, _ = _resolve_person(_Me(uid), body.person)
    date = fc._parse_date(profile.get("birthDate"))
    time_ = fc._parse_time(profile.get("birthTime"))
    place = None
    try:
        place = _d().resolve_place(profile)
    except Exception:
        place = None
    try:
        out = ac.birth_chart(date, time_, place)
    except Exception:
        logger.exception("chart failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    out["name"] = name
    out["birthDate"] = profile.get("birthDate")
    out["birthTime"] = profile.get("birthTime") or None
    out["placeKnown"] = bool(place and place.get("precision") == "city")
    return {"ok": True, "chart": out}


# ------------------------------------------------------------------ /match
def _verdict(total: float) -> Dict[str, str]:
    if total >= 28:
        return {"word": "Excellent match", "tone": "great", "text": "Very few pairs score this high. The stars are on your side."}
    if total >= 24:
        return {"word": "Very good match", "tone": "good", "text": "Strong on most counts: a comfortable, caring match."}
    if total >= 18:
        return {"word": "Good match", "tone": "ok", "text": "18 or more is the traditional pass mark. Look at the doshas below."}
    return {"word": "Needs a closer look", "tone": "low", "text": "Under 18 points. Many happy couples score here: "
            "families usually read the full chart before deciding."}


def _named(text: str, a_name: str, b_name: str) -> str:
    return (text.replace("You are both", f"{a_name} and {b_name} are both")
            .replace("You are Manglik", f"{a_name} is Manglik").replace("They are Manglik", f"{b_name} is Manglik")
            .replace("Neither of you", "Neither of them"))


def _match(me: _Me, body: MatchReq) -> Dict[str, Any]:
    pa, a_name, a_gender = _resolve_person(me, body.a)
    pb, b_name, b_gender = _resolve_person(me, body.b)
    ca, _ = _chart_for(pa)
    cb, _ = _chart_for(pb)
    res = fates_astro.match_charts(ca, cb, a_gender, b_gender)
    viewer_is_a = body.a is None or body.a.me
    if not viewer_is_a:
        for d in res["doshas"]:
            d["why"] = _named(d["why"], a_name, b_name)
    res["verdict"] = _verdict(res["total"])
    res["a"] = {"name": a_name, "gender": a_gender, "manglik": fates_astro.manglik_status(ca), "timeKnown": ca.get("timeKnown")}
    res["b"] = {"name": b_name, "gender": b_gender, "manglik": fates_astro.manglik_status(cb), "timeKnown": cb.get("timeKnown")}
    # A member's Moon pada narrows down their birth time: their side shows the sign and nakshatra only.
    for side, person in (("aMoon", body.a), ("bMoon", body.b)):
        if _is_member(person) and isinstance(res.get(side), dict):
            res[side]["pada"] = 0
    return res


@router.post("/match")
def match(body: MatchReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    try:
        return {"ok": True, "match": _match(_Me(uid), body)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("match failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")


REPORT_SYSTEM = (
    "You are Mithu, the warm, witty parrot astrologer of the Shaadi Parrot app. Write a kundli match reading "
    "in simple English for an Indian family. Use ONLY the facts given (Ashtakoota points, doshas, Moon signs). "
    "Never invent planets, houses or dates. Never frighten: no curses, no 'danger', no paid remedies or pujas. "
    "Doshas are explained calmly with their cancellations. Speak about the two people by name. "
    "Format: exactly these five headings, each on its own line starting with '## ', then 2-4 short sentences: "
    "## The short answer\n## Where they click\n## Where they need patience\n## Doshas, honestly\n## Mithu's advice. "
    "Under 330 words. No emoji."
)

_report_hits: Dict[str, Tuple[str, int]] = {}


def _report_slot(uid: str) -> bool:
    """This instance's own count of fresh readings a day. The shared Firestore counter lets a call through when
    Firestore can't be reached, so the daily cap still holds here."""
    day = datetime.now(timezone.utc).date().isoformat()
    with _hits_lock:
        last_day, n = _report_hits.get(uid, ("", 0))
        n = n if last_day == day else 0
        if n >= REPORTS_PER_DAY:
            return False
        _report_hits[uid] = (day, n + 1)
        if len(_report_hits) > 5000:
            for k in [k for k, v in _report_hits.items() if v[0] != day]:
                _report_hits.pop(k, None)
    return True


@router.post("/match/report")
def match_report(body: MatchReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    me = _Me(uid)
    merged, _, user = me.docs()
    if not _d().is_premium(user, merged, datetime.now(timezone.utc)):
        raise HTTPException(status_code=402, detail="premium_required")
    try:
        res = _match(me, body)
    except HTTPException:
        raise
    except Exception:
        logger.exception("match failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    facts = {
        "a": res["a"]["name"], "b": res["b"]["name"], "total": res["total"], "range": [res["totalMin"], res["totalMax"]],
        "kootas": [(k["name"], k["points"], k["max"], k["meaning"]) for k in res["kootas"]],
        "doshas": [(d["name"], d["present"], d["cancelled"], d["why"]) for d in res["doshas"]],
        "aMoon": res["aMoon"], "bMoon": res["bMoon"], "precision": res["precision"],
    }
    key = hashlib.sha256(repr(sorted(facts.items())).encode("utf-8")).hexdigest()[:24]
    db = _d().db()
    ref = db.collection("astroReports").document(f"{uid}__{key}") if db else None
    if ref is not None:
        try:
            snap = ref.get()
            if snap.exists and (snap.to_dict() or {}).get("text"):
                return {"ok": True, "match": res, "report": snap.to_dict()["text"], "cached": True}
        except Exception:
            logger.warning("report cache read failed", exc_info=True)
    if not _report_slot(uid) or not _d().take_rate(uid, "astroReport", REPORTS_PER_DAY, "day"):
        raise HTTPException(status_code=429, detail="daily_cap")
    msgs = [{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": f"FACTS: {facts}"}]
    try:
        text, _ = _d().complete(msgs, 700, 0.6, "astro_match_report")
    except Exception:
        logger.exception("report llm failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    if ref is not None:
        try:
            ref.set({"uid": uid, "text": text, "a": res["a"]["name"], "b": res["b"]["name"], "total": res["total"],
                     "createdAtIso": datetime.now(timezone.utc).isoformat()})
        except Exception:
            logger.warning("report cache write failed", exc_info=True)
    return {"ok": True, "match": res, "report": text, "cached": False}


# ------------------------------------------------------------------ /good-days
@router.post("/good-days")
def good_days(body: GoodDaysReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    me = _Me(uid)
    place = _place(body.place, me.merged)
    start = _local_day(place["tz"])
    people = []
    me_done = False
    for m in body.people[:6]:
        if m.me:
            if me_done:
                continue
            me_done = True
        try:
            prof, name, _ = _resolve_person(me, m)
            c, _ = _chart_for(prof)
            people.append({"name": name, "rashi": c["rashi"], "nak": c["nak"]})
        except HTTPException:
            continue   # no usable birth date (or a member we can't read): the dates come without them
        except Exception:
            logger.warning("good days person failed", exc_info=True)
    try:
        out = ac.good_days(body.event, start, max(14, min(int(body.days or 60), 120)), place["lat"], place["lon"], place["tz"], people)
    except Exception:
        logger.exception("good days failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    out["place"] = {"name": place["name"], "tz": place["tz"]}
    return {"ok": True, **out}


# ------------------------------------------------------------------ /question
def _results(doc: Dict[str, Any], n: int, city_key: str, city_name: str) -> Dict[str, Any]:
    counts = doc.get("counts") or {}
    all_counts = [int(counts.get(str(i)) or 0) for i in range(n)]
    city = ((doc.get("cities") or {}).get(city_key) or {})
    city_counts = [int(city.get(str(i)) or 0) for i in range(n)]
    return {"total": sum(all_counts), "counts": all_counts,
            "city": {"name": city_name, "total": sum(city_counts), "counts": city_counts}}


@router.post("/question")
def question(body: QuestionReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    place = _place(body.place, _Me(uid).merged)
    day = datetime.now(ac.zone(place["tz"])).date()
    q = astro_content.question_for(day)
    out = {"ok": True, "question": {"id": q["id"], "day": day.isoformat(), "text": q["text"], "options": q["options"]},
           "voted": None, "results": None}
    db = _d().db()
    if db is None:
        return out
    from google.api_core import exceptions as gexc
    from google.cloud import firestore as fs
    qref = db.collection("dailyQuestions").document(day.isoformat())
    vref = db.collection("dailyQuestionVotes").document(f"{day.isoformat()}__{uid}")
    ck = place.get("key") or "india"
    city_name = place["name"] if ck != "india" else "India"
    vote = body.vote
    if vote is not None and body.qid and body.qid != q["id"]:
        # The day turned while yesterday's question was on screen: show today's, count nothing.
        vote = None
    try:
        vote_snap = vref.get()
        voted = (vote_snap.to_dict() or {}).get("option") if vote_snap.exists else None
        if voted is None and vote is not None and 0 <= vote < len(q["options"]):
            inc = {"qid": q["id"], "counts": {str(vote): fs.Increment(1)}}
            if ck != "india":
                inc["cities"] = {ck: {str(vote): fs.Increment(1)}}
            # The vote and the count are one write: never a vote without its count, or a count twice.
            batch = db.batch()
            batch.create(vref, {"uid": uid, "option": vote, "city": ck, "qid": q["id"],
                                "createdAtIso": datetime.now(timezone.utc).isoformat()})
            batch.set(qref, inc, merge=True)
            try:
                batch.commit()
                voted = vote
            except gexc.Conflict:   # AlreadyExists: a vote from another device got there first
                again = vref.get()
                voted = (again.to_dict() or {}).get("option") if again.exists else None
        if voted is not None:
            out["voted"] = voted
            out["results"] = _results(qref.get().to_dict() or {}, len(q["options"]), ck, city_name)
    except Exception:
        logger.warning("question vote failed", exc_info=True)
    return out


# ------------------------------------------------------------------ /places
@router.post("/places")
def places(body: PlacesReq, authorization: Optional[str] = Header(default=None)):
    _auth(authorization)
    q = " ".join(str(body.q or "").casefold().split())
    if len(q) < 2:
        return {"ok": True, "places": []}
    cc = (body.countryIso2 or "IN").strip().upper()[:2]
    # A prefix as a range on the (casefolded) name: uses the (cc, city) index, and % or _ typed in mean nothing.
    sql = ("SELECT c.city, s.state, c.cc, c.lat, c.lon, c.tz, c.pop FROM cities c LEFT JOIN states s ON s.id = c.state_id "
           "WHERE {where} AND c.city >= ? AND c.city < ? ORDER BY (c.city = ?) DESC, COALESCE(c.pop, 0) DESC LIMIT 12")
    upper = q + "\U0010ffff"
    try:
        rows = _d().places_query(sql.format(where="c.cc = ?"), (cc, q, upper, q))
        if len(rows) < 5:
            # Country by country through the same index (a plain "cc <> ?" would read the whole table).
            more = _d().places_query(sql.format(where="c.cc IN (SELECT cc FROM countries WHERE cc <> ?)"), (cc, q, upper, q))
            rows = list(rows) + list(more)[: 12 - len(rows)]
    except Exception:
        logger.warning("places search failed", exc_info=True)
        rows = []
    seen = set()
    out = []
    for city, state, ccode, lat, lon, tz, _ in rows:
        key = (city, state, ccode)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": _title(city), "state": _title(state or ""), "countryIso2": ccode,
                    "lat": lat, "lon": lon, "tz": ac.zone(tz).key})
    return {"ok": True, "places": out}


# ------------------------------------------------------------------ /scores
_scores_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_scores_lock = threading.Lock()
_SCORES_TTL = 3 * 3600.0
_SCORES_MAX = 20000
_SIG_KEYS = ("birthDate", "birthTime", "birthCityName", "birthStateName", "birthCountryIso2", "birthCountryName", "gender")


def _birth_sig(doc: Dict[str, Any]) -> str:
    """A short fingerprint of my own birth details: editing them gives fresh scores instead of cached ones."""
    raw = "|".join(str(doc.get(k) or "") for k in _SIG_KEYS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _trim_scores(now: float) -> None:
    """Called under _scores_lock: drop stale scores, then the oldest, to stay under the cap."""
    if len(_scores_cache) <= _SCORES_MAX:
        return
    for k in [k for k, v in _scores_cache.items() if now - v[0] > _SCORES_TTL]:
        _scores_cache.pop(k, None)
    extra = len(_scores_cache) - int(_SCORES_MAX * 0.9)
    if extra > 0:
        for k, _ in sorted(_scores_cache.items(), key=lambda kv: kv[1][0])[:extra]:
            _scores_cache.pop(k, None)


@router.post("/scores")
def scores(body: ScoresReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    merged, _, _ = _own_profile(uid)
    if not fc._parse_date(merged.get("birthDate")):
        return {"ok": True, "scores": {}, "missing": "me"}
    want = [u for u in dict.fromkeys(body.uids) if isinstance(u, str) and _UID_RE.match(u) and u != uid][:40]
    db = _d().db()
    if not want or db is None:
        return {"ok": True, "scores": {}}
    # Blocks either way are checked on every call (two short lists), so a block hides a score at once.
    blocked = _block_ids(db, uid)
    if blocked is None:
        return {"ok": True, "scores": {}}
    want = [u for u in want if u not in blocked]
    sig = _birth_sig(merged)
    out: Dict[str, Any] = {}
    now = time.monotonic()
    todo = []
    with _scores_lock:
        for u in want:
            hit = _scores_cache.get(f"{uid}|{sig}|{u}")
            if hit and now - hit[0] < _SCORES_TTL:
                out[u] = hit[1]
            else:
                todo.append(u)
    if todo and not _d().take_rate(uid, "astroScores", SCORES_PER_DAY, "day"):
        todo = []   # over today's cap: the scores already worked out still show
    fresh: Dict[str, Dict[str, Any]] = {}
    if todo:
        try:
            me_chart, _ = _chart_for(merged)
            me_gender = _gender(merged.get("gender"))
            refs = [(u, db.collection("profiles").document(u), db.collection("publicProfiles").document(u)) for u in todo]
            got = _get_docs(db, [r for _, pr, pu in refs for r in (pr, pu)])
        except Exception:
            logger.warning("scores failed", exc_info=True)
            refs, got = [], {}
        for u, pr, pu in refs:
            profile, public = got.get(pr.path), got.get(pu.path)
            if profile is None or public is None or not _visible(public, profile):
                continue
            p = {**public, **profile}
            if not fc._parse_date(p.get("birthDate")):
                continue
            try:   # one bad profile skips only itself
                c, _ = _chart_for(p)
                m = fates_astro.match_charts(me_chart, c, me_gender, _gender(p.get("gender")))
            except Exception:
                logger.warning("score failed for one profile", exc_info=True)
                continue
            fresh[u] = {"total": m["total"], "min": m["totalMin"], "max": m["totalMax"]}
    if fresh:
        with _scores_lock:
            for u, s in fresh.items():
                _scores_cache[f"{uid}|{sig}|{u}"] = (now, s)
            _trim_scores(now)
        out.update(fresh)
    return {"ok": True, "scores": {u: out[u] for u in want if u in out}}
