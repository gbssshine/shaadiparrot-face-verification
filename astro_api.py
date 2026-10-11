"""Today / Kundli / Family HTTP API (Cloud Run). main.py wires the shared helpers in with configure().

  POST /astro/today       city panchang, festivals and vrats ahead, a day line for each family member
  POST /astro/chart       a birth chart (planets by house, Lagna or Moon based), Vimshottari dashas
  POST /astro/match       36-guna match of any two people (or me and someone), doshas
  POST /astro/match/report Mithu's full written match reading (Parrot+), kept so it is never charged twice
  POST /astro/good-days   dates that suit an event (wedding, griha pravesh, naming...) by the panchang
  POST /astro/question    Mithu's question of the day: vote once, then see your city and India
  POST /astro/places      city search for the panchang place and for birth places (India first)

Data (server-only): dailyQuestions/{dayKey} vote counts, dailyQuestionVotes/{dayKey}__{uid},
astroReports/{uid}__{hash} written readings. Birth details sent for other people are used for the
answer and not stored.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

import astro_content
import astro_core as ac
import fates_astro
import fates_astro_chart as fc

logger = logging.getLogger("shaadiparrot-cloudrun.astro")
router = APIRouter(prefix="/astro")

DEFAULT_PLACE = {"name": "New Delhi", "lat": 28.6139, "lon": 77.209, "tz": "Asia/Kolkata"}
PER_MINUTE = 40
REPORTS_PER_DAY = 12


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
class PlaceIn(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    city: Optional[str] = None
    state: Optional[str] = None
    countryIso2: Optional[str] = None
    tz: Optional[str] = None


class PersonIn(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = Field(default=None, max_length=60)
    relation: Optional[str] = Field(default=None, max_length=30)
    gender: Optional[str] = None
    birthDate: Optional[str] = None
    birthTime: Optional[str] = None
    birthCity: Optional[str] = Field(default=None, max_length=80)
    birthState: Optional[str] = Field(default=None, max_length=80)
    birthCountryIso2: Optional[str] = Field(default=None, max_length=2)
    me: bool = False


class TodayReq(BaseModel):
    place: Optional[PlaceIn] = None
    date: Optional[str] = None
    members: List[PersonIn] = Field(default_factory=list, max_length=8)


class ChartReq(BaseModel):
    person: Optional[PersonIn] = None


class MatchReq(BaseModel):
    a: Optional[PersonIn] = None
    b: PersonIn


class GoodDaysReq(BaseModel):
    event: str = "griha_pravesh"
    place: Optional[PlaceIn] = None
    days: int = 60
    people: List[PersonIn] = Field(default_factory=list, max_length=6)


class QuestionReq(BaseModel):
    vote: Optional[int] = None
    place: Optional[PlaceIn] = None


class PlacesReq(BaseModel):
    q: str = Field(default="", max_length=60)
    countryIso2: Optional[str] = "IN"


# ------------------------------------------------------------------ places
def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in str(name or "").split())


def _nearest_city(lat: float, lon: float) -> Optional[Dict[str, Any]]:
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
            return {"name": _title(best[0]), "lat": float(lat), "lon": float(lon), "tz": best[3] or "Asia/Kolkata"}
    return None


def _place(p: Optional[PlaceIn], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if p and p.lat is not None and p.lon is not None and -90 <= p.lat <= 90 and -180 <= p.lon <= 180:
            near = _nearest_city(round(p.lat, 3), round(p.lon, 3))
            if near:
                if p.city:
                    near["name"] = p.city.strip()[:60]
                return near
            return {"name": (p.city or "Your city").strip()[:60], "lat": p.lat, "lon": p.lon, "tz": p.tz or "Asia/Kolkata"}
        if p and p.city:
            found = _d().resolve_place({"birthCityName": p.city, "birthStateName": p.state or "",
                                        "birthCountryIso2": (p.countryIso2 or "IN")})
            if found:
                return {"name": p.city.strip()[:60], "lat": found["lat"], "lon": found["lon"], "tz": found["tz"] or "Asia/Kolkata"}
        if user:
            lat, lon = user.get("lat"), user.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                near = _nearest_city(round(float(lat), 2), round(float(lon), 2))
                if near:
                    return near
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


# ------------------------------------------------------------------ people
def _person_profile(p: PersonIn) -> Dict[str, Any]:
    return {"birthDate": p.birthDate or "", "birthTime": p.birthTime or "", "birthCityName": p.birthCity or "",
            "birthStateName": p.birthState or "", "birthCountryIso2": (p.birthCountryIso2 or "").upper(),
            "birthCountryName": "India" if not p.birthCountryIso2 else ""}


def _gender(v: Any) -> str:
    s = str(v or "").strip().lower()
    return "male" if s in ("male", "man", "m", "boy") else ("female" if s in ("female", "woman", "f", "girl") else "")


def _own_profile(uid: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    profile, public, user = _d().load_user_docs(uid)
    merged = {**public, **user, **profile}
    return merged, profile, user


def _name_of(doc: Dict[str, Any]) -> str:
    return str(doc.get("firstName") or doc.get("name") or doc.get("displayName") or "").strip()[:40]


def _resolve_person(uid: str, p: Optional[PersonIn]) -> Tuple[Dict[str, Any], str, str]:
    """(profile-like dict, name, gender) for a person sent by the app, or the caller when p is empty or 'me'."""
    if p is None or p.me or not p.birthDate:
        merged, _, _ = _own_profile(uid)
        if not fc._parse_date(merged.get("birthDate")):
            raise HTTPException(status_code=422, detail="missing_birth_data")
        return merged, (p.name if p and p.name else _name_of(merged)) or "You", _gender(merged.get("gender"))
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
    user = None
    if not body.place or (body.place.lat is None and not body.place.city):
        _, _, user = _d().load_user_docs(uid)
    place = _place(body.place, user)
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
    profile, name, _ = _resolve_person(uid, body.person)
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


def _match(uid: str, body: MatchReq) -> Dict[str, Any]:
    pa, a_name, a_gender = _resolve_person(uid, body.a)
    pb, b_name, b_gender = _resolve_person(uid, body.b)
    ca, _ = _chart_for(pa)
    cb, _ = _chart_for(pb)
    res = fates_astro.match_charts(ca, cb, a_gender, b_gender)
    viewer_is_a = body.a is None or body.a.me or not body.a.birthDate
    if not viewer_is_a:
        for d in res["doshas"]:
            d["why"] = _named(d["why"], a_name, b_name)
    res["verdict"] = _verdict(res["total"])
    res["a"] = {"name": a_name, "gender": a_gender, "manglik": fates_astro.manglik_status(ca), "timeKnown": ca.get("timeKnown")}
    res["b"] = {"name": b_name, "gender": b_gender, "manglik": fates_astro.manglik_status(cb), "timeKnown": cb.get("timeKnown")}
    return res


@router.post("/match")
def match(body: MatchReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    try:
        return {"ok": True, "match": _match(uid, body)}
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


@router.post("/match/report")
def match_report(body: MatchReq, authorization: Optional[str] = Header(default=None)):
    uid = _auth(authorization)
    profile, _, user = _own_profile(uid)
    if not _d().is_premium(user, profile, datetime.now(timezone.utc)):
        raise HTTPException(status_code=402, detail="premium_required")
    res = _match(uid, body)
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
    if not _d().take_rate(uid, "astroReport", REPORTS_PER_DAY, "day"):
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
    user = None
    if not body.place or (body.place.lat is None and not body.place.city):
        _, _, user = _d().load_user_docs(uid)
    place = _place(body.place, user)
    start = _local_day(place["tz"])
    people = []
    for m in body.people[:6]:
        try:
            prof, name, _ = _resolve_person(uid, m)
            c, _ = _chart_for(prof)
            people.append({"name": name, "rashi": c["rashi"], "nak": c["nak"]})
        except HTTPException:
            continue
    try:
        out = ac.good_days(body.event, start, max(14, min(int(body.days or 60), 120)), place["lat"], place["lon"], place["tz"], people)
    except Exception:
        logger.exception("good days failed")
        raise HTTPException(status_code=503, detail="temporarily_unavailable")
    out["place"] = {"name": place["name"], "tz": place["tz"]}
    return {"ok": True, **out}


# ------------------------------------------------------------------ /question
def _city_key(name: str) -> str:
    k = "".join(ch for ch in str(name or "").lower() if ch.isalnum())[:40]
    return k or "india"


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
    place = _place(body.place, None) if body.place else dict(DEFAULT_PLACE)
    day = datetime.now(ac.zone(place["tz"])).date()
    q = astro_content.question_for(day)
    out = {"ok": True, "question": {"id": q["id"], "day": day.isoformat(), "text": q["text"], "options": q["options"]},
           "voted": None, "results": None}
    db = _d().db()
    if db is None:
        return out
    from google.cloud import firestore as fs
    qref = db.collection("dailyQuestions").document(day.isoformat())
    vref = db.collection("dailyQuestionVotes").document(f"{day.isoformat()}__{uid}")
    ck = _city_key(place["name"]) if body.place else "india"
    city_name = place["name"] if body.place else "India"
    try:
        vote_snap = vref.get()
        voted = (vote_snap.to_dict() or {}).get("option") if vote_snap.exists else None
        if voted is None and body.vote is not None and 0 <= body.vote < len(q["options"]):
            try:
                vref.create({"uid": uid, "option": body.vote, "city": ck, "qid": q["id"],
                             "createdAtIso": datetime.now(timezone.utc).isoformat()})
                inc = {"qid": q["id"], "counts": {str(body.vote): fs.Increment(1)}}
                if ck != "india":
                    inc["cities"] = {ck: {str(body.vote): fs.Increment(1)}}
                qref.set(inc, merge=True)
                voted = body.vote
            except Exception:   # already voted from another device
                voted = (vref.get().to_dict() or {}).get("option")
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
    sql = ("SELECT c.city, s.state, c.cc, c.lat, c.lon, c.tz, c.pop FROM cities c LEFT JOIN states s ON s.id = c.state_id "
           "WHERE {where} AND c.city LIKE ? ORDER BY (c.city = ?) DESC, COALESCE(c.pop, 0) DESC LIMIT 12")
    try:
        rows = _d().places_query(sql.format(where="c.cc = ?"), (cc, q + "%", q))
        if len(rows) < 5:
            more = _d().places_query(sql.format(where="c.cc <> ?"), (cc, q + "%", q))
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
                    "lat": lat, "lon": lon, "tz": tz or "Asia/Kolkata"})
    return {"ok": True, "places": out}
