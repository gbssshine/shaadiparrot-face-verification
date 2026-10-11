"""Mithu's week ahead (engagement N4): the week's facts, the AI prompt, the validator and a cached writer.

Everything the reader sees as a fact (the day tiles, best days, events) comes from astro_core and is exact. The AI
writes only the short text on top, from those facts, and validate() throws away any text that names a planet or
sign the facts don't have, a date outside the week, a frightening word or markdown. After one stricter retry the
text is None and the app shows the day strip alone.

astro_api.py serves POST /astro/forecast with this module: build_week() for the payload, week_facts() for the AI
facts, cached_week_text() for the text, cached in forecasts/{uid}__self__week__{periodKey} under the lease
forecast_{uid}_{periodKey}. No I/O here except through the callables and the db handed in.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date as Date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import astro_core as ac
from fates_astro import NAKSHATRAS, RASHIS

logger = logging.getLogger("shaadiparrot-cloudrun.forecast")

FORECAST_COLLECTION = "forecasts"
LEASE_SEC = 60.0
REGEN_KEY = "horoscope_regen"   # a birth-detail change regenerates the week: it shares the horoscope's daily cap
REGEN_PER_DAY = 2
WORDS_MIN, WORDS_MAX = 100, 210   # the prompt asks for 120-180 words
MAX_TOKENS = 450
TEMPERATURE = 0.6
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# ------------------------------------------------------------------ the week
def week_period(today: Date) -> Tuple[Date, bool]:
    """(Monday, next_week): Monday to Sunday holding today; on Saturday and Sunday the week after."""
    monday = today - timedelta(days=today.weekday())
    if today.weekday() >= 5:
        return monday + timedelta(days=7), True
    return monday, False


def _span_words(monday: Date) -> str:
    """'12–18 Oct', '28 Sep – 4 Oct', '28 Dec – 3 Jan'."""
    sunday = monday + timedelta(days=6)
    if monday.month == sunday.month:
        return f"{monday.day}–{sunday.day} {sunday:%b}"
    return f"{monday.day} {monday:%b} – {sunday.day} {sunday:%b}"


def week_title(monday: Date, next_week: bool, name: Optional[str] = None) -> str:
    """'Your week · 12–18 Oct', 'Next week · 19–25 Oct'; for a profile made for someone else
    "Ananya's week · …" / "Ananya's next week · …"."""
    who = (name or "").strip()
    if who:
        head = f"{who}'s next week" if next_week else f"{who}'s week"
    else:
        head = "Next week" if next_week else "Your week"
    return f"{head} · {_span_words(monday)}"


def week_events(monday: Date, natal_rashi: Optional[int], festivals: Optional[List[Dict[str, Any]]] = None,
                name: Optional[str] = None) -> List[Dict[str, str]]:
    """Own-faith festivals (filtered by the caller, from astro_festivals) and big sky events inside the week.
    name: the person a managed profile is for, so a sky line reads "Ananya's 5th house"."""
    sunday = monday + timedelta(days=6)
    a, b = monday.isoformat(), sunday.isoformat()
    out: List[Dict[str, str]] = []
    seen = set()
    for f in festivals or []:
        # fday/fname, never day/name: name is the managed person's first name, used by the sky lines below.
        fday, fname = str(f.get("date") or "")[:10], str(f.get("name") or "").strip()
        if not fname or not (a <= fday <= b) or (fday, fname) in seen:
            continue
        seen.add((fday, fname))
        out.append({"date": fday, "kind": "festival", "id": str(f.get("id") or ""), "text": fname})
    for e in ac.sky_events_between(monday, sunday):
        out.append({"date": e["date"], "kind": "sky", "id": e["planet"].lower(),
                    "text": ac.sky_line(e, natal_rashi, name)})
    out.sort(key=lambda e: (e["date"], e["kind"] != "festival"))
    return out


def build_week(natal_rashi: int, natal_nak: int, today: Date, place: Dict[str, Any], *, name: Optional[str] = None,
               relation: str = "", precision: str = "exact",
               festivals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The /astro/forecast payload without ok and text: {span, periodKey, title, precision, days, best, events}.
    name: the first name of the profile when it was made for someone else (managed), else None.
    relation: profileFor, accepted from the caller; a managed rishta profile is an adult, so the day lines are the
    adult ones whatever it says (see astro_core.week_days).
    precision: 'exact' only for a chart with birth time and city; anything else is 'approximate'."""
    monday, nxt = week_period(today)
    days = ac.week_days(natal_rashi, natal_nak, monday, place["lat"], place["lon"], place["tz"], name=name,
                        relation=relation)
    return {"span": "week", "periodKey": monday.isoformat(), "title": week_title(monday, nxt, name),
            "precision": "exact" if precision == "exact" else "approximate",
            "days": days, "best": ac.best_days(days), "events": week_events(monday, natal_rashi, festivals, name)}


# ------------------------------------------------------------------ facts for the AI
def _long_day(iso: str) -> str:
    d = Date.fromisoformat(iso)
    return f"{ac.WEEKDAYS[d.weekday()]} {d.day} {MONTHS[d.month - 1]}"


def week_facts(week: Dict[str, Any], *, natal_rashi: int, natal_nak: int, name: Optional[str] = None,
               astro_mode: str = "full") -> Dict[str, Any]:
    """Everything the AI may use, as plain data (also the base of factsHash)."""
    monday = Date.fromisoformat(week["periodKey"])
    sunday = monday + timedelta(days=6)
    who = (name or "").strip()[:30]
    return {
        "reader": f"family of {who}" if who else "self",
        "name": who,
        "astroMode": "fun" if astro_mode == "fun" else "full",
        "monday": monday.isoformat(),
        "sunday": sunday.isoformat(),
        "natal": {"moonSign": f"{RASHIS[natal_rashi][0]} ({RASHIS[natal_rashi][1]})",
                  "nakshatra": NAKSHATRAS[natal_nak][0]},
        "precision": week.get("precision") or "approximate",
        "days": [{"day": _long_day(d["date"]), "moonSign": d["moonSign"], "house": d["house"],
                  "theme": ac.HOUSE_THEMES[d["house"]], "tara": d["tara"], "tone": d["tone"], "tithi": d["tithi"]}
                 for d in week["days"]],
        "best": {k: [_long_day(x) for x in v] for k, v in (week.get("best") or {}).items()},
        "events": [{"day": _long_day(e["date"]), "text": e["text"]} for e in week.get("events") or []],
    }


def facts_text(facts: Dict[str, Any]) -> str:
    """The FACTS block of the prompt (and the list of words the text may use)."""
    reader = ("READER: self" if facts["reader"] == "self"
              else f"READER: {facts['reader']} (the profile was made by a parent or relative)")
    mode = "ASTRO_MODE: fun (keep it light, entertainment framing)" if facts["astroMode"] == "fun" else "ASTRO_MODE: full"
    lines = [
        reader, mode,
        f"WEEK: {_long_day(facts['monday'])} to {_long_day(facts['sunday'])} {Date.fromisoformat(facts['sunday']).year}",
        f"NATAL: Moon sign {facts['natal']['moonSign']}, {facts['natal']['nakshatra']} nakshatra",
        "PRECISION: exact" if facts["precision"] == "exact"
        else "PRECISION: approximate (no exact birth time: read from the Moon only)",
        "DAYS:",
    ]
    for d in facts["days"]:
        lines.append(f"- {d['day']}: Moon in {d['moonSign']}, house {d['house']} from the natal Moon ({d['theme']}); "
                     f"tara {d['tara']}; tone {d['tone']}; tithi {d['tithi']}")
    best = facts.get("best") or {}
    labels = (("love", "love & family"), ("work", "work & money"), ("quiet", "quiet days"))
    lines.append("BEST: " + "; ".join(f"{label}: {', '.join(best.get(k) or []) or 'none'}" for k, label in labels))
    events = facts.get("events") or []
    lines.append("EVENTS: " + ("; ".join(f"{e['day']}: {e['text']}" for e in events) if events else "none"))
    return "\n".join(lines)


WEEK_SYSTEM = (
    "You are Mithu, the warm, witty parrot astrologer of the Shaadi Parrot app, writing a Vedic (Jyotish) week "
    "ahead for an Indian family in simple English.\n"
    "Use ONLY the FACTS. Never invent planets, signs, houses, festivals or dates; name only planets and signs that "
    "appear in the FACTS, and only dates inside WEEK.\n"
    "Format: plain text, no markdown, no asterisks, no hashes, no emoji. 120-180 words in total. Four sections, "
    "each a heading line followed by 1-2 short sentences:\n"
    "Love & family\nWork & money\nHealth\nBest day for…\n"
    "In 'Best day for…' name the days from BEST by weekday.\n"
    "Rules:\n"
    "- Warm, practical and calm. A day with tone gentle is a quiet day for rest and patience; never call a day "
    "bad.\n"
    "- Never mention death, dying, accidents, illness, divorce, curses, danger or doshas; no medical, legal or "
    "money advice beyond common sense; no remedies, pujas or gemstones to buy.\n"
    "- If READER is family, write about that person by first name in the third person and give tips the family "
    "can act on.\n"
    "- If ASTRO_MODE is fun, keep it light: entertainment, not prediction.\n"
    "- If PRECISION is approximate, say once, gently, that the reading is from the Moon only.\n"
    "- Never mention being an AI, prompts or data fields."
)


def week_prompt(facts: Dict[str, Any], stricter: Optional[str] = None) -> List[Dict[str, str]]:
    """Chat messages for the weekly text. stricter: why the last answer was refused (the one retry)."""
    user = "FACTS:\n" + facts_text(facts)
    if stricter:
        user += (f"\n\nYour last answer was refused ({stricter}). Write it again and follow every rule exactly: "
                 "only the facts above, plain text, 120-180 words.")
    return [{"role": "system", "content": WEEK_SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ the validator
PLANET_WORDS = ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu", "Uranus", "Neptune",
                "Pluto", "Surya", "Chandra", "Mangal", "Budh", "Budha", "Guru", "Brihaspati", "Shukra", "Shani")
SIGN_WORDS = tuple(r[0] for r in RASHIS) + tuple(r[1] for r in RASHIS) + (
    "Vrishchika", "Vrischika", "Vrishabh", "Mithun", "Kark", "Karkata", "Makar", "Kumbh", "Meen")
_SKY_RE = re.compile(r"\b(" + "|".join(sorted(set(PLANET_WORDS + SIGN_WORDS), key=len, reverse=True)) + r")\b")
_BANNED_RE = re.compile(
    r"\b(deaths?|die|dies|died|dying|accidents?|accidental|divorc\w*|curse\w*|danger\w*|dosha\w*|illness\w*|"
    r"bad\s+days?)\b", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*\*|__|`|\]\(|^\s{0,3}#|^\s*[-*+]\s", re.MULTILINE)
_MONTH = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH + r"\b")
_MONTH_DAY_RE = re.compile(r"\b" + _MONTH + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’-][A-Za-z0-9À-ɏ]+)*")


def _month_no(word: str) -> int:
    return next(i for i, m in enumerate(MONTHS, 1) if m.lower().startswith(word.lower()[:3]))


def _dates_in(text: str, years: Tuple[int, ...]) -> List[Optional[Date]]:
    """Every calendar date the text names (None for one that can't exist)."""
    found: List[Tuple[int, int, Optional[int]]] = []
    for m in _DAY_MONTH_RE.finditer(text):
        found.append((int(m.group(1)), _month_no(m.group(2)), None))
    for m in _MONTH_DAY_RE.finditer(text):
        found.append((int(m.group(2)), _month_no(m.group(1)), None))
    for m in _ISO_RE.finditer(text):
        found.append((int(m.group(3)), int(m.group(2)), int(m.group(1))))
    out: List[Optional[Date]] = []
    for day, month, year in found:
        best: Optional[Date] = None
        for y in ([year] if year else list(years)):
            try:
                d = Date(y, month, day)
            except ValueError:
                continue
            best = d if best is None else best
            if d.year in years:
                best = d
                break
        out.append(best)
    return out


def problems(text: Optional[str], facts: Dict[str, Any]) -> Optional[str]:
    """Why the weekly text can't be shown, or None when it can."""
    t = (text or "").strip()
    if not t:
        return "empty"
    words = len(_WORD_RE.findall(t))
    if words < WORDS_MIN or words > WORDS_MAX:
        return f"{words} words"
    if _MARKDOWN_RE.search(t):
        return "markdown"
    bad = _BANNED_RE.search(t)
    if bad:
        return f"banned word '{bad.group(0)}'"
    allowed = set(_SKY_RE.findall(facts_text(facts)))
    for name in _SKY_RE.findall(t):
        if name not in allowed:
            return f"'{name}' is not in the facts"
    monday, sunday = Date.fromisoformat(facts["monday"]), Date.fromisoformat(facts["sunday"])
    years = tuple(sorted({monday.year, sunday.year}))
    for d in _dates_in(t, years):
        if d is None or not (monday <= d <= sunday):
            return "a date outside the week"
    for y in _YEAR_RE.findall(t):
        if int(y) not in years:
            return f"the year {y}"
    return None


def validate(text: Optional[str], facts: Dict[str, Any]) -> bool:
    return problems(text, facts) is None


def _clean(text: Optional[str]) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", t)


def write_week(complete: Callable[..., Tuple[str, str]], facts: Dict[str, Any]) -> Optional[str]:
    """The weekly text from the AI, checked; one stricter retry; None when both answers are refused.
    complete(messages, max_tokens, temperature, kind) -> (text, finish_reason), as astro_api's Deps.complete."""
    why: Optional[str] = None
    for _ in range(2):
        try:
            raw, _finish = complete(week_prompt(facts, stricter=why), MAX_TOKENS, TEMPERATURE, "astro_week")
        except Exception:
            logger.warning("week text call failed", exc_info=True)
            return None
        text = _clean(raw)
        why = problems(text, facts)
        if why is None:
            return text
        logger.info("week text refused: %s", why)
    return None


# ------------------------------------------------------------------ cache
def birth_sig(birth_date: Any, birth_time: Any, birth_city: Any, approx: Any) -> str:
    """The N2 birth signature: sha1('{birthDate}|{birthTime}|{birthCityName}|{birthTimeApprox}')[:12]."""
    raw = f"{birth_date or ''}|{birth_time or ''}|{birth_city or ''}|{approx or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def facts_hash(facts: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def forecast_doc_id(uid: str, period_key: str) -> str:
    return f"{uid}__self__week__{period_key}"


def _read(ref: Any) -> Dict[str, Any]:
    try:
        snap = ref.get()
        return (snap.to_dict() or {}) if snap.exists else {}
    except Exception:
        logger.warning("forecast cache read failed", exc_info=True)
        return {}


def cached_week_text(db: Any, uid: str, period_key: str, facts: Dict[str, Any], sig: str,
                     complete: Callable[..., Tuple[str, str]], *,
                     take_lease: Callable[[str, float], bool], release_lease: Callable[[str], None],
                     take_rate: Callable[[str, str, int, str], bool],
                     wait_sec: float = 12.0, poll_sec: float = 0.5) -> Tuple[Optional[str], bool]:
    """(text, cached). One AI call per uid and week: parallel requests wait for the one holding the lease. A changed
    birth signature regenerates (counted in horoscope_regen, 2 a day); otherwise the stored text is served."""
    if db is None:
        return write_week(complete, facts), False
    ref = db.collection(FORECAST_COLLECTION).document(forecast_doc_id(uid, period_key))
    old = _read(ref)
    old_text = old.get("text") or None
    if old_text and (not old.get("birthSig") or old.get("birthSig") == sig):
        return old_text, True
    lease = f"forecast_{uid}_{period_key}" if not old_text else f"forecast_{uid}_{period_key}_{sig}"
    if not take_lease(lease, LEASE_SEC):
        # Someone else is writing this week right now: wait a little for their text.
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            time.sleep(poll_sec)
            cur = _read(ref)
            if cur.get("text") and (not old_text or cur.get("birthSig") == sig):
                return cur["text"], True
        return old_text, bool(old_text)
    if old_text and not take_rate(uid, REGEN_KEY, REGEN_PER_DAY, "day"):
        release_lease(lease)
        return old_text, True
    text = write_week(complete, facts)
    if not text:
        release_lease(lease)
        return old_text, bool(old_text)
    try:
        ref.set({"uid": uid, "span": "week", "periodKey": period_key, "text": text, "factsHash": facts_hash(facts),
                 "birthSig": sig, "createdAtIso": datetime.now(timezone.utc).isoformat()})
    except Exception:
        logger.warning("forecast cache write failed", exc_info=True)
    return text, False
