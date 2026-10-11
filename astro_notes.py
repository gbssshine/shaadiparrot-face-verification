"""Mithu writes first (engagement N5): one short note a day inside the app, from templates (no AI, no push).

The highest priority that qualifies wins:
  P1  an own-faith festival today or tomorrow (the caller passes the festivals, already filtered for the
      reader's faith and region by astro_festivals), with the city time when the festival has one;
  P2  a birthday today or tomorrow: a family member, the person a managed profile is for, or the reader;
  P3  a big sky event (a Jupiter, Saturn or Rahu sign change) within 2 days, astroMode 'full' with a birth date;
  P5  the wedding season opens (Dev Uthani Ekadashi), for single and managed accounts, 'full' or 'fun'.
astroMode 'off' gets only festival greetings and birthdays.

Caps: one note per uid per local day, at most 4 in any 7 days (mithuNotesState/{uid}, server-only). Storage is
one batch: create() mithuNotes/{uid}__{day} plus the chat message (kind 'mithu_note') in the Parrot thread, so
parallel calls end with one note and one message, like the daily horoscope. Asks never come from here: the app's
ProfileAsks owns them.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date as Date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import astro_core as ac
import fates_astro_chart as fc

logger = logging.getLogger("shaadiparrot-cloudrun.notes")

NOTES_COLLECTION = "mithuNotes"
STATE_COLLECTION = "mithuNotesState"
CHATS_COLLECTION = "parrotChats"
WEEK_CAP = 4                 # notes in any 7 local days
SKY_AHEAD_DAYS = 2           # P3: a sky event today, tomorrow or the day after
MANAGED = frozenset({"son", "daughter", "brother", "sister", "relative", "friend"})
MARRIED_STAGES = frozenset({"newly_married", "relationship", "married", "married_kids"})
# Faiths whose people are asked first whether they want kundli at all ("Kundli, your way"); 'fun' until they answer.
ASK_FIRST = frozenset({"muslim", "christian", "sikh", "parsi", "jewish", "buddhist", "spiritual"})
# The wedding season (muhurats) is a Hindu, Jain and Sikh family custom; unknown faith counts as "All of India".
WEDDING_SEASON_FAITHS = frozenset({"hindu", "jain", "sikh"})
DEV_UTHANI_DAYS = frozenset(f["date"] for f in ac.FESTIVALS if f["id"] == "dev_uthani")
# Festival ids whose name is the start of a period ("Sharad Navratri begins", "Ramzan begins"): the title reads
# "<name> tomorrow", not "<name> is tomorrow". Keyed by id, never by the displayed name (CR5).
STARTS_A_PERIOD = frozenset({"navratri", "ramzan_start"})


# ------------------------------------------------------------------ small readers
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """A field of a dict or of a request model (PersonIn)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def faith_key(religion: Any) -> str:
    """'Hindu' -> 'hindu', 'Parsi / Zoroastrian' -> 'parsi', 'No religion / Spiritual' -> 'spiritual'."""
    s = str(religion or "").strip().lower()
    if not s or s.startswith("prefer"):
        return ""
    if s.startswith("no religion") or "spiritual" in s:
        return "spiritual"
    return s.split("/")[0].split()[0]


def _celebrates(prefs: Dict[str, Any], me: Dict[str, Any]) -> List[str]:
    raw = prefs.get("celebrates")
    if raw is None:
        raw = me.get("celebrates")
    keys = [faith_key(x) for x in (raw if isinstance(raw, list) else [])]
    keys = [k for k in keys if k]
    if not keys:
        k = faith_key(me.get("religion"))
        keys = [k] if k else []
    return keys


def astro_mode_of(prefs: Dict[str, Any], me: Dict[str, Any]) -> str:
    """'full' | 'fun' | 'off': the user's answer, else 'fun' for ask-first faiths, else 'full'."""
    mode = str(prefs.get("astroMode") or me.get("astroMode") or "").strip().lower()
    if mode in ("full", "fun", "off"):
        return mode
    return "fun" if any(k in ASK_FIRST for k in _celebrates(prefs, me)) else "full"


def _first_name(v: Any, cap: int = 30) -> str:
    return str(v or "").strip().split(" ")[0][:cap] if str(v or "").strip() else ""


def _is_managed(me: Dict[str, Any]) -> bool:
    return str(me.get("profileFor") or "").strip().lower() in MANAGED


def _single_or_managed(me: Dict[str, Any]) -> bool:
    if _is_managed(me):
        return True
    stage = str(me.get("lifeStage") or "").strip().lower()
    if stage in MARRIED_STAGES:
        return False
    if stage == "single":
        return True
    # Accounts from before the life-stage question count as single when they have a profile (MyBasics rule).
    return not stage and str(me.get("profileFor") or "").strip().lower() != "none"


def _birthday_in(birth_date: Any, day: Date) -> Optional[int]:
    """0 when day is the birthday, 1 when tomorrow is, else None. 29 Feb counts on 28 Feb in other years."""
    parsed = fc._parse_date(birth_date)
    if not parsed:
        return None
    _, m, d = parsed
    for offset in (0, 1):
        x = day + timedelta(days=offset)
        if (x.month, x.day) == (m, d) or ((m, d) == (2, 29) and (x.month, x.day) == (2, 28)
                                          and not _leap(x.year)):
            return offset
    return None


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _action(type_: str, label: str, arg: Optional[str] = None) -> Dict[str, str]:
    a = {"type": type_, "label": label}
    if arg:
        a["arg"] = arg
    return a


# ------------------------------------------------------------------ P1: festivals
def _city_time(fest: Dict[str, Any], day: Date, place: Dict[str, Any]) -> Optional[str]:
    """'Moonrise in Jaipur: 20:31.' for the festival's own date, when it has a time kind."""
    kind = str(fest.get("timeKind") or "").strip().lower()
    if kind not in ("moonrise", "sunset", "pradosh") or not place:
        return None
    try:
        tz = ac.zone(place.get("tz"))
        f = ac._day_frame(day.year, day.month, day.day, round(float(place["lat"]), 3), round(float(place["lon"]), 3),
                          tz.key)
    except Exception:
        logger.warning("festival time failed", exc_info=True)
        return None
    city = str(place.get("name") or "").strip() or "your city"
    if kind == "moonrise":
        if not f["moonrise"]:
            return None
        return f"Moonrise in {city}: {ac.moment_of(f['moonrise'], tz):%H:%M}."
    sunset = ac.moment_of(f["sunset"], tz)
    if kind == "sunset":
        return f"Sunset in {city}: {sunset:%H:%M}."
    end = ac.moment_of(f["sunset"] + ac.PRADOSH_KAAL, tz)
    return f"Pradosh time in {city}: {sunset:%H:%M}–{end:%H:%M}."


def _festival_note(fest: Dict[str, Any], offset: int, day: Date, place: Dict[str, Any],
                   mode: str) -> Optional[Dict[str, Any]]:
    name = str(fest.get("name") or "").strip()
    if not name:
        return None
    # An empty greeting is on purpose (astro_festivals): days of mourning or remembrance get no greeting and no
    # share action, never a made-up "Happy …".
    greeting = str(fest.get("greeting") or "").strip().rstrip("!")
    if str(fest.get("id") or "").strip() in STARTS_A_PERIOD:
        head = f"{name} today" if offset == 0 else f"{name} tomorrow"
    else:
        head = f"{name} is today" if offset == 0 else f"{name} is tomorrow"
    # The share line never repeats the display name ("a happy Sharad Navratri begins").
    share = f"{greeting}! Wishing you and your family joy and blessings." if greeting else ""
    # Moon-sighting days (Eid, Muharram, …) always carry the honesty label. Rows from astro_festivals.upcoming()
    # (and the app's UpcomingDto) say it as `tentative`; the raw table entries as `moonSighting`.
    tentative = bool(fest.get("moonSighting") or fest.get("tentative"))
    if mode == "off":
        if not greeting:
            return None
        text = (f"{head}. Expected, depends on the moon. {greeting} to you and your family!" if tentative
                else f"{head}. {greeting} to you and your family!")
        return {"topic": "festival", "title": head, "text": text,
                "actions": [_action("share_greeting", "Send a greeting", share)]}
    parts = [f"{greeting}! {head}." if greeting and offset == 0 else f"{head}."]
    if tentative:
        parts.append("Expected, depends on the moon.")
    when = _city_time(fest, day + timedelta(days=offset), place)
    if when:
        parts.append(when)
        parts.append("I'll keep the time on Today for you.")
    actions = [_action("open_panchang", "See the times")] if when else []
    if share:
        actions.append(_action("share_greeting", "Send a greeting", share))
    return {"topic": "festival", "title": head, "text": " ".join(parts), "actions": actions}


def _p1(festivals: Optional[List[Dict[str, Any]]], day: Date, place: Dict[str, Any], mode: str,
        skip_ids_today: frozenset) -> Optional[Dict[str, Any]]:
    ranked: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, f in enumerate(festivals or []):
        if not isinstance(f, dict) or str(f.get("type") or "festival") == "vrat":
            continue
        try:
            offset = (Date.fromisoformat(str(f.get("date") or "")[:10]) - day).days
        except ValueError:
            continue
        if offset not in (0, 1) or (offset == 0 and str(f.get("id") or "") in skip_ids_today):
            continue
        ranked.append((offset, i, f))
    for offset, _, f in sorted(ranked, key=lambda x: (x[0], x[1])):
        note = _festival_note(f, offset, day, place, mode)
        if note:
            return note
    return None


# ------------------------------------------------------------------ P2: birthdays
def _blessing(name: str, relation: str) -> str:
    group = ac.RELATION_GROUP.get((relation or "").strip().lower(), "any")
    if group == "child":
        return f"Happy birthday, {name}! May you grow happy, healthy and wise. Lots of love and blessings."
    if group == "parent":
        return f"Happy birthday, {name}! Thank you for everything. Wishing you good health and a peaceful year."
    if group == "spouse":
        return f"Happy birthday, {name}! Here's to another year of love and laughter together."
    return f"Happy birthday, {name}! Wishing you good health, happiness and many blessings this year."


def _person_birthday(name: str, relation: str, offset: int, member_id: str = "") -> Dict[str, Any]:
    title = f"{name}'s birthday is tomorrow" if offset else f"It's {name}'s birthday today"
    text = (f"{name}'s birthday is tomorrow. Want a blessing line to send in the morning?" if offset
            else f"It's {name}'s birthday today. Want a blessing line to send?")
    actions = [_action("share_blessing", "Get a blessing line", _blessing(name, relation))]
    if member_id:
        actions.append(_action("open_family_member", f"See {name}'s day", member_id))
    return {"topic": "birthday", "title": title, "text": text, "actions": actions}


def _p2(me: Dict[str, Any], day: Date, members: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
    found: List[Tuple[int, int, Dict[str, Any]]] = []   # (offset, rank, note)
    if _is_managed(me):
        name = _first_name(me.get("firstName") or me.get("name"))
        off = _birthday_in(me.get("birthDate"), day)
        if name and off is not None:
            # A managed rishta profile is always an adult: the adult blessing, whatever profileFor says.
            found.append((off, 0, _person_birthday(name, "", off)))
    else:
        off = _birthday_in(me.get("birthDate"), day)
        if off is not None:
            found.append((off, 0, {
                "topic": "birthday",
                "title": "Happy birthday!" if off == 0 else "Your birthday is tomorrow",
                "text": ("Happy birthday! I wish you a year of good health, love and blessings." if off == 0
                         else "Your birthday is tomorrow! I wish you a wonderful year ahead."),
                "actions": []}))
    for i, m in enumerate(members or []):
        if _get(m, "me") or _get(m, "self") or _get(m, "isSelf"):
            continue
        name = _first_name(_get(m, "name"))
        off = _birthday_in(_get(m, "birthDate"), day)
        if not name or off is None:
            continue
        found.append((off, 1 + i, _person_birthday(name, str(_get(m, "relation") or ""), off,
                                                    str(_get(m, "id") or "")[:128])))
    if not found:
        return None
    return sorted(found, key=lambda x: (x[0], x[1]))[0][2]


# ------------------------------------------------------------------ P3: big sky events
def _natal_rashi(me: Dict[str, Any], natal: Optional[Dict[str, Any]]) -> Optional[int]:
    if natal and isinstance(natal.get("rashi"), int):
        return int(natal["rashi"]) % 12
    date = fc._parse_date(me.get("birthDate"))
    if not date:
        return None
    time_ = fc._parse_time(me.get("birthTime")) or ac.approx_time(me.get("birthTimeApprox"))
    try:
        return int(fc.chart_from_birth(date, time_, None)["rashi"])
    except Exception:
        logger.warning("natal moon failed", exc_info=True)
        return None


def _p3(me: Dict[str, Any], day: Date, natal: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rashi = _natal_rashi(me, natal)
    if rashi is None:
        return None
    events = ac.sky_events_between(day, day + timedelta(days=SKY_AHEAD_DAYS))
    if not events:
        return None
    e = events[0]
    when = Date.fromisoformat(e["date"])
    lead = {0: "Today", 1: "Tomorrow"}.get((when - day).days, f"On {when.day} {when:%B}")
    house = (e["sign"] - rashi) % 12 + 1
    # A retrograde return or a re-entry gets its own timeline-free words (ac.sky_meaning), never "a warm year".
    move, theme = ac.sky_move(e), ac.sky_meaning(e, house)
    # A profile made for someone else reads that person's chart: the parent reads "Ananya's 5th house".
    name = _first_name(me.get("firstName") or me.get("name")) if _is_managed(me) else ""
    whose = f"{name}'s" if name else "your"
    return {"topic": "sky", "title": move,
            "text": f"{lead} {move}, {whose} {ac.ordinal(house)} house from the Moon: {theme}.",
            "actions": []}


# ------------------------------------------------------------------ P5: the wedding season
def _p5_applies(me: Dict[str, Any], prefs: Dict[str, Any], day: Date, mode: str) -> bool:
    if mode not in ("full", "fun") or day.isoformat() not in DEV_UTHANI_DAYS or not _single_or_managed(me):
        return False
    faiths = _celebrates(prefs, me)
    return not faiths or any(k in WEDDING_SEASON_FAITHS for k in faiths)


def _p5() -> Dict[str, Any]:
    return {"topic": "wedding_season", "title": "The wedding season opens",
            "text": "The wedding season opens today. Want to see good wedding dates?",
            "actions": [_action("open_good_dates", "See wedding dates", "wedding")]}


# ------------------------------------------------------------------ the note
def compose(me: Dict[str, Any], place: Dict[str, Any], day: Date, members: Optional[List[Any]],
            prefs: Optional[Dict[str, Any]], festivals: Optional[List[Dict[str, Any]]] = None,
            natal: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Today's note by priority (P1 > P2 > P3 > P5), or None. Pure: no storage, no caps.
    me: the caller's merged docs; place: the /astro/today place; prefs: {celebrates, community, stateName,
    astroMode}; festivals: own-faith festivals from astro_festivals.upcoming (today and tomorrow are used);
    natal: {'rashi': birth Moon sign} when the caller already has the chart."""
    me = me or {}
    prefs = prefs or {}
    mode = astro_mode_of(prefs, me)
    p5 = _p5_applies(me, prefs, day, mode)
    # On Dev Uthani itself the wedding-season note says more than the festival line to singles and families.
    note = _p1(festivals, day, place, mode, frozenset({"dev_uthani"}) if p5 else frozenset())
    if note is None:
        note = _p2(me, day, members)
    if note is None and mode == "full" and fc._parse_date(me.get("birthDate")):
        note = _p3(me, day, natal)
    if note is None and p5:
        note = _p5()
    return note


def chat_doc_id(uid: str, thread_id: str) -> str:
    """The parrotChats doc of a thread: the same rule as main._chat_doc_id."""
    tid = (thread_id or "default").strip() or "default"
    if tid == "default":
        return uid
    return f"{uid}__t_{hashlib.sha256(tid.encode('utf-8')).hexdigest()[:16]}"


def _public(note_id: str, day: Date, doc: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": note_id, "day": day.isoformat(), "topic": doc.get("topic") or "", "title": doc.get("title") or "",
            "text": doc.get("text") or "", "actions": list(doc.get("actions") or [])}


def _recent_days(state: Dict[str, Any], day: Date) -> List[str]:
    """Days with a note in the 7 local days ending today."""
    first = (day - timedelta(days=6)).isoformat()
    return sorted({d for d in (state.get("days") or []) if isinstance(d, str) and first <= d <= day.isoformat()})


def note_for(uid: str, me: Dict[str, Any], place: Dict[str, Any], day: Date, members: Optional[List[Any]],
             prefs: Optional[Dict[str, Any]], *, festivals: Optional[List[Dict[str, Any]]] = None, db: Any = None,
             natal: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Today's Mithu note for /astro/today ({id, day, topic, title, text, actions}) or None. With a db it is
    stored once a day (idempotent) and capped; without one (local runs) it is only composed."""
    if db is None:
        note = compose(me, place, day, members, prefs, festivals, natal)
        return _public(f"{uid}__{day.isoformat()}", day, note) if note else None
    from google.api_core import exceptions as gexc

    note_id = f"{uid}__{day.isoformat()}"
    note_ref = db.collection(NOTES_COLLECTION).document(note_id)
    state_ref = db.collection(STATE_COLLECTION).document(uid)
    try:
        got = {s.reference.path: (s.to_dict() or {}) for s in db.get_all([note_ref, state_ref]) if s.exists}
    except Exception:
        logger.warning("mithu note read failed", exc_info=True)
        return None
    existing = got.get(note_ref.path)
    if existing:
        return _public(note_id, day, existing)
    recent = _recent_days(got.get(state_ref.path) or {}, day)
    if len(recent) >= WEEK_CAP:
        return None
    note = compose(me, place, day, members, prefs, festivals, natal)
    if note is None:
        return None

    tid = (thread_id or "").strip() or f"mobile_{uid}"
    chat_ref = db.collection(CHATS_COLLECTION).document(chat_doc_id(uid, tid))
    ms = int(time.time() * 1000)
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg_id = f"{ms}_a_{hashlib.md5(note['text'].encode('utf-8')).hexdigest()[:8]}"
    batch = db.batch()
    batch.create(note_ref, {"uid": uid, "dayKey": day.isoformat(), "topic": note["topic"], "title": note["title"],
                            "text": note["text"], "actions": note["actions"], "createdAtIso": iso})
    batch.set(chat_ref.collection("messages").document(msg_id), {
        "role": "assistant", "kind": "mithu_note", "noteId": note_id, "topic": note["topic"], "text": note["text"],
        "createdAtIso": iso, "createdAtMs": ms})
    batch.set(chat_ref, {"uid": uid, "threadId": tid, "updatedAtIso": iso, "updatedAtMs": ms}, merge=True)
    batch.set(state_ref, {"uid": uid, "days": (recent + [day.isoformat()])[-WEEK_CAP:], "lastDay": day.isoformat(),
                          "updatedAtIso": iso}, merge=True)
    try:
        batch.commit()
    except gexc.AlreadyExists:   # a parallel call stored today's note first: serve that one
        try:
            snap = note_ref.get()
            return _public(note_id, day, snap.to_dict() or {}) if snap.exists else None
        except Exception:
            logger.warning("mithu note re-read failed", exc_info=True)
            return None
    except Exception:
        logger.warning("mithu note store failed", exc_info=True)
        return None
    return _public(note_id, day, note)
