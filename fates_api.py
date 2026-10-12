"""Daily Fates HTTP API (Cloud Run). main.py wires the shared helpers in with configure().

Data (all server-only, the app talks to these endpoints and never reads the collections):
  dailyFates/{uid}__{dayKey}  today's paths: teasers, and for opened paths the report and verdict
  fatesMemory/{uid}           who was shown when, streak, today's extra opens and per-day counters (aiCalls,
                              modeSwitch, verifyTeaser)
  fateExposure/{dayKey}       how many people got each person as a fate today (fairness cap)
Someone who accepted the viewer as a fate ("chose you") is brought into the viewer's own paths the
next morning, one a day: that is the free way to meet them. Parrot+ or a crown shows them at once.
A fate day starts at 07:30 IST, when the morning push goes out.
Two modes (plan §2 C1): /today without a mode (1.0.7 apps send {}) is the dating Fates and brings only people in the
quick pool; {mode: 'rishta'} (1.0.8, a profile that wants marriage) brings only rishta-pool people who fit both
families' must-haves, with both kundlis and 18+ gunas on every path. A stored day keeps its `mode`; asked in the other
mode, an untouched day is made again once a fate day, after that only its opened paths stay.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import math
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, Header, HTTPException
from pydantic import BaseModel

import face_live
import photo_fetch
import fates_ai
import fates_astro
import fates_engine as fe

logger = logging.getLogger("shaadiparrot-cloudrun.fates")
router = APIRouter(prefix="/fates")

IST = ZoneInfo("Asia/Kolkata")
DAY_START = (7, 30)
FREE_OPENS_PER_DAY = 1
EXPOSURE_CAP = int(os.getenv("FATES_EXPOSURE_CAP") or "8")
CANDIDATE_POOL_LIMIT = int(os.getenv("FATES_POOL_LIMIT") or "400")
SEEN_COOLDOWN_DAYS = 14
AI_ENABLED = (os.getenv("FATES_AI") or "1").strip() != "0"
# Cost guard: DeepSeek calls per person per fate day (1 pick + a verdict per opened path).
AI_DAILY_CAP = int(os.getenv("FATES_AI_DAILY_CAP") or "6")
# People who chose the viewer: how many are looked at per request (each costs reads and a kundli),
# how many are listed, and how many mornings one is brought before Mithu stops insisting.
CHOOSER_CHECKS = 8
CHOSEN_LIST_LIMIT = 20
CHOOSER_MAX_BRINGS = 2
# Fates modes (plan §2 C1). A request without a mode (1.0.7, or a 1.0.8 app without a marriage profile) is the dating
# Fates and brings quick-pool people only; {mode: 'rishta'} brings rishta-pool people who fit both families' wishes.
MODE_QUICK = "quick"
MODE_RISHTA = "rishta"
# An untouched day is made again for the other mode at most this often a fate day (fatesMemory.modeSwitch): a new day
# reads the pool, blurs photos, may call the AI and spends other people's exposure, so flipping modes must not remake it.
MODE_SWITCH_REMAKES = 1
POOLS_CONFIG_TTL_SEC = 600
_pools_config: Dict[str, Any] = {"at": 0.0, "quickMaleMinAge": fe.QUICK_MALE_MIN_AGE}


STAT_SHARDS = 10
EXPOSURE_SHARDS = 16


def _stat(db, key: str, **counts: int) -> None:
    """Best-effort daily counters in fatesStats/{dayKey}/shards/{0..9} (one document per day would be written by
    everyone at 07:30 IST and stall); sum the shards to read them. Never fails a request."""
    try:
        from google.cloud import firestore as fs
        db.document(f"fatesStats/{key}/shards/{random.randrange(STAT_SHARDS)}").set(
            {k: fs.Increment(v) for k, v in counts.items() if v}, merge=True)
    except Exception:
        logger.warning("fates stat write failed", exc_info=True)


def _exposure_shard(uid: str) -> int:
    return int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % EXPOSURE_SHARDS


def _exposure_counts(db, key: str) -> Dict[str, int]:
    """How many people got each person as a fate today: fateExposure/{dayKey}/shards/{n}, a person always in the
    same shard, so the morning's writes spread over 16 documents."""
    counts: Dict[str, int] = {}
    try:
        refs = [db.document(f"fateExposure/{key}/shards/{i}") for i in range(EXPOSURE_SHARDS)]
        for snap in db.get_all(refs):
            for uid, n in ((snap.to_dict() or {}).get("counts") or {}).items():
                counts[uid] = counts.get(uid, 0) + int(n or 0)
    except Exception:
        logger.warning("fates exposure read failed", exc_info=True)
    return counts


def _add_exposure(db, key: str, uids: List[str]) -> None:
    """Best effort, after the day's paths are stored: a busy counter must never fail someone's fates."""
    from google.cloud import firestore as fs
    by_shard: Dict[int, Dict[str, Any]] = {}
    for u in uids:
        by_shard.setdefault(_exposure_shard(u), {})[u] = fs.Increment(1)
    for shard, inc in by_shard.items():
        try:
            db.document(f"fateExposure/{key}/shards/{shard}").set({"counts": inc}, merge=True)
        except Exception:
            logger.warning("fates exposure write failed", exc_info=True)


def _today_n(memory: Dict[str, Any], field: str, key: str) -> int:
    """Today's value of a per-day counter in fatesMemory, {day, n}; 0 when it was counted on another day."""
    c = memory.get(field) or {}
    return int(c.get("n") or 0) if isinstance(c, dict) and c.get("day") == key else 0


def _take_today(db, uid: str, key: str, field: str, cap: int) -> bool:
    """Counts one use of a per-day allowance in fatesMemory/{uid}.{field} ({day, n}) in a transaction; False once
    today's cap is reached, or when the count can't be written (then nothing is spent)."""
    from google.cloud import firestore as fs
    ref = db.document(f"fatesMemory/{uid}")

    @fs.transactional
    def take(tx) -> bool:
        n = _today_n(ref.get(transaction=tx).to_dict() or {}, field, key)
        if n >= cap:
            return False
        tx.set(ref, {field: {"day": key, "n": n + 1}}, merge=True)
        return True

    try:
        return take(db.transaction())
    except Exception:
        logger.warning("fates %s count failed", field, exc_info=True)
        return False


def _ai_allowed(db, uid: str, key: str) -> bool:
    """Counts one AI call against today's cap; False once the cap is reached."""
    return AI_ENABLED and _take_today(db, uid, key, "aiCalls", AI_DAILY_CAP)


@dataclass
class Deps:
    db: Callable[[], Any]                                   # -> firestore.Client or None
    verify_uid: Callable[[Optional[str]], str]              # Authorization header -> uid (raises 401)
    complete: Callable[..., Tuple[str, str]]                # (messages, max_tokens, temperature, kind) -> (text, finish)
    resolve_place: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    is_premium: Callable[[Dict[str, Any], Dict[str, Any], datetime], bool]


_deps: Optional[Deps] = None


def configure(deps: Deps) -> None:
    global _deps
    _deps = deps


def _d() -> Deps:
    if _deps is None:
        raise HTTPException(status_code=503, detail="fates not configured")
    return _deps


def _db():
    client = _d().db()
    if client is None:
        raise HTTPException(status_code=503, detail="firestore unavailable")
    return client


# =========================
# DAYS
# =========================
def day_key(now: Optional[datetime] = None) -> str:
    """IST date of the fate day; the day turns over at 07:30 IST."""
    local = (now or datetime.now(timezone.utc)).astimezone(IST)
    return (local - timedelta(hours=DAY_START[0], minutes=DAY_START[1])).date().isoformat()


def day_closes_at(key: str) -> datetime:
    d = datetime.fromisoformat(key).replace(tzinfo=IST) + timedelta(days=1)
    return d.replace(hour=DAY_START[0], minute=DAY_START[1]).astimezone(timezone.utc)


def _prev_day(key: str) -> str:
    return (datetime.fromisoformat(key) - timedelta(days=1)).date().isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =========================
# LOADING PEOPLE
# =========================
def _doc(db, path: str) -> Dict[str, Any]:
    try:
        return db.document(path).get().to_dict() or {}
    except Exception:
        logger.exception("fates read failed %s", path)
        return {}


def _chart(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return fates_astro.chart_for_profile(profile, _d().resolve_place)
    except Exception:
        logger.exception("fates chart failed")
        return None


def _load(db, uid: str, public: Optional[Dict[str, Any]] = None) -> Tuple[fe.Person, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(person, publicProfiles doc, profiles doc, users doc)."""
    pub = public if public is not None else _doc(db, f"publicProfiles/{uid}")
    priv = _doc(db, f"profiles/{uid}")
    usr = _doc(db, f"users/{uid}")
    person = fe.person_from_docs(uid, pub, priv, usr)
    person.chart = _chart({**pub, **usr, **priv})
    return person, pub, priv, usr


def load_person(db, uid: str, public: Optional[Dict[str, Any]] = None) -> Tuple[fe.Person, Dict[str, Any], Dict[str, Any]]:
    person, _, priv, usr = _load(db, uid, public)
    return person, priv, usr


QUICK_MALE_MIN_AGE_BOUNDS = (18, 30)   # the same bounds as feed.ts parseQuickMaleMinAge


def parse_quick_male_min_age(v: Any) -> int:
    """serverState/pools.quickMaleMinAge as feed.ts parseQuickMaleMinAge reads it, so Python and the functions agree
    on every value: a number, or a string that is one, rounded like JS Math.round (half up); 18..30, else the
    default 21. Booleans, empty or odd values give the default."""
    n: Optional[float] = None
    if isinstance(v, bool) or v is None:
        n = None
    elif isinstance(v, (int, float)):
        n = float(v)
    elif isinstance(v, str) and v.strip() and "_" not in v:
        s = v.strip()
        try:
            n = float(int(s, 0)) if s[:2].lower() in ("0x", "0o", "0b") else float(s)
        except ValueError:
            n = None
    if n is None or not math.isfinite(n):
        return fe.QUICK_MALE_MIN_AGE
    r = math.floor(n + 0.5)
    lo, hi = QUICK_MALE_MIN_AGE_BOUNDS
    return int(r) if lo <= r <= hi else fe.QUICK_MALE_MIN_AGE


def quick_male_min_age(db) -> int:
    """serverState/pools.quickMaleMinAge: from what age men are in the quick pool (default 21; the owner may set
    18..30, the bounds feed.ts uses). Cached for 10 minutes; after a failed read the last value is kept and the read
    is tried a minute later."""
    now = time.time()
    if now - float(_pools_config["at"]) < POOLS_CONFIG_TTL_SEC:
        return int(_pools_config["quickMaleMinAge"])
    try:
        v = (db.document("serverState/pools").get().to_dict() or {}).get("quickMaleMinAge")
        _pools_config.update(at=now, quickMaleMinAge=parse_quick_male_min_age(v))
    except Exception:
        logger.warning("fates pools config read failed", exc_info=True)
        _pools_config["at"] = now - POOLS_CONFIG_TTL_SEC + 60
    return int(_pools_config["quickMaleMinAge"])


def _doc_mode(doc: Optional[Dict[str, Any]]) -> str:
    """The mode a stored day was made in (days stored before 1.0.8 are dating days)."""
    return MODE_RISHTA if (doc or {}).get("mode") == MODE_RISHTA else MODE_QUICK


def _mode_fits(viewer: fe.Person, cand: fe.Person, pub: Dict[str, Any], priv: Dict[str, Any], today: str, qmin: int) -> bool:
    """The candidate is in the viewer's pool and passes the mode's rules (fates_engine.candidate_gate)."""
    if viewer.rishta:
        fe.rishta_ready(cand)
    return fe.candidate_gate(viewer, cand, pub, priv, today, qmin) is None


def _ids(db, path: str, limit: int = 2000) -> set:
    try:
        return {d.id for d in db.collection(path).limit(limit).stream()}
    except Exception:
        logger.exception("fates list failed %s", path)
        return set()


def _candidates(db, viewer: fe.Person, now_ms: int, memory: Dict[str, Any]) -> List[fe.Person]:
    excluded = set()
    for sub in ("blocks", "matches", "outgoing", "passes"):
        excluded |= _ids(db, f"users/{viewer.uid}/{sub}")
    # Reported by several people and waiting for a moderator (the functions' report hold): no fates until then.
    excluded |= _ids(db, "reportHolds")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SEEN_COOLDOWN_DAYS)).date().isoformat()
    seen = memory.get("seen") or {}
    excluded |= {u for u, v in seen.items() if str((v or {}).get("day", "")) >= cutoff}
    today = fe.ist_day()
    qmin = quick_male_min_age(db)
    pool_key = "inRishta" if viewer.rishta else "inQuick"

    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        q = db.collection("publicProfiles").where(filter=FieldFilter("isDiscoverable", "==", True))
        if viewer.rishta:
            # Only cards the functions indexed into the rishta pool (pools_of still checks the age day below).
            q = q.where(filter=FieldFilter("feedRishta", "==", True))
        # Equality filters on two fields need no composite index (Firestore merges single-field indexes).
        if viewer.looking_for in ("male", "female"):
            q = q.where(filter=FieldFilter("gender", "==", viewer.looking_for.capitalize()))
        if fe.VERIFIED_ONLY:
            # Only verified people can be a fate: asking for them keeps the pool from filling up with the others.
            q = q.where(filter=FieldFilter("isFaceVerified", "==", True))
        snaps = list(q.limit(CANDIDATE_POOL_LIMIT).stream())
    except Exception:
        logger.exception("fates pool query failed")
        return []

    quick = []
    for s in snaps:
        if s.id == viewer.uid or s.id in excluded:
            continue
        pub = s.to_dict() or {}
        # The pool first (dating: the quick pool only; rishta mode: the rishta pool only), from the server-written index.
        if not fe.pools_of(pub, today, qmin)[pool_key]:
            continue
        p = fe.person_from_docs(s.id, pub)
        # Cheap public checks first; the private docs are only read for people who pass them.
        if not p.photos or not (fe._wants(viewer, p) and fe._wants(p, viewer)):
            continue
        quick.append((s.id, pub))

    people: List[fe.Person] = []
    if not quick:
        return people
    priv_refs = [db.document(f"profiles/{u}") for u, _ in quick]
    user_refs = [db.document(f"users/{u}") for u, _ in quick]
    try:
        privs = {d.id: (d.to_dict() or {}) for d in db.get_all(priv_refs)}
        users = {d.id: (d.to_dict() or {}) for d in db.get_all(user_refs)}
    except Exception:
        logger.exception("fates batch read failed")
        return people
    for uid, pub in quick:
        priv = privs.get(uid) or {}
        p = fe.person_from_docs(uid, pub, priv, users.get(uid))
        if not _mode_fits(viewer, p, pub, priv, today, qmin):
            continue
        if fe.hard_filter(viewer, p, now_ms) is None:
            p.chart = _chart({**pub, **users.get(uid, {}), **privs.get(uid, {})})
            people.append(p)
    return people


def _blocked_by(db, target: str, viewer: str) -> bool:
    try:
        return db.document(f"users/{target}/blocks/{viewer}").get().exists
    except Exception:
        return False


def _incoming_fate_likes(db, uid: str) -> List[Dict[str, Any]]:
    """Fate likes still waiting for the viewer's answer, oldest first (that's the order Mithu brings them)."""
    out: List[Dict[str, Any]] = []
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        q = db.collection(f"users/{uid}/incoming").where(filter=FieldFilter("viaFate", "==", True)).limit(200)
        for d in q.stream():
            x = d.to_dict() or {}
            if str(x.get("type") or x.get("action") or "").lower() not in ("like", "superlike"):
                continue
            out.append({"uid": d.id, "note": str(x.get("note") or "")[:200], "path": str(x.get("fatePath") or ""),
                        "createdAtIso": str(x.get("createdAtIso") or x.get("updatedAtIso") or "")})
    except Exception:
        logger.exception("fates incoming read failed")
    out.sort(key=lambda c: c["createdAtIso"])
    return out


def _answered(db, viewer_uid: str, target: str) -> bool:
    """The viewer already acted on this person (liked, passed, matched or blocked), or is blocked by them."""
    for sub in ("blocks", "matches", "outgoing", "passes"):
        try:
            if db.document(f"users/{viewer_uid}/{sub}/{target}").get().exists:
                return True
        except Exception:
            return True
    return _blocked_by(db, target, viewer_uid)


def _chooser_pair(db, viewer: fe.Person, uid: str, now_ms: int, memory: Dict[str, Any]) -> Optional[fe.Pair]:
    """A person who chose the viewer, as a pair Mithu may bring: not answered, not already opened or shown
    by a reveal, inside the viewer's filters, and fitting at least one path."""
    seen = ((memory.get("seen") or {}).get(uid)) or {}
    if seen.get("opened") or uid in (memory.get("revealed") or {}):
        return None
    if int((memory.get("chooserBrought") or {}).get(uid, 0)) >= CHOOSER_MAX_BRINGS:
        return None
    if _answered(db, viewer.uid, uid):
        return None
    cand, pub, priv, _ = _load(db, uid)
    if not _mode_fits(viewer, cand, pub, priv, fe.ist_day(), quick_male_min_age(db)):
        return None
    if fe.hard_filter(viewer, cand, now_ms) is not None:
        return None
    pair = fe.evaluate_pair(viewer, cand, now_ms, _match_fn)
    pair.chose_you = True
    return pair if fe.chooser_path(viewer, pair) else None


# =========================
# THUMBNAILS
# =========================
THUMB_FETCH_SEC = 8.0      # the whole photo download for one thumbnail (the paths wait for it)


def thumb_source_ok(url: str, uid: str) -> bool:
    """Only the person's own Storage photo is ever fetched for a thumbnail: face_live.own_photo_url (a Storage host,
    https except on the local emulator, the project's own bucket, users/{uid}/photos/). The photos Daily Fates shows
    are exactly those (fates_engine.own_private_photos), so this never drops a real one; it keeps the server from
    fetching any other host, bucket or member's file should a foreign link ever reach it."""
    return face_live.own_photo_url(url, uid)


def blur_thumbs(url: str, uid: str) -> Tuple[Optional[str], Optional[str]]:
    """Two tiny JPEGs (data URIs) for a closed card: a heavy blur, and a lighter "peek" that shows while
    the card is held. Both are 64 px wide, so neither can reveal who it is; the real photo only
    arrives when the path is opened. uid: the person the photo belongs to."""
    if not thumb_source_ok(url, uid):
        return None, None
    try:
        import cv2
        # Bounded (photo_fetch): no redirects, 12 MB and THUMB_FETCH_SEC in all, never decoded above 40 megapixels,
        # and decoded at a fraction of its size when that is still wide enough for a 64 px blur.
        data = photo_fetch.fetch(url, total_sec=THUMB_FETCH_SEC)
        img = photo_fetch.decode_cv2(data, min_width=256)
        if img is None:
            return None, None
        h, w = img.shape[:2]
        tw = 64
        th = max(1, int(h * tw / w))
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        def enc(sigma: float) -> Optional[str]:
            ok, buf = cv2.imencode(".jpg", cv2.GaussianBlur(small, (0, 0), sigmaX=sigma), [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

        return enc(4.5), enc(2.2)
    except photo_fetch.FetchError as e:
        logger.warning("fates blur: photo not fetched (%s)", e.reason)
        return None, None
    except Exception:
        logger.exception("fates blur failed")
        return None, None


# =========================
# GENERATION
# =========================
def _match_fn(a: fe.Person, b: fe.Person) -> Optional[Dict[str, Any]]:
    try:
        return fates_astro.match_people(a, b)
    except Exception:
        logger.exception("fates ashtakoota failed")
        return None


def _complete_short(messages, max_tokens, temperature, kind):
    return _d().complete(messages, max_tokens, temperature, kind)


def _viewer_facts(v: fe.Person) -> Dict[str, Any]:
    return {"first_name": v.first_name, "age": v.age, "city": v.city, "bio": v.bio[:200],
            "interests": v.interests[:8], "stars_matter": v.stars_pref}


def generate(db, viewer: fe.Person, memory: Dict[str, Any], key: str, now_ms: int) -> Dict[str, Any]:
    pool = _candidates(db, viewer, now_ms, memory)
    exposure = _exposure_counts(db, key)
    pool = [p for p in pool if int(exposure.get(p.uid, 0)) < fe.exposure_cap(p, EXPOSURE_CAP)]

    learned = memory.get("learned") or {}
    pairs = [fe.evaluate_pair(viewer, c, now_ms, _match_fn) for c in pool]
    short = fe.shortlist(viewer, pairs, learned=learned)

    # Mutual fates: people who already got the viewer today are brought back to them first.
    uids = {p.cand.uid for opts in short.values() for p in opts}
    for u in uids:
        other = _doc(db, f"fatesMemory/{u}").get("seen") or {}
        if str((other.get(viewer.uid) or {}).get("day", "")) == key:
            for opts in short.values():
                for p in opts:
                    if p.cand.uid == u:
                        p.mutual = True
    for path in short:
        short[path] = [p for p in short[path] if not _blocked_by(db, p.cand.uid, viewer.uid)]
        short[path].sort(key=lambda p: fe.rank_key(p, path, learned), reverse=True)

    # One person who chose the viewer is brought today, oldest first, past the exposure cap and the
    # usual bar: they already said yes, so the viewer meets them for free by waiting a day.
    chooser_uid: Optional[str] = None
    chooser_route: Optional[str] = None
    waiting = _incoming_fate_likes(db, viewer.uid)
    for c in waiting[:CHOOSER_CHECKS]:
        pair = _chooser_pair(db, viewer, c["uid"], now_ms, memory)
        if pair is None:
            continue
        chooser_route = fe.place_chooser(viewer, short, pair)
        if chooser_route:
            chooser_uid = pair.cand.uid
            break

    ai: Dict[str, Dict[str, Any]] = {}
    if any(short.values()) and _ai_allowed(db, viewer.uid, key):
        options = {path: [fe.facts_for_ai(viewer, path, p) for p in opts] for path, opts in short.items()}
        ai = fates_ai.pick_and_hooks(_complete_short, _viewer_facts(viewer), options)
        _stat(db, key, aiPickOk=1 if ai else 0, aiPickFallback=0 if ai else 1)
    choice = {p: v["pick"] for p, v in ai.items()}
    if chooser_route:
        choice[chooser_route] = 0   # the AI may prefer someone else there; the chooser stays
    chosen = fe.resolve_paths(short, choice)

    # Anyone on today's paths who chose the viewer is marked on the card, not only the one brought for it.
    choosers = {c["uid"] for c in waiting}
    for p in chosen.values():
        p.chose_you = p.cand.uid in choosers

    paths: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        thumbs = {path: ex.submit(blur_thumbs, p.cand.photos[0], p.cand.uid) for path, p in chosen.items()}
        for path, p in chosen.items():
            ai_item = ai.get(path) or {}
            # The AI's hook belongs to the person it picked; if resolve_paths had to take someone else,
            # fall back to the template for that path.
            picked_uid = short[path][ai_item["pick"]].cand.uid if "pick" in ai_item else None
            own = picked_uid == p.cand.uid
            paths[path] = {
                "targetUid": p.cand.uid,
                "teaser": fe.teaser(path, p, viewer),
                "hook": (ai_item.get("hook") if own else None) or fe.template_hook(path, p),
                "why": (ai_item.get("why") if own else None) or fe.template_why(path, p),
                "pickedBy": "ai" if own and ai_item.get("hook") else "engine",
                "blurThumb": thumbs[path].result()[0],
                "peekThumb": thumbs[path].result()[1],
                "opened": False,
                "decision": None,
            }
    order = [p for p in fe.preferred_path_order(learned) if p in paths]
    _stat(db, key, generated=1, **{f"paths{len(order)}": 1}, poolTotal=len(pool))
    return {
        "uid": viewer.uid, "dayKey": key, "createdAtIso": _now_iso(), "closesAtIso": day_closes_at(key).isoformat(),
        "mode": MODE_RISHTA if viewer.rishta else MODE_QUICK,
        "paths": paths, "order": order, "openedCount": 0, "poolSize": len(pool),
        "chooserUid": chooser_uid if chooser_route in paths else None,
    }


def _store_generation(db, doc: Dict[str, Any], key: str) -> Dict[str, Any]:
    from google.api_core import exceptions as gexc
    from google.cloud import firestore as fs

    uid = doc["uid"]
    ref = db.document(f"dailyFates/{uid}__{key}")
    batch = db.batch()
    batch.create(ref, doc)
    seen_patch: Dict[str, Any] = {f"seen.{p['targetUid']}": {"day": key, "opened": False} for p in doc["paths"].values()}
    if doc.get("chooserUid"):
        seen_patch[f"chooserBrought.{doc['chooserUid']}"] = fs.Increment(1)
    if seen_patch:
        batch.set(db.document(f"fatesMemory/{uid}"), {"uid": uid}, merge=True)
        batch.update(db.document(f"fatesMemory/{uid}"), seen_patch)
    try:
        batch.commit()
        _add_exposure(db, key, [p["targetUid"] for p in doc["paths"].values()])
        return doc
    except gexc.AlreadyExists:
        return ref.get().to_dict() or doc


# =========================
# VIEWS
# =========================
def _allowance(opened: int, extra: int, premium: bool) -> int:
    if premium:
        return len(fe.PATHS)
    return max(0, FREE_OPENS_PER_DAY + extra - opened)


def _extra_today(memory: Dict[str, Any], key: str) -> Dict[str, Any]:
    ex = memory.get("extra") or {}
    if ex.get("day") != key:
        return {"day": key, "count": 0, "ad": False}
    return {"day": key, "count": int(ex.get("count") or 0), "ad": bool(ex.get("ad"))}


def _accuracy(v: fe.Person, profile: Dict[str, Any]) -> Dict[str, Any]:
    stars = 0
    if profile.get("birthDate"):
        stars = 40
        if str(profile.get("birthTime") or "").strip():
            stars += 35
        if str(profile.get("birthCityName") or "").strip():
            stars += 25
    cats = {fe.tests_catalog()["tests"][t]["category"] for t in v.tests if t in fe.tests_catalog()["tests"]}
    heart = round(100 * len(cats) / 9)
    home_fields = [v.intent, v.want_children, v.religion, v.community, v.relocate, v.smoking, v.drinking,
                   ",".join(v.languages)]
    home = round(100 * sum(1 for f in home_fields if f and "prefer not" not in f.casefold()) / len(home_fields))
    total = round((stars + heart + home) / 3)
    return {"total": total, "stars": stars, "heart": heart, "home": home, "testCategories": sorted(cats)}


def client_view(doc: Dict[str, Any], memory: Dict[str, Any], premium: bool, accuracy: Dict[str, Any]) -> Dict[str, Any]:
    key = doc["dayKey"]
    extra = _extra_today(memory, key)
    out_paths = []
    for path in doc.get("order", []):
        p = doc["paths"][path]
        item = {"path": path, "opened": p.get("opened", False), "decision": p.get("decision"),
                "teaser": p["teaser"], "hook": p.get("hook"), "blurThumb": p.get("blurThumb"),
                "peekThumb": p.get("peekThumb")}
        if p.get("opened"):
            item["why"] = p.get("why")
            item["report"] = p.get("report")
            item["verdict"] = p.get("verdict")
        out_paths.append(item)
    opened = int(doc.get("openedCount") or 0)
    return {
        "ok": True, "dayKey": key, "closesAtIso": doc.get("closesAtIso"), "paths": out_paths, "mode": _doc_mode(doc),
        "opensLeft": _allowance(opened, extra["count"], premium), "premium": premium,
        "canUnlock": {"ad": not extra["ad"], "crown": True},
        "streak": _streak_value(memory, key), "accuracy": accuracy,
    }


def _streak_value(memory: Dict[str, Any], key: str) -> int:
    day = str(memory.get("streakDay") or "")
    if day in (key, _prev_day(key)):
        return int(memory.get("streak") or 0)
    return 0


# =========================
# ENDPOINTS
# =========================
class OpenRequest(BaseModel):
    path: str


class UnlockRequest(BaseModel):
    method: str  # "ad" | "crown"


class DecisionRequest(BaseModel):
    path: str
    decision: str            # "accepted" | "skipped"
    reason: Optional[str] = None
    dayKey: Optional[str] = None   # the day the path belongs to (a decision sent just after 07:30 IST is yesterday's)


class TodayRequest(BaseModel):
    mode: Optional[str] = None      # "rishta" from 1.0.8 for a profile that wants marriage; anything else = dating


def request_mode(body: Optional[TodayRequest]) -> str:
    return MODE_RISHTA if body is not None and str(body.mode or "").strip().lower() == MODE_RISHTA else MODE_QUICK


def viewer_block(mode: str, pub: Dict[str, Any], priv: Dict[str, Any], today: str) -> Optional[str]:
    """Who may have today's Fates in this mode (plan §2 C1): rishta mode needs profiles.marriageIntent ('needs_intent')
    and, for a man's own profile, 21 years ('groom_age'), each a 403; the dating Fates are only for one's own profile
    ('no_quick', an empty day)."""
    if mode == MODE_RISHTA:
        return fe.rishta_viewer_block(priv, pub, today)
    return fe.quick_viewer_block(priv)


def _reconcile_mode(db, ref, doc: Optional[Dict[str, Any]], mode: str, memory: Dict[str, Any], uid: str,
                    key: str) -> Optional[Dict[str, Any]]:
    """A fate day belongs to one mode. When the app asks in the other one (an update during the day, a second phone),
    nothing unopened from the old mode is shown. An untouched day is made again (None), but only MODE_SWITCH_REMAKES
    times a fate day; otherwise only the opened paths stay, which after that one remake may be none: an empty day in
    the new mode, looked at again by the quiet-day re-check like any empty day."""
    if not doc or _doc_mode(doc) == mode:
        return doc
    opened = {k: v for k, v in (doc.get("paths") or {}).items() if (v or {}).get("opened")}
    if not opened and _today_n(memory, "modeSwitch", key) < MODE_SWITCH_REMAKES \
            and _take_today(db, uid, key, "modeSwitch", MODE_SWITCH_REMAKES):
        ref.delete()
        return None
    order = [p for p in doc.get("order", []) if p in opened]
    ref.update({"paths": opened, "order": order, "mode": mode})
    return {**doc, "paths": opened, "order": order, "mode": mode}


@router.post("/today")
def fates_today(body: Optional[TodayRequest] = Body(default=None), authorization: Optional[str] = Header(default=None)):
    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    now_ms = int(time.time() * 1000)
    mode = request_mode(body)
    viewer, pub, priv, usr = _load(db, uid)
    memory = _doc(db, f"fatesMemory/{uid}")
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))
    accuracy = _accuracy(viewer, {**usr, **priv})
    blocked = viewer_block(mode, pub, priv, fe.ist_day())
    if blocked and mode == MODE_RISHTA:
        # Plan §2 C1: rishta-mode Fates refuse like getRishtas and swipeAction, 403 'needs_intent' / 'groom_age'.
        raise HTTPException(status_code=403, detail=blocked)
    if blocked:
        # Dating mode for a family-made profile ('no_quick'): an empty day, as 1.0.7 reads any other 403 as an error.
        return _empty_view(memory, key, premium, accuracy, mode, blocked)
    if mode == MODE_RISHTA:
        fe.rishta_ready(viewer, viewer=True)
    if fe.VERIFIED_ONLY and not viewer.face_verified:
        return _verify_first(db, viewer, memory, key, now_ms, premium, accuracy, priv)
    ref = db.document(f"dailyFates/{uid}__{key}")
    doc = _reconcile_mode(db, ref, ref.get().to_dict(), mode, memory, uid, key)
    if doc and not doc.get("order") and _older_than(doc.get("createdAtIso"), EMPTY_DAY_RECHECK_HOURS):
        # A quiet day is looked at again a few hours later: someone new may have joined since.
        ref.delete()
        doc = None
    if not doc:
        doc = generate(db, viewer, memory, key, now_ms)
        doc = _store_generation(db, doc, key)
        memory = _doc(db, f"fatesMemory/{uid}")
    return client_view(doc, memory, premium, accuracy)


EMPTY_DAY_RECHECK_HOURS = 3


def _empty_view(memory: Dict[str, Any], key: str, premium: bool, accuracy: Dict[str, Any], mode: str,
                reason: str) -> Dict[str, Any]:
    """No dating Fates for this viewer (nobody is picked or stored): 'no_quick'. Rishta mode refuses with a 403."""
    return {
        "ok": True, "dayKey": key, "closesAtIso": day_closes_at(key).isoformat(), "paths": [], "mode": mode,
        "reason": reason, "opensLeft": 0, "premium": premium, "canUnlock": {"ad": False, "crown": False},
        "streak": _streak_value(memory, key), "accuracy": accuracy,
    }


def _verify_reason(priv: Dict[str, Any]) -> str:
    flagged = bool(priv.get("faceVerified") or priv.get("isFaceVerified"))
    if flagged:
        return "photos_changed"
    reason = str(priv.get("faceVerifiedReason") or "")
    return "" if reason in ("", "ok") else reason


def _teaser_counts(teaser: Any, key: str) -> Dict[str, int]:
    """Today's verify-teaser counts per mode from fatesMemory.verifyTeaser = {day, n: {quick, rishta}}; a plain number
    is a count stored before 1.0.8, when only the dating Fates existed."""
    if not isinstance(teaser, dict) or teaser.get("day") != key:
        return {}
    n = teaser.get("n")
    if isinstance(n, dict):
        return {m: int(v or 0) for m, v in n.items() if m in (MODE_QUICK, MODE_RISHTA)}
    return {MODE_QUICK: int(n or 0)}


def _verify_first(db, viewer: fe.Person, memory: Dict[str, Any], key: str, now_ms: int, premium: bool,
                  accuracy: Dict[str, Any], priv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Daily Fates is for verified people only. Before that, only how many verified people near the
    viewer Mithu could bring (counted once a day per mode, no one is picked and nothing is stored as a fate)."""
    mode = MODE_RISHTA if viewer.rishta else MODE_QUICK
    counts = _teaser_counts(memory.get("verifyTeaser"), key)
    if mode in counts:
        n = counts[mode]
    else:
        n = len(_candidates(db, viewer, now_ms, memory))
        try:
            # The whole map is replaced, so another day's count of the other mode never stays.
            db.document(f"fatesMemory/{viewer.uid}").set({"verifyTeaser": {"day": key, "n": {**counts, mode: n}}},
                                                          merge=["verifyTeaser"])
        except Exception:
            logger.warning("fates verify teaser cache failed", exc_info=True)
    _stat(db, key, needsVerification=1)
    return {
        "ok": True, "dayKey": key, "closesAtIso": day_closes_at(key).isoformat(), "paths": [], "mode": mode,
        "needsVerification": True, "verifiedNearby": n,
        # "photos_changed": verified, but the photos aren't the ones checked (the app re-checks them by itself);
        # otherwise the last check's reason ("main_photo_not_you", "photo_3_not_you", ...) or "".
        "verifyReason": _verify_reason(priv or {}),
        "opensLeft": 0, "premium": premium, "canUnlock": {"ad": False, "crown": False},
        "streak": _streak_value(memory, key), "accuracy": accuracy,
    }


def _older_than(iso: Optional[str], hours: float) -> bool:
    try:
        created = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return True
    return datetime.now(timezone.utc) - created > timedelta(hours=hours)


def _pair_for(db, viewer: fe.Person, target_uid: str, now_ms: int) -> fe.Pair:
    cand, _, _ = load_person(db, target_uid)
    return fe.evaluate_pair(viewer, cand, now_ms, _match_fn)


def _require_verified(db, uid: str, viewer: Optional[fe.Person] = None) -> None:
    """Opening, accepting and revealing need a verification that still holds: the team may have removed it
    after today's paths were made (the app then shows the verify screen again)."""
    if not fe.VERIFIED_ONLY:
        return
    if viewer is None:
        viewer, _, _ = load_person(db, uid)
    if not viewer.face_verified:
        raise HTTPException(status_code=403, detail="not_verified")


@router.post("/open")
def fates_open(body: OpenRequest, authorization: Optional[str] = Header(default=None)):
    from google.cloud import firestore as fs

    uid = _d().verify_uid(authorization)
    db = _db()
    _require_verified(db, uid)
    key = day_key()
    ref = db.document(f"dailyFates/{uid}__{key}")
    mem_ref = db.document(f"fatesMemory/{uid}")
    usr = _doc(db, f"users/{uid}")
    priv = _doc(db, f"profiles/{uid}")
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))

    @fs.transactional
    def claim(tx) -> Tuple[str, Dict[str, Any]]:
        snap = ref.get(transaction=tx)
        doc = snap.to_dict() if snap.exists else None
        if not doc or body.path not in (doc.get("paths") or {}):
            raise HTTPException(status_code=404, detail="no_such_path")
        p = doc["paths"][body.path]
        if p.get("opened"):
            return "already", doc
        memory = mem_ref.get(transaction=tx).to_dict() or {}
        extra = _extra_today(memory, key)
        if _allowance(int(doc.get("openedCount") or 0), extra["count"], premium) <= 0:
            raise HTTPException(status_code=402, detail="no_opens_left")
        tx.update(ref, {f"paths.{body.path}.opened": True, f"paths.{body.path}.openedAtIso": _now_iso(),
                        "openedCount": int(doc.get("openedCount") or 0) + 1})
        streak_day = str(memory.get("streakDay") or "")
        patch: Dict[str, Any] = {"learned": fe.learn(memory.get("learned"), "open", body.path)}
        if streak_day != key:
            patch["streak"] = int(memory.get("streak") or 0) + 1 if streak_day == _prev_day(key) else 1
            patch["streakDay"] = key
        tx.set(mem_ref, patch, merge=True)
        tx.update(mem_ref, {f"seen.{p['targetUid']}": {"day": key, "opened": True}})
        return "claimed", doc

    status, doc = claim(db.transaction())
    p = doc["paths"][body.path]
    if status == "already" and p.get("report"):
        memory = _doc(db, f"fatesMemory/{uid}")
        return client_view(doc, memory, premium, {})
    if status == "claimed":
        _stat(db, key, opens=1, **{f"opens_{body.path}": 1})

    # The report is plain math and returns at once; the AI verdict comes from /fates/verdict while the
    # person reads the first scenes.
    viewer, _, _ = load_person(db, uid)
    viewer.rishta = _doc_mode(doc) == MODE_RISHTA     # the report compares what families compare in rishta mode
    pair = _pair_for(db, viewer, p["targetUid"], int(time.time() * 1000))
    ref.update({f"paths.{body.path}.report": fe.report(viewer, body.path, pair)})
    doc = ref.get().to_dict() or doc
    memory = _doc(db, f"fatesMemory/{uid}")
    return client_view(doc, memory, premium, {})


class VerdictRequest(BaseModel):
    path: str


@router.post("/verdict")
def fates_verdict(body: VerdictRequest, authorization: Optional[str] = Header(default=None)):
    """Mithu's verdict for an opened path: DeepSeek within the daily cap, else a template. Cached."""
    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    ref = db.document(f"dailyFates/{uid}__{key}")
    doc = ref.get().to_dict() or {}
    p = (doc.get("paths") or {}).get(body.path)
    if not p or not p.get("opened"):
        raise HTTPException(status_code=409, detail="not_opened")
    if p.get("verdict"):
        return {"ok": True, "verdict": p["verdict"], "why": p.get("why") or ""}
    viewer, _, _ = load_person(db, uid)
    viewer.rishta = _doc_mode(doc) == MODE_RISHTA
    pair = _pair_for(db, viewer, p["targetUid"], int(time.time() * 1000))
    pair.chose_you = bool((p.get("teaser") or {}).get("choseYou"))
    facts = fe.facts_for_ai(viewer, body.path, pair)
    verdict = fates_ai.verdict(_complete_short, facts) if _ai_allowed(db, uid, key) else None
    _stat(db, key, aiVerdictOk=1 if verdict else 0, aiVerdictFallback=0 if verdict else 1)
    verdict = verdict or fates_ai.template_verdict(facts)
    ref.update({f"paths.{body.path}.verdict": verdict})
    return {"ok": True, "verdict": verdict, "why": p.get("why") or ""}


def _take_crown(tx, profile_ref) -> int:
    """Spends one crown inside a transaction (same balance fields as the rest of the app); 402 when none."""
    prof = profile_ref.get(transaction=tx).to_dict() or {}
    balance = 0
    for k in ("crownsBalance", "crownsCount", "crowns"):
        try:
            balance = max(balance, int(prof.get(k) or 0))
        except (TypeError, ValueError):
            pass
    if balance <= 0:
        raise HTTPException(status_code=402, detail="no_crowns")
    nb = balance - 1
    now = _now_iso()
    tx.set(profile_ref, {"crownsBalance": nb, "crownsCount": nb, "crowns": nb, "lastCrownUsedAt": now, "updatedAt": now}, merge=True)
    return nb


@router.post("/unlock")
def fates_unlock(body: UnlockRequest, authorization: Optional[str] = Header(default=None)):
    from google.cloud import firestore as fs

    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    mem_ref = db.document(f"fatesMemory/{uid}")
    profile_ref = db.document(f"profiles/{uid}")
    if body.method not in ("ad", "crown"):
        raise HTTPException(status_code=400, detail="bad_method")

    @fs.transactional
    def grant(tx) -> Dict[str, Any]:
        memory = mem_ref.get(transaction=tx).to_dict() or {}
        extra = _extra_today(memory, key)
        if body.method == "ad":
            if extra["ad"]:
                raise HTTPException(status_code=429, detail="ad_used_today")
            extra["ad"] = True
        else:
            _take_crown(tx, profile_ref)
        extra["count"] += 1
        tx.set(mem_ref, {"extra": extra}, merge=True)
        return extra

    extra = grant(db.transaction())
    _stat(db, key, **{f"unlock_{body.method}": 1})
    return {"ok": True, "extraOpens": extra["count"], "adUsed": extra["ad"]}


@router.post("/decision")
def fates_decision(body: DecisionRequest, authorization: Optional[str] = Header(default=None)):
    uid = _d().verify_uid(authorization)
    db = _db()
    if body.decision not in ("accepted", "skipped"):
        raise HTTPException(status_code=400, detail="bad_decision")
    if body.decision == "accepted":
        _require_verified(db, uid)
    key = day_key()
    yesterday = day_key(datetime.now(timezone.utc) - timedelta(days=1))
    if body.dayKey in (key, yesterday):
        key = body.dayKey
    ref = db.document(f"dailyFates/{uid}__{key}")
    doc = ref.get().to_dict() or {}
    p = (doc.get("paths") or {}).get(body.path)
    if not p or not p.get("opened"):
        raise HTTPException(status_code=409, detail="not_opened")
    if p.get("decision"):
        return {"ok": True, "already": True}
    reason = body.reason if body.reason in fe.SKIP_REASONS else ("other" if body.decision == "skipped" else "")
    ref.update({f"paths.{body.path}.decision": body.decision,
                f"paths.{body.path}.decisionReason": reason,
                f"paths.{body.path}.decidedAtIso": _now_iso()})
    mem_ref = db.document(f"fatesMemory/{uid}")
    memory = _doc(db, f"fatesMemory/{uid}")
    event = "accept" if body.decision == "accepted" else "skip"
    mem_ref.set({"learned": fe.learn(memory.get("learned"), event, body.path, reason)}, merge=True)
    _stat(db, key, **({"accepts": 1} if event == "accept" else {"skips": 1, f"skip_{reason}": 1}))
    return {"ok": True}


@router.post("/journal")
def fates_journal(authorization: Optional[str] = Header(default=None)):
    """The last 30 fate days, newest first. Closed paths stay anonymous."""
    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    days = [(datetime.fromisoformat(key) - timedelta(days=i)).date().isoformat() for i in range(30)]
    refs = [db.document(f"dailyFates/{uid}__{d}") for d in days]
    items = []
    for snap in db.get_all(refs):
        if not snap.exists:
            continue
        doc = snap.to_dict() or {}
        for path in doc.get("order", []):
            p = doc["paths"][path]
            entry = {"dayKey": doc["dayKey"], "path": path, "opened": p.get("opened", False),
                     "decision": p.get("decision"), "teaser": p["teaser"]}
            if p.get("opened") and p.get("report"):
                person = p["report"]["person"]
                entry["person"] = {"firstName": person["firstName"], "age": person["age"],
                                   "photo": (person.get("photos") or [None])[0]}
            items.append(entry)
    items.sort(key=lambda e: e["dayKey"], reverse=True)
    return {"ok": True, "items": items}


# =========================
# CHOSE YOU
# =========================
class RevealRequest(BaseModel):
    uid: str


def _chooser_card(c: Dict[str, Any], pub: Dict[str, Any], viewer: fe.Person, revealed: bool) -> Dict[str, Any]:
    p = fe.person_from_docs(c["uid"], pub)
    km = fe.distance_km(viewer, p)
    dist = "" if not viewer.show_distance else fe.approx_distance(km, viewer.geo_precise and p.geo_precise, viewer.units)
    card: Dict[str, Any] = {
        "uid": c["uid"], "path": c["path"], "note": c["note"], "createdAtIso": c["createdAtIso"], "revealed": revealed,
        "teaser": {"age": p.age, "city": p.city, "distance": dist, "verified": p.face_verified},
    }
    if revealed:
        card["person"] = {"firstName": p.first_name, "age": p.age, "city": p.city, "photo": (p.photos or [None])[0]}
    return card


@router.post("/chosen")
def fates_chosen(authorization: Optional[str] = Header(default=None)):
    """Everyone waiting who chose the viewer as a fate, newest first. Without Parrot+ (or a crown for
    that person) they stay blurred, with their note and when Mithu brings them."""
    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    now_ms = int(time.time() * 1000)
    viewer, pub, priv, usr = _load(db, uid)
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))
    memory = _doc(db, f"fatesMemory/{uid}")
    today = _doc(db, f"dailyFates/{uid}__{key}")
    # Who Mithu can bring follows today's mode (before today's paths exist: rishta for a profile that wants marriage).
    if today.get("mode") == MODE_RISHTA or (not today and fe.rishta_viewer_block(priv, pub, fe.ist_day()) is None):
        fe.rishta_ready(viewer, viewer=True)
    in_today = {p.get("targetUid"): (path, p) for path, p in (today.get("paths") or {}).items()}
    likes = _incoming_fate_likes(db, uid)
    revealed = memory.get("revealed") or {}

    # Who Mithu can bring, in the order he will (only the oldest few are checked).
    queue: List[str] = []
    checked: set = set()
    for c in likes:
        if c["uid"] in in_today or len(checked) >= CHOOSER_CHECKS:
            continue
        checked.add(c["uid"])
        if _chooser_pair(db, viewer, c["uid"], now_ms, memory) is not None:
            queue.append(c["uid"])

    items: List[Dict[str, Any]] = []
    for c in list(reversed(likes))[:CHOSEN_LIST_LIMIT]:
        pub = _doc(db, f"publicProfiles/{c['uid']}")
        if not pub:
            continue
        today_hit = in_today.get(c["uid"])
        is_revealed = premium or c["uid"] in revealed or bool(today_hit and today_hit[1].get("opened"))
        card = _chooser_card(c, pub, viewer, is_revealed)
        if today_hit:
            card["status"], card["todayPath"], card["opened"] = "today", today_hit[0], bool(today_hit[1].get("opened"))
        elif c["uid"] in revealed:
            card["status"] = "revealed"
        elif c["uid"] in queue:
            card["status"], card["position"] = ("next" if queue[0] == c["uid"] else "queue"), queue.index(c["uid"]) + 1
        elif c["uid"] in checked:
            card["status"] = "later"      # outside the viewer's filters, or already brought twice
        else:
            card["status"] = "queue"
        if not is_revealed:
            card["_photo"] = (fe.person_from_docs(c["uid"], pub).photos or [None])[0]
        items.append(card)

    # Blurred thumbnails for the hidden ones, cached per person in fatesMemory.
    cache = dict(memory.get("chooserThumbs") or {})
    todo = [it for it in items if not it["revealed"] and it.get("_photo") and it["uid"] not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:
            got = {it["uid"]: ex.submit(blur_thumbs, it["_photo"], it["uid"]) for it in todo}
            for u, fut in got.items():
                blur = fut.result()[0]
                if blur:
                    cache[u] = blur
        keep = {it["uid"] for it in items}
        pruned = {k: v for k, v in cache.items() if k in keep}
        ref = db.document(f"fatesMemory/{uid}")
        try:
            ref.update({"chooserThumbs": pruned})      # replaces the whole map, so answered people drop out
        except Exception:
            try:
                ref.set({"chooserThumbs": pruned}, merge=True)
            except Exception:
                logger.warning("fates chooser thumbs cache failed", exc_info=True)
    for it in items:
        it.pop("_photo", None)
        if not it["revealed"]:
            it["blurThumb"] = cache.get(it["uid"])
    _stat(db, key, chosenViews=1)
    return {"ok": True, "premium": premium, "bringsAtIso": day_closes_at(key).isoformat(), "items": items}


@router.post("/chosen/reveal")
def fates_chosen_reveal(body: RevealRequest, authorization: Optional[str] = Header(default=None)):
    """Shows one person who chose the viewer now: free with Parrot+, else one crown. Marks them as an
    opened fate, so liking back is free (and they are not brought again as a path)."""
    from google.cloud import firestore as fs

    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    likes = {c["uid"]: c for c in _incoming_fate_likes(db, uid)}
    c = likes.get(body.uid)
    if c is None:
        raise HTTPException(status_code=404, detail="not_found")
    viewer, priv, usr = load_person(db, uid)
    _require_verified(db, uid, viewer)
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))
    mem_ref = db.document(f"fatesMemory/{uid}")
    profile_ref = db.document(f"profiles/{uid}")

    @fs.transactional
    def pay(tx) -> Tuple[bool, Optional[int]]:
        memory = mem_ref.get(transaction=tx).to_dict() or {}
        if body.uid in (memory.get("revealed") or {}):
            return False, None
        left = None if premium else _take_crown(tx, profile_ref)
        tx.set(mem_ref, {"revealed": {body.uid: key}, "seen": {body.uid: {"day": key, "opened": True}}}, merge=True)
        return True, left

    charged, crowns_left = pay(db.transaction())
    if charged:
        _stat(db, key, **({"revealPremium": 1} if premium else {"revealCrown": 1}))
    pub = _doc(db, f"publicProfiles/{body.uid}")
    card = _chooser_card(c, pub, viewer, True)
    card["status"] = "revealed"
    return {"ok": True, "item": card, "charged": charged and not premium, "crownsLeft": crowns_left}

