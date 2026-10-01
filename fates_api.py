"""Daily Fates HTTP API (Cloud Run). main.py wires the shared helpers in with configure().

Data (all server-only, the app talks to these endpoints and never reads the collections):
  dailyFates/{uid}__{dayKey}  today's paths: teasers, and for opened paths the report and verdict
  fatesMemory/{uid}           who was shown when, streak, today's extra opens
  fateExposure/{dayKey}       how many people got each person as a fate today (fairness cap)
Someone who accepted the viewer as a fate ("chose you") is brought into the viewer's own paths the
next morning, one a day: that is the free way to meet them. Parrot+ or a crown shows them at once.
A fate day starts at 07:30 IST, when the morning push goes out.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

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


def _stat(db, key: str, **counts: int) -> None:
    """Best-effort daily counters in fatesStats/{dayKey}; never fails a request."""
    try:
        from google.cloud import firestore as fs
        db.document(f"fatesStats/{key}").set({k: fs.Increment(v) for k, v in counts.items() if v}, merge=True)
    except Exception:
        logger.warning("fates stat write failed", exc_info=True)


def _ai_allowed(db, uid: str, key: str) -> bool:
    """Counts one AI call against today's cap; False once the cap is reached."""
    if not AI_ENABLED:
        return False
    from google.cloud import firestore as fs
    ref = db.document(f"fatesMemory/{uid}")

    @fs.transactional
    def take(tx) -> bool:
        mem = ref.get(transaction=tx).to_dict() or {}
        calls = mem.get("aiCalls") or {}
        n = int(calls.get("n") or 0) if calls.get("day") == key else 0
        if n >= AI_DAILY_CAP:
            return False
        tx.set(ref, {"aiCalls": {"day": key, "n": n + 1}}, merge=True)
        return True

    try:
        return take(db.transaction())
    except Exception:
        logger.warning("fates ai cap check failed", exc_info=True)
        return False


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


def load_person(db, uid: str, public: Optional[Dict[str, Any]] = None) -> Tuple[fe.Person, Dict[str, Any], Dict[str, Any]]:
    pub = public if public is not None else _doc(db, f"publicProfiles/{uid}")
    priv = _doc(db, f"profiles/{uid}")
    usr = _doc(db, f"users/{uid}")
    person = fe.person_from_docs(uid, pub, priv, usr)
    person.chart = _chart({**pub, **usr, **priv})
    return person, priv, usr


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
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SEEN_COOLDOWN_DAYS)).date().isoformat()
    seen = memory.get("seen") or {}
    excluded |= {u for u, v in seen.items() if str((v or {}).get("day", "")) >= cutoff}

    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        q = db.collection("publicProfiles").where(filter=FieldFilter("isDiscoverable", "==", True))
        # Equality filters on two fields need no composite index (Firestore merges single-field indexes).
        if viewer.looking_for in ("male", "female"):
            q = q.where(filter=FieldFilter("gender", "==", viewer.looking_for.capitalize()))
        snaps = list(q.limit(CANDIDATE_POOL_LIMIT).stream())
    except Exception:
        logger.exception("fates pool query failed")
        return []

    quick = []
    for s in snaps:
        if s.id == viewer.uid or s.id in excluded:
            continue
        pub = s.to_dict() or {}
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
        p = fe.person_from_docs(uid, pub, privs.get(uid), users.get(uid))
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
    cand, _, _ = load_person(db, uid)
    if fe.hard_filter(viewer, cand, now_ms) is not None:
        return None
    pair = fe.evaluate_pair(viewer, cand, now_ms, _match_fn)
    pair.chose_you = True
    return pair if fe.chooser_path(viewer, pair) else None


# =========================
# THUMBNAILS
# =========================
def blur_thumbs(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Two tiny JPEGs (data URIs) for a closed card: a heavy blur, and a lighter "peek" that shows while
    the card is held. Both are 64 px wide, so neither can reveal who it is; the real photo only
    arrives when the path is opened."""
    try:
        import cv2
        import numpy as np
        r = requests.get(url, timeout=6)
        if r.status_code != 200 or len(r.content) > 12 * 1024 * 1024:
            return None, None
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
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
    exposure = (_doc(db, f"fateExposure/{key}").get("counts") or {})
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
        thumbs = {path: ex.submit(blur_thumbs, p.cand.photos[0]) for path, p in chosen.items()}
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
        batch.set(db.document(f"fateExposure/{key}"),
                  {"counts": {p["targetUid"]: fs.Increment(1) for p in doc["paths"].values()}}, merge=True)
    try:
        batch.commit()
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
        "ok": True, "dayKey": key, "closesAtIso": doc.get("closesAtIso"), "paths": out_paths,
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


@router.post("/today")
def fates_today(authorization: Optional[str] = Header(default=None)):
    uid = _d().verify_uid(authorization)
    db = _db()
    key = day_key()
    now_ms = int(time.time() * 1000)
    viewer, priv, usr = load_person(db, uid)
    memory = _doc(db, f"fatesMemory/{uid}")
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))
    if fe.VERIFIED_ONLY and not viewer.face_verified:
        return _verify_first(db, viewer, memory, key, now_ms, premium, _accuracy(viewer, {**usr, **priv}), priv)
    ref = db.document(f"dailyFates/{uid}__{key}")
    doc = ref.get().to_dict()
    if doc and not doc.get("order") and _older_than(doc.get("createdAtIso"), EMPTY_DAY_RECHECK_HOURS):
        # A quiet day is looked at again a few hours later: someone new may have joined since.
        ref.delete()
        doc = None
    if not doc:
        doc = generate(db, viewer, memory, key, now_ms)
        doc = _store_generation(db, doc, key)
        memory = _doc(db, f"fatesMemory/{uid}")
    return client_view(doc, memory, premium, _accuracy(viewer, {**usr, **priv}))


EMPTY_DAY_RECHECK_HOURS = 3


def _verify_reason(priv: Dict[str, Any]) -> str:
    flagged = bool(priv.get("faceVerified") or priv.get("isFaceVerified"))
    if flagged:
        return "photos_changed"
    reason = str(priv.get("faceVerifiedReason") or "")
    return "" if reason in ("", "ok") else reason


def _verify_first(db, viewer: fe.Person, memory: Dict[str, Any], key: str, now_ms: int, premium: bool,
                  accuracy: Dict[str, Any], priv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Daily Fates is for verified people only. Before that, only how many verified people near the
    viewer Mithu could bring (counted once a day, no one is picked and nothing is stored as a fate)."""
    teaser = memory.get("verifyTeaser") or {}
    if teaser.get("day") == key:
        n = int(teaser.get("n") or 0)
    else:
        n = len(_candidates(db, viewer, now_ms, memory))
        try:
            db.document(f"fatesMemory/{viewer.uid}").set({"verifyTeaser": {"day": key, "n": n}}, merge=True)
        except Exception:
            logger.warning("fates verify teaser cache failed", exc_info=True)
    _stat(db, key, needsVerification=1)
    return {
        "ok": True, "dayKey": key, "closesAtIso": day_closes_at(key).isoformat(), "paths": [],
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


@router.post("/open")
def fates_open(body: OpenRequest, authorization: Optional[str] = Header(default=None)):
    from google.cloud import firestore as fs

    uid = _d().verify_uid(authorization)
    db = _db()
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
    key = day_key()
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
    viewer, priv, usr = load_person(db, uid)
    premium = _d().is_premium(usr, priv, datetime.now(timezone.utc))
    memory = _doc(db, f"fatesMemory/{uid}")
    today = _doc(db, f"dailyFates/{uid}__{key}")
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
            got = {it["uid"]: ex.submit(blur_thumbs, it["_photo"]) for it in todo}
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

