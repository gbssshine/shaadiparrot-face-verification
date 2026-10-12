"""Ashtakoota Guna Milan (North Indian, 36 points, Moon-based, sidereal Lahiri) and doshas.

Tables were cross-checked in Sept 2026 against AstroSage and Prokerala software output and several
independent published tables (see docs in tests/ashtakoota_fixtures.json). Every matrix here is
[boy][girl]. Raw koota points are kept as the software shows them; dosha cancellations are reported
separately and never change the 36-point total.

Manglik (audit 9, plan C3): a status per chart, "yes" | "partial" | "no" | "check". Only the Lagna gives a
full "yes"; Mars counted from the Moon or Venus alone is "partial"; without the Lagna (no birth time or
place) it stays "check", never a verdict, even when a cancellation applies (its reason is still listed);
with the Lagna known, the common cancellations make it "no".
"""
from __future__ import annotations

import re
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import swisseph as swe

import fates_astro_chart as _fc
from fates_astro_chart import _parse_date, _parse_time, house_from  # noqa: F401  (house_from is re-exported)

# ---------------------------------------------------------------- nakshatras 0..26
NAKSHATRAS: List[Tuple[str, str, str, str]] = [   # name, gana, nadi, yoni animal
    ("Ashwini", "Deva", "Adi", "Horse"), ("Bharani", "Manushya", "Madhya", "Elephant"),
    ("Krittika", "Rakshasa", "Antya", "Sheep"), ("Rohini", "Manushya", "Antya", "Serpent"),
    ("Mrigashira", "Deva", "Madhya", "Serpent"), ("Ardra", "Manushya", "Adi", "Dog"),
    ("Punarvasu", "Deva", "Adi", "Cat"), ("Pushya", "Deva", "Madhya", "Sheep"),
    ("Ashlesha", "Rakshasa", "Antya", "Cat"), ("Magha", "Rakshasa", "Antya", "Rat"),
    ("Purva Phalguni", "Manushya", "Madhya", "Rat"), ("Uttara Phalguni", "Manushya", "Adi", "Cow"),
    ("Hasta", "Deva", "Adi", "Buffalo"), ("Chitra", "Rakshasa", "Madhya", "Tiger"),
    ("Swati", "Deva", "Antya", "Buffalo"), ("Vishakha", "Rakshasa", "Antya", "Tiger"),
    ("Anuradha", "Deva", "Madhya", "Deer"), ("Jyeshtha", "Rakshasa", "Adi", "Deer"),
    ("Mula", "Rakshasa", "Adi", "Dog"), ("Purva Ashadha", "Manushya", "Madhya", "Monkey"),
    ("Uttara Ashadha", "Manushya", "Antya", "Mongoose"), ("Shravana", "Deva", "Antya", "Monkey"),
    ("Dhanishta", "Rakshasa", "Madhya", "Lion"), ("Shatabhisha", "Rakshasa", "Adi", "Horse"),
    ("Purva Bhadrapada", "Manushya", "Adi", "Lion"), ("Uttara Bhadrapada", "Manushya", "Madhya", "Cow"),
    ("Revati", "Deva", "Antya", "Elephant"),
]

# ---------------------------------------------------------------- rashis 0..11
RASHIS: List[Tuple[str, str, str, str]] = [   # name, western, lord, varna
    ("Mesha", "Aries", "Mars", "Kshatriya"), ("Vrishabha", "Taurus", "Venus", "Vaishya"),
    ("Mithuna", "Gemini", "Mercury", "Shudra"), ("Karka", "Cancer", "Moon", "Brahmin"),
    ("Simha", "Leo", "Sun", "Kshatriya"), ("Kanya", "Virgo", "Mercury", "Vaishya"),
    ("Tula", "Libra", "Venus", "Shudra"), ("Vrischika", "Scorpio", "Mars", "Brahmin"),
    ("Dhanu", "Sagittarius", "Jupiter", "Kshatriya"), ("Makara", "Capricorn", "Saturn", "Vaishya"),
    ("Kumbha", "Aquarius", "Saturn", "Shudra"), ("Meena", "Pisces", "Jupiter", "Brahmin"),
]
VASHYA_BY_RASHI = ["Chatushpada", "Chatushpada", "Manava", "Jalachara", "Vanachara", "Manava",
                   "Manava", "Keeta", None, None, "Manava", "Jalachara"]   # Dhanu/Makara split at 15 deg
VARNA_RANK = {"Shudra": 0, "Vaishya": 1, "Kshatriya": 2, "Brahmin": 3}

VASHYA_ORDER = ["Chatushpada", "Manava", "Jalachara", "Vanachara", "Keeta"]
VASHYA = [[2, 1, 1, 0, 1], [1, 2, 0.5, 0, 1], [1, 0.5, 2, 1, 1], [0, 0, 1, 2, 0], [1, 1, 1, 0, 2]]

YONI_ORDER = ["Horse", "Elephant", "Sheep", "Serpent", "Dog", "Cat", "Rat", "Cow", "Buffalo", "Tiger",
              "Deer", "Monkey", "Mongoose", "Lion"]
YONI = [
    [4, 2, 2, 3, 2, 2, 2, 1, 0, 1, 1, 3, 2, 1], [2, 4, 3, 3, 2, 2, 2, 2, 3, 1, 2, 3, 2, 0],
    [2, 3, 4, 2, 1, 2, 1, 3, 3, 1, 2, 0, 3, 1], [3, 3, 2, 4, 2, 1, 1, 1, 1, 2, 2, 2, 0, 2],
    [2, 2, 1, 2, 4, 2, 1, 2, 2, 1, 0, 2, 1, 1], [2, 2, 2, 1, 2, 4, 0, 2, 2, 1, 3, 3, 2, 1],
    [2, 2, 1, 1, 1, 0, 4, 2, 2, 2, 2, 2, 1, 2], [1, 2, 3, 1, 2, 2, 2, 4, 3, 0, 3, 2, 2, 1],
    [0, 3, 3, 1, 2, 2, 2, 3, 4, 1, 2, 2, 2, 2], [1, 1, 1, 2, 1, 1, 2, 0, 1, 4, 1, 1, 2, 1],
    [3, 2, 2, 2, 0, 3, 2, 3, 2, 1, 4, 2, 2, 1], [3, 3, 0, 2, 2, 3, 2, 2, 2, 1, 2, 4, 3, 2],
    [2, 2, 3, 0, 1, 2, 1, 2, 2, 2, 2, 3, 4, 2], [1, 0, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 4],
]

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
GRAHA_MAITRI = [
    [5, 5, 5, 4, 5, 0, 0], [5, 5, 4, 1, 4, 0.5, 0.5], [5, 4, 5, 0.5, 5, 3, 0.5], [4, 1, 0.5, 5, 0.5, 5, 4],
    [5, 4, 5, 0.5, 5, 0.5, 3], [0, 0.5, 3, 5, 0.5, 5, 5], [0, 0.5, 0.5, 4, 3, 5, 5],
]
FRIENDS = {
    "Sun": {"Moon", "Mars", "Jupiter"}, "Moon": {"Sun", "Mercury"}, "Mars": {"Sun", "Moon", "Jupiter"},
    "Mercury": {"Sun", "Venus"}, "Jupiter": {"Sun", "Moon", "Mars"}, "Venus": {"Mercury", "Saturn"},
    "Saturn": {"Mercury", "Venus"},
}

GANA_ORDER = ["Deva", "Manushya", "Rakshasa"]
GANA = [[6, 6, 0], [5, 6, 0], [1, 0, 6]]

BHAKOOT_BAD = {2: "2/12", 12: "2/12", 5: "5/9", 9: "5/9", 6: "6/8", 8: "6/8"}
MANGLIK_HOUSES = {1, 2, 4, 7, 8, 12}
MANGLIK_FULL_HOUSES = {1, 4, 7, 8, 12}   # from the Lagna; the 2nd alone is partial
# The commonly published house-sign exceptions: Mars in these signs in that house is not Manglik.
MANGLIK_SIGN_EXCEPTIONS = {1: {0}, 2: {2, 5}, 4: {0, 7}, 7: {3, 9}, 8: {8, 11}, 12: {1, 6}}
MARS_OWN_OR_EXALTED = {0, 7, 9}           # Mesha, Vrischika (own), Makara (exalted)
JUPITER_ON_MARS = {1, 5, 7, 9}            # Mars with Jupiter, or in the 5th, 7th or 9th house from it (its aspects)
CANCEL_OWN_SIGN = "Mars in its own or exalted sign"
CANCEL_JUPITER = "Jupiter with or aspecting Mars"
MOON_KEYS_MANGLIK = {"yes": "yes", "partial": "mild", "no": "no", "check": "unknown"}

KOOTA_MAX = {"Varna": 1, "Vashya": 2, "Tara": 3, "Yoni": 4, "Graha Maitri": 5, "Gana": 6, "Bhakoot": 7, "Nadi": 8}


def vashya_group(rashi: int, moon_lon: Optional[float]) -> str:
    g = VASHYA_BY_RASHI[rashi]
    if g:
        return g
    within = (moon_lon % 30.0) if moon_lon is not None else 7.5
    if rashi == 8:   # Dhanu: first half human, second half four-footed
        return "Manava" if within < 15 else "Chatushpada"
    return "Chatushpada" if within < 15 else "Jalachara"   # Makara


def _tara_good(count: int) -> bool:
    return count % 9 not in (3, 5, 7)


def koota_points(boy: Dict[str, Any], girl: Dict[str, Any]) -> Dict[str, float]:
    """boy/girl: {"rashi": 0..11, "nak": 0..26, "lon": moon longitude or None}."""
    br, gr, bn, gn = boy["rashi"], girl["rashi"], boy["nak"], girl["nak"]
    pts: Dict[str, float] = {}
    pts["Varna"] = 1 if VARNA_RANK[RASHIS[br][3]] >= VARNA_RANK[RASHIS[gr][3]] else 0
    pts["Vashya"] = VASHYA[VASHYA_ORDER.index(vashya_group(br, boy.get("lon")))][VASHYA_ORDER.index(vashya_group(gr, girl.get("lon")))]
    g2b = (bn - gn) % 27 + 1
    b2g = (gn - bn) % 27 + 1
    pts["Tara"] = 1.5 * (int(_tara_good(g2b)) + int(_tara_good(b2g)))
    pts["Yoni"] = YONI[YONI_ORDER.index(NAKSHATRAS[bn][3])][YONI_ORDER.index(NAKSHATRAS[gn][3])]
    pts["Graha Maitri"] = GRAHA_MAITRI[PLANETS.index(RASHIS[br][2])][PLANETS.index(RASHIS[gr][2])]
    pts["Gana"] = GANA[GANA_ORDER.index(NAKSHATRAS[bn][1])][GANA_ORDER.index(NAKSHATRAS[gn][1])]
    d = (gr - br) % 12 + 1
    pts["Bhakoot"] = 0 if d in BHAKOOT_BAD else 7
    pts["Nadi"] = 0 if NAKSHATRAS[bn][2] == NAKSHATRAS[gn][2] else 8
    return pts


def _mutual_friends(a: str, b: str) -> bool:
    return b in FRIENDS[a] and a in FRIENDS[b]


# ---------------------------------------------------------------- doshas
def nadi_dosha(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    nadi = NAKSHATRAS[a["nak"]][2]
    if nadi != NAKSHATRAS[b["nak"]][2]:
        return {"name": "Nadi dosha", "present": False, "cancelled": False, "why": "Your Nadis differ."}
    same_rashi, same_nak = a["rashi"] == b["rashi"], a["nak"] == b["nak"]
    reason = None
    if same_rashi and not same_nak:
        reason = "same Moon sign, different nakshatras"
    elif same_nak and not same_rashi:
        reason = "same nakshatra, different Moon signs"
    elif same_nak and a.get("pada") and b.get("pada") and a["pada"] != b["pada"]:
        reason = "same nakshatra, different padas"
    elif not same_rashi and RASHIS[a["rashi"]][2] == RASHIS[b["rashi"]][2]:
        reason = f"both Moon signs are ruled by {RASHIS[a['rashi']][2]}"
    if reason:
        return {"name": "Nadi dosha", "present": True, "cancelled": True, "why": f"Same Nadi ({nadi}), cancelled: {reason}."}
    return {"name": "Nadi dosha", "present": True, "cancelled": False,
            "why": f"You share the same Nadi ({nadi}). Traditionally the weightiest dosha; a family pandit can advise remedies."}


def bhakoot_dosha(boy: Dict[str, Any], girl: Dict[str, Any]) -> Dict[str, Any]:
    d = (girl["rashi"] - boy["rashi"]) % 12 + 1
    if d not in BHAKOOT_BAD:
        return {"name": "Bhakoot dosha", "present": False, "cancelled": False, "why": "Your Moon signs sit well together."}
    la, lb = RASHIS[boy["rashi"]][2], RASHIS[girl["rashi"]][2]
    rel = BHAKOOT_BAD[d]
    if la == lb:
        return {"name": "Bhakoot dosha", "present": True, "cancelled": True,
                "why": f"Your Moon signs sit {rel}, but both are ruled by {la}, which cancels it."}
    if _mutual_friends(la, lb):
        return {"name": "Bhakoot dosha", "present": True, "cancelled": True,
                "why": f"Your Moon signs sit {rel}, but their lords, {la} and {lb}, are friends, which cancels it."}
    return {"name": "Bhakoot dosha", "present": True, "cancelled": False,
            "why": f"Your Moon signs sit {rel}: plan money and family matters together."}


def gana_dosha(pts: Dict[str, float]) -> Optional[Dict[str, Any]]:
    if pts["Gana"] > 1:
        return None
    if pts["Graha Maitri"] >= 5 or pts["Tara"] >= 3:
        return {"name": "Gana dosha", "present": True, "cancelled": True,
                "why": "Your temperaments differ, cancelled by " + ("friendly Moon lords." if pts["Graha Maitri"] >= 5 else "kind birth stars.")}
    return {"name": "Gana dosha", "present": True, "cancelled": False, "why": "Your temperaments differ: give each other room."}


def _sign_exception(label: str) -> str:
    return f"Mars's sign is an accepted exception for its house from {'Venus' if label == 'Venus' else 'the ' + label}"


def manglik_status(chart: Dict[str, Any]) -> Dict[str, Any]:
    """One chart's Manglik (plan C3). Keeps the old fields (manglik, mild, from, cancelledBy) for 1.0.7 apps
    and adds status ("yes" | "partial" | "no" | "check"), cancelReasons and sure (the Lagna is known).

    - Mars in a Manglik house from the Lagna (1, 4, 7, 8, 12): "yes"; the 2nd house from the Lagna: "partial".
    - Only from the Moon or Venus: "partial".
    - No Lagna (no birth time or place): "check", whatever the Moon, Venus and the cancellations say. The
      cancellations that apply are listed in cancelReasons, but there is no cancelledBy (no verdict).
    - With the Lagna known, cancellations make it "no": Mars in its own or exalted sign, Jupiter with or
      aspecting Mars, and the house-sign exceptions (MANGLIK_SIGN_EXCEPTIONS), which drop the house they apply to.
    The reasons never name a sign or a house, so a rishta member's side gives nothing of their chart away.
    """
    lagna = chart.get("lagna")
    sure = lagna is not None
    mars = chart.get("marsRashi")
    if mars is None:
        return {"manglik": False, "mild": False, "from": [], "status": "check", "cancelReasons": [], "sure": False}
    refs = []
    for label, ref in (("Lagna", lagna), ("Moon", chart.get("rashi")), ("Venus", chart.get("venusRashi"))):
        h = house_from(mars, ref)
        if h in MANGLIK_HOUSES:
            refs.append((label, h))
    kept = [(label, h) for label, h in refs if mars not in MANGLIK_SIGN_EXCEPTIONS[h]]
    excepted = [_sign_exception(label) for label, h in refs if mars in MANGLIK_SIGN_EXCEPTIONS[h]]
    whole = []
    if mars in MARS_OWN_OR_EXALTED:
        whole.append(CANCEL_OWN_SIGN)
    jupiter = chart.get("jupiterRashi")
    if jupiter is not None and house_from(mars, jupiter) in JUPITER_ON_MARS:
        whole.append(CANCEL_JUPITER)

    # Something to cancel: a house found, or an unknown Lagna that could still give one.
    reasons = (whole if (refs or not sure) else []) + excepted
    if not sure:
        status = "check"          # C3 row 1: no birth time, no verdict, not even a cancelled one
    elif whole and refs:
        status = "no"
    elif not kept:
        status = "no"
    else:
        lagna_house = next((h for label, h in kept if label == "Lagna"), None)
        status = "yes" if lagna_house in MANGLIK_FULL_HOUSES else "partial"

    manglik = bool(kept) and not whole
    out: Dict[str, Any] = {
        "manglik": manglik, "mild": manglik and all(h == 2 for _, h in kept),
        "from": [label for label, _ in (kept if manglik else refs)],
    }
    if status == "no" and reasons:
        out["cancelledBy"] = reasons[0]
    out.update({"status": status, "cancelReasons": reasons, "sure": sure})
    return out


MANGLIK_CHECK_WHY = ("The Manglik check needs both birth times and places, because it counts Mars from the Lagna. "
                     "Until then it stays a check, not a verdict.")


def manglik_dosha(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """The Manglik row of a match. status: "clear" | "cancelled" | "present" | "check" ("check" when either side
    is "check": never "present" without both Lagnas). present/cancelled keep their meaning for 1.0.7 apps."""
    ma, mb = manglik_status(a), manglik_status(b)
    sa, sb = ma["status"], mb["status"]
    if "check" in (sa, sb):
        # 1.0.7 apps still read present/cancelled: the old Moon-and-Venus reading, with the new cancellations.
        strong_a, strong_b = ma["manglik"] and not ma["mild"], mb["manglik"] and not mb["mild"]
        present = ma["manglik"] or mb["manglik"]
        cancelled = present and ((ma["manglik"] and mb["manglik"]) or not (strong_a or strong_b))
        return {"name": "Manglik", "present": present, "cancelled": cancelled, "why": MANGLIK_CHECK_WHY, "status": "check"}
    ya, yb = sa in ("yes", "partial"), sb in ("yes", "partial")
    if not ya and not yb:
        return {"name": "Manglik", "present": False, "cancelled": False, "why": "Neither of you is Manglik.", "status": "clear"}
    if ya and yb:
        return {"name": "Manglik", "present": True, "cancelled": True, "why": "You are both Manglik, which cancels it out.",
                "status": "cancelled"}
    if "yes" not in (sa, sb):
        return {"name": "Manglik", "present": True, "cancelled": True, "status": "cancelled",
                "why": "A partial Manglik placement on one side only, which most families don’t count as a dosha."}
    who = "You are" if sa == "yes" else "They are"
    return {"name": "Manglik", "present": True, "cancelled": False, "status": "present",
            "why": f"{who} Manglik and the other isn’t. Worth knowing: many families talk it over with a pandit."}


_MOON_OPTION_RE = re.compile(r"(\d{1,2})_(\d{1,2})(?:_[01])?")


def moon_option(o: Any) -> Optional[Tuple[int, int]]:
    """One option Moon as (rashi, nak), from any form it is kept in: [r, n] (a chart's options; a Moon doc written
    before the maps), {rashi, nak} (how users/{uid}/feedState/moon stores it: Firestore refuses an array inside an
    array), {r, n} or "r_n" (the forms guna.ts optionPair also reads). None when it is none of these or not a Moon
    at all (rashi 0-11, nakshatra 0-26)."""
    try:
        if isinstance(o, (list, tuple)):
            r, n = int(o[0]), int(o[1])
        elif isinstance(o, dict):
            key = ("rashi", "nak") if ("rashi" in o or "nak" in o) else ("r", "n")
            r, n = int(o[key[0]]), int(o[key[1]])
        elif isinstance(o, str):
            m = _MOON_OPTION_RE.fullmatch(o.strip())
            if not m:
                return None
            r, n = int(m.group(1)), int(m.group(2))
        else:
            return None
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (r, n) if 0 <= r < 12 and 0 <= n < 27 else None


def moon_options(chart: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Every (rashi, nak) the chart's day allows, in any form moon_option reads; the main Moon when there are none."""
    opts = [p for p in (moon_option(o) for o in (chart.get("options") or [])) if p is not None]
    return opts or [(int(chart["rashi"]), int(chart["nak"]))]


def moon_keys(chart: Dict[str, Any]) -> Dict[str, Any]:
    """What getRishtas needs to work out the 36 gunas without the astro server (matching §1/§3):
    main = [rashi, nak, half], where half (1 from 15 deg) matters only for Dhanu and Makara (the Vashya split);
    options = every [rashi, nak] the day allows; manglik in the matching vocabulary (yes | mild | no | unknown,
    from the C3 status yes | partial | no | check). This is the in-memory form (the guna fixture is built from
    it); the Firestore document is moon_keys_doc."""
    r, n = int(chart["rashi"]), int(chart["nak"])
    lon = chart.get("moonLon")
    half = 1 if r in (8, 9) and lon is not None and float(lon) % 30.0 >= 15.0 else 0
    options = [[o[0], o[1]] for o in moon_options(chart)]
    return {"main": [r, n, half], "options": options, "exact": chart.get("precision") == "exact",
            "manglik": MOON_KEYS_MANGLIK[manglik_status(chart)["status"]]}


def moon_keys_doc(chart: Dict[str, Any], sig: str, at: str) -> Dict[str, Any]:
    """The users/{uid}/feedState/moon document (matching §1): moon_keys with each option stored as a map
    {rashi, nak}, because Firestore refuses an array inside an array (INVALID_ARGUMENT "Cannot convert an array
    value in an array value"); main stays the flat [rashi, nak, half]. The one writer for /astro/scores
    (astro_api._remember_moons) and tools/backfill_moon_keys.py; moon_keys_of reads it back."""
    keys = moon_keys(chart)
    return {**keys, "options": [{"rashi": int(o[0]), "nak": int(o[1])} for o in keys["options"]],
            "sig": sig, "atIso": at}


def moon_keys_of(doc: Any) -> Optional[Dict[str, Any]]:
    """A stored users/{uid}/feedState/moon doc read back in the moon_keys form (options as [[rashi, nak], ...]):
    the {rashi, nak} maps moon_keys_doc writes and the old [[rashi, nak], ...] form alike (guna.ts moonKeysOf reads
    the same forms). Missing or empty options mean the main Moon only. None when the doc is missing or malformed:
    the kundli then counts as unknown, never as a wrong score."""
    if not isinstance(doc, dict):
        return None
    main = doc.get("main")
    if not isinstance(main, (list, tuple)) or len(main) != 3:
        return None
    head, half = moon_option(main[:2]), main[2]
    if head is None or isinstance(half, bool) or half not in (0, 1):
        return None
    options: List[List[int]] = []
    raw = doc.get("options")
    if isinstance(raw, (list, tuple)):
        for o in raw:
            p = moon_option(o)
            if p is None:
                return None
            options.append([p[0], p[1]])
    elif raw is not None:
        return None
    manglik = doc.get("manglik")
    return {"main": [head[0], head[1], int(half)], "options": options or [[head[0], head[1]]],
            "exact": doc.get("exact") is True,
            "manglik": manglik if manglik in MOON_KEYS_MANGLIK.values() else "unknown"}


# ---------------------------------------------------------------- words
def _meaning(name: str, pts: float, boy: Dict[str, Any], girl: Dict[str, Any], a_is_boy: bool) -> str:
    a, b = (boy, girl) if a_is_boy else (girl, boy)
    if name == "Varna":
        return "Work and ego: no fight over who leads." if pts >= 1 else "Work and ego: agree early on how you share decisions."
    if name == "Vashya":
        return {2: "A natural pull towards each other.", 1: "The pull between you is steady rather than magnetic.",
                0.5: "One of you may feel more drawn than the other."}.get(pts, "Little natural pull; attraction grows with time together.")
    if name == "Tara":
        return {3: "Your birth stars are good for each other’s luck and health.",
                1.5: "One birth star favours the other more; care keeps it balanced."}.get(pts, "Your birth stars don’t favour each other; small kindnesses matter more.")
    if name == "Yoni":
        ya, yb = NAKSHATRAS[a["nak"]][3], NAKSHATRAS[b["nak"]][3]
        if pts >= 4:
            return f"Same Yoni, {ya}: physical ease comes naturally."
        if pts >= 3:
            return f"Friendly Yonis ({ya} and {yb}): physical ease comes easily."
        if pts >= 2:
            return f"Neutral Yonis ({ya} and {yb}): closeness grows with time."
        if pts >= 1:
            return f"Different instincts ({ya} and {yb}): go at a pace you both like."
        return f"Opposite Yonis ({ya} and {yb}): be patient with each other’s rhythm."
    if name == "Graha Maitri":
        la, lb = RASHIS[a["rashi"]][2], RASHIS[b["rashi"]][2]
        if la == lb:
            return f"The same Moon lord, {la}: you think alike."
        if pts >= 5:
            return f"{la} and {lb} are friends: you think alike."
        if pts >= 4:
            return f"{la} and {lb} get on: easy to understand each other."
        if pts >= 3:
            return f"{la} and {lb} are neutral: you’ll learn each other’s way of thinking."
        return f"{la} and {lb} don’t get on: explain, don’t assume."
    if name == "Gana":
        ga, gb = NAKSHATRAS[a["nak"]][1], NAKSHATRAS[b["nak"]][1]
        if ga == gb:
            return f"Same nature, {ga}: at ease with each other."
        if pts >= 5:
            return f"{ga} and {gb}: gentle with each other."
        return f"{ga} and {gb}: different temperaments, give each other room."
    if name == "Bhakoot":
        d = (girl["rashi"] - boy["rashi"]) % 12 + 1
        return "Your Moon signs sit well together: good for family and money." if pts >= 7 else f"Your Moon signs sit {BHAKOOT_BAD[d]}: plan money and family together."
    if name == "Nadi":
        return "Different Nadi: no Nadi dosha." if pts >= 8 else f"Same Nadi ({NAKSHATRAS[a['nak']][2]}): see the doshas below."
    return ""


def moon_info(chart: Dict[str, Any]) -> Dict[str, Any]:
    r, n = chart["rashi"], chart["nak"]
    return {"rashi": RASHIS[r][0], "rashiEnglish": RASHIS[r][1], "nakshatra": NAKSHATRAS[n][0], "pada": chart.get("pada") or 0,
            "lord": RASHIS[r][2], "gana": NAKSHATRAS[n][1], "nadi": NAKSHATRAS[n][2], "yoni": NAKSHATRAS[n][3]}


def _moon(chart: Dict[str, Any], rashi: Optional[int] = None, nak: Optional[int] = None) -> Dict[str, Any]:
    return {"rashi": chart["rashi"] if rashi is None else rashi, "nak": chart["nak"] if nak is None else nak,
            "lon": chart.get("moonLon") if rashi is None else None, "pada": chart.get("pada") if rashi is None else None}


def gender_roles(a_gender: str, b_gender: str) -> List[bool]:
    """Who is the groom, as a list of "a is the boy" runs: one run when the genders say it, both directions
    (averaged by the 36 gunas; the first one taken by porutham) when they are the same or unknown."""
    if a_gender == "male" and b_gender != "male":
        return [True]
    if b_gender == "male" and a_gender != "male":
        return [False]
    if a_gender == "female" and b_gender not in ("female",):
        return [False]
    return [True, False]


def match_charts(a: Dict[str, Any], b: Dict[str, Any], a_gender: str, b_gender: str) -> Dict[str, Any]:
    """a = the viewer. Roles follow gender; same or unknown genders average both directions."""
    roles = gender_roles(a_gender, b_gender)

    ma, mb = _moon(a), _moon(b)

    def score(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, float]:
        runs = [koota_points(x, y) if o else koota_points(y, x) for o in roles]
        return {k: sum(r[k] for r in runs) / len(runs) for k in KOOTA_MAX}

    pts = score(ma, mb)
    total = sum(pts.values())

    # Range over every Moon position the day allows when a birth time is missing ([r, n] or a stored {rashi, nak}).
    opts_a, opts_b = moon_options(a), moon_options(b)
    totals = [sum(score(_moon(a, ra, na), _moon(b, rb, nb)).values()) for (ra, na), (rb, nb) in product(opts_a, opts_b)]

    a_is_boy = roles[0]
    boy, girl = (ma, mb) if a_is_boy else (mb, ma)
    kootas = [{"name": k, "points": pts[k], "max": KOOTA_MAX[k], "meaning": _meaning(k, pts[k], boy, girl, a_is_boy)} for k in KOOTA_MAX]
    doshas = [nadi_dosha(ma, mb), bhakoot_dosha(boy, girl), manglik_dosha(a, b)]
    g = gana_dosha(pts)
    if g:
        doshas.append(g)
    precisions = {a.get("precision"), b.get("precision")}
    precision = "exact" if precisions == {"exact"} else ("date" if "date" in precisions else "time")
    return {
        "total": round(total * 2) / 2, "totalMin": min(totals), "totalMax": max(totals), "precision": precision,
        "kootas": kootas, "doshas": doshas, "aMoon": moon_info(a), "bMoon": moon_info(b),
    }


def match_people(a, b) -> Optional[Dict[str, Any]]:
    """Engine hook: a, b are fates_engine.Person with .chart and .gender."""
    if not a.chart or not b.chart:
        return None
    return match_charts(a.chart, b.chart, a.gender, b.gender)


# ---------------------------------------------------------------- charts
def _jupiter_rashi(date: Tuple[int, int, int], time: Optional[Tuple[int, int]], place: Optional[Dict[str, Any]]) -> int:
    """Jupiter's sidereal sign at the same moment fates_astro_chart uses (local noon without a time)."""
    tz = _fc._zone(place.get("tz") if place else None)
    h, mi = time if time else (12, 0)
    jd = _fc._jd(datetime(date[0], date[1], date[2], h, mi, tzinfo=tz))
    return int(_fc._lon(jd, swe.JUPITER) // 30.0) % 12


def chart_from_birth(date: Tuple[int, int, int], time: Optional[Tuple[int, int]],
                     place: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """fates_astro_chart.chart_from_birth plus jupiterRashi, which the Manglik cancellations need."""
    chart = _fc.chart_from_birth(date, time, place)
    try:
        chart["jupiterRashi"] = _jupiter_rashi(date, time, place)
    except Exception:   # the chart still works; only the Jupiter cancellation is skipped
        pass
    return chart


def chart_for_profile(profile: Dict[str, Any], resolve_place) -> Optional[Dict[str, Any]]:
    """fates_astro_chart.chart_for_profile, with Jupiter (see chart_from_birth)."""
    date = _parse_date(profile.get("birthDate"))
    if not date:
        return None
    place = None
    try:
        place = resolve_place(profile)
    except Exception:
        place = None
    return chart_from_birth(date, _parse_time(profile.get("birthTime")), place)
