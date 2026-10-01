"""Ashtakoota Guna Milan (North Indian, 36 points, Moon-based, sidereal Lahiri) and doshas.

Tables were cross-checked in Sept 2026 against AstroSage and Prokerala software output and several
independent published tables (see docs in tests/ashtakoota_fixtures.json). Every matrix here is
[boy][girl]. Raw koota points are kept as the software shows them; dosha cancellations are reported
separately and never change the 36-point total.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from fates_astro_chart import chart_for_profile, chart_from_birth, house_from  # noqa: F401  (re-exported)

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


def manglik_status(chart: Dict[str, Any]) -> Dict[str, Any]:
    mars = chart.get("marsRashi")
    if mars is None:
        return {"manglik": False, "mild": False, "from": []}
    refs = []
    for label, ref in (("Lagna", chart.get("lagna")), ("Moon", chart.get("rashi")), ("Venus", chart.get("venusRashi"))):
        h = house_from(mars, ref)
        if h in MANGLIK_HOUSES:
            refs.append((label, h))
    if not refs:
        return {"manglik": False, "mild": False, "from": []}
    # Mars in its own sign (Mesha, Vrischika) or exalted (Makara) is widely held to cancel it.
    if mars in (0, 7, 9):
        return {"manglik": False, "mild": False, "from": [r[0] for r in refs], "cancelledBy": "Mars in its own or exalted sign"}
    mild = all(h == 2 for _, h in refs)
    return {"manglik": True, "mild": mild, "from": [r[0] for r in refs]}


def manglik_dosha(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ma, mb = manglik_status(a), manglik_status(b)
    strong_a, strong_b = ma["manglik"] and not ma["mild"], mb["manglik"] and not mb["mild"]
    lagna_note = "" if a.get("lagna") is not None and b.get("lagna") is not None else " (from Moon and Venus; a birth time adds the Lagna check)"
    if not ma["manglik"] and not mb["manglik"]:
        return {"name": "Manglik", "present": False, "cancelled": False, "why": "Neither of you is Manglik" + lagna_note + "."}
    if ma["manglik"] and mb["manglik"]:
        return {"name": "Manglik", "present": True, "cancelled": True, "why": "You are both Manglik, which cancels it out."}
    if not (strong_a or strong_b):
        return {"name": "Manglik", "present": True, "cancelled": True, "why": "A mild Manglik placement (2nd house only) on one side."}
    who = "You are" if strong_a else "They are"
    return {"name": "Manglik", "present": True, "cancelled": False, "why": f"{who} Manglik and the other isn’t{lagna_note}. Many families ask a pandit about this."}


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


def match_charts(a: Dict[str, Any], b: Dict[str, Any], a_gender: str, b_gender: str) -> Dict[str, Any]:
    """a = the viewer. Roles follow gender; same or unknown genders average both directions."""
    if a_gender == "male" and b_gender != "male":
        orientations = [True]
    elif b_gender == "male" and a_gender != "male":
        orientations = [False]
    elif a_gender == "female" and b_gender not in ("female",):
        orientations = [False]
    else:
        orientations = [True, False]

    ma, mb = _moon(a), _moon(b)

    def score(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, float]:
        runs = [koota_points(x, y) if o else koota_points(y, x) for o in orientations]
        return {k: sum(r[k] for r in runs) / len(runs) for k in KOOTA_MAX}

    pts = score(ma, mb)
    total = sum(pts.values())

    # Range over every Moon position the day allows when a birth time is missing.
    opts_a = [(o[0], o[1]) for o in a.get("options") or [[a["rashi"], a["nak"]]]]
    opts_b = [(o[0], o[1]) for o in b.get("options") or [[b["rashi"], b["nak"]]]]
    totals = [sum(score(_moon(a, ra, na), _moon(b, rb, nb)).values()) for (ra, na), (rb, nb) in product(opts_a, opts_b)]

    a_is_boy = orientations[0]
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
