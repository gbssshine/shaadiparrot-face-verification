"""South Indian Dasa Porutham: the ten poruthams Tamil and Kerala families check before a marriage.

Pure: no I/O. Everything is read from the bride's and the groom's Moon (sign 0..11 and nakshatra 0..26), and
from the Moon options a day allows when a birth time is missing. The output never carries a longitude, a pada,
the options themselves or any birth detail, so it is safe for a rishta member's side of a match.

Counts follow the almanacs: c = the groom's star counted from the bride's (her star is 1), r = the groom's
Moon sign counted from hers. Traditions differ on a few cells (see tests/porutham_fixtures.json): the ones
here are fixed by the 1.0.8 design, and Rajju, the one families treat as a must, sends people to their
family astrologer rather than giving a verdict on its own.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Sequence, Tuple

from fates_astro import GRAHA_MAITRI, NAKSHATRAS, PLANETS, RASHIS, YONI, YONI_ORDER, gender_roles

KEYS = ["dina", "gana", "mahendra", "stree_deergha", "yoni", "rasi", "rasi_adhipathi", "vasya", "rajju", "vedha"]
NAMES = {
    "dina": "Dina", "gana": "Gana", "mahendra": "Mahendra", "stree_deergha": "Stree Deergha", "yoni": "Yoni",
    "rasi": "Rasi", "rasi_adhipathi": "Rasi Adhipathi", "vasya": "Vasya", "rajju": "Rajju", "vedha": "Vedha",
}
ABOUT = {
    "dina": "Health and day-to-day harmony",
    "gana": "Temperament",
    "mahendra": "Children and the family's growth",
    "stree_deergha": "A long married life",
    "yoni": "Physical closeness",
    "rasi": "Family harmony",
    "rasi_adhipathi": "How they think together",
    "vasya": "Mutual attraction",
    "rajju": "The most-checked porutham",
    "vedha": "Stars that block each other",
}

GOOD, FAIR, NO = "good", "fair", "no"

DINA_GOOD = {2, 4, 6, 8, 9, 11, 13, 15, 18, 20, 24, 26}
SAME_STAR_GOOD = {3, 5, 9, 12, 15, 20, 21, 26}     # Rohini, Ardra, Magha, Hasta, Vishakha, U. Ashadha, Shravana, Revati
SAME_STAR_FAIR = {0, 2, 4, 6, 7, 11, 13, 19}       # Ashwini, Krittika, Mrigashira, Punarvasu, Pushya, U. Phalguni, Chitra, P. Ashadha
MAHENDRA_GOOD = {4, 7, 10, 13, 16, 19, 22, 25}
RASI_GOOD = {1, 7, 9, 10, 11}
RASI_SEVENTH_NO = (frozenset({3, 9}), frozenset({4, 10}))   # Karka-Makara and Simha-Kumbha face each other
# The bride's sign -> the signs that are Vasya to it.
VASYA = [[4, 7], [3, 6], [5], [7, 8], [6], [2, 11], [5, 9], [3], [11], [0, 10], [0], [9]]
RAJJU_BY_POS = ["Pada", "Kati", "Nabhi", "Kantha", "Siro", "Kantha", "Nabhi", "Kati", "Pada"]
RAJJU_WORD = {"Pada": "feet", "Kati": "waist", "Nabhi": "navel", "Kantha": "neck", "Siro": "head"}
VEDHA = frozenset(frozenset(p) for p in (
    (0, 17), (1, 16), (2, 15), (3, 14), (4, 13), (4, 22), (13, 22), (5, 21), (6, 20), (7, 19), (8, 18),
    (9, 26), (10, 25), (11, 24), (12, 23)))


def _ord(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _star(n: int) -> str:
    return NAKSHATRAS[n][0]


def _sign(r: int) -> str:
    return RASHIS[r][0]


def _lord(r: int) -> str:
    return RASHIS[r][2]


def rajju(nak: int) -> str:
    return RAJJU_BY_POS[nak % 9]


def rajju_label(nak: int) -> str:
    r = rajju(nak)
    return f"{r} ({RAJJU_WORD[r]})"


def star_count(boy_nak: int, girl_nak: int) -> int:
    """The groom's star counted from the bride's, hers being 1."""
    return (boy_nak - girl_nak) % 27 + 1


def sign_count(boy_rashi: int, girl_rashi: int) -> int:
    """The groom's Moon sign counted from the bride's, hers being 1."""
    return (boy_rashi - girl_rashi) % 12 + 1


# ---------------------------------------------------------------- the ten
def _dina(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    c = star_count(boy[1], girl[1])
    if c == 1:
        s = _star(girl[1])
        if girl[1] in SAME_STAR_GOOD:
            return GOOD, f"Both share {s}, one of the stars where a shared birth star is welcome."
        if girl[1] in SAME_STAR_FAIR:
            return FAIR, f"Both share {s}: almanacs accept a shared {s} with a little care."
        return NO, f"Both share {s}: almanacs don't count a shared {s} as Dina. Families weigh it with the rest."
    if c in DINA_GOOD:
        return GOOD, f"The groom's star is the {_ord(c)} from the bride's: a good Dina count for everyday harmony."
    return NO, f"The groom's star is the {_ord(c)} from the bride's: Dina doesn't agree, a small point families weigh with the rest."


def _gana(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    gb, gg = NAKSHATRAS[boy[1]][1], NAKSHATRAS[girl[1]][1]
    if gb == gg:
        return GOOD, f"The same nature, {gb}, for both: at ease with each other."
    if {gb, gg} == {"Deva", "Manushya"}:
        return FAIR, f"{gb} for the groom and {gg} for the bride: gentle natures that get along."
    return NO, f"{gb} for the groom and {gg} for the bride: different temperaments that need room and patience."


def _mahendra(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    c = star_count(boy[1], girl[1])
    if c in MAHENDRA_GOOD:
        return GOOD, f"The groom's star is the {_ord(c)} from the bride's: Mahendra agrees, a good sign for the family's growth."
    return NO, f"The groom's star is the {_ord(c)} from the bride's: Mahendra doesn't agree. Families count it as a small point."


def _stree_deergha(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    c = star_count(boy[1], girl[1])
    if c > 13:
        return GOOD, f"The groom's star is the {_ord(c)} from the bride's, beyond the 13th: Stree Deergha agrees."
    if c >= 10:
        return FAIR, (f"The groom's star is the {_ord(c)} from the bride's: Kerala practice accepts beyond the 9th, "
                      "Tamil practice asks for beyond the 13th.")
    return NO, f"The groom's star is the {_ord(c)} from the bride's: Stree Deergha asks for more distance between the stars."


def _yoni(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    yb, yg = NAKSHATRAS[boy[1]][3], NAKSHATRAS[girl[1]][3]
    if yb == yg:
        return GOOD, f"The same Yoni, {yb}: physical ease comes naturally."
    if YONI[YONI_ORDER.index(yb)][YONI_ORDER.index(yg)] == 0:
        return NO, f"{yb} and {yg} are opposite Yonis: patience with each other's rhythm helps."
    return FAIR, f"{yb} for the groom and {yg} for the bride: closeness grows with time."


def _rasi(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    r = sign_count(boy[0], girl[0])
    if r == 1:
        return GOOD, f"Both Moons are in {_sign(girl[0])}: Rasi agrees."
    if r == 7 and frozenset({boy[0], girl[0]}) in RASI_SEVENTH_NO:
        return NO, f"{_sign(girl[0])} and {_sign(boy[0])} face each other, a pair the Rasi porutham doesn't accept."
    if r in RASI_GOOD:
        return GOOD, f"The groom's Moon sign is the {_ord(r)} from the bride's: Rasi agrees, good for family harmony."
    return NO, f"The groom's Moon sign is the {_ord(r)} from the bride's: Rasi doesn't agree. Families read it with the other nine."


def _rasi_adhipathi(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    lb, lg = _lord(boy[0]), _lord(girl[0])
    m = GRAHA_MAITRI[PLANETS.index(lb)][PLANETS.index(lg)]
    if lb == lg:
        return GOOD, f"Both Moon signs are ruled by {lb}: they think alike."
    if m >= 4:
        return GOOD, f"{lb} and {lg}, the lords of the two Moon signs, are friends: they think alike."
    if m == 3:
        return FAIR, f"{lb} and {lg}, the lords of the two Moon signs, are neutral: each learns the other's way of thinking."
    return NO, f"{lb} and {lg}, the lords of the two Moon signs, don't get on: talking things through matters more."


def _vasya(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    if boy[0] in VASYA[girl[0]]:
        return GOOD, f"The groom's {_sign(boy[0])} is a Vasya sign of the bride's {_sign(girl[0])}: a natural pull between the two."
    return NO, f"The groom's {_sign(boy[0])} isn't a Vasya sign of the bride's {_sign(girl[0])}: attraction grows with time together."


def _rajju(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    rb, rg = rajju(boy[1]), rajju(girl[1])
    if rb == rg:
        return NO, (f"Same Rajju ({rb}). Most South Indian families treat Rajju as a must and talk it over "
                    "with their family astrologer.")
    return GOOD, f"Different Rajju ({rb} and {rg}): the most-checked porutham is clear."


def _vedha(boy: Tuple[int, int], girl: Tuple[int, int]) -> Tuple[str, str]:
    a, b = _star(boy[1]), _star(girl[1])
    if frozenset({boy[1], girl[1]}) in VEDHA:
        return NO, f"{a} and {b} are a Vedha pair. Families read this together with Rajju."
    if boy[1] == girl[1]:
        return GOOD, f"A shared star, {a}, is not a Vedha pair."
    return GOOD, f"{a} and {b} are not a Vedha pair: no block between the stars."


_RULES = {
    "dina": _dina, "gana": _gana, "mahendra": _mahendra, "stree_deergha": _stree_deergha, "yoni": _yoni,
    "rasi": _rasi, "rasi_adhipathi": _rasi_adhipathi, "vasya": _vasya, "rajju": _rajju, "vedha": _vedha,
}


def items(boy: Dict[str, Any], girl: Dict[str, Any]) -> List[Dict[str, str]]:
    """The ten poruthams in the fixed order. boy / girl: {"rashi": 0..11, "nak": 0..26}."""
    b, g = (int(boy["rashi"]), int(boy["nak"])), (int(girl["rashi"]), int(girl["nak"]))
    out = []
    for key in KEYS:
        level, meaning = _RULES[key](b, g)
        out.append({"key": key, "name": NAMES[key], "about": ABOUT[key], "level": level, "meaning": meaning})
    return out


def _levels(boy: Tuple[int, int], girl: Tuple[int, int]) -> Dict[str, str]:
    return {k: _RULES[k](boy, girl)[0] for k in KEYS}


def count_of(levels: Dict[str, str]) -> int:
    """How many of the ten agree. 'fair' (partly) counts as agreeing."""
    return sum(1 for k in KEYS if levels[k] != NO)


def verdict(count: int, rajju_ok: bool, vedha_ok: bool) -> Dict[str, str]:
    """Three count bands and one Rajju override. Every Vedha pair also shares a Rajju, so Vedha never stands alone."""
    if not rajju_ok:
        return {"word": "Rajju and Vedha don't match" if not vedha_ok else "Rajju doesn't match", "tone": "low",
                "text": "Most South Indian families treat Rajju as a must, whatever the count. "
                        "Talk it over with the family astrologer."}
    if count >= 8:
        return {"word": "Excellent match", "tone": "great",
                "text": "Eight or more of the ten poruthams agree, Rajju included."}
    if count >= 6:
        return {"word": "Good match", "tone": "good", "text": "Most of the poruthams agree and Rajju is clear."}
    if count == 5:
        return {"word": "Fair match", "tone": "ok",
                "text": "Half of the poruthams agree and Rajju is clear: many families go ahead with this."}
    return {"word": "Needs a closer look", "tone": "low",
            "text": "Fewer than half agree. Families usually read the full charts with their astrologer before deciding."}


def _options(chart: Dict[str, Any]) -> List[Tuple[int, int]]:
    main = (int(chart["rashi"]), int(chart["nak"]))
    opts: List[Tuple[int, int]] = []
    for o in (chart.get("options") or []):
        try:
            pair = (int(o[0]), int(o[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= pair[0] < 12 and 0 <= pair[1] < 27 and pair not in opts:
            opts.append(pair)
    if main not in opts:
        opts.insert(0, main)
    return opts


def dasa_porutham(a_chart: Dict[str, Any], b_chart: Dict[str, Any], a_gender: str, b_gender: str) -> Dict[str, Any]:
    """The ten poruthams of a and b. The bride is the female side (fates_astro.gender_roles, the same rule as the
    36 gunas); with the same or unknown genders the first person counts as the groom. Without a birth time
    the count becomes a range over every Moon the day allows, and mustSure says whether Rajju and Vedha could
    change."""
    a_is_boy = gender_roles(a_gender, b_gender)[0]
    a_main = (int(a_chart["rashi"]), int(a_chart["nak"]))
    b_main = (int(b_chart["rashi"]), int(b_chart["nak"]))
    boy, girl = (a_chart, b_chart) if a_is_boy else (b_chart, a_chart)
    its = items(boy, girl)
    count = sum(1 for i in its if i["level"] != NO)
    rajju_ok = its[KEYS.index("rajju")]["level"] != NO
    vedha_ok = its[KEYS.index("vedha")]["level"] != NO

    counts: List[int] = []
    musts = set()
    for oa, ob in product(_options(a_chart), _options(b_chart)):
        ob_, og_ = (oa, ob) if a_is_boy else (ob, oa)
        lv = _levels(ob_, og_)
        counts.append(count_of(lv))
        musts.add((lv["rajju"] != NO, lv["vedha"] != NO))
    musts.add((rajju_ok, vedha_ok))
    return {
        "count": count, "countMin": min(counts + [count]), "countMax": max(counts + [count]), "items": its,
        "rajjuOk": rajju_ok, "vedhaOk": vedha_ok, "mustSure": len(musts) == 1,
        "aRajju": rajju_label(a_main[1]), "bRajju": rajju_label(b_main[1]),
        "verdict": verdict(count, rajju_ok, vedha_ok),
    }
