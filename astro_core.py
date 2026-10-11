"""Panchang, birth charts, Vimshottari dashas, family day lines and good dates (muhurat guide).

Pure computation on Swiss Ephemeris, sidereal Lahiri, no I/O: astro_api.py serves it over HTTP.
Times are local to the place asked for. Panchang rules follow the common North Indian practice:
the day's tithi, nakshatra and yoga are the ones running at sunrise; Rahu kaal, Yamaganda and
Gulika are eighths of the daylight; Abhijit is the 8th of the 15 daytime muhurtas.
"""
from __future__ import annotations

from datetime import date as Date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import swisseph as swe

from fates_astro import NAKSHATRAS, RASHIS, manglik_status

swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

NAK_SPAN = 360.0 / 27.0
FLAGS = swe.FLG_SWIEPH | swe.FLG_SIDEREAL

TITHIS = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami",
          "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi"]
YOGAS = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarma", "Dhriti", "Shula",
         "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan",
         "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
KARANAS_MOVING = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]
VARAS = ["Somvar", "Mangalvar", "Budhvar", "Guruvar", "Shukravar", "Shanivar", "Ravivar"]   # Python weekday order
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Amanta lunar month by the Sun's sidereal sign at the new moon that starts it.
MASA_BY_SUN_SIGN = ["Vaishakha", "Jyeshtha", "Ashadha", "Shravana", "Bhadrapada", "Ashwin", "Kartika",
                    "Margashirsha", "Pausha", "Magha", "Phalguna", "Chaitra"]

# Eighths of daylight (1-based), Monday first.
RAHU_PART = [2, 7, 5, 6, 4, 3, 8]
YAMA_PART = [4, 3, 2, 1, 7, 6, 5]
GULIKA_PART = [6, 5, 4, 3, 2, 1, 7]
CHOGHADIYA = ["Udveg", "Char", "Labh", "Amrit", "Kaal", "Shubh", "Rog"]
CHOGHADIYA_START = [3, 6, 2, 5, 1, 4, 0]   # Monday starts with Amrit, Sunday with Udveg
GOOD_CHOGHADIYA = {"Amrit", "Shubh", "Labh"}

PLANETS = [("Su", "Sun", swe.SUN), ("Mo", "Moon", swe.MOON), ("Ma", "Mars", swe.MARS),
           ("Me", "Mercury", swe.MERCURY), ("Ju", "Jupiter", swe.JUPITER), ("Ve", "Venus", swe.VENUS),
           ("Sa", "Saturn", swe.SATURN), ("Ra", "Rahu", swe.MEAN_NODE)]
DASHA = [("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10), ("Mars", 7), ("Rahu", 18), ("Jupiter", 16),
         ("Saturn", 19), ("Mercury", 17)]
YEAR_DAYS = 365.25


# ------------------------------------------------------------------ basics
def zone(name: Any) -> ZoneInfo:
    try:
        return ZoneInfo(str(name or "").strip() or "Asia/Kolkata")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("Asia/Kolkata")


def jd_of(moment: datetime) -> float:
    u = moment.astimezone(timezone.utc)
    return swe.julday(u.year, u.month, u.day, u.hour + u.minute / 60.0 + u.second / 3600.0)


def moment_of(jd: float, tz: ZoneInfo) -> datetime:
    y, m, d, h = swe.revjul(jd)
    secs = int(round(h * 3600))
    return (datetime(y, m, d, tzinfo=timezone.utc) + timedelta(seconds=secs)).astimezone(tz)


def sid(jd: float, body: int) -> float:
    res, _ = swe.calc_ut(jd, body, FLAGS)
    return float(res[0]) % 360.0


def sid_speed(jd: float, body: int) -> Tuple[float, float]:
    res, _ = swe.calc_ut(jd, body, FLAGS | swe.FLG_SPEED)
    return float(res[0]) % 360.0, float(res[3])


def tithi_angle(jd: float) -> float:
    return (sid(jd, swe.MOON) - sid(jd, swe.SUN)) % 360.0


def yoga_angle(jd: float) -> float:
    return (sid(jd, swe.MOON) + sid(jd, swe.SUN)) % 360.0


def moon_lon(jd: float) -> float:
    return sid(jd, swe.MOON)


def _end_of(fn, jd0: float, span: float, max_days: float = 1.6) -> float:
    """When fn (a longitude-like angle growing over time) reaches the next multiple of span after jd0."""
    a0 = fn(jd0)
    need = span - (a0 % span)
    lo, hi = jd0, jd0 + max_days
    for _ in range(32):
        mid = (lo + hi) / 2
        if (fn(mid) - a0) % 360.0 >= need:
            hi = mid
        else:
            lo = mid
    return hi


def _rise_set(jd_start: float, body: int, lat: float, lon: float, rise: bool) -> Optional[float]:
    flag = swe.CALC_RISE if rise else swe.CALC_SET
    try:
        res, tret = swe.rise_trans(jd_start, body, flag, (lon, lat, 0.0))
    except Exception:
        return None
    return float(tret[0]) if res == 0 else None


def tithi_name(n: int) -> Tuple[str, str]:
    """n = 1..30 -> (name, paksha)."""
    if n == 15:
        return "Purnima", "Shukla"
    if n == 30:
        return "Amavasya", "Krishna"
    return TITHIS[(n - 1) % 15], ("Shukla" if n <= 15 else "Krishna")


def karana_name(k: int) -> str:
    """k = 1..60, the half-tithi of the lunar month."""
    if k == 1:
        return "Kimstughna"
    if k >= 58:
        return ["Shakuni", "Chatushpada", "Naga"][k - 58]
    return KARANAS_MOVING[(k - 2) % 7]


def hhmm(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%H:%M") if dt else None


def _span(a: datetime, b: datetime) -> Dict[str, str]:
    return {"start": a.strftime("%H:%M"), "end": b.strftime("%H:%M"), "startIso": a.isoformat(), "endIso": b.isoformat()}


# ------------------------------------------------------------------ panchang
@lru_cache(maxsize=4096)
def _day_frame(y: int, m: int, d: int, lat: float, lon: float, tz_name: str) -> Dict[str, Any]:
    tz = zone(tz_name)
    midnight = datetime(y, m, d, tzinfo=tz)
    jd0 = jd_of(midnight)
    sunrise = _rise_set(jd0, swe.SUN, lat, lon, True)
    sunset = _rise_set(sunrise or jd0, swe.SUN, lat, lon, False)
    if sunrise is None or sunset is None:   # polar day / night: fall back to 06:00-18:00
        sunrise, sunset = jd_of(midnight + timedelta(hours=6)), jd_of(midnight + timedelta(hours=18))
    next_rise = _rise_set(sunset, swe.SUN, lat, lon, True) or (sunrise + 1.0)
    moonrise = _rise_set(jd0, swe.MOON, lat, lon, True)
    if moonrise is not None and moonrise - jd0 > 1.0:
        moonrise = None   # no moonrise on this calendar day
    return {"jd0": jd0, "sunrise": sunrise, "sunset": sunset, "nextRise": next_rise, "moonrise": moonrise}


def tithi_at(jd: float) -> int:
    return int(tithi_angle(jd) // 12.0) + 1


def panchang(day: Date, lat: float, lon: float, tz_name: str) -> Dict[str, Any]:
    tz = zone(tz_name)
    f = _day_frame(day.year, day.month, day.day, round(lat, 3), round(lon, 3), tz.key)
    rise, set_, nxt = f["sunrise"], f["sunset"], f["nextRise"]
    t_rise, t_set = moment_of(rise, tz), moment_of(set_, tz)

    ta = tithi_angle(rise)
    tn = int(ta // 12.0) + 1
    t_name, paksha = tithi_name(tn)
    t_end = moment_of(_end_of(tithi_angle, rise, 12.0), tz)
    nxt_name, _ = tithi_name(tn % 30 + 1)

    ml = moon_lon(rise)
    nak = int(ml // NAK_SPAN) % 27
    n_end = moment_of(_end_of(moon_lon, rise, NAK_SPAN), tz)
    ya = yoga_angle(rise)
    yoga = int(ya // NAK_SPAN) % 27
    y_end = moment_of(_end_of(yoga_angle, rise, NAK_SPAN), tz)
    karana = int(ta // 6.0) + 1

    wd = day.weekday()
    daylight = (set_ - rise)
    part = daylight / 8.0

    def eighth(n: int) -> Dict[str, str]:
        return _span(moment_of(rise + part * (n - 1), tz), moment_of(rise + part * n, tz))

    abhijit = _span(moment_of(rise + daylight * 7 / 15, tz), moment_of(rise + daylight * 8 / 15, tz))
    brahma = _span(moment_of(rise - 96 / 1440.0, tz), moment_of(rise - 48 / 1440.0, tz))

    chog = []
    for i in range(8):
        name = CHOGHADIYA[(CHOGHADIYA_START[wd] + i) % 7]
        chog.append({"name": name, "good": name in GOOD_CHOGHADIYA, **eighth(i + 1)})
    # Choghadiya cycles Udveg, Char, Labh, Amrit, Kaal, Shubh, Rog; the 8th repeats the 1st.
    chog[7]["name"] = chog[0]["name"]
    chog[7]["good"] = chog[0]["good"]

    rahu = eighth(RAHU_PART[wd])
    good_times = []
    if wd != 2:   # Abhijit is skipped on Wednesdays
        good_times.append({"name": "Abhijit muhurat", **abhijit})
    for c in chog:
        if c["good"] and c["start"] != rahu["start"]:
            good_times.append({"name": c["name"] + " choghadiya", "start": c["start"], "end": c["end"],
                               "startIso": c["startIso"], "endIso": c["endIso"]})
    good_times.sort(key=lambda g: g["startIso"])

    sun_sign = int(sid(rise, swe.SUN) // 30.0)
    moon_sign = int(ml // 30.0)
    return {
        "date": day.isoformat(), "weekday": WEEKDAYS[wd], "vara": VARAS[wd],
        "sunrise": hhmm(t_rise), "sunset": hhmm(t_set),
        "moonrise": hhmm(moment_of(f["moonrise"], tz)) if f["moonrise"] else None,
        "tithi": {"number": tn, "name": t_name, "paksha": paksha, "endsAt": hhmm(t_end), "endsAtIso": t_end.isoformat(),
                  "endsNextDay": t_end.date() > day, "next": nxt_name},
        "nakshatra": {"index": nak, "name": NAKSHATRAS[nak][0], "endsAt": hhmm(n_end), "endsAtIso": n_end.isoformat(),
                      "endsNextDay": n_end.date() > day, "lord": DASHA[nak % 9][0],
                      "next": NAKSHATRAS[(nak + 1) % 27][0]},
        "yoga": {"name": YOGAS[yoga], "endsAt": hhmm(y_end), "endsNextDay": y_end.date() > day},
        "karana": {"name": karana_name(karana)},
        "moonSign": {"index": moon_sign, "name": RASHIS[moon_sign][0], "english": RASHIS[moon_sign][1]},
        "sunSign": {"index": sun_sign, "name": RASHIS[sun_sign][0], "english": RASHIS[sun_sign][1]},
        "masa": lunar_month(rise),
        "rahuKaal": rahu, "yamaganda": eighth(YAMA_PART[wd]), "gulika": eighth(GULIKA_PART[wd]),
        "abhijit": abhijit, "abhijitSkipped": wd == 2, "brahmaMuhurta": brahma,
        "choghadiya": chog, "goodTimes": good_times,
        "dayLengthMinutes": int(round(daylight * 1440)),
        "_jd": {"sunrise": rise, "sunset": set_, "nextRise": nxt, "moonrise": f["moonrise"]},
    }


def _prev_new_moon(jd: float) -> float:
    """The last new moon (tithi angle 0) before jd."""
    a = tithi_angle(jd)
    guess = jd - a / 12.19   # ~12.19 deg a day
    lo, hi = guess - 2.0, min(guess + 2.0, jd)
    for _ in range(40):
        mid = (lo + hi) / 2
        # past the new moon when the angle is small (just after 0), before it when near 360
        if tithi_angle(mid) < 180.0:
            hi = mid
        else:
            lo = mid
    return hi


def lunar_month(jd: float) -> Dict[str, Any]:
    nm = _prev_new_moon(jd)
    sign = int(sid(nm, swe.SUN) // 30.0)
    nm_next = _prev_new_moon(nm + 31.0)
    adhik = int(sid(nm_next, swe.SUN) // 30.0) == sign
    amanta = MASA_BY_SUN_SIGN[sign]
    paksha_krishna = tithi_angle(jd) >= 180.0
    # Purnimanta (North India) months run full moon to full moon: the dark half belongs to the next month.
    purnimanta = MASA_BY_SUN_SIGN[(sign + 1) % 12] if paksha_krishna else amanta
    return {"amanta": amanta, "purnimanta": purnimanta, "adhik": adhik}


# ------------------------------------------------------------------ vrats and festivals
def vrats_between(start: Date, days: int, lat: float, lon: float, tz_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        tz = zone(tz_name)
        f = _day_frame(d.year, d.month, d.day, round(lat, 3), round(lon, 3), tz.key)
        t_rise = tithi_at(f["sunrise"])
        t_set = tithi_at(f["sunset"])
        t_moon = tithi_at(f["moonrise"]) if f["moonrise"] else None
        wd = d.weekday()
        items = []
        if t_rise in (11, 26):
            items.append(("ekadashi", "Ekadashi vrat", "A day of fasting and Vishnu puja."))
        if t_set in (13, 28):
            nm = {0: "Som Pradosh", 5: "Shani Pradosh"}.get(wd, "Pradosh vrat")
            items.append(("pradosh", nm, "Shiva puja at dusk, after sunset."))
        if t_set == 15 or (t_rise == 15 and t_set == 16):
            items.append(("purnima", "Purnima", "Full moon: Satyanarayan katha in many homes."))
        if t_rise == 30:
            items.append(("amavasya", "Amavasya", "New moon: a day to remember the ancestors."))
        if t_moon == 19:
            mr = moment_of(f["moonrise"], tz)
            items.append(("sankashti", "Sankashti Chaturthi", f"Ganesha vrat, broken at moonrise ({mr.strftime('%H:%M')})."))
        for kind, name, note in items:
            # one entry per tithi: the first day it falls on
            if any(o["kind"] == kind and (d - Date.fromisoformat(o["date"])).days <= 1 for o in out):
                continue
            out.append({"date": d.isoformat(), "kind": kind, "type": "vrat", "name": name, "note": note})
    return out


FESTIVALS: List[Dict[str, str]] = [
    {"date": "2026-10-11", "id": "navratri", "name": "Sharad Navratri begins", "note": "Nine nights of Durga: ghatasthapana in the morning."},
    {"date": "2026-10-17", "id": "durga_puja", "name": "Durga Puja (Saptami)", "note": "Pandal days begin in Bengal and beyond."},
    {"date": "2026-10-20", "id": "dussehra", "name": "Dussehra", "note": "Vijayadashami: good wins over evil."},
    {"date": "2026-10-25", "id": "sharad_purnima", "name": "Sharad Purnima", "note": "Kheer under the full moon."},
    {"date": "2026-10-29", "id": "karva_chauth", "name": "Karva Chauth", "note": "The fast ends at moonrise: see the time for your city."},
    {"date": "2026-11-06", "id": "dhanteras", "name": "Dhanteras", "note": "Buy something new for the home, light the first diya."},
    {"date": "2026-11-08", "id": "diwali", "name": "Diwali", "note": "Lakshmi puja in the evening."},
    {"date": "2026-11-11", "id": "bhai_dooj", "name": "Bhai Dooj", "note": "Tilak for brothers, blessings for sisters."},
    {"date": "2026-11-15", "id": "chhath", "name": "Chhath Puja", "note": "Evening arghya to the setting Sun."},
    {"date": "2026-11-20", "id": "dev_uthani", "name": "Dev Uthani Ekadashi", "note": "Tulsi Vivah: the wedding season opens."},
    {"date": "2026-11-24", "id": "gurpurab", "name": "Guru Nanak Jayanti", "note": "Kartik Purnima."},
    {"date": "2026-12-25", "id": "christmas", "name": "Christmas", "note": ""},
    {"date": "2027-01-01", "id": "new_year", "name": "New Year 2027", "note": ""},
    {"date": "2027-01-14", "id": "lohri", "name": "Lohri", "note": "Bonfire, rewri and gajak."},
    {"date": "2027-01-15", "id": "sankranti", "name": "Makar Sankranti · Pongal", "note": "The Sun enters Makara: til-gud and kites."},
    {"date": "2027-01-26", "id": "republic_day", "name": "Republic Day", "note": ""},
    {"date": "2027-02-11", "id": "basant_panchami", "name": "Basant Panchami", "note": "Saraswati puja: wear yellow."},
    {"date": "2027-03-06", "id": "shivratri", "name": "Maha Shivratri", "note": "Night of Shiva."},
    {"date": "2027-03-10", "id": "eid_ul_fitr", "name": "Eid ul-Fitr", "note": "Depends on the moon sighting (10 or 11 March)."},
    {"date": "2027-03-21", "id": "holika_dahan", "name": "Holika Dahan", "note": "The bonfire on the eve of Holi."},
    {"date": "2027-03-22", "id": "holi", "name": "Holi", "note": "Colours day (some lists say 23 March)."},
    {"date": "2027-04-07", "id": "ugadi", "name": "Ugadi · Gudi Padwa", "note": "Hindu new year."},
    {"date": "2027-04-14", "id": "baisakhi", "name": "Baisakhi", "note": "Harvest festival."},
]


# Festivals that already are the day's vrat: the generic entry within a day of them is left out.
FESTIVAL_COVERS = {"dev_uthani": {"ekadashi"}, "sharad_purnima": {"purnima"}, "gurpurab": {"purnima"},
                   "diwali": {"amavasya"}, "karva_chauth": {"sankashti"}, "holika_dahan": {"purnima"},
                   "shivratri": {"pradosh"}}


def upcoming(start: Date, days: int, lat: float, lon: float, tz_name: str, limit: int = 8) -> List[Dict[str, Any]]:
    end = start + timedelta(days=days)
    fest = [{**f, "kind": f["id"], "type": "festival"} for f in FESTIVALS
            if start <= Date.fromisoformat(f["date"]) < end]
    covered = [(Date.fromisoformat(f["date"]), FESTIVAL_COVERS.get(f["id"], set())) for f in fest]
    vrats = [v for v in vrats_between(start, days, lat, lon, tz_name)
             if not any(v["kind"] in kinds and abs((Date.fromisoformat(v["date"]) - d).days) <= 1 for d, kinds in covered)]
    items = fest + vrats
    items.sort(key=lambda i: (i["date"], i["type"] != "festival"))
    for i in items:
        i["inDays"] = (Date.fromisoformat(i["date"]) - start).days
    return items[:limit]


# ------------------------------------------------------------------ birth chart and dashas
def birth_moment(date: Tuple[int, int, int], time: Optional[Tuple[int, int]], place: Optional[Dict[str, Any]]) -> datetime:
    tz = zone(place.get("tz") if place else None)
    h, mi = time if time else (12, 0)
    return datetime(date[0], date[1], date[2], h, mi, tzinfo=tz)


def dashas(moon: float, born: datetime, now: datetime) -> Dict[str, Any]:
    nak = int(moon // NAK_SPAN) % 27
    frac = (moon % NAK_SPAN) / NAK_SPAN
    idx = nak % 9
    start = born - timedelta(days=frac * DASHA[idx][1] * YEAR_DAYS)
    periods = []
    t = start
    i = idx
    while t < born + timedelta(days=121 * YEAR_DAYS) and len(periods) < 12:
        lord, years = DASHA[i % 9]
        end = t + timedelta(days=years * YEAR_DAYS)
        if end > born:
            periods.append({"lord": lord, "years": years, "start": max(t, born).date().isoformat(), "end": end.date().isoformat(),
                            "_s": t, "_e": end})
        t = end
        i += 1
    cur = next((p for p in periods if p["_s"] <= now < p["_e"]), None)
    antar = None
    antars: List[Dict[str, Any]] = []
    if cur:
        li = [d[0] for d in DASHA].index(cur["lord"])
        t = cur["_s"]
        for k in range(9):
            lord, years = DASHA[(li + k) % 9]
            length = cur["years"] * years / 120.0
            end = t + timedelta(days=length * YEAR_DAYS)
            a = {"lord": lord, "start": t.date().isoformat(), "end": end.date().isoformat(), "_s": t, "_e": end}
            antars.append(a)
            if t <= now < end:
                antar = a
            t = end
    clean = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}   # noqa: E731
    return {
        "periods": [clean(p) for p in periods],
        "current": clean(cur) if cur else None,
        "antar": clean(antar) if antar else None,
        "antars": [clean(a) for a in antars if a["_e"] > now][:4],
        "birthBalanceYears": round((1 - frac) * DASHA[idx][1], 2),
    }


DASHA_THEME = {
    "Ketu": "letting go, inner focus and spiritual turns",
    "Venus": "love, marriage, comfort and beauty",
    "Sun": "recognition, father figures and self-confidence",
    "Moon": "home, mother, feelings and travel",
    "Mars": "energy, property, courage and quick action",
    "Rahu": "big ambitions, new places and sudden changes",
    "Jupiter": "growth, wisdom, children and teachers",
    "Saturn": "hard work, patience and results that last",
    "Mercury": "learning, business, friends and clever talk",
}

RASHI_NATURE = [
    "bold, quick to start, a born leader", "steady, loyal, loves comfort and good food",
    "curious, chatty, two ideas at once", "caring, emotional, home comes first",
    "proud, warm, generous with family", "practical, careful, notices every detail",
    "fair, charming, wants peace at home", "deep, private, fiercely loyal",
    "honest, optimistic, loves freedom", "hard-working, patient, plans for years",
    "original, friendly, thinks for the group", "gentle, intuitive, full of empathy",
]


def birth_chart(date: Tuple[int, int, int], time: Optional[Tuple[int, int]], place: Optional[Dict[str, Any]],
                now: Optional[datetime] = None) -> Dict[str, Any]:
    born = birth_moment(date, time, place)
    jd = jd_of(born)
    planets = []
    for code, name, body in PLANETS:
        lon, speed = sid_speed(jd, body)
        planets.append({"code": code, "name": name, "lon": round(lon, 3), "rashi": int(lon // 30.0),
                        "degree": round(lon % 30.0, 2), "nak": int(lon // NAK_SPAN) % 27,
                        "retro": speed < 0 and body not in (swe.MEAN_NODE,)})
    ra = planets[-1]
    ke_lon = (ra["lon"] + 180.0) % 360.0
    planets.append({"code": "Ke", "name": "Ketu", "lon": round(ke_lon, 3), "rashi": int(ke_lon // 30.0),
                    "degree": round(ke_lon % 30.0, 2), "nak": int(ke_lon // NAK_SPAN) % 27, "retro": False})

    lagna = None
    if time and place and place.get("precision") == "city":
        _, ascmc = swe.houses_ex(jd, float(place["lat"]), float(place["lon"]), b"W", swe.FLG_SIDEREAL)
        lagna = int((float(ascmc[0]) % 360.0) // 30.0)
    moon = planets[1]
    basis = lagna if lagna is not None else moon["rashi"]
    for p in planets:
        p["house"] = (p["rashi"] - basis) % 12 + 1
        p["rashiName"] = RASHIS[p["rashi"]][0]
        p["nakName"] = NAKSHATRAS[p["nak"]][0]

    chart_like = {"marsRashi": planets[2]["rashi"], "venusRashi": planets[5]["rashi"], "rashi": moon["rashi"], "lagna": lagna}
    mang = manglik_status(chart_like)
    now = now or datetime.now(timezone.utc)
    ds = dashas(moon["lon"], born, now)
    precision = "exact" if lagna is not None else ("time" if time else "date")
    # Without a time the Moon may sit up to ~6.5 deg away: its nakshatra (and so the dasha dates) can shift.
    moon_sure = True
    if not time:
        tz = born.tzinfo
        lo = sid(jd_of(datetime(date[0], date[1], date[2], 0, 0, tzinfo=tz)), swe.MOON)
        hi = sid(jd_of(datetime(date[0], date[1], date[2], 23, 59, tzinfo=tz)), swe.MOON)
        moon_sure = int(lo // NAK_SPAN) == int(hi // NAK_SPAN)
    return {
        "precision": precision, "timeKnown": time is not None, "lagna": lagna,
        "lagnaName": RASHIS[lagna][0] if lagna is not None else None,
        "lagnaEnglish": RASHIS[lagna][1] if lagna is not None else None,
        "basis": "lagna" if lagna is not None else "moon",
        "planets": planets,
        "moon": {"rashi": moon["rashi"], "rashiName": RASHIS[moon["rashi"]][0], "rashiEnglish": RASHIS[moon["rashi"]][1],
                 "nakshatra": NAKSHATRAS[moon["nak"]][0], "nak": moon["nak"],
                 "pada": int((moon["lon"] % NAK_SPAN) // (NAK_SPAN / 4)) + 1, "sure": moon_sure},
        "sun": {"rashi": planets[0]["rashi"], "rashiName": RASHIS[planets[0]["rashi"]][0], "rashiEnglish": RASHIS[planets[0]["rashi"]][1]},
        "manglik": mang,
        "dasha": ds,
        "dashaApproximate": not time,
        "lines": chart_lines(lagna, moon, planets[0], ds, mang, time is not None),
    }


def _month_year(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%b %Y")


def chart_lines(lagna: Optional[int], moon: Dict[str, Any], sun: Dict[str, Any], ds: Dict[str, Any],
                mang: Dict[str, Any], time_known: bool) -> List[Dict[str, str]]:
    lines = []
    if lagna is not None:
        lines.append({"title": f"Rising sign: {RASHIS[lagna][0]} ({RASHIS[lagna][1]})",
                      "text": f"How the world sees you: {RASHI_NATURE[lagna]}."})
    else:
        lines.append({"title": "Rising sign: add your birth time",
                      "text": "Your Lagna changes every two hours. Add the time and city of birth to see it."})
    r = moon["rashi"]
    lines.append({"title": f"Moon sign: {RASHIS[r][0]} ({RASHIS[r][1]}) · {NAKSHATRAS[moon['nak']][0]}",
                  "text": f"Your heart and mind: {RASHI_NATURE[r]}. This is the sign Indian horoscopes use."})
    s = sun["rashi"]
    lines.append({"title": f"Sun sign: {RASHIS[s][0]} ({RASHIS[s][1]})",
                  "text": f"Your will and purpose: {RASHI_NATURE[s]}."})
    cur, antar = ds.get("current"), ds.get("antar")
    if cur:
        t = f"Until {_month_year(cur['end'])}: a time of {DASHA_THEME[cur['lord']]}."
        if antar:
            t += f" Inside it, {antar['lord']} until {_month_year(antar['end'])}: {DASHA_THEME[antar['lord']]}."
        lines.append({"title": f"Now: {cur['lord']} mahadasha", "text": t})
    if mang.get("manglik"):
        frm = " and ".join(mang.get("from") or [])
        lines.append({"title": "Mangal: Manglik" + (" (mild)" if mang.get("mild") else ""),
                      "text": f"Mars sits in a Manglik house from your {frm}. Families look at this when matching; "
                              "two Manglik charts cancel it out."})
    else:
        lines.append({"title": "Mangal: not Manglik", "text": "No Mangal dosha in your chart" +
                      ("." if time_known else " (from Moon and Venus; a birth time adds the Lagna check).")})
    return lines


# ------------------------------------------------------------------ family lines
HOUSE_GOOD = {1, 3, 6, 7, 10, 11}
HOUSE_HARD = {4, 8, 12}
TARA_HARD = {3, 5, 7}

LINES = {
    1: {"any": "{n} feels the day personally: good for self-care and fresh starts."},
    2: {"any": "Family talk and good food go well for {n} today.",
        "child": "{n} will enjoy family time and favourite food today."},
    3: {"any": "{n} has extra courage today: good for calls, short trips and trying something new.",
        "child": "{n} is brave today: a good day for a new activity or sport."},
    4: {"any": "{n} will want home comforts today; keep the evening calm.",
        "parent": "{n} would love a quiet evening at home with you today."},
    5: {"any": "A creative, light-hearted day for {n}.",
        "child": "A bright day for {n}'s studies and creative play.",
        "spouse": "A playful, romantic day: make {n} smile."},
    6: {"any": "{n} gets things done today: good for routines, health and pending work.",
        "child": "{n} can finish homework fast today: keep the routine."},
    7: {"any": "{n} connects easily with people today.",
        "spouse": "A warm day for the two of you: plan a little time together."},
    8: {"any": "{n} may feel low on energy; an early night will help.",
        "child": "{n} may feel tired or moody; extra sleep and a hug will help."},
    9: {"any": "A lucky day for {n}: blessings from elders and good news.",
        "parent": "{n} is in a giving mood today: ask for their blessings."},
    10: {"any": "{n} shines at work today.", "child": "{n} shines at school today: praise the effort."},
    11: {"any": "{n} gains from friends today: good news may come.",
         "child": "{n} has fun with friends today."},
    12: {"any": "{n} needs rest today; avoid big spending.",
         "child": "{n} needs rest today: an early bedtime helps."},
}
RELATION_GROUP = {"son": "child", "daughter": "child", "child": "child", "kid": "child",
                  "husband": "spouse", "wife": "spouse", "spouse": "spouse", "partner": "spouse",
                  "mother": "parent", "father": "parent", "mom": "parent", "dad": "parent",
                  "mother-in-law": "parent", "father-in-law": "parent", "grandmother": "parent", "grandfather": "parent"}


def day_line(name: str, relation: str, natal_rashi: int, natal_nak: int, today_rashi: int, today_nak: int) -> Dict[str, Any]:
    house = (today_rashi - natal_rashi) % 12 + 1
    tara = ((today_nak - natal_nak) % 27) % 9 + 1
    group = RELATION_GROUP.get((relation or "").strip().lower(), "any")
    t = LINES[house]
    text = (t.get(group) or t["any"]).format(n=name or "They")
    good = house in HOUSE_GOOD and tara not in TARA_HARD
    hard = house in HOUSE_HARD and tara in TARA_HARD
    return {"text": text, "tone": "good" if good else ("gentle" if hard else "steady"), "house": house, "tara": tara}


# ------------------------------------------------------------------ good dates (muhurat guide)
EVENTS: Dict[str, Dict[str, Any]] = {
    "wedding": {"title": "Wedding", "naks": {3, 4, 9, 11, 12, 14, 16, 18, 20, 25, 26},
                "tithis": {2, 3, 5, 7, 10, 11, 12, 13, 16, 17, 18, 20, 22, 25},
                "days": {0, 2, 3, 4}, "season": True, "planets": True},
    "griha_pravesh": {"title": "Griha pravesh (housewarming)", "naks": {3, 4, 11, 13, 16, 20, 22, 23, 25, 26},
                      "tithis": {2, 3, 5, 7, 10, 11, 12, 13, 17, 18, 20, 22},
                      "days": {0, 2, 3, 4, 5}, "season": True, "planets": False},
    "naming": {"title": "Naming ceremony (namkaran)", "naks": {0, 3, 4, 6, 7, 11, 12, 13, 14, 16, 20, 21, 22, 23, 25, 26},
               "tithis": {1, 2, 3, 5, 7, 10, 11, 12, 13, 16, 17, 18, 20, 22, 25},
               "days": {0, 2, 3, 4}, "season": False, "planets": False},
    "mundan": {"title": "Mundan (first haircut)", "naks": {0, 4, 6, 7, 12, 13, 14, 21, 22, 23, 26},
               "tithis": {2, 3, 5, 7, 10, 11, 12, 13},
               "days": {0, 2, 3, 4}, "season": True, "planets": False},
    "vehicle": {"title": "New vehicle", "naks": {0, 3, 4, 6, 7, 12, 13, 14, 16, 21, 22, 23, 26},
                "tithis": {1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 25},
                "days": {0, 2, 3, 4, 6}, "season": False, "planets": False},
    "business": {"title": "New business or job", "naks": {0, 3, 7, 11, 12, 13, 16, 20, 21, 25, 26},
                 "tithis": {1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 25},
                 "days": {2, 3, 4, 6}, "season": False, "planets": False},
    "property": {"title": "Buying property", "naks": {3, 4, 9, 11, 15, 18, 20, 23, 25, 26},
                 "tithis": {2, 3, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 25},
                 "days": {0, 3, 4, 5}, "season": False, "planets": False},
}


def _blocked(jd: float, ev: Dict[str, Any]) -> Optional[str]:
    sun = sid(jd, swe.SUN)
    if ev["season"]:
        sign = int(sun // 30.0)
        if sign in (8, 11):
            return "Kharmas (Sun in " + ("Dhanu" if sign == 8 else "Meena") + "): no ceremonies"
        m = lunar_month(jd)
        t = tithi_at(jd)
        if m["adhik"]:
            return "Adhik maas: ceremonies wait"
        mon = m["amanta"]
        if mon in ("Shravana", "Bhadrapada", "Ashwin") or (mon == "Ashadha" and t >= 11) or (mon == "Kartika" and t <= 11):
            return "Chaturmas: the gods rest until Dev Uthani Ekadashi"
    if ev["planets"]:
        for body, name, orb in ((swe.VENUS, "Venus", 10.0), (swe.JUPITER, "Jupiter", 11.0)):
            d = abs((sid(jd, body) - sun + 180.0) % 360.0 - 180.0)
            if d < orb:
                return f"{name} is combust (too close to the Sun)"
    return None


def _good_windows(rise: float, next_rise: float, ev: Dict[str, Any]) -> List[Tuple[float, float, int, int]]:
    """Stretches of the Hindu day (sunrise to sunrise) when both the tithi and the nakshatra suit the event."""
    out = []
    t = rise
    for _ in range(8):
        if t >= next_rise - 1e-4:
            break
        tn, nk = tithi_at(t), int(moon_lon(t) // NAK_SPAN) % 27
        end = min(_end_of(tithi_angle, t, 12.0), _end_of(moon_lon, t, NAK_SPAN), next_rise)
        if tn in ev["tithis"] and nk in ev["naks"] and end - t >= 1 / 24:
            out.append((t, end, tn, nk))
        t = end + 1e-5
    return out


def good_days(event: str, start: Date, days: int, lat: float, lon: float, tz_name: str,
              people: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    ev = EVENTS.get(event) or EVENTS["griha_pravesh"]
    tz = zone(tz_name)
    found: List[Dict[str, Any]] = []
    blocked_note: Optional[str] = None
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() not in ev["days"]:
            continue
        f = _day_frame(d.year, d.month, d.day, round(lat, 3), round(lon, 3), tz.key)
        wins = _good_windows(f["sunrise"], f["nextRise"], ev)
        if not wins:
            continue
        why_block = _blocked(f["sunrise"], ev)
        if why_block:
            blocked_note = blocked_note or why_block
            continue
        w0, w1, tn, nk = max(wins, key=lambda w: w[1] - w[0])
        t_name, paksha = tithi_name(tn)
        reasons = [f"{NAKSHATRAS[nk][0]} nakshatra", f"{paksha} {t_name}", WEEKDAYS[d.weekday()]]
        score = 3 + (1 if w1 - w0 >= 0.5 else 0)
        personal = []
        mid = (w0 + w1) / 2
        today_rashi = int(moon_lon(mid) // 30.0)
        for person in people or []:
            pr, pn = person.get("rashi"), person.get("nak")
            if pr is None or pn is None:
                continue
            house = (today_rashi - pr) % 12 + 1
            tara = ((nk - pn) % 27) % 9 + 1
            ok = house not in HOUSE_HARD and tara not in TARA_HARD
            score += 1 if ok else -1
            personal.append({"name": person.get("name") or "", "ok": ok})
        s_t, e_t = moment_of(w0, tz), moment_of(w1, tz)
        rise_t, set_t = moment_of(f["sunrise"], tz), moment_of(f["sunset"], tz)
        daylight = f["sunset"] - f["sunrise"]
        rahu_n = RAHU_PART[d.weekday()]
        rahu = (moment_of(f["sunrise"] + daylight / 8 * (rahu_n - 1), tz), moment_of(f["sunrise"] + daylight / 8 * rahu_n, tz))
        # Dated by the calendar day the window starts on, the way muhurat lists print it.
        on = s_t.date()
        item = {"date": on.isoformat(), "weekday": WEEKDAYS[on.weekday()], "score": score, "reasons": reasons,
                "window": {"start": s_t.strftime("%H:%M"), "end": e_t.strftime("%H:%M"),
                           "endsNextDay": e_t.date() > on, "night": s_t >= set_t or e_t <= rise_t},
                "rahuKaal": {"start": rahu[0].strftime("%H:%M"), "end": rahu[1].strftime("%H:%M")} if on == d else None,
                "people": personal, "_len": w1 - w0}
        same = next((x for x in found if x["date"] == item["date"]), None)
        if same is None:
            found.append(item)
        elif item["_len"] > same["_len"]:
            found[found.index(same)] = item
    for x in found:
        x.pop("_len", None)
    best = sorted(sorted(found, key=lambda x: (-x["score"], x["date"]))[:3], key=lambda x: x["date"])
    return {"event": event if event in EVENTS else "griha_pravesh", "title": ev["title"], "days": found,
            "best": [b["date"] for b in best], "blockedNote": blocked_note if not found else None,
            "note": "A traditional guide from the panchang. For the exact muhurat of a wedding or griha pravesh, "
                    "confirm the time with your family pandit."}
