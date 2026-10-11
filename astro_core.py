"""Panchang, birth charts, Vimshottari dashas, family day lines, the week ahead, big sky events (sign
ingresses of Jupiter, Saturn and Rahu) and good dates (muhurat guide).

Pure computation on Swiss Ephemeris, sidereal Lahiri, no I/O: astro_api.py serves it over HTTP.
Times are local to the place asked for. Panchang rules follow the common North Indian practice:
the day's tithi, nakshatra and yoga are the ones running at sunrise; Rahu kaal, Yamaganda and
Gulika are eighths of the daylight; Abhijit is the 8th of the 15 daytime muhurtas.
"""
from __future__ import annotations

import threading
from bisect import bisect_right
from datetime import date as Date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import swisseph as swe

import fates_astro_chart
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
    """A tz database zone; anything unknown or malformed (the app may send any text) is India time."""
    key = str(name or "").strip()
    if not key or len(key) > 64:
        return ZoneInfo("Asia/Kolkata")
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        # OSError: a folder such as "Asia" (PermissionError on Windows, IsADirectoryError on Linux) or a bad name.
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

    rahu, yama, gulika = eighth(RAHU_PART[wd]), eighth(YAMA_PART[wd]), eighth(GULIKA_PART[wd])
    good_times = []
    if wd != 2:   # Abhijit is skipped on Wednesdays
        # On Fridays Rahu kaal covers the first half of Abhijit: only the part outside Rahu kaal is a good time.
        a0, a1 = rise + daylight * 7 / 15, rise + daylight * 8 / 15
        r0, r1 = rise + part * (RAHU_PART[wd] - 1), rise + part * RAHU_PART[wd]
        free = [(a0, min(a1, r0)), (max(a0, r1), a1)] if r0 < a1 and a0 < r1 else [(a0, a1)]
        s0, s1 = max(free, key=lambda w: w[1] - w[0])
        if s1 - s0 >= 10 / 1440.0:
            good_times.append({"name": "Abhijit muhurat", **_span(moment_of(s0, tz), moment_of(s1, tz))})
    # A good choghadiya that falls on Rahu kaal, Yamaganda or Gulika is left out: the app shows those as times
    # to avoid, and the same slot can't be both.
    bad = {rahu["startIso"], yama["startIso"], gulika["startIso"]}
    for c in chog:
        if c["good"] and c["startIso"] not in bad:
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
        "rahuKaal": rahu, "yamaganda": yama, "gulika": gulika,
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


def _masa(nm: float, nm_next: float, krishna: bool) -> Dict[str, Any]:
    """The lunar month that starts at the new moon nm (the next one is nm_next), in the dark or bright half."""
    sign = int(sid(nm, swe.SUN) // 30.0)
    adhik = int(sid(nm_next, swe.SUN) // 30.0) == sign
    amanta = MASA_BY_SUN_SIGN[sign]
    # Purnimanta (North India) months run full moon to full moon: the dark half belongs to the next month.
    # An adhik month runs new moon to new moon in both systems, so its dark half keeps the adhik name.
    purnimanta = MASA_BY_SUN_SIGN[(sign + 1) % 12] if krishna and not adhik else amanta
    return {"amanta": amanta, "purnimanta": purnimanta, "adhik": adhik}


def lunar_month(jd: float) -> Dict[str, Any]:
    nm = _prev_new_moon(jd)
    return _masa(nm, _prev_new_moon(nm + 31.0), tithi_angle(jd) >= 180.0)


# ------------------------------------------------------------------ vrats and festivals
PRADOSH_KAAL = 0.1   # the three muhurtas (2 h 24 min) after sunset


def vrats_between(start: Date, days: int, lat: float, lon: float, tz_name: str) -> List[Dict[str, Any]]:
    """Monthly vrats from start for days. Each vrat goes by its tithi at the moment that matters for it (sunrise,
    sunset or moonrise). A tithi can also begin after one such moment and end before the next (kshaya): it is
    then placed on the day it fell within, so no Ekadashi, Amavasya, Purnima, Pradosh or Sankashti goes missing."""
    tz = zone(tz_name)
    la, lo = round(lat, 3), round(lon, 3)
    # The day before and the day after too: a skipped tithi is found by comparing neighbours.
    dates = [start + timedelta(days=i) for i in range(-1, days + 1)]
    frames = [_day_frame(x.year, x.month, x.day, la, lo, tz.key) for x in dates]
    at_rise = [tithi_at(f["sunrise"]) for f in frames]
    at_set = [tithi_at(f["sunset"]) for f in frames]
    at_moon = [tithi_at(f["moonrise"]) if f["moonrise"] else None for f in frames]

    def skipped(a: Optional[int], b: Optional[int]) -> Optional[int]:
        """The tithi that ran wholly between two moments with tithis a and b, if one did."""
        return a % 30 + 1 if a is not None and b is not None and (b - a) % 30 == 2 else None

    found: List[Tuple[int, str, str, str]] = []   # (index into dates, kind, name, note)
    for i in range(1, days + 1):
        f = frames[i]
        t_rise, t_set, t_moon = at_rise[i], at_set[i], at_moon[i]
        k_rise = skipped(t_rise, at_rise[i + 1])   # began after this sunrise, ended before the next
        if t_rise in (11, 26) or k_rise in (11, 26):
            found.append((i, "ekadashi", "Ekadashi vrat", "A day of fasting and Vishnu puja."))
        k_set = skipped(at_set[i - 1], t_set)
        if t_set in (13, 28):
            found.append((i, "pradosh", "", ""))
        elif k_set in (13, 28):
            # Trayodashi began after yesterday's sunset and ended before today's: the evening whose pradosh
            # kaal it touched, else today.
            touched = tithi_at(frames[i - 1]["sunset"] + PRADOSH_KAAL) == k_set
            found.append((i - 1 if touched else i, "pradosh", "", ""))
        if t_set == 15 or (t_rise == 15 and t_set == 16) or k_rise == 15:
            found.append((i, "purnima", "Purnima", "Full moon: Satyanarayan katha in many homes."))
        if t_rise == 30 or k_rise == 30:
            found.append((i, "amavasya", "Amavasya", "New moon: a day to remember the ancestors."))
        if t_moon == 19:
            found.append((i, "sankashti", "", ""))
        elif at_moon[i - 1] == 18 and (t_moon == 20 or (t_moon is None and at_moon[i + 1] != 19)):
            # Chaturthi came and went between two moonrises (a day without moonrise in between counts as
            # none): today if it held at sunrise or sunset, else yesterday.
            found.append((i if 19 in (t_rise, t_set) else i - 1, "sankashti", "", ""))

    out: List[Dict[str, Any]] = []
    for i, kind, name, note in sorted(found, key=lambda x: x[0]):
        if i < 1:   # the day before the range
            continue
        d = dates[i]
        if kind == "pradosh":
            name, note = {0: "Som Pradosh", 5: "Shani Pradosh"}.get(d.weekday(), "Pradosh vrat"), "Shiva puja at dusk, after sunset."
        elif kind == "sankashti":
            mr = frames[i]["moonrise"]
            name = "Sankashti Chaturthi"
            note = (f"Ganesha vrat, broken at moonrise ({moment_of(mr, tz).strftime('%H:%M')})." if mr
                    else "Ganesha vrat, broken at moonrise.")
        # One entry per tithi: the first day it falls on. The same vrat comes back after 13 days at the soonest,
        # so anything within 3 days is the same tithi seen twice (two moonrises far north, for one).
        if any(o["kind"] == kind and abs((d - Date.fromisoformat(o["date"])).days) <= 3 for o in out):
            continue
        out.append({"date": d.isoformat(), "kind": kind, "type": "vrat", "name": name, "note": note})
    return out


# Festivals by date (New Delhi). Extend the list before it runs out: a test fails 60 days before its last date.
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
    # From here on worked out by the same panchang rules for New Delhi (udaya tithi, or the tithi in aparahna,
    # pradosh, nishita or at moonrise, Bhadra for Rakhi) and checked against Drik Panchang and the DoPT 2027 list.
    {"date": "2027-04-15", "id": "ram_navami", "name": "Ram Navami", "note": "Lord Rama's birthday."},
    {"date": "2027-04-19", "id": "mahavir_jayanti", "name": "Mahavir Jayanti", "note": "The birth of Bhagwan Mahavir."},
    {"date": "2027-04-20", "id": "hanuman_jayanti", "name": "Hanuman Jayanti", "note": "Chaitra Purnima: Hanuman Chalisa in many homes."},
    {"date": "2027-05-09", "id": "akshaya_tritiya", "name": "Akshaya Tritiya", "note": "A lucky day for gold and new beginnings."},
    {"date": "2027-05-17", "id": "bakrid", "name": "Eid ul-Adha (Bakrid)", "note": "Depends on the moon sighting."},
    {"date": "2027-05-20", "id": "buddha_purnima", "name": "Buddha Purnima", "note": "Vaishakha Purnima."},
    {"date": "2027-06-04", "id": "vat_savitri", "name": "Vat Savitri", "note": "Wives pray at the banyan tree for their husband's long life."},
    {"date": "2027-07-18", "id": "guru_purnima", "name": "Guru Purnima", "note": "Thank your teachers and elders."},
    {"date": "2027-08-04", "id": "hariyali_teej", "name": "Hariyali Teej", "note": "Green bangles, swings and mehndi."},
    {"date": "2027-08-15", "id": "independence_day", "name": "Independence Day", "note": ""},
    {"date": "2027-08-17", "id": "raksha_bandhan", "name": "Raksha Bandhan", "note": "Rakhi for brothers, blessings and gifts for sisters."},
    {"date": "2027-08-25", "id": "janmashtami", "name": "Janmashtami", "note": "Krishna's birthday: the fast ends at midnight."},
    {"date": "2027-09-04", "id": "ganesh_chaturthi", "name": "Ganesh Chaturthi", "note": "Ganpati comes home."},
    {"date": "2027-09-29", "id": "mahalaya", "name": "Mahalaya Amavasya", "note": "Pitru Paksha ends: a day to remember the ancestors."},
    {"date": "2027-09-30", "id": "navratri", "name": "Sharad Navratri begins", "note": "Nine nights of Durga: ghatasthapana in the morning."},
    {"date": "2027-10-06", "id": "durga_puja", "name": "Durga Puja (Saptami)", "note": "Pandal days begin in Bengal and beyond."},
    {"date": "2027-10-09", "id": "dussehra", "name": "Dussehra", "note": "Vijayadashami: good wins over evil."},
    {"date": "2027-10-14", "id": "sharad_purnima", "name": "Sharad Purnima", "note": "Kheer under the full moon."},
    {"date": "2027-10-18", "id": "karva_chauth", "name": "Karva Chauth", "note": "The fast ends at moonrise: see the time for your city."},
    {"date": "2027-10-27", "id": "dhanteras", "name": "Dhanteras", "note": "Buy something new for the home, light the first diya."},
    {"date": "2027-10-29", "id": "diwali", "name": "Diwali", "note": "Lakshmi puja in the evening."},
    {"date": "2027-10-31", "id": "bhai_dooj", "name": "Bhai Dooj", "note": "Tilak for brothers, blessings for sisters."},
    {"date": "2027-11-04", "id": "chhath", "name": "Chhath Puja", "note": "Evening arghya to the setting Sun."},
    {"date": "2027-11-10", "id": "dev_uthani", "name": "Dev Uthani Ekadashi", "note": "Tulsi Vivah: the wedding season opens."},
    {"date": "2027-11-14", "id": "gurpurab", "name": "Guru Nanak Jayanti", "note": "Kartik Purnima."},
    {"date": "2027-12-25", "id": "christmas", "name": "Christmas", "note": ""},
    {"date": "2028-01-01", "id": "new_year", "name": "New Year 2028", "note": ""},
    {"date": "2028-01-14", "id": "lohri", "name": "Lohri", "note": "Bonfire, rewri and gajak."},
    {"date": "2028-01-15", "id": "sankranti", "name": "Makar Sankranti · Pongal", "note": "The Sun enters Makara: til-gud and kites."},
    {"date": "2028-01-26", "id": "republic_day", "name": "Republic Day", "note": ""},
    {"date": "2028-01-31", "id": "basant_panchami", "name": "Basant Panchami", "note": "Saraswati puja: wear yellow."},
    {"date": "2028-02-23", "id": "shivratri", "name": "Maha Shivratri", "note": "Night of Shiva."},
    {"date": "2028-03-10", "id": "holika_dahan", "name": "Holika Dahan", "note": "The bonfire on the eve of Holi."},
    {"date": "2028-03-11", "id": "holi", "name": "Holi", "note": "Colours day."},
    {"date": "2028-03-27", "id": "ugadi", "name": "Ugadi · Gudi Padwa", "note": "Hindu new year."},
    {"date": "2028-04-03", "id": "ram_navami", "name": "Ram Navami", "note": "Lord Rama's birthday."},
    {"date": "2028-04-07", "id": "mahavir_jayanti", "name": "Mahavir Jayanti", "note": "The birth of Bhagwan Mahavir."},
    {"date": "2028-04-09", "id": "hanuman_jayanti", "name": "Hanuman Jayanti", "note": "Chaitra Purnima: Hanuman Chalisa in many homes."},
    {"date": "2028-04-13", "id": "baisakhi", "name": "Baisakhi", "note": "Harvest festival."},
    {"date": "2028-04-27", "id": "akshaya_tritiya", "name": "Akshaya Tritiya", "note": "A lucky day for gold and new beginnings."},
    {"date": "2028-05-08", "id": "buddha_purnima", "name": "Buddha Purnima", "note": "Vaishakha Purnima."},
    {"date": "2028-05-24", "id": "vat_savitri", "name": "Vat Savitri", "note": "Wives pray at the banyan tree for their husband's long life."},
    {"date": "2028-07-06", "id": "guru_purnima", "name": "Guru Purnima", "note": "Thank your teachers and elders."},
    {"date": "2028-07-24", "id": "hariyali_teej", "name": "Hariyali Teej", "note": "Green bangles, swings and mehndi."},
    {"date": "2028-08-05", "id": "raksha_bandhan", "name": "Raksha Bandhan", "note": "Rakhi for brothers, blessings and gifts for sisters."},
    {"date": "2028-08-13", "id": "janmashtami", "name": "Janmashtami", "note": "Krishna's birthday: the fast ends at midnight."},
    {"date": "2028-08-15", "id": "independence_day", "name": "Independence Day", "note": ""},
    {"date": "2028-08-23", "id": "ganesh_chaturthi", "name": "Ganesh Chaturthi", "note": "Ganpati comes home."},
    {"date": "2028-09-18", "id": "mahalaya", "name": "Mahalaya Amavasya", "note": "Pitru Paksha ends: a day to remember the ancestors."},
    {"date": "2028-09-19", "id": "navratri", "name": "Sharad Navratri begins", "note": "Nine nights of Durga: ghatasthapana in the morning."},
    {"date": "2028-09-25", "id": "durga_puja", "name": "Durga Puja (Saptami)", "note": "Pandal days begin in Bengal and beyond."},
    {"date": "2028-09-27", "id": "dussehra", "name": "Dussehra", "note": "Vijayadashami: good wins over evil."},
    {"date": "2028-10-02", "id": "sharad_purnima", "name": "Sharad Purnima", "note": "Kheer under the full moon."},
    {"date": "2028-10-07", "id": "karva_chauth", "name": "Karva Chauth", "note": "The fast ends at moonrise: see the time for your city."},
    {"date": "2028-10-15", "id": "dhanteras", "name": "Dhanteras", "note": "Buy something new for the home, light the first diya."},
    {"date": "2028-10-17", "id": "diwali", "name": "Diwali", "note": "Lakshmi puja in the evening."},
    {"date": "2028-10-19", "id": "bhai_dooj", "name": "Bhai Dooj", "note": "Tilak for brothers, blessings for sisters."},
    {"date": "2028-10-23", "id": "chhath", "name": "Chhath Puja", "note": "Evening arghya to the setting Sun."},
    {"date": "2028-10-28", "id": "dev_uthani", "name": "Dev Uthani Ekadashi", "note": "Tulsi Vivah: the wedding season opens."},
    {"date": "2028-11-02", "id": "gurpurab", "name": "Guru Nanak Jayanti", "note": "Kartik Purnima."},
    {"date": "2028-12-25", "id": "christmas", "name": "Christmas", "note": ""},
]


# Festivals that already are the day's vrat: the generic entry within a day of them is left out.
FESTIVAL_COVERS = {"dev_uthani": {"ekadashi"}, "sharad_purnima": {"purnima"}, "gurpurab": {"purnima"},
                   "diwali": {"amavasya"}, "karva_chauth": {"sankashti"}, "holika_dahan": {"purnima"},
                   "shivratri": {"pradosh"}, "hanuman_jayanti": {"purnima"}, "buddha_purnima": {"purnima"},
                   "guru_purnima": {"purnima"}, "raksha_bandhan": {"purnima"}, "vat_savitri": {"amavasya"},
                   "mahalaya": {"amavasya"}}


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
# "I don't know my birth time" answered with a part of the day (users/{uid}.birthTimeApprox): the Moon-only chart
# is drawn for the middle of that part instead of noon. It never gives a Lagna, and the chart stays "approximate".
APPROX_TIMES: Dict[str, Tuple[int, int]] = {"morning": (9, 0), "afternoon": (14, 30), "evening": (19, 0), "night": (3, 0)}


def approx_time(key: Any) -> Optional[Tuple[int, int]]:
    """The clock time a birthTimeApprox answer stands for, or None for anything else."""
    return APPROX_TIMES.get(str(key or "").strip().lower())


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
                now: Optional[datetime] = None, approx: Optional[str] = None) -> Dict[str, Any]:
    """approx: users/{uid}.birthTimeApprox, used only when time is None (see APPROX_TIMES)."""
    approx_t = None if time else approx_time(approx)
    born = birth_moment(date, time or approx_t, place)
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
    precision = "exact" if lagna is not None else ("time" if time else ("approximate" if approx_t else "date"))
    # Without a time the Moon may sit up to ~6.5 deg away: its nakshatra (and so the dasha dates) can shift.
    moon_sure = True
    options: Optional[List[List[int]]] = None
    if not time:
        tz = born.tzinfo
        lo = sid(jd_of(datetime(date[0], date[1], date[2], 0, 0, tzinfo=tz)), swe.MOON)
        hi = sid(jd_of(datetime(date[0], date[1], date[2], 23, 59, tzinfo=tz)), swe.MOON)
        moon_sure = int(lo // NAK_SPAN) == int(hi // NAK_SPAN)
        options = moon_options(date, place, (moon["rashi"], moon["nak"]))
    moon_out: Dict[str, Any] = {
        "rashi": moon["rashi"], "rashiName": RASHIS[moon["rashi"]][0], "rashiEnglish": RASHIS[moon["rashi"]][1],
        "nakshatra": NAKSHATRAS[moon["nak"]][0], "nak": moon["nak"],
        "pada": int((moon["lon"] % NAK_SPAN) // (NAK_SPAN / 4)) + 1, "sure": moon_sure,
        "vh": moon_half(moon["rashi"], moon["lon"]),
    }
    if options is not None:
        moon_out["options"] = options
    out = {
        "precision": precision, "timeKnown": time is not None, "lagna": lagna,
        "lagnaName": RASHIS[lagna][0] if lagna is not None else None,
        "lagnaEnglish": RASHIS[lagna][1] if lagna is not None else None,
        "basis": "lagna" if lagna is not None else "moon",
        "planets": planets,
        "moon": moon_out,
        "sun": {"rashi": planets[0]["rashi"], "rashiName": RASHIS[planets[0]["rashi"]][0], "rashiEnglish": RASHIS[planets[0]["rashi"]][1]},
        "manglik": mang,
        "dasha": ds,
        "dashaApproximate": not time,
        "lines": chart_lines(lagna, moon, planets[0], ds, mang, time is not None),
        "transits": transits(moon["rashi"], now, sure=len({o[0] for o in options or [[moon["rashi"]]]}) == 1),
    }
    if approx_t:
        out["timeApprox"] = str(approx).strip().lower()
    return out


def moon_options(date: Tuple[int, int, int], place: Optional[Dict[str, Any]], own: Tuple[int, int]) -> List[List[int]]:
    """Every (rashi, nakshatra) the Moon takes on a birth day without a known time, the chart's own Moon first.
    The same scan as fates_astro_chart.chart_from_birth, so the app, the match and the site agree."""
    out = [[own[0], own[1]]]
    for r, n in fates_astro_chart.chart_from_birth(date, None, place)["options"]:
        if [r, n] not in out:
            out.append([r, n])
    return out


def moon_half(rashi: int, lon: float) -> Optional[int]:
    """Which half of Dhanu or Makara the Moon is in (0 = first 15 degrees, 1 = the rest): Vashya splits those two
    signs. None for every other sign."""
    if rashi not in (8, 9):
        return None
    return 0 if (lon % 30.0) < 15.0 else 1


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
    text = (t.get(group) or t["any"]).format(n=name or "your loved one")
    if not name:   # the lines are written for one person; a line may open with the name
        text = text[0].upper() + text[1:]
    good = house in HOUSE_GOOD and tara not in TARA_HARD
    hard = house in HOUSE_HARD and tara in TARA_HARD
    return {"text": text, "tone": "good" if good else ("gentle" if hard else "steady"), "house": house, "tara": tara}


# ------------------------------------------------------------------ the week ahead
# Chandra gochara themes: the Moon counted from the birth Moon sign (the same table the daily horoscope uses).
HOUSE_THEMES = {
    1: "mood, self-care and fresh starts",
    2: "money, family and how you speak",
    3: "courage, messages and initiative",
    4: "home, comfort and inner peace",
    5: "romance, creativity and fun",
    6: "work routines, health habits and small hurdles",
    7: "partnerships and one-to-one connections",
    8: "a sensitive day (Chandrashtama): patience and caution",
    9: "luck, blessings and learning",
    10: "career, visibility and duty",
    11: "gains, friends and wishes coming true",
    12: "rest, spending and letting go",
}

# One second-person line per house for the reader's own week. A hard day is "sensitive" or "quiet", never "bad".
SELF_LINES = {
    1: "A personal day for you: good for self-care and fresh starts.",
    2: "Family talk, good food and money matters go well for you.",
    3: "You have extra courage: good for calls, messages and taking the first step.",
    4: "You'll want home comforts: keep the evening calm and close to family.",
    5: "A light, creative day for you: good for romance, hobbies and fun.",
    6: "You get things done: good for routines, health habits and pending work.",
    7: "You connect easily with people: good for meeting someone or a long talk.",
    8: "A sensitive day for you: go slow, be patient and plan an early night.",
    9: "A lucky day for you: blessings from elders, learning and good news.",
    10: "You shine at work: a good day to be seen and to finish duties.",
    11: "Friends bring you gains and good news; wishes come closer.",
    12: "A day for you to rest: go easy on spending and let old worries go.",
}

# Best days of the week by the Moon's house from the birth Moon: (key, houses, quiet). Two days at most each.
BEST_RULES = (("love", frozenset({5, 7, 9, 11}), False),
              ("work", frozenset({3, 6, 10, 11}), False),
              ("quiet", frozenset({4, 8, 12}), True))
BEST_MAX = 2


def week_tone(house: int, tara: int) -> str:
    """The day_line rule, plus the 8th house from the birth Moon (Chandrashtama), which is always a gentle day."""
    if house == 8:
        return "gentle"
    if house in HOUSE_GOOD and tara not in TARA_HARD:
        return "good"
    if house in HOUSE_HARD and tara in TARA_HARD:
        return "gentle"
    return "steady"


def _tithi_label(n: int) -> str:
    name, paksha = tithi_name(n)
    return name if n in (15, 30) else f"{paksha} {name}"


def week_days(natal_rashi: int, natal_nak: int, monday: Date, lat: float, lon: float, tz: str,
              name: Optional[str] = None, relation: str = "") -> List[Dict[str, Any]]:
    """Seven days from monday: the Moon at sunrise + 6 h against the birth Moon gives the house, the tara and the
    tone. The line is the reader's own (SELF_LINES) or, for a profile made for someone else (name given), the
    family line with that name. The day's tithi is the one running at sunrise."""
    tzz = zone(tz)
    la, lo = round(float(lat), 3), round(float(lon), 3)
    group = RELATION_GROUP.get((relation or "").strip().lower(), "any")
    who = (name or "").strip()[:30]
    out: List[Dict[str, Any]] = []
    for i in range(7):
        d = monday + timedelta(days=i)
        f = _day_frame(d.year, d.month, d.day, la, lo, tzz.key)
        ml = moon_lon(f["sunrise"] + 0.25)
        rashi, nak = int(ml // 30.0) % 12, int(ml // NAK_SPAN) % 27
        house = (rashi - natal_rashi) % 12 + 1
        tara = ((nak - natal_nak) % 27) % 9 + 1
        if who:
            t = LINES[house]
            line = (t.get(group) or t["any"]).format(n=who)
        else:
            line = SELF_LINES[house]
        out.append({"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "house": house, "tara": tara,
                    "tone": week_tone(house, tara), "line": line, "tithi": _tithi_label(tithi_at(f["sunrise"])),
                    "moonSign": RASHIS[rashi][0]})
    return out


def best_days(days: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """{love, work, quiet}: up to two dates each. Love and work skip gentle days and take good days first; quiet
    days take the gentlest first. Each list is in date order."""
    rank = {"good": 0, "steady": 1, "gentle": 2}
    out: Dict[str, List[str]] = {}
    for key, houses, quiet in BEST_RULES:
        pool = [d for d in days if d["house"] in houses and (quiet or d["tone"] != "gentle")]
        pool.sort(key=lambda d: ((-rank[d["tone"]]) if quiet else rank[d["tone"]], d["date"]))
        out[key] = sorted(d["date"] for d in pool[:BEST_MAX])
    return out


# ------------------------------------------------------------------ big sky events (Jupiter, Saturn, Rahu)
IST = ZoneInfo("Asia/Kolkata")
SKY_BODIES = (("Jupiter", swe.JUPITER), ("Saturn", swe.SATURN), ("Rahu", swe.MEAN_NODE))
SKY_DAYS = 450


def _sign_at(jd: float, body: int) -> int:
    return int(sid(jd, body) // 30.0) % 12


def _crossing(body: int, lo: float, hi: float, sign_lo: int) -> float:
    """The moment between lo and hi (a day apart) when body leaves sign_lo, to about a second."""
    for _ in range(28):
        mid = (lo + hi) / 2
        if _sign_at(mid, body) == sign_lo:
            lo = mid
        else:
            hi = mid
    return hi


def _ingress(planet: str, frm: int, to: int, jd: float) -> Dict[str, Any]:
    at = moment_of(jd, IST)
    # Jupiter and Saturn going back a sign are retrograde; the mean Rahu always moves back through the signs.
    back = (frm - to) % 12 == 1
    return {"planet": planet, "sign": to, "signName": RASHIS[to][0], "signEnglish": RASHIS[to][1], "from": frm,
            "date": at.date().isoformat(), "atIso": at.isoformat(timespec="minutes"),
            "retro": back and planet != "Rahu"}


@lru_cache(maxsize=32)
def _ingress_scan(start_ord: int, days: int) -> Tuple[Dict[str, Any], ...]:
    start = Date.fromordinal(start_ord)
    jd0 = jd_of(datetime(start.year, start.month, start.day, tzinfo=IST))
    found: List[Dict[str, Any]] = []
    for planet, body in SKY_BODIES:
        prev = _sign_at(jd0, body)
        for i in range(1, days + 1):
            s = _sign_at(jd0 + i, body)
            if s != prev:
                found.append(_ingress(planet, prev, s, _crossing(body, jd0 + i - 1, jd0 + i, prev)))
                prev = s
    found.sort(key=lambda e: e["atIso"])
    return tuple(found)


def sky_events(start: Date, days: int = SKY_DAYS) -> List[Dict[str, Any]]:
    """Sign ingresses of Jupiter, Saturn and the mean Rahu from start (IST midnight) for days, by time."""
    return [dict(e) for e in _ingress_scan(start.toordinal(), int(days))]


def _today_ist() -> Date:
    return datetime.now(IST).date()


# Scanned once at import for the next 450 days (a few thousand ephemeris calls, well under a second).
SKY_EVENTS_START = _today_ist()
SKY_EVENTS: List[Dict[str, Any]] = sky_events(SKY_EVENTS_START, SKY_DAYS)


def sky_events_between(first: Date, last: Date) -> List[Dict[str, Any]]:
    """Ingresses dated (IST) first..last, both included: from SKY_EVENTS when it covers the range."""
    if SKY_EVENTS_START <= first and last < SKY_EVENTS_START + timedelta(days=SKY_DAYS):
        pool = SKY_EVENTS
    else:
        pool = sky_events(first, (last - first).days + 2)
    a, b = first.isoformat(), last.isoformat()
    return [dict(e) for e in pool if a <= e["date"] <= b]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# Positive themes for a big planet arriving in a house counted from the birth Moon (never fear, never "dosha").
SKY_HOUSE_THEMES = {
    1: "you, your health and fresh starts",
    2: "family, savings and kind words",
    3: "courage, siblings and short trips",
    4: "home, mother and peace of mind",
    5: "love, children and creativity",
    6: "work routines and good health habits",
    7: "marriage and partnerships",
    8: "shared money and inner growth",
    9: "luck, elders' blessings and learning",
    10: "career and standing",
    11: "gains, friends and wishes",
    12: "rest, travel and letting go",
}
SKY_PLANET_LINES = {
    "Jupiter": "a warm year for {theme} begins",
    "Saturn": "slow, steady effort on {theme} pays off",
    "Rahu": "a time of fresh ideas around {theme} begins",
}


def sky_line(event: Dict[str, Any], natal_rashi: Optional[int], name: Optional[str] = None) -> str:
    """'Jupiter moves into Simha, your 5th house from the Moon: a warm year for love, children and creativity
    begins.' name: the person a managed profile is for ("Ananya's 5th house"). Without a birth Moon only the move."""
    head = f"{event['planet']} moves into {event['signName']}"
    if natal_rashi is None:
        return f"{head} ({event['signEnglish']})."
    house = (event["sign"] - natal_rashi) % 12 + 1
    theme = SKY_PLANET_LINES[event["planet"]].format(theme=SKY_HOUSE_THEMES[house])
    whose = f"{name.strip()}'s" if name and name.strip() else "your"
    return f"{head}, {whose} {ordinal(house)} house from the Moon: {theme}."


# ------------------------------------------------------------------ transits for the chart
SADE_SATI_PHASE = {12: 1, 1: 2, 2: 3}   # Saturn's sign counted from the birth Moon sign -> phase


@lru_cache(maxsize=64)
def _final_exit(body: int, sign: int, start_ord: int, horizon: int = 1200) -> Optional[float]:
    """When body leaves sign for the last time within horizon days of start (retrograde returns included)."""
    start = Date.fromordinal(start_ord)
    jd0 = jd_of(datetime(start.year, start.month, start.day, tzinfo=IST))
    last_in = None
    for i in range(horizon + 1):
        if _sign_at(jd0 + i, body) == sign:
            last_in = i
    if last_in is None or last_in == horizon:
        return None
    return _crossing(body, jd0 + last_in, jd0 + last_in + 1, sign)


def _short_date(d: Date) -> str:
    return f"{d.day} {d:%b %Y}"


def transits(natal_rashi: int, now: Optional[datetime] = None, sure: bool = True) -> List[Dict[str, Any]]:
    """Sade Sati (Saturn in the 12th, 1st or 2nd sign from the birth Moon: phases 1-3, until Saturn leaves that
    sign for good) and the next Jupiter ingress with its house from the birth Moon. sure=False: the birth Moon
    sign itself is not certain (no birth time)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    out: List[Dict[str, Any]] = []
    try:
        jd = jd_of(now)
        sat = _sign_at(jd, swe.SATURN)
        phase = SADE_SATI_PHASE.get((sat - natal_rashi) % 12 + 1)
        if phase:
            item: Dict[str, Any] = {"kind": "sade_sati", "title": f"Sade Sati · phase {phase} of 3", "phase": phase,
                                    "text": "Slow, steady years: effort pays.", "sign": sat, "signName": RASHIS[sat][0]}
            exit_jd = _final_exit(swe.SATURN, sat, now.astimezone(IST).date().toordinal())
            if exit_jd is not None:
                item["until"] = moment_of(exit_jd, IST).date().isoformat()
            if not sure:
                item["approximate"] = True
            out.append(item)
        today = now.astimezone(IST).date()
        stamp = now.astimezone(IST).isoformat(timespec="minutes")
        nxt = next((e for e in sky_events_between(today, today + timedelta(days=SKY_DAYS - 1))
                    if e["planet"] == "Jupiter" and e["atIso"] > stamp), None)
        if nxt:
            house = (nxt["sign"] - natal_rashi) % 12 + 1
            when = Date.fromisoformat(nxt["date"])
            item = {"kind": "jupiter", "title": f"Jupiter moves into {nxt['signName']} · {_short_date(when)}",
                    "date": nxt["date"], "house": house, "sign": nxt["sign"], "signName": nxt["signName"],
                    "text": f"Your {ordinal(house)} house from the Moon: "
                            f"{SKY_PLANET_LINES['Jupiter'].format(theme=SKY_HOUSE_THEMES[house])}."}
            if not sure:
                item["approximate"] = True
            out.append(item)
    except Exception:   # a transit must never cost the chart
        return out
    return out


# ------------------------------------------------------------------ good dates (muhurat guide)
EVENTS: Dict[str, Dict[str, Any]] = {
    "wedding": {"title": "Wedding", "naks": {3, 4, 9, 11, 12, 14, 16, 18, 20, 25, 26},
                "tithis": {2, 3, 5, 7, 10, 11, 12, 13, 16, 17, 18, 20, 22, 25},
                "days": {0, 2, 3, 4}, "season": True, "planets": True},
    # Roka / sagai: the wedding's nakshatras, tithis and weekdays, and its season; no Venus/Jupiter check
    # (the conservative choice from the content design; needs the owner's or a pandit's sign-off).
    "engagement": {"title": "Roka / engagement", "naks": {3, 4, 9, 11, 12, 14, 16, 18, 20, 25, 26},
                   "tithis": {2, 3, 5, 7, 10, 11, 12, 13, 16, 17, 18, 20, 22, 25},
                   "days": {0, 2, 3, 4}, "season": True, "planets": False},
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


# Tithi and nakshatra changes are moments in UT, the same for every place and event. They are found once a
# day for the whole process (yesterday to about four months ahead) and looked up with bisect, so a good-dates
# search doesn't redo thousands of ephemeris calls for every request.
BOUNDS_BACK_DAYS = 2.0
BOUNDS_AHEAD_DAYS = 126.0
_bounds_lock = threading.Lock()
_bounds: Optional[Dict[str, Any]] = None


def _chain(fn, span: float, lo: float, hi: float) -> Tuple[List[float], List[int]]:
    """Every moment from lo until past hi when fn reaches a multiple of span, with the segment that ends there."""
    ends: List[float] = []
    segs: List[int] = []
    t = lo
    while True:
        seg = int(fn(t) // span)
        e = _end_of(fn, t, span)
        ends.append(e)
        segs.append(seg)
        if e > hi:
            return ends, segs
        t = e + 1e-5


def _build_bounds(day: Date) -> Dict[str, Any]:
    lo = jd_of(datetime(day.year, day.month, day.day, tzinfo=timezone.utc)) - BOUNDS_BACK_DAYS
    hi = lo + BOUNDS_BACK_DAYS + BOUNDS_AHEAD_DAYS
    tb, ts = _chain(tithi_angle, 12.0, lo, hi)
    nb, ns = _chain(moon_lon, NAK_SPAN, lo, hi)
    tn = [s % 30 + 1 for s in ts]
    # New moons (the end of Amavasya) mark the lunar months: one before the table and one after it as well.
    moons = [_prev_new_moon(lo)] + [e for e, n in zip(tb, tn) if n == 30]
    moons.append(_prev_new_moon(moons[-1] + 31.0))
    return {"day": day, "lo": lo, "hi": min(tb[-1], nb[-1]), "tb": tb, "tn": tn, "nb": nb,
            "nn": [s % 27 for s in ns], "moons": moons}


def shared_bounds(jd_from: float, jd_to: float) -> Optional[Dict[str, Any]]:
    """Today's table of tithi and nakshatra changes when it covers jd_from..jd_to, else None."""
    global _bounds
    today = datetime.now(timezone.utc).date()
    b = _bounds
    if b is None or b["day"] != today:
        with _bounds_lock:
            b = _bounds
            if b is None or b["day"] != today:
                b = _build_bounds(today)
                _bounds = b
    return b if b["lo"] <= jd_from and jd_to < b["hi"] else None


def _tithi_in(b: Dict[str, Any], jd: float) -> Tuple[int, float]:
    """The tithi running at jd and when it ends, from the table."""
    i = bisect_right(b["tb"], jd)
    return b["tn"][i], b["tb"][i]


def _nak_in(b: Dict[str, Any], jd: float) -> Tuple[int, float]:
    i = bisect_right(b["nb"], jd)
    return b["nn"][i], b["nb"][i]


def _blocked(jd: float, ev: Dict[str, Any], b: Optional[Dict[str, Any]] = None) -> Optional[str]:
    sun = sid(jd, swe.SUN)
    if ev["season"]:
        sign = int(sun // 30.0)
        if sign in (8, 11):
            return "Kharmas (Sun in " + ("Dhanu" if sign == 8 else "Meena") + "): no ceremonies"
        if b is not None:
            t = _tithi_in(b, jd)[0]
            i = bisect_right(b["moons"], jd) - 1
            m = _masa(b["moons"][i], b["moons"][i + 1], t > 15)
        else:
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
                return f"{name} is too close to the Sun (combust): families traditionally skip these weeks"
    return None


SEASON_BLOCKS = ("Kharmas", "Adhik maas", "Chaturmas")


def _is_season_block(reason: Optional[str]) -> bool:
    return bool(reason) and reason.startswith(SEASON_BLOCKS)


def _pieces(rise: float, next_rise: float, b: Optional[Dict[str, Any]]) -> List[Tuple[float, float, int, int]]:
    """The Hindu day (sunrise to sunrise) cut at every tithi and nakshatra change: (start, end, tithi, nakshatra)."""
    out = []
    t = rise
    for _ in range(8):
        if t >= next_rise - 1e-4:
            break
        if b is not None:
            (tn, te), (nk, ne) = _tithi_in(b, t), _nak_in(b, t)
        else:
            tn, nk = tithi_at(t), int(moon_lon(t) // NAK_SPAN) % 27
            te, ne = _end_of(tithi_angle, t, 12.0), _end_of(moon_lon, t, NAK_SPAN)
        end = min(te, ne, next_rise)
        out.append((t, end, tn, nk))
        t = end + 1e-5
    return out


def _good_windows(rise: float, next_rise: float, ev: Dict[str, Any],
                  b: Optional[Dict[str, Any]] = None) -> List[Tuple[float, float, List[int], List[int]]]:
    """Stretches of the Hindu day when both the tithi and the nakshatra suit the event, with every tithi and
    nakshatra inside each. A change from one good tithi (or nakshatra) to another doesn't break a stretch:
    neighbouring good pieces are joined first, and only then is anything under an hour left out."""
    wins: List[List[Any]] = []
    prev_good = False
    for t0, t1, tn, nk in _pieces(rise, next_rise, b):
        good = tn in ev["tithis"] and nk in ev["naks"]
        if good and prev_good:
            w = wins[-1]
            w[1] = t1
            if tn not in w[2]:
                w[2].append(tn)
            if nk not in w[3]:
                w[3].append(nk)
        elif good:
            wins.append([t0, t1, [tn], [nk]])
        prev_good = good
    return [(w[0], w[1], w[2], w[3]) for w in wins if w[1] - w[0] >= 1 / 24]


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _tithi_words(tns: List[int]) -> str:
    """"Krishna Pratipada", "Krishna Pratipada and Dwitiya", "Purnima and Krishna Pratipada"."""
    named = [tithi_name(t) for t in tns]
    if len({p for _, p in named}) == 1 and not any(t in (15, 30) for t in tns):
        return f"{named[0][1]} " + _and([n for n, _ in named])
    return _and([n if t in (15, 30) else f"{p} {n}" for t, (n, p) in zip(tns, named)])


def good_days(event: str, start: Date, days: int, lat: float, lon: float, tz_name: str,
              people: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    ev = EVENTS.get(event) or EVENTS["griha_pravesh"]
    tz = zone(tz_name)
    last = start + timedelta(days=days)
    bounds = shared_bounds(jd_of(datetime(start.year, start.month, start.day, tzinfo=tz)) - 0.1,
                           jd_of(datetime(last.year, last.month, last.day, tzinfo=tz)) + 1.5)
    found: List[Dict[str, Any]] = []
    blocked_note: Optional[str] = None
    season_note: Optional[str] = None
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() not in ev["days"]:
            continue
        f = _day_frame(d.year, d.month, d.day, round(lat, 3), round(lon, 3), tz.key)
        wins = _good_windows(f["sunrise"], f["nextRise"], ev, bounds)
        if not wins:
            continue
        why_block = _blocked(f["sunrise"], ev, bounds)
        if why_block:
            blocked_note = blocked_note or why_block
            if season_note is None and _is_season_block(why_block):
                season_note = why_block
            continue
        w0, w1, tns, nks = max(wins, key=lambda w: w[1] - w[0])
        reasons = [_and([NAKSHATRAS[n][0] for n in nks]) + (" nakshatra" if len(nks) == 1 else " nakshatras"),
                   _tithi_words(tns), WEEKDAYS[d.weekday()]]
        score = 3 + (1 if w1 - w0 >= 0.5 else 0)
        personal = []
        mid = (w0 + w1) / 2
        today_rashi = int(moon_lon(mid) // 30.0)
        for person in people or []:
            pr, pn = person.get("rashi"), person.get("nak")
            if pr is None or pn is None:
                continue
            house = (today_rashi - pr) % 12 + 1
            # Every nakshatra of the window counts: one hard tara is enough to say "careful".
            ok = house not in HOUSE_HARD and not any(((nk - pn) % 27) % 9 + 1 in TARA_HARD for nk in nks)
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
                "naks": sorted(nks), "tithis": sorted(tns),
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
            # The first season pause (Kharmas, Adhik maas, Chaturmas) met in the window, even when days were found.
            "seasonNote": season_note,
            "note": "A traditional guide from the panchang. For the exact muhurat of a wedding or griha pravesh, "
                    "confirm the time with your family pandit."}
