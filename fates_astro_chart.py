"""Sidereal (Lahiri) birth chart basics for kundli matching: Moon rashi, nakshatra and pada, lagna,
Mars and Venus. Without a birth time the Moon can change nakshatra during the day, so every nakshatra
the Moon passes through that day is kept as a possibility and matching reports a range.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import swisseph as swe

swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

NAK_SPAN = 360.0 / 27.0
PADA_SPAN = NAK_SPAN / 4.0


def _zone(name: Any) -> ZoneInfo:
    try:
        return ZoneInfo(str(name or "").strip() or "Asia/Kolkata")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("Asia/Kolkata")


def _jd(moment: datetime) -> float:
    u = moment.astimezone(timezone.utc)
    return swe.julday(u.year, u.month, u.day, u.hour + u.minute / 60.0 + u.second / 3600.0)


def _lon(jd: float, planet: int) -> float:
    res, _ = swe.calc_ut(jd, planet, swe.FLG_SWIEPH | swe.FLG_SIDEREAL)
    return float(res[0]) % 360.0


def _parse_date(v: Any) -> Optional[Tuple[int, int, int]]:
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", str(v or ""))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


def _parse_time(v: Any) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(v or ""))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return (h, mi) if h <= 23 and mi <= 59 else None


def moon_facts(moon_lon: float) -> Dict[str, int]:
    nak = int(moon_lon // NAK_SPAN) % 27
    return {"rashi": int(moon_lon // 30.0) % 12, "nak": nak, "pada": int((moon_lon % NAK_SPAN) // PADA_SPAN) + 1}


def chart_from_birth(date: Tuple[int, int, int], time: Optional[Tuple[int, int]],
                     place: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    tz = _zone(place.get("tz") if place else None)
    y, mo, d = date
    if time:
        moment = datetime(y, mo, d, time[0], time[1], tzinfo=tz)
    else:
        moment = datetime(y, mo, d, 12, 0, tzinfo=tz)
    jd = _jd(moment)
    moon = _lon(jd, swe.MOON)
    facts = moon_facts(moon)

    # Every Moon position the day allows when the time is unknown (the Moon moves ~13 deg a day).
    options = [(facts["rashi"], facts["nak"])]
    if not time:
        start = datetime(y, mo, d, 0, 0, tzinfo=tz)
        for step in range(0, 25, 2):
            m2 = moon_facts(_lon(_jd(start + timedelta(hours=step) - timedelta(minutes=1 if step == 24 else 0)), swe.MOON))
            pair = (m2["rashi"], m2["nak"])
            if pair not in options:
                options.append(pair)

    lagna = None
    if time and place and place.get("precision") == "city":
        _, ascmc = swe.houses_ex(jd, float(place["lat"]), float(place["lon"]), b"W", swe.FLG_SIDEREAL)
        lagna = int((float(ascmc[0]) % 360.0) // 30.0)

    mars = _lon(jd, swe.MARS)
    precision = "exact" if time and place and place.get("precision") == "city" else ("time" if time else "date")
    return {
        "moonLon": round(moon, 4), "rashi": facts["rashi"], "nak": facts["nak"], "pada": facts["pada"],
        "options": [list(o) for o in options], "lagna": lagna,
        "marsRashi": int(mars // 30.0) % 12, "venusRashi": int(_lon(jd, swe.VENUS) // 30.0) % 12,
        "timeKnown": time is not None, "precision": precision,
    }


def chart_for_profile(profile: Dict[str, Any],
                      resolve_place: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    date = _parse_date(profile.get("birthDate"))
    if not date:
        return None
    place = None
    try:
        place = resolve_place(profile)
    except Exception:
        place = None
    return chart_from_birth(date, _parse_time(profile.get("birthTime")), place)


def house_from(planet_rashi: int, ref_rashi: Optional[int]) -> Optional[int]:
    if ref_rashi is None:
        return None
    return (planet_rashi - ref_rashi) % 12 + 1
