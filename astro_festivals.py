"""Festivals and fasts for all of India: which days each family keeps, and what Today shows them.

Self-contained data and filtering (engagement design N3). It never imports astro_core, so astro_core (or a later
loader) can read this table without an import cycle: the monthly Hindu vrats come from a function the caller
passes in (astro_core.vrats_between).

The table
- One entry per day and id, tagged with who keeps it: faiths (FAITHS keys), communities (COMMUNITIES keys,
  the mother-tongue groups of the app's "community" field) and states (ISO 3166-2:IN codes without "IN-").
- tier: 'national' (everyone), 'major' (the big days of each faith), 'faith' (the rest of a faith's calendar),
  'regional' (a community's or a state's own days).
- kind: 'festival', 'observance' (a jayanti, a day of remembrance, the start of a holy month) or 'national'.
- Date rule: tithi and solar days follow the panchang (or the region's own rule); a central or state office list
  that differs goes into officeNote. Days that depend on the moon being seen are moonSighting:true and their note
  starts with "Expected, depends on the moon".
- Greetings are faith-correct and left empty where a greeting would be wrong (days of mourning or remembrance).

What Today shows (upcoming): the person's own festivals and fasts as two lists, so a crowded month of vrats can
never push Diwali off the card, the next big day, and a greeting for other faiths' big days today or tomorrow.
legacy_merged keeps the old merged list for 1.0.7 apps; observance_for gives the horoscope and the chat today's
line (TODAY_OBSERVANCE).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

FAITHS = ("hindu", "muslim", "sikh", "christian", "jain", "buddhist", "parsi", "jewish")
TIERS = ("national", "major", "faith", "regional")
KINDS = ("festival", "observance", "national")
TIME_KINDS = ("moonrise", "sunset", "pradosh")
COMMUNITIES = ("hindi", "punjabi", "bengali", "marathi", "gujarati", "tamil", "telugu", "kannada", "malayali",
               "urdu", "rajasthani", "sindhi", "kashmiri", "odia", "assamese", "konkani", "tulu", "maithili",
               "bhojpuri", "manipuri", "dogri", "nepali")
STATES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh",
    "DH": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TG": "Telangana", "TR": "Tripura", "UK": "Uttarakhand",
    "UP": "Uttar Pradesh", "WB": "West Bengal",
}

EXPECTED = "Expected, depends on the moon"   # every moonSighting note starts with it

FESTIVAL_DAYS = 40     # festivals: 40 days ahead,
FESTIVAL_LIMIT = 6     # at most 6 rows
FAST_DAYS = 14         # Hindu monthly vrats: 14 days ahead,
FAST_LIMIT = 4         # at most 4 rows
BIG_DAY_DAYS = 21      # the "Next big day" card looks 21 days ahead

# Monthly Hindu vrats, as astro_core.vrats_between(start, days, lat, lon, tz_name).
VratsFn = Callable[[Date, int, float, float, str], List[Dict[str, Any]]]

_BENGAL = ("WB", "AS", "TR", "OD", "JH")   # where Durga Puja's Bengali days are kept
_NORTH = ("DL", "HR", "PB", "RJ", "UP", "MP", "UK", "HP", "CH", "JK")
_PUNJAB = ("PB", "HR", "HP", "DL", "JK", "CH")
_HINDI_BELT = ("RJ", "UP", "MP", "HR", "DL", "BR", "UK", "JH")
_VAT_SAVITRI = ("UP", "BR", "MP", "JH", "OD", "DL", "HR", "RJ", "UK")


def _e(day: str, id_: str, name: str, note: str = "", greeting: str = "", *, faiths: Tuple[str, ...] = (),
       communities: Tuple[str, ...] = (), states: Tuple[str, ...] = (), tier: str = "faith", kind: str = "festival",
       moon: bool = False, time: Optional[str] = None, office: Optional[str] = None,
       source: str = "panchang") -> Dict[str, Any]:
    """One table entry in the full schema."""
    return {"date": day, "id": id_, "name": name, "note": note, "greeting": greeting, "faiths": list(faiths),
            "communities": list(communities), "states": list(states), "tier": tier, "kind": kind,
            "moonSighting": moon, "timeKind": time, "officeNote": office, "source": source}


_H = ("hindu",)
_NAT = {"tier": "national", "kind": "national"}

# Sources: DoPT O.M. F.No.12/2/2023-JCA of 03.07.2025 (2026 lists) and of 16.07.2026 (2027 lists), read through
# staffnews.in on 2026-10-11 (confirm against the DoPT PDF before release); West Bengal notification
# 4188-F(P2) of 27.11.2025. "panchang" = the app's own panchang rules for New Delhi (udaya tithi, or the tithi in
# aparahna, pradosh, nishita or at moonrise), checked against Drik Panchang. Keep the list sorted by date and
# extend it before it runs out: a test fails 60 days before its last date.
FESTIVALS: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- 2026
    _e("2026-10-11", "navratri", "Sharad Navratri begins", "Nine nights of Durga: ghatasthapana in the morning.",
       "Happy Navratri", faiths=_H),
    _e("2026-10-18", "durga_puja", "Durga Puja (Saptami)", "Pandal days begin in Bengal and beyond.",
       "Happy Durga Puja", faiths=_H, communities=("bengali", "assamese", "odia"), states=_BENGAL, tier="major",
       source="DoPT RH 2026; WB 4188-F(P2)"),
    _e("2026-10-19", "durga_ashtami", "Durga Ashtami", "Maha Ashtami: pushpanjali in the morning, Sandhi puja at the turn of the tithi.",
       "Shubho Ashtami", faiths=_H, communities=("bengali",), states=_BENGAL, tier="regional", source="WB 4188-F(P2)"),
    _e("2026-10-20", "durga_navami", "Durga Navami", "Maha Navami: the last full day of the Puja.",
       "Shubho Navami", faiths=_H, communities=("bengali",), states=_BENGAL, tier="regional", source="WB 4188-F(P2)"),
    _e("2026-10-20", "dussehra", "Dussehra", "Vijayadashami: good wins over evil.", "Happy Dussehra",
       faiths=_H, tier="major", source="DoPT gazetted 2026"),
    _e("2026-10-21", "bijoya_dashami", "Bijoya Dashami", "Sindoor khela, then the immersion: sweets and hugs for everyone.",
       "Shubho Bijoya", faiths=_H, communities=("bengali",), states=_BENGAL, tier="regional", source="WB 4188-F(P2)"),
    _e("2026-10-25", "sharad_purnima", "Sharad Purnima", "Kheer under the full moon.", "Happy Sharad Purnima",
       faiths=_H, time="moonrise"),
    _e("2026-10-25", "kojagari_lakshmi_puja", "Kojagari Lakshmi Puja", "Lakshmi puja at night under the full moon.",
       "Happy Lakshmi Puja", faiths=_H, communities=("bengali",), states=("WB",), tier="regional",
       source="WB 4188-F(P2)"),
    _e("2026-10-26", "valmiki_jayanti", "Valmiki Jayanti", "The birthday of Maharishi Valmiki, who wrote the Ramayana.",
       "Happy Valmiki Jayanti", faiths=_H, kind="observance", source="DoPT RH 2026"),
    _e("2026-10-29", "karva_chauth", "Karva Chauth", "The fast ends at moonrise: see the time for your city.",
       "Happy Karva Chauth", faiths=("hindu", "sikh"), communities=("hindi", "punjabi", "rajasthani"), states=_NORTH,
       tier="regional", time="moonrise", source="DoPT RH 2026"),
    _e("2026-11-06", "dhanteras", "Dhanteras", "Buy something new for the home, light the first diya.",
       "Happy Dhanteras", faiths=_H, time="pradosh"),
    _e("2026-11-08", "diwali", "Diwali", "Lakshmi puja in the evening.", "Shubh Deepavali",
       faiths=("hindu", "jain"), tier="major", time="pradosh", source="DoPT gazetted 2026"),
    _e("2026-11-08", "kali_puja", "Kali Puja", "Kali puja through the night, with diyas on every ledge.",
       "Shubho Kali Puja", faiths=_H, communities=("bengali",), states=("WB", "TR", "AS"), tier="regional",
       source="WB 4188-F(P2)"),
    _e("2026-11-08", "bandi_chhor_divas", "Bandi Chhor Divas", "The release of Guru Hargobind Ji: diyas and kirtan at the gurdwara.",
       "Happy Bandi Chhor Divas", faiths=("sikh",), source="panchang (Diwali day)"),
    _e("2026-11-09", "govardhan", "Govardhan Puja", "Annakut: a mountain of food offered to Krishna.",
       "Happy Govardhan Puja", faiths=_H, source="DoPT RH 2026"),
    _e("2026-11-11", "bhai_dooj", "Bhai Dooj", "Tilak for brothers, blessings for sisters.", "Happy Bhai Dooj",
       faiths=_H),
    _e("2026-11-11", "bhai_phonta", "Bhai Phonta", "Sisters put a phonta on their brothers' foreheads and wish them a long life.",
       "Happy Bhai Phonta", faiths=_H, communities=("bengali",), states=("WB", "TR"), tier="regional",
       source="WB 4188-F(P2)"),
    _e("2026-11-15", "chhath", "Chhath Puja", "Evening arghya to the setting Sun.", "Happy Chhath Puja",
       faiths=_H, communities=("bhojpuri", "maithili", "hindi"), states=("BR", "JH", "UP", "DL"), tier="regional",
       time="sunset"),
    _e("2026-11-20", "dev_uthani", "Dev Uthani Ekadashi", "Tulsi Vivah: the wedding season opens.", faiths=_H),
    _e("2026-11-24", "gurpurab", "Guru Nanak Jayanti", "Kartik Purnima.", "Gurpurab di lakh lakh vadhaiyan",
       faiths=("sikh",), communities=("punjabi",), tier="major", source="DoPT gazetted 2026"),
    _e("2026-11-24", "guru_teg_bahadur", "Guru Teg Bahadur's Martyrdom Day",
       "Shaheedi Divas: the ninth Guru is remembered with kirtan and langar.",
       faiths=("sikh",), kind="observance", source="DoPT RH 2026"),
    _e("2026-12-23", "hazarat_ali", "Hazarat Ali's Birthday", EXPECTED + ".", faiths=("muslim",),
       kind="observance", moon=True, source="DoPT RH 2026"),
    _e("2026-12-24", "christmas_eve", "Christmas Eve", "Carols and midnight Mass.", "Merry Christmas",
       faiths=("christian",), source="DoPT RH 2026"),
    _e("2026-12-25", "christmas", "Christmas", "", "Merry Christmas", faiths=("christian",), tier="major",
       source="DoPT gazetted 2026"),
    # ---------------------------------------------------------------- 2027
    _e("2027-01-01", "new_year", "New Year 2027", "", "Happy New Year", **_NAT, source="calendar"),
    _e("2027-01-14", "lohri", "Lohri", "Bonfire, rewri and gajak.", "Happy Lohri",
       communities=("punjabi", "dogri"), states=_PUNJAB, tier="regional"),
    _e("2027-01-15", "sankranti", "Makar Sankranti", "The Sun enters Makara: til-gud and kites.",
       "Happy Makar Sankranti", faiths=_H, office="Central list: 14 January",
       source="panchang (the Sun enters Makara 14 Jan 21:10 IST)"),
    _e("2027-01-15", "pongal", "Pongal", "Thai Pongal: the new rice boils over in the pot, a sign of plenty.",
       "Happy Pongal", communities=("tamil",), states=("TN", "PY"), tier="regional", source="Tamil rule (Thai 1)"),
    _e("2027-01-15", "guru_gobind_singh", "Guru Gobind Singh Jayanti", "The tenth Guru's Prakash Purab: kirtan and langar.",
       "Gurpurab di lakh lakh vadhaiyan", faiths=("sikh",), source="DoPT RH 2027"),
    _e("2027-01-26", "republic_day", "Republic Day", "", "Happy Republic Day", **_NAT, source="DoPT gazetted 2027"),
    _e("2027-02-09", "ramzan_start", "Ramzan begins", EXPECTED + ": the holy month of fasting begins.", "Ramzan Mubarak",
       faiths=("muslim",), kind="observance", moon=True, source="moon-sighting projection"),
    _e("2027-02-10", "ash_wednesday", "Ash Wednesday", "Lent begins, forty days before Easter.",
       faiths=("christian",), kind="observance", source="Easter - 46 days"),
    _e("2027-02-11", "basant_panchami", "Basant Panchami", "Saraswati puja: wear yellow.", "Happy Basant Panchami",
       faiths=_H),
    _e("2027-02-19", "shivaji_jayanti", "Shivaji Jayanti", "Chhatrapati Shivaji Maharaj's birthday.",
       communities=("marathi",), states=("MH",), tier="regional", kind="observance", source="Maharashtra list"),
    _e("2027-02-20", "guru_ravidas", "Guru Ravidas Jayanti", "The birthday of Sant Guru Ravidas: shabads and processions.",
       "Happy Guru Ravidas Jayanti", faiths=("hindu", "sikh"), communities=("punjabi",),
       states=("PB", "HR", "HP", "CH", "UP"), tier="regional", source="DoPT RH 2027"),
    _e("2027-03-05", "jamat_ul_vida", "Jamat-ul-Vida", EXPECTED + ": the last Friday prayers of Ramzan.",
       faiths=("muslim",), kind="observance", moon=True, source="DoPT RH 2027"),
    _e("2027-03-06", "shivratri", "Maha Shivratri", "Night of Shiva.", "Happy Maha Shivratri", faiths=_H),
    _e("2027-03-10", "eid_ul_fitr", "Eid ul-Fitr", EXPECTED + " (10 or 11 March).", "Eid Mubarak",
       faiths=("muslim",), tier="major", moon=True, source="DoPT gazetted 2027"),
    _e("2027-03-21", "holika_dahan", "Holika Dahan", "The bonfire on the eve of Holi.", "Happy Holika Dahan", faiths=_H,
       source="panchang (Purnima from 21 Mar 18:30 IST)"),
    _e("2027-03-21", "jamshedi_navroz", "Jamshedi Navroz", "The Parsi spring new year.", "Navroz Mubarak",
       faiths=("parsi",), source="calendar (21 March)"),
    _e("2027-03-22", "holi", "Holi", "Colours day.", "Happy Holi", faiths=_H, tier="major",
       office="Central government holiday: 23 March", source="panchang (Purnima ends 22 Mar 16:30 IST)"),
    _e("2027-03-22", "dol_jatra", "Dol Jatra", "Dol Purnima: abir colours and Krishna on a swing.",
       "Happy Dol Jatra", faiths=_H, communities=("bengali", "odia"), states=("WB", "OD", "TR"), tier="regional"),
    _e("2027-03-26", "good_friday", "Good Friday", "A day of prayer and remembrance.", faiths=("christian",),
       tier="major", kind="observance", source="DoPT gazetted 2027"),
    _e("2027-03-28", "easter", "Easter", "Easter Sunday: church, family and a festive meal.", "Happy Easter",
       faiths=("christian",), tier="major", source="DoPT RH 2027"),
    _e("2027-04-07", "ugadi", "Ugadi · Gudi Padwa", "Hindu new year.", "Happy Ugadi and Gudi Padwa", faiths=_H,
       communities=("telugu", "kannada", "marathi", "konkani"), states=("AP", "TG", "KA", "MH", "GA"),
       tier="regional"),
    _e("2027-04-14", "baisakhi", "Baisakhi", "Harvest festival.", "Happy Baisakhi", faiths=("sikh", "hindu"),
       communities=("punjabi",), states=("PB", "HR"), tier="major", source="DoPT RH 2027"),
    _e("2027-04-14", "puthandu", "Puthandu", "Tamil New Year: a tray of fruit, flowers and gold to see first thing.",
       "Happy Puthandu", communities=("tamil",), states=("TN", "PY"), tier="regional",
       source="Tamil rule (sankranti 15:28 IST, before sunset)"),
    _e("2027-04-14", "ambedkar_jayanti", "Ambedkar Jayanti", "Dr B. R. Ambedkar's birthday.", tier="national",
       kind="observance", source="DoPT 2027"),
    _e("2027-04-15", "ram_navami", "Ram Navami", "Lord Rama's birthday.", "Happy Ram Navami", faiths=_H),
    _e("2027-04-15", "poila_boishakh", "Poila Boishakh", "Bengali New Year: new clothes, and shops open a new halkhata.",
       "Shubho Noboborsho", communities=("bengali",), states=("WB", "TR", "AS"), tier="regional",
       source="DoPT RH 2027"),
    _e("2027-04-15", "bohag_bihu", "Bohag Bihu", "Rongali Bihu: the Assamese New Year, with dhol and Bihu dance.",
       "Happy Bohag Bihu", communities=("assamese",), states=("AS",), tier="regional", source="DoPT RH 2027"),
    _e("2027-04-19", "mahavir_jayanti", "Mahavir Jayanti", "The birth of Bhagwan Mahavir.", "Happy Mahavir Jayanti",
       faiths=("jain",), tier="major", source="DoPT gazetted 2027"),
    _e("2027-04-20", "hanuman_jayanti", "Hanuman Jayanti", "Chaitra Purnima: Hanuman Chalisa in many homes.",
       "Happy Hanuman Jayanti", faiths=_H),
    _e("2027-05-09", "akshaya_tritiya", "Akshaya Tritiya", "A lucky day for gold and new beginnings.",
       "Happy Akshaya Tritiya", faiths=_H),
    _e("2027-05-09", "tagore_jayanti", "Rabindra Jayanti", "Rabindranath Tagore's birthday: his songs and poems everywhere.",
       communities=("bengali",), states=("WB", "TR"), tier="regional", kind="observance", source="DoPT RH 2027"),
    _e("2027-05-17", "bakrid", "Eid ul-Adha (Bakrid)", EXPECTED + ".", "Eid Mubarak", faiths=("muslim",),
       tier="major", moon=True, source="DoPT gazetted 2027"),
    _e("2027-05-20", "buddha_purnima", "Buddha Purnima", "Vaishakha Purnima.", "Happy Buddha Purnima",
       faiths=("buddhist",), tier="major", source="DoPT gazetted 2027"),
    _e("2027-06-04", "vat_savitri", "Vat Savitri", "Wives pray at the banyan tree for their husband's long life.",
       faiths=_H, communities=("hindi", "maithili", "bhojpuri", "odia"), states=_VAT_SAVITRI, tier="regional"),
    _e("2027-06-16", "muharram", "Muharram (Ashura)", EXPECTED + ": a day of remembrance.", faiths=("muslim",),
       kind="observance", moon=True, source="DoPT gazetted 2027"),
    _e("2027-07-05", "rath_yatra", "Rath Yatra", "Lord Jagannath's chariots roll through Puri.", "Happy Rath Yatra",
       faiths=_H, communities=("odia",), states=("OD",), tier="regional", source="DoPT RH 2027"),
    _e("2027-07-18", "guru_purnima", "Guru Purnima", "Thank your teachers and elders.", "Happy Guru Purnima",
       faiths=_H),
    _e("2027-08-04", "hariyali_teej", "Hariyali Teej", "Green bangles, swings and mehndi.", "Happy Teej", faiths=_H,
       communities=("hindi", "rajasthani"), states=_HINDI_BELT, tier="regional"),
    _e("2027-08-13", "varalakshmi_vratam", "Varalakshmi Vratam",
       "Married women pray to Goddess Lakshmi for the family's well-being.", "Happy Varalakshmi Vratam", faiths=_H,
       communities=("telugu", "kannada", "tamil"), states=("AP", "TG", "KA", "TN"), tier="regional",
       source="panchang (the Friday before Shravana Purnima)"),
    _e("2027-08-15", "independence_day", "Independence Day", "", "Happy Independence Day", **_NAT,
       source="DoPT gazetted 2027"),
    _e("2027-08-15", "milad_un_nabi", "Milad-un-Nabi", EXPECTED + ": the Prophet's birthday.",
       "Eid Milad-un-Nabi Mubarak", faiths=("muslim",), moon=True, source="DoPT gazetted 2027"),
    _e("2027-08-15", "parsi_navroz", "Parsi New Year (Navroz)", "The Parsi New Year: the fire temple, then a family feast.",
       "Navroz Mubarak", faiths=("parsi",), tier="major", source="DoPT RH 2027"),
    _e("2027-08-17", "raksha_bandhan", "Raksha Bandhan", "Rakhi for brothers, blessings and gifts for sisters.",
       "Happy Raksha Bandhan", faiths=_H),
    _e("2027-08-25", "janmashtami", "Janmashtami", "Krishna's birthday: the fast ends at midnight.",
       "Happy Janmashtami", faiths=_H),
    _e("2027-09-04", "ganesh_chaturthi", "Ganesh Chaturthi", "Ganpati comes home.", "Happy Ganesh Chaturthi",
       faiths=_H),
    _e("2027-09-12", "onam", "Onam", "Thiruvonam: a pookalam at the door and sadya on a banana leaf.", "Happy Onam",
       communities=("malayali",), states=("KL",), tier="regional", source="DoPT RH 2027"),
    _e("2027-09-29", "mahalaya", "Mahalaya Amavasya", "Pitru Paksha ends: a day to remember the ancestors.",
       faiths=_H, kind="observance"),
    _e("2027-09-30", "navratri", "Sharad Navratri begins", "Nine nights of Durga: ghatasthapana in the morning.",
       "Happy Navratri", faiths=_H),
    _e("2027-10-02", "gandhi_jayanti", "Gandhi Jayanti", "Mahatma Gandhi's birthday.", **_NAT,
       source="DoPT gazetted 2027"),
    _e("2027-10-06", "durga_puja", "Durga Puja (Saptami)", "Pandal days begin in Bengal and beyond.",
       "Happy Durga Puja", faiths=_H, communities=("bengali", "assamese", "odia"), states=_BENGAL, tier="major",
       source="DoPT RH 2027"),
    _e("2027-10-07", "durga_ashtami", "Durga Ashtami", "Maha Ashtami: pushpanjali in the morning, Sandhi puja at the turn of the tithi.",
       "Shubho Ashtami", faiths=_H, communities=("bengali",), states=_BENGAL, tier="regional", source="DoPT RH 2027"),
    _e("2027-10-08", "durga_navami", "Durga Navami", "Maha Navami: the last full day of the Puja.",
       "Shubho Navami", faiths=_H, communities=("bengali",), states=_BENGAL, tier="regional", source="DoPT RH 2027"),
    _e("2027-10-09", "dussehra", "Dussehra", "Vijayadashami: good wins over evil.", "Happy Dussehra",
       faiths=_H, tier="major", source="DoPT gazetted 2027"),
    _e("2027-10-14", "sharad_purnima", "Sharad Purnima", "Kheer under the full moon.", "Happy Sharad Purnima",
       faiths=_H, time="moonrise"),
    _e("2027-10-15", "valmiki_jayanti", "Valmiki Jayanti", "The birthday of Maharishi Valmiki, who wrote the Ramayana.",
       "Happy Valmiki Jayanti", faiths=_H, kind="observance", source="DoPT RH 2027"),
    _e("2027-10-18", "karva_chauth", "Karva Chauth", "The fast ends at moonrise: see the time for your city.",
       "Happy Karva Chauth", faiths=("hindu", "sikh"), communities=("hindi", "punjabi", "rajasthani"), states=_NORTH,
       tier="regional", time="moonrise", source="DoPT RH 2027"),
    _e("2027-10-27", "dhanteras", "Dhanteras", "Buy something new for the home, light the first diya.",
       "Happy Dhanteras", faiths=_H, time="pradosh"),
    _e("2027-10-28", "naraka_chaturdashi", "Deepavali (Naraka Chaturdashi)",
       "Tamil Deepavali: an oil bath before dawn, new clothes and sweets.", "Happy Deepavali", faiths=_H,
       communities=("tamil",), states=("TN", "PY"), tier="regional", source="DoPT RH 2027"),
    _e("2027-10-29", "diwali", "Diwali", "Lakshmi puja in the evening.", "Shubh Deepavali",
       faiths=("hindu", "jain"), tier="major", time="pradosh", source="DoPT gazetted 2027"),
    _e("2027-10-29", "bandi_chhor_divas", "Bandi Chhor Divas", "The release of Guru Hargobind Ji: diyas and kirtan at the gurdwara.",
       "Happy Bandi Chhor Divas", faiths=("sikh",), source="panchang (Diwali day)"),
    _e("2027-10-30", "govardhan", "Govardhan Puja", "Annakut: a mountain of food offered to Krishna.",
       "Happy Govardhan Puja", faiths=_H, source="DoPT RH 2027"),
    _e("2027-10-31", "bhai_dooj", "Bhai Dooj", "Tilak for brothers, blessings for sisters.", "Happy Bhai Dooj",
       faiths=_H),
    _e("2027-11-04", "chhath", "Chhath Puja", "Evening arghya to the setting Sun.", "Happy Chhath Puja",
       faiths=_H, communities=("bhojpuri", "maithili", "hindi"), states=("BR", "JH", "UP", "DL"), tier="regional",
       time="sunset"),
    _e("2027-11-10", "dev_uthani", "Dev Uthani Ekadashi", "Tulsi Vivah: the wedding season opens.", faiths=_H),
    _e("2027-11-14", "gurpurab", "Guru Nanak Jayanti", "Kartik Purnima.", "Gurpurab di lakh lakh vadhaiyan",
       faiths=("sikh",), communities=("punjabi",), tier="major", source="DoPT gazetted 2027"),
    _e("2027-11-24", "guru_teg_bahadur", "Guru Teg Bahadur's Martyrdom Day",
       "Shaheedi Divas: the ninth Guru is remembered with kirtan and langar.",
       faiths=("sikh",), kind="observance", source="DoPT RH 2027"),
    _e("2027-12-12", "hazarat_ali", "Hazarat Ali's Birthday", EXPECTED + ".", faiths=("muslim",),
       kind="observance", moon=True, source="DoPT RH 2027"),
    _e("2027-12-24", "christmas_eve", "Christmas Eve", "Carols and midnight Mass.", "Merry Christmas",
       faiths=("christian",), source="DoPT RH 2027"),
    _e("2027-12-25", "christmas", "Christmas", "", "Merry Christmas", faiths=("christian",), tier="major",
       source="DoPT gazetted 2027"),
    # ---------------------------------------------------------------- 2028 (panchang; office lists not out yet)
    _e("2028-01-01", "new_year", "New Year 2028", "", "Happy New Year", **_NAT, source="calendar"),
    _e("2028-01-14", "lohri", "Lohri", "Bonfire, rewri and gajak.", "Happy Lohri",
       communities=("punjabi", "dogri"), states=_PUNJAB, tier="regional"),
    _e("2028-01-15", "sankranti", "Makar Sankranti", "The Sun enters Makara: til-gud and kites.",
       "Happy Makar Sankranti", faiths=_H, source="panchang (the Sun enters Makara 15 Jan 03:24 IST)"),
    _e("2028-01-15", "pongal", "Pongal", "Thai Pongal: the new rice boils over in the pot, a sign of plenty.",
       "Happy Pongal", communities=("tamil",), states=("TN", "PY"), tier="regional", source="Tamil rule (Thai 1)"),
    _e("2028-01-26", "republic_day", "Republic Day", "", "Happy Republic Day", **_NAT, source="calendar"),
    _e("2028-01-31", "basant_panchami", "Basant Panchami", "Saraswati puja: wear yellow.", "Happy Basant Panchami",
       faiths=_H),
    _e("2028-02-23", "shivratri", "Maha Shivratri", "Night of Shiva.", "Happy Maha Shivratri", faiths=_H),
    _e("2028-03-01", "ash_wednesday", "Ash Wednesday", "Lent begins, forty days before Easter.",
       faiths=("christian",), kind="observance", source="Easter - 46 days"),
    _e("2028-03-10", "holika_dahan", "Holika Dahan", "The bonfire on the eve of Holi.", "Happy Holika Dahan", faiths=_H),
    _e("2028-03-11", "holi", "Holi", "Colours day.", "Happy Holi", faiths=_H, tier="major"),
    _e("2028-03-27", "ugadi", "Ugadi · Gudi Padwa", "Hindu new year.", "Happy Ugadi and Gudi Padwa", faiths=_H,
       communities=("telugu", "kannada", "marathi", "konkani"), states=("AP", "TG", "KA", "MH", "GA"),
       tier="regional"),
    _e("2028-04-03", "ram_navami", "Ram Navami", "Lord Rama's birthday.", "Happy Ram Navami", faiths=_H),
    _e("2028-04-07", "mahavir_jayanti", "Mahavir Jayanti", "The birth of Bhagwan Mahavir.", "Happy Mahavir Jayanti",
       faiths=("jain",), tier="major"),
    _e("2028-04-09", "hanuman_jayanti", "Hanuman Jayanti", "Chaitra Purnima: Hanuman Chalisa in many homes.",
       "Happy Hanuman Jayanti", faiths=_H),
    _e("2028-04-13", "baisakhi", "Baisakhi", "Harvest festival.", "Happy Baisakhi", faiths=("sikh", "hindu"),
       communities=("punjabi",), states=("PB", "HR"), tier="major"),
    _e("2028-04-14", "good_friday", "Good Friday", "A day of prayer and remembrance.", faiths=("christian",),
       tier="major", kind="observance", source="computus (Easter - 2 days)"),
    _e("2028-04-14", "ambedkar_jayanti", "Ambedkar Jayanti", "Dr B. R. Ambedkar's birthday.", tier="national",
       kind="observance", source="calendar (14 April)"),
    _e("2028-04-16", "easter", "Easter", "Easter Sunday: church, family and a festive meal.", "Happy Easter",
       faiths=("christian",), tier="major", source="computus"),
    _e("2028-04-27", "akshaya_tritiya", "Akshaya Tritiya", "A lucky day for gold and new beginnings.",
       "Happy Akshaya Tritiya", faiths=_H),
    _e("2028-05-08", "buddha_purnima", "Buddha Purnima", "Vaishakha Purnima.", "Happy Buddha Purnima",
       faiths=("buddhist",), tier="major"),
    _e("2028-05-24", "vat_savitri", "Vat Savitri", "Wives pray at the banyan tree for their husband's long life.",
       faiths=_H, communities=("hindi", "maithili", "bhojpuri", "odia"), states=_VAT_SAVITRI, tier="regional"),
    _e("2028-07-06", "guru_purnima", "Guru Purnima", "Thank your teachers and elders.", "Happy Guru Purnima",
       faiths=_H),
    _e("2028-07-24", "hariyali_teej", "Hariyali Teej", "Green bangles, swings and mehndi.", "Happy Teej", faiths=_H,
       communities=("hindi", "rajasthani"), states=_HINDI_BELT, tier="regional"),
    _e("2028-08-04", "varalakshmi_vratam", "Varalakshmi Vratam",
       "Married women pray to Goddess Lakshmi for the family's well-being.", "Happy Varalakshmi Vratam", faiths=_H,
       communities=("telugu", "kannada", "tamil"), states=("AP", "TG", "KA", "TN"), tier="regional",
       source="panchang (the Friday before Shravana Purnima)"),
    _e("2028-08-05", "raksha_bandhan", "Raksha Bandhan", "Rakhi for brothers, blessings and gifts for sisters.",
       "Happy Raksha Bandhan", faiths=_H),
    _e("2028-08-13", "janmashtami", "Janmashtami", "Krishna's birthday: the fast ends at midnight.",
       "Happy Janmashtami", faiths=_H),
    _e("2028-08-15", "independence_day", "Independence Day", "", "Happy Independence Day", **_NAT,
       source="calendar"),
    _e("2028-08-23", "ganesh_chaturthi", "Ganesh Chaturthi", "Ganpati comes home.", "Happy Ganesh Chaturthi",
       faiths=_H),
    _e("2028-09-18", "mahalaya", "Mahalaya Amavasya", "Pitru Paksha ends: a day to remember the ancestors.",
       faiths=_H, kind="observance"),
    _e("2028-09-19", "navratri", "Sharad Navratri begins", "Nine nights of Durga: ghatasthapana in the morning.",
       "Happy Navratri", faiths=_H),
    _e("2028-09-25", "durga_puja", "Durga Puja (Saptami)", "Pandal days begin in Bengal and beyond.",
       "Happy Durga Puja", faiths=_H, communities=("bengali", "assamese", "odia"), states=_BENGAL, tier="major",
       source="panchang (Saptami at Kolkata sunrise)"),
    _e("2028-09-27", "dussehra", "Dussehra", "Vijayadashami: good wins over evil.", "Happy Dussehra",
       faiths=_H, tier="major"),
    _e("2028-10-02", "sharad_purnima", "Sharad Purnima", "Kheer under the full moon.", "Happy Sharad Purnima",
       faiths=_H, time="moonrise"),
    _e("2028-10-02", "gandhi_jayanti", "Gandhi Jayanti", "Mahatma Gandhi's birthday.", **_NAT,
       source="calendar (2 October)"),
    _e("2028-10-07", "karva_chauth", "Karva Chauth", "The fast ends at moonrise: see the time for your city.",
       "Happy Karva Chauth", faiths=("hindu", "sikh"), communities=("hindi", "punjabi", "rajasthani"), states=_NORTH,
       tier="regional", time="moonrise"),
    _e("2028-10-15", "dhanteras", "Dhanteras", "Buy something new for the home, light the first diya.",
       "Happy Dhanteras", faiths=_H, time="pradosh"),
    _e("2028-10-17", "diwali", "Diwali", "Lakshmi puja in the evening.", "Shubh Deepavali",
       faiths=("hindu", "jain"), tier="major", time="pradosh"),
    _e("2028-10-17", "bandi_chhor_divas", "Bandi Chhor Divas", "The release of Guru Hargobind Ji: diyas and kirtan at the gurdwara.",
       "Happy Bandi Chhor Divas", faiths=("sikh",), source="panchang (Diwali day)"),
    _e("2028-10-19", "bhai_dooj", "Bhai Dooj", "Tilak for brothers, blessings for sisters.", "Happy Bhai Dooj",
       faiths=_H),
    _e("2028-10-23", "chhath", "Chhath Puja", "Evening arghya to the setting Sun.", "Happy Chhath Puja",
       faiths=_H, communities=("bhojpuri", "maithili", "hindi"), states=("BR", "JH", "UP", "DL"), tier="regional",
       time="sunset"),
    _e("2028-10-28", "dev_uthani", "Dev Uthani Ekadashi", "Tulsi Vivah: the wedding season opens.", faiths=_H),
    _e("2028-11-02", "gurpurab", "Guru Nanak Jayanti", "Kartik Purnima.", "Gurpurab di lakh lakh vadhaiyan",
       faiths=("sikh",), communities=("punjabi",), tier="major"),
    _e("2028-12-24", "christmas_eve", "Christmas Eve", "Carols and midnight Mass.", "Merry Christmas",
       faiths=("christian",), source="calendar"),
    _e("2028-12-25", "christmas", "Christmas", "", "Merry Christmas", faiths=("christian",), tier="major",
       source="calendar"),
]

# Worked out, but shown only once the state's own 2027 list confirms the date: then move the entry into FESTIVALS.
AWAITING_CONFIRMATION: List[Dict[str, Any]] = [
    # Mesha sankranti is 14 Apr 15:28 IST, after 3/5 of the day, so the Malayalam rule gives 15 April (as in 2023).
    _e("2027-04-15", "vishu", "Vishu", "Vishukkani at dawn, then the sadya.", "Happy Vishu", faiths=_H,
       communities=("malayali",), states=("KL",), tier="regional", source="Malayalam rule; awaiting the Kerala 2027 list"),
    # The Sun enters Makara 14 Jan 21:10 IST: Magh Bihu is 15 January unless the Assam 2027 list says otherwise.
    _e("2027-01-15", "magh_bihu", "Magh Bihu", "Bhogali Bihu: the harvest feast after the Uruka night.",
       "Happy Magh Bihu", communities=("assamese",), states=("AS",), tier="regional",
       source="Assamese rule; awaiting the Assam 2027 list"),
]

# Festivals that already are the day's vrat: the generic vrat within a day of them is left out.
FESTIVAL_COVERS: Dict[str, FrozenSet[str]] = {
    "dev_uthani": frozenset({"ekadashi"}), "sharad_purnima": frozenset({"purnima"}),
    "gurpurab": frozenset({"purnima"}), "diwali": frozenset({"amavasya"}), "karva_chauth": frozenset({"sankashti"}),
    "holika_dahan": frozenset({"purnima"}), "shivratri": frozenset({"pradosh"}),
    "hanuman_jayanti": frozenset({"purnima"}), "buddha_purnima": frozenset({"purnima"}),
    "guru_purnima": frozenset({"purnima"}), "raksha_bandhan": frozenset({"purnima"}),
    "vat_savitri": frozenset({"amavasya"}), "mahalaya": frozenset({"amavasya"}),
    # days added with this table
    "kojagari_lakshmi_puja": frozenset({"purnima"}), "valmiki_jayanti": frozenset({"purnima"}),
    "kali_puja": frozenset({"amavasya"}), "dol_jatra": frozenset({"purnima"}),
}

# On the same day the regional name stands for the pan-Indian one: a Tamil family sees Pongal, not Makar
# Sankranti as well; a Bengali family sees Bhai Phonta, not Bhai Dooj as well.
SAME_DAY_SUPERSEDES: Dict[str, Tuple[str, ...]] = {"pongal": ("sankranti",), "bhai_phonta": ("bhai_dooj",)}

# The ids of the old single list (astro_core.FESTIVALS) that 1.0.7 apps read: legacy_merged serves only these.
LEGACY_IDS: FrozenSet[str] = frozenset({
    "navratri", "durga_puja", "dussehra", "sharad_purnima", "karva_chauth", "dhanteras", "diwali", "bhai_dooj",
    "chhath", "dev_uthani", "gurpurab", "christmas", "new_year", "lohri", "sankranti", "republic_day",
    "basant_panchami", "shivratri", "eid_ul_fitr", "holika_dahan", "holi", "ugadi", "baisakhi", "ram_navami",
    "mahavir_jayanti", "hanuman_jayanti", "akshaya_tritiya", "bakrid", "buddha_purnima", "vat_savitri",
    "guru_purnima", "hariyali_teej", "independence_day", "raksha_bandhan", "janmashtami", "ganesh_chaturthi",
    "mahalaya",
})

_TIER_RANK = {"major": 3, "faith": 2, "regional": 2, "national": 1}
_VRAT_TIME = {"pradosh": "pradosh", "sankashti": "moonrise"}

_DAY_CACHE: Dict[str, Date] = {}


def _day(e: Mapping[str, Any]) -> Date:
    s = e["date"]
    d = _DAY_CACHE.get(s)
    if d is None:
        d = _DAY_CACHE[s] = Date.fromisoformat(s)
    return d


# ------------------------------------------------------------------ who the person is
_FAITH_WORDS = {"hindu": "hindu", "hinduism": "hindu", "muslim": "muslim", "islam": "muslim", "sikh": "sikh",
                "sikhism": "sikh", "christian": "christian", "christianity": "christian", "jain": "jain",
                "jainism": "jain", "buddhist": "buddhist", "buddhism": "buddhist", "parsi": "parsi",
                "zoroastrian": "parsi", "jewish": "jewish", "judaism": "jewish"}
_ALL_WORDS = {"all", "all of india", "all_india", "all-india", "allindia"}

_COMMUNITY_WORDS = {
    "hindi belt": "hindi", "hindi": "hindi", "north indian": "hindi", "punjabi": "punjabi", "bengali": "bengali",
    "bangla": "bengali", "marathi": "marathi", "gujarati": "gujarati", "tamil": "tamil", "telugu": "telugu",
    "kannada": "kannada", "malayalam": "malayali", "malayali": "malayali", "urdu": "urdu",
    "rajasthani": "rajasthani", "sindhi": "sindhi", "kashmiri": "kashmiri", "odia": "odia", "oriya": "odia",
    "assamese": "assamese", "asomiya": "assamese", "konkani": "konkani", "tulu": "tulu", "maithili": "maithili",
    "bhojpuri": "bhojpuri", "manipuri": "manipuri", "meitei": "manipuri", "dogri": "dogri", "nepali": "nepali",
}

_STATE_ALIASES = {
    "nct of delhi": "DL", "national capital territory of delhi": "DL", "new delhi": "DL", "orissa": "OD",
    "pondicherry": "PY", "uttaranchal": "UK", "jammu & kashmir": "JK", "andaman & nicobar islands": "AN",
    "andaman and nicobar": "AN", "dadra and nagar haveli": "DH", "daman and diu": "DH",
    # older ISO 3166-2:IN codes
    "or": "OD", "ct": "CG", "ut": "UK", "ts": "TG", "dd": "DH", "dn": "DH",
}
_STATE_BY_NAME = {name.lower(): code for code, name in STATES.items()}


def faith_key(value: Any) -> Optional[str]:
    """A FAITHS key from a key or a religion as the app shows it ("Parsi / Zoroastrian"), "all" for All of India,
    else None ("No religion / Spiritual", "Prefer not to say", anything unknown)."""
    s = " ".join(str(value or "").strip().lower().split())
    if not s:
        return None
    if s in _ALL_WORDS:
        return "all"
    for part in s.replace("/", " ").replace(",", " ").split():
        if part in _FAITH_WORDS:
            return _FAITH_WORDS[part]
    return None


def community_keys(value: Any) -> FrozenSet[str]:
    """COMMUNITIES keys from a community as the app stores it ("Hindi belt / North Indian", "Bhojpuri / Maithili")."""
    s = " ".join(str(value or "").strip().lower().split())
    if not s:
        return frozenset()
    if s in _COMMUNITY_WORDS:
        return frozenset({_COMMUNITY_WORDS[s]})
    out = set()
    for part in s.replace(",", "/").replace(" and ", "/").split("/"):
        k = _COMMUNITY_WORDS.get(part.strip())
        if k:
            out.add(k)
    return frozenset(out)


def state_code(value: Any) -> Optional[str]:
    """A STATES code from a code ("WB", "IN-WB") or a state name as geocoders and the app write it."""
    s = " ".join(str(value or "").strip().lower().replace("state of ", "").split())
    if not s:
        return None
    if s.startswith("in-"):
        s = s[3:]
    if s.upper() in STATES:
        return s.upper()
    if s in _STATE_ALIASES:
        return _STATE_ALIASES[s]
    s = s.replace("&", "and")
    return _STATE_BY_NAME.get(s) or _STATE_ALIASES.get(s)


@dataclass(frozen=True)
class Prefs:
    """What the festival lists are picked by."""
    faiths: FrozenSet[str] = frozenset()      # FAITHS keys
    all_india: bool = False                   # "All of India" chosen
    communities: FrozenSet[str] = frozenset()
    community: str = ""                       # as stored, for "Showing: …"
    state: Optional[str] = None               # STATES code
    source: str = "unknown"                   # 'chosen' | 'religion' | 'unknown'
    astro_mode: str = ""

    @property
    def all_majors(self) -> bool:
        """Unknown faith, or All of India: every faith's big days are shown."""
        return self.all_india or not self.faiths


def read_prefs(prefs: Optional[Mapping[str, Any]] = None) -> Prefs:
    """Prefs from the caller's merged profile (astro_api _Me.merged) or a plain dict with the same keys:
    celebrates (list of faith keys, or religions as the app shows them; "all" = All of India), religion,
    community, stateName (home, else birthStateName), astroMode."""
    p = prefs or {}
    raw = p.get("celebrates")
    if isinstance(raw, str):
        raw = [raw]
    keys = [faith_key(x) for x in raw] if isinstance(raw, (list, tuple)) else []
    faiths = frozenset(k for k in keys if k and k != "all")
    all_india = "all" in keys
    source = "chosen" if faiths or all_india else "unknown"
    if source == "unknown":
        k = faith_key(p.get("religion"))
        if k and k != "all":
            faiths, source = frozenset({k}), "religion"
    community = " ".join(str(p.get("community") or "").split())[:40]
    state = state_code(p.get("stateName") or p.get("state")) or state_code(p.get("birthStateName"))
    return Prefs(faiths=faiths, all_india=all_india, communities=community_keys(community), community=community,
                 state=state, source=source, astro_mode=str(p.get("astroMode") or ""))


def _as_prefs(prefs: Any) -> Prefs:
    return prefs if isinstance(prefs, Prefs) else read_prefs(prefs)


# ------------------------------------------------------------------ the filter
def _region_hit(e: Mapping[str, Any], me: Prefs) -> bool:
    return bool(me.communities.intersection(e["communities"])) or (me.state is not None and me.state in e["states"])


def shown(e: Mapping[str, Any], me: Prefs) -> bool:
    """Whether a person keeps this day.
    - National days: everyone.
    - A day of the person's own faith, unless it is a regional day of another region.
    - A day of their community or state, when it belongs to their faith or to no faith in particular (Onam,
      Pongal, Poila Boishakh): a Bengali Muslim family is not handed Kali Puja.
    - Unknown faith or All of India: national days plus every faith's big days."""
    tier = e["tier"]
    if tier == "national":
        return True
    faiths = e["faiths"]
    own = bool(me.faiths.intersection(faiths))
    if (own or not faiths) and _region_hit(e, me):
        return True
    if tier == "major":
        return own or me.all_majors
    return own and tier == "faith"


def _region_fit(e: Mapping[str, Any], me: Prefs) -> int:
    """2: the person's own region; 1: a day for all of India; 0: a day tagged for other regions."""
    if _region_hit(e, me):
        return 2
    return 0 if e["communities"] or e["states"] else 1


def _drop_superseded(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gone = {(e["date"], old) for e in items for old in SAME_DAY_SUPERSEDES.get(e["id"], ())}
    return [e for e in items if (e["date"], e["id"]) not in gone] if gone else items


def _mine(start: Date, end: Date, me: Prefs) -> List[Dict[str, Any]]:
    """The person's days in [start, end), table order (by date)."""
    return _drop_superseded([e for e in FESTIVALS if start <= _day(e) < end and shown(e, me)])


def _row(e: Mapping[str, Any], start: Date) -> Dict[str, Any]:
    """A festival as the app reads it (UpcomingDto). kind stays the festival id, as 1.0.7 reads it; the entry's
    own kind goes out as category."""
    return {"date": e["date"], "id": e["id"], "kind": e["id"], "type": "festival", "name": e["name"],
            "note": e["note"], "greeting": e["greeting"], "tier": e["tier"], "category": e["kind"],
            "faiths": list(e["faiths"]), "communities": list(e["communities"]), "states": list(e["states"]),
            "tentative": bool(e["moonSighting"]), "timeKind": e["timeKind"], "officeNote": e["officeNote"],
            "inDays": (_day(e) - start).days}


def _covered(v: Mapping[str, Any], fests: Iterable[Mapping[str, Any]]) -> bool:
    d = Date.fromisoformat(v["date"])
    return any(v["kind"] in FESTIVAL_COVERS.get(f["id"], ()) and abs((d - _day(f)).days) <= 1 for f in fests)


def _big_day(mine: List[Dict[str, Any]], start: Date, me: Prefs) -> Optional[Dict[str, Any]]:
    """The person's biggest festival in the next BIG_DAY_DAYS days (today included): the highest tier, then their
    own region over all-India over other regions, then the sooner.
    - Only their own days count: their faith's, their region's and national ones (every faith's majors only when
      the faith is unknown), so "All of India" never makes another faith's day their big day.
    - A major day tagged for other regions (Durga Puja for a family in Delhi) ranks like a faith day.
    - Observances (jayantis, days of remembrance) are not big days."""
    end = start + timedelta(days=BIG_DAY_DAYS)
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[int, int, int]] = None
    for e in mine:
        d = _day(e)
        if not (start <= d < end) or e["kind"] == "observance":
            continue
        fit = _region_fit(e, me)
        own = bool(me.faiths.intersection(e["faiths"])) or fit == 2 or e["tier"] == "national"
        if not own and not (not me.faiths and e["tier"] == "major"):
            continue
        rank = _TIER_RANK[e["tier"]] - (1 if e["tier"] == "major" and fit == 0 else 0)
        key = (rank, fit, -(d - start).days)
        if best_key is None or key > best_key:
            best, best_key = e, key
    return best


def _pick(mine: List[Dict[str, Any]], start: Date, limit: int, big: Optional[Dict[str, Any]],
          me: Prefs) -> List[Dict[str, Any]]:
    """At most limit rows, in date order. When there are more, today's days come first, then the big days (every
    major and the next big day), then their own community's days, then festivals before observances, the
    soonest first: Diwali is never crowded out."""
    if len(mine) <= limit:
        return mine

    def rank(e: Dict[str, Any]) -> Tuple[int, Date]:
        if _day(e) == start:
            r = 0
        elif e["tier"] == "major" or e is big:
            r = 1
        elif me.communities.intersection(e["communities"]):
            r = 2
        elif e["kind"] != "observance":
            r = 3
        else:
            r = 4
        return r, _day(e)

    keep = {id(e) for e in sorted(mine, key=rank)[:max(limit, 0)]}
    return [e for e in mine if id(e) in keep]


def _fast_row(v: Dict[str, Any], start: Date) -> Dict[str, Any]:
    return {**v, "id": v["kind"], "greeting": "", "tier": "faith", "category": "observance", "faiths": ["hindu"],
            "communities": [], "states": [], "tentative": False, "timeKind": _VRAT_TIME.get(v["kind"]),
            "officeNote": None, "inDays": (Date.fromisoformat(v["date"]) - start).days}


def showing(me: Prefs) -> Dict[str, Any]:
    """What the lists are picked by, for "Showing: Hindu · Bengali · West Bengal · Change"."""
    faiths = [f for f in FAITHS if f in me.faiths] + (["all"] if me.all_india else [])
    return {"faiths": faiths, "community": me.community, "state": STATES.get(me.state or "", ""),
            "source": me.source}


# ------------------------------------------------------------------ public
def upcoming(start: Date, days: int, lat: float, lon: float, tz_name: str,
             prefs: Optional[Mapping[str, Any]] = None, limit: int = FESTIVAL_LIMIT, *,
             vrats: Optional[VratsFn]) -> Dict[str, Any]:
    """Today's lists for one person.

    - festivals: their days in the next `days` days (FESTIVAL_DAYS on Today), at most `limit`, date order.
    - fasts: the Hindu monthly vrats in the next FAST_DAYS days (at most FAST_LIMIT), only when they keep Hindu
      days; none for an unknown faith. A vrat that one of their festivals already is (FESTIVAL_COVERS) is left out.
    - bigDay: their biggest festival in the next BIG_DAY_DAYS days, today included, or None. It is also in
      festivals when inside the window: the app skips that row by id and date.
    - alsoInIndia: other faiths' big days today or tomorrow that carry a greeting, for one quiet greeting row.
    - showing: what the lists are picked by (see read_prefs).

    vrats is the monthly-vrat function (astro_core.vrats_between); with None there are no fasts. The place only
    matters for the vrats (sunrise, moonrise)."""
    me = _as_prefs(prefs)
    days = max(0, int(days))
    horizon = max(days, BIG_DAY_DAYS, FAST_DAYS) + 1
    mine = _mine(start - timedelta(days=1), start + timedelta(days=horizon), me)
    ahead = [e for e in mine if start <= _day(e) < start + timedelta(days=days)]
    big = _big_day(mine, start, me)
    festivals = [_row(e, start) for e in _pick(ahead, start, limit, big, me)]

    fasts: List[Dict[str, Any]] = []
    if "hindu" in me.faiths and vrats is not None:
        for v in vrats(start, FAST_DAYS, lat, lon, tz_name):
            if not _covered(v, mine):
                fasts.append(_fast_row(v, start))
            if len(fasts) >= FAST_LIMIT:
                break

    soon = (start, start + timedelta(days=1))
    also = [_row(e, start) for e in _drop_superseded([e for e in FESTIVALS if _day(e) in soon])
            if e["tier"] == "major" and e["greeting"] and not shown(e, me)]
    return {"festivals": festivals, "fasts": fasts, "bigDay": _row(big, start) if big else None,
            "alsoInIndia": also, "showing": showing(me)}


def legacy_merged(start: Date, days: int, lat: float, lon: float, tz_name: str, limit: int = 8, *,
                  vrats: Optional[VratsFn]) -> List[Dict[str, Any]]:
    """The old single list for 1.0.7 apps (astro_api `upcoming`): the old festivals and every monthly vrat, merged
    by date, in exactly the old shape {date, id, name, note, kind, type, inDays}. Same rules as before, on this
    table's dates (Durga Puja 2026 on its corrected day)."""
    end = start + timedelta(days=days)
    legacy = [e for e in FESTIVALS if e["id"] in LEGACY_IDS and start <= _day(e) < end]
    vr = [v for v in (vrats(start, days, lat, lon, tz_name) if vrats is not None else [])
          if not _covered(v, legacy)]
    items = [{"date": e["date"], "id": e["id"], "name": e["name"], "note": e["note"], "kind": e["id"],
              "type": "festival"} for e in legacy] + vr
    items.sort(key=lambda i: (i["date"], i["type"] != "festival"))
    for i in items:
        i["inDays"] = (Date.fromisoformat(i["date"]) - start).days
    return items[:limit]


def _own_day(e: Mapping[str, Any], me: Prefs) -> bool:
    """Whether a day the lists show is the person's own, for a wish: a national day, a day of their faith, a day of
    their region that belongs to no faith in particular (Onam, Pongal), or any faith's big day when they chose
    All of India. An unknown faith ("Prefer not to say", "No religion / Spiritual") never gets another faith's
    big day here: the lists may show it, but Mithu does not wish them Eid Mubarak or Merry Christmas."""
    return (e["tier"] == "national" or bool(me.faiths.intersection(e["faiths"]))
            or (not e["faiths"] and _region_hit(e, me)) or (me.all_india and e["tier"] == "major"))


def observance_for(day: Date, celebrates: Any = None, community: Any = None, *, religion: Any = None,
                   state: Any = None, fasts: Optional[Iterable[Mapping[str, Any]]] = None) -> str:
    """Today's observance line for the TODAY_OBSERVANCE fact: the person's own festivals today (the upcoming
    filter, narrowed to their own days by _own_day), then any of today's vrats passed in `fasts`
    (astro_core.vrats_between rows; only for people who keep Hindu days), or "none".
    - An unknown faith gets national days and their region's no-faith days only, never every faith's majors.
    - An explicit All of India choice keeps every faith's big days.

    celebrates: faith keys or religions (a single string works too); religion is used when celebrates is empty."""
    if isinstance(celebrates, str):
        celebrates = [celebrates]
    me = read_prefs({"celebrates": list(celebrates or []), "religion": religion, "community": community,
                     "stateName": state})
    # a vrat within a day of a festival is it
    near = [e for e in _mine(day - timedelta(days=1), day + timedelta(days=2), me) if _own_day(e, me)]
    iso = day.isoformat()
    parts = []
    for e in (x for x in near if x["date"] == iso):
        text = e["name"] + (": " + e["note"] if e["note"] else "")
        if e["greeting"]:
            text += (" " if text.endswith(".") else ". ") + "Greeting: " + e["greeting"] + "."
        parts.append(text)
    if "hindu" in me.faiths:
        for v in fasts or []:
            if v.get("date") == iso and v.get("name") and not _covered(v, near):
                parts.append(str(v["name"]) + (": " + str(v["note"]) if v.get("note") else ""))
    line = " | ".join(parts)
    return line[:400] if line else "none"
