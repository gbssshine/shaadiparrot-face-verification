"""The biodata PDF (plan §2 C9, features W2-S3): one A4 page (two at most) made from the signed-in account's own
profile, for families to forward on WhatsApp. Nothing is stored.

Building (build: which facts go on it, in which words) is pure and separate from drawing (render, fpdf2), so the
rules are tested without a PDF. biodata_route.py is the HTTP side; there is no FastAPI here.

What goes on it is chosen with the C9 chips:
- always: age, height, marital status, religion and community, education, work, family (empty fields and
  "Prefer not..." answers are left out);
- photo: the main photo (private photos come from the profile's own links; a blur link never);
- kundli: Moon sign and nakshatra; manglik: the C3 Manglik status;
- birth: date, time and place of birth, plus the pada and the Lagna (both give the birth time away);
- caste: caste, sub-caste and gotra; income: the yearly band, only when its visibility is Everyone;
- contact: the account's own sign-in phone, labelled "Me" or with the creator's relation (the route reads it);
- withLink: a footer line "Check your kundli match with {Name}, free" that opens the kundli tool; only with the
  kundli chip on, since the link itself carries the Moon sign and nakshatra (footer_link).
Never on it: e-mail, uid, address, exact location, complexion, weight or a gender line.
"""
from __future__ import annotations

import io
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests
from fontTools.ttLib import TTFont
from fpdf import FPDF
from PIL import Image, ImageOps

import fates_astro
import fates_astro_chart as fc

logger = logging.getLogger("shaadiparrot-cloudrun.biodata")

HERE = Path(__file__).resolve().parent
ASSETS = "assets/biodata"
FONT_FILES = {
    "Body": "assets/biodata/fonts/PlusJakartaSans-Regular.ttf",
    "BodySemi": "assets/biodata/fonts/PlusJakartaSans-SemiBold.ttf",
    "BodyBold": "assets/biodata/fonts/PlusJakartaSans-Bold.ttf",
    "Display": "assets/biodata/fonts/BricolageGrotesque-Bold.ttf",
    "Icons": "assets/biodata/fonts/Lucide.ttf",
}
APP_ICON = "assets/biodata/appicon_128.png"
MITHU_ART = "assets/biodata/mithu_256.png"

# MainUi colours.
PALETTE = {
    "Ink": "#1F1A17", "Muted": "#7A6F68", "CardLine": "#EEDFD8", "RedDeep": "#C62828", "Gold": "#B7791F",
    "GoldSoft": "#FFF6E6", "Rose": "#D6336C", "RoseSoft": "#FFE6EE", "Marigold": "#C77700",
    "MarigoldSoft": "#FFF1D2", "Peacock": "#0E8A83", "PeacockSoft": "#DFF4F1", "Blush": "#FFF0EC",
    "Leaf": "#2F8F46", "LeafSoft": "#E4F5E8", "Cream": "#FFF8F4", "Swoosh": "#F5A524", "White": "#FFFFFF",
}

# Lucide codepoints, the same as MauiApp2 Services/Icons.cs (all in the copied subset font).
GLYPHS = {
    "UserRound": "\ue468", "ScrollText": "\ue45f", "GraduationCap": "\ue234", "Users": "\ue1a4",
    "MoonStar": "\ue410", "Phone": "\ue133", "BadgeCheck": "\ue241",
}

# The link rule of firebase-functions/src/safety.ts LINK_RX (ContactSafety.Link in the app). E-mail addresses and
# phone numbers are not links. A link typed into a free text never goes on the branded PDF.
LINK_RX = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:t\.me|wa\.me|telegram\.me|instagram\.com|snapchat\.com|bit\.ly|"
                     r"linktr\.ee)/\S*", re.I)

# C9: the only link the footer takes (the app's AstroWire.IsKundliWithLink).
WITH_LINK_PREFIX = "https://www.shaadiparrot.com/kundli-match.html#with=1&"
WITH_LINK_RX = re.compile(r"^https://www\.shaadiparrot\.com/kundli-match\.html#with=1&")
WITH_LINK_MAX = 160

PHOTO_HOST = "photos.shaadiparrot.com"
STORAGE_HOST = "firebasestorage.googleapis.com"
MASCOT_RX = re.compile(r"(?:^|/|%2F)mascot\.png$", re.I)     # the app's placeholder when there is no photo
PHOTO_MAX_BYTES = 4 * 1024 * 1024
PHOTO_MAX_PIXELS = 40_000_000
PHOTO_SIZE = (600, 750)

ABOUT_MAX, FAMILY_MAX, VALUE_MAX, NAME_MAX = 500, 300, 60, 40
ABOUT_SHORT, FAMILY_SHORT = 300, 150      # the second try when the first render runs past two pages
VALUE_COMPACT, FACTS_COMPACT = 34, 60      # the third: one line per value
MAX_PAGES = 2
SQUEEZE_LEVELS = 4                         # build(short=0..3)

MANAGED = ("son", "daughter", "brother", "sister", "relative", "friend")
NEVER_MARRIED = "never married"
CHILDREN = ("No children", "Yes, living together", "Yes, living separately")

# Biodata.IncomeRanges (code -> label).
INCOME_LABELS = {
    "0-2L": "Up to ₹2 lakh", "2-4L": "₹2–4 lakh", "4-7L": "₹4–7 lakh", "7-10L": "₹7–10 lakh",
    "10-15L": "₹10–15 lakh", "15-20L": "₹15–20 lakh", "20-30L": "₹20–30 lakh", "30-50L": "₹30–50 lakh",
    "50L-1Cr": "₹50 lakh–1 crore", "1Cr+": "₹1 crore and above",
}

# The only profile fields the PDF reads (docs/design-1.0.8/feature-pdf.md 4.1, plus the C9 chips).
FIELDS = (
    "firstName", "gender", "birthDate", "birthTime", "birthCityName", "birthStateName", "birthCountryIso2",
    "birthCountryName", "cityName", "city", "stateName", "heightDisplayText", "height", "maritalStatus", "children",
    "religion", "settings_religion", "community", "settings_community", "diet", "education", "profession",
    "familyType", "familyValues", "fatherOccupation", "motherOccupation", "brothers", "sisters", "familyCity",
    "aboutFamily", "aboutMe", "profileFor", "creatorRelation", "caste", "subCaste", "gotra", "annualIncome",
    "annualIncomeVisibility",
)

KUNDLI_NOTE = "Vedic (sidereal, Lahiri). Worked out by Mithu, the Shaadi Parrot astrologer."
DISCLAIMER = "Details as given in the profile; not checked by Shaadi Parrot."
DISCLAIMER_BADGE = "Details as given in the profile; only the Photo verified badge is checked by Shaadi Parrot."
BRAND_LINE = "Made with Shaadi Parrot · www.shaadiparrot.com"
BRAND_URL = "https://www.shaadiparrot.com"
TOOL_TEXT = "shaadiparrot.com/kundli-match"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class UnsupportedText(Exception):
    """A letter, mark or digit the PDF font can't print (Devanagari, for now): the app offers the text share."""


def _cmap(path: str) -> frozenset:
    font = TTFont(str(HERE / path), lazy=True)
    try:
        return frozenset(font.getBestCmap() or {})
    finally:
        font.close()


# Loaded once: the body font decides what can be printed; the display font only what the name line uses.
BODY_CHARS = _cmap(FONT_FILES["Body"])
DISPLAY_CHARS = _cmap(FONT_FILES["Display"])


# ------------------------------------------------------------------ text
def _invisible(ch: str) -> bool:
    """Emoji helpers that are marks by category but never letters: variation selectors and the keycap."""
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or o == 0x20E3


def clean(value: Any, limit: int = VALUE_MAX) -> str:
    """One printable line: spaces and newlines collapsed, links removed, characters the font lacks dropped when they
    aren't letters, marks or digits (emoji, symbols, ZWJ, variation selectors), cut to `limit` on a word.
    A letter, mark or digit the font lacks raises UnsupportedText."""
    if value is None or isinstance(value, (bool, dict, list, tuple, set)):
        return ""
    text = LINK_RX.sub(" ", unicodedata.normalize("NFC", str(value)))   # NFC: "e" + U+0301 is the font's "é"
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif ord(ch) in BODY_CHARS and not _invisible(ch):
            out.append(ch)
        elif _invisible(ch) or unicodedata.category(ch)[0] not in "LMN":
            continue
        else:
            raise UnsupportedText(f"U+{ord(ch):04X}")
    return _cut(" ".join("".join(out).split()), limit)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    space = head.rfind(" ")
    if space >= limit * 0.6:
        head = head[:space]
    return head.rstrip(" ,.;:-") + "…"


def skip(v: Any) -> bool:
    """Not stated: empty, or a "Prefer not..." answer."""
    t = str(v or "").strip() if not isinstance(v, (bool, dict, list)) else ""
    return not t or t.lower().startswith("prefer not")


def _unstated(v: str) -> bool:
    """Biodata.IsUnstated: empty, "Prefer not..." or "Don't know"."""
    return skip(v) or v.strip().lower() == "don't know"


def first_word(v: Any) -> str:
    """The name on the biodata: the first word only (a surname typed into the first-name box stays off it)."""
    words = clean(v, 200).split()
    return words[0][:NAME_MAX] if words else ""


_TWELVE = re.compile(r"""^\s*(\d)\s*'\s*12\s*"?\s*$""")
_CM = re.compile(r"^(\d{2,3})\s*cm$", re.I)


def height(text: Any) -> str:
    """ProfileDocMapper.NormalizeHeightText (5'12" -> 6'0") then Biodata.Height(withCm: true) ("163 cm" ->
    5'4" (163 cm))."""
    if skip(text):
        return ""
    t = clean(text)
    m = _TWELVE.match(t)
    if m:
        return f"{int(m.group(1)) + 1}'0\""
    m = _CM.match(t)
    if m:
        cm = int(m.group(1))
        if 120 <= cm <= 230:
            inches = round(cm / 2.54)
            return f"{inches // 12}'{inches % 12}\" ({cm} cm)"
    return t


def _profile_for(doc: Mapping[str, Any]) -> str:
    return str(doc.get("profileFor") or "").strip().lower()


def is_managed(doc: Mapping[str, Any]) -> bool:
    return _profile_for(doc) in MANAGED


_RELATION_LABELS = {
    "mother": "Mother", "father": "Father", "brother": "Brother", "sister": "Sister", "grandmother": "Grandmother",
    "grandfather": "Grandfather", "uncle": "Uncle", "aunt": "Aunt", "cousin": "Cousin", "other": "Relative",
    "friend": "Friend",
}


def managed_by(profile_for: Any, creator_relation: Any = None) -> str:
    """Biodata.ManagedBy: "Managed by mother", "Managed by a parent"...; "" for one's own profile."""
    pf = str(profile_for or "").strip().lower()
    if pf not in MANAGED:
        return ""
    rel = str(creator_relation or "").strip().lower()
    exact = {"mother": "Managed by mother", "father": "Managed by father", "brother": "Managed by brother",
             "sister": "Managed by sister", "grandmother": "Managed by grandmother",
             "grandfather": "Managed by grandfather", "uncle": "Managed by uncle", "aunt": "Managed by aunt",
             "cousin": "Managed by a cousin", "other": "Managed by a relative", "friend": "Managed by a friend"}
    if rel in exact:
        return exact[rel]
    return {"son": "Managed by a parent", "daughter": "Managed by a parent", "brother": "Managed by a sibling",
            "sister": "Managed by a sibling", "relative": "Managed by a relative",
            "friend": "Managed by a friend"}[pf]


def contact_label(doc: Mapping[str, Any]) -> str:
    """Whose number it is (C9): "Me" on one's own profile, else the creator's relation ("Mother"), else what the
    profile type says ("Parent", "Sibling", "Relative", "Friend")."""
    pf = _profile_for(doc)
    if pf not in MANAGED:
        return "Me"
    rel = str(doc.get("creatorRelation") or "").strip().lower()
    if rel in _RELATION_LABELS:
        return _RELATION_LABELS[rel]
    return {"son": "Parent", "daughter": "Parent", "brother": "Sibling", "sister": "Sibling",
            "relative": "Relative", "friend": "Friend"}[pf]


def format_phone(e164: Any) -> str:
    """"+919820000000" -> "+91 98200 00000"; other numbers stay as they are. "" for anything that isn't one."""
    p = re.sub(r"[\s-]", "", str(e164 or ""))
    if not re.fullmatch(r"\+\d{8,15}", p):
        return ""
    if p.startswith("+91") and len(p) == 13:
        return f"+91 {p[3:8]} {p[8:]}"
    return p


def siblings(brothers: Any, sisters: Any) -> str:
    """BiodataEditPage.Siblings: "1 brother, 2 sisters"; both "None" -> "No siblings"."""
    b, s = str(brothers or "").strip(), str(sisters or "").strip()

    def one(count: str, word: str) -> str:
        if count in ("", "None"):
            return ""
        return f"1 {word}" if count == "1" else f"{clean(count, 4)} {word}s"

    if b == "None" and s == "None":
        return "No siblings"
    return ", ".join(x for x in (one(b, "brother"), one(s, "sister")) if x)


def family_line(doc: Mapping[str, Any]) -> str:
    """"Joint family, traditional values" (BiodataEditPage.FamilyLine's head)."""
    kind = "" if skip(doc.get("familyType")) else clean(doc.get("familyType"))
    values = "" if skip(doc.get("familyValues")) else clean(doc.get("familyValues"))
    return ", ".join(x for x in (kind, values.lower() + " values" if values else "") if x)


def marital_line(status: Any, children: Any, gender: Any = None) -> str:
    """Biodata.MaritalLine: "" for "Never married" or not stated, else the status with the children answer."""
    s = "" if skip(status) else clean(status)
    if not s or s.lower() == NEVER_MARRIED:
        return ""
    c = str(children or "").strip()
    if c.lower() == CHILDREN[0].lower():
        return f"{s} · no children"
    if c.lower() == CHILDREN[1].lower():
        g = str(gender or "").strip().lower()
        who = "him" if g == "male" else "her" if g == "female" else "them"
        return f"{s} · has children, living with {who}"
    if c.lower() == CHILDREN[2].lower():
        return f"{s} · has children, living separately"
    return s


def age(born: Tuple[int, int, int], today: Date) -> int:
    y, m, d = born
    return today.year - y - ((today.month, today.day) < (m, d))


def _date_text(born: Tuple[int, int, int]) -> str:
    return f"{born[2]} {MONTHS[born[1] - 1]} {born[0]}"


def _time_text(t: Tuple[int, int]) -> str:
    h, mi = t
    return f"{h % 12 or 12}:{mi:02d} {'am' if h < 12 else 'pm'}"


def income_label(doc: Mapping[str, Any]) -> str:
    """The band, only when its visibility is Everyone (missing counts as Everyone, as in the app)."""
    if str(doc.get("annualIncomeVisibility") or "").strip() == "matches":
        return ""
    return INCOME_LABELS.get(str(doc.get("annualIncome") or "").strip(), "")


def valid_with_link(link: Any) -> Optional[str]:
    """C9: the footer link, or None (ignored) when it isn't a kundli-match "with" link of at most 160 characters."""
    if not isinstance(link, str):
        return None
    t = link.strip()
    if (len(t) <= len(WITH_LINK_PREFIX) or len(t) > WITH_LINK_MAX or not WITH_LINK_RX.match(t)
            or any(ch.isspace() for ch in t) or any(ord(ch) < 33 or ord(ch) > 126 for ch in t)):
        return None
    return t


def footer_link(link: Any, opts: "Options") -> Optional[str]:
    """The "Check your kundli match with {Name}, free" line's link, only with the Kundli chip on. The link carries
    the Moon itself (m=<rashi>.<nakshatra> groups, t=0 when the birth time isn't known), and feature-pdf.md makes
    the Kundli chip the way to keep "Moon sign, nakshatra and Manglik" off a forwarded document. With the chip off
    the whole line goes: the plain tool text can't keep its "with {Name}" promise without that Moon, and the
    footer is then exactly feature-pdf.md's own (the brand row with www.shaadiparrot.com, the disclaimer)."""
    return valid_with_link(link) if opts.kundli else None


def pick(public: Mapping[str, Any], user: Mapping[str, Any], profile: Mapping[str, Any]) -> Dict[str, Any]:
    """{**public, **user, **profile} (the private profile wins), read only through FIELDS."""
    out: Dict[str, Any] = {}
    for src in (public or {}, user or {}, profile or {}):
        for k in FIELDS:
            if k in src:
                out[k] = src[k]
    return out


# ------------------------------------------------------------------ the kundli card
def _in_day_order(idxs: Iterable[int], mod: int) -> List[int]:
    """The distinct signs (or nakshatras) of one day in the order the Moon passes them: it only moves forward,
    and less than two of them in a day, so the start is the one the others follow most closely."""
    uniq = list(dict.fromkeys(idxs))
    if len(uniq) < 2:
        return uniq
    start = min(uniq, key=lambda s: max((i - s) % mod for i in uniq))
    return sorted(uniq, key=lambda i: (i - start) % mod)


def _names(idxs: Iterable[int], table, western: bool) -> List[str]:
    out = []
    for i in idxs:
        name = table[i][0] + (f" ({table[i][1]})" if western else "")
        if name not in out:
            out.append(name)
    return out


def _either(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def manglik_text(chart: Mapping[str, Any]) -> str:
    """The Manglik row in the C3 words: Yes, Partial (Anshik), No (with the cancellation), Check (needs birth time)."""
    st = fates_astro.manglik_status(dict(chart))
    status = st.get("status")
    if status == "yes":
        return "Yes"
    if status == "partial":
        return "Partial (Anshik)"
    if status == "no":
        why = st.get("cancelledBy")
        return f"No ({why})" if why else "No"
    return "Check (needs birth time)"


def kundli_rows(chart: Mapping[str, Any], birth: bool, kundli: bool = True,
                manglik: bool = False) -> List[Tuple[str, str]]:
    """Moon sign and nakshatra (kundli), the Lagna (birth), the Manglik status (manglik).

    The same chart, tables and noon default as the Kundli tab. The wording is deliberately more careful on a
    forwarded document: without a birth time it names every sign or nakshatra the day allows ("Rohini or
    Mrigashira (birth time not given)") instead of a flat guess, and the pada and the Lagna, which give the birth
    time away, are only printed with the birth chip."""
    rows: List[Tuple[str, str]] = []
    if kundli:
        time_known = bool(chart.get("timeKnown"))
        options = [tuple(o[:2]) for o in (chart.get("options") or []) if len(o) >= 2] or \
                  [(chart["rashi"], chart["nak"])]
        rashis = _names(_in_day_order([r for r, _ in options], 12), fates_astro.RASHIS, True)
        naks = _names(_in_day_order([n for _, n in options], 27), fates_astro.NAKSHATRAS, False)
        unsure = " (birth time not given)"
        if not time_known and len(rashis) > 1:
            rows.append(("Moon sign (Rashi)", " or ".join(r.split(" (")[0] for r in rashis) + unsure))
        else:
            rows.append(("Moon sign (Rashi)", _names([chart["rashi"]], fates_astro.RASHIS, True)[0]))
        if not time_known and len(naks) > 1:
            rows.append(("Nakshatra", _either(naks) + unsure))
        else:
            nak = fates_astro.NAKSHATRAS[chart["nak"]][0]
            rows.append(("Nakshatra", f"{nak}, pada {chart['pada']}" if birth and time_known else nak))
        if birth and chart.get("lagna") is not None:
            rows.append(("Rising sign (Lagna)", _names([chart["lagna"]], fates_astro.RASHIS, True)[0]))
    if manglik:
        rows.append(("Manglik", manglik_text(chart)))
    return rows


# ------------------------------------------------------------------ the biodata
@dataclass
class Options:
    """The C9 chips."""
    photo: bool = True
    kundli: bool = True
    manglik: bool = False
    birth: bool = False
    caste: bool = False
    income: bool = False


@dataclass
class Section:
    key: str
    title: str
    glyph: str
    fg: str
    bg: str
    rows: List[Tuple[str, str]] = field(default_factory=list)
    paragraph: str = ""
    paragraph_first: bool = True
    art: Optional[str] = None
    note: str = ""


@dataclass
class Biodata:
    name: str
    facts_lines: List[str]
    pills: List[str]
    sections: List[Section]
    contact: Optional[Tuple[str, str]] = None
    link: Optional[str] = None
    link_text: str = ""

    def section(self, key: str) -> Optional[Section]:
        return next((s for s in self.sections if s.key == key), None)

    def text(self) -> str:
        """Everything printed (tests)."""
        parts = [self.name, *self.facts_lines, *self.pills, self.link_text]
        for s in self.sections:
            parts += [s.title, s.paragraph, s.note, *(f"{a}: {b}" for a, b in s.rows)]
        return "\n".join(p for p in parts if p)


def build(doc: Mapping[str, Any], chart: Optional[Mapping[str, Any]], opts: Options,
          contact: Optional[Tuple[str, str]] = None, *, with_link: Optional[str] = None,
          today: Optional[Date] = None, short: int = 0) -> Biodata:
    """The biodata from the allow-listed doc (pick). Pure. Raises UnsupportedText for a letter the font lacks.

    `short` is how hard make() has to squeeze a profile that ran past two pages: 1 cuts the about texts to
    300 / 150 characters (the spec's second try), 2 also keeps every typed value to one line, 3 leaves the two
    about texts out. Only a modified client's oversized fields ever get past 1."""
    today = today or Date.today()
    short = int(short or 0)

    def val(*keys: str) -> str:
        for k in keys:
            v = doc.get(k)
            if not skip(v):
                return clean(v)
        return ""

    name = first_word(doc.get("firstName"))
    born = fc._parse_date(doc.get("birthDate"))
    managed = is_managed(doc)

    tall = height(doc.get("heightDisplayText") or doc.get("height"))
    status = val("maritalStatus")
    religion, community = val("religion", "settings_religion"), val("community", "settings_community")
    city, state = val("cityName", "city"), val("stateName")
    facts = [
        " · ".join(x for x in (f"{age(born, today)} yrs" if born else "", tall,
                               marital_line(status, doc.get("children"), doc.get("gender"))) if x),
        " · ".join(x for x in (religion, community, f"Lives in {city}" if city else "") if x),
    ]
    pills = [p for p in (managed_by(doc.get("profileFor"), doc.get("creatorRelation")),) if p]

    sections: List[Section] = []
    about = clean(doc.get("aboutMe"), ABOUT_SHORT if short else ABOUT_MAX) \
        if not skip(doc.get("aboutMe")) and short < 3 else ""
    if about:
        title = f"About {name}" if managed and name else ("About" if managed else "About me")
        sections.append(Section("about", title, GLYPHS["UserRound"], PALETTE["Rose"], PALETTE["RoseSoft"],
                                paragraph=about))

    personal: List[Tuple[str, str]] = []
    if opts.birth and born:
        personal.append(("Date of birth", _date_text(born)))
        t = fc._parse_time(doc.get("birthTime"))
        if t:
            personal.append(("Time of birth", _time_text(t)))
        place = [val("birthCityName"), val("birthStateName")]
        iso2 = str(doc.get("birthCountryIso2") or "").strip().upper()
        if iso2 not in ("", "IN"):
            place.append(val("birthCountryName"))
        place_text = ", ".join(x for x in place if x)
        if place_text:
            personal.append(("Place of birth", place_text))
    if tall:
        personal.append(("Height", tall))
    if status:
        personal.append(("Marital status", status))
        kids = val("children")
        if kids and status.lower() != NEVER_MARRIED:
            personal.append(("Children", kids))
    if religion:
        personal.append(("Religion", religion))
    if community:
        personal.append(("Mother tongue", community))
    if opts.caste:
        for label, key in (("Caste", "caste"), ("Sub-caste", "subCaste"), ("Gotra", "gotra")):
            v = str(doc.get(key) or "")
            if not _unstated(v):
                personal.append((label, clean(v)))
    diet = val("diet")
    if diet:
        personal.append(("Food", diet))
    lives = ", ".join(x for x in (city, state) if x)
    if lives:
        personal.append(("Lives in", lives))
    if personal:
        sections.append(Section("personal", "Personal details", GLYPHS["ScrollText"], PALETTE["Gold"],
                                PALETTE["GoldSoft"], rows=personal))

    work = [(label, v) for label, v in (("Education", val("education")), ("Work", val("profession"))) if v]
    if opts.income:
        band = income_label(doc)
        if band:
            work.append(("Annual income", band))
    if work:
        sections.append(Section("work", "Education & work", GLYPHS["GraduationCap"], PALETTE["Marigold"],
                                PALETTE["MarigoldSoft"], rows=work))

    fam: List[Tuple[str, str]] = []
    for label, v in (("Family type", family_line(doc)), ("Father", val("fatherOccupation")),
                     ("Mother", val("motherOccupation")), ("Siblings", siblings(doc.get("brothers"), doc.get("sisters"))),
                     ("Family lives in", val("familyCity"))):
        if v:
            fam.append((label, v))
    about_family = clean(doc.get("aboutFamily"), FAMILY_SHORT if short else FAMILY_MAX) \
        if not skip(doc.get("aboutFamily")) and short < 3 else ""
    if fam or about_family:
        sections.append(Section("family", "Family", GLYPHS["Users"], PALETTE["Peacock"], PALETTE["PeacockSoft"],
                                rows=fam, paragraph=about_family, paragraph_first=False))

    if short >= 2:      # every typed value on one line (only oversized fields from a modified client get here)
        for s in sections:
            s.rows = [(label, _cut(v, VALUE_COMPACT)) for label, v in s.rows]
        facts = [_cut(f, FACTS_COMPACT) for f in facts]

    if chart is not None and born and (opts.kundli or opts.manglik):
        rows = kundli_rows(chart, opts.birth, opts.kundli, opts.manglik)
        if rows:
            sections.append(Section("kundli", "Kundli", GLYPHS["MoonStar"], PALETTE["RedDeep"], PALETTE["Blush"],
                                    rows=rows, art=MITHU_ART, note=KUNDLI_NOTE))

    if contact:
        sections.append(Section("contact", "Contact", GLYPHS["Phone"], PALETTE["Leaf"], PALETTE["LeafSoft"],
                                rows=[contact]))

    link = footer_link(with_link, opts)
    link_text = ""
    if link:
        who = name if name else "this biodata"
        link_text = f"Check your kundli match with {who}, free: {TOOL_TEXT}"
    return Biodata(name=name, facts_lines=[f for f in facts if f], pills=pills, sections=sections, contact=contact,
                   link=link, link_text=link_text)


# ------------------------------------------------------------------ the photo
def _first(v: Any) -> str:
    if isinstance(v, list) and v and isinstance(v[0], str):
        return v[0].strip()
    return ""


def is_private(profile: Mapping[str, Any]) -> bool:
    return str((profile or {}).get("photoPrivacy") or "").strip() == "accepted"


def _own_source(profile: Mapping[str, Any], public: Mapping[str, Any]) -> str:
    """The profile photo the PDF stands for: the card's first photo, or the private profile's in private mode."""
    src = _first((profile if is_private(profile) else public or {}).get("photos"))
    return "" if not src or MASCOT_RX.search(src.split("?")[0]) else src


def allowed_photo_url(url: str, uid: str) -> bool:
    """https only: the photo CDN in the user's own folder, or Firebase Storage in the user's own folder (in the
    project bucket when FIREBASE_STORAGE_BUCKET is set). A blur link is never one, whoever's it is."""
    if not isinstance(url, str) or not uid or "/" in uid or "%" in uid:
        return False
    try:
        u = urlparse(url.strip())
    except ValueError:
        return False
    path = u.path or ""
    plain = unquote(path)
    if u.scheme != "https" or "/blur/" in plain.lower() or ".." in plain:
        return False
    try:
        if u.port not in (None, 443) or u.username or u.password:
            return False
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    if host == PHOTO_HOST:
        return path.startswith(f"/users/{uid}/")
    if host == STORAGE_HOST:
        parts = path.split("/")
        # /v0/b/{bucket}/o/{object}; the object's first folder must be users/{uid} (the uid compared exactly)
        if len(parts) < 6 or parts[1] != "v0" or parts[2] != "b" or parts[4] != "o" or not parts[3]:
            return False
        bucket = os.getenv("FIREBASE_STORAGE_BUCKET") or ""
        if bucket and parts[3] != bucket:
            return False
        m = re.match(r"users%2[fF]([^%/]+)%2[fF]", "/".join(parts[5:]))
        return bool(m) and m.group(1) == uid
    return False


def photo_url(profile: Mapping[str, Any], public: Mapping[str, Any], uid: str) -> Optional[str]:
    """The main photo's 960 px variant to print, or None (nothing is fetched then).
    Private photos (photoPrivacy "accepted"): the profile's own links, since the card only has blurs. Otherwise
    the public card's, which is what was moderated. Candidates: photoMediumCdn[0], photoMedium[0], photos[0]."""
    if not _own_source(profile or {}, public or {}):
        return None
    src = (profile or {}) if is_private(profile or {}) else (public or {})
    for key in ("photoMediumCdn", "photoMedium", "photos"):
        url = _first(src.get(key))
        if url and allowed_photo_url(url, uid):
            return url
    return None


def fetch_photo(url: str) -> Optional[bytes]:
    """The photo as a 600x750 JPEG (EXIF turned, 4:5 crop a little above the centre, no metadata), or None on any
    problem. Redirects are not followed; at most 4 MB is read."""
    resp = None
    try:
        resp = requests.get(url, timeout=(3, 8), allow_redirects=False, stream=True)
        if resp.status_code != 200:
            return None
        if int(resp.headers.get("Content-Length") or 0) > PHOTO_MAX_BYTES:
            return None
        buf = bytearray()
        for chunk in resp.iter_content(64 * 1024):
            buf += chunk
            if len(buf) > PHOTO_MAX_BYTES:
                return None
        img = Image.open(io.BytesIO(bytes(buf)))
        if img.size[0] * img.size[1] > PHOTO_MAX_PIXELS:
            return None
        img.draft("RGB", (PHOTO_SIZE[0] * 2, PHOTO_SIZE[1] * 2))
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if w < 2 or h < 2:
            return None
        if w / h > 0.8:
            nw = int(round(h * 0.8))
            img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(round(w / 0.8))
            top = int((h - nh) * 0.4)          # 10 % of the slack above the centre: faces sit high
            img = img.crop((0, top, w, top + nh))
        img = img.resize(PHOTO_SIZE, Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=82, optimize=True)
        return out.getvalue()
    except Exception:
        logger.warning("biodata photo fetch failed", exc_info=False)
        return None
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


def photo_verified(printed: bool, profile: Mapping[str, Any], public: Mapping[str, Any]) -> bool:
    """The "Photo verified" pill: the photo is on the PDF, the public card says verified (the card flag is the one
    cleared when photos change), and the photo it stands for is one the face check verified."""
    if not printed or (public or {}).get("isFaceVerified") is not True:
        return False
    src = _own_source(profile or {}, public or {})
    verified = (profile or {}).get("faceVerifiedPhotos") or []
    return bool(src) and isinstance(verified, list) and src in [str(x) for x in verified]


# ------------------------------------------------------------------ drawing
def _rgb(hex_: str) -> Tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PAGE_W, PAGE_H = 210.0, 297.0
MARGIN, TOP = 16.0, 16.0
CONTENT_W = PAGE_W - 2 * MARGIN
PHOTO_W, PHOTO_H = 42.0, 52.5          # 4:5, a white-mat polaroid
PAD = 4.2                              # inside a card
TILE = 8.0                             # the icon tile
TITLE_GAP = 2.0                        # under the title row
GAP = 3.2                              # between cards
COL_GAP = 6.0
ROW_GAP = 1.8
SIDE_OVERHANG = 30.0                   # how far a card beside the photo may run below it
INLINE_PAD = 3.4                       # a one-line card (Contact)
LABEL_H, VALUE_H, PARA_H, NOTE_H = 4.0, 4.9, 5.1, 3.4
# The footer, from the bottom: the disclaimer, the brand row (icon, link, date), the kundli link line.
FOOT_DISCLAIMER_Y, FOOT_BRAND_Y, FOOT_LINK_Y = 284.0, 277.5, 272.5
BOTTOM_NO_LINK, BOTTOM_WITH_LINK = PAGE_H - 275.5, PAGE_H - 271.0


class _Pdf(FPDF):
    def __init__(self, bio: Biodata, verified: bool, today: Date):
        super().__init__(format="A4", unit="mm")
        self.bio, self.verified, self.today = bio, verified, today
        for fam, path in FONT_FILES.items():
            self.add_font(fam, "", str(HERE / path))
        self.set_margins(MARGIN, TOP, MARGIN)
        self.set_auto_page_break(True, BOTTOM_WITH_LINK if bio.link else BOTTOM_NO_LINK)
        self.set_title(f"{bio.name or 'Shaadi Parrot'} - Biodata")
        self.set_author("Shaadi Parrot")
        self.set_creator("Shaadi Parrot")
        self.set_lang("en")

    @property
    def bottom(self) -> float:
        return PAGE_H - self.b_margin

    def color(self, name: str, what: str = "text") -> None:
        r, g, b = _rgb(PALETTE[name] if name in PALETTE else name)
        {"text": self.set_text_color, "fill": self.set_fill_color, "draw": self.set_draw_color}[what](r, g, b)

    def font(self, fam: str, size: float, text: str = "") -> None:
        # A name the display font can't draw (an accented letter it lacks) is set in the body bold instead.
        if fam == "Display" and text and any(ord(ch) not in DISPLAY_CHARS for ch in text if not ch.isspace()):
            fam = "BodyBold"
        self.set_font(fam, size=size)

    def header(self) -> None:
        self.color("Cream", "fill")
        self.rect(0, 0, PAGE_W, PAGE_H, style="F")
        self.color("Gold", "draw")
        self.set_line_width(0.6 * 0.3528)
        self.rect(6, 6, PAGE_W - 12, PAGE_H - 12)
        self.color("Swoosh", "draw")
        self.set_line_width(0.3 * 0.3528)
        self.rect(7.4, 7.4, PAGE_W - 14.8, PAGE_H - 14.8)

    def footer(self) -> None:
        auto, margin = self.auto_page_break, self.b_margin
        self.set_auto_page_break(False)
        try:
            if self.bio.link:
                self.font("BodySemi", 9)
                self.color("RedDeep")
                self.set_xy(MARGIN, FOOT_LINK_Y)
                self.cell(CONTENT_W, 4.5, self.bio.link_text, align="L", link=self.bio.link)
            self.image(str(HERE / APP_ICON), MARGIN, FOOT_BRAND_Y, 5.5, 5.5)
            self.font("BodySemi", 8.5)
            self.color("Ink")
            self.set_xy(MARGIN + 7.5, FOOT_BRAND_Y)
            self.cell(110, 5.5, BRAND_LINE, align="L", link=BRAND_URL)
            self.font("Body", 8.5)
            self.color("Muted")
            self.set_xy(PAGE_W - MARGIN - 50, FOOT_BRAND_Y)
            d = self.today
            self.cell(50, 5.5, f"{d.day} {MONTHS[d.month - 1]} {d.year}", align="R")
            self.font("Body", 7.5)
            self.set_xy(MARGIN, FOOT_DISCLAIMER_Y)
            self.cell(CONTENT_W, 3.5, DISCLAIMER_BADGE if self.verified else DISCLAIMER, align="L")
        finally:
            self.set_auto_page_break(auto, margin)

    # -------------------------------------------------------------- blocks
    def text_height(self, fam: str, size: float, w: float, h: float, text: str) -> float:
        self.font(fam, size, text)
        return float(self.multi_cell(w, h, text, dry_run=True, output="HEIGHT", align="L"))

    def pill(self, x: float, y: float, text: str, fg: str, bg: str, glyph: str = "") -> float:
        self.font("BodySemi", 8.5)
        tw = self.get_string_width(text)
        gw = 0.0
        if glyph:
            self.font("Icons", 9)
            gw = self.get_string_width(glyph) + 1.2
        w = tw + gw + 6
        self.color(bg, "fill")
        self.rect(x, y, w, 6, style="F", round_corners=True, corner_radius=3)
        cx = x + 3
        if glyph:
            self.font("Icons", 9)
            self.color(fg)
            self.set_xy(cx, y)
            self.cell(gw, 6, glyph)
            cx += gw
        self.font("BodySemi", 8.5)
        self.color(fg)
        self.set_xy(cx, y)
        self.cell(tw, 6, text)
        return w

    def head(self, photo: Optional[bytes]) -> None:
        bio = self.bio
        text_w = CONTENT_W - (PHOTO_W + 6 if photo else 0)
        y = TOP + 1
        if bio.name:
            self.font("BodyBold", 9)
            self.color("Gold")
            self.set_char_spacing(1.6)
            self.set_xy(MARGIN, y)
            self.cell(text_w, 4, "BIODATA")
            self.set_char_spacing(0)
            y += 5
        self.font("Display", 28, bio.name or "Biodata")
        self.color("RedDeep")
        self.set_xy(MARGIN, y)
        self.multi_cell(text_w, 11, bio.name or "Biodata", align="L")
        y = self.get_y() + 1
        self.color("Swoosh", "fill")
        self.rect(MARGIN, y, 22, 1.6, style="F", round_corners=True, corner_radius=0.8)
        y += 5
        for i, line in enumerate(bio.facts_lines):
            self.font("BodySemi" if i == 0 else "Body", 11 if i == 0 else 10.5)
            self.color("Ink" if i == 0 else "Muted")
            self.set_xy(MARGIN, y)
            self.multi_cell(text_w, 5.4, line, align="L")
            y = self.get_y() + 0.8
        pills = [(p, "Rose", "RoseSoft", "") for p in bio.pills]
        if self.verified:
            pills.append(("Photo verified", "Leaf", "LeafSoft", GLYPHS["BadgeCheck"]))
        if pills:
            y += 1.5
            x = MARGIN
            for text, fg, bg, glyph in pills:
                x += self.pill(x, y, text, fg, bg, glyph) + 2.5
            y += 6
        bottom = y
        if photo:
            px = PAGE_W - MARGIN - PHOTO_W
            self.color("White", "fill")
            self.color("Gold", "draw")
            self.set_line_width(0.2)
            self.rect(px, TOP, PHOTO_W, PHOTO_H, style="DF")
            mat = 2.5
            iw = PHOTO_W - 2 * mat
            self.image(io.BytesIO(photo), px + mat, TOP + mat, iw, iw * 1.25)
            bottom = max(bottom, TOP + PHOTO_H)
        self.set_y(bottom + 5)
        return y, (TOP + PHOTO_H if photo else 0.0)

    def _grid_h(self, rows: List[Tuple[str, str]], cols: int, width: float) -> Tuple[float, float]:
        """(height, the tallest value) of the grid in `cols` columns."""
        col_w = (width - 2 * PAD - (cols - 1) * COL_GAP) / cols
        h, tallest = 0.0, 0.0
        for i in range(0, len(rows), cols):
            values = [self.text_height("BodySemi", 10.5, col_w, VALUE_H, v) for _, v in rows[i:i + cols]]
            tallest = max([tallest, *values])
            h += LABEL_H + max(values) + ROW_GAP
        return h, tallest

    def columns(self, rows: List[Tuple[str, str]], width: float = CONTENT_W) -> int:
        """Three label/value columns when that makes the card shorter (no value over two lines, every label on
        one), else two."""
        if len(rows) < 3:
            return 2
        third = (width - 2 * PAD - 2 * COL_GAP) / 3
        self.font("Body", 9.5)
        if any(self.get_string_width(label) > third - 1 for label, _ in rows):
            return 2
        three, tallest = self._grid_h(rows, 3, width)
        return 3 if tallest <= 2 * VALUE_H + 0.01 and three < self._grid_h(rows, 2, width)[0] - 0.01 else 2

    def grid(self, rows: List[Tuple[str, str]], x: float, y: float, draw: bool, width: float = CONTENT_W) -> float:
        """The label/value grid at (x, y) in a card `width` wide; returns its height. With draw=False it only
        measures."""
        cols = self.columns(rows, width)
        col_w = (width - 2 * PAD - (cols - 1) * COL_GAP) / cols
        h = 0.0
        for i in range(0, len(rows), cols):
            line = rows[i:i + cols]
            row_h = max(LABEL_H + self.text_height("BodySemi", 10.5, col_w, VALUE_H, v) for _, v in line)
            if draw:
                for j, (label, value) in enumerate(line):
                    cx = x + j * (col_w + COL_GAP)
                    self.font("Body", 9.5)
                    self.color("Muted")
                    self.set_xy(cx, y + h)
                    self.cell(col_w, LABEL_H, label)
                    self.font("BodySemi", 10.5)
                    self.color("Ink")
                    self.set_xy(cx, y + h + LABEL_H)
                    self.multi_cell(col_w, VALUE_H, value, align="L")
            h += row_h + ROW_GAP
        return h

    @staticmethod
    def inline(s: Section) -> bool:
        """A card with one short fact (Contact, mostly) keeps it on the title line."""
        return len(s.rows) == 1 and not s.paragraph and not s.note and len(s.rows[0][0]) + len(s.rows[0][1]) <= 60

    def note_box(self, s: Section, width: float) -> Optional[Tuple[float, float]]:
        """(x offset, width) of the note in the title row, right of the title and left of the art, when it fits
        there in two lines; None puts it under the rows."""
        if not s.note:
            return None
        self.font("Display", 13, s.title)
        left = PAD + TILE + 3 + self.get_string_width(s.title) + 6
        w = width - left - PAD - (17 if s.art else 0)
        if w < 40 or self.text_height("Body", 7.5, w, NOTE_H, s.note) > 2 * NOTE_H + 0.01:
            return None
        return left, w

    def card_height(self, s: Section, width: float = CONTENT_W) -> float:
        if self.inline(s):
            return 2 * INLINE_PAD + TILE
        inner = width - 2 * PAD
        h = PAD + TILE + TITLE_GAP
        if s.paragraph:
            h += self.text_height("Body", 10.5, inner, PARA_H, s.paragraph) + ROW_GAP
        if s.rows:
            h += self.grid(s.rows, 0, 0, draw=False, width=width)
        if s.note and self.note_box(s, width) is None:
            h += self.text_height("Body", 7.5, inner - (17 if s.art else 0), NOTE_H, s.note) + 0.5
        return h + PAD - ROW_GAP

    def card(self, s: Section, width: float = CONTENT_W) -> None:
        h = self.card_height(s, width)
        if self.get_y() + h > self.bottom:
            self.add_page()
            self.set_y(TOP + 2)
        x, y = MARGIN, self.get_y()
        inner = width - 2 * PAD
        self.color("White", "fill")
        self.color("CardLine", "draw")
        self.set_line_width(0.25)
        self.rect(x, y, width, h, style="DF", round_corners=True, corner_radius=4)
        # icon tile and title
        ty = y + (INLINE_PAD if self.inline(s) else PAD)
        self.color(s.bg, "fill")
        self.rect(x + PAD, ty, TILE, TILE, style="F", round_corners=True, corner_radius=2)
        self.font("Icons", 12)
        self.color(s.fg)
        self.set_xy(x + PAD, ty)
        self.cell(TILE, TILE, s.glyph, align="C")
        self.font("Display", 13, s.title)
        self.color("Ink")
        self.set_xy(x + PAD + TILE + 3, ty)
        if self.inline(s):
            title_w = self.get_string_width(s.title)
            self.cell(title_w, TILE, s.title)
            label, value = s.rows[0]
            lx = x + PAD + TILE + 3 + max(title_w + 8, 40)
            self.font("Body", 9.5)
            self.color("Muted")
            label_w = self.get_string_width(label)
            self.set_xy(lx, ty)
            self.cell(label_w, TILE, label)
            self.font("BodySemi", 10.5)
            self.color("Ink")
            self.set_xy(lx + label_w + 3, ty)
            self.cell(max(10.0, x + width - PAD - (lx + label_w + 3)), TILE, value)
            self.set_y(y + h + GAP)
            return
        self.cell(inner - TILE - 3 - (17 if s.art else 0), TILE, s.title)
        if s.art:
            # 14 mm in the corner, ending above the first label row (y + PAD + TILE + TITLE_GAP)
            self.image(str(HERE / s.art), x + width - PAD - 14, y + 2.0, 14, 14)
        box = self.note_box(s, width)
        if box:
            self.font("Body", 7.5)
            self.color("Muted")
            lines = self.text_height("Body", 7.5, box[1], NOTE_H, s.note) / NOTE_H
            self.set_xy(x + box[0], ty + (TILE - lines * NOTE_H) / 2)
            self.multi_cell(box[1], NOTE_H, s.note, align="L")
        cy = y + PAD + TILE + TITLE_GAP

        def para() -> float:
            self.font("Body", 10.5)
            self.color("Ink")
            self.set_xy(x + PAD, cy)
            self.multi_cell(inner, PARA_H, s.paragraph, align="L")
            return self.get_y() + ROW_GAP

        if s.paragraph and s.paragraph_first:
            cy = para()
        if s.rows:
            cy += self.grid(s.rows, x + PAD, cy, draw=True, width=width)
        if s.paragraph and not s.paragraph_first:
            cy = para()
        if s.note and not box:
            self.font("Body", 7.5)
            self.color("Muted")
            self.set_xy(x + PAD, cy)
            self.multi_cell(inner - (17 if s.art else 0), NOTE_H, s.note, align="L")
        self.set_y(y + h + GAP)


def render(bio: Biodata, photo: Optional[bytes], verified: bool, today: Optional[Date] = None) -> Tuple[bytes, int]:
    """(PDF bytes, pages). Every card is measured first and moved whole to the next page when it doesn't fit."""
    pdf = _Pdf(bio, verified and photo is not None, today or Date.today())
    pdf.add_page()
    text_bottom, photo_bottom = pdf.head(photo)
    sections = list(bio.sections)
    if photo is not None and sections and sections[0].paragraph and not sections[0].rows:
        # A short "About" fits beside the photo, under the name, instead of leaving that corner empty.
        side_w, y0 = CONTENT_W - PHOTO_W - 5, text_bottom + 3
        if y0 + pdf.card_height(sections[0], side_w) <= photo_bottom + SIDE_OVERHANG:
            pdf.set_y(y0)
            pdf.card(sections.pop(0), side_w)
            pdf.set_y(max(pdf.get_y(), photo_bottom + 5))
    for s in sections:
        pdf.card(s)
    pages = pdf.page_no()
    return bytes(pdf.output()), pages


def make(doc: Mapping[str, Any], chart: Optional[Mapping[str, Any]], opts: Options,
         contact: Optional[Tuple[str, str]], photo: Optional[bytes], verified: bool, *,
         with_link: Optional[str] = None, today: Optional[Date] = None) -> Tuple[bytes, int, Biodata]:
    """build + render. Past two pages it is built again, each time a little shorter (build's `short`): first the
    about texts are cut to 300 / 150 characters, then every typed value is kept to one line, then the about texts
    are left out. Real profiles stop at the first or second try."""
    for level in range(SQUEEZE_LEVELS):
        bio = build(doc, chart, opts, contact, with_link=with_link, today=today, short=level)
        pdf, pages = render(bio, photo, verified, today)
        if pages <= MAX_PAGES:
            break
    return pdf, pages, bio
