"""POST /astro/biodata  my biodata as a PDF to share (nothing stored).

Plan §2 C9: the request carries the chips {photo, kundli, manglik, birth, caste, income, contact} (on/off) and
withLink (the "Check your kundli match with {Name}, free" footer link, ignored unless it is a kundli-match "with"
link of at most 160 characters). The answer is {ok, pdf (base64), pages, photo, contact}: photo and contact say
whether they made it onto the PDF.

It shares astro_api's sign-in check, per-minute limit and Deps (configured by main.py); this router only adds the
route. The caller's own docs are read once (profiles, publicProfiles, users), through biodata_pdf.FIELDS only.
The one write is the serverRate/{uid}.biodataPdf counter (30 a day). Logs carry the uid and the on/off flags,
never a profile value.

Refusals, in this order:
- 503 temporarily_unavailable: neither the profile nor the card could be read;
- 403 no_rishta_profile: a kundli-only account (profileFor "none"; a missing profileFor is a rishta profile);
- 403 not_allowed: banned, or suspended until later (on the profile or on the card);
- 422 missing_birth_data: no valid birth date (the age is printed);
- 422 too_young: a groom under 21 or anyone under 18 (India time);
- 429 daily_cap: 30 PDFs a day (429 slow_down is the shared per-minute limit);
- 422 unsupported_text: a letter the PDF font can't print (Devanagari, for now).
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Annotated, Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, BeforeValidator

import astro_api
import biodata_pdf as bp
import fates_astro_chart as fc

logger = logging.getLogger("shaadiparrot-cloudrun.biodata")
router = APIRouter(prefix="/astro")

DAILY_CAP = 30
INDIA = ZoneInfo("Asia/Kolkata")


def _on(v: Any) -> bool:
    """A chip is on only when it is JSON true: a string, a number or null leaves it off (an older contact text
    never turns the phone line on)."""
    return v is True


Chip = Annotated[bool, BeforeValidator(_on)]
LinkText = Annotated[Optional[str], BeforeValidator(lambda v: v if isinstance(v, str) else None)]


class BiodataReq(BaseModel):
    photo: Chip = True
    kundli: Chip = True
    manglik: Chip = False
    birth: Chip = False
    caste: Chip = False
    income: Chip = False
    contact: Chip = False
    withLink: LinkText = None


def auth_phone(uid: str) -> str:
    """The account's own verified sign-in phone (Firebase Auth), or "" (a Google-only account, or Auth not reached).
    Read only when the contact chip is on."""
    try:
        from firebase_admin import auth
        return str(auth.get_user(uid).phone_number or "")
    except Exception:
        logger.warning("biodata_pdf auth phone read failed uid=%s", uid)
        return ""


def _refuse(status: int, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail=code)


@router.post("/biodata")
def biodata(body: BiodataReq, authorization: Optional[str] = Header(default=None)):
    uid = astro_api._auth(authorization)
    profile, public, user = astro_api._d().load_user_docs(uid)
    profile, public, user = profile or {}, public or {}, user or {}
    if not profile and not public:
        raise _refuse(503, "temporarily_unavailable")
    doc = bp.pick(public, user, profile)

    if str(doc.get("profileFor") or "").strip().lower() == "none":
        raise _refuse(403, "no_rishta_profile")
    if astro_api._moderated(profile) or astro_api._moderated(public):
        raise _refuse(403, "not_allowed")
    born = fc._parse_date(doc.get("birthDate"))
    if born is None:
        raise _refuse(422, "missing_birth_data")
    today = datetime.now(INDIA).date()
    if bp.age(born, today) < (21 if astro_api._gender(doc.get("gender")) == "male" else 18):
        raise _refuse(422, "too_young")
    if not astro_api._d().take_rate(uid, "biodataPdf", DAILY_CAP, "day"):
        raise _refuse(429, "daily_cap")

    contact = None
    if body.contact:
        phone = bp.format_phone(auth_phone(uid))
        if phone:
            contact = (bp.contact_label(doc), phone)

    chart = None
    if body.kundli or body.manglik:
        try:
            chart = astro_api._chart_for(doc)[0]
        except Exception:
            logger.warning("biodata_pdf chart failed uid=%s", uid)   # the Kundli card is left out
            chart = None

    opts = bp.Options(photo=body.photo, kundli=body.kundli, manglik=body.manglik, birth=body.birth,
                      caste=body.caste, income=body.income)
    link = bp.valid_with_link(body.withLink)
    try:
        bp.build(doc, chart, opts, contact, with_link=link, today=today)
    except bp.UnsupportedText:
        raise _refuse(422, "unsupported_text")

    photo = None
    if body.photo:
        url = bp.photo_url(profile, public, uid)
        if url:
            photo = bp.fetch_photo(url)
    verified = bp.photo_verified(photo is not None, profile, public)

    try:
        pdf, pages, _ = bp.make(doc, chart, opts, contact, photo, verified, with_link=link, today=today)
    except bp.UnsupportedText:
        raise _refuse(422, "unsupported_text")
    except Exception:
        logger.exception("biodata_pdf render failed uid=%s", uid)
        raise _refuse(503, "temporarily_unavailable")

    logger.info("biodata_pdf uid=%s pages=%s photo=%s kundli=%s manglik=%s birth=%s caste=%s income=%s contact=%s "
                "link=%s", uid, pages, photo is not None, body.kundli, body.manglik, body.birth, body.caste,
                body.income, contact is not None, link is not None)
    return {"ok": True, "pdf": base64.b64encode(pdf).decode("ascii"), "pages": pages, "photo": photo is not None,
            "contact": contact is not None}
