"""The biodata PDF (plan §2 C9, features W2-S3) and the TODAY_OBSERVANCE wiring in main.py (W1b-S-PDF).

The route runs on a bare FastAPI app with astro_api's Deps faked (the caller's three docs, the rate counter); the
Auth phone and the photo fetch are replaced. Firestore is an in-memory fake for the chat and horoscope wiring.
"""
import base64
import dataclasses
import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import astro_api
import astro_core
import astro_festivals
import biodata_pdf as bp
import biodata_route
import fates_astro
import main

ROOT = Path(__file__).resolve().parents[1]
ME = "me123456"
H = {"Authorization": "Bearer ok"}
TODAY = datetime.now(biodata_route.INDIA).date()
CDN = f"https://photos.shaadiparrot.com/users/{ME}/variants/p1_960.jpg"
OWN = f"https://firebasestorage.googleapis.com/v0/b/shaadiparrot.appspot.com/o/users%2F{ME}%2Fphotos%2Fp1.jpg?alt=media"
LINK = "https://www.shaadiparrot.com/kundli-match.html#with=1&m=10.22&g=f&n=Asha&s=p"
EMAIL = "asha.patil@example.com"

PROFILE = {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21", "birthTime": "07:40",
           "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN",
           "cityName": "Bengaluru", "stateName": "Karnataka", "countryIso2": "IN", "lat": 12.9719, "lon": 77.5937,
           "religion": "Hindu", "community": "Marathi", "heightDisplayText": "163 cm"}
PUBLIC = {"firstName": "Asha", "gender": "Female", "cityName": "Bengaluru", "lat": 12.97, "lon": 77.59,
          "isDiscoverable": True}
USER = {"gender": "Female", "email": EMAIL, "lifeStage": "single"}


def years_ago(n: int, plus_days: int = 0) -> str:
    try:
        d = TODAY.replace(year=TODAY.year - n)
    except ValueError:                       # 29 February
        d = TODAY.replace(year=TODAY.year - n, day=28)
    return (d + timedelta(days=plus_days)).isoformat()


def jpeg(w=900, h=1200, color=(200, 120, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def keep_astro_deps():
    saved = astro_api._deps
    yield
    astro_api.configure(saved) if saved is not None else None


@pytest.fixture()
def env(monkeypatch):
    """A client for POST /astro/biodata. env.make(profile=..., public=..., user=...) sets the caller's docs."""
    state = NS(profile=dict(PROFILE), public=dict(PUBLIC), user=dict(USER), rate_ok=True, rates=[], phone="",
               phone_reads=[], fetched=[], photo=jpeg(), renders=[])

    def load(uid):
        return (dict(state.profile), dict(state.public), dict(state.user)) if uid == ME else ({}, {}, {})

    def take_rate(*a):
        state.rates.append(a)
        return state.rate_ok

    astro_api._hits.clear()
    astro_api.configure(astro_api.Deps(
        db=lambda: None,
        verify_uid=lambda h: ME if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        resolve_place=main._resolve_birth_place,
        places_query=main._places_query,
        load_user_docs=load,
        is_premium=lambda u, p, now: False,
        complete=lambda *a, **k: ("", "stop"),
        take_rate=take_rate,
    ))

    def phone(uid):
        state.phone_reads.append(uid)
        return state.phone

    def fetch(url):
        state.fetched.append(url)
        return state.photo

    real_render = bp.render

    def render(bio, photo, verified, today=None):
        state.renders.append(NS(bio=bio, photo=photo, verified=verified))
        return real_render(bio, photo, verified, today)

    monkeypatch.setattr(biodata_route, "auth_phone", phone)
    monkeypatch.setattr(bp, "fetch_photo", fetch)
    monkeypatch.setattr(bp, "render", render)
    app = FastAPI()
    app.include_router(biodata_route.router)
    client = TestClient(app)

    def post(**body):
        return client.post("/astro/biodata", json=body, headers=H)

    def make(**docs):
        for k, v in docs.items():
            setattr(state, k, v)

    state.post, state.make = post, make
    state.bio = lambda: state.renders[-1].bio
    return state


def ok(r) -> dict:
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and base64.b64decode(body["pdf"]).startswith(b"%PDF-")
    return body


def build(doc=None, chart=None, **opts) -> bp.Biodata:
    o = bp.Options(**{k: v for k, v in opts.items() if k in {f.name for f in dataclasses.fields(bp.Options)}})
    return bp.build(bp.pick({}, {}, doc if doc is not None else PROFILE), chart, o, opts.get("contact"),
                    with_link=opts.get("with_link"), today=date(2026, 10, 11))


def chart_of(doc):
    return fates_astro.chart_from_birth(bp.fc._parse_date(doc["birthDate"]), bp.fc._parse_time(doc.get("birthTime")),
                                        main._resolve_birth_place(doc))


# ------------------------------------------------------------------ basics
def test_the_default_profile_makes_a_one_page_pdf(env):
    assert "profileFor" not in PROFILE            # a 1.0.7-era account still has a rishta profile
    body = ok(env.post())
    assert body["pages"] == 1 and body["photo"] is False and body["contact"] is False
    assert set(body) == {"ok", "pdf", "pages", "photo", "contact"}
    pdf = base64.b64decode(body["pdf"])
    assert b"/Title (Asha - Biodata)" in pdf and b"/Author (Shaadi Parrot)" in pdf and b"/Creator (Shaadi Parrot)" in pdf
    assert env.rates == [(ME, "biodataPdf", 30, "day")]
    text = env.bio().text()
    assert "Asha" in text and "Moon sign (Rashi)" in text      # the kundli chip is on by default


def test_the_route_is_on_the_main_app():
    assert any(getattr(r, "path", "") == "/astro/biodata" and "POST" in getattr(r, "methods", set())
               for r in main.app.routes)


def test_nothing_identifying_is_printed(env):
    env.make(profile={**PROFILE, "aboutMe": "Write to me at " + EMAIL + " any time."})
    body = ok(env.post(birth=True, caste=True, income=True))
    pdf = base64.b64decode(body["pdf"])
    assert ME.encode() not in pdf and EMAIL.encode() not in pdf and b"12.97" not in pdf
    text = env.bio().text()
    assert ME not in text and "77.59" not in text and "Female" not in text
    assert EMAIL in text                      # only because the person typed it into their own about text


# ------------------------------------------------------------------ gates
def test_a_kundli_only_account_gets_no_pdf(env):
    env.make(profile={**PROFILE, "profileFor": "none"})
    r = env.post()
    assert r.status_code == 403 and r.json()["detail"] == "no_rishta_profile"
    assert env.rates == []


@pytest.mark.parametrize("where,fields,code", [
    ("profile", {"moderationStatus": "banned"}, 403),
    ("public", {"moderationStatus": "banned"}, 403),
    ("public", {"moderationStatus": "suspended", "moderationUntil": "2999-01-01T00:00:00Z"}, 403),
    ("profile", {"moderationStatus": "suspended"}, 403),
    ("profile", {"moderationStatus": "suspended", "moderationUntil": "2001-01-01T00:00:00Z"}, 200),
])
def test_moderated_accounts_are_refused(env, where, fields, code):
    env.make(**{where: {**getattr(env, where), **fields}})
    r = env.post()
    assert r.status_code == code, r.text
    if code == 403:
        assert r.json()["detail"] == "not_allowed"


def test_unreadable_docs_are_temporarily_unavailable(env):
    env.make(profile={}, public={}, user={})
    r = env.post()
    assert r.status_code == 503 and r.json()["detail"] == "temporarily_unavailable"


def test_the_age_comes_from_the_birth_date(env):
    env.make(profile={k: v for k, v in PROFILE.items() if k != "birthDate"})
    r = env.post()
    assert r.status_code == 422 and r.json()["detail"] == "missing_birth_data"
    for gender, born, code in (("Male", years_ago(20), 422), ("Male", years_ago(21), 200),
                               ("Female", years_ago(18), 200), ("Female", years_ago(18, 1), 422),
                               ("", years_ago(18, 1), 422)):
        env.make(profile={**PROFILE, "gender": gender, "birthDate": born}, public={**PUBLIC, "gender": gender})
        r = env.post()
        assert r.status_code == code, (gender, born, r.text)
        if code == 422:
            assert r.json()["detail"] == "too_young"
    assert len(env.rates) == 2                # refused requests never count


def test_the_daily_cap(env):
    env.rate_ok = False
    r = env.post(contact=True)
    assert r.status_code == 429 and r.json()["detail"] == "daily_cap"
    assert env.phone_reads == [] and env.fetched == []


def test_text_the_font_cant_print_is_refused_and_emoji_are_dropped(env):
    env.make(profile={**PROFILE, "firstName": "आशा"}, public={**PUBLIC, "firstName": "आशा"})
    r = env.post()
    assert r.status_code == 422 and r.json()["detail"] == "unsupported_text"
    env.make(profile={**PROFILE, "aboutMe": "मुझे संगीत पसंद है"}, public=dict(PUBLIC))
    assert env.post().json()["detail"] == "unsupported_text"
    env.make(profile={**PROFILE, "aboutMe": "Love music 🎶❤️ and chai ☕, family first 👨‍👩‍👧 1️⃣"})
    ok(env.post())
    assert env.bio().section("about").paragraph == "Love music and chai , family first 1"
    # a field that isn't printed never refuses the PDF
    env.make(profile={**PROFILE, "caste": "ब्राह्मण"})
    ok(env.post(caste=False))


# ------------------------------------------------------------------ contact (C9)
def test_the_contact_line_is_the_accounts_own_phone(env):
    env.phone = "+919820000000"
    body = ok(env.post(contact=True))
    assert body["contact"] is True and env.phone_reads == [ME]
    assert env.bio().section("contact").rows == [("Me", "+91 98200 00000")]


@pytest.mark.parametrize("profile_for,relation,label", [
    ("son", "Mother", "Mother"), ("daughter", "father", "Father"), ("son", "", "Parent"),
    ("brother", "", "Sibling"), ("sister", "Sister", "Sister"), ("relative", "Uncle", "Uncle"),
    ("relative", "Other", "Relative"), ("friend", "Friend", "Friend"), ("self", "", "Me"),
])
def test_the_contact_label_is_the_creators_relation(env, profile_for, relation, label):
    env.phone = "+447700900123"
    env.make(profile={**PROFILE, "profileFor": profile_for, "creatorRelation": relation})
    ok(env.post(contact=True))
    assert env.bio().section("contact").rows == [(label, "+447700900123")]


def test_no_auth_phone_means_no_contact_line(env):
    env.phone = ""
    body = ok(env.post(contact=True))
    assert body["contact"] is False and env.bio().section("contact") is None and env.phone_reads == [ME]


def test_the_phone_is_only_read_when_the_chip_is_on(env):
    env.phone = "+919820000000"
    for body in ({}, {"contact": False}, {"contact": "Sunita (mother), +91 98200 00000"}, {"contact": 1}):
        assert ok(env.post(**body))["contact"] is False
    assert env.phone_reads == [] and env.bio().section("contact") is None


def test_phone_formatting():
    assert bp.format_phone("+919820000000") == "+91 98200 00000"
    assert bp.format_phone("+44 7700 900123") == "+447700900123"
    assert bp.format_phone("98200 00000") == "" and bp.format_phone(None) == "" and bp.format_phone("+91abc") == ""


# ------------------------------------------------------------------ chips
def test_birth_details_only_with_the_birth_chip():
    chart = chart_of(PROFILE)
    assert chart["timeKnown"] and chart["lagna"] is not None
    off = build(PROFILE, chart).text()
    for word in ("Date of birth", "Time of birth", "Place of birth", "21 Aug 1994", "7:40 am", "Pune", "pada",
                 "Rising sign"):
        assert word not in off, word
    on = build(PROFILE, chart, birth=True).text()
    for word in ("Date of birth: 21 Aug 1994", "Time of birth: 7:40 am", "Place of birth: Pune, Maharashtra",
                 "pada", "Rising sign (Lagna): "):
        assert word in on, word


def test_the_route_passes_the_birth_chip(env):
    ok(env.post())
    assert "pada" not in env.bio().text() and "Date of birth" not in env.bio().text()
    ok(env.post(birth=True))
    assert "pada" in env.bio().text() and "Time of birth: 7:40 am" in env.bio().text()


def test_without_a_birth_time_the_kundli_says_or():
    doc = {k: v for k, v in PROFILE.items() if k != "birthTime"}
    chart = chart_of(doc)
    assert not chart["timeKnown"] and len({n for _, n in chart["options"]}) > 1
    rows = dict(build(doc, chart, manglik=True, birth=True).section("kundli").rows)
    assert " or " in rows["Nakshatra"] and rows["Nakshatra"].endswith("(birth time not given)")
    assert "pada" not in rows["Nakshatra"] and "Rising sign (Lagna)" not in rows
    assert rows["Manglik"] == "Check (needs birth time)"
    # a day whose Moon changes sign: both signs, in the order the Moon passed them
    for day in range(1, 60):
        d = date(1994, 1, 1) + timedelta(days=day)
        c = fates_astro.chart_from_birth((d.year, d.month, d.day), None, None)
        if len({r for r, _ in c["options"]}) > 1:
            moon = dict(bp.kundli_rows(c, birth=False))["Moon sign (Rashi)"]
            first, second = moon.replace(" (birth time not given)", "").split(" or ")
            names = [r[0] for r in fates_astro.RASHIS]
            assert (names.index(second) - names.index(first)) % 12 == 1, moon
            break
    else:
        pytest.fail("no sign change in 60 days")


def test_the_kundli_and_manglik_chips():
    chart = chart_of(PROFILE)
    assert build(PROFILE, chart, kundli=False).section("kundli") is None
    only = build(PROFILE, chart, kundli=False, manglik=True).section("kundli")
    assert [label for label, _ in only.rows] == ["Manglik"]
    assert "Manglik" not in build(PROFILE, chart).text()          # off by default
    assert build(PROFILE, None).section("kundli") is None          # a chart that failed: no card
    card = build(PROFILE, chart, manglik=True).section("kundli")
    assert card.note == bp.KUNDLI_NOTE and card.art == bp.MITHU_ART


@pytest.mark.parametrize("status,extra,text", [
    ("yes", {}, "Yes"), ("partial", {}, "Partial (Anshik)"),
    ("no", {"cancelledBy": fates_astro.CANCEL_OWN_SIGN}, "No (Mars in its own or exalted sign)"),
    ("no", {}, "No"), ("check", {}, "Check (needs birth time)"),
])
def test_the_manglik_row_uses_the_c3_words(monkeypatch, status, extra, text):
    monkeypatch.setattr(fates_astro, "manglik_status", lambda c: {"status": status, **extra})
    assert bp.manglik_text({}) == text


def test_mars_in_its_own_sign_reads_no():
    chart = {"lagna": 0, "marsRashi": 0, "rashi": 6, "venusRashi": 6, "nak": 14, "pada": 1, "options": [[6, 14]],
             "timeKnown": True}
    assert dict(bp.kundli_rows(chart, birth=False, manglik=True))["Manglik"] == "No (Mars in its own or exalted sign)"


def test_caste_and_gotra_only_with_the_caste_chip():
    doc = {**PROFILE, "caste": "Brahmin - Iyer", "subCaste": "Vadama", "gotra": "Kashyap"}
    assert "Caste" not in build(doc).text() and "Kashyap" not in build(doc).text()
    rows = dict(build(doc, caste=True).section("personal").rows)
    assert rows["Caste"] == "Brahmin - Iyer" and rows["Sub-caste"] == "Vadama" and rows["Gotra"] == "Kashyap"
    rows = dict(build({**doc, "caste": "Prefer not to say", "gotra": "Don't know"}, caste=True).section("personal").rows)
    assert "Caste" not in rows and "Gotra" not in rows and rows["Sub-caste"] == "Vadama"
    assert dict(build({**doc, "caste": "Inter-caste / Mixed"}, caste=True).section("personal").rows)["Caste"] == \
        "Inter-caste / Mixed"


def test_income_only_with_the_chip_and_visibility_everyone():
    doc = {**PROFILE, "education": "MBA", "annualIncome": "10-15L"}
    assert "Annual income" not in build(doc).text()
    assert dict(build(doc, income=True).section("work").rows)["Annual income"] == "₹10–15 lakh"
    assert dict(build({**doc, "annualIncomeVisibility": "everyone"}, income=True).section("work").rows)[
        "Annual income"] == "₹10–15 lakh"
    for hidden in ({"annualIncomeVisibility": "matches"}, {"annualIncome": "prefer_not"}, {"annualIncome": "junk"}):
        assert "Annual income" not in build({**doc, **hidden}, income=True).text(), hidden


def test_the_route_reads_income_visibility_from_the_private_profile(env):
    env.make(profile={**PROFILE, "annualIncome": "10-15L", "annualIncomeVisibility": "matches"},
             public={**PUBLIC, "annualIncome": "10-15L"})
    ok(env.post(income=True))
    assert "Annual income" not in env.bio().text()


# ------------------------------------------------------------------ withLink (C9)
def test_a_valid_with_link_becomes_the_footer_line(env):
    body = ok(env.post(withLink=LINK))
    bio = env.bio()
    assert bio.link == LINK and bio.link_text == "Check your kundli match with Asha, free: shaadiparrot.com/kundli-match"
    assert f"/URI ({LINK})".encode() in base64.b64decode(body["pdf"])
    env.make(profile={**PROFILE, "profileFor": "son", "firstName": "Rahul Sharma", "gender": "Male"},
             public={**PUBLIC, "gender": "Male"})
    ok(env.post(withLink=LINK))
    assert env.bio().link_text.startswith("Check your kundli match with Rahul, free")


@pytest.mark.parametrize("link", [
    "https://evil.example/kundli-match.html#with=1&m=1.3",
    "http://www.shaadiparrot.com/kundli-match.html#with=1&m=1.3",
    "https://www.shaadiparrot.com/kundli-match.html#with=2&m=1.3",
    "https://www.shaadiparrot.com/kundli-match.html#with=1&",
    "https://www.shaadiparrot.com/kundli-match.html#with=1&m=1.3 click here",
    LINK + "x" * (161 - len(LINK)),
    123, None, "",
])
def test_any_other_link_is_ignored(env, link):
    body = ok(env.post(withLink=link))
    assert env.bio().link is None and env.bio().link_text == ""
    assert b"kundli-match" not in base64.b64decode(body["pdf"])


def test_the_link_rule():
    assert bp.valid_with_link(LINK) == LINK and bp.valid_with_link("  " + LINK + " ") == LINK
    assert bp.valid_with_link(LINK + "x" * (160 - len(LINK))) is not None
    assert bp.valid_with_link(LINK + "é") is None


# ------------------------------------------------------------------ photo and badge
def test_a_public_photo_is_fetched_from_the_card(env):
    env.make(public={**PUBLIC, "photos": [OWN], "photoMediumCdn": [CDN]})
    body = ok(env.post())
    assert body["photo"] is True and env.fetched == [CDN] and env.renders[-1].photo == env.photo
    ok(env.post(photo=False))
    assert env.fetched == [CDN] and env.renders[-1].photo is None


def test_a_private_profile_uses_its_own_links_never_the_blurs(env):
    blur = f"https://photos.shaadiparrot.com/users/{ME}/blur/p1.jpg"
    env.make(profile={**PROFILE, "photoPrivacy": "accepted", "photos": [OWN], "photoMediumCdn": [CDN]},
             public={**PUBLIC, "photos": [blur], "photoMediumCdn": [blur], "photoPrivacy": "accepted"})
    ok(env.post())
    assert env.fetched == [CDN]


@pytest.mark.parametrize("url", [
    f"https://photos.shaadiparrot.com/users/other999/variants/p1_960.jpg",
    f"https://photos.shaadiparrot.com.evil.com/users/{ME}/variants/p1_960.jpg",
    f"http://photos.shaadiparrot.com/users/{ME}/variants/p1_960.jpg",
    f"https://evil.com/users/{ME}/variants/p1_960.jpg",
    f"https://photos.shaadiparrot.com/users/{ME}/blur/p1.jpg",
    f"https://photos.shaadiparrot.com/users/{ME}/x/../../other999/p1.jpg",
    f"https://photos.shaadiparrot.com:8443/users/{ME}/variants/p1_960.jpg",
    f"https://firebasestorage.googleapis.com/v0/b/b/o/users%2F{ME}%2Fblur%2Fp1.jpg",
    f"https://firebasestorage.googleapis.com/v0/b/b/o/users%2F{ME}9%2Fphotos%2Fp1.jpg",
    f"https://firebasestorage.googleapis.com/v0/b/b/o/users%2Fother%2Fusers%2F{ME}%2Fp1.jpg",
    f"https://photos.shaadiparrot.com/users/{ME.upper()}/variants/p1_960.jpg",
])
def test_other_photo_links_are_never_fetched(env, url):
    env.make(public={**PUBLIC, "photos": [url], "photoMediumCdn": [url], "photoMedium": [url]})
    body = ok(env.post())
    assert body["photo"] is False and env.fetched == []
    assert not bp.allowed_photo_url(url, ME)


def test_the_mascot_is_no_photo(env):
    env.make(public={**PUBLIC, "photos": ["mascot.png"], "photoMediumCdn": [CDN]})
    assert ok(env.post())["photo"] is False and env.fetched == []
    assert bp.photo_url({}, {"photos": [f"https://x/o/users%2F{ME}%2Fmascot.png?alt=media"],
                             "photoMediumCdn": [CDN]}, ME) is None


def test_storage_links_need_the_own_folder_and_the_project_bucket(monkeypatch):
    url = f"https://firebasestorage.googleapis.com/v0/b/shaadiparrot.appspot.com/o/users%2F{ME}%2Fvariants%2Fp_960.jpg?alt=media&token=t"
    assert bp.allowed_photo_url(url, ME) and bp.allowed_photo_url(CDN, ME)
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "shaadiparrot.appspot.com")
    assert bp.allowed_photo_url(url, ME)
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "other-bucket")
    assert not bp.allowed_photo_url(url, ME)
    assert bp.photo_url({}, {"photos": [url], "photoMedium": [url]}, ME) is None


def test_the_photo_verified_badge(env):
    env.make(public={**PUBLIC, "photos": [OWN], "photoMediumCdn": [CDN], "isFaceVerified": False},
             profile={**PROFILE, "isFaceVerified": True, "faceVerified": True, "faceVerifiedPhotos": [OWN]})
    ok(env.post())
    assert env.renders[-1].verified is False             # only the card's flag counts (cleared on a photo change)
    env.make(public={**env.public, "isFaceVerified": True})
    ok(env.post())
    assert env.renders[-1].verified is True
    env.make(profile={**env.profile, "faceVerifiedPhotos": ["https://other/photo.jpg"]})
    ok(env.post())
    assert env.renders[-1].verified is False             # the printed photo is not the verified one
    env.make(profile={**env.profile, "faceVerifiedPhotos": [OWN]})
    env.photo = None                                     # the fetch failed: no photo, no pill
    body = ok(env.post())
    assert body["photo"] is False and env.renders[-1].verified is False
    ok(env.post(photo=False))
    assert env.renders[-1].verified is False


def test_photo_verified_in_private_mode_follows_the_private_photo():
    profile = {"photoPrivacy": "accepted", "photos": [OWN], "faceVerifiedPhotos": [OWN]}
    assert bp.photo_verified(True, profile, {"isFaceVerified": True, "photos": ["https://blur"]})
    assert not bp.photo_verified(False, profile, {"isFaceVerified": True})
    assert not bp.photo_verified(True, profile, {"isFaceVerified": "true"})


class FakeResp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._body, self.headers, self.closed = status, body, headers or {}, False

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def close(self):
        self.closed = True


def test_fetch_photo(monkeypatch):
    calls = []

    def get(url, **kw):
        calls.append((url, kw))
        return resp

    monkeypatch.setattr(bp.requests, "get", get)
    resp = FakeResp(200, jpeg(1200, 900))
    out = bp.fetch_photo(CDN)
    assert calls[0] == (CDN, {"timeout": (3, 8), "allow_redirects": False, "stream": True}) and resp.closed
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and img.size == (600, 750) and not img.info.get("exif")
    resp = FakeResp(302, b"", {"Location": "https://evil.com/x.jpg"})
    assert bp.fetch_photo(CDN) is None
    resp = FakeResp(200, b"x" * (bp.PHOTO_MAX_BYTES + 10))
    assert bp.fetch_photo(CDN) is None
    resp = FakeResp(200, b"not an image")
    assert bp.fetch_photo(CDN) is None

    def boom(url, **kw):
        raise OSError("down")

    monkeypatch.setattr(bp.requests, "get", boom)
    assert bp.fetch_photo(CDN) is None


# ------------------------------------------------------------------ text
def test_prefer_not_answers_are_left_out_and_settings_fill_in():
    rows = dict(build({**PROFILE, "religion": "Prefer not to say", "settings_religion": "Jain",
                       "community": "", "settings_community": "Gujarati"}).section("personal").rows)
    assert rows["Religion"] == "Jain" and rows["Mother tongue"] == "Gujarati"
    bio = build({**PROFILE, "religion": "Prefer not to say", "settings_religion": ""})
    assert "Religion" not in dict(bio.section("personal").rows) and "Prefer not" not in bio.text()


def test_heights_names_and_lines():
    assert bp.height("163 cm") == "5'4\" (163 cm)" and bp.height("5'12\"") == "6'0\"" and bp.height("5'4\"") == "5'4\""
    assert bp.height("4'12") == "5'0\"" and bp.height("99 cm") == "99 cm"
    assert bp.first_word("  Asha   Patil ") == "Asha"
    assert bp.first_word("Rene" + chr(0x301)) == "Ren" + chr(0xE9)     # a decomposed accent is composed first
    bio = build({**PROFILE, "firstName": "Asha Patil", "maritalStatus": "Never married"})
    assert bio.name == "Asha" and "Patil" not in bio.text()
    assert bio.facts_lines[0] == f"{bp.age((1994, 8, 21), date(2026, 10, 11))} yrs · 5'4\" (163 cm)"
    assert dict(bio.section("personal").rows)["Marital status"] == "Never married"
    assert bio.facts_lines[1] == "Hindu · Marathi · Lives in Bengaluru"
    divorced = build({**PROFILE, "maritalStatus": "Divorced", "children": "No children"})
    assert divorced.facts_lines[0].endswith("Divorced · no children")
    assert dict(divorced.section("personal").rows)["Children"] == "No children"


def test_family_and_managed_wording():
    doc = {**PROFILE, "profileFor": "son", "creatorRelation": "Mother", "firstName": "Rahul", "gender": "Male",
           "aboutMe": "Calm and kind.", "familyType": "Joint family", "familyValues": "Traditional",
           "fatherOccupation": "Business", "brothers": "None", "sisters": "None", "aboutFamily": "We live in Pune."}
    bio = build(doc)
    assert bio.pills == ["Managed by mother"] and bio.section("about").title == "About Rahul"
    fam = bio.section("family")
    assert dict(fam.rows)["Family type"] == "Joint family, traditional values" and dict(fam.rows)["Siblings"] == \
        "No siblings" and fam.paragraph == "We live in Pune."
    assert bp.siblings("1", "2") == "1 brother, 2 sisters" and bp.siblings("3+", "") == "3+ brothers"
    assert build({**PROFILE, "aboutMe": "Hi"}).section("about").title == "About me"
    assert bp.managed_by("daughter") == "Managed by a parent" and bp.managed_by("self", "Mother") == ""


def test_links_never_go_on_the_pdf():
    bio = build({**PROFILE, "aboutMe": "Find me on wa.me/919820000000 or www.example.com/asha, thanks",
                 "aboutFamily": "See https://t.me/family"})
    assert bio.section("about").paragraph == "Find me on or thanks" and "t.me" not in bio.text()
    assert bio.section("family").paragraph == "See"


def test_long_texts_are_cut():
    bio = build({**PROFILE, "aboutMe": "word " * 400, "aboutFamily": "fam " * 200, "education": "x" * 200})
    assert len(bio.section("about").paragraph) <= bp.ABOUT_MAX and bio.section("about").paragraph.endswith("…")
    assert len(bio.section("family").paragraph) <= bp.FAMILY_MAX
    assert len(dict(bio.section("work").rows)["Education"]) <= bp.VALUE_MAX


# ------------------------------------------------------------------ limits
def test_the_worst_case_stays_within_two_pages(env):
    long = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
    worst = {**PROFILE, **{k: long for k in ("heightDisplayText", "maritalStatus", "religion", "community", "diet",
                                               "education", "profession", "familyType", "familyValues",
                                               "fatherOccupation", "motherOccupation", "familyCity", "cityName",
                                               "stateName", "subCaste", "gotra", "children", "birthCityName",
                                               "birthStateName")},
             "maritalStatus": "Divorced", "caste": long, "brothers": "3+", "sisters": "3+", "profileFor": "daughter",
             "creatorRelation": "Mother", "aboutMe": ("I love long walks and longer talks. " * 60)[:2000],
             "aboutFamily": ("We are a big, warm family. " * 80)[:2000], "annualIncome": "1Cr+",
             "photoMediumCdn": [CDN]}
    env.phone = "+919820000000"
    env.make(profile=worst, public={**PUBLIC, "photos": [OWN], "photoMediumCdn": [CDN], "isFaceVerified": True})
    body = ok(env.post(photo=True, kundli=True, manglik=True, birth=True, caste=True, income=True, contact=True,
                       withLink=LINK))
    assert body["pages"] <= 2 and body["photo"] is True and body["contact"] is True


def test_a_typical_full_profile_fits_one_page():
    doc = {"firstName": "Rahul", "gender": "Male", "birthDate": "1996-02-10", "cityName": "Pune",
           "stateName": "Maharashtra", "heightDisplayText": "178 cm", "maritalStatus": "Never married",
           "religion": "Hindu", "community": "Marathi", "diet": "Vegetarian", "education": "Bachelor's",
           "profession": "Engineer", "familyType": "Nuclear family", "familyValues": "Moderate",
           "fatherOccupation": "Retired", "motherOccupation": "Homemaker", "brothers": "None", "sisters": "1",
           "familyCity": "Pune", "profileFor": "self",
           "aboutMe": "Engineer at an auto company, cricket on Sundays, and the family's designated chai maker. "
                      "Looking for someone warm and honest who likes small adventures."}
    chart = fates_astro.chart_from_birth((1996, 2, 10), None, None)
    pdf, pages, bio = bp.make(bp.pick({}, {}, doc), chart, bp.Options(manglik=True), ("Me", "+91 98200 00000"),
                              jpeg(600, 750), True, with_link=LINK, today=date(2026, 10, 11))
    assert pages == 1 and [s.key for s in bio.sections] == ["about", "personal", "work", "family", "kundli", "contact"]


def test_render_shortens_the_about_texts_past_two_pages(monkeypatch):
    seen = []
    real = bp.render

    def render(bio, photo, verified, today=None):
        seen.append(len(bio.section("about").paragraph))
        pdf, pages = real(bio, photo, verified, today)
        return pdf, (3 if len(seen) == 1 else pages)

    monkeypatch.setattr(bp, "render", render)
    doc = bp.pick({}, {}, {**PROFILE, "aboutMe": "word " * 400})
    _, pages, bio = bp.make(doc, None, bp.Options(), None, None, False, today=date(2026, 10, 11))
    assert seen[0] > bp.ABOUT_SHORT >= seen[1] and pages == 1 and len(seen) == 2


def test_oversized_fields_are_squeezed_until_two_pages(monkeypatch):
    seen = []
    real = bp.render

    def render(bio, photo, verified, today=None):
        seen.append(bio)
        pdf, pages = real(bio, photo, verified, today)
        return pdf, (3 if len(seen) < 4 else pages)

    monkeypatch.setattr(bp, "render", render)
    long = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do"
    doc = bp.pick({}, {}, {**PROFILE, "aboutMe": "word " * 400, "aboutFamily": "fam " * 100, "education": long})
    chart = chart_of(PROFILE)
    _, pages, bio = bp.make(doc, chart, bp.Options(manglik=True), None, None, False, today=date(2026, 10, 11))
    assert len(seen) == 4 and pages <= 2
    assert len(dict(seen[1].section("work").rows)["Education"]) > bp.VALUE_COMPACT
    assert len(dict(seen[2].section("work").rows)["Education"]) <= bp.VALUE_COMPACT
    assert seen[2].section("about") is not None and seen[3].section("about") is None
    assert seen[3].section("family") is None or seen[3].section("family").paragraph == ""
    kundli = dict(seen[3].section("kundli").rows)          # the kundli words are never cut
    assert kundli["Manglik"] == dict(seen[0].section("kundli").rows)["Manglik"]


def test_the_assets_ship_with_their_licences():
    for path in (*bp.FONT_FILES.values(), bp.APP_ICON, bp.MITHU_ART):
        assert (ROOT / path).is_file(), path
    for lic in ("PlusJakartaSans-OFL.txt", "BricolageGrotesque-OFL.txt", "Lucide-ISC.txt"):
        assert (ROOT / "assets" / "biodata" / "licenses" / lic).is_file(), lic
    assert all(ord(g) in bp._cmap(bp.FONT_FILES["Icons"]) for g in bp.GLYPHS.values())
    assert all(ord(c) in bp.BODY_CHARS for c in "₹–·…'\"")
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").split()
    assert "fpdf2==2.8.9" in reqs and any(r.startswith("Pillow==") for r in reqs)
    assert any(r.startswith("fonttools==") for r in reqs) and any(r.startswith("defusedxml==") for r in reqs)


# ------------------------------------------------------------------ main.py: astro_api Deps
def test_the_lease_helpers_go_to_astro_api_only_when_deps_has_them():
    @dataclasses.dataclass
    class Old:
        db: object = None

    @dataclasses.dataclass
    class New:
        db: object = None
        take_lease: object = None
        release_lease: object = None

    assert main._astro_deps_extra(Old) == {}
    assert main._astro_deps_extra(New) == {"take_lease": main._take_lease, "release_lease": main._release_lease}
    extra = main._astro_deps_extra(astro_api.Deps)        # whichever W1b merge order this tree has
    deps = astro_api.Deps(db=lambda: None, verify_uid=lambda h: ME, resolve_place=lambda p: None,
                          places_query=lambda s, a: [], load_user_docs=lambda u: ({}, {}, {}),
                          is_premium=lambda *a: False, complete=lambda *a, **k: ("", "stop"),
                          take_rate=lambda *a: True, **extra)
    if "take_lease" in astro_api.Deps.__dataclass_fields__:
        assert deps.take_lease is main._take_lease and deps.release_lease is main._release_lease
    else:
        assert extra == {}


# ------------------------------------------------------------------ main.py: TODAY_OBSERVANCE
DELHI_HINDU = {"religion": "Hindu", "community": "Hindi belt / North Indian", "cityName": "New Delhi",
               "stateName": "Delhi", "countryIso2": "IN"}
NO_PLACE = {"religion": "Hindu", "community": "Hindi belt / North Indian", "stateName": "Delhi"}


def _minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def test_karva_chauth_carries_the_moonrise_of_the_readers_city():
    line = main._today_observance(DELHI_HINDU, date(2026, 10, 29))
    assert line.startswith("Karva Chauth:")
    m = re.search(r"Moonrise in New Delhi: (\d\d:\d\d)\.", line)
    assert m and abs(_minutes(m.group(1)) - _minutes("20:12")) <= 3, line   # published: about 20:12
    # without a place, no time is made up
    bare = main._today_observance(NO_PLACE, date(2026, 10, 29))
    assert bare.startswith("Karva Chauth:") and "Moonrise in" not in bare and not re.search(r"\d\d:\d\d", bare)


def test_a_festival_without_a_time_kind_gets_no_time():
    line = main._today_observance(DELHI_HINDU, date(2026, 10, 11))
    assert line.startswith("Sharad Navratri begins") and not re.search(r"\d\d:\d\d", line)


def test_diwali_carries_the_pradosh_window():
    line = main._today_observance(DELHI_HINDU, date(2026, 11, 8))
    assert line.startswith("Diwali:") and re.search(r"Pradosh time in New Delhi: \d\d:\d\d–\d\d:\d\d\.", line), line


def test_the_readers_own_days_only():
    assert main._today_observance({"cityName": "New Delhi", "countryIso2": "IN"}, date(2026, 10, 29)) == "none"
    assert main._today_observance({**DELHI_HINDU, "religion": "Muslim"}, date(2026, 11, 8)) == "none"
    # users.celebrates counts when the religion isn't given
    celebrates = {k: v for k, v in DELHI_HINDU.items() if k != "religion"}
    assert main._today_observance({**celebrates, "celebrates": ["hindu"]}, date(2026, 10, 29)).startswith("Karva")
    # the profile's coordinates find the city when there is no city name
    point = {**NO_PLACE, "lat": 28.6139, "lon": 77.209}
    assert re.search(r"Moonrise in .+: \d\d:\d\d\.", main._today_observance(point, date(2026, 10, 29)))


def test_fasts_need_a_known_place():
    vr = astro_core.vrats_between(date(2026, 12, 1), 45, 28.6139, 77.209, "Asia/Kolkata")
    fest_days = {f["date"] for f in astro_festivals.FESTIVALS}
    eka = next(v for v in vr if v["kind"] == "ekadashi" and v["date"] not in fest_days)
    day = date.fromisoformat(eka["date"])
    assert eka["name"] in main._today_observance(DELHI_HINDU, day)
    assert main._today_observance(NO_PLACE, day) == "none"
    pradosh = next(v for v in vr if v["kind"] == "pradosh" and v["date"] not in fest_days)
    line = main._today_observance(DELHI_HINDU, date.fromisoformat(pradosh["date"]))
    assert line.startswith(pradosh["name"]) and "Pradosh time in New Delhi: " in line, line


def test_the_observance_never_fails_a_reply(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("table broken")

    monkeypatch.setattr(astro_festivals, "observance_for", boom)
    assert main._today_observance(DELHI_HINDU, date(2026, 10, 29)) == "none"


def test_the_observance_line_is_not_cut_short():
    long = "Karva Chauth: " + "x" * 380 + " Moonrise in New Delhi: 20:12."
    assert main._observance_text(long).endswith("20:12.")
    assert main._day_facts({}, observance="Diwali") == "ASTRO_MODE=full | TODAY_OBSERVANCE=Diwali"


# ------------------------------------------------------------------ main.py: wiring into the chat and the horoscope
class _Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d, self.exists = ref, ref.id, data, data is not None

    def to_dict(self):
        return None if self._d is None else json.loads(json.dumps(self._d))


class _Ref:
    def __init__(self, db, path):
        self._db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def collection(self, name):
        return _Col(self._db, f"{self.path}/{name}")

    def get(self, transaction=None):
        return _Snap(self, self._db.docs.get(self.path))

    def set(self, data, merge=False):
        self._db.write(self.path, data, merge)

    def delete(self):
        self._db.docs.pop(self.path, None)


class _Col:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return _Ref(self._db, f"{self.path}/{id_}")

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        return self

    def where(self, *a, **k):
        return self

    def stream(self):
        prefix = self.path + "/"
        return [_Snap(_Ref(self._db, p), d) for p, d in list(self._db.docs.items())
                if p.startswith(prefix) and "/" not in p[len(prefix):]]


class _Batch:
    def __init__(self, db):
        self._db, self._ops = db, []

    def create(self, ref, data):
        self._ops.append((ref, data, False))

    def set(self, ref, data, merge=False):
        self._ops.append((ref, data, merge))

    def commit(self):
        for ref, data, merge in self._ops:
            self._db.write(ref.path, data, merge)


class _Tx:
    def __init__(self):
        self.ops = []

    def set(self, ref, data, merge=False):
        self.ops.append((ref.path, data, merge))


class _Db:
    def __init__(self, docs):
        self.docs = {k: dict(v) for k, v in docs.items()}

    def collection(self, name):
        return _Col(self, name)

    def get_all(self, refs):
        return [_Snap(r, self.docs.get(r.path)) for r in list(refs)]

    def batch(self):
        return _Batch(self)

    def write(self, path, data, merge):
        cur = dict(self.docs.get(path) or {}) if merge else {}
        cur.update(data)
        self.docs[path] = cur

    def run(self, work):
        tx = _Tx()
        out = work(tx)
        for path, data, merge in tx.ops:
            self.write(path, data, merge)
        return out


@pytest.fixture()
def wired(monkeypatch):
    uid = "obs123456"
    db = _Db({f"profiles/{uid}": {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21",
                                  "profileFor": "self", "religion": "Hindu", "cityName": "New Delhi"},
              f"users/{uid}": {"gender": "Female", "celebrates": ["hindu"]}})
    sent, asked = [], []

    def complete(messages, max_tokens, temperature=None, kind="chat", timeout=None):
        sent.append({"kind": kind, "messages": messages})
        return f"Good morning!\nTheme: Reading {len(sent)}\n🌙 Overall mood\n• Calm.", "stop"

    def observance(doc, day=None):
        asked.append((dict(doc), day))
        return "Karva Chauth: The fast ends at moonrise. Moonrise in New Delhi: 20:12."

    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_verify_firebase_token_or_401", lambda a: uid)
    monkeypatch.setattr(main, "_take_rate", lambda *a, **k: True)
    monkeypatch.setattr(main, "_reserve_parrot_reply", lambda *a, **k: None)
    monkeypatch.setattr(main, "_deepseek_complete", complete)
    monkeypatch.setattr(main, "_take_lease", lambda name, ttl: True)
    monkeypatch.setattr(main, "_release_lease", lambda name: None)
    monkeypatch.setattr(main, "_run_transaction", db.run)
    monkeypatch.setattr(main, "_today_observance", observance)
    return NS(uid=uid, db=db, sent=sent, asked=asked, client=TestClient(main.app))


def test_the_chat_gets_todays_observance(wired):
    r = wired.client.post("/ai-chat", json={"text": "How is my day?", "thread_id": "t1"}, headers=H)
    assert r.status_code == 200, r.text
    facts = [m["content"] for m in wired.sent[-1]["messages"] if m["content"].startswith("ASTRO_MODE=")]
    assert facts == ["ASTRO_MODE=full | TODAY_OBSERVANCE=Karva Chauth: The fast ends at moonrise. "
                     "Moonrise in New Delhi: 20:12."]
    doc, day = wired.asked[-1]
    assert doc["celebrates"] == ["hindu"] and doc["cityName"] == "New Delhi" and day is None


def test_the_horoscope_and_its_rewrite_get_todays_observance(wired):
    tz = "Asia/Kolkata"
    r = wired.client.post("/daily-horoscope", json={"tz": tz}, headers=H)
    assert r.status_code == 200 and r.json()["ok"], r.text
    assert "TODAY_OBSERVANCE: Karva Chauth: The fast ends at moonrise. Moonrise in New Delhi: 20:12." in \
        wired.sent[-1]["messages"][1]["content"]
    doc, day = wired.asked[-1]
    assert doc["celebrates"] == ["hindu"] and day == datetime.now(main._zone(tz)).date()
    # new birth details the same day: the rewrite gets it too
    wired.db.docs[f"profiles/{wired.uid}"]["birthTime"] = "07:40"
    r = wired.client.post("/daily-horoscope", json={"tz": tz}, headers=H)
    assert r.status_code == 200 and r.json()["cached"] is False, r.text
    assert len(wired.sent) == 2 and "Moonrise in New Delhi: 20:12." in wired.sent[-1]["messages"][1]["content"]
    assert "Give a clock time for it only when TODAY_OBSERVANCE has one." in main._HOROSCOPE_SYSTEM_PROMPT


def test_the_approximate_birth_time_moves_the_horoscope_moon():
    base = {"birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN"}
    for day in range(1, 29):                                  # a day whose Moon changes nakshatra before noon
        base["birthDate"] = f"1994-08-{day:02d}"
        noon = main._natal_chart(base)
        night = main._natal_chart({**base, "birthTimeApprox": "Night"})
        if night["nakshatra"] != noon["nakshatra"]:
            break
    else:
        pytest.fail("no nakshatra change between 03:00 and 12:00 in August 1994")
    assert night["lagna"] is None and not night["time_known"]
    chart = astro_core.birth_chart((1994, 8, day), None, main._resolve_birth_place(base), approx="night")
    assert chart["moon"]["nak"] == night["nakshatra_idx"]     # the same hour as the Kundli tab's chart
    assert main._natal_chart({**base, "birthTimeApprox": "sometime"})["nakshatra"] == noon["nakshatra"]
    assert main._MOON_HOUSE_THEMES is astro_core.HOUSE_THEMES
