"""astro_festivals: the all-India festival table, the per-person lists for Today, the old list for 1.0.7 apps and
the TODAY_OBSERVANCE line (engagement design N3, tests from N8)."""
from datetime import date, timedelta

import pytest

import astro_core as ac
import astro_festivals as af

DELHI = (28.6139, 77.209, "Asia/Kolkata")
KOLKATA = (22.5726, 88.3639, "Asia/Kolkata")
BLR = (12.9719, 77.5937, "Asia/Kolkata")

SCHEMA = {"date", "id", "name", "note", "greeting", "faiths", "communities", "states", "tier", "kind",
          "moonSighting", "timeKind", "officeNote", "source"}
HINDU_DELHI = {"religion": "Hindu", "stateName": "Delhi"}
BENGALI = {"religion": "Hindu", "community": "Bengali", "stateName": "West Bengal"}
MUSLIM = {"religion": "Muslim"}


def up(start, prefs=None, place=DELHI, limit=af.FESTIVAL_LIMIT, vrats=ac.vrats_between):
    return af.upcoming(start, af.FESTIVAL_DAYS, *place, prefs, limit, vrats=vrats)


def keys(rows):
    return {(r["date"], r["id"]) for r in rows}


def by_id(id_):
    return {e["date"]: e for e in af.FESTIVALS if e["id"] == id_}


def easter(year):
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l_) // 433
    month = (h + l_ - 7 * m + 90) // 25
    return date(year, month, (h + l_ - 7 * m + 33 * month + 19) % 32)


# ------------------------------------------------------------------ the table
def test_every_entry_has_the_full_schema_and_who_keeps_it():
    for e in af.FESTIVALS + af.AWAITING_CONFIRMATION:
        assert set(e) == SCHEMA, e
        date.fromisoformat(e["date"])
        for f in ("id", "name", "source"):
            assert isinstance(e[f], str) and e[f].strip(), (f, e)
        assert isinstance(e["note"], str) and isinstance(e["greeting"], str), e
        assert e["tier"] in af.TIERS and e["kind"] in af.KINDS, e
        assert e["timeKind"] in af.TIME_KINDS or e["timeKind"] is None, e
        assert isinstance(e["moonSighting"], bool), e
        assert e["officeNote"] is None or (isinstance(e["officeNote"], str) and e["officeNote"]), e
        assert set(e["faiths"]) <= set(af.FAITHS), e
        assert set(e["communities"]) <= set(af.COMMUNITIES), e
        assert set(e["states"]) <= set(af.STATES), e
        # who keeps it: a faith, a community or a state, or everyone (national)
        assert e["faiths"] or e["communities"] or e["states"] or e["tier"] == "national", e
        if e["kind"] == "national":
            assert e["tier"] == "national", e
        if e["tier"] == "regional":
            assert e["communities"] or e["states"], e
        if e["tier"] in ("major", "faith"):
            assert e["faiths"], e


def test_ids_are_unique_per_date_and_the_table_is_sorted():
    days = [e["date"] for e in af.FESTIVALS]
    assert days == sorted(days)
    pairs = [(e["date"], e["id"]) for e in af.FESTIVALS]
    assert len(pairs) == len(set(pairs))
    assert days[0] <= "2026-10-11"


def test_festival_list_runs_ahead():
    # Today's festival card goes quiet when the table runs out: extend astro_festivals.FESTIVALS before this fails.
    last = max(date.fromisoformat(e["date"]) for e in af.FESTIVALS)
    assert last - date.today() >= timedelta(days=60)


def test_tiers():
    majors = {e["id"] for e in af.FESTIVALS if e["tier"] == "major"}
    assert majors == {"diwali", "holi", "dussehra", "durga_puja", "eid_ul_fitr", "bakrid", "christmas",
                      "good_friday", "easter", "gurpurab", "baisakhi", "mahavir_jayanti", "buddha_purnima",
                      "parsi_navroz"}
    national = {e["id"] for e in af.FESTIVALS if e["tier"] == "national"}
    assert {"republic_day", "independence_day", "gandhi_jayanti", "new_year"} <= national
    assert national <= {"republic_day", "independence_day", "gandhi_jayanti", "new_year", "ambedkar_jayanti"}


def test_durga_puja_follows_the_bengal_saptami():
    # Saptami at Kolkata sunrise is 18 Oct 2026 (DoPT RH; WB 4188-F(P2)); the old list had 17 Oct (Shashthi).
    assert sorted(by_id("durga_puja")) == ["2026-10-18", "2027-10-06", "2028-09-25"]
    for d in by_id("durga_puja"):
        assert ac.panchang(date.fromisoformat(d), *KOLKATA)["tithi"]["name"] == "Saptami", d
    assert "2026-10-19" in by_id("durga_ashtami") and "2026-10-20" in by_id("durga_navami")
    assert "2026-10-21" in by_id("bijoya_dashami")
    assert "2027-10-07" in by_id("durga_ashtami") and "2027-10-08" in by_id("durga_navami")
    for id_ in ("durga_ashtami", "durga_navami", "bijoya_dashami"):
        for e in by_id(id_).values():
            assert e["communities"] == ["bengali"] and set(e["states"]) == {"WB", "AS", "TR", "OD", "JH"}


def test_gazetted_days_of_2027():
    assert "2027-03-26" in by_id("good_friday")
    assert "2027-06-16" in by_id("muharram")
    assert "2027-08-15" in by_id("milad_un_nabi")


def test_moon_sighting_days_say_expected():
    moon_ids = {"eid_ul_fitr", "bakrid", "muharram", "milad_un_nabi", "ramzan_start", "jamat_ul_vida", "hazarat_ali"}
    for e in af.FESTIVALS:
        if e["id"] in moon_ids:
            assert e["moonSighting"] is True, e
        if e["moonSighting"]:
            assert e["id"] in moon_ids and e["note"].startswith(af.EXPECTED), e
        else:
            assert not e["note"].startswith(af.EXPECTED), e
    assert {e["id"] for e in af.FESTIVALS if e["moonSighting"]} == moon_ids


def test_office_notes_where_the_central_list_differs():
    assert by_id("holi")["2027-03-22"]["officeNote"] == "Central government holiday: 23 March"
    assert "2027-03-21" in by_id("holika_dahan")
    assert by_id("sankranti")["2027-01-15"]["officeNote"] == "Central list: 14 January"
    assert by_id("govardhan")["2026-11-10"]["officeNote"] == "Central list: 9 November"


def test_govardhan_is_shukla_pratipada_at_delhi_sunrise():
    # 2026: Amavasya runs to 12:32 IST on 9 Nov (the DoPT RH date), Pratipada holds at sunrise on 10 Nov.
    assert sorted(by_id("govardhan")) == ["2026-11-10", "2027-10-30"]
    for d in by_id("govardhan"):
        t = ac.panchang(date.fromisoformat(d), *DELHI)["tithi"]
        assert (t["paksha"], t["name"]) == ("Shukla", "Pratipada"), d
        day_before = ac.panchang(date.fromisoformat(d) - timedelta(days=1), *DELHI)["tithi"]
        assert day_before["name"] == "Amavasya", d
    assert by_id("govardhan")["2027-10-30"]["officeNote"] is None
    # Mithu wishes it on the panchang day, not a day early
    assert af.observance_for(date(2026, 11, 10), ["hindu"], None).startswith("Govardhan Puja")
    assert "Govardhan" not in af.observance_for(date(2026, 11, 9), ["hindu"], None)


def test_christian_days_follow_easter():
    for y in (2027, 2028):
        e = easter(y)
        assert e.isoformat() in by_id("easter"), y
        assert (e - timedelta(days=2)).isoformat() in by_id("good_friday"), y
        assert (e - timedelta(days=46)).isoformat() in by_id("ash_wednesday"), y


def test_varalakshmi_vratam_is_the_friday_before_shravana_purnima():
    days = sorted(by_id("varalakshmi_vratam"))
    assert days == ["2027-08-13", "2028-08-04"]
    for d in days:
        day = date.fromisoformat(d)
        rakhi = next(date.fromisoformat(x) for x in by_id("raksha_bandhan") if x[:4] == d[:4])
        assert day.weekday() == 4 and 1 <= (rakhi - day).days <= 7, d
        p = ac.panchang(day, *BLR)
        assert p["masa"]["amanta"] == "Shravana" and p["tithi"]["paksha"] == "Shukla", d
        assert set(by_id("varalakshmi_vratam")[d]["states"]) == {"AP", "TG", "KA", "TN"}


def test_days_still_waiting_for_a_state_list_are_not_shown():
    waiting = {e["id"] for e in af.AWAITING_CONFIRMATION}
    assert {"vishu", "magh_bihu"} <= waiting
    assert not waiting & {e["id"] for e in af.FESTIVALS}
    r = up(date(2027, 4, 10), {"religion": "Hindu", "community": "Malayalam", "stateName": "Kerala"})
    assert not any(x["id"] == "vishu" for x in r["festivals"])


def test_south_and_regional_days_of_the_audit_list():
    assert "2027-09-12" in by_id("onam") and "2027-04-14" in by_id("puthandu")
    assert "2027-08-13" in by_id("varalakshmi_vratam")
    for id_, day in (("kojagari_lakshmi_puja", "2026-10-25"), ("valmiki_jayanti", "2026-10-26"),
                     ("kali_puja", "2026-11-08"), ("bandi_chhor_divas", "2026-11-08"), ("govardhan", "2026-11-10"),
                     ("bhai_phonta", "2026-11-11"), ("guru_teg_bahadur", "2026-11-24"), ("hazarat_ali", "2026-12-23"),
                     ("christmas_eve", "2026-12-24"), ("guru_gobind_singh", "2027-01-15"), ("pongal", "2027-01-15"),
                     ("ramzan_start", "2027-02-09"), ("ash_wednesday", "2027-02-10"), ("shivaji_jayanti", "2027-02-19"),
                     ("guru_ravidas", "2027-02-20"), ("jamat_ul_vida", "2027-03-05"), ("jamshedi_navroz", "2027-03-21"),
                     ("dol_jatra", "2027-03-22"), ("easter", "2027-03-28"), ("ambedkar_jayanti", "2027-04-14"),
                     ("poila_boishakh", "2027-04-15"), ("bohag_bihu", "2027-04-15"), ("tagore_jayanti", "2027-05-09"),
                     ("rath_yatra", "2027-07-05"), ("parsi_navroz", "2027-08-15"), ("gandhi_jayanti", "2027-10-02"),
                     ("naraka_chaturdashi", "2027-10-28"), ("govardhan", "2027-10-30"),
                     ("guru_teg_bahadur", "2027-11-24"), ("hazarat_ali", "2027-12-12"),
                     ("christmas_eve", "2027-12-24")):
        assert day in by_id(id_), (id_, day)


def test_greetings_fit_the_faith():
    want = {"eid_ul_fitr": "Eid Mubarak", "bakrid": "Eid Mubarak", "christmas": "Merry Christmas",
            "gurpurab": "Gurpurab di lakh lakh vadhaiyan", "parsi_navroz": "Navroz Mubarak",
            "jamshedi_navroz": "Navroz Mubarak", "diwali": "Shubh Deepavali", "pongal": "Happy Pongal",
            "bijoya_dashami": "Shubho Bijoya", "onam": "Happy Onam"}
    for e in af.FESTIVALS:
        if e["id"] in want:
            assert e["greeting"] == want[e["id"]], e
    # days of mourning or remembrance carry no greeting
    for e in af.FESTIVALS:
        if e["id"] in ("muharram", "good_friday", "guru_teg_bahadur", "mahalaya", "ash_wednesday", "jamat_ul_vida"):
            assert e["greeting"] == "", e


def test_covers_and_supersedes_name_real_days():
    ids = {e["id"] for e in af.FESTIVALS}
    assert set(af.FESTIVAL_COVERS) <= ids
    assert set(af.SAME_DAY_SUPERSEDES) <= ids
    assert {x for v in af.SAME_DAY_SUPERSEDES.values() for x in v} <= ids
    assert af.LEGACY_IDS <= ids
    assert af.LEGACY_IDS == {f["id"] for f in ac.FESTIVALS}


# ------------------------------------------------------------------ who the person is
def test_read_prefs():
    p = af.read_prefs({"religion": "Parsi / Zoroastrian", "community": "Hindi belt / North Indian",
                       "stateName": "National Capital Territory of Delhi"})
    assert p.faiths == {"parsi"} and p.source == "religion" and p.communities == {"hindi"} and p.state == "DL"
    p = af.read_prefs({"celebrates": ["hindu", "Sikh"], "religion": "Muslim", "community": "Malayalam"})
    assert p.faiths == {"hindu", "sikh"} and p.source == "chosen" and p.communities == {"malayali"}
    p = af.read_prefs({"celebrates": ["All of India"]})
    assert p.all_india and not p.faiths and p.source == "chosen" and p.all_majors
    for religion in ("No religion / Spiritual", "Prefer not to say", "", None):
        p = af.read_prefs({"religion": religion})
        assert not p.faiths and p.source == "unknown" and p.all_majors
    assert af.read_prefs({"birthStateName": "Orissa"}).state == "OD"
    assert af.read_prefs({"stateName": "WB"}).state == "WB"
    assert af.read_prefs({"stateName": "IN-TN"}).state == "TN"
    assert af.read_prefs({"stateName": "Jammu & Kashmir"}).state == "JK"
    assert af.read_prefs({"stateName": "Atlantis"}).state is None
    assert af.community_keys("Bhojpuri / Maithili") == {"bhojpuri", "maithili"}
    assert af.community_keys("Other / Mixed") == frozenset()
    assert af.faith_key("hindu") == "hindu" and af.faith_key("ISLAM") == "muslim"


# ------------------------------------------------------------------ Today's lists
@pytest.mark.parametrize("start", [date(2026, 10, 11), date(2026, 10, 20)])
@pytest.mark.parametrize("prefs", [None, HINDU_DELHI, BENGALI], ids=["unknown", "hindu_delhi", "bengali"])
def test_diwali_is_never_crowded_out(start, prefs):
    # The old merged list ended at Ekadashi 5 Nov (11 Oct) and Pradosh 6 Nov (20 Oct): Diwali never showed.
    r = up(start, prefs)
    diwali = ("2026-11-08", "diwali")
    assert diwali in keys(r["festivals"]) or (r["bigDay"] and diwali in keys([r["bigDay"]]))
    assert len(r["festivals"]) <= af.FESTIVAL_LIMIT


def test_every_day_before_diwali_shows_it():
    d = date(2026, 10, 11)
    while d <= date(2026, 11, 8):
        assert ("2026-11-08", "diwali") in keys(up(d, HINDU_DELHI)["festivals"]), d
        d += timedelta(days=1)


def test_a_muslim_family_gets_eid_and_national_days_and_no_vrats():
    r = up(date(2027, 2, 20), MUSLIM)
    assert ("2027-03-10", "eid_ul_fitr") in keys(r["festivals"])
    assert r["bigDay"]["id"] == "eid_ul_fitr" and r["bigDay"]["tentative"] is True
    r = up(date(2027, 1, 20), MUSLIM)
    assert ("2027-01-26", "republic_day") in keys(r["festivals"])
    assert ("2027-02-09", "ramzan_start") in keys(r["festivals"])
    for start in (date(2026, 10, 11), date(2026, 10, 20), date(2026, 11, 7), date(2026, 11, 8), date(2027, 1, 20)):
        r = up(start, MUSLIM)
        assert r["fasts"] == []
        assert not any(x["kind"] in ("ekadashi", "pradosh") for x in r["festivals"])
        assert all(set(x["faiths"]) & {"muslim"} or x["tier"] == "national" for x in r["festivals"]), start
        assert "diwali" not in {x["id"] for x in r["festivals"]}
        assert not (r["bigDay"] and r["bigDay"]["id"] == "diwali")
    # Diwali reaches them only as a greeting, on its day (and the eve)
    assert "diwali" not in {x["id"] for x in up(date(2026, 10, 20), MUSLIM)["alsoInIndia"]}
    also = up(date(2026, 11, 8), MUSLIM)["alsoInIndia"]
    assert [(x["id"], x["greeting"], x["inDays"]) for x in also] == [("diwali", "Shubh Deepavali", 0)]
    assert up(date(2026, 11, 8), MUSLIM)["showing"] == {"faiths": ["muslim"], "community": "", "state": "",
                                                       "source": "religion"}


def test_a_bengali_hindu_family_in_kolkata():
    r = up(date(2026, 10, 11), BENGALI, KOLKATA)
    assert ("2026-10-18", "durga_puja") in keys(r["festivals"])
    assert r["bigDay"]["id"] == "durga_puja" and r["bigDay"]["date"] == "2026-10-18"
    assert r["showing"] == {"faiths": ["hindu"], "community": "Bengali", "state": "West Bengal", "source": "religion"}
    wide = up(date(2026, 10, 11), BENGALI, KOLKATA, limit=20)["festivals"]
    assert {("2026-10-18", "durga_puja"), ("2026-11-08", "kali_puja"), ("2026-10-25", "kojagari_lakshmi_puja"),
            ("2026-11-11", "bhai_phonta"), ("2026-10-21", "bijoya_dashami")} <= keys(wide)
    assert ("2026-11-11", "bhai_dooj") not in keys(wide)          # Bhai Phonta stands for it
    assert ("2026-10-29", "karva_chauth") not in keys(wide)       # a North Indian fast, not theirs
    r = up(date(2026, 10, 25), BENGALI, KOLKATA)
    assert ("2026-11-08", "kali_puja") in keys(r["festivals"])
    # Karva Chauth isn't on their list, so that night's Sankashti is their fast
    assert ("2026-10-29", "sankashti") in {(v["date"], v["kind"]) for v in r["fasts"]}


def test_a_hindu_family_in_delhi():
    r = up(date(2026, 10, 11), HINDU_DELHI)
    assert r["bigDay"]["id"] == "dussehra"                         # not Bengal's Saptami
    r = up(date(2026, 10, 20), HINDU_DELHI)
    assert r["bigDay"]["id"] == "dussehra" and r["bigDay"]["inDays"] == 0
    assert {("2026-10-29", "karva_chauth"), ("2026-11-06", "dhanteras")} <= keys(r["festivals"])
    r = up(date(2026, 10, 27), HINDU_DELHI)
    assert ("2026-10-29", "karva_chauth") in keys(r["festivals"])
    assert not any(v["kind"] == "sankashti" for v in r["fasts"])  # Karva Chauth already is that vrat
    for start in (date(2026, 10, 11), date(2026, 12, 20), date(2027, 6, 1)):
        fasts = up(start, HINDU_DELHI)["fasts"]
        assert 1 <= len(fasts) <= af.FAST_LIMIT
        for v in fasts:
            assert v["type"] == "vrat" and 0 <= v["inDays"] < af.FAST_DAYS and v["faiths"] == ["hindu"]
            assert v["timeKind"] == {"pradosh": "pradosh", "sankashti": "moonrise"}.get(v["kind"])


def test_unknown_faith_gets_national_days_and_every_faiths_big_days():
    for start in (date(2026, 10, 11), date(2026, 11, 20), date(2027, 3, 1), date(2027, 8, 1)):
        r = up(start)
        assert r["festivals"], start
        assert all(x["tier"] in ("national", "major") for x in r["festivals"]), start
        assert r["fasts"] == [] and r["alsoInIndia"] == []
        assert r["showing"]["source"] == "unknown" and r["showing"]["faiths"] == []
    assert ("2026-11-24", "gurpurab") in keys(up(date(2026, 11, 20))["festivals"])
    assert {("2027-03-10", "eid_ul_fitr"), ("2027-03-22", "holi")} <= keys(up(date(2027, 3, 1))["festivals"])


def test_regional_days_follow_the_community_and_respect_the_faith():
    tamil = up(date(2027, 1, 10), {"religion": "Hindu", "community": "Tamil"})
    assert ("2027-01-15", "pongal") in keys(tamil["festivals"])
    assert ("2027-01-15", "sankranti") not in keys(tamil["festivals"])   # Pongal stands for it
    assert ("2027-01-14", "lohri") not in keys(tamil["festivals"])
    assert tamil["bigDay"]["id"] == "pongal"
    north = up(date(2027, 1, 10), {"religion": "Hindu", "community": "Punjabi"})
    assert {("2027-01-14", "lohri"), ("2027-01-15", "sankranti")} <= keys(north["festivals"])
    assert ("2027-01-15", "pongal") not in keys(north["festivals"])
    # Days of a culture are for everyone in it; a faith's puja is not handed to another faith.
    christian_tamil = up(date(2027, 1, 10), {"religion": "Christian", "community": "Tamil"})
    assert ("2027-01-15", "pongal") in keys(christian_tamil["festivals"])
    assert ("2027-01-15", "sankranti") not in keys(christian_tamil["festivals"])
    muslim_bengali = up(date(2026, 10, 25), {"religion": "Muslim", "community": "Bengali", "stateName": "West Bengal"},
                        KOLKATA, limit=20)
    assert not keys(muslim_bengali["festivals"]) & {("2026-11-08", "kali_puja"), ("2026-10-25", "kojagari_lakshmi_puja")}
    assert ("2027-04-15", "poila_boishakh") in keys(
        up(date(2027, 4, 1), {"religion": "Muslim", "community": "Bengali"})["festivals"])
    # A Hindu whose region is unknown gets the all-India Hindu days, not every region's
    plain = up(date(2027, 1, 10), {"religion": "Hindu"}, limit=20)
    assert ("2027-01-15", "sankranti") in keys(plain["festivals"])
    assert not {"pongal", "lohri"} & {x["id"] for x in plain["festivals"]}
    telugu = up(date(2027, 8, 1), {"religion": "Hindu", "community": "Telugu", "stateName": "Telangana"})
    assert telugu["bigDay"]["id"] == "varalakshmi_vratam"


def test_all_of_india_adds_every_faiths_big_days_but_keeps_your_own_big_day():
    r = up(date(2027, 4, 12), {"celebrates": ["hindu", "all"], "community": "Tamil"})
    assert {("2027-04-19", "mahavir_jayanti"), ("2027-05-17", "bakrid")} <= keys(r["festivals"])
    assert r["bigDay"]["id"] == "puthandu"
    assert r["showing"] == {"faiths": ["hindu", "all"], "community": "Tamil", "state": "", "source": "chosen"}
    assert r["fasts"]                                                  # still a Hindu: the vrats stay


def test_other_faiths_big_days_are_a_greeting_today_or_tomorrow():
    hindu = up(date(2026, 12, 24), {"religion": "Hindu"})
    assert [(x["id"], x["greeting"], x["inDays"]) for x in hindu["alsoInIndia"]] == [("christmas", "Merry Christmas", 1)]
    sikh = up(date(2026, 11, 8), {"religion": "Sikh", "community": "Punjabi"})
    assert ("2026-11-08", "bandi_chhor_divas") in keys(sikh["festivals"])
    assert [x["id"] for x in sikh["alsoInIndia"]] == ["diwali"]
    # Good Friday is a major day without a greeting: it is never a "greeting" row
    assert up(date(2027, 3, 26), {"religion": "Hindu"})["alsoInIndia"] == []


def test_rows_carry_what_the_app_reads():
    r = up(date(2026, 10, 25), BENGALI, KOLKATA)
    for x in r["festivals"] + [r["bigDay"]]:
        assert x["kind"] == x["id"] and x["type"] == "festival"     # 1.0.7 reads kind as the festival id
        for f in ("date", "name", "note", "greeting", "tier", "category", "faiths", "tentative", "officeNote",
                  "timeKind", "inDays"):
            assert f in x, f
        assert "source" not in x and "moonSighting" not in x
    rows = sorted(r["festivals"], key=lambda x: x["date"])
    assert rows == r["festivals"]
    assert r["bigDay"]["category"] == "festival" and r["bigDay"]["timeKind"] == "pradosh"
    holi = next(x for x in up(date(2027, 3, 20), {"religion": "Hindu"})["festivals"] if x["id"] == "holi")
    assert holi["officeNote"] == "Central government holiday: 23 March"


def test_no_vrat_function_means_no_fasts():
    assert up(date(2026, 10, 11), HINDU_DELHI, vrats=None)["fasts"] == []
    items = af.legacy_merged(date(2026, 11, 15), 40, *DELHI, vrats=None)
    assert items and all(i["type"] == "festival" for i in items)


# ------------------------------------------------------------------ the old list (1.0.7 apps)
# 2027-01-10, 2027-03-08, 2027-05-10 and 2028-01-10 reach the days whose wording 1.0.7 keeps (LEGACY_TEXT).
@pytest.mark.parametrize("start", [date(2026, 11, 15), date(2026, 12, 1), date(2027, 1, 10), date(2027, 3, 8),
                                   date(2027, 5, 10), date(2027, 6, 1), date(2027, 8, 1), date(2027, 9, 1),
                                   date(2028, 1, 10)])
@pytest.mark.parametrize("place", [DELHI, BLR], ids=["delhi", "bengaluru"])
def test_legacy_merged_equals_the_old_list(start, place):
    assert af.legacy_merged(start, 40, *place, limit=8, vrats=ac.vrats_between) == ac.upcoming(start, 40, *place, limit=8)


def test_the_sample_days_reach_the_reworded_days():
    seen = set()
    for start in (date(2027, 1, 10), date(2027, 3, 8), date(2027, 5, 10), date(2028, 1, 10)):
        seen |= {(i["date"], i["id"]) for i in ac.upcoming(start, 40, *DELHI, limit=8) if i["type"] == "festival"}
    assert set(af.LEGACY_TEXT) <= seen


def test_the_old_list_changes_only_where_it_was_wrong():
    """1.0.7 apps get the old list word for word; the only change is Durga Puja 2026 moving from 17 Oct (Shashthi)
    to 18 Oct (Saptami)."""
    first, last = date(2026, 10, 11), max(date.fromisoformat(e["date"]) for e in af.FESTIVALS)
    days = (last - first).days + 1
    new = af.legacy_merged(first, days, *DELHI, limit=10_000, vrats=None)
    old = [{**f, "kind": f["id"], "type": "festival", "inDays": (date.fromisoformat(f["date"]) - first).days}
           for f in ac.FESTIVALS if first <= date.fromisoformat(f["date"]) <= last]
    moved = ("2026-10-18", "durga_puja")
    assert [i for i in new if (i["date"], i["id"]) != moved] == [i for i in old if (i["date"], i["id"]) != ("2026-10-17", "durga_puja")]
    old_puja = next(i for i in old if (i["date"], i["id"]) == ("2026-10-17", "durga_puja"))
    new_puja = next(i for i in new if (i["date"], i["id"]) == moved)
    assert new_puja == {**old_puja, "date": "2026-10-18", "inDays": old_puja["inDays"] + 1}
    # The table words these days anew (Pongal its own Tamil day, "Expected" moon notes, Holi's office date in
    # officeNote); LEGACY_TEXT keeps the old words for exactly these and nothing else.
    table = {(e["date"], e["id"]): e for e in af.FESTIVALS if e["id"] in af.LEGACY_IDS}
    old_by_key = {(f["date"], f["id"]): f for f in ac.FESTIVALS}
    reworded = {k for k in set(table) & set(old_by_key)
                if (table[k]["name"], table[k]["note"]) != (old_by_key[k]["name"], old_by_key[k]["note"])}
    assert reworded == set(af.LEGACY_TEXT)
    items = af.legacy_merged(date(2026, 10, 11), 40, *DELHI, limit=30, vrats=ac.vrats_between)
    assert ("2026-10-18", "durga_puja") in {(i["date"], i.get("id")) for i in items}
    assert all(set(i) == {"date", "id", "name", "note", "kind", "type", "inDays"}
               for i in items if i["type"] == "festival")
    # Tamil 1.0.7 users still see Pongal: in the old Sankranti name
    jan = af.legacy_merged(date(2027, 1, 10), 40, *DELHI, vrats=None)
    assert next(i for i in jan if i["id"] == "sankranti")["name"] == "Makar Sankranti · Pongal"


# ------------------------------------------------------------------ TODAY_OBSERVANCE
def test_observance_line():
    diwali = date(2026, 11, 8)
    line = af.observance_for(diwali, ["hindu"], "Hindi belt / North Indian")
    assert line.startswith("Diwali: Lakshmi puja in the evening.") and "Greeting: Shubh Deepavali." in line
    assert "Kali Puja" not in line
    assert "Kali Puja" in af.observance_for(diwali, "Hindu", "Bengali")
    assert af.observance_for(diwali, ["muslim"], None) == "none"
    assert af.observance_for(diwali, [], None, religion="Sikh").startswith("Bandi Chhor Divas")
    eid = af.observance_for(date(2027, 3, 10), None, None, religion="Muslim")
    assert eid.startswith("Eid ul-Fitr: Expected, depends on the moon") and "Eid Mubarak" in eid
    assert af.observance_for(date(2027, 1, 26), ["christian"], None).startswith("Republic Day")
    assert af.observance_for(date(2026, 10, 12), ["hindu"], None) == "none"


def test_observance_line_never_wishes_an_unknown_faith_another_faiths_day():
    """TODAY_OBSERVANCE is the reader's own day: Mithu wishes them for it, so "Prefer not to say" and
    "No religion / Spiritual" never get Merry Christmas, Eid Mubarak or Shubh Deepavali."""
    diwali, christmas, eid = date(2026, 11, 8), date(2026, 12, 25), date(2027, 3, 10)
    for religion in (None, "", "Prefer not to say", "No religion / Spiritual"):
        for day in (diwali, christmas, eid):
            assert af.observance_for(day, None, None, religion=religion) == "none", (religion, day)
            assert af.observance_for(day, [], "Hindi belt / North Indian", religion=religion, state="DL") == "none"
    assert af.observance_for(christmas, [], None, religion="No religion / Spiritual") == "none"
    assert af.observance_for(diwali, None, "Bengali", state="WB") == "none"              # not Kali Puja either
    # national days stay for everyone
    rd = af.observance_for(date(2027, 1, 26), None, None)
    assert rd.startswith("Republic Day")
    assert af.observance_for(date(2027, 1, 26), [], None, religion="Prefer not to say") == rd
    # their own region's days that belong to no faith in particular stay too
    assert af.observance_for(date(2027, 4, 15), None, "Bengali").startswith("Poila Boishakh")
    # the upcoming lists still show every faith's big days for an unknown faith
    assert any(f["id"] == "christmas" for f in af.upcoming(christmas, 1, *DELHI, {}, vrats=None)["festivals"])
    # an explicit All of India choice keeps every faith's big days
    assert af.observance_for(christmas, ["all"], None).startswith("Christmas")
    assert "Eid Mubarak" in af.observance_for(eid, ["all"], None)
    assert af.observance_for(diwali, ["all"], None).startswith("Diwali")
    # their own faith still gets theirs
    assert af.observance_for(christmas, [], None, religion="Christian").startswith("Christmas")
    assert af.observance_for(christmas, ["hindu"], None) == "none"


def test_observance_line_takes_todays_vrat_only_for_hindu_days():
    day = date(2026, 10, 22)                                       # Ekadashi in New Delhi
    vr = ac.vrats_between(day, 1, *DELHI)
    assert [v["kind"] for v in vr] == ["ekadashi"]
    assert af.observance_for(day, ["hindu"], None, fasts=vr) == "Ekadashi vrat: A day of fasting and Vishnu puja."
    assert af.observance_for(day, ["jain"], None, fasts=vr) == "none"
    kc = date(2026, 10, 29)                                        # Karva Chauth already is that Sankashti
    vr = ac.vrats_between(kc, 1, *DELHI)
    assert [v["kind"] for v in vr] == ["sankashti"]
    line = af.observance_for(kc, ["hindu"], "Punjabi", fasts=vr)
    assert line.startswith("Karva Chauth") and "Sankashti" not in line
    assert "Sankashti" in af.observance_for(kc, ["hindu"], "Tamil", fasts=vr)
    assert len(af.observance_for(date(2026, 11, 8), ["hindu", "jain", "sikh"], "Bengali", state="WB")) <= 400


# ------------------------------------------------------------------ own_days (what may be wished)
def _ids(rows):
    return [r["id"] for r in rows]


def test_own_days_never_hands_an_unknown_faith_another_faiths_big_day():
    """The lists show an unknown faith every faith's big days; own_days keeps only national days and their region's
    days that belong to no faith, so notes never wish "Prefer not to say" Eid Mubarak or Merry Christmas."""
    for prefs in ({}, {"religion": "Prefer not to say"}, {"religion": "No religion / Spiritual"},
                  {"celebrates": [], "religion": ""}):
        for start in (date(2026, 11, 1), date(2026, 12, 20), date(2027, 3, 5), date(2027, 8, 10)):
            rows = up(start, prefs)["festivals"]
            assert any(r["tier"] == "major" and r["faiths"] for r in rows), (prefs, start)   # the lists show them
            own = af.own_days(rows, prefs)
            assert all(r["tier"] == "national" or not r["faiths"] for r in own), (prefs, start, _ids(own))
    # Independence Day 2027 stays for everyone; Milad-un-Nabi and Parsi New Year the same day do not.
    own = af.own_days(up(date(2027, 8, 14), {})["festivals"], {})
    assert "independence_day" in _ids(own) and not {"milad_un_nabi", "parsi_navroz"} & set(_ids(own))
    # A Bengali with no faith given keeps Poila Boishakh (a day of no faith in particular), never Kali Puja.
    bengali = {"community": "Bengali"}
    assert "poila_boishakh" in _ids(af.own_days(up(date(2027, 4, 10), bengali)["festivals"], bengali))
    assert "kali_puja" not in _ids(af.own_days([e for e in af.FESTIVALS if e["id"] == "kali_puja"], bengali))


def test_own_days_keeps_a_faiths_own_days_and_drops_the_rest():
    hindu_rows = up(date(2026, 11, 1), HINDU_DELHI)["festivals"]
    assert af.own_days(hindu_rows, HINDU_DELHI) == hindu_rows          # their lists are their own days
    every = up(date(2026, 12, 20), {})["festivals"] + up(date(2027, 3, 5), {})["festivals"]
    assert "christmas" in _ids(af.own_days(every, {"religion": "Christian"}))
    assert "christmas" not in _ids(af.own_days(every, MUSLIM))
    assert "eid_ul_fitr" in _ids(af.own_days(every, MUSLIM))
    assert "eid_ul_fitr" not in _ids(af.own_days(every, {"celebrates": ["hindu"]}))
    # An explicit All of India choice keeps every faith's big days.
    assert {"christmas", "eid_ul_fitr"} <= set(_ids(af.own_days(every, {"celebrates": ["all"]})))
    # Read through the same Prefs object too, and in the order given.
    me = af.read_prefs({"celebrates": ["all"]})
    assert _ids(af.own_days(every, me)) == [i for i in _ids(every) if i in set(_ids(af.own_days(every, me)))]


def test_own_days_reads_untagged_rows_by_the_table_and_vrats_by_faith():
    diwali = {"date": "2026-11-08", "id": "diwali", "name": "Diwali"}
    assert _ids(af.own_days([diwali], HINDU_DELHI)) == ["diwali"]
    assert af.own_days([diwali], MUSLIM) == []
    assert af.own_days([{"date": "2026-11-09", "id": "diwali", "name": "Diwali"}], HINDU_DELHI) == []   # wrong day
    assert af.own_days([{"date": "2026-11-08", "id": "made_up", "name": "X"}], {"celebrates": ["all"]}) == []
    assert af.own_days([None, "x", 3], HINDU_DELHI) == [] and af.own_days(None, HINDU_DELHI) == []
    vrat = {"date": "2026-10-22", "id": "ekadashi", "kind": "ekadashi", "type": "vrat", "name": "Ekadashi vrat"}
    assert af.own_days([vrat], HINDU_DELHI) == [vrat] and af.own_days([vrat], {"celebrates": ["jain"]}) == []
    out = af.own_days([diwali], HINDU_DELHI)
    out[0]["name"] = "changed"
    assert diwali["name"] == "Diwali"   # copies, never the caller's rows


@pytest.mark.parametrize("prefs", [HINDU_DELHI, BENGALI, MUSLIM, {"religion": "Prefer not to say"},
                                   {"celebrates": ["all"]}, {"religion": "Sikh", "community": "Punjabi"}])
def test_own_days_agree_with_the_observance_line(prefs):
    """Mithu's note (own_days over the Today rows) and the horoscope's TODAY_OBSERVANCE wish the same days."""
    day = date(2026, 10, 11)
    while day < date(2027, 4, 1):
        own = af.own_days(af.upcoming(day, 1, *DELHI, prefs, vrats=None)["festivals"], prefs)
        line = af.observance_for(day, prefs.get("celebrates"), prefs.get("community"), religion=prefs.get("religion"),
                                 state=prefs.get("stateName"))
        if own:
            parts = line.split(" | ")
            assert [p.split(":")[0].split(". Greeting")[0] for p in parts] == [r["name"] for r in own], (day, line)
        else:
            assert line == "none", (day, line)
        day += timedelta(days=1)
