"""Mithu writes first (engagement N5): templated notes by priority, caps and idempotent storage."""
from datetime import date, timedelta

import pytest
from google.api_core import exceptions as gexc

import astro_core as ac
import astro_notes as an
import main

DELHI = {"name": "New Delhi", "lat": 28.6139, "lon": 77.209, "tz": "Asia/Kolkata"}
JAIPUR = {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "tz": "Asia/Kolkata"}
# A single Hindu reader, Moon in Kanya (born 15 Sep 1996 at noon in India), birthday far from the test days.
ME = {"firstName": "Asha", "gender": "female", "religion": "Hindu", "birthDate": "1996-09-15", "lifeStage": "single",
      "profileFor": "self"}
HINDU = {"celebrates": ["hindu"], "astroMode": "full"}
DIWALI = {"date": "2026-11-08", "id": "diwali", "name": "Diwali", "greeting": "Shubh Deepavali", "timeKind": "pradosh"}
KARVA = {"date": "2026-10-29", "id": "karva_chauth", "name": "Karva Chauth", "greeting": "Happy Karva Chauth",
         "timeKind": "moonrise"}
EID = {"date": "2027-03-10", "id": "eid_ul_fitr", "name": "Eid ul-Fitr", "greeting": "Eid Mubarak", "moonSighting": True}
DEV_UTHANI = {"date": "2026-11-20", "id": "dev_uthani", "name": "Dev Uthani Ekadashi"}


# ------------------------------------------------------------------ a small Firestore stand-in
class Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self.exists, self._d = ref, ref.id, data is not None, data

    def to_dict(self):
        return None if self._d is None else dict(self._d)


class Ref:
    def __init__(self, db, path):
        self.db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def collection(self, name):
        return Col(self.db, f"{self.path}/{name}")

    def get(self):
        return Snap(self, self.db.docs.get(self.path))


class Col:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def document(self, id_):
        return Ref(self.db, f"{self.path}/{id_}")


class Batch:
    def __init__(self, db):
        self.db, self.ops = db, []

    def create(self, ref, data):
        self.ops.append(("create", ref, data, False))

    def set(self, ref, data, merge=False):
        self.ops.append(("set", ref, data, merge))

    def commit(self):
        if self.db.fail:
            raise RuntimeError("firestore down")
        if any(op == "create" and ref.path in self.db.docs for op, ref, _, _ in self.ops):
            raise gexc.AlreadyExists("Document already exists")
        for _, ref, data, merge in self.ops:
            self.db.docs[ref.path] = {**(self.db.docs.get(ref.path) or {}), **data} if merge else dict(data)
        self.db.commits += 1


class Db:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.commits = 0
        self.fail = False
        self.hide_once = set()   # another request writes these right after this one reads

    def collection(self, name):
        return Col(self, name)

    def get_all(self, refs):
        out = []
        for r in refs:
            if r.path in self.hide_once:
                self.hide_once.discard(r.path)
                out.append(Snap(r, None))
            else:
                out.append(Snap(r, self.docs.get(r.path)))
        return list(reversed(out))   # any order, like Firestore

    def batch(self):
        return Batch(self)

    def messages(self, uid="u1"):
        chat = f"parrotChats/{main._chat_doc_id(uid, f'mobile_{uid}')}/messages/"
        return [v for k, v in self.docs.items() if k.startswith(chat)]


def _note(day, festivals=None, members=None, me=ME, prefs=HINDU, place=DELHI, **kw):
    return an.compose(me, place, day, members, prefs, festivals, **kw)


# ------------------------------------------------------------------ priority
def test_priority_p1_over_p2_over_p3():
    day = date(2026, 10, 29)   # Jupiter moves into Simha the day after tomorrow
    kid = [{"id": "m1", "name": "Aarav", "relation": "Son", "birthDate": "2019-10-30"}]
    fest = [{"date": "2026-10-29", "id": "x_fest", "name": "Test Festival", "greeting": "Happy Test"}]
    assert _note(day, fest, kid)["topic"] == "festival"
    assert _note(day, None, kid)["topic"] == "birthday"
    sky = _note(day)
    assert sky["topic"] == "sky"
    assert sky["text"] == ("On 31 October Jupiter moves into Simha, your 12th house from the Moon: a warm year for "
                           "rest, travel and letting go begins.")
    assert sky["actions"] == []


def test_p3_over_p5(monkeypatch):
    monkeypatch.setattr(an, "DEV_UTHANI_DAYS", frozenset({"2026-10-30"}))
    assert _note(date(2026, 10, 30))["topic"] == "sky"
    assert _note(date(2026, 10, 30), me={**ME, "birthDate": ""})["topic"] == "wedding_season"


def test_p5_on_dev_uthani_for_singles_and_families():
    day = date(2026, 11, 20)
    note = _note(day, [DEV_UTHANI])
    assert note == {"topic": "wedding_season", "title": "The wedding season opens",
                    "text": "The wedding season opens today. Want to see good wedding dates?",
                    "actions": [{"type": "open_good_dates", "label": "See wedding dates", "arg": "wedding"}]}
    managed = {**ME, "profileFor": "son", "firstName": "Rohan", "lifeStage": ""}
    assert _note(day, [DEV_UTHANI], me=managed)["topic"] == "wedding_season"
    assert _note(day, [DEV_UTHANI], prefs={"astroMode": "fun", "celebrates": ["hindu"]})["topic"] == "wedding_season"
    # Married readers get the festival line instead; other faiths get nothing from the wedding season.
    married = _note(day, [DEV_UTHANI], me={**ME, "lifeStage": "married"})
    assert married["topic"] == "festival" and married["title"] == "Dev Uthani Ekadashi is today"
    assert _note(day, [], prefs={"celebrates": ["muslim"], "astroMode": "fun"}) is None
    # A birthday or another festival today still comes first.
    kid = [{"id": "m1", "name": "Aarav", "relation": "son", "birthDate": "2019-11-21"}]
    assert _note(day, [DEV_UTHANI], kid)["topic"] == "birthday"
    other = {"date": "2026-11-20", "id": "x", "name": "Another Day", "greeting": "Happy day"}
    assert _note(day, [DEV_UTHANI, other])["title"] == "Another Day is today"


def test_off_gets_no_sky_and_no_wedding_season():
    off = {"celebrates": ["hindu"], "astroMode": "off"}
    assert _note(date(2026, 10, 30), prefs=off) is None
    assert _note(date(2026, 11, 20), [DEV_UTHANI], prefs=off) is None   # no greeting on that entry either
    fun = {"celebrates": ["hindu"], "astroMode": "fun"}
    assert _note(date(2026, 10, 30), prefs=fun) is None                 # P3 is for 'full' only


def test_off_gets_only_the_greeting():
    note = _note(date(2026, 11, 8), [DIWALI], prefs={"celebrates": ["hindu"], "astroMode": "off"})
    assert note["text"] == "Diwali is today. Shubh Deepavali to you and your family!"
    assert [a["type"] for a in note["actions"]] == ["share_greeting"]
    assert "Pradosh" not in note["text"]


def test_nothing_qualifies():
    assert _note(date(2026, 10, 15)) is None
    assert _note(date(2026, 10, 15), [DIWALI]) is None                    # Diwali is weeks away
    assert _note(date(2026, 10, 30), me={**ME, "birthDate": ""}) is None  # no chart, no sky note


# ------------------------------------------------------------------ P1 festivals
def test_festival_tomorrow_with_the_city_moonrise():
    note = _note(date(2026, 10, 28), [KARVA], place=JAIPUR)
    mr = ac.panchang(date(2026, 10, 29), JAIPUR["lat"], JAIPUR["lon"], JAIPUR["tz"])["moonrise"]
    assert note["title"] == "Karva Chauth is tomorrow"
    assert note["text"] == f"Karva Chauth is tomorrow. Moonrise in Jaipur: {mr}. I'll keep the time on Today for you."
    assert [a["type"] for a in note["actions"]] == ["open_panchang", "share_greeting"]
    assert note["actions"][1]["arg"] == "Happy Karva Chauth! Wishing you and your family joy and blessings."
    # Without the table's greeting there is nothing to send: no made-up "Happy …" line, only the times.
    bare = _note(date(2026, 10, 28), [{**KARVA, "greeting": ""}], place=JAIPUR)
    assert bare["text"] == note["text"]
    assert [a["type"] for a in bare["actions"]] == ["open_panchang"]


def test_festival_today_with_greeting_and_pradosh_time():
    note = _note(date(2026, 11, 8), [DIWALI])
    p = ac.panchang(date(2026, 11, 8), DELHI["lat"], DELHI["lon"], DELHI["tz"])
    assert note["text"].startswith(f"Shubh Deepavali! Diwali is today. Pradosh time in New Delhi: {p['sunset']}–")
    assert note["actions"][-1] == {"type": "share_greeting", "label": "Send a greeting",
                                   "arg": "Shubh Deepavali! Wishing you and your family joy and blessings."}


def _fest_row(day, id_, name, greeting, faiths, *, tentative=False, category="festival", tier="faith", in_days=1):
    """A row as astro_festivals.upcoming()['festivals'] gives it (W1-S-FEST _row)."""
    return {"date": day, "id": id_, "kind": id_, "type": "festival", "name": name, "note": "", "greeting": greeting,
            "tier": tier, "category": category, "faiths": list(faiths), "communities": [], "states": [],
            "tentative": tentative, "timeKind": None, "officeNote": None, "inDays": in_days}


# Days of mourning or remembrance: astro_festivals leaves the greeting empty on purpose.
MUHARRAM = _fest_row("2027-06-16", "muharram", "Muharram (Ashura)", "", ["muslim"], tentative=True,
                     category="observance")
GOOD_FRIDAY = _fest_row("2027-03-26", "good_friday", "Good Friday", "", ["christian"], category="observance",
                        tier="major")
TEG_BAHADUR = _fest_row("2027-11-24", "guru_teg_bahadur", "Guru Teg Bahadur's Martyrdom Day", "", ["sikh"],
                        category="observance")


@pytest.mark.parametrize("row, faith, eve, text", [
    (MUHARRAM, "muslim", date(2027, 6, 15), "Muharram (Ashura) is tomorrow. Expected, depends on the moon."),
    (GOOD_FRIDAY, "christian", date(2027, 3, 25), "Good Friday is tomorrow."),
    (TEG_BAHADUR, "sikh", date(2027, 11, 23), "Guru Teg Bahadur's Martyrdom Day is tomorrow."),
])
def test_solemn_days_get_no_greeting(row, faith, eve, text):
    me = {**ME, "religion": faith.title()}
    for mode in ("full", "fun"):
        prefs = {"celebrates": [faith], "astroMode": mode}
        note = _note(eve, [row], me=me, prefs=prefs)
        assert note["topic"] == "festival" and note["text"] == text
        assert note["actions"] == []
        today = _note(eve + timedelta(days=1), [{**row, "inDays": 0}], me=me, prefs=prefs)
        assert today["title"] == row["name"] + " is today" and today["actions"] == []
        for n in (note, today):
            assert "Happy" not in n["title"] + n["text"] and "Wishing" not in n["text"]
    # astroMode 'off' gets greetings only: nothing for these days.
    assert _note(eve, [row], me=me, prefs={"celebrates": [faith], "astroMode": "off"}) is None


def test_days_that_begin_a_period_read_by_id():
    navratri = _fest_row("2026-10-11", "navratri", "Sharad Navratri begins", "Happy Navratri", ["hindu"])
    eve = _note(date(2026, 10, 10), [navratri])
    assert eve["title"] == "Sharad Navratri begins tomorrow" and eve["text"] == "Sharad Navratri begins tomorrow."
    assert eve["actions"] == [{"type": "share_greeting", "label": "Send a greeting",
                               "arg": "Happy Navratri! Wishing you and your family joy and blessings."}]
    day = _note(date(2026, 10, 11), [{**navratri, "inDays": 0}])
    assert day["title"] == "Sharad Navratri begins today"
    assert day["text"] == "Happy Navratri! Sharad Navratri begins today."

    ramzan = _fest_row("2027-02-09", "ramzan_start", "Ramzan begins", "Ramzan Mubarak", ["muslim"], tentative=True,
                       category="observance")
    muslim = {**ME, "religion": "Muslim"}
    eve = _note(date(2027, 2, 8), [ramzan], me=muslim, prefs={"celebrates": ["muslim"]})
    assert eve["title"] == "Ramzan begins tomorrow"
    assert eve["text"] == "Ramzan begins tomorrow. Expected, depends on the moon."
    assert eve["actions"] == [{"type": "share_greeting", "label": "Send a greeting",
                               "arg": "Ramzan Mubarak! Wishing you and your family joy and blessings."}]
    off = _note(date(2027, 2, 8), [ramzan], me=muslim, prefs={"celebrates": ["muslim"], "astroMode": "off"})
    assert off["text"] == "Ramzan begins tomorrow. Expected, depends on the moon. Ramzan Mubarak to you and your family!"
    assert off["actions"] == eve["actions"]
    # The share line never repeats the display name ("a happy Sharad Navratri begins", "a happy Ramzan begins").
    for n in (eve, off, day):
        for a in n["actions"]:
            assert "begins" not in a["arg"]


def test_moon_sighting_is_marked_expected():
    note = _note(date(2027, 3, 9), [EID], me={**ME, "religion": "Muslim"}, prefs={"celebrates": ["muslim"]})
    assert note["text"] == "Eid ul-Fitr is tomorrow. Expected, depends on the moon."
    begins = _note(date(2026, 10, 10), [{"date": "2026-10-11", "id": "navratri", "name": "Sharad Navratri begins"}])
    assert begins["title"] == "Sharad Navratri begins tomorrow"


# A row as astro_festivals.upcoming()['festivals'] (and the app's UpcomingDto) carries it: `tentative`, no moonSighting.
EID_ROW = {"date": "2027-03-10", "id": "eid_ul_fitr", "kind": "eid_ul_fitr", "type": "festival", "name": "Eid ul-Fitr",
           "note": "", "greeting": "Eid Mubarak", "tier": "major", "category": "festival", "faiths": ["muslim"],
           "communities": [], "states": [], "tentative": True, "timeKind": "", "officeNote": "", "inDays": 1}


def test_upcoming_rows_marked_tentative_carry_the_moon_label():
    muslim, prefs = {**ME, "religion": "Muslim"}, {"celebrates": ["muslim"]}
    off = {**prefs, "astroMode": "off"}
    assert _note(date(2027, 3, 9), [EID_ROW], me=muslim, prefs=prefs)["text"] == \
        "Eid ul-Fitr is tomorrow. Expected, depends on the moon."
    assert _note(date(2027, 3, 10), [EID_ROW], me=muslim, prefs=prefs)["text"] == \
        "Eid Mubarak! Eid ul-Fitr is today. Expected, depends on the moon."
    # astroMode 'off' gets the greeting only, and the label too (from either flag).
    for row in (EID_ROW, EID):
        assert _note(date(2027, 3, 9), [row], me=muslim, prefs=off)["text"] == \
            "Eid ul-Fitr is tomorrow. Expected, depends on the moon. Eid Mubarak to you and your family!"
    # A fixed-date day has no label.
    sure = _note(date(2027, 3, 9), [{**EID_ROW, "tentative": False}], me=muslim, prefs=prefs)
    assert sure["text"] == "Eid ul-Fitr is tomorrow."
    assert _note(date(2027, 3, 9), [{**EID_ROW, "tentative": False}], me=muslim, prefs=off)["text"] == \
        "Eid ul-Fitr is tomorrow. Eid Mubarak to you and your family!"


def test_today_beats_tomorrow_and_fasts_are_skipped():
    fests = [{"date": "2026-11-09", "id": "govardhan", "name": "Govardhan Puja"}, DIWALI,
             {"date": "2026-11-08", "id": "amavasya", "name": "Amavasya", "type": "vrat"}]
    assert _note(date(2026, 11, 8), fests)["title"] == "Diwali is today"


# ------------------------------------------------------------------ P2 birthdays
def test_member_birthday_tomorrow():
    kid = [{"id": "m1", "name": "Aarav Sharma", "relation": "Son", "birthDate": "2019-10-16"}]
    note = _note(date(2026, 10, 15), None, kid)
    assert note["title"] == "Aarav's birthday is tomorrow"
    assert note["text"] == "Aarav's birthday is tomorrow. Want a blessing line to send in the morning?"
    assert note["actions"] == [
        {"type": "share_blessing", "label": "Get a blessing line",
         "arg": "Happy birthday, Aarav! May you grow happy, healthy and wise. Lots of love and blessings."},
        {"type": "open_family_member", "label": "See Aarav's day", "arg": "m1"}]


def test_own_and_managed_birthdays():
    mine = _note(date(2026, 9, 15))
    assert mine["topic"] == "birthday" and mine["title"] == "Happy birthday!" and mine["actions"] == []
    managed = {**ME, "profileFor": "daughter", "firstName": "Ananya"}
    theirs = _note(date(2026, 9, 14), me=managed)
    assert theirs["text"] == "Ananya's birthday is tomorrow. Want a blessing line to send in the morning?"
    assert [a["type"] for a in theirs["actions"]] == ["share_blessing"]
    # A managed rishta profile is an adult: the adult blessing, never "May you grow …".
    assert theirs["actions"][0]["arg"] == ("Happy birthday, Ananya! Wishing you good health, happiness and many "
                                           "blessings this year.")
    # Today's birthday beats tomorrow's; the reader's own self row in the family list isn't counted twice.
    rows = [{"id": "s", "name": "Asha", "birthDate": "1996-09-16", "me": True},
            {"id": "m2", "name": "Papa", "relation": "Father", "birthDate": "1960-09-16"}]
    assert _note(date(2026, 9, 15), None, rows)["title"] == "Happy birthday!"
    assert _note(date(2026, 9, 15), None, rows, me={**ME, "birthDate": ""})["title"] == "Papa's birthday is tomorrow"


def test_leap_day_birthday_counts_on_28_february():
    leap = [{"id": "m1", "name": "Leela", "relation": "wife", "birthDate": "1992-02-29"}]
    assert _note(date(2027, 2, 28), None, leap)["title"] == "It's Leela's birthday today"
    assert _note(date(2028, 2, 28), None, leap)["title"] == "Leela's birthday is tomorrow"


def test_members_can_be_request_models():
    class Person:
        id, name, relation, birthDate, me = "m9", "Dadi", "grandmother", "1950-10-16", False
    note = _note(date(2026, 10, 16), None, [Person()])
    assert note["title"] == "It's Dadi's birthday today" and note["actions"][1]["arg"] == "m9"


# ------------------------------------------------------------------ P3 sky
def test_sky_note_for_a_managed_profile_names_the_person():
    managed = {**ME, "profileFor": "daughter", "firstName": "Ananya"}
    note = _note(date(2026, 10, 29), me=managed, natal={"rashi": 0})
    assert note["text"] == ("On 31 October Jupiter moves into Simha, Ananya's 5th house from the Moon: a warm year "
                            "for love, children and creativity begins.")
    tomorrow = _note(date(2026, 10, 30), natal={"rashi": 0})
    assert tomorrow["text"].startswith("Tomorrow Jupiter moves into Simha, your 5th house")
    assert _note(date(2026, 10, 31), natal={"rashi": 0})["text"].startswith("Today Jupiter moves into Simha")
    assert "dosha" not in note["text"].lower() and "sade" not in note["text"].lower()


def test_sky_note_for_a_retrograde_return_names_no_year():
    # Jupiter goes back from Simha into Karka on 25 Jan 2027 (Karka is the 11th from a Kanya Moon).
    note = _note(date(2027, 1, 23))
    assert note["title"] == "Jupiter moves back into Karka"
    assert note["text"] == ("On 25 January Jupiter moves back into Karka, your 11th house from the Moon: a gentle "
                            "second look at gains, friends and wishes.")
    managed = _note(date(2027, 1, 24), me={**ME, "profileFor": "daughter", "firstName": "Ananya"})
    assert managed["text"] == ("Tomorrow Jupiter moves back into Karka, Ananya's 11th house from the Moon: a gentle "
                               "second look at gains, friends and wishes.")
    # The forward re-entry on 26 Jun 2027 is no new year either.
    again = _note(date(2027, 6, 26))
    assert again["title"] == "Jupiter moves into Simha again"
    assert again["text"] == ("Today Jupiter moves into Simha again, your 12th house from the Moon: warmth returns to "
                             "rest, travel and letting go.")
    for n in (note, managed, again):
        assert "year" not in n["text"] and "begins" not in n["text"]


def test_astro_mode_defaults():
    assert an.astro_mode_of({}, {"religion": "Muslim"}) == "fun"
    assert an.astro_mode_of({}, {"religion": "No religion / Spiritual"}) == "fun"
    assert an.astro_mode_of({"celebrates": ["christian"]}, {"religion": "Hindu"}) == "fun"
    assert an.astro_mode_of({}, {"religion": "Hindu"}) == "full"
    assert an.astro_mode_of({}, {"religion": "Prefer not to say"}) == "full"
    assert an.astro_mode_of({}, {}) == "full"
    assert an.astro_mode_of({"astroMode": "off"}, {"religion": "Hindu"}) == "off"
    assert an.astro_mode_of({}, {"astroMode": "full", "religion": "Muslim"}) == "full"
    assert an.faith_key("Parsi / Zoroastrian") == "parsi"


# ------------------------------------------------------------------ storage and caps
def test_one_note_per_day_is_stored_once():
    db = Db()
    day = date(2026, 11, 8)
    first = an.note_for("u1", ME, DELHI, day, [], HINDU, festivals=[DIWALI], db=db)
    assert first["id"] == "u1__2026-11-08" and first["day"] == "2026-11-08" and first["topic"] == "festival"
    again = an.note_for("u1", ME, DELHI, day, [], HINDU, festivals=[DIWALI], db=db)
    assert again == first and db.commits == 1
    doc = db.docs["mithuNotes/u1__2026-11-08"]
    assert doc["text"] == first["text"] and doc["actions"] == first["actions"] and doc["topic"] == "festival"
    assert doc["title"] == "Diwali is today" and doc["createdAtIso"].endswith("Z")
    msgs = db.messages()
    assert len(msgs) == 1
    m = msgs[0]
    assert m["role"] == "assistant" and m["kind"] == "mithu_note" and m["noteId"] == "u1__2026-11-08"
    assert m["topic"] == "festival" and m["text"] == first["text"] and isinstance(m["createdAtMs"], int)
    chat = db.docs[f"parrotChats/{main._chat_doc_id('u1', 'mobile_u1')}"]
    assert chat["threadId"] == "mobile_u1" and chat["uid"] == "u1"
    assert db.docs["mithuNotesState/u1"]["days"] == ["2026-11-08"]


def test_a_parallel_request_that_stored_first_wins():
    db = Db()
    day = date(2026, 11, 8)
    winner = an.note_for("u1", ME, DELHI, day, [], HINDU, festivals=[DIWALI], db=db)
    db.hide_once.add("mithuNotes/u1__2026-11-08")   # the late request read before the winner wrote
    late = an.note_for("u1", ME, DELHI, day, [], HINDU, festivals=[DIWALI], db=db)
    assert late == winner and db.commits == 1 and len(db.messages()) == 1


def test_at_most_four_in_seven_days():
    db = Db()
    fest = [{"date": (date(2026, 12, 1) + timedelta(days=i)).isoformat(), "id": f"f{i}", "name": f"Day {i}",
             "greeting": "Hello"} for i in range(12)]
    got = [an.note_for("u1", ME, DELHI, date(2026, 12, 1) + timedelta(days=i), [], HINDU, festivals=fest, db=db)
           for i in range(12)]
    # 1-4 Dec, then nothing while four sit in the last 7 days; the window rolls from 8 Dec.
    assert [g is not None for g in got] == [True, True, True, True, False, False, False, True, True, True, True, False]
    days = sorted(k.split("__")[1] for k in db.docs if k.startswith("mithuNotes/"))
    for i in range(12):
        first, last = date(2026, 12, 1) + timedelta(days=i - 6), date(2026, 12, 1) + timedelta(days=i)
        assert sum(first.isoformat() <= d <= last.isoformat() for d in days) <= 4
    assert len(db.messages()) == 8


def test_nothing_qualifies_stores_nothing():
    db = Db()
    assert an.note_for("u1", ME, DELHI, date(2026, 10, 15), [], HINDU, festivals=[], db=db) is None
    assert db.docs == {} and db.commits == 0


def test_store_failure_shows_no_note():
    db = Db()
    db.fail = True
    assert an.note_for("u1", ME, DELHI, date(2026, 11, 8), [], HINDU, festivals=[DIWALI], db=db) is None


def test_without_a_db_the_note_is_only_composed():
    note = an.note_for("u1", ME, DELHI, date(2026, 11, 8), [], HINDU, festivals=[DIWALI])
    assert note["id"] == "u1__2026-11-08" and note["topic"] == "festival"


@pytest.mark.parametrize("thread", ["mobile_u1", "default", "some thread"])
def test_chat_doc_id_matches_main(thread):
    assert an.chat_doc_id("u1", thread) == main._chat_doc_id("u1", thread)
