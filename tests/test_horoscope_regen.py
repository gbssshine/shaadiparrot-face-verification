"""The daily horoscope: rewritten the same day when the birth details change (at most twice), the family reader of
a managed profile, the ASTRO_MODE and TODAY_OBSERVANCE facts, and the Theme / Do / Avoid shape of the prompt.

Firestore is an in-memory fake; the transaction runs through main._run_transaction. The last test runs the real
transaction against the Firestore emulator and is skipped without it.
"""
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from google.api_core import exceptions as gexc

import main

UID = "reg123456"
H = {"Authorization": "Bearer ok"}
TZ = "Asia/Kolkata"


class Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d, self.exists = ref, ref.id, data, data is not None

    def to_dict(self):
        return None if self._d is None else json.loads(json.dumps(self._d))


class Ref:
    def __init__(self, db, path):
        self._db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def collection(self, name):
        return Col(self._db, f"{self.path}/{name}")

    def get(self, transaction=None):
        return Snap(self, self._db.docs.get(self.path))

    def set(self, data, merge=False):
        self._db.write(self.path, data, merge)

    def delete(self):
        self._db.docs.pop(self.path, None)


class Col:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")

    def limit(self, n):
        return self

    def stream(self):
        prefix = self.path + "/"
        return [Snap(Ref(self._db, p), d) for p, d in list(self._db.docs.items())
                if p.startswith(prefix) and "/" not in p[len(prefix):]]


class Batch:
    def __init__(self, db):
        self._db, self._ops = db, []

    def create(self, ref, data):
        self._ops.append(("create", ref, data, False))

    def set(self, ref, data, merge=False):
        self._ops.append(("set", ref, data, merge))

    def commit(self):
        if any(op == "create" and ref.path in self._db.docs for op, ref, _, _ in self._ops):
            raise gexc.AlreadyExists("Document already exists")
        for _, ref, data, merge in self._ops:
            self._db.write(ref.path, data, merge)


class Tx:
    """Writes are applied when the work returns, like a commit."""
    def __init__(self, db):
        self._db, self.ops = db, []

    def set(self, ref, data, merge=False):
        self.ops.append((ref.path, data, merge))


class Db:
    def __init__(self, docs=None):
        self.docs = {k: dict(v) for k, v in (docs or {}).items()}

    def collection(self, name):
        return Col(self, name)

    def get_all(self, refs):
        return [Snap(r, self.docs.get(r.path)) for r in list(refs)]

    def batch(self):
        return Batch(self)

    def write(self, path, data, merge):
        cur = dict(self.docs.get(path) or {}) if merge else {}
        cur.update(data)
        self.docs[path] = cur

    def run(self, work):
        tx = Tx(self)
        out = work(tx)
        for path, data, merge in tx.ops:
            self.write(path, data, merge)
        return out

    def messages(self):
        prefix = f"parrotChats/{UID}__"
        return sorted((d for p, d in self.docs.items() if p.startswith(prefix) and "/messages/" in p),
                      key=lambda d: d["createdAtMs"])


PROFILE = {"firstName": "Asha", "gender": "Female", "lifeStage": "single", "profileFor": "self",
           "birthDate": "1994-08-21"}


@pytest.fixture()
def env(monkeypatch):
    db = Db({f"profiles/{UID}": dict(PROFILE), f"users/{UID}": {"gender": "Female"}})
    calls, rates, leases = [], {}, set()

    def complete(messages, max_tokens, temperature=None, kind="chat", timeout=None):
        calls.append({"kind": kind, "messages": messages})
        return f"Good morning, Asha! Happy day.\nTheme: Reading number {len(calls)}\n🌙 Overall mood\n• Calm.", "stop"

    def take_rate(uid, key, limit, per="day"):
        rates[key] = rates.get(key, 0) + 1
        return rates[key] <= limit

    def take_lease(name, ttl):
        if name in leases:
            return False
        leases.add(name)
        return True

    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_verify_firebase_token_or_401", lambda a: UID)
    monkeypatch.setattr(main, "_deepseek_complete", complete)
    monkeypatch.setattr(main, "_take_rate", take_rate)
    monkeypatch.setattr(main, "_take_lease", take_lease)
    monkeypatch.setattr(main, "_release_lease", lambda name: leases.discard(name))
    monkeypatch.setattr(main, "_run_transaction", db.run)
    client = TestClient(main.app)       # not entered: no startup (no real Vision or Firestore clients)

    def ask():
        r = client.post("/daily-horoscope", json={"tz": TZ}, headers=H)
        assert r.status_code == 200, r.text
        return r.json()

    def day_doc():
        day = datetime.now(ZoneInfo(TZ)).date().isoformat()
        return db.docs.get(f"dailyHoroscopes/{UID}__{day}") or {}

    def change(**fields):
        db.docs[f"profiles/{UID}"].update(fields)

    def regen_leases():
        return [name for name in leases if name.count("_") == 3]     # horoscope_{uid}_{day}_{sig}

    from types import SimpleNamespace as NS
    return NS(db=db, calls=calls, rates=rates, leases=leases, ask=ask, day_doc=day_doc, change=change,
              regen_leases=regen_leases)


def test_first_reading_keeps_the_birth_signature(env):
    r = env.ask()
    assert r["ok"] and r["cached"] is False and "Reading number 1" in r["text"]
    doc = env.day_doc()
    assert doc["birthSig"] == main._birth_sig(PROFILE) and doc["regenCount"] == 0
    assert [m["kind"] for m in env.db.messages()] == ["daily_horoscope"]
    again = env.ask()
    assert again["cached"] is True and again["text"] == r["text"] and len(env.calls) == 1


def test_new_birth_details_rewrite_todays_reading_twice_then_it_stays(env):
    first = env.ask()["text"]
    env.change(birthTime="07:40", birthCityName="Pune", birthStateName="Maharashtra", birthCountryIso2="IN")
    second = env.ask()
    assert second["cached"] is False and second["text"] != first and "Reading number 2" in second["text"]
    doc = env.day_doc()
    assert doc["regenCount"] == 1 and doc["birthSig"] == main._birth_sig(env.db.docs[f"profiles/{UID}"])
    assert doc["text"] == second["text"] and doc["uid"] == UID and "createdAtIso" in doc and doc["updatedAtIso"]
    msgs = env.db.messages()
    assert len(msgs) == 2 and msgs[-1]["kind"] == "daily_horoscope" and msgs[-1]["updated"] is True
    assert msgs[-1]["text"] == second["text"]
    # the same details again: no new call
    assert env.ask() == {**second, "cached": True} and len(env.calls) == 2

    env.change(birthDate="1994-08-22")
    third = env.ask()
    assert third["cached"] is False and "Reading number 3" in third["text"] and env.day_doc()["regenCount"] == 2

    env.change(birthDate="1994-08-23")                   # a third change the same day: the reading stays
    fourth = env.ask()
    assert fourth == {**third, "cached": True}
    assert len(env.calls) == 3 and len(env.db.messages()) == 3
    assert env.rates["horoscope_regen"] == 2


def test_the_daily_counter_also_limits_rewrites(env):
    env.ask()
    env.rates["horoscope_regen"] = 2                     # spent today (another day key, another tz)
    env.change(birthTime="10:00")
    r = env.ask()
    assert r["cached"] is True and len(env.calls) == 1
    assert not env.regen_leases()                        # the rewrite's lease was given back


def test_an_approximate_birth_time_changes_the_signature(env):
    env.ask()
    env.db.docs[f"users/{UID}"]["birthTimeApprox"] = "night"
    r = env.ask()
    assert r["cached"] is False and env.day_doc()["regenCount"] == 1


def test_a_reading_written_before_signatures_is_served_as_it_is(env):
    day = datetime.now(ZoneInfo(TZ)).date().isoformat()
    old = {"uid": UID, "dayKey": day, "text": "Yesterday's style reading", "tz": TZ, "createdAtIso": "x"}
    env.db.docs[f"dailyHoroscopes/{UID}__{day}"] = dict(old)
    env.change(birthTime="05:00")
    r = env.ask()
    assert r == {"ok": True, "text": "Yesterday's style reading", "dayKey": day, "cached": True}
    assert env.calls == [] and env.day_doc() == old


def test_a_failed_rewrite_keeps_the_cached_reading(env, monkeypatch):
    first = env.ask()

    def down(*a, **k):
        raise main.HTTPException(status_code=502, detail="DeepSeek error")

    monkeypatch.setattr(main, "_deepseek_complete", down)
    env.change(birthTime="08:15")
    r = env.ask()
    assert r["cached"] is True and r["text"] == first["text"]
    assert env.day_doc()["regenCount"] == 0 and not env.regen_leases()


def test_a_rewrite_another_request_already_wrote_wins(env):
    env.ask()
    env.change(birthTime="11:11")
    sig = main._birth_sig({**env.db.docs[f"profiles/{UID}"], **env.db.docs[f"users/{UID}"]})
    day = datetime.now(ZoneInfo(TZ)).date().isoformat()
    # the other request finished between our read and our transaction
    text, mine = main._store_regenerated_horoscope(UID, f"mobile_{UID}", day, sig, "Mine")
    assert mine is True
    text, mine = main._store_regenerated_horoscope(UID, f"mobile_{UID}", day, sig, "Second writer")
    assert (text, mine) == ("Mine", False) and env.day_doc()["regenCount"] == 1


def test_birth_signature_rule():
    p = {"birthDate": "1990-01-02", "birthTime": "05:30", "birthCityName": "Agra", "birthTimeApprox": ""}
    import hashlib
    assert main._birth_sig(p) == hashlib.sha1(b"1990-01-02|05:30|Agra|").hexdigest()[:12]
    assert main._birth_sig({}) == hashlib.sha1(b"|||").hexdigest()[:12]
    assert main._birth_sig({**p, "birthTimeApprox": "morning"}) != main._birth_sig(p)
    assert main._birth_sig({**p, "firstName": "Other", "cityName": "Delhi"}) == main._birth_sig(p)


def test_an_approximate_time_moves_only_the_moon_never_a_lagna():
    base = {"birthDate": "1994-08-21", "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN"}
    morning = main._natal_chart({**base, "birthTimeApprox": "morning"})
    nine = main._natal_chart({**base, "birthTime": "09:00"})
    noon = main._natal_chart(dict(base))
    assert morning["lagna"] is None and morning["time_known"] is False and nine["lagna"] is not None
    assert (morning["moon_sign"], morning["nakshatra"]) == (nine["moon_sign"], nine["nakshatra"])
    assert noon["lagna"] is None
    assert main._natal_chart({**base, "birthTimeApprox": "sometime"}) == noon


# ------------------------------------------------------------------ the facts and the prompt
NATAL = {"sun_sign": "Leo", "moon_sign": "Taurus", "moon_sign_idx": 1, "nakshatra": "Rohini", "nakshatra_idx": 3,
         "lagna": None, "time_known": False}
SKY = {"weekday": "Sunday", "date": "11 October 2026", "lord": "Sun", "tithi": "Shukla Pratipada",
       "moon_sign": "Virgo", "moon_sign_idx": 5, "moon_nakshatra": "Chitra", "moon_nak_idx": 13, "moon_sign_later": None}


def test_managed_profile_reader_snapshot():
    managed = {"firstName": "Ananya", "gender": "Female", "lifeStage": "single", "profileFor": "daughter",
               "religion": "Hindu"}
    assert main._horoscope_facts(managed, NATAL, SKY).split("\n") == [
        "USER: first_name=Ananya; gender=Female; life_stage=single",
        "READER: family of Ananya (the profile was made by a parent or relative)",
        "ASTRO_MODE: full",
        "TODAY_OBSERVANCE: none",
        "TODAY: Sunday, 11 October 2026",
        "WEEKDAY_LORD: Sun",
        "TITHI: Shukla Pratipada",
        "MOON_TRANSIT: Virgo, Chitra nakshatra",
        "NATAL: Moon sign (Rashi)=Taurus; Nakshatra=Rohini; Sun sign=Leo; Lagna unknown (no birth time or place)",
        "MOON_FROM_NATAL_MOON: house 5, theme: romance, creativity and fun",
        "TARA: Sampat, supportive: good for gains and progress",
        "LUCKY_COLOR: pearl white",
        "LUCKY_NUMBER: 2",
    ]


@pytest.mark.parametrize("profile_for", ["son", "daughter", "brother", "sister", "relative", "friend", " Son "])
def test_every_family_made_profile_reads_as_family(profile_for):
    facts = main._horoscope_facts({"firstName": "Rahul", "profileFor": profile_for}, NATAL, SKY)
    assert "READER: family of Rahul (the profile was made by a parent or relative)" in facts


@pytest.mark.parametrize("profile_for", ["self", "", None, "none"])
def test_my_own_profile_reads_as_self(profile_for):
    assert "READER: self\n" in main._horoscope_facts({"firstName": "Asha", "profileFor": profile_for}, NATAL, SKY)


def test_astro_mode_and_observance_facts():
    def mode(**p):
        return main._horoscope_facts(p, NATAL, SKY).split("\n")[2]

    assert mode(religion="Hindu") == "ASTRO_MODE: full"
    assert mode() == "ASTRO_MODE: full"                                          # unknown: full
    assert mode(religion="Prefer not to say") == "ASTRO_MODE: full"
    assert mode(religion="Muslim") == "ASTRO_MODE: fun (keep it light, entertainment framing)"   # until they answer
    assert mode(religion="Parsi / Zoroastrian").startswith("ASTRO_MODE: fun")
    assert mode(religion="No religion / Spiritual").startswith("ASTRO_MODE: fun")
    assert mode(religion="Hindu", celebrates=["hindu", "sikh"]).startswith("ASTRO_MODE: fun")
    assert mode(religion="Muslim", astroMode="full") == "ASTRO_MODE: full"       # their own answer wins
    assert mode(religion="Hindu", astroMode="off") == "ASTRO_MODE: off"
    assert mode(religion="Hindu", astroMode="weird") == "ASTRO_MODE: full"
    facts = main._horoscope_facts({}, NATAL, SKY, observance="  Karva Chauth (fast until moonrise)  ")
    assert "TODAY_OBSERVANCE: Karva Chauth (fast until moonrise)\n" in facts


def test_the_prompt_asks_for_theme_do_and_avoid():
    msgs = main._horoscope_messages({"firstName": "Asha"}, NATAL, SKY)
    system = msgs[0]["content"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and "READER: self" in msgs[1]["content"]
    assert "Line 2: 'Theme: ' followed by today's theme in at most 7 words." in system
    assert "\n✅ Do\n" in system and "\n⛔ Avoid" in system and "exactly 2 lines" in system and "at most 8 words" in system
    assert "Today's tip" not in system and "180–250 words" in system
    assert "five sections" in system and "🎨 Lucky color & number" in system
    assert "If READER is family, write about the person named in READER in the third person" in system
    assert "Greet the reader warmly without a name. Give tips the family can act on." in system
    assert "If ASTRO_MODE is fun, keep it light" in system and "TODAY_OBSERVANCE" in system
    assert "Respect every faith" in system


def test_the_generation_uses_that_prompt(env):
    env.db.docs[f"profiles/{UID}"].update(profileFor="daughter", firstName="Ananya")
    env.ask()
    sent = env.calls[0]["messages"]
    assert sent[0]["content"] == main._HOROSCOPE_SYSTEM_PROMPT
    assert "READER: family of Ananya" in sent[1]["content"] and "TODAY_OBSERVANCE: none" in sent[1]["content"]


# ------------------------------------------------------------------ the real transaction (emulator)
@pytest.mark.skipif(not os.getenv("FIRESTORE_EMULATOR_HOST"), reason="needs the Firestore emulator")
def test_the_rewrite_transaction_against_the_emulator(monkeypatch):
    from google.cloud import firestore
    project = os.getenv("GCLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "demo-fates"
    db = firestore.Client(project=project)
    monkeypatch.setattr(main, "firestore_client", db)
    uid, day = f"regen_{time.time_ns()}", "2026-10-11"
    thread = f"mobile_{uid}"
    text, cached = main._store_daily_horoscope(uid, thread, day, TZ, "First", birth_sig="aaaaaaaaaaaa")
    assert (text, cached) == ("First", False)
    assert main._store_regenerated_horoscope(uid, thread, day, "bbbbbbbbbbbb", "Second") == ("Second", True)
    assert main._store_regenerated_horoscope(uid, thread, day, "bbbbbbbbbbbb", "Again") == ("Second", False)
    assert main._store_regenerated_horoscope(uid, thread, day, "cccccccccccc", "Third") == ("Third", True)
    assert main._store_regenerated_horoscope(uid, thread, day, "dddddddddddd", "Fourth") == ("Third", False)
    doc = db.document(f"dailyHoroscopes/{uid}__{day}").get().to_dict()
    assert doc["text"] == "Third" and doc["regenCount"] == 2 and doc["birthSig"] == "cccccccccccc"
    assert doc["tz"] == TZ and doc["createdAtIso"]                   # set with merge: the first fields stay
    msgs = [m.to_dict() for m in main._chat_msgs_col_ref(uid, thread).stream()]
    assert sorted(m["text"] for m in msgs) == ["First", "Second", "Third"]
    assert sum(1 for m in msgs if m.get("updated") is True) == 2
