"""Mithu's full kundli reading, money phase 1 (monetization-money §3.6, §5.1, §7, §12 Python):

- Parrot+ or a running promo (the legacy promise) reads every pair, and the pair stays unlocked;
- a server-written users/{uid}/kundliUnlocks/{pairKey} unlocks the pair for good (a doc not written by the server
  is ignored);
- useCredit spends one of users.reportCredits in a transaction, with the unlock doc, before the reading is
  written, so a failed reading is asked for again without another credit;
- otherwise 402 {credits: n}; REPORTS_OPEN=1 opens every reading (via 'open').

Firestore is an in-memory fake; the transaction runs through astro_api._run_transaction under a lock, the way
Firestore serialises two transactions on the same docs. The last test runs the real transaction against the
Firestore emulator and is skipped without it.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import astro_api
import main

ME = "pay123456"
H = {"Authorization": "Bearer ok"}
PROFILE = {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21", "birthTime": "07:40",
           "birthCityName": "Pune", "birthStateName": "Maharashtra", "birthCountryIso2": "IN"}
VIKRAM = {"name": "Vikram", "gender": "male", "birthDate": "1992-02-14", "birthTime": "09:10", "birthCity": "Chennai",
          "birthState": "Tamil Nadu"}
REPORT = "## The short answer\nA warm match."


# ------------------------------------------------------------------ a small Firestore stand-in
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
        self._db.docs[self.path] = {**(self._db.docs.get(self.path) or {}), **data} if merge else dict(data)


class Col:
    def __init__(self, db, path):
        self._db, self.path = db, path

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")


class Tx:
    """Writes are applied at commit, after work() returns, like Firestore."""

    def __init__(self, db):
        self._db, self.ops = db, []

    def update(self, ref, data):
        if ref.path not in self._db.docs:
            raise RuntimeError("update of a missing doc")
        self.ops.append((ref.path, data, True))

    def set(self, ref, data, merge=False):
        self.ops.append((ref.path, data, merge))


class Db:
    def __init__(self, docs=None):
        self.docs = {k: dict(v) for k, v in (docs or {}).items()}
        self.lock = threading.Lock()
        self.transactions = 0

    def collection(self, name):
        return Col(self, name)

    def run(self, work):
        with self.lock:
            self.transactions += 1
            tx = Tx(self)
            out = work(tx)
            for path, data, merge in tx.ops:
                self.docs[path] = {**(self.docs.get(path) or {}), **data} if merge else dict(data)
            return out


def make_client(monkeypatch, db, user=None, premium=False, complete=None, rate=None, profile=PROFILE):
    astro_api._hits.clear()
    astro_api._report_hits.clear()
    monkeypatch.setattr(astro_api, "_run_transaction", lambda d, work: d.run(work))
    monkeypatch.delenv("REPORTS_OPEN", raising=False)
    calls = []

    def default_complete(msgs, *a, **k):
        calls.append(msgs)
        return REPORT, "stop"

    astro_api.configure(astro_api.Deps(
        db=lambda: db,
        verify_uid=lambda h: ME if h == "Bearer ok" else (_ for _ in ()).throw(astro_api.HTTPException(401)),
        resolve_place=main._resolve_birth_place,
        places_query=main._places_query,
        load_user_docs=lambda uid: (dict(profile), {}, dict(db.docs.get(f"users/{ME}") or user or {})),
        is_premium=lambda u, p, now: premium() if callable(premium) else premium,
        complete=complete or default_complete,
        take_rate=rate or (lambda *a: True),
    ))
    app = FastAPI()
    app.include_router(astro_api.router)
    c = TestClient(app)
    c.calls = calls
    return c


def read(c, b=VIKRAM, **kw):
    return c.post("/astro/match/report", json={"b": b, **kw}, headers=H)


def unlocks(db):
    return {p: d for p, d in db.docs.items() if p.startswith(f"users/{ME}/kundliUnlocks/")}


def credits(db):
    return db.docs.get(f"users/{ME}", {}).get("reportCredits")


# ------------------------------------------------------------------ who may read
def test_legacy_parrot_plus_reads_and_the_pair_stays_unlocked(monkeypatch):
    db = Db({f"users/{ME}": {"premiumActive": True, "reportCredits": 2}})
    premium = {"on": True}
    c = make_client(monkeypatch, db, premium=lambda: premium["on"])
    r = read(c)
    assert r.status_code == 200, r.text
    assert r.json()["report"] == REPORT and r.json()["cached"] is False and r.json()["credits"] == 2
    (path, doc), = unlocks(db).items()
    assert doc["by"] == "server" and doc["via"] == "legacy" and doc["atIso"]
    assert credits(db) == 2                                         # nothing spent
    # Parrot+ ends: the pair stays unlocked (here in poruthams, so it isn't the kept reading).
    premium["on"] = False
    r = read(c, system="porutham")
    assert r.status_code == 200 and r.json()["cached"] is False and credits(db) == 2 and len(c.calls) == 2


def test_locked_reading_answers_402_with_the_credits(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 3}})
    c = make_client(monkeypatch, db)
    r = read(c)
    assert r.status_code == 402 and r.json() == {"detail": "credits_needed", "credits": 3}
    assert not c.calls and not unlocks(db) and credits(db) == 3
    none = Db({f"users/{ME}": {}})
    r = read(make_client(monkeypatch, none), useCredit=True)
    assert r.status_code == 402 and r.json()["credits"] == 0
    for odd in (-4, "5", True, None, 2.7):
        r = read(make_client(monkeypatch, Db({f"users/{ME}": {"reportCredits": odd}})))
        assert r.status_code == 402 and r.json()["credits"] == (2 if odd == 2.7 else 0), odd


def test_a_credit_is_spent_once_per_pair(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 2}})
    c = make_client(monkeypatch, db)
    r = read(c, useCredit=True)
    assert r.status_code == 200, r.text
    assert r.json()["credits"] == 1 and credits(db) == 1
    (path, doc), = unlocks(db).items()
    assert doc["by"] == "server" and doc["via"] == "credit"
    # The same pair again (another system, so not the kept reading): no second credit.
    assert read(c, useCredit=True, system="porutham").status_code == 200 and credits(db) == 1
    # Without useCredit too: the pair is unlocked.
    assert read(c, system="porutham").json()["cached"] is True and credits(db) == 1
    # Another pair costs another credit; with none left it is locked.
    other = {**VIKRAM, "birthDate": "1991-01-01"}
    assert read(c, b=other, useCredit=True).status_code == 200 and credits(db) == 0 and len(unlocks(db)) == 2
    r = read(c, b={**VIKRAM, "birthDate": "1990-05-05"}, useCredit=True)
    assert r.status_code == 402 and r.json()["credits"] == 0


def test_the_pair_key_is_the_same_either_way_round(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 1}})
    c = make_client(monkeypatch, db)
    asha = {"name": "Asha", "gender": "female", "birthDate": "1994-08-21", "birthTime": "07:40", "birthCity": "Pune",
            "birthState": "Maharashtra"}
    r = c.post("/astro/match/report", json={"a": asha, "b": VIKRAM, "useCredit": True}, headers=H)
    assert r.status_code == 200 and credits(db) == 0
    swapped = c.post("/astro/match/report", json={"a": VIKRAM, "b": asha, "system": "porutham"}, headers=H)
    assert swapped.status_code == 200, swapped.text          # unlocked: no credit left, none needed
    assert astro_api.pair_key("a", "b") == astro_api.pair_key("b", "a") != astro_api.pair_key("a", "c")
    # personKey: dob|time|lat4|lon4 for someone typed in, the uid for me and for members.
    me = astro_api._Me(ME)
    key = astro_api._person_key(me, astro_api.PersonIn(**VIKRAM))
    place = main._resolve_birth_place({"birthCityName": "Chennai", "birthStateName": "Tamil Nadu",
                                       "birthCountryIso2": "", "birthCountryName": "India", "birthDate": "1992-02-14"})
    assert key == f"1992-02-14|09:10|{place['lat']:.4f}|{place['lon']:.4f}"
    assert astro_api._person_key(me, None) == ME and astro_api._person_key(me, astro_api.PersonIn(me=True)) == ME
    assert astro_api._person_key(me, astro_api.PersonIn(uid="bride1234")) == "bride1234"
    assert astro_api._person_key(me, astro_api.PersonIn(name="X", birthDate="2001-01-01")).startswith("2001-01-01||")


def test_a_failed_reading_is_not_charged_twice(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 1}})
    state = {"fail": True}

    def complete(msgs, *a, **k):
        if state["fail"]:
            raise RuntimeError("the AI is down")
        return REPORT, "stop"

    c = make_client(monkeypatch, db, complete=complete)
    assert read(c, useCredit=True).status_code == 503
    assert credits(db) == 0 and len(unlocks(db)) == 1          # spent before writing, the pair unlocked
    state["fail"] = False
    r = read(c, useCredit=True)
    assert r.status_code == 200 and r.json()["report"] == REPORT and credits(db) == 0


def test_an_unlock_not_written_by_the_server_is_ignored(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 0}})
    c = make_client(monkeypatch, db)
    pair = astro_api.pair_key(ME, astro_api._person_key(astro_api._Me(ME), astro_api.PersonIn(**VIKRAM)))
    db.docs[f"users/{ME}/kundliUnlocks/{pair}"] = {"by": "client", "via": "credit"}
    assert read(c).status_code == 402
    db.docs[f"users/{ME}/kundliUnlocks/{pair}"] = {"via": "credit"}
    assert read(c).status_code == 402
    db.docs[f"users/{ME}/kundliUnlocks/{pair}"] = {"by": "server", "via": "credit", "atIso": "2026-10-11T00:00:00Z"}
    assert read(c).status_code == 200


def test_reports_open(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 2}})
    c = make_client(monkeypatch, db)
    monkeypatch.setenv("REPORTS_OPEN", "1")
    r = read(c, useCredit=True)
    assert r.status_code == 200 and credits(db) == 2                  # nothing spent while readings are open
    (path, doc), = unlocks(db).items()
    assert doc["via"] == "open" and doc["by"] == "server"
    monkeypatch.setenv("REPORTS_OPEN", "0")
    assert read(c, b={**VIKRAM, "birthDate": "1991-01-01"}).status_code == 402


def test_the_daily_cap_comes_before_a_credit_is_spent(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 2}})
    c = make_client(monkeypatch, db, rate=lambda uid, key, limit, per: key != "astroReport")
    r = read(c, useCredit=True)
    assert r.status_code == 429 and credits(db) == 2 and not unlocks(db)


def test_a_402_never_uses_a_daily_slot(monkeypatch):
    """The daily slots (this instance's and the shared serverRate counter) are taken only once the reader may read:
    a locked reading never touches them, and a credit found gone inside the spend transaction (another device spent
    the last one meanwhile) gives back the slots it took before answering 402. The cap still comes before a credit is
    spent (test_the_daily_cap_comes_before_a_credit_is_spent)."""
    import dataclasses
    db = Db({f"users/{ME}": {"reportCredits": 1}})
    counts, given_back = {}, []

    def rate(uid, key, limit, per):
        counts[key] = counts.get(key, 0) + 1
        return counts[key] <= limit

    def give_back(uid, key, per):
        given_back.append((uid, key, per))
        counts[key] -= 1

    c = make_client(monkeypatch, db, rate=rate)
    astro_api.configure(dataclasses.replace(astro_api._d(), give_back_rate=give_back))
    assert read(c).status_code == 402 and "astroReport" not in counts          # locked: no slot looked at
    real_spend = astro_api._spend_credit
    monkeypatch.setattr(astro_api, "_spend_credit", lambda d, uid, pair: None)   # the credit went meanwhile
    for i in range(astro_api.REPORTS_PER_DAY + 2):
        r = read(c, b={**VIKRAM, "birthDate": f"1990-01-{i + 1:02d}"}, useCredit=True)
        assert r.status_code == 402 and r.json()["credits"] == 0, (i, r.text)
    assert counts["astroReport"] == 0 and given_back[-1] == (ME, "astroReport", "day")
    assert astro_api._report_hits.get(ME, ("", 0))[1] == 0
    # Every slot is still there: the reading the account can pay for now is served.
    monkeypatch.setattr(astro_api, "_spend_credit", real_spend)
    r = read(c, useCredit=True)
    assert r.status_code == 200 and credits(db) == 0 and counts["astroReport"] == 1


def test_a_busy_counter_gives_back_this_instances_slot(monkeypatch):
    """The shared counter answering 503 busy_try_again (it couldn't be written) leaves this instance's slot unused."""
    import dataclasses
    db = Db({f"users/{ME}": {"premiumActive": True}})

    def busy(uid, key, limit, per):
        raise astro_api.HTTPException(status_code=503, detail="busy_try_again")

    c = make_client(monkeypatch, db, premium=True, rate=lambda *a: True)
    astro_api.configure(dataclasses.replace(astro_api._d(), take_rate_strict=busy))
    for _ in range(astro_api.REPORTS_PER_DAY + 1):
        assert read(c).status_code == 503
    assert astro_api._report_hits.get(ME, ("", 0))[1] == 0 and not c.calls


def test_a_kept_reading_is_never_charged_again(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 1}})
    c = make_client(monkeypatch, db)
    assert read(c, useCredit=True).status_code == 200 and credits(db) == 0
    # Even with the unlock gone (deleted by hand), the reading this account already has is served.
    for p in list(unlocks(db)):
        del db.docs[p]
    r = read(c)
    assert r.status_code == 200 and r.json()["cached"] is True and r.json()["report"] == REPORT


def test_parallel_spends_for_one_pair_take_one_credit(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 3}})
    make_client(monkeypatch, db)
    req = astro_api.MatchReq(b=astro_api.PersonIn(**VIKRAM), useCredit=True)

    def one(_):
        try:
            return astro_api.match_report(req, "Bearer ok")
        except astro_api.HTTPException as e:
            return e.status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        out = list(pool.map(one, range(6)))
    assert all(isinstance(o, dict) and o["ok"] for o in out), out
    assert credits(db) == 2 and len(unlocks(db)) == 1


def test_spend_credit_transaction_rules(monkeypatch):
    db = Db({f"users/{ME}": {"reportCredits": 1}})
    make_client(monkeypatch, db)
    assert astro_api._spend_credit(db, ME, "p1") == 0
    assert db.docs[f"users/{ME}/kundliUnlocks/p1"]["via"] == "credit"
    assert astro_api._spend_credit(db, ME, "p1") == 0             # unlocked already: nothing spent
    assert astro_api._spend_credit(db, ME, "p2") is None          # none left
    assert f"users/{ME}/kundliUnlocks/p2" not in db.docs
    assert astro_api._spend_credit(Db({}), ME, "p3") is None      # no users doc: nothing to spend


# ------------------------------------------------------------------ the real transaction (emulator)
@pytest.mark.skipif(not os.getenv("FIRESTORE_EMULATOR_HOST"), reason="needs the Firestore emulator")
def test_spend_credit_against_the_emulator():
    from google.cloud import firestore
    db = firestore.Client(project="demo-shaadi")
    uid = "emu_credit_user"
    user = db.collection("users").document(uid)
    user.set({"reportCredits": 2})
    unlock = user.collection("kundliUnlocks").document("pairA")
    unlock.delete()
    try:
        assert astro_api._spend_credit(db, uid, "pairA") == 1
        assert astro_api._spend_credit(db, uid, "pairA") == 1        # the pair is unlocked: nothing spent
        assert user.get().to_dict()["reportCredits"] == 1
        assert unlock.get().to_dict()["by"] == "server"
    finally:
        unlock.delete()
        user.delete()
