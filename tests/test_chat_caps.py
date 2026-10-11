"""The Mithu chat contract for 1.0.8: caps ["actions"] turn the birth hint into a button (at most once per 14 days),
hint 'birth_details' whenever it applies, Hindi questions trigger it too, /history passes the message kind, the
language rule and the per-user facts in the prompt, and the strict (fail-closed) rate counter.

Firestore is an in-memory fake; DeepSeek, the token check and the quota are replaced.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import main

UID = "chat123456"
H = {"Authorization": "Bearer ok"}
THREAD = f"mobile_{UID}"
HINT_TEXT = "add your birth time and place in your profile"
BUTTON = [{"type": "add_birth_time", "label": "Add my birth time"}]


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


class Col:
    def __init__(self, db, path, order=None, n=None):
        self._db, self.path, self._order, self._n = db, path, order, n

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")

    def order_by(self, field, direction=None):
        return Col(self._db, self.path, (field, direction == "DESCENDING"), self._n)

    def limit(self, n):
        return Col(self._db, self.path, self._order, n)

    def stream(self):
        prefix = self.path + "/"
        snaps = [Snap(Ref(self._db, p), d) for p, d in list(self._db.docs.items())
                 if p.startswith(prefix) and "/" not in p[len(prefix):]]
        if self._order:
            field, desc = self._order
            snaps.sort(key=lambda s: s._d.get(field) or 0, reverse=desc)
        return snaps[: self._n] if self._n else snaps


class Batch:
    def __init__(self, db):
        self._db, self._ops = db, []

    def set(self, ref, data, merge=False):
        self._ops.append((ref, data, merge))

    def commit(self):
        for ref, data, merge in self._ops:
            self._db.write(ref.path, data, merge)


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
        def apply(dst, src):
            for k, v in src.items():
                if merge and isinstance(v, dict) and isinstance(dst.get(k), dict):
                    apply(dst[k], v)
                else:
                    dst[k] = v
        cur = json.loads(json.dumps(self.docs.get(path) or {})) if merge else {}
        apply(cur, data)
        self.docs[path] = cur

    def state(self):
        return self.docs.get(f"parrotChats/{main._chat_doc_id(UID, THREAD)}") or {}

    def messages(self):
        prefix = f"parrotChats/{main._chat_doc_id(UID, THREAD)}/messages/"
        return sorted((d for p, d in self.docs.items() if p.startswith(prefix)), key=lambda d: d["createdAtMs"])


ENGLISH_READING = "🌙 Your day\n• The Moon in your rashi brings calm.\n• Venus favours family talks."
HINDI_READING = "🌙 आपका दिन\n• आज चंद्र आपकी राशि में है।\n• परिवार के साथ समय बिताइए।"


@pytest.fixture()
def env(monkeypatch):
    db = Db({f"profiles/{UID}": {"firstName": "Asha", "gender": "Female", "birthDate": "1994-08-21",
                                 "lifeStage": "single", "profileFor": "self"},
             f"users/{UID}": {"gender": "Female"}})
    sent = []
    reply = {"text": ENGLISH_READING}

    def complete(messages, max_tokens, temperature=None, kind="chat", timeout=None):
        sent.append({"kind": kind, "messages": messages})
        return reply["text"], "stop"

    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_verify_firebase_token_or_401", lambda a: UID)
    monkeypatch.setattr(main, "_take_rate", lambda *a, **k: True)
    monkeypatch.setattr(main, "_reserve_parrot_reply", lambda *a, **k: None)
    monkeypatch.setattr(main, "_deepseek_complete", complete)
    client = TestClient(main.app)       # not entered: no startup (no real Vision or Firestore clients)

    def chat(text, caps=None, **extra):
        body = {"text": text, "locale": "en", "thread_id": THREAD, **extra}
        if caps is not None:
            body["caps"] = caps
        r = client.post("/ai-chat", json=body, headers=H)
        assert r.status_code == 200, r.text
        return r.json()

    from types import SimpleNamespace as NS
    return NS(db=db, sent=sent, reply=reply, chat=chat, client=client)


# ------------------------------------------------------------------ the birth hint
def test_an_old_app_gets_the_hint_as_text(env):
    r = env.chat("What does my horoscope say today?")
    assert HINT_TEXT in r["reply_text"] and r["hint"] == "birth_details" and r["actions"] is None
    assert set(r) == {"reply_text", "blocked", "reason", "thread_id", "actions", "hint"}
    assert HINT_TEXT in env.db.messages()[-1]["text"]
    # not twice in a row (the old rule), and then no hint field either
    r2 = env.chat("And my horoscope for love?")
    assert HINT_TEXT not in r2["reply_text"] and r2["hint"] is None


def test_a_new_app_gets_a_button_instead_once_per_14_days(env):
    r = env.chat("What does my horoscope say today?", caps=["actions"])
    assert HINT_TEXT not in r["reply_text"] and r["reply_text"] == ENGLISH_READING
    assert r["actions"] == BUTTON and r["hint"] == "birth_details"
    asked = env.db.state()["asked"]["birth_time"]
    assert main._to_utc_datetime(asked) is not None
    assert env.db.messages()[-1]["text"] == ENGLISH_READING          # the stored reply has no hint text
    # within 14 days: no button, no hint, still no text
    r2 = env.chat("Tell me about my nakshatra", caps=["actions"])
    assert r2["actions"] is None and r2["hint"] is None and HINT_TEXT not in r2["reply_text"]
    assert env.db.state()["asked"]["birth_time"] == asked
    # 15 days later it may ask again
    old = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    env.db.docs[f"parrotChats/{main._chat_doc_id(UID, THREAD)}"]["asked"] = {"birth_time": old}
    r3 = env.chat("Vedic forecast for tomorrow?", caps=["Actions", "something_else"])
    assert r3["actions"] == BUTTON and r3["hint"] == "birth_details"
    assert env.db.state()["asked"]["birth_time"] != old
    assert env.db.state()["turns"] == 3                                # the rest of the state patch still lands


def test_only_the_place_missing_asks_for_the_place(env):
    env.db.docs[f"profiles/{UID}"]["birthTime"] = "07:40"
    r = env.chat("My horoscope please", caps=["actions"])
    assert r["actions"] == [{"type": "add_birth_time", "label": "Add my birth place"}]


def test_no_hint_when_the_birth_details_are_complete_or_unknown_or_off_topic(env):
    env.db.docs[f"profiles/{UID}"].update(birthTime="07:40", birthCityName="Pune")
    assert env.chat("My horoscope please", caps=["actions"])["hint"] is None
    assert env.chat("My horoscope please")["hint"] is None
    env.db.docs[f"profiles/{UID}"] = {"firstName": "Asha"}                   # no birth date: nothing to sharpen
    assert env.chat("My horoscope please", caps=["actions"])["actions"] is None
    env.db.docs[f"profiles/{UID}"] = {"firstName": "Asha", "birthDate": "1994-08-21"}
    r = env.chat("How should I talk to my in-laws about the wedding budget?", caps=["actions"])
    assert r["actions"] is None and r["hint"] is None                         # not an astro question


@pytest.mark.parametrize("question", [
    "मेरा आज का राशिफल बताइए",
    "मेरी कुंडली में क्या है?",
    "मेरा नक्षत्र कौन सा है",
    "mera rashifal batao",
    "meri kundali dekho",
    "aaj ka din kaisa rahega?",
])
def test_a_hindi_question_triggers_the_hint(env, question):
    env.reply["text"] = HINDI_READING
    assert main._infer_intent(question) == main.Intent.ASTRO
    r = env.chat(question, caps=["actions"])
    assert r["actions"] == BUTTON and r["hint"] == "birth_details"
    assert env.sent[-1]["kind"] == "chat:astro"


def test_a_hindi_question_on_an_old_app_gets_the_text_hint(env):
    env.reply["text"] = HINDI_READING
    r = env.chat("मेरा आज का राशिफल बताइए")
    assert r["reply_text"].startswith(HINDI_READING) and HINT_TEXT in r["reply_text"] and r["hint"] == "birth_details"


def test_english_intents_are_unchanged():
    assert main._infer_intent("What should I reply to her message?") == main.Intent.TEXTING
    assert main._infer_intent("Can you fix my bio?") == main.Intent.PROFILE
    assert main._infer_intent("Tell me about Mumbai") == main.Intent.GENERAL
    assert main._infer_intent("Show me a photograph of a graph") == main.Intent.GENERAL
    assert main._infer_intent("Is my kundli good?") == main.Intent.ASTRO


# ------------------------------------------------------------------ the prompt
def test_the_language_rule_replaces_reply_in_english(env):
    env.chat("नमस्ते मिठू")
    system = "\n".join(m["content"] for m in env.sent[-1]["messages"] if m["role"] == "system")
    assert main.CHAT_LANGUAGE_RULE in system and "Reply in English." not in system
    assert main.CHAT_LANGUAGE_RULE == (
        "Reply in the language and script the user wrote in (Hindi in Devanagari if they wrote Hindi; never Roman "
        "Hinglish unless they did). Mithu speaks of himself in the masculine.")
    assert "Reply in Russian." in main._build_intent_prompt("ru", main.Intent.GENERAL)
    assert main.CHAT_LANGUAGE_RULE not in main._build_intent_prompt("ru", main.Intent.GENERAL)
    for intent in (main.Intent.MATCH, main.Intent.ASTRO, main.Intent.GENERAL):
        assert main._build_intent_prompt("en", intent).endswith(main.CHAT_LANGUAGE_RULE + "\n")


def test_the_faith_rule_and_the_day_facts(env):
    assert ("Respect every faith. Use the user's own festival names. If ASTRO_MODE is off, don't bring up astrology "
            "unless asked.") in main._SYSTEM_PROMPT_BASE
    env.chat("Hello Mithu")
    contents = [m["content"] for m in env.sent[-1]["messages"]]
    assert contents[0] == main._SYSTEM_PROMPT_BASE                    # the cached prefix stays first
    assert "ASTRO_MODE=full | TODAY_OBSERVANCE=none" in contents
    env.db.docs[f"users/{UID}"]["astroMode"] = "off"
    env.chat("Hello again")
    assert "ASTRO_MODE=off (no astrology unless the user asks) | TODAY_OBSERVANCE=none" in \
        [m["content"] for m in env.sent[-1]["messages"]]
    env.db.docs[f"users/{UID}"].pop("astroMode")
    env.db.docs[f"profiles/{UID}"]["religion"] = "Christian"
    env.chat("Hello once more")
    assert "ASTRO_MODE=fun (keep it light, entertainment framing) | TODAY_OBSERVANCE=none" in \
        [m["content"] for m in env.sent[-1]["messages"]]
    facts = main._day_facts({}, observance="Diwali")
    assert facts == "ASTRO_MODE=full | TODAY_OBSERVANCE=Diwali"


# ------------------------------------------------------------------ /history
def test_history_passes_the_kind_and_keeps_old_rows_as_they_were(env):
    col = f"parrotChats/{main._chat_doc_id(UID, THREAD)}/messages"
    env.db.docs.update({
        f"{col}/1": {"role": "user", "text": "Hi", "createdAtMs": 1},
        f"{col}/2": {"role": "assistant", "text": "Good morning reading", "createdAtMs": 2, "kind": "daily_horoscope",
                     "updated": True},
        f"{col}/3": {"role": "assistant", "text": "Karva Chauth is tomorrow.", "createdAtMs": 3, "kind": "mithu_note",
                     "noteId": f"{UID}__2026-10-28"},
        f"{col}/4": {"role": "assistant", "text": "Plain reply", "createdAtMs": 4, "kind": ""},
    })
    r = env.client.get(f"/history?thread_id={THREAD}", headers=H)
    assert r.status_code == 200
    assert r.json() == {"thread_id": THREAD, "messages": [
        {"role": "user", "text": "Hi"},
        {"role": "assistant", "text": "Good morning reading", "kind": "daily_horoscope"},
        {"role": "assistant", "text": "Karva Chauth is tomorrow.", "kind": "mithu_note"},
        {"role": "assistant", "text": "Plain reply"},
    ]}


def test_the_prompt_window_ignores_the_kind():
    rows = [{"role": "assistant", "content": "Reading", "ms": 1, "kind": "daily_horoscope"},
            {"role": "user", "content": "Thanks", "ms": 2}]
    assert main._history_window(rows) == [{"role": "assistant", "content": "Reading"},
                                          {"role": "user", "content": "Thanks"}]


# ------------------------------------------------------------------ the strict rate counter
def test_the_strict_counter_counts_like_the_open_one(monkeypatch):
    db = Db()

    class Tx:       # the counter's transaction, run through main._run_transaction
        def set(self, ref, data, merge=False):
            db.write(ref.path, data, merge)

    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_run_transaction", lambda work: work(Tx()))
    assert [main._take_rate_strict("u1", "mithuTake", 2) for _ in range(3)] == [True, True, False]
    assert db.docs["serverRate/u1"]["mithuTake"]["n"] == 2
    assert [main._take_rate("u2", "photoChecks", 1) for _ in range(2)] == [True, False]


def test_the_strict_counter_fails_closed(monkeypatch):
    class Broken:
        def collection(self, name):
            raise RuntimeError("firestore is down")

    monkeypatch.setattr(main, "firestore_client", Broken())
    assert main._take_rate_strict("u1", "mithuTakeAll", 1000) is False
    assert main._take_rate("u1", "aiChatMinute", 8, per="minute") is True          # the old one stays open

    def aborted(work):
        raise main.google_exceptions.Aborted("contention")

    monkeypatch.setattr(main, "firestore_client", Db())
    monkeypatch.setattr(main, "_run_transaction", aborted)
    assert main._take_rate_strict("u1", "mithuTake", 8) is False
    assert main._take_rate("u1", "mithuTake", 8) is True
    monkeypatch.setattr(main, "firestore_client", None)
    assert main._take_rate_strict("u1", "mithuTake", 8) is False
    assert main._take_rate("u1", "mithuTake", 8) is True
