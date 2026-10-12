"""What reaches DeepSeek from a profile, and which photos the face check uses.

- Caste, sub-caste, gotra, Manglik and income never leave the server in a chat prompt, whatever the intent
  (the _profile_context_compact key allow-list).
- The /match breakdown carries no birth details of either person (only the computed MATCH_ASTRO signs).
- While a user's photos are private the card holds blurs; the face check reads the real links from the profile.
"""
import json
from urllib.parse import quote, unquote, urlparse

import pytest

import face_live
import face_match
import main

# Values that can't appear in a prompt by accident.
PRIVATE = {
    "caste": "Zqcaste Brahmin",
    "subCaste": "Zqsub Iyengar",
    "gotra": "Zqgotra Kashyap",
    "manglik": "Zqmanglik Partial",
    "annualIncome": "Zqincome 2530",
    "annualIncomeVisibility": "Zqvisibility matches",
}
BIRTH = {"birthDate": "1995-03-14", "birthTime": "06:45", "birthCityName": "Pune",
         "birthStateName": "Maharashtra", "birthCountryName": "India", "birthCountryIso2": "IN"}
PROFILE = {
    "firstName": "Asha", "age": 29, "gender": "Female", "lifeStage": "single", "profileFor": "self",
    "maritalStatus": "Never married", "cityName": "Bengaluru", "countryName": "India", "languages": ["Kannada"],
    "bio": "Loves music", "aboutMe": "Teacher", "interests": ["music"], "tags": ["calm"], "education": "MA",
    "jobTitle": "Teacher", "occupation": "Education", "workout": "Sometimes", "smoking": "No", "drinking": "No",
    **BIRTH, **PRIVATE,
}
INTENTS = [v for k, v in vars(main.Intent).items() if k.isupper()]


def _no_private(text: str):
    for key, value in PRIVATE.items():
        assert f"{key}=" not in text, key
        assert value not in text and value.split()[0] not in text, key


def test_every_intent_is_covered():
    assert set(INTENTS) == {"match", "astro", "texting", "profile", "relation", "general"}


@pytest.mark.parametrize("intent", INTENTS)
def test_context_line_never_carries_caste_gotra_manglik_or_income(intent):
    for prefix in ("USER", "MATCH"):
        line = main._profile_context_compact(dict(PROFILE), intent, prefix=prefix)
        assert line.startswith(f"{prefix}_CONTEXT: ")
        _no_private(line)


@pytest.mark.parametrize("intent", INTENTS)
def test_the_whole_prompt_never_carries_them_either(intent):
    msgs = main._build_llm_messages(
        locale="en", intent=intent, summary="", user_profile=dict(PROFILE),
        match_profile=dict(PROFILE) if intent == main.Intent.MATCH else None, distance_km=12,
        user_text="Tell me about my day", history=[], family_ctx="")
    _no_private(json.dumps(msgs, ensure_ascii=False))


def test_match_context_has_no_birth_details_of_either_person():
    for prefix in ("USER", "MATCH"):
        line = main._profile_context_compact(dict(PROFILE), main.Intent.MATCH, prefix=prefix)
        assert "birthDate=" not in line and "birthTime=" not in line and "birthCity" not in line
        assert "birthState" not in line and "birthCountry" not in line
        assert "firstName=Asha" in line and "education=MA" in line         # the rest of the breakdown stays


def test_the_match_prompt_keeps_the_signs_but_no_birth_details():
    other = {**PROFILE, "firstName": "Ravi", "gender": "Male", "birthDate": "1992-11-02", "birthTime": "21:10",
             "birthCityName": "Mysuru", "birthStateName": "Karnataka"}
    msgs = main._build_llm_messages(
        locale="en", intent=main.Intent.MATCH, summary="", user_profile=dict(PROFILE), match_profile=other,
        distance_km=140, user_text="/match abcdef123", history=[], family_ctx="")
    text = "\n".join(m["content"] for m in msgs)
    for p in (PROFILE, other):
        assert p["birthDate"] not in text and p["birthTime"] not in text
    assert "birthDate=" not in text and "birthTime=" not in text and "birthCity" not in text
    assert "USER_ASTRO: Sun=" in text and "MATCH_ASTRO: Sun=" in text and "Nakshatra=" in text
    assert "DISTANCE_KM: 140" in text


def test_my_own_astro_question_still_uses_my_birth_details():
    line = main._profile_context_compact(dict(PROFILE), main.Intent.ASTRO, prefix="USER")
    assert "birthDate=1995-03-14" in line and "birthTime=06:45" in line and "birthCityName=Pune" in line


# ------------------------------------------------------------------ photos the face check uses
BUCKET = "shaadiparrot.firebasestorage.app"


def link(path: str) -> str:
    return f"https://firebasestorage.googleapis.com/v0/b/{BUCKET}/o/{quote(path, safe='')}?alt=media&token=t"


class Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d, self.exists = ref, ref.id, data, data is not None

    def to_dict(self):
        return None if self._d is None else json.loads(json.dumps(self._d))


class Ref:
    def __init__(self, db, path):
        self._db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def get(self, transaction=None):
        self._db.reads.append(self.path)
        return Snap(self, self._db.docs.get(self.path))


class Col:
    def __init__(self, db, path, filters=()):
        self._db, self.path, self._filters = db, path, filters

    def document(self, id_):
        return Ref(self._db, f"{self.path}/{id_}")

    def where(self, filter):
        return Col(self._db, self.path, self._filters + ((filter.field_path, filter.value),))

    def limit(self, n):
        return self

    def stream(self):
        prefix = self.path + "/"
        out = []
        for path, doc in self._db.docs.items():
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                if all(doc.get(f) == v for f, v in self._filters):
                    out.append(Snap(Ref(self._db, path), doc))
        return out


class Db:
    def __init__(self, docs):
        self.docs, self.reads = docs, []

    def collection(self, name):
        return Col(self, name)

    def get_all(self, refs):
        refs = list(refs)
        self.reads.extend(r.path for r in refs)
        return [Snap(r, self.docs.get(r.path)) for r in reversed(refs)]    # any order, like Firestore


def test_private_photos_come_from_the_profile_never_the_blurs(monkeypatch):
    real = [link(f"users/u1/photos/{i}.jpg") for i in range(8)]
    blurs = [link(f"users/u1/blur/{i:040d}.jpg") for i in range(8)]
    db = Db({"profiles/u1": {"photoPrivacy": "accepted", "photos": real},
             "publicProfiles/u1": {"photos": blurs}})
    monkeypatch.setattr(main, "firestore_client", db)
    got = main._own_photos("u1")
    assert got == real[:6]
    assert not any("/blur/" in unquote(u) for u in got)
    assert db.reads == ["profiles/u1", "publicProfiles/u1"]          # one get_all, both docs


def test_public_photos_come_from_the_card(monkeypatch):
    card = [link("users/u2/photos/a.jpg"), link("users/u2/photos/b.jpg")]
    db = Db({"profiles/u2": {"photoPrivacy": "all", "photos": [link("users/u2/photos/old.jpg")]},
             "publicProfiles/u2": {"photos": card}})
    monkeypatch.setattr(main, "firestore_client", db)
    assert main._own_photos("u2") == card


def test_only_my_own_storage_photos_reach_the_face_check(monkeypatch):
    mine = link("users/u3/photos/me.jpg")
    db = Db({"profiles/u3": {"photoPrivacy": "accepted",
                             "photos": [link("users/other/photos/x.jpg"), "https://evil.example/users/u3/photos/a.jpg",
                                        "mascot.png", "", None, mine]},
             "publicProfiles/u3": {"photos": [link("users/u3/photos/card.jpg")]}})
    monkeypatch.setattr(main, "firestore_client", db)
    assert main._own_photos("u3") == [mine]


@pytest.mark.parametrize("private", [True, False])
def test_a_non_own_link_among_the_first_6_is_dropped_not_replaced(monkeypatch, private):
    """First the 6 http links, then the own-photo filter (fates_engine.own_private_photos' order): a blur or a
    foreign link among the first 6 leaves 5, and the 7th photo never moves up. Otherwise faceVerifiedPhotos would
    hold the 7th while Daily Fates checks the first 6 against it."""
    import fates_engine
    uid = "u4"
    own = [link(f"users/{uid}/photos/{i}.jpg") for i in range(6)]
    photos = own[:2] + [link(f"users/{uid}/blur/{0:040d}.jpg")] + own[2:]   # 7 links, the blur 3rd
    profile = {"photoPrivacy": "accepted" if private else "all", "photos": photos if private else []}
    card = {"photos": [] if private else photos}
    monkeypatch.setattr(main, "firestore_client", Db({f"profiles/{uid}": profile, f"publicProfiles/{uid}": card}))
    got = main._own_photos(uid)
    assert got == own[:5] and own[5] not in got
    assert got == fates_engine.own_private_photos(uid, {"photos": photos})
    # The faceVerifiedPhotos saved from this list cover what Daily Fates checks for a private profile.
    private_doc = {"isFaceVerified": True, "faceVerifiedPhotos": got, "photos": photos}
    assert fates_engine.verified_now(private_doc, {}, fates_engine.own_private_photos(uid, private_doc))


def test_a_public_card_keeps_daily_fates_verified_with_a_variant_on_it(monkeypatch):
    """The chain for public photos: the list main._own_photos hands the face check (saved as faceVerifiedPhotos) is
    the list Daily Fates checks and shows, so a resized variant or someone else's blur among the card's photos never
    pauses it and is never shown."""
    import fates_engine
    uid = "u5"
    own = [link(f"users/{uid}/photos/{i}.jpg") for i in range(3)]
    photos = own[:1] + [link(f"users/{uid}/variants/0_960.jpg"), link(f"users/other/blur/{0:040d}.jpg")] + own[1:]
    profile = {"photoPrivacy": "all", "photos": photos}
    card = {"gender": "Female", "isDiscoverable": True, "photos": photos}
    monkeypatch.setattr(main, "firestore_client", Db({f"profiles/{uid}": profile, f"publicProfiles/{uid}": card}))
    got = main._own_photos(uid)
    assert got == own
    person = fates_engine.person_from_docs(uid, card, {**profile, "faceVerified": True, "faceVerifiedPhotos": got})
    assert person.face_verified and person.photos == own


def test_no_profile_docs_means_no_photos(monkeypatch):
    monkeypatch.setattr(main, "firestore_client", Db({}))
    assert main._own_photos("nobody") == []


def test_cohort_skips_blurred_cards_through_own_photo_url(monkeypatch):
    """_cohort has no blur code of its own: a card whose main photo is a blur fails own_photo_url and is skipped.
    W1-S-FATES tightens own_photo_url to users/{uid}/photos/; this test runs with that rule."""
    real_rule = face_live.own_photo_url

    def tightened(url, uid, hosts=None):
        return real_rule(url, uid, hosts) and f"users/{uid}/photos/" in unquote(urlparse(url).path)

    monkeypatch.setattr(face_live, "own_photo_url", tightened)
    db = Db({
        "publicProfiles/me": {"gender": "Female", "isDiscoverable": True},
        "publicProfiles/open": {"gender": "Female", "isDiscoverable": True, "photos": [link("users/open/photos/1.jpg")]},
        "publicProfiles/private": {"gender": "Female", "isDiscoverable": True,
                                   "photos": [link("users/private/blur/" + "a" * 40 + ".jpg")]},
        "publicProfiles/variant": {"gender": "Female", "isDiscoverable": True,
                                   "photos": [link("users/variant/variants/" + "b" * 40 + "_480.jpg")]},
        "publicProfiles/man": {"gender": "Male", "isDiscoverable": True, "photos": [link("users/man/photos/1.jpg")]},
    })
    downloaded = []
    monkeypatch.setattr(main, "firestore_client", db)
    monkeypatch.setattr(main, "_download_image_bytes", lambda url, max_mb=10: downloaded.append(url) or b"img")
    monkeypatch.setattr(face_match, "decode", lambda data: data)
    monkeypatch.setattr(face_match, "face_features", lambda img: [[1.0, 0.0]])
    main._cohort_cache.clear()
    try:
        faces = main._cohort("me")
    finally:
        main._cohort_cache.clear()
    assert downloaded == [link("users/open/photos/1.jpg")]
    assert faces == [[1.0, 0.0]]
