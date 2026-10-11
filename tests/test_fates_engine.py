import time

import fates_engine as fe

NOW = int(time.time() * 1000)
DAY = 86_400_000


def person(uid, gender="female", looking="male", **kw):
    base = dict(
        uid=uid, first_name=uid.title(), age=26, gender=gender, looking_for=looking, city="Bengaluru",
        lat=12.97, lon=77.59, photos=["https://x/p.jpg"], discoverable=True, last_active_ms=NOW - DAY // 2,
        intent="serious", religion="Hindu", community="Kannada", want_children="Yes",
        smoking="Never", drinking="Socially", languages=["Hindi", "English"], interests=["Trekking", "Coffee"],
        face_verified=True,
    )
    base.update(kw)
    return fe.Person(**base)


def viewer(**kw):
    return person("arjun", gender="male", looking="female", **kw)


def stars_match(total, doshas=()):
    return {"total": total, "kootas": [{"name": "Nadi", "points": 8, "max": 8}], "doshas": list(doshas)}


# ---------------- normalisation

def test_intent_normalisation_matches_app_values():
    assert fe.norm_intent("Serious long-term partner") == "serious"
    assert fe.norm_intent("Long-term or short-term, open to both") == "both"
    assert fe.norm_intent("Dating, see where it goes") == "dating"
    assert fe.norm_intent("Friends / social circle") == "friends"
    assert fe.norm_intent("Not sure yet") == "unsure"
    assert fe.norm_intent("") == ""


def test_looking_for_values():
    assert fe.norm_looking_for("Any") == "any"
    assert fe.norm_looking_for("Everyone") == "any"
    assert fe.norm_looking_for("Female") == "female"
    assert fe.norm_looking_for("Women only") == "female"
    assert fe.norm_looking_for("Men only") == "male"


def test_person_from_docs_reads_tests_and_private_fields():
    pub = {"firstName": "Anaya Rao", "gender": "Female", "lookingForGender": "Male", "isDiscoverable": True,
           "photos": ["https://a/1.jpg", "not-a-url"], "tests_attachment_closeness_scorePercent": 61, "lat": 12.9, "lon": 77.6}
    priv = {"settings_wantChildren": "Yes", "settings_ageMin": 24, "settings_ageMax": 32, "birthDate": "1999-05-02",
            "fatesPrefs": {"stars": "lot", "useTests": True}, "faceVerifiedPhotos": ["https://a/1.jpg"]}
    usr = {"lastAppOpenAtUtcIso": "2026-09-30T08:00:00Z", "isFaceVerified": True}
    p = fe.person_from_docs("u1", pub, priv, usr)
    assert p.first_name == "Anaya"
    assert p.gender == "female" and p.looking_for == "male"
    assert p.photos == ["https://a/1.jpg"]
    assert p.tests == {"attachment_closeness": 61}
    assert p.age_min == 24 and p.age_max == 32
    assert p.face_verified and p.stars_pref == "lot"
    assert p.last_active_ms is not None


# ---------------- hard filters

def test_mutual_gender_and_age():
    v = viewer(age=28, age_min=22, age_max=30)
    assert fe.hard_filter(v, person("a", age=26), NOW) is None
    assert fe.hard_filter(v, person("b", age=35), NOW) == "age"
    assert fe.hard_filter(v, person("c", looking="female"), NOW) == "gender"
    assert fe.hard_filter(v, person("d", age=26, age_min=30, age_max=40), NOW) == "age"


def test_inactive_friends_religion_children_distance():
    v = viewer(require_religion=True, radius_km=20)
    assert fe.hard_filter(v, person("a", last_active_ms=NOW - 30 * DAY), NOW) == "inactive"
    assert fe.hard_filter(v, person("b", intent="friends"), NOW) == "intent"
    assert fe.hard_filter(v, person("c", religion="Muslim"), NOW) == "religion"
    assert fe.hard_filter(v, person("d", want_children="No"), NOW) == "children"
    assert fe.hard_filter(v, person("e", lat=13.5, lon=77.6), NOW) == "distance"   # ~59 km
    assert fe.hard_filter(v, person("f", photos=[]), NOW) == "no_photos"
    assert fe.hard_filter(v, person("g", discoverable=False), NOW) == "not_discoverable"


def test_unknown_children_is_not_a_clash():
    v = viewer(want_children="Prefer not to say")
    assert fe.hard_filter(v, person("a", want_children="No"), NOW) is None
    v2 = viewer(want_children="Only if we are a perfect match")
    assert fe.hard_filter(v2, person("a", want_children="No"), NOW) is None


# ---------------- components

def _tid(category):
    return next(t for t, m in fe.tests_catalog()["tests"].items() if m["category"] == category)


def test_heart_rows_and_verdicts():
    att, love, conf = _tid("attachment"), _tid("love_style"), _tid("conflict")
    v = viewer(tests={att: 50, love: 20, conf: 30, "not_a_test": 10})
    c = person("a", tests={att: 60, love: 75, conf: 90, "not_a_test": 90})
    h = fe.heart_component(v, c)
    by = {r["category"]: r for r in h["rows"]}
    assert set(by) == {"attachment", "love_style", "conflict"}          # unknown ids are ignored
    assert by["attachment"]["verdict"] == "alike"
    assert by["love_style"]["verdict"] == "complete"                    # a different love style can complete
    assert by["conflict"]["verdict"] == "differs"
    assert [r["category"] for r in h["rows"]] == ["attachment", "love_style", "conflict"]
    assert by["attachment"]["lowPole"] and by["attachment"]["highPole"]
    assert "attachment" not in h["missing"] and "values" in h["missing"]
    assert h["score"] == round(100 * (0.9 + 0.8 + 0.4) / 3)


def test_heart_uses_real_catalog_ids():
    cat = fe.tests_catalog()["tests"]
    tid = next(t for t, m in cat.items() if m["category"] == "attachment")
    v = viewer(tests={tid: 50})
    c = person("a", tests={tid: 58})
    h = fe.heart_component(v, c)
    assert h["rows"][0]["verdict"] == "alike" and h["score"] >= 90
    c2 = person("b", tests={tid: 99})
    assert fe.heart_component(v, c2)["rows"][0]["verdict"] == "differs"


def test_heart_respects_opt_out():
    cat = fe.tests_catalog()["tests"]
    tid = next(iter(cat))
    v = viewer(tests={tid: 50})
    c = person("a", tests={tid: 50}, use_tests=False)
    assert fe.heart_component(v, c)["score"] is None


def test_home_statuses():
    v = viewer(relocate="No", drinking="Never")
    c = person("a", relocate="Yes", community="Tamil", languages=["Tamil", "English"], drinking="Regularly")
    h = fe.home_component(v, c)
    st = {r["key"]: r["status"] for r in h["rows"]}
    assert st["marriage"] == "aligns" and st["children"] == "aligns" and st["faith"] == "aligns"
    assert st["community"] == "talk" and st["moving"] == "aligns" and st["languages"] == "aligns"
    assert st["drinking"] == "talk"
    assert h["score"] is not None and 0 <= h["score"] <= 100


def test_home_unknown_when_too_little_known():
    v = fe.Person(uid="v", gender="male", looking_for="female")
    c = fe.Person(uid="c", gender="female", looking_for="male")
    assert fe.home_component(v, c)["score"] is None


def test_everyday_shared_interests():
    e = fe.everyday_component(viewer(interests=["Trekking", "Cricket"]), person("a", interests=["trekking", "Yoga"]))
    assert e["shared"] == ["Trekking"] and e["onlyYou"] == ["Cricket"] and e["onlyThem"] == ["Yoga"]


def test_stars_score_penalises_uncancelled_dosha():
    assert fe.stars_score(stars_match(27)) == 75
    assert fe.stars_score(stars_match(27, [{"name": "Nadi", "present": True, "cancelled": False}])) == 63
    assert fe.stars_score(stars_match(27, [{"name": "Bhakoot", "present": True, "cancelled": True}])) == 75
    assert fe.stars_score(None) is None


def test_fate_weights_renormalise_missing_parts():
    assert fe.fate_score({"stars": None, "heart": 80, "home": 80, "everyday": 80}, "little") == 80
    assert fe.fate_score({"stars": 100, "heart": 50, "home": 50, "everyday": 50}, "none") == 50
    assert fe.fate_score({}, "little") is None


# ---------------- selection

def test_paths_get_distinct_people_and_respect_eligibility():
    cat = fe.tests_catalog()["tests"]
    tid = next(t for t, m in cat.items() if m["category"] == "values")
    v = viewer(tests={tid: 40})
    a = person("a", tests={tid: 42})              # heart + home
    b = person("b")                                # home only
    c = person("c", tests={tid: 45})
    def mfn(x, y):
        return stars_match(30 if y.uid == "c" else 12)
    v.chart = {"x": 1}
    for p in (a, b, c):
        p.chart = {"x": 1}
    pairs = [fe.evaluate_pair(v, p, NOW, mfn) for p in (a, b, c)]
    short = fe.shortlist(v, pairs)
    assert [p.cand.uid for p in short["stars"]] == ["c"]           # 12 gunas is below the minimum
    chosen = fe.resolve_paths(short)
    uids = [p.cand.uid for p in chosen.values()]
    assert len(uids) == len(set(uids))
    assert chosen["stars"].cand.uid == "c"


def test_ai_choice_is_honoured_when_possible():
    v = viewer()
    pairs = [fe.evaluate_pair(v, person(u), NOW) for u in ("a", "b", "c")]
    short = fe.shortlist(v, pairs)
    assert short["home"]
    first = short["home"][0].cand.uid
    chosen = fe.resolve_paths({"home": short["home"]}, {"home": len(short["home"]) - 1})
    assert chosen["home"].cand.uid == short["home"][-1].cand.uid
    assert fe.resolve_paths({"home": short["home"]}, {"home": 99})["home"].cand.uid == first


def test_stars_path_skipped_when_viewer_does_not_care():
    v = viewer(stars_pref="none")
    v.chart = {"x": 1}
    c = person("a")
    c.chart = {"x": 1}
    pairs = [fe.evaluate_pair(v, c, NOW, lambda x, y: stars_match(33))]
    assert fe.shortlist(v, pairs)["stars"] == []


def test_weak_pairs_are_never_brought():
    v = viewer(intent="serious", want_children="Yes", religion="Hindu", smoking="Never", languages=["Hindi"])
    c = person("a", intent="dating", religion="Christian", community="Goan", smoking="Regularly",
               drinking="Regularly", languages=["Konkani"], relocate="No", interests=["Gaming"])
    pairs = [fe.evaluate_pair(v, c, NOW)]
    assert pairs[0].fate is not None and pairs[0].fate < fe.MIN_FATE_SCORE
    assert all(not v for v in fe.shortlist(v, pairs).values())


def test_teaser_never_contains_name_or_photo():
    p = fe.evaluate_pair(viewer(), person("anaya"), NOW)
    t = fe.teaser("home", p)
    assert "Anaya" not in str(t) and "http" not in str(t)


def test_pronouns():
    assert fe.pronoun(person("a"))["subj"] == "she"
    assert fe.pronoun(person("b", gender="male"))["obj"] == "him"
    assert fe.pronoun(fe.Person(uid="c"))["subj"] == "they"


# ---------------- learning

def test_learning_is_bounded_and_reversible():
    l = {}
    for _ in range(20):
        l = fe.learn(l, "skip", "home", "too_far")
    assert l["farPenaltyPerKm"] == 0.6                      # clamped
    assert l["skipReasons"]["too_far"] == 20
    l = fe.learn(l, "accept", "home")
    assert l["farPenaltyPerKm"] == 0.56
    l = fe.learn(l, "skip", "home", "nonsense")
    assert l["skipReasons"]["other"] == 1


def test_open_counts_reorder_paths():
    l = fe.learn({}, "open", "home")
    l = fe.learn(l, "open", "home")
    l = fe.learn(l, "open", "heart")
    assert fe.preferred_path_order(l) == ["home", "heart", "stars"]
    assert fe.preferred_path_order({}) == ["stars", "heart", "home"]


def test_far_penalty_prefers_closer_people():
    v = viewer()
    near = fe.evaluate_pair(v, person("near", lat=12.98, lon=77.60), NOW)
    far = fe.evaluate_pair(v, person("far", lat=13.25, lon=77.60), NOW)      # ~31 km
    learned = {"farPenaltyPerKm": 0.6}
    assert fe.rank_key(near, "home", learned) > fe.rank_key(far, "home", learned)
    assert abs(fe.rank_key(near, "home") - fe.rank_key(far, "home")) < 1.0   # no penalty without learning


def test_template_hooks_read_naturally():
    att, love = _tid("attachment"), _tid("love_style")
    v = viewer(tests={att: 50, love: 20})
    c = person("anaya", tests={att: 58, love: 80}, community="Tamil")
    p = fe.evaluate_pair(v, c, NOW)          # no charts here, so the Moon reading is set by hand
    p.stars = {"total": 27, "totalMin": 27, "totalMax": 27, "doshas": [],
               "kootas": [{"name": "Varna", "points": 1, "max": 1}, {"name": "Gana", "points": 6, "max": 6},
                          {"name": "Nadi", "points": 0, "max": 8}]}
    assert fe.template_hook("stars", p) == "27 of 36 gunas, and your temperaments match."
    p.stars = {**p.stars, "totalMin": 22, "totalMax": 29}
    assert fe.template_hook("stars", p).startswith("22–29 of 36 gunas")
    p.stars = {**p.stars, "kootas": [{"name": "Varna", "points": 1, "max": 1}]}
    assert fe.template_hook("stars", p).endswith("of 36 gunas between you.")
    assert fe.template_hook("heart", p) == "You feel safe in love the same way."
    home = fe.template_hook("home", p)
    assert home == "You agree on marriage plans and children."
    for path in ("stars", "heart", "home"):
        assert "Anaya" not in fe.template_hook(path, p)


def test_home_hook_mentions_a_shared_language_and_habits_once():
    v = viewer(intent="", want_children="", religion="", community="")
    c = person("a", intent="", want_children="", religion="", community="")
    p = fe.evaluate_pair(v, c, NOW)
    assert fe.template_hook("home", p) == "You agree on habits and share a language."


def test_template_why_gives_the_whole_picture_this_path_first():
    att = _tid("attachment")
    v = viewer(tests={att: 50}, interests=["Trekking", "Bollywood", "Coffee"])
    c = person("anaya", tests={att: 58}, interests=["Bollywood", "Trekking"])
    p = fe.evaluate_pair(v, c, NOW)
    p.stars = {"total": 25.5, "kootas": [], "doshas": []}
    why = fe.template_why("heart", p)
    assert why.startswith("Hearts 92% in tune, 25.5 of 36 gunas and ")
    assert "life answers alike." in why
    assert why.endswith("You both love trekking and Bollywood.")
    assert "Anaya" not in why
    assert fe.template_why("stars", p).startswith("25.5 of 36 gunas, hearts")


def test_template_why_leaves_out_low_gunas():
    att = _tid("attachment")
    p = fe.evaluate_pair(viewer(tests={att: 50}), person("priya", tests={att: 54}), NOW)
    p.stars = {"total": 14.5, "kootas": [], "doshas": []}
    why = fe.template_why("heart", p)
    assert "gunas" not in why and why.startswith("Hearts 96% in tune")



def test_a_chooser_gets_a_softer_bar_and_their_best_path_first():
    att = _tid("attachment")
    v = viewer(tests={att: 50})
    strong = fe.evaluate_pair(v, person("strong", tests={att: 52}), NOW)
    chooser = fe.evaluate_pair(v, person("chooser", tests={att: 58}, religion="", community="", want_children=""), NOW)
    chooser.fate = 48                      # below the normal bar, above the chooser bar
    assert not fe._eligible_for_path(v, chooser, "heart")
    chooser.chose_you = True
    assert fe._eligible_for_path(v, chooser, "heart")
    short = fe.shortlist(v, [strong])
    path = fe.place_chooser(v, short, chooser)
    best = max((k for k in fe.PATHS if fe._eligible_for_path(v, chooser, k)), key=lambda k: chooser.part(k) or 0)
    assert path == best and short[path][0].cand.uid == "chooser"
    assert all(p.cand.uid != "chooser" for k, opts in short.items() if k != path for p in opts)
    # generate() pins the chooser by forcing the choice for that path to 0.
    assert fe.resolve_paths(short, {path: 0})[path].cand.uid == "chooser"
    assert fe.teaser(path, chooser)["choseYou"] is True


def test_a_chooser_who_fits_no_path_is_not_placed():
    v = viewer()
    c = fe.evaluate_pair(v, person("c"), NOW)
    c.chose_you = True
    c.fate = 20
    short = {"stars": [], "heart": [], "home": []}
    assert fe.place_chooser(v, short, c) is None and not any(short.values())



def test_only_verified_people_are_brought():
    v = viewer()
    assert fe.hard_filter(v, person("ok"), NOW) is None
    assert fe.hard_filter(v, person("x", face_verified=False), NOW) == "not_verified"


def test_the_apps_own_verification_copy_is_not_trusted():
    pub = {"uid": "a", "gender": "Female", "isDiscoverable": True, "photos": ["https://x/a.jpg"]}
    checked = {"faceVerifiedPhotos": ["https://x/a.jpg"]}
    assert fe.person_from_docs("a", pub, {"faceVerified": True, **checked}).face_verified
    assert fe.person_from_docs("a", pub, checked, {"isFaceVerified": True}).face_verified
    assert not fe.person_from_docs("a", pub, {"faceVerificationPassed": True, **checked}, {"faceVerificationPassed": True}).face_verified


def test_a_new_photo_pauses_verification_until_the_server_rechecks():
    checked = {"faceVerified": True, "faceVerifiedPhotos": ["https://x/a.jpg", "https://x/b.jpg"]}
    pub = {"uid": "a", "gender": "Female", "isDiscoverable": True}
    assert fe.person_from_docs("a", {**pub, "photos": ["https://x/b.jpg", "https://x/a.jpg"]}, checked).face_verified  # reordered
    assert fe.person_from_docs("a", {**pub, "photos": ["https://x/a.jpg"]}, checked).face_verified                    # one removed
    assert not fe.person_from_docs("a", {**pub, "photos": ["https://x/a.jpg", "https://x/new.jpg"]}, checked).face_verified
    assert not fe.person_from_docs("a", {**pub, "photos": ["https://x/a.jpg"]}, {"faceVerified": True}).face_verified  # never checked


def test_a_boost_buys_reach_never_fit():
    from datetime import datetime, timedelta, timezone
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    pub = {"uid": "b", "gender": "Female", "isDiscoverable": True, "photos": ["https://x/b.jpg"], "boostActiveUntilUtcIso": soon}
    assert fe.person_from_docs("b", pub).boosted
    assert not fe.person_from_docs("b", {**pub, "boostActiveUntilUtcIso": "2020-01-01T00:00:00+00:00"}).boosted

    att = _tid("attachment")
    v = viewer(tests={att: 50})
    plain = fe.evaluate_pair(v, person("plain", tests={att: 52}), NOW)
    paid = fe.evaluate_pair(v, person("paid", tests={att: 52}, boosted=True), NOW)
    assert paid.fate == plain.fate                                   # the number is the same
    assert fe.rank_key(paid, "heart") > fe.rank_key(plain, "heart")  # the order among equals is not
    weak = fe.evaluate_pair(v, person("weak", tests={att: 99}, boosted=True, religion="", community="", want_children=""), NOW)
    weak.fate = fe.MIN_FATE_SCORE - 1
    assert not fe._eligible_for_path(v, weak, "heart")               # a boost never lowers the bar
    assert fe.exposure_cap(paid.cand, 8) == 16 and fe.exposure_cap(plain.cand, 8) == 8


def test_at_most_one_promoted_person_a_day():
    att = _tid("attachment")
    v = viewer(tests={att: 50})
    a = fe.evaluate_pair(v, person("a", tests={att: 50}, boosted=True), NOW)
    b = fe.evaluate_pair(v, person("b", tests={att: 50}, boosted=True), NOW)
    c = fe.evaluate_pair(v, person("c", tests={att: 50}), NOW)
    short = {"stars": [], "heart": [a, b], "home": [b, c]}
    got = fe.resolve_paths(short)
    uids = [p.cand.uid for p in got.values()]
    assert sum(1 for p in got.values() if fe.promoted(p)) == 1 and "c" in uids
    b.chose_you = True                                               # someone who chose you is never "promoted"
    got = fe.resolve_paths({"stars": [], "heart": [a], "home": [b]})
    assert {p.cand.uid for p in got.values()} == {"a", "b"}



def test_distance_is_approximate_like_the_app():
    assert fe.approx_distance(0.4, True) == "<2 km"
    assert fe.approx_distance(3.6, True) == "~4 km"
    assert fe.approx_distance(13, True) == "~15 km"
    assert fe.approx_distance(846, False) == "~850 km"
    assert fe.approx_distance(1349, True) == "~1,300 km"
    assert fe.approx_distance(12, False) == ""            # a side picked a city by hand: no made-up number
    assert fe.approx_distance(5, True, "mi") == "~3 mi"
    assert fe.approx_distance(None, True) == ""
    assert fe.precise_geo_source("gps_auto") and fe.precise_geo_source("lastKnown") and not fe.precise_geo_source("manual_geocode")


def test_teasers_never_carry_an_exact_distance():
    v = viewer(geo_precise=True)
    p = fe.evaluate_pair(v, person("a", lat=12.99, lon=77.61, geo_precise=True), NOW)
    t = fe.teaser("home", p, v)
    assert "km" not in t and t["distance"].startswith(("<", "~"))
    hidden = viewer(geo_precise=True, show_distance=False)
    assert fe.teaser("home", fe.evaluate_pair(hidden, person("a"), NOW), hidden)["distance"] == ""


# ======================================================================================================================
# 1.0.8: photo privacy, pools (plan §2 C1), rishta mode and partner preferences (§2 C11)
# ======================================================================================================================
from urllib.parse import quote  # noqa: E402

import pytest  # noqa: E402

TODAY = "2026-10-11"


def storage_link(uid, folder, name):
    return f"https://firebasestorage.googleapis.com/v0/b/app.appspot.com/o/{quote(f'users/{uid}/{folder}/{name}', safe='')}?alt=media&token=t"


# ---------------- photo privacy (features W1-S2)

def test_a_blurred_card_brings_no_photos_and_a_private_viewer_stays_verified():
    real = [storage_link("u1", "photos", "a.jpg"), storage_link("u1", "photos", "b.jpg")]
    blur = [storage_link("u1", "blur", "a.jpg"), storage_link("u1", "blur", "b.jpg")]
    pub = {"gender": "Female", "isDiscoverable": True, "photos": blur, "photoPrivacy": "accepted"}
    priv = {"photos": real, "faceVerified": True, "faceVerifiedPhotos": real}
    p = fe.person_from_docs("u1", pub, priv)
    assert p.photos == [] and p.face_verified                       # a private viewer keeps Daily Fates
    assert fe.hard_filter(viewer(), p, NOW) == "no_photos"          # and is never a candidate: nothing is revealed
    # Blur links alone give it away (both URL kinds), even before photoPrivacy reaches the card.
    assert fe.person_from_docs("u1", {**pub, "photoPrivacy": "all"}, priv).photos == []
    cdn = {**pub, "photoPrivacy": "", "photos": ["https://photos.shaadiparrot.com/users/u1/blur/a.jpg"]}
    assert fe.person_from_docs("u1", cdn, priv).photos == []
    # Private photos are checked against the real links, never a blur.
    assert not fe.person_from_docs("u1", pub, {**priv, "photos": blur, "faceVerifiedPhotos": blur}).face_verified
    assert not fe.person_from_docs("u1", pub, {**priv, "photos": real + [storage_link("u1", "photos", "new.jpg")]}).face_verified
    # Public photos: the card's links for both, as before.
    shown = fe.person_from_docs("u1", {**pub, "photos": real, "photoPrivacy": "all"}, priv)
    assert shown.photos == real and shown.face_verified


def test_someone_elses_blur_link_does_not_hide_a_card():
    pub = {"gender": "Female", "isDiscoverable": True, "photos": ["https://x/a.jpg", storage_link("u2", "blur", "b.jpg")]}
    assert fe.person_from_docs("u1", pub).photos == pub["photos"]


# ---------------- pools: the truth table shared with feed.ts poolsOf

RISHTA_W = {"gender": "Female", "feedRishta": True, "feedRishtaFromDay": "2026-01-01", "feedQuick": False, "feedQuickFromDay": "2026-01-01"}
GROOM_20 = {"gender": "Male", "feedRishta": True, "feedRishtaFromDay": "2027-03-01", "feedQuick": False, "feedQuickFromDay": "2027-03-01"}

POOL_VECTORS = [
    # (case, card, today, quickMaleMinAge, inRishta, inQuick)
    ("groom of 20 with intent waits for his 21st birthday", GROOM_20, TODAY, 21, False, False),
    ("and joins on it with no rewrite", GROOM_20, "2027-03-01", 21, True, False),
    ("bride with intent", RISHTA_W, TODAY, 21, True, False),
    ("intent and quickSwipeOk: both pools", {**RISHTA_W, "feedQuick": True}, TODAY, 21, True, True),
    ("family-made profile: never quick", {**RISHTA_W, "profileFor": "daughter"}, TODAY, 21, True, False),
    ("legacy card: quick only", {"gender": "Female", "age": 25}, TODAY, 21, False, True),
    ("legacy man of 20, quick from 21", {"gender": "Male", "age": 20}, TODAY, 21, False, False),
    ("legacy man of 20, quick from 18", {"gender": "Male", "age": 20}, TODAY, 18, False, True),
    ("legacy man a day before 21 (birth date wins over age)", {"gender": "Male", "birthDate": "2005-10-12", "age": 30}, TODAY, 21, False, False),
    ("legacy man on his 21st birthday", {"gender": "Male", "birthDate": "2005-10-11"}, TODAY, 21, False, True),
    ("legacy family-made card", {"gender": "Female", "age": 25, "profileFor": "daughter"}, TODAY, 21, False, False),
    ("legacy self card", {"gender": "Female", "age": 25, "profileFor": "self"}, TODAY, 21, False, True),
    ("legacy card marked for marriage", {"gender": "Female", "age": 25, "marriageIntent": True}, TODAY, 21, False, False),
    ("a client-written marriageIntent never enters rishta", {"gender": "Female", "age": 25, "marriageIntent": True, "feedQuick": True, "feedQuickFromDay": ""}, TODAY, 21, False, True),
    ("feedRishta without a day", {"gender": "Female", "feedRishta": True, "feedRishtaFromDay": "", "feedQuick": False}, TODAY, 21, False, False),
    ("feedRishta written as a string", {"gender": "Female", "feedRishta": "true", "feedRishtaFromDay": "2020-01-01", "feedQuick": False}, TODAY, 21, False, False),
    ("legacy woman without an age counts as 18", {"gender": "Female"}, TODAY, 21, False, True),
    ("legacy man without an age counts as 18", {"gender": "Male"}, TODAY, 21, False, False),
    ("legacy card under 18", {"gender": "Female", "age": 17}, TODAY, 21, False, False),
    ("quick from a later day", {"gender": "Male", "feedQuick": True, "feedQuickFromDay": "2027-01-01"}, TODAY, 21, False, False),
    ("quick with an unknown birth day", {"gender": "Female", "feedQuick": True, "feedQuickFromDay": ""}, TODAY, 21, False, True),
    ("indexed as not quick", {"gender": "Female", "age": 30, "feedQuick": False, "feedQuickFromDay": ""}, TODAY, 21, False, False),
    # C1 `todayDay >= feedQuickFromDay`: a missing or null day compares false in JS, so the age gate stays shut.
    ("quick without feedQuickFromDay", {"gender": "Male", "age": 19, "feedQuick": True}, TODAY, 21, False, False),
    ("quick with a null feedQuickFromDay", {"gender": "Male", "age": 19, "feedQuick": True, "feedQuickFromDay": None}, TODAY, 21, False, False),
]


@pytest.mark.parametrize("case,card,today,qmin,rishta,quick", POOL_VECTORS, ids=[v[0] for v in POOL_VECTORS])
def test_pools_of_truth_table(case, card, today, qmin, rishta, quick):
    assert fe.pools_of(card, today, qmin) == {"inRishta": rishta, "inQuick": quick}


def test_pools_default_to_men_from_21():
    assert fe.QUICK_MALE_MIN_AGE == 21
    assert fe.pools_of({"gender": "Male", "age": 20}, TODAY) == {"inRishta": False, "inQuick": False}
    assert fe.pools_of({"gender": "Male", "age": 20}, TODAY, 17)["inQuick"]       # never below 18
    assert not fe.pools_of({"gender": "Male", "age": 17}, TODAY, 17)["inQuick"]
    assert fe.pools_of(None, TODAY) == {"inRishta": False, "inQuick": True}      # an empty card: legacy rule, age 18


def test_card_age_like_the_feed():
    assert fe.card_age({"birthDate": "2000-10-12"}, TODAY) == 25
    assert fe.card_age({"birthDate": "2000-10-11T00:00:00Z"}, TODAY) == 26
    assert fe.card_age({"age": "30"}, TODAY) == 30
    assert fe.card_age({}, TODAY) == 18
    assert fe.card_age({"age": 16}, TODAY) == 0
    assert fe.card_age({"birthDate": "garbage", "age": 22}, TODAY) == 22
    assert fe.card_age({"birthDate": "2030-01-01", "age": 22}, TODAY) == 22         # a future date is unusable


# ---------------- viewer rules

SELF_GROOM = {"profileFor": "self", "marriageIntent": True, "gender": "male", "birthDate": "2000-01-01",
              "intentConsentAtIso": "2026-10-01T00:00:00Z"}


def test_rishta_mode_viewer_needs_intent_and_a_groom_needs_21():
    assert fe.rishta_viewer_block(SELF_GROOM, {}, TODAY) is None
    assert fe.rishta_viewer_block({**SELF_GROOM, "marriageIntent": False}, {}, TODAY) == "needs_intent"
    assert fe.rishta_viewer_block({**SELF_GROOM, "marriageIntent": "true"}, {}, TODAY) == "needs_intent"   # a real bool only
    assert fe.rishta_viewer_block({k: v for k, v in SELF_GROOM.items() if k != "marriageIntent"},
                                  {"marriageIntent": True}, TODAY) == "needs_intent"   # never the card copy
    assert fe.rishta_viewer_block({**SELF_GROOM, "profileFor": "none"}, {}, TODAY) == "needs_intent"
    assert fe.rishta_viewer_block({**SELF_GROOM, "birthDate": "2005-10-12"}, {}, TODAY) == "groom_age"     # a day short
    assert fe.rishta_viewer_block({**SELF_GROOM, "birthDate": "2005-10-11"}, {}, TODAY) is None           # 21 today
    assert fe.rishta_viewer_block({**SELF_GROOM, "birthDate": ""}, {}, TODAY) == "groom_age"
    assert fe.rishta_viewer_block({**SELF_GROOM, "profileFor": "", "gender": "", "birthDate": "2006-01-01"},
                                  {"gender": "Male"}, TODAY) == "groom_age"     # gender from the card when the profile has none
    assert fe.rishta_viewer_block({**SELF_GROOM, "gender": "female", "birthDate": "2008-01-01"}, {}, TODAY) is None
    assert fe.rishta_viewer_block({**SELF_GROOM, "profileFor": "son", "birthDate": "2006-01-01"}, {}, TODAY) is None


def test_dating_fates_are_for_ones_own_profile_only():
    assert fe.quick_viewer_block({}) is None and fe.quick_viewer_block({"profileFor": "self"}) is None
    assert fe.quick_viewer_block({"profileFor": "son"}) == "no_quick"
    assert fe.quick_viewer_block({"profileFor": "none"}) == "no_quick"


# ---------------- candidates per mode

RISHTA_PRIV = {"marriageIntent": True, "intentConsentAtIso": "2026-10-01T00:00:00Z", "profileFor": "self"}


def rishta_viewer(**kw):
    v = viewer(**kw)
    v.rishta = True
    return v


def test_a_dating_viewer_never_gets_a_family_made_or_marriage_only_profile():
    v = viewer()
    legacy = {"gender": "Female", "age": 26}
    assert fe.candidate_gate(v, person("a"), legacy, {}, TODAY) is None
    # The card index says so...
    assert fe.candidate_gate(v, person("b"), RISHTA_W, RISHTA_PRIV, TODAY) == "pool"
    managed = {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00Z", "profileFor": "daughter"}
    assert fe.candidate_gate(v, person("c"), {**RISHTA_W, "profileFor": "daughter"}, managed, TODAY) == "pool"
    # ...and so do the private facts where an old or stale card doesn't.
    assert fe.candidate_gate(v, person("d"), legacy, {"profileFor": "daughter"}, TODAY) == "pool"
    assert fe.candidate_gate(v, person("e"), legacy, {"marriageIntent": True}, TODAY) == "pool"
    quick_card = {"gender": "Female", "feedQuick": True, "feedQuickFromDay": "2020-01-01"}
    assert fe.candidate_gate(v, person("f"), quick_card, {"marriageIntent": True}, TODAY) == "pool"
    assert fe.candidate_gate(v, person("g"), quick_card, {"profileFor": "son"}, TODAY) == "pool"
    # Marriage and Quick swipe both: fine.
    assert fe.candidate_gate(v, person("h"), {**RISHTA_W, "feedQuick": True}, {**RISHTA_PRIV, "quickSwipeOk": True}, TODAY) is None


def test_rishta_mode_brings_only_rishta_pool_people_who_still_want_marriage():
    v = rishta_viewer()
    assert fe.candidate_gate(v, person("a"), RISHTA_W, RISHTA_PRIV, TODAY) is None
    assert fe.candidate_gate(v, person("b"), {"gender": "Female", "age": 26}, {}, TODAY) == "pool"          # quick only
    assert fe.candidate_gate(v, person("c"), {**RISHTA_W, "feedRishtaFromDay": "2027-01-01"}, RISHTA_PRIV, TODAY) == "pool"
    assert fe.candidate_gate(v, person("d"), RISHTA_W, {**RISHTA_PRIV, "profileFor": "none"}, TODAY) == "kundli_only"
    assert fe.candidate_gate(v, person("e"), RISHTA_W, {**RISHTA_PRIV, "marriageIntent": False}, TODAY) == "pool"   # stale index
    assert fe.candidate_gate(v, person("f"), RISHTA_W, {"marriageIntent": True, "profileFor": "self"}, TODAY) == "pool"  # no consent
    # A parent's account carrying a daughter's profile keeps the parent's life stage: it never counts.
    managed = {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00Z", "profileFor": "daughter",
               "lifeStage": "MarriedKids"}
    assert fe.candidate_gate(v, person("g"), {**RISHTA_W, "profileFor": "daughter"}, managed, TODAY) is None
    # A card copy of marriageIntent never puts anyone in the rishta pool.
    assert fe.candidate_gate(v, person("h"), {"gender": "Female", "age": 26, "marriageIntent": True}, RISHTA_PRIV, TODAY) == "pool"


def test_life_stage_is_never_read():
    import inspect

    import fates_api
    for mod in (fe, fates_api):
        assert "lifeStage" not in inspect.getsource(mod)


# ---------------- partner preferences both ways (§2 C11)

CASTES = {"entries": [{"name": "Brahmin", "umbrella": "", "religions": ["Hindu"], "tongues": ["*"]},
                      {"name": "Brahmin - Iyer", "umbrella": "Brahmin", "religions": ["Hindu"], "tongues": ["Tamil"]},
                      {"name": "Bhumihar", "umbrella": "", "religions": ["Hindu"], "tongues": ["Hindi belt / North Indian"]}],
          "blocked": []}


@pytest.fixture
def caste_list():
    fe.load_caste_list(CASTES)
    yield
    fe._caste_state.update(loaded=False, umbrellas=None)


@pytest.fixture
def no_caste_list():
    fe.load_caste_list(None)
    yield
    fe._caste_state.update(loaded=False, umbrellas=None)


def rp(uid="c", prefs=None, **kw):
    """A person with partner preferences from a profiles doc."""
    p = person(uid, **kw)
    p.prefs = fe.prefs_from_doc(prefs or {})
    return p


def test_prefs_from_doc():
    p = fe.prefs_from_doc({})
    assert p["strict"] is None and p["childrenOk"] == "any" and p["manglik"] == "" and p["kundli"] == ""
    assert p["incomeMinLakh"] == 0 and p["educationMin"] == 0 and p["castes"] == []
    assert fe.prefs_from_doc({"pref_strict": []})["strict"] == []                       # saved with nothing marked
    q = fe.prefs_from_doc({"pref_manglik": "Match", "pref_kundli": "24", "pref_childrenOk": "none", "pref_incomeMinLakh": 12,
                           "pref_castes": ["Brahmin"], "pref_strict": ["religions", "diets"], "pref_educationMin": "3"})
    assert q["manglik"] == "match" and q["kundli"] == "24" and q["childrenOk"] == "none" and q["incomeMinLakh"] == 10
    assert q["castes"] == ["Brahmin"] and q["strict"] == ["religions", "diets"] and q["educationMin"] == 3
    assert fe.prefs_from_doc({"pref_kundli": "99", "pref_manglik": "yes"})["kundli"] == ""
    assert fe.prefs_from_doc({"pref_manglik": "yes"})["manglik"] == ""


def test_religion_and_marital_status_hide_by_default():
    v = rp("me", {"pref_religions": ["Hindu"], "pref_maritalStatuses": ["Never married"]})
    assert fe._pref_gate(v, rp(religion="Hindu", marital_status="Never married")) is None
    assert fe._pref_gate(v, rp(religion="Muslim")) == "pref_religions"
    assert fe._pref_gate(v, rp(religion="Prefer not to say")) is None                  # unknown never hides
    assert fe._pref_gate(v, rp(religion="")) is None
    assert fe._pref_gate(v, rp(religion="hindu")) is None
    assert fe._pref_gate(v, rp(marital_status="Divorced")) == "pref_maritalStatuses"


def test_community_food_and_education_only_rank_unless_marked():
    prefs = {"pref_communities": ["Marathi"], "pref_diets": ["Vegetarian"], "pref_educationMin": 4}
    c = rp(community="Gujarati", diet="Non-vegetarian", education="Bachelor's")
    assert fe._pref_gate(rp("me", prefs), c) is None
    assert set(fe.pref_misses(rp("me", prefs).prefs, c, "")) == {"communities", "diets", "education"}
    assert fe._pref_gate(rp("me", {**prefs, "pref_strict": ["communities"]}), c) == "pref_communities"
    assert fe._pref_gate(rp("me", {**prefs, "pref_strict": ["diets"]}), c) == "pref_diets"
    assert fe._pref_gate(rp("me", {**prefs, "pref_strict": ["education"]}), c) == "pref_education"
    assert fe._pref_gate(rp("me", {**prefs, "pref_strict": ["education"]}), rp(education="Master's")) is None
    assert fe._pref_gate(rp("me", {**prefs, "pref_strict": ["education"]}), rp(education="")) is None
    # Marking parts replaces the default: religion then only ranks.
    assert fe._pref_gate(rp("me", {"pref_religions": ["Hindu"], "pref_strict": ["diets"]}), rp(religion="Muslim")) is None


def test_caste_fits_like_the_app(caste_list):
    cases = [([], "Bhumihar", True), (["Brahmin"], "Brahmin - Iyer", True), (["brahmin"], "BRAHMIN", True),
             (["Brahmin"], "Bhumihar", False), (["Brahmin"], "Prefer not to say", True), (["Brahmin"], "", True),
             (["Brahmin"], None, True), (["Brahmin"], "Inter-caste / Mixed", True), (["Brahmin - Iyer"], "Brahmin", False),
             (["Brahmin"], "Brahmin - Vadama", True)]                     # a typed "X - Y" under a wanted umbrella
    for wanted, theirs, fits in cases:
        assert fe.caste_fits(wanted, theirs) is fits, (wanted, theirs)


def test_caste_preference_with_the_shared_list(caste_list):
    v = rp("me", {"pref_castes": ["Brahmin"]})
    assert fe._pref_gate(v, rp(caste="brahmin - iyer")) is None                          # the umbrella fits its communities
    assert fe._pref_gate(v, rp(caste="Bhumihar")) == "pref_castes"                       # a must-have once chosen
    for unknown in ("", "Prefer not to say", "Inter-caste / Mixed"):
        assert fe._pref_gate(v, rp(caste=unknown)) is None
    assert fe._pref_gate(rp("me", {"pref_castes": ["Brahmin - Iyer"]}), rp(caste="Brahmin")) == "pref_castes"
    assert fe._pref_gate(rp("me", {"pref_castes": []}), rp(caste="Bhumihar")) is None    # caste no bar
    assert fe._pref_gate(rp("me", {"pref_castes": ["Brahmin"], "pref_strict": []}), rp(caste="Bhumihar")) == "pref_castes"


def test_caste_preference_without_the_list_matches_exactly(no_caste_list):
    v = rp("me", {"pref_castes": ["Brahmin"]})
    assert fe.caste_umbrellas() is None
    assert fe._pref_gate(v, rp(caste="Brahmin")) is None
    assert fe._pref_gate(v, rp(caste="Brahmin - Iyer")) == "pref_castes"
    assert fe._pref_gate(v, rp(caste="Inter-caste / Mixed")) is None


def test_the_shared_caste_file_loads_when_it_is_there():
    import os
    fe._caste_state.update(loaded=False, umbrellas=None)
    try:
        got = fe.caste_umbrellas()
        assert (got is not None) == os.path.exists(fe._CASTE_LIST_PATH)
    finally:
        fe._caste_state.update(loaded=False, umbrellas=None)


def test_manglik_fits_every_pair():
    yes, no, part, dk = "Yes", "No", "Partial (Anshik)", "Don't know"
    assert fe.manglik_fits(yes, yes) and not fe.manglik_fits(yes, no) and fe.manglik_fits(yes, part)
    assert not fe.manglik_fits(no, yes) and fe.manglik_fits(no, no) and fe.manglik_fits(no, part)
    for other in (yes, no, part, dk, "", "Prefer not to say"):
        assert fe.manglik_fits(part, other) and fe.manglik_fits(dk, other)
        assert fe.manglik_fits(other, dk) and fe.manglik_fits(other, "")


def test_manglik_must_match_uses_my_own_manglik():
    v = rp("me", {"pref_manglik": "match"}, manglik="Yes")
    assert fe._pref_gate(v, rp(manglik="No")) == "pref_manglik"
    assert fe._pref_gate(v, rp(manglik="Partial (Anshik)")) is None
    assert fe._pref_gate(v, rp(manglik="Yes")) is None
    assert fe._pref_gate(v, rp(manglik="")) is None
    assert fe._pref_gate(rp("me", {"pref_manglik": "match"}, manglik=""), rp(manglik="No")) is None   # mine unknown: all fit
    assert fe._pref_gate(rp("me", {}, manglik="Yes"), rp(manglik="No")) is None                     # "Doesn't matter"


def test_income_minimum():
    v = rp("me", {"pref_incomeMinLakh": 12})
    assert v.prefs["incomeMinLakh"] == 10
    assert fe._pref_gate(v, rp(income="7-10L")) == "pref_income"
    for fine in ("10-15L", "1Cr+", "", "prefer_not", "junk"):
        assert fe._pref_gate(v, rp(income=fine)) is None, fine
    assert fe.normal_min_lakh(3) == 0 and fe.normal_min_lakh(100) == 100 and fe.normal_min_lakh(49) == 30
    assert fe.income_at_least("7-10L", 0) and not fe.income_at_least("0-2L", 4)


def test_income_is_the_public_card_value_only():
    priv = {"annualIncome": "1Cr+", "annualIncomeVisibility": "matches"}
    assert fe.person_from_docs("u", {"annualIncome": ""}, priv).income == ""
    assert fe.person_from_docs("u", {}, priv).income == ""
    assert fe.person_from_docs("u", {"annualIncome": "10-15L"}, {}).income == "10-15L"


def test_children_preference():
    v = rp("me", {"pref_childrenOk": "none"})
    assert fe._pref_gate(v, rp(children="Yes, living together")) == "pref_children"
    assert fe._pref_gate(v, rp(children="Yes, living separately")) == "pref_children"
    assert fe._pref_gate(v, rp(children="No children")) is None
    assert fe._pref_gate(v, rp(children="")) is None
    assert fe._pref_gate(rp("me", {"pref_childrenOk": "any"}), rp(children="Yes, living together")) is None


def test_work_only_ranks():
    v = rp("me", {"pref_professions": ["Doctor"], "pref_strict": ["professions"]})
    c = rp(profession="Engineer")
    assert "professions" in fe.pref_misses(v.prefs, c, "")
    assert fe._pref_gate(v, c) is None


def test_their_wishes_hide_only_when_they_saved_must_haves():
    me = rp("me", {}, religion="Hindu", caste="Bhumihar", marital_status="Divorced", diet="Non-vegetarian")
    jain_only = {"pref_religions": ["Jain"]}
    assert fe._pref_gate(me, rp("c", jain_only)) is None                                 # pref_strict never saved
    assert fe._pref_gate(me, rp("c", {**jain_only, "pref_strict": []})) is None          # saved, nothing marked
    assert fe._pref_gate(me, rp("c", {**jain_only, "pref_strict": ["religions"]})) == "their_pref_religions"
    assert fe._pref_gate(me, rp("c", {**jain_only, "pref_strict": ["diets"]})) is None
    assert fe._pref_gate(me, rp("c", {"pref_maritalStatuses": ["Never married"], "pref_strict": ["maritalStatuses"]})) \
        == "their_pref_maritalStatuses"
    assert fe._pref_gate(me, rp("c", {"pref_diets": ["Vegetarian"], "pref_strict": ["diets"]})) == "their_pref_diets"


def test_their_unmarkable_wishes_never_hide_my_profile():
    """Matching owner decision 3: only parts a family marked "Must have" hide their profile. Caste, Manglik, income and
    children have no toggle, so on their side they never hide anyone, whatever pref_strict holds."""
    me = rp("me", {}, religion="Hindu", caste="Bhumihar", manglik="Yes", income="0-2L", children="Yes, living together")
    wishes = {"pref_castes": ["Brahmin"], "pref_manglik": "match", "pref_incomeMinLakh": 20, "pref_childrenOk": "none"}
    for strict in (None, [], ["religions", "maritalStatuses", "diets", "communities", "education"],
                   ["castes", "manglik", "income", "children"]):                         # unmarkable tokens are ignored
        doc = wishes if strict is None else {**wishes, "pref_strict": strict}
        assert fe._pref_gate(me, rp("c", doc, manglik="No")) is None, strict
    # The same wishes are still must-haves on my own side.
    mine = rp("me", {"pref_castes": ["Brahmin"]}, caste="Bhumihar")
    assert fe._pref_gate(mine, rp("c", {}, caste="Rajput")) == "pref_castes"
    assert fe.marked_must_haves({"strict": None}) == set()
    assert fe.marked_must_haves({"strict": ["castes", "religions", "professions"]}) == {"religions"}


def test_my_must_haves_come_first_in_the_reason():
    v = rp("me", {"pref_religions": ["Jain"], "pref_childrenOk": "none"})
    assert fe._pref_gate(v, rp(religion="Hindu", children="Yes, living together")) == "pref_religions"
    assert fe.PREF_LABELS["professions"] == "Work" and fe.PREF_LABELS["communities"] == "Mother tongue"


# ---------------- rishta mode: kundli on every path

def charted(p):
    p.chart = {"x": 1}
    return p


def _tests_pair(v, c, total):
    return fe.evaluate_pair(v, c, NOW, (lambda a, b: stars_match(total)) if total is not None else None)


def test_rishta_mode_needs_both_charts_and_18_gunas_on_every_path():
    att = _tid("attachment")
    v = charted(rp("me", {}, tests={att: 50}))                       # Hindu: kundli "Best first" by default
    c = charted(rp("c", {}, tests={att: 52}))
    v.rishta = True
    low, ok = _tests_pair(v, c, 16), _tests_pair(v, c, 24)
    assert not any(fe._eligible_for_path(v, low, k) for k in fe.PATHS)
    assert all(fe._eligible_for_path(v, ok, k) for k in fe.PATHS)
    no_chart = fe.evaluate_pair(v, rp("n", {}, tests={att: 52}), NOW, lambda a, b: stars_match(30))
    assert no_chart.stars is None and not any(fe._eligible_for_path(v, no_chart, k) for k in fe.PATHS)
    # Dating (no mode) keeps today's rule: only the stars path needs the gunas.
    v.rishta = False
    assert fe._eligible_for_path(v, low, "heart") and not fe._eligible_for_path(v, low, "stars")


def test_kundli_modes_in_rishta_mode():
    att = _tid("attachment")
    c = charted(rp("c", {}, tests={att: 52}))

    def v_with(prefs, **kw):
        v = charted(rp("me", prefs, tests={att: 50}, **kw))
        v.rishta = True
        return v

    show = v_with({"pref_kundli": "show"})
    assert fe._eligible_for_path(show, _tests_pair(show, c, 12), "heart")               # "Just show the score"
    assert fe._eligible_for_path(show, _tests_pair(show, charted(rp("n", {}, tests={att: 52})), None), "heart")
    other_faith = v_with({}, religion="Christian")                                        # 'show' by default
    assert fe._eligible_for_path(other_faith, _tests_pair(other_faith, c, 12), "heart")
    strong = v_with({"pref_kundli": "24"})
    assert not fe._eligible_for_path(strong, _tests_pair(strong, c, 22), "heart")
    assert fe._eligible_for_path(strong, _tests_pair(strong, c, 24), "heart")
    # Their "24+ gunas only" holds whenever the gunas are known.
    picky = charted(rp("p", {"pref_kundli": "24"}, tests={att: 52}))
    assert not fe._eligible_for_path(show, _tests_pair(show, picky, 20), "heart")
    assert fe._eligible_for_path(show, _tests_pair(show, picky, 26), "heart")


def test_effective_kundli_defaults():
    for faith in ("Hindu", "Jain", "Sikh", "Buddhist", "", "Prefer not to say"):
        assert fe.effective_kundli(rp(religion=faith)) == "rank", faith
    for faith in ("Muslim", "Christian", "Parsi / Zoroastrian"):
        assert fe.effective_kundli(rp(religion=faith)) == "show", faith
    assert fe.effective_kundli(rp("c", {"pref_kundli": "18"}, religion="Muslim")) == "18"
    assert fe.rishta_min_gunas(rp("c", {"pref_kundli": "24"})) == 24
    assert fe.rishta_min_gunas(rp("c", {})) == 18 and fe.rishta_min_gunas(rp("c", {"pref_kundli": "show"})) is None


def test_rishta_ready_reads_the_preferences_page():
    v = fe.rishta_ready(rp("me", {"pref_ageMin": 24, "pref_ageMax": 30}), viewer=True)
    assert v.rishta and v.age_min == 24 and v.age_max == 30 and v.radius_km == 300.0      # "My state" by default
    assert fe.rishta_ready(rp("me", {"pref_scope": "near"}), viewer=True).radius_km == 50.0
    assert fe.rishta_ready(rp("me", {"pref_scope": "india"}), viewer=True).radius_km >= 3000
    c = fe.rishta_ready(rp("c", {"pref_ageMin": 30}, age_min=22))
    assert c.age_min == 30 and not c.rishta
    me = viewer(age=28)
    assert fe.hard_filter(me, c, NOW) == "age"                                             # their wish about my age


# ---------------- rishta mode: what families compare

def test_rishta_home_rows_are_scored():
    v = rp("me", {}, marital_status="Never married", diet="Vegetarian", family_values="Traditional", education="Master's")
    v.rishta = True
    c = rp("c", {}, marital_status="Never married", diet="Non-vegetarian", family_values="Liberal", education="Bachelor's")
    h = fe.home_component(v, c)
    st = {r["key"]: r["status"] for r in h["rows"]}
    assert st["marital"] == "aligns" and st["food"] == "differs" and st["family_values"] == "differs"
    assert st["education"] == "aligns" and h["counts"]["differs"] >= 2
    c2 = rp("c2", {}, marital_status="Divorced", diet="Eggetarian", family_values="Moderate", education="High school")
    st2 = {r["key"]: r["status"] for r in fe.home_component(v, c2)["rows"]}
    assert (st2["marital"], st2["food"], st2["family_values"], st2["education"]) == ("talk", "talk", "talk", "talk")
    c3 = rp("c3", {}, marital_status="Prefer not to say", diet="", family_values="", education="")
    st3 = {r["key"]: r["status"] for r in fe.home_component(v, c3)["rows"]}
    assert all(st3[k] is None for k in ("marital", "food", "family_values", "education"))
    assert fe.home_component(v, rp("c4", {}, diet="Jain food"))["rows"][-3]["status"] == "talk"
    # Facts for the AI carry the topic and status only, never the values.
    p = fe.evaluate_pair(v, c, NOW)
    facts = fe.facts_for_ai(v, "home", p)
    assert {"topic": "Food", "status": "differs"} in facts["home"]
    assert "Non-vegetarian" not in str(facts["home"])


def test_dating_picks_are_unchanged_by_matrimony_facts():
    v, c = viewer(), person("a")
    plain = fe.evaluate_pair(v, c, NOW)
    v2 = viewer(marital_status="Never married", diet="Vegetarian", family_values="Traditional", education="MBA")
    c2 = person("a", marital_status="Divorced", diet="Non-vegetarian", family_values="Liberal", education="Diploma",
                caste="Bhumihar", income="1Cr+", prefs=fe.prefs_from_doc({"pref_religions": ["Jain"], "pref_strict": ["religions"]}))
    rich = fe.evaluate_pair(v2, c2, NOW)
    assert rich.home == plain.home and rich.fate == plain.fate
    assert [r["key"] for r in plain.home["rows"]] == ["marriage", "children", "faith", "community", "moving", "smoking",
                                                     "drinking", "languages"]
    assert [fe._eligible_for_path(v2, rich, k) for k in fe.PATHS] == [fe._eligible_for_path(v, plain, k) for k in fe.PATHS]
    assert fe.template_hook("home", rich) == fe.template_hook("home", plain)


def test_rishta_home_hook_names_family_values_and_food():
    v = rp("me", {}, intent="", want_children="", religion="", community="", languages=[], smoking="", drinking="",
           family_values="Traditional", diet="Vegetarian")
    v.rishta = True
    c = rp("c", {}, intent="", want_children="", religion="", community="", languages=[], smoking="", drinking="",
           family_values="Traditional", diet="Vegetarian")
    p = fe.evaluate_pair(v, c, NOW)
    assert fe.template_hook("home", p) == "You agree on family values and food at home."


# ---------------- the API's mode plumbing (no emulator)

class _Snap:
    def __init__(self, path, data):
        self.id = path.rsplit("/", 1)[-1]
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class _Ref:
    def __init__(self, db, path):
        self.db, self.path, self.id = db, path, path.rsplit("/", 1)[-1]

    def get(self, transaction=None):
        return _Snap(self.path, self.db.data.get(self.path))

    def set(self, data, merge=False):
        self.db.data[self.path] = {**(self.db.data.get(self.path) or {}), **data} if merge else dict(data)

    def update(self, data):
        self.db.data[self.path] = {**self.db.data[self.path], **data}

    def delete(self):
        self.db.data.pop(self.path, None)


class _Query:
    def __init__(self, db, path, filters=()):
        self.db, self.path, self.filters, self.n = db, path, filters, None

    def where(self, filter):
        return _Query(self.db, self.path, self.filters + ((filter.field_path, filter.op_string, filter.value),))

    def limit(self, n):
        self.n = n
        return self

    def stream(self):
        out = []
        for path, d in sorted(self.db.data.items()):
            parent, _, _ = path.rpartition("/")
            if parent == self.path and all(op == "==" and d.get(f) == val for f, op, val in self.filters):
                out.append(_Snap(path, d))
        return out[: self.n] if self.n else out


class FakeDb:
    def __init__(self, data):
        self.data = {k: dict(v) for k, v in data.items()}
        self.reads = []

    def document(self, path):
        self.reads.append(path)
        return _Ref(self, path)

    def collection(self, path):
        return _Query(self, path)

    def get_all(self, refs):
        return [r.get() for r in refs]


def _card(uid, **kw):
    link = storage_link(uid, "photos", "a.jpg")
    base = {"uid": uid, "firstName": uid.title(), "age": 26, "gender": "Female", "lookingForGender": "Male",
            "isDiscoverable": True, "isFaceVerified": True, "photos": [link], "cityName": "Pune", "lat": 18.52, "lon": 73.85}
    base.update(kw)
    return base


def _people(cards, privs):
    recent = "2026-10-11T00:00:00+00:00"
    data = {}
    for uid, card in cards.items():
        data[f"publicProfiles/{uid}"] = card
        data[f"profiles/{uid}"] = {"faceVerified": True, "faceVerifiedPhotos": card["photos"], **privs.get(uid, {})}
        data[f"users/{uid}"] = {"lastAppOpenAtUtcIso": recent}
    return data


@pytest.fixture
def api(monkeypatch):
    import fates_api
    monkeypatch.setattr(fates_api, "_chart", lambda profile: None)
    monkeypatch.setattr(fates_api, "_pools_config", {"at": 0.0, "quickMaleMinAge": fe.QUICK_MALE_MIN_AGE})
    monkeypatch.setattr(fe, "ist_day", lambda now=None: TODAY)
    monkeypatch.setattr(fe, "ACTIVE_DAYS", 100000)
    return fates_api


def test_candidates_follow_the_mode(api):
    managed = {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00Z", "profileFor": "daughter"}
    cards = {
        "legacy": _card("legacy"),                                                            # 1.0.7, never indexed
        "bride": _card("bride", **RISHTA_W),                                                  # marriage only
        "both": _card("both", **{**RISHTA_W, "feedQuick": True}),                             # marriage + Quick swipe
        "daughter": _card("daughter", **{**RISHTA_W, "profileFor": "daughter"}),              # made by a parent
        "stale": _card("stale"),                                                              # old card, profile says daughter
        "hindu_only": _card("hindu_only", **RISHTA_W),                                        # her family marked religion
        "private": _card("private", **RISHTA_W, photoPrivacy="accepted",
                         photos=[storage_link("private", "blur", "a.jpg")]),
    }
    privs = {"bride": RISHTA_PRIV, "both": {**RISHTA_PRIV, "quickSwipeOk": True}, "daughter": managed,
             "stale": {"profileFor": "daughter"}, "private": RISHTA_PRIV,
             "hindu_only": {**RISHTA_PRIV, "pref_religions": ["Hindu"], "pref_strict": ["religions"]}}
    db = FakeDb(_people(cards, privs))
    me = fe.person_from_docs("me", {"gender": "Male", "lookingForGender": "Female", "lat": 18.5, "lon": 73.8, "religion": "Sikh"})

    dating = {p.uid for p in api._candidates(db, me, NOW, {})}
    assert dating == {"legacy", "both"}

    me.rishta = True
    fe.rishta_ready(me, viewer=True)
    rishta = {p.uid for p in api._candidates(db, me, NOW, {})}
    assert rishta == {"bride", "both", "daughter"}            # never legacy, never a private card, never against her wishes


def test_quick_male_min_age_is_read_and_cached(api, monkeypatch):
    db = FakeDb({"serverState/pools": {"quickMaleMinAge": 18}})
    assert api.quick_male_min_age(db) == 18
    db.data["serverState/pools"] = {"quickMaleMinAge": 25}
    assert api.quick_male_min_age(db) == 18                    # cached for 10 minutes
    monkeypatch.setattr(api, "_pools_config", {"at": 0.0, "quickMaleMinAge": 21})
    db.data["serverState/pools"] = {"quickMaleMinAge": 5}
    assert api.quick_male_min_age(db) == fe.QUICK_MALE_MIN_AGE  # out of range: the default
    monkeypatch.setattr(api, "_pools_config", {"at": 0.0, "quickMaleMinAge": 21})
    assert api.quick_male_min_age(FakeDb({})) == 21


def test_request_mode_and_viewer_rules(api):
    assert api.request_mode(None) == "quick"
    assert api.request_mode(api.TodayRequest()) == "quick"
    assert api.request_mode(api.TodayRequest(mode="rishta")) == "rishta"
    assert api.request_mode(api.TodayRequest(mode=" Rishta ")) == "rishta"
    assert api.request_mode(api.TodayRequest(mode="dating")) == "quick"
    assert api.viewer_block("rishta", {}, {}, TODAY) == "needs_intent"
    assert api.viewer_block("rishta", {}, SELF_GROOM, TODAY) is None
    assert api.viewer_block("quick", {}, {"profileFor": "none"}, TODAY) == "no_quick"
    assert api.viewer_block("quick", {}, SELF_GROOM, TODAY) is None


def test_today_refuses_rishta_mode_with_a_403_and_dating_gets_an_empty_day(api, monkeypatch):
    """Plan §2 C1: rishta-mode Fates answer 403 'needs_intent' / 'groom_age' (like getRishtas and swipeAction); the
    dating Fates of a family-made profile stay a 200 empty day ('no_quick'), which 1.0.7 already shows."""
    from fastapi import HTTPException
    privs = {"sam": {"profileFor": "self", "gender": "male", "birthDate": "1996-01-01"},
             "kid": {**SELF_GROOM, "birthDate": "2006-01-01"},
             "mom": {"marriageIntent": True, "consentDeclaredAtIso": "2026-10-01T00:00:00Z", "profileFor": "son"}}
    db = FakeDb({f"profiles/{u}": p for u, p in privs.items()})
    who = {"uid": ""}
    monkeypatch.setattr(api, "_deps", api.Deps(db=lambda: db, verify_uid=lambda h: who["uid"],
                                                complete=lambda *a, **k: ("{}", "stop"), resolve_place=lambda p: None,
                                                is_premium=lambda *a: False))
    for uid, reason in (("sam", "needs_intent"), ("kid", "groom_age")):
        who["uid"] = uid
        with pytest.raises(HTTPException) as err:
            api.fates_today(api.TodayRequest(mode="rishta"), authorization="Bearer x")
        assert (err.value.status_code, err.value.detail) == (403, reason)
    who["uid"] = "mom"
    body = api.fates_today(None, authorization="Bearer x")
    assert body["paths"] == [] and body["reason"] == "no_quick" and body["mode"] == "quick"
    assert not [k for k in db.data if k.startswith("dailyFates/")]                      # nothing picked or stored


def test_a_day_made_in_the_other_mode_never_shows_its_unopened_people(api):
    db = FakeDb({})
    ref = db.document("dailyFates/me__2026-10-11")
    untouched = {"mode": "quick", "order": ["heart", "home"], "openedCount": 0,
                 "paths": {"heart": {"targetUid": "a", "opened": False}, "home": {"targetUid": "b", "opened": False}}}
    ref.set(untouched)
    assert api._reconcile_mode(ref, ref.get().to_dict(), "quick") == untouched          # same mode: as it is
    assert api._reconcile_mode(ref, ref.get().to_dict(), "rishta") is None              # made again
    assert "dailyFates/me__2026-10-11" not in db.data
    legacy_day = {k: v for k, v in untouched.items() if k != "mode"}                     # stored before 1.0.8 = dating
    ref.set({**legacy_day, "openedCount": 1, "paths": {**untouched["paths"], "home": {"targetUid": "b", "opened": True}}})
    kept = api._reconcile_mode(ref, ref.get().to_dict(), "rishta")
    assert kept["order"] == ["home"] and set(kept["paths"]) == {"home"} and kept["mode"] == "rishta"
    assert db.data["dailyFates/me__2026-10-11"]["paths"].keys() == {"home"}
