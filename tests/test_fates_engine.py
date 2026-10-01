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
