"""South Indian Dasa Porutham (porutham.py): tables, verdicts, roles, ranges and the Prokerala cross-check."""
import json
import os
import re

import pytest

import fates_astro as fa
import porutham as po

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "porutham_fixtures.json"), encoding="utf-8"))
STAR = {n[0]: i for i, n in enumerate(fa.NAKSHATRAS)}
SIGN = {r[0]: i for i, r in enumerate(fa.RASHIS)}


def moon(rashi, nak):
    return {"rashi": rashi, "nak": nak}


def level(key, boy, girl):
    return next(i["level"] for i in po.items(boy, girl) if i["key"] == key)


def any_sign(nak):
    """A sign the star actually sits in (its first pada)."""
    return (nak * 4) // 9


# ---------------------------------------------------------------- Rajju and Vedha
def test_rajju_groups_are_the_published_lists():
    published = {
        "Pada": ["Ashwini", "Ashlesha", "Magha", "Jyeshtha", "Mula", "Revati"],
        "Kati": ["Bharani", "Pushya", "Purva Phalguni", "Anuradha", "Purva Ashadha", "Uttara Bhadrapada"],
        "Nabhi": ["Krittika", "Punarvasu", "Uttara Phalguni", "Vishakha", "Uttara Ashadha", "Purva Bhadrapada"],
        "Kantha": ["Rohini", "Ardra", "Hasta", "Swati", "Shravana", "Shatabhisha"],
        "Siro": ["Mrigashira", "Chitra", "Dhanishta"],
    }
    groups = {}
    for n in range(27):
        groups.setdefault(po.rajju(n), set()).add(fa.NAKSHATRAS[n][0])
    assert groups == {k: set(v) for k, v in published.items()}
    assert sorted(len(v) for v in groups.values()) == [3, 6, 6, 6, 6]
    assert po.rajju_label(STAR["Rohini"]) == "Kantha (neck)" and po.rajju_label(STAR["Revati"]) == "Pada (feet)"


def test_vedha_pairs():
    assert len(po.VEDHA) == 15 and all(len(p) == 2 for p in po.VEDHA)
    for pair in po.VEDHA:
        a, b = sorted(pair)
        assert po.rajju(a) == po.rajju(b), (a, b)          # every Vedha pair also shares a Rajju
        for boy, girl in ((a, b), (b, a)):                    # symmetric
            assert level("vedha", moon(any_sign(boy), boy), moon(any_sign(girl), girl)) == "no"
    named = {frozenset({STAR[a], STAR[b]}) for a, b in (("Rohini", "Swati"), ("Mrigashira", "Chitra"),
                                                       ("Chitra", "Dhanishta"), ("Mrigashira", "Dhanishta"),
                                                       ("Ashwini", "Jyeshtha"), ("Magha", "Revati"))}
    assert named <= po.VEDHA
    assert level("vedha", moon(0, 0), moon(0, 0)) == "good"   # a shared star is not a Vedha pair


# ---------------------------------------------------------------- counted poruthams
def test_dina_all_27_counts_and_the_same_star_lists():
    good = {2, 4, 6, 8, 9, 11, 13, 15, 18, 20, 24, 26}
    for girl in range(27):
        for c in range(2, 28):
            boy = (girl + c - 1) % 27
            assert po.star_count(boy, girl) == c
            assert level("dina", moon(0, boy), moon(0, girl)) == ("good" if c in good else "no"), (girl, c)
    same_good = {"Rohini", "Ardra", "Magha", "Hasta", "Vishakha", "Uttara Ashadha", "Shravana", "Revati"}
    same_fair = {"Ashwini", "Krittika", "Mrigashira", "Punarvasu", "Pushya", "Uttara Phalguni", "Chitra", "Purva Ashadha"}
    for n in range(27):
        name = fa.NAKSHATRAS[n][0]
        want = "good" if name in same_good else ("fair" if name in same_fair else "no")
        assert level("dina", moon(0, n), moon(0, n)) == want, name


def test_mahendra_has_exactly_eight_good_counts():
    good = {c for c in range(1, 28) if level("mahendra", moon(0, c - 1), moon(0, 0)) == "good"}
    assert good == {4, 7, 10, 13, 16, 19, 22, 25}


def test_stree_deergha():
    def at(c):
        return level("stree_deergha", moon(0, c - 1), moon(0, 0))
    assert at(14) == "good" and at(27) == "good"
    assert [at(c) for c in (10, 11, 12, 13)] == ["fair"] * 4
    assert at(9) == "no" and at(1) == "no"
    twelfth = next(i for i in po.items(moon(0, 11), moon(0, 0)) if i["key"] == "stree_deergha")
    assert twelfth["meaning"] == ("The groom's star is the 12th from the bride's: Kerala practice accepts beyond "
                                  "the 9th, Tamil practice asks for beyond the 13th.")


def test_gana():
    deva, manushya, rakshasa = STAR["Ashwini"], STAR["Bharani"], STAR["Krittika"]
    assert level("gana", moon(0, deva), moon(0, STAR["Mrigashira"])) == "good"
    assert level("gana", moon(0, rakshasa), moon(0, STAR["Ashlesha"])) == "good"
    assert level("gana", moon(0, deva), moon(0, manushya)) == "fair"
    assert level("gana", moon(0, manushya), moon(0, deva)) == "fair"
    assert level("gana", moon(0, rakshasa), moon(0, deva)) == "no"
    assert level("gana", moon(0, manushya), moon(0, rakshasa)) == "no"


def test_yoni_enemies_are_the_zero_cells():
    enemy = set()
    for b in range(27):
        for g in range(27):
            yb, yg = fa.NAKSHATRAS[b][3], fa.NAKSHATRAS[g][3]
            lv = level("yoni", moon(0, b), moon(0, g))
            zero = fa.YONI[fa.YONI_ORDER.index(yb)][fa.YONI_ORDER.index(yg)] == 0
            if yb == yg:
                assert lv == "good"
            else:
                assert lv == ("no" if zero else "fair"), (yb, yg)
            if zero:
                enemy.add(frozenset({yb, yg}))
    assert enemy == {frozenset(p) for p in (("Horse", "Buffalo"), ("Elephant", "Lion"), ("Sheep", "Monkey"),
                                            ("Serpent", "Mongoose"), ("Dog", "Deer"), ("Cat", "Rat"), ("Cow", "Tiger"))}


def test_rasi():
    for r in range(12):
        assert level("rasi", moon(r, 0), moon(r, 0)) == "good"                 # the same sign
    for girl in range(12):
        for count in (2, 6, 8, 12):
            assert level("rasi", moon((girl + count - 1) % 12, 0), moon(girl, 0)) == "no", (girl, count)
    assert level("rasi", moon(SIGN["Tula"], 0), moon(SIGN["Mesha"], 0)) == "good"   # Tula is the 7th from Mesha
    assert level("rasi", moon(SIGN["Makara"], 0), moon(SIGN["Karka"], 0)) == "no"   # Karka bride, Makara groom
    assert level("rasi", moon(SIGN["Karka"], 0), moon(SIGN["Makara"], 0)) == "no"
    assert level("rasi", moon(SIGN["Kumbha"], 0), moon(SIGN["Simha"], 0)) == "no"
    for count in (9, 10, 11):
        assert level("rasi", moon((count - 1) % 12, 0), moon(0, 0)) == "good"


def test_rasi_adhipathi():
    assert level("rasi_adhipathi", moon(SIGN["Vrischika"], 0), moon(SIGN["Mesha"], 0)) == "good"   # both Mars
    assert level("rasi_adhipathi", moon(SIGN["Tula"], 0), moon(SIGN["Mesha"], 0)) == "fair"        # Venus, Mars
    assert level("rasi_adhipathi", moon(SIGN["Makara"], 0), moon(SIGN["Simha"], 0)) == "no"        # Saturn, Sun


def test_vasya_has_a_direction():
    assert level("vasya", moon(SIGN["Vrischika"], 0), moon(SIGN["Mesha"], 0)) == "good"   # Mesha bride, Vrischika groom
    assert level("vasya", moon(SIGN["Mesha"], 0), moon(SIGN["Vrischika"], 0)) == "no"     # the reverse


# ---------------------------------------------------------------- count, verdict, roles, range
def test_count_is_the_number_of_agreeing_items():
    for b in range(0, 27, 2):
        for g in range(1, 27, 3):
            boy, girl = moon(any_sign(b), b), moon(any_sign(g), g)
            res = po.dasa_porutham(boy, girl, "male", "female")
            assert [i["key"] for i in res["items"]] == po.KEYS
            assert res["count"] == sum(1 for i in res["items"] if i["level"] != "no")
            assert 0 <= res["count"] <= 10
            assert set(res["items"][0]) == {"key", "name", "about", "level", "meaning"}


def test_verdict_bands_and_the_rajju_override():
    assert po.verdict(8, False, True) == {"word": "Rajju doesn't match", "tone": "low",
                                          "text": "Most South Indian families treat Rajju as a must, whatever the count. "
                                                  "Talk it over with the family astrologer."}
    assert po.verdict(9, False, False)["word"] == "Rajju and Vedha don't match"
    assert po.verdict(8, True, True)["word"] == "Excellent match" and po.verdict(10, True, True)["tone"] == "great"
    assert po.verdict(7, True, True)["word"] == "Good match" and po.verdict(6, True, True)["tone"] == "good"
    assert po.verdict(5, True, True) == {"word": "Fair match", "tone": "ok",
                                         "text": "Half of the poruthams agree and Rajju is clear: many families go ahead with this."}
    assert po.verdict(4, True, True)["word"] == "Needs a closer look" and po.verdict(0, True, True)["tone"] == "low"
    for count in range(11):
        assert po.verdict(count, True, True)["text"]


def test_a_vedha_pair_reads_rajju_and_vedha():
    res = po.dasa_porutham(moon(SIGN["Tula"], STAR["Swati"]), moon(SIGN["Vrishabha"], STAR["Rohini"]), "male", "female")
    assert not res["rajjuOk"] and not res["vedhaOk"]
    assert res["verdict"]["word"] == "Rajju and Vedha don't match" and res["verdict"]["tone"] == "low"
    vedha = res["items"][-1]
    assert vedha["meaning"] == "Swati and Rohini are a Vedha pair. Families read this together with Rajju."


def test_eight_agreeing_with_the_same_rajju_is_still_low():
    # Find real pairs with 8+ agreeing poruthams and Rajju failing: the count never hides Rajju.
    found = 0
    for b in range(27):
        for g in range(27):
            for rb in {(b * 4) // 9, (b * 4 + 3) // 9}:
                for rg in {(g * 4) // 9, (g * 4 + 3) // 9}:
                    res = po.dasa_porutham(moon(rb, b), moon(rg, g), "male", "female")
                    if res["count"] >= 8 and not res["rajjuOk"]:
                        found += 1
                        assert res["verdict"]["word"] in ("Rajju doesn't match", "Rajju and Vedha don't match")
                        assert res["verdict"]["tone"] == "low"
    assert found


def test_five_with_rajju_is_fair():
    for b in range(27):
        for g in range(27):
            res = po.dasa_porutham(moon(any_sign(b), b), moon(any_sign(g), g), "male", "female")
            if res["count"] == 5 and res["rajjuOk"]:
                assert res["verdict"]["word"] == "Fair match"
                return
    pytest.fail("no pair with 5 and Rajju")


def test_roles_follow_gender():
    priya, karthik = moon(SIGN["Kanya"], STAR["Hasta"]), moon(SIGN["Makara"], STAR["Shravana"])
    as_bride = po.dasa_porutham(priya, karthik, "female", "male")          # a female viewer is the bride
    assert as_bride["items"] == po.items(karthik, priya)
    assert po.dasa_porutham(karthik, priya, "male", "female")["items"] == as_bride["items"]
    assert po.dasa_porutham(priya, karthik, "female", "")["items"] == as_bride["items"]
    # Same or unknown genders: the first person is the groom.
    assert po.dasa_porutham(priya, karthik, "", "")["items"] == po.items(priya, karthik)
    # Swapping the genders changes Stree Deergha (the count runs from the bride's star).
    a, b = moon(0, 0), moon(SIGN["Simha"], STAR["Purva Phalguni"])
    sd = [next(i["level"] for i in po.dasa_porutham(a, b, ga, gb)["items"] if i["key"] == "stree_deergha")
          for ga, gb in (("male", "female"), ("female", "male"))]
    assert sd[0] != sd[1]
    # aRajju / bRajju stay with a and b as sent, whoever is the bride.
    res = po.dasa_porutham(priya, karthik, "female", "male")
    assert res["aRajju"] == "Kantha (neck)" and res["bRajju"] == "Kantha (neck)"


def test_range_without_a_birth_time():
    fixed = moon(SIGN["Mesha"], STAR["Ashwini"])
    # The day lets the Moon sit in Bharani (Kati) or Krittika (Nabhi): Rajju with Ashwini (Pada) is clear either way.
    day = {"rashi": 0, "nak": 1, "options": [[0, 1], [0, 2]]}
    res = po.dasa_porutham(fixed, day, "male", "female")
    assert res["countMin"] <= res["count"] <= res["countMax"]
    assert res["mustSure"] is True
    # Ashwini (Pada, like the groom's Magha) or Bharani (Kati) the same day: an option flips Rajju, so it isn't sure.
    flip = {"rashi": 0, "nak": 0, "options": [[0, 0], [0, 1]]}
    res2 = po.dasa_porutham(moon(SIGN["Simha"], STAR["Magha"]), flip, "male", "female")
    assert res2["rajjuOk"] is False
    assert res2["mustSure"] is False
    assert res2["countMin"] <= res2["count"] <= res2["countMax"]
    # An option list without the main Moon still counts the main Moon.
    odd = {"rashi": 0, "nak": 1, "options": [[0, 2], ["x", None]]}
    res3 = po.dasa_porutham(fixed, odd, "male", "female")
    assert res3["countMin"] <= res3["count"] <= res3["countMax"]


def test_meanings_are_person_neutral_and_calm():
    banned = re.compile(r"\b(you|your|yours|his|her|hers|him|death|die|widow|danger|curse|bad luck)\b", re.I)
    for b in range(27):
        for g in range(27):
            for rb in {(b * 4) // 9, (b * 4 + 3) // 9}:
                for rg in {(g * 4) // 9, (g * 4 + 3) // 9}:
                    for i in po.items(moon(rb, b), moon(rg, g)):
                        assert i["meaning"] and not banned.search(i["meaning"]), i
    for count in range(11):
        for r in (True, False):
            for v in (True, False):
                assert not banned.search(json.dumps(po.verdict(count, r, v)))
    assert po.ABOUT["rajju"] == "The most-checked porutham" and po.ABOUT["vedha"] == "Stars that block each other"


def test_output_has_no_birth_details():
    a = {"rashi": 1, "nak": 3, "pada": 2, "moonLon": 43.21, "options": [[1, 3], [1, 4]], "lagna": 5,
         "marsRashi": 2, "birthDate": "1996-04-11", "birthTime": "05:20"}
    b = {"rashi": 6, "nak": 14, "pada": 3, "moonLon": 190.5, "options": [[6, 14]], "precision": "exact"}
    res = po.dasa_porutham(a, b, "female", "male")
    assert set(res) == {"count", "countMin", "countMax", "items", "rajjuOk", "vedhaOk", "mustSure", "aRajju", "bRajju",
                        "verdict"}
    keys = set()

    def walk(v):
        if isinstance(v, dict):
            keys.update(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(res)
    assert not keys & {"lon", "moonLon", "pada", "options", "birthDate", "birthTime", "lagna", "marsRashi", "precision"}
    text = json.dumps(res)
    for secret in ("43.21", "190.5", "1996", "05:20"):
        assert secret not in text, secret


# ---------------------------------------------------------------- Prokerala cross-check
def test_fixture_is_documented():
    assert FIX["source"]["url"].startswith("https://www.prokerala.com/") and FIX["source"]["checked"]
    assert sum(1 for p in FIX["pairs"] if p["agrees"]) >= 3


@pytest.mark.parametrize("pair", FIX["pairs"], ids=[p["id"] for p in FIX["pairs"]])
def test_prokerala_pairs(pair):
    girl, boy = moon(pair["girl"]["rashi"], pair["girl"]["nak"]), moon(pair["boy"]["rashi"], pair["boy"]["nak"])
    res = po.dasa_porutham(boy, girl, "male", "female")
    pk_no = {k for k, v in pair["prokerala"].items() if v == "Bad"}
    our_no = {i["key"] for i in res["items"] if i["level"] == "no"}
    # Known per-item variants stay exactly as listed: a change to a table shows here.
    assert sorted(pk_no ^ our_no) == sorted(pair["variantDiffs"])
    if pair["agrees"]:
        assert res["count"] == pair["prokeralaCount"]
        assert res["rajjuOk"] == (pair["prokerala"]["rajju"] != "Bad")
        assert res["vedhaOk"] == (pair["prokerala"]["vedha"] != "Bad")
