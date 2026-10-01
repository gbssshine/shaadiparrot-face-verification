import json
import os

import pytest

import fates_astro as fa
import fates_astro_chart as fc

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "ashtakoota_fixtures.json"), encoding="utf-8"))
KEYS = {"varna": "Varna", "vashya": "Vashya", "tara": "Tara", "yoni": "Yoni", "graha_maitri": "Graha Maitri",
        "gana": "Gana", "bhakoot": "Bhakoot", "nadi": "Nadi"}


def moon(nak_1based, rashi_1based, lon=None):
    return {"rashi": rashi_1based - 1, "nak": nak_1based - 1, "lon": lon}


@pytest.mark.parametrize("ex", FIX["examples"], ids=[e["id"] for e in FIX["examples"]])
def test_published_software_examples(ex):
    """AstroSage / Prokerala / Divine API outputs, koota by koota."""
    pts = fa.koota_points(moon(ex["boy"]["nakshatra"], ex["boy"]["rashi"]), moon(ex["girl"]["nakshatra"], ex["girl"]["rashi"]))
    expected = ex.get("expected") or ex["expected_partial_verified"]
    for key, name in KEYS.items():
        if key in expected:
            assert pts[name] == expected[key], (key, pts)
    if "total" in expected:
        assert sum(pts.values()) == expected["total"]


def test_tables_are_complete_and_bounded():
    assert len(fa.NAKSHATRAS) == 27 and len(fa.RASHIS) == 12
    for m, size, mx in ((fa.VASHYA, 5, 2), (fa.YONI, 14, 4), (fa.GRAHA_MAITRI, 7, 5), (fa.GANA, 3, 6)):
        assert len(m) == size and all(len(r) == size for r in m)
        assert all(0 <= v <= mx for r in m for v in r)
        assert all(m[i][i] == mx for i in range(size))           # same group always scores full
    for animal in {n[3] for n in fa.NAKSHATRAS}:
        assert animal in fa.YONI_ORDER
    assert fa.GRAHA_MAITRI == [list(r) for r in zip(*fa.GRAHA_MAITRI)]   # symmetric


def test_sworn_enemy_yonis_score_zero():
    for a, b in (("Horse", "Buffalo"), ("Elephant", "Lion"), ("Sheep", "Monkey"), ("Serpent", "Mongoose"),
                 ("Dog", "Deer"), ("Cat", "Rat"), ("Cow", "Tiger")):
        i, j = fa.YONI_ORDER.index(a), fa.YONI_ORDER.index(b)
        assert fa.YONI[i][j] == 0 and fa.YONI[j][i] == 0, (a, b)


def test_mockup_pair_rohini_mrigashira_is_27():
    # Arjun: Rohini, Vrishabha. Anaya: Mrigashira in Mithuna. The pair shown in the design mockups.
    pts = fa.koota_points(moon(4, 2), moon(5, 3, lon=62.0))
    assert pts == {"Varna": 1, "Vashya": 1, "Tara": 3, "Yoni": 4, "Graha Maitri": 5, "Gana": 5, "Bhakoot": 0, "Nadi": 8}
    d = fa.bhakoot_dosha(moon(4, 2), moon(5, 3))
    assert d["present"] and d["cancelled"] and "Venus and Mercury" in d["why"]


def test_tara_rule_counts_both_ways():
    assert fa.koota_points(moon(1, 1), moon(1, 1))["Tara"] == 3          # same star: Janma both ways
    # boy 3 stars after girl: g2b = 3 (Vipat, bad); b2g = 26 -> 26 % 9 = 8 (good)
    assert fa.koota_points(moon(3, 1), moon(1, 1))["Tara"] == 1.5


def test_half_sign_vashya_split():
    assert fa.vashya_group(8, 245.0) == "Manava" and fa.vashya_group(8, 258.0) == "Chatushpada"   # Dhanu
    assert fa.vashya_group(9, 275.0) == "Chatushpada" and fa.vashya_group(9, 290.0) == "Jalachara"  # Makara


def test_nadi_dosha_and_cancellations():
    same = fa.nadi_dosha({"rashi": 0, "nak": 1, "pada": 1}, {"rashi": 0, "nak": 1, "pada": 1})
    assert same["present"] and not same["cancelled"]
    diff_pada = fa.nadi_dosha({"rashi": 0, "nak": 1, "pada": 1}, {"rashi": 0, "nak": 1, "pada": 3})
    assert diff_pada["cancelled"]
    # Ashwini (Adi, Mesha) and Mula (Adi, Dhanu): different signs, lords Mars and Jupiter -> not cancelled
    assert not fa.nadi_dosha({"rashi": 0, "nak": 0}, {"rashi": 8, "nak": 18})["cancelled"]
    assert not fa.nadi_dosha({"rashi": 0, "nak": 0}, {"rashi": 1, "nak": 3})["present"]


def test_bhakoot_same_lord_cancels():
    # Mesha boy, Vrischika girl: 8th, both ruled by Mars
    d = fa.bhakoot_dosha(moon(1, 1), moon(17, 8))
    assert d["present"] and d["cancelled"] and "Mars" in d["why"]
    # Simha boy, Makara girl: 6th, Sun and Saturn are enemies
    d2 = fa.bhakoot_dosha(moon(10, 5), moon(22, 10))
    assert d2["present"] and not d2["cancelled"]


def test_manglik():
    chart = {"marsRashi": 3, "rashi": 3, "venusRashi": 5, "lagna": None}    # Mars in Karka = 1st from Moon
    assert fa.manglik_status(chart)["manglik"]
    own = {"marsRashi": 0, "rashi": 0, "venusRashi": 0, "lagna": None}      # Mars in Mesha (own sign)
    assert not fa.manglik_status(own)["manglik"]
    clear = {"marsRashi": 2, "rashi": 3, "venusRashi": 1, "lagna": None}    # 12th from Moon -> manglik
    assert fa.manglik_status(clear)["manglik"]
    none = {"marsRashi": 5, "rashi": 3, "venusRashi": 1, "lagna": None}     # 3rd from Moon, 5th from Venus
    assert not fa.manglik_status(none)["manglik"]
    both = fa.manglik_dosha(chart, clear)
    assert both["present"] and both["cancelled"]
    one = fa.manglik_dosha(chart, none)
    assert one["present"] and not one["cancelled"]


def test_match_charts_roles_and_range():
    arjun = fc.chart_from_birth((1996, 11, 20), (6, 30), {"lat": 12.97, "lon": 77.59, "tz": "Asia/Kolkata", "precision": "city"})
    anaya = fc.chart_from_birth((1999, 5, 2), None, {"lat": 12.97, "lon": 77.59, "tz": "Asia/Kolkata", "precision": "city"})
    m = fa.match_charts(arjun, anaya, "male", "female")
    assert 0 <= m["total"] <= 36
    assert m["totalMin"] <= m["total"] <= m["totalMax"]
    assert m["precision"] == "date"
    assert [k["name"] for k in m["kootas"]] == list(fa.KOOTA_MAX)
    assert sum(k["points"] for k in m["kootas"]) == pytest.approx(m["total"], abs=0.5)
    assert all(k["meaning"] for k in m["kootas"])
    reverse = fa.match_charts(anaya, arjun, "female", "male")
    assert reverse["total"] == m["total"]                      # the viewer does not change the kundli
    assert m["aMoon"]["nakshatra"] == fa.NAKSHATRAS[arjun["nak"]][0]


def test_same_gender_pairs_average_both_directions():
    a = {"rashi": 4, "nak": 9, "moonLon": 125.0, "precision": "exact"}
    b = {"rashi": 0, "nak": 0, "moonLon": 5.0, "precision": "exact"}
    m = fa.match_charts(a, b, "female", "female")
    one = sum(fa.koota_points(fa._moon(a), fa._moon(b)).values())
    two = sum(fa.koota_points(fa._moon(b), fa._moon(a)).values())
    assert m["total"] == round((one + two) / 2 * 2) / 2
