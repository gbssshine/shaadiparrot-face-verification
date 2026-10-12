"""Partner preferences (plan §2 C11): fates_engine._pref_gate and its helpers against the shared vectors.

The vectors below are the one contract the three sides must give the same answer for: C# (PartnerPrefsLogic /
Biodata / CasteList), TS (rishtas.ts) and Python (fates_engine). They are plain data (JSON-shaped) so the other
sides can copy them verbatim into their own tests. The caste cases run against the shared caste_list.json (the
W1-CASTE export copied from firebase-functions/src/casteList.json), not a stub.

The must-have rule (W1-S-FATES deviation 2, which TS and C# have to adopt, or the integrator picks one):
- my side: what I marked "Must have" (pref_strict, only the markable tokens), or religion + marital status while
  pref_strict was never saved; plus caste, Manglik "must match", the income minimum and "no children", which are
  must-haves by what they say; work (professions) and height never hide anyone;
- their side: only the parts their family marked "Must have" (pref_strict); never the defaults and never the four
  that have no toggle;
- an unknown, unstated or private value never hides anyone.

PREF_GATE_VECTORS is a verbatim copy of the TS PREF_GATE_VECTORS (APP firebase-tests/tests/rishtas.functions.test.mjs,
same names, keys and order): edit both together. EDUCATION_RANK pins RishtaMatch.EducationRank / educationRank /
education_rank, including the curly apostrophe a phone keyboard types.
"""
import json
import os

import pytest

import fates_engine as fe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASTE_FILE = os.path.join(ROOT, "caste_list.json")

# ------------------------------------------------------------------ the shared vectors
# [wanted, theirs, fits]: Biodata.PrefFits / pref_fits (religion, mother tongue, marital status, food, work).
PREF_FITS = [
    [[], "Muslim", True],
    [["Hindu"], "Hindu", True],
    [["Hindu"], "hindu", True],
    [["Hindu", "Jain"], "Jain", True],
    [["Hindu"], "Muslim", False],
    [["Hindu"], "", True],
    [["Hindu"], None, True],
    [["Hindu"], "Prefer not to say", True],
    [["Never married"], "Divorced", False],
    [["Never married"], "Never married", True],
    [["Vegetarian"], "Non-vegetarian", False],
    [["Vegetarian"], "Don't know", True],
]

# [wanted, theirs, fits]: CasteList.Fits / caste_fits with the shared list.
CASTE_FITS = [
    [[], "Bhumihar", True],                                  # Caste no bar
    [["", "  "], "Bhumihar", True],                          # only blanks wanted = no bar
    [["Brahmin"], "Brahmin", True],
    [["Brahmin"], "BRAHMIN", True],
    [["Brahmin"], "Brahmin - Iyer", True],                   # a listed community under its umbrella
    [["brahmin"], "Brahmin - Iyengar", True],
    [["Brahmin"], "Brahmin - Vadama", True],                 # typed "X - Y": the prefix rule
    [["Brahmin - Iyer"], "Brahmin", False],                  # never the other way
    [["Brahmin - Iyer"], "Brahmin - Iyengar", False],
    [["Brahmin"], "Bhumihar", False],
    [["Brahmin"], "Brahminical", False],                     # a prefix only with " - "
    [["Patel"], "Patel - Leuva", True],
    [["Sunni"], "Sunni - Hanafi", True],
    [["Sunni"], "Shia - Ismaili", False],
    [["Brahmin", "Rajput"], "Rajput", True],
    [["Brahmin"], "", True],
    [["Brahmin"], None, True],
    [["Brahmin"], "Prefer not to say", True],
    [["Brahmin"], "Don't know", True],
    [["Brahmin"], "Inter-caste / Mixed", True],
    [["Brahmin"], "inter-caste / mixed", True],
    [["Brahmin"], "  Brahmin - Iyer  ", True],               # trimmed
]

# [mine, theirs, fits]: Biodata.ManglikFits / manglik_fits ("Manglik must match").
MANGLIK_FITS = [
    ["Yes", "Yes", True], ["Yes", "No", False], ["No", "Yes", False], ["No", "No", True],
    ["Yes", "Partial (Anshik)", True], ["Partial (Anshik)", "No", True], ["No", "Partial (Anshik)", True],
    ["Yes", "", True], ["", "No", True], ["Don't know", "Yes", True], ["No", "Prefer not to say", True],
    ["yes", "YES", True],
]

# [their public income code, minimum lakh, fits]: Biodata.IncomeAtLeast / income_at_least.
INCOME_AT_LEAST = [
    ["7-10L", 0, True], ["7-10L", 10, False], ["10-15L", 10, True], ["1Cr+", 100, True], ["50L-1Cr", 100, False],
    ["0-2L", 4, False], ["", 20, True], ["prefer_not", 20, True], ["junk", 20, True],
]

# [stored pref_incomeMinLakh, normal minimum]: Biodata.NormalMinLakh / normal_min_lakh.
NORMAL_MIN_LAKH = [[0, 0], [3, 0], [4, 4], [12, 10], [49, 30], [100, 100], [250, 100]]

# [pref_childrenOk, their children answer, fits]: PartnerPrefsLogic.ChildrenFit.
CHILDREN_FIT = [
    ["any", "Yes, living together", True], ["none", "Yes, living together", False],
    ["none", "Yes, living separately", False], ["none", "No children", True], ["none", "", True],
    ["none", "yes", False], ["", "Yes, living together", True],
]

# Both ways through _pref_gate. "me" and "them" are {card: publicProfiles fields, profile: profiles fields};
# expect: None, "pref_<token>" (my must-have) or "their_pref_<token>" (their marked must-have).
_HINDU_CARD = {"religion": "Hindu", "maritalStatus": "Never married", "caste": "Brahmin - Iyer", "manglik": "No",
               "diet": "Vegetarian", "education": "Master's", "profession": "Engineer", "community": "Tamil",
               "children": "No children", "annualIncome": "10-15L"}
GATE = [
    {"case": "nothing set", "me": {}, "them": {}, "expect": None},
    {"case": "religion hides by default", "me": {"profile": {"pref_religions": ["Muslim"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_religions"},
    {"case": "marital status hides by default", "me": {"profile": {"pref_maritalStatuses": ["Divorced"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_maritalStatuses"},
    {"case": "unstated religion never hides", "me": {"profile": {"pref_religions": ["Muslim"]}},
     "them": {"card": {**_HINDU_CARD, "religion": "Prefer not to say"}}, "expect": None},
    {"case": "mother tongue only ranks unless marked", "me": {"profile": {"pref_communities": ["Marathi"]}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "mother tongue marked", "me": {"profile": {"pref_communities": ["Marathi"], "pref_strict": ["communities"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_communities"},
    {"case": "food marked", "me": {"profile": {"pref_diets": ["Non-vegetarian"], "pref_strict": ["diets"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_diets"},
    {"case": "education marked", "me": {"profile": {"pref_educationMin": 5, "pref_strict": ["education"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_education"},
    {"case": "education only ranks", "me": {"profile": {"pref_educationMin": 5}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "marking replaces the default", "me": {"profile": {"pref_religions": ["Muslim"], "pref_strict": ["diets"]}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "saved with nothing marked", "me": {"profile": {"pref_religions": ["Muslim"], "pref_strict": []}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "caste umbrella fits its community", "me": {"profile": {"pref_castes": ["Brahmin"]}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "caste is a must-have once chosen", "me": {"profile": {"pref_castes": ["Rajput"]}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_castes"},
    {"case": "caste even with nothing marked", "me": {"profile": {"pref_castes": ["Rajput"], "pref_strict": []}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_castes"},
    {"case": "caste no bar", "me": {"profile": {"pref_castes": []}}, "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "mixed caste never hides", "me": {"profile": {"pref_castes": ["Rajput"]}},
     "them": {"card": {**_HINDU_CARD, "caste": "Inter-caste / Mixed"}}, "expect": None},
    {"case": "manglik must match", "me": {"card": {"manglik": "Yes"}, "profile": {"pref_manglik": "match"}},
     "them": {"card": _HINDU_CARD}, "expect": "pref_manglik"},
    {"case": "manglik partial fits", "me": {"card": {"manglik": "Yes"}, "profile": {"pref_manglik": "match"}},
     "them": {"card": {**_HINDU_CARD, "manglik": "Partial (Anshik)"}}, "expect": None},
    {"case": "my manglik unknown: all fit", "me": {"profile": {"pref_manglik": "match"}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "manglik doesn't matter", "me": {"card": {"manglik": "Yes"}}, "them": {"card": _HINDU_CARD},
     "expect": None},
    {"case": "income minimum", "me": {"profile": {"pref_incomeMinLakh": 20}}, "them": {"card": _HINDU_CARD},
     "expect": "pref_income"},
    {"case": "income minimum rounds down to a band", "me": {"profile": {"pref_incomeMinLakh": 12}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "private income never hides", "me": {"profile": {"pref_incomeMinLakh": 20}},
     "them": {"card": {**_HINDU_CARD, "annualIncome": ""}, "profile": {"annualIncome": "0-2L"}}, "expect": None},
    {"case": "no children", "me": {"profile": {"pref_childrenOk": "none"}},
     "them": {"card": {**_HINDU_CARD, "maritalStatus": "Divorced", "children": "Yes, living together"}},
     "expect": "pref_children"},
    {"case": "work never hides", "me": {"profile": {"pref_professions": ["Doctor"], "pref_strict": ["professions"]}},
     "them": {"card": _HINDU_CARD}, "expect": None},
    {"case": "their unsaved wishes never hide me", "me": {"card": _HINDU_CARD},
     "them": {"profile": {"pref_religions": ["Jain"]}}, "expect": None},
    {"case": "their marked religion hides me", "me": {"card": _HINDU_CARD},
     "them": {"profile": {"pref_religions": ["Jain"], "pref_strict": ["religions"]}}, "expect": "their_pref_religions"},
    {"case": "their marked food hides me", "me": {"card": _HINDU_CARD},
     "them": {"profile": {"pref_diets": ["Non-vegetarian"], "pref_strict": ["diets"]}}, "expect": "their_pref_diets"},
    {"case": "their caste wish never hides me", "me": {"card": _HINDU_CARD},
     "them": {"profile": {"pref_castes": ["Rajput"], "pref_strict": ["castes"]}}, "expect": None},
    {"case": "their income and children wishes never hide me",
     "me": {"card": {**_HINDU_CARD, "annualIncome": "0-2L", "children": "Yes, living together"}},
     "them": {"profile": {"pref_incomeMinLakh": 50, "pref_childrenOk": "none", "pref_manglik": "match",
                          "pref_strict": ["religions", "maritalStatuses", "income", "children", "manglik"]},
              "card": {"manglik": "Yes"}}, "expect": None},
    {"case": "mine first", "me": {"card": _HINDU_CARD, "profile": {"pref_religions": ["Muslim"]}},
     "them": {"card": {**_HINDU_CARD, "religion": "Sikh"}, "profile": {"pref_religions": ["Jain"],
                                                                       "pref_strict": ["religions"]}},
     "expect": "pref_religions"},
]

# The TS shape: prefGate(mine, me, theirs, them) -> null | 'pref_<part>' | 'their_pref_<part>'. mine / theirs:
# profiles/{uid} preference fields; me / them: public-card facts. Each side is person_from_docs(uid, public=<facts>,
# private=<prefs>), then _pref_gate(me, them).
PREF_GATE_VECTORS = [
    {"name": "nothing set", "mine": {}, "me": {"religion": "Hindu"}, "theirs": {}, "them": {"religion": "Muslim"}, "want": None},
    {"name": "religion: a must-have by default", "mine": {"pref_religions": ["Hindu"]}, "me": {}, "theirs": {}, "them": {"religion": "Muslim"}, "want": "pref_religions"},
    {"name": "religion unstated never hides", "mine": {"pref_religions": ["Hindu"]}, "me": {}, "theirs": {}, "them": {"religion": "Prefer not to say"}, "want": None},
    {"name": "religion ignoring case", "mine": {"pref_religions": ["Hindu"]}, "me": {}, "theirs": {}, "them": {"religion": "hindu"}, "want": None},
    {"name": "marital status: a must-have by default", "mine": {"pref_maritalStatuses": ["Never married"]}, "me": {}, "theirs": {}, "them": {"maritalStatus": "Divorced"}, "want": "pref_maritalStatuses"},
    {"name": "mother tongue only ranks", "mine": {"pref_communities": ["Marathi"]}, "me": {}, "theirs": {}, "them": {"community": "Gujarati"}, "want": None},
    {"name": "mother tongue marked", "mine": {"pref_communities": ["Marathi"], "pref_strict": ["communities"]}, "me": {}, "theirs": {}, "them": {"community": "Gujarati"}, "want": "pref_communities"},
    {"name": "marks replace the default", "mine": {"pref_religions": ["Hindu"], "pref_strict": ["diets"]}, "me": {}, "theirs": {}, "them": {"religion": "Muslim"}, "want": None},
    {"name": "food marked", "mine": {"pref_diets": ["Vegetarian"], "pref_strict": ["diets"]}, "me": {}, "theirs": {}, "them": {"diet": "Non-vegetarian"}, "want": "pref_diets"},
    {"name": "education marked", "mine": {"pref_educationMin": 4, "pref_strict": ["education"]}, "me": {}, "theirs": {}, "them": {"education": "Bachelor's"}, "want": "pref_education"},
    {"name": "education marked, unknown", "mine": {"pref_educationMin": 4, "pref_strict": ["education"]}, "me": {}, "theirs": {}, "them": {"education": ""}, "want": None},
    {"name": "education only ranks", "mine": {"pref_educationMin": 4}, "me": {}, "theirs": {}, "them": {"education": "Bachelor's"}, "want": None},
    {"name": "education marked, a curly apostrophe", "mine": {"pref_educationMin": 4, "pref_strict": ["education"]}, "me": {}, "theirs": {}, "them": {"education": "Bachelor’s"}, "want": "pref_education"},
    {"name": "caste: a listed community under its umbrella", "mine": {"pref_castes": ["Brahmin"]}, "me": {}, "theirs": {}, "them": {"caste": "Brahmin - Iyer"}, "want": None},
    {"name": "caste: another caste", "mine": {"pref_castes": ["Brahmin"]}, "me": {}, "theirs": {}, "them": {"caste": "Rajput"}, "want": "pref_castes"},
    {"name": "caste: a typed \"X - Y\" under a wanted umbrella", "mine": {"pref_castes": ["Brahmin"]}, "me": {}, "theirs": {}, "them": {"caste": "Brahmin - Vadama"}, "want": None},
    {"name": "caste: never the other way", "mine": {"pref_castes": ["Brahmin - Iyer"]}, "me": {}, "theirs": {}, "them": {"caste": "Brahmin"}, "want": "pref_castes"},
    {"name": "caste: inter-caste passes", "mine": {"pref_castes": ["Brahmin"]}, "me": {}, "theirs": {}, "them": {"caste": "Inter-caste / Mixed"}, "want": None},
    {"name": "caste: unstated passes", "mine": {"pref_castes": ["Brahmin"]}, "me": {}, "theirs": {}, "them": {"caste": "Prefer not to say"}, "want": None},
    {"name": "caste is a must-have even when marks were saved", "mine": {"pref_castes": ["Brahmin"], "pref_strict": []}, "me": {}, "theirs": {}, "them": {"caste": "Bhumihar"}, "want": "pref_castes"},
    {"name": "Manglik must match: Yes vs No", "mine": {"pref_manglik": "match"}, "me": {"manglik": "Yes"}, "theirs": {}, "them": {"manglik": "No"}, "want": "pref_manglik"},
    {"name": "Manglik must match: partial fits", "mine": {"pref_manglik": "match"}, "me": {"manglik": "Yes"}, "theirs": {}, "them": {"manglik": "Partial (Anshik)"}, "want": None},
    {"name": "Manglik must match: mine unknown", "mine": {"pref_manglik": "match"}, "me": {"manglik": ""}, "theirs": {}, "them": {"manglik": "No"}, "want": None},
    {"name": "Manglik: doesn't matter", "mine": {}, "me": {"manglik": "Yes"}, "theirs": {}, "them": {"manglik": "No"}, "want": None},
    {"name": "income minimum (12 → 10 lakh)", "mine": {"pref_incomeMinLakh": 12}, "me": {}, "theirs": {}, "them": {"annualIncome": "7-10L"}, "want": "pref_income"},
    {"name": "income at the minimum", "mine": {"pref_incomeMinLakh": 12}, "me": {}, "theirs": {}, "them": {"annualIncome": "10-15L"}, "want": None},
    {"name": "income private or unknown", "mine": {"pref_incomeMinLakh": 12}, "me": {}, "theirs": {}, "them": {"annualIncome": "prefer_not"}, "want": None},
    {"name": "prefer no children", "mine": {"pref_childrenOk": "none"}, "me": {}, "theirs": {}, "them": {"children": "Yes, living together"}, "want": "pref_children"},
    {"name": "prefer no children: none", "mine": {"pref_childrenOk": "none"}, "me": {}, "theirs": {}, "them": {"children": "No children"}, "want": None},
    {"name": "work only ranks, even marked", "mine": {"pref_professions": ["Doctor"], "pref_strict": ["professions"]}, "me": {}, "theirs": {}, "them": {"profession": "Engineer"}, "want": None},
    {"name": "their wish without marks never hides me", "mine": {}, "me": {"religion": "Hindu"}, "theirs": {"pref_religions": ["Jain"]}, "them": {}, "want": None},
    {"name": "their marks saved empty", "mine": {}, "me": {"religion": "Hindu"}, "theirs": {"pref_religions": ["Jain"], "pref_strict": []}, "them": {}, "want": None},
    {"name": "their marked religion", "mine": {}, "me": {"religion": "Hindu"}, "theirs": {"pref_religions": ["Jain"], "pref_strict": ["religions"]}, "them": {}, "want": "their_pref_religions"},
    {"name": "their marked food", "mine": {}, "me": {"diet": "Non-vegetarian"}, "theirs": {"pref_diets": ["Vegetarian"], "pref_strict": ["diets"]}, "them": {}, "want": "their_pref_diets"},
    {"name": "their caste, Manglik, income and children wishes never hide me", "mine": {}, "me": {"caste": "Bhumihar", "manglik": "Yes", "annualIncome": "0-2L", "children": "Yes, living together"}, "theirs": {"pref_castes": ["Brahmin"], "pref_manglik": "match", "pref_incomeMinLakh": 20, "pref_childrenOk": "none", "pref_strict": ["castes", "manglik", "income", "children"]}, "them": {"manglik": "No"}, "want": None},
    {"name": "my must-haves come first", "mine": {"pref_religions": ["Jain"], "pref_childrenOk": "none"}, "me": {}, "theirs": {}, "them": {"religion": "Hindu", "children": "Yes, living together"}, "want": "pref_religions"},
]

# [education as typed, rank]: RishtaMatch.EducationRank / rishtas.ts educationRank (the curly apostrophe reads as ').
EDUCATION_RANK = [
    ["High school", 1], ["Diploma", 2], ["Bachelor's", 3], ["Bachelor’s", 3], ["BACHELOR’S", 3],
    ["  Bachelor’s ", 3], ["Law (LLB)", 3], ["Master's", 4], ["Master’s", 4], ["MBA", 4], ["CA / CS", 4],
    ["MBBS / Medical", 4], ["Doctorate", 5], ["PhD", 0], ["Bachelors", 0], ["Bachelor‘s", 0], ["", 0], [None, 0],
]


# ------------------------------------------------------------------ the shared caste list
@pytest.fixture(scope="module", autouse=True)
def shared_caste_list():
    """The real caste_list.json, loaded fresh, so these vectors never run against a stub or a previous test's list."""
    assert os.path.exists(CASTE_FILE), "caste_list.json (the W1-CASTE export) must sit at the repo root"
    with open(CASTE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    fe.load_caste_list(data)
    yield data
    fe._caste_state.update(loaded=False, umbrellas=None)


def test_the_shared_list_has_the_umbrellas_the_vectors_use(shared_caste_list):
    by_name = {e["name"]: e["umbrella"] for e in shared_caste_list["entries"]}
    for name, umbrella in (("Brahmin - Iyer", "Brahmin"), ("Brahmin - Iyengar", "Brahmin"), ("Patel - Leuva", "Patel"),
                           ("Sunni - Hanafi", "Sunni"), ("Shia - Ismaili", "Shia")):
        assert by_name.get(name) == umbrella, name
    for name in ("Brahmin", "Patel", "Sunni", "Rajput", "Bhumihar"):
        assert by_name.get(name) == "", name
    # The CasteList rule: every listed "X - Y" name sits under the umbrella X, so the prefix rule and the umbrella
    # give the same answer for listed names.
    for e in shared_caste_list["entries"]:
        if " - " in e["name"]:
            assert e["umbrella"] == e["name"].split(" - ")[0], e
        else:
            assert e["umbrella"] == "", e


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize("wanted, theirs, fits", PREF_FITS)
def test_pref_fits(wanted, theirs, fits):
    assert fe.pref_fits(wanted, theirs) is fits


@pytest.mark.parametrize("wanted, theirs, fits", CASTE_FITS)
def test_caste_fits(wanted, theirs, fits):
    assert fe.caste_fits(wanted, theirs) is fits


@pytest.mark.parametrize("mine, theirs, fits", MANGLIK_FITS)
def test_manglik_fits(mine, theirs, fits):
    assert fe.manglik_fits(mine, theirs) is fits


@pytest.mark.parametrize("code, min_lakh, fits", INCOME_AT_LEAST)
def test_income_at_least(code, min_lakh, fits):
    assert fe.income_at_least(code, min_lakh) is fits


@pytest.mark.parametrize("stored, normal", NORMAL_MIN_LAKH)
def test_normal_min_lakh(stored, normal):
    assert fe.normal_min_lakh(stored) == normal
    assert fe.prefs_from_doc({"pref_incomeMinLakh": stored})["incomeMinLakh"] == normal


@pytest.mark.parametrize("children_ok, theirs, fits", CHILDREN_FIT)
def test_children_fit(children_ok, theirs, fits):
    me = fe.person_from_docs("me", {}, {"pref_childrenOk": children_ok})
    them = fe.person_from_docs("them", {"children": theirs})
    assert ("children" not in fe.pref_misses(me.prefs, them, "")) is fits


def test_labels_cover_every_token_in_order():
    assert list(fe.PREF_LABELS.values()) == ["Religion", "Mother tongue", "Caste", "Manglik", "Marital status",
                                             "Children", "Food", "Education", "Work", "Height", "Income"]
    assert set(fe.STRICT_TOKENS) | set(fe.ALWAYS_MUST) | {"professions", "height"} == set(fe.PREF_LABELS)


# ------------------------------------------------------------------ both ways
def _person(uid, side):
    return fe.person_from_docs(uid, dict(side.get("card") or {}), dict(side.get("profile") or {}))


@pytest.mark.parametrize("v", GATE, ids=[v["case"] for v in GATE])
def test_pref_gate(v):
    me, them = _person("me", v["me"]), _person("them", v["them"])
    assert fe._pref_gate(me, them) == v["expect"]


@pytest.mark.parametrize("v", PREF_GATE_VECTORS, ids=[v["name"] for v in PREF_GATE_VECTORS])
def test_the_shared_ts_vectors(v):
    me = fe.person_from_docs("me", dict(v["me"]), dict(v["mine"]))
    them = fe.person_from_docs("them", dict(v["them"]), dict(v["theirs"]))
    assert fe._pref_gate(me, them) == v["want"]


def test_the_shared_ts_vectors_are_well_formed():
    names = [v["name"] for v in PREF_GATE_VECTORS]
    assert len(names) == len(set(names)) == 36
    for v in PREF_GATE_VECTORS:
        assert list(v) == ["name", "mine", "me", "theirs", "them", "want"], v["name"]


@pytest.mark.parametrize("typed, rank", EDUCATION_RANK)
def test_education_rank(typed, rank):
    assert fe.education_rank(typed) == rank


def test_a_curly_apostrophe_education_is_a_must_have_like_the_straight_one():
    """“Bachelor’s” typed on a phone is rank 3, so a marked Master's minimum hides it (an unknown rank never would)."""
    me = fe.person_from_docs("me", {}, {"pref_educationMin": 4, "pref_strict": ["education"]})
    for typed in ("Bachelor's", "Bachelor’s"):
        assert fe._pref_gate(me, fe.person_from_docs("them", {"education": typed})) == "pref_education", typed
        assert "education" in fe.pref_misses(me.prefs, fe.person_from_docs("them", {"education": typed}), "")


def test_the_vectors_are_plain_json():
    """So C# and TS can copy them as they are."""
    blob = json.dumps({"prefFits": PREF_FITS, "casteFits": CASTE_FITS, "manglikFits": MANGLIK_FITS,
                       "incomeAtLeast": INCOME_AT_LEAST, "normalMinLakh": NORMAL_MIN_LAKH,
                       "childrenFit": CHILDREN_FIT, "gate": GATE, "prefGateVectors": PREF_GATE_VECTORS,
                       "educationRank": EDUCATION_RANK}, ensure_ascii=False)
    back = json.loads(blob)
    assert back["gate"][0]["case"] == "nothing set"
    assert back["prefGateVectors"] == PREF_GATE_VECTORS and back["educationRank"] == EDUCATION_RANK
