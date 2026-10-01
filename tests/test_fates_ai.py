import json

import fates_ai


def fake(reply):
    def complete(messages, max_tokens, temperature, kind):
        fake.calls.append((messages, kind))
        if isinstance(reply, Exception):
            raise reply
        return reply, "stop"
    fake.calls = []
    return complete


FACTS = {
    "path": "stars", "their_first_name": "Anaya", "pronoun": "she", "their_city": "Bengaluru",
    "shared_interests": ["Trekking"], "home": [{"topic": "Marriage and plans", "status": "aligns"},
                                               {"topic": "Food", "status": "talk"}],
    "stars": {"gunas": 27, "of": 36, "kootas": [], "doshas": []},
}


def test_extract_json_handles_fences_and_noise():
    assert fates_ai.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert fates_ai.extract_json('Sure! {"a": 2} hope it helps') == {"a": 2}
    assert fates_ai.extract_json("no json") is None
    assert fates_ai.extract_json('{"a": }') is None


def test_verdict_accepts_valid_reply():
    reply = json.dumps({
        "verdict": "Anaya is a calm match for you. Your kundli gives 27 of 36 gunas and you both want to marry soon.",
        "strengths": ["Same plans", "Both love trekking", "Calm together"],
        "talk_about": ["Food at home"],
        "openers": ["Nandi Hills or Skandagiri for a first trek?", "What does a good Sunday look like?", "Filter coffee first?"],
        "date_idea": "A Sunday morning walk in Cubbon Park, then filter coffee nearby.",
    })
    v = fates_ai.verdict(fake(reply), FACTS)
    assert v and v["by"] == "ai" and len(v["openers"]) == 3 and v["dateIdea"]


def test_verdict_rejects_invented_guna_numbers():
    reply = json.dumps({"verdict": "You share 32 of 36 gunas, a rare match.", "strengths": [], "openers": []})
    assert fates_ai.verdict(fake(reply), FACTS) is None


def test_verdict_rejects_banned_topics_and_bad_json():
    bad = json.dumps({"verdict": "Her caste fits yours well.", "strengths": [], "openers": []})
    assert fates_ai.verdict(fake(bad), FACTS) is None
    assert fates_ai.verdict(fake("not json at all"), FACTS) is None
    assert fates_ai.verdict(fake(RuntimeError("down")), FACTS) is None


def test_verdict_drops_bad_items_but_keeps_good_ones():
    reply = json.dumps({
        "verdict": "Anaya fits your plans well. Start with trekking.",
        "strengths": ["Same plans", "x" * 80],
        "openers": ["Trek this Sunday? Or next? Or never?", "What got you into trekking?"],
        "date_idea": "🌸 " + "a" * 200,
    })
    v = fates_ai.verdict(fake(reply), FACTS)
    assert v["strengths"] == ["Same plans"]
    assert v["openers"] == ["What got you into trekking?"]
    assert v["dateIdea"] is None


def test_template_verdict_is_grounded():
    t = fates_ai.template_verdict(FACTS)
    assert "27 of 36" in t["verdict"] and "Anaya" in t["verdict"] and t["by"] == "template"
    assert "food" in " ".join(t["talkAbout"]).lower()


def test_pick_and_hooks_validates_indexes_and_names():
    options = {
        "stars": [{"their_first_name": "Anaya", "stars": {"gunas": 27}}, {"their_first_name": "Riya", "stars": {"gunas": 25}}],
        "home": [{"their_first_name": "Tanisha"}],
    }
    reply = json.dumps({"paths": [
        {"path": "stars", "pick": 1, "hook": "You both love the hills, 25 of 36 gunas.", "why": "Riya shares your plans."},
        {"path": "home", "pick": 3, "hook": "x", "why": "y"},
    ]})
    out = fates_ai.pick_and_hooks(fake(reply), {"first_name": "Arjun"}, options)
    assert out["stars"]["pick"] == 1 and out["stars"]["hook"]
    assert "home" not in out


def test_pick_hook_with_name_or_wrong_gunas_is_dropped():
    options = {"stars": [{"their_first_name": "Anaya", "stars": {"gunas": 27}}]}
    reply = json.dumps({"paths": [{"path": "stars", "pick": 0, "hook": "Anaya and you: 30 gunas!", "why": "ok"}]})
    out = fates_ai.pick_and_hooks(fake(reply), {}, options)
    assert out["stars"]["pick"] == 0 and out["stars"]["hook"] is None


def test_pick_and_hooks_survives_failures():
    assert fates_ai.pick_and_hooks(fake(RuntimeError("x")), {}, {"stars": [{}]}) == {}
    assert fates_ai.pick_and_hooks(fake("{}"), {}, {}) == {}


def test_template_verdict_reads_naturally():
    facts = {**FACTS, "pronoun": "they", "shared_interests": ["Bollywood", "Trekking"], "home": [
        {"topic": "Marriage and plans", "status": "aligns"}, {"topic": "Children", "status": "aligns"},
        {"topic": "Smoking", "status": "aligns"}, {"topic": "Languages", "status": "aligns"},
        {"topic": "Moving cities", "status": "talk"}, {"topic": "Drinking", "status": "differs"}]}
    t = fates_ai.template_verdict(facts)
    assert "You agree on marriage plans and children." in t["verdict"]
    assert "Talk early about where to live and drinking." in t["verdict"]
    assert "Start with Bollywood: they like it too." in t["verdict"]
    assert t["strengths"] == ["Both into Bollywood", "Both into trekking", "Agree on marriage plans"]


def test_template_verdict_is_honest_about_a_weak_kundli():
    facts = {**FACTS, "stars": {"gunas": 14.5, "of": 36, "kootas": [], "doshas": [
        {"name": "Nadi dosha", "present": True, "cancelled": False}, {"name": "Manglik", "present": True, "cancelled": True}]}}
    t = fates_ai.template_verdict(facts)
    assert "steady match" not in t["verdict"]
    assert "14.5 of 36 gunas, below the usual 18" in t["verdict"]
    assert "ask a pandit about the Nadi dosha." in t["verdict"] and "Manglik" not in t["verdict"]


def test_template_verdict_opens_with_a_chooser():
    t = fates_ai.template_verdict({**FACTS, "they_chose_you": True})
    assert t["verdict"].startswith("Anaya already chose you as a fate. Accept, and it’s a match.")
    low = fates_ai.template_verdict({**FACTS, "they_chose_you": True, "stars": {"gunas": 12, "of": 36, "kootas": [], "doshas": []}})
    assert low["verdict"].startswith("Anaya already chose you") and "below the usual 18" in low["verdict"]
