"""Daily Fates: the AI layer. DeepSeek chooses among the engine's shortlist and writes the words.

Everything here is guarded: the model only sees computed facts, must answer in JSON, and any reply
that breaks the rules (bad JSON, invented guna numbers, too long) is thrown away for a template.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

Complete = Callable[[List[Dict[str, str]], int, float, str], Tuple[str, str]]

PICK_SYSTEM = (
    "You are Mithu, the parrot matchmaker inside Shaadi Parrot, an Indian dating app for people who want "
    "something real. Each morning you bring a person up to three 'paths': Stars (kundli), Heart (personality "
    "tests) and Home (the life they want). For every path you get up to 3 candidates that already passed all "
    "filters, with computed facts. Choose the ONE candidate per path who is the best fate for the viewer: "
    "weigh the facts, the bios and how naturally they would get on. Then write:\n"
    "- hook: a teaser for a closed card, max 70 characters, no names, second person ('You both ...'), "
    "grounded in one concrete fact of that path.\n"
    "- why: why Mithu picked this person, max 130 characters, warm and specific, may use the first name.\n"
    "Rules: use only the facts given; never invent numbers, places, jobs or hobbies; never mention caste, "
    "income, looks or body; no emojis; English only.\n"
    "Answer with JSON only: {\"paths\":[{\"path\":\"stars\",\"pick\":0,\"hook\":\"...\",\"why\":\"...\"}]}"
)

VERDICT_SYSTEM = (
    "You are Mithu, the warm, honest parrot matchmaker inside Shaadi Parrot, an Indian dating app. Write the "
    "verdict for ONE pair from the computed facts. Speak to the viewer in the second person, about the other "
    "person by first name and the given pronoun. Be specific and kind, and name one real thing to talk about.\n"
    "Return JSON only with these keys:\n"
    "verdict: 3-5 sentences, max 75 words;\n"
    "strengths: 3 short phrases (max 5 words each);\n"
    "talk_about: 0-2 short phrases (max 5 words each), only from facts marked talk or differs;\n"
    "openers: 3 first messages the viewer could send, each max 90 characters, natural and specific, "
    "at most one question mark each;\n"
    "date_idea: one first-date idea in their city, max 120 characters, simple and safe (public place, daytime).\n"
    "Rules: use only the facts given; if you mention gunas use exactly the given number; never invent jobs, "
    "places, hobbies or family details; never mention caste, income, looks or body; never promise outcomes; "
    "no emojis; English only. If they_chose_you is true, this person already accepted the viewer as their fate: "
    "say so warmly in the first sentence (accepting back makes it a match), without pressure."
)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(t[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def clean_line(v: Any, max_len: int) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = _EMOJI.sub("", " ".join(v.split())).strip().strip('"')
    if not s or len(s) > max_len:
        return None
    return s


def _numbers_ok(text: str, allowed: set) -> bool:
    """Every 'N of 36' / 'N gunas' in the text must be the real total."""
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:of 36|/36|gunas?)", text, flags=re.I):
        if float(m.group(1)) not in allowed:
            return False
    return True


_BANNED = re.compile(r"\b(caste|salary|income|sexy|hot body|fair skin|dowry)\b", re.I)


def _safe(text: str, allowed_gunas: set) -> bool:
    return bool(text) and not _BANNED.search(text) and _numbers_ok(text, allowed_gunas)


def pick_and_hooks(complete: Complete, viewer_facts: Dict[str, Any],
                   options: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Returns {path: {"pick": i, "hook": str|None, "why": str|None}} for the paths the AI answered well."""
    if not any(options.values()):
        return {}
    payload = {"viewer": viewer_facts, "paths": {p: [dict(o, index=i) for i, o in enumerate(opts)] for p, opts in options.items() if opts}}
    msgs = [{"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    try:
        text, _ = complete(msgs, 700, 0.5, "fates_pick")
    except Exception:
        return {}
    data = extract_json(text) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("paths") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        opts = options.get(path) or []
        try:
            idx = int(item.get("pick"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(opts)):
            continue
        gunas = {float(opts[idx]["stars"]["gunas"])} if opts[idx].get("stars") else set()
        hook = clean_line(item.get("hook"), 80)
        why = clean_line(item.get("why"), 150)
        out[path] = {
            "pick": idx,
            "hook": hook if hook and _safe(hook, gunas) and opts[idx].get("their_first_name", "") not in hook else None,
            "why": why if why and _safe(why, gunas) else None,
        }
    return out


def verdict(complete: Complete, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    msgs = [{"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, separators=(",", ":"))}]
    try:
        text, _ = complete(msgs, 650, 0.7, "fates_verdict")
    except Exception:
        return None
    data = extract_json(text)
    if not data:
        return None
    gunas = {float(facts["stars"]["gunas"])} if facts.get("stars") else set()
    v = clean_line(data.get("verdict"), 600)
    if not v or len(v.split()) > 90 or not _safe(v, gunas):
        return None

    def phrases(key: str, n: int, max_len: int) -> List[str]:
        items = data.get(key) if isinstance(data.get(key), list) else []
        res = []
        for x in items:
            s = clean_line(x, max_len)
            if s and _safe(s, gunas):
                res.append(s)
        return res[:n]

    openers = [o for o in phrases("openers", 3, 100) if o.count("?") <= 1]
    date_idea = clean_line(data.get("date_idea"), 140)
    return {
        "verdict": v,
        "strengths": phrases("strengths", 3, 40),
        "talkAbout": phrases("talk_about", 2, 40),
        "openers": openers,
        "dateIdea": date_idea if date_idea and _safe(date_idea, gunas) else None,
        "by": "ai",
    }


# Home topics as they read inside a sentence ("You agree on ...", "Talk early about ...").
AGREE_NOUNS = {"Marriage and plans": "marriage plans", "Children": "children", "Faith": "faith", "Community": "community",
               "Moving cities": "where to live", "Smoking": "habits", "Drinking": "habits"}
TALK_NOUNS = {"Marriage and plans": "what you each want", "Children": "children", "Faith": "faith", "Community": "community",
              "Moving cities": "where to live", "Smoking": "smoking", "Drinking": "drinking", "Languages": "the language at home"}


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def template_verdict(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Used when the AI is unavailable or its answer failed validation."""
    from fates_engine import _interest_in_sentence
    name = facts.get("their_first_name") or "They"
    subj = facts.get("pronoun") or "they"
    home = facts.get("home", [])
    aligned = _unique([AGREE_NOUNS.get(h["topic"], h["topic"].lower()) for h in home
                       if h["status"] == "aligns" and h["topic"] != "Languages"])[:2]
    share_language = any(h["topic"] == "Languages" and h["status"] == "aligns" for h in home)
    talks = _unique([TALK_NOUNS.get(h["topic"], h["topic"].lower()) for h in home if h["status"] in ("talk", "differs")])[:2]
    parts = [f"{name} already chose you as a fate. Accept, and it’s a match." if facts.get("they_chose_you")
             else f"{name} looks like a steady match for you."]
    stars = facts.get("stars")
    if stars:
        gunas = stars["gunas"]
        open_doshas = [d["name"] for d in stars.get("doshas", []) if d.get("present") and not d.get("cancelled")]
        if gunas >= 18:
            parts.append(f"Your kundli gives {gunas:g} of 36 gunas.")
        else:
            # Honest about a weak kundli: the pick came from the heart or the home, not the stars.
            if not facts.get("they_chose_you"):
                parts[0] = f"{name} fits you in heart and in the life you want."
            parts.append(f"The kundli gives {gunas:g} of 36 gunas, below the usual 18.")
        dosha_note = f"If the stars matter to your family, ask a pandit about the {_join(open_doshas)}." if open_doshas else ""
    else:
        dosha_note = ""
    if aligned:
        parts.append(f"You agree on {_join(aligned)}" + (" and share a language." if share_language and len(aligned) < 2 else "."))
    elif share_language:
        parts.append("You share a language.")
    if talks:
        parts.append(f"Talk early about {_join(talks)}.")
    if dosha_note:
        parts.append(dosha_note)
    shared = [_interest_in_sentence(s) for s in facts.get("shared_interests") or [] if s.strip()]
    if shared:
        parts.append(f"Start with {shared[0]}: {subj} {'like' if subj == 'they' else 'likes'} it too.")
    strengths = [f"Both into {s}" for s in shared[:2]] + [f"Agree on {a}" for a in aligned]
    return {
        "verdict": " ".join(parts),
        "strengths": strengths[:3],
        "talkAbout": talks,
        "openers": ([f"I saw you like {shared[0]} too. What got you into it?"] if shared else [])
                   + ["What does a perfect Sunday look like for you?"],
        "dateIdea": None,
        "by": "template",
    }
