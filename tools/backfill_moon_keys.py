"""Backfill users/{uid}/feedState/moon for the rishta cards, so getRishtas can score kundli from day one.

    python tools/backfill_moon_keys.py            # dry run (default): counts only, writes nothing
    python tools/backfill_moon_keys.py --apply    # writes

Covers publicProfiles cards whose profile is not kundli-only (profileFor != "none") and has a birth date. The
keys and their sig are exactly what /astro/scores writes (astro_api._moon_sig over {**card, **profile},
fates_astro.moon_keys of the same chart), so a card that already has keys with the same sig is left alone.
Needs Application Default Credentials (gcloud auth application-default login) with Firestore access; the
project comes from --project, else GOOGLE_CLOUD_PROJECT, else "shaadiparrot".
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fates_astro  # noqa: E402
import fates_astro_chart as fc  # noqa: E402

CHUNK = 100       # cards read per getAll round
BATCH = 400       # writes per batch (Firestore allows 500)


def _sig(doc: Dict[str, Any]) -> str:
    import astro_api   # the one definition getRishtas mirrors
    return astro_api._moon_sig(doc)


def run(db: Any, resolve_place: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]], apply: bool = False,
        limit: Optional[int] = None, out: Callable[[str], None] = print) -> Counter:
    counts: Counter = Counter()
    at = datetime.now(timezone.utc).isoformat()
    cards = []
    for snap in db.collection("publicProfiles").stream():
        cards.append((snap.id, snap.to_dict() or {}))
        if limit and len(cards) >= limit:
            break
    counts["cards"] = len(cards)
    pending: List[Any] = []

    def flush() -> None:
        if not pending:
            return
        batch = db.batch()
        for ref, data in pending:
            batch.set(ref, data)
        batch.commit()
        counts["written"] += len(pending)
        pending.clear()

    for i in range(0, len(cards), CHUNK):
        chunk = cards[i:i + CHUNK]
        refs = []
        for uid, _ in chunk:
            refs.append(db.collection("profiles").document(uid))
            refs.append(db.collection("users").document(uid).collection("feedState").document("moon"))
        got = {s.reference.path: (s.to_dict() or {}) for s in db.get_all(refs) if s.exists}
        for uid, public in chunk:
            profile = got.get(f"profiles/{uid}")
            if profile is None:
                counts["no_profile"] += 1
                continue
            if any(str(d.get("profileFor") or "").strip().lower() == "none" for d in (profile, public)):
                counts["kundli_only"] += 1
                continue
            doc = {**public, **profile}
            if not fc._parse_date(doc.get("birthDate")):
                counts["no_birth_date"] += 1
                continue
            sig = _sig(doc)
            if got.get(f"users/{uid}/feedState/moon", {}).get("sig") == sig:
                counts["up_to_date"] += 1
                continue
            try:
                chart = fates_astro.chart_for_profile(doc, resolve_place)
                data = {**fates_astro.moon_keys(chart), "sig": sig, "atIso": at}
            except Exception as e:   # one odd profile never stops the run
                counts["errors"] += 1
                out(f"  {uid}: {type(e).__name__}")
                continue
            counts["due"] += 1
            if apply:
                pending.append((db.collection("users").document(uid).collection("feedState").document("moon"), data))
                if len(pending) >= BATCH:
                    flush()
    if apply:
        flush()
    out(("APPLIED" if apply else "DRY RUN (nothing written; add --apply to write)") + ": " +
        ", ".join(f"{k} {counts[k]}" for k in ("cards", "due", "written", "up_to_date", "kundli_only", "no_birth_date",
                                               "no_profile", "errors")))
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill the Moon keys getRishtas scores kundli with.")
    ap.add_argument("--apply", action="store_true", help="write (the default is a dry run)")
    ap.add_argument("--dry-run", action="store_true", help="count only (the default)")
    ap.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT") or "shaadiparrot")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cards (for a trial)")
    args = ap.parse_args(argv)
    if args.apply and args.dry_run:
        ap.error("--apply and --dry-run together")
    from google.cloud import firestore

    import main as server   # the same birth-place lookup the server uses (places.sqlite3)
    db = firestore.Client(project=args.project)
    print(f"project {args.project}")
    run(db, server._resolve_birth_place, apply=args.apply, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
