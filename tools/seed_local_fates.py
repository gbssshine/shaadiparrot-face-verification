"""Local emulator only: gives the cloned test profiles varied birth data, tests and habits so Daily
Fates readings differ. Refuses to run unless FIRESTORE_EMULATOR_HOST is set."""
import json
import os
import sys
from datetime import datetime, timezone

if not os.getenv("FIRESTORE_EMULATOR_HOST"):
    sys.exit("FIRESTORE_EMULATOR_HOST is not set: refusing to touch a real database")

from google.cloud import firestore

db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "demo-shaadiparrot"))
cat = json.load(open(os.path.join(os.path.dirname(__file__), "..", "fates_tests_catalog.json"), encoding="utf-8"))["tests"]
by_cat = {}
for tid, m in cat.items():
    by_cat.setdefault(m["category"], tid)
now = datetime.now(timezone.utc).isoformat()

PEOPLE = {
    "iqvuqhxOKIbhwkHrkVtFU8nV4T5l": dict(birthDate="1996-11-20", birthTime="06:30", city="Bengaluru",
        tests={"attachment": 52, "love_style": 20, "communication": 36, "values": 40}, interests=["Trekking", "Coffee", "Cricket", "Indie music"],
        languages=["Hindi", "English", "Kannada"], want="Yes", relocate="No", smoking="Never", drinking="Socially",
        bio="Product engineer who treks on weekends and makes decent filter coffee."),
    "test_ananya": dict(birthDate="1999-05-02", birthTime="10:15", city="Bengaluru",
        tests={"attachment": 61, "love_style": 74, "communication": 70, "values": 46}, interests=["Trekking", "Coffee", "Painting", "Yoga"],
        languages=["Hindi", "English"], want="Yes", relocate="Yes", smoking="Never", drinking="Socially",
        bio="Designer, early riser, happiest on a hill at sunrise."),
    "test_meera": dict(birthDate="2000-01-14", birthTime="18:40", city="Mumbai",
        tests={"attachment": 58, "love_style": 25, "conflict": 40}, interests=["Coffee", "Board games", "Indie music"],
        languages=["English", "Marathi", "Hindi"], want="Yes", relocate="Only if we are a perfect match", smoking="Never", drinking="Rarely",
        bio="Architect. I show love with food and long walks."),
    "test_kavya": dict(birthDate="1998-03-09", birthTime=None, city="Chennai",
        tests={"family_focus": 80, "values": 30}, interests=["Cooking", "Temples", "Coffee"],
        languages=["Tamil", "English"], want="Yes", relocate="Yes", smoking="Never", drinking="Never",
        bio="Teacher, big family, bigger heart. Looking for something real."),
    "test_neha": dict(birthDate="1997-12-25", birthTime="07:05", city="Delhi",
        tests={"attachment": 30, "communication": 85}, interests=["Cricket", "Travel", "Indie music"],
        languages=["Hindi", "English", "Punjabi"], want="No", relocate="No", smoking="Socially", drinking="Regularly",
        bio="Marketing, travel, too many playlists."),
    "5EYurmlJQwujh3m8cPuovb1Ofz4Y": dict(birthDate="1999-08-22", birthTime="14:20", city="Pune",
        tests={"attachment": 55, "values": 44, "long_term": 70}, interests=["Trekking", "Books", "Coffee"],
        languages=["Hindi", "English", "Marathi"], want="Yes", relocate="Yes", smoking="Never", drinking="Socially",
        bio="Doctor in training. Weekends are for hills and books."),
}

for uid, p in PEOPLE.items():
    pub, priv = {}, {"birthDate": p["birthDate"], "birthCityName": p["city"], "birthCountryIso2": "IN", "birthCountryName": "India",
                     "settings_wantChildren": p["want"], "settings_relocate": p["relocate"],
                     "faceVerified": True, "isFaceVerified": True}   # Daily Fates is for verified people only
    if p["birthTime"]:
        priv["birthTime"] = p["birthTime"]
    for c, pct in p["tests"].items():
        tid = by_cat[c]
        for doc in (pub, priv):
            doc[f"tests_{tid}_scorePercent"] = pct
            doc[f"tests_{tid}_categoryId"] = c
            doc[f"tests_{tid}_resultKey"] = "low" if pct <= 35 else ("high" if pct >= 65 else "mid")
    pub.update({"interests": p["interests"], "languages": p["languages"], "smoking": p["smoking"], "drinking": p["drinking"],
                "bio": p["bio"], "religion": "Hindu", "cityName": "Bengaluru", "lat": 12.97, "lon": 77.60,
                "isFaceVerified": True})   # the public badge, as the server writes it together with the profile flags
    db.document(f"publicProfiles/{uid}").set(pub, merge=True)
    current = (db.document(f"publicProfiles/{uid}").get().to_dict() or {}).get("photos") or []
    priv["faceVerifiedPhotos"] = [p for p in current if isinstance(p, str)]
    db.document(f"profiles/{uid}").set({**priv, **pub}, merge=True)
    db.document(f"users/{uid}").set({"lastAppOpenAtUtcIso": now}, merge=True)
    print("seeded", uid)

# Start the day fresh for the viewer.
viewer = "iqvuqhxOKIbhwkHrkVtFU8nV4T5l"
for d in db.collection("dailyFates").stream():
    if d.id.startswith(viewer + "__"):
        d.reference.delete()
db.document(f"fatesMemory/{viewer}").delete()
print("reset today's fates for the viewer")
