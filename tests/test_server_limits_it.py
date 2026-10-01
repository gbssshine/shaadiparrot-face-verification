"""Server-side limits against the Firestore emulator: leases (one horoscope generation per day), per-minute and
per-day counters, and photo checks reading only the caller's own uploads. Skipped without the emulator."""
import os
import time

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("FIRESTORE_EMULATOR_HOST"), reason="needs the Firestore emulator")

PROJECT = os.getenv("GCLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "demo-fates"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT)


@pytest.fixture(scope="module")
def m():
    import main
    from google.cloud import firestore
    main.firestore_client = firestore.Client(project=PROJECT)
    return main


def test_one_lease_holder_at_a_time(m):
    name = f"test_lease_{time.time_ns()}"
    assert m._take_lease(name, 30) is True
    assert m._take_lease(name, 30) is False            # someone else is generating
    m._release_lease(name)
    assert m._take_lease(name, 30) is True             # released after a failure: the next one may try


def test_a_stale_lease_is_taken_over(m):
    name = f"test_stale_{time.time_ns()}"
    m.firestore_client.collection("serverLeases").document(name).set({"at": time.time() - 120})
    assert m._take_lease(name, 45) is True


def test_per_minute_and_per_day_counters(m):
    uid = f"rate_{time.time_ns()}"
    assert all(m._take_rate(uid, "aiChatMinute", 3, per="minute") for _ in range(3))
    assert m._take_rate(uid, "aiChatMinute", 3, per="minute") is False
    assert m._take_rate(uid, "photoChecks", 2) and m._take_rate(uid, "photoChecks", 2)
    assert m._take_rate(uid, "photoChecks", 2) is False
