from types import SimpleNamespace as NS

import face_live as fl


def face(pan=0.0, size=300, conf=0.9, roll=0.0, tilt=0.0, x=0):
    verts = [NS(x=x, y=0), NS(x=x + size, y=0), NS(x=x + size, y=size), NS(x=x, y=size)]
    return NS(pan_angle=pan, roll_angle=roll, tilt_angle=tilt, detection_confidence=conf, bounding_poly=NS(vertices=verts))


SAFE = ("VERY_UNLIKELY", "UNLIKELY", "VERY_UNLIKELY")


def shots(*pans):
    return [(*SAFE, [face(p)]) for p in pans]


def test_three_real_poses_pass():
    ok, reason, pans = fl.judge(shots(3, 24, -21))
    assert ok and reason == "ok" and pans == [3, 24, -21]


def test_the_same_photo_three_times_fails():
    assert fl.judge(shots(2, 2, 2))[1] == "turn_your_head_more"
    assert fl.judge(shots(2, 22, 25))[1] == "turn_to_both_sides"
    assert fl.judge(shots(30, 22, -25))[1] == "look_straight_first"


def test_faces_must_be_clear_single_and_safe():
    assert fl.judge([(*SAFE, [])] + shots(20, -20))[1] == "no_face_detected_1"
    two = [(*SAFE, [face(0), face(0, size=280, x=400)])] + shots(20, -20)
    assert fl.judge(two)[1] == "more_than_one_face_1"
    assert fl.judge([(*SAFE, [face(0, size=50)])] + shots(20, -20))[1] == "face_too_small_1"
    assert fl.judge([("VERY_LIKELY", "UNLIKELY", "VERY_UNLIKELY", [face(0)])] + shots(20, -20))[1] == "adult_content"
    assert fl.judge(shots(0, 20))[1] == "three_selfies_needed"


def test_only_the_users_own_verification_uploads(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")          # the emulator's demo project (demo.appspot.com)
    hosts = ("firebasestorage.googleapis.com",)
    good = "https://firebasestorage.googleapis.com/v0/b/shaadiparrot.firebasestorage.app/o/users%2Fu1%2Fverification%2Fa.jpg?alt=media&token=t"
    assert fl.selfie_url_ok(good, "u1", hosts)
    assert not fl.selfie_url_ok(good, "u2", hosts)                                                  # someone else's
    assert not fl.selfie_url_ok(good.replace("verification", "photos"), "u1", hosts)                # a profile photo
    assert not fl.selfie_url_ok("https://example.com/users/u1/verification/a.jpg", "u1", hosts)    # not Storage
    assert not fl.selfie_url_ok(good.replace("https", "http"), "u1", hosts)                         # prod must be https
    emu = "http://10.0.2.2:9199/v0/b/demo.appspot.com/o/users%2Fu1%2Fverification%2Fa.jpg?alt=media"
    assert fl.selfie_url_ok(emu, "u1", ("10.0.2.2:9199",))



def smiley(pan=0.0, joy=5):
    f = face(pan)
    f.joy_likelihood = joy
    return f


def test_challenges_are_random_and_well_formed():
    import random
    seen = set()
    for seed in range(40):
        c = fl.new_challenge(random.Random(seed))
        assert c[0] == "front" and len(c) == 3 and len(set(c)) == 3 and set(c[1:]) <= set(fl.STEP_POOL)
        seen.add(tuple(c))
    assert len(seen) == 6                                   # every order of two of three steps shows up


def test_a_challenge_is_judged_step_by_step():
    ok = fl.judge_challenge(["front", "smile", "left"], [(*SAFE, [face(2)]), (*SAFE, [smiley(4)]), (*SAFE, [face(-23)])])
    assert ok[0] and ok[1] == "ok"
    assert fl.judge_challenge(["front", "smile", "left"], [(*SAFE, [face(2)]), (*SAFE, [smiley(4, joy=2)]), (*SAFE, [face(-23)])])[1] == "smile_please"
    assert fl.judge_challenge(["front", "right", "left"], [(*SAFE, [face(2)]), (*SAFE, [face(22)]), (*SAFE, [face(20)])])[1] == "turn_to_both_sides"
    assert fl.judge_challenge(["front", "right", "smile"], [(*SAFE, [face(2)]), (*SAFE, [face(4)]), (*SAFE, [smiley()])])[1] == "turn_your_head_more"
    assert fl.judge_challenge(["front", "right"], shots(0, 20, -20))[1] == "challenge_mismatch"


def test_storage_object_from_a_download_url():
    url = "https://firebasestorage.googleapis.com/v0/b/app.appspot.com/o/users%2Fu1%2Fverification%2Fa.jpg?alt=media&token=t"
    assert fl.storage_object(url) == ("app.appspot.com", "users/u1/verification/a.jpg")
    assert fl.storage_object("http://127.0.0.1:9199/v0/b/demo.appspot.com/o/users%2Fu1%2Fx.jpg") == ("demo.appspot.com", "users/u1/x.jpg")
    assert fl.storage_object("https://example.com/a.jpg") is None
    assert fl.storage_object("https://firebasestorage.googleapis.com/v0/b/app/o/users%2F..%2Fx") is None



def _fb(path: str, host: str = "firebasestorage.googleapis.com", scheme: str = "https") -> str:
    from urllib.parse import quote
    return f"{scheme}://{host}/v0/b/shaadiparrot.firebasestorage.app/o/{quote(path, safe='')}?alt=media&token=t"


def test_own_photo_url_takes_only_the_users_own_photos_folder(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")          # the emulator's demo project (demo.appspot.com)
    hosts = ("firebasestorage.googleapis.com", "storage.googleapis.com")
    assert fl.own_photo_url(_fb("users/u1/photos/a.jpg"), "u1", hosts)
    assert fl.own_photo_url("https://storage.googleapis.com/shaadiparrot.firebasestorage.app/users/u1/photos/a.jpg", "u1", hosts)
    assert not fl.own_photo_url(_fb("users/u1/photos/a.jpg"), "u2", hosts)                 # someone else's
    assert not fl.own_photo_url(_fb("users/u1/blur/a.jpg"), "u1", hosts)                   # a server-made blur
    assert not fl.own_photo_url(_fb("users/u1/variants/abc_480.jpg"), "u1", hosts)         # a resized variant
    assert not fl.own_photo_url(_fb("users/u1/verification/a.jpg"), "u1", hosts)           # a selfie
    assert not fl.own_photo_url(_fb("users/u1/a.jpg"), "u1", hosts)                        # not in photos/
    assert not fl.own_photo_url(_fb("users/u2/photos/users/u1/photos/a.jpg"), "u1", hosts)  # nested in u2's folder
    assert not fl.own_photo_url(_fb("users/u1/photos/../../u2/a.jpg"), "u1", hosts)
    assert not fl.own_photo_url("https://photos.shaadiparrot.com/users/u1/photos/a.jpg", "u1", hosts)  # not Storage
    assert not fl.own_photo_url(_fb("users/u1/photos/a.jpg", scheme="http"), "u1", hosts)  # prod must be https
    assert not fl.own_photo_url(_fb("users/u1/photos/a.jpg"), "", hosts)
    emu = "http://10.0.2.2:9199/v0/b/demo.appspot.com/o/users%2Fu1%2Fphotos%2Fa.jpg?alt=media"
    assert fl.own_photo_url(emu, "u1", ("10.0.2.2:9199",))


def test_photo_checks_only_read_the_callers_own_uploads():
    import main
    assert main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u1/photos/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u2/photos/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u1/../u2/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://other-bucket/private/a.jpg", "u1")
    assert not main._own_gcs_photo("https://example.com/users/u1/a.jpg", "u1")


# ------------------------------------------------------------------ only the project's own buckets
PROD = "shaadiparrot.firebasestorage.app"
FOREIGN = "attacker-photos.appspot.com"
ALL_HOSTS = ("firebasestorage.googleapis.com", "storage.googleapis.com", "127.0.0.1:9199", "10.0.2.2:9199")


def _forms(bucket: str, obj: str):
    """Every URL form a Storage object can be linked by: the Firebase download link on both Google hosts, the plain
    storage.googleapis.com/{bucket}/{object} link, and the local emulator's download link (this PC and the Android
    emulator's name for it)."""
    from urllib.parse import quote
    enc = quote(obj, safe="")
    return [f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{enc}?alt=media&token=t",
            f"https://storage.googleapis.com/v0/b/{bucket}/o/{enc}?alt=media",
            f"https://storage.googleapis.com/{bucket}/{obj}",
            f"http://127.0.0.1:9199/v0/b/{bucket}/o/{enc}?alt=media",
            f"http://10.0.2.2:9199/v0/b/{bucket}/o/{enc}?alt=media"]


def _no_bucket_env(monkeypatch):
    for k in ("FIREBASE_STORAGE_BUCKET", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        monkeypatch.delenv(k, raising=False)


def test_an_own_photo_lives_in_the_projects_own_bucket_in_every_url_form(monkeypatch):
    """users/{uid}/photos/ in someone else's bucket is not the user's photo, whatever the link looks like: the
    server never fetches it, checks a face against it or shows it."""
    _no_bucket_env(monkeypatch)
    obj = "users/u1/photos/a.jpg"
    for url in _forms(PROD, obj) + _forms("shaadiparrot.appspot.com", obj):
        assert fl.own_photo_url(url, "u1", ALL_HOSTS), url
    for bucket in (FOREIGN, "shaadiparrot.firebasestorage.app.evil.com", "evil-shaadiparrot.firebasestorage.app",
                   "SHAADIPARROT.FIREBASESTORAGE.APP%2F..%2Fx", ""):
        for url in _forms(bucket, obj):
            assert not fl.own_photo_url(url, "u1", ALL_HOSTS), url
            assert fl.own_photo_object(url, "u1", ALL_HOSTS) is None, url


def test_the_bucket_env_and_the_emulator_project_add_buckets(monkeypatch):
    _no_bucket_env(monkeypatch)
    obj = "users/u1/photos/a.jpg"
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "gs://renamed-bucket/")
    assert all(fl.own_photo_url(u, "u1", ALL_HOSTS) for u in _forms("renamed-bucket", obj))
    assert all(fl.own_photo_url(u, "u1", ALL_HOSTS) for u in _forms(PROD, obj))          # the project's, always
    assert not any(fl.own_photo_url(u, "u1", ALL_HOSTS) for u in _forms(FOREIGN, obj))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-fates")                              # the emulator tests' project
    assert all(fl.own_photo_url(u, "u1", ALL_HOSTS) for u in _forms("demo-fates.appspot.com", obj))
    assert not any(fl.own_photo_url(u, "u1", ALL_HOSTS) for u in _forms(FOREIGN, obj))


def test_own_photo_object_is_the_object_path(monkeypatch):
    _no_bucket_env(monkeypatch)
    for url in _forms(PROD, "users/u1/photos/a b.jpg"):
        assert fl.own_photo_object(url, "u1", ALL_HOSTS) == "users/u1/photos/a b.jpg"
    # An unencoded object in a download link is not one Firebase serves (media.ts storageRef reads one segment).
    assert fl.own_photo_object(f"https://firebasestorage.googleapis.com/v0/b/{PROD}/o/users/u1/photos/a.jpg", "u1",
                               ALL_HOSTS) is None
    assert fl.own_photo_object(f"https://firebasestorage.googleapis.com/v0/b/{PROD}/o/users%2Fu1%2Fphotos%2F%2Fa.jpg",
                               "u1", ALL_HOSTS) is None                                     # an empty segment


def test_selfies_only_in_the_projects_own_bucket(monkeypatch):
    _no_bucket_env(monkeypatch)
    obj = "users/u1/verification/a.jpg"
    assert all(fl.selfie_url_ok(u, "u1", ALL_HOSTS) for u in _forms(PROD, obj))
    assert not any(fl.selfie_url_ok(u, "u1", ALL_HOSTS) for u in _forms(FOREIGN, obj))
    # The folder must be the object's own first folder, not text anywhere in the link.
    assert not fl.selfie_url_ok(f"https://firebasestorage.googleapis.com/v0/b/{PROD}/o/x%2Fusers%2Fu1%2Fverification%2Fa.jpg",
                                "u1", ALL_HOSTS)


def test_thumbnails_and_photo_checks_never_read_a_foreign_bucket(monkeypatch):
    import fates_api
    import main
    _no_bucket_env(monkeypatch)
    for url in _forms(FOREIGN, "users/u1/photos/a.jpg"):
        assert not fates_api.thumb_source_ok(url, "u1"), url
    assert fates_api.thumb_source_ok(_forms(PROD, "users/u1/photos/a.jpg")[0], "u1")
    assert not fates_api.thumb_source_ok(_forms(PROD, "users/u1/photos/a.jpg")[0], "u2")    # someone else's photo
    assert main._own_gcs_photo(f"gs://{PROD}/users/u1/photos/a.jpg", "u1")
    assert not main._own_gcs_photo(f"gs://{FOREIGN}/users/u1/photos/a.jpg", "u1")


def test_fates_never_shows_or_checks_a_photo_in_a_foreign_bucket(monkeypatch):
    import fates_engine as fe
    _no_bucket_env(monkeypatch)
    own = _forms(PROD, "users/u1/photos/a.jpg")[0]
    foreign = [u for u in _forms(FOREIGN, "users/u1/photos/b.jpg") if u.startswith("https")]
    pub = {"gender": "Female", "isDiscoverable": True, "photos": foreign + [own]}
    assert fe.own_private_photos("u1", pub) == [own]
    p = fe.person_from_docs("u1", pub, {"faceVerified": True, "faceVerifiedPhotos": foreign + [own]})
    assert p.photos == [own] and p.face_verified


def test_the_biodata_photo_is_never_read_from_a_foreign_bucket(monkeypatch):
    import biodata_pdf as bp
    _no_bucket_env(monkeypatch)
    url = f"https://firebasestorage.googleapis.com/v0/b/{FOREIGN}/o/users%2Fme1%2Fvariants%2Fk_960.jpg?alt=media&token=t"
    assert not bp.allowed_photo_url(url, "me1")
    assert bp.allowed_photo_url(url.replace(FOREIGN, PROD), "me1")
