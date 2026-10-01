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


def test_only_the_users_own_verification_uploads():
    hosts = ("firebasestorage.googleapis.com",)
    good = "https://firebasestorage.googleapis.com/v0/b/app.appspot.com/o/users%2Fu1%2Fverification%2Fa.jpg?alt=media&token=t"
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



def test_photo_checks_only_read_the_callers_own_uploads():
    import main
    assert main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u1/photos/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u2/photos/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://shaadiparrot.firebasestorage.app/users/u1/../u2/a.jpg", "u1")
    assert not main._own_gcs_photo("gs://other-bucket/private/a.jpg", "u1")
    assert not main._own_gcs_photo("https://example.com/users/u1/a.jpg", "u1")
