import math

import pytest

import face_match as fm


def vec(angle_deg, dims=8):
    """A unit vector at `angle_deg` from the first axis: cosine to vec(0) is cos(angle)."""
    a = math.radians(angle_deg)
    v = [0.0] * dims
    v[0], v[1] = math.cos(a), math.sin(a)
    return v


ME = vec(0)
ME_SIDE = vec(55)            # ~0.57, a turned head
ME_PHOTO = vec(40)           # ~0.77
BORDER = vec(64)             # ~0.44, below the main-photo bar but above "same person"
STRANGER = vec(80)           # ~0.17


def test_similarity_is_cosine():
    assert fm.similarity(ME, ME) == pytest.approx(1.0)
    assert fm.similarity(ME, vec(90)) == pytest.approx(0.0, abs=1e-9)
    assert fm.similarity(ME, [0] * 8) == 0.0


def test_you_in_selfies_and_photos_pass():
    ok, reason, meta = fm.judge(ME, [ME_SIDE, ME_SIDE], [[ME_PHOTO], [ME_PHOTO], []])
    assert ok and reason == "ok"
    assert meta["photos"][2] is None                       # a photo without a face doesn't count


def test_someone_else_in_the_selfies_fails():
    ok, reason, _ = fm.judge(ME, [ME_SIDE, STRANGER], [[ME_PHOTO]])
    assert not ok and reason == "selfies_not_same_person"


def test_the_main_photo_must_clearly_be_you():
    assert fm.judge(ME, [], [[STRANGER], [ME_PHOTO]])[1] == "main_photo_not_you"
    assert fm.judge(ME, [], [[BORDER], [ME_PHOTO]])[1] == "review_main_photo"      # just under: a person looks
    assert fm.judge(ME, [], [[], [ME_PHOTO]])[0]                                   # no face in the main photo: another can carry it
    assert fm.judge(ME, [], [[], [BORDER]])[1] == "review_main_photo"
    assert fm.judge(ME, [], [[], [vec(75)]])[1] == "photos_not_you"


def test_every_other_photo_with_a_face_must_be_you():
    ok, reason, _ = fm.judge(ME, [], [[ME_PHOTO], [ME_PHOTO], [STRANGER]])
    assert not ok and reason == "photo_3_not_you"
    assert fm.judge(ME, [], [[ME_PHOTO], [BORDER]])[0]                               # "same person" bar for the rest
    assert fm.judge(ME, [], [[ME_PHOTO], [STRANGER, ME_PHOTO]])[0]                   # group photo: one of them is you


def test_no_face_anywhere_fails():
    assert fm.judge(ME, [], [[], []])[1] == "no_face_in_photos"


def test_a_look_alike_crowd_raises_the_bar():
    look_alike = vec(57)                                     # ~0.54: passes the fixed bars
    assert fm.judge(ME, [], [[look_alike]])[0]
    crowd = [0.50, 0.52, 0.48, 0.55, 0.47, 0.53]             # strangers who look about as close
    ok, reason, meta = fm.judge(ME, [], [[look_alike]], crowd)
    assert not ok and reason == "main_photo_not_you" and meta["cohort"]["n"] == 6
    assert fm.judge(ME, [], [[ME_PHOTO]], crowd)[0]          # the real you is far closer than the crowd


def test_the_cohort_bar_is_capped_and_needs_enough_people():
    twins = [0.9] * 10
    main_bar, other_bar, _ = fm.bars(twins)
    assert main_bar == fm.COHORT_CAP and other_bar == fm.COHORT_CAP
    assert fm.bars([0.9, 0.9])[:2] == (fm.MAIN_PHOTO, fm.SAME_PERSON)   # too few to judge by


@pytest.mark.skipif(not fm.available(), reason="models not downloaded (see Dockerfile)")
def test_the_models_load_and_find_no_face_in_a_blank_image():
    import numpy as np
    blank = np.full((240, 320, 3), 200, dtype=np.uint8)
    assert fm.face_features(blank) == []



def test_review_only_just_under_a_bar():
    assert fm.judge(ME, [], [[vec(62)]])[0]                           # ~0.47 vs the 0.45 bar: passes
    assert fm.judge(ME, [], [[vec(65)]])[1] == "review_main_photo"   # ~0.42
    assert fm.judge(ME, [], [[vec(70)]])[1] == "main_photo_not_you"  # ~0.34: clearly not


def test_the_same_face_on_another_account_is_found():
    found = fm.duplicates(ME, {"me": ME, "twin": vec(10), "other": STRANGER, "close": vec(45)}, exclude="me")
    assert [u for u, _ in found] == ["twin", "close"]
    assert fm.duplicates(ME, {"other": STRANGER}) == []
