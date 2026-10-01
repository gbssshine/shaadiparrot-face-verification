"""Runs the Cloud Run app on this PC against the local Firebase emulators (docs/local-backend.md in
MauiApp2). Photo URLs stored for the Android emulator (10.0.2.2) are fetched via 127.0.0.1 here."""
import os
import sys

os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8085")
os.environ.setdefault("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "demo-shaadiparrot")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import uvicorn

_get = requests.get
requests.get = lambda url, *a, **k: _get(str(url).replace("://10.0.2.2:", "://127.0.0.1:"), *a, **k)

import main  # noqa: E402

_session_get = main._HTTP.get
main._HTTP.get = lambda url, *a, **k: _session_get(str(url).replace("://10.0.2.2:", "://127.0.0.1:"), *a, **k)

# No Cloud Vision locally (the dummy credentials can't reach it): each selfie is read as a clear face doing
# what the challenge asked (straight, turned, smiling), so the app's verification flow can be walked through.
if os.getenv("LOCAL_FAKE_VISION", "1") == "1":
    from types import SimpleNamespace as _NS
    _pose = {"i": 0, "steps": ["front", "right", "left"]}
    _PAN = {"front": 0.0, "right": 24.0, "left": -22.0, "smile": 2.0}
    _take = main._take_challenge

    def _remember_steps(uid, cid):
        ch = _take(uid, cid)
        if ch:
            _pose.update(i=0, steps=list(ch.get("steps") or _pose["steps"]))
        return ch

    def _fake_vision(_bytes):
        step = _pose["steps"][_pose["i"] % len(_pose["steps"])]
        _pose["i"] += 1
        verts = [_NS(x=0, y=0), _NS(x=320, y=0), _NS(x=320, y=320), _NS(x=0, y=320)]
        face = _NS(pan_angle=_PAN[step], roll_angle=0.0, tilt_angle=0.0, detection_confidence=0.95,
                   joy_likelihood=5 if step == "smile" else 1, bounding_poly=_NS(vertices=verts))
        return "VERY_UNLIKELY", "VERY_UNLIKELY", "VERY_UNLIKELY", [face]

    main._take_challenge = _remember_steps
    main._detect_faces_and_safety_from_bytes = _fake_vision
    os.environ.setdefault("FIREBASE_STORAGE_EMULATOR_HOST", "127.0.0.1:9199")

    # The emulator's camera films a test scene, not a face: the selfies are read as the profile's own face.
    if os.getenv("LOCAL_FAKE_FACE_MATCH", "1") == "1":
        import numpy as _np
        import face_match as _fm
        _real, _real_sim = _fm.face_features, _fm.similarity
        _fm.face_features = lambda img: _real(img) or [[1.0, 0.0, 0.0]]       # "a face" where the test scene has none
        # that stand-in face is close to everyone (it only exists locally); real faces compare for real
        _fm.similarity = lambda a, b: 0.9 if _np.asarray(a).size != _np.asarray(b).size else _real_sim(a, b)

if __name__ == "__main__":
    uvicorn.run(main.app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")), log_level="info")
