"""Downloads the face matching models (face_match.py) from the official OpenCV model zoo, pinned to one commit,
and checks each file's sha256, so a changed or broken download fails loudly. Used by the Dockerfile and locally:

    python tools/fetch_models.py [target_dir]     (default: ./models)

YuNet (face detection): MIT. SFace (face recognition): Apache-2.0.
"""
import hashlib
import os
import sys
import urllib.request

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/25f423d0e04c31a17254620e58febd7386da523b/models"
FILES = {
    "face_detection_yunet/face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface/face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(target, exist_ok=True)
    for path, sha in FILES.items():
        dest = os.path.join(target, path.split("/")[-1])
        if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha:
            print("model ok (cached)", dest)
            continue
        data = urllib.request.urlopen(f"{ZOO}/{path}", timeout=180).read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{path}: sha256 {got} != expected {sha}")
        with open(dest, "wb") as f:
            f.write(data)
        print("model ok", dest, len(data))


if __name__ == "__main__":
    main()
