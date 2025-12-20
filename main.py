from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

app = FastAPI()

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.get("/")
def root():
    return {"status": "ok", "service": "shaadiparrot-face-verification"}

@app.post("/verify-photo")
async def verify_photo(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "reason": "invalid_image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return {"ok": False, "reason": "no_face_detected"}

    if len(faces) > 1:
        return {"ok": False, "reason": "multiple_faces_detected"}

    return {
        "ok": True,
        "faces_detected": len(faces)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
