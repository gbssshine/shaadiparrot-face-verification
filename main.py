from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI(title="ShaadiParrot Face Verification")

# ===== логгер =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-verification")

# ===== ML globals =====
face_app = None


class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: str


@app.on_event("startup")
def startup_event():
    """
    ВАЖНО:
    Всё тяжёлое — ТОЛЬКО тут.
    Если упадёт — сервис всё равно стартует.
    """
    global face_app
    try:
        from insightface.app import FaceAnalysis

        logger.info("Initializing InsightFace model...")
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace loaded successfully")

    except Exception as e:
        face_app = None
        logger.exception("Face model failed to load, service will run without verification")


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "shaadiparrot-face-verification",
        "face_model_loaded": face_app is not None
    }


@app.post("/verify-face")
def verify_face(data: VerifyFaceRequest):
    if face_app is None:
        raise HTTPException(
            status_code=503,
            detail="Face verification model not available"
        )

    # ⚠️ ТУТ будет реальная логика:
    # 1. скачать image_url (Firebase Storage signed URL)
    # 2. cv2.imread / decode
    # 3. faces = face_app.get(img)
    # 4. сравнение эмбеддингов с уже загруженными фото

    return {
        "status": "verified",
        "user_id": data.user_id,
        "image_url": data.image_url
    }
