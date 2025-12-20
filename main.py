from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()


# ---------- MODELS ----------

class FaceVerificationRequest(BaseModel):
    user_id: str
    image_url: str


# ---------- ROUTES ----------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "shaadiparrot-face-verification"
    }


@app.post("/verify-face")
def verify_face(data: FaceVerificationRequest):
    """
    Here later you will:
    - download image
    - run face detection / matching
    - return result
    """

    # TEMP mock logic
    if not data.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    return {
        "verified": True,
        "user_id": data.user_id,
        "confidence": 0.97,
        "message": "Face verified successfully"
    }


# ---------- ENTRYPOINT ----------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
