from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()


class VerifyFaceRequest(BaseModel):
    user_id: str
    image_url: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "shaadiparrot-face-verification"
    }


@app.post("/verify-face")
def verify_face(data: VerifyFaceRequest):
    # ⛔️ пока без реального face recognition
    # тут будет логика позже

    return {
        "status": "verified",
        "user_id": data.user_id,
        "image_url": data.image_url
    }
