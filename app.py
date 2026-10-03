"""Local inference API for AI-Media Guard."""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/AI_Media_Guard_Image_Final.keras"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime"}
IMAGE_SIZE = (128, 128)
app = FastAPI(title="AI-Media Guard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}. Train it first (see README.md).")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="TensorFlow is not installed. Install requirements.txt to enable inference.") from exc
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc
    return _model


def score_image(image: Image.Image) -> float:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    return float(np.asarray(get_model().predict(batch, verbose=0)).reshape(-1)[0])


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_exists": MODEL_PATH.is_file(), "model_path": str(MODEL_PATH)}


@app.get("/")
def frontend():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if content_type not in IMAGE_TYPES | VIDEO_TYPES:
        raise HTTPException(status_code=415, detail="Supported formats: JPG, PNG, WEBP, MP4 and MOV.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 25 MB limit.")

    scores = []
    if content_type in IMAGE_TYPES:
        try:
            scores.append(score_image(Image.open(io.BytesIO(data))))
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Could not read this image.") from exc
    else:
        try:
            import cv2
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Video analysis requires OpenCV. Install requirements.txt.") from exc
        temp_path = None
        try:
            import tempfile
            suffix = ".mp4" if content_type == "video/mp4" else ".mov"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(data)
                temp_path = temp.name
            capture = cv2.VideoCapture(temp_path)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or total <= 0:
                raise HTTPException(status_code=400, detail="Could not read frames from this video.")
            for index in np.linspace(0, total - 1, min(8, total), dtype=int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    scores.append(score_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            capture.release()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        if not scores:
            raise HTTPException(status_code=400, detail="No readable frames found in this video.")

    # The directory loader's verified class order is FAKE=0, REAL=1, so the
    # sigmoid output represents the REAL class probability.
    real_probability = float(np.mean(scores))
    verdict = "real" if real_probability >= 0.5 else "ai_generated"
    confidence = real_probability if verdict == "real" else 1 - real_probability
    return {
        "verdict": verdict,
        "label": "Likely AI-generated" if verdict == "ai_generated" else "Likely real",
        "confidence": confidence,
        "breakdown": [
            {"label": "AI-generated", "score": 1 - real_probability},
            {"label": "Real", "score": real_probability},
        ],
        "frames_analyzed": len(scores),
        "model": "EfficientNetB0 CIFAKE binary classifier",
        "limitations": "Estimates AI-generated vs real appearance; it does not independently verify provenance or detect every form of manipulation.",
    }
