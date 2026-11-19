"""FastAPI inference service for crop disease prediction."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from crop_disease.inference import Predictor

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/best.pt")
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

state: dict[str, Predictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(MODEL_PATH)
    if path.exists():
        state["predictor"] = Predictor.from_checkpoint(path, device=DEVICE)
    yield
    state["predictor"] = None


app = FastAPI(title="Crop Disease Detection API", version="0.1.0", lifespan=lifespan)


class TopK(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: list[TopK]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state["predictor"] is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3) -> PredictionResponse:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode image.")

    top = predictor.predict(image, top_k=max(1, min(top_k, 10)))
    return PredictionResponse(
        predicted_class=top[0]["class_name"],
        confidence=top[0]["probability"],
        top_k=[TopK(**t) for t in top],
    )
