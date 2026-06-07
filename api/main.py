from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.schemas import (
    HealthResponse,
    PredictGridRequest,
    PredictPixelsRequest,
    PredictResponse,
)
from s4p_mnist.config import PROJECT_ROOT
from s4p_mnist.models.model import Model

app = FastAPI(title="S4P MNIST API", version="1.0.0")
_model: Model | None = None


def _model_path() -> Path:
    raw = os.environ.get("S4P_MODEL_PATH", "models/model.joblib")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_model() -> Model:
    global _model
    if _model is not None:
        return _model
    path = _model_path()
    if not path.is_file():
        raise FileNotFoundError(f"missing model file: {path}")
    _model = Model.load(path)
    return _model


def _run_predict(x: np.ndarray) -> tuple[int, float]:
    model = _load_model()
    device = model._device()
    model._net.to(device)
    model._net.eval()
    batch = Model._prepare_x(x)
    with torch.no_grad():
        logits = model._net(batch.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        digit = int(probs.argmax().item())
        confidence = float(probs[digit].item())
    return digit, confidence


def _ensure_model_from_gcs() -> None:
    gcs_uri = os.environ.get("S4P_GCS_MODEL_URI", "").strip()
    if not gcs_uri:
        return
    local = _model_path()
    if local.is_file():
        return
    from s4p_mnist.utils.io import download_gcs_file

    download_gcs_file(gcs_uri, local)


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("S4P_SKIP_MODEL_LOAD", "").lower() in {"1", "true", "yes"}:
        return
    try:
        _ensure_model_from_gcs()
        _load_model()
    except FileNotFoundError:
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="ok", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict_pixels(body: PredictPixelsRequest) -> PredictResponse:
    try:
        x = np.asarray(body.pixels, dtype=np.float32).reshape(1, 28, 28)
        digit, confidence = _run_predict(x)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(digit=digit, confidence=confidence)


@app.post("/predict/grid", response_model=PredictResponse)
def predict_grid(body: PredictGridRequest) -> PredictResponse:
    try:
        x = np.asarray(body.image, dtype=np.float32).reshape(1, 28, 28)
        digit, confidence = _run_predict(x)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(digit=digit, confidence=confidence)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize((28, 28))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    arr = np.asarray(img, dtype=np.float32)[None, :, :]
    try:
        digit, confidence = _run_predict(arr)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(digit=digit, confidence=confidence)


# Cloud Functions Gen 2 entry point (gcloud --entry-point=api.main.handler).
from mangum import Mangum

handler = Mangum(app, lifespan="off")
