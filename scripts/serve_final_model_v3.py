from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from final_model_v3_inference import FINAL_MODEL_ROOT, FinalModelV3Predictor, load_signal


app = FastAPI(title="final_model_v3_service", version="1.0.0")
predictor = FinalModelV3Predictor.load()


class PredictRequest(BaseModel):
    mat_path: str


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_root": str(FINAL_MODEL_ROOT),
        "positive_threshold": predictor.positive_threshold,
        "negative_threshold": predictor.negative_threshold,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    mat_path = Path(request.mat_path)
    result = predictor.predict_tensor(load_signal(mat_path))
    result["mat_path"] = str(mat_path)
    return result
