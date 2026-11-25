import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TrainedModel
from app.schemas import PredictionRead, PredictionRequest


router = APIRouter(prefix="/api/v1/models", tags=["predictions"])


@router.post("/{model_id}/predict", response_model=PredictionRead)
def predict(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionRead:
    record = db.get(TrainedModel, model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Model is {record.status}")
    if not record.artifact_path or not Path(record.artifact_path).is_file():
        raise HTTPException(status_code=500, detail="Model artifact is unavailable")

    features = json.loads(record.feature_columns_json)
    missing = [name for name in features if name not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing)}")
    frame = pd.DataFrame([{name: request.features[name] for name in features}])

    bundle = joblib.load(record.artifact_path)
    if bundle["kind"] == "sklearn_pipeline":
        prediction = bundle["pipeline"].predict(frame)[0]
        probabilities = None
        if hasattr(bundle["pipeline"], "predict_proba"):
            values = bundle["pipeline"].predict_proba(frame)[0]
            classes = bundle["pipeline"].classes_
            probabilities = {str(label): float(value) for label, value in zip(classes, values)}
    else:
        matrix = bundle["preprocessor"].transform(frame)
        tensor = torch.tensor(np.asarray(matrix), dtype=torch.float32)
        network = bundle["network"]
        network.eval()
        with torch.no_grad():
            raw = network(tensor)
        if record.problem_type == "classification":
            values = torch.softmax(raw, dim=1)[0].numpy()
            index = int(np.argmax(values))
            prediction = bundle["classes"][index]
            probabilities = {
                str(label): float(value)
                for label, value in zip(bundle["classes"], values)
            }
        else:
            prediction = float(raw.numpy().reshape(-1)[0])
            probabilities = None

    if isinstance(prediction, np.generic):
        prediction = prediction.item()
    return PredictionRead(prediction=prediction, probabilities=probabilities)
