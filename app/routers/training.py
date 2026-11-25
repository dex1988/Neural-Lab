import json
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, TrainedModel
from app.schemas import TrainedModelRead, TrainingRequest
from app.training.trainer import run_training


router = APIRouter(prefix="/api/v1", tags=["training"])


def _model_read(record: TrainedModel) -> TrainedModelRead:
    return TrainedModelRead(
        id=record.id,
        name=record.name,
        model_type=record.model_type,
        problem_type=record.problem_type,
        feature_columns=json.loads(record.feature_columns_json),
        target_column=record.target_column,
        hyperparameters=json.loads(record.hyperparameters_json),
        metrics=json.loads(record.metrics_json),
        status=record.status,
        error=record.error,
        duration_seconds=record.duration_seconds,
        created_at=record.created_at,
    )


@router.post("/training", response_model=TrainedModelRead, status_code=status.HTTP_202_ACCEPTED)
def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainedModelRead:
    dataset = db.get(Dataset, request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if request.target_column not in json.loads(dataset.columns_json):
        raise HTTPException(status_code=400, detail="Target column is not in the dataset")
    invalid_features = set(request.feature_columns) - set(json.loads(dataset.columns_json))
    if invalid_features:
        raise HTTPException(status_code=400, detail="One or more feature columns are not in the dataset")
    if request.target_column in request.feature_columns:
        raise HTTPException(status_code=400, detail="Target column cannot also be a feature")

    record = TrainedModel(
        dataset_id=request.dataset_id,
        name=request.name,
        model_type=request.model_type,
        problem_type=request.problem_type,
        artifact_path="",
        feature_columns_json=json.dumps(request.feature_columns),
        target_column=request.target_column,
        hyperparameters_json=json.dumps(request.hyperparameters),
        metrics_json="{}",
        status="queued",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    background_tasks.add_task(run_training, record.id, request.model_dump())
    return _model_read(record)


@router.get("/models", response_model=list[TrainedModelRead])
def list_models(db: Session = Depends(get_db)) -> list[TrainedModelRead]:
    records = db.scalars(select(TrainedModel).order_by(TrainedModel.created_at.desc()))
    return [_model_read(record) for record in records]


@router.get("/models/{model_id}", response_model=TrainedModelRead)
def get_model(model_id: int, db: Session = Depends(get_db)) -> TrainedModelRead:
    record = db.get(TrainedModel, model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_read(record)


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    record = db.get(TrainedModel, model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if record.artifact_path:
        Path(record.artifact_path).unlink(missing_ok=True)
    db.delete(record)
    db.commit()
