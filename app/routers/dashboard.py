import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, Experiment, TrainedModel


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary")
def dashboard_summary(db: Session = Depends(get_db)) -> dict[str, object]:
    datasets = list(db.scalars(select(Dataset)))
    experiments = list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc())))
    models = list(db.scalars(select(TrainedModel).order_by(TrainedModel.created_at.desc())))
    completed = [model for model in models if model.status == "completed"]
    best = None
    if completed:
        candidates: list[tuple[float, TrainedModel]] = []
        for model in completed:
            metrics = json.loads(model.metrics_json)
            score = metrics.get("accuracy") if model.problem_type == "classification" else metrics.get("r2")
            if isinstance(score, (int, float)):
                candidates.append((float(score), model))
        if candidates:
            best_score, best_model = max(candidates, key=lambda item: item[0])
            best = {
                "id": best_model.id,
                "name": best_model.name,
                "model_type": best_model.model_type,
                "score": best_score,
                "problem_type": best_model.problem_type,
            }

    recent_experiments = []
    for experiment in experiments[:5]:
        recent_experiments.append({
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "created_at": experiment.created_at.isoformat(),
        })

    history = []
    now = datetime.now(timezone.utc)
    for offset in range(6, -1, -1):
        day = now - timedelta(days=offset)
        count = sum(
            1 for model in models
            if model.created_at.replace(tzinfo=timezone.utc).date() == day.date()
        )
        history.append({"date": day.strftime("%a"), "models": count})

    return {
        "total_datasets": len(datasets),
        "total_models": len(models),
        "total_experiments": len(experiments),
        "completed_models": len(completed),
        "recent_experiments": recent_experiments,
        "best_model": best,
        "training_history": history,
    }
