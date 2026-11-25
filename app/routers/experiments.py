from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experiment
from app.schemas import ExperimentCreate, ExperimentRead


router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
def create_experiment(
    experiment: ExperimentCreate, db: Session = Depends(get_db)
) -> Experiment:
    record = Experiment(**experiment.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("", response_model=list[ExperimentRead])
def list_experiments(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Experiment]:
    records = db.scalars(
        select(Experiment).order_by(Experiment.id).offset(offset).limit(limit)
    )
    return list(records)


@router.get("/{experiment_id}", response_model=ExperimentRead)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)) -> Experiment:
    record = db.get(Experiment, experiment_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return record


@router.put("/{experiment_id}", response_model=ExperimentRead)
def update_experiment(
    experiment_id: int, experiment: ExperimentCreate, db: Session = Depends(get_db)
) -> Experiment:
    record = db.get(Experiment, experiment_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    for field, value in experiment.model_dump().items():
        setattr(record, field, value)
    db.commit()
    db.refresh(record)
    return record


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)) -> Response:
    record = db.get(Experiment, experiment_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    db.delete(record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
