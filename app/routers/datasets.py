import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.schemas import DatasetPreview, DatasetRead


router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _column_info(frame: pd.DataFrame) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        dtype = str(series.dtype)
        result.append({
            "name": str(column),
            "dtype": dtype,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "example": None if series.dropna().empty else str(series.dropna().iloc[0]),
        })
    return result


def _dataset_read(record: Dataset) -> DatasetRead:
    return DatasetRead(
        id=record.id,
        name=record.name,
        filename=record.filename,
        size_bytes=record.size_bytes,
        row_count=record.row_count,
        column_count=record.column_count,
        columns=json.loads(record.columns_json),
        created_at=record.created_at,
    )


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = None,
    db: Session = Depends(get_db),
) -> DatasetRead:
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="CSV exceeds the 25 MB upload limit")
                target.write(chunk)
        frame = pd.read_csv(destination)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    if frame.empty or len(frame.columns) == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="CSV must contain at least one row and column")

    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="CSV column names must be unique")

    safe_name = (name or Path(file.filename).stem).strip()[:255]
    record = Dataset(
        name=safe_name or "Untitled dataset",
        filename=Path(file.filename).name,
        file_path=str(destination),
        size_bytes=size,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns_json=json.dumps(list(frame.columns)),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _dataset_read(record)


@router.get("", response_model=list[DatasetRead])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetRead]:
    records = db.scalars(select(Dataset).order_by(Dataset.created_at.desc()))
    return [_dataset_read(record) for record in records]


@router.get("/{dataset_id}", response_model=DatasetPreview)
def inspect_dataset(dataset_id: int, db: Session = Depends(get_db)) -> DatasetPreview:
    record = db.get(Dataset, dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        frame = pd.read_csv(record.file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Stored dataset could not be read") from exc

    numeric = frame.select_dtypes(include="number")
    statistics: dict[str, object] = {
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_columns": len(numeric.columns),
        "categorical_columns": len(frame.columns) - len(numeric.columns),
        "numeric_summary": numeric.describe().to_dict() if not numeric.empty else {},
    }
    preview = frame.head(50).where(pd.notna(frame.head(50)), None).to_dict(orient="records")
    return DatasetPreview(
        dataset=_dataset_read(record),
        columns=_column_info(frame),
        statistics=statistics,
        preview=preview,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    record = db.get(Dataset, dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    Path(record.file_path).unlink(missing_ok=True)
    db.delete(record)
    db.commit()
