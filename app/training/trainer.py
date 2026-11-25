import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.database import SessionLocal
from app.models import Dataset, TrainedModel, TrainingConfiguration


ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class TorchMLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, problem_type: str):
        super().__init__()
        self.problem_type = problem_type
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _preprocessor(frame: pd.DataFrame, features: list[str], scaling: str, missing: str):
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    numeric_fill = "median" if missing == "median" else "most_frequent"
    scaler = StandardScaler() if scaling == "standard" else RobustScaler() if scaling == "robust" else "passthrough"
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy=numeric_fill)),
            ("scaler", scaler),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers=transformers)


def run_training(model_id: int, request_data: dict[str, object]) -> None:
    started = time.monotonic()
    with SessionLocal() as db:
        record = db.get(TrainedModel, model_id)
        if record is None:
            return
        record.status = "running"
        db.commit()
        try:
            dataset = db.get(Dataset, int(request_data["dataset_id"]))
            if dataset is None:
                raise ValueError("Dataset not found")
            frame = pd.read_csv(dataset.file_path)
            target = str(request_data["target_column"])
            features = list(request_data["feature_columns"])
            if target not in frame or any(column not in frame for column in features):
                raise ValueError("Target or feature column does not exist")

            frame = frame[features + [target]].copy()
            if request_data["missing_strategy"] == "drop":
                frame = frame.dropna()
            frame = frame.dropna(subset=[target])
            if len(frame) < 4:
                raise ValueError("At least 4 complete rows are required")

            X = frame[features]
            y = frame[target]
            problem_type = str(request_data["problem_type"])
            hyperparameters = dict(request_data.get("hyperparameters") or {})
            seed = int(request_data["random_seed"])
            test_size = float(request_data["test_size"])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed,
                stratify=y if problem_type == "classification" and y.nunique() > 1 else None,
            )

            preprocessor = _preprocessor(
                X_train, features, str(request_data["scaling"]),
                str(request_data["missing_strategy"]),
            )
            model_type = str(request_data["model_type"])
            loss_history: list[dict[str, float]] = []

            if model_type == "linear_regression":
                estimator = LinearRegression()
            elif model_type == "logistic_regression":
                estimator = LogisticRegression(
                    C=float(hyperparameters.get("C", 1.0)),
                    max_iter=int(hyperparameters.get("max_iter", 500)),
                    random_state=seed,
                )
            elif model_type == "random_forest":
                estimator = (RandomForestClassifier if problem_type == "classification" else RandomForestRegressor)(
                    n_estimators=int(hyperparameters.get("n_estimators", 100)),
                    max_depth=hyperparameters.get("max_depth"),
                    random_state=seed,
                )
            elif model_type == "knn":
                estimator = (KNeighborsClassifier if problem_type == "classification" else KNeighborsRegressor)(
                    n_neighbors=int(hyperparameters.get("n_neighbors", 5)),
                )
            elif model_type == "decision_tree":
                estimator = (DecisionTreeClassifier if problem_type == "classification" else DecisionTreeRegressor)(
                    max_depth=hyperparameters.get("max_depth"),
                    random_state=seed,
                )
            elif model_type == "mlp":
                estimator = None
            else:
                raise ValueError("Unsupported model type")

            artifact_path = ARTIFACT_DIR / f"model-{model_id}.joblib"
            metrics: dict[str, object]

            if model_type == "mlp":
                X_train_matrix = preprocessor.fit_transform(X_train)
                X_test_matrix = preprocessor.transform(X_test)
                X_train_tensor = torch.tensor(np.asarray(X_train_matrix), dtype=torch.float32)
                X_test_tensor = torch.tensor(np.asarray(X_test_matrix), dtype=torch.float32)

                if problem_type == "classification":
                    classes = np.unique(y_train)
                    class_to_index = {value: index for index, value in enumerate(classes)}
                    y_train_tensor = torch.tensor([class_to_index[value] for value in y_train], dtype=torch.long)
                    y_test_tensor = torch.tensor([class_to_index[value] for value in y_test], dtype=torch.long)
                    output_size = len(classes)
                    criterion = torch.nn.CrossEntropyLoss()
                    mlp = TorchMLP(X_train_tensor.shape[1], output_size, problem_type)
                else:
                    y_train_tensor = torch.tensor(y_train.to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)
                    y_test_tensor = torch.tensor(y_test.to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)
                    output_size = 1
                    criterion = torch.nn.MSELoss()
                    mlp = TorchMLP(X_train_tensor.shape[1], output_size, problem_type)

                optimizer = torch.optim.Adam(
                    mlp.parameters(), lr=float(hyperparameters.get("learning_rate", 0.001))
                )
                epochs = int(hyperparameters.get("epochs", 20))
                batch_size = int(hyperparameters.get("batch_size", 32))
                mlp.train()
                for epoch in range(epochs):
                    permutation = torch.randperm(len(X_train_tensor))
                    epoch_loss = 0.0
                    for start in range(0, len(permutation), batch_size):
                        indices = permutation[start:start + batch_size]
                        optimizer.zero_grad()
                        outputs = mlp(X_train_tensor[indices])
                        loss = criterion(
                            outputs,
                            y_train_tensor[indices].view(-1, 1) if problem_type == "regression"
                            else y_train_tensor[indices],
                        )
                        loss.backward()
                        optimizer.step()
                        epoch_loss += float(loss.item())
                    loss_history.append({"epoch": float(epoch + 1), "loss": epoch_loss / max(1, len(X_train_tensor) // batch_size)})

                mlp.eval()
                with torch.no_grad():
                    raw_predictions = mlp(X_test_tensor)
                    if problem_type == "classification":
                        predictions = np.asarray(classes[torch.argmax(raw_predictions, dim=1).numpy()])
                        probabilities = torch.softmax(raw_predictions, dim=1).numpy()
                    else:
                        predictions = raw_predictions.numpy().reshape(-1)
                        probabilities = None
                bundle = {
                    "kind": "pytorch_mlp",
                    "preprocessor": preprocessor,
                    "network": mlp,
                    "classes": np.asarray(classes) if problem_type == "classification" else None,
                }
                joblib.dump(bundle, artifact_path)
            else:
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ])
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                probabilities = pipeline.predict_proba(X_test) if problem_type == "classification" and hasattr(pipeline, "predict_proba") else None
                joblib.dump({
                    "kind": "sklearn_pipeline",
                    "pipeline": pipeline,
                    "classes": getattr(pipeline, "classes_", None),
                }, artifact_path)

            if problem_type == "classification":
                labels = sorted(np.unique(y_test).tolist())
                metrics = {
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                    "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                    "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
                    "labels": labels,
                    "loss_history": loss_history,
                }
            else:
                metrics = {
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "mse": float(mean_squared_error(y_test, predictions)),
                    "r2": float(r2_score(y_test, predictions)),
                    "loss_history": loss_history,
                }

            record.status = "completed"
            record.metrics_json = json.dumps(metrics)
            record.artifact_path = str(artifact_path)
            record.duration_seconds = round(time.monotonic() - started, 3)
            record.hyperparameters_json = json.dumps(hyperparameters)

            db.add(TrainingConfiguration(
                dataset_id=int(request_data["dataset_id"]),
                target_column=target,
                feature_columns_json=json.dumps(features),
                problem_type=problem_type,
                test_size=test_size,
                random_seed=seed,
                missing_strategy=str(request_data["missing_strategy"]),
                scaling=str(request_data["scaling"]),
            ))
            db.commit()
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.duration_seconds = round(time.monotonic() - started, 3)
            db.commit()
