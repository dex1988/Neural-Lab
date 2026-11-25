import sys
import types

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_experiment_crud(client: TestClient) -> None:
    created = client.post(
        "/api/v1/experiments",
        json={"name": "Baseline", "description": "First run"},
    )
    assert created.status_code == 201
    experiment = created.json()
    assert experiment["name"] == "Baseline"
    assert experiment["id"] == 1

    listed = client.get("/api/v1/experiments")
    assert [item["id"] for item in listed.json()] == [1]

    updated = client.put(
        "/api/v1/experiments/1",
        json={"name": "Updated", "description": None},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Updated"

    deleted = client.delete("/api/v1/experiments/1")
    assert deleted.status_code == 204
    assert client.get("/api/v1/experiments/1").status_code == 404


def test_create_requires_non_empty_name(client: TestClient) -> None:
    response = client.post("/api/v1/experiments", json={"name": ""})
    assert response.status_code == 422


def test_chat_reports_missing_hugging_face_token(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Summarize my dataset"}]},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "HF_TOKEN is not configured"
    assert "message" in response.json()


def test_chat_sends_dataset_and_prediction_context(client: TestClient, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeInferenceClient:
        def __init__(self, model: str, token: str):
            captured["model"] = model
            captured["token"] = token

        def chat_completion(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Churn context received.")
                    )
                ]
            )

    hf_module = types.ModuleType("huggingface_hub")
    hf_module.InferenceClient = FakeInferenceClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
    monkeypatch.setenv("HF_TOKEN", "environment-token")

    upload = client.post(
        "/api/v1/datasets?name=Customers",
        files={
            "file": (
                "customers.csv",
                "age,churn\n25,0\n30,1\n",
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]
    response = client.post(
        "/api/chat",
        json={
            "dataset_id": dataset_id,
            "model_id": "test/model",
            "hf_token": "request-token",
            "messages": [{"role": "user", "content": "Explain churn"}],
            "latest_prediction": {"prediction": 1, "model_name": "Churn forest"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "Churn context received.", "model": "test/model"}
    assert captured["model"] == "test/model"
    assert captured["token"] == "request-token"
    system_prompt = captured["messages"][0]["content"]
    assert '"row_count": 2' in system_prompt
    assert '"column_names": ["age", "churn"]' in system_prompt
    assert '"latest_prediction": {"prediction": 1, "model_name": "Churn forest"}' in system_prompt

    client.delete(f"/api/v1/datasets/{dataset_id}")


def test_standalone_chat_does_not_attach_workspace_context(client: TestClient, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeInferenceClient:
        def __init__(self, model: str, token: str):
            captured["model"] = model

        def chat_completion(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="Standalone reply.")
                    )
                ]
            )

    hf_module = types.ModuleType("huggingface_hub")
    hf_module.InferenceClient = FakeInferenceClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)

    response = client.post(
        "/api/chat",
        json={
            "context_mode": "standalone",
            "model_id": "test/standalone-model",
            "hf_token": "test-token",
            "messages": [{"role": "user", "content": "Write a sorting function"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "Standalone reply."
    system_prompt = captured["messages"][0]["content"]
    assert "Do not reference datasets" in system_prompt
    assert "Workspace context" not in system_prompt


def test_chat_returns_json_when_inference_api_fails(client: TestClient, monkeypatch) -> None:
    class FailingInferenceClient:
        def __init__(self, model: str, token: str):
            pass

        def chat_completion(self, **kwargs):
            raise RuntimeError("provider unavailable")

    hf_module = types.ModuleType("huggingface_hub")
    hf_module.InferenceClient = FailingInferenceClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "Hugging Face Inference API request failed"


def test_list_experiments_rejects_out_of_range_pagination(client: TestClient) -> None:
    response = client.get("/api/v1/experiments?limit=501&offset=-1")
    assert response.status_code == 422


def test_dashboard_assets_are_served(client: TestClient) -> None:
    assert "NeuralLab" in client.get("/").text
    assert client.get("/training").status_code == 200
    assert client.get("/api/v1/unknown").status_code == 404
    assert client.get("/assets").status_code in (200, 404)
    assert client.get("/docs").status_code == 200


def test_dataset_upload_inspection_and_delete(client: TestClient) -> None:
    csv = "age,income,purchased\n25,50000,yes\n30,62000,no\n40,80000,yes\n"
    response = client.post(
        "/api/v1/datasets?name=Customers",
        files={"file": ("customers.csv", csv, "text/csv")},
    )
    assert response.status_code == 201
    dataset = response.json()
    assert dataset["row_count"] == 3
    assert dataset["columns"] == ["age", "income", "purchased"]

    listed = client.get("/api/v1/datasets")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "Customers"

    preview = client.get(f"/api/v1/datasets/{dataset['id']}")
    assert preview.status_code == 200
    assert len(preview.json()["columns"]) == 3

    deleted = client.delete(f"/api/v1/datasets/{dataset['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/datasets/{dataset['id']}").status_code == 404


def test_real_training_request(client: TestClient) -> None:
    csv = (
        "x,y\n"
        "1,0\n2,0\n3,0\n4,1\n5,1\n6,1\n"
        "7,1\n8,1\n9,1\n10,0\n"
    )
    upload = client.post(
        "/api/v1/datasets?name=Binary",
        files={"file": ("binary.csv", csv, "text/csv")},
    )
    dataset_id = upload.json()["id"]
    trained = client.post(
        "/api/v1/training",
        json={
            "dataset_id": dataset_id,
            "name": "Test forest",
            "model_type": "random_forest",
            "problem_type": "classification",
            "target_column": "y",
            "feature_columns": ["x"],
            "test_size": 0.3,
            "random_seed": 42,
            "missing_strategy": "median",
            "scaling": "standard",
            "hyperparameters": {"n_estimators": 10},
        },
    )
    assert trained.status_code == 202
    model = trained.json()
    assert model["status"] in ("queued", "running", "completed")

    model_detail = client.get(f"/api/v1/models/{model['id']}")
    assert model_detail.status_code == 200
    assert model_detail.json()["status"] in ("queued", "running", "completed", "failed")

    prediction = client.post(
        f"/api/v1/models/{model['id']}/predict",
        json={"features": {"x": 5}},
    )
    assert prediction.status_code in (409, 500)
