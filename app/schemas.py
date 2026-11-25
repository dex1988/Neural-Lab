from datetime import datetime

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None


class ExperimentRead(ExperimentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class HealthRead(BaseModel):
    status: str


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    filename: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[str]
    created_at: datetime


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing: int
    unique: int
    example: str | None = None


class DatasetPreview(BaseModel):
    dataset: DatasetRead
    columns: list[ColumnInfo]
    statistics: dict[str, object]
    preview: list[dict[str, object]]


class TrainingRequest(BaseModel):
    dataset_id: int
    name: str = Field(min_length=1, max_length=120)
    model_type: str = Field(pattern="^(linear_regression|logistic_regression|random_forest|knn|decision_tree|mlp)$")
    problem_type: str = Field(pattern="^(classification|regression)$")
    target_column: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)
    missing_strategy: str = Field(default="median", pattern="^(drop|median|most_frequent)$")
    scaling: str = Field(default="standard", pattern="^(none|standard|robust)$")
    hyperparameters: dict[str, object] = Field(default_factory=dict)


class TrainedModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    model_type: str
    problem_type: str
    feature_columns: list[str]
    target_column: str
    hyperparameters: dict[str, object]
    metrics: dict[str, object]
    status: str
    error: str | None
    duration_seconds: float | None
    created_at: datetime


class PredictionRequest(BaseModel):
    features: dict[str, object]


class PredictionRead(BaseModel):
    prediction: object
    probabilities: dict[str, float] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    dataset_id: int | None = Field(default=None, gt=0)
    model_id: str | None = Field(default=None, max_length=200)
    hf_token: str | None = Field(default=None, max_length=512)
    latest_prediction: dict[str, object] | None = None
    context_mode: Literal["data", "standalone"] = "data"


class WebAISourceInput(BaseModel):
    kind: Literal["url", "query", "text"]
    value: str = Field(min_length=1, max_length=60000)


class WebAIModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    topic: str = Field(min_length=1, max_length=2000)
    urls: list[Annotated[str, Field(min_length=4, max_length=2048)]] = Field(
        default_factory=list, max_length=20
    )
    queries: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=20
    )


class WebAISourceSelection(BaseModel):
    selected: bool


class WebAIChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    hf_token: str | None = Field(default=None, max_length=512)


class WebAILearnRequest(BaseModel):
    brave_search_api_key: str | None = Field(default=None, max_length=512)


class WebAISourceRead(BaseModel):
    id: int
    model_id: int
    parent_id: int | None
    kind: str
    value: str
    url: str | None
    title: str | None
    status: str
    selected: bool
    error: str | None
    retrieved_at: datetime | None
    chunk_count: int


class WebAIModelRead(BaseModel):
    id: int
    name: str
    description: str | None
    topic: str
    status: str
    status_message: str | None
    created_at: datetime
    updated_at: datetime
    source_count: int
    chunk_count: int
    discovered_count: int
    retrieved_count: int
    processed_count: int
    error_count: int
    skipped_count: int
    chat_ready: bool


class WebAISourceDetail(WebAISourceRead):
    content: str | None


class WebAIChatRead(BaseModel):
    reply: str
    model: str
    sources: list[dict[str, str]]
