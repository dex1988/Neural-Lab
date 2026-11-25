# NeuralLab

NeuralLab is a local machine-learning training studio for uploading datasets, configuring preprocessing, training real models, inspecting results, and generating predictions. It is built with React, TypeScript, FastAPI, SQLAlchemy, scikit-learn, and PyTorch.

## Features

- CSV upload with safe filenames and a 25 MB size limit
- Dataset preview, column type detection, missing-value and duplicate statistics
- Configurable train/test split, random seed, missing-value handling, and scaling
- Scikit-learn pipelines that fit preprocessing only on training data
- Linear/logistic regression, random forest, KNN, decision tree, and PyTorch MLP
- Classification and regression metrics, confusion matrices, and MLP loss history
- Persistent datasets, training configurations, experiments, and model artifacts
- Model prediction playground with class probabilities
- Chat with Data assistant powered by the Hugging Face Inference API (no local LLM weights)
- Dark, responsive React dashboard using Recharts and Lucide

## Architecture

```text
app/
  api/                 API package
  models/              SQLAlchemy entities
  schemas/             Pydantic request/response models
  services/            Shared application services
  training/            Training implementations
  preprocessing/       Pipeline and preprocessing helpers
  database.py          SQLAlchemy engine and session factory
  main.py              FastAPI application and static frontend
frontend/
  src/components/      Reusable UI components
  src/pages/           Route-level application screens
  src/services/        Typed API client
  src/types/           Shared frontend types
  src/charts/          Recharts visualizations
tests/                 API and ML tests
migrations/            Alembic migrations
data/example/iris.csv  Example dataset
```

## Local setup

Prerequisites: Python 3.12+ and Node.js 20+.

For Chat with Data, create a Hugging Face access token with Inference Providers access. In a terminal, stop any running server with `Ctrl+C`, then set the token and restart the API on port 8080:

```bash
export HF_TOKEN="hf_your_token"
.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

The chat uses `meta-llama/Llama-3.2-3B-Instruct` by default. Set `HF_MODEL` to choose another default model, or enter a model ID in the chat panel. The latest prediction generated in Prediction playground is included automatically.

You can also enter a Hugging Face token directly in Chat with Data; it overrides `HF_TOKEN` for chat requests only and is kept in page memory, not browser storage.

To add another chat model, enter its public Hugging Face Hub repository ID in **Add a Hugging Face model**. The app saves the ID in browser storage and uses the model remotely through an Inference Provider; it does not upload model weights. Access and provider support depend on the selected model.

Use **Hugging Face playground** for general chat that is independent of the NeuralLab workspace. It sends only the conversation to the selected Hugging Face model and does not include uploaded datasets or latest prediction context.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API documentation is at `http://localhost:8000/docs`.

## Docker

```bash
docker compose up --build
```

The application is available at `http://localhost:8000`.

## Example workflow

1. Open **Datasets** and upload `data/example/iris.csv`.
2. Open **Training studio**.
3. Select the Iris dataset, choose `target` as the target column, and select the feature columns.
4. Choose classification and a model such as Random Forest or PyTorch MLP.
5. Start training and open **Models** or **Prediction playground** when the run completes.

## API

- `POST /api/v1/datasets` — upload a CSV
- `GET /api/v1/datasets` — list datasets
- `GET /api/v1/datasets/{id}` — inspect data
- `DELETE /api/v1/datasets/{id}` — delete a dataset
- `POST /api/v1/training` — start a background training job
- `GET /api/v1/models` — list trained models
- `GET /api/v1/models/{id}` — read model status and metrics
- `DELETE /api/v1/models/{id}` — delete a model artifact
- `POST /api/v1/models/{id}/predict` — generate a prediction
- `GET /api/v1/dashboard/summary` — dashboard statistics
- `GET /api/v1/experiments` — manage experiments

## Tests

```bash
pytest -q
cd frontend && npm run build
```

## Screenshots

Add application screenshots to `docs/screenshots/` and link them here before publishing a release.

## Database

SQLite is the default and is stored in `data/neural_lab.db`. Set `DATABASE_URL` to a PostgreSQL URL such as:

```text
postgresql+psycopg://user:password@host:5432/neural_lab
```

Then run `alembic upgrade head`.
