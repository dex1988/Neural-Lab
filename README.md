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
- Independent Web AI Trainer that researches public websites and builds isolated, cited RAG knowledge bases
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

### Web AI Trainer

Web AI Trainer is a separate workspace from CSV datasets and Training Studio. For each assistant, add your own text, retrieve allowed public webpages, and build a persisted, searchable knowledge base using local text hashing and an isolated FAISS vector index; it does **not** retrain a foundation model. A website URL crawls up to 25 allowed pages on the same origin. Indexing does not require a Hugging Face token. Enter a Hugging Face read or inference-enabled Hugging Face token in the trainer for grounded AI answers and enable the Brave Search API key checkbox when using search queries; both are kept in page memory. Server-side `HF_TOKEN` and `BRAVE_SEARCH_API_KEY` remain available as fallbacks. Direct website URLs and manually provided text do not need a search key. The trainer defaults to `Qwen/Qwen2.5-7B-Instruct`; optional `WEB_AI_LLM_MODEL` and `WEB_AI_VECTOR_DIR` select another answer model or persistent FAISS storage directory.

Adding a website URL automatically starts its crawl and indexing; website URLs supplied when creating an assistant do the same. Research obeys each website's `robots.txt`, applies public-network URL checks and size/rate limits, and does not follow redirects or access login/paywall-protected pages. Only add sources you have permission to use. Extracted page content is untrusted input and is never executed.

Source progress moves through pending, retrieving, retrieved, processing, processed, or failed and is polled by the trainer. Chat is enabled only when selected source content has been indexed successfully; missing provider configuration is reported on the affected source.

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

Set `HF_TOKEN` and optionally `BRAVE_SEARCH_API_KEY` in the environment or a local Compose `.env` file to enable Web AI Trainer providers. The application is available at `http://localhost:8000`.

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
- `/api/v1/web-ai/models` — create and manage independent web-trained AI models
- `/api/v1/web-ai/models/{id}/sources` — manage, select, and inspect sources
- `/api/v1/web-ai/models/{id}/learn` — research and index selected sources
- `/api/v1/web-ai/models/{id}/chat` — chat with retrieved knowledge and source citations

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
