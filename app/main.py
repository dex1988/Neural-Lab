from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import text
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from app.database import engine
from app.routers.experiments import router as experiments_router
from app.routers.datasets import router as datasets_router
from app.routers.training import router as training_router
from app.routers.predictions import router as predictions_router
from app.routers.dashboard import router as dashboard_router
from app.routers.chat import router as chat_router
from app.routers.web_ai import router as web_ai_router
from app.schemas import HealthRead


class FrontendStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if (
                error.status_code != 404
                or path == "api"
                or path.startswith("api/")
                or Path(path).suffix
            ):
                raise
            return await super().get_response("index.html", scope)


app = FastAPI(title="Neural-Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(experiments_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(predictions_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(web_ai_router)


@app.get("/health", response_model=HealthRead, tags=["health"])
def health_check() -> HealthRead:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return HealthRead(status="ok")


frontend_directory = Path(__file__).resolve().parent.parent / "frontend"
frontend_build = frontend_directory / "dist"
if frontend_build.exists():
    app.mount("/", FrontendStaticFiles(directory=frontend_build, html=True), name="frontend")
else:
    app.mount("/", FrontendStaticFiles(directory=frontend_directory, html=True), name="frontend")
