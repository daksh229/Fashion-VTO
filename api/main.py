"""FastAPI entry point for the Fashion Try-On service.

Boot order matters: the MediaPipe PoseLandmarker is heavy to instantiate
(model file load + native init), so we build it once during the lifespan
startup and stash it on `app.state` for routers to reuse via Depends.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.measurement import _create_pose_solution  # noqa: E402

from .jobs import OUTPUT_DIR  # noqa: E402
from .routers import health, measure, tryon  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pose_solution = _create_pose_solution()
    app.state.jobs = {}
    try:
        yield
    finally:
        try:
            app.state.pose_solution.close()
        except Exception:
            pass


app = FastAPI(
    title="Fashion Try-On API",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(measure.router, prefix="/api", tags=["measure"])
app.include_router(tryon.router, prefix="/api", tags=["tryon"])

app.mount(
    "/static/output",
    StaticFiles(directory=OUTPUT_DIR),
    name="static-output",
)
