"""Liveness/readiness probe.

Reports whether the MediaPipe PoseLandmarker finished loading. Useful for
orchestration (don't route traffic until pose is ready) and for the
upcoming Django side to short-circuit before forwarding requests.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pose_loaded = getattr(request.app.state, "pose_solution", None) is not None
    return HealthResponse(status="ok", pose_loaded=pose_loaded)
