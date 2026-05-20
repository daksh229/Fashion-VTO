"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request


def get_pose_solution(request: Request):
    """Return the cached MediaPipe PoseLandmarker built during app startup."""
    return request.app.state.pose_solution
