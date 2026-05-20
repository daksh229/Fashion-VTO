"""Pydantic request/response schemas for the API.

Mirrors the dataclasses in `src/measurement/types.py` but with all-optional
measurement fields to match the engine's "report what we could compute"
behavior.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MeasurementsOut(BaseModel):
    shoulder_width_cm: Optional[float] = None
    chest_width_cm: Optional[float] = None
    chest_circumference_cm: Optional[float] = None
    arm_length_cm: Optional[float] = None
    torso_length_cm: Optional[float] = None
    waist_width_cm: Optional[float] = None
    waist_circumference_cm: Optional[float] = None
    hip_width_cm: Optional[float] = None
    hip_circumference_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    outseam_cm: Optional[float] = None
    thigh_width_cm: Optional[float] = None
    thigh_circumference_cm: Optional[float] = None


class MeasureResponse(BaseModel):
    measurements: MeasurementsOut
    confidence: dict[str, float]
    warnings: list[str]
    annotated_image_b64: Optional[str] = None
    used_side_photo: bool


class HealthResponse(BaseModel):
    status: str
    pose_loaded: bool


class TryonStartResponse(BaseModel):
    job_id: str
    status: str


class TryonStatusResponse(BaseModel):
    job_id: str
    status: str
    image_url: Optional[str] = None
    fit: Optional[dict] = None
    prompt_used: Optional[str] = None
    # Live-streaming fields. `prompt_text` accumulates as Groq writes the
    # prompt; `prompt_streaming` is True only while Groq is actively yielding.
    prompt_text: str = ""
    prompt_streaming: bool = False
    error: Optional[str] = None
