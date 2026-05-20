"""POST /api/measure — run the body-measurement pipeline on uploaded photos.

Wraps `src.measurement.run_measurement_pipeline` and returns the JSON
contract Django will consume. The annotated pose image is base64-encoded
inline so the caller doesn't need a second round-trip to fetch it.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.measurement import MeasurementError, run_measurement_pipeline

from ..deps import get_pose_solution
from ..schemas import MeasureResponse, MeasurementsOut

router = APIRouter()


def _coerce_optional_float(value: Optional[float]) -> Optional[float]:
    """Treat 0/None as 'not provided'.

    Streamlit's number_input emits 0.0 for empty optional fields, so we
    keep the same convention here: 0 means "skip this anchor".
    """
    if value is None or value <= 0:
        return None
    return value


async def _read_optional_upload(upload: Optional[UploadFile]) -> Optional[bytes]:
    """Return bytes only if the upload is a real file with content."""
    if upload is None or not upload.filename:
        return None
    data = await upload.read()
    return data or None


@router.post("/measure", response_model=MeasureResponse)
async def measure(
    frontal_image: UploadFile = File(...),
    height_cm: float = Form(...),
    side_image: Optional[UploadFile] = File(None),
    shoulder_cm: Optional[float] = Form(None),
    waist_cm: Optional[float] = Form(None),
    pose_solution=Depends(get_pose_solution),
) -> MeasureResponse:
    if not frontal_image.filename:
        raise HTTPException(status_code=400, detail="frontal_image is required.")

    frontal_bytes = await frontal_image.read()
    if not frontal_bytes:
        raise HTTPException(status_code=400, detail="frontal_image is empty.")

    side_bytes = await _read_optional_upload(side_image)

    try:
        result = run_measurement_pipeline(
            frontal_bytes=frontal_bytes,
            height_cm=height_cm,
            side_bytes=side_bytes,
            shoulder_cm=_coerce_optional_float(shoulder_cm),
            waist_cm=_coerce_optional_float(waist_cm),
            pose_solution=pose_solution,
        )
    except MeasurementError as e:
        raise HTTPException(status_code=422, detail=str(e))

    annotated_b64 = (
        base64.b64encode(result.annotated_image_bytes).decode("ascii")
        if result.annotated_image_bytes
        else None
    )

    return MeasureResponse(
        measurements=MeasurementsOut(**asdict(result.measurements)),
        confidence=result.confidence,
        warnings=result.warnings,
        annotated_image_b64=annotated_b64,
        used_side_photo=result.used_side_photo,
    )
