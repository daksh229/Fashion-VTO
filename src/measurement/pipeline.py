"""End-to-end orchestrator: bytes in → MeasurementResult out.

The Streamlit UI calls `run_measurement_pipeline` exactly once per detection.
Per-layer errors are caught and folded into `result.warnings` rather than
raising, so the UI can always show *something* and let the user fall back to
manual entry.
"""

from __future__ import annotations

from .engine import measure
from .pose import annotate_image, detect_pose
from .preprocess import decode_image
from .scale import calibrate_scale
from .types import MeasurementError, MeasurementResult


def run_measurement_pipeline(
    frontal_bytes: bytes,
    height_cm: float,
    side_bytes: bytes | None = None,
    shoulder_cm: float | None = None,
    waist_cm: float | None = None,
    annotate: bool = True,
    pose_solution=None,
) -> MeasurementResult:
    frontal_img = decode_image(frontal_bytes)
    pose = detect_pose(frontal_img, pose_solution=pose_solution)
    scale = calibrate_scale(pose, height_cm, shoulder_cm=shoulder_cm, waist_cm=waist_cm)

    side_pose = None
    side_scale = None
    used_side = False
    side_warning = None
    if side_bytes:
        try:
            side_img = decode_image(side_bytes)
            side_pose = detect_pose(side_img, pose_solution=pose_solution)
            side_scale = calibrate_scale(
                side_pose, height_cm, shoulder_cm=shoulder_cm, waist_cm=waist_cm
            )
            used_side = True
        except MeasurementError as e:
            side_warning = f"Side photo skipped: {e}"
            side_pose = None
            side_scale = None

    measurements, confidence = measure(pose, scale, side_pose=side_pose, side_scale=side_scale)

    warnings = list(pose.warnings)
    if not pose.full_body_visible:
        warnings.append("Full body not detected — some measurements may be missing or imprecise.")
    if side_warning:
        warnings.append(side_warning)
    if not used_side:
        warnings.append(
            "No side photo provided — circumferences (chest, waist, hip, thigh) "
            "are estimated using a fixed depth ratio."
        )

    annotated = annotate_image(frontal_img, pose) if annotate else None

    return MeasurementResult(
        measurements=measurements,
        confidence=confidence,
        warnings=warnings,
        pose=pose,
        scale=scale,
        annotated_image_bytes=annotated,
        used_side_photo=used_side,
    )
