"""Pixel → cm scale calibration.

Primary method: nose-to-heel pixel distance × 1.087 → estimated full pixel height.
The 1.087 factor compensates for the missing nose-to-crown segment, which is
empirically ~8% of total adult height. This is a fixed correction (not derived
from segmentation) — see the body-measurement memory file for the trade-off.

Optional shoulder/waist anchors produce regional correction multipliers that
the measurement engine can apply per region.
"""

from __future__ import annotations

import math

from .types import MeasurementError, PoseResult, ScaleCalibration


NOSE_TO_CROWN_FACTOR = 1.087
HEIGHT_MIN_CM = 100.0
HEIGHT_MAX_CM = 230.0


def _euclidean(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def calibrate_scale(
    pose: PoseResult,
    height_cm: float,
    shoulder_cm: float | None = None,
    waist_cm: float | None = None,
) -> ScaleCalibration:
    if not (HEIGHT_MIN_CM <= height_cm <= HEIGHT_MAX_CM):
        raise MeasurementError(
            f"Height {height_cm:.1f} cm is out of plausible range "
            f"({HEIGHT_MIN_CM}–{HEIGHT_MAX_CM} cm)."
        )

    nose = pose.landmarks["nose"]
    left_heel = pose.landmarks["left_heel"]
    right_heel = pose.landmarks["right_heel"]
    left_shoulder = pose.landmarks["left_shoulder"]
    right_shoulder = pose.landmarks["right_shoulder"]

    heels_visible = max(left_heel.visibility, right_heel.visibility) >= 0.5
    nose_visible = nose.visibility >= 0.3

    if heels_visible and nose_visible:
        heel_y = (
            left_heel.y if left_heel.visibility >= right_heel.visibility else right_heel.y
        )
        pixel_nose_to_heel = abs(heel_y - nose.y)
        pixel_height = pixel_nose_to_heel * NOSE_TO_CROWN_FACTOR
        cm_per_pixel = height_cm / pixel_height
        method = "nose_to_heel_corrected"
        ref_pixel = pixel_height
    elif shoulder_cm is not None:
        pixel_shoulder = _euclidean(left_shoulder, right_shoulder)
        if pixel_shoulder <= 0:
            raise MeasurementError("Could not measure shoulder pixels for fallback calibration.")
        cm_per_pixel = shoulder_cm / pixel_shoulder
        method = "shoulder_anchor"
        ref_pixel = pixel_shoulder
    else:
        raise MeasurementError(
            "Cannot calibrate scale: feet are not visible and no shoulder anchor was "
            "provided. Either retake the photo with feet in frame or enter a shoulder "
            "width in the optional anchors."
        )

    shoulder_correction = 1.0
    if shoulder_cm is not None and method != "shoulder_anchor":
        pixel_shoulder = _euclidean(left_shoulder, right_shoulder)
        measured_shoulder_cm = pixel_shoulder * cm_per_pixel
        if measured_shoulder_cm > 0:
            shoulder_correction = shoulder_cm / measured_shoulder_cm

    waist_correction = 1.0
    if waist_cm is not None:
        left_hip = pose.landmarks["left_hip"]
        right_hip = pose.landmarks["right_hip"]
        pixel_hip = _euclidean(left_hip, right_hip)
        # waist_cm is provided as circumference; back out an implied width using the
        # default depth ratio so we can compute a correction against the measured width
        from .engine import DEPTH_TO_WIDTH_RATIO, ellipse_perimeter
        if pixel_hip > 0:
            measured_hip_width = pixel_hip * cm_per_pixel
            implied_waist_width = _invert_circumference(waist_cm, DEPTH_TO_WIDTH_RATIO)
            if implied_waist_width and measured_hip_width > 0:
                waist_correction = implied_waist_width / measured_hip_width

    return ScaleCalibration(
        cm_per_pixel=cm_per_pixel,
        reference_pixel_height=ref_pixel,
        reference_real_height_cm=height_cm,
        method=method,  # type: ignore[arg-type]
        shoulder_correction=shoulder_correction,
        waist_correction=waist_correction,
    )


def _invert_circumference(circumference_cm: float, depth_ratio: float) -> float | None:
    """Given a circumference and a fixed depth-to-width ratio, solve for width
    such that ellipse_perimeter(width, depth=depth_ratio*width) = circumference.

    Closed form: for ellipse with a=W/2, b=ratio*W/2, Ramanujan #1 gives
    C ≈ π*(a+b) = π*W*(1+ratio)/2, so W ≈ 2C / (π*(1+ratio)).
    """
    if circumference_cm <= 0:
        return None
    return (2.0 * circumference_cm) / (math.pi * (1.0 + depth_ratio))
