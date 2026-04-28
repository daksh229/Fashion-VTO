"""Shared dataclasses for the body-measurement pipeline.

Each layer (preprocess → pose → scale → engine) reads/writes these types so
the seams between layers stay typed and easy to test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    landmarks: dict[str, Landmark]
    image_width: int
    image_height: int
    full_body_visible: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScaleCalibration:
    cm_per_pixel: float
    reference_pixel_height: float
    reference_real_height_cm: float
    method: Literal["nose_to_heel_corrected", "shoulder_anchor", "waist_anchor"]
    shoulder_correction: float = 1.0
    waist_correction: float = 1.0


@dataclass
class Measurements:
    """Raw measurements in cm. Widths are direct from MediaPipe; circumferences
    are computed via ellipse approximation using either side-photo depth or the
    fixed depth-to-width ratio of 0.78.
    """

    shoulder_width_cm: float | None = None
    chest_width_cm: float | None = None
    chest_circumference_cm: float | None = None
    arm_length_cm: float | None = None
    torso_length_cm: float | None = None

    waist_width_cm: float | None = None
    waist_circumference_cm: float | None = None
    hip_width_cm: float | None = None
    hip_circumference_cm: float | None = None
    inseam_cm: float | None = None
    outseam_cm: float | None = None
    thigh_width_cm: float | None = None
    thigh_circumference_cm: float | None = None


@dataclass
class MeasurementResult:
    measurements: Measurements
    confidence: dict[str, float]
    warnings: list[str]
    pose: PoseResult
    scale: ScaleCalibration
    annotated_image_bytes: bytes | None = None
    used_side_photo: bool = False


class MeasurementError(Exception):
    """Raised when the pipeline cannot produce any usable measurements."""
