"""Body measurement pipeline: photo bytes + reference height → cm measurements.

Public surface intentionally small — `run_measurement_pipeline` is the single
entry point for the Streamlit UI; the rest is exposed for tests and tooling.
"""

from .catalog_mapping import (
    confidence_for_catalog_keys,
    measurements_to_person_dimensions,
)
from .pipeline import run_measurement_pipeline
from .pose import _create_pose_solution
from .types import (
    Landmark,
    MeasurementError,
    MeasurementResult,
    Measurements,
    PoseResult,
    ScaleCalibration,
)

__all__ = [
    "run_measurement_pipeline",
    "measurements_to_person_dimensions",
    "confidence_for_catalog_keys",
    "_create_pose_solution",
    "Landmark",
    "Measurements",
    "MeasurementResult",
    "MeasurementError",
    "PoseResult",
    "ScaleCalibration",
]
