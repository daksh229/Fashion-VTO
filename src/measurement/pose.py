"""MediaPipe Pose wrapper (Tasks API).

Newer mediapipe wheels (0.10.30+) on Windows ship only the Tasks API and drop
the legacy `mp.solutions.pose` module, so we use `mp.tasks.vision.PoseLandmarker`
directly. The model file is downloaded once on first use and cached under the
user's home directory.

Returns a `PoseResult` with semantic landmark names (e.g. `"left_shoulder"`)
in pixel coordinates so the rest of the pipeline never reaches into MediaPipe's
33-index enum. The detector is heavy to instantiate, so the caller (Streamlit)
should cache it via `@st.cache_resource`.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from .types import Landmark, MeasurementError, PoseResult


_POSE_LANDMARKS: dict[str, int] = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

VISIBILITY_THRESHOLD = 0.5

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe"
_MODEL_FILE = _MODEL_CACHE_DIR / "pose_landmarker_lite.task"


def _ensure_model() -> str:
    if _MODEL_FILE.exists() and _MODEL_FILE.stat().st_size > 0:
        return str(_MODEL_FILE)
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_FILE.with_suffix(".task.part")
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        os.replace(tmp, _MODEL_FILE)
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise MeasurementError(
            f"Could not download MediaPipe pose model from {_MODEL_URL}: {e}"
        )
    return str(_MODEL_FILE)


def _create_pose_solution(model_complexity: int = 1):
    """Return a configured `PoseLandmarker` (Tasks API).

    `model_complexity` is accepted for API compatibility with the legacy
    `solutions` API but is unused — the lite model is hardcoded for now.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def detect_pose(img: np.ndarray, pose_solution=None) -> PoseResult:
    import mediapipe as mp

    own_solution = pose_solution is None
    detector = pose_solution or _create_pose_solution()

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
        result = detector.detect(mp_image)
    finally:
        if own_solution:
            detector.close()

    if not result.pose_landmarks:
        raise MeasurementError(
            "No body detected in the photo. Make sure the full body is visible, "
            "the background is reasonably plain, and the lighting is even."
        )

    h, w = img.shape[:2]
    landmarks: dict[str, Landmark] = {}
    person_landmarks = result.pose_landmarks[0]
    for name, idx in _POSE_LANDMARKS.items():
        lm = person_landmarks[idx]
        landmarks[name] = Landmark(
            x=lm.x * w,
            y=lm.y * h,
            z=lm.z * w,
            visibility=getattr(lm, "visibility", 1.0),
        )

    warnings: list[str] = []
    nose_v = landmarks["nose"].visibility
    left_heel_v = landmarks["left_heel"].visibility
    right_heel_v = landmarks["right_heel"].visibility
    full_body = (
        nose_v >= VISIBILITY_THRESHOLD
        and (left_heel_v >= VISIBILITY_THRESHOLD or right_heel_v >= VISIBILITY_THRESHOLD)
    )

    if nose_v < VISIBILITY_THRESHOLD:
        warnings.append("Face not clearly visible — height calibration may be less accurate.")
    if left_heel_v < VISIBILITY_THRESHOLD and right_heel_v < VISIBILITY_THRESHOLD:
        warnings.append("Feet not visible — provide a shoulder anchor for better calibration.")

    return PoseResult(
        landmarks=landmarks,
        image_width=w,
        image_height=h,
        full_body_visible=full_body,
        warnings=warnings,
    )


def annotate_image(img: np.ndarray, pose: PoseResult) -> bytes:
    import cv2

    annotated = img.copy()
    skeleton_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]
    for a, b in skeleton_pairs:
        la, lb = pose.landmarks[a], pose.landmarks[b]
        if la.visibility < VISIBILITY_THRESHOLD or lb.visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.line(
            annotated,
            (int(la.x), int(la.y)), (int(lb.x), int(lb.y)),
            (0, 200, 0), 2,
        )
    for lm in pose.landmarks.values():
        if lm.visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.circle(annotated, (int(lm.x), int(lm.y)), 4, (255, 80, 80), -1)

    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return b""
    return buf.tobytes()
