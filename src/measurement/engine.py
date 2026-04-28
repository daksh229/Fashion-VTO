"""Compute body measurements (cm) from pose landmarks + scale calibration.

Widths come straight from MediaPipe via Euclidean distance × scale. Lengths
(arm, inseam) sum two segments. Circumferences are estimated as the perimeter
of an ellipse whose semi-axes are (width/2) and (depth/2). Depth is taken from
a side-photo pose if available, else estimated as `DEPTH_TO_WIDTH_RATIO * width`.
"""

from __future__ import annotations

import math
from typing import Tuple

from .types import Landmark, Measurements, PoseResult, ScaleCalibration


DEPTH_TO_WIDTH_RATIO = 0.78
THIGH_WIDTH_FACTOR = 0.575
LANDMARK_VISIBILITY_THRESHOLD = 0.5
THIGH_DERIVED_CONFIDENCE = 0.6
DERIVED_CIRCUMFERENCE_PENALTY = 0.85


def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _avg_visibility(*landmarks: Landmark) -> float:
    if not landmarks:
        return 0.0
    return sum(lm.visibility for lm in landmarks) / len(landmarks)


def _all_visible(*landmarks: Landmark, threshold: float = LANDMARK_VISIBILITY_THRESHOLD) -> bool:
    return all(lm.visibility >= threshold for lm in landmarks)


def ellipse_perimeter(width: float, depth: float) -> float:
    """Ramanujan's second approximation for ellipse perimeter."""
    if width <= 0 or depth <= 0:
        return 0.0
    a, b = width / 2.0, depth / 2.0
    h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0.0
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def _interpolate(a: Landmark, b: Landmark, t: float) -> Tuple[float, float]:
    return a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t


def _measure_side_depth(
    side_pose: PoseResult,
    side_scale: ScaleCalibration,
    landmark_a: str,
    landmark_b: str,
) -> float | None:
    """Return body depth in cm at the given landmarks from a side photo.

    On a side view, the same body landmarks (e.g. shoulders) project so that
    the front-back depth shows up as horizontal distance in the image. We use
    the absolute x-distance (in pixels) between the two landmarks as a depth
    proxy and convert via the side-photo scale.
    """
    if landmark_a not in side_pose.landmarks or landmark_b not in side_pose.landmarks:
        return None
    la = side_pose.landmarks[landmark_a]
    lb = side_pose.landmarks[landmark_b]
    if la.visibility < LANDMARK_VISIBILITY_THRESHOLD or lb.visibility < LANDMARK_VISIBILITY_THRESHOLD:
        return None
    pixel_depth = abs(la.x - lb.x)
    return pixel_depth * side_scale.cm_per_pixel


def _circumference_from_width(width_cm: float, depth_cm: float | None) -> tuple[float, bool]:
    if depth_cm is None or depth_cm <= 0:
        return ellipse_perimeter(width_cm, width_cm * DEPTH_TO_WIDTH_RATIO), False
    return ellipse_perimeter(width_cm, depth_cm), True


def measure(
    pose: PoseResult,
    scale: ScaleCalibration,
    side_pose: PoseResult | None = None,
    side_scale: ScaleCalibration | None = None,
) -> tuple[Measurements, dict[str, float]]:
    m = Measurements()
    conf: dict[str, float] = {}

    L = pose.landmarks
    s_corr = scale.shoulder_correction
    w_corr = scale.waist_correction

    # Shoulder width
    if _all_visible(L["left_shoulder"], L["right_shoulder"]):
        m.shoulder_width_cm = (
            _dist(L["left_shoulder"], L["right_shoulder"]) * scale.cm_per_pixel * s_corr
        )
        conf["shoulder"] = _avg_visibility(L["left_shoulder"], L["right_shoulder"])

    # Chest width — interpolate ~15% down from shoulder midpoint toward hip midpoint
    if _all_visible(L["left_shoulder"], L["right_shoulder"], L["left_hip"], L["right_hip"]):
        ls, rs = L["left_shoulder"], L["right_shoulder"]
        lh, rh = L["left_hip"], L["right_hip"]
        # Estimate chest as the shoulder-line shrunk slightly (chest is narrower than shoulders).
        chest_axis = _dist(ls, rs)
        chest_width_px = chest_axis * 0.92
        m.chest_width_cm = chest_width_px * scale.cm_per_pixel * s_corr
        conf["chest_width"] = _avg_visibility(ls, rs, lh, rh)

    # Arm length (sum of shoulder→elbow + elbow→wrist on the more-visible side)
    arm_pairs = []
    for side in ("left", "right"):
        s, e, w = L[f"{side}_shoulder"], L[f"{side}_elbow"], L[f"{side}_wrist"]
        if _all_visible(s, e, w):
            arm_pairs.append((side, s, e, w, _avg_visibility(s, e, w)))
    if arm_pairs:
        arm_pairs.sort(key=lambda x: x[-1], reverse=True)
        _, s, e, w, vis = arm_pairs[0]
        m.arm_length_cm = (_dist(s, e) + _dist(e, w)) * scale.cm_per_pixel * s_corr
        conf["arm_length"] = vis

    # Torso length: mid-shoulder → mid-hip
    if _all_visible(L["left_shoulder"], L["right_shoulder"], L["left_hip"], L["right_hip"]):
        ls, rs = L["left_shoulder"], L["right_shoulder"]
        lh, rh = L["left_hip"], L["right_hip"]
        mid_shoulder_y = (ls.y + rs.y) / 2.0
        mid_hip_y = (lh.y + rh.y) / 2.0
        torso_px = abs(mid_hip_y - mid_shoulder_y)
        m.torso_length_cm = torso_px * scale.cm_per_pixel
        conf["length"] = _avg_visibility(ls, rs, lh, rh)

    # Hip width (direct)
    if _all_visible(L["left_hip"], L["right_hip"]):
        m.hip_width_cm = _dist(L["left_hip"], L["right_hip"]) * scale.cm_per_pixel
        conf["hip_width"] = _avg_visibility(L["left_hip"], L["right_hip"])

    # Waist width — interpolate roughly 35% up from hip to shoulder midline
    if _all_visible(L["left_shoulder"], L["right_shoulder"], L["left_hip"], L["right_hip"]):
        ls, rs = L["left_shoulder"], L["right_shoulder"]
        lh, rh = L["left_hip"], L["right_hip"]
        # Waist sits above the hip line; assume hip width × 0.92 as proxy.
        m.waist_width_cm = _dist(lh, rh) * scale.cm_per_pixel * 0.92 * w_corr
        conf["waist_width"] = _avg_visibility(ls, rs, lh, rh) * 0.9

    # Inseam: mid-hip → knee → ankle on the more-visible side
    leg_pairs = []
    for side in ("left", "right"):
        h, k, a = L[f"{side}_hip"], L[f"{side}_knee"], L[f"{side}_ankle"]
        if _all_visible(h, k, a):
            leg_pairs.append((side, h, k, a, _avg_visibility(h, k, a)))
    if leg_pairs:
        leg_pairs.sort(key=lambda x: x[-1], reverse=True)
        _, h, k, a, vis = leg_pairs[0]
        m.inseam_cm = (_dist(h, k) + _dist(k, a)) * scale.cm_per_pixel
        # Outseam ≈ vertical distance from hip to ankle (outer leg)
        m.outseam_cm = abs(a.y - h.y) * scale.cm_per_pixel
        conf["inseam"] = vis
        conf["outseam"] = vis

    # Thigh width — derived from hip width (no direct landmark)
    if m.hip_width_cm is not None:
        m.thigh_width_cm = m.hip_width_cm * THIGH_WIDTH_FACTOR
        conf["thigh_width"] = THIGH_DERIVED_CONFIDENCE

    # Circumferences (chest, waist, hip, thigh)
    chest_depth = (
        _measure_side_depth(side_pose, side_scale, "left_shoulder", "right_shoulder")
        if side_pose is not None and side_scale is not None
        else None
    )
    waist_depth = (
        _measure_side_depth(side_pose, side_scale, "left_hip", "right_hip")
        if side_pose is not None and side_scale is not None
        else None
    )
    hip_depth = waist_depth

    if m.chest_width_cm is not None:
        circ, used_real_depth = _circumference_from_width(m.chest_width_cm, chest_depth)
        m.chest_circumference_cm = circ
        conf["chest"] = conf.get("chest_width", 0.0) * (1.0 if used_real_depth else DERIVED_CIRCUMFERENCE_PENALTY)

    if m.waist_width_cm is not None:
        circ, used_real_depth = _circumference_from_width(m.waist_width_cm, waist_depth)
        m.waist_circumference_cm = circ
        conf["waist"] = conf.get("waist_width", 0.0) * (1.0 if used_real_depth else DERIVED_CIRCUMFERENCE_PENALTY)

    if m.hip_width_cm is not None:
        circ, used_real_depth = _circumference_from_width(m.hip_width_cm, hip_depth)
        m.hip_circumference_cm = circ
        conf["hip"] = conf.get("hip_width", 0.0) * (1.0 if used_real_depth else DERIVED_CIRCUMFERENCE_PENALTY)

    if m.thigh_width_cm is not None:
        thigh_depth = m.thigh_width_cm * DEPTH_TO_WIDTH_RATIO
        m.thigh_circumference_cm = ellipse_perimeter(m.thigh_width_cm, thigh_depth)
        conf["thigh"] = THIGH_DERIVED_CONFIDENCE * DERIVED_CIRCUMFERENCE_PENALTY

    return m, conf
