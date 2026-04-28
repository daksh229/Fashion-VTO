"""Map pipeline measurements (cm) to the dimension keys the garment catalog uses.

Catalog units are cm (post-migration), so this layer is purely a key rename
plus a width→circumference choice for chest/waist/hip/thigh. Splitting it out
of the engine keeps unit/key concerns at a single boundary that's easy to
adjust if the catalog schema changes.
"""

from __future__ import annotations

from .types import Measurements


_TOP_KEY_MAP = {
    "length": "torso_length_cm",
    "chest": "chest_circumference_cm",
    "shoulder": "shoulder_width_cm",
    "arm_length": "arm_length_cm",
}

_BOTTOM_KEY_MAP = {
    "length": "outseam_cm",
    "waist": "waist_circumference_cm",
    "hip": "hip_circumference_cm",
    "inseam": "inseam_cm",
    "outseam": "outseam_cm",
    "thigh": "thigh_circumference_cm",
}

_CONFIDENCE_KEY_FOR_CATALOG_KEY = {
    "length": "length",
    "chest": "chest",
    "shoulder": "shoulder",
    "arm_length": "arm_length",
    "waist": "waist",
    "hip": "hip",
    "inseam": "inseam",
    "outseam": "outseam",
    "thigh": "thigh",
}


def measurements_to_person_dimensions(
    m: Measurements,
    fit_relevant_keys: list[str],
    category: str = "top",
) -> dict[str, float | None]:
    key_map = _BOTTOM_KEY_MAP if category == "bottom" else _TOP_KEY_MAP
    out: dict[str, float | None] = {}
    for key in fit_relevant_keys:
        attr = key_map.get(key)
        out[key] = getattr(m, attr, None) if attr else None
    return out


def confidence_for_catalog_keys(
    confidence: dict[str, float],
    fit_relevant_keys: list[str],
) -> dict[str, float]:
    return {
        key: confidence.get(_CONFIDENCE_KEY_FOR_CATALOG_KEY.get(key, key), 0.0)
        for key in fit_relevant_keys
    }
