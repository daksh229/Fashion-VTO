"""Forms for the measurement upload flow.

`MeasureForm` accepts the same inputs the FastAPI `/api/measure` endpoint
expects: a frontal photo, a height, and optional side photo + correction
anchors. `VerifyForm` collects the four top-category dimension keys after
the user has reviewed detected values.
"""

from __future__ import annotations

from django import forms

# Hardcoded for Phase 3 (top-only catalog). Phase 4 will dispatch by
# garment.category and use the matching set of keys.
TOP_DIMENSION_KEYS: list[tuple[str, str]] = [
    ("length", "Length / torso (cm)"),
    ("chest", "Chest circumference (cm)"),
    ("shoulder", "Shoulder width (cm)"),
    ("arm_length", "Arm length (cm)"),
]


class MeasureForm(forms.Form):
    frontal_image = forms.ImageField(label="Frontal photo")
    side_image = forms.ImageField(label="Side photo (optional)", required=False)
    height_cm = forms.FloatField(
        label="Height (cm)",
        min_value=100.0,
        max_value=230.0,
        initial=170.0,
    )
    shoulder_cm = forms.FloatField(
        label="Shoulder width (cm) — optional",
        required=False,
        min_value=0.0,
        max_value=80.0,
    )
    waist_cm = forms.FloatField(
        label="Waist circumference (cm) — optional",
        required=False,
        min_value=0.0,
        max_value=200.0,
    )


class VerifyForm(forms.Form):
    """Editable detected values. Field names match the catalog keys so the
    cleaned payload can be passed straight to `/api/tryon` later.
    """

    def __init__(self, *args, initial_dims: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for key, label in TOP_DIMENSION_KEYS:
            self.fields[key] = forms.FloatField(
                label=label,
                min_value=0.0,
                max_value=300.0,
                initial=(initial_dims or {}).get(key),
            )
