"""Calculate the fit delta between a person and a garment.

Delta = garment_dimension - person_dimension
  negative -> garment is smaller than person (tight)
  positive -> garment is larger than person (loose)
"""

import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "cloth" / "garments.json"


def calculate_fit(person_dimensions: dict, cloth_id: str, catalog_path: Path = CATALOG_PATH) -> dict:
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    garment = next((g for g in catalog["garments"] if g["id"] == cloth_id), None)
    if garment is None:
        raise ValueError(f"Garment id '{cloth_id}' not found in catalog")

    g_dims = garment["dimensions"]
    deltas = {
        key: round(g_dims[key] - person_dimensions[key], 2)
        for key in g_dims
        if key in person_dimensions
    }

    return {
        "garment_id": garment["id"],
        "garment_name": garment["name"],
        "garment_size": garment.get("size"),
        "garment_fit_style": garment.get("fit_style"),
        "garment_category": garment.get("category", "Top"),
        "garment_type": garment.get("type"),
        "image_path": garment["path"],
        "units": catalog.get("units", "inches"),
        "person_dimensions": person_dimensions,
        "garment_dimensions": g_dims,
        "deltas": deltas,
    }


if __name__ == "__main__":
    person = {"length": 26, "chest": 36, "shoulder": 15, "arm_length": 9}
    result = calculate_fit(person, "G2")
    print(json.dumps(result, indent=2))
