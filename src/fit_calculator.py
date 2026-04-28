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
    fit_keys = garment.get("fit_relevant_keys") or list(g_dims.keys())
    deltas = {
        key: round(g_dims[key] - person_dimensions[key], 2)
        for key in fit_keys
        if key in g_dims and key in person_dimensions
    }

    return {
        "garment_id": garment["id"],
        "garment_name": garment["name"],
        "garment_size": garment.get("size"),
        "garment_fit_style": garment.get("fit_style"),
        "category": garment.get("category", "top"),
        "image_path": garment["path"],
        "units": catalog.get("units", "cm"),
        "person_dimensions": {k: person_dimensions[k] for k in fit_keys if k in person_dimensions},
        "garment_dimensions": {k: g_dims[k] for k in fit_keys if k in g_dims},
        "deltas": deltas,
    }


if __name__ == "__main__":
    person = {"length": 66, "chest": 91, "shoulder": 38, "arm_length": 23}
    result = calculate_fit(person, "G2")
    print(json.dumps(result, indent=2))
