"""Extend cloth/garments.json with the bulk-imported class folders.

For each of the six class folders (T-Shirt, Blouse, Crop Top, Sweater,
Bodysuit, Tank Top), assigns each image a size round-robin across
XS/S/M/L/XL/XXL/XXXL so every class ends up with 8-9 garments per size.
Per-garment dimensions come from a women's-tops size chart, with small
deterministic jitter so no two garments share identical measurements.

Existing G1-G11 entries are preserved untouched. The original catalog is
backed up to garments.json.bak on the first run.

Usage:
    venv/Scripts/python.exe tools/build_catalog.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOTH_DIR = PROJECT_ROOT / "cloth"
CATALOG_PATH = CLOTH_DIR / "garments.json"
BACKUP_PATH = CATALOG_PATH.with_suffix(".json.bak")

CLASSES = ["T-Shirt", "Blouse", "Crop Top", "Sweater", "Bodysuit", "Tank Top"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

CLASS_PREFIX = {
    "T-Shirt":  "TSH",
    "Blouse":   "BLO",
    "Crop Top": "CRP",
    "Sweater":  "SWT",
    "Bodysuit": "BDS",
    "Tank Top": "TNK",
}

# Authoritative size charts, transcribed from cloth/garment_dimensions.xlsx.
# Each tuple holds the seven values for XS / S / M / L / XL / XXL / XXXL.
# Tank Top has no sleeve length (sleeveless) — arm_length is fixed at 0.
_CHARTS: dict[str, dict[str, tuple[int, int, int, int, int, int, int]]] = {
    "T-Shirt": {
        "length":      (61, 63, 66, 69, 72, 74,  76),
        "chest":       (84, 90, 96, 102, 108, 116, 124),
        "shoulder":    (36, 38, 40, 42, 44, 46,  48),
        "arm_length":  (18, 19, 20, 21, 22, 23,  24),
    },
    "Blouse": {
        "length":      (56, 58, 60, 62, 64, 66,  68),
        "chest":       (82, 88, 94, 100, 106, 114, 122),  # bust circumference
        "shoulder":    (34, 36, 38, 40, 42, 44,  46),
        "arm_length":  (56, 57, 58, 59, 60, 61,  62),
    },
    "Crop Top": {
        "length":      (38, 40, 42, 44, 46, 48,  50),
        "chest":       (76, 82, 88, 94, 100, 108, 116),
        "shoulder":    (33, 35, 37, 39, 41, 43,  45),
        "arm_length":  (14, 15, 16, 17, 18, 19,  20),
    },
    "Sweater": {
        "length":      (62, 64, 67, 70, 73, 75,  77),
        "chest":       (90, 96, 102, 108, 116, 124, 132),
        "shoulder":    (39, 41, 43, 45, 47, 49,  51),
        "arm_length":  (60, 61, 62, 63, 64, 65,  66),
    },
    "Bodysuit": {
        "length":      (58, 60, 62, 64, 66, 68,  70),  # torso length
        "chest":       (80, 86, 92, 98, 104, 112, 120),  # bust circumference
        "shoulder":    (33, 35, 37, 39, 41, 43,  45),
        "arm_length":  (15, 16, 17, 18, 19, 20,  21),
    },
    "Tank Top": {
        "length":      (58, 60, 63, 66, 69, 71,  73),
        "chest":       (78, 84, 90, 96, 102, 110, 118),
        "shoulder":    (32, 34, 36, 38, 40, 42,  44),
        "arm_length":  ( 0,  0,  0,  0,  0,  0,   0),  # sleeveless
    },
}


def _jitter(seed: str, amplitude: int) -> int:
    """Deterministic +/- amplitude offset, hashed from `seed`. Returns 0 when
    amplitude is 0 (used for sleeveless arm_length)."""
    if amplitude <= 0:
        return 0
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)
    return (h % (2 * amplitude + 1)) - amplitude


def _dimensions(cls: str, size: str, filename: str) -> dict[str, int]:
    """Look up the size-chart values for `cls`/`size`, then add tiny per-file
    jitter so two same-size garments differ by 1-2 cm. Stays well within
    realistic manufacturing tolerance."""
    chart = _CHARTS[cls]
    i = SIZES.index(size)

    length    = chart["length"][i]     + _jitter(f"{filename}:length", 1)
    chest     = chart["chest"][i]      + _jitter(f"{filename}:chest", 2)
    shoulder  = chart["shoulder"][i]   + _jitter(f"{filename}:shoulder", 1)
    arm       = chart["arm_length"][i]
    if arm > 0:
        arm += _jitter(f"{filename}:arm", 1)

    return {
        "length": int(length),
        "chest": int(chest),
        "shoulder": int(shoulder),
        "arm_length": int(arm),
    }


def main() -> int:
    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found")
        return 1

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not BACKUP_PATH.exists():
        shutil.copy2(CATALOG_PATH, BACKUP_PATH)
        print(f"Backed up original catalog to {BACKUP_PATH.name}")

    existing = catalog.get("garments", [])
    existing_ids = {g["id"] for g in existing}

    # Drop any previously-generated bulk entries so this script is idempotent
    # (re-running won't double-insert). They're identifiable by their id prefix.
    bulk_prefixes = tuple(f"{p}-" for p in CLASS_PREFIX.values())
    legacy = [g for g in existing if not g["id"].startswith(bulk_prefixes)]
    dropped_bulk = len(existing) - len(legacy)
    if dropped_bulk:
        print(f"Removed {dropped_bulk} previously-generated bulk entries (re-run)")

    new_entries: list[dict] = []

    for cls in CLASSES:
        cls_dir = CLOTH_DIR / cls
        if not cls_dir.is_dir():
            print(f"WARN: {cls_dir} not found, skipping")
            continue

        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() == ".jpg")
        prefix = CLASS_PREFIX[cls]
        per_size_seq: dict[str, int] = {s: 0 for s in SIZES}

        for idx, f in enumerate(files):
            size = SIZES[idx % len(SIZES)]
            per_size_seq[size] += 1
            seq = per_size_seq[size]
            gid = f"{prefix}-{size}-{seq:02d}"

            entry = {
                "id": gid,
                "name": f"{cls} {size}-{seq:02d}",
                "type": cls,
                "gender": "Women",
                "size": size,
                "fit_style": "Regular",
                "category": "top",
                "fit_relevant_keys": ["length", "chest", "shoulder", "arm_length"],
                "image": f"{cls}/{f.name}",
                "path": f"{cls}/{f.name}",
                "dimensions": _dimensions(cls, size, f.name),
            }
            new_entries.append(entry)

    catalog["garments"] = legacy + new_entries

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Summary
    print(f"\nLegacy entries kept:  {len(legacy)}")
    print(f"New entries written:  {len(new_entries)}")
    print(f"Total in catalog:     {len(legacy) + len(new_entries)}")
    print()

    cs = Counter((g["type"], g["size"]) for g in new_entries)
    header = "| Class       | " + " | ".join(f"{s:>4}" for s in SIZES) + " | Total |"
    sep    = "|" + "---|" * (len(SIZES) + 2)
    print(header)
    print(sep)
    for cls in CLASSES:
        row = [cls.ljust(11)]
        total = 0
        for s in SIZES:
            n = cs.get((cls, s), 0)
            total += n
            row.append(f"{n:>4}")
        row.append(f"{total:>5}")
        print("| " + " | ".join(row) + " |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
