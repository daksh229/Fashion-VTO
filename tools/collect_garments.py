"""Collect N images per class from cloth/cloth/, copy them into cloth/<class>/.

Reuses output/classifications.csv from a prior run so the first 100 (already
labeled) are free. Then it classifies more images in parallel batches until
every target class has reached the per-class quota, or we hit a hard scan
cap. Newly-labeled rows are appended back into the same CSV, so successive
runs only ever pay for new images.

Usage:
    venv/Scripts/python.exe tools/collect_garments.py
        [--per-class 30] [--max-scan 3000] [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CLOTH_DIR = PROJECT_ROOT / "cloth"           # destination root (frontend serves this)
SOURCE_DIR = CLOTH_DIR / "cloth"             # full dataset
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "classifications.csv"

# Active set: classes we want N of. Match the buckets that actually showed up
# in the first-100 sample. Other taxonomy buckets (Shirt, Dress, ...) are
# tracked-but-not-targeted; if any appear they get logged in the CSV but
# don't gate the loop.
TARGET_CLASSES = [
    "T-Shirt", "Blouse", "Crop Top", "Sweater",
    "Bodysuit", "Tank Top",
]

# Same closed taxonomy as the classifier, so labels normalize cleanly.
TAXONOMY = [
    "T-Shirt", "Polo", "Blouse", "Shirt", "Tank Top",
    "Sweater", "Hoodie", "Cardigan", "Jacket", "Coat",
    "Dress", "Crop Top", "Bodysuit", "Vest", "Other",
]

PROMPT = (
    "Classify this clothing item into exactly ONE of these categories:\n"
    f"{', '.join(TAXONOMY)}\n\n"
    "Respond with only the category name from the list above. "
    "Nothing else - no punctuation, no explanation."
)


def _normalize_label(raw: str) -> str:
    s = (raw or "").strip().strip(".").strip()
    for t in TAXONOMY:
        if s.lower() == t.lower():
            return t
    s_low = s.lower()
    for t in TAXONOMY:
        if t.lower() in s_low:
            return t
    return "Other"


def _classify_one(client: genai.Client, path: Path) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    img_for_api = Image.open(buf)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PROMPT, img_for_api],
    )
    return path.name, _normalize_label((resp.text or "").strip())


def _load_existing_labels() -> dict[str, str]:
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        return {row["file"]: row["category"] for row in csv.DictReader(f) if row.get("file")}


def _save_labels(labels: dict[str, str]) -> None:
    rows = sorted(labels.items())
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "category"])
        w.writerows(rows)


def _all_targets_full(buckets: dict[str, list[str]], per_class: int) -> bool:
    return all(len(buckets[c]) >= per_class for c in TARGET_CLASSES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--max-scan", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: no Gemini API key in .env (api_key= or GEMINI_API_KEY=).")
        return 1

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: source dir not found: {SOURCE_DIR}")
        return 1

    files = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() == ".jpg")
    print(f"Source pool: {len(files)} images in {SOURCE_DIR}")
    print(f"Target: {args.per_class} per class for {TARGET_CLASSES}")
    print(f"Hard cap: scan at most {args.max_scan} images\n")

    labels = _load_existing_labels()
    print(f"Loaded {len(labels)} pre-labeled rows from {CSV_PATH.name}")

    buckets: dict[str, list[str]] = {c: [] for c in TARGET_CLASSES}
    # Pre-fill from existing labels.
    for fname, label in labels.items():
        if label in buckets and len(buckets[label]) < args.per_class:
            buckets[label].append(fname)

    print("Starting bucket counts (from cache):")
    for c in TARGET_CLASSES:
        print(f"  {c:10s}  {len(buckets[c])}/{args.per_class}")
    print()

    if _all_targets_full(buckets, args.per_class):
        print("All targets already filled from cache — skipping classification.\n")
    else:
        client = genai.Client(api_key=api_key)
        unlabeled = [p for p in files if p.name not in labels]
        scanned = len(labels)
        started = time.time()

        for batch_start in range(0, len(unlabeled), args.batch):
            if scanned >= args.max_scan:
                print(f"\nHit max-scan cap of {args.max_scan}. Stopping.")
                break
            if _all_targets_full(buckets, args.per_class):
                print("\nAll targets reached. Stopping classification.")
                break

            batch = unlabeled[batch_start : batch_start + args.batch]
            print(f"-- batch {batch_start // args.batch + 1}: classifying {len(batch)} (scanned so far: {scanned})")

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(_classify_one, client, p): p for p in batch}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        name, label = fut.result()
                    except Exception as e:
                        labels[p.name] = "ERROR"
                        print(f"   {p.name} ERROR: {type(e).__name__}: {e}")
                        continue
                    labels[name] = label
                    if label in buckets and len(buckets[label]) < args.per_class:
                        buckets[label].append(name)

            scanned = len(labels)
            elapsed = time.time() - started
            print(f"   bucket fill: " + "  ".join(
                f"{c}={len(buckets[c])}" for c in TARGET_CLASSES
            ))
            print(f"   elapsed: {elapsed:.0f}s")
            _save_labels(labels)  # persist after every batch — recoverable

        _save_labels(labels)
        print(f"\nClassified total: {len(labels)} images, {time.time() - started:.0f}s elapsed")

    # ------------------------------------------------------------------ copy
    print("\nCopying selected files into cloth/<class>/ ...")
    copied_total = 0
    for cls in TARGET_CLASSES:
        dest = CLOTH_DIR / cls
        dest.mkdir(exist_ok=True)
        chosen = buckets[cls][: args.per_class]
        for fname in chosen:
            src = SOURCE_DIR / fname
            dst = dest / fname
            if not dst.exists():
                shutil.copy2(src, dst)
            copied_total += 1
        print(f"  {cls:10s}  {len(chosen):3d} files  ->  cloth/{cls}/")

    # ------------------------------------------------------------------ table
    print(f"\nDone. {copied_total} files placed under cloth/<class>/.\n")
    print("| Category | Selected | Available | Quota |")
    print("|---|---:|---:|---:|")
    full_counts = Counter(labels.values())
    for cls in TARGET_CLASSES:
        sel = len(buckets[cls])
        avail = full_counts[cls]
        print(f"| {cls} | {sel} | {avail} | {args.per_class} |")
    other_total = sum(n for c, n in full_counts.items() if c not in TARGET_CLASSES)
    if other_total:
        print(f"\nLogged outside targets (not copied): {other_total}")
        for c, n in full_counts.most_common():
            if c not in TARGET_CLASSES:
                print(f"  {c}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
