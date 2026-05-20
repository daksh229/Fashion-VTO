"""Classify each garment image in cloth/cloth/ using Gemini vision.

One-off analysis tool. Reads a slice of the dataset, sends each image to
gemini-2.5-flash with a fixed taxonomy prompt, and prints a markdown
count table at the end. Per-file results are also written to
output/classifications.csv for follow-up.

Usage:
    venv/Scripts/python.exe tools/classify_garments.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
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

CLOTH_DIR = PROJECT_ROOT / "cloth" / "cloth"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "classifications.csv"

# Closed taxonomy: forces Gemini to pick one bucket so we get clean counts
# instead of a long tail of similar labels ("tee", "t shirt", "graphic tee").
TAXONOMY = [
    "T-Shirt",
    "Polo",
    "Blouse",
    "Shirt",         # button-down / dress shirt / casual long-sleeve
    "Tank Top",
    "Sweater",       # knit pullover, jumper
    "Hoodie",        # incl. sweatshirt
    "Cardigan",
    "Jacket",        # casual / denim / bomber
    "Coat",          # heavy outerwear
    "Dress",
    "Crop Top",
    "Bodysuit",
    "Vest",
    "Other",
]

PROMPT = (
    "Classify this clothing item into exactly ONE of these categories:\n"
    f"{', '.join(TAXONOMY)}\n\n"
    "Respond with only the category name from the list above. "
    "Nothing else — no punctuation, no explanation."
)


def _normalize_label(raw: str) -> str:
    """Snap whatever Gemini returns to a known taxonomy label, or 'Other'."""
    s = (raw or "").strip().strip(".").strip()
    # Fast path: exact match (case-insensitive).
    for t in TAXONOMY:
        if s.lower() == t.lower():
            return t
    # Loose match: substring.
    s_low = s.lower()
    for t in TAXONOMY:
        if t.lower() in s_low:
            return t
    return "Other"


def _classify_one(client: genai.Client, path: Path) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))  # smaller payload, no quality loss for classification
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    img_for_api = Image.open(buf)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PROMPT, img_for_api],
    )
    raw = (resp.text or "").strip()
    return path.name, _normalize_label(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: no Gemini API key in .env (api_key= or GEMINI_API_KEY=).")
        return 1

    files = sorted(p for p in CLOTH_DIR.iterdir() if p.suffix.lower() == ".jpg")
    files = files[: args.limit]
    if not files:
        print(f"No .jpg files found in {CLOTH_DIR}")
        return 1

    print(f"Classifying {len(files)} images with gemini-2.5-flash...\n")
    client = genai.Client(api_key=api_key)

    results: list[tuple[str, str]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_classify_one, client, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            f = futures[fut]
            done += 1
            try:
                name, label = fut.result()
            except Exception as e:
                name, label = f.name, "ERROR"
                print(f"  [{done:3d}/{len(files)}] {f.name}  ERROR: {type(e).__name__}: {e}")
                results.append((name, label))
                continue
            results.append((name, label))
            print(f"  [{done:3d}/{len(files)}] {name}  ->  {label}")

    elapsed = time.time() - started

    # CSV dump
    results.sort(key=lambda r: r[0])
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "category"])
        w.writerows(results)
    print(f"\nWrote {CSV_PATH}")

    # Markdown count table
    counts = Counter(label for _, label in results)
    total = sum(counts.values())
    print(f"\nClassified {total} images in {elapsed:.1f}s\n")
    print("| Category | Count | % |")
    print("|---|---:|---:|")
    for cat, n in counts.most_common():
        pct = n / total * 100 if total else 0
        print(f"| {cat} | {n} | {pct:.1f}% |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
