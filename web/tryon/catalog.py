"""Read-only access to `cloth/garments.json`.

Per the project decision (no `Garment` model, JSON stays canonical),
this module is the only place the web side touches the catalog file.
Cached by mtime so we re-read transparently if the user edits the JSON
between requests in dev.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

from django.conf import settings


_lock = threading.Lock()
_cache: dict[str, object] = {"mtime": None, "data": None}


def load_catalog() -> dict:
    """Return the parsed catalog dict, re-reading the file if it changed."""
    path = settings.CATALOG_PATH
    mtime = path.stat().st_mtime
    with _lock:
        if _cache["mtime"] != mtime:
            with open(path, "r", encoding="utf-8") as f:
                _cache["data"] = json.load(f)
            _cache["mtime"] = mtime
        return _cache["data"]  # type: ignore[return-value]


def list_garments(category: Optional[str] = None) -> list[dict]:
    """Return all garments, optionally filtered by `category` (e.g. 'top')."""
    garments = load_catalog().get("garments", [])
    if category is None:
        return garments
    return [g for g in garments if g.get("category", "top") == category]


def find_garment(garment_id: str) -> Optional[dict]:
    return next((g for g in load_catalog().get("garments", []) if g["id"] == garment_id), None)
