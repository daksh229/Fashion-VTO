"""HTTP client wrapper around the FastAPI service.

Centralizing this here keeps views free of httpx specifics and gives a
single seam to mock in tests later. Timeouts are tuned to the slowest
real call (Gemini try-on can take 30 s).
"""

from __future__ import annotations

import json
from typing import Optional

import httpx
from django.conf import settings


class ApiError(Exception):
    """Raised when FastAPI responds non-2xx or the network call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FastAPIClient:
    """Thin wrapper. One instance per request is fine — httpx.Client is cheap."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.API_BASE_URL).rstrip("/")

    def measure(
        self,
        frontal_bytes: bytes,
        frontal_filename: str,
        height_cm: float,
        side_bytes: Optional[bytes] = None,
        side_filename: Optional[str] = None,
        shoulder_cm: Optional[float] = None,
        waist_cm: Optional[float] = None,
    ) -> dict:
        files = [
            ("frontal_image", (frontal_filename, frontal_bytes, "application/octet-stream")),
        ]
        if side_bytes:
            files.append(
                ("side_image", (side_filename or "side.jpg", side_bytes, "application/octet-stream"))
            )

        data: dict[str, str] = {"height_cm": str(height_cm)}
        if shoulder_cm is not None:
            data["shoulder_cm"] = str(shoulder_cm)
        if waist_cm is not None:
            data["waist_cm"] = str(waist_cm)

        with httpx.Client(timeout=30.0) as client:
            try:
                resp = client.post(f"{self.base_url}/api/measure", files=files, data=data)
            except httpx.HTTPError as e:
                raise ApiError(f"could not reach FastAPI ({self.base_url}): {e}") from e

        if resp.status_code >= 400:
            detail = _extract_detail(resp)
            raise ApiError(detail, status_code=resp.status_code)
        return resp.json()


    def tryon_start(
        self,
        person_bytes: bytes,
        person_filename: str,
        garment_id: str,
        person_dimensions: dict,
        use_llm_prompt: bool = True,
    ) -> dict:
        """Enqueue a try-on job. Returns {'job_id': ..., 'status': 'queued'}."""
        files = [
            ("person_image", (person_filename, person_bytes, "application/octet-stream")),
        ]
        data = {
            "garment_id": garment_id,
            "person_dimensions": json.dumps(person_dimensions),
            "use_llm_prompt": "true" if use_llm_prompt else "false",
        }
        with httpx.Client(timeout=30.0) as client:
            try:
                resp = client.post(f"{self.base_url}/api/tryon", files=files, data=data)
            except httpx.HTTPError as e:
                raise ApiError(f"could not reach FastAPI ({self.base_url}): {e}") from e
        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status_code=resp.status_code)
        return resp.json()

    def tryon_status(self, job_id: str) -> dict:
        """Fetch current status of a try-on job."""
        with httpx.Client(timeout=10.0) as client:
            try:
                resp = client.get(f"{self.base_url}/api/tryon/status/{job_id}")
            except httpx.HTTPError as e:
                raise ApiError(f"could not reach FastAPI ({self.base_url}): {e}") from e
        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status_code=resp.status_code)
        return resp.json()


def _extract_detail(resp: httpx.Response) -> str:
    """FastAPI conventionally returns {"detail": "..."} on errors. Fall back
    to body text if the response isn't JSON.
    """
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"
