"""In-memory async job model for the try-on flow.

The Gemini call is the slow link (10–30 s). Rather than block the HTTP
response, `POST /api/tryon` enqueues a `JobRecord` here and returns a
`job_id` immediately. The actual work runs as an `asyncio` task that
calls into the existing sync code (`fit_calculator` → `prompt_builder`
→ `gemini_client`) via `asyncio.to_thread` so the event loop stays
unblocked.

Trade-offs:
- Job state is in-memory; an API restart loses in-flight jobs. Acceptable
  while a single `uvicorn` worker handles all traffic.
- No retention policy; old job records accumulate. Add a TTL sweep when
  this matters. (~1 KB per record, so practically a non-issue at v1 traffic.)
- Multi-worker deployments would need a shared store (Redis/DB) — at that
  point swap this module for a real queue.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    image_url: Optional[str] = None
    fit: Optional[dict] = None
    prompt_used: Optional[str] = None
    # Live token-by-token text from Groq while it's writing the prompt. Status
    # pollers read this every ~1 s and render it with a typing-cursor effect.
    # `prompt_streaming` flips False once Groq is finished — the image
    # generation phase is opaque (Gemini doesn't stream).
    prompt_text: str = ""
    prompt_streaming: bool = False
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOTH_DIR = PROJECT_ROOT / "cloth"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CATALOG_PATH = CLOTH_DIR / "garments.json"


def garment_exists(garment_id: str) -> bool:
    """Eager catalog check so we 404 at submit time, not at poll time."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return any(g["id"] == garment_id for g in catalog["garments"])


def _do_tryon_sync(
    job: JobRecord,
    person_bytes: bytes,
    garment_id: str,
    person_dimensions: dict,
    use_llm_prompt: bool,
    output_path: Path,
) -> tuple[dict, str]:
    """Synchronous worker. Imports inside the function so we don't pay the
    Groq/Gemini SDK import cost at API startup.

    The `job` is mutated in place so HTTP status pollers see streamed Groq
    tokens land in `job.prompt_text` as Groq emits them. CPython attribute
    assignment is GIL-atomic, so the reader either sees the previous string
    or the new one — never a half-written value.
    """
    from src.fit_calculator import calculate_fit
    from src.gemini_client import generate_tryon
    from src.groq_client import generate_text_stream
    from src.prompt_builder import build_fit_prompt, build_meta_prompt

    fit_result = calculate_fit(person_dimensions, garment_id)

    if use_llm_prompt:
        meta_prompt = build_meta_prompt(fit_result)
        job.prompt_streaming = True
        accumulated = ""
        try:
            for chunk in generate_text_stream(meta_prompt):
                accumulated += chunk
                job.prompt_text = accumulated
        finally:
            job.prompt_streaming = False
        prompt = accumulated
    else:
        prompt = build_fit_prompt(fit_result)
        job.prompt_text = prompt

    garment_image_path = str(CLOTH_DIR / fit_result["image_path"])
    generate_tryon(
        person_image=person_bytes,
        garment_image_path=garment_image_path,
        prompt=prompt,
        output_path=str(output_path),
    )
    return fit_result, prompt


async def run_tryon_job(
    jobs: dict[str, JobRecord],
    job_id: str,
    person_bytes: bytes,
    garment_id: str,
    person_dimensions: dict,
    use_llm_prompt: bool,
    output_filename: str,
) -> None:
    """Drive a single job through running → done|failed.

    Runs the sync work in a worker thread so the FastAPI event loop stays
    responsive for other requests (notably status polls from Django).
    """
    job = jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    output_path = OUTPUT_DIR / output_filename

    try:
        fit_result, prompt = await asyncio.to_thread(
            _do_tryon_sync,
            job,
            person_bytes,
            garment_id,
            person_dimensions,
            use_llm_prompt,
            output_path,
        )
        job.fit = fit_result
        job.prompt_used = prompt
        job.image_url = f"/static/output/{output_filename}"
        job.status = JobStatus.DONE
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = datetime.now(timezone.utc)
