"""Async try-on endpoints.

Flow:
    POST /api/tryon                  → 202 with {job_id, status:"queued"}
                                       launches an asyncio task in the
                                       background; returns immediately.
    GET  /api/tryon/status/{job_id}  → current state (queued/running/done/failed)
                                       and, when done, image_url + fit + prompt_used.

The image itself is served by the StaticFiles mount in main.py; status
returns the relative URL (`/static/output/...`) the caller hot-links.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..jobs import JobRecord, JobStatus, garment_exists, run_tryon_job
from ..schemas import TryonStartResponse, TryonStatusResponse

router = APIRouter()


@router.post("/tryon", response_model=TryonStartResponse, status_code=202)
async def tryon_start(
    request: Request,
    person_image: UploadFile = File(...),
    garment_id: str = Form(...),
    person_dimensions: str = Form(...),
    use_llm_prompt: bool = Form(True),
) -> TryonStartResponse:
    if not person_image.filename:
        raise HTTPException(status_code=400, detail="person_image is required.")
    person_bytes = await person_image.read()
    if not person_bytes:
        raise HTTPException(status_code=400, detail="person_image is empty.")

    try:
        dims = json.loads(person_dimensions)
        if not isinstance(dims, dict):
            raise ValueError("must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"person_dimensions invalid: {e}")

    if not garment_exists(garment_id):
        raise HTTPException(status_code=404, detail=f"garment '{garment_id}' not found")

    job_id = str(uuid.uuid4())
    output_filename = f"tryon_{garment_id}_{job_id[:8]}.png"
    job = JobRecord(job_id=job_id)
    request.app.state.jobs[job_id] = job

    asyncio.create_task(
        run_tryon_job(
            jobs=request.app.state.jobs,
            job_id=job_id,
            person_bytes=person_bytes,
            garment_id=garment_id,
            person_dimensions=dims,
            use_llm_prompt=use_llm_prompt,
            output_filename=output_filename,
        )
    )

    return TryonStartResponse(job_id=job_id, status=job.status.value)


@router.get("/tryon/status/{job_id}", response_model=TryonStatusResponse)
def tryon_status(job_id: str, request: Request) -> TryonStatusResponse:
    job: JobRecord | None = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return TryonStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        image_url=job.image_url,
        fit=job.fit,
        prompt_used=job.prompt_used,
        prompt_text=job.prompt_text,
        prompt_streaming=job.prompt_streaming,
        error=job.error,
    )
