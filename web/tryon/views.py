"""User-facing views for the try-on flow.

Pages:
    GET/POST /                                upload form → calls /api/measure
    GET/POST /verify/                         editable detected values
    GET      /wardrobe/                       garment grid (top-only for now)
    POST     /tryon/start/<garment_id>/       kick off /api/tryon
    GET      /tryon/result/<job_id>/          "generating…" page with JS polling
    GET      /tryon/status/<job_id>.json      polling proxy to /api/tryon/status

Session keys used:
    measure_result          — last /api/measure response (incl. annotated PNG b64)
    verified_dimensions     — user-edited dimensions, keyed by catalog keys
                              ("length", "chest", "shoulder", "arm_length")
    person_image_b64        — person photo bytes, base64-encoded (set on
                              successful /api/measure submission so the wardrobe
                              view doesn't re-prompt for the photo)
"""

from __future__ import annotations

import base64

from django.conf import settings
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .catalog import find_garment, list_garments

# Phase-3 mapping: which catalog key maps to which raw measurement field
# coming back from /api/measure. Mirrors src/measurement/catalog_mapping.py
# but kept locally here so the web layer doesn't reach into FastAPI's src.
_TOP_KEY_TO_RAW = {
    "length": "torso_length_cm",
    "chest": "chest_circumference_cm",
    "shoulder": "shoulder_width_cm",
    "arm_length": "arm_length_cm",
}

_CONF_KEY_FOR_CATALOG_KEY = {
    "length": "length",
    "chest": "chest",
    "shoulder": "shoulder",
    "arm_length": "arm_length",
}

from .forms import TOP_DIMENSION_KEYS, MeasureForm, VerifyForm
from .services import ApiError, FastAPIClient


def measure_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = MeasureForm(request.POST, request.FILES)
        if form.is_valid():
            frontal = form.cleaned_data["frontal_image"]
            side = form.cleaned_data.get("side_image")
            frontal_bytes = frontal.read()
            client = FastAPIClient()
            try:
                result = client.measure(
                    frontal_bytes=frontal_bytes,
                    frontal_filename=frontal.name,
                    height_cm=form.cleaned_data["height_cm"],
                    side_bytes=side.read() if side else None,
                    side_filename=side.name if side else None,
                    shoulder_cm=form.cleaned_data.get("shoulder_cm") or None,
                    waist_cm=form.cleaned_data.get("waist_cm") or None,
                )
            except ApiError as e:
                messages.error(request, f"Measurement failed: {e}")
                return render(request, "tryon/measure.html", {"form": form})

            request.session["measure_result"] = result
            # Stash the photo so the try-on step (next phase) doesn't make the
            # user re-upload. Base64 to keep the session JSON-serializable.
            request.session["person_image_b64"] = base64.b64encode(frontal_bytes).decode("ascii")
            request.session["person_image_filename"] = frontal.name
            return redirect(reverse("tryon:verify"))
    else:
        form = MeasureForm()

    return render(request, "tryon/measure.html", {"form": form})


def verify_view(request: HttpRequest) -> HttpResponse:
    result = request.session.get("measure_result")
    if not result:
        messages.warning(request, "Upload a photo first to detect measurements.")
        return redirect(reverse("tryon:measure"))

    raw = result.get("measurements", {})
    confidences = result.get("confidence", {})

    detected = {
        cat_key: raw.get(_TOP_KEY_TO_RAW[cat_key])
        for cat_key, _ in TOP_DIMENSION_KEYS
    }

    if request.method == "POST":
        form = VerifyForm(request.POST, initial_dims=detected)
        if form.is_valid():
            request.session["verified_dimensions"] = {
                key: form.cleaned_data[key] for key, _ in TOP_DIMENSION_KEYS
            }
            return redirect(reverse("tryon:wardrobe"))
    else:
        form = VerifyForm(initial_dims=detected)

    rows = []
    for key, label in TOP_DIMENSION_KEYS:
        conf = float(confidences.get(_CONF_KEY_FOR_CATALOG_KEY[key], 0.0))
        rows.append(
            {
                "key": key,
                "label": label,
                "field": form[key],
                "detected_value": detected.get(key),
                "confidence": conf,
                "badge": _confidence_badge(conf),
            }
        )

    context = {
        "form": form,
        "rows": rows,
        "warnings": result.get("warnings", []),
        "annotated_b64": result.get("annotated_image_b64"),
        "used_side_photo": result.get("used_side_photo", False),
        "verified_dimensions": request.session.get("verified_dimensions"),
    }
    return render(request, "tryon/verify.html", context)


def _confidence_badge(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    if conf >= 0.6:
        return "medium"
    return "low"


def wardrobe_view(request: HttpRequest) -> HttpResponse:
    dims = request.session.get("verified_dimensions")
    if not dims:
        messages.warning(request, "Verify your measurements before picking a garment.")
        return redirect(reverse("tryon:measure"))

    garments = list_garments(category="top")
    return render(
        request,
        "tryon/wardrobe.html",
        {
            "garments": garments,
            "dims": dims,
            "cloth_url": settings.CLOTH_URL,
        },
    )


@require_POST
def tryon_start_view(request: HttpRequest, garment_id: str) -> HttpResponse:
    garment = find_garment(garment_id)
    if not garment:
        messages.error(request, f"Garment {garment_id!r} not found.")
        return redirect(reverse("tryon:wardrobe"))

    dims = request.session.get("verified_dimensions")
    person_b64 = request.session.get("person_image_b64")
    if not dims or not person_b64:
        messages.warning(request, "Session expired — please start from the beginning.")
        return redirect(reverse("tryon:measure"))

    use_llm_prompt = request.POST.get("use_llm_prompt") == "on"
    person_filename = request.session.get("person_image_filename") or "person.jpg"
    person_bytes = base64.b64decode(person_b64)

    client = FastAPIClient()
    try:
        start_resp = client.tryon_start(
            person_bytes=person_bytes,
            person_filename=person_filename,
            garment_id=garment_id,
            person_dimensions=dims,
            use_llm_prompt=use_llm_prompt,
        )
    except ApiError as e:
        messages.error(request, f"Try-on failed to start: {e}")
        return redirect(reverse("tryon:wardrobe"))

    return redirect(reverse("tryon:result", args=[start_resp["job_id"]]))


def tryon_result_view(request: HttpRequest, job_id: str) -> HttpResponse:
    """Render the polling page. JS in the template hits the status JSON
    endpoint every ~2 s and swaps the image in once the job finishes.
    """
    return render(
        request,
        "tryon/result.html",
        {
            "job_id": job_id,
            "tryon_output_url": settings.TRYON_OUTPUT_URL,
            "status_url": reverse("tryon:status_json", args=[job_id]),
            "wardrobe_url": reverse("tryon:wardrobe"),
        },
    )


def tryon_status_json_view(request: HttpRequest, job_id: str) -> JsonResponse:
    """JSON proxy in front of FastAPI's /api/tryon/status, with the
    image_url path rewritten so the browser fetches it from Django's
    /tryon-output/ mount instead of FastAPI's /static/output/ mount.
    """
    client = FastAPIClient()
    try:
        data = client.tryon_status(job_id)
    except ApiError as e:
        return JsonResponse(
            {"status": "failed", "error": str(e), "job_id": job_id},
            status=e.status_code or 502,
        )

    image_url = data.get("image_url")
    if image_url and image_url.startswith("/static/output/"):
        filename = image_url.rsplit("/", 1)[-1]
        data["image_url"] = settings.TRYON_OUTPUT_URL + filename

    return JsonResponse(data)
