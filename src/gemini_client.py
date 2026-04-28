"""Call Gemini 2.5 Flash Image (a.k.a. 'Nano Banana') for virtual try-on generation.

Requires:
    pip install google-genai pillow
    export GEMINI_API_KEY=...

Two generation functions:
    generate_tryon(person, garment, prompt, ...)
        -> the original try-on call. Used for the FRONT view.
           Takes the user photo + the garment reference photo.

    rotate_view(front_tryon, person, prompt, view, ...)
        -> rotates an already-generated front try-on to a different camera
           angle. Takes the front-view render as the primary canvas and
           (optionally) the original person photo as a facial-feature
           reference. No garment reference needed — the garment is already
           correctly rendered in the front view.

    generate_with_retry(fn, *args, max_attempts=3, backoff=1.5, **kwargs)
        -> wraps either call with exponential backoff retries. Retries on
           every RuntimeError from the two functions above (safety blocks,
           empty responses, rate limits).
"""

import os
import time
import random
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = "gemini-2.5-flash-image"


def _get_client() -> genai.Client:
    api_key = os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set `api_key` (or `GEMINI_API_KEY`) in your .env file."
        )
    return genai.Client(api_key=api_key)


def _extract_image_or_raise(response, context: str) -> bytes:
    """Pull the first inline_data part out of a Gemini response, or raise a
    RuntimeError whose message identifies the failure mode.
    """
    if not response.candidates:
        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise RuntimeError(
            f"Gemini returned no candidates for {context} — request was blocked "
            f"(block_reason={block!r})."
        )

    candidate = response.candidates[0]
    finish = getattr(candidate, "finish_reason", None)
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None

    if not parts:
        raise RuntimeError(
            f"Gemini returned a candidate with no content for {context} "
            f"(finish_reason={finish!r}). Likely a safety/recitation filter."
        )

    text_parts = []
    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data
        if getattr(part, "text", None):
            text_parts.append(part.text)

    extra = f" Model said: {' '.join(text_parts)[:300]!r}" if text_parts else ""
    raise RuntimeError(
        f"Gemini response contained no image data for {context} "
        f"(finish_reason={finish!r}).{extra}"
    )


def generate_tryon(
    person_image_path: str,
    garment_image_path: str,
    prompt: str,
    output_path: str = "output.png",
    view: str = "front",
) -> str:
    """Front-view try-on: user photo + garment reference → rendered front view.

    This is the original try-on call. The `view` argument is carried for
    uniform signatures but this function should only be called for the
    front view now — side/back/right views use rotate_view().
    """
    client = _get_client()
    person_img = Image.open(person_image_path)
    garment_img = Image.open(garment_image_path)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            "REFERENCE B — GARMENT DESIGN SOURCE. Use this image ONLY to extract the garment's design: sleeve length and style (preserve exactly), hemline position, neckline, collar, silhouette, print/logo, color, fabric texture, cuffs, buttons, pockets, closures. IGNORE everything else in this image — the reference model's face, hair, body, pose, background, lighting, and any pants/skirt/shorts/accessories she is paired with. She is NOT in the output.",
            garment_img,
            "REFERENCE A — THE CANVAS. This is the scene to edit. Start from this image. The output keeps this person exactly as shown: same face, same hair, same earrings/accessories, same skin tone, same body, same pose and arm position, same camera angle, same lighting, same background, and same non-target clothing (e.g., the pants/skirt they are already wearing). Replace ONLY the top/bottom being tried on.",
            person_img,
            "OUTPUT RULES: The person in the output MUST be the person from REFERENCE A — their face, hair, skin, earrings, and body must match REFERENCE A exactly. The face, hair, and body of REFERENCE B's model must NOT appear in the output. The garment's design comes from REFERENCE B. If the output face does not match REFERENCE A's face, the result is incorrect. Produce one photorealistic try-on image.",
        ],
    )

    data = _extract_image_or_raise(response, f"view={view!r}")
    Path(output_path).write_bytes(data)
    return output_path


def rotate_view(
    front_tryon_path: str,
    person_image_path: str | None,
    prompt: str,
    view: str,
    output_path: str = "output.png",
) -> str:
    """Rotate an already-generated front try-on to a different camera angle.

    Inputs:
        front_tryon_path: path to the FRONT try-on render. This is the
            authoritative source for identity, garment, pose-base, lighting,
            and background.
        person_image_path: optional path to the original user photo. For
            left/right profile views, passing this gives Gemini a second
            look at facial features from the only real reference we have.
            For back view, pass None — the face won't be visible.
        prompt: fit-aware prompt already customized for this view.
        view: one of "left", "back", "right". Do not call this for "front".
    """
    if view not in ("left", "back", "right"):
        raise ValueError(
            f"rotate_view() is only for non-front views; got {view!r}"
        )

    client = _get_client()
    front_img = Image.open(front_tryon_path)

    contents: list = [
        prompt,
        (
            "REFERENCE F — THE FRONT TRY-ON (PRIMARY CANVAS). This is an already-"
            "rendered front view of the target person wearing the target garment. "
            "It is the authoritative source for: the person's identity (face, hair, "
            "skin tone, earrings, body), the garment (color, design, print/logo, "
            "silhouette, fit — tight/loose/cropped/draped — and all visible "
            "details), the lower-body clothing, the lighting, and the background. "
            "Your job is to produce the SAME person wearing the SAME garment in "
            "the SAME scene, with ONLY the camera angle changed. Do not alter "
            "identity. Do not alter the garment. Do not alter the background or "
            "lighting. Do not add or remove accessories."
        ),
        front_img,
    ]

    if person_image_path is not None:
        person_img = Image.open(person_image_path)
        contents.extend([
            (
                "REFERENCE A — ORIGINAL FACE / HAIR REFERENCE (supplementary). "
                "This is an additional reference for the person's face, hair color, "
                "hair length, skin tone, and facial features. Use this to keep the "
                "person's identity locked. The garment shown in this image is NOT "
                "the target garment — ignore the clothing here; the garment comes "
                "entirely from REFERENCE F."
            ),
            person_img,
        ])

    contents.append(
        "OUTPUT RULES: Produce ONE photorealistic image of the same person from "
        "REFERENCE F, wearing the same garment from REFERENCE F, in the same "
        "scene from REFERENCE F — but viewed from the camera angle specified in "
        "the prompt. The face must match REFERENCE F (and REFERENCE A if "
        "provided). The garment color, print, and fit must match REFERENCE F "
        "exactly. No other changes."
    )

    response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    data = _extract_image_or_raise(response, f"view={view!r} (rotate)")
    Path(output_path).write_bytes(data)
    return output_path


def generate_with_retry(
    fn: Callable,
    *args,
    max_attempts: int = 3,
    backoff: float = 1.5,
    **kwargs,
) -> str:
    """Call `fn(*args, **kwargs)` with up to `max_attempts` attempts and
    exponential backoff with jitter between attempts.

    Retries on any RuntimeError raised by generate_tryon or rotate_view —
    those cover safety blocks, empty responses, and most transient API
    errors.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            last_err = e
            if attempt < max_attempts:
                wait = (backoff ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
            else:
                break
    # Re-raise the last error with attempt context.
    raise RuntimeError(
        f"All {max_attempts} attempts failed. Last error: {last_err}"
    ) from last_err


if __name__ == "__main__":
    out = generate_tryon(
        person_image_path="person.jpg",
        garment_image_path="cloth/00041_00.jpg",
        prompt="Place the garment on the person, photorealistic, natural lighting.",
        output_path="output.png",
    )
    print(f"Saved: {out}")
