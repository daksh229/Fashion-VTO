"""Call Gemini 2.5 Flash Image (a.k.a. 'Nano Banana') for virtual try-on generation.

Requires:
    pip install google-genai pillow
    export GEMINI_API_KEY=...
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = "gemini-3.1-flash-image-preview"


def _get_client() -> genai.Client:
    api_key = os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set `api_key` (or `GEMINI_API_KEY`) in your .env file."
        )
    return genai.Client(api_key=api_key)


def generate_tryon(
    person_image_path: str,
    garment_image_path: str,
    prompt: str,
    output_path: str = "output.png",
) -> str:
    client = _get_client()
    person_img = Image.open(person_image_path)
    garment_img = Image.open(garment_image_path)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, person_img, garment_img],
    )

    if not response.candidates:
        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise RuntimeError(
            f"Gemini returned no candidates — request was blocked "
            f"(block_reason={block!r}). Try a less extreme prompt or different inputs."
        )

    candidate = response.candidates[0]
    finish = getattr(candidate, "finish_reason", None)
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None

    if not parts:
        raise RuntimeError(
            f"Gemini returned a candidate with no content "
            f"(finish_reason={finish!r}). Likely a safety/recitation filter — "
            f"try softer prompt language or a different garment/person combination."
        )

    text_parts = []
    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            Path(output_path).write_bytes(part.inline_data.data)
            return output_path
        if getattr(part, "text", None):
            text_parts.append(part.text)

    extra = f" Model said: {' '.join(text_parts)[:300]!r}" if text_parts else ""
    raise RuntimeError(
        f"Gemini response contained no image data (finish_reason={finish!r}).{extra}"
    )


if __name__ == "__main__":
    out = generate_tryon(
        person_image_path="person.jpg",
        garment_image_path="cloth/00041_00.jpg",
        prompt="Place the garment on the person, photorealistic, natural lighting.",
        output_path="output.png",
    )
    print(f"Saved: {out}")
