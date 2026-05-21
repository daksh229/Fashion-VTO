"""Call FLUX.1 Kontext [dev] via Hugging Face Inference Providers.

FLUX.1 Kontext [dev] is the open-weights multi-image edit model from Black
Forest Labs — the open equivalent of Kontext Pro. HF Inference Providers
routes the call to a partner backend (fal-ai, Replicate, Together, Nebius)
and bills against your HF credit balance.

Multi-image strategy: the routed HF API standardises on a single input image.
We stitch the person photo on the left and the garment photo on the right
into one composite, then instruct the prompt to output only the left-half
person wearing the right-half garment. This is a well-known FLUX Kontext
trick and works on any single-image edit model.

Requires:
    pip install huggingface_hub
    .env must contain:  HF_TOKEN=...

Get a token at https://huggingface.co/settings/tokens (read-only is fine).
"""

import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


def _ensure_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set `HF_TOKEN` in your .env file. "
            "Generate one at https://huggingface.co/settings/tokens (read-only scope)."
        )
    return token


def _open_image(source) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (bytes, bytearray)):
        return Image.open(BytesIO(source))
    return Image.open(source)


def _to_rgb_max(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _stitch_side_by_side(
    person: Image.Image, garment: Image.Image, target_h: int = 1024
) -> Image.Image:
    def _resize_to_h(img: Image.Image, h: int) -> Image.Image:
        ratio = h / img.size[1]
        return img.resize((max(1, int(img.size[0] * ratio)), h), Image.LANCZOS)

    p = _resize_to_h(person, target_h)
    g = _resize_to_h(garment, target_h)
    combined = Image.new("RGB", (p.size[0] + g.size[0], target_h), "white")
    combined.paste(p, (0, 0))
    combined.paste(g, (p.size[0], 0))
    return combined


def generate_tryon(
    person_image,
    garment_image_path: str,
    prompt: str,
    output_path: str = "output.png",
) -> str:
    """Generate a try-on image via FLUX.1 Kontext [dev] on HF Inference Providers.

    Signature matches `src.gemini_client.generate_tryon`.
    """
    from huggingface_hub import InferenceClient

    token = _ensure_token()
    client = InferenceClient(provider="auto", api_key=token)

    person = _to_rgb_max(_open_image(person_image))
    garment = _to_rgb_max(_open_image(garment_image_path))
    composite = _stitch_side_by_side(person, garment)

    composite_prompt = (
        "The reference image is a side-by-side composite: the PERSON is on the "
        "LEFT half and a flat product photo of the GARMENT is on the RIGHT half. "
        "Apply the garment from the right half onto the person on the left half. "
        "Output ONLY the person wearing the garment — do NOT include the garment "
        "product photo in the final image, and do NOT keep the side-by-side layout. "
        "Preserve the person's face, hair, skin tone, pose, and background exactly.\n\n"
        f"{prompt}"
    )

    result = client.image_to_image(
        image=composite,
        prompt=composite_prompt,
        model=MODEL_ID,
    )

    if isinstance(result, Image.Image):
        result.save(output_path, format="PNG")
        return output_path
    if isinstance(result, (bytes, bytearray)):
        Path(output_path).write_bytes(result)
        return output_path
    if hasattr(result, "read"):
        Path(output_path).write_bytes(result.read())
        return output_path

    raise RuntimeError(
        f"Unexpected response type from HF Inference Providers: {type(result)!r}. "
        f"Check that the model `{MODEL_ID}` is available via Inference Providers "
        f"and that your HF credit balance is positive."
    )


if __name__ == "__main__":
    out = generate_tryon(
        person_image="person.jpg",
        garment_image_path="cloth/00041_00.jpg",
        prompt="Place the garment on the person, photorealistic, natural lighting.",
        output_path="output_flux.png",
    )
    print(f"Saved: {out}")
