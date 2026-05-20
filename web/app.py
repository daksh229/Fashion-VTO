"""Fit-Aware Virtual Try-On — Streamlit frontend.

Flow:
    1. User picks photo source (upload or camera) and provides a frontal photo
       (optional side photo for circumference accuracy).
    2. User enters height (mandatory pixel→cm scale anchor) plus optional
       shoulder/waist correction anchors.
    3. "Detect Measurements" runs the MediaPipe pipeline; results are shown
       as editable inputs in the sidebar for last-moment user verification.
    4. User selects the image generation model.
    5. "Load Wardrobe" → grid of all garments from cloth/garments.json.
    6. "Try Now" on a garment:
        - fit_calculator.calculate_fit()  -> deltas
        - Groq writes the detailed prompt (or rule-based fallback is used)
        - Selected model generates the final try-on image
    7. Final try-on image is displayed alongside person + garment.

Supported generation models:
    - Gemini 2.5 Flash Image  ("Nano Banana")        — via google-genai SDK
    - Gemini 3 Pro Image      ("Nano Banana Pro")     — via google-genai SDK
    - FLUX.1 Kontext Pro                              — via fal-client SDK

Manual fallback: a sidebar checkbox bypasses detection entirely and shows the
classic 4-input form.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit_calculator import calculate_fit
from src.prompt_builder import build_fit_prompt, build_meta_prompt
from src.groq_client import generate_text_stream
from src.measurement import (
    MeasurementError,
    _create_pose_solution,
    confidence_for_catalog_keys,
    measurements_to_person_dimensions,
    run_measurement_pipeline,
)

CLOTH_DIR = PROJECT_ROOT / "cloth"
CATALOG_PATH = CLOTH_DIR / "garments.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

MODEL_OPTIONS = {
    "Gemini 2.5 Flash Image (Nano Banana) — $0.039/img": "gemini-2.5-flash",
    "Gemini 3 Pro Image (Nano Banana Pro) — $0.134/img": "gemini-3-pro",
    "FLUX.1 Kontext Pro (fal.ai) — $0.04/img":           "flux-kontext-pro",
}

MODEL_DESCRIPTIONS = {
    "gemini-2.5-flash":  "Gemini 2.5 Flash Image · Multimodal edit · ~10–20 s · $0.039/img",
    "gemini-3-pro":      "Gemini 3 Pro Image · Best quality, 4K, 14 ref imgs · ~15–30 s · $0.134/img",
    "flux-kontext-pro":  "FLUX.1 Kontext Pro · 12B params, precise local edits · ~8–15 s · $0.04/img",
}


# ─────────────────────────────────────────────
# Generation backends
# ─────────────────────────────────────────────

def _generate_gemini(
    person_image_bytes: bytes,
    garment_image_path: str,
    prompt: str,
    output_path: str,
    model_id: str,
) -> str:
    """Generate via Gemini (both 2.5 Flash and 3 Pro share the same SDK call)."""
    from src.gemini_client import generate_tryon as _gemini_tryon

    # gemini_client.py uses MODEL_NAME = "gemini-2.5-flash-image" by default.
    # We monkey-patch to support Gemini 3 Pro without touching the original file.
    import src.gemini_client as _gc

    model_name_map = {
        "gemini-2.5-flash": "gemini-2.5-flash-image",
        "gemini-3-pro":     "gemini-3.0-pro-image",   # official model string
    }
    original_model = _gc.MODEL_NAME
    _gc.MODEL_NAME = model_name_map[model_id]
    try:
        result = _gemini_tryon(
            person_image=person_image_bytes,
            garment_image_path=garment_image_path,
            prompt=prompt,
            output_path=output_path,
        )
    finally:
        _gc.MODEL_NAME = original_model
    return result


def _generate_flux_kontext(
    person_image_bytes: bytes,
    garment_image_path: str,
    prompt: str,
    output_path: str,
) -> str:
    """Generate via FLUX.1 Kontext Pro on fal.ai.

    Input images are uploaded as base64 data-URIs. fal accepts these
    directly in the `image_url` field so no separate upload step is needed.

    Requires:
        pip install fal-client
        FAL_KEY in .env
    """
    import base64
    import fal_client  # type: ignore

    fal_api_key = os.environ.get("FAL_KEY")
    if not fal_api_key:
        raise RuntimeError(
            "FAL_KEY not found in .env. Add it to use FLUX.1 Kontext Pro."
        )
    os.environ["FAL_KEY"] = fal_api_key  # fal_client reads from env

    # Encode person image
    person_b64 = base64.b64encode(person_image_bytes).decode("ascii")
    person_data_uri = f"data:image/jpeg;base64,{person_b64}"

    # Encode garment image
    garment_bytes = Path(garment_image_path).read_bytes()
    garment_b64 = base64.b64encode(garment_bytes).decode("ascii")
    garment_data_uri = f"data:image/jpeg;base64,{garment_b64}"

    # Build a combined prompt that includes garment reference instruction
    combined_prompt = (
        f"{prompt}\n\n"
        "The second image is the garment reference. "
        "Dress the person in the first image with that exact garment, "
        "preserving garment texture, color, and print faithfully."
    )

    result = fal_client.subscribe(
        "fal-ai/flux-pro/kontext",
        arguments={
            "prompt":      combined_prompt,
            "image_url":   person_data_uri,         # person photo
            "image_url_2": garment_data_uri,         # garment reference
            "num_inference_steps": 28,
            "guidance_scale":      3.5,
            "num_images":          1,
            "output_format":       "jpeg",
        },
    )

    # fal returns a list of image dicts with "url" keys
    images = result.get("images") or []
    if not images:
        raise RuntimeError("FLUX.1 Kontext Pro returned no images. Check your FAL_KEY and account balance.")

    import urllib.request
    image_url = images[0]["url"]
    urllib.request.urlretrieve(image_url, output_path)
    return output_path


def generate_tryon_with_model(
    model_id: str,
    person_image_bytes: bytes,
    garment_image_path: str,
    prompt: str,
    output_path: str,
) -> str:
    """Dispatch to the correct backend based on model_id."""
    if model_id in ("gemini-2.5-flash", "gemini-3-pro"):
        return _generate_gemini(
            person_image_bytes, garment_image_path, prompt, output_path, model_id
        )
    elif model_id == "flux-kontext-pro":
        return _generate_flux_kontext(
            person_image_bytes, garment_image_path, prompt, output_path
        )
    else:
        raise ValueError(f"Unknown model_id: {model_id!r}")


# ─────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────

@st.cache_data
def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_pose_solution():
    return _create_pose_solution(model_complexity=1)


def stream_words(text: str, delay: float = 0.02):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def confidence_badge(conf: float) -> str:
    if conf >= 0.8:
        return "🟢"
    if conf >= 0.6:
        return "🟡"
    return "🔴"


# ─────────────────────────────────────────────
# Loading animation HTML
# ─────────────────────────────────────────────

def _tailoring_html(model_label: str) -> str:
    return f"""
<style>
  .tailoring-wrap {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 22px;
    background: linear-gradient(90deg, #f8f9fb, #eef2f8, #f8f9fb);
    background-size: 200% 100%;
    border: 1px solid #e1e4ea;
    border-radius: 10px;
    animation: tailor-shimmer 3.2s ease-in-out infinite;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .tailor-icon {{
    font-size: 22px;
    animation: tailor-pulse 1.6s ease-in-out infinite;
  }}
  .tailor-model-tag {{
    font-size: 11px;
    color: #6b7280;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 2px 7px;
    white-space: nowrap;
  }}
  .tailor-msg-stack {{
    position: relative;
    height: 22px;
    flex: 1;
    overflow: hidden;
  }}
  .tailor-msg {{
    position: absolute;
    inset: 0;
    font-size: 15px;
    color: #2a2f3a;
    font-weight: 500;
    opacity: 0;
    animation: tailor-rotate 16s ease-in-out infinite;
    white-space: nowrap;
  }}
  .tailor-msg.m1 {{ animation-delay: 0s; }}
  .tailor-msg.m2 {{ animation-delay: 4s; }}
  .tailor-msg.m3 {{ animation-delay: 8s; }}
  .tailor-msg.m4 {{ animation-delay: 12s; }}
  @keyframes tailor-rotate {{
    0%, 25%, 100% {{ opacity: 0; transform: translateY(8px); }}
    3%, 22% {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes tailor-pulse {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.15); }}
  }}
  @keyframes tailor-shimmer {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
  }}
</style>
<div class="tailoring-wrap">
  <span class="tailor-icon">&#9986;&#65039;</span>
  <span class="tailor-model-tag">{model_label}</span>
  <div class="tailor-msg-stack">
    <span class="tailor-msg m1">AI is tailoring the garment to your measurements&hellip;</span>
    <span class="tailor-msg m2">Matching fabric drape to your silhouette&hellip;</span>
    <span class="tailor-msg m3">Preserving every print, stitch, and colour&hellip;</span>
    <span class="tailor-msg m4">Rendering your final try-on&hellip;</span>
  </div>
</div>
"""


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fit-Aware Virtual Try-On",
    page_icon="👗",
    layout="wide",
)

st.title("Fit-Aware Virtual Try-On")
st.caption(
    "Take or upload your photo, enter your height, and we'll auto-detect your "
    "measurements. Verify them, pick a generation model, then try on any garment."
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

for key, default in [
    ("wardrobe_loaded",      False),
    ("selected_garment_id",  None),
    ("last_result",          None),
    ("measurement_result",   None),
    ("person_image_bytes",   None),
    ("person_dimensions",    None),
    ("selected_model_id",    "gemini-2.5-flash"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Sidebar — user inputs
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Your Details")

    manual_mode = st.checkbox(
        "Skip detection — enter manually",
        value=False,
        help="Bypass the photo-based pipeline and enter measurements directly.",
    )

    person_dimensions: dict[str, float] | None = None
    person_image_bytes: bytes | None = None

    if manual_mode:
        # ── Manual entry path ──
        uploaded = st.file_uploader(
            "Upload your photo",
            type=["jpg", "jpeg", "png"],
            help="Used as the person reference for try-on rendering only.",
        )
        st.subheader("Your measurements (cm)")
        length_cm    = st.number_input("Length / torso",       min_value=20.0,  max_value=120.0, value=66.0,  step=0.5)
        chest_cm     = st.number_input("Chest circumference",  min_value=50.0,  max_value=180.0, value=91.0,  step=0.5)
        shoulder_cm_in = st.number_input("Shoulder width",     min_value=20.0,  max_value=80.0,  value=38.0,  step=0.5)
        arm_length_cm = st.number_input("Arm length",          min_value=10.0,  max_value=90.0,  value=23.0,  step=0.5)

        if uploaded is not None:
            person_image_bytes = uploaded.getvalue()
        person_dimensions = {
            "length":     length_cm,
            "chest":      chest_cm,
            "shoulder":   shoulder_cm_in,
            "arm_length": arm_length_cm,
        }

    else:
        # ── Auto-detect path ──
        st.subheader("1. Frontal photo")
        frontal_source = st.radio("Source", ["Upload", "Camera"], horizontal=True, key="frontal_src")
        if frontal_source == "Upload":
            frontal_file = st.file_uploader("Upload frontal photo", type=["jpg", "jpeg", "png"], key="frontal_upload")
        else:
            frontal_file = st.camera_input("Take a frontal photo", key="frontal_cam")

        st.subheader("2. Side photo (optional)")
        st.caption("Improves circumference accuracy for chest, waist, hip, thigh.")
        side_source = st.radio("Source", ["None", "Upload", "Camera"], horizontal=True, key="side_src")
        side_file = None
        if side_source == "Upload":
            side_file = st.file_uploader("Upload side photo", type=["jpg", "jpeg", "png"], key="side_upload")
        elif side_source == "Camera":
            side_file = st.camera_input("Take a side photo", key="side_cam")

        st.subheader("3. Reference height")
        height_cm = st.number_input("Your height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=0.5)

        st.subheader("4. Optional anchors")
        anchor_shoulder = st.number_input(
            "Shoulder width (cm, 0 = skip)", min_value=0.0, max_value=80.0, value=0.0, step=0.5,
            help="Measured seam-to-seam across the back. Improves shoulder + arm accuracy."
        )
        anchor_waist = st.number_input(
            "Waist circumference (cm, 0 = skip)", min_value=0.0, max_value=200.0, value=0.0, step=0.5,
            help="Measured at the natural waist. Improves waist and hip circumference accuracy."
        )

        if frontal_file is not None:
            if st.button("Detect Measurements", type="primary", use_container_width=True):
                with st.spinner("Running MediaPipe pose detection…"):
                    try:
                        result = run_measurement_pipeline(
                            frontal_bytes=frontal_file.getvalue(),
                            height_cm=height_cm,
                            side_bytes=side_file.getvalue() if side_file else None,
                            shoulder_cm=anchor_shoulder if anchor_shoulder > 0 else None,
                            waist_cm=anchor_waist if anchor_waist > 0 else None,
                            pose_solution=get_pose_solution(),
                        )
                        st.session_state.measurement_result = result
                        st.session_state.person_image_bytes = frontal_file.getvalue()
                        for stale_key in [k for k in st.session_state if k.startswith("verify_")]:
                            del st.session_state[stale_key]
                        st.success("Detection complete — verify the values below.")
                    except MeasurementError as e:
                        st.error(f"Detection failed: {e}")
                        st.session_state.measurement_result = None

        if frontal_file is not None and st.session_state.person_image_bytes is None:
            st.session_state.person_image_bytes = frontal_file.getvalue()
        person_image_bytes = st.session_state.person_image_bytes

        # ── Verification panel ──
        mr = st.session_state.measurement_result
        if mr is not None:
            st.divider()
            st.subheader("Verify measurements")
            st.caption("Edit any value before loading the wardrobe. Confidence: 🟢 high · 🟡 medium · 🔴 low.")
            for w in mr.warnings:
                st.warning(w, icon="⚠️")

            verification_keys = [
                ("length",     "Length / torso (cm)"),
                ("chest",      "Chest circumference (cm)"),
                ("shoulder",   "Shoulder width (cm)"),
                ("arm_length", "Arm length (cm)"),
            ]
            detected = measurements_to_person_dimensions(
                mr.measurements,
                fit_relevant_keys=[k for k, _ in verification_keys],
                category="top",
            )
            confs = confidence_for_catalog_keys(
                mr.confidence,
                fit_relevant_keys=[k for k, _ in verification_keys],
            )

            edited: dict[str, float] = {}
            for key, label in verification_keys:
                detected_val = detected.get(key)
                conf = confs.get(key, 0.0)
                badge = confidence_badge(conf)
                fallback_value = float(detected_val) if detected_val else 0.0
                edited[key] = st.number_input(
                    f"{badge} {label} · conf {conf:.2f}",
                    min_value=0.0,
                    max_value=300.0,
                    value=round(fallback_value, 1),
                    step=0.5,
                    key=f"verify_{key}",
                )
            person_dimensions = edited

            if mr.annotated_image_bytes:
                with st.expander("Detected pose"):
                    st.image(mr.annotated_image_bytes, caption="Pose landmarks", use_container_width=True)

    # ── Model selector ──
    st.divider()
    st.subheader("🤖 Image generation model")
    selected_model_label = st.selectbox(
        "Choose model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.values()).index(st.session_state.selected_model_id),
        help="Each model has different quality, speed, and cost characteristics.",
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]
    st.session_state.selected_model_id = selected_model_id
    st.caption(MODEL_DESCRIPTIONS[selected_model_id])

    # ── API key warnings per model ──
    if selected_model_id in ("gemini-2.5-flash", "gemini-3-pro"):
        if not (os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")):
            st.warning("Gemini API key not found in `.env` (`GEMINI_API_KEY`).")
    elif selected_model_id == "flux-kontext-pro":
        if not os.environ.get("FAL_KEY"):
            st.warning("fal.ai API key not found in `.env` (`FAL_KEY`).")

    # ── Prompt method ──
    st.divider()
    use_llm_prompt = st.toggle(
        "Use Groq to write the prompt",
        value=True,
        help=(
            "ON: Groq (Llama 3.3 70B) writes the fit-aware prompt which is then "
            "sent to the selected image model.\n"
            "OFF: use the deterministic rule-based prompt builder."
        ),
    )
    if use_llm_prompt and not os.environ.get("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` not found in `.env`. Groq prompt generation will fail.")

    # ── Load Wardrobe ──
    st.divider()
    load_disabled = (
        person_image_bytes is None
        or person_dimensions is None
        or not all(v and v > 0 for v in person_dimensions.values())
    )
    if st.button("Load Wardrobe →", type="primary", use_container_width=True, disabled=load_disabled):
        st.session_state.person_image_bytes  = person_image_bytes
        st.session_state.person_dimensions   = person_dimensions
        st.session_state.wardrobe_loaded     = True
        st.session_state.last_result         = None
        st.session_state.selected_garment_id = None


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────

if not st.session_state.wardrobe_loaded:
    st.info("← Enter your details in the sidebar and click **Load Wardrobe** to begin.")
    st.stop()

# Person preview + dimensions
col_a, col_b = st.columns([1, 3])
with col_a:
    st.image(st.session_state.person_image_bytes, caption="You", width=220)
with col_b:
    st.subheader("Your dimensions")
    pd_cols = st.columns(4)
    pd = st.session_state.person_dimensions
    for c, (label, val) in zip(pd_cols, pd.items()):
        c.metric(label.replace("_", " ").title(), f"{val} cm")

    # Show active model as a small info box
    active_model_label = next(
        (k for k, v in MODEL_OPTIONS.items() if v == st.session_state.selected_model_id),
        ""
    )
    st.info(f"🤖 **Active model:** {active_model_label}", icon=None)

st.divider()

# ─────────────────────────────────────────────
# Wardrobe grid
# ─────────────────────────────────────────────

st.header("Available Wardrobe")
catalog  = load_catalog()
garments = catalog["garments"]
units    = catalog.get("units", "cm")

cols_per_row = 3
for i in range(0, len(garments), cols_per_row):
    row = st.columns(cols_per_row)
    for col, garment in zip(row, garments[i:i + cols_per_row]):
        with col:
            with st.container(border=True):
                img_path = CLOTH_DIR / garment["path"]
                st.image(str(img_path), use_container_width=True)
                st.markdown(f"**{garment['name']}**")
                st.caption(
                    f"{garment['id']} · {garment['type']} · "
                    f"Size {garment['size']} · {garment['fit_style']}"
                )
                d       = garment["dimensions"]
                relevant = garment.get("fit_relevant_keys", list(d.keys()))
                spec_pairs = " · ".join(
                    f"{k.replace('_', ' ').title()}: <b>{d[k]} {units}</b>"
                    for k in relevant if k in d
                )
                st.markdown(f"<small>{spec_pairs}</small>", unsafe_allow_html=True)
                if st.button("Try Now", key=f"try_{garment['id']}", use_container_width=True):
                    st.session_state.selected_garment_id = garment["id"]
                    st.session_state.last_result = None


# ─────────────────────────────────────────────
# Generation flow
# ─────────────────────────────────────────────

if st.session_state.selected_garment_id:
    gid        = st.session_state.selected_garment_id
    model_id   = st.session_state.selected_model_id
    model_label = next((k for k, v in MODEL_OPTIONS.items() if v == model_id), model_id)

    st.divider()
    status_box = st.empty()
    loader_box = st.empty()
    error_box  = st.empty()

    status_box.markdown(
        f"### ✨ Generating try-on for **{gid}** using _{model_label}_\n"
        f"<span style='color:#888'>This usually takes 15–60 seconds depending on the model.</span>",
        unsafe_allow_html=True,
    )
    loader_box.markdown(_tailoring_html(model_label), unsafe_allow_html=True)

    fit              = calculate_fit(st.session_state.person_dimensions, gid)
    garment_img_path = str(CLOTH_DIR / fit["image_path"])

    if use_llm_prompt:
        meta   = build_meta_prompt(fit)
        prompt = "".join(generate_text_stream(meta)).strip()
    else:
        prompt = build_fit_prompt(fit)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{gid}_{model_id}_{timestamp}.png"

    try:
        result_path = generate_tryon_with_model(
            model_id=model_id,
            person_image_bytes=st.session_state.person_image_bytes,
            garment_image_path=garment_img_path,
            prompt=prompt,
            output_path=str(output_file),
        )
        st.session_state.last_result = {
            "garment_id": gid,
            "model_id":   model_id,
            "model_label": model_label,
            "fit":        fit,
            "prompt":     prompt,
            "image_path": result_path,
        }
    except Exception as e:
        error_box.error(f"Generation error ({model_label}): {e}")
        st.session_state.last_result = None

    status_box.empty()
    loader_box.empty()
    st.session_state.selected_garment_id = None


# ─────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────

if st.session_state.last_result:
    st.divider()
    r = st.session_state.last_result
    st.header(f"Your Try-On Result  ·  {r.get('model_label', '')}")

    result_cols = st.columns(3)
    with result_cols[0]:
        st.image(st.session_state.person_image_bytes, caption="You", use_container_width=True)
    with result_cols[1]:
        g_img = CLOTH_DIR / r["fit"]["image_path"]
        st.image(str(g_img), caption=r["fit"]["garment_name"], use_container_width=True)
    with result_cols[2]:
        st.image(r["image_path"], caption="Try-On", use_container_width=True)

    with st.expander("Fit analysis"):
        st.write(
            f"**Garment:** {r['fit']['garment_name']} "
            f"(size {r['fit']['garment_size']}, {r['fit']['garment_fit_style']})"
        )
        st.write(f"**Deltas (garment − you), {r['fit']['units']}:**")
        st.json(r["fit"]["deltas"])

    with st.expander(f"Prompt sent to {r.get('model_label', 'model')}"):
        st.code(r["prompt"], language="text")

    with open(r["image_path"], "rb") as f:
        st.download_button(
            "Download result",
            data=f.read(),
            file_name=Path(r["image_path"]).name,
            mime="image/png",
        )
