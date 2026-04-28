"""Fit-Aware Virtual Try-On — Streamlit frontend.

Flow:
    1. User picks photo source (upload or camera) and provides a frontal photo
       (optional side photo for circumference accuracy).
    2. User enters height (mandatory pixel→cm scale anchor) plus optional
       shoulder/waist correction anchors.
    3. "Detect Measurements" runs the MediaPipe pipeline; results are shown
       as editable inputs in the sidebar for last-moment user verification.
    4. "Load Wardrobe" → grid of all garments from cloth/garments.json.
    5. "Try Now" on a garment:
        - fit_calculator.calculate_fit()  -> deltas
        - Groq writes the detailed prompt (or rule-based fallback is used)
        - gemini_client.generate_tryon()    -> saves image to output/
    6. Final try-on image is displayed alongside person + garment.

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
from src.gemini_client import generate_tryon
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


TAILORING_HTML = """
<style>
  .tailoring-wrap {
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
  }
  .tailor-icon {
    font-size: 22px;
    animation: tailor-pulse 1.6s ease-in-out infinite;
  }
  .tailor-msg-stack {
    position: relative;
    height: 22px;
    flex: 1;
    overflow: hidden;
  }
  .tailor-msg {
    position: absolute;
    inset: 0;
    font-size: 15px;
    color: #2a2f3a;
    font-weight: 500;
    opacity: 0;
    animation: tailor-rotate 16s ease-in-out infinite;
    white-space: nowrap;
  }
  .tailor-msg.m1 { animation-delay: 0s; }
  .tailor-msg.m2 { animation-delay: 4s; }
  .tailor-msg.m3 { animation-delay: 8s; }
  .tailor-msg.m4 { animation-delay: 12s; }
  @keyframes tailor-rotate {
    0%, 25%, 100% { opacity: 0; transform: translateY(8px); }
    3%, 22% { opacity: 1; transform: translateY(0); }
  }
  @keyframes tailor-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.15); }
  }
  @keyframes tailor-shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
</style>
<div class="tailoring-wrap">
  <span class="tailor-icon">&#9986;&#65039;</span>
  <div class="tailor-msg-stack">
    <span class="tailor-msg m1">AI is tailoring the garment to your measurements&hellip;</span>
    <span class="tailor-msg m2">Matching fabric drape to your silhouette&hellip;</span>
    <span class="tailor-msg m3">Preserving every print, stitch, and color&hellip;</span>
    <span class="tailor-msg m4">Rendering your final try-on&hellip;</span>
  </div>
</div>
"""


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Fit-Aware Virtual Try-On",
    page_icon="👗",
    layout="wide",
)

st.title("Fit-Aware Virtual Try-On")
st.caption(
    "Take or upload your photo, enter your height, and we'll auto-detect your "
    "measurements. You can verify and edit them before picking a garment."
)

# ---------------- Session state ----------------
for key, default in [
    ("wardrobe_loaded", False),
    ("selected_garment_id", None),
    ("last_result", None),
    ("measurement_result", None),
    ("person_image_bytes", None),
    ("person_dimensions", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- Sidebar: user inputs ----------------
with st.sidebar:
    st.header("Your Details")

    manual_mode = st.checkbox(
        "Skip detection — enter manually",
        value=False,
        help="Bypass the photo-based pipeline and enter measurements directly.",
    )

    person_dimensions: dict[str, float] | None = None
    person_image_bytes: bytes | None = None
    category_for_input = "top"

    if manual_mode:
        # ----- Manual entry path (legacy) -----
        uploaded = st.file_uploader(
            "Upload your photo",
            type=["jpg", "jpeg", "png"],
            help="Used as the person reference for try-on rendering. No measurements derived from it.",
        )
        st.subheader("Your measurements (cm)")
        length_cm = st.number_input("Length / torso", min_value=20.0, max_value=120.0, value=66.0, step=0.5)
        chest_cm = st.number_input("Chest circumference", min_value=50.0, max_value=180.0, value=91.0, step=0.5)
        shoulder_cm_in = st.number_input("Shoulder width", min_value=20.0, max_value=80.0, value=38.0, step=0.5)
        arm_length_cm = st.number_input("Arm length", min_value=10.0, max_value=90.0, value=23.0, step=0.5)

        if uploaded is not None:
            person_image_bytes = uploaded.getvalue()
        person_dimensions = {
            "length": length_cm,
            "chest": chest_cm,
            "shoulder": shoulder_cm_in,
            "arm_length": arm_length_cm,
        }

    else:
        # ----- Auto-detect path -----
        st.subheader("1. Frontal photo")
        frontal_source = st.radio(
            "Source",
            ["Upload", "Camera"],
            horizontal=True,
            key="frontal_src",
        )
        if frontal_source == "Upload":
            frontal_file = st.file_uploader(
                "Upload frontal photo",
                type=["jpg", "jpeg", "png"],
                key="frontal_upload",
            )
        else:
            frontal_file = st.camera_input("Take a frontal photo", key="frontal_cam")

        st.subheader("2. Side photo (optional)")
        st.caption("Improves circumference accuracy for chest, waist, hip, thigh.")
        side_source = st.radio(
            "Source",
            ["None", "Upload", "Camera"],
            horizontal=True,
            key="side_src",
        )
        if side_source == "Upload":
            side_file = st.file_uploader(
                "Upload side photo",
                type=["jpg", "jpeg", "png"],
                key="side_upload",
            )
        elif side_source == "Camera":
            side_file = st.camera_input("Take a side photo", key="side_cam")
        else:
            side_file = None

        st.subheader("3. Reference height")
        height_unit = st.radio("Unit", ["cm", "ft/in"], horizontal=True, key="height_unit")
        if height_unit == "cm":
            height_cm = st.number_input("Height (cm)", 100.0, 230.0, 170.0, 0.5)
        else:
            ft_col, in_col = st.columns(2)
            ft = ft_col.number_input("Feet", 3, 8, 5)
            inch = in_col.number_input("Inches", 0.0, 11.5, 7.0, 0.5)
            height_cm = (ft * 12 + inch) * 2.54

        with st.expander("Optional correction anchors"):
            st.caption("Leave at 0 to skip. Improves accuracy where provided.")
            shoulder_anchor_cm = st.number_input(
                "Shoulder width (cm)", 0.0, 80.0, 0.0, 0.5, key="shoulder_anchor"
            )
            waist_anchor_cm = st.number_input(
                "Waist circumference (cm)", 0.0, 200.0, 0.0, 0.5, key="waist_anchor"
            )

        st.subheader("4. Detect")
        detect_disabled = frontal_file is None
        detect_clicked = st.button(
            "Detect Measurements",
            disabled=detect_disabled,
            type="secondary",
            use_container_width=True,
        )

        if detect_clicked and frontal_file is not None:
            try:
                with st.spinner("Detecting body landmarks…"):
                    pose_solution = get_pose_solution()
                    result = run_measurement_pipeline(
                        frontal_bytes=frontal_file.getvalue(),
                        height_cm=float(height_cm),
                        side_bytes=side_file.getvalue() if side_file is not None else None,
                        shoulder_cm=shoulder_anchor_cm or None,
                        waist_cm=waist_anchor_cm or None,
                        pose_solution=pose_solution,
                    )
                st.session_state.measurement_result = result
                st.session_state.person_image_bytes = frontal_file.getvalue()
                # Clear stale verification widget state so the freshly-detected
                # values are shown, not whatever the widgets cached from a prior
                # detection run.
                for stale_key in [k for k in st.session_state if k.startswith("verify_")]:
                    del st.session_state[stale_key]
                st.success("Detection complete — verify the values below.")
            except MeasurementError as e:
                st.error(f"Detection failed: {e}")
                st.session_state.measurement_result = None

        # Always carry the latest frontal photo through (so try-on works even
        # without re-detection).
        if frontal_file is not None and st.session_state.person_image_bytes is None:
            st.session_state.person_image_bytes = frontal_file.getvalue()
        person_image_bytes = st.session_state.person_image_bytes

        # ----- Verification panel -----
        mr = st.session_state.measurement_result
        if mr is not None:
            st.divider()
            st.subheader("Verify measurements")
            st.caption("Edit any value before loading the wardrobe. Confidence: 🟢 high · 🟡 medium · 🔴 low.")
            for w in mr.warnings:
                st.warning(w, icon="⚠️")

            # For now the catalog only has tops; expose the four top keys.
            verification_keys = [
                ("length", "Length / torso (cm)"),
                ("chest", "Chest circumference (cm)"),
                ("shoulder", "Shoulder width (cm)"),
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

    st.divider()
    load_disabled = person_image_bytes is None or person_dimensions is None or not all(
        v and v > 0 for v in person_dimensions.values()
    )
    if st.button("Load Wardrobe →", type="primary", use_container_width=True, disabled=load_disabled):
        st.session_state.person_image_bytes = person_image_bytes
        st.session_state.person_dimensions = person_dimensions
        st.session_state.wardrobe_loaded = True
        st.session_state.last_result = None
        st.session_state.selected_garment_id = None

    st.divider()
    use_llm_prompt = st.toggle(
        "Use Groq to write the prompt",
        value=True,
        help=(
            "ON: Groq writes the fit-aware text prompt, then Gemini Nano Banana uses that "
            "prompt to generate the final try-on image. OFF: use the deterministic rule-based prompt."
        ),
    )

    if not (os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")):
        st.warning("Gemini API key not found in `.env`. Image generation will fail until you add it.")
    if use_llm_prompt and not os.environ.get("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` not found in `.env`. Groq prompt generation will fail until you add it.")


# ---------------- Main area ----------------
if not st.session_state.wardrobe_loaded:
    st.info("← Enter your details in the sidebar and click **Load Wardrobe** to begin.")
    st.stop()

# Person preview
col_a, col_b = st.columns([1, 3])
with col_a:
    st.image(st.session_state.person_image_bytes, caption="You", width=220)
with col_b:
    st.subheader("Your dimensions")
    pd_cols = st.columns(4)
    pd = st.session_state.person_dimensions
    units_label = "cm"
    for c, (label, val) in zip(pd_cols, pd.items()):
        c.metric(label.replace("_", " ").title(), f"{val} {units_label}")

st.divider()

# ---------------- Wardrobe grid ----------------
st.header("Available Wardrobe")
catalog = load_catalog()
garments = catalog["garments"]
units = catalog.get("units", "cm")

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
                d = garment["dimensions"]
                relevant = garment.get("fit_relevant_keys", list(d.keys()))
                spec_pairs = " · ".join(
                    f"{k.replace('_', ' ').title()}: <b>{d[k]} {units}</b>"
                    for k in relevant
                    if k in d
                )
                st.markdown(
                    f"<small>{spec_pairs}</small>",
                    unsafe_allow_html=True,
                )
                if st.button("Try Now", key=f"try_{garment['id']}", use_container_width=True):
                    st.session_state.selected_garment_id = garment["id"]
                    st.session_state.last_result = None

# ---------------- Generation flow ----------------
if st.session_state.selected_garment_id:
    gid = st.session_state.selected_garment_id
    st.divider()

    status_box = st.empty()
    loader_box = st.empty()
    error_box = st.empty()

    status_box.markdown(
        f"### ✨ Generating your try-on for **{gid}**\n"
        f"<span style='color:#888'>This usually takes 30–60 seconds.</span>",
        unsafe_allow_html=True,
    )
    loader_box.markdown(TAILORING_HTML, unsafe_allow_html=True)

    fit = calculate_fit(st.session_state.person_dimensions, gid)
    garment_img_path = str(CLOTH_DIR / fit["image_path"])

    if use_llm_prompt:
        meta = build_meta_prompt(fit)
        prompt = "".join(generate_text_stream(meta)).strip()
    else:
        prompt = build_fit_prompt(fit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{gid}_{timestamp}.png"

    try:
        result_path = generate_tryon(
            person_image=st.session_state.person_image_bytes,
            garment_image_path=garment_img_path,
            prompt=prompt,
            output_path=str(output_file),
        )
        st.session_state.last_result = {
            "garment_id": gid,
            "fit": fit,
            "prompt": prompt,
            "image_path": result_path,
        }
    except Exception as e:
        error_box.error(f"Generation error: {e}")
        st.session_state.last_result = None

    status_box.empty()
    loader_box.empty()
    st.session_state.selected_garment_id = None

# ---------------- Result display ----------------
if st.session_state.last_result:
    st.divider()
    st.header("Your Try-On Result")
    r = st.session_state.last_result
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

    with st.expander("Prompt sent to Gemini"):
        st.code(r["prompt"], language="text")

    with open(r["image_path"], "rb") as f:
        st.download_button(
            "Download result",
            data=f.read(),
            file_name=Path(r["image_path"]).name,
            mime="image/png",
        )
