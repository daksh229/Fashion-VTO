"""Fit-Aware Virtual Try-On — Streamlit frontend.

Flow:
    1. User uploads photo + enters 4 body dimensions in the sidebar.
    2. Clicks "Load Wardrobe" → grid of all garments from cloth/garments.json.
    3. Clicks "Try Now" on a garment:
        - fit_calculator.calculate_fit()  -> deltas
        - Groq writes the detailed prompt (or rule-based fallback is used)
        - gemini_client.generate_tryon()    -> saves image to output/
    4. Final try-on image is displayed alongside person + garment.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit_calculator import calculate_fit
from src.prompt_builder import build_fit_prompt, build_meta_prompt
from src.gemini_client import generate_tryon
from src.groq_client import generate_text_stream

CLOTH_DIR = PROJECT_ROOT / "cloth_3D"
CATALOG_PATH = CLOTH_DIR / "garments.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_PERSON_PATH = OUTPUT_DIR / "_person_temp.png"


@st.cache_data
def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_uploaded_person(uploaded_file) -> Path:
    Image.open(uploaded_file).convert("RGB").save(TEMP_PERSON_PATH)
    return TEMP_PERSON_PATH


STATUS_STREAM_HTML = """
<!doctype html>
<html>
<head>
<style>
  html, body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .card {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 18px; padding: 36px 24px; margin: 0 auto; max-width: 640px;
    background: #ffffff; border: 1px solid #ececec; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .spinner {
    width: 44px; height: 44px; border-radius: 50%;
    border: 3px solid rgba(255, 75, 75, 0.15);
    border-top-color: #ff4b4b;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status {
    font-size: 16px; color: #2c2c2c; min-height: 24px;
    text-align: center; max-width: 540px; line-height: 1.4;
    transition: opacity 0.45s ease;
  }
  .status.fade { opacity: 0; }
  .meta {
    font-size: 11px; color: #999; letter-spacing: 0.08em;
    text-transform: uppercase; font-variant-numeric: tabular-nums;
  }
  .dot {
    display: inline-block; width: 4px; height: 4px; border-radius: 50%;
    background: #ff4b4b; margin: 0 3px; animation: pulse 1.2s ease-in-out infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.15s; }
  .dot:nth-child(3) { animation-delay: 0.3s; }
  @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
</style>
</head>
<body>
<div class="card">
  <div class="spinner"></div>
  <div id="status" class="status">Analyzing your body measurements…</div>
  <div class="meta">
    elapsed <span id="elapsed">0:00</span>
    &nbsp;·&nbsp;
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>
</div>
<script>
(function(){
  const messages = [
    "Analyzing your body measurements…",
    "Calculating fit deltas across every landmark…",
    "Tailoring the garment to your frame…",
    "Studying how the fabric will tension, drape, and fold…",
    "Drafting fit-aware rendering instructions for the image model…",
    "Composing the final try-on scene…",
    "Sending the brief to the image model for a photorealistic render…",
    "Polishing lighting, seams, and texture detail…",
    "Finalizing your virtual try-on…",
  ];
  const el = document.getElementById('status');
  const elapsedEl = document.getElementById('elapsed');
  let i = 0;
  const start = Date.now();

  function rotate(){
    el.classList.add('fade');
    setTimeout(() => {
      i = (i + 1) % messages.length;
      el.textContent = messages[i];
      el.classList.remove('fade');
    }, 450);
  }
  setInterval(rotate, 2800);

  function tick(){
    const s = Math.floor((Date.now() - start) / 1000);
    const m = Math.floor(s / 60);
    elapsedEl.textContent = m + ':' + String(s % 60).padStart(2, '0');
  }
  setInterval(tick, 1000);
  tick();
})();
</script>
</body>
</html>
"""


def render_status_stream():
    components.html(STATUS_STREAM_HTML, height=220, scrolling=False)


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Fit-Aware Virtual Try-On",
    page_icon="👗",
    layout="wide",
)

st.title("Fit-Aware Virtual Try-On")
st.caption(
    "Upload your photo, enter your measurements, pick a garment, "
    "and preview how it will actually fit you — not just how it looks pasted on."
)

# ---------------- Session state ----------------
for key, default in [
    ("wardrobe_loaded", False),
    ("selected_garment_id", None),
    ("last_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- Sidebar: user inputs ----------------
with st.sidebar:
    st.header("Your Details")

    uploaded = st.file_uploader(
        "Upload your photo",
        type=["jpg", "jpeg", "png"],
        help="Clear, front-facing, upper-body photo works best.",
    )

    st.subheader("Your measurements (inches)")

    st.markdown("**Upper body** — used for tops")
    length = st.number_input("Length (torso)", min_value=10.0, max_value=40.0, value=26.0, step=0.5)
    chest = st.number_input("Chest", min_value=20.0, max_value=60.0, value=36.0, step=0.5)
    shoulder = st.number_input("Shoulder", min_value=10.0, max_value=30.0, value=15.0, step=0.5)
    arm_length = st.number_input("Arm length", min_value=5.0, max_value=30.0, value=9.0, step=0.5)

    st.markdown("**Lower body** — used for pants / shorts")
    waist = st.number_input("Waist", min_value=20.0, max_value=60.0, value=32.0, step=0.5)
    hip = st.number_input("Hip", min_value=24.0, max_value=70.0, value=38.0, step=0.5)
    thigh = st.number_input("Thigh", min_value=14.0, max_value=40.0, value=22.0, step=0.5)
    inseam = st.number_input("Inseam (leg)", min_value=10.0, max_value=40.0, value=30.0, step=0.5)

    if st.button("Load Wardrobe →", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Please upload your photo first.")
        else:
            person_path = save_uploaded_person(uploaded)
            st.session_state.person_image_path = str(person_path)
            st.session_state.person_dimensions = {
                "length": length,
                "chest": chest,
                "shoulder": shoulder,
                "arm_length": arm_length,
                "waist": waist,
                "hip": hip,
                "thigh": thigh,
                "inseam": inseam,
            }
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
    st.image(st.session_state.person_image_path, caption="You", width=220)
with col_b:
    st.subheader("Your dimensions")
    pd = st.session_state.person_dimensions
    top_keys = ["length", "chest", "shoulder", "arm_length"]
    bot_keys = ["waist", "hip", "thigh", "inseam"]
    st.caption("Upper body")
    top_cols = st.columns(4)
    for c, k in zip(top_cols, top_keys):
        c.metric(k.replace("_", " ").title(), f'{pd[k]}"')
    st.caption("Lower body")
    bot_cols = st.columns(4)
    for c, k in zip(bot_cols, bot_keys):
        c.metric(k.replace("_", " ").title(), f'{pd[k]}"')

st.divider()

# ---------------- Wardrobe grid ----------------
st.header("Available Wardrobe")
catalog = load_catalog()
garments = catalog["garments"]
units = catalog.get("units", "inches")

cols_per_row = 3
for i in range(0, len(garments), cols_per_row):
    row = st.columns(cols_per_row)
    for col, garment in zip(row, garments[i:i + cols_per_row]):
        with col:
            with st.container(border=True):
                img_path = CLOTH_DIR / garment["path"]
                st.image(str(img_path), use_container_width=True)
                st.markdown(f"**{garment['name']}**")
                cat = garment.get("category", "Top")
                st.caption(
                    f"{garment['id']} · {cat} · {garment['type']} · "
                    f"Size {garment['size']} · {garment['fit_style']}"
                )
                d = garment["dimensions"]
                pairs = [f'{k.replace("_", " ").title()}: <b>{v}"</b>' for k, v in d.items()]
                rows = [" · ".join(pairs[i:i + 2]) for i in range(0, len(pairs), 2)]
                st.markdown(
                    "<small>" + "<br>".join(rows) + "</small>",
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
    stream_box = st.empty()
    error_box = st.empty()

    status_box.markdown(
        f"### Generating your try-on for **{gid}**\n"
        f"<span style='color:#888'>This usually takes 30–60 seconds.</span>",
        unsafe_allow_html=True,
    )
    with stream_box.container():
        render_status_stream()

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
            person_image_path=st.session_state.person_image_path,
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
    stream_box.empty()
    st.session_state.selected_garment_id = None

# ---------------- Result display ----------------
if st.session_state.last_result:
    st.divider()
    st.header("Your Try-On Result")
    r = st.session_state.last_result
    result_cols = st.columns(3)
    with result_cols[0]:
        st.image(st.session_state.person_image_path, caption="You", use_container_width=True)
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
        st.write("**Deltas (garment − you), inches:**")
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
