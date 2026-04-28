"""Fit-Aware Virtual Try-On — Streamlit frontend.

Flow:
    1. User uploads photo + enters 8 body dimensions in the sidebar.
    2. Clicks "Load Wardrobe" → grid of all garments from cloth_3D/garments.json.
    3. Clicks "Try Now" on a garment:
        - fit_calculator.calculate_fit()                  -> deltas
        - PHASE 1: generate_tryon() for the FRONT view with person + garment
                   references and the full fit-aware prompt. Retries up to 3x.
        - PHASE 2: rotate_view() for LEFT, BACK, and RIGHT views in parallel,
                   each anchored on the front try-on as the identity/garment
                   source. Each retries up to 3x. Left/right also pass the
                   original person photo as a supplementary face reference.
    4. Results are displayed in an interactive HTML rotator — the user can
       click arrows, press ← / →, or drag the image to rotate between the
       four views.
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit_calculator import calculate_fit
from src.prompt_builder import (
    build_fit_prompt,
    build_meta_prompt,
    build_rotate_prompt,
    VALID_VIEWS,
)
from src.gemini_client import generate_tryon, rotate_view, generate_with_retry
from src.groq_client import generate_text

CLOTH_DIR = PROJECT_ROOT / "cloth_3D"
CATALOG_PATH = CLOTH_DIR / "garments.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_PERSON_PATH = OUTPUT_DIR / "_person_temp.png"

MAX_ATTEMPTS_PER_VIEW = 3
STAGGER_SECONDS = 0.4  # small delay between launching rotation workers


@st.cache_data
def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_uploaded_person(uploaded_file) -> Path:
    Image.open(uploaded_file).convert("RGB").save(TEMP_PERSON_PATH)
    return TEMP_PERSON_PATH


# ---------- Phase 1: Front view ----------

def generate_front_view(
    fit: dict,
    person_image_path: str,
    garment_image_path: str,
    output_path: str,
    use_llm_prompt: bool,
) -> tuple[str, str]:
    """Generate the anchor front view. Returns (output_path, prompt_used)."""
    if use_llm_prompt:
        meta = build_meta_prompt(fit, view="front")
        prompt = generate_text(meta).strip()
    else:
        prompt = build_fit_prompt(fit, view="front")

    result_path = generate_with_retry(
        generate_tryon,
        person_image_path=person_image_path,
        garment_image_path=garment_image_path,
        prompt=prompt,
        output_path=output_path,
        view="front",
        max_attempts=MAX_ATTEMPTS_PER_VIEW,
    )
    return result_path, prompt


# ---------- Phase 2: Rotation (runs in worker thread) ----------

def rotate_one_view(
    view: str,
    fit: dict,
    front_tryon_path: str,
    person_image_path: str | None,
    output_path: str,
    launch_delay: float = 0.0,
) -> tuple[str, str, str]:
    """Rotate the front try-on to `view`. Returns (view, output_path, prompt)."""
    if launch_delay > 0:
        time.sleep(launch_delay)

    prompt = build_rotate_prompt(fit, view=view)

    # Back view: face isn't visible, dropping the person photo simplifies
    # the prompt and reduces the chance Gemini renders a second body.
    person_ref = person_image_path if view in ("left", "right") else None

    result_path = generate_with_retry(
        rotate_view,
        front_tryon_path=front_tryon_path,
        person_image_path=person_ref,
        prompt=prompt,
        view=view,
        output_path=output_path,
        max_attempts=MAX_ATTEMPTS_PER_VIEW,
    )
    return view, result_path, prompt


# ---------- Status stream shown during generation ----------

STATUS_STREAM_HTML = r"""
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
  .phase {
    font-size: 11px; color: #ff4b4b; letter-spacing: 0.12em;
    text-transform: uppercase; font-weight: 600;
  }
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
  <div class="phase" id="phase">PHASE 1 · FRONT VIEW</div>
  <div id="status" class="status">Rendering the front view as the identity anchor…</div>
  <div class="meta">
    elapsed <span id="elapsed">0:00</span>
    &nbsp;·&nbsp;
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>
</div>
<script>
(function(){
  const phase1 = [
    "Rendering the front view as the identity anchor…",
    "Analyzing your body measurements and fit deltas…",
    "Drafting fit-aware instructions for the image model…",
    "Composing the front view — this is the identity anchor for all other views…",
    "Studying how the fabric tensions, drapes, and folds on your frame…",
  ];
  const phase2 = [
    "Rotating to left profile, back, and right profile in parallel…",
    "Anchoring each new view to the identity from the front render…",
    "Preserving your face, hair, and the garment's exact fit from every angle…",
    "Rendering the back view — the garment's reverse side needs extra care…",
    "Polishing lighting and seam detail across all three rotations…",
    "Finalizing your 360° try-on…",
  ];
  const phaseEl  = document.getElementById('phase');
  const el       = document.getElementById('status');
  const elapsedEl= document.getElementById('elapsed');
  const start = Date.now();

  // Swap to phase 2 after ~18 seconds (typical front-view generation time).
  const PHASE2_AT = 18;
  let i = 0;

  function current(){
    const s = Math.floor((Date.now() - start) / 1000);
    if (s >= PHASE2_AT) { phaseEl.textContent = 'PHASE 2 · SIDE / BACK / RIGHT VIEWS'; return phase2; }
    return phase1;
  }

  function rotate(){
    el.classList.add('fade');
    setTimeout(() => {
      const msgs = current();
      i = (i + 1) % msgs.length;
      el.textContent = msgs[i];
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
    components.html(STATUS_STREAM_HTML, height=240, scrolling=False)


# ---------- The HTML rotator (4-view interactive viewer) ----------

ROTATOR_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<style>
  html, body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .viewer { position: relative; max-width: 520px; margin: 0 auto; user-select: none; -webkit-user-select: none; }
  .stage { position: relative; width: 100%; aspect-ratio: 3 / 4; background: #fafafa; border: 1px solid #ececec; border-radius: 12px; overflow: hidden; cursor: grab; touch-action: pan-y; }
  .stage.dragging { cursor: grabbing; }
  .stage img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; transition: opacity 0.25s ease; }
  .stage img.active { opacity: 1; }
  .arrow { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.9); border: 1px solid #ddd; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 20px; color: #333; box-shadow: 0 1px 4px rgba(0,0,0,0.1); z-index: 3; transition: background 0.15s ease; }
  .arrow:hover { background: #fff; }
  .arrow.prev { left: 10px; }
  .arrow.next { right: 10px; }
  .labels { display: flex; justify-content: center; gap: 8px; margin-top: 14px; }
  .label { padding: 6px 14px; border-radius: 16px; background: #f4f4f4; font-size: 13px; color: #666; cursor: pointer; border: 1px solid transparent; transition: all 0.15s ease; }
  .label.active { background: #ff4b4b; color: #fff; border-color: #ff4b4b; }
  .hint { text-align: center; font-size: 11px; color: #aaa; margin-top: 10px; letter-spacing: 0.04em; }
  .missing { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #999; font-size: 13px; text-align: center; padding: 20px; opacity: 0; transition: opacity 0.25s ease; }
  .missing.active { opacity: 1; }
</style>
</head>
<body>
<div class="viewer" id="viewer">
  <div class="stage" id="stage">
    __SLIDES__
    <div class="arrow prev" id="prev">‹</div>
    <div class="arrow next" id="next">›</div>
  </div>
  <div class="labels" id="labels">
    <div class="label" data-i="0">Front</div>
    <div class="label" data-i="1">Left</div>
    <div class="label" data-i="2">Back</div>
    <div class="label" data-i="3">Right</div>
  </div>
  <div class="hint">drag · click arrows · ← / → keys</div>
</div>
<script>
(function(){
  const order = ['front', 'left', 'back', 'right'];
  const slides = document.querySelectorAll('.stage > img, .stage > .missing');
  const labels = document.querySelectorAll('.label');
  const stage = document.getElementById('stage');
  let idx = 0;

  function render(){
    slides.forEach((s, i) => s.classList.toggle('active', i === idx));
    labels.forEach((l, i) => l.classList.toggle('active', i === idx));
  }

  function go(step){ idx = (idx + step + order.length) % order.length; render(); }

  document.getElementById('prev').addEventListener('click', () => go(-1));
  document.getElementById('next').addEventListener('click', () => go(1));
  labels.forEach(l => l.addEventListener('click', () => { idx = parseInt(l.dataset.i, 10); render(); }));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft')  { go(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { go(1);  e.preventDefault(); }
  });

  let dragStartX = null;
  const DRAG_THRESHOLD = 40;
  function onDragStart(x){ dragStartX = x; stage.classList.add('dragging'); }
  function onDragEnd(x){
    if (dragStartX === null) return;
    const dx = x - dragStartX;
    if      (dx >  DRAG_THRESHOLD) go(-1);
    else if (dx < -DRAG_THRESHOLD) go(1);
    dragStartX = null; stage.classList.remove('dragging');
  }
  stage.addEventListener('mousedown', (e) => { onDragStart(e.clientX); e.preventDefault(); });
  window.addEventListener('mouseup',   (e) => onDragEnd(e.clientX));
  stage.addEventListener('touchstart', (e) => onDragStart(e.touches[0].clientX), { passive: true });
  stage.addEventListener('touchend',   (e) => onDragEnd(e.changedTouches[0].clientX), { passive: true });

  render();
})();
</script>
</body>
</html>
"""


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def render_rotator(image_paths: dict):
    order = ["front", "left", "back", "right"]
    slides_html = []
    for view in order:
        path = image_paths.get(view)
        if path and Path(path).exists():
            b64 = _img_to_b64(path)
            slides_html.append(f'<img src="data:image/png;base64,{b64}" alt="{view} view" />')
        else:
            slides_html.append(
                f'<div class="missing">{view.title()} view failed to generate. '
                f'Try another garment or toggle prompt mode.</div>'
            )
    html = ROTATOR_TEMPLATE.replace("__SLIDES__", "\n    ".join(slides_html))
    components.html(html, height=780, scrolling=False)


# ---------------- Page config ----------------
st.set_page_config(page_title="Fit-Aware Virtual Try-On", page_icon="👗", layout="wide")

st.title("Fit-Aware Virtual Try-On")
st.caption(
    "Upload your photo, enter your measurements, pick a garment, "
    "and preview how it will actually fit you — from all four sides."
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
                "length": length, "chest": chest, "shoulder": shoulder, "arm_length": arm_length,
                "waist": waist, "hip": hip, "thigh": thigh, "inseam": inseam,
            }
            st.session_state.wardrobe_loaded = True
            st.session_state.last_result = None
            st.session_state.selected_garment_id = None

    st.divider()
    use_llm_prompt = st.toggle(
        "Use Groq to write the prompt",
        value=True,
        help=(
            "ON: Groq writes the fit-aware text prompt, then Gemini uses that "
            "prompt to generate the front view. OFF: use the deterministic rule-based prompt."
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
                st.markdown("<small>" + "<br>".join(rows) + "</small>", unsafe_allow_html=True)
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
        f"### Generating 4-view try-on for **{gid}**\n"
        f"<span style='color:#888'>Front view generates first (identity anchor), then "
        f"left/back/right rotate in parallel. Usually 30–50 seconds total.</span>",
        unsafe_allow_html=True,
    )
    with stream_box.container():
        render_status_stream()

    fit = calculate_fit(st.session_state.person_dimensions, gid)
    garment_img_path = str(CLOTH_DIR / fit["image_path"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_paths: dict[str, str] = {}
    prompts: dict[str, str] = {}
    errors: dict[str, str] = {}

    # ---- PHASE 1: Front view ----
    try:
        front_output = str(OUTPUT_DIR / f"{gid}_front_{timestamp}.png")
        front_path, front_prompt = generate_front_view(
            fit=fit,
            person_image_path=st.session_state.person_image_path,
            garment_image_path=garment_img_path,
            output_path=front_output,
            use_llm_prompt=use_llm_prompt,
        )
        image_paths["front"] = front_path
        prompts["front"] = front_prompt
    except Exception as e:
        errors["front"] = str(e)

    # ---- PHASE 2: Rotation views (only if front succeeded) ----
    if "front" in image_paths:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for i, view in enumerate(("left", "back", "right")):
                delay = i * STAGGER_SECONDS  # stagger launches to ease rate limits
                futures[pool.submit(
                    rotate_one_view,
                    view=view,
                    fit=fit,
                    front_tryon_path=image_paths["front"],
                    person_image_path=st.session_state.person_image_path,
                    output_path=str(OUTPUT_DIR / f"{gid}_{view}_{timestamp}.png"),
                    launch_delay=delay,
                )] = view

            for fut in as_completed(futures):
                view = futures[fut]
                try:
                    _, path, prompt = fut.result()
                    image_paths[view] = path
                    prompts[view] = prompt
                except Exception as e:
                    errors[view] = str(e)

    stream_box.empty()
    status_box.empty()

    if errors:
        err_lines = []
        for v in VALID_VIEWS:
            if v in errors:
                err_lines.append(f"- **{v}**: {errors[v]}")
        error_box.warning(
            "Some views failed after all retries. Successful views will still display below.\n\n"
            + "\n".join(err_lines)
        )

    if image_paths:
        st.session_state.last_result = {
            "garment_id": gid,
            "fit": fit,
            "prompts": prompts,
            "image_paths": image_paths,
        }
    else:
        st.session_state.last_result = None

    st.session_state.selected_garment_id = None

# ---------------- Result display ----------------
if st.session_state.last_result:
    st.divider()
    st.header("Your 360° Try-On Result")

    r = st.session_state.last_result
    top_cols = st.columns([1, 1, 2])
    with top_cols[0]:
        st.image(st.session_state.person_image_path, caption="You", use_container_width=True)
    with top_cols[1]:
        g_img = CLOTH_DIR / r["fit"]["image_path"]
        st.image(str(g_img), caption=r["fit"]["garment_name"], use_container_width=True)
    with top_cols[2]:
        st.markdown("#### Rotate your try-on")
        render_rotator(r["image_paths"])

    with st.expander("Fit analysis"):
        st.write(
            f"**Garment:** {r['fit']['garment_name']} "
            f"(size {r['fit']['garment_size']}, {r['fit']['garment_fit_style']})"
        )
        st.write("**Deltas (garment − you), inches:**")
        st.json(r["fit"]["deltas"])

    with st.expander("Prompts sent to Gemini (per view)"):
        for view in VALID_VIEWS:
            if view in r["prompts"]:
                st.markdown(f"**{view.upper()}**")
                st.code(r["prompts"][view], language="text")

    st.markdown("#### Download")
    successful_views = [v for v in VALID_VIEWS if v in r["image_paths"]]
    if successful_views:
        dl_cols = st.columns(len(successful_views))
        for col, view in zip(dl_cols, successful_views):
            with col:
                with open(r["image_paths"][view], "rb") as f:
                    st.download_button(
                        f"{view.title()}",
                        data=f.read(),
                        file_name=Path(r["image_paths"][view]).name,
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{view}",
                    )
