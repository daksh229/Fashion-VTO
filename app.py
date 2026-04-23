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
import time
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

CLOTH_DIR = PROJECT_ROOT / "cloth"
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


def stream_words(text: str, delay: float = 0.02):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


DINO_GAME_HTML = """
<!doctype html>
<html>
<head>
<style>
  html, body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 8px 0; }
  .hud { display: flex; justify-content: space-between; width: 600px; max-width: 95vw; color: #444; font-size: 13px; }
  canvas { background: #fafafa; border: 1px solid #e6e6e6; border-radius: 6px; max-width: 95vw; touch-action: manipulation; }
  .hint { color: #888; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="hud">
    <span id="status">Press <b>Space</b> or <b>tap</b> to jump &nbsp;·&nbsp; <b>↓</b> to duck</span>
    <span>Score: <b id="score">0</b> &nbsp;·&nbsp; HI <b id="hi">0</b></span>
  </div>
  <canvas id="game" width="600" height="180"></canvas>
  <div class="hint">A little something to do while your try-on renders.</div>
</div>
<script>
(function(){
  const cvs = document.getElementById('game');
  const ctx = cvs.getContext('2d');
  const W = cvs.width, H = cvs.height;
  const GROUND_Y = H - 30;
  const scoreEl = document.getElementById('score');
  const hiEl = document.getElementById('hi');
  const statusEl = document.getElementById('status');

  let hi = parseInt(localStorage.getItem('dino_hi') || '0', 10);
  hiEl.textContent = hi;

  const dino = {
    x: 50, y: GROUND_Y - 40, w: 28, h: 40,
    vy: 0, onGround: true, ducking: false,
  };
  const GRAVITY = 0.7;
  const JUMP_V = -12.5;

  let obstacles = [];
  let clouds = [];
  let frame = 0;
  let speed = 5;
  let score = 0;
  let gameOver = false;
  let started = false;

  function reset(){
    obstacles = []; clouds = []; frame = 0; speed = 5; score = 0;
    gameOver = false; dino.y = GROUND_Y - 40; dino.vy = 0; dino.onGround = true; dino.ducking = false;
    statusEl.innerHTML = 'Press <b>Space</b> or <b>tap</b> to jump &nbsp;·&nbsp; <b>↓</b> to duck';
  }

  function jump(){
    if (gameOver) { reset(); started = true; return; }
    if (dino.onGround) { dino.vy = JUMP_V; dino.onGround = false; started = true; }
  }
  function duckOn(){ if (dino.onGround) { dino.ducking = true; dino.h = 22; dino.y = GROUND_Y - 22; } }
  function duckOff(){ if (dino.ducking) { dino.ducking = false; dino.h = 40; dino.y = GROUND_Y - 40; } }

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' || e.code === 'ArrowUp') { e.preventDefault(); jump(); }
    if (e.code === 'ArrowDown') { e.preventDefault(); duckOn(); }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'ArrowDown') { duckOff(); }
  });
  cvs.addEventListener('mousedown', jump);
  cvs.addEventListener('touchstart', (e) => { e.preventDefault(); jump(); }, {passive: false});

  function spawn(){
    if (frame % Math.max(45, 90 - Math.floor(speed * 4)) === 0 && Math.random() < 0.7) {
      const isBird = Math.random() < 0.25 && score > 200;
      if (isBird) {
        const flyY = Math.random() < 0.5 ? GROUND_Y - 55 : GROUND_Y - 30;
        obstacles.push({ x: W + 10, y: flyY, w: 28, h: 18, type: 'bird', flap: 0 });
      } else {
        const big = Math.random() < 0.4;
        const w = big ? 22 : 14;
        const h = big ? 38 : 28;
        obstacles.push({ x: W + 10, y: GROUND_Y - h, w, h, type: 'cactus' });
      }
    }
    if (frame % 110 === 0) {
      clouds.push({ x: W + 10, y: 20 + Math.random() * 50, w: 36 });
    }
  }

  function rectsHit(a, b){
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function step(){
    if (started && !gameOver) {
      frame++;
      score++;
      if (frame % 200 === 0) speed += 0.4;
      dino.vy += GRAVITY;
      dino.y += dino.vy;
      const floor = GROUND_Y - dino.h;
      if (dino.y >= floor) { dino.y = floor; dino.vy = 0; dino.onGround = true; }

      spawn();
      for (const o of obstacles) o.x -= speed;
      for (const c of clouds) c.x -= speed * 0.4;
      obstacles = obstacles.filter(o => o.x + o.w > -10);
      clouds = clouds.filter(c => c.x + c.w > -10);

      for (const o of obstacles) {
        const dHit = { x: dino.x + 3, y: dino.y + 3, w: dino.w - 6, h: dino.h - 6 };
        if (rectsHit(dHit, o)) {
          gameOver = true;
          if (score > hi) { hi = score; localStorage.setItem('dino_hi', hi); hiEl.textContent = hi; }
          statusEl.innerHTML = '<b>Game over</b> — press <b>Space</b> or <b>tap</b> to play again';
        }
      }
      scoreEl.textContent = Math.floor(score / 4);
    }
    draw();
    requestAnimationFrame(step);
  }

  function draw(){
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#bbb';
    for (const c of clouds) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 8, 0, Math.PI*2);
      ctx.arc(c.x+10, c.y+2, 10, 0, Math.PI*2);
      ctx.arc(c.x+22, c.y, 7, 0, Math.PI*2);
      ctx.fill();
    }
    ctx.strokeStyle = '#888';
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y + 1);
    ctx.lineTo(W, GROUND_Y + 1);
    ctx.stroke();

    ctx.fillStyle = '#444';
    if (dino.ducking) {
      ctx.fillRect(dino.x, dino.y, 36, dino.h);
      ctx.fillRect(dino.x + 30, dino.y - 4, 8, 8);
    } else {
      ctx.fillRect(dino.x, dino.y, dino.w, dino.h);
      ctx.fillRect(dino.x + dino.w - 4, dino.y - 6, 12, 10);
      ctx.fillStyle = '#fafafa';
      ctx.fillRect(dino.x + dino.w + 2, dino.y - 3, 2, 2);
      ctx.fillStyle = '#444';
      ctx.fillRect(dino.x - 4, dino.y + dino.h - 6, 6, 4);
      ctx.fillRect(dino.x + dino.w - 8, dino.y + dino.h - 6, 6, 4);
    }

    for (const o of obstacles) {
      if (o.type === 'cactus') {
        ctx.fillStyle = '#3b6b3b';
        ctx.fillRect(o.x, o.y, o.w, o.h);
        ctx.fillRect(o.x - 4, o.y + 6, 4, o.h * 0.4);
        ctx.fillRect(o.x + o.w, o.y + 10, 4, o.h * 0.35);
      } else {
        ctx.fillStyle = '#666';
        const wing = (Math.floor(frame / 8) % 2 === 0) ? -6 : 6;
        ctx.fillRect(o.x, o.y, o.w, o.h);
        ctx.fillRect(o.x + 4, o.y + wing, 16, 4);
      }
    }
  }

  step();
})();
</script>
</body>
</html>
"""


def render_dino_game():
    components.html(DINO_GAME_HTML, height=240, scrolling=False)


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
    length = st.number_input("Length (torso)", min_value=10.0, max_value=40.0, value=26.0, step=0.5)
    chest = st.number_input("Chest", min_value=20.0, max_value=60.0, value=36.0, step=0.5)
    shoulder = st.number_input("Shoulder", min_value=10.0, max_value=30.0, value=15.0, step=0.5)
    arm_length = st.number_input("Arm length", min_value=5.0, max_value=30.0, value=9.0, step=0.5)

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
    pd_cols = st.columns(4)
    pd = st.session_state.person_dimensions
    for c, (label, val) in zip(pd_cols, pd.items()):
        c.metric(label.replace("_", " ").title(), f'{val}"')

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
                st.caption(
                    f"{garment['id']} · {garment['type']} · "
                    f"Size {garment['size']} · {garment['fit_style']}"
                )
                d = garment["dimensions"]
                st.markdown(
                    f"<small>"
                    f"Length: <b>{d['length']}\"</b> · Chest: <b>{d['chest']}\"</b><br>"
                    f"Shoulder: <b>{d['shoulder']}\"</b> · Arm: <b>{d['arm_length']}\"</b>"
                    f"</small>",
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
    game_box = st.empty()
    error_box = st.empty()

    status_box.markdown(
        f"### ✨ Generating your try-on for **{gid}**…\n"
        f"<span style='color:#888'>This usually takes 30–60 seconds. "
        f"Play the mini-game below while you wait.</span>",
        unsafe_allow_html=True,
    )
    with game_box.container():
        render_dino_game()

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
    game_box.empty()
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
