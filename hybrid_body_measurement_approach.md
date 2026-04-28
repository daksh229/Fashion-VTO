# Hybrid Body Measurement Approach
> Estimate body measurements from a photo using pose estimation + user-provided reference inputs.
> **Status:** Implemented in `src/measurement/` and wired into the Streamlit app (`app.py`).

---

## Overview

The hybrid approach combines **open-source pose estimation** (MediaPipe Pose Landmarker, Tasks API) with **user-supplied reference measurements** (height, and optionally shoulder width / waist circumference) to produce body measurements without 3D scanning hardware or paid APIs.

The core idea:
- Pose estimation gives **2D body landmark positions in pixels**
- User-provided height gives a **pixel-to-cm scale factor**
- Optional shoulder/waist inputs act as **regional correction multipliers**
- An optional **side photo** unlocks depth-aware circumferences via ellipse approximation

Measurements cover **both upper and lower body** so the pipeline serves shirts/jackets and pants/jeans from a single detection.

> **Dimensionality note:** measurements are computed in **2D** (frontal projection) — see [`engine._dist`](src/measurement/engine.py#L24-L25). MediaPipe's `z` coordinate is captured but **not used** by any measurement. Circumferences are **2.5D**: a frontal width plus a side-photo depth (or a fixed `0.78 × width` fallback) fed into Ramanujan's ellipse-perimeter approximation. There is no SMPL or true 3D body model.

---

## Implemented Architecture

```
Streamlit UI (app.py)
 │
 ├─── Frontal photo (upload or st.camera_input)
 ├─── Side photo (optional, upload or camera)
 ├─── Height in cm or ft/in (mandatory)
 └─── Optional anchors: shoulder width, waist circumference
         │
         ▼
┌────────────────────────────────────────────────┐
│ src/measurement/preprocess.py                  │
│ • PIL decode + EXIF transpose (phone rotation) │
│ • Convert to RGB ndarray                       │
│ • Reject < 480 px on either side               │
└────────────────┬───────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────┐
│ src/measurement/pose.py                        │
│ • MediaPipe Tasks PoseLandmarker (lite model)  │
│ • Auto-downloads .task file to                 │
│   ~/.cache/mediapipe/ on first run             │
│ • Returns 33 named landmarks                   │
│   (e.g. "left_shoulder") in pixel coords       │
│ • Annotated PNG produced for the UI            │
└────────────────┬───────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────┐
│ src/measurement/scale.py                       │
│ • Primary: pixel(nose→heel) × 1.087            │
│   ≈ true crown-to-heel pixel height            │
│ • Fallback: shoulder anchor (if feet hidden)   │
│ • Computes shoulder_correction +               │
│   waist_correction multipliers                 │
└────────────────┬───────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────┐
│ src/measurement/engine.py                      │
│ • Widths: 2D Euclidean × cm_per_pixel          │
│ • Lengths: sum of 2D segments (more-visible    │
│   side wins for arm and inseam)                │
│ • Circumferences: ellipse perimeter from       │
│   width + side-depth (or 0.78×width fallback)  │
│ • Derived thigh width = 0.575 × hip width      │
│ • Per-measurement confidence (visibility-based)│
└────────────────┬───────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────┐
│ src/measurement/catalog_mapping.py             │
│ • Map raw measurements → catalog dimension     │
│   keys per category (top vs. bottom)           │
│ • Category-aware fit_relevant_keys dispatch    │
└────────────────┬───────────────────────────────┘
                 ▼
        Sidebar verification panel
       (always-editable number_inputs +
        🟢/🟡/🔴 confidence badges)
                 │
                 ▼
        Load Wardrobe → fit_calculator
                       → prompt_builder
                       → gemini_client.generate_tryon (bytes in-memory)
```

`run_measurement_pipeline` in [`pipeline.py`](src/measurement/pipeline.py) is the single entry point that stitches preprocess → pose → scale → engine and folds layer errors into `result.warnings` instead of raising — the UI always renders something and lets the user fall back to manual entry.

---

## Locked Design Decisions (2026-04-28)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Manual entry kept as permanent fallback (sidebar toggle: *Skip detection — enter manually*) | Escape hatch for low-confidence detections and users without a webcam |
| 2 | After detection, detected values are shown as **always-editable** `st.number_input` widgets | Last-moment human verification before "Load Wardrobe" |
| 3 | **Side photo is in v1** (not deferred) | Real depth >> heuristic depth for circumferences |
| 4 | Depth fallback (no side photo): fixed **0.78 × width** ratio | Average chest/waist depth-to-width ratio for adult populations |
| 5 | Height calibration: nose-to-heel × **1.087** | Adds back the ~8% nose-to-crown segment without segmentation; hair/hat-robust |
| 6 | Catalog migrated **inches → cm** | Aligns with pipeline's natural unit; removed a class of conversion bugs |
| 7 | Garment schema: `category: "top" \| "bottom"` + explicit `fit_relevant_keys` | Calculator and prompt builder dispatch by category, only consider relevant dimensions |
| 8 | Photos handled **in-memory only** (no `output/_person_temp.png` write) | `gemini_client.generate_tryon` accepts bytes for person_image |
| 9 | Lower-body scope (waist, hip, inseam, outseam, thigh) shipped now | Pants/jeans wardrobe expansion is imminent |

See `memory/project_body_measurement_flow.md` for the source-of-truth project memo.

---

## Pose Backend

- **Library:** `mediapipe>=0.10.0` — **Tasks API** (`mp.tasks.vision.PoseLandmarker`).
- **Why Tasks, not Solutions?** Newer MediaPipe wheels (0.10.30+) on Windows ship only the Tasks API; the legacy `mp.solutions.pose` module is gone.
- **Model:** `pose_landmarker_lite.task` (float16). Downloaded once from `storage.googleapis.com/mediapipe-models/...` to `~/.cache/mediapipe/` on first run.
- **Detector caching:** instantiation is heavy, so the Streamlit UI memoizes it with `@st.cache_resource` and passes `pose_solution=` into the pipeline.
- **Mode:** `RunningMode.IMAGE`, `num_poses=1`, all confidence thresholds 0.5.

---

## Landmarks → Measurements (as implemented)

Visibility threshold = **0.5**. Each measurement is computed only if all required landmarks pass.

### Upper body

| Catalog key | Internal field | Computation |
|---|---|---|
| `shoulder` | `shoulder_width_cm` | Euclidean(L_shoulder, R_shoulder) × scale × `shoulder_correction` |
| `chest` (circumference) | `chest_circumference_cm` | Ellipse perimeter from `chest_width_cm` (= 0.92 × shoulder line) and side-depth between shoulders, or `0.78 × width` fallback |
| `arm_length` | `arm_length_cm` | (shoulder→elbow) + (elbow→wrist) on the **higher-visibility** side, × scale × `shoulder_correction` |
| `length` (torso) | `torso_length_cm` | Vertical pixel delta between mid-shoulder and mid-hip × scale |

### Lower body

| Catalog key | Internal field | Computation |
|---|---|---|
| `waist` (circumference) | `waist_circumference_cm` | Ellipse perimeter from `waist_width_cm` (= 0.92 × hip line × `waist_correction`) and side-depth between hips |
| `hip` (circumference) | `hip_circumference_cm` | Ellipse perimeter from hip width and side-depth (same depth as waist) |
| `inseam` | `inseam_cm` | (hip→knee) + (knee→ankle) on the higher-visibility side × scale |
| `outseam` | `outseam_cm` | Vertical pixel delta hip → ankle × scale (used as `length` for bottoms) |
| `thigh` (circumference) | `thigh_circumference_cm` | Width = `0.575 × hip_width`; depth = `0.78 × thigh_width`; ellipse perimeter |

> Width-only attributes (`chest_width_cm`, `waist_width_cm`, `hip_width_cm`, `thigh_width_cm`) remain on the `Measurements` dataclass for completeness, but the catalog mapping uses circumferences for fit comparison.

---

## Scale Calibration Detail

```python
# src/measurement/scale.py
NOSE_TO_CROWN_FACTOR = 1.087
HEIGHT_MIN_CM = 100.0
HEIGHT_MAX_CM = 230.0
```

1. **Validate** height is in `[100, 230]` cm.
2. **Primary** (`method = "nose_to_heel_corrected"`): if `max(left_heel_v, right_heel_v) ≥ 0.5` and `nose_v ≥ 0.3`, use the more-visible heel and compute
   `pixel_height = |heel.y − nose.y| × 1.087`.
3. **Fallback** (`method = "shoulder_anchor"`): if feet aren't visible **and** the user supplied a shoulder anchor, calibrate from shoulder pixels directly.
4. **Hard fail**: feet hidden and no shoulder anchor → `MeasurementError` is raised and surfaced as a UI error.
5. **Regional corrections:**
   - `shoulder_correction = shoulder_cm / measured_shoulder_cm` (only applied when calibrated by feet)
   - `waist_correction = implied_waist_width / measured_hip_width`, where `implied_waist_width` is solved from the user's circumference using Ramanujan #1: `W ≈ 2C / (π·(1+ratio))`

These multipliers are applied per-region inside the engine — they don't rescale the whole image.

---

## Confidence Model

Per-measurement confidences flow through the pipeline as a `dict[str, float]`:

- **Direct measurements** (shoulder, hip width, arm, inseam, torso): mean of the contributing landmarks' visibilities.
- **Width-derived chest/waist**: visibilities × **0.9** penalty (interpolation noise).
- **Derived thigh**: fixed `THIGH_DERIVED_CONFIDENCE = 0.6` (no direct landmark).
- **Circumferences without a side photo**: multiplied by `DERIVED_CIRCUMFERENCE_PENALTY = 0.85` to reflect the fixed-ratio assumption.
- **UI rendering:** thresholds → 🟢 ≥ 0.85, 🟡 ≥ 0.6, 🔴 below.

---

## Catalog Integration

`measurements_to_person_dimensions(m, fit_relevant_keys, category)` in [`catalog_mapping.py`](src/measurement/catalog_mapping.py) projects raw measurements to the keys the garment catalog uses, with `category="top" | "bottom"` deciding the key map:

```
TOP    : length → torso_length_cm
         chest  → chest_circumference_cm
         shoulder → shoulder_width_cm
         arm_length → arm_length_cm

BOTTOM : length → outseam_cm
         waist → waist_circumference_cm
         hip → hip_circumference_cm
         inseam → inseam_cm
         outseam → outseam_cm
         thigh → thigh_circumference_cm
```

`confidence_for_catalog_keys` mirrors this so each catalog key carries the right confidence into the verification panel.

---

## Sidebar UX (app.py)

```
[ Skip detection — enter manually ] (toggle, persists in session_state)
   ↓ ON                            ↓ OFF
manual 4-input form         ┌──────────────────────────────────┐
                            │ 1. Frontal photo  (upload | cam) │
                            │ 2. Side photo (optional)         │
                            │ 3. Reference height (cm | ft/in) │
                            │ 4. Optional correction anchors   │
                            │    • Shoulder width              │
                            │    • Waist circumference         │
                            │ [ Detect Measurements ]          │
                            ├──────────────────────────────────┤
                            │ Verify measurements              │
                            │ • 4 always-editable inputs       │
                            │ • 🟢/🟡/🔴 confidence per row    │
                            │ • warnings shown above           │
                            │ • collapsible "Detected pose"    │
                            │   image with skeleton overlay    │
                            └──────────────────────────────────┘
                            [ Load Wardrobe → ] (primary)
```

After detection completes, stale `verify_*` widget keys are cleared from `st.session_state` so each new run renders fresh values.

---

## Image Preprocessing Notes

- **EXIF transpose** is critical: phone JPEGs commonly carry orientation metadata that viewers honor but PIL/MediaPipe ignore unless explicitly applied. Without `ImageOps.exif_transpose`, MediaPipe sees a sideways body and the pose either fails or maps shoulders to hips.
- **No background removal** in v1. `rembg` was on the original wishlist but isn't required for the current accuracy target — landmark detection is robust to plain backgrounds.
- **Min size 480 px** on either side; smaller images get a clear `MeasurementError`.

---

## Tech Stack (current)

```
Frontend / runtime : Streamlit  (long-running server, NOT serverless)
Pose backend       : mediapipe ≥ 0.10  (Tasks API, lite model)
Image I/O          : Pillow + OpenCV
Numerics           : NumPy
Try-on generation  : google-genai (Gemini Nano Banana)
Prompt synthesis   : groq (optional toggle in sidebar)
Config             : python-dotenv
```

`requirements.txt` (live):

```txt
streamlit>=1.36
google-genai>=0.3.0
Pillow>=10.0
python-dotenv>=1.0
pandas>=2.0
groq
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## Deployment Notes

- **Vercel is incompatible.** Streamlit needs a long-running server with WebSocket session state; Vercel's Python runtime is serverless-only and looks for an `app` ASGI/WSGI export. Build fails with `No python entrypoint found`.
- **Recommended targets:** Streamlit Community Cloud, Hugging Face Spaces, Render, Railway, or Fly.io. Run command:
  ```
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

---

## Known Limitations

- **2D-only.** Limb foreshortening from off-axis stance reads as shorter measurements. The pipeline assumes the subject faces the camera squarely.
- **Baggy clothing** still inflates widths because pose landmarks anchor to silhouette joints.
- **Single frontal photo** can't produce true circumference — without a side photo, depth defaults to `0.78 × width` (chest/waist/hip) or `0.78 × thigh_width` (thigh).
- **Pose detection fails** in poor lighting or with body partially out of frame; the pipeline raises `MeasurementError` and the UI prompts a retake.
- **Tall/short outliers** may exceed the `[100, 230] cm` validator — adjust `HEIGHT_MIN_CM`/`HEIGHT_MAX_CM` in [`scale.py`](src/measurement/scale.py) if needed.
- **Streamlit deprecation noise:** seven `use_container_width=True` call sites in [`app.py`](app.py) will need to migrate to `width="stretch"` before Streamlit removes the old kwarg.
- **MediaPipe console spam:** native code emits `portable_clearcut_uploader.cc` errors when its telemetry endpoint rate-limits. Functionally harmless. Silence with `GLOG_minloglevel=3` in the environment.

---

## Roadmap

1. **MVP (shipped):** frontal + height → all width-based measurements + ellipse circumferences with fixed-ratio depth.
2. **Side-photo depth (shipped):** optional second photo for accurate chest/waist/hip circumferences.
3. **Lower-body wardrobe:** pants/jeans entries in `cloth/garments.json` using the bottom key map (in progress).
4. **Calibration learning:** collect user edits from the verification panel as ground truth; fit a per-region scale correction model.
5. **Optional 3D upgrade:** swap `pose_landmarker_lite` for the full model and start using `z` for limb foreshortening correction. (Not on the immediate roadmap.)

---

*Last updated: 2026-04-28*
