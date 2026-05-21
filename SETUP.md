# Setup

Local-development setup for **Fit-Aware Virtual Try-On**. The app is a Streamlit frontend backed by Python modules in `src/` — no external services to run locally, only API keys to configure.

---

## Prerequisites

| Requirement | Minimum | Notes |
| --- | --- | --- |
| Python | 3.10 | `mediapipe` wheels target 3.10–3.12 on Windows |
| Git | any | for cloning |
| Disk | ~1 GB | venv + MediaPipe model cache + sample generations |
| Network | yes | first run downloads the MediaPipe Pose Landmarker model |
| OS | Windows / macOS / Linux | tested primarily on Windows 11 |

API keys (free tiers available for both):

| Key | Where to get it | Required for |
| --- | --- | --- |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey | Try-on image generation (mandatory) |
| `GROQ_API_KEY` | https://console.groq.com/keys | LLM-written prompt (optional — rule-based fallback works without it) |

---

## 1. Clone the repository

```powershell
git clone <repo-url> Fashion_Image
cd Fashion_Image
```

---

## 2. Create and activate a virtual environment

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Verify:

```powershell
python --version          # 3.10.x or newer
which python              # should point inside ./venv
```

---

## 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

What gets installed (see [requirements.txt](requirements.txt)):

- `streamlit>=1.36` — UI framework
- `google-genai>=0.3.0` — Gemini 2.5 Flash Image client
- `groq` — Llama 3.3 70B prompt-writing client
- `mediapipe>=0.10.0` — Pose Landmarker (Tasks API)
- `opencv-python>=4.8.0` — pose annotation overlay
- `Pillow>=10.0`, `numpy>=1.24.0`, `pandas>=2.0`, `python-dotenv>=1.0`

**Windows note:** if `mediapipe` fails to install, you are likely on Python 3.13+ which has no wheel yet. Install Python 3.12 from python.org and recreate the venv.

---

## 4. Configure API keys

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

Legacy variable name `api_key` is also accepted for Gemini (see [src/gemini_client.py:21](src/gemini_client.py#L21)).

The `.env` file is gitignored — never commit it.

**Quick verification (optional):**

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('GEMINI:', bool(os.getenv('GEMINI_API_KEY') or os.getenv('api_key'))); print('GROQ:  ', bool(os.getenv('GROQ_API_KEY')))"
```

Both should print `True`.

---

## 5. Run the app

```powershell
streamlit run app.py
```

Streamlit prints a local URL (default `http://localhost:8501`) and opens it in the browser. Stop the app with `Ctrl+C`.

On the **first measurement detection** the app downloads the MediaPipe Pose Landmarker model (~6 MB) to `~/.cache/mediapipe/pose_landmarker_lite.task`. Subsequent runs are offline-fast.

---

## 6. Smoke test the flow

1. In the sidebar, leave **Skip detection** unchecked.
2. Upload a frontal full-body photo (or use the camera input).
3. Enter your height in cm.
4. Click **Detect Measurements** — pose landmarks should overlay in the "Detected pose" expander, and values should populate the verification panel with confidence badges.
5. Click **Load Wardrobe** — the garment grid should appear.
6. Click **Try Now** on any garment. The "tailoring…" animation runs for ~30–60 seconds; the result appears below with the original, the garment, and the try-on side by side.

If any step fails, see **Troubleshooting** below.

---

## 7. Run individual modules (optional)

The fit calculator and prompt builder are runnable standalone for quick verification:

```powershell
python -m src.fit_calculator        # prints a sample fit dict for garment G2
python -m src.prompt_builder        # prints the rule-based prompt for a sample fit
```

The Gemini client can also be exercised directly (writes `output.png`):

```powershell
python -m src.gemini_client
```

This uses `person.jpg` + `cloth/00041_00.jpg` baked into the `__main__` block.

---

## Troubleshooting

### "No API key found. Set `api_key` (or `GEMINI_API_KEY`) in your .env file."
Your `.env` is missing or the venv was activated before the key was added. Re-create the file and restart Streamlit (Ctrl+C, then `streamlit run app.py`).

### "Could not download MediaPipe pose model from …"
First-run model fetch failed. Check your network/proxy, delete any partial file at `~/.cache/mediapipe/pose_landmarker_lite.task.part`, and retry detection.

### "No body detected in the photo."
The frontal photo doesn't show a full body, or the subject is heavily cropped/occluded. Use a plain background, even lighting, and frame head-to-feet.

### "Cannot calibrate scale: feet are not visible…"
Either retake with feet in frame, or enter a **Shoulder width** in the optional anchors expander — the calibrator will fall back to the shoulder anchor automatically.

### Detection succeeds but values are red (low confidence)
MediaPipe couldn't see those landmarks clearly. The verification panel lets you edit any value before loading the wardrobe — this is by design.

### Gemini returns "no candidates" or "no content"
Safety/recitation filter triggered. Try a less extreme person/garment combination, or flip the **Use Groq to write the prompt** toggle off to use the more conservative rule-based prompt.

### `mediapipe` import fails after install on Windows
The `mp.solutions.pose` legacy API was removed in newer wheels. This project uses the Tasks API (`mp.tasks.vision.PoseLandmarker`) which works on `mediapipe>=0.10.0`. If imports still fail, ensure you are on Python 3.10–3.12.

### Streamlit shows "stale" measurements after a re-detection
The app clears verification widget state on each successful detection ([app.py:285](app.py#L285)). If you see stale values, use the sidebar menu → **Clear cache** and re-detect.

---

## Adding a garment to the catalog

1. Drop the product image into `cloth/` (or a subfolder).
2. Append an entry to [cloth/garments.json](cloth/garments.json):

   ```json
   {
     "id": "G99",
     "name": "Your Garment",
     "type": "T-Shirt",
     "size": "M",
     "fit_style": "Regular",
     "category": "top",
     "fit_relevant_keys": ["length", "chest", "shoulder", "arm_length"],
     "path": "your_image.jpg",
     "dimensions": { "length": 64, "chest": 91, "shoulder": 38, "arm_length": 20 }
   }
   ```

3. Restart Streamlit (the catalog is cached via `@st.cache_data`).

For lower-body items, set `category: "bottom"` and use keys from `{length, waist, hip, inseam, outseam, thigh}`; the measurement→catalog key mapping switches automatically ([src/measurement/catalog_mapping.py:14-28](src/measurement/catalog_mapping.py#L14-L28)).

---

## Where things live

| You want to… | Edit / inspect |
| --- | --- |
| Change the Streamlit layout or copy | [app.py](app.py) |
| Tweak the fit deltas formula | [src/fit_calculator.py](src/fit_calculator.py) |
| Adjust severity bands or rewrite the meta-prompt | [src/prompt_builder.py](src/prompt_builder.py) |
| Swap the Gemini model or compression settings | [src/gemini_client.py](src/gemini_client.py) |
| Swap the Groq model | [src/groq_client.py:19](src/groq_client.py#L19) |
| Change scale calibration math | [src/measurement/scale.py](src/measurement/scale.py) |
| Change circumference / width math | [src/measurement/engine.py](src/measurement/engine.py) |
| Add a new measurement key | [src/measurement/types.py](src/measurement/types.py) + [engine.py](src/measurement/engine.py) + [catalog_mapping.py](src/measurement/catalog_mapping.py) |

Generated try-on images are saved to `output/<garment_id>_<timestamp>.png` and are gitignored. Clear the folder when you no longer need them.
