# Hybrid Body Measurement Approach
> Estimate body measurements from a photo using pose estimation + user-provided reference inputs.

---

## Overview

The hybrid approach combines **open-source pose estimation** (MediaPipe Pose) with **user-supplied reference measurements** (height, and optionally shoulder/waist) to produce accurate body measurements without requiring expensive 3D scanning hardware or paid APIs.

The core idea:
- Pose estimation gives us **body landmark positions in pixels**
- User-provided height gives us a **pixel-to-cm scale factor**
- Optional shoulder/waist inputs act as **correction anchors** to improve accuracy

The measurement set covers **both upper and lower body** so the same pipeline can fit tops (shirts, jackets) and bottoms (jeans, pants) as the wardrobe expands.

---

## Requirements

### 1. Functional Requirements

| # | Requirement | Priority |
|---|-------------|----------|
| F1 | Accept 1 frontal full-body photo (JPEG/PNG) | Must have |
| F2 | Accept 1 optional side-profile photo for depth-based measurements | Nice to have |
| F3 | Accept user height as mandatory reference input (cm or ft/in) | Must have |
| F4 | Accept optional shoulder width and waist circumference as correction inputs | Nice to have |
| F5 | Detect and return 33 body pose landmarks from the image | Must have |
| F6 | Calculate pixel-to-cm scale ratio using the height reference | Must have |
| F7 | Output the following measurements: | Must have |
|   | **Upper body** | |
|   | — Shoulder width | |
|   | — Chest width (frontal estimate) | |
|   | — Arm length (shoulder to wrist) | |
|   | — Torso length (shoulder to hip) | |
|   | **Lower body** (for pants / jeans) | |
|   | — Waist width | |
|   | — Hip width | |
|   | — Inseam (crotch to ankle) | |
|   | — Outseam (hip to ankle, outer leg) | |
|   | — Thigh width (frontal estimate) | |
| F8 | Return a confidence score (0–1) per measurement | Nice to have |
| F9 | Flag if body is not fully visible or pose detection failed | Must have |

---

### 2. Technical Requirements

| # | Requirement | Detail |
|---|-------------|--------|
| T1 | **Pose estimation library** | MediaPipe Pose (Google) — free, 33 landmarks, runs locally |
| T2 | **Language & runtime** | Python 3.9+ |
| T3 | **Backend framework** | FastAPI (recommended) or Flask |
| T4 | **Image preprocessing** | Resize to standard height, convert to RGB, optional background removal via `rembg` |
| T5 | **Segmentation (optional)** | `rembg` or `Segment Anything` for cleaner silhouette |
| T6 | **Scale calibration module** | Maps pixel distance (head-to-toe landmarks) → known height in cm |
| T7 | **Measurement engine** | Euclidean distance between landmark pairs, multiplied by scale factor |
| T8 | **Correction module** | Applies user-supplied shoulder/waist to adjust scale or offset per region |
| T9 | **API interface** | REST endpoint: `POST /measure` accepting multipart image + JSON inputs |
| T10 | **Frontend** | Simple HTML form or React component — photo upload + input fields |
| T11 | **Output format** | JSON response with measurement names, values (cm), and confidence scores |

---

### 3. Data / Input Requirements

#### Photo Requirements
- **Angle**: Straight-on frontal (and optionally a 90° side view)
- **Lighting**: Even, no harsh shadows — natural light preferred
- **Clothing**: Tight-fitting or form-fitting (baggy clothes reduce accuracy)
- **Background**: Plain or contrasting with the person
- **Pose**: Standing upright, arms slightly away from body, feet shoulder-width apart
- **Full body visible**: Head to toe must be in frame
- **Resolution**: Minimum 480×640px, recommended 720p or higher

#### Reference Inputs
| Input | Type | Required | Notes |
|-------|------|----------|-------|
| Height | number (cm or ft/in) | **Yes** | Primary scale anchor |
| Shoulder width | number (cm) | No | Correction anchor for upper body |
| Waist circumference | number (cm) | No | Correction anchor for mid-body |

---

### 4. Non-Functional Requirements

| # | Requirement | Target |
|---|-------------|--------|
| NF1 | Processing time | < 5 seconds per image |
| NF2 | Accuracy | ±2–3 cm margin of error for major measurements |
| NF3 | Offline capability | Core ML pipeline must run without internet access |
| NF4 | Privacy | Images must not be stored server-side without explicit user consent |
| NF5 | GDPR compliance | Provide image deletion option post-processing |
| NF6 | Scalability | API should handle concurrent requests via async workers |
| NF7 | Platform support | Works on Linux, macOS, Windows; deployable via Docker |

---

## Architecture

```
User
 │
 ├─── Frontal photo (+ optional side photo)
 ├─── Height (mandatory)
 └─── Shoulder / Waist (optional)
         │
         ▼
┌─────────────────────────────────────────┐
│           Preprocessing Layer           │
│  • Resize & normalize image             │
│  • Background removal (optional)        │
│  • Validate full-body visibility        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        Pose Estimation (MediaPipe)      │
│  • Detect 33 body landmarks             │
│  • Extract (x, y) pixel coordinates     │
│  • Compute visibility scores            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Scale Calibration Module        │
│  • Pixel height = distance(nose → heel) │
│  • Scale = real_height_cm / pixel_height│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Measurement Engine              │
│  • Compute Euclidean distances between  │
│    relevant landmark pairs              │
│  • Multiply by scale factor             │
│  • Apply correction from optional inputs│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            Output Layer                 │
│  • JSON with measurements + confidence  │
│  • Annotated image (optional)           │
└─────────────────────────────────────────┘
```

---

## Landmark Pairs Used for Each Measurement

| Measurement | Landmarks Used | Notes |
|-------------|---------------|-------|
| Shoulder width | Left shoulder ↔ Right shoulder | Direct horizontal distance |
| Chest width | Chest-level horizontal estimate | Interpolated between shoulders |
| Torso length | Mid-shoulder ↔ Mid-hip | Vertical distance |
| Arm length | Shoulder → Elbow → Wrist | Sum of two segments |
| Waist width | Left hip ↔ Right hip (adjusted) | Approx at waist level (above hip line) |
| Hip width | Left hip ↔ Right hip | Direct horizontal distance |
| Inseam | Mid-hip → Knee → Ankle | Crotch line to ankle, sum of two segments |
| Outseam | Hip → Ankle (outer leg) | Hip landmark to ankle landmark vertically |
| Thigh width | Hip-width × body-ratio factor | No direct landmark — estimated (~0.55–0.6 × hip width) |

> For circumference (chest, waist, hip), a **depth factor** is applied using body-type ratios or side-photo depth measurements. Without a side photo, an average depth-to-width ratio (typically 0.7–0.8) is used as an approximation.

---

## Tech Stack Summary

```
Backend:      Python 3.9+  |  FastAPI  |  Uvicorn
ML / Vision:  MediaPipe Pose  |  OpenCV  |  NumPy
Segmentation: rembg (optional)
Frontend:     React (or plain HTML)
Deployment:   Docker  |  Linux server or cloud VM
```

---

## Python Dependencies

```txt
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
pillow>=10.0.0
rembg>=2.0.50       # optional: background removal
python-multipart    # for file upload in FastAPI
```

---

## Sample API Contract

### Endpoint
```
POST /measure
Content-Type: multipart/form-data
```

### Request
```
frontal_image   : file      (required)
side_image      : file      (optional)
height_cm       : float     (required)
shoulder_cm     : float     (optional)
waist_cm        : float     (optional)
```

### Response
```json
{
  "status": "success",
  "measurements": {
    "shoulder_width_cm": 42.3,
    "chest_width_cm": 38.1,
    "arm_length_cm": 61.2,
    "torso_length_cm": 52.4,
    "waist_width_cm": 33.5,
    "hip_width_cm": 39.8,
    "inseam_cm": 79.1,
    "outseam_cm": 102.7,
    "thigh_width_cm": 22.4
  },
  "confidence": {
    "shoulder_width": 0.94,
    "waist_width": 0.87,
    "inseam": 0.91
  },
  "warnings": []
}
```

---

## Known Limitations

- **Baggy clothing** reduces measurement accuracy significantly
- **Single frontal photo** cannot produce true circumference — depth estimation is approximate
- **Pose detection fails** in poor lighting or when body is partially out of frame
- **Accuracy degrades** for very tall/short individuals if landmark visibility is low

---

## Accuracy Improvement Roadmap

1. **Phase 1 (MVP):** Frontal photo + height → width-based measurements
2. **Phase 2:** Add side photo → depth-aware circumference estimation
3. **Phase 3:** Collect user-verified measurements to fine-tune scale correction model
4. **Phase 4:** Train a custom regression model on top of landmark features for higher accuracy

---

*Last updated: April 2026*
