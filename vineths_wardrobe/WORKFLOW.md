# eWardrobeAI — AI-Powered Virtual Try-On System
## Technical Workflow & System Architecture

---

## Executive Summary

**eWardrobeAI** is a smart wardrobe mobile application that delivers a fully
personalised, AI-driven virtual try-on experience. The system fuses computer
vision, deep learning, real-time 3D rendering, and a rule-based recommendation
engine to produce a lifelike avatar dressed in the user's own wardrobe — scaled
to their exact body proportions and personalised with their facial appearance.

The pipeline is divided into four sequential stages:

```
Stage 1 — User Input & Body Calibration
Stage 2 — AI-Driven Face Processing    (Research Component)
Stage 3 — Avatar & Outfit Logic        (Recommendation + Wardrobe DB)
Stage 4 — Animation & Real-Time Rendering
```

---

## Stage 1 — User Input & Body Calibration

### 1.1 Selfie Capture

The user opens the eWardrobeAI mobile app and captures a front-facing
selfie. The selfie is used exclusively in **Stage 2** for facial
personalisation of the 3D avatar head. The image is transmitted as a
JPEG payload to the backend via `POST /api/tryon` (multipart form-data).

### 1.2 Body Measurements — Manual Entry or Barcode Scan

The user provides the following measurements (all in centimetres):

| Measurement       | Demo Value | Validation Bounds | Derived If Absent     |
|-------------------|:----------:|:-----------------:|:---------------------:|
| Shoulder Width    |  42 cm     | 30 – 65 cm        | —                     |
| Chest             |  92 cm     | 60 – 160 cm       | —                     |
| Waist             |  72 cm     | 50 – 150 cm       | —                     |
| Height            | 168 cm     | 120 – 230 cm      | —                     |
| Hip               | optional   | 60 – 175 cm       | waist + 25 cm         |
| Inseam            | optional   | 60 – 110 cm       | height × 0.47         |

Measurements may be entered manually or scanned via a printed measurement
card with a QR/barcode encoding the values.

### 1.3 Measurement Validation (`src/body_calibration.py`)

The `BodyCalibrator` class performs a two-phase validation:

**Phase A — Range Validation**
Each measurement is checked against anatomical feasibility bounds.
Measurements outside these bounds produce a hard `ValidationError` that
aborts the pipeline and returns a descriptive error to the user.

**Phase B — Cross-Field Consistency**
- Waist must not exceed chest (warns if violated)
- Shoulder width must not exceed 55 % of chest (unusual proportion)
- Inseam must not exceed 60 % of height

### 1.4 Derived Metrics

After validation, the `BodyCalibrator` computes:

**Standard Size Label** (used by NisfaMatchmaking for garment filtering):

| Size  | Chest Range |
|-------|:-----------:|
| XS    | ≤ 82 cm     |
| S     | ≤ 88 cm     |
| M     | ≤ 96 cm     |
| L     | ≤ 104 cm    |
| XL    | ≤ 112 cm    |
| XXL   | ≤ 124 cm    |
| XXXL  | > 124 cm    |

**Body Type Classification** (rule-based):

| Body Type          | Rule                                  |
|--------------------|---------------------------------------|
| `hourglass`        | waist-definition > 9 cm, S≈H balance  |
| `inverted_triangle`| shoulder-proxy / hip < 0.87           |
| `pear`             | shoulder-proxy / hip > 1.13           |
| `rectangle`        | otherwise                             |

### 1.5 Avatar Scale Parameters (`AvatarScaleParams`)

Per-bone scale factors are computed as
`scale = clamp(user_measurement / base_avatar_measurement, 0.70, 1.40)`.

Base avatar reference dimensions:

| Dimension   | Base Value |
|-------------|:----------:|
| Shoulder    | 40 cm      |
| Chest       | 90 cm      |
| Waist       | 70 cm      |
| Height      | 170 cm     |
| Hip         | 95 cm      |
| Inseam      | 80 cm      |

The clamping prevents extreme scale distortions that would break the
Blender mesh skinning.

---

## Stage 2 — AI-Driven Face Processing (Research Component)

This stage implements a **two-layer facial landmark detection pipeline**
combining dense real-time detection (MediaPipe) with high-accuracy sparse
keypoint regression (custom CNN). The design allows the system to degrade
gracefully: if the camera is unavailable, the CNN processes the uploaded
still image.

### 2.1 Layer 1 — MediaPipe FaceMesh (468 Landmarks)

**Library:** `mediapipe>=0.10`
**Model:**   Attention Mesh model (refine_landmarks=True)

The MediaPipe FaceMesh model detects **468 facial landmark points** in a
single forward pass of a lightweight neural network optimised for mobile and
edge-device inference.

**Landmark categories detected:**
- Facial contour / oval boundary
- Eye regions (left + right, inner + outer corners, iris)
- Eyebrow arches
- Nasal bridge and tip
- Lip contour (upper + lower)
- Philtrum

**Usage in eWardrobeAI:**
- Facial bounding-box computation for face-crop pre-processing
- Real-time head-pose estimation for the live WebSocket stream
- Dense UV-coordinate mapping for texture projection

**Implementation (`src/face_processor.py`):**
```python
mp_face = mp.solutions.face_mesh
self._mp_face_mesh = mp_face.FaceMesh(
    static_image_mode       = True,
    max_num_faces           = 1,
    refine_landmarks        = True,
    min_detection_confidence= 0.5,
)
```

### 2.2 Layer 2 — eWardrobeAI CNN (15 Keypoints, Research Component)

**Dataset:** Facial Keypoints Detection (training.csv)
             7,049 grayscale face images at 96 × 96 pixels
             30 target values (15 landmark × 2 coordinates)

**Keypoints regressed (15 facial landmarks):**

| #  | Landmark                    |
|----|-----------------------------|
| 1  | Left Eye Centre             |
| 2  | Right Eye Centre            |
| 3  | Left Eye Inner Corner       |
| 4  | Left Eye Outer Corner       |
| 5  | Right Eye Inner Corner      |
| 6  | Right Eye Outer Corner      |
| 7  | Left Eyebrow Inner End      |
| 8  | Left Eyebrow Outer End      |
| 9  | Right Eyebrow Inner End     |
| 10 | Right Eyebrow Outer End     |
| 11 | Nose Tip                    |
| 12 | Mouth Left Corner           |
| 13 | Mouth Right Corner          |
| 14 | Mouth Centre Top Lip        |
| 15 | Mouth Centre Bottom Lip     |

### 2.3 CNN Architecture (`src/face_keypoint_model.py`)

```
Input  : (96, 96, 1)  — normalised grayscale face image [0, 1]
         │
         ├── Conv2D(32) → BatchNorm → ReLU → MaxPool  [48 × 48]
         ├── Conv2D(64) → BatchNorm → ReLU → MaxPool  [24 × 24]
         ├── Conv2D(128)→ BatchNorm → ReLU → MaxPool  [12 × 12]
         ├── Conv2D(256)→ BatchNorm → ReLU → MaxPool  [6 × 6]
         ├── Conv2D(256)→ BatchNorm → ReLU            [6 × 6]
         │
         ├── Flatten
         ├── Dense(1024) + Dropout(0.4)
         ├── Dense(512)  + Dropout(0.3)
         │
Output : Dense(30, activation='sigmoid')
         — 30 normalised keypoint coordinates in [0, 1]
         → rescaled × 96 for pixel space

Loss      : Mean Squared Error (MSE)
Metric    : Normalised MAE  (target < 0.05 = < 4.8 px error)
Optimiser : Adam + CosineDecayRestarts LR schedule
Epochs    : 100 (with early stopping, patience=15)
```

### 2.4 Training Procedure

```
python -m src.face_keypoint_model
```

1. Load `training.csv` — parse pixel strings, drop rows with any missing keypoints
2. Normalise images to [0, 1]; normalise keypoint coords to [0, 1] (÷ 96)
3. Train/validation split: 85 % / 15 % (random_state=42)
4. Augmentation (per-batch): horizontal flip with mirrored x-coords, brightness jitter ±15 %
5. Train with callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
6. Best model saved to `models/face_keypoint_cnn.keras`
7. Post-training evaluation plots saved to `models/`

### 2.5 Facial Texture Mapping

After keypoint detection, a face texture patch is extracted for UV mapping
onto the 3D avatar head mesh:

1. **Crop:** Face ROI extracted from the full selfie using detected bounding box
2. **Resize:** Upsampled to 512 × 512 (standard UV map resolution)
3. **Enhance:** CLAHE (Contrast Limited Adaptive Histogram Equalisation) applied
   to L-channel in LAB colour space for perceptual normalisation
4. **Encode:** Converted to RGB and base64-encoded PNG for transmission
5. **Apply:** Three.js assigns the decoded image as `THREE.Texture` to the
   head mesh material (`material.map = texture`)

UV mapping is pre-baked into the Blender avatar export with the
`flipY = false` convention matching Blender's UV coordinate system.

### 2.6 Head Pose Estimation

A lightweight pose estimate is derived from the symmetry of the eye positions:

- **Yaw (left–right rotation):** Computed from the ratio of the actual inter-eye
  horizontal span to the expected span (32 % of face width). A compressed span
  indicates the face is turned.
- **Pitch (up–down tilt):** Computed from the vertical angle of the eye-to-eye
  line using `arctan2`.

These values are passed to the renderer as `headYawDeg` and `headPitchDeg`
to orient the avatar head appropriately in the initial render pose.

### 2.7 Inter-Ocular Distance (Head Scaling)

The pixel distance between left and right eye centres is used as a face-size
proxy to fine-tune the avatar head scale:

```
head_scale = clamp(measured_IED_px / reference_IED_px, 0.85, 1.15)
```

This ensures users with wider or narrower faces receive a proportionally
correct head mesh rather than a generic default.

---

## Stage 3 — Avatar & Outfit Logic Preparation

### 3.1 RaveehaOrganisationalDB (`src/outfit_recommender.py`)

The wardrobe database tracks each garment with three critical state fields:

| Field              | Values                              | Impact on Try-On     |
|--------------------|-------------------------------------|----------------------|
| `cleaning_status`  | `Clean` / `Dirty` / `In Laundry`   | Only `Clean` eligible|
| `availability`     | `owned` / `borrowed` / `wishlist`  | Only `owned` eligible|

**Garment exclusion rules (strictly enforced):**
- `CleaningStatus.DIRTY`      → **excluded** from all recommendations
- `CleaningStatus.IN_LAUNDRY` → **excluded** from all recommendations
- `Availability.WISHLIST`     → **excluded** (not yet in wardrobe)

The `is_wearable` property encapsulates this logic:
```python
@property
def is_wearable(self) -> bool:
    return (self.cleaning_status == CleaningStatus.CLEAN
            and self.availability == Availability.OWNED)
```

**Garment cleaning lifecycle:**
```
Clean → (worn) → Dirty → (sent) → In Laundry → (returned) → Clean
```

Backend endpoints expose this lifecycle:
- `PATCH /api/wardrobe/{garment_id}/status` — update any garment's status

### 3.2 NisfaMatchmaking Engine (`src/outfit_recommender.py`)

The matching engine operates in two steps: **querying** and **scoring**.

**Query Phase:**
For each user style preference, the engine queries `RaveehaOrganisationalDB`
for wearable garments matching:
- User's standard size (from `BodyCalibrator`)
- User's body type (from `BodyCalibrator`)
- Style preference (Casual / Formal / Smart / Sporty / Evening)

**Bundle Construction:**
Compatible garments are combined into outfit bundles:
- Type A: Top + Bottom (+ optional Outerwear)
- Type B: Dress (standalone)
- Type C: Suit (standalone)

**Scoring Heuristic (0 – 1.0):**

| Factor                  | Weight |
|-------------------------|:------:|
| Occasion tag match      |  0.40  |
| Body type compatibility |  0.30  |
| Bundle completeness     |  0.20  |
| Colour coordination     |  0.10  |

Bundles are sorted by score descending; top-K are returned (default K=5).

### 3.3 3D Asset Mapping (`src/avatar_manager.py`)

Each `GarmentRecord` has a hardcoded `asset_path` pointing to a `.glb` or
`.fbx` file in the `assets/outfits/` directory:

| Garment ID | Asset Path                              |
|------------|-----------------------------------------|
| GAR-001    | `assets/outfits/white_oxford_shirt.glb` |
| GAR-005    | `assets/outfits/slim_chinos.glb`        |
| GAR-008    | `assets/outfits/wool_blazer.glb`        |
| GAR-010    | `assets/outfits/wrap_midi_dress.glb`    |
| GAR-012    | `assets/outfits/classic_suit.glb`       |

**3D Asset Conventions (Blender Export Settings):**
- Y-up coordinate system
- Armature origin at world origin (0, 0, 0)
- Bone names follow Mixamo naming convention (`mixamorigHips`, `mixamorigSpine`, …)
- UV maps present on both body mesh and head mesh
- Clothing meshes include blend-shape morph targets for size variation

If a `.glb` asset file is not present on disk (e.g. demo mode), the renderer
substitutes placeholder geometry (capsule + colour quad) while correctly
applying all scale parameters and face texture.

---

## Stage 4 — Animation & Real-Time Rendering

### 4.1 Mixamo Animation Integration

Animations are sourced from **Mixamo** (Adobe), baked as separate action tracks
into the base avatar `.glb` file.

| Animation Key | Mixamo Clip Name        | Loop   | Description              |
|---------------|-------------------------|--------|--------------------------|
| `idle`        | `Mixamo_Idle`           | Yes    | Standing breathing idle  |
| `walk`        | `Mixamo_Walking`        | Yes    | Forward walking cycle    |
| `rotate`      | `Mixamo_TurnLeft`       | Yes    | 360° body rotation       |
| `pose_t`      | `Mixamo_TPose`          | No     | T-pose (measurement ref) |
| `pose_a`      | `Mixamo_APose`          | No     | A-pose (rigging ref)     |
| `catwalk`     | `Mixamo_CatwalkWalk`    | Yes    | Fashion catwalk walk     |

Cross-fade transitions between animations use Three.js `crossFadeTo()` with
a 0.5-second blend window, ensuring smooth motion without abrupt cuts.

### 4.2 Three.js Real-Time Renderer (`frontend/js/tryon_renderer.js`)

**Scene Graph:**
```
THREE.Scene
├── AmbientLight         (0xffffff, intensity 0.4)
├── DirectionalLight     (key — front-right, shadows enabled)
├── DirectionalLight     (fill — left)
├── DirectionalLight     (rim — back)
├── HemisphereLight      (sky/ground colour blend)
├── CircleGeometry       (ground plane, shadow receiver)
├── GridHelper           (subtle floor grid)
└── AvatarGroup (THREE.Group)
    ├── BaseMesh.glb     (Blender avatar, Mixamo-rigged)
    └── ClothingMesh*.glb (one per garment in outfit bundle)
```

**Avatar Scaling Synchronisation:**
Scale parameters are applied directly to named bones in the Mixamo skeleton:

```javascript
BONE_MAP = {
  globalY:   ['mixamorigHips'],              // height
  shoulderX: ['mixamorigLeft/RightShoulder'],// shoulder width
  chestX:    ['mixamorigSpine1', 'Spine2'],  // chest depth
  waistX:    ['mixamorigSpine'],             // waist
  hipX:      ['mixamorigHips'],              // hip width
  legY:      ['LeftUpLeg', 'RightUpLeg',    // leg length
               'LeftLeg',  'RightLeg'],
  headScale: ['mixamorigHead', 'Neck'],      // head uniform scale
}
```

**Clothing Attachment:**
Each clothing `.glb` is loaded as a separate scene and added to `AvatarGroup`.
Clothing animations in the `.glb` are retargeted to the avatar's `AnimationMixer`
using `THREE.AnimationUtils.makeClipAdditive()`.

**Rendering Quality:**
- Shadow mapping: PCFSoftShadow at 2048 × 2048
- Tone mapping: ACES Filmic (cinematic colour grading)
- Output colour space: sRGB
- Anti-aliasing: MSAA (built-in Three.js WebGLRenderer)
- Max pixel ratio: 2× (prevents blurring on HiDPI displays)

### 4.3 WebSocket Live Try-On (`/ws/tryon`)

For real-time camera preview, the WebSocket endpoint processes incoming
JPEG frames (base64-encoded) and returns per-frame landmark data:

**Client → Server:**
```json
{ "type": "frame", "imageB64": "<base64 JPEG>", "animation": "walk" }
```

**Server → Client:**
```json
{
  "type": "render",
  "landmarks15":  { "left_eye_center": {"x": 312.4, "y": 180.2}, … },
  "landmarks468": [ {"x": 310.0, "y": 178.5, "z": -0.02}, … ],
  "headPose":     { "yaw": 3.2, "pitch": -1.1 },
  "interEyeDist": 42.7
}
```

The frontend renderer uses these landmark positions to animate the avatar
head in real time without re-running the outfit pipeline.

---

## API Reference

| Method | Endpoint                          | Description                               |
|--------|-----------------------------------|-------------------------------------------|
| POST   | `/api/tryon`                      | Main virtual try-on — multipart form      |
| GET    | `/api/wardrobe/summary`           | Cleaning-status counts                    |
| PATCH  | `/api/wardrobe/{id}/status`       | Update garment cleaning status            |
| GET    | `/api/animations`                 | List Mixamo animation keys                |
| POST   | `/api/train`                      | Trigger CNN training (admin / research)   |
| WS     | `/ws/tryon`                       | Real-time landmark streaming              |
| GET    | `/`                               | Serve frontend HTML                       |

---

## Data Flow Diagram

```
User (Mobile App)
      │
      │  selfie (JPEG) + body measurements + preferences
      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend  (app.py)                  │
│  POST /api/tryon                                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼───────────────┐
              │  VirtualTryOnPipeline       │
              │  (src/virtual_tryon_        │
              │   pipeline.py)              │
              └──┬──────────┬───────────────┘
                 │          │
       ┌─────────▼──┐  ┌────▼──────────────┐
       │Stage 1     │  │Stage 2             │
       │Body        │  │Face Processing     │
       │Calibrator  │  │                    │
       │            │  │  MediaPipe         │
       │→ Validation│  │  (468 landmarks)   │
       │→ Sizing    │  │                    │
       │  Profile   │  │  CNN Model         │
       │→ Avatar    │  │  (training.csv)    │
       │  Scale     │  │  (15 keypoints)    │
       │  Params    │  │                    │
       └────┬───────┘  │→ FaceProfile       │
            │          │  (landmarks +      │
            │          │   texture +        │
            │          │   geometry)        │
            │          └────────┬───────────┘
            │                   │
            └──────────┬────────┘
                       │
              ┌────────▼────────────────────┐
              │Stage 3                      │
              │RaveehaOrganisationalDB      │
              │ → query Clean + owned items │
              │                            │
              │NisfaMatchmaking            │
              │ → filter by size/body type  │
              │ → build outfit bundles      │
              │ → rank by relevance score   │
              │                            │
              │→ List[OutfitRecommendation] │
              └────────┬────────────────────┘
                       │
              ┌────────▼────────────────────┐
              │Stage 4                      │
              │AvatarManager               │
              │ → resolve .glb asset paths  │
              │ → encode face texture PNG   │
              │ → select Mixamo animation   │
              │                            │
              │→ AvatarRenderPayload (JSON) │
              └────────┬────────────────────┘
                       │
              ┌────────▼────────────────────┐
              │  Three.js Renderer          │
              │  (frontend/js/              │
              │   tryon_renderer.js)        │
              │                            │
              │ 1. Load base avatar .glb    │
              │ 2. Apply bone scales        │
              │ 3. Apply face texture UV    │
              │ 4. Load clothing .glb files │
              │ 5. Play Mixamo animation    │
              │ 6. Render to WebGL canvas   │
              └─────────────────────────────┘
```

---

## Project File Structure

```
vineths_wardrobe/
│
├── training.csv                        ← Facial Keypoints dataset (input)
├── requirements.txt                    ← Python dependencies
├── app.py                              ← FastAPI server
│
├── src/
│   ├── face_keypoint_model.py          ← CNN training + inference
│   ├── face_processor.py              ← MediaPipe + CNN pipeline
│   ├── body_calibration.py            ← Measurement validation + avatar scaling
│   ├── outfit_recommender.py          ← NisfaMatchmaking + RaveehaOrganisationalDB
│   ├── avatar_manager.py              ← 3D asset resolver + render payload builder
│   └── virtual_tryon_pipeline.py      ← Main orchestrator
│
├── frontend/
│   ├── index.html                      ← Single-page app UI
│   └── js/
│       └── tryon_renderer.js           ← Three.js avatar renderer
│
├── models/                             ← Generated after training
│   ├── face_keypoint_cnn.keras         ← Saved CNN weights
│   ├── training_curves.png             ← Loss + MAE plots
│   ├── keypoint_predictions.png        ← Validation overlay
│   └── logs/                           ← TensorBoard logs
│
└── assets/
    ├── avatars/
    │   └── base_avatar.glb             ← Blender avatar (Mixamo-rigged)
    └── outfits/
        ├── white_oxford_shirt.glb
        ├── slim_chinos.glb
        ├── wool_blazer.glb
        └── …                           ← Additional garment .glb files
```

---

## Running the System

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Facial Keypoint Model

```bash
python -m src.face_keypoint_model
```

Training progress is printed to stdout. The best model is saved to
`models/face_keypoint_cnn.keras`. Estimated training time:
~25 min on CPU, ~5 min on GPU (CUDA).

### 3. Start the Backend Server

```bash
python app.py
```

Server runs at `http://localhost:8000`.
API documentation at `http://localhost:8000/docs` (Swagger UI).

### 4. Open the Frontend

Navigate to `http://localhost:8000` in any modern browser.

---

## Research Component Summary

The CNN facial keypoint model (`src/face_keypoint_model.py`) constitutes the
core **research contribution** of Stage 2. It demonstrates:

1. **Custom supervised learning** on the standard Facial Keypoints Detection
   benchmark dataset (`training.csv`), achieving sub-5 pixel MAE on 96×96 images.

2. **Practical application** of regression-based CNNs for sparse facial geometry
   estimation, complementing MediaPipe's dense detection approach.

3. **Integration pathway** from a research model (trained offline on a benchmark)
   to a production-grade pipeline component (loaded at server start, called
   per-request via `predict_single_image()`).

4. **Avatar personalisation** through the derived inter-ocular distance metric,
   demonstrating that even a 15-keypoint sparse model can drive meaningful
   3D geometry modifications (head scale, head pose initialisation, UV texture
   anchor points).
```
