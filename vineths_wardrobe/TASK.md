# eWardrobeAI — Project Task List

## Project: AI-Powered Virtual Try-On System
**App Name:** eWardrobeAI  
**Stack:** Python · PyTorch · MediaPipe · FastAPI · Three.js · SQLite  
**Dataset:** Facial Keypoints Detection (`training.csv`)

---

## Phase 1 — Environment Setup

- [x] Install Python 3.14
- [x] Install PyTorch (`python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)
- [x] Install remaining dependencies (`python -m pip install pandas scikit-learn matplotlib opencv-python mediapipe fastapi uvicorn python-multipart pydantic tqdm pytest httpx`)
- [x] Verify environment (`python -c "import torch; print(torch.__version__)"`)

---

## Phase 2 — Stage 1: Body Calibration

**File:** `src/body_calibration.py`

- [x] Define `BodyMeasurements` dataclass (shoulder, chest, waist, height, hip, inseam)
- [x] Implement range validation against anatomical bounds
- [x] Implement cross-field consistency checks (waist vs chest, shoulder vs chest)
- [x] Compute standard size label (XS → XXXL) from chest measurement
- [x] Classify body type (hourglass / inverted_triangle / pear / rectangle)
- [x] Compute `AvatarScaleParams` (per-bone scale factors for Blender rig)
- [x] Return `SizingProfile` for NisfaMatchmaking
- [x] Write unit tests for boundary values → `tests/test_body_calibration.py`
- [x] Test invalid measurement rejection (e.g. waist = 200)

---

## Phase 3 — Stage 2: AI Face Processing (Research Component)

### 3A — CNN Model Training

**File:** `src/face_keypoint_model.py`  
**Dataset:** `training.csv` (7,049 samples · 96×96 px · 15 landmarks)

- [x] Parse `training.csv` — pixel strings → numpy arrays
- [x] Drop rows with missing keypoints
- [x] Normalise images to [0, 1] and keypoints to [0, 1]
- [x] Train/validation split (85% / 15%)
- [x] Implement data augmentation (horizontal flip + brightness jitter)
- [x] Build CNN architecture (5 conv blocks → 2 FC layers → Sigmoid output)
- [x] Train with Adam + CosineAnnealingLR + early stopping
- [x] Save best model to `models/face_keypoint_cnn.pth`
- [x] Generate training curve plots (`models/training_curves.png`)
- [x] Generate keypoint prediction overlays (`models/keypoint_predictions.png`)
- [ ] **Run training:** `python -m src.face_keypoint_model`
- [ ] Confirm val MAE < 5 px on validation set
- [ ] Review `models/keypoint_predictions.png` for visual accuracy

### 3B — Face Processor

**File:** `src/face_processor.py`

- [x] Integrate MediaPipe FaceMesh (468 landmarks) with graceful fallback
- [x] Integrate CNN keypoint inference (15 sparse landmarks)
- [x] Face bounding-box detection (MediaPipe → Haar cascade fallback)
- [x] Facial texture extraction (512×512, CLAHE-enhanced, base64 PNG)
- [x] Inter-ocular distance computation (head scale driver)
- [x] Head pose estimation (yaw + pitch from eye symmetry)
- [x] `draw_landmarks()` debug visualisation utility
- [ ] Test on real selfie image (`python scripts/generate_test_assets.py` then upload `assets/test_selfie.jpg`)
- [ ] Confirm face texture is extracted correctly

---

## Phase 4 — Stage 3: Outfit Recommendation

### 4A — RaveehaOrganisationalDB

**File:** `src/outfit_recommender.py`

- [x] Define `GarmentRecord` dataclass with cleaning status + availability
- [x] Seed catalogue with 12 demo garments (tops, bottoms, dresses, suits, outerwear)
- [x] Implement `is_wearable` — excludes Dirty + In Laundry items
- [x] Implement `query_available()` — filter by size, style, category, body type
- [x] Implement `update_cleaning_status()` — Clean / Dirty / In Laundry lifecycle
- [x] Test via `GET /api/wardrobe/items` in Swagger
- [x] Verified GAR-003 (Striped T-Shirt) excluded — seeded as Dirty
- [x] Verified GAR-004 (Graphic Hoodie) excluded — seeded as In Laundry
- [x] Verified GAR-011 (Evening Gown) excluded — seeded as Dirty

### 4B — NisfaMatchmaking Engine

- [x] Query available garments by size + body type + style
- [x] Build outfit bundles: Top+Bottom, Dress, Suit
- [x] Score bundles (occasion 35% + body type 25% + completeness 20% + colour 20%)
- [x] Colour palette harmony scoring using colour theory rules
- [x] Return top-K ranked `OutfitRecommendation` objects
- [x] Tests in `tests/test_wardrobe_db.py`
- [ ] Verify via `POST /api/demo/tryon` in Swagger
- [ ] Verify score values are between 0 and 1 ← automated in Swagger test script

---

## Phase 5 — Stage 4: Avatar & Rendering

### 5A — Avatar Manager

**File:** `src/avatar_manager.py`

- [x] Map `GarmentRecord.asset_path` → `ClothingAsset` with existence check
- [x] Build `AvatarRenderPayload` JSON (scale params + face texture + clothing + animation)
- [x] Assign layer order (base=1, outerwear=2, accessory=3)
- [x] Encode face texture as base64 PNG
- [x] Select Mixamo animation config from key
- [ ] Add real `.glb` avatar file to `assets/avatars/base_avatar.glb`
- [ ] Add clothing `.glb` files to `assets/outfits/`

> **Note:** Renderer degrades gracefully to placeholder geometry when .glb files are absent.

### 5B — Three.js Renderer

**File:** `frontend/js/tryon_renderer.js`

- [x] Scene setup (lights, ground, grid, fog)
- [x] Load base avatar `.glb` via `GLTFLoader`
- [x] Apply per-bone scale transforms from `AvatarScaleParams`
- [x] Apply face texture to head mesh UV map
- [x] Load clothing `.glb` assets and attach to skeleton
- [x] Mixamo animation playback with cross-fade transitions
- [x] Demo fallback geometry (capsule + colour quads) when `.glb` absent
- [x] Responsive canvas resize
- [ ] Test demo mode in browser at `http://localhost:8000`
- [ ] Verify outfit carousel switches between recommendations
- [ ] Verify animation buttons (walk, rotate, catwalk) work

---

## Phase 6 — Backend API

**File:** `app.py`

- [x] `GET  /api/health` — server + pipeline status
- [x] `POST /api/demo/tryon` — full pipeline, no selfie needed (Swagger-friendly)
- [x] `POST /api/tryon` — full pipeline with selfie upload
- [x] `GET  /api/wardrobe/summary` — cleaning status counts
- [x] `GET  /api/wardrobe/items` — full garment list
- [x] `PATCH /api/wardrobe/{id}/status` — update cleaning status
- [x] `GET  /api/wardrobe/history` — full status change history (SQLite)
- [x] `GET  /api/wardrobe/{id}/history` — per-garment history + wear count
- [x] `GET  /api/wardrobe/analytics/most-worn` — top worn garments
- [x] `POST /api/sizing/validate` — Stage 1 only
- [x] `GET  /api/animations` — list Mixamo keys
- [x] `GET  /api/model/status` — check CNN model file
- [x] `POST /api/train` — background training trigger
- [x] `WS   /ws/tryon` — real-time landmark streaming
- [x] Swagger UI at `/docs`
- [x] ReDoc UI at `/redoc`

---

## Phase 7 — Swagger Testing Checklist

Run `python app.py` then open `http://localhost:8000/docs`  
**Or run automated:** `python scripts/run_swagger_tests.py`

| # | Endpoint | Expected Result | Status |
|---|----------|-----------------|--------|
| 1 | `GET /api/health` | `pipelineReady: true` | [ ] |
| 2 | `GET /api/wardrobe/summary` | `{"Clean":8,"Dirty":3,"In Laundry":1}` | [ ] |
| 3 | `GET /api/wardrobe/items` | 12 garments listed | [ ] |
| 4 | `POST /api/sizing/validate` (default values) | size=M, bodyType returned | [ ] |
| 5 | `POST /api/sizing/validate` (waist=200) | 422 validation error | [ ] |
| 6 | `POST /api/demo/tryon` (default values) | recommendations + renderPayload | [ ] |
| 7 | `POST /api/demo/tryon` (occasion=formal) | formal outfit recommended | [ ] |
| 8 | `PATCH /api/wardrobe/GAR-001/status` → Dirty | `isWearable: false` | [ ] |
| 9 | `PATCH /api/wardrobe/GAR-001/status` → Clean | `isWearable: true` | [ ] |
| 10 | `GET /api/model/status` | exists=true (after training) | [ ] |
| 11 | `GET /api/animations` | 6 animation keys listed | [ ] |

---

## Phase 8 — Model Training & Evaluation

Run from project root:

```powershell
python -m src.face_keypoint_model
```

| Checkpoint | Target | Status |
|------------|--------|--------|
| Training starts without error | — | [ ] |
| Epoch 1 prints train_loss + val_loss | — | [ ] |
| Early stopping triggers OR 100 epochs complete | — | [ ] |
| `models/face_keypoint_cnn.pth` saved | — | [ ] |
| `models/training_curves.png` generated | — | [ ] |
| `models/keypoint_predictions.png` generated | — | [ ] |
| Final val MAE < 5 px | < 5 px | [ ] |
| Visual keypoint overlay looks correct | Green ≈ Red dots | [ ] |

---

## Phase 9 — Integration Testing

```powershell
# Run all pytest unit + integration tests
python -m pip install pytest httpx
python -m pytest tests/ -v

# Run automated Swagger tests (requires running server)
python app.py          # terminal 1
python scripts/run_swagger_tests.py   # terminal 2
```

- [ ] All pytest tests pass (`tests/test_body_calibration.py`)
- [ ] All pytest tests pass (`tests/test_wardrobe_db.py`)
- [ ] All pytest tests pass (`tests/test_pipeline.py`)
- [ ] All pytest tests pass (`tests/test_api.py`)
- [ ] All 11 Swagger tests pass (`scripts/run_swagger_tests.py`)
- [ ] Upload real selfie to `POST /api/tryon` — `hasFaceTexture: true`
- [ ] Open `http://localhost:8000` and run full UI try-on
- [ ] Switch between outfit recommendations in carousel
- [ ] Toggle all 6 animation modes in renderer
- [ ] Mark garment Dirty → re-run → confirm excluded from recommendations
- [ ] Verify `GET /api/wardrobe/history` logs status changes

---

## Phase 10 — Future Enhancements

- [x] SQLite wardrobe persistence (`src/wardrobe_database.py`)
- [x] Garment wear history + analytics endpoints
- [x] Colour theory-based palette matching in NisfaMatchmaking
- [ ] Add real Blender `.glb` avatar to `assets/avatars/base_avatar.glb`
- [ ] Add clothing `.glb` files to `assets/outfits/`
- [ ] Fine-tune CNN on a larger facial landmark dataset
- [ ] Add user authentication (JWT)
- [ ] Mobile app (React Native) consuming the FastAPI backend
- [ ] Add outfit history / wear-count UI dashboard
- [ ] Integrate real Mixamo FBX-to-GLB export pipeline

---

## Quick Command Reference

```powershell
# Install all dependencies
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas scikit-learn matplotlib opencv-python mediapipe fastapi uvicorn[standard] python-multipart pydantic tqdm pytest httpx requests

# Generate asset directories + test selfie
python scripts/generate_test_assets.py

# Train the CNN model
python -m src.face_keypoint_model

# Run all unit + integration tests
python -m pytest tests/ -v

# Start the backend server
python app.py

# Run automated Swagger tests (server must be running)
python scripts/run_swagger_tests.py

# Open Swagger UI
start http://localhost:8000/docs

# Open frontend UI
start http://localhost:8000
```

---

*Last updated: 2026-05-11*
