# eWardrobe Avatar — Full Stack

A photo + body measurements in, a stylized dressed avatar out. This repo has
two layers:

1. **`avatar_pipeline/`** — six TensorFlow/Keras models (body shape, pose,
   skin tone, avatar generation, virtual try-on, 3D body reconstruction)
   plus the integration glue (`controller.py`, `pipeline_types.py`). See
   [avatar_pipeline/README.md](avatar_pipeline/README.md) for the model
   architectures, data-flow diagram, and training/inference details. **This
   layer requires TensorFlow and trained model artifacts in `saved_models/`,
   and is intended to run on Colab/cloud GPU.**
2. **`server/` + `mobile/`** — a Flask API and an Expo (React Native +
   TypeScript) mobile app that consume the pipeline. This is the "product"
   layer covered by this document.

## Why Flask + Expo (not Express)

The avatar pipeline is TF/Keras (Python). Re-implementing six trained
models in Node/Express isn't practical, so the API layer is a thin Flask
wrapper around `avatar_pipeline.controller` / `pipeline_types`. The frontend
is an Expo (React Native) app so the same TypeScript codebase runs on iOS,
Android, and web from one `npx expo start`.

## Mock mode vs. real mode

Running the real pipeline requires TensorFlow + trained weights in
`saved_models/` (see [avatar_pipeline/README.md §4](avatar_pipeline/README.md#4-recommended-end-to-end-folder-structure)).
For local development without either, `server/app.py` has a **mock mode**
(default, `AVATAR_PIPELINE_MOCK=1`):

| | Mock mode (default) | Real mode (`AVATAR_PIPELINE_MOCK=0`) |
|---|---|---|
| Body shape (Model 1) | `classify_body_shape` — the same rule-based logic Model 1's CNN was trained to imitate | trained CNN (`saved_models/model1_body_shape/`) |
| Pose (Model 2) | `CANONICAL_POSE` — a fixed standing pose | MoveNet via TF Hub |
| Skin tone (Model 3) | real face detection + dominant-color palette match (Model 3's algorithm, no CNN) | trained CNN refinement |
| Avatar render (Model 4) | `render_avatar` — the programmatic paper-doll renderer Model 4's GAN was trained to imitate | trained cVAE-GAN decoder |
| Try-on (Model 5) | **identical** — classical TPS warp, no trained weights either way | same |
| 3D body mesh (Model 6) | `default_params_from_measurements` — the same rule-based anthropometric approximation Model 6's CNN was trained to imitate, fed straight into `mesh_builder` + `glb_export` (no CNN). The head is textured with the user's own face (`face_features.extract_face_features`, OpenCV) and procedural hair/eyes/nose/mouth in both modes | trained CNN (`saved_models/model6_body3d/`) for body shape; face/hair texture is identical to mock mode |
| Dependencies | Flask, numpy, pandas, pillow, opencv-python | + TensorFlow, tensorflow-hub, scikit-learn, joblib |

Mock mode is not a stub — it's the real non-CNN halves of Models 1/3/4/5/6,
so avatars genuinely vary with the user's photo and measurements.

## Quick start

### 1. Backend (Flask)

```bash
pip install -r server/requirements.txt
python -m server.app                # mock mode -> http://localhost:5000
```

To run against the real trained models instead:

```bash
pip install -r requirements.txt              # adds TensorFlow etc.
set AVATAR_PIPELINE_MOCK=0                    # PowerShell: $env:AVATAR_PIPELINE_MOCK=0
set AVATAR_SAVED_MODELS_DIR=saved_models      # optional, this is the default
python -m server.app
```

Check it's up: `GET http://localhost:5000/api/health` -> `{"status": "ok", "mock": true}`.

## Phase 2: Photorealistic Avatars (Optional)

By default, avatars are generated procedurally. Phase 2 enables **photorealistic avatars** by combining realistic 3D base humanoid models with detected hairstyles and custom materials.

### Enable Phase 2

Set the environment variable before running the server:

```bash
# PowerShell (Windows)
$env:AVATAR_USE_REALISTIC=1
python -m server.app

# Bash/Linux
export AVATAR_USE_REALISTIC=1
python -m server.app
```

### Phase 2 Requirements

Phase 2 requires 3D base models and hairstyles:

1. **Base avatar models**: Male and female humanoid models in `.glb` format
2. **Hairstyle models**: Multiple hairstyles per gender in `.glb` format
3. **Asset structure**: Place models in `assets/` (see [assets/README.md](assets/README.md))

```
assets/
├── avatars/
│   ├── male/base.glb
│   └── female/base.glb
├── hair/
│   ├── male/*.glb
│   └── female/*.glb
```

**Without these assets, Phase 2 gracefully falls back to Phase 1 (procedural avatars).**

### Implementation Status

- ✅ **Complete**: Asset loading infrastructure, avatar builder framework, integration with mock pipeline
- ⏳ **TODO**: Mesh deformation, material application, face texture mapping, hairstyle merging
- 🔄 **TODO**: Asset acquisition (3D models)

See [PHASE2_IMPLEMENTATION.md](PHASE2_IMPLEMENTATION.md) for implementation details and development guide.

### Performance Impact

- Phase 1 (procedural): ~150ms
- Phase 2 (realistic): ~300-500ms (depends on assets and mesh complexity)

Phase 1 recommended for development. Phase 2 for production with photorealistic requirements.

### 2. Demo wardrobe assets (one-time)

The mobile app ships with three sample garments used for the try-on demo.
They're already generated and committed under `mobile/assets/clothing/`, but
if you ever need to regenerate them:

```bash
python -m scripts.generate_sample_clothing
```

This writes `tshirt_red.png`, `jeans_blue.png`, and `dress_green.png`
(256x256 RGBA, transparent background) to `mobile/assets/clothing/`.

### 3. Mobile app (Expo)

```bash
cd mobile
npm install
npx expo start --web      # or: npm run android / npm run ios
```

The app needs to reach the Flask backend:

- **Web** (`--web`): defaults to `http://localhost:5000`, which works as
  long as the backend is running on the same machine.
- **Physical device via Expo Go**: the app auto-detects the dev machine's
  LAN IP from the Metro bundler (`Constants.expoConfig.hostUri`) and points
  at `http://<that-ip>:5000`. Make sure the Flask server is reachable on
  your LAN (it binds `0.0.0.0` by default) and your firewall allows
  port 5000.
- **Android emulator**: falls back to `http://10.0.2.2:5000` automatically.
- **Override**: set `EXPO_PUBLIC_API_URL` (e.g. in `mobile/.env`) to point
  at any backend, e.g. a deployed instance:
  ```
  EXPO_PUBLIC_API_URL=https://your-deployed-backend.example.com
  ```

### 4. Try the flow

1. **Profile screen** — pick a full-body photo and enter bust/waist/hips/
   height (cm). Submitting calls `POST /api/avatars`.
2. **Avatar screen** — shows the generated avatar, a rotatable 3D body model
   (drag to spin) with a head textured from the user's own photo (face crop +
   sampled hair color) plus procedural eyes/eyebrows/nose/mouth/ears,
   detected body shape, and matched skin tone.
3. **Wardrobe screen** — pick one of the three sample garments.
4. **Try-on screen** — calls `POST /api/avatars/<id>/tryon` and shows the
   avatar wearing that item.

## API reference

All endpoints are implemented in [server/app.py](server/app.py).

| Method & path | Body | Returns |
|---|---|---|
| `GET /api/health` | — | `{status, mock}` |
| `POST /api/avatars` | multipart: `photo` (image file), `bust`, `waist`, `hips`, `height` (cm, > 0) | `{avatar_id, avatar_image, body_shape, body_shape_confidence, skin_tone, avatar_mesh_url, body3d_params}` |
| `GET /api/avatars/<avatar_id>` | — | same shape as above, or 404 |
| `GET /api/avatars/<avatar_id>/mesh.glb` | — | the avatar's 3D body mesh as a binary `.glb` file (`model/gltf-binary`), or 404 |
| `POST /api/avatars/<avatar_id>/tryon` | multipart: `clothing` (RGBA PNG, alpha = mask), `category` (`upper_body` \| `lower_body` \| `dress`) | `{dressed_image}` |

`avatar_image` / `dressed_image` are `data:image/png;base64,...` URIs, ready
to drop straight into an `<Image source={{ uri }} />`. `avatar_mesh_url` is a
path (e.g. `/api/avatars/<id>/mesh.glb`) — resolve it against the API base
URL and load it with a glTF loader (see the mobile app's `AvatarViewer3D`).
`body3d_params` is a dict of 13 floats (fractions of `height`) keyed by
`avatar_pipeline.model6_body3d.params.PARAM_NAMES`. The mesh's head is
textured with a crop of the user's own face (detected via OpenCV from
`photo`, falling back to a flat skin-color texture if no face is found) and
its hair/eyebrows use a hair color sampled from the photo (falling back to
`avatar_pipeline.model6_body3d.face_features.DEFAULT_HAIR_RGB`).

Each created avatar is persisted under `server/sessions/<avatar_id>/`
(`avatar.png` + `avatar_mesh.glb` + `avatar_meta.json`, via
`pipeline_types.save_avatar_result` / `load_avatar_result`) so try-on and
mesh requests can run later without recomputing Phase A.

## Project layout

```
New avatar/                          (project root)
├── README.md                        <- this file (full-stack overview)
├── requirements.txt                 <- TensorFlow pipeline deps (real mode)
├── avatar_pipeline/                 <- Models 1-6 + controller (see its README)
├── data/                            <- training data per model
├── scripts/
│   └── generate_sample_clothing.py  <- generates mobile/assets/clothing/*.png
├── server/                          <- Flask API
│   ├── requirements.txt             <- Flask + mock-mode deps (no TF)
│   ├── app.py                       <- routes (see API reference above)
│   ├── mock_pipeline.py             <- TF-free Phase A
│   ├── storage.py                   <- per-avatar session persistence
│   └── sessions/                    <- created avatars (gitignored)
└── mobile/                          <- Expo (React Native + TypeScript) app
    ├── App.tsx                      <- navigation stack
    ├── src/
    │   ├── api/client.ts            <- Flask API client
    │   ├── theme.ts                 <- colors/spacing/typography
    │   ├── types.ts                 <- shared TS types (mirrors API JSON)
    │   ├── data/                    <- wardrobe items, body-shape copy
    │   ├── components/              <- reusable UI building blocks,
    │   │                                including AvatarViewer3D.tsx
    │   │                                (expo-gl + three.js GLB viewer)
    │   ├── navigation/types.ts      <- React Navigation param list
    │   └── screens/                 <- Profile, Avatar, Wardrobe, TryOn
    └── assets/clothing/             <- sample garment PNGs
```
#   3 d - a v t a r  
 # 3d-avtar
