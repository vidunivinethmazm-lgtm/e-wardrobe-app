# Face Texture Application Pipeline

## How the User's Selfie Face Gets Onto the 3D Avatar Head

This document describes the complete end-to-end flow: from the user taking a
selfie in the mobile app to the final face-textured 3D avatar mesh (GLB).

---

## High-Level Flow

```
Mobile App (selfie)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 1: Initial Avatar Creation                    │
│  POST /api/avatars                                   │
│  (controller.py → predict.py → mesh_builder.py)     │
│                                                      │
│  1. Face detection + landmark extraction              │
│  2. Procedural mesh with Delaunay warp                │
│  3. [Optional] Realistic MakeHuman avatar             │
│     with MakeHuman UV warp                            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 2: Face Customization                         │
│  POST /api/avatars/<id>/customize-face               │
│  (app.py → face_customization.py → makehuman_mesh.py)│
│                                                      │
│  1. Decode 256×256 face crop from client              │
│  2. Client provides real crop dimensions              │
│     (faceCropWidth × faceCropHeight in original photo)│
│  3. Rebuild MakeHuman mesh with scaled head            │
│  4. Delaunay warp with MakeHuman UV anchors            │
└─────────────────────────────────────────────────────┘
    │
    ▼
Mobile App → AvatarViewer3D renders the GLB
```

---

## Phase 1: POST /api/avatars

### Entry Point
```
server/app.py → controller.build_avatar()
```

### Step 1: Body + Face Prediction
**File**: `avatar_pipeline/model6_body3d/predict.py` — `predict_body3d()`

```python
# 1. Extract face features with MediaPipe landmarks
face = extract_face_features(photo, estimate_landmarks=True)
# Returns:
#   face_crop: 128×128 RGB crop of the face
#   face_width, face_height: pixel dimensions of crop in original photo
#   landmarks_2d: (478, 2) MediaPipe face-mesh points
#   hair_rgb, facial_analysis...

# 2. Build the procedural 3D mesh
mesh = build_avatar_mesh(
    params, height, skin_rgb,
    face_crop=face["face_crop"],
    hair_rgb=face["hair_rgb"],
    selfie_rgb=photo,           # ← original full photo
    landmarks_2d=landmarks_2d,   # ← 478 landmarks
    face_width=face_width,       # ← real pixel dimensions
    face_height=face_height,
)
```

### Step 2: Procedural Mesh Warp Path
**File**: `avatar_pipeline/model6_body3d/mesh_builder.py` — `_build_head_texture()`

```
_build_head_texture(skin_rgb, face_crop, selfie_rgb, landmarks_2d)
    │
    ├── landmarks >= 468?
    │   YES → build_head_texture_warped(selfie_rgb, landmarks_2d)
    │          │
    │          ├── warp_face_to_uv(selfie_rgb, landmarks_2d)
    │          │   │  Uses _FACE_UV_ANCHORS (procedural head)
    │          │   │  Delaunay triangulation in UV space
    │          │   │  Piecewise-affine warp per triangle
    │          │   ▼
    │          │  (512×512 warped face in UV layout)
    │          │
    │          ├── blend_face_with_skin(warped, skin_rgb)
    │          │   │  Poisson or feather blending
    │          │   ▼
    │          │  (face blended onto skin-tone canvas)
    │          │
    │          └── _VERTICAL_OFFSET = 0.12 (only for procedural anchors)
    │              (shifts face down to match MakeHuman UV layout)
    │
    └── NO → Centre-paste fallback
             (resize face crop → paste on skin canvas)
```

**UV Anchor System** (`face_texture_builder.py`):
- `_FACE_UV_ANCHORS` — 132 MediaPipe landmark indices mapped to UV
  coordinates on the procedural UV-sphere head
- Procedural head UV: `v=0=bottom(neck), v=1=top(crown)`
- Front of face at `u=0.5`

### Step 3: Realistic MakeHuman Avatar (Optional)
**File**: `avatar_pipeline/controller.py` — `build_avatar()`

```python
# After procedural mesh is built, also try realistic avatar
avatar_builder = create_avatar_builder(use_realistic=True)
if avatar_builder:
    realistic_glb = avatar_builder.build_realistic_avatar(
        gender=gender,
        body3d_params=body3d_result["params"],
        height=height,
        skin_rgb=skin_rgb,
        face_crop=face["face_crop"],
        selfie_rgb=photo,                    # ← full original photo
        landmarks_2d=face.get("landmarks_2d"),# ← 478 landmarks
        face_width=face.get("face_width"),
        face_height=face.get("face_height"),
    )
    if realistic_glb:
        mesh_glb = realistic_glb  # ← replaces procedural mesh!
```

**File**: `avatar_pipeline/model6_body3d/avatar_builder.py` — `_apply_face_texture()`

```
_apply_face_texture(glb, face_crop, skin_rgb, selfie_rgb, landmarks_2d)
    │
    ├── selfie_rgb + landmarks_2d available?
    │   YES → build_head_texture_warped(selfie_rgb, landmarks_2d,
    │           uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN)
    │          │
    │          │  Uses MakeHuman-specific UV anchors!
    │          │  NO vertical offset applied (anchors already correct)
    │          ▼
    │          (MakeHuman texture PNG)
    │
    └── NO → Centre-paste with feather mask
```

---

## Phase 2: POST /api/avatars/<id>/customize-face

### Entry Point
```
server/app.py → customize_face()
```

### Step 1: Decode Client Request
**File**: `server/app.py`

```python
features = request.get_json()
# features contains:
#   faceImage: base64-encoded 256×256 JPEG face crop
#   faceCropWidth, faceCropHeight: real pixel dims from original selfie
#   faceShape, jawWidth, noseWidth, eyeSpacing: normalized measurements
#   skinTone, hairColor: detected colors
#   gender: user-selected gender

# Decode face image → 256×256 RGB numpy array
selfie_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

# Detect MediaPipe landmarks on the face crop
landmarks_2d = estimate_face_landmarks(selfie_rgb)
# → (478, 2) float32 — landmarks in 256×256 coordinate space
```

### Step 2: Apply Customization
**File**: `avatar_pipeline/model6_body3d/face_customization.py`

```python
apply_face_customization(avatar_result, features,
                         selfie_rgb, landmarks_2d):

    # 1. Read real crop dimensions from client
    client_w = features.get("faceCropWidth")   # e.g., 228
    client_h = features.get("faceCropHeight")  # e.g., 300
    # → face_width=228, face_height=300

    # 2. Nudge head_radius from jawWidth
    head_radius *= features["jawWidth"] / 0.8

    # 3. Rebuild MakeHuman mesh with head scaling
    build_personalized_glb(
        body3d_params, height_cm, skin_rgb, face_shape,
        face_crop=selfie_rgb,          # ← 256×256 crop
        selfie_rgb=selfie_rgb,         # ← same 256×256 image
        landmarks_2d=landmarks_2d,     # ← 478 landmarks
        face_width=face_width,         # ← 228 (real! not 256)
        face_height=face_height,       # ← 300 (real! not 256)
    )
```

### Step 3: MakeHuman Mesh with Head Scaling
**File**: `avatar_pipeline/model6_body3d/makehuman_mesh.py` — `build_personalized_glb()`

```
# 1. Compute 6 morph weights from body3d_params
#    (shoulderWidth, hipWidth, bodyType, headWidth, etc.)
weights = compute_morph_weights(body3d_params, face_shape)

# 2. Adjust headWidth based on face crop aspect ratio
user_aspect = face_width / face_height          # e.g., 228/300 = 0.76
DEFAULT_ASPECT_RATIO = 0.83                     # MakeHuman default
factor = user_aspect / DEFAULT_ASPECT_RATIO     # e.g., 0.76/0.83 = 0.916
head_width_adj = clip((factor - 1.0) * 2.0, -0.5, 0.5)
weights["headWidth"] = clip(weights["headWidth"] + head_width_adj, -1, 1)
# → narrower face = negative headWidth adjustment

# 3. Apply morph deltas to base mesh vertices
positions = base_positions.copy()
for name, weight in weights.items():
    positions += deltas[name] * weight

# 4. Scale to user's height
positions *= (height_cm / 100.0) / base_height

# 5. Build texture (with Delaunay warp)
texture_png = _build_texture(skin_rgb, face_crop,
                             selfie_rgb, landmarks_2d,
                             face_width, face_height)
```

### Step 4: MakeHuman Texture with Delaunay Warp
**File**: `avatar_pipeline/model6_body3d/makehuman_mesh.py` — `_build_texture()`

```
_build_texture(skin_rgb, face_crop, selfie_rgb, landmarks_2d, face_width, face_height)
    │
    ├── landmarks >= 468?
    │   YES → build_head_texture_warped(selfie_rgb, landmarks_2d,
    │           uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN)
    │          │
    │          │  MakeHuman UV anchors have:
    │          │    u = 0.5 + atan2(x, z) / 2π  (front = u=0.5)
    │          │    v = 0.5 - asin(y) / π       (v=0=top, v=1=bottom)
    │          │  NO vertical offset (v offset = 0)
    │          ▼
    │          Returns PNG bytes with warped face texture
    │
    └── NO → Centre-paste fallback
             │  face crop → resize keeping aspect ratio
             │  paste at center with feathered edge
             │  vertical offset = 0.12 (for MakeHuman UV)
             ▼
             Returns PNG bytes
```

---

## MakeHuman UV Anchor System

### Why Separate Anchors?
The `face_texture_builder.py` originally had one UV anchor set calibrated for
the **procedural UV-sphere head** (`mesh_builder._build_head()`). The
MakeHuman head uses a different UV projection:

| Property | Procedural Head | MakeHuman Head |
|----------|:---------------:|:--------------:|
| V direction | v=0=bottom, v=1=top | v=0=top, v=1=bottom |
| Front face | u=0.5 | u=0.5 |
| UV formula | Grid-based (n_lat×n_lon) | Spherical projection |
| Anchors | `_FACE_UV_ANCHORS` | `_FACE_UV_ANCHORS_MAKEHUMAN` |

### Anchor Generation
**File**: `scripts/compute_mh_uv_anchors.py`

The MakeHuman anchors were generated by converting procedural anchors
through spherical coordinate transformation:

```python
# Procedural UV → 3D direction
phi = 2π * u_proc + 1.5π          # azimuth angle
theta = π * (1 - v_proc)          # polar angle
x = sin(theta) * cos(phi)
y = cos(theta)
z = sin(theta) * sin(phi)

# 3D direction → MakeHuman UV
u_mh = 0.5 + atan2(x, z) / (2π)
v_mh = 0.5 - asin(y) / π
```

---

## Mobile App: How the GLB Reaches the User

### AvatarViewer3D Component
**File**: `mobile/src/components/AvatarViewer3D.tsx`

```typescript
<AvatarViewer3D
  config={avatarConfig}
  remoteAvatarUrl={meshUrl}       // ← URL to /api/avatars/<id>/mesh.glb
  remoteTextureUrl={textureUrl}   // ← URL to /api/avatars/<id>/face-texture.png
/>
```

The component:
1. Loads GLB via `GLTFLoader` from `remoteAvatarUrl`
2. If `remoteTextureUrl` is provided:
   - Loads face texture PNG via expo-three's `TextureLoader`
   - Applies to all meshes via `MeshBasicMaterial`
3. If no remote texture:
   - If `config.faceTextureUri` exists → composites face onto skin canvas
     using WebGLRenderTarget (client-side centre-paste)
   - Otherwise → flat skin tint

### Face Texture URL Server Endpoint
**File**: `server/app.py` — `GET /api/avatars/<id>/face-texture.png`

Extracts the embedded PNG texture from the GLB's binary blob and returns it
as a standalone PNG file. This allows the mobile app to load the
server-baked face texture separately, bypassing GLTFLoader's inability to
decode embedded bufferView images on React Native.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `server/app.py` | HTTP endpoints: create avatar, customize face, serve mesh/texture |
| `avatar_pipeline/controller.py` | Phase 1 orchestrator: calls models 1-6 |
| `model6_body3d/predict.py` | Face extraction + mesh building entry point |
| `model6_body3d/mesh_builder.py` | Procedural avatar mesh + head texture builder |
| `model6_body3d/makehuman_mesh.py` | MakeHuman-based avatar mesh with head scaling + texture |
| `model6_body3d/face_customization.py` | Customize-face logic for Phase 2 |
| `model6_body3d/face_features.py` | MediaPipe face detection + feature extraction |
| `model6_body3d/face_texture_builder.py` | Delaunay warp engine + UV anchors (procedural + MakeHuman) |
| `model6_body3d/avatar_builder.py` | Realistic avatar builder (trimesh + pygltflib) |
| `mobile/src/components/AvatarViewer3D.tsx` | Mobile 3D rendering with texture support |
| `mobile/src/services/faceAnalysis.ts` | Mobile-side face analysis |
| `mobile/src/screens/FacePreviewScreen.tsx` | Face preview + customize-face call |
| `scripts/compute_mh_uv_anchors.py` | MakeHuman UV anchor generator |
| `scripts/bake_makehuman_morphs.py` | MakeHuman base mesh UV layout definition |

---

## Data Flow Diagram (Texture Paths)

```
                     ┌─────────────────────────┐
                     │   User Selfie (photo)     │
                     └────────┬────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │ MediaPipe FaceMesh │          │ Mobile analyzeFace()  │
    │ (478 landmarks)    │          │ (TFJS MediaPipe)      │
    └────────┬─────────┘          └──────────┬───────────┘
             │                               │
             ▼                               ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │ Server-side       │          │ faceCropWidth/Height  │
    │ estimate_face_    │          │ + faceImage base64    │
    │ landmarks()       │          │ + normalized features │
    └────────┬─────────┘          └──────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
              ┌─────────────────────────────┐
              │   build_personalized_glb()   │
              │   or build_avatar_mesh()     │
              └────────────┬────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐     ┌─────────────────────┐
    │ Delaunay Warp    │     │ Centre-paste         │
    │ (MakeHuman UV)   │     │ (fallback)           │
    │                  │     │                      │
    │ MediaPipe LM →   │     │ Resize face crop     │
    │ UV anchors       │     │ Paste on skin canvas │
    │ Affine warp per  │     │ Feather edges        │
    │ triangle         │     │                      │
    └────────┬────────┘     └──────────┬──────────┘
             │                         │
             └──────────┬──────────────┘
                        ▼
              ┌─────────────────────┐
              │ 512×512 Texture PNG  │
              │ (embedded in GLB)    │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ AvatarViewer3D.tsx   │
              │ Three.js rendering   │
              │ on mobile            │
              └─────────────────────┘
```
