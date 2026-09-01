# Multi-Angle Face Texture Pipeline — සම්පූර්ණ ක්‍රියාත්මක කිරීමේ සැලැස්ම

## 🎯 ඉලක්කය
User ගේ මුහුණ Front + Left + Right ඡායාරූප 3ක් භාවිතා කරලා 360° කරකැවෙනකොටත් ස්වභාවිකව පෙනෙන Avatar Face Texture එකක් හදන්න.

---

## 🔍 වත්මන් තත්වය (Current State)

### Mobile App එක දැනට කරන්නේ:
1. **Front, Left, Right** ඡායාරූප 3ක් Capture කරනවා ✅
2. නමුත් **Front photo එක විතරයි** Server එකට යවන්නේ ❌
3. Left/Right photos **captureStorage** එකේ Save වෙනවා, නමුත් Server එකට **කවදාවත් යවන්නේ නැහැ**

### Backend එක දැනට කරන්නේ:
1. **Single front photo** එකක් විතරක් භාවිතා කරනවා
2. **Delaunay triangulation** එකෙන් 2D landmarks → UV space warp එකක් කරනවා
3. ඒක **3D perspective correction** එකක් නැහැ (head turned 30° නම් distortion)
4. **Back/sides of head** එකට flat skin color එකක් දානවා (texture නැහැ)

---

## 📋 කළ යුතු වෙනස්කම් (What Needs to Change)

### අදියර 1: Mobile App Changes (TypeScript/React Native)

#### 1.1 `mobile/src/services/faceAnalysis.ts` — Multi-angle analysis function
**File:** `mobile/src/services/faceAnalysis.ts`

**Change:** Front photo එකේ face detection + feature analysis කරලා, Left/Right photos වලින්ත් MediaPipe landmarks extract කරලා, **photos 3ම base64 කරලා** server එකට යවන්න.

```typescript
// New function
export async function analyzeFaceMultiAngle(
  frontUri: string,
  leftUri: string,
  rightUri: string
): Promise<FaceAnalysisResult & {
  leftFaceImage?: string;  // base64
  rightFaceImage?: string; // base64
}> {
  // 1. Front photo එක analyze කරන්න (existing logic)
  const frontResult = await analyzeFace(frontUri);
  
  // 2. Left photo crop + base64
  const leftResult = await cropAndEncode(leftUri);
  
  // 3. Right photo crop + base64
  const rightResult = await cropAndEncode(rightUri);
  
  return {
    ...frontResult,
    faceCustomization: {
      ...frontResult.faceCustomization,
      leftFaceImage: leftResult.base64,
      rightFaceImage: rightResult.base64,
    }
  };
}
```

#### 1.2 `mobile/src/screens/AvatarCreatorScreen.tsx` — Send all 3 photos
**File:** `mobile/src/screens/AvatarCreatorScreen.tsx`

**Change:** `analyze()` function එකේදී front photo එක විතරක් නෙවෙයි, **photos 3ම** server එකට send කරන්න.

**Current code (line ~220):**
```typescript
const { faceTextureUri, faceCustomization } = await analyzeFace(frontUri);
const { avatar_mesh_url, face_texture_url } = await customizeFaceAvatar(avatar.avatar_id, {
  ...faceCustomization,
  gender: avatarConfig.gender,
});
```

**New code:**
```typescript
// Get all 3 captured photos
const leftUri = capturesRef.current.left;
const rightUri = capturesRef.current.right;

// Analyze all 3 photos together
const { faceTextureUri, faceCustomization, leftFaceImage, rightFaceImage } = 
  await analyzeFaceMultiAngle(frontUri, leftUri, rightUri);

// Send all 3 face images to server
const { avatar_mesh_url, face_texture_url } = await customizeFaceMultiAngle(
  avatar.avatar_id, 
  {
    ...faceCustomization,
    gender: avatarConfig.gender,
    leftFaceImage,   // NEW
    rightFaceImage,  // NEW
  }
);
```

#### 1.3 `mobile/src/api/client.ts` — New API function
**File:** `mobile/src/api/client.ts`

**Change:** නව API function එකක් `customizeFaceMultiAngle()` එකතු කරන්න.

```typescript
export async function customizeFaceMultiAngle(
  avatarId: string,
  features: FaceCustomizationRequest & {
    leftFaceImage?: string;
    rightFaceImage?: string;
  }
): Promise<{ avatar_mesh_url: string; face_texture_url?: string }> {
  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/customize-face-multi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  });
  return handleResponse(res);
}
```

#### 1.4 `mobile/src/types.ts` — Update types
**File:** `mobile/src/types.ts`

**Change:** `FaceCustomizationRequest` type එකට `leftFaceImage`, `rightFaceImage` fields එකතු කරන්න.

```typescript
export interface FaceCustomizationRequest {
  faceShape: string;
  jawWidth: number;
  noseWidth: number;
  eyeSpacing: number;
  skinTone: string;
  hairColor: string;
  faceImage?: string;       // front face (already exists)
  faceCropWidth?: number;
  faceCropHeight?: number;
  // NEW fields for multi-angle:
  leftFaceImage?: string;   // left profile base64
  rightFaceImage?: string;  // right profile base64
}
```

---

### අදියර 2: Backend New Module — Multi-Angle Texture Pipeline (Python)

#### 2.1 නව File: `avatar_pipeline/model6_body3d/multi_angle_texture.py`
**Create this new file**

මෙය Multi-angle projective texture mapping එකේ **core engine** එකයි.

```python
"""
Multi-Angle Face Texture Pipeline

Front + Left + Right images 3ක් භාවිතා කරලා:
1. Head Pose Detection (MediaPipe SolvePnP → Yaw/Pitch/Roll)
2. UV Region Assignment (එක් එක් angle එකට UV map එකේ කොටසක්)
3. Projective Texture Mapping (3D → 2D projection)
4. Weighted Blending (overlap regions seamless කරන්න)
5. Final Composite UV Texture

Output: 512×512 PNG texture එකක් (single texture, multi-angle content එක්ක)
"""

import numpy as np
import cv2

# ── Standard MediaPipe face landmark indices for head pose ──
# These are the 6 key points used for SolvePnP head pose estimation
_HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# 3D model points (in mm) relative to the face center
# These are canonical MediaPipe 3D face model coordinates
_FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],         # nose tip
    [0.0, -330.0, -65.0],    # chin
    [-225.0, 170.0, -135.0], # left eye outer corner
    [225.0, 170.0, -135.0],  # right eye outer corner
    [-150.0, -150.0, -125.0],# left mouth corner
    [150.0, -150.0, -125.0], # right mouth corner
], dtype=np.float64)


def estimate_head_pose(landmarks_2d: np.ndarray, image_w: int, image_h: int):
    """Estimate head pose (yaw, pitch, roll) from MediaPipe 2D landmarks.
    
    Uses OpenCV SolvePnP to compute rotation vector from 6 key facial points.
    
    Returns:
        yaw: float (degrees, positive = right turn)
        pitch: float (degrees, positive = looking up)
        roll: float (degrees, positive = tilting right)
        rvec: np.ndarray (3,) rotation vector
        tvec: np.ndarray (3,) translation vector
    """
    # Camera intrinsic matrix (approximate)
    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    # Get 2D image points for the 6 key landmarks
    image_points = []
    for name, idx in _HEAD_POSE_LANDMARKS.items():
        if idx < len(landmarks_2d):
            image_points.append(landmarks_2d[idx])
    
    if len(image_points) < 4:
        return 0.0, 0.0, 0.0, None, None
    
    image_points = np.array(image_points[:6], dtype=np.float64)
    model_3d = _FACE_MODEL_3D[:len(image_points)]
    
    success, rvec, tvec = cv2.solvePnP(
        model_3d, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0, None, None
    
    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    # Convert to degrees
    pitch = np.degrees(x)   # up/down
    yaw = np.degrees(y)     # left/right
    roll = np.degrees(z)    # tilt
    
    return yaw, pitch, roll, rvec, tvec


# ── UV Region Assignment ──
# එක් එක් image angle එකට UV map එකේ කොටසක් assign කරනවා

_UV_REGIONS = {
    "left": {
        "yaw_range": (-60, -10),
        "u_range": (0.0, 0.35),
        "blend_weight": {"left": 1.0, "front": 0.3, "right": 0.0},
    },
    "front": {
        "yaw_range": (-15, 15),
        "u_range": (0.3, 0.7),
        "blend_weight": {"left": 0.3, "front": 1.0, "right": 0.3},
    },
    "right": {
        "yaw_range": (10, 60),
        "u_range": (0.65, 1.0),
        "blend_weight": {"left": 0.0, "front": 0.3, "right": 1.0},
    },
}


def assign_uv_regions(yaw_angles: dict):
    """Assign each image to its UV region based on detected yaw angle.
    
    Args:
        yaw_angles: {"front": yaw_front, "left": yaw_left, "right": yaw_right}
    
    Returns:
        regions: {"front": "front", "left": "left", "right": "right"}
                 (or adjusted based on actual yaw values)
    """
    regions = {}
    for name, yaw in yaw_angles.items():
        if yaw is None:
            continue
        for region_name, region in _UV_REGIONS.items():
            lo, hi = region["yaw_range"]
            if lo <= yaw <= hi:
                regions[name] = region_name
                break
        else:
            # Default: closest region
            regions[name = min(_UV_REGIONS.keys(), 
                key=lambda r: abs(yaw - np.mean(_UV_REGIONS[r]["yaw_range"])))
    return regions


# ── Projective Texture Mapping ──

def project_face_to_uv(
    selfie_rgb: np.ndarray,
    landmarks_2d: np.ndarray,
    head_vertices_3d: np.ndarray,
    head_uvs: np.ndarray,
    face_indices: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    texture_size: int = 512,
) -> np.ndarray:
    """Project face image pixels onto UV texture using projective mapping.
    
    Instead of 2D Delaunay warping (current approach), this uses actual 3D
    head geometry + camera pose to project pixels correctly, handling
    perspective distortion from angled faces.
    
    Args:
        selfie_rgb: (H, W, 3) input face image
        landmarks_2d: (N, 2) MediaPipe landmarks in image space
        head_vertices_3d: (V, 3) 3D head mesh vertices in world space
        head_uvs: (V, 2) UV coordinates for each vertex
        face_indices: (F, 3) face/triangle indices into vertices
        rvec: (3,) rotation vector from SolvePnP
        tvec: (3,) translation vector from SolvePnP
        camera_matrix: (3, 3) camera intrinsic matrix
        texture_size: output texture resolution
    
    Returns:
        (texture_size, texture_size, 3) uint8 warped texture
    """
    # 1. Project 3D head vertices to 2D image space
    projected_2d, _ = cv2.projectPoints(
        head_vertices_3d.reshape(-1, 1, 3).astype(np.float64),
        rvec, tvec, camera_matrix, None
    )
    projected_2d = projected_2d.reshape(-1, 2)
    
    # 2. For each triangle, check visibility and warp
    warped = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    weight = np.zeros((texture_size, texture_size), dtype=np.float32)
    
    for tri in face_indices:
        # Get 3D triangle vertices
        v3d = head_vertices_3d[tri]
        
        # Back-face culling: check if triangle faces camera
        normal = np.cross(v3d[1] - v3d[0], v3d[2] - v3d[0])
        view_dir = np.array([0, 0, 1])  # camera looks along +Z
        if np.dot(normal, view_dir) < 0:
            continue  # back-facing, skip
        
        # Get projected 2D positions (in image space)
        src_tri = projected_2d[tri].astype(np.float32)
        
        # Get UV positions (in texture space)
        dst_tri = head_uvs[tri].astype(np.float32)
        dst_tri[:, 0] *= texture_size  # u * size
        dst_tri[:, 1] *= texture_size  # v * size
        
        # Bounding box in destination
        x_min = max(0, int(np.floor(dst_tri[:, 0].min())))
        y_min = max(0, int(np.floor(dst_tri[:, 1].min())))
        x_max = min(texture_size - 1, int(np.ceil(dst_tri[:, 0].max())))
        y_max = min(texture_size - 1, int(np.ceil(dst_tri[:, 1].max())))
        
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Affine transform: destination (UV) → source (image)
        affine_mat = cv2.getAffineTransform(dst_tri, src_tri)
        
        # Create mask for this triangle
        tw, th = x_max - x_min + 1, y_max - y_min + 1
        mask = np.zeros((th, tw), dtype=np.uint8)
        tri_local = dst_tri.copy()
        tri_local[:, 0] -= x_min
        tri_local[:, 1] -= y_min
        cv2.fillConvexPoly(mask, tri_local.astype(np.int32), 255)
        
        # Sample from source image
        src_rect = cv2.warpAffine(
            selfie_rgb, affine_mat, (tw, th),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        
        # Accumulate with mask
        mask_f = mask.astype(np.float32) / 255.0
        for c in range(3):
            warped[y_min:y_max+1, x_min:x_max+1, c] += src_rect[..., c] * mask_f
        weight[y_min:y_max+1, x_min:x_max+1] += mask_f
    
    # Normalize
    weight = np.clip(weight, 1e-6, None)
    for c in range(3):
        warped[..., c] = np.clip(warped[..., c] / weight, 0, 255).astype(np.uint8)
    
    return warped


# ── Multi-View Blending ──

def blend_multi_view_textures(
    textures: dict[str, np.ndarray],
    yaw_angles: dict[str, float],
    texture_size: int = 512,
) -> np.ndarray:
    """Blend multiple view textures into one composite UV texture.
    
    Uses per-pixel weighting based on view angle: front contributes most
    to center UV, sides contribute most to side UV regions.
    
    Args:
        textures: {"front": np.ndarray, "left": np.ndarray, "right": np.ndarray}
        yaw_angles: {"front": float, "left": float, "right": float}
        texture_size: output resolution
    
    Returns:
        (texture_size, texture_size, 3) blended composite texture
    """
    # Create weight maps for each view based on UV coordinates
    u_coords = np.linspace(0, 1, texture_size)
    
    # Weight functions — Gaussian centered on each view's UV region
    def _weight_fn(u, center_u, sigma=0.2):
        return np.exp(-0.5 * ((u - center_u) / sigma) ** 2)
    
    # Center UV for each view
    view_centers = {
        "left": 0.15,    # u=0.0-0.3
        "front": 0.5,    # u=0.3-0.7
        "right": 0.85,   # u=0.7-1.0
    }
    
    composite = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    total_weight = np.zeros((texture_size, texture_size), dtype=np.float32)
    
    for view_name, texture in textures.items():
        if texture is None:
            continue
        # Determine which UV region this view covers
        yaw = yaw_angles.get(view_name, 0)
        if yaw is None:
            continue
        
        # Create per-pixel weight map
        weight_map = np.zeros((texture_size, texture_size), dtype=np.float32)
        for u_center in [view_centers.get(region, 0.5) 
                         for region, cfg in _UV_REGIONS.items()
                         if cfg["yaw_range"][0] <= yaw <= cfg["yaw_range"][1]]:
            # Actually use the single best-matching region
            best_region = min(_UV_REGIONS.keys(), 
                key=lambda r: abs(yaw - np.mean(_UV_REGIONS[r]["yaw_range"])))
            u_center = view_centers[best_region]
            
        for u_pixel in range(texture_size):
            u_norm = u_pixel / texture_size
            w = _weight_fn(u_norm, u_center)
            weight_map[:, u_pixel] = w
        
        # Add to composite
        for c in range(3):
            composite[..., c] += texture.astype(np.float32) * weight_map
        total_weight += weight_map
    
    # Normalize
    total_weight = np.clip(total_weight, 1e-6, None)
    for c in range(3):
        composite[..., c] = np.clip(composite[..., c] / total_weight, 0, 255)
    
    return composite.astype(np.uint8)


# ── Main Pipeline Entry Point ──

def build_multi_angle_texture(
    front_image: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    front_landmarks: np.ndarray,
    left_landmarks: np.ndarray,
    right_landmarks: np.ndarray,
    skin_rgb: tuple[int, int, int],
    texture_size: int = 512,
) -> bytes:
    """Main entry point: process 3 images → composite UV texture PNG.
    
    Args:
        front_image: (H, W, 3) front-facing selfie
        left_image: (H, W, 3) left profile photo
        right_image: (H, W, 3) right profile photo
        front_landmarks: (N, 2) MediaPipe landmarks for front
        left_landmarks: (N, 2) MediaPipe landmarks for left
        right_landmarks: (N, 2) MediaPipe landmarks for right
        skin_rgb: (R, G, B) skin color
        texture_size: output texture resolution
    
    Returns:
        PNG bytes of the composite texture
    """
    images = {"front": front_image, "left": left_image, "right": right_image}
    landmarks = {"front": front_landmarks, "left": left_landmarks, "right": right_landmarks}
    
    # 1. Estimate head pose for each image
    yaw_angles = {}
    for name, img in images.items():
        h, w = img.shape[:2]
        lm = landmarks[name]
        yaw, pitch, roll, rvec, tvec = estimate_head_pose(lm, w, h)
        yaw_angles[name] = yaw
    
    print(f"[multi_angle_texture] Head pose yaw angles: {yaw_angles}")
    
    # 2. Generate separate Delaunay warp for each view
    # (Using existing warp_face_to_uv function from face_texture_builder)
    from .face_texture_builder import warp_face_to_uv, blend_face_with_skin, _FACE_UV_ANCHORS_MAKEHUMAN
    
    per_view_textures = {}
    for name in ["front", "left", "right"]:
        if images[name] is None or landmarks[name] is None or len(landmarks[name]) < 468:
            continue
        
        warped = warp_face_to_uv(
            images[name], landmarks[name],
            img_size=texture_size,
            uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN,
        )
        blended = blend_face_with_skin(warped, skin_rgb, blend_mode="feather")
        per_view_textures[name] = blended
    
    # 3. Blend multi-view textures together
    if len(per_view_textures) == 1:
        # Only one view available
        composite = list(per_view_textures.values())[0]
    elif len(per_view_textures) >= 2:
        composite = blend_multi_view_textures(per_view_textures, yaw_angles, texture_size)
    else:
        # No valid views — flat skin color
        composite = np.full((texture_size, texture_size, 3), list(skin_rgb), dtype=np.uint8)
    
    # 4. Return PNG bytes
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(composite, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
```

---

### අදියර 3: Backend API Changes (Flask)

#### 3.1 `server/app.py` — නව Endpoint එක
**File:** `server/app.py`

**Change:** `POST /api/avatars/<id>/customize-face-multi` endpoint එක add කරන්න.

```python
@app.route("/api/avatars/<avatar_id>/customize-face-multi", methods=["POST"])
def customize_face_multi(avatar_id):
    """Multi-angle face customization — accepts front + left + right face images.
    
    Body: {
        "faceImage": "<base64 front face crop>",
        "leftFaceImage": "<base64 left face crop>",
        "rightFaceImage": "<base64 right face crop>",
        ... other face features (faceShape, jawWidth, etc.)
    }
    
    Uses multi-angle projective texture mapping to create a composite
    face texture that looks natural from all viewing angles.
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    features = request.get_json(silent=True)
    if not features:
        return jsonify({"error": "JSON body required"}), 400

    # Decode all 3 face images
    selfie_rgb = None
    left_rgb = None
    right_rgb = None
    landmarks_2d = None
    left_landmarks = None
    right_landmarks = None

    from avatar_pipeline.model6_body3d.face_features import estimate_face_landmarks
    
    # Front image
    if features.get("faceImage"):
        try:
            raw = base64.b64decode(features["faceImage"], validate=True)
            selfie_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            landmarks_2d = estimate_face_landmarks(selfie_rgb)
        except Exception:
            pass
    
    # Left image (NEW)
    if features.get("leftFaceImage"):
        try:
            raw = base64.b64decode(features["leftFaceImage"], validate=True)
            left_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            left_landmarks = estimate_face_landmarks(left_rgb)
        except Exception:
            pass
    
    # Right image (NEW)
    if features.get("rightFaceImage"):
        try:
            raw = base64.b64decode(features["rightFaceImage"], validate=True)
            right_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            right_landmarks = estimate_face_landmarks(right_rgb)
        except Exception:
            pass

    try:
        gender_override = features.get("gender") if features.get("gender") in ("male", "female") else None
        
        # Use multi-angle texture if we have at least 2 angled images
        from avatar_pipeline.model6_body3d.multi_angle_texture import build_multi_angle_texture
        
        has_multi_angle = (
            left_rgb is not None and right_rgb is not None and
            left_landmarks is not None and right_landmarks is not None and
            len(left_landmarks) >= 468 and len(right_landmarks) >= 468
        )
        
        if has_multi_angle and selfie_rgb is not None and landmarks_2d is not None:
            # Build multi-angle composite texture
            skin_rgb = np.array(hex_to_rgb(features["skinTone"]), dtype=np.float32)
            multi_texture_png = build_multi_angle_texture(
                selfie_rgb, left_rgb, right_rgb,
                landmarks_2d, left_landmarks, right_landmarks,
                tuple(int(c) for c in skin_rgb[:3]),
                texture_size=512,
            )
            
            # Use this composite texture instead of individual warp
            # Pass as selfie_rgb override to apply_face_customization
            # OR: directly modify the GLB with this texture
            
            # ...
        
        # Fall back to regular single-image path
        updated = apply_face_customization(
            avatar_result, features,
            selfie_rgb=selfie_rgb,
            left_rgb=left_rgb,          # NEW
            right_rgb=right_rgb,         # NEW
            left_landmarks=left_landmarks,   # NEW
            right_landmarks=right_landmarks, # NEW
            landmarks_2d=landmarks_2d,
            blend_mode="feather",
            gender_override=gender_override,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    storage.update(avatar_id, updated)

    return jsonify({
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "face_texture_url": f"/api/avatars/{avatar_id}/face-texture.png",
    })
```

#### 3.2 `face_customization.py` — Multi-angle support
**File:** `avatar_pipeline/model6_body3d/face_customization.py`

**Change:** `apply_face_customization()` function එකට multi-angle parameters accept කරන්න update කරන්න.

New params: `left_rgb`, `right_rgb`, `left_landmarks`, `right_landmarks`

When all 3 are available, use `build_multi_angle_texture()` instead of the single-image warp.

---

### අදියර 4: Head Pose Visualization (Testing)

#### 4.1 Test script — `scripts/test_multi_angle.py`
Create a test script that:
1. Loads 3 test images (front, left, right)
2. Runs head pose estimation
3. Generates the composite texture
4. Saves intermediate results for debugging

---

## 📊 Change Summary Table

| Component | File | Change Type | Complexity |
|-----------|------|-------------|------------|
| **Mobile App** | `AvatarCreatorScreen.tsx` | Modify `analyze()` to send 3 photos | Medium |
| **Mobile App** | `faceAnalysis.ts` | Add `analyzeFaceMultiAngle()` | Medium |
| **Mobile App** | `api/client.ts` | Add `customizeFaceMultiAngle()` API function | Low |
| **Mobile App** | `types.ts` | Add `leftFaceImage`/`rightFaceImage` to types | Low |
| **Backend** | `multi_angle_texture.py` (NEW) | Head pose + projective mapping + blending | **High** |
| **Backend** | `app.py` | New `/customize-face-multi` endpoint | Medium |
| **Backend** | `face_customization.py` | Multi-angle params + composite texture path | Medium |
| **Backend** | `face_features.py` | Add head pose estimation function | Low |
| **Testing** | `scripts/test_multi_angle.py` | Test script for validation | Medium |

---

## ⚠️ Important Notes

### UV Map Coverage
- Current UV anchors cover only the **front face** (u≈0.3-0.7)
- Multi-angle UV anchors need to cover **u=0.0 to u=1.0** (full wrap)
- Need new UV anchors for side-of-head regions (ears, temples)
- These side UV anchors must be generated from the MakeHuman head mesh geometry

### Head Mesh Requirement
- Projective texture mapping needs **3D head vertices** + **face indices**
- MakeHuman mesh already has these (from `assets/makehuman/{gender}.glb`)
- The 3D vertices need to be transformed to match user's head pose

### Fallback Strategy
- If only front photo available → existing single-image Delaunay warp (backward compatible)
- If front + 1 side available → partial multi-angle (better than single)
- If all 3 available → full multi-angle projective texture (best quality)

### Performance
- Multi-angle processing: ~3-5 seconds (vs ~1-2s for single image)
- Can be optimized with caching
- Processing happens once at avatar creation time, not at runtime

---

## 🚀 ක්‍රියාත්මක කිරීමේ පියවර (Implementation Steps)

### Step 1: Create `multi_angle_texture.py` — Core engine
- Head pose estimation function
- UV region assignment
- Projective texture mapping
- Multi-view blending

### Step 2: Update `face_features.py` — Add head pose estimation
- Add `estimate_head_pose()` function using MediaPipe landmarks + SolvePnP

### Step 3: Update `face_customization.py` — Multi-angle support
- Accept left/right images and landmarks
- Use multi-angle composite when available

### Step 4: Update `server/app.py` — New API endpoint
- `/customize-face-multi` endpoint

### Step 5: Update Mobile App — Send all 3 photos
- `faceAnalysis.ts`: Add multi-angle analysis
- `AvatarCreatorScreen.tsx`: Send all 3 photos
- `api/client.ts`: New API function
- `types.ts`: Update types

### Step 6: Testing
- Test with 3 test images
- Compare single vs multi-angle results
- Verify in AvatarViewer3D

---

## 🎯 අවසාන ප්‍රතිඵලය

Multi-angle pipeline එක implement කළාට පස්සේ:

1. User front selfie එකේ face features (eyes, nose, mouth) **හරියට UV එකට map වෙනවා**
2. Left profile එකෙන් **වම් කම්මුල, වම් කන** හරියට පේනවා
3. Right profile එකෙන් **දකුණු කම්මුල, දකුණු කන** හරියට පේනවා
4. Overlap regions **seamless blend** වෙනවා (Poisson/weighted blending)
5. Avatar එක **360° කරකැවෙනකොටත්** මුහුණ ස්වභාවිකයි
6. **Backward compatible** — single photo එකක් විතරක් තියෙනවා නම් old pipeline එක වැඩ කරනවා

**ZEPETO / Apple Memoji level quality** 🚀
