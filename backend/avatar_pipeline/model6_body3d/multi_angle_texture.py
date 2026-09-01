"""
Multi-Angle Face Texture Pipeline

Front + Left + Right ඡායාරූප 3ක් භාවිතා කරලා, එක් එක් angle එකෙන්
හොඳම quality එක තියෙන face region එක UV texture එකට map කරලා,
overlap regions seamless ලෙස blend කරනවා.

Result එක: 360° කරකැවෙනකොටත් ස්වභාවිකව පෙනෙන face texture එකක්.

Pipeline:
  1. Head Pose Estimation (MediaPipe SolvePnP → Yaw angle)
  2. Per-view Delaunay warp (reuses existing warp_face_to_uv)
  3. View-dependent weighted blending based on UV region + yaw angle
  4. Final composite UV texture (single 512×512 PNG)
"""

from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Head Pose Estimation — MediaPipe SolvePnP
# ---------------------------------------------------------------------------
# 6 key landmark indices used for Perspective-n-Point (PnP) solving.
# These correspond to canonical 3D face model coordinates defined in
# the MediaPipe face mesh topology.

_FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],           # 1  — nose tip
    [0.0, -330.0, -65.0],      # 152 — chin
    [-225.0, 170.0, -135.0],   # 33  — left eye outer corner
    [225.0, 170.0, -135.0],    # 263 — right eye outer corner
    [-150.0, -150.0, -125.0],  # 61  — left mouth corner
    [150.0, -150.0, -125.0],   # 291 — right mouth corner
], dtype=np.float64)

_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(
    landmarks_2d: np.ndarray,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate head pose (yaw, pitch, roll) from MediaPipe 2D landmarks.

    Uses OpenCV SolvePnP with the 6 key facial points (nose tip, chin,
    eye corners, mouth corners) and a canonical 3D face model.

    Parameters
    ----------
    landmarks_2d : (N, 2) float32
        MediaPipe face-mesh landmark pixel positions.
    image_w, image_h : int
        Image dimensions for camera intrinsic approximation.

    Returns
    -------
    yaw : float
        Degrees — positive = looking right (counter-clockwise from above).
    pitch : float
        Degrees — positive = looking up.
    roll : float
        Degrees — positive = tilting right.
    rvec : (3,) float64 or None
        Rotation vector from SolvePnP.
    tvec : (3,) float64 or None
        Translation vector from SolvePnP.
    """
    if landmarks_2d is None or len(landmarks_2d) < 4:
        return 0.0, 0.0, 0.0, None, None

    # Build camera intrinsic matrix (approximate)
    focal_length = max(image_w, image_h)
    center = (image_w / 2.0, image_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Collect 2D image points for available landmarks
    img_pts = []
    model_pts = []
    for i, mp_idx in enumerate(_POSE_LANDMARK_INDICES):
        if mp_idx < len(landmarks_2d):
            img_pts.append(landmarks_2d[mp_idx])
            model_pts.append(_FACE_MODEL_3D[i])

    if len(img_pts) < 4:
        return 0.0, 0.0, 0.0, None, None

    img_pts = np.array(img_pts, dtype=np.float64)
    model_pts = np.array(model_pts, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0, None, None

    # Convert rotation vector → Euler angles (yaw, pitch, roll)
    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)

    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])  # pitch
        y = np.arctan2(-rot_mat[2, 0], sy)              # yaw
        z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])   # roll
    else:
        x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = 0.0

    pitch = float(np.degrees(x))
    yaw = float(np.degrees(y))
    roll = float(np.degrees(z))

    return yaw, pitch, roll, rvec, tvec


# ---------------------------------------------------------------------------
# UV Region Definitions
# ---------------------------------------------------------------------------
# එක් එක් angle එකට UV map එකේ කොටසක් assign කරනවා.
# Yaw angle determines which UV region each view contributes to.

# Each region: (u_start, u_end) — the UV-u range this view dominates.
# Blend weights: how much each view contributes to each region.

_UV_REGIONS = {
    "left": {
        "yaw_center": -30.0,       # ideal yaw for left profile
        "yaw_range": (-70, -10),   # acceptable yaw range
        "u_range": (0.0, 0.30),    # UV-u range this view covers best
        "sigma": 0.15,             # Gaussian falloff for blending
    },
    "front": {
        "yaw_center": 0.0,
        "yaw_range": (-15, 15),
        "u_range": (0.25, 0.75),
        "sigma": 0.20,
    },
    "right": {
        "yaw_center": 30.0,
        "yaw_range": (10, 70),
        "u_range": (0.70, 1.0),
        "sigma": 0.15,
    },
}


def _compute_view_weight_shape(
    texture_size: int,
    u_center: float,
    sigma: float,
) -> np.ndarray:
    """Create a 2D weight map for a single view.

    Higher weight near *u_center*, lower weight further away.
    Uses a Gaussian falloff across the U axis (uniform along V).

    Returns
    -------
    (texture_size, texture_size) float32 weight map (0..1).
    """
    u_coords = np.linspace(0.0, 1.0, texture_size, dtype=np.float32)
    # Gaussian centered on u_center
    weights = np.exp(-0.5 * ((u_coords - u_center) / sigma) ** 2)
    # Broadcast to 2D: same weight for all V rows
    return np.tile(weights, (texture_size, 1))


def _match_view_to_region(yaw: float) -> str:
    """Match a detected yaw angle to the closest UV region.

    Returns region name: 'left', 'front', or 'right'.
    """
    if yaw is None:
        return "front"
    best = min(_UV_REGIONS.keys(),
               key=lambda r: abs(yaw - _UV_REGIONS[r]["yaw_center"]))
    return best


# ---------------------------------------------------------------------------
# Multi-View Blending
# ---------------------------------------------------------------------------

def blend_multi_view_textures(
    per_view_textures: dict[str, np.ndarray],
    yaw_angles: dict[str, float],
    texture_size: int = 512,
) -> np.ndarray:
    """Blend multiple view textures into one composite UV texture.

    Strategy: එක් එක් view එකට UV map එකේ specific region එකක් assign
    කරලා, overlap regions වලදී Gaussian-weighted blending එකක් කරනවා.
    Front = center UV, Left = left UV, Right = right UV.

    Parameters
    ----------
    per_view_textures : dict
        {"front": (S,S,3), "left": (S,S,3), "right": (S,S,3)} warped textures.
        Missing entries are skipped.
    yaw_angles : dict
        {"front": yaw, "left": yaw, "right": yaw} in degrees.
    texture_size : int
        Output texture resolution (square).

    Returns
    -------
    (texture_size, texture_size, 3) uint8 composite texture.
    """
    if not per_view_textures:
        return np.full((texture_size, texture_size, 3), 128, dtype=np.uint8)

    # If only one view, return it directly
    if len(per_view_textures) == 1:
        return list(per_view_textures.values())[0]

    composite = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    total_weight = np.zeros((texture_size, texture_size), dtype=np.float32)

    for view_name, texture in per_view_textures.items():
        if texture is None:
            continue

        # Determine which UV region this view maps to
        yaw = yaw_angles.get(view_name, 0.0)
        region_name = _match_view_to_region(yaw)
        region_cfg = _UV_REGIONS.get(region_name, _UV_REGIONS["front"])

        # Create weight map: Gaussian centered on this region's ideal u
        u_center = (region_cfg["u_range"][0] + region_cfg["u_range"][1]) / 2.0
        weight_map = _compute_view_weight_shape(
            texture_size, u_center, region_cfg["sigma"]
        )

        # Accumulate weighted texture
        for c in range(3):
            composite[..., c] += texture[..., c].astype(np.float32) * weight_map
        total_weight += weight_map

    # Normalize
    total_weight = np.clip(total_weight, 1e-6, None)
    for c in range(3):
        composite[..., c] = np.clip(
            composite[..., c] / total_weight, 0, 255
        )

    return composite.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main Pipeline Entry Point
# ---------------------------------------------------------------------------

def build_multi_angle_texture(
    front_image: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    front_landmarks: np.ndarray,
    left_landmarks: np.ndarray,
    right_landmarks: np.ndarray,
    skin_rgb: tuple[int, int, int],
    texture_size: int = 512,
    blend_mode: str = "feather",
) -> bytes:
    """Main entry point: process 3 angle images → composite UV texture PNG.

    ක්‍රියා කරන විදිය:
    1. එක් එක් image එකේ Head Pose (yaw) detect කරනවා
    2. එක් එක් image එක Delaunay warp කරලා UV space එකට map කරනවා
       (reuses ``face_texture_builder.warp_face_to_uv``)
    3. තුන් view එකම weighted blending කරලා composite texture එක හදනවා
    4. PNG bytes විදියට return කරනවා

    Parameters
    ----------
    front_image : (H, W, 3) uint8
        Front-facing selfie.
    left_image : (H, W, 3) uint8
        Left profile photo.
    right_image : (H, W, 3) uint8
        Right profile photo.
    front_landmarks : (N, 2) float32
        MediaPipe landmarks for front image.
    left_landmarks : (N, 2) float32
        MediaPipe landmarks for left image.
    right_landmarks : (N, 2) float32
        MediaPipe landmarks for right image.
    skin_rgb : (R, G, B) 0-255
        Skin color for canvas background.
    texture_size : int
        Output texture resolution (square). Default 512.
    blend_mode : str
        ``"feather"`` or ``"poisson"`` — passed to ``blend_face_with_skin``.

    Returns
    -------
    png_bytes : bytes
        PNG-encoded composite texture, ready for GLB embedding.
    """
    from .face_texture_builder import (
        warp_face_to_uv,
        blend_face_with_skin,
        _FACE_UV_ANCHORS_MAKEHUMAN,
    )

    images = {
        "front": (front_image, front_landmarks),
        "left": (left_image, left_landmarks),
        "right": (right_image, right_landmarks),
    }

    # ── Step 1: Estimate head pose for each image ──
    yaw_angles: dict[str, float] = {}
    for name, (img, lm) in images.items():
        if img is not None and lm is not None and len(lm) >= 468:
            h, w = img.shape[:2]
            yaw, pitch, roll, _, _ = estimate_head_pose(lm, w, h)
            yaw_angles[name] = yaw
            print(f"[multi_angle] {name}: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
        else:
            yaw_angles[name] = 0.0
            print(f"[multi_angle] {name}: skipped (landmarks={lm.shape if lm is not None else 'None'})")

    # ── Step 2: Per-view Delaunay warp ──
    per_view_textures: dict[str, np.ndarray] = {}
    for name, (img, lm) in images.items():
        if img is None or lm is None or len(lm) < 468:
            continue

        # Warp this view to UV space
        warped = warp_face_to_uv(
            img, lm,
            img_size=texture_size,
            uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN,
        )

        # Blend with skin tone (fills gaps where no landmarks cover)
        blended = blend_face_with_skin(warped, skin_rgb, blend_mode=blend_mode)
        per_view_textures[name] = blended

        print(f"[multi_angle] {name}: warped ({np.count_nonzero(np.max(warped, axis=2))} non-zero pixels)")

    # ── Step 3: Blend multi-view textures ──
    if len(per_view_textures) == 0:
        # No valid views — flat skin color
        print("[multi_angle] No valid views — returning flat skin texture")
        flat = np.full((texture_size, texture_size, 3), list(skin_rgb), dtype=np.uint8)
        return _to_png(flat)

    composite = blend_multi_view_textures(
        per_view_textures, yaw_angles, texture_size
    )

    # ── Step 4: Return PNG bytes ──
    print(f"[multi_angle] Composite texture created: {len(per_view_textures)} views blended")
    return _to_png(composite)


def _to_png(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
