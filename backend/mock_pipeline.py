"""
TensorFlow-free stand-in for `avatar_pipeline.controller.build_avatar`
(Phase A — Models 1, 2, 3, 4, 6), built entirely from real, already-verified
utilities elsewhere in `avatar_pipeline/`.

Phase 1 (Default): Procedural gender-aware avatars
- Detects gender from facial analysis
- Applies gender-specific body proportions
- Generates 3D mesh procedurally
- Works without 3D assets

Phase 2 (Optional): Photorealistic avatars
- Requires base GLB models + hairstyle assets
- Enable with AVATAR_USE_REALISTIC=1 environment variable
- Combines realistic base models with hairstyles
- Uses detected gender and facial features

Set AVATAR_USE_REALISTIC=1 to enable Phase 2 (requires assets in /assets/).
"""

import os
import numpy as np

from backend.avatar_pipeline.model1_body_shape.synthetic_data import classify_body_shape
from backend.avatar_pipeline.model3_skin_tone.color_utils import (
    MONK_SKIN_TONE_PALETTE,
    dominant_skin_color,
    hex_to_rgb,
    nearest_palette_match,
    white_balance_scene,
)
from backend.avatar_pipeline.model3_skin_tone.face_crop import detect_and_crop_face
from backend.avatar_pipeline.model4_avatar.synthetic_avatars import CANONICAL_POSE, render_avatar
from backend.avatar_pipeline.model6_body3d.face_features import extract_face_features
from backend.avatar_pipeline.model6_body3d.glb_export import mesh_to_glb_bytes
from backend.avatar_pipeline.model6_body3d.mesh_builder import build_avatar_mesh
from backend.avatar_pipeline.model6_body3d.params import default_params_from_measurements
from backend.avatar_pipeline.model6_body3d.avatar_builder import create_avatar_builder
from backend.avatar_pipeline.pipeline_types import AvatarResult

# Used when no face is detected in the photo: a mid-tone Lab value roughly
# central in the Monk Skin Tone scale (~MST-5), so the avatar still renders
# with a plausible skin color rather than failing the whole request.
_FALLBACK_SKIN_LAB = np.array([70.0, 8.0, 20.0], dtype=np.float32)


def _shade(rgb, factor):
    """factor > 1 lightens, < 1 darkens (simple multiplicative shading)."""
    return tuple(int(np.clip(c * factor, 0, 255)) for c in rgb)


def _skin_tone_result(photo):
    """Mirrors model3_skin_tone.predict.predict_skin_tone's output shape
    using the real color-matching utilities directly, skipping the CNN."""
    balanced = white_balance_scene(photo)
    face = detect_and_crop_face(balanced, output_size=128)

    if face is None:
        lab = _FALLBACK_SKIN_LAB
    else:
        lab = dominant_skin_color(face)

    match = nearest_palette_match(lab, MONK_SKIN_TONE_PALETTE)
    base_rgb = hex_to_rgb(match["hex"])

    return {
        **match,
        "avatar_render": {
            "base_color": "#%02x%02x%02x" % base_rgb,
            "shadow_color": "#%02x%02x%02x" % _shade(base_rgb, 0.78),
            "highlight_color": "#%02x%02x%02x" % _shade(base_rgb, 1.15),
        },
    }


def build_avatar(photo, bust, waist, hips, height, silhouette_path=None, seed=None):
    """Same signature and return type (`AvatarResult`) as
    `controller.build_avatar`, computed without TensorFlow.

    Now includes gender detection and facial analysis for realistic avatars.
    `silhouette_path` and `seed` are accepted for interface compatibility
    with `server.app` but are not used by the rule-based classifier or the
    deterministic renderer. `height` sets the 3D mesh's scale (Model 6).
    
    Phase 1 (default): Procedural gender-aware avatars
    Phase 2 (optional): Photorealistic avatars (enable with AVATAR_USE_REALISTIC=1)
    """
    body_shape = classify_body_shape(bust, waist, hips)
    keypoints_dict = {name: list(coords) for name, coords in CANONICAL_POSE.items()}
    skin_tone_result = _skin_tone_result(photo)

    skin_rgb = np.array(hex_to_rgb(skin_tone_result["hex"]), dtype=np.float32)
    avatar_rgba = render_avatar(body_shape, CANONICAL_POSE, skin_rgb, img_size=128)

    # Extract comprehensive facial analysis + approximate landmarks (for
    # Delaunay-triangulation face-texture warping — see face_texture_builder.py)
    face_features = extract_face_features(photo, estimate_landmarks=True)
    facial_analysis = face_features.get("facial_analysis", {
        "gender": "neutral",
        "age_group": "20s",
        "hair_color": "brown",
        "hair_style": "short",
        "facial_hair": "none",
        "eye_color": "brown",
        "face_shape": "oval",
        "confidence": 0.0,
    })
    landmarks_2d = face_features.get("landmarks_2d", None)
    
    gender = facial_analysis.get("gender", "neutral")
    
    # Generate body parameters with gender-specific proportions
    body3d_params = default_params_from_measurements(body_shape, bust, waist, hips, height, gender=gender)
    face = extract_face_features(photo, estimate_landmarks=True)
    landmarks_2d = face.get("landmarks_2d", landmarks_2d)
    
    # Check if Phase 2 (realistic avatars) is enabled
    use_realistic = os.getenv("AVATAR_USE_REALISTIC", "0") == "1"
    
    # Shared kwargs for procedural avatar mesh (with Delaunay warping support)
    _hair_style = facial_analysis.get("hair_style", "medium")
    _mesh_kwargs = dict(
        face_crop=face["face_crop"],
        hair_rgb=face["hair_rgb"],
        selfie_rgb=photo,
        landmarks_2d=landmarks_2d,
        blend_mode="feather",
        hair_style=_hair_style,
    )
    
    if use_realistic:
        # Phase 2: Try to use realistic avatars
        avatar_builder = create_avatar_builder(use_realistic=True)
        if avatar_builder is not None:
            # Attempt to build realistic avatar
            realistic_glb = avatar_builder.build_realistic_avatar(
                gender=gender,
                facial_analysis=facial_analysis,
                body3d_params=body3d_params,
                height=height,
                skin_rgb=skin_rgb,
                hair_rgb=face["hair_rgb"],
                face_crop=face.get("face_crop", None),
                selfie_rgb=photo,
                landmarks_2d=landmarks_2d,
                blend_mode="feather",
            )
            
            if realistic_glb is not None:
                avatar_mesh_glb = realistic_glb
            else:
                # Fallback to procedural if realistic failed
                print("Warning: Realistic avatar generation failed, falling back to procedural")
                mesh = build_avatar_mesh(body3d_params, height, skin_rgb, **_mesh_kwargs)
                avatar_mesh_glb = mesh_to_glb_bytes(mesh)
        else:
            # Fallback to procedural if no builder
            mesh = build_avatar_mesh(body3d_params, height, skin_rgb, **_mesh_kwargs)
            avatar_mesh_glb = mesh_to_glb_bytes(mesh)
    else:
        # Phase 1: Use procedural avatars (default) with Delaunay warping
        mesh = build_avatar_mesh(body3d_params, height, skin_rgb, **_mesh_kwargs)
        avatar_mesh_glb = mesh_to_glb_bytes(mesh)

    return AvatarResult(
        avatar_rgba=avatar_rgba,
        body_shape=body_shape,
        body_shape_confidence=1.0,
        keypoints_dict=keypoints_dict,
        skin_tone_result=skin_tone_result,
        avatar_mesh_glb=avatar_mesh_glb,
        body3d_params=body3d_params,
        gender=gender,
        facial_analysis=facial_analysis,
    )
