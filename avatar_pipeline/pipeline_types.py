"""
Shared data types for the avatar pipeline, plus the operations on them that
need no TensorFlow: Model 5's TPS-based try-on (`dress_avatar`) and
`AvatarResult` persistence.

Splitting these out from controller.py means a lightweight "dress avatar"
service — which only needs a previously-cached `AvatarResult` and Model 5 —
can import this module without pulling in Models 1/3/4's TensorFlow
dependencies. See controller.py / README.md for the full pipeline.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from avatar_pipeline.model5_tryon.predict import try_on_avatar


@dataclass
class AvatarResult:
    """Output of `controller.build_avatar()` — the user's reusable base
    avatar plus the intermediate results Model 5 (and the app UI, e.g. for
    showing the detected body shape / skin tone) need.

    `avatar_mesh_glb` (Model 6) is a complete `.glb` file (bytes) — see
    `model6_body3d.glb_export.mesh_to_glb_bytes` — and `body3d_params` is the
    `model6_body3d.params.PARAM_NAMES` dict it was built from.
    
    New fields include facial analysis results for realistic avatar generation:
    - gender: 'male', 'female', or 'neutral'
    - facial_analysis: comprehensive facial features dict
    """

    avatar_rgba: Image.Image
    body_shape: str
    body_shape_confidence: float
    keypoints_dict: dict
    skin_tone_result: dict
    avatar_mesh_glb: bytes
    body3d_params: dict
    gender: str = "neutral"
    facial_analysis: dict = None
    
    def __post_init__(self):
        if self.facial_analysis is None:
            self.facial_analysis = {
                "gender": self.gender,
                "age_group": "20s",
                "hair_color": "brown",
                "hair_style": "short",
                "facial_hair": "none",
                "eye_color": "brown",
                "face_shape": "oval",
                "confidence": 0.0,
            }


@dataclass
class ClothingItem:
    """A single recommended clothing item, as handed off by the
    recommendation team."""

    rgb: np.ndarray
    mask: np.ndarray
    category: str = "upper_body"


def dress_avatar(avatar_result, clothing_item):
    """Composites `clothing_item` onto `avatar_result.avatar_rgba` using
    Model 5's TPS warp (`try_on_avatar`). Returns an RGBA PIL.Image, same
    size as the base avatar."""
    avatar_rgba = np.array(avatar_result.avatar_rgba)
    return try_on_avatar(
        avatar_rgba,
        clothing_item.rgb,
        clothing_item.mask,
        avatar_result.keypoints_dict,
        category=clothing_item.category,
    )


def save_avatar_result(avatar_result, output_dir):
    """Persists an `AvatarResult` (base avatar PNG + 3D mesh GLB + JSON
    metadata) so `dress_avatar` can be called later — e.g. from a separate
    API request — without rerunning Models 1-6."""
    os.makedirs(output_dir, exist_ok=True)
    avatar_result.avatar_rgba.save(os.path.join(output_dir, "avatar.png"))
    with open(os.path.join(output_dir, "avatar_mesh.glb"), "wb") as f:
        f.write(avatar_result.avatar_mesh_glb)
    meta = {
        "body_shape": avatar_result.body_shape,
        "body_shape_confidence": avatar_result.body_shape_confidence,
        "keypoints_dict": avatar_result.keypoints_dict,
        "skin_tone_result": avatar_result.skin_tone_result,
        "body3d_params": avatar_result.body3d_params,
        "gender": avatar_result.gender,
        "facial_analysis": avatar_result.facial_analysis,
    }
    with open(os.path.join(output_dir, "avatar_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_avatar_result(output_dir):
    avatar_rgba = Image.open(os.path.join(output_dir, "avatar.png")).convert("RGBA")
    with open(os.path.join(output_dir, "avatar_mesh.glb"), "rb") as f:
        avatar_mesh_glb = f.read()
    with open(os.path.join(output_dir, "avatar_meta.json")) as f:
        meta = json.load(f)
    
    # Handle both old and new metadata formats
    gender = meta.get("gender", "neutral")
    facial_analysis = meta.get("facial_analysis")
    
    return AvatarResult(
        avatar_rgba=avatar_rgba,
        avatar_mesh_glb=avatar_mesh_glb,
        gender=gender,
        facial_analysis=facial_analysis,
        **{k: v for k, v in meta.items() if k not in ["gender", "facial_analysis"]}
    )
