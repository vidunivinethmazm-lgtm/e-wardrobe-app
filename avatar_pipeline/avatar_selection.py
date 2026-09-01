"""
Avatar Selection — ties Model 1's body-shape output into Model 6's existing
body3d_params system, and resolves which base GLB to load for the user's
gender + detected body shape.

Each of Model 1's 5 body-shape classes (Hourglass/Pear/Apple/Rectangle/
InvertedTriangle) now has its own dedicated GLB per gender (10 total - see
scripts/generate_makehuman_avatars.py's BODY_SHAPE_PRESETS +
scripts/bake_makehuman_morphs.py), copied to mobile/assets/avatars/ as flat
files named ``{gender}_{body_shape}.glb``. This replaces the earlier
5-class-to-3-size-category approximation (a BODY_SHAPE_TO_SIZE_CATEGORY
grouping onto slim/average/curvy mesh variants) - now that a real
shape-specific mesh exists per class, no grouping/approximation is needed,
and `body_shape` maps directly to a filename.

Known limitation, documented rather than chased further given the ship
deadline: Hourglass, Apple, and Rectangle are only reliably differentiated
in side/depth silhouette, not the front view (their front-view outlines
read as near-identical at a glance) - see the bust/waist target ceiling
research earlier in this effort (2-5cm max displacement per target, a fine
sculpting adjustment, not a dramatic macro-level change). Pear and
InvertedTriangle are front-view distinct and fully approved.
"""

from pathlib import Path

from .model1_body_shape.architecture import CLASS_NAMES as BODY_SHAPE_NAMES
from .model6_body3d.params import default_params_from_measurements


def select_avatar(gender, body_shape, bust, waist, hips, height, assets_dir="mobile/assets"):
    """gender: 'male', 'female', or 'neutral' (maps to 'male').
    body_shape: one of model1_body_shape's CLASS_NAMES — typically
        predict_body_shape(...)["body_shape"] from Step 2's trained model.
    bust/waist/hips/height: cm, from the user's profile.
    assets_dir: defaults to the actual app asset root (mobile/assets), where
        the 10 {gender}_{body_shape}.glb files now live.

    Returns:
        {
            "gender": resolved gender ("male"/"female"),
            "body_shape": the input body_shape (validated against BODY_SHAPE_NAMES),
            "base_glb_path": ".../avatars/<gender>_<body_shape>.glb",
            "base_glb_bytes": GLB file bytes, or None if the asset is missing,
            "body3d_params": dict of 13 morph params (model6_body3d.params.PARAM_NAMES)
        }
    """
    if body_shape not in BODY_SHAPE_NAMES:
        raise ValueError(f"Unknown body_shape {body_shape!r}, expected one of {BODY_SHAPE_NAMES}")

    resolved_gender = "male" if gender == "neutral" else gender

    glb_path = Path(assets_dir) / "avatars" / f"{resolved_gender}_{body_shape}.glb"
    base_glb_bytes = glb_path.read_bytes() if glb_path.exists() else None

    body3d_params = default_params_from_measurements(
        body_shape, bust, waist, hips, height, gender=resolved_gender
    )

    return {
        "gender": resolved_gender,
        "body_shape": body_shape,
        "base_glb_path": str(glb_path),
        "base_glb_bytes": base_glb_bytes,
        "body3d_params": body3d_params,
    }
