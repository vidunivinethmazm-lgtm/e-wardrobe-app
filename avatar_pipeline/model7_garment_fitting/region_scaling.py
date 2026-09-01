"""
Model 7 — region-wise adaptive scaling.

Turns `NormalizedGarmentFeatures` (the *extracted garment's* proportions)
plus an avatar's `body3d_params` (the *avatar's* proportions, both already
dimensionless — see `params.py` and `garment_features.py`) into six
independent per-region scale multipliers (`RegionScales`). This is
deliberately not a single global Scale X/Y/Z transform: a garment cut
generously through the chest but snug at the waist should scale those two
vertex groups differently once fitted to a specific body.

Each region's scale is the product of two independent ratios:

- `garment_factor` = how this garment's own region compares to the
  category's template garment (`garment_template.TEMPLATE_FEATURES`) — a
  garment cut wider in the chest than the template scales that region up
  regardless of who wears it.
- `avatar_factor` = how the target avatar's body region compares to the
  reference body the template garment fits (`garment_template.
  REFERENCE_BODY_PARAMS`) — a broader-shouldered avatar needs the shoulder
  region scaled up regardless of which garment is worn.

Both factors, and their product, are clamped to `_SCALE_RANGE` so a noisy
feature extraction (e.g. a background sliver misread as extra chest width)
can't produce a degenerate mesh deformation.
"""

from __future__ import annotations

from .fitting_types import NormalizedGarmentFeatures, RegionScales
from .garment_template import REFERENCE_BODY_PARAMS, get_template_features

# Deformation beyond this range (60% shrink .. 60% growth) is more likely a
# pipeline error than a real fit difference, and risks a self-intersecting
# mesh once Blender's Surface Deform / Shrinkwrap / Cloth stack runs on it.
_SCALE_RANGE = (0.6, 1.6)

# Avatar body3d_params proxy used for each garment region. `sleeve` has no
# direct body3d_params equivalent (no arm-length parameter), so it uses the
# average of the two arm *radius* params as the closest available proxy for
# "how much bigger this avatar's arms are than the reference" — an
# imperfect but documented approximation pending an arm-length parameter in
# Model 6.
_AVATAR_PARAM_FOR_REGION = {
    "shoulder": ("shoulder_width",),
    "chest": ("chest_width",),
    "waist": ("waist_width",),
    "hip": ("hip_width",),
    "sleeve": ("upper_arm_radius", "forearm_radius"),
}


def _clip(value: float) -> float:
    lo, hi = _SCALE_RANGE
    return float(min(max(value, lo), hi))


def _avatar_param_value(body3d_params: dict, region: str) -> float:
    names = _AVATAR_PARAM_FOR_REGION[region]
    values = [body3d_params[name] for name in names if name in body3d_params]
    if not values:
        return 1.0
    return sum(values) / len(values)


def _region_scale(
    garment_value: float, template_value: float,
    avatar_value: float, reference_value: float,
) -> float:
    garment_factor = garment_value / template_value if template_value > 1e-6 else 1.0
    avatar_factor = avatar_value / reference_value if reference_value > 1e-6 else 1.0
    return _clip(garment_factor * avatar_factor)


def compute_region_scales(
    features: NormalizedGarmentFeatures, body3d_params: dict, garment_type: str,
) -> RegionScales:
    """Computes independent shoulder/chest/waist/hip/sleeve/length scale
    multipliers for `blender_runner.fit_garment_mesh`'s region-wise vertex
    group scaling step."""
    template = get_template_features(garment_type)

    shoulder = _region_scale(
        features.shoulder_width, template.shoulder_width,
        _avatar_param_value(body3d_params, "shoulder"), REFERENCE_BODY_PARAMS["shoulder_width"],
    )
    chest = _region_scale(
        features.chest_width, template.chest_width,
        _avatar_param_value(body3d_params, "chest"), REFERENCE_BODY_PARAMS["chest_width"],
    )
    waist = _region_scale(
        features.waist_width, template.waist_width,
        _avatar_param_value(body3d_params, "waist"), REFERENCE_BODY_PARAMS["waist_width"],
    )
    hip = _region_scale(
        features.hip_width, template.hip_width,
        _avatar_param_value(body3d_params, "hip"), REFERENCE_BODY_PARAMS["hip_width"],
    )
    reference_sleeve = (REFERENCE_BODY_PARAMS["upper_arm_radius"] + REFERENCE_BODY_PARAMS["forearm_radius"]) / 2.0
    sleeve = _region_scale(
        features.sleeve_length, template.sleeve_length,
        _avatar_param_value(body3d_params, "sleeve"), reference_sleeve,
    ) if garment_type != "lower_body" else 1.0

    # `garment_length` is already height-relative on both sides (the
    # garment's own diagonal, the avatar's own height), so length_scale is
    # garment-factor-only — there is no separate avatar "length" parameter
    # to fold in (height is the normalizing unit itself, see params.py).
    length = _clip(
        features.garment_length / template.garment_length if template.garment_length > 1e-6 else 1.0
    )

    return RegionScales(
        shoulder_scale=shoulder,
        chest_scale=chest,
        waist_scale=waist,
        hip_scale=hip,
        sleeve_scale=sleeve,
        length_scale=length,
    )
