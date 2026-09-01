"""
Model 7 — garment feature reference tables.

NOTE: this module no longer supplies a *mesh* — the actual garment mesh now
always comes from `garment_mesh_generation.py` (Unique3D, or the explicitly
non-production `MockGarmentMeshProvider`), never a category template (see
that module's docstring for why: a generic shirt/dress/pants shell must
never stand in for the uploaded garment's real geometry/colour/pattern).

What's left here is purely descriptive: `TEMPLATE_FEATURES` /
`REFERENCE_BODY_PARAMS` are reference numbers `region_scaling.py` uses for
an independent, diagnostic-only 2D-photo-based `NormalizedGarmentFeatures`
comparison (surfaced in the API response's `garment_features` field for
research-demo purposes) — unrelated to, and not used by, the actual
mesh-fitting pipeline in `pipeline.py` (see `garment_region_fitting.py` for
that).
"""

from __future__ import annotations

from .fitting_types import NormalizedGarmentFeatures

# The template garment's own proportions, in the same normalized units
# `garment_features.compute_normalized_features` produces (fraction of the
# garment's own bounding diagonal). A `region_scaling.compute_region_scales`
# call with a garment whose extracted features exactly equal these, worn by
# `REFERENCE_BODY_PARAMS`, produces all scales == 1.0 (no deformation).
TEMPLATE_FEATURES = {
    "upper_body": NormalizedGarmentFeatures(
        shoulder_width=0.55, chest_width=0.60, waist_width=0.56, hip_width=0.58,
        sleeve_length=0.45, garment_length=0.85, neck_width=0.18, hem_width=0.58,
    ),
    "lower_body": NormalizedGarmentFeatures(
        shoulder_width=0.0, chest_width=0.0, waist_width=0.50, hip_width=0.54,
        sleeve_length=0.0, garment_length=0.95, neck_width=0.0, hem_width=0.30,
    ),
    "dress": NormalizedGarmentFeatures(
        shoulder_width=0.50, chest_width=0.56, waist_width=0.48, hip_width=0.58,
        sleeve_length=0.30, garment_length=0.95, neck_width=0.16, hem_width=0.62,
    ),
}

# The reference avatar's `body3d_params` (fractions of total height, see
# `model6_body3d.params.PARAM_RANGES`) the template meshes above were
# authored to fit — roughly the midpoint of each parameter's plausible
# range for a "neutral" gender.
REFERENCE_BODY_PARAMS = {
    "head_radius": 0.130,
    "neck_radius": 0.065,
    "shoulder_width": 0.250,
    "chest_width": 0.240,
    "chest_depth": 0.150,
    "waist_width": 0.220,
    "waist_depth": 0.135,
    "hip_width": 0.250,
    "hip_depth": 0.150,
    "upper_arm_radius": 0.050,
    "forearm_radius": 0.043,
    "thigh_radius": 0.085,
    "calf_radius": 0.058,
}


def get_template_features(garment_type: str) -> NormalizedGarmentFeatures:
    if garment_type not in TEMPLATE_FEATURES:
        raise ValueError(f"unknown garment_type {garment_type!r}")
    return TEMPLATE_FEATURES[garment_type]
