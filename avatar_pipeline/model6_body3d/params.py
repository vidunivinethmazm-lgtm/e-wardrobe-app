"""
Model 6 — 3D Body Reconstruction: body-mesh parameter space with gender support.

The 3D avatar mesh (`mesh_builder.build_avatar_mesh`) is controlled by a
small set of named measurements, each expressed as the *full width
(diameter), as a fraction of total height* for the body part it controls.

Now supports gender-specific proportions:
- 'male': Wider shoulders, narrower hips, larger frame
- 'female': Narrower shoulders, wider hips, smaller frame  
- 'neutral': Balanced proportions (default)

`PARAM_NAMES` / `PARAM_DIM` define the regression target for Model 6's CNN.
`default_params_from_measurements` generates rule-based parameters from
measurements and detected gender, used both in training and directly in
`server.mock_pipeline`.
"""

import numpy as np

from avatar_pipeline.model4_avatar.condition_utils import (
    body_shape_to_onehot,
    keypoints_to_pose_vector,
)

# Normalizes bust/waist/hips/height (cm) into a roughly [0, 1] range for the
# "aux" input branch (architecture.py's `build_aux_branch`). 200cm comfortably
# covers adult human measurements/heights without any per-feature scaler.
MEASUREMENT_SCALE = 200.0

PARAM_NAMES = [
    "head_radius",
    "neck_radius",
    "shoulder_width",
    "chest_width",
    "chest_depth",
    "waist_width",
    "waist_depth",
    "hip_width",
    "hip_depth",
    "upper_arm_radius",
    "forearm_radius",
    "thigh_radius",
    "calf_radius",
]
PARAM_DIM = len(PARAM_NAMES)

# Plausible (min, max) for each parameter, as a fraction of total height.
PARAM_RANGES = {
    "head_radius": (0.10, 0.16),
    "neck_radius": (0.04, 0.09),
    "shoulder_width": (0.18, 0.32),
    "chest_width": (0.16, 0.32),
    "chest_depth": (0.08, 0.20),
    "waist_width": (0.12, 0.32),
    "waist_depth": (0.07, 0.20),
    "hip_width": (0.16, 0.34),
    "hip_depth": (0.08, 0.22),
    "upper_arm_radius": (0.03, 0.07),
    "forearm_radius": (0.025, 0.06),
    "thigh_radius": (0.05, 0.12),
    "calf_radius": (0.035, 0.08),
}

# Average head height is ~1/7.5 of total height ("7.5 heads tall" figure).
_HEAD_RADIUS_FRACTION = 1.0 / 7.5

# Shoulder width relative to chest (bust) width, by body shape — mirrors
# model4_avatar.synthetic_avatars.BODY_SHAPE_PROFILE's waist/hip multipliers.
_SHOULDER_FACTOR = {
    "Hourglass": 1.00,
    "Pear": 0.95,
    "Apple": 1.00,
    "Rectangle": 1.00,
    "InvertedTriangle": 1.12,
}

# Gender-specific multipliers for key proportions
_GENDER_FACTORS = {
    "male": {
        "shoulder_width_mult": 1.12,      # Males have wider shoulders
        "hip_width_mult": 0.88,           # Males have narrower hips
        "chest_width_mult": 1.08,         # Slightly wider chest
        "waist_depth_mult": 1.0,
        "upper_arm_radius_mult": 1.10,    # Larger arms
        "thigh_radius_mult": 1.05,
        "calf_radius_mult": 1.05,
        "head_radius_mult": 0.98,         # Slightly smaller head relative to body
    },
    "female": {
        "shoulder_width_mult": 0.90,      # Females have narrower shoulders
        "hip_width_mult": 1.12,           # Females have wider hips (curvy)
        "chest_width_mult": 0.95,         # Slightly narrower chest
        "waist_depth_mult": 1.02,
        "upper_arm_radius_mult": 0.92,    # Smaller arms
        "thigh_radius_mult": 0.95,
        "calf_radius_mult": 0.95,
        "head_radius_mult": 1.02,         # Slightly larger head relative to body
    },
    "neutral": {
        "shoulder_width_mult": 1.0,
        "hip_width_mult": 1.0,
        "chest_width_mult": 1.0,
        "waist_depth_mult": 1.0,
        "upper_arm_radius_mult": 1.0,
        "thigh_radius_mult": 1.0,
        "calf_radius_mult": 1.0,
        "head_radius_mult": 1.0,
    }
}


def _clip(name, value):
    lo, hi = PARAM_RANGES[name]
    return float(np.clip(value, lo, hi))


def default_params_from_measurements(body_shape, bust, waist, hips, height, gender="neutral"):
    """Rule-based body-mesh parameters from bust/waist/hips/height (cm),
    `body_shape` (one of model4_avatar.condition_utils.BODY_SHAPE_NAMES),
    and gender ('male', 'female', 'neutral').

    Approximates each torso cross-section as an ellipse with gender-specific
    proportions applied for anatomically correct avatars.
    """
    # Ensure gender is valid
    if gender not in _GENDER_FACTORS:
        gender = "neutral"
    
    gender_mult = _GENDER_FACTORS[gender]
    
    chest_width = _clip("chest_width", (bust / np.pi * 1.10) / height * gender_mult["chest_width_mult"])
    chest_depth = _clip("chest_depth", chest_width * 0.62)

    waist_width = _clip("waist_width", (waist / np.pi * 1.05) / height)
    waist_depth = _clip("waist_depth", waist_width * 0.68 * gender_mult["waist_depth_mult"])

    hip_width = _clip("hip_width", (hips / np.pi * 1.10) / height * gender_mult["hip_width_mult"])
    hip_depth = _clip("hip_depth", hip_width * 0.62)

    shoulder_factor = _SHOULDER_FACTOR.get(body_shape, 1.0) * gender_mult["shoulder_width_mult"]
    shoulder_width = _clip("shoulder_width", chest_width * shoulder_factor)

    neck_radius = _clip("neck_radius", shoulder_width * 0.22)
    head_radius = _clip("head_radius", _HEAD_RADIUS_FRACTION * gender_mult["head_radius_mult"])

    upper_arm_radius = _clip("upper_arm_radius", chest_width * 0.22 * gender_mult["upper_arm_radius_mult"])
    forearm_radius = _clip("forearm_radius", chest_width * 0.16 * gender_mult["upper_arm_radius_mult"])
    thigh_radius = _clip("thigh_radius", hip_width * 0.32 * gender_mult["thigh_radius_mult"])
    calf_radius = _clip("calf_radius", hip_width * 0.20 * gender_mult["calf_radius_mult"])

    return {
        "head_radius": head_radius,
        "neck_radius": neck_radius,
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "chest_depth": chest_depth,
        "waist_width": waist_width,
        "waist_depth": waist_depth,
        "hip_width": hip_width,
        "hip_depth": hip_depth,
        "upper_arm_radius": upper_arm_radius,
        "forearm_radius": forearm_radius,
        "thigh_radius": thigh_radius,
        "calf_radius": calf_radius,
    }


def params_to_vector(params):
    return np.array([params[name] for name in PARAM_NAMES], dtype=np.float32)


def params_to_sigmoid_vector(params):
    """Inverse of `vector_to_params(..., sigmoid_input=True)`: maps a
    physical params dict back to a (PARAM_DIM,) vector in [0, 1], i.e. the
    training target for Model 6's sigmoid output layer."""
    vector = np.zeros(PARAM_DIM, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        vector[i] = np.clip((params[name] - lo) / (hi - lo), 0.0, 1.0)
    return vector


def vector_to_params(vector, sigmoid_input=True):
    """vector: (PARAM_DIM,) array.

    If `sigmoid_input` (Model 6's raw output, each in [0, 1]), rescale each
    entry into its `PARAM_RANGES` interval. Otherwise treat `vector` as
    already-physical fractions of height (e.g. round-tripping
    `params_to_vector`'s output).
    """
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        value = float(vector[i])
        if sigmoid_input:
            lo, hi = PARAM_RANGES[name]
            value = lo + value * (hi - lo)
        params[name] = value
    return params
