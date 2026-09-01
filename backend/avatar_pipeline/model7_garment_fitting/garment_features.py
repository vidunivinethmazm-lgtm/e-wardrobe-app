"""
Model 7 — normalized garment feature extraction.

Combines the front + back `GarmentKeypoints` (see `garment_keypoints.py`)
into `NormalizedGarmentFeatures`: 8 *dimensionless* ratios, each a garment
dimension divided by the garment's own bounding diagonal.

Scientific constraint (see `fitting_types.py` module docstring): a single
2D photo cannot yield centimetre-accurate measurements. Every value
returned here is therefore a unitless proportion in (0, 1] relative to the
garment's own silhouette — never a claimed physical length. Region-wise
avatar-fitting scale factors (which *do* need to relate the garment to an
avatar's actual body proportions) are derived downstream in
`region_scaling.py`, by comparing these ratios against the avatar's own
`body3d_params` ratios (also normalized, as fractions of height) — no
absolute unit ever enters the pipeline.
"""

from __future__ import annotations

import math

from .fitting_types import FEATURE_NAMES, GarmentFittingError, GarmentKeypoints, NormalizedGarmentFeatures

# A feature ratio outside this range signals a keypoint-extraction failure
# (e.g. background noise misread as a wide "shoulder" row) rather than a
# real garment — validated in `compute_normalized_features`.
_PLAUSIBLE_RATIO_RANGE = (0.0, 1.5)


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    x_min, y_min, x_max, y_max = bbox
    return math.hypot(x_max - x_min, y_max - y_min) or 1.0


def _average(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def compute_normalized_features(
    front_keypoints: GarmentKeypoints,
    back_keypoints: GarmentKeypoints,
    front_bbox: tuple[int, int, int, int],
    back_bbox: tuple[int, int, int, int],
    garment_type: str,
) -> NormalizedGarmentFeatures:
    """Averages the same measurement taken from the front and back image
    (when both are available) and normalizes by each image's own bounding
    diagonal, so front/back photos taken at different zoom levels are still
    comparable. Missing landmarks (e.g. no separate waist pinch on a
    straight-cut dress) fall back to the nearest available measurement
    rather than 0, so a garment lacking one silhouette feature still yields
    a complete, plausible feature vector.
    """
    front_diag = _bbox_diagonal(front_bbox)
    back_diag = _bbox_diagonal(back_bbox)

    def normalized(front_px: float | None, back_px: float | None) -> float | None:
        front_ratio = None if front_px is None else front_px / front_diag
        back_ratio = None if back_px is None else back_px / back_diag
        return _average(front_ratio, back_ratio)

    shoulder = normalized(
        _dist(front_keypoints.left_shoulder, front_keypoints.right_shoulder),
        _dist(back_keypoints.left_shoulder, back_keypoints.right_shoulder),
    )
    chest = normalized(
        _dist(front_keypoints.left_chest, front_keypoints.right_chest),
        _dist(back_keypoints.left_chest, back_keypoints.right_chest),
    )
    waist = normalized(
        _dist(front_keypoints.left_waist, front_keypoints.right_waist),
        _dist(back_keypoints.left_waist, back_keypoints.right_waist),
    )
    hip = normalized(
        _dist(front_keypoints.left_hip, front_keypoints.right_hip),
        _dist(back_keypoints.left_hip, back_keypoints.right_hip),
    )
    neck = normalized(
        _dist(front_keypoints.left_neck, front_keypoints.right_neck),
        _dist(back_keypoints.left_neck, back_keypoints.right_neck),
    )
    hem = normalized(
        _dist(front_keypoints.left_hem, front_keypoints.right_hem),
        _dist(back_keypoints.left_hem, back_keypoints.right_hem),
    )

    sleeve_front = _sleeve_length_px(front_keypoints)
    sleeve_back = _sleeve_length_px(back_keypoints)
    sleeve = normalized(sleeve_front, sleeve_back) if garment_type != "lower_body" else 0.0

    length_front = _dist(front_keypoints.top_center, front_keypoints.bottom_center)
    length_back = _dist(back_keypoints.top_center, back_keypoints.bottom_center)
    length = normalized(length_front, length_back)

    # Fallbacks: a garment silhouette always has *some* overall extent, so
    # `length` should never be missing; the rest fall back to a neighboring
    # region rather than an arbitrary 0, which would look like "no garment
    # there" instead of "not distinguishable from its neighbor".
    values = {
        "shoulder_width": shoulder if shoulder is not None else chest,
        "chest_width": chest if chest is not None else shoulder,
        "waist_width": waist if waist is not None else _average(chest, hip),
        "hip_width": hip if hip is not None else waist,
        "sleeve_length": sleeve if sleeve is not None else 0.0,
        "garment_length": length,
        "neck_width": neck if neck is not None else 0.0,
        "hem_width": hem if hem is not None else waist,
    }

    if values["garment_length"] is None:
        raise GarmentFittingError(
            "could not determine the garment's overall length from either image — "
            "check that the garment is clearly visible against its background"
        )

    for name in FEATURE_NAMES:
        if values[name] is None:
            values[name] = 0.0

    lo, hi = _PLAUSIBLE_RATIO_RANGE
    for name in FEATURE_NAMES:
        values[name] = float(min(max(values[name], lo), hi))

    return NormalizedGarmentFeatures(**values)


def _sleeve_length_px(keypoints: GarmentKeypoints) -> float | None:
    """Sleeve length: shoulder-to-cuff distance, using whichever side
    (left/right) has both points available (a folded/angled photo may only
    show one sleeve clearly)."""
    left = _dist(keypoints.left_shoulder, keypoints.left_sleeve_end)
    right = _dist(keypoints.right_shoulder, keypoints.right_sleeve_end)
    return _average(left, right)
