"""
Model 7 — garment keypoint extraction from a segmented silhouette.

Phase 1 implementation: deterministic silhouette analysis. For each row of
the garment mask we compute the left/right silhouette edge (a "width
profile"), then locate landmarks as local extrema of that profile at
garment-type-specific height bands (e.g. a t-shirt's shoulders are the
widest row near the top; a pair of trousers' waistband is the widest row at
the very top; a dress's waist is the narrowest row in the lower-middle).

This is a classical silhouette heuristic, not a trained keypoint model — it
works because product garment photos are near-universally laid flat (or on
a mannequin/hanger) front-on with no significant perspective distortion.
Swapping in a trained pose/keypoint model (e.g. a MediaPipe-style landmark
network fine-tuned on garments) later only means replacing
`extract_keypoints`'s body; `garment_features.py` only depends on
`GarmentKeypoints`'s field names.
"""

from __future__ import annotations

import numpy as np

from .fitting_types import GarmentFittingError, GarmentKeypoints, SegmentationResult

# (y_start_frac, y_end_frac) height bands (0=top of bbox, 1=bottom), searched
# for the widest ("_MAX") or narrowest ("_MIN") row, per garment type + side.
_UPPER_BODY_BANDS = {
    "neck": (0.0, 0.08, "min"),
    "shoulder": (0.05, 0.22, "max"),
    "chest": (0.20, 0.45, "max"),
    "waist": (0.45, 0.70, "min"),
    "hem": (0.90, 1.0, "max"),
}
_DRESS_BANDS = {
    "neck": (0.0, 0.06, "min"),
    "shoulder": (0.04, 0.18, "max"),
    "chest": (0.18, 0.38, "max"),
    "waist": (0.38, 0.60, "min"),
    "hip": (0.55, 0.75, "max"),
    "hem": (0.92, 1.0, "max"),
}
_LOWER_BODY_BANDS = {
    "waist": (0.0, 0.08, "max"),
    "hip": (0.08, 0.28, "max"),
    "chest": (0.08, 0.28, "max"),  # unused conceptually for pants; mirrors hip
    "hem": (0.90, 1.0, "avg"),
}

_BANDS_BY_TYPE = {
    "upper_body": _UPPER_BODY_BANDS,
    "dress": _DRESS_BANDS,
    "lower_body": _LOWER_BODY_BANDS,
}


def _width_profile(mask: np.ndarray, bbox: tuple[int, int, int, int]):
    """Returns (rows_y, left_x, right_x, width) arrays, one entry per row of
    `mask` inside `bbox`, in original image pixel coordinates. Rows with no
    foreground pixel are omitted."""
    x_min, y_min, x_max, y_max = bbox
    rows_y, left_x, right_x, width = [], [], [], []
    for y in range(y_min, y_max + 1):
        xs = np.where(mask[y, x_min:x_max + 1])[0]
        if xs.size == 0:
            continue
        lo, hi = int(xs.min()) + x_min, int(xs.max()) + x_min
        rows_y.append(y)
        left_x.append(lo)
        right_x.append(hi)
        width.append(hi - lo)
    if not rows_y:
        raise GarmentFittingError("segmented garment mask has no rows with foreground pixels")
    return (
        np.array(rows_y), np.array(left_x), np.array(right_x), np.array(width, dtype=np.float64),
    )


def _band_indices(rows_y: np.ndarray, y_min: int, y_max: int, y0_frac: float, y1_frac: float) -> np.ndarray:
    span = max(y_max - y_min, 1)
    lo = y_min + y0_frac * span
    hi = y_min + y1_frac * span
    idx = np.where((rows_y >= lo) & (rows_y <= hi))[0]
    return idx


def _pick_row(width: np.ndarray, idx: np.ndarray, mode: str) -> int | None:
    if idx.size == 0:
        return None
    band_widths = width[idx]
    if mode == "max":
        return int(idx[np.argmax(band_widths)])
    if mode == "min":
        return int(idx[np.argmin(band_widths)])
    if mode == "avg":
        # Row whose width is closest to the band's mean, for a stable
        # "representative" landmark (used for pant-leg hems, which are two
        # separate blobs — a single min/max row would only capture one leg).
        target = band_widths.mean()
        return int(idx[np.argmin(np.abs(band_widths - target))])
    raise ValueError(f"unknown band mode {mode!r}")


def extract_keypoints(segmentation: SegmentationResult, garment_type: str, side: str) -> GarmentKeypoints:
    """Extracts silhouette-derived keypoints from `segmentation.mask`, using
    height bands tuned per `garment_type`. Returns None for any landmark
    whose band has no rows (e.g. a lower_body photo has no 'neck')."""
    bands = _BANDS_BY_TYPE.get(garment_type)
    if bands is None:
        raise GarmentFittingError(f"unknown garment_type {garment_type!r}")

    mask, bbox = segmentation.mask, segmentation.bbox
    x_min, y_min, x_max, y_max = bbox
    rows_y, left_x, right_x, width = _width_profile(mask, bbox)

    def landmark_row(name: str) -> int | None:
        band = bands.get(name)
        if band is None:
            return None
        y0f, y1f, mode = band
        idx = _band_indices(rows_y, y_min, y_max, y0f, y1f)
        return _pick_row(width, idx, mode)

    def left_right_at(row_idx: int | None):
        if row_idx is None:
            return None, None
        y = int(rows_y[row_idx])
        return (float(left_x[row_idx]), float(y)), (float(right_x[row_idx]), float(y))

    neck_row = landmark_row("neck")
    shoulder_row = landmark_row("shoulder")
    chest_row = landmark_row("chest")
    waist_row = landmark_row("waist")
    hip_row = landmark_row("hip")
    hem_row = landmark_row("hem")

    left_shoulder, right_shoulder = left_right_at(shoulder_row)
    left_chest, right_chest = left_right_at(chest_row)
    left_waist, right_waist = left_right_at(waist_row)
    left_hip, right_hip = left_right_at(hip_row)
    left_hem, right_hem = left_right_at(hem_row)
    left_neck, right_neck = left_right_at(neck_row)

    neck_center = None
    if neck_row is not None:
        y = float(rows_y[neck_row])
        neck_center = (float((left_x[neck_row] + right_x[neck_row]) / 2.0), y)

    hem_center = None
    if hem_row is not None:
        y = float(rows_y[hem_row])
        hem_center = (float((left_x[hem_row] + right_x[hem_row]) / 2.0), y)

    # Sleeve ends: the overall leftmost/rightmost foreground pixel in the
    # image (a t-shirt/dress laid flat with sleeves out extends furthest
    # sideways at the cuff, not at the shoulder seam).
    left_sleeve_end = (float(left_x.min()), float(rows_y[int(np.argmin(left_x))]))
    right_sleeve_end = (float(right_x.max()), float(rows_y[int(np.argmax(right_x))]))
    if garment_type == "lower_body":
        left_sleeve_end = None
        right_sleeve_end = None

    top_center = (float((left_x[0] + right_x[0]) / 2.0), float(rows_y[0]))
    bottom_center = (float((left_x[-1] + right_x[-1]) / 2.0), float(rows_y[-1]))

    return GarmentKeypoints(
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_neck=left_neck,
        right_neck=right_neck,
        neck_center=neck_center,
        left_chest=left_chest,
        right_chest=right_chest,
        left_hem=left_hem,
        right_hem=right_hem,
        hem_center=hem_center,
        left_waist=left_waist,
        right_waist=right_waist,
        left_hip=left_hip,
        right_hip=right_hip,
        left_sleeve_end=left_sleeve_end,
        right_sleeve_end=right_sleeve_end,
        top_center=top_center,
        bottom_center=bottom_center,
    )
