"""
Model 7 — garment segmentation: turns a background-removed mask into a
validated `SegmentationResult` (mask + tight bounding box), with basic
sanity checks that the segmentation is plausibly a single garment.

Phase 1 implementation wraps `background_removal.remove_background` — same
deterministic OpenCV approach, no learned segmentation model. Swapping in a
trained model (SAM / a garment-specific U-Net) later means replacing
`segment_garment`'s body; `GarmentKeypoints`/`garment_features.py` only
depend on `SegmentationResult.mask` + `.bbox`.
"""

from __future__ import annotations

import numpy as np

from .background_removal import remove_background
from .fitting_types import GarmentFittingError, SegmentationResult

# A garment silhouette that covers less than this fraction of the image is
# almost certainly a failed segmentation (background misclassified as
# foreground, or the actual garment is too small/off-frame).
_MIN_COVERAGE = 0.02
_MAX_COVERAGE = 0.98


def segment_garment(rgb: np.ndarray, side: str) -> SegmentationResult:
    """Segments the garment out of `rgb` (HxWx3 uint8). Raises
    `GarmentFittingError` if the resulting mask is implausible (empty, the
    whole frame, or too fragmented to be a single garment)."""
    mask = remove_background(rgb)

    coverage = float(mask.mean())
    if coverage < _MIN_COVERAGE:
        raise GarmentFittingError(
            f"could not detect a garment in the {side} image "
            f"(foreground covers only {coverage:.1%} of the frame)"
        )
    if coverage > _MAX_COVERAGE:
        raise GarmentFittingError(
            f"could not separate the garment from the background in the {side} image "
            f"(foreground covers {coverage:.1%} of the frame)"
        )

    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bbox_w < 8 or bbox_h < 8:
        raise GarmentFittingError(f"segmented garment region in the {side} image is too small to analyze")

    return SegmentationResult(mask=mask, bbox=bbox)
