"""
Model 7 — Garment Fitting: shared dataclasses/types for the adaptive garment
fitting pipeline (research focus: "AI-Based Personalized 3D Garment Fitting
Using Automatic Garment Feature Extraction and Adaptive Region-Wise Scaling").

Pipeline stages (see `pipeline.run_garment_fitting`):

    front/back garment images
    -> background_removal.remove_background
    -> garment_segmentation.segment_garment
    -> garment_mesh_generation: Unique3D (or MockGarmentMeshProvider, dev-only)
       image-to-3D garment mesh generation -> project_front_back_texture
       -> validate_garment_mesh
    -> garment_region_fitting: avatar region landmarks + region-wise ratios
    -> garment_fit_runner: Blender region-wise fitting (or MockGarmentFitRunner)
    -> fitted garment .glb (the actual uploaded garment, not a template)

    (in parallel, purely diagnostic/descriptive — see garment_keypoints.py /
    garment_features.py / region_scaling.py: 2D-photo-derived
    NormalizedGarmentFeatures shown in the API response for research-demo
    purposes, unrelated to the mesh above)

Scientific constraint: a single 2D garment photo cannot yield centimetre-
accurate measurements (no depth, unknown camera distance/lens, garment can be
laid flat or worn, folded, etc). Every geometric quantity here is therefore a
*normalized, unitless* ratio (fraction of garment length or of image size),
never a claimed physical measurement — see `garment_features.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

GarmentType = str  # "dress" | "upper_body" | "lower_body"
VALID_GARMENT_TYPES = ("dress", "upper_body", "lower_body")

# Feature flag selecting the garment-fitting pipeline used by
# `POST /api/avatars/<id>/fit-garment` (see `server/app.py`). Default is the
# existing, production adaptive-template pipeline (`pipeline.py`); the
# experimental research pipeline (`multiview/pipeline.py`) is only used when
# a caller opts in, either via this env var or per-request `pipeline_mode`.
PIPELINE_MODES = ("adaptive_template", "multiview_tryon")
GARMENT_PIPELINE_MODE = os.environ.get("GARMENT_PIPELINE_MODE", "adaptive_template")

# Order matches the spec's required feature list.
FEATURE_NAMES = [
    "shoulder_width",
    "chest_width",
    "waist_width",
    "hip_width",
    "sleeve_length",
    "garment_length",
    "neck_width",
    "hem_width",
]

REGION_NAMES = ["shoulder", "chest", "waist", "hip", "sleeve", "length"]


@dataclass(frozen=True)
class GarmentImage:
    """A single decoded garment photo (front or back) plus its role."""

    rgb: "object"  # np.ndarray, HxWx3 uint8 — typed loosely to avoid a hard numpy import here
    side: str  # "front" | "back"


@dataclass(frozen=True)
class SegmentationResult:
    """Output of `garment_segmentation.segment_garment`."""

    mask: "object"  # np.ndarray, HxW bool — True where the garment is
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), pixels


@dataclass(frozen=True)
class GarmentKeypoints:
    """Pixel-space keypoints extracted from one garment side's silhouette.

    Every key maps to an (x, y) pixel tuple or None if it couldn't be
    located reliably (e.g. a dress front image has no separate waist
    pinch-in for a straight-cut silhouette).
    """

    left_shoulder: tuple[float, float] | None
    right_shoulder: tuple[float, float] | None
    left_neck: tuple[float, float] | None
    right_neck: tuple[float, float] | None
    neck_center: tuple[float, float] | None
    left_chest: tuple[float, float] | None
    right_chest: tuple[float, float] | None
    left_hem: tuple[float, float] | None
    right_hem: tuple[float, float] | None
    hem_center: tuple[float, float] | None
    left_waist: tuple[float, float] | None
    right_waist: tuple[float, float] | None
    left_hip: tuple[float, float] | None
    right_hip: tuple[float, float] | None
    left_sleeve_end: tuple[float, float] | None
    right_sleeve_end: tuple[float, float] | None
    top_center: tuple[float, float] | None
    bottom_center: tuple[float, float] | None


@dataclass(frozen=True)
class NormalizedGarmentFeatures:
    """Normalized (dimensionless) garment feature ratios — every value is a
    fraction of the garment's own bounding silhouette (width relative to
    max/shoulder width, length relative to total garment length), *not* a
    centimetre measurement. See module docstring."""

    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    sleeve_length: float
    garment_length: float
    neck_width: float
    hem_width: float

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class RegionScales:
    """Region-wise scale multipliers applied to an avatar's garment template,
    one per body region — *not* a single global Scale X/Y/Z transform."""

    shoulder_scale: float
    chest_scale: float
    waist_scale: float
    hip_scale: float
    sleeve_scale: float
    length_scale: float

    def as_dict(self) -> dict:
        return {
            "shoulder_scale": self.shoulder_scale,
            "chest_scale": self.chest_scale,
            "waist_scale": self.waist_scale,
            "hip_scale": self.hip_scale,
            "sleeve_scale": self.sleeve_scale,
            "length_scale": self.length_scale,
        }


@dataclass
class GarmentFittingResult:
    """End-to-end output of `pipeline.run_garment_fitting`."""

    fit_id: str
    garment_type: GarmentType
    features: NormalizedGarmentFeatures
    region_scales: RegionScales
    glb_bytes: bytes
    texture_png: bytes | None = None
    status: str = "ready"  # "ready" | "processing"
    warnings: list[str] = field(default_factory=list)
    # True when the garment mesh came from MockGarmentMeshProvider (no
    # Unique3D configured) and/or MockGarmentFitRunner (no Blender) —
    # callers (API response, mobile UI) must surface this, never present
    # mock output as the real fitted garment. See garment_mesh_generation.py.
    is_mock: bool = True


class GarmentFittingError(ValueError):
    """Raised for invalid/missing/inconsistent garment images or garment_type
    — caught by the Flask route and turned into a 400 response."""
