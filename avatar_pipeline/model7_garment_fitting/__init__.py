"""
Model 7 — AI-based personalized 3D garment fitting.

Replaces the catalog-based garment mapping in `model6_body3d.garment_mesh`
with an adaptive pipeline driven by user-uploaded front/back garment
photos: background removal -> segmentation -> keypoint extraction ->
normalized feature extraction -> region-wise adaptive scaling ->
Blender-based 3D fitting. See `pipeline.run_garment_fitting` for the entry
point, and `fitting_types.py` for the shared data types.
"""

from .fitting_types import GARMENT_PIPELINE_MODE, PIPELINE_MODES, GarmentFittingError, GarmentFittingResult
from .pipeline import run_garment_fitting

__all__ = [
    "run_garment_fitting", "GarmentFittingError", "GarmentFittingResult",
    "GARMENT_PIPELINE_MODE", "PIPELINE_MODES",
]
