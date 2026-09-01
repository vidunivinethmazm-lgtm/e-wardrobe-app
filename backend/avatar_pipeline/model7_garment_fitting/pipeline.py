"""
Model 7 — end-to-end adaptive garment fitting pipeline.

    front/back garment images
    -> background_removal.remove_background
    -> garment_segmentation.segment_garment
    -> garment_mesh_generation.get_garment_mesh_provider().generate()
       (Unique3D image-to-3D garment mesh generation, or the explicitly
       non-production MockGarmentMeshProvider)
    -> garment_mesh_generation.project_front_back_texture
       (front photo -> front-facing surface, back photo -> rear surface)
    -> garment_mesh_generation.validate_garment_mesh
    -> garment_region_fitting.extract_avatar_region_landmarks +
       compute_region_fit_ratios
    -> garment_fit_runner.get_garment_fit_runner().fit()
       (Blender region-wise Surface Deform/Shrinkwrap/Cloth on the
       GENERATED mesh, or MockGarmentFitRunner's Python deformation)
    -> GarmentFittingResult (fit_id, features, region_scales, .glb bytes,
       is_mock)

`run_garment_fitting` is the single entry point `server/app.py`'s
`POST /api/avatars/<id>/fit-garment` route calls; the sub-stage modules stay
independently unit-testable. `garment_features`/`region_scaling.py`'s
2D-photo-derived diagnostics (`garment_features` in the API response) run
alongside the mesh pipeline above but do not drive it — see those modules'
docstrings.
"""

from __future__ import annotations

import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError

from .fitting_types import GarmentFittingError, GarmentFittingResult, VALID_GARMENT_TYPES
from .garment_features import compute_normalized_features
from .garment_fit_runner import get_garment_fit_runner
from .garment_keypoints import extract_keypoints
from .garment_mesh_generation import (
    get_garment_mesh_provider, project_front_back_texture, validate_garment_mesh,
)
from .garment_region_fitting import compute_region_fit_ratios, extract_avatar_region_landmarks
from .garment_segmentation import segment_garment

# Matches model6_body3d.garment_mesh._HEIGHT_CM_BY_GENDER — the fitted
# garment must scale to the same fixed per-gender height the body/other
# garments use, see that module's docstring.
_HEIGHT_CM_BY_GENDER = {"male": 170, "female": 160, "neutral": 160}

MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 6000


def decode_garment_image(file_storage, side: str) -> np.ndarray:
    """Validates and decodes an uploaded garment photo. Raises
    `GarmentFittingError` (-> 400 at the route level) for anything that
    isn't a readable, reasonably-sized RGB image."""
    if file_storage is None or not getattr(file_storage, "filename", None):
        raise GarmentFittingError(f"'garment_{side}' file is required")

    try:
        image = Image.open(file_storage.stream)
        image.load()
        rgb = np.array(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise GarmentFittingError(f"'garment_{side}' could not be read as an image") from exc

    h, w = rgb.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise GarmentFittingError(
            f"'garment_{side}' is too small ({w}x{h}px, minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}px)"
        )
    if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
        raise GarmentFittingError(
            f"'garment_{side}' is too large ({w}x{h}px, maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}px)"
        )

    return rgb


def validate_garment_type(garment_type: str | None) -> str:
    if garment_type not in VALID_GARMENT_TYPES:
        raise GarmentFittingError(
            f"'garment_type' must be one of {VALID_GARMENT_TYPES}, got {garment_type!r}"
        )
    return garment_type


def _check_front_back_consistency(front_bbox, back_bbox, side_a="front", side_b="back") -> list[str]:
    """Soft consistency check between the two images' segmented aspect
    ratios — a mismatch (e.g. the "back" photo is actually an unrelated
    item) doesn't fail the request, but is surfaced as a warning since the
    averaged features in that case are less trustworthy."""
    def aspect(bbox):
        x0, y0, x1, y1 = bbox
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        return w / h

    a, b = aspect(front_bbox), aspect(back_bbox)
    ratio = max(a, b) / max(min(a, b), 1e-6)
    if ratio > 2.5:
        return [
            f"the {side_a} and {side_b} garment images have very different proportions "
            "(they may not be the same garment) — extracted features may be unreliable"
        ]
    return []


def run_garment_fitting(
    front_bytes_storage,
    back_bytes_storage,
    garment_type: str | None,
    body3d_params: dict,
    gender: str = "neutral",
    avatar_mesh_glb: bytes = b"",
) -> GarmentFittingResult:
    """Runs the full pipeline on the uploaded front/back garment photos and
    returns a `GarmentFittingResult`. Raises `GarmentFittingError` for any
    invalid/missing/inconsistent input, for a garment mesh provider that's
    configured but fails, and — when `UNIQUE3D_ENABLED` is unset — the
    caller still gets a result, but `is_mock=True` throughout so it's never
    mistaken for a real Unique3D fit (see `garment_mesh_generation.py`).
    `avatar_mesh_glb` (the avatar's own personalized GLB) is required by the
    Blender fitting backend; the mock backend works without it.
    """
    garment_type = validate_garment_type(garment_type)

    front_rgb = decode_garment_image(front_bytes_storage, "front")
    back_rgb = decode_garment_image(back_bytes_storage, "back")

    front_seg = segment_garment(front_rgb, "front")
    back_seg = segment_garment(back_rgb, "back")

    warnings = _check_front_back_consistency(front_seg.bbox, back_seg.bbox)

    # ── Diagnostic 2D features (shown in the API response, doesn't drive
    #    the mesh below) ────────────────────────────────────────────────
    front_keypoints = extract_keypoints(front_seg, garment_type, "front")
    back_keypoints = extract_keypoints(back_seg, garment_type, "back")
    features = compute_normalized_features(
        front_keypoints, back_keypoints, front_seg.bbox, back_seg.bbox, garment_type,
    )

    # ── Real garment mesh: image-to-3D generation, never a template ────
    provider = get_garment_mesh_provider()
    generated_mesh = provider.generate(
        front_rgb, back_rgb, garment_type,
        front_mask=front_seg.mask, back_mask=back_seg.mask,
        front_bbox=front_seg.bbox, back_bbox=back_seg.bbox,
    )
    if generated_mesh is None:
        raise GarmentFittingError(
            "garment mesh generation is unavailable — set UNIQUE3D_ENABLED=1 with a configured "
            "UNIQUE3D_ENDPOINT to fit the actual uploaded garment, or use the explicitly "
            "non-production mock path for local testing"
        )

    generated_mesh = project_front_back_texture(generated_mesh, front_rgb, back_rgb)
    validate_garment_mesh(generated_mesh)

    # ── Region-wise fit onto the avatar's own body ──────────────────────
    height_cm = _HEIGHT_CM_BY_GENDER.get(gender, _HEIGHT_CM_BY_GENDER["neutral"])
    avatar_landmarks = extract_avatar_region_landmarks(body3d_params, height_cm)
    ratios = compute_region_fit_ratios(avatar_landmarks, generated_mesh.landmarks)

    runner = get_garment_fit_runner()
    glb_bytes, texture_png, used_blender = runner.fit(
        generated_mesh, avatar_mesh_glb, avatar_landmarks, ratios,
    )

    is_mock = generated_mesh.is_mock or not used_blender
    if is_mock:
        warnings = warnings + [
            "this is a non-production preview "
            + ("(mock garment mesh)" if generated_mesh.is_mock else "(Python-only fitting, no Blender)")
            + " — not the final production-quality fitted garment"
        ]

    return GarmentFittingResult(
        fit_id=uuid.uuid4().hex,
        garment_type=garment_type,
        features=features,
        region_scales=ratios,
        glb_bytes=glb_bytes,
        texture_png=texture_png,
        status="ready",
        warnings=warnings,
        is_mock=is_mock,
    )
