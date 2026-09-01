"""
Model 6 — optional Unique3D-based 3D face fitting pipeline:

    User face images (front, left, right)
    -> face_mesh_generation.FaceMeshProvider.generate (image-to-3D face mesh)
    -> Generated 3D face mesh + landmarks
    -> face_mesh_fitting.extract_avatar_head_landmarks (existing avatar side)
    -> face_mesh_fitting.compute_scale_ratios (scale_x/y/z)
    -> face_mesh_fitting.align_face_mesh (scale, then landmark-weighted
       rigid alignment: eyes first, nose second, chin/jaw third)
    -> face_fit_runner.FaceFitRunner.fit (Blender region-limited blend +
       texture bake, or MockFaceFitRunner passthrough)
    -> FaceFittingResult (status + possibly-updated avatar GLB)

This is entirely additive and optional: it does not replace, call, or
modify `face_customization.apply_face_customization` (the existing face
texture-transfer pipeline) or `makehuman_mesh.py`/`avatar_builder.py` (the
existing measurement-based avatar body). `run_face_fitting` never changes
`avatar_result.body3d_params` or any body geometry — only, at most, the
face/head region of `avatar_result.avatar_mesh_glb`, and only when a real
`FaceFitRunner` backend actually modified it (`geometry_modified=True`).

If no face mesh provider is configured (`face_mesh_generation.
get_face_mesh_provider().generate(...)` returns `None`), `run_face_fitting`
returns a `status="unavailable"` result with the avatar completely
untouched — callers should keep using the existing
`apply_face_customization` texture-transfer path in that case. This
function never silently pretends a real 3D face was generated: `status`
always distinguishes "unavailable" (no provider), "fitted_metadata_only"
(fitting math ran, geometry not yet blended in — the mock backend), and
"ready" (a real Blender pass merged the fitted face into the avatar mesh).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .face_fit_runner import get_face_fit_runner
from .face_mesh_fitting import (
    FitTransform,
    align_face_mesh,
    extract_avatar_head_landmarks,
    compute_scale_ratios,
    landmarks_from_dict,
)
from .face_mesh_generation import FaceMeshProvider, get_face_mesh_provider

_HEIGHT_CM_BY_GENDER = {"male": 170, "female": 160, "neutral": 160}


@dataclass
class FaceFittingResult:
    status: str  # "unavailable" | "fitted_metadata_only" | "ready"
    avatar_result: object  # the (possibly updated) AvatarResult
    scale: tuple[float, float, float] | None = None
    transform: FitTransform | None = None
    warnings: list[str] = field(default_factory=list)


def run_face_fitting(
    avatar_result, front_rgb: np.ndarray,
    left_rgb: np.ndarray | None = None, right_rgb: np.ndarray | None = None,
    provider: FaceMeshProvider | None = None,
) -> FaceFittingResult:
    """Runs the optional Unique3D-based face fitting pipeline. `avatar_result`
    is read for `body3d_params`/`gender`/`avatar_mesh_glb` and, only on a
    "ready" result, replaced (via `dataclasses.replace`, never mutated in
    place) with an updated `avatar_mesh_glb` — `body3d_params` is always
    passed through unchanged, preserving the existing measurement-based
    body exactly.
    """
    provider = provider or get_face_mesh_provider()
    generated = provider.generate(front_rgb, left_rgb, right_rgb)

    if generated is None:
        return FaceFittingResult(
            status="unavailable", avatar_result=avatar_result,
            warnings=["no face mesh provider configured — use the existing face-texture-transfer path"],
        )

    height_cm = _HEIGHT_CM_BY_GENDER.get(avatar_result.gender, _HEIGHT_CM_BY_GENDER["neutral"])
    avatar_landmarks = extract_avatar_head_landmarks(avatar_result.body3d_params, height_cm)
    generated_landmarks = landmarks_from_dict(generated.landmarks)

    scale = compute_scale_ratios(avatar_landmarks, generated_landmarks)
    fitted = align_face_mesh(
        generated.vertices, generated.faces, generated_landmarks, avatar_landmarks, scale=scale,
    )

    runner = get_face_fit_runner()
    updated_glb, geometry_modified = runner.fit(
        avatar_result.avatar_mesh_glb, fitted, generated.texture_png,
    )

    if geometry_modified:
        updated_avatar = dataclasses.replace(avatar_result, avatar_mesh_glb=updated_glb)
        status = "ready"
        warnings = []
    else:
        # body3d_params/avatar_mesh_glb are untouched by design here — see
        # MockFaceFitRunner's docstring for why a not-yet-verified mesh
        # blend is never silently applied.
        updated_avatar = avatar_result
        status = "fitted_metadata_only"
        warnings = ["fitting transform computed but not yet blended into the avatar mesh (mock backend)"]

    return FaceFittingResult(
        status=status, avatar_result=updated_avatar, scale=scale, transform=fitted.transform, warnings=warnings,
    )
