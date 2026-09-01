"""
Model 7 — avatar-to-garment region-wise fitting math: compares the
avatar's own body region landmarks against the *generated garment mesh's*
landmarks (see `garment_mesh_generation.GeneratedGarmentMesh`) to compute
independent shoulder/chest/waist/hip/sleeve/length ratios, then applies a
region-weighted local deformation directly to the generated mesh's own
vertices — never swapping in a template mesh. Vertex count, face
connectivity, and UVs are untouched; only vertex *positions* move, blended
smoothly between named body regions so there's no hard seam between (e.g.)
the waist and hip deformation zones.

Mirrors `model6_body3d.face_mesh_fitting`'s avatar-landmark-extraction
pattern (there: the avatar's head; here: the avatar's torso/hip regions),
using `mesh_builder.py`'s own `ANCHORS` height fractions so the landmarks
line up with where the body mesh actually places those features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting_types import RegionScales

# Mirrors mesh_builder.ANCHORS (height fractions, 0=feet, 1=head-top) for
# the regions this module cares about. Duplicated rather than imported
# because ANCHORS is a private implementation detail of the procedural body
# mesh; this module only needs the same relative layout.
_ANCHOR_Y_FRACTION = {"shoulder": 0.82, "chest": 0.72, "waist": 0.60, "hip": 0.50}

_SCALE_RANGE = (0.6, 1.6)


@dataclass(frozen=True)
class AvatarRegionLandmarks:
    """The avatar's own 3D body-region landmarks, in meters — analogous to
    `GeneratedGarmentMesh.landmarks` but for the body being fitted."""

    left_shoulder: tuple[float, float, float]
    right_shoulder: tuple[float, float, float]
    left_chest: tuple[float, float, float]
    right_chest: tuple[float, float, float]
    left_waist: tuple[float, float, float]
    right_waist: tuple[float, float, float]
    left_hip: tuple[float, float, float]
    right_hip: tuple[float, float, float]
    sleeve_length: float
    body_length: float  # shoulder-to-hip vertical span, meters

    def as_region_centers(self) -> dict[str, np.ndarray]:
        """One representative 3D point per named region — the center used
        for proximity-weighted vertex blending in `region_deform_mesh`."""
        def mid(a, b):
            return np.array([(a[i] + b[i]) / 2.0 for i in range(3)])

        return {
            "shoulder": mid(self.left_shoulder, self.right_shoulder),
            "chest": mid(self.left_chest, self.right_chest),
            "waist": mid(self.left_waist, self.right_waist),
            "hip": mid(self.left_hip, self.right_hip),
        }


def extract_avatar_region_landmarks(body3d_params: dict, height_cm: float) -> AvatarRegionLandmarks:
    """Derives the avatar's shoulder/chest/waist/hip 3D landmarks (meters)
    from `body3d_params` (see `model6_body3d.params.PARAM_NAMES`), using the
    same height fractions `mesh_builder.py`'s `ANCHORS` places those
    features at. Read-only — never modifies `body3d_params` or any avatar
    geometry."""
    required = ("shoulder_width", "chest_width", "waist_width", "hip_width",
                "upper_arm_radius", "forearm_radius")
    missing = [name for name in required if name not in body3d_params]
    if missing:
        raise ValueError(f"body3d_params missing required keys: {missing}")

    height_m = height_cm / 100.0

    def side_points(param_name: str, region: str) -> tuple[tuple, tuple]:
        half_width = body3d_params[param_name] / 2.0 * height_m
        y = _ANCHOR_Y_FRACTION[region] * height_m
        return (-half_width, y, 0.0), (half_width, y, 0.0)

    left_shoulder, right_shoulder = side_points("shoulder_width", "shoulder")
    left_chest, right_chest = side_points("chest_width", "chest")
    left_waist, right_waist = side_points("waist_width", "waist")
    left_hip, right_hip = side_points("hip_width", "hip")

    # No direct arm-length parameter in Model 6 (see region_scaling.py's
    # same documented gap) — approximate sleeve length from the arm radii
    # and shoulder-to-hip span, a reasonable proportion for an average arm.
    body_length = abs(_ANCHOR_Y_FRACTION["shoulder"] - _ANCHOR_Y_FRACTION["hip"]) * height_m
    sleeve_length = (body3d_params["upper_arm_radius"] + body3d_params["forearm_radius"]) * height_m * 3.0

    return AvatarRegionLandmarks(
        left_shoulder=left_shoulder, right_shoulder=right_shoulder,
        left_chest=left_chest, right_chest=right_chest,
        left_waist=left_waist, right_waist=right_waist,
        left_hip=left_hip, right_hip=right_hip,
        sleeve_length=sleeve_length, body_length=body_length,
    )


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _clip(value: float) -> float:
    lo, hi = _SCALE_RANGE
    return float(min(max(value, lo), hi))


def compute_region_fit_ratios(avatar: AvatarRegionLandmarks, garment_landmarks: dict) -> RegionScales:
    """Computes independent shoulder/chest/waist/hip/sleeve/length ratios:
    `avatar_region_width / garment_region_width`. This is what actually
    drives `region_deform_mesh` below — a garment cut narrower through the
    chest than the avatar's own chest gets `chest_scale > 1` (stretched to
    fit), independent of every other region."""
    def region_ratio(avatar_left, avatar_right, garment_left_key, garment_right_key) -> float:
        avatar_width = _dist(avatar_left, avatar_right)
        garment_width = _dist(garment_landmarks[garment_left_key], garment_landmarks[garment_right_key])
        if garment_width <= 1e-6:
            return 1.0
        return _clip(avatar_width / garment_width)

    shoulder = region_ratio(avatar.left_shoulder, avatar.right_shoulder, "left_shoulder", "right_shoulder")
    chest = region_ratio(avatar.left_chest, avatar.right_chest, "left_chest", "right_chest")
    waist = region_ratio(avatar.left_waist, avatar.right_waist, "left_waist", "right_waist")
    hip = region_ratio(avatar.left_hip, avatar.right_hip, "left_hip", "right_hip")

    garment_sleeve = _dist(garment_landmarks["left_shoulder"], garment_landmarks["left_sleeve_end"])
    sleeve = _clip(avatar.sleeve_length / garment_sleeve) if garment_sleeve > 1e-6 else 1.0

    garment_length = _dist(garment_landmarks["top_center"], garment_landmarks["bottom_center"])
    length = _clip(avatar.body_length / garment_length) if garment_length > 1e-6 else 1.0

    return RegionScales(
        shoulder_scale=shoulder, chest_scale=chest, waist_scale=waist,
        hip_scale=hip, sleeve_scale=sleeve, length_scale=length,
    )


_REGION_SCALE_KEY = {"shoulder": "shoulder_scale", "chest": "chest_scale", "waist": "waist_scale", "hip": "hip_scale"}


def region_deform_mesh(
    vertices: np.ndarray, avatar: AvatarRegionLandmarks, ratios: RegionScales,
) -> np.ndarray:
    """Applies `ratios` directly to `vertices` (the generated garment
    mesh's own vertices — same array shape in, same shape out, topology
    untouched) via inverse-distance blending between the four named region
    centers (shoulder/chest/waist/hip): each vertex's effective scale is a
    weighted average of the regions it's closest to, so there's a smooth
    transition rather than a hard boundary between (e.g.) waist- and
    hip-scaled zones. Scaling is applied radially from the garment's own
    central vertical (Y) axis, plus a uniform `length_scale` stretch along
    Y from the mesh's vertical center — preserves the garment's silhouette
    shape while resizing it region-by-region.
    """
    centers = avatar.as_region_centers()
    region_names = list(centers.keys())
    center_array = np.stack([centers[name] for name in region_names])  # (4, 3)
    scale_by_region = np.array([getattr(ratios, _REGION_SCALE_KEY[name]) for name in region_names])

    # Inverse-distance weights (small epsilon avoids divide-by-zero exactly
    # at a region center) -> each vertex's blended scale factor.
    dists = np.linalg.norm(vertices[:, None, :] - center_array[None, :, :], axis=2)  # (N, 4)
    weights = 1.0 / (dists + 1e-3)
    weights /= weights.sum(axis=1, keepdims=True)
    vertex_scale = weights @ scale_by_region  # (N,)

    y_center = float(vertices[:, 1].mean())
    out = vertices.copy()
    out[:, 0] *= vertex_scale
    out[:, 2] *= vertex_scale
    out[:, 1] = y_center + (out[:, 1] - y_center) * ratios.length_scale
    return out.astype(np.float32)
