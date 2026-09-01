"""
Model 6 — face mesh fitting math: scales and aligns a `GeneratedFaceMesh`
(see `face_mesh_generation.py`) onto the existing MakeHuman avatar's head
region. Pure geometry/linear algebra, no I/O — `face_fitting_pipeline.py`
wires this together with a provider and a `face_fit_runner.py` backend.

Both "sides" (the avatar's head and the generated face) are described the
same way, as a `FaceLandmarks` anchor set (eye_left/eye_right/nose_bridge/
chin/jaw_left/jaw_right) plus the derived face_width/face_height/face_depth
those anchors imply. That symmetry is what lets `compute_scale_ratios` and
`align_face_mesh` treat "the avatar's head" and "the generated face" as two
instances of the same shape, one being fit onto the other.

`extract_avatar_head_landmarks` derives the avatar side from
`body3d_params["head_radius"]` using the same relative eye/nose layout
constants `mesh_builder.py` uses to place those features on the avatar's own
head mesh (see that module's `_EYE_X`/`_EYE_Y`/`_EYE_Z`/`_NOSE_Y`/`_NOSE_Z`).
The avatar's head is built as a near-spherical ellipsoid, so these are
documented approximate anchor points, not measurements of specific mesh
vertices — sufficient for the region-fitting math below, which only needs
consistent relative proportions, not exact geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Mirrors mesh_builder.py's facial-feature layout constants (fractions of
# head_radius from head_center) — see that module's docstring. Duplicated
# here (rather than imported) because mesh_builder's are private,
# implementation-detail constants of the procedural mesh builder; this
# module only needs the same relative layout, not a hard dependency on it.
_EYE_X, _EYE_Y, _EYE_Z = 0.40, 0.05, 0.85
_NOSE_Y, _NOSE_Z = 0.0, 0.90
_CHIN_Y, _CHIN_Z = -0.85, 0.55
_JAW_X, _JAW_Y, _JAW_Z = 0.55, -0.55, 0.55

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class FaceLandmarks:
    """Six anchor points describing a face/head region, in some local
    coordinate system (meters for the avatar side, provider-local units for
    a generated face — `compute_scale_ratios` reconciles the two)."""

    eye_left: Point3
    eye_right: Point3
    nose_bridge: Point3
    chin: Point3
    jaw_left: Point3
    jaw_right: Point3

    @property
    def eye_center(self) -> Point3:
        return tuple((a + b) / 2.0 for a, b in zip(self.eye_left, self.eye_right))

    @property
    def eye_distance(self) -> float:
        return _dist(self.eye_left, self.eye_right)

    @property
    def jaw_width(self) -> float:
        return _dist(self.jaw_left, self.jaw_right)

    @property
    def face_width(self) -> float:
        """Overall face width — the wider of the eye span and jaw span, so
        neither a narrow-eyed/wide-jawed nor wide-eyed/narrow-jawed face
        underestimates its own width."""
        return max(self.eye_distance, self.jaw_width)

    @property
    def face_height(self) -> float:
        """Eye-line to chin distance, doubled — the eye line sits roughly
        at face-height midpoint, so this approximates forehead-to-chin
        without requiring a separate forehead landmark."""
        return 2.0 * abs(self.eye_center[1] - self.chin[1])

    @property
    def face_depth(self) -> float:
        """Bounding depth (Z extent) across all six anchors — captures how
        far the nose/chin protrude relative to the jaw/eye plane."""
        zs = [p[2] for p in (self.eye_left, self.eye_right, self.nose_bridge,
                              self.chin, self.jaw_left, self.jaw_right)]
        return max(zs) - min(zs)

    def as_array(self) -> np.ndarray:
        """(6, 3) array in the fixed order `compute_scale_ratios`/
        `align_face_mesh` correspond points by."""
        return np.array([
            self.eye_left, self.eye_right, self.nose_bridge,
            self.chin, self.jaw_left, self.jaw_right,
        ], dtype=np.float64)


@dataclass(frozen=True)
class FitTransform:
    """The rigid+anisotropic-scale transform `align_face_mesh` solved:
    `aligned = rotation @ (scale * point - scale_pivot) + translation`."""

    scale: tuple[float, float, float]
    scale_pivot: Point3
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pivot = np.asarray(self.scale_pivot, dtype=np.float64)
        scaled = (points - pivot) * np.asarray(self.scale, dtype=np.float64) + pivot
        return scaled @ self.rotation.T + self.translation


@dataclass(frozen=True)
class FittedFaceMesh:
    vertices: np.ndarray  # (N, 3) float32, in the avatar's coordinate system
    faces: np.ndarray  # (M, 3) uint32
    landmarks: FaceLandmarks  # transformed into the avatar's coordinate system
    transform: FitTransform


def _dist(a: Point3, b: Point3) -> float:
    return math.dist(a, b)


def extract_avatar_head_landmarks(body3d_params: dict, height_cm: float) -> FaceLandmarks:
    """Derives approximate 3D head landmarks (meters) for the existing
    MakeHuman avatar from `body3d_params["head_radius"]` (see
    `model6_body3d.params.PARAM_NAMES`), using the same relative
    eye/nose/jaw layout `mesh_builder.py` places its head features at.
    Read-only: never modifies `body3d_params` or any avatar geometry — this
    is only used to compute a fitting transform for a *separately* generated
    face mesh (see `face_fitting_pipeline.py`)."""
    if "head_radius" not in body3d_params:
        raise ValueError("body3d_params missing 'head_radius'")

    height_m = height_cm / 100.0
    hr = body3d_params["head_radius"] / 2.0 * height_m
    head_center = (0.0, height_m - hr, 0.0)

    def offset(fx, fy, fz):
        return (head_center[0] + fx * hr, head_center[1] + fy * hr, head_center[2] + fz * hr)

    return FaceLandmarks(
        eye_left=offset(-_EYE_X, _EYE_Y, _EYE_Z),
        eye_right=offset(_EYE_X, _EYE_Y, _EYE_Z),
        nose_bridge=offset(0.0, _NOSE_Y, _NOSE_Z),
        chin=offset(0.0, _CHIN_Y, _CHIN_Z),
        jaw_left=offset(-_JAW_X, _JAW_Y, _JAW_Z),
        jaw_right=offset(_JAW_X, _JAW_Y, _JAW_Z),
    )


def landmarks_from_dict(landmarks: dict) -> FaceLandmarks:
    """Builds a `FaceLandmarks` from a `GeneratedFaceMesh.landmarks` dict
    (see `face_mesh_generation.LANDMARK_NAMES`)."""
    return FaceLandmarks(
        eye_left=tuple(landmarks["eye_left"]),
        eye_right=tuple(landmarks["eye_right"]),
        nose_bridge=tuple(landmarks["nose_bridge"]),
        chin=tuple(landmarks["chin"]),
        jaw_left=tuple(landmarks["jaw_left"]),
        jaw_right=tuple(landmarks["jaw_right"]),
    )


def compute_scale_ratios(
    avatar_landmarks: FaceLandmarks, generated_landmarks: FaceLandmarks,
) -> tuple[float, float, float]:
    """Independent per-axis scale ratios bringing the generated face's
    dimensions onto the avatar's head dimensions:

        scale_x = avatar_face_width  / generated_face_width
        scale_y = avatar_face_height / generated_face_height
        scale_z = avatar_face_depth  / generated_face_depth

    Anisotropic by design (see module docstring / region_scaling.py's
    equivalent rationale for garments) — a generated face that's
    proportionally wider than the avatar's head shouldn't also get taller.
    """
    def ratio(avatar_value, generated_value, axis):
        if generated_value <= 1e-9:
            raise ValueError(f"generated face has ~zero {axis} extent — cannot compute a scale ratio")
        return float(avatar_value / generated_value)

    return (
        ratio(avatar_landmarks.face_width, generated_landmarks.face_width, "width"),
        ratio(avatar_landmarks.face_height, generated_landmarks.face_height, "height"),
        ratio(avatar_landmarks.face_depth, generated_landmarks.face_depth, "depth"),
    )


# Landmark correspondence weights for the alignment solve below — higher
# weight pulls that landmark's post-scale position closer to the avatar's,
# implementing "align eye centers first, nose bridge second, chin/jaw
# region third" as a single weighted least-squares fit rather than three
# sequential passes (equivalent result, one linear solve).
_LANDMARK_WEIGHTS = {
    "eye_left": 3.0, "eye_right": 3.0,
    "nose_bridge": 2.0,
    "chin": 1.0, "jaw_left": 1.0, "jaw_right": 1.0,
}
_LANDMARK_ORDER = ["eye_left", "eye_right", "nose_bridge", "chin", "jaw_left", "jaw_right"]


def _weighted_kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Kabsch/Procrustes: the rotation `R` and translation `t`
    minimizing `sum(w_i * ||R @ src_i + t - dst_i||^2)`. Standard SVD-based
    solution; `weights` is what lets eye/nose landmarks dominate the fit
    over chin/jaw (see `_LANDMARK_WEIGHTS`)."""
    w = weights / weights.sum()
    src_centroid = (src * w[:, None]).sum(axis=0)
    dst_centroid = (dst * w[:, None]).sum(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    H = (src_c * w[:, None]).T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    t = dst_centroid - R @ src_centroid
    return R, t


def align_face_mesh(
    generated_vertices: np.ndarray,
    generated_faces: np.ndarray,
    generated_landmarks: FaceLandmarks,
    avatar_landmarks: FaceLandmarks,
    scale: tuple[float, float, float] | None = None,
) -> FittedFaceMesh:
    """Scales (per-axis, about the generated face's eye-center pivot) then
    rigidly aligns (weighted Kabsch over the 6 landmark correspondences —
    see `_LANDMARK_WEIGHTS`) `generated_vertices`/`generated_landmarks` onto
    `avatar_landmarks`'s coordinate system. `scale` defaults to
    `compute_scale_ratios(avatar_landmarks, generated_landmarks)`.

    This is landmark alignment, not bounding-box-only scaling: after
    scaling, the six anchors are least-squares-fit onto their avatar
    counterparts (weighted toward eyes, then nose, then chin/jaw), so
    proportions between landmarks are preserved rather than forced into a
    box.
    """
    if scale is None:
        scale = compute_scale_ratios(avatar_landmarks, generated_landmarks)
    scale_arr = np.asarray(scale, dtype=np.float64)
    pivot = np.asarray(generated_landmarks.eye_center, dtype=np.float64)

    scaled_vertices = (generated_vertices.astype(np.float64) - pivot) * scale_arr + pivot
    scaled_landmark_pts = (generated_landmarks.as_array() - pivot) * scale_arr + pivot

    avatar_pts = avatar_landmarks.as_array()
    weights = np.array([_LANDMARK_WEIGHTS[name] for name in _LANDMARK_ORDER])

    R, t = _weighted_kabsch(scaled_landmark_pts, avatar_pts, weights)

    aligned_vertices = (scaled_vertices @ R.T + t).astype(np.float32)
    aligned_landmark_pts = scaled_landmark_pts @ R.T + t

    aligned_landmarks = FaceLandmarks(
        eye_left=tuple(aligned_landmark_pts[0]), eye_right=tuple(aligned_landmark_pts[1]),
        nose_bridge=tuple(aligned_landmark_pts[2]), chin=tuple(aligned_landmark_pts[3]),
        jaw_left=tuple(aligned_landmark_pts[4]), jaw_right=tuple(aligned_landmark_pts[5]),
    )

    transform = FitTransform(
        scale=tuple(scale_arr.tolist()), scale_pivot=tuple(pivot.tolist()),
        rotation=R, translation=t,
    )

    return FittedFaceMesh(
        vertices=aligned_vertices, faces=generated_faces, landmarks=aligned_landmarks, transform=transform,
    )
