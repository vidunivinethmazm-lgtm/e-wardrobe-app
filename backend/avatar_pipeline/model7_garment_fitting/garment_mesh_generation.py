"""
Model 7 — image-to-3D **garment** mesh generation: turns the uploaded
front/back garment photos into an actual textured 3D mesh of *that*
garment (its own silhouette, neckline, sleeves, colour/pattern), not a
generic category placeholder.

    garment front/back images (already background-removed/segmented)
    -> GarmentMeshProvider.generate()
    -> GeneratedGarmentMesh (real geometry + its own UVs/texture)
    -> project_front_back_texture() (front photo -> front-facing surface,
       back photo -> back-facing surface)
    -> validate_garment_mesh()

`Unique3DGarmentMeshProvider` is the real workflow: an externally-hosted
Unique3D-style image-to-3D *garment mesh generation* service (never bundled
here, and never described as "the fitting" — fitting/deformation is a
separate, later stage in `garment_region_fitting.py`). Gated by
`UNIQUE3D_ENABLED` (default unset/"0"): when disabled, `get_garment_mesh_
provider()` returns `MockGarmentMeshProvider` instead — a **non-production,
tests/dev-only** procedural stand-in, never to be presented to a user as a
real fitting result (`GeneratedGarmentMesh.is_mock` makes this explicit and
is threaded through the whole pipeline into the API response).

When `UNIQUE3D_ENABLED=1` but generation genuinely fails (bad response,
network error, degenerate mesh), this module raises a clear error rather
than silently falling back to the mock mesh — see `pipeline.py`.
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .fitting_types import GarmentFittingError
from .glb_writer import MIN_BOUNDING_EXTENT, MIN_FACES, MIN_VERTICES, validate_mesh_geometry

UNIQUE3D_ENABLED = os.environ.get("UNIQUE3D_ENABLED", "0") == "1"
UNIQUE3D_ENDPOINT = os.environ.get("UNIQUE3D_ENDPOINT")
UNIQUE3D_TIMEOUT_S = float(os.environ.get("UNIQUE3D_TIMEOUT_S", "120"))

# 3D landmark names a garment mesh (mock or real) must provide — mirrors
# `fitting_types.GarmentKeypoints`' 2D names, now as 3D points on the
# generated mesh, so `garment_region_fitting.py` can compare them directly
# against the avatar's own 3D region landmarks.
LANDMARK_NAMES = (
    "left_shoulder", "right_shoulder", "neck_center",
    "left_chest", "right_chest",
    "left_waist", "right_waist",
    "left_hip", "right_hip",
    "left_sleeve_end", "right_sleeve_end",
    "hem_center", "top_center", "bottom_center",
)

_ATLAS_SIZE = 512


@dataclass(frozen=True)
class GeneratedGarmentMesh:
    """The actual reconstructed garment: real geometry + its own UVs, ready
    for region-wise fitting (`garment_region_fitting.py`) — never a
    stand-in category shape. `is_mock=True` marks output from
    `MockGarmentMeshProvider` (tests/dev only); every caller (`pipeline.py`,
    the API response, the mobile UI) must propagate that flag rather than
    presenting mock output as a real fitted garment.
    """

    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) uint32
    uvs: np.ndarray  # (N, 2) float32
    texture_png: bytes | None
    landmarks: dict  # LANDMARK_NAMES -> (x, y, z)
    garment_type: str
    is_mock: bool

    def __post_init__(self):
        missing = [name for name in LANDMARK_NAMES if name not in self.landmarks]
        if missing:
            raise ValueError(f"GeneratedGarmentMesh.landmarks missing required keys: {missing}")


class GarmentMeshProvider(ABC):
    """Interface for image-to-3D garment mesh generation. Never responsible
    for region-wise fitting to a specific avatar — see
    `garment_region_fitting.py` / `garment_fit_runner.py` for that."""

    @abstractmethod
    def generate(
        self, front_rgb: np.ndarray, back_rgb: np.ndarray, garment_type: str,
        front_mask: np.ndarray | None = None, back_mask: np.ndarray | None = None,
        front_bbox: tuple[int, int, int, int] | None = None,
        back_bbox: tuple[int, int, int, int] | None = None,
    ) -> GeneratedGarmentMesh | None:
        """Returns `None` only when this provider isn't configured/available
        at all (e.g. Unique3D disabled) — callers must treat that as "cannot
        proceed", never as an empty/placeholder mesh. A *configured*
        provider that fails to produce a usable mesh must raise instead."""


# ── Mock provider (non-production) ───────────────────────────────────────

class MockGarmentMeshProvider(GarmentMeshProvider):
    """NON-PRODUCTION / TESTS-DEV-ONLY stand-in. Builds a category-shaped
    procedural shell (torso+sleeves for `upper_body`, bodice+flared skirt
    for `dress`, waist/hip+two legs for `lower_body`) with the uploaded
    photos projected onto it, purely so the pipeline/API/mobile app have
    something to exercise without a real Unique3D deployment. Always sets
    `is_mock=True` — never to be shown to a user as "your garment", only as
    an explicitly-labelled preview/dev placeholder (see `pipeline.py` /
    `server/app.py`'s `is_mock` response field).
    """

    _SEGMENTS = 24
    _TORSO_RINGS_BY_TYPE = {
        "upper_body": [("hem", 0.55), ("waist", 0.62), ("chest", 0.74), ("shoulder", 0.86)],
        "dress": [("hem", 0.35), ("hip", 0.50), ("waist", 0.62), ("chest", 0.74), ("shoulder", 0.86)],
        "lower_body": [("hip", 0.46), ("waist", 0.53)],
    }
    _BASE_RADIUS_BY_REGION = {"shoulder": 0.19, "chest": 0.17, "waist": 0.15, "hip": 0.17, "hem": 0.14}
    _DRESS_HEM_RADIUS = 0.24
    _HEIGHT_M = 1.6

    def generate(
        self, front_rgb, back_rgb, garment_type,
        front_mask=None, back_mask=None, front_bbox=None, back_bbox=None,
    ) -> GeneratedGarmentMesh:
        rings_spec = self._TORSO_RINGS_BY_TYPE.get(garment_type)
        if rings_spec is None:
            raise GarmentFittingError(f"unknown garment_type {garment_type!r}")

        builder = _ShellBuilder()
        landmarks: dict[str, tuple[float, float, float]] = {}

        if garment_type == "lower_body":
            self._build_lower_body(builder, rings_spec, landmarks)
        else:
            self._build_torso(builder, garment_type, rings_spec, landmarks)
            if garment_type == "upper_body":
                self._build_sleeves(builder, landmarks)
            else:
                landmarks.setdefault("left_sleeve_end", landmarks["left_shoulder"])
                landmarks.setdefault("right_sleeve_end", landmarks["right_shoulder"])

        positions, faces, uvs = builder.build()

        for name in LANDMARK_NAMES:
            landmarks.setdefault(name, (0.0, 0.0, 0.0))

        texture_png = build_front_back_atlas(front_rgb, front_mask, front_bbox, back_rgb, back_mask, back_bbox)

        return GeneratedGarmentMesh(
            vertices=positions, faces=faces, uvs=uvs, texture_png=texture_png,
            landmarks=landmarks, garment_type=garment_type, is_mock=True,
        )

    def _region_radius(self, region: str, garment_type: str) -> float:
        base = self._BASE_RADIUS_BY_REGION[region]
        if garment_type == "dress" and region == "hem":
            base = self._DRESS_HEM_RADIUS
        return base

    def _build_torso(self, builder, garment_type, rings_spec, landmarks) -> None:
        n_rings = len(rings_spec)
        ring_indices = []
        for i, (region, y_frac) in enumerate(rings_spec):
            radius = self._region_radius(region, garment_type)
            y = self._HEIGHT_M * y_frac
            ring_idx = builder.add_ring(y, radius, radius, i / max(n_rings - 1, 1), segments=self._SEGMENTS)
            ring_indices.append(ring_idx)
            if region == "shoulder":
                landmarks["left_shoulder"] = (-radius, y, 0.0)
                landmarks["right_shoulder"] = (radius, y, 0.0)
                landmarks["neck_center"] = (0.0, y, radius)
                landmarks["top_center"] = (0.0, y, 0.0)
            elif region == "chest":
                landmarks["left_chest"] = (-radius, y, 0.0)
                landmarks["right_chest"] = (radius, y, 0.0)
            elif region == "waist":
                landmarks["left_waist"] = (-radius, y, 0.0)
                landmarks["right_waist"] = (radius, y, 0.0)
            elif region == "hip":
                landmarks["left_hip"] = (-radius, y, 0.0)
                landmarks["right_hip"] = (radius, y, 0.0)
            elif region == "hem":
                landmarks["hem_center"] = (0.0, y, radius)
                landmarks["bottom_center"] = (0.0, y, 0.0)
        for i in range(n_rings - 1):
            builder.loft(ring_indices[i], ring_indices[i + 1])
        landmarks.setdefault("left_waist", landmarks.get("left_chest", (-0.15, self._HEIGHT_M * 0.6, 0.0)))
        landmarks.setdefault("right_waist", landmarks.get("right_chest", (0.15, self._HEIGHT_M * 0.6, 0.0)))
        landmarks.setdefault("left_hip", landmarks["left_waist"])
        landmarks.setdefault("right_hip", landmarks["right_waist"])

    def _build_sleeves(self, builder, landmarks) -> None:
        shoulder_radius = self._region_radius("shoulder", "upper_body")
        shoulder_y = self._HEIGHT_M * 0.86
        sleeve_length = shoulder_radius * 1.6

        for side, key in ((-1, "left_sleeve_end"), (1, "right_sleeve_end")):
            attach_x = side * shoulder_radius * 0.85
            cuff_x = side * (shoulder_radius * 0.85 + sleeve_length)
            attach_ring = builder.add_ring(shoulder_y, shoulder_radius * 0.55, shoulder_radius * 0.55, 1.0,
                                            segments=self._SEGMENTS, cx=attach_x)
            cuff_ring = builder.add_ring(shoulder_y - shoulder_radius * 0.15, shoulder_radius * 0.47,
                                          shoulder_radius * 0.47, 1.0, segments=self._SEGMENTS, cx=cuff_x)
            builder.loft(attach_ring, cuff_ring)
            landmarks[key] = (cuff_x, shoulder_y - shoulder_radius * 0.15, 0.0)

    def _build_lower_body(self, builder, rings_spec, landmarks) -> None:
        n_rings = len(rings_spec)
        ring_indices = []
        for i, (region, y_frac) in enumerate(rings_spec):
            radius = self._region_radius(region, "lower_body")
            y = self._HEIGHT_M * y_frac
            ring_idx = builder.add_ring(y, radius, radius, 0.6 + 0.4 * (i / max(n_rings - 1, 1)), segments=self._SEGMENTS)
            ring_indices.append(ring_idx)
            if region == "hip":
                landmarks["left_hip"] = (-radius, y, 0.0)
                landmarks["right_hip"] = (radius, y, 0.0)
            elif region == "waist":
                landmarks["left_waist"] = (-radius, y, 0.0)
                landmarks["right_waist"] = (radius, y, 0.0)
        for i in range(n_rings - 1):
            builder.loft(ring_indices[i], ring_indices[i + 1])

        hip_radius = self._region_radius("hip", "lower_body")
        hip_y = self._HEIGHT_M * rings_spec[0][1]
        leg_gap = hip_radius * 0.55
        leg_radius = hip_radius * 0.55
        hem_y = self._HEIGHT_M * 0.04

        for side in (-1, 1):
            thigh_ring = builder.add_ring(hip_y, leg_radius, leg_radius, 0.55, segments=self._SEGMENTS, cx=side * leg_gap)
            hem_ring = builder.add_ring(hem_y, leg_radius * 0.85, leg_radius * 0.85, 0.0, segments=self._SEGMENTS, cx=side * leg_gap)
            builder.loft(ring_indices[0], thigh_ring)
            builder.loft(thigh_ring, hem_ring)

        landmarks["hem_center"] = (0.0, hem_y, 0.0)
        landmarks["bottom_center"] = (0.0, hem_y, 0.0)
        landmarks["top_center"] = (0.0, self._HEIGHT_M * rings_spec[-1][1], 0.0)
        landmarks["neck_center"] = landmarks["top_center"]
        landmarks["left_shoulder"] = landmarks["left_waist"]
        landmarks["right_shoulder"] = landmarks["right_waist"]
        landmarks["left_chest"] = landmarks["left_waist"]
        landmarks["right_chest"] = landmarks["right_waist"]


class _ShellBuilder:
    def __init__(self):
        self._positions: list[tuple[float, float, float]] = []
        self._uvs: list[tuple[float, float]] = []
        self._faces: list[tuple[int, int, int]] = []
        self._rings: list[list[int]] = []

    def add_ring(self, y, rx, rz, height_within_shell, segments, cx=0.0, cz=0.0) -> int:
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring_indices = []
        for theta in angles:
            x = cx + rx * np.cos(theta)
            z = cz + rz * np.sin(theta)
            ring_indices.append(len(self._positions))
            self._positions.append((x, y, z))
            self._uvs.append((0.5, 0.5))  # placeholder — real UVs assigned by project_front_back_texture
        self._rings.append(ring_indices)
        return len(self._rings) - 1

    def loft(self, ring_a_index: int, ring_b_index: int) -> None:
        ring_a, ring_b = self._rings[ring_a_index], self._rings[ring_b_index]
        n = len(ring_a)
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = ring_a[i], ring_a[j], ring_b[j], ring_b[i]
            self._faces.append((a, b, c))
            self._faces.append((a, c, d))

    def build(self):
        positions = np.asarray(self._positions, dtype=np.float32)
        faces = np.asarray(self._faces, dtype=np.uint32)
        uvs = np.asarray(self._uvs, dtype=np.float32)
        return positions, faces, uvs


# ── Real provider ─────────────────────────────────────────────────────────

class Unique3DGarmentMeshProvider(GarmentMeshProvider):
    """Real image-to-3D garment mesh generation provider, backed by an
    externally-hosted Unique3D-style inference service. Framed strictly as
    *image-to-3D garment mesh generation* — its output is handed to
    `garment_region_fitting.py`/`garment_fit_runner.py` for the actual
    avatar-fitting step, never claimed to already be "the fitted result".

    Multi-view contract: always sends both the front and back photos in one
    request (`{"front_png_base64", "back_png_base64", "garment_type"}`).
    If the endpoint reports it only supports single-view generation (JSON
    response `{"multi_view_supported": false}`), falls back to
    `_generate_fused_from_single_views`: two separate single-image calls
    (front-only, back-only) whose results are fused — see that function's
    docstring — so the back photo is never silently dropped.

    Returns `None` only when `UNIQUE3D_ENABLED` is off or no endpoint is
    configured. A configured-but-failing call raises `GarmentFittingError`
    (network error, bad response, degenerate mesh) — never a silent
    downgrade to a placeholder.
    """

    def __init__(self, endpoint: str | None = None, timeout_s: float | None = None):
        self.endpoint = endpoint or UNIQUE3D_ENDPOINT
        self.timeout_s = timeout_s or UNIQUE3D_TIMEOUT_S

    def generate(
        self, front_rgb, back_rgb, garment_type,
        front_mask=None, back_mask=None, front_bbox=None, back_bbox=None,
    ) -> GeneratedGarmentMesh | None:
        if not UNIQUE3D_ENABLED or not self.endpoint:
            return None

        import requests

        payload = {
            "garment_type": garment_type,
            "front_png_base64": _encode_png(front_rgb),
            "back_png_base64": _encode_png(back_rgb),
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise GarmentFittingError(f"Unique3D garment mesh generation request failed: {exc}") from exc

        if body.get("multi_view_supported") is False:
            return self._generate_fused_from_single_views(front_rgb, back_rgb, garment_type)

        return self._mesh_from_response(body, garment_type)

    def _generate_fused_from_single_views(self, front_rgb, back_rgb, garment_type) -> GeneratedGarmentMesh:
        """Multi-view adapter for a Unique3D deployment that only accepts one
        image per call: generates a full garment guess from the front photo
        alone, then a second full guess from the back photo alone, and fuses
        them — front-facing vertices come from the front-image mesh,
        back-facing vertices are replaced with their nearest counterpart in
        the back-image mesh (nearest-neighbour match in the shared local
        coordinate frame, after aligning the two meshes on their shared
        landmarks). This means the back photo actually determines the rear
        geometry/texture rather than being discarded — see module
        docstring's "never silently ignore the back image" requirement."""
        import requests

        def _single_view_call(rgb, side):
            resp = requests.post(
                self.endpoint, json={"garment_type": garment_type, f"{side}_png_base64": _encode_png(rgb)},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return resp.json()

        front_body = _single_view_call(front_rgb, "front")
        back_body = _single_view_call(back_rgb, "back")

        front_mesh = self._mesh_from_response(front_body, garment_type)
        back_mesh = self._mesh_from_response(back_body, garment_type)
        return fuse_front_back_meshes(front_mesh, back_mesh)

    def _mesh_from_response(self, body: dict, garment_type: str) -> GeneratedGarmentMesh:
        try:
            vertices = np.asarray(body["vertices"], dtype=np.float32)
            faces = np.asarray(body["faces"], dtype=np.uint32)
            uvs = np.asarray(body.get("uvs", []), dtype=np.float32)
            if uvs.shape != vertices[:, :2].shape:
                uvs = np.full((len(vertices), 2), 0.5, dtype=np.float32)
            landmarks = {name: tuple(point) for name, point in body["landmarks"].items()}
            texture_png = None
            if body.get("texture_png_base64"):
                import base64
                texture_png = base64.b64decode(body["texture_png_base64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GarmentFittingError(f"Unique3D response missing/invalid mesh data: {exc}") from exc

        return GeneratedGarmentMesh(
            vertices=vertices, faces=faces, uvs=uvs, texture_png=texture_png,
            landmarks=landmarks, garment_type=garment_type, is_mock=False,
        )


def fuse_front_back_meshes(front_mesh: GeneratedGarmentMesh, back_mesh: GeneratedGarmentMesh) -> GeneratedGarmentMesh:
    """Fuses two single-view garment mesh guesses into one: keeps
    `front_mesh`'s topology (vertex count/faces/UVs) throughout, but
    replaces every back-facing vertex's position with its nearest
    counterpart from `back_mesh` (after aligning `back_mesh` onto
    `front_mesh`'s coordinate frame via their shared shoulder/hem
    landmarks). This is a pragmatic, always-preserves-one-topology fusion
    rather than full mesh stitching — real rear geometry from the back
    photo, without the topological risk of merging two independently
    triangulated surfaces."""
    front_landmarks = np.array([front_mesh.landmarks[name] for name in LANDMARK_NAMES], dtype=np.float64)
    back_landmarks = np.array([back_mesh.landmarks[name] for name in LANDMARK_NAMES], dtype=np.float64)

    # Align back_mesh onto front_mesh's frame (simple centroid + scale
    # match — a full rigid Kabsch fit is unnecessary here since both meshes
    # already share the same "avatar-space-like" convention from the same
    # provider).
    front_centroid, back_centroid = front_landmarks.mean(axis=0), back_landmarks.mean(axis=0)
    front_scale = np.linalg.norm(front_landmarks - front_centroid) or 1.0
    back_scale = np.linalg.norm(back_landmarks - back_centroid) or 1.0
    scale = front_scale / back_scale

    aligned_back_vertices = (back_mesh.vertices.astype(np.float64) - back_centroid) * scale + front_centroid

    fused_vertices = front_mesh.vertices.copy()
    front_facing = fused_vertices[:, 2] >= 0
    back_facing_idx = np.where(~front_facing)[0]

    if len(back_facing_idx) > 0 and len(aligned_back_vertices) > 0:
        # Brute-force nearest neighbour — garment meshes here are a few
        # hundred vertices, well within a dense O(N*M) search.
        diffs = fused_vertices[back_facing_idx][:, None, :] - aligned_back_vertices[None, :, :]
        nearest = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
        fused_vertices[back_facing_idx] = aligned_back_vertices[nearest].astype(np.float32)

    fused_landmarks = dict(front_mesh.landmarks)
    for name in ("left_hip", "right_hip", "hem_center", "bottom_center"):
        if name in back_mesh.landmarks:
            back_point = np.array(back_mesh.landmarks[name], dtype=np.float64)
            fused_landmarks[name] = tuple(((back_point - back_centroid) * scale + front_centroid).tolist())

    return GeneratedGarmentMesh(
        vertices=fused_vertices, faces=front_mesh.faces, uvs=front_mesh.uvs,
        texture_png=front_mesh.texture_png, landmarks=fused_landmarks,
        garment_type=front_mesh.garment_type, is_mock=front_mesh.is_mock,
    )


# ── Front/back texture projection (explicit post-generation step) ────────

def project_front_back_texture(mesh: GeneratedGarmentMesh, front_rgb, back_rgb=None) -> GeneratedGarmentMesh:
    """Explicit "project front texture to front, back texture to back" step
    (run regardless of provider — even a real Unique3D mesh's own texture
    guess is replaced with the user's actual photos here, so colours/
    patterns/logos are guaranteed to be the real ones, not a generative
    guess). Classifies each vertex as front- or back-facing by its Z sign
    relative to the mesh's own centroid (works for any garment mesh
    oriented with +Z roughly "front", the same convention `MockGarmentMesh
    Provider` and this pipeline use throughout), then re-derives UVs into a
    front-on-top/back-on-bottom atlas — see `build_front_back_atlas`.
    """
    atlas_png = build_front_back_atlas(front_rgb, None, None, back_rgb, None, None)
    if atlas_png is None:
        return mesh

    vertices = mesh.vertices
    centroid_z = float(np.median(vertices[:, 2])) if len(vertices) else 0.0
    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    y_span = max(y_max - y_min, 1e-6)

    front_facing = vertices[:, 2] >= centroid_z
    height_frac = (vertices[:, 1] - y_min) / y_span

    uvs = np.empty((len(vertices), 2), dtype=np.float32)
    x_min, x_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    x_span = max(x_max - x_min, 1e-6)
    u = np.clip((vertices[:, 0] - x_min) / x_span, 0.0, 1.0)

    v_within_half = 0.5 * (1.0 - height_frac)
    uvs[:, 0] = u
    uvs[:, 1] = np.where(front_facing, v_within_half, 0.5 + v_within_half)

    return GeneratedGarmentMesh(
        vertices=mesh.vertices, faces=mesh.faces, uvs=uvs.astype(np.float32), texture_png=atlas_png,
        landmarks=mesh.landmarks, garment_type=mesh.garment_type, is_mock=mesh.is_mock,
    )


def build_front_back_atlas(
    front_rgb, front_mask, front_bbox, back_rgb, back_mask, back_bbox, size: int = _ATLAS_SIZE,
) -> bytes | None:
    """Builds a square PNG atlas: front photo (cropped to its segmented
    bbox, if given) on the top half, back photo on the bottom half —
    background pixels outside the segmentation mask are filled with the
    garment's own median colour so no studio-backdrop halo shows through.
    Returns `None` if neither photo is available."""
    if front_rgb is None and back_rgb is None:
        return None

    atlas = Image.new("RGB", (size, size), (200, 200, 200))

    def _prepare_half(rgb, mask, bbox):
        if rgb is None:
            return None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            crop = rgb[y0:y1 + 1, x0:x1 + 1]
            crop_mask = mask[y0:y1 + 1, x0:x1 + 1] if mask is not None else None
        else:
            crop, crop_mask = rgb, mask
        crop = crop.copy()
        if crop_mask is not None and crop_mask.any():
            median_color = np.median(crop[crop_mask], axis=0).astype(np.uint8)
            crop[~crop_mask] = median_color
        return Image.fromarray(crop).resize((size, size // 2))

    front_half = _prepare_half(front_rgb, front_mask, front_bbox)
    back_half = _prepare_half(back_rgb, back_mask, back_bbox)

    if front_half is not None:
        atlas.paste(front_half, (0, 0))
    if back_half is not None:
        atlas.paste(back_half, (0, size // 2))
    elif front_half is not None:
        atlas.paste(front_half, (0, size // 2))
    if front_half is None and back_half is not None:
        atlas.paste(back_half, (0, 0))

    buf = io.BytesIO()
    atlas.save(buf, format="PNG")
    return buf.getvalue()


def _encode_png(rgb: np.ndarray) -> str:
    import base64
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Mesh quality validation ───────────────────────────────────────────────
# The numeric checks themselves live in `glb_writer.validate_mesh_geometry`
# (shared with the EXPERIMENTAL multiview/avatar3d_providers.py full-avatar
# path, which has no landmark contract to also check) — `MIN_VERTICES` etc.
# stay re-exported under their original names since other modules already
# import them from here.


def validate_garment_mesh(mesh: GeneratedGarmentMesh) -> None:
    """Raises `GarmentFittingError` if `mesh` is too degenerate to fit onto
    an avatar (empty, near-zero extent, or faces referencing out-of-range
    vertices) — the "Validate mesh quality" pipeline stage."""
    validate_mesh_geometry(mesh.vertices, mesh.faces)


def get_garment_mesh_provider() -> GarmentMeshProvider:
    return Unique3DGarmentMeshProvider() if UNIQUE3D_ENABLED else MockGarmentMeshProvider()
