"""EXPERIMENTAL — garment-only isolation for the `multiview_tryon` pipeline.

`mesh3d_providers.py` reconstructs a mesh from *whole-body* virtual-try-on
photos, so its output still contains reconstructed human-body geometry
(head, hands, skin) alongside the actual garment surface. This module strips
that out before the mesh is ever handed to the avatar-fitting stage —
requirement: "never overlay a complete reconstructed human mesh on top of
the existing avatar".

`isolate_garment_geometry` re-derives each vertex's front/back UV using the
same convention `garment_mesh_generation.project_front_back_texture` uses
(front/back by Z-sign relative to the mesh centroid, height fraction by Y),
then samples the corresponding try-on-image garment mask (front mask for
front-facing vertices, back mask for back-facing ones) at that position.
Vertices landing outside the mask (i.e. classified as body/background, not
garment, in the source try-on photo) are dropped, along with any face that
references a dropped vertex — vertex/face indices are then compacted.

Because this heuristic has no real human-parsing model behind it (see
`multiview/pipeline.py` for how the masks themselves are derived), a
mesh whose isolation barely removes anything *and* whose overall
proportions look like a full standing figure (tall relative to its width)
is rejected outright with `GarmentFittingError` rather than silently
passed through as "the garment" — this is the concrete guard against
handing a full-body reconstruction to the avatar-fitting stage.
"""

from __future__ import annotations

import numpy as np

from ..fitting_types import GarmentFittingError
from ..garment_mesh_generation import MIN_FACES, MIN_VERTICES, GeneratedGarmentMesh

# A kept-vertex fraction above this, combined with a bounding-box aspect
# ratio (height / width) above this, indicates isolation did essentially
# nothing to a mesh shaped like a full standing human figure.
_FULL_BODY_ASPECT_RATIO_THRESHOLD = 2.2
_FULL_BODY_KEEP_FRACTION_THRESHOLD = 0.85


def isolate_garment_geometry(
    mesh: GeneratedGarmentMesh,
    front_garment_mask: np.ndarray | None,
    back_garment_mask: np.ndarray | None,
) -> GeneratedGarmentMesh:
    """Returns a new `GeneratedGarmentMesh` containing only the vertices/
    faces classified as garment surface. Raises `GarmentFittingError` if
    isolation would remove everything, or if the input looks like an
    unisolated full-body reconstruction (see module docstring)."""
    vertices = mesh.vertices
    n = len(vertices)
    keep = np.ones(n, dtype=bool)

    if front_garment_mask is not None or back_garment_mask is not None:
        centroid_z = float(np.median(vertices[:, 2])) if n else 0.0
        y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
        y_span = max(y_max - y_min, 1e-6)
        x_min, x_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
        x_span = max(x_max - x_min, 1e-6)

        front_facing = vertices[:, 2] >= centroid_z
        u = np.clip((vertices[:, 0] - x_min) / x_span, 0.0, 1.0)
        v = np.clip(1.0 - (vertices[:, 1] - y_min) / y_span, 0.0, 1.0)

        for side_mask, side_selector in (
            (front_garment_mask, front_facing),
            (back_garment_mask, ~front_facing),
        ):
            if side_mask is None or not side_selector.any():
                continue
            h, w = side_mask.shape
            rows = np.clip((v[side_selector] * (h - 1)).astype(int), 0, h - 1)
            cols = np.clip((u[side_selector] * (w - 1)).astype(int), 0, w - 1)
            keep[side_selector] = side_mask[rows, cols]

    kept_fraction = float(keep.mean()) if n else 0.0
    extent = vertices.max(axis=0) - vertices.min(axis=0) if n else np.zeros(3)
    horizontal_extent = max(float(extent[0]), float(extent[2]), 1e-6)
    aspect_ratio = float(extent[1]) / horizontal_extent

    if (
        aspect_ratio > _FULL_BODY_ASPECT_RATIO_THRESHOLD
        and kept_fraction > _FULL_BODY_KEEP_FRACTION_THRESHOLD
    ):
        raise GarmentFittingError(
            "garment isolation left the mesh looking like a full-body reconstruction "
            f"(aspect ratio {aspect_ratio:.2f}, kept {kept_fraction:.0%} of vertices) — "
            "refusing to fit a whole reconstructed human mesh onto the avatar"
        )

    if not keep.any():
        raise GarmentFittingError("garment isolation removed all mesh geometry — check the segmentation masks")

    if keep.all():
        return mesh

    kept_indices = np.where(keep)[0]
    new_index = np.full(n, -1, dtype=np.int64)
    new_index[kept_indices] = np.arange(len(kept_indices))

    face_keep = keep[mesh.faces].all(axis=1)
    remapped_faces = new_index[mesh.faces[face_keep]].astype(np.uint32)

    new_vertices = vertices[kept_indices]
    new_uvs = mesh.uvs[kept_indices] if mesh.uvs is not None and len(mesh.uvs) == n else mesh.uvs

    if len(new_vertices) < MIN_VERTICES or len(remapped_faces) < MIN_FACES:
        raise GarmentFittingError(
            "garment isolation left too little geometry to be usable "
            f"({len(new_vertices)} vertices, {len(remapped_faces)} faces)"
        )

    return GeneratedGarmentMesh(
        vertices=new_vertices, faces=remapped_faces, uvs=new_uvs, texture_png=mesh.texture_png,
        landmarks=mesh.landmarks, garment_type=mesh.garment_type, is_mock=mesh.is_mock,
    )
