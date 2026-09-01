"""
Model 7 — final region-wise fitting of the **generated garment mesh**
(see `garment_mesh_generation.GeneratedGarmentMesh`) onto the existing
avatar, behind a small interface so a Python-only mock backend can run in
development/tests without Blender installed.

Both backends operate on the *actual* generated mesh — its vertex/face
count, UVs, and texture are preserved throughout; only vertex positions
change. Neither backend ever substitutes a category template mesh.

Real backend (`BlenderGarmentFitRunner`): shells out to
`blender --background --python scripts/blender_fit_garment_mesh.py -- ...`,
importing the *generated* garment GLB (not a template) alongside the
avatar GLB, and runs: region-proximity vertex-group weighting (using the
avatar's own shoulder/chest/waist/hip landmarks as group centers) ->
region-wise local scaling -> Surface Deform -> Shrinkwrap (positive
offset) -> Collision + Cloth -> Corrective Smooth -> apply + export,
preserving the garment's topology/UVs/material throughout.

Mock backend (`MockGarmentFitRunner`, used by default — see
`GARMENT_FIT_MOCK` below): applies the same region ratios via
`garment_region_fitting.region_deform_mesh` directly in Python/numpy —
real per-region deformation of the real generated mesh, just without
Blender's Surface Deform/Shrinkwrap/Cloth refinement. Still clearly a
**non-production approximation** (no true avatar-surface conforming,
no collision resolution) — never presented as the final production result
without Blender in the loop; see `pipeline.py`'s `is_mock` propagation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .fitting_types import GarmentFittingError, RegionScales
from .garment_mesh_generation import GeneratedGarmentMesh
from .garment_region_fitting import AvatarRegionLandmarks, region_deform_mesh
from .glb_writer import write_mesh_glb as _write_garment_glb

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "blender_fit_garment_mesh.py"

BLENDER_EXECUTABLE = os.environ.get("BLENDER_EXECUTABLE", "blender")
BLENDER_TIMEOUT_S = int(os.environ.get("BLENDER_TIMEOUT_S", "180"))

# Same convention as model6_body3d.face_fit_runner.FACE_FIT_MOCK.
GARMENT_FIT_MOCK = os.environ.get("GARMENT_FIT_MOCK", "1") != "0"

# Small positive push-out so the fitted garment sits just outside the
# avatar's skin instead of clipping/z-fighting with it.
_OUTWARD_OFFSET_M = 0.008


class GarmentFitRunner(ABC):
    """Interface both backends implement — `pipeline.py` only depends on
    this. Returns `(glb_bytes, texture_png, used_blender)` — `used_blender`
    lets callers/tests honestly report whether real Surface Deform/
    Shrinkwrap/Cloth refinement ran, independent of `is_mock` (which is
    about the *mesh source*, Unique3D vs procedural placeholder)."""

    @abstractmethod
    def fit(
        self, generated_mesh: GeneratedGarmentMesh, avatar_glb_bytes: bytes,
        avatar_landmarks: AvatarRegionLandmarks, ratios: RegionScales,
    ) -> tuple[bytes, bytes | None, bool]:
        ...


class MockGarmentFitRunner(GarmentFitRunner):
    """Python-only region deformation — see module docstring."""

    def fit(self, generated_mesh, avatar_glb_bytes, avatar_landmarks, ratios) -> tuple[bytes, bytes | None, bool]:
        deformed = region_deform_mesh(generated_mesh.vertices.astype(np.float64), avatar_landmarks, ratios)
        deformed = _push_outward(deformed)
        glb_bytes = _write_garment_glb(deformed, generated_mesh.faces, generated_mesh.uvs, generated_mesh.texture_png)
        return glb_bytes, generated_mesh.texture_png, False


class BlenderGarmentFitRunner(GarmentFitRunner):
    """Shells out to Blender to run the real modifier stack on the
    *generated* garment mesh (see module docstring)."""

    def fit(self, generated_mesh, avatar_glb_bytes, avatar_landmarks, ratios) -> tuple[bytes, bytes | None, bool]:
        if not BLENDER_SCRIPT.exists():
            raise GarmentFittingError(f"Blender garment-mesh fitting script missing: {BLENDER_SCRIPT}")
        if not avatar_glb_bytes:
            raise GarmentFittingError("Blender garment fitting requires the avatar's own GLB")

        garment_glb = _write_garment_glb(
            generated_mesh.vertices, generated_mesh.faces, generated_mesh.uvs, generated_mesh.texture_png,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            avatar_path = tmp_dir / "avatar.glb"
            avatar_path.write_bytes(avatar_glb_bytes)
            garment_path = tmp_dir / "garment.glb"
            garment_path.write_bytes(garment_glb)

            centers = avatar_landmarks.as_region_centers()
            config_path = tmp_dir / "fit_config.json"
            config_path.write_text(json.dumps({
                "avatar_glb": str(avatar_path),
                "garment_glb": str(garment_path),
                "region_centers": {name: point.tolist() for name, point in centers.items()},
                "region_scales": ratios.as_dict(),
                "output_glb": str(tmp_dir / "fitted.glb"),
            }))

            cmd = [BLENDER_EXECUTABLE, "--background", "--python", str(BLENDER_SCRIPT), "--", str(config_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=BLENDER_TIMEOUT_S, check=False)
            except FileNotFoundError as exc:
                raise GarmentFittingError(f"Blender executable not found: {BLENDER_EXECUTABLE!r}") from exc
            except subprocess.TimeoutExpired as exc:
                raise GarmentFittingError("Blender garment fitting timed out") from exc

            output_path = tmp_dir / "fitted.glb"
            if result.returncode != 0 or not output_path.exists():
                stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
                raise GarmentFittingError(f"Blender garment fitting failed: {stderr}")

            return output_path.read_bytes(), generated_mesh.texture_png, True


def get_garment_fit_runner() -> GarmentFitRunner:
    return MockGarmentFitRunner() if GARMENT_FIT_MOCK else BlenderGarmentFitRunner()


def _push_outward(vertices: np.ndarray) -> np.ndarray:
    centroid_xz = vertices[:, [0, 2]].mean(axis=0)
    radial = vertices[:, [0, 2]] - centroid_xz
    dist = np.linalg.norm(radial, axis=1, keepdims=True)
    dist_safe = np.where(dist < 1e-6, 1.0, dist)
    direction = radial / dist_safe
    out = vertices.copy()
    out[:, [0, 2]] += direction * _OUTWARD_OFFSET_M
    return out
