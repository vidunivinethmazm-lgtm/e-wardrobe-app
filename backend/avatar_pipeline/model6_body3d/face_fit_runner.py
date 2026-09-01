"""
Model 6 — Blender-based integration of a fitted 3D face
(`face_mesh_fitting.FittedFaceMesh`) into the existing MakeHuman avatar's
head region, behind a small interface so a mock backend can run in
development/tests without Blender installed (same pattern as
`model7_garment_fitting.blender_runner`).

Real backend (`BlenderFaceFitRunner`): shells out to
`blender --background --python scripts/blender_fit_face.py -- <config.json>`
(see that script for the full modifier stack — region-limited Shrinkwrap/
Surface Deform, boundary blending near forehead/cheeks/jaw/neck, Corrective
Smooth, face-UV texture bake, export).

Mock backend (`MockFaceFitRunner`, used by default — see
`FACE_FIT_MOCK` below): does **not** modify the avatar's mesh geometry.
Real region-limited mesh blending (shrinkwrap/surface-deform confined to a
face vertex group, boundary feathering) is exactly the kind of operation
that's easy to get subtly wrong outside Blender's modifier stack, and the
critical constraint here is that the avatar's body/neck must never change —
so the safe default is a passthrough that still runs (and returns) the real
scale/rotation/translation fitting math, without gambling on an approximate
from-scratch mesh edit. This means `MockFaceFitRunner` output is provably
identical to the input avatar everywhere outside the face, at the cost of
not actually visualizing the fitted face until the real Blender backend
runs. Never claim otherwise — see `FaceFitRunner.fit`'s return contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .face_mesh_fitting import FittedFaceMesh

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "blender_fit_face.py"

BLENDER_EXECUTABLE = os.environ.get("BLENDER_EXECUTABLE", "blender")
BLENDER_TIMEOUT_S = int(os.environ.get("BLENDER_TIMEOUT_S", "180"))

# Same convention as model7_garment_fitting.blender_runner.GARMENT_FITTING_MOCK.
FACE_FIT_MOCK = os.environ.get("FACE_FIT_MOCK", "1") != "0"


class FaceFitError(ValueError):
    """Raised for a face-fitting integration failure (e.g. Blender missing/
    failed) — callers should surface this rather than falling back to a
    geometry change they can't verify."""


class FaceFitRunner(ABC):
    """Interface both backends implement. `fit` returns
    `(avatar_glb_bytes, geometry_modified)` — `geometry_modified` is False
    for any backend (like the mock) that didn't actually touch the mesh, so
    callers/tests can assert on that honestly instead of trusting a status
    string alone."""

    @abstractmethod
    def fit(
        self, avatar_glb_bytes: bytes, fitted_face: FittedFaceMesh,
        texture_png: bytes | None,
    ) -> tuple[bytes, bool]:
        """Returns `(resulting_avatar_glb_bytes, geometry_modified)`."""


class MockFaceFitRunner(FaceFitRunner):
    """Passthrough — see module docstring. Returns the input avatar GLB
    completely unchanged, with `geometry_modified=False`."""

    def fit(self, avatar_glb_bytes, fitted_face, texture_png) -> tuple[bytes, bool]:
        return avatar_glb_bytes, False


class BlenderFaceFitRunner(FaceFitRunner):
    """Shells out to Blender to run the real region-limited fitting stack
    (see module docstring)."""

    def fit(self, avatar_glb_bytes, fitted_face, texture_png) -> tuple[bytes, bool]:
        if not BLENDER_SCRIPT.exists():
            raise FaceFitError(f"Blender face-fit script missing: {BLENDER_SCRIPT}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            avatar_path = tmp_dir / "avatar.glb"
            avatar_path.write_bytes(avatar_glb_bytes)

            face_glb_path = tmp_dir / "face.glb"
            face_glb_path.write_bytes(_write_face_glb(fitted_face, texture_png))

            config_path = tmp_dir / "fit_config.json"
            config_path.write_text(json.dumps({
                "avatar_glb": str(avatar_path),
                "generated_face_glb": str(face_glb_path),
                "eye_center": list(fitted_face.landmarks.eye_center),
                "chin": list(fitted_face.landmarks.chin),
                "jaw_left": list(fitted_face.landmarks.jaw_left),
                "jaw_right": list(fitted_face.landmarks.jaw_right),
                "output_glb": str(tmp_dir / "fitted_avatar.glb"),
            }))

            cmd = [
                BLENDER_EXECUTABLE, "--background", "--python", str(BLENDER_SCRIPT),
                "--", str(config_path),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=BLENDER_TIMEOUT_S, check=False)
            except FileNotFoundError as exc:
                raise FaceFitError(f"Blender executable not found: {BLENDER_EXECUTABLE!r}") from exc
            except subprocess.TimeoutExpired as exc:
                raise FaceFitError("Blender face fitting timed out") from exc

            output_path = tmp_dir / "fitted_avatar.glb"
            if result.returncode != 0 or not output_path.exists():
                stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
                raise FaceFitError(f"Blender face fitting failed: {stderr}")

            return output_path.read_bytes(), True


def get_face_fit_runner() -> FaceFitRunner:
    return MockFaceFitRunner() if FACE_FIT_MOCK else BlenderFaceFitRunner()


def _write_face_glb(fitted_face: FittedFaceMesh, texture_png: bytes | None) -> bytes:
    """Minimal single-mesh GLB writer for the fitted (already scaled/
    aligned) generated face mesh, so the Blender script can import it as a
    standalone object alongside the avatar."""
    import numpy as np
    from pygltflib import (
        GLTF2, Accessor, Asset, Buffer, BufferView,
        Material, Mesh, Node, PbrMetallicRoughness, Primitive, Scene,
    )

    positions = fitted_face.vertices.astype(np.float32)
    indices = fitted_face.faces.astype(np.uint32).reshape(-1)

    blob = bytearray()

    def _append(data: bytes, alignment: int = 4) -> int:
        pad = (-len(data)) % alignment
        offset = len(blob)
        blob.extend(data + b"\x00" * pad)
        return offset

    pos_bytes = positions.tobytes()
    pos_offset = _append(pos_bytes)
    pos_bv = BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes))

    idx_bytes = indices.tobytes()
    idx_offset = _append(idx_bytes)
    idx_bv = BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes))

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1, material=0)])],
        materials=[Material(
            pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[0.85, 0.75, 0.65, 1.0]),
            doubleSided=True,
        )],
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=[pos_bv, idx_bv],
        accessors=[
            Accessor(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                     min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
            Accessor(bufferView=1, componentType=5125, count=len(indices), type="SCALAR"),
        ],
    )
    gltf.set_binary_blob(bytes(blob))
    return b"".join(gltf.save_to_bytes())
