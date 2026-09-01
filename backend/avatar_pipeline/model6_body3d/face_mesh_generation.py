"""
Model 6 — optional image-to-3D face mesh generation, upstream of the face
*fitting* stage (`face_mesh_fitting.py`) that scales/aligns a generated face
onto the existing MakeHuman avatar's head (`face_fitting_pipeline.py`).

This module only produces a candidate 3D face mesh from the user's
front/left/right photos — it never touches the avatar's body or decides how
that mesh gets merged into the avatar. `Unique3DFaceMeshProvider` describes
how an optional *image-to-3D face mesh generation provider* (such as
Unique3D) plugs in here. It is not described or used as a "face
reconstruction model": the identity-preserving guarantee for the final
avatar still comes from the fitting/texture-transfer stages downstream,
which work directly from the user's own photo pixels. This provider's job
is strictly to turn 2D photos into a rough 3D mesh + landmark set that those
downstream stages can then scale and align onto the avatar's actual head.

Defaults to `MockFaceMeshProvider` (`FACE_MESH_PROVIDER=mock`), a
deterministic, dependency-free synthetic mesh generator for tests/dev.
`Unique3DFaceMeshProvider` (`FACE_MESH_PROVIDER=unique3d`) is the real
workflow; if it isn't configured (`FACE_MESH_ENDPOINT` unset), `generate()`
returns `None` — a clear "not available" signal — rather than silently
fabricating a result. Callers (see `face_fitting_pipeline.run_face_fitting`)
must fall back to the existing face-texture-transfer path when `generate()`
returns `None`.
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

FACE_MESH_PROVIDER = os.environ.get("FACE_MESH_PROVIDER", "mock")

# Landmark keys every provider must return — matches the anchor points
# `face_mesh_fitting.py` fits against on the avatar side.
LANDMARK_NAMES = ("eye_left", "eye_right", "nose_bridge", "chin", "jaw_left", "jaw_right")


@dataclass(frozen=True)
class GeneratedFaceMesh:
    """A candidate 3D face mesh + landmarks produced by a `FaceMeshProvider`,
    in the provider's own arbitrary local coordinate system/scale — never
    assumed to already match the avatar's units. `face_mesh_fitting.py`
    computes the scale/rotation/translation needed to bring it onto the
    avatar's head.
    """

    vertices: np.ndarray  # (N, 3) float32, provider-local units
    faces: np.ndarray  # (M, 3) uint32
    landmarks: dict  # LANDMARK_NAMES -> (x, y, z) in the same local units
    texture_png: bytes | None = None  # optional face appearance texture

    def __post_init__(self):
        missing = [name for name in LANDMARK_NAMES if name not in self.landmarks]
        if missing:
            raise ValueError(f"GeneratedFaceMesh.landmarks missing required keys: {missing}")


class FaceMeshProvider(ABC):
    """Interface for image-to-3D face mesh generation. Never responsible for
    fitting the mesh to an avatar or for body geometry — see
    `face_mesh_fitting.py` / `face_fitting_pipeline.py` for that."""

    @abstractmethod
    def generate(
        self, front_rgb: np.ndarray,
        left_rgb: np.ndarray | None = None,
        right_rgb: np.ndarray | None = None,
    ) -> GeneratedFaceMesh | None:
        """Given the user's captured front (required) and left/right
        (optional) photos, returns a `GeneratedFaceMesh`, or `None` if this
        provider isn't available/configured — callers must treat `None` as
        "fall back to the existing texture-transfer path", never as an
        empty/degenerate mesh."""


class MockFaceMeshProvider(FaceMeshProvider):
    """Deterministic, dependency-free synthetic face mesh generator for
    tests/development — no ML model, no network call. Builds a small
    ellipsoid "face" mesh sized from the front photo's aspect ratio, with
    landmarks placed at fixed, documented fractional positions (the same
    role real Unique3D output would play, just not photorealistic)."""

    # Fractions of the synthetic face's own half-extents (rx, ry, rz) —
    # mirrors the layout `mesh_builder.py` uses for the avatar's head so the
    # two landmark sets are directly comparable.
    _EYE_X, _EYE_Y, _EYE_Z = 0.40, 0.05, 0.85
    _NOSE_Y, _NOSE_Z = 0.0, 0.90
    _CHIN_Y, _CHIN_Z = -0.85, 0.55
    _JAW_X, _JAW_Y, _JAW_Z = 0.55, -0.55, 0.55
    _N_LAT, _N_LON = 8, 12

    def generate(self, front_rgb, left_rgb=None, right_rgb=None) -> GeneratedFaceMesh:
        h, w = front_rgb.shape[:2]
        aspect = w / max(h, 1)

        # A face mesh roughly 1 "unit" tall; width/depth derived from the
        # photo's aspect ratio so a wider photo produces a wider mesh —
        # enough variation to make scale-ratio tests meaningful without any
        # real face detection.
        rx = 0.35 * np.clip(aspect, 0.6, 1.6)
        ry = 0.45
        rz = 0.30

        vertices = _ellipsoid_vertices((0.0, 0.0, 0.0), rx, ry, rz, self._N_LAT, self._N_LON)
        faces = _ellipsoid_faces(self._N_LAT, self._N_LON)

        landmarks = {
            "eye_left": (-self._EYE_X * rx, self._EYE_Y * ry, self._EYE_Z * rz),
            "eye_right": (self._EYE_X * rx, self._EYE_Y * ry, self._EYE_Z * rz),
            "nose_bridge": (0.0, self._NOSE_Y * ry, self._NOSE_Z * rz),
            "chin": (0.0, self._CHIN_Y * ry, self._CHIN_Z * rz),
            "jaw_left": (-self._JAW_X * rx, self._JAW_Y * ry, self._JAW_Z * rz),
            "jaw_right": (self._JAW_X * rx, self._JAW_Y * ry, self._JAW_Z * rz),
        }

        texture_png = None
        try:
            buf = io.BytesIO()
            Image.fromarray(front_rgb).resize((64, 64)).save(buf, format="PNG")
            texture_png = buf.getvalue()
        except Exception:
            pass

        return GeneratedFaceMesh(vertices=vertices, faces=faces, landmarks=landmarks, texture_png=texture_png)


class Unique3DFaceMeshProvider(FaceMeshProvider):
    """Real image-to-3D face mesh generation provider, backed by an
    externally-hosted Unique3D-style inference service (never bundled — this
    class only knows how to call it). Framed strictly as an *image-to-3D
    face mesh generation* step, not a "face reconstruction model": its
    output is a rough geometric mesh + landmarks for the fitting stage to
    scale/align onto the avatar's real head, not a finished, identity-baked
    avatar.

    Requires `FACE_MESH_ENDPOINT` (an HTTP endpoint accepting the front/
    left/right photos and returning `{"vertices": [[x,y,z],...],
    "faces": [[i,j,k],...], "landmarks": {name: [x,y,z], ...},
    "texture_png_base64": "..."}`). If unset, `generate()` returns `None`
    — a clear "not configured" status — instead of silently producing a
    placeholder mesh that could be mistaken for a real one.
    """

    def __init__(self, endpoint: str | None = None, timeout_s: float = 60.0):
        self.endpoint = endpoint or os.environ.get("FACE_MESH_ENDPOINT")
        self.timeout_s = timeout_s

    def generate(self, front_rgb, left_rgb=None, right_rgb=None) -> GeneratedFaceMesh | None:
        if not self.endpoint:
            return None

        import base64

        import requests

        def _encode(rgb):
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        payload = {"front_png_base64": _encode(front_rgb)}
        if left_rgb is not None:
            payload["left_png_base64"] = _encode(left_rgb)
        if right_rgb is not None:
            payload["right_png_base64"] = _encode(right_rgb)

        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            # Configured but unreachable/erroring — a real failure, not "not
            # configured": raise loudly rather than silently falling back,
            # so a broken deployment doesn't quietly look like it worked.
            raise RuntimeError(f"Unique3D face mesh generation request failed: {exc}") from exc

        vertices = np.asarray(body["vertices"], dtype=np.float32)
        faces = np.asarray(body["faces"], dtype=np.uint32)
        landmarks = {name: tuple(point) for name, point in body["landmarks"].items()}
        texture_png = base64.b64decode(body["texture_png_base64"]) if body.get("texture_png_base64") else None

        return GeneratedFaceMesh(vertices=vertices, faces=faces, landmarks=landmarks, texture_png=texture_png)


def get_face_mesh_provider() -> FaceMeshProvider:
    if FACE_MESH_PROVIDER == "mock":
        return MockFaceMeshProvider()
    if FACE_MESH_PROVIDER == "unique3d":
        return Unique3DFaceMeshProvider()
    raise ValueError(f"unknown FACE_MESH_PROVIDER {FACE_MESH_PROVIDER!r}")


def _ellipsoid_vertices(center, rx, ry, rz, n_lat, n_lon) -> np.ndarray:
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        y = np.cos(theta)
        r = np.sin(theta)
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = r * np.cos(phi)
            z = r * np.sin(phi)
            verts.append((center[0] + x * rx, center[1] + y * ry, center[2] + z * rz))
    return np.asarray(verts, dtype=np.float32)


def _ellipsoid_faces(n_lat, n_lon) -> np.ndarray:
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(faces, dtype=np.uint32)
