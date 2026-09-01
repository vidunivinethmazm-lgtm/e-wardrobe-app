"""EXPERIMENTAL — multi-view image-to-3D step of the `multiview_tryon`
pipeline: turns the two virtual-try-on images (front/back photo of the
avatar's own person wearing the uploaded garment) into a `GeneratedGarmentMesh`
— the same mesh contract `avatar_pipeline.model7_garment_fitting.
garment_mesh_generation` already defines, so downstream code (texture
projection, region fitting, the Blender runner) needs no changes.

Note: the mesh produced here still contains reconstructed human-body
geometry (the try-on images are whole-body photos) — `garment_isolation.py`
strips that before the mesh is fitted onto the avatar.

`IMAGE_TO_3D_MV_PROVIDER` (default "mock"):
- mock: `MockMultiViewImageTo3DProvider` delegates to the existing
  `garment_mesh_generation.MockGarmentMeshProvider` (no network call).
- hunyuan3d_2mv: `Hunyuan3D2MVProvider` calls an externally-hosted
  Hunyuan3D-2mv-style multi-view image-to-3D inference service. Requires
  `HUNYUAN3D_2MV_ENDPOINT`.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from ..fitting_types import GarmentFittingError
from ..garment_mesh_generation import GeneratedGarmentMesh, MockGarmentMeshProvider

HUNYUAN3D_2MV_ENDPOINT = os.environ.get("HUNYUAN3D_2MV_ENDPOINT")
HUNYUAN3D_2MV_TIMEOUT_S = float(os.environ.get("HUNYUAN3D_2MV_TIMEOUT_S", "300"))

IMAGE_TO_3D_MV_PROVIDER = os.environ.get("IMAGE_TO_3D_MV_PROVIDER", "mock")


class MultiViewImageTo3DProvider(ABC):
    """Interface for multi-view (front+back) image-to-3D garment mesh
    reconstruction from virtual-try-on images. Never responsible for
    isolating the garment from the reconstructed body — see
    `garment_isolation.py` for that, run right after this stage."""

    @abstractmethod
    def generate(
        self,
        front_tryon_rgb: np.ndarray,
        back_tryon_rgb: np.ndarray,
        garment_type: str,
        front_mask: np.ndarray | None = None,
        back_mask: np.ndarray | None = None,
    ) -> GeneratedGarmentMesh | None:
        """Returns `None` only when this provider isn't configured/available
        at all. A configured provider that fails to produce a usable mesh
        must raise instead."""


# ── Mock provider (non-production) ───────────────────────────────────────

class MockMultiViewImageTo3DProvider(MultiViewImageTo3DProvider):
    """NON-PRODUCTION / TESTS-DEV-ONLY stand-in. Delegates straight to
    `garment_mesh_generation.MockGarmentMeshProvider` — the same category-
    shaped procedural shell used by the adaptive-template pipeline — rather
    than duplicating that shell-building logic. Always `is_mock=True`."""

    def generate(self, front_tryon_rgb, back_tryon_rgb, garment_type, front_mask=None, back_mask=None):
        return MockGarmentMeshProvider().generate(
            front_tryon_rgb, back_tryon_rgb, garment_type, front_mask=front_mask, back_mask=back_mask,
        )


# ── Real provider ─────────────────────────────────────────────────────────

class Hunyuan3D2MVProvider(MultiViewImageTo3DProvider):
    """Real multi-view image-to-3D provider, backed by an externally-hosted
    Hunyuan3D-2mv-style inference service. Sends both try-on views in one
    request. Returns `None` only when `HUNYUAN3D_2MV_ENDPOINT` isn't
    configured; a configured-but-failing call raises `GarmentFittingError`.
    """

    def __init__(self, endpoint: str | None = None, timeout_s: float | None = None):
        self.endpoint = endpoint or HUNYUAN3D_2MV_ENDPOINT
        self.timeout_s = timeout_s or HUNYUAN3D_2MV_TIMEOUT_S

    def generate(self, front_tryon_rgb, back_tryon_rgb, garment_type, front_mask=None, back_mask=None):
        if not self.endpoint:
            return None

        import requests

        payload = {
            "garment_type": garment_type,
            "front_png_base64": _encode_png(front_tryon_rgb),
            "back_png_base64": _encode_png(back_tryon_rgb),
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise GarmentFittingError(f"Hunyuan3D-2mv mesh generation request failed: {exc}") from exc

        try:
            vertices = np.asarray(body["vertices"], dtype=np.float32)
            faces = np.asarray(body["faces"], dtype=np.uint32)
            uvs = np.asarray(body.get("uvs", []), dtype=np.float32)
            if uvs.shape != vertices[:, :2].shape:
                uvs = np.full((len(vertices), 2), 0.5, dtype=np.float32)
            landmarks = {name: tuple(point) for name, point in body["landmarks"].items()}
            texture_png = None
            if body.get("texture_png_base64"):
                texture_png = base64.b64decode(body["texture_png_base64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GarmentFittingError(f"Hunyuan3D-2mv response missing/invalid mesh data: {exc}") from exc

        return GeneratedGarmentMesh(
            vertices=vertices, faces=faces, uvs=uvs, texture_png=texture_png,
            landmarks=landmarks, garment_type=garment_type, is_mock=False,
        )


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_multiview_mesh_provider() -> MultiViewImageTo3DProvider:
    if IMAGE_TO_3D_MV_PROVIDER == "mock":
        return MockMultiViewImageTo3DProvider()
    if IMAGE_TO_3D_MV_PROVIDER == "hunyuan3d_2mv":
        return Hunyuan3D2MVProvider()
    raise NotImplementedError(
        f"IMAGE_TO_3D_MV_PROVIDER={IMAGE_TO_3D_MV_PROVIDER!r} is not implemented. "
        "Set IMAGE_TO_3D_MV_PROVIDER=mock or hunyuan3d_2mv."
    )
