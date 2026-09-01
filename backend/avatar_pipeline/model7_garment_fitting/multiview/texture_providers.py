"""EXPERIMENTAL — texture generation step of the `multiview_tryon` pipeline.

Runs after `garment_isolation.isolate_garment_geometry` and always textures
the isolated garment mesh from the ORIGINAL uploaded garment front/back
photos — never from the virtual-try-on images — so the garment's actual
colours/pattern/logos are guaranteed to be the real ones, not whatever the
try-on or mesh-generation step guessed.

`TEXTURE_PROVIDER` (default "mock"):
- mock: delegates to the existing `garment_mesh_generation.
  project_front_back_texture` / `build_front_back_atlas` (no network call).
- hunyuan3d_paint: `Hunyuan3DPaintProvider` calls an externally-hosted
  Hunyuan3D-Paint-style texture-generation service. Requires
  `HUNYUAN3D_PAINT_ENDPOINT`.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod

import numpy as np

from ..fitting_types import GarmentFittingError
from ..garment_mesh_generation import GeneratedGarmentMesh, build_front_back_atlas

HUNYUAN3D_PAINT_ENDPOINT = os.environ.get("HUNYUAN3D_PAINT_ENDPOINT")
HUNYUAN3D_PAINT_TIMEOUT_S = float(os.environ.get("HUNYUAN3D_PAINT_TIMEOUT_S", "180"))

TEXTURE_PROVIDER = os.environ.get("TEXTURE_PROVIDER", "mock")


class TextureGenerationProvider(ABC):
    """Interface for garment mesh texture generation from the original
    garment photos. Never responsible for mesh geometry — see
    `mesh3d_providers.py` / `garment_isolation.py` for that."""

    @abstractmethod
    def generate(
        self,
        mesh: GeneratedGarmentMesh,
        garment_front_rgb: np.ndarray,
        garment_back_rgb: np.ndarray,
    ) -> bytes | None:
        """Returns a PNG texture atlas, or `None` only when this provider
        isn't configured/available at all."""


# ── Mock provider (non-production) ───────────────────────────────────────

class MockTextureGenerationProvider(TextureGenerationProvider):
    """NON-PRODUCTION / TESTS-DEV-ONLY stand-in. Reuses the same
    front-on-top/back-on-bottom atlas the adaptive-template pipeline builds
    from the raw garment photos — no learned texture-painting model."""

    def generate(self, mesh, garment_front_rgb, garment_back_rgb):
        return build_front_back_atlas(garment_front_rgb, None, None, garment_back_rgb, None, None)


# ── Real provider ─────────────────────────────────────────────────────────

class Hunyuan3DPaintProvider(TextureGenerationProvider):
    """Real texture-generation provider, backed by an externally-hosted
    Hunyuan3D-Paint-style inference service: paints the isolated garment
    mesh's UVs using the original garment front/back photos as reference.
    Returns `None` only when `HUNYUAN3D_PAINT_ENDPOINT` isn't configured; a
    configured-but-failing call raises `GarmentFittingError`.
    """

    def __init__(self, endpoint: str | None = None, timeout_s: float | None = None):
        self.endpoint = endpoint or HUNYUAN3D_PAINT_ENDPOINT
        self.timeout_s = timeout_s or HUNYUAN3D_PAINT_TIMEOUT_S

    def generate(self, mesh, garment_front_rgb, garment_back_rgb):
        if not self.endpoint:
            return None

        import requests

        from .mesh3d_providers import _encode_png

        payload = {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "uvs": mesh.uvs.tolist(),
            "garment_front_png_base64": _encode_png(garment_front_rgb),
            "garment_back_png_base64": _encode_png(garment_back_rgb),
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise GarmentFittingError(f"Hunyuan3D-Paint texture request failed: {exc}") from exc

        try:
            return base64.b64decode(body["texture_png_base64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GarmentFittingError(f"Hunyuan3D-Paint response missing/invalid texture: {exc}") from exc


def get_texture_provider() -> TextureGenerationProvider:
    if TEXTURE_PROVIDER == "mock":
        return MockTextureGenerationProvider()
    if TEXTURE_PROVIDER == "hunyuan3d_paint":
        return Hunyuan3DPaintProvider()
    raise NotImplementedError(
        f"TEXTURE_PROVIDER={TEXTURE_PROVIDER!r} is not implemented. "
        "Set TEXTURE_PROVIDER=mock or hunyuan3d_paint."
    )
