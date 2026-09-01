"""EXPERIMENTAL — shared result type for `multiview/pipeline.py`. Mirrors
`fitting_types.GarmentFittingResult`, extended with the virtual-try-on
preview images and per-stage provider metadata the mobile app / API
response must surface (never claim real AI processing happened when a mock
provider ran — see each field's docstring).

PIVOT: `region_scales` is now optional and `texture_provider` may be `None`
— since `avatar3d_providers.py` replaced the garment-isolation-then-fit-
onto-existing-avatar design with a full-avatar reconstruction (Unique3D),
there is no more avatar-to-garment region fitting (nothing left to compute
ratios against) and no more separate texture-generation stage (the
reconstructed avatar mesh already carries its own baked texture). See
`is_full_avatar_replacement`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fitting_types import NormalizedGarmentFeatures, RegionScales


@dataclass
class MultiviewFittingResult:
    """End-to-end output of `multiview.pipeline.run_multiview_tryon_fitting`."""

    fit_id: str
    garment_type: str
    features: NormalizedGarmentFeatures
    glb_bytes: bytes
    texture_png: bytes | None
    tryon_front_png: bytes
    tryon_back_png: bytes
    virtual_tryon_provider: str  # "mock" | "idm_vton"
    image_to_3d_provider: str  # "mock" | "unique3d" — the full-avatar 3D provider
    # True only when the avatar mesh came from a real Unique3D call —
    # independent of `is_mock` below, which also covers the virtual try-on
    # provider.
    is_real_3d_generation: bool
    # True: this result REPLACES the existing avatar (a full reconstructed
    # human mesh from Unique3D), rather than being a garment fitted onto it
    # — callers (mobile UI) must render it as the avatar, not an overlay.
    is_full_avatar_replacement: bool = True
    # No longer computed (nothing is fitted onto a separate avatar anymore).
    region_scales: RegionScales | None = None
    # No longer a separate stage — the reconstructed avatar mesh carries its
    # own baked texture.
    texture_provider: str | None = None
    status: str = "ready"
    warnings: list[str] = field(default_factory=list)
    # True unless every stage (virtual try-on, avatar reconstruction) ran
    # its real backend — callers (API response, mobile UI) must surface
    # this, never present mock output as a real result.
    is_mock: bool = True
