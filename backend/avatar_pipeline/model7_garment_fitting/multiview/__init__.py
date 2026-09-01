"""Model 7 — EXPERIMENTAL `multiview_tryon` garment-fitting research pipeline.

Not the default: `server/app.py`'s `POST /api/avatars/<id>/fit-garment`
still runs the existing adaptive-template pipeline
(`avatar_pipeline.model7_garment_fitting.pipeline.run_garment_fitting`)
unless the caller opts in via `pipeline_mode=multiview_tryon` (or the
`GARMENT_PIPELINE_MODE` env var). See `pipeline.run_multiview_tryon_fitting`
for the entry point and module docstrings in this package for the stage
breakdown:

    person front/back + garment front/back photos
    -> tryon_providers.VirtualTryOnProvider (IDM-VTON, or mock)
    -> mesh3d_providers.MultiViewImageTo3DProvider (Hunyuan3D-2mv, or mock)
    -> garment_isolation.isolate_garment_geometry (strip reconstructed body)
    -> texture_providers.TextureGenerationProvider (Hunyuan3D-Paint, or mock)
       — always textured from the ORIGINAL garment photos, never the
         try-on images
    -> garment_region_fitting + garment_fit_runner (reused, unchanged)
       -> fitted garment .glb
"""
