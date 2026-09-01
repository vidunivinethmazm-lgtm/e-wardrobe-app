# EXPERIMENTAL: `multiview_tryon` garment-fitting pipeline — setup

This document covers the **experimental research pipeline** added alongside
the existing, production `adaptive_template` garment-fitting pipeline (Model
7, `backend/avatar_pipeline/model7_garment_fitting/`). It is **not required** to run
the app: every provider defaults to a mock implementation, so
`POST /api/avatars/<id>/fit-garment` with `pipeline_mode=multiview_tryon`
works end to end with no external services or model installs. This doc
explains what to set up if you want the real backends.

## PIVOT: full-avatar reconstruction, not garment fitting

**As of the current design, `multiview_tryon` reconstructs a full 3D human
avatar directly from the virtual-try-on images and uses that mesh as the
avatar going forward** — it no longer fits an isolated garment mesh onto
the pre-existing MakeHuman avatar. This was an explicit, deliberate
decision (see `backend/avatar_pipeline/model7_garment_fitting/multiview/
avatar3d_providers.py`'s module docstring) that intentionally departs from
"existing avatar stays unchanged." The response's `is_full_avatar_replacement`
field is always `true` for this mode; `region_scales` is no longer computed
(`null`); there's no separate texture-generation stage (the reconstructed
mesh carries its own baked texture).

The *original* garment-isolation-then-fit design (Stage 2/3 below,
`mesh3d_providers.py` / `garment_isolation.py` / `texture_providers.py`) is
still present in the codebase and still tested, but no longer called by
`multiview/pipeline.py`. It's kept in case a garment-only mode is wanted
again later.

## Pipeline mode

| Env var | Values | Default |
|---|---|---|
| `GARMENT_PIPELINE_MODE` | `adaptive_template` \| `multiview_tryon` | `adaptive_template` |

`GARMENT_PIPELINE_MODE` sets the server-wide default; a per-request
`pipeline_mode` form field on `POST /api/avatars/<id>/fit-garment` overrides
it for that call. The existing `adaptive_template` behavior is completely
unchanged.

## Stage 1 — Virtual try-on (IDM-VTON)

| Env var | Purpose | Default |
|---|---|---|
| `VIRTUAL_TRYON_PROVIDER` | `mock` \| `idm_vton` \| `gemini` | `mock` |
| `IDM_VTON_HF_SPACE` | Free public Gradio Space id, called via `gradio_client` | `yisol/IDM-VTON` |
| `IDM_VTON_HF_TOKEN` (or `HF_TOKEN`) | Free HuggingFace account read token — **required** in practice, see below | unset |
| `IDM_VTON_HF_FALLBACKS` | Comma-separated extra Space ids to try if the primary fails | empty |
| `IDM_VTON_ENDPOINT` | URL of a custom, self-hosted IDM-VTON-style inference service — takes priority over `IDM_VTON_HF_SPACE` when set | unset |
| `IDM_VTON_API_KEY` | optional bearer token for `IDM_VTON_ENDPOINT` | unset |
| `IDM_VTON_TIMEOUT_S` | request timeout, seconds | `120` |

Setting `VIRTUAL_TRYON_PROVIDER=idm_vton` (no self-hosting/payment needed)
calls the free, public **yisol/IDM-VTON** community Gradio Space
(https://huggingface.co/spaces/yisol/IDM-VTON) via the `gradio_client`
package — one call for the front person/garment pair, one for the back
pair (that Space's demo API only accepts a single pair per call).

**`IDM_VTON_HF_TOKEN` is required in practice.** That Space runs on
HuggingFace's ZeroGPU (shared, dynamically-allocated GPU), which grants
real GPU quota only to authenticated requests — anonymous calls routinely
fail with `AcceleratorError`. To fix:
1. Create a free account at https://huggingface.co/join (no payment info).
2. Generate a **read** token at https://huggingface.co/settings/tokens.
3. Set `IDM_VTON_HF_TOKEN=hf_xxxxxxxxxxxx` before starting the server.

Even authenticated, this is a shared community resource: it can be slow,
queued, temporarily down, or have its API shape changed by its owner at
any time — a failure here raises `GarmentFittingError` with a clear
message, it never silently falls back to mock output. Not suitable as a
reliable production backend; set `IDM_VTON_ENDPOINT` (a self-hosted
deployment) for that.

Mock behavior (`VIRTUAL_TRYON_PROVIDER=mock`, the default): returns the two
uploaded person photos unchanged — no try-on actually happens, clearly
reported via `virtual_tryon_provider: "mock"` in the API response.

### Alternative: Gemini (`VIRTUAL_TRYON_PROVIDER=gemini`)

Reuses the **existing** `backend/ai_tryon/gemini_client.py` (already used by
the older `/api/ai-tryon` endpoint) — a pragmatic fallback when IDM-VTON's
free Space is unavailable or its ZeroGPU quota is exhausted. Requires:

| Env var | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key (https://aistudio.google.com/apikey, free tier) | unset |
| `AI_TRYON_MOCK` | Must be `0` for real calls — `gemini_client`'s own mock flag | `1` |

If `AI_TRYON_MOCK` is left at its default (`1`), this provider reports
itself as unconfigured (returns `None`, surfaced as a clear "unavailable"
error) rather than silently echoing the person photos as a fake try-on
result.

## Stage 2 — Full-avatar image-to-3D (Unique3D)

| Env var | Purpose | Default |
|---|---|---|
| `FULL_AVATAR_3D_PROVIDER` | `mock` \| `unique3d` | `mock` |
| `UNIQUE3D_HF_SPACE` | Free public Gradio Space id, called via `gradio_client` | `Wuvin/Unique3D` |
| `UNIQUE3D_HF_TOKEN` (or `HF_TOKEN`) | Free HuggingFace account read token — required for the same ZeroGPU reason as IDM-VTON's | unset |
| `UNIQUE3D_HF_FALLBACKS` | Comma-separated extra Space ids to try if the primary fails | empty |
| `UNIQUE3D_AVATAR_ENDPOINT` | URL of a custom, self-hosted Unique3D-style inference service — takes priority over `UNIQUE3D_HF_SPACE` | unset |
| `UNIQUE3D_AVATAR_API_KEY` | optional bearer token for `UNIQUE3D_AVATAR_ENDPOINT` | unset |
| `UNIQUE3D_AVATAR_TIMEOUT_S` | request timeout, seconds | `180` |

Setting `FULL_AVATAR_3D_PROVIDER=unique3d` calls the free, public
**Wuvin/Unique3D** community Gradio Space
(https://github.com/AiuniAI/Unique3D) via `gradio_client`, using the front
try-on image (Unique3D's public demo reconstructs from a single reference
image; the back try-on image isn't sent to this integration). Same
ZeroGPU/`UNIQUE3D_HF_TOKEN` requirement as Stage 1 — anonymous calls fail
with `AcceleratorError`.

The Gradio call shape (`api_name="/generate3dv2"`, `input_image`/
`input_processing`/`setable_seed`/`render_video`/`do_refine`/
`expansion_weight`/`init_type`, returning a `.glb` path) is verified
against the Space's own source (`gradio_app/gradio_3dgen.py`'s
`fullrunv2_btn` handler), not just guessed. A community fork added via
`UNIQUE3D_HF_FALLBACKS` may still differ if its owner changed the
interface — that raises a clear error rather than misparsing.

Mock behavior (default): delegates to the existing
`garment_mesh_generation.MockGarmentMeshProvider`'s "dress" category shell
— the roughest full-body-shaped placeholder already available.

### Alternative: self-hosting on Kaggle's free GPU quota

`scripts/kaggle_unique3d_endpoint.ipynb` runs the official Unique3D Gradio
demo on Kaggle's free weekly 30-GPU-hour quota and tunnels it out via
ngrok. Since `UNIQUE3D_HF_SPACE` is passed straight to `gradio_client.Client`
(which accepts any reachable Gradio URL, not just an HF Space id), point it
at the printed tunnel URL — no code changes and no `UNIQUE3D_HF_TOKEN`
needed (that's only for the public Space's shared ZeroGPU auth). Useful
when the public `Wuvin/Unique3D` Space is queued/down or its ZeroGPU quota
is exhausted.

## API response metadata

`POST /api/avatars/<id>/fit-garment` with `pipeline_mode=multiview_tryon`
returns, in addition to the existing `adaptive_template` fields:

```json
{
  "pipeline_mode": "multiview_tryon",
  "virtual_tryon_provider": "mock" | "idm_vton",
  "image_to_3d_provider": "mock" | "unique3d",
  "texture_provider": null,
  "region_scales": null,
  "is_real_3d_generation": true | false,
  "is_full_avatar_replacement": true,
  "garment_tryon_front_url": "/api/avatars/<id>/fitted-garment/<fit_id>-tryon-front.png",
  "garment_tryon_back_url": "/api/avatars/<id>/fitted-garment/<fit_id>-tryon-back.png"
}
```

`garment_mesh_url` in this mode points to a **full replacement avatar
mesh**, not a garment overlay — the mobile app must render it as the
avatar (see `is_full_avatar_replacement`), not alongside the existing one.

`is_mock` (also present in the `adaptive_template` response) is `true`
unless *every* stage above ran its real backend. Never treat a `true`
`is_mock`/non-`unique3d`/non-`idm_vton` response as a validated final
result — the mobile app must surface this distinction to the user.

## Running the tests

No external services required — everything defaults to mock:

```
pytest tests/test_model7_garment_fitting.py tests/test_multiview_tryon_pipeline.py
```

## Unused-by-pipeline reference: original garment-isolation design

Kept in the codebase (fully tested, matches the original spec's required
interfaces) but no longer called from `multiview/pipeline.py` — useful if
a garment-only (not full-avatar-replacing) mode is wanted again.

### Multi-view image-to-3D (Hunyuan3D-2mv)

| Env var | Purpose | Default |
|---|---|---|
| `IMAGE_TO_3D_MV_PROVIDER` | `mock` \| `hunyuan3d_2mv` | `mock` |
| `HUNYUAN3D_2MV_ENDPOINT` | URL of an externally-hosted Hunyuan3D-2mv inference service | unset |
| `HUNYUAN3D_2MV_TIMEOUT_S` | request timeout, seconds | `300` |

Deploy Hunyuan3D-2mv (https://github.com/Tencent/Hunyuan3D-2) behind an
HTTP endpoint accepting `{"garment_type", "front_png_base64",
"back_png_base64"}` and returning `{"vertices", "faces", "uvs"?,
"landmarks", "texture_png_base64"?}`. The mesh this stage produces still
contains reconstructed human-body geometry —
`multiview/garment_isolation.py` strips that before fitting.

### Texture generation (Hunyuan3D-Paint)

| Env var | Purpose | Default |
|---|---|---|
| `TEXTURE_PROVIDER` | `mock` \| `hunyuan3d_paint` | `mock` |
| `HUNYUAN3D_PAINT_ENDPOINT` | URL of an externally-hosted Hunyuan3D-Paint inference service | unset |
| `HUNYUAN3D_PAINT_TIMEOUT_S` | request timeout, seconds | `180` |

Deploy Hunyuan3D-Paint (same repo as Hunyuan3D-2mv) behind an HTTP endpoint
accepting `{"vertices", "faces", "uvs", "garment_front_png_base64",
"garment_back_png_base64"}` and returning `{"texture_png_base64"}`. Always
textures from the original uploaded garment photos, never the try-on
images.

### Avatar fitting (Blender)

| Env var | Purpose | Default |
|---|---|---|
| `GARMENT_FIT_MOCK` | `1` = Python-only region deformation, `0` = real Blender | `1` |
| `BLENDER_EXECUTABLE` | path to the `blender` binary | `blender` |
| `BLENDER_TIMEOUT_S` | subprocess timeout, seconds | `180` |

See `scripts/blender_fit_garment_mesh.py` for the align → region scaling →
Surface Deform → Shrinkwrap → Cloth → Corrective Smooth → export sequence.
