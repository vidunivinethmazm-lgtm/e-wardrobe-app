"""Disk-backed storage for fitted-garment `.glb` (+ optional texture atlas
`.png`) results, keyed by an opaque `fit_id` (see
`avatar_pipeline.model7_garment_fitting.pipeline.run_garment_fitting`).
Mirrors `server.storage`'s pattern for avatar sessions: `POST
/api/avatars/<id>/fit-garment` computes and saves a result once, `GET
/api/avatars/<id>/fitted-garment/<fit_id>.glb` (and the sibling `-texture.png`
route) serve it back without recomputing the pipeline. The texture is
stored separately (not only embedded in the GLB) because React Native's
GLTFLoader can't decode a GLB's embedded bufferView image — the mobile app
loads it out-of-band, the same way the avatar's face texture is served."""

import re
import uuid
from pathlib import Path

FIT_STORAGE_DIR = Path(__file__).resolve().parent / "garment_fits"

# fit_ids are always uuid4().hex (see `save`); reject anything else so a
# crafted URL segment can't be used to read files outside FIT_STORAGE_DIR.
_VALID_ID = re.compile(r"^[0-9a-f]{32}$")


def save(glb_bytes: bytes, texture_png: bytes | None = None, fit_id: str | None = None) -> str:
    fit_id = fit_id or uuid.uuid4().hex
    FIT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    (FIT_STORAGE_DIR / f"{fit_id}.glb").write_bytes(glb_bytes)
    if texture_png is not None:
        (FIT_STORAGE_DIR / f"{fit_id}.png").write_bytes(texture_png)
    return fit_id


def get(fit_id: str) -> bytes | None:
    if not _VALID_ID.match(fit_id):
        return None
    path = FIT_STORAGE_DIR / f"{fit_id}.glb"
    if not path.is_file():
        return None
    return path.read_bytes()


def get_texture(fit_id: str) -> bytes | None:
    if not _VALID_ID.match(fit_id):
        return None
    path = FIT_STORAGE_DIR / f"{fit_id}.png"
    if not path.is_file():
        return None
    return path.read_bytes()


def save_tryon_previews(fit_id: str, front_png: bytes, back_png: bytes) -> None:
    """EXPERIMENTAL — stores the two virtual-try-on preview images produced
    by `pipeline_mode=multiview_tryon` (see `avatar_pipeline.model7_garment_
    fitting.multiview.pipeline`), keyed by the same `fit_id` as the fitted
    garment GLB/texture."""
    FIT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    (FIT_STORAGE_DIR / f"{fit_id}-tryon-front.png").write_bytes(front_png)
    (FIT_STORAGE_DIR / f"{fit_id}-tryon-back.png").write_bytes(back_png)


def get_tryon_preview(fit_id: str, side: str) -> bytes | None:
    if not _VALID_ID.match(fit_id) or side not in ("front", "back"):
        return None
    path = FIT_STORAGE_DIR / f"{fit_id}-tryon-{side}.png"
    if not path.is_file():
        return None
    return path.read_bytes()
