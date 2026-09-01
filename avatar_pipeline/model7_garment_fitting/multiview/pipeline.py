"""
EXPERIMENTAL — Model 7 "multiview_tryon" research pipeline. NOT the default
(`GARMENT_PIPELINE_MODE=adaptive_template` remains the default — see
`fitting_types.py`); only used when a caller explicitly opts in.

PIVOT: this pipeline reconstructs a **full 3D human avatar** directly from
the virtual-try-on image(s) via Unique3D, and that mesh becomes the avatar
going forward — it is not fitted onto the pre-existing MakeHuman avatar.
This was an explicit, repeated user decision (see `avatar3d_providers.py`'s
docstring); the original garment-isolation-then-fit design
(`garment_isolation.py`, `mesh3d_providers.py`, `texture_providers.py`) is
still present in the codebase but no longer called from here.

    user full-body front/back photos + garment front/back photos
    -> tryon_providers.VirtualTryOnProvider.generate()
       (front/back photos of the SAME user wearing the uploaded garment)
    -> avatar3d_providers.FullAvatarImageTo3DProvider.generate()
       (Unique3D image-to-3D reconstruction of the person wearing the
       garment — this mesh IS the avatar, not a garment to fit onto one)
    -> glb_writer.validate_mesh_geometry() / write_mesh_glb()
    -> MultiviewFittingResult (is_full_avatar_replacement=True)

This module never silently substitutes a template/mock result for a
provider that's configured but fails — see each provider module's
docstring. It also never claims real AI processing happened when a mock
provider ran: `MultiviewFittingResult` carries per-stage provider names and
`is_real_3d_generation`/`is_mock`, which `server/app.py` passes straight
through to the API response.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..fitting_types import GarmentFittingError
from ..garment_features import compute_normalized_features
from ..garment_keypoints import extract_keypoints
from ..garment_segmentation import segment_garment
from ..glb_writer import validate_mesh_geometry, write_mesh_glb
from ..pipeline import MAX_IMAGE_DIM, MIN_IMAGE_DIM, decode_garment_image, validate_garment_type
from .avatar3d_providers import FULL_AVATAR_3D_PROVIDER, get_full_avatar_provider
from .tryon_providers import VIRTUAL_TRYON_PROVIDER, get_virtual_tryon_provider
from .types import MultiviewFittingResult


def decode_person_image(file_storage, side: str, required: bool = True) -> np.ndarray | None:
    """Validates and decodes an uploaded full-body person photo — same
    bounds as `pipeline.decode_garment_image`, distinct error field name.

    `side="back"` is optional (`required=False`): when omitted, the caller
    (`run_multiview_tryon_fitting`) auto-generates a back view from the
    front photo via Gemini instead of rejecting the request — the mobile
    app only asks the user for one photo of themselves."""
    if file_storage is None or not getattr(file_storage, "filename", None):
        if not required:
            return None
        raise GarmentFittingError(f"'person_{side}' file is required for pipeline_mode=multiview_tryon")

    try:
        image = Image.open(file_storage.stream)
        image.load()
        rgb = np.array(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise GarmentFittingError(f"'person_{side}' could not be read as an image") from exc

    h, w = rgb.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise GarmentFittingError(f"'person_{side}' is too small ({w}x{h}px, minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}px)")
    if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
        raise GarmentFittingError(f"'person_{side}' is too large ({w}x{h}px, maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}px)")

    return rgb


def _encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(png_bytes: bytes, label: str) -> np.ndarray:
    try:
        return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise GarmentFittingError(f"{label} image returned by the virtual try-on provider is invalid") from exc


def _normalize_person_photo(person_rgb: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Runs `gemini_client.normalize_person_photo` on an uploaded person
    photo before it reaches a virtual-try-on provider — real phone photos
    (cropped limbs, non-frontal angle, busy background, ...) routinely fail
    a try-on model's own pose-detection step (e.g. Replicate's IDM-VTON
    raising "list index out of range"); regenerating a clean, front-facing,
    full-body studio version of the SAME person fixes the vast majority of
    those failures.

    Returns `(possibly-normalized rgb, warning|None)`. Mock mode or a
    missing GEMINI_API_KEY: returns the original image unchanged with a
    warning explaining pose-detection failures are more likely. A
    configured-but-failing Gemini call also falls back to the original
    image (with a warning) rather than blocking the whole pipeline on a
    best-effort preprocessing step."""
    from server.ai_tryon import gemini_client

    if gemini_client.MOCK or not gemini_client.GEMINI_API_KEY:
        return person_rgb, (
            "photo normalization (Gemini) isn't configured (set GEMINI_API_KEY and AI_TRYON_MOCK=0) — "
            "your photo was sent to the virtual try-on model as-is, which is more likely to fail its "
            "pose-detection step on non-studio photos"
        )

    try:
        normalized_png = gemini_client.normalize_person_photo(_encode_png(person_rgb))
        return _decode_png(normalized_png, "normalized person photo"), None
    except Exception as exc:
        return person_rgb, f"could not normalize your photo via Gemini ({exc}) — sent it to the virtual try-on model as-is"


def _resolve_person_back(person_front_rgb: np.ndarray, person_back_storage) -> tuple[np.ndarray, bool, str | None]:
    """Returns `(person_back_rgb, was_auto_generated, warning)`. If
    `person_back_storage` is a real upload, decodes and returns it
    unchanged (`was_auto_generated=False`, no warning). Otherwise
    synthesizes a back view from `person_front_rgb` via Gemini — the mobile
    app only asks the user for one photo of themselves. `was_auto_generated`
    is only `True` when that synthesis actually ran Gemini for real (not
    its identity-echo mock mode); when it didn't, a warning string explains
    the front photo was reused unchanged as a stand-in back view."""
    person_back_rgb = decode_person_image(person_back_storage, "back", required=False)
    if person_back_rgb is not None:
        person_back_rgb, normalize_warning = _normalize_person_photo(person_back_rgb)
        return person_back_rgb, False, normalize_warning

    from server.ai_tryon import gemini_client

    try:
        back_view_png = gemini_client.generate_back_view_image(_encode_png(person_front_rgb))
    except Exception as exc:
        raise GarmentFittingError(f"could not auto-generate a back view from 'person_front': {exc}") from exc
    person_back_rgb = _decode_png(back_view_png, "auto-generated person back")

    if gemini_client.MOCK:
        return person_back_rgb, False, (
            "no back photo of yourself was uploaded and Gemini back-view generation isn't "
            "configured (set GEMINI_API_KEY and AI_TRYON_MOCK=0) — your front photo was reused "
            "unchanged as a stand-in back view"
        )
    return person_back_rgb, True, None


def run_multiview_tryon_fitting(
    person_front_storage,
    person_back_storage,
    garment_front_storage,
    garment_back_storage,
    garment_type: str | None,
) -> MultiviewFittingResult:
    """Runs the full EXPERIMENTAL multiview_tryon pipeline. Raises
    `GarmentFittingError` for any invalid/missing input, or for a
    configured-but-failing provider at any stage — never silently falls
    back to a different provider than the one selected via env vars."""
    garment_type = validate_garment_type(garment_type)

    person_front_rgb = decode_person_image(person_front_storage, "front")
    garment_front_rgb = decode_garment_image(garment_front_storage, "front")
    garment_back_rgb = decode_garment_image(garment_back_storage, "back")
    person_front_rgb, normalize_warning = _normalize_person_photo(person_front_rgb)
    person_back_rgb, _, back_view_warning = _resolve_person_back(person_front_rgb, person_back_storage)

    # 2D diagnostic features (as in the adaptive_template response) — purely
    # descriptive, doesn't drive avatar reconstruction below.
    front_seg = segment_garment(garment_front_rgb, "front")
    back_seg = segment_garment(garment_back_rgb, "back")
    front_keypoints = extract_keypoints(front_seg, garment_type, "front")
    back_keypoints = extract_keypoints(back_seg, garment_type, "back")
    features = compute_normalized_features(
        front_keypoints, back_keypoints, front_seg.bbox, back_seg.bbox, garment_type,
    )

    # ── Stage 1: virtual try-on ─────────────────────────────────────────
    tryon_provider = get_virtual_tryon_provider()
    tryon_result = tryon_provider.generate(
        person_front_rgb, person_back_rgb, garment_front_rgb, garment_back_rgb, garment_type,
    )
    if tryon_result is None:
        raise GarmentFittingError(
            "virtual try-on is unavailable — set VIRTUAL_TRYON_PROVIDER=idm_vton with a configured "
            "IDM_VTON_ENDPOINT (or the default free IDM_VTON_HF_SPACE), or use the default mock "
            "provider for local dev"
        )
    tryon_front_png, tryon_back_png = tryon_result
    tryon_front_rgb = _decode_png(tryon_front_png, "front try-on")
    tryon_back_rgb = _decode_png(tryon_back_png, "back try-on")

    # ── Stage 2: full-avatar image-to-3D reconstruction (Unique3D) ──────
    # This mesh IS the avatar going forward — not fitted onto a separate
    # existing one. See avatar3d_providers.py's docstring for that decision.
    avatar_provider = get_full_avatar_provider()
    avatar_mesh = avatar_provider.generate(tryon_front_rgb, tryon_back_rgb)
    if avatar_mesh is None:
        raise GarmentFittingError(
            "full-avatar 3D reconstruction is unavailable — set FULL_AVATAR_3D_PROVIDER=unique3d "
            "with a configured UNIQUE3D_HF_TOKEN (or UNIQUE3D_AVATAR_ENDPOINT), or use the default "
            "mock provider for local dev"
        )
    is_real_3d_generation = not avatar_mesh.is_mock

    validate_mesh_geometry(avatar_mesh.vertices, avatar_mesh.faces)
    glb_bytes = write_mesh_glb(avatar_mesh.vertices, avatar_mesh.faces, avatar_mesh.uvs, avatar_mesh.texture_png)

    is_mock = avatar_mesh.is_mock
    warnings: list[str] = []
    if is_mock:
        warnings.append(
            "this is an EXPERIMENTAL, non-production preview (mock virtual try-on and/or "
            "avatar 3D generation) — not a validated final result"
        )
    if normalize_warning:
        warnings.append(normalize_warning)
    if back_view_warning:
        warnings.append(back_view_warning)

    return MultiviewFittingResult(
        fit_id=uuid.uuid4().hex,
        garment_type=garment_type,
        features=features,
        glb_bytes=glb_bytes,
        texture_png=avatar_mesh.texture_png,
        tryon_front_png=tryon_front_png,
        tryon_back_png=tryon_back_png,
        virtual_tryon_provider=VIRTUAL_TRYON_PROVIDER,
        image_to_3d_provider=FULL_AVATAR_3D_PROVIDER,
        is_real_3d_generation=is_real_3d_generation,
        is_full_avatar_replacement=True,
        status="ready",
        warnings=warnings,
        is_mock=is_mock,
    )


def run_multiview_tryon_fitting_top_and_bottom(
    person_front_storage,
    person_back_storage,
    top_front_storage,
    top_back_storage,
    bottom_front_storage,
    bottom_back_storage,
) -> MultiviewFittingResult:
    """Same EXPERIMENTAL pipeline as `run_multiview_tryon_fitting`, but for
    a top + bottom outfit (two separate garments) instead of one garment
    (a dress, or a single top/bottom). Chains two virtual-try-on passes
    through `tryon_providers.VirtualTryOnProvider.generate()` — the top is
    applied to the person photos first, then the bottom is applied to THAT
    result (not the original person photos), so the final image shows both
    garments worn together:

        person photos + top photos  -> generate(..., "upper_body")
                                     -> (person wearing the top)
        (person wearing the top) + bottom photos -> generate(..., "lower_body")
                                     -> (person wearing both)
        -> avatar3d_providers.FullAvatarImageTo3DProvider.generate()

    Raises `GarmentFittingError` for any invalid/missing input, or for a
    configured-but-failing provider at either try-on pass or the 3D
    reconstruction stage.
    """
    person_front_rgb = decode_person_image(person_front_storage, "front")
    top_front_rgb = decode_garment_image(top_front_storage, "front")
    top_back_rgb = decode_garment_image(top_back_storage, "back")
    bottom_front_rgb = decode_garment_image(bottom_front_storage, "front")
    bottom_back_rgb = decode_garment_image(bottom_back_storage, "back")
    person_front_rgb, normalize_warning = _normalize_person_photo(person_front_rgb)
    person_back_rgb, _, back_view_warning = _resolve_person_back(person_front_rgb, person_back_storage)

    # 2D diagnostic features — computed from the top only (purely
    # descriptive, doesn't drive avatar reconstruction below; there's no
    # single-garment feature set that meaningfully covers two garments).
    front_seg = segment_garment(top_front_rgb, "front")
    back_seg = segment_garment(top_back_rgb, "back")
    front_keypoints = extract_keypoints(front_seg, "upper_body", "front")
    back_keypoints = extract_keypoints(back_seg, "upper_body", "back")
    features = compute_normalized_features(
        front_keypoints, back_keypoints, front_seg.bbox, back_seg.bbox, "upper_body",
    )

    tryon_provider = get_virtual_tryon_provider()

    # ── Pass 1: apply the top ────────────────────────────────────────────
    top_tryon_result = tryon_provider.generate(
        person_front_rgb, person_back_rgb, top_front_rgb, top_back_rgb, "upper_body",
    )
    if top_tryon_result is None:
        raise GarmentFittingError(
            "virtual try-on is unavailable — set VIRTUAL_TRYON_PROVIDER=idm_vton, gemini, or "
            "replicate (with the matching credentials configured), or use the default mock "
            "provider for local dev"
        )
    top_tryon_front_png, top_tryon_back_png = top_tryon_result
    top_tryon_front_rgb = _decode_png(top_tryon_front_png, "top front try-on")
    top_tryon_back_rgb = _decode_png(top_tryon_back_png, "top back try-on")

    # ── Pass 2: apply the bottom on top of pass 1's result ───────────────
    final_tryon_result = tryon_provider.generate(
        top_tryon_front_rgb, top_tryon_back_rgb, bottom_front_rgb, bottom_back_rgb, "lower_body",
    )
    if final_tryon_result is None:
        raise GarmentFittingError(
            "virtual try-on is unavailable for the bottom garment pass — see the top-garment error "
            "above for provider setup"
        )
    tryon_front_png, tryon_back_png = final_tryon_result
    tryon_front_rgb = _decode_png(tryon_front_png, "front try-on")
    tryon_back_rgb = _decode_png(tryon_back_png, "back try-on")

    # ── Stage 2: full-avatar image-to-3D reconstruction (Unique3D) ──────
    avatar_provider = get_full_avatar_provider()
    avatar_mesh = avatar_provider.generate(tryon_front_rgb, tryon_back_rgb)
    if avatar_mesh is None:
        raise GarmentFittingError(
            "full-avatar 3D reconstruction is unavailable — set FULL_AVATAR_3D_PROVIDER=unique3d "
            "with a configured UNIQUE3D_HF_TOKEN (or UNIQUE3D_AVATAR_ENDPOINT), or use the default "
            "mock provider for local dev"
        )
    is_real_3d_generation = not avatar_mesh.is_mock

    validate_mesh_geometry(avatar_mesh.vertices, avatar_mesh.faces)
    glb_bytes = write_mesh_glb(avatar_mesh.vertices, avatar_mesh.faces, avatar_mesh.uvs, avatar_mesh.texture_png)

    is_mock = avatar_mesh.is_mock
    warnings: list[str] = []
    if is_mock:
        warnings.append(
            "this is an EXPERIMENTAL, non-production preview (mock virtual try-on and/or "
            "avatar 3D generation) — not a validated final result"
        )
    if normalize_warning:
        warnings.append(normalize_warning)
    if back_view_warning:
        warnings.append(back_view_warning)

    return MultiviewFittingResult(
        fit_id=uuid.uuid4().hex,
        garment_type="upper_body",  # descriptive only — this result covers a top + bottom outfit
        features=features,
        glb_bytes=glb_bytes,
        texture_png=avatar_mesh.texture_png,
        tryon_front_png=tryon_front_png,
        tryon_back_png=tryon_back_png,
        virtual_tryon_provider=VIRTUAL_TRYON_PROVIDER,
        image_to_3d_provider=FULL_AVATAR_3D_PROVIDER,
        is_real_3d_generation=is_real_3d_generation,
        is_full_avatar_replacement=True,
        status="ready",
        warnings=warnings,
        is_mock=is_mock,
    )
