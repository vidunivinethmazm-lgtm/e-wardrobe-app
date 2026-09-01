"""
Model 6 — 3D Body Reconstruction: personalizes the baked MakeHuman base mesh
(`assets/makehuman/{male,female}.glb`, see `scripts/generate_makehuman_avatars.py`
and `scripts/bake_makehuman_morphs.py`) for one user.

Given `body3d_params` (see `params.py`) and the detected face shape, derives
the same 6 morph-target weights `mobile/src/services/bodyScaling.ts` computes
for the on-device base model, bakes those deltas into final vertex positions,
uniformly scales to `height_cm`, and writes a head+body texture (flat skin
tone, with the user's face crop pasted over the head's UV region — the head
spans the full 0-1 UV via the spherical projection `bake_makehuman_morphs.py`
applied, while the body is pinned to a single texel at `BODY_UV`). Returns a
self-contained GLB, ready to serve as `avatar_mesh_glb`.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFilter
from pygltflib import GLTF2, BufferView, Image as GLTFImage, PbrMetallicRoughness, Sampler, Texture, TextureInfo

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "makehuman"

# Order matches mesh.extras.targetNames in assets/makehuman/{gender}.glb and
# mobile/src/types.ts's MorphTargetName.
MORPH_TARGET_NAMES = ["shoulderWidth", "hipWidth", "armLength", "legLength", "bodyType", "headWidth"]

# Ported from mobile/src/services/bodyScaling.ts, which derives these same 6
# weights for the on-device base model from body3d_params-equivalent
# fractions and the detected face shape.
_NEUTRAL_WIDTH_FRACTION = 0.25  # PARAM_RANGES midpoint -> morph weight 0
_SHOULDER_WIDTH_RANGE = (0.18, 0.32)  # params.PARAM_RANGES["shoulder_width"]
_HIP_WIDTH_RANGE = (0.16, 0.34)  # params.PARAM_RANGES["hip_width"]
_HEAD_WIDTH_BY_FACE_SHAPE = {"round": 0.6, "square": 0.5, "oblong": -0.5, "oval": 0.0, "heart": 0.1}

# `gender` values that don't have their own baked asset (params.PARAM_RANGES'
# "neutral" -> the female asset, matching face_customization's
# _HEIGHT_CM_BY_GENDER neutral==female height).
_ASSET_GENDER = {"male": "male", "female": "female", "neutral": "female"}

TEXTURE_SIZE = 512

# Fraction of the texture's width/height the face crop covers when pasted,
# centered on the head's front-facing UV (0.5, 0.5) -
# see bake_makehuman_morphs.py's _remap_uv_for_face_overlay.
_FACE_CROP_FRACTION = 0.5

# The head UV's v=0.5 (texture-vertical-centre) lands on the sphere's
# equator (_remap_uv_for_face_overlay's `0.5 - asin(norm_y)/pi`), which is
# ear-level, not eye-level: for norm_y in [-1, 1] over the head's vertical
# extent, forehead/eyes/chin actually fall around v=[0.25, 0.39, 0.67]
# (asin-weighted, not linear) - i.e. the face is naturally centred close to
# v=0.5 already with NO shift needed. A previous version of this constant
# (0.12) shifted the crop down so its centre landed at v=0.62, well past the
# chin and onto the neck/body UV - keep this at 0 (or use a small negative
# value to bias toward the forehead) instead of a positive downward shift.
_FACE_VERTICAL_OFFSET = 0.0


def _read_vec3_accessor(gltf: GLTF2, blob: bytes, accessor_index: int) -> np.ndarray:
    acc = gltf.accessors[accessor_index]
    bv = gltf.bufferViews[acc.bufferView]
    offset = bv.byteOffset + (acc.byteOffset or 0)
    count = acc.count
    stride = bv.byteStride or 12

    if stride == 12:
        data = blob[offset: offset + count * 12]
        return np.frombuffer(data, dtype=np.float32).reshape(count, 3).copy()

    positions = np.zeros((count, 3), dtype=np.float32)
    for i in range(count):
        o = offset + i * stride
        positions[i] = np.frombuffer(blob[o:o + 12], dtype=np.float32)
    return positions


def extract_base_color_texture(glb_bytes: bytes) -> bytes | None:
    """Pulls the embedded baseColorTexture PNG out of a personalized avatar
    GLB (as produced by `build_personalized_glb`/`_write_glb`), or `None` if
    it has none. Shared by the face-texture route and
    `garment_texture_paint`'s repaint step, both of which need to read back
    whatever texture is currently baked into a stored avatar."""
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = gltf.binary_blob() or b""
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.material is None:
                continue
            material = gltf.materials[prim.material]
            pbr = material.pbrMetallicRoughness
            if pbr and pbr.baseColorTexture is not None:
                texture = gltf.textures[pbr.baseColorTexture.index]
                image = gltf.images[texture.source]
                if image.bufferView is not None:
                    bv = gltf.bufferViews[image.bufferView]
                    return blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
    return None


def repaint_avatar_texture(avatar_mesh_glb: bytes, new_texture_png: bytes) -> bytes:
    """Returns a copy of `avatar_mesh_glb` with its baseColorTexture swapped
    for `new_texture_png`, geometry unchanged. Used by
    `garment_texture_paint` to bake a garment-painted texture back onto a
    stored avatar without touching its mesh."""
    gltf = GLTF2.load_from_bytes(avatar_mesh_glb)
    blob = gltf.binary_blob()
    prim = gltf.meshes[0].primitives[0]
    positions = _read_vec3_accessor(gltf, blob, prim.attributes.POSITION)
    return _write_glb(avatar_mesh_glb, positions, new_texture_png)


@lru_cache(maxsize=2)
def _load_base(gender: str) -> tuple[bytes, np.ndarray, dict[str, np.ndarray]]:
    """Returns (glb_bytes, base_positions (N,3), {morph name: deltas (N,3)})."""
    glb_bytes = (ASSETS_DIR / f"{gender}.glb").read_bytes()
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = gltf.binary_blob()

    prim = gltf.meshes[0].primitives[0]
    positions = _read_vec3_accessor(gltf, blob, prim.attributes.POSITION)

    target_names = gltf.meshes[0].extras["targetNames"]
    deltas = {
        name: _read_vec3_accessor(gltf, blob, target["POSITION"])
        for name, target in zip(target_names, prim.targets)
    }
    return glb_bytes, positions, deltas


def _width_fraction_to_weight(fraction: float, value_range: tuple[float, float]) -> float:
    lo, hi = value_range
    clipped = min(hi, max(lo, fraction))
    if clipped >= _NEUTRAL_WIDTH_FRACTION:
        span = hi - _NEUTRAL_WIDTH_FRACTION
        return min(1.0, max(0.0, (clipped - _NEUTRAL_WIDTH_FRACTION) / span)) if span > 0 else 0.0
    span = _NEUTRAL_WIDTH_FRACTION - lo
    return max(-1.0, min(0.0, (clipped - _NEUTRAL_WIDTH_FRACTION) / span)) if span > 0 else 0.0


def compute_morph_weights(body3d_params: dict, face_shape: str) -> dict[str, float]:
    """6 morph-target weights for `assets/makehuman/{gender}.glb`, mirroring
    `bodyScaling.ts`'s `computeBodyScale`. `armLength`/`legLength` stay 0:
    unlike the mobile app, this pipeline doesn't have the user's absolute
    height (just a per-gender default, see `_HEIGHT_CM_BY_GENDER`), so there's
    no height deviation to derive limb-length weights from."""
    shoulder_width = _width_fraction_to_weight(body3d_params["shoulder_width"], _SHOULDER_WIDTH_RANGE)
    hip_width = _width_fraction_to_weight(body3d_params["hip_width"], _HIP_WIDTH_RANGE)

    frame = (body3d_params["chest_width"] + body3d_params["hip_width"]) / 2
    waist_to_frame = body3d_params["waist_width"] / frame if frame > 0 else 1.0
    body_type = max(-1.0, min(1.0, (waist_to_frame - 1.0) * 2.5))

    head_width = _HEAD_WIDTH_BY_FACE_SHAPE.get(face_shape, 0.0)

    return {
        "shoulderWidth": shoulder_width,
        "hipWidth": hip_width,
        "armLength": 0.0,
        "legLength": 0.0,
        "bodyType": body_type,
        "headWidth": head_width,
    }


def _build_texture(skin_rgb, face_crop: np.ndarray | None,
                   selfie_rgb: np.ndarray | None = None,
                   landmarks_2d: np.ndarray | list | None = None,
                   blend_mode: str = "feather",
                   face_width: int | None = None,
                   face_height: int | None = None) -> bytes:
    """A `TEXTURE_SIZE`x`TEXTURE_SIZE` PNG: flat `skin_rgb` fill (covers the
    body's pinned UV texel and the head where no face crop is pasted), with
    the user's face warped onto the head's front-facing UV region.

    Uses Delaunay-triangulation warping with MakeHuman-specific UV anchors
    (``face_texture_builder._FACE_UV_ANCHORS_MAKEHUMAN``) when real MediaPipe
    landmarks are available, falling back to centre-paste + feathered edge
    when they are not.  The MakeHuman anchors are calibrated for the spherical
    UV projection baked by ``bake_makehuman_morphs.py`` (front = u=0.5,
    v=0=top, v=1=bottom).

    When ``face_width``/``face_height`` are provided, the crop size respects
    the selfie's face aspect ratio for a more natural fit on the 3D head.
    """
    # ── Delaunay-triangulation warp path (MakeHuman UV anchors) ─────────
    # ``_FACE_UV_ANCHORS_MAKEHUMAN`` is calibrated for the MakeHuman head's
    # spherical UV projection (see bake_makehuman_morphs.py's
    # ``_remap_uv_for_face_overlay``: front = u=0.5, v=0=top, v=1=bottom).
    # This gives a photorealistic warp instead of a simple centre-paste.
    use_warp = (
        selfie_rgb is not None
        and landmarks_2d is not None
        and hasattr(landmarks_2d, "shape")
        and landmarks_2d.shape[0] >= 468
    )
    print(f"[_build_texture] MakeHuman warp: use_warp={use_warp} "
          f"(selfie={'set' if selfie_rgb is not None else 'None'}, "
          f"landmarks={getattr(landmarks_2d, 'shape', 'None')})")
    if use_warp:
        from .face_texture_builder import build_head_texture_warped, _FACE_UV_ANCHORS_MAKEHUMAN
        warped_png = build_head_texture_warped(
            selfie_rgb, landmarks_2d,
            tuple(int(round(c)) for c in skin_rgb),
            texture_size=TEXTURE_SIZE,
            blend_mode=blend_mode,
            uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN,
        )
        # If the warp produced actual face content (not just flat fill), return it.
        if len(warped_png) > 900:
            print(f"[_build_texture] MakeHuman Delaunay warp path used ({len(warped_png)} bytes)")
            return warped_png
        print("[_build_texture] warp produced a flat/trivial result, falling back to centre-paste")

    # ── Legacy centre-paste fallback ────────────────────────────────────
    # Shift down by _FACE_VERTICAL_OFFSET so the crop's eye-line lands near
    # the head UV's equator (v=0.5), with a feathered edge so the crop blends
    # into the skin-tone fill instead of looking like a pasted sticker.
    fill = tuple(int(round(c)) for c in skin_rgb)
    image = PILImage.new("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), fill)

    if face_crop is not None:
        # ✅ Face Crop Aspect Ratio අනුව Target Size එක Adjust කරන්න
        base_size = int(TEXTURE_SIZE * _FACE_CROP_FRACTION)
        if face_width is not None and face_height is not None and face_width > 0 and face_height > 0:
            user_aspect = face_width / face_height
            if user_aspect > 1.0:
                # පළල් මුහුණ (Width > Height)
                new_w = base_size
                new_h = int(base_size / user_aspect)
            else:
                # දිගටි මුහුණ (Height >= Width)
                new_h = base_size
                new_w = int(base_size * user_aspect)
            new_w = max(8, new_w)
            new_h = max(8, new_h)
            face_image = PILImage.fromarray(face_crop.astype(np.uint8), "RGB").resize((new_w, new_h))
        else:
            # Fallback: හරි හතරැස්
            new_w = new_h = base_size
            face_image = PILImage.fromarray(face_crop.astype(np.uint8), "RGB").resize((new_w, new_h))

        start_x = (TEXTURE_SIZE - new_w) // 2
        start_y = (TEXTURE_SIZE - new_h) // 2 + int(TEXTURE_SIZE * _FACE_VERTICAL_OFFSET)
        start_y = max(0, min(start_y, TEXTURE_SIZE - new_h))

        # Elliptical mask: larger top inset (22%) excludes hair above forehead;
        # sides/bottom use 6% so ears and chin remain visible.
        top_inset = max(2, int(new_h * 0.22))
        side_inset = max(2, int(new_w * 0.06))
        bot_inset = max(2, int(new_h * 0.06))
        mask = PILImage.new("L", (new_w, new_h), 0)
        ImageDraw.Draw(mask).ellipse(
            [side_inset, top_inset, new_w - side_inset, new_h - bot_inset], fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(3, int(min(new_w, new_h) * 0.06))))

        image.paste(face_image, (start_x, start_y), mask)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _write_glb(glb_bytes: bytes, positions: np.ndarray, texture_png: bytes) -> bytes:
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = bytearray(gltf.binary_blob() or b"")

    prim = gltf.meshes[0].primitives[0]
    pos_acc = gltf.accessors[prim.attributes.POSITION]
    pos_bv = gltf.bufferViews[pos_acc.bufferView]
    offset = pos_bv.byteOffset + (pos_acc.byteOffset or 0)

    data = positions.astype(np.float32).tobytes()
    if len(data) != pos_acc.count * 12:
        raise ValueError(f"position size mismatch: {len(data)} != {pos_acc.count * 12}")
    blob[offset:offset + len(data)] = data
    pos_acc.min = positions.min(axis=0).tolist()
    pos_acc.max = positions.max(axis=0).tolist()

    pad = (-len(texture_png)) % 4
    image_offset = len(blob)
    blob.extend(texture_png + b"\x00" * pad)

    image_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=image_offset, byteLength=len(texture_png)))

    image_idx = len(gltf.images)
    gltf.images.append(GLTFImage(bufferView=image_bv_idx, mimeType="image/png"))

    sampler_idx = len(gltf.samplers)
    gltf.samplers.append(Sampler())

    texture_idx = len(gltf.textures)
    gltf.textures.append(Texture(source=image_idx, sampler=sampler_idx))

    material = gltf.materials[prim.material]
    if material.pbrMetallicRoughness is None:
        material.pbrMetallicRoughness = PbrMetallicRoughness()
    material.pbrMetallicRoughness.baseColorTexture = TextureInfo(index=texture_idx)
    material.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))

    return b"".join(gltf.save_to_bytes())


def build_personalized_glb(
    body3d_params: dict,
    height_cm: float,
    skin_rgb,
    face_shape: str,
    face_crop: np.ndarray | None = None,
    gender: str = "female",
    selfie_rgb: np.ndarray | None = None,
    landmarks_2d: np.ndarray | list | None = None,
    blend_mode: str = "feather",
    face_width: int | None = None,
    face_height: int | None = None,
    texture_png_override: bytes | None = None,
) -> bytes:
    """Personalized GLB: the baked MakeHuman base mesh for `gender`, deformed
    by the 6 morph weights derived from `body3d_params`/`face_shape`, scaled
    to `height_cm`, and textured with `skin_rgb` (+ warped face on the head).

    When ``selfie_rgb`` and ``landmarks_2d`` are provided, uses Delaunay-
    triangulation warping (``face_texture_builder.build_head_texture_warped``)
    for photorealistic face textures.  Otherwise falls back to centre-paste.

    When ``face_width`` and ``face_height`` are provided (pixel dimensions of
    the face crop from the user's selfie), the head is additionally scaled to
    match the face crop aspect ratio — ensuring the face texture fits
    naturally on the 3D head without UV warping distortion.

    When ``texture_png_override`` is provided (pre-built multi-angle composite
    texture PNG bytes), it is used directly — bypassing both the Delaunay warp
    and the centre-paste paths.  This is used by the multi-angle face texture
    pipeline (``multi_angle_texture.build_multi_angle_texture``).
    """
    glb_bytes, base_positions, deltas = _load_base(_ASSET_GENDER.get(gender, "female"))
    weights = compute_morph_weights(body3d_params, face_shape)

    # ✅ Selfie Face Crop එකේ Aspect Ratio අනුව Head Scale Factor ගණනය කරන්න
    if face_width is not None and face_height is not None and face_height > 0 and face_width > 0:
        user_aspect = face_width / face_height
        # Default aspect ratio for an "oval" face (neutral morph weight = 0.0).
        # MakeHuman base head ~0.25m wide × ~0.30m tall → aspect ≈ 0.83
        DEFAULT_ASPECT_RATIO = 0.83
        aspect_ratio_factor = user_aspect / DEFAULT_ASPECT_RATIO

        # Adjust headWidth morph weight: wider face (aspect > 1) → more headWidth
        # Scale factor of 1.0 → no change; 1.2 → +0.2; 0.8 → -0.2
        head_width_adj = np.clip((aspect_ratio_factor - 1.0) * 2.0, -0.5, 0.5)
        weights["headWidth"] = np.clip(weights["headWidth"] + head_width_adj, -1.0, 1.0)

        print(f"[build_personalized_glb] Head scaled by aspect ratio: user={user_aspect:.3f}, "
              f"default={DEFAULT_ASPECT_RATIO:.2f}, factor={aspect_ratio_factor:.3f}, "
              f"headWidth_adj={head_width_adj:+.3f} -> final={weights['headWidth']:.3f}")

    positions = base_positions.copy()
    for name, weight in weights.items():
        if weight:
            positions += deltas[name] * weight

    base_height = float(positions[:, 1].max() - positions[:, 1].min())
    if base_height > 0:
        positions *= (height_cm / 100.0) / base_height

    # ── Use pre-built texture (multi-angle composite) if provided ───────
    if texture_png_override is not None:
        print(f"[build_personalized_glb] Using pre-built texture ({len(texture_png_override)} bytes) — "
              f"bypassing Delaunay warp and centre-paste")
        return _write_glb(glb_bytes, positions, texture_png_override)

    # ── Standard texture building (Delaunay warp or centre-paste) ───────
    texture_png = _build_texture(skin_rgb, face_crop,
                                 selfie_rgb=selfie_rgb,
                                 landmarks_2d=landmarks_2d,
                                 blend_mode=blend_mode,
                                 face_width=face_width,
                                 face_height=face_height)
    return _write_glb(glb_bytes, positions, texture_png)
