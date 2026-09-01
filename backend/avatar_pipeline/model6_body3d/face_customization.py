"""
Model 6 — 3D Body Reconstruction: re-customizes an existing avatar mesh from
the normalized facial-feature JSON produced by the mobile app's guided
capture flow (`mobile/src/services/facialFeatures.ts`).

`apply_face_customization` rebuilds `avatar_result.avatar_mesh_glb` from the
baked MakeHuman base mesh via `makehuman_mesh.build_personalized_glb`, using
the detected skin tone, face shape, a `head_radius` nudge from the jaw-width
ratio, and (if provided) the user's face photo as the head texture.
"""

import base64
import binascii
import dataclasses
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.avatar_pipeline.model3_skin_tone.color_utils import hex_to_rgb
from backend.avatar_pipeline.model6_body3d.makehuman_mesh import build_personalized_glb
from backend.avatar_pipeline.model6_body3d.params import PARAM_RANGES

# Decoded `faceImage` is resized to this before being passed as `face_crop`,
# matching `face_features.extract_face_features`'s `texture_size=128`.
_FACE_CROP_SIZE = 128

# `normalized.jawWidth` from facialFeatures.ts is `jawWidth / faceWidth`.
# faceMath.ts's `estimateFaceShape` treats ~0.78-0.85 as an "oval" face, so
# 0.8 is the ratio a deformation scale of 1.0 (i.e. the avatar's existing
# `head_radius`, unchanged) should correspond to.
_NEUTRAL_JAW_RATIO = 0.8

# height_cm used to rebuild the mesh -- matches avatar_builder.py's
# `_deform_base_model` base_height and mobile's bodyScaling.ts
# REFERENCE_HEIGHT. AvatarResult doesn't persist the user's actual height, so
# this is a per-gender approximation rather than their true height.
_HEIGHT_CM_BY_GENDER = {"male": 170, "female": 160, "neutral": 160}

_REQUIRED_FIELDS = {
    "faceShape": str,
    "jawWidth": (int, float),
    "noseWidth": (int, float),
    "eyeSpacing": (int, float),
    "skinTone": str,
    "hairColor": str,
}


def _validate_features(features):
    if not isinstance(features, dict):
        raise ValueError("features must be a JSON object")

    for key, types in _REQUIRED_FIELDS.items():
        if key not in features:
            raise ValueError(f"'{key}' is required")
        if isinstance(features[key], bool) or not isinstance(features[key], types):
            raise ValueError(f"'{key}' has the wrong type")

    for key in ("jawWidth", "noseWidth", "eyeSpacing"):
        if not 0 < features[key] < 2:
            raise ValueError(f"'{key}' must be between 0 and 2")

    try:
        hex_to_rgb(features["skinTone"])
    except ValueError:
        raise ValueError("'skinTone' must be a '#rrggbb' hex color")

    if "faceImage" in features and not isinstance(features["faceImage"], str):
        raise ValueError("'faceImage' must be a base64-encoded image")


def _decode_face_crop(face_image_b64):
    """Decodes a base64 JPEG/PNG into a `(_FACE_CROP_SIZE, _FACE_CROP_SIZE, 3)`
    uint8 RGB array for `mesh_builder.build_avatar_mesh`'s `face_crop`, or
    raises `ValueError` if it isn't a valid image.
    """
    try:
        raw = base64.b64decode(face_image_b64, validate=True)
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB").resize((_FACE_CROP_SIZE, _FACE_CROP_SIZE))
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        raise ValueError("'faceImage' must be a valid base64-encoded image")
    return np.asarray(image, dtype=np.uint8)


def apply_face_customization(avatar_result, features,
                             selfie_rgb=None, landmarks_2d=None,
                             blend_mode="feather", gender_override=None,
                             left_rgb=None, right_rgb=None,
                             left_landmarks=None, right_landmarks=None):
    """Rebuilds `avatar_result`'s mesh/facial_analysis from the normalized
    facial-feature JSON `features` (see `NormalizedFacialFeatures` in
    `mobile/src/types.ts`). If `features['faceImage']` (base64 JPEG/PNG) is
    present, it's textured onto the rebuilt head mesh - see
    `makehuman_mesh._build_texture`. Raises `ValueError` if `features` is
    malformed.

    When ``selfie_rgb`` and ``landmarks_2d`` are provided, uses Delaunay-
    triangulation warping (``face_texture_builder.build_head_texture_warped``)
    for photorealistic face textures instead of simple center-paste.

    When ``left_rgb``/``left_landmarks`` and ``right_rgb``/``right_landmarks``
    are ALSO provided (multi-angle mode), uses projective multi-angle texture
    mapping (``multi_angle_texture.build_multi_angle_texture``) to create a
    composite UV texture from all 3 angles — front + left + right profiles.
    This produces a 360°-natural face texture that looks correct from every
    viewing angle (see ``multi-angle-texture-plan.md``).
    """
    _validate_features(features)

    print(f"[apply_face_customization] selfie_rgb={'set ' + str(np.asarray(selfie_rgb).shape) if selfie_rgb is not None else 'None'}, "
          f"landmarks_2d={'None' if landmarks_2d is None else (np.asarray(landmarks_2d).shape if hasattr(landmarks_2d, '__len__') else landmarks_2d)}, "
          f"multi_angle={'yes' if (left_rgb is not None and right_rgb is not None) else 'no'}")

    skin_rgb = np.array(hex_to_rgb(features["skinTone"]), dtype=np.float32)

    face_crop = None
    face_width = None
    face_height = None
    prebuilt_texture_png = None  # for multi-angle composite

    # ── Multi-angle mode: build composite texture from 3 angles ─────────
    has_multi_angle = (
        selfie_rgb is not None and landmarks_2d is not None
        and left_rgb is not None and left_landmarks is not None
        and right_rgb is not None and right_landmarks is not None
        and len(landmarks_2d) >= 468
        and len(left_landmarks) >= 468
        and len(right_landmarks) >= 468
    )

    if has_multi_angle:
        from .multi_angle_texture import build_multi_angle_texture
        print(f"[apply_face_customization] Using MULTI-ANGLE texture pipeline "
              f"(front={landmarks_2d.shape}, left={left_landmarks.shape}, right={right_landmarks.shape})")
        prebuilt_texture_png = build_multi_angle_texture(
            selfie_rgb, left_rgb, right_rgb,
            landmarks_2d, left_landmarks, right_landmarks,
            tuple(int(round(c)) for c in skin_rgb[:3]),
            texture_size=512,
            blend_mode=blend_mode,
        )
        # Multi-angle already produced a full UV texture (skin + face composited).
        # Don't pass selfie_rgb/landmarks to build_personalized_glb — use the
        # pre-built texture instead.
        face_crop = None
        selfie_rgb_multi = None
        landmarks_2d_multi = None
    else:
        # ── Standard single-image path ──────────────────────────────────
        prebuilt_texture_png = None
        selfie_rgb_multi = selfie_rgb
        landmarks_2d_multi = landmarks_2d

        if features.get("faceImage"):
            face_crop = selfie_rgb if selfie_rgb is not None else _decode_face_crop(features["faceImage"])
            client_w = features.get("faceCropWidth")
            client_h = features.get("faceCropHeight")
            if client_w and client_h and client_w > 0 and client_h > 0:
                face_width = int(client_w)
                face_height = int(client_h)
                print(f"[apply_face_customization] Face crop size (from client): {face_width}×{face_height} px, "
                      f"aspect={face_width/face_height:.3f}")

    body3d_params = dict(avatar_result.body3d_params)
    lo, hi = PARAM_RANGES["head_radius"]
    head_radius = body3d_params.get("head_radius", (lo + hi) / 2)
    head_radius *= features["jawWidth"] / _NEUTRAL_JAW_RATIO
    body3d_params["head_radius"] = float(np.clip(head_radius, lo, hi))

    _gender = gender_override or avatar_result.gender
    height_cm = _HEIGHT_CM_BY_GENDER.get(_gender, _HEIGHT_CM_BY_GENDER["neutral"])
    avatar_mesh_glb = build_personalized_glb(
        body3d_params, height_cm, skin_rgb, features["faceShape"],
        face_crop=face_crop, gender=_gender,
        selfie_rgb=selfie_rgb_multi, landmarks_2d=landmarks_2d_multi,
        blend_mode=blend_mode,
        face_width=face_width,
        face_height=face_height,
        texture_png_override=prebuilt_texture_png,
    )

    facial_analysis = dict(avatar_result.facial_analysis or {})
    facial_analysis["face_shape"] = features["faceShape"]
    facial_analysis["hair_color"] = features["hairColor"]

    return dataclasses.replace(
        avatar_result,
        avatar_mesh_glb=avatar_mesh_glb,
        body3d_params=body3d_params,
        facial_analysis=facial_analysis,
    )
