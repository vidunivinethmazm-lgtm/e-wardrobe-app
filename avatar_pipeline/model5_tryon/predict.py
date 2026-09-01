"""
Model 5 — Virtual Try-On: inference.

Two entry points:

- `try_on_avatar` (no trained model needed): warps a flat clothing product
  image onto Model 4's stylized avatar using classical TPS (tps_utils,
  geometry only — no trained weights) driven by Model 2's pose keypoints,
  then alpha-composites it on top. This is what the integration controller
  (Step 5) calls for the eWardrobe avatar, because the trained GMM+TOM
  (below) learn a photorealistic appearance model from VITON photos that
  does not transfer to a flat-shaded paper-doll avatar — but TPS warping
  driven by body landmarks is geometry, which transfers fine.

- `try_on_photo`: runs the trained GMM + TOM (architecture.py, train.py) on
  a real photo + person representation, exactly as in VITON/CP-VTON. Useful
  if/when the team points the avatar pipeline at real photos instead of (or
  in addition to) the stylized avatar.
"""

import argparse
import os

import numpy as np
from PIL import Image

from avatar_pipeline.model4_avatar.condition_utils import JOINT_NAMES
from .tps_utils import tps_warp

# Approximate landmark positions on a typical flat-lay / front-view product
# photo, normalized to [0, 1] (x, y) of the clothing image.
GARMENT_LANDMARKS = {
    "upper_body": {
        "left_shoulder": (0.18, 0.05),
        "right_shoulder": (0.82, 0.05),
        "left_hem": (0.12, 0.95),
        "right_hem": (0.88, 0.95),
        "center": (0.50, 0.45),
    },
    "lower_body": {
        "left_waist": (0.20, 0.05),
        "right_waist": (0.80, 0.05),
        "left_hem": (0.25, 0.95),
        "right_hem": (0.75, 0.95),
        "center": (0.50, 0.50),
    },
    "dress": {
        "left_shoulder": (0.18, 0.03),
        "right_shoulder": (0.82, 0.03),
        "left_hem": (0.15, 0.97),
        "right_hem": (0.85, 0.97),
        "center": (0.50, 0.50),
    },
}

# Each garment landmark maps to a point on the avatar, derived from Model 2's
# keypoints (either directly, e.g. "left_shoulder", or a derived midpoint).
AVATAR_ANCHORS = {
    "upper_body": {
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_hem": "left_hip",
        "right_hem": "right_hip",
        "center": "torso_center",
    },
    "lower_body": {
        "left_waist": "left_hip",
        "right_waist": "right_hip",
        "left_hem": "left_ankle",
        "right_hem": "right_ankle",
        "center": "hip_center",
    },
    "dress": {
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_hem": "left_ankle",
        "right_hem": "right_ankle",
        "center": "hip_center",
    },
}


def _midpoint(a, b):
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def _resolve_anchor(anchor, keypoints_dict):
    if anchor == "torso_center":
        shoulder_center = _midpoint(keypoints_dict["left_shoulder"], keypoints_dict["right_shoulder"])
        hip_center = _midpoint(keypoints_dict["left_hip"], keypoints_dict["right_hip"])
        return _midpoint(shoulder_center, hip_center)
    if anchor == "hip_center":
        return _midpoint(keypoints_dict["left_hip"], keypoints_dict["right_hip"])
    if anchor in JOINT_NAMES:
        return keypoints_dict[anchor][:2]
    raise ValueError(f"Unknown avatar anchor: {anchor}")


def try_on_avatar(avatar_rgba, clothing_rgb, clothing_mask, keypoints_dict, category="upper_body"):
    """avatar_rgba: (H,W,4) uint8, Model 4's output
    clothing_rgb: (Hc,Wc,3) uint8, the recommended clothing item's product photo
    clothing_mask: (Hc,Wc) or (Hc,Wc,1) uint8 {0,255}, foreground mask of the
        clothing photo (e.g. from a background-removal step in the
        recommendation pipeline)
    keypoints_dict: {joint_name: [x,y]} normalized [0,1], from Model 2
    category: one of GARMENT_LANDMARKS ("upper_body", "lower_body", "dress")

    Returns an RGBA PIL.Image, same size as `avatar_rgba`, with the warped
    clothing composited over the avatar.
    """
    if category not in GARMENT_LANDMARKS:
        raise ValueError(f"Unknown category '{category}', expected one of {list(GARMENT_LANDMARKS)}")

    h, w = avatar_rgba.shape[:2]
    ch, cw = clothing_rgb.shape[:2]

    landmarks = GARMENT_LANDMARKS[category]
    anchors = AVATAR_ANCHORS[category]

    src_points, dst_points = [], []
    for name, (lx, ly) in landmarks.items():
        src_points.append([lx * cw, ly * ch])
        ax, ay = _resolve_anchor(anchors[name], keypoints_dict)
        dst_points.append([ax * w, ay * h])

    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    if clothing_mask.ndim == 3:
        clothing_mask = clothing_mask[..., 0]

    warped_cloth = tps_warp(clothing_rgb, src_points, dst_points, output_size=(h, w))
    warped_mask = tps_warp(clothing_mask, src_points, dst_points, output_size=(h, w))

    alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
    avatar_rgb = avatar_rgba[..., :3].astype(np.float32)
    composite_rgb = alpha * warped_cloth.astype(np.float32) + (1 - alpha) * avatar_rgb

    composite_alpha = np.maximum(avatar_rgba[..., 3].astype(np.float32), warped_mask.astype(np.float32))

    result = np.dstack([composite_rgb, composite_alpha]).astype(np.uint8)
    return Image.fromarray(result, "RGBA")


# ---------------------------------------------------------------------------
# Trained GMM + TOM, for real photos (CP-VTON / VITON style)
# ---------------------------------------------------------------------------

def load_tryon_models(model_dir):
    import tensorflow as tf

    gmm = tf.keras.models.load_model(os.path.join(model_dir, "gmm.keras"))
    tom = tf.keras.models.load_model(os.path.join(model_dir, "tom.keras"))
    return gmm, tom


def try_on_photo(gmm, tom, person_repr, cloth, cloth_mask):
    """person_repr: (H,W,16) float32 [0,1] (pose_repr.build_person_representation)
    cloth: (H,W,3) float32/uint8 [0,255]
    cloth_mask: (H,W,1) float32/uint8 {0,255}
    Returns the composited person image (H,W,3) uint8.
    """
    import tensorflow as tf

    from .architecture import apply_gmm_warp

    person_repr_b = tf.expand_dims(tf.cast(person_repr, tf.float32), 0)
    cloth_b = tf.expand_dims(tf.cast(cloth, tf.float32), 0)
    mask_b = tf.expand_dims(tf.cast(cloth_mask, tf.float32), 0)

    control_points = gmm([person_repr_b, cloth_b, mask_b], training=False)
    warped_cloth, warped_mask = apply_gmm_warp(cloth_b, mask_b, control_points)

    rendered, comp_mask = tom([person_repr_b, warped_cloth, warped_mask], training=False)
    composite = comp_mask * warped_cloth + (1.0 - comp_mask) * rendered
    composite = tf.clip_by_value(composite, 0, 255)
    return composite.numpy()[0].astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar", required=True, help="Path to Model 4's RGBA avatar PNG")
    parser.add_argument("--clothing", required=True, help="Path to the clothing product photo")
    parser.add_argument("--clothing_mask", required=True, help="Path to the clothing foreground mask")
    parser.add_argument("--category", default="upper_body", choices=list(GARMENT_LANDMARKS))
    parser.add_argument("--output", default="dressed_avatar.png")
    args = parser.parse_args()

    from avatar_pipeline.model4_avatar.synthetic_avatars import CANONICAL_POSE

    avatar = np.array(Image.open(args.avatar).convert("RGBA"))
    clothing = np.array(Image.open(args.clothing).convert("RGB"))
    clothing_mask = np.array(Image.open(args.clothing_mask).convert("L"))

    keypoints_dict = {name: list(coords) for name, coords in CANONICAL_POSE.items()}

    result = try_on_avatar(avatar, clothing, clothing_mask, keypoints_dict, category=args.category)
    result.save(args.output)
    print(f"Saved dressed avatar to {args.output}")
