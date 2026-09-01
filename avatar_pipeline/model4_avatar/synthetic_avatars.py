"""
Model 4 — Avatar Generation: synthetic "paper doll" dataset generator.

There is no public dataset of stylized 2D avatars labeled with
(body shape, pose, skin tone) — the exact condition triple Models 1-3
produce. As with Models 1 and 3, we bootstrap with a programmatically
rendered dataset: a simple paper-doll figure whose proportions, pose, and
skin color are driven directly by the same condition vector the CVAE-GAN is
trained to consume (see condition_utils.build_condition_vector).

WHAT THIS BUYS YOU, AND WHAT IT DOESN'T:
A fully working, end-to-end-trainable pipeline (architecture, losses, data
plumbing, save/export) that you can point at a REAL stylized-avatar dataset
later (e.g. commissioned character art spanning body types, poses, and skin
tones, or a licensed asset pack) by swapping out only this module. Trained on
synthetic data alone, the CVAE-GAN will mostly learn to reproduce this
programmatic renderer's style — useful as a working proof-of-concept and as
infrastructure to fine-tune from, but it will not "invent" an art style the
renderer doesn't already have. The architecture, condition vector, and
training loop do not change when you swap in real data.

Output: `images/*.png` (RGBA, transparent background) and `conditions.npz`
holding the matching (CONDITION_DIM,) condition vector per image.

Usage:
    python -m avatar_pipeline.model4_avatar.synthetic_avatars \
        --output_dir data/model4_avatar --n_samples 4000
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from .condition_utils import BODY_SHAPE_NAMES, JOINT_NAMES, build_condition_vector

# Canonical standing pose, normalized to [0, 1] (y increases downward).
# `sample_pose` jitters this per-sample for pose diversity.
CANONICAL_POSE = {
    "left_shoulder": (0.40, 0.28),
    "right_shoulder": (0.60, 0.28),
    "left_elbow": (0.32, 0.45),
    "right_elbow": (0.68, 0.45),
    "left_wrist": (0.28, 0.62),
    "right_wrist": (0.72, 0.62),
    "left_hip": (0.43, 0.55),
    "right_hip": (0.57, 0.55),
    "left_knee": (0.42, 0.75),
    "right_knee": (0.58, 0.75),
    "left_ankle": (0.41, 0.95),
    "right_ankle": (0.59, 0.95),
}

# Torso/hip width multipliers per body shape, relative to shoulder width.
# These mirror the bust/waist/hip relationships used to LABEL Model 1's
# training data (synthetic_data.classify_body_shape), so a given body-shape
# class renders with the silhouette that Model 1 was trained to recognize.
BODY_SHAPE_PROFILE = {
    "Hourglass": {"waist": 0.62, "hip": 1.02},
    "Pear": {"waist": 0.78, "hip": 1.18},
    "Apple": {"waist": 1.05, "hip": 0.95},
    "Rectangle": {"waist": 0.92, "hip": 0.98},
    "InvertedTriangle": {"waist": 0.85, "hip": 0.78},
}

LIMB_PAIRS = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def sample_pose(rng, jitter=0.06):
    """Returns a {joint_name: (x, y)} dict, jittered around CANONICAL_POSE."""
    pose = {}
    for name, (x, y) in CANONICAL_POSE.items():
        pose[name] = (
            float(np.clip(x + rng.uniform(-jitter, jitter), 0.05, 0.95)),
            float(np.clip(y + rng.uniform(-jitter, jitter), 0.05, 0.98)),
        )
    return pose


def pose_to_vector(pose):
    """Inverse of `condition_utils.keypoints_to_pose_vector` for a
    {joint_name: (x, y)} pose dict — flattens in JOINT_NAMES order."""
    return np.array(
        [coord for name in JOINT_NAMES for coord in pose[name]], dtype=np.float32
    )


def render_avatar(body_shape, pose, skin_rgb, img_size=128, limb_width_frac=0.05):
    """Renders an RGBA paper-doll avatar.

    body_shape: one of BODY_SHAPE_NAMES
    pose: {joint_name: (x, y)} normalized [0, 1] coordinates
    skin_rgb: (3,) array-like, 0-255
    Returns a PIL.Image in "RGBA" mode, transparent background.
    """
    img = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    skin = tuple(int(c) for c in skin_rgb) + (255,)

    def px(p):
        return (p[0] * img_size, p[1] * img_size)

    ls, rs = pose["left_shoulder"], pose["right_shoulder"]
    lh, rh = pose["left_hip"], pose["right_hip"]
    shoulder_w = abs(rs[0] - ls[0]) * img_size
    profile = BODY_SHAPE_PROFILE[body_shape]
    waist_w = shoulder_w * profile["waist"]
    hip_w = shoulder_w * profile["hip"]

    shoulder_y = (ls[1] + rs[1]) / 2 * img_size
    hip_y = (lh[1] + rh[1]) / 2 * img_size
    waist_y = (shoulder_y + hip_y) / 2
    cx = (ls[0] + rs[0] + lh[0] + rh[0]) / 4 * img_size

    # Head
    head_r = shoulder_w * 0.32
    head_cy = shoulder_y - head_r * 1.6
    draw.ellipse(
        [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=skin
    )

    # Torso: shoulders -> waist -> hips, widths shaped by body_shape
    torso_pts = [
        (cx - shoulder_w / 2, shoulder_y),
        (cx - waist_w / 2, waist_y),
        (cx - hip_w / 2, hip_y),
        (cx + hip_w / 2, hip_y),
        (cx + waist_w / 2, waist_y),
        (cx + shoulder_w / 2, shoulder_y),
    ]
    draw.polygon(torso_pts, fill=skin)

    # Limbs as thick rounded lines connecting pose joints
    limb_width = max(2, int(img_size * limb_width_frac))
    for a, b in LIMB_PAIRS:
        draw.line([px(pose[a]), px(pose[b])], fill=skin, width=limb_width)
        for p in (pose[a], pose[b]):
            r = limb_width / 2
            cxp, cyp = px(p)
            draw.ellipse([cxp - r, cyp - r, cxp + r, cyp + r], fill=skin)

    return img


def generate_dataset(output_dir, n_samples=4000, img_size=128, seed=42):
    from avatar_pipeline.model3_skin_tone.color_utils import (
        MONK_SKIN_TONE_PALETTE,
        hex_to_rgb,
    )

    rng = np.random.default_rng(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    from .condition_utils import CONDITION_DIM

    conditions = np.zeros((n_samples, CONDITION_DIM), dtype=np.float32)
    rel_paths = []

    for i in range(n_samples):
        body_shape = BODY_SHAPE_NAMES[rng.integers(0, len(BODY_SHAPE_NAMES))]
        pose = sample_pose(rng)

        _, hex_color = MONK_SKIN_TONE_PALETTE[rng.integers(0, len(MONK_SKIN_TONE_PALETTE))]
        skin_rgb = np.array(hex_to_rgb(hex_color), dtype=np.float32)
        # Small color jitter so the model sees a continuum, not 10 fixed colors.
        skin_rgb = np.clip(skin_rgb + rng.normal(0, 6, size=3), 0, 255)

        img = render_avatar(body_shape, pose, skin_rgb, img_size=img_size)
        rel_path = os.path.join("images", f"{i:05d}.png")
        img.save(os.path.join(output_dir, rel_path))
        rel_paths.append(rel_path)

        conditions[i] = build_condition_vector(
            body_shape, pose_to_vector(pose), skin_rgb / 255.0
        )

    np.savez(
        os.path.join(output_dir, "conditions.npz"),
        conditions=conditions,
        paths=np.array(rel_paths),
    )
    print(f"Wrote {n_samples} avatars to {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/model4_avatar")
    parser.add_argument("--n_samples", type=int, default=4000)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.n_samples, args.img_size, args.seed)
