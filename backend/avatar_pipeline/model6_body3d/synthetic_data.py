"""
Model 6 — 3D Body Reconstruction: synthetic dataset generator.

There is no public dataset pairing user photos with the body-mesh
parameters in `params.PARAM_NAMES`. As with Models 1 and 4, we bootstrap
with a programmatically generated dataset:

- "photo": Model 4's paper-doll renderer (`synthetic_avatars.render_avatar`)
  composited onto a neutral background — the same stylized stand-in Model 1
  uses for its silhouette branch, just RGB instead of a binary mask, since
  architecture.py's photo branch expects a 3-channel image.
- "aux": measurements (bust/waist/hips/height, scaled by
  `params.MEASUREMENT_SCALE`) + Model 1's body-shape one-hot
  (`condition_utils.body_shape_to_onehot`) + the same pose vector used to
  pose the renderer (`synthetic_avatars.pose_to_vector`) — exactly
  `architecture.AUX_DIM` values, in the same order
  `model6_body3d.predict.build_aux_vector` assembles them at inference time.
- target: `params.default_params_from_measurements(...)` — the same
  rule-based anthropometric approximation `server.mock_pipeline` uses
  directly (no CNN) — converted to the model's [0, 1] sigmoid output space
  via `params.params_to_sigmoid_vector`.

WHAT THIS BUYS YOU, AND WHAT IT DOESN'T: a fully working, end-to-end
regression pipeline (architecture, training loop, save/export) that can be
pointed at real (photo, body-mesh measurement) pairs later by swapping out
only this module — the architecture and target space (`params.PARAM_NAMES`)
do not change. Trained on synthetic data alone, the CNN mostly learns to
recover `default_params_from_measurements`'s rule from the paper-doll
render's body-shape silhouette and the aux measurements/pose, which are
already strong (in fact sufficient) signals for that rule — useful as a
working proof-of-concept and as infrastructure to fine-tune from.

Output: `images/*.png` (RGB, `img_size`x`img_size`) and `targets.npz` holding
the matching `aux` (N, AUX_DIM) and `targets` (N, PARAM_DIM) float32 arrays
plus `paths` (N,) image paths, relative to `output_dir`.

Usage:
    python -m avatar_pipeline.model6_body3d.synthetic_data \
        --output_dir data/model6_body3d --n_per_class 400
"""

import argparse
import os

import numpy as np
from PIL import Image

from backend.avatar_pipeline.model1_body_shape.synthetic_data import _sample_measurements
from backend.avatar_pipeline.model3_skin_tone.color_utils import MONK_SKIN_TONE_PALETTE, hex_to_rgb
from backend.avatar_pipeline.model4_avatar.condition_utils import BODY_SHAPE_NAMES, body_shape_to_onehot
from backend.avatar_pipeline.model4_avatar.synthetic_avatars import pose_to_vector, render_avatar, sample_pose

from .params import MEASUREMENT_SCALE, PARAM_DIM, default_params_from_measurements, params_to_sigmoid_vector

# Background the paper-doll render is composited onto — real photos rarely
# have a transparent background, so the photo branch should never see one.
_BACKGROUND_RGB = (235, 235, 235)


def _to_rgb(avatar_rgba):
    bg = Image.new("RGB", avatar_rgba.size, _BACKGROUND_RGB)
    bg.paste(avatar_rgba, mask=avatar_rgba.split()[3])
    return bg


def generate_dataset(output_dir, n_per_class=400, img_size=128, seed=42):
    rng = np.random.default_rng(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    n_samples = n_per_class * len(BODY_SHAPE_NAMES)
    targets = np.zeros((n_samples, PARAM_DIM), dtype=np.float32)
    aux_rows = []
    rel_paths = []

    idx = 0
    for body_shape in BODY_SHAPE_NAMES:
        for _ in range(n_per_class):
            bust, waist, hips = _sample_measurements(rng, body_shape)
            height = float(np.clip(rng.normal(162, 7), 145, 195))

            pose = sample_pose(rng)

            _, hex_color = MONK_SKIN_TONE_PALETTE[rng.integers(0, len(MONK_SKIN_TONE_PALETTE))]
            skin_rgb = np.clip(
                np.array(hex_to_rgb(hex_color), dtype=np.float32) + rng.normal(0, 6, size=3), 0, 255
            )

            avatar_rgba = render_avatar(body_shape, pose, skin_rgb, img_size=img_size)
            rel_path = os.path.join("images", f"{idx:05d}.png")
            _to_rgb(avatar_rgba).save(os.path.join(output_dir, rel_path))
            rel_paths.append(rel_path)

            measurements = np.array([bust, waist, hips, height], dtype=np.float32) / MEASUREMENT_SCALE
            aux_rows.append(
                np.concatenate([measurements, body_shape_to_onehot(body_shape), pose_to_vector(pose)])
            )

            target_params = default_params_from_measurements(body_shape, bust, waist, hips, height)
            targets[idx] = params_to_sigmoid_vector(target_params)

            idx += 1

    perm = rng.permutation(n_samples)
    aux = np.stack(aux_rows).astype(np.float32)[perm]
    targets = targets[perm]
    rel_paths = np.array(rel_paths)[perm]

    npz_path = os.path.join(output_dir, "targets.npz")
    np.savez(npz_path, aux=aux, targets=targets, paths=rel_paths)
    print(f"Wrote {n_samples} samples ({n_per_class} per class) to {images_dir}")
    return aux, targets, rel_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/model6_body3d")
    parser.add_argument("--n_per_class", type=int, default=400)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.n_per_class, args.img_size, args.seed)
