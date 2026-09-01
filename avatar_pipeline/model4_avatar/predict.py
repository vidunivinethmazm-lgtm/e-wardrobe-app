"""
Model 4 — Avatar Generation: inference.

Given Model 1's body shape, Model 2's pose keypoints, and Model 3's skin
tone, build a condition vector and decode an avatar from the trained
generator (decoder).

`generate_avatar` is what the integration controller (Step 4 of the
pipeline) calls directly.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from .architecture import LATENT_DIM
from .condition_utils import (
    build_condition_vector,
    keypoints_to_pose_vector,
    skin_tone_to_rgb,
)


def load_decoder(model_dir):
    return tf.keras.models.load_model(os.path.join(model_dir, "decoder.keras"))


def generate_avatar(decoder, body_shape, keypoints_dict, skin_tone_result, seed=None):
    """body_shape: str from Model 1, e.g. "Hourglass"
    keypoints_dict: {joint_name: [x, y]}, from Model 2's
        `keypoints_to_avatar_params(...)["keypoints"]`
    skin_tone_result: dict from Model 3's `predict_skin_tone(...)`
        (must contain a "hex" key)

    Returns an RGBA PIL.Image — transparent background, opaque avatar —
    ready for Model 5 to dress.
    """
    pose_vector = keypoints_to_pose_vector(keypoints_dict)
    skin_rgb = skin_tone_to_rgb(skin_tone_result)
    condition = build_condition_vector(body_shape, pose_vector, skin_rgb)
    condition = tf.expand_dims(tf.constant(condition, dtype=tf.float32), 0)

    if seed is not None:
        z = tf.random.Generator.from_seed(seed).normal((1, LATENT_DIM))
    else:
        z = tf.random.normal((1, LATENT_DIM))

    image = decoder([z, condition], training=False).numpy()[0]
    image = np.clip(image, 0, 255).astype("uint8")
    return Image.fromarray(image, "RGBA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_models/model4_avatar")
    parser.add_argument("--body_shape", default="Hourglass")
    parser.add_argument("--skin_hex", default="#d7bd96")
    parser.add_argument("--output", default="avatar.png")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    decoder = load_decoder(args.model_dir)

    # CLI smoke test: render with the canonical standing pose.
    from .synthetic_avatars import CANONICAL_POSE

    keypoints_dict = {name: list(coords) for name, coords in CANONICAL_POSE.items()}
    skin_tone_result = {"hex": args.skin_hex}

    avatar = generate_avatar(decoder, args.body_shape, keypoints_dict, skin_tone_result, seed=args.seed)
    avatar.save(args.output)
    print(f"Saved avatar to {args.output}")
