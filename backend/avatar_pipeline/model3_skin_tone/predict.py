"""
Model 3 — Skin Tone Detection: inference.

Pipeline: face crop -> white-balance -> CNN predicts dominant skin LAB ->
nearest palette match -> avatar rendering parameters (base/shadow/highlight
colors for the avatar's skin layer).

`predict_skin_tone` is what the integration controller (Step 4 of the
pipeline) calls directly.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from .color_utils import (
    MONK_SKIN_TONE_PALETTE,
    hex_to_rgb,
    nearest_palette_match,
    white_balance_scene,
)
from .face_crop import detect_and_crop_face


def load_model(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return model, config


def _shade(rgb, factor):
    """factor > 1 lightens, < 1 darkens (simple multiplicative shading)."""
    return tuple(int(np.clip(c * factor, 0, 255)) for c in rgb)


def predict_skin_tone(image, model, config, palette=MONK_SKIN_TONE_PALETTE):
    """image: HxWx3 uint8 RGB numpy array (full photo)."""
    image = white_balance_scene(image)
    face = detect_and_crop_face(image, output_size=config["image_size"])
    if face is None:
        raise ValueError("No face detected in the input image.")

    inp = tf.expand_dims(tf.cast(face, tf.float32), 0)

    lab = model.predict(inp, verbose=0)[0] * 255.0
    match = nearest_palette_match(lab, palette)

    base_rgb = hex_to_rgb(match["hex"])
    return {
        **match,
        "avatar_render": {
            "base_color": "#%02x%02x%02x" % base_rgb,
            "shadow_color": "#%02x%02x%02x" % _shade(base_rgb, 0.78),
            "highlight_color": "#%02x%02x%02x" % _shade(base_rgb, 1.15),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_dir", default="saved_models/model3_skin_tone")
    args = parser.parse_args()

    model, config = load_model(args.model_dir)
    image = np.array(Image.open(args.image).convert("RGB"))
    result = predict_skin_tone(image, model, config)
    print(json.dumps(result, indent=2))
