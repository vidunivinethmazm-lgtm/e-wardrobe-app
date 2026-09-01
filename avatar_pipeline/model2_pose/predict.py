"""
Model 2 — Pose Estimation: inference.

Default path uses MoveNet (no training required). Pass --method finetuned
to use a custom model produced by train.py instead.

`extract_keypoints` + `keypoints_to_avatar_params` are what the integration
controller (Step 4 of the pipeline) calls directly.
"""

import argparse
import json

import numpy as np
from PIL import Image

from .keypoint_utils import NUM_KEYPOINTS, keypoints_to_avatar_params
from .movenet_inference import MoveNetPoseEstimator


def extract_keypoints(image, method="movenet", model_dir=None, movenet_variant="thunder"):
    """image: HxWx3 uint8 RGB numpy array. Returns (17, 3) array of (x, y, score)."""
    if method == "movenet":
        estimator = MoveNetPoseEstimator(variant=movenet_variant)
        return estimator.predict(image)

    if method == "finetuned":
        import tensorflow as tf

        model = tf.keras.models.load_model(model_dir, compile=False)
        input_size = model.input_shape[1]
        resized = tf.image.resize(image, (input_size, input_size))
        resized = tf.cast(resized, tf.float32)
        preds = model.predict(tf.expand_dims(resized, 0), verbose=0)[0]
        coords = preds.reshape(NUM_KEYPOINTS, 2)
        scores = np.ones((NUM_KEYPOINTS, 1))  # regression model has no built-in confidence
        return np.concatenate([coords, scores], axis=1)

    raise ValueError(method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--method", choices=["movenet", "finetuned"], default="movenet")
    parser.add_argument("--model_dir", default="saved_models/model2_pose/saved_model")
    parser.add_argument("--movenet_variant", choices=["lightning", "thunder"], default="thunder")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    keypoints = extract_keypoints(image, args.method, args.model_dir, args.movenet_variant)
    params = keypoints_to_avatar_params(keypoints)
    print(json.dumps(params, indent=2))
