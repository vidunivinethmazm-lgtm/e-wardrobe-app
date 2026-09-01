"""
Model 2 — Pose Estimation: MoveNet inference (RECOMMENDED path).

MoveNet (TF Hub) is pretrained on COCO and is fast and accurate enough for
full-body keypoint extraction with NO training required. Use this module
directly. Only fall back to the fine-tuning path in `architecture.py` /
`train.py` if MoveNet measurably underperforms on your app's photo style
(e.g. unusual cropping, low-light selfies).
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .keypoint_utils import keypoints_to_avatar_params, movenet_to_xy

# (TF Hub URL, model input resolution)
MOVENET_MODELS = {
    "lightning": ("https://tfhub.dev/google/movenet/singlepose/lightning/4", 192),
    "thunder": ("https://tfhub.dev/google/movenet/singlepose/thunder/4", 256),
}


class MoveNetPoseEstimator:
    """Thin wrapper around TF Hub MoveNet SinglePose.

    `thunder` is more accurate (256x256 input); `lightning` is faster
    (192x192) and better suited to mobile/edge inference.
    """

    def __init__(self, variant="thunder"):
        if variant not in MOVENET_MODELS:
            raise ValueError(f"variant must be one of {list(MOVENET_MODELS)}")
        url, self.input_size = MOVENET_MODELS[variant]
        self.variant = variant
        self.model = hub.load(url)
        self.movenet = self.model.signatures["serving_default"]

    def predict(self, image):
        """image: HxWx3 uint8 RGB numpy array (any size).

        Returns (17, 3) array of (x, y, score), normalized to [0, 1] relative
        to the (letterboxed) input MoveNet sees.
        """
        img = tf.expand_dims(image, axis=0)
        img = tf.image.resize_with_pad(img, self.input_size, self.input_size)
        img = tf.cast(img, dtype=tf.int32)

        outputs = self.movenet(img)
        keypoints = outputs["output_0"].numpy()[0, 0, :, :]  # (17, 3) = (y, x, score)
        return movenet_to_xy(keypoints)

    def predict_avatar_params(self, image, score_threshold=0.3):
        keypoints = self.predict(image)
        return keypoints_to_avatar_params(keypoints, score_threshold=score_threshold)


if __name__ == "__main__":
    import argparse
    import json
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--variant", choices=["lightning", "thunder"], default="thunder")
    args = parser.parse_args()

    estimator = MoveNetPoseEstimator(variant=args.variant)
    image = np.array(Image.open(args.image).convert("RGB"))
    params = estimator.predict_avatar_params(image)
    print(json.dumps(params, indent=2))
