"""
Model 4 — Avatar Generation: condition vector assembly.

Combines the outputs of Model 1 (body shape), Model 2 (pose), and Model 3
(skin tone) into the single condition vector that the conditional VAE-GAN in
architecture.py is conditioned on. Keeping this mapping in one place means
the controller (predict.py here, and the end-to-end pipeline) only has to
call `build_condition_vector`.
"""

import numpy as np

BODY_SHAPE_NAMES = ["Hourglass", "Pear", "Apple", "Rectangle", "InvertedTriangle"]
NUM_BODY_SHAPES = len(BODY_SHAPE_NAMES)

# Reduced 12-joint pose (x, y) used for both rendering the synthetic dataset
# and conditioning the generator. All 12 names are a subset of Model 2's
# 17-keypoint COCO output, so `keypoints_to_pose_vector` below can read
# directly from `keypoints_to_avatar_params(...)["keypoints"]`.
JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
NUM_JOINTS = len(JOINT_NAMES)
POSE_DIM = NUM_JOINTS * 2

CONDITION_DIM = NUM_BODY_SHAPES + POSE_DIM + 3  # body shape one-hot + pose + skin RGB


def body_shape_to_onehot(body_shape):
    onehot = np.zeros(NUM_BODY_SHAPES, dtype=np.float32)
    onehot[BODY_SHAPE_NAMES.index(body_shape)] = 1.0
    return onehot


def keypoints_to_pose_vector(keypoints_dict):
    """keypoints_dict: {joint_name: [x, y]}, e.g. from Model 2's
    `keypoints_to_avatar_params(...)["keypoints"]`. Returns a (POSE_DIM,)
    vector in JOINT_NAMES order, normalized to [0, 1]."""
    return np.array(
        [coord for name in JOINT_NAMES for coord in keypoints_dict[name][:2]],
        dtype=np.float32,
    )


def skin_tone_to_rgb(predict_skin_tone_result):
    """Extracts a normalized [0, 1] RGB triple from Model 3's
    `predict_skin_tone(...)` output (uses the matched palette's base color)."""
    from backend.avatar_pipeline.model3_skin_tone.color_utils import hex_to_rgb

    rgb = hex_to_rgb(predict_skin_tone_result["hex"])
    return np.array(rgb, dtype=np.float32) / 255.0


def build_condition_vector(body_shape, pose_vector, skin_rgb_normalized):
    """body_shape: str, one of BODY_SHAPE_NAMES
    pose_vector: (POSE_DIM,) float array, normalized [0, 1] coordinates
    skin_rgb_normalized: (3,) float array in [0, 1]

    Returns a (CONDITION_DIM,) float32 vector.
    """
    return np.concatenate(
        [
            body_shape_to_onehot(body_shape),
            np.asarray(pose_vector, dtype=np.float32),
            np.asarray(skin_rgb_normalized, dtype=np.float32),
        ]
    ).astype(np.float32)
