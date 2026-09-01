"""
Model 2 — Pose Estimation: shared keypoint constants and conversions.

Uses the standard COCO 17-keypoint layout (also what MoveNet outputs), so
this module is the single source of truth consumed by both the MoveNet
inference path and the fine-tuning path, and by Models 4/5 downstream.
"""

import numpy as np

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Pairs of connected keypoints — for skeleton drawing and for Model 5's warp anchors.
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def movenet_to_xy(movenet_keypoints):
    """MoveNet returns (17, 3) as (y, x, score). Convert to (17, 3) as (x, y, score)."""
    kp = np.asarray(movenet_keypoints)
    return np.stack([kp[:, 1], kp[:, 0], kp[:, 2]], axis=1)


def keypoints_to_avatar_params(keypoints, score_threshold=0.3):
    """Convert normalized (17, 3) [x, y, score] keypoints into the proportions
    and anchor points that Models 4 & 5 use to build and dress the avatar.

    All coordinates are normalized to [0, 1] relative to the input image.
    """
    kp = {name: np.asarray(keypoints[i]) for i, name in enumerate(KEYPOINT_NAMES)}

    def mid(a, b):
        return (kp[a][:2] + kp[b][:2]) / 2

    def dist(a, b):
        return float(np.hypot(*(kp[a][:2] - kp[b][:2])))

    def visible(*names):
        return all(kp[n][2] >= score_threshold for n in names)

    shoulder_mid = mid("left_shoulder", "right_shoulder")
    hip_mid = mid("left_hip", "right_hip")

    return {
        "keypoints": {name: kp[name][:2].tolist() for name in KEYPOINT_NAMES},
        "scores": {name: float(kp[name][2]) for name in KEYPOINT_NAMES},
        "shoulder_center": shoulder_mid.tolist(),
        "hip_center": hip_mid.tolist(),
        "shoulder_width": dist("left_shoulder", "right_shoulder") if visible("left_shoulder", "right_shoulder") else None,
        "hip_width": dist("left_hip", "right_hip") if visible("left_hip", "right_hip") else None,
        "torso_length": float(np.hypot(*(shoulder_mid - hip_mid))),
        "left_arm_length": (
            dist("left_shoulder", "left_elbow") + dist("left_elbow", "left_wrist")
            if visible("left_shoulder", "left_elbow", "left_wrist")
            else None
        ),
        "right_arm_length": (
            dist("right_shoulder", "right_elbow") + dist("right_elbow", "right_wrist")
            if visible("right_shoulder", "right_elbow", "right_wrist")
            else None
        ),
        "left_leg_length": (
            dist("left_hip", "left_knee") + dist("left_knee", "left_ankle")
            if visible("left_hip", "left_knee", "left_ankle")
            else None
        ),
        "right_leg_length": (
            dist("right_hip", "right_knee") + dist("right_knee", "right_ankle")
            if visible("right_hip", "right_knee", "right_ankle")
            else None
        ),
        "torso_tilt_deg": float(
            np.degrees(
                np.arctan2(
                    kp["right_shoulder"][1] - kp["left_shoulder"][1],
                    kp["right_shoulder"][0] - kp["left_shoulder"][0],
                )
            )
        ),
    }
