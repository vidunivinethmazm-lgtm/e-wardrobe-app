"""
Model 5 — Virtual Try-On: clothing-agnostic "person representation".

Both the GMM and TOM (architecture.py) need a representation of the wearer
that does NOT depend on the clothing being tried on (otherwise the network
could just copy the original garment instead of warping the new one). We
build a (H, W, PERSON_REPR_CHANNELS) tensor from:

- channel 0:        a coarse, blurred body silhouette (so exact garment
                     edges from the original photo/avatar are not leaked)
- channels 1..12:   one Gaussian heatmap per joint in
                     model4_avatar.condition_utils.JOINT_NAMES, built from
                     Model 2's pose keypoints
- channels 13..15:  the wearer's skin RGB (Model 3), broadcast as a
                     constant-color image

This is a simplified analogue of CP-VTON's 22-channel (pose + body-shape +
face/hair) agnostic representation, sized down to match what Models 1-4
already produce. Default size is (256, 192) — the standard VITON/CP-VTON
aspect ratio used throughout Model 5.
"""

import cv2
import numpy as np

from backend.avatar_pipeline.model4_avatar.condition_utils import JOINT_NAMES

NUM_POSE_CHANNELS = len(JOINT_NAMES)  # 12
PERSON_REPR_CHANNELS = 1 + NUM_POSE_CHANNELS + 3  # 16


def _gaussian_heatmap(size, center, sigma):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = center
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def build_pose_heatmaps(keypoints_dict, img_height, img_width, sigma_frac=0.03):
    """keypoints_dict: {joint_name: [x, y]} normalized to [0, 1] (Model 2's
    `keypoints_to_avatar_params(...)["keypoints"]`).
    Returns (img_height, img_width, NUM_POSE_CHANNELS) float32 in [0, 1].
    """
    sigma = sigma_frac * min(img_height, img_width)
    heatmaps = np.zeros((img_height, img_width, NUM_POSE_CHANNELS), dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        x, y = keypoints_dict[name][:2]
        heatmaps[..., i] = _gaussian_heatmap(
            (img_height, img_width), (x * img_width, y * img_height), sigma=sigma
        )
    return heatmaps


def build_silhouette_channel(mask, img_height, img_width, blur_frac=0.04):
    """mask: HxW array (any dtype), nonzero where the body/avatar is.
    Returns (img_height, img_width, 1) float32 in [0, 1], resized and
    blurred to give a coarse body-shape signal without sharp garment edges.
    """
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    ksize = max(1, int(blur_frac * min(img_height, img_width))) | 1  # force odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask[..., None]


def build_person_representation(silhouette_mask, keypoints_dict, skin_rgb_normalized, img_height=256, img_width=192):
    """silhouette_mask: HxW array, nonzero where the body/avatar is (e.g.
        Model 4's avatar alpha channel, or a parsing-derived body mask)
    keypoints_dict: {joint_name: [x, y]} normalized [0, 1], from Model 2
    skin_rgb_normalized: (3,) in [0, 1], from Model 3

    Returns (img_height, img_width, PERSON_REPR_CHANNELS) float32 in [0, 1].
    """
    silhouette = build_silhouette_channel(silhouette_mask, img_height, img_width)
    heatmaps = build_pose_heatmaps(keypoints_dict, img_height, img_width)
    skin = np.ones((img_height, img_width, 3), dtype=np.float32) * np.asarray(
        skin_rgb_normalized, dtype=np.float32
    )
    return np.concatenate([silhouette, heatmaps, skin], axis=-1)
