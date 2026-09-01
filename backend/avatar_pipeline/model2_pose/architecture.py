"""
Model 2 — Pose Estimation: OPTIONAL fine-tunable keypoint regressor.

This is a transfer-learning model (MobileNetV2 backbone + dense regression
head) for the 17 COCO keypoints. Only train this if MoveNet
(`movenet_inference.py`) underperforms on your app's photo style — it is
NOT required for the avatar pipeline to work.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from .keypoint_utils import NUM_KEYPOINTS


def build_pose_regression_model(input_shape=(224, 224, 3), num_keypoints=NUM_KEYPOINTS, freeze_backbone=True):
    """Returns (model, base_model). `base_model.trainable` can be flipped
    later for a second fine-tuning phase (see train.py)."""
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_keypoints * 2, activation="sigmoid", name="keypoints")(x)

    model = Model(inputs, outputs, name="pose_regression_mobilenetv2")
    return model, base


def masked_keypoint_loss(y_true, y_pred):
    """y_true: (batch, K*3) = (x, y, visibility) per keypoint.
    y_pred: (batch, K*2) = (x, y) per keypoint.

    Keypoints with visibility == 0 (not labeled) are excluded from the loss.
    """
    y_true = tf.reshape(y_true, (-1, NUM_KEYPOINTS, 3))
    y_pred = tf.reshape(y_pred, (-1, NUM_KEYPOINTS, 2))

    coords_true = y_true[..., :2]
    visibility = tf.cast(y_true[..., 2] > 0, tf.float32)

    sq_err = tf.reduce_sum(tf.square(coords_true - y_pred), axis=-1)
    masked_err = sq_err * visibility
    return tf.reduce_sum(masked_err) / (tf.reduce_sum(visibility) + 1e-6)


def pck_metric(threshold=0.1):
    """Percentage of Correct Keypoints within `threshold` (normalized distance)."""

    def pck(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, NUM_KEYPOINTS, 3))
        y_pred = tf.reshape(y_pred, (-1, NUM_KEYPOINTS, 2))

        coords_true = y_true[..., :2]
        visibility = tf.cast(y_true[..., 2] > 0, tf.float32)

        dist = tf.norm(coords_true - y_pred, axis=-1)
        correct = tf.cast(dist < threshold, tf.float32) * visibility
        return tf.reduce_sum(correct) / (tf.reduce_sum(visibility) + 1e-6)

    pck.__name__ = f"pck_at_{threshold}"
    return pck
