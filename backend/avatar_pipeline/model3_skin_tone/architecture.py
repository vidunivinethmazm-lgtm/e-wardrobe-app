"""
Model 3 — Skin Tone Detection: Keras model architecture.

A MobileNetV2-backed regressor that predicts the dominant skin color of a
face crop as normalized L*a*b* values in [0, 1] (multiply by 255 to recover
OpenCV-convention L*a*b*). Regression onto a continuous color space, rather
than direct classification into palette buckets, means:

- training labels come for free from `pseudo_labels.py` (no manual
  annotation), and
- the palette (Monk, Fitzpatrick, or custom) can be changed at INFERENCE
  time without retraining — `nearest_palette_match` just re-buckets the
  predicted LAB color.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_skin_tone_model(input_shape=(128, 128, 3), freeze_backbone=True):
    """Returns (model, base_model). Flip `base_model.trainable` for a second
    fine-tuning phase (see train.py)."""
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="sigmoid", name="lab_normalized")(x)

    model = Model(inputs, outputs, name="skin_tone_lab_regressor")
    return model, base
