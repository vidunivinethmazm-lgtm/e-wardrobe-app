"""
Model 1 — Body Shape Estimation: Keras model architectures.

Two models are provided:

- `build_measurement_only_model`: an MLP over (bust, waist, hips, height +
  engineered ratios). This is the recommended PRIMARY model — measurements
  are the most reliable signal for body shape and the labels themselves are
  derived from measurement ratios, so this model is fast, small, and easy to
  fine-tune later on real labeled data.

- `build_fusion_model`: the MLP branch above, fused with a small CNN over a
  body silhouette image. Use this once you have (or can derive) a silhouette
  image per user, to let the model also pick up on shape cues that aren't
  fully captured by 4 raw measurements.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

CLASS_NAMES = ["Hourglass", "Pear", "Apple", "Rectangle", "InvertedTriangle"]
NUM_CLASSES = len(CLASS_NAMES)


def build_measurement_branch(input_dim):
    inp = layers.Input(shape=(input_dim,), name="measurements")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    return inp, x


def build_silhouette_branch(image_shape=(96, 96, 1)):
    inp = layers.Input(shape=image_shape, name="silhouette")
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    return inp, x


def build_measurement_only_model(input_dim, num_classes=NUM_CLASSES):
    inp, x = build_measurement_branch(input_dim)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="body_shape")(x)
    return Model(inputs=inp, outputs=out, name="body_shape_measurement_mlp")


def build_fusion_model(measurement_dim, image_shape=(96, 96, 1), num_classes=NUM_CLASSES):
    meas_inp, meas_feat = build_measurement_branch(measurement_dim)
    img_inp, img_feat = build_silhouette_branch(image_shape)

    merged = layers.Concatenate()([meas_feat, img_feat])
    x = layers.Dense(64, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="body_shape")(x)

    return Model(inputs=[meas_inp, img_inp], outputs=out, name="body_shape_fusion_model")


if __name__ == "__main__":
    m = build_fusion_model(measurement_dim=7)
    m.summary()
