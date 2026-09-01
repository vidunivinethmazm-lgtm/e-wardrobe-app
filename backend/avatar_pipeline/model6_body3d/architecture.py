"""
Model 6 — 3D Body Reconstruction: Keras model architecture.

A small CNN over the user's photo, fused with an MLP over the "aux" vector
(measurements + Model 1's body-shape one-hot + Model 2's pose), regresses
`params.PARAM_DIM` body-mesh parameters (see params.py).

The output layer is a sigmoid (each value in [0, 1]); `predict.py` rescales
these into physical units via `params.vector_to_params`. This mirrors Model
1's measurement+silhouette fusion model and Model 4's CNN-decoder condition
vector — the same "photo + structured features in, fixed-size vector out"
shape, just with a different (continuous, regression) target.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from backend.avatar_pipeline.model4_avatar.condition_utils import NUM_BODY_SHAPES, POSE_DIM

from .params import PARAM_DIM

NUM_MEASUREMENTS = 4  # bust, waist, hips, height
AUX_DIM = NUM_MEASUREMENTS + NUM_BODY_SHAPES + POSE_DIM


def build_photo_branch(image_shape=(128, 128, 3)):
    inp = layers.Input(shape=image_shape, name="photo")
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    return inp, x


def build_aux_branch(input_dim=AUX_DIM):
    inp = layers.Input(shape=(input_dim,), name="aux")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    return inp, x


def build_body3d_model(image_shape=(128, 128, 3), aux_dim=AUX_DIM, param_dim=PARAM_DIM):
    photo_inp, photo_feat = build_photo_branch(image_shape)
    aux_inp, aux_feat = build_aux_branch(aux_dim)

    merged = layers.Concatenate()([photo_feat, aux_feat])
    x = layers.Dense(64, activation="relu")(merged)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(param_dim, activation="sigmoid", name="body3d_params")(x)

    return Model(inputs=[photo_inp, aux_inp], outputs=out, name="body3d_regressor")


if __name__ == "__main__":
    m = build_body3d_model()
    m.summary()
