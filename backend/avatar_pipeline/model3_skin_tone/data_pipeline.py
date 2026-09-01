"""
Model 3 — Skin Tone Detection: data loading, splitting, and tf.data pipeline.

Handles class imbalance via `compute_sample_weights` (inverse-frequency
weighting per palette bucket), passed as the 3rd element of each dataset
element — `model.fit` applies these automatically as `sample_weight`.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .color_utils import MONK_SKIN_TONE_PALETTE


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    palette_index = {name: i for i, (name, _) in enumerate(MONK_SKIN_TONE_PALETTE)}
    df["palette_idx"] = df["palette_label"].map(palette_index)
    return df


def split_dataframe(df, test_size=0.15, val_size=0.15, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["palette_idx"], random_state=seed)
    val_relative = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_relative, stratify=train_df["palette_idx"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_sample_weights(df):
    """Inverse-frequency weight per palette bucket, to counter skew toward
    over-represented skin tones in the source face dataset."""
    counts = df["palette_idx"].value_counts()
    weights = df["palette_idx"].map(lambda i: len(df) / (len(counts) * counts[i]))
    return weights.values.astype("float32")


def _load_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    return img


def _augment(img, lab, weight):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, lab, weight


def make_dataset(df, faces_dir, image_size=(128, 128), batch_size=32, training=False, use_sample_weights=True):
    paths = [os.path.join(faces_dir, f) for f in df["file_name"].values]
    lab = (df[["L", "a", "b"]].values / 255.0).astype("float32")

    if training and use_sample_weights:
        weights = compute_sample_weights(df)
    else:
        weights = np.ones(len(df), dtype="float32")

    img_ds = tf.data.Dataset.from_tensor_slices(paths).map(
        lambda p: _load_image(p, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    lab_ds = tf.data.Dataset.from_tensor_slices(lab)
    weight_ds = tf.data.Dataset.from_tensor_slices(weights)

    ds = tf.data.Dataset.zip((img_ds, lab_ds, weight_ds))

    if training:
        ds = ds.shuffle(len(df))
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
