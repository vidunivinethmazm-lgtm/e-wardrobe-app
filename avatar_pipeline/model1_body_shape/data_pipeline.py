"""
Model 1 — Body Shape Estimation: data loading, splitting, and tf.data pipelines.

Handles the multi-input (measurements + silhouette image) case by zipping a
measurements tensor dataset with an image dataset and mapping both into a
dict of named inputs that matches the Keras `Input(name=...)` layers in
`architecture.py`.
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .architecture import CLASS_NAMES

ENGINEERED_COLUMNS = [
    "bust",
    "waist",
    "hips",
    "height",
    "bust_waist_ratio",
    "waist_hip_ratio",
    "bust_hip_ratio",
]


def add_engineered_features(df):
    df = df.copy()
    df["bust_waist_ratio"] = df["bust"] / df["waist"]
    df["waist_hip_ratio"] = df["waist"] / df["hips"]
    df["bust_hip_ratio"] = df["bust"] / df["hips"]
    return df


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)
    df["label_idx"] = df["body_shape"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    return df


def split_dataframe(df, test_size=0.15, val_size=0.15, seed=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label_idx"], random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_relative, stratify=train_df["label_idx"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def fit_scaler(train_df):
    scaler = StandardScaler()
    scaler.fit(train_df[ENGINEERED_COLUMNS].values)
    return scaler


def _load_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size)
    return img


def _augment(xy, y):
    meas, img = xy
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return (meas, img), y


def make_dataset(df, scaler, image_size=(96, 96), batch_size=32, training=False, fusion=True):
    measurements = scaler.transform(df[ENGINEERED_COLUMNS].values).astype("float32")
    labels = df["label_idx"].values.astype("int32")

    if not fusion:
        ds = tf.data.Dataset.from_tensor_slices((measurements, labels))
        if training:
            ds = ds.shuffle(len(df))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    paths = df["image_path"].values
    img_ds = tf.data.Dataset.from_tensor_slices(paths).map(
        lambda p: _load_image(p, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    meas_ds = tf.data.Dataset.from_tensor_slices(measurements)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip(((meas_ds, img_ds), label_ds))

    if training:
        ds = ds.shuffle(len(df))
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        lambda xy, y: ({"measurements": xy[0], "silhouette": xy[1]}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
