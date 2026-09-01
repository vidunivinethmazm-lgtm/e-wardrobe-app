"""
Model 4 — Avatar Generation: tf.data pipeline over the synthetic avatar
dataset produced by synthetic_avatars.generate_dataset.
"""

import os

import numpy as np
import tensorflow as tf


def _load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=4)
    img = tf.image.resize(img, (img_size, img_size), method="nearest")
    img = tf.cast(img, tf.float32)
    img.set_shape((img_size, img_size, 4))
    return img


def make_dataset(data_dir, img_size=128, batch_size=32, val_fraction=0.1, seed=42):
    """Returns (train_ds, val_ds), each yielding (image, condition) pairs.

    image: (img_size, img_size, 4) float32 in [0, 255]
    condition: (CONDITION_DIM,) float32
    """
    npz = np.load(os.path.join(data_dir, "conditions.npz"), allow_pickle=True)
    rel_paths = npz["paths"]
    conditions = npz["conditions"].astype("float32")
    abs_paths = np.array([os.path.join(data_dir, p) for p in rel_paths])

    n = len(abs_paths)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    def build(idx, training):
        ds_paths = tf.data.Dataset.from_tensor_slices(abs_paths[idx])
        img_ds = ds_paths.map(
            lambda p: _load_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE
        )
        cond_ds = tf.data.Dataset.from_tensor_slices(conditions[idx])
        ds = tf.data.Dataset.zip((img_ds, cond_ds))
        if training:
            ds = ds.shuffle(min(2000, len(idx)))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build(train_idx, training=True), build(val_idx, training=False)
