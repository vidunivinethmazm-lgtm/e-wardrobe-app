"""
Model 5 — Virtual Try-On: tf.data pipeline.

Expects a directory of preprocessed `.npz` files (one per training sample),
produced by preprocess_viton.py from a raw VITON / VITON-HD download. Each
`.npz` contains:

  person               (256,192,3) uint8   ground-truth person photo (TOM target)
  cloth                (256,192,3) uint8   in-shop clothing image (GMM/TOM input)
  cloth_mask           (256,192,1) uint8   binary clothing mask {0,255}
  person_repr          (256,192,16) float32 [0,1]  agnostic person representation
                                            (see pose_repr.build_person_representation)
  cloth_on_person      (256,192,3) uint8   clothing region cropped from `person`
                                            (GMM warp target)
  cloth_on_person_mask (256,192,1) uint8   mask of that region {0,255}
"""

import glob
import os

import numpy as np
import tensorflow as tf

_FIELDS = {
    "person": (256, 192, 3),
    "cloth": (256, 192, 3),
    "cloth_mask": (256, 192, 1),
    "person_repr": (256, 192, 16),
    "cloth_on_person": (256, 192, 3),
    "cloth_on_person_mask": (256, 192, 1),
}
_UINT8_FIELDS = {"person", "cloth", "cloth_mask", "cloth_on_person", "cloth_on_person_mask"}


def _load_npz(path):
    def _load(p):
        data = np.load(p.numpy().decode("utf-8"))
        return tuple(data[name].astype("float32") for name in _FIELDS)

    values = tf.py_function(_load, [path], [tf.float32] * len(_FIELDS))
    sample = {}
    for name, value in zip(_FIELDS, values):
        value.set_shape(_FIELDS[name])
        sample[name] = value
    return sample


def make_dataset(data_dir, batch_size=8, val_fraction=0.1, seed=42):
    """Returns (train_ds, val_ds), each yielding dicts of the fields above
    (all float32; the `_UINT8_FIELDS` are still in [0, 255] / {0, 255})."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = np.random.default_rng(seed)
    paths = np.array(paths)
    indices = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_fraction))
    val_paths, train_paths = paths[indices[:n_val]], paths[indices[n_val:]]

    def build(p, training):
        ds = tf.data.Dataset.from_tensor_slices(p)
        ds = ds.map(_load_npz, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(min(1000, len(p)))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build(train_paths, training=True), build(val_paths, training=False)
