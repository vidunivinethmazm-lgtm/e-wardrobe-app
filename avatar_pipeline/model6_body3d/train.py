"""
Model 6 — 3D Body Reconstruction: training script.

Usage:
    # 1. Generate the synthetic dataset (run once)
    python -m avatar_pipeline.model6_body3d.synthetic_data

    # 2. Train
    python -m avatar_pipeline.model6_body3d.train

Trains `architecture.build_body3d_model` as a plain regression model:
inputs are a photo + the aux vector (measurements + body shape + pose),
target is the `params.PARAM_DIM` sigmoid vector
(`params.params_to_sigmoid_vector`) produced by `synthetic_data.py`. Loss is
MSE over the whole vector; the per-parameter MAE breakdown in
`test_report.json` shows which body-mesh measurements (see
`params.PARAM_NAMES`) are hardest to predict.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from .architecture import build_body3d_model
from .params import PARAM_NAMES


def _load_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    return img


def make_dataset(data_dir, paths, aux, targets, image_size, batch_size, training):
    full_paths = np.array([os.path.join(data_dir, p) for p in paths])
    ds = tf.data.Dataset.from_tensor_slices((full_paths, aux, targets))

    def _load(path, a, t):
        return {"photo": _load_image(path, image_size), "aux": a}, t

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/model6_body3d")
    parser.add_argument("--output_dir", default="saved_models/model6_body3d")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npz = np.load(os.path.join(args.data_dir, "targets.npz"))
    aux, targets, paths = npz["aux"], npz["targets"], npz["paths"]

    n = len(paths)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]

    train_ds = make_dataset(
        args.data_dir, paths[train_idx], aux[train_idx], targets[train_idx],
        args.image_size, args.batch_size, training=True,
    )
    val_ds = make_dataset(
        args.data_dir, paths[val_idx], aux[val_idx], targets[val_idx],
        args.image_size, args.batch_size, training=False,
    )
    test_ds = make_dataset(
        args.data_dir, paths[test_idx], aux[test_idx], targets[test_idx],
        args.image_size, args.batch_size, training=False,
    )

    model = build_body3d_model(image_shape=(args.image_size, args.image_size, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Reload the best checkpoint before final evaluation/export.
    model = tf.keras.models.load_model(os.path.join(args.output_dir, "best_model.keras"))

    test_loss, test_mae = model.evaluate(test_ds)
    y_pred = model.predict(test_ds)
    mae_per_param = np.mean(np.abs(y_pred - targets[test_idx]), axis=0)

    report = {
        "test_mse": float(test_loss),
        "test_mae": float(test_mae),
        "mae_per_param": {name: float(m) for name, m in zip(PARAM_NAMES, mae_per_param)},
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.output_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Inference-only SavedModel export (for serving).
    model.export(os.path.join(args.output_dir, "saved_model"))

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"image_size": args.image_size, "param_names": PARAM_NAMES}, f, indent=2)

    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
