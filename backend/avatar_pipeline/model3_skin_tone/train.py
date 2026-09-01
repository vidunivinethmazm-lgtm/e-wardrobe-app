"""
Model 3 — Skin Tone Detection: training script.

1. Generate pseudo-labels from a face image directory:
    python -m avatar_pipeline.model3_skin_tone.pseudo_labels \
        --images_dir path/to/face_dataset --output_dir data/model3_skin_tone

2. Train:
    python -m avatar_pipeline.model3_skin_tone.train
"""

import argparse
import json
import os

import tensorflow as tf
from sklearn.metrics import classification_report, mean_absolute_error

from .architecture import build_skin_tone_model
from .color_utils import MONK_SKIN_TONE_PALETTE, nearest_palette_match
from .data_pipeline import load_dataframe, make_dataset, split_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/model3_skin_tone/pseudo_labels.csv")
    parser.add_argument("--faces_dir", default="data/model3_skin_tone/faces")
    parser.add_argument("--output_dir", default="saved_models/model3_skin_tone")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--head_epochs", type=int, default=15)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataframe(args.csv)
    train_df, val_df, test_df = split_dataframe(df)

    image_size = (args.image_size, args.image_size)
    train_ds = make_dataset(train_df, args.faces_dir, image_size, args.batch_size, training=True)
    val_ds = make_dataset(val_df, args.faces_dir, image_size, args.batch_size, training=False)
    test_ds = make_dataset(test_df, args.faces_dir, image_size, args.batch_size, training=False)

    model, base = build_skin_tone_model(input_shape=(args.image_size, args.image_size, 3), freeze_backbone=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks)

    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mse", metrics=["mae"])
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=args.head_epochs,
        epochs=args.head_epochs + args.finetune_epochs,
        callbacks=callbacks,
    )

    model = tf.keras.models.load_model(os.path.join(args.output_dir, "best_model.keras"))

    y_true_lab = test_df[["L", "a", "b"]].values.astype("float32")
    y_pred_lab = model.predict(test_ds, verbose=0) * 255.0

    mae = mean_absolute_error(y_true_lab, y_pred_lab)
    print(f"Test LAB MAE: {mae:.2f}")

    true_labels = test_df["palette_label"].values
    pred_labels = [nearest_palette_match(lab, MONK_SKIN_TONE_PALETTE)["label"] for lab in y_pred_lab]

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    print(classification_report(true_labels, pred_labels, zero_division=0))

    with open(os.path.join(args.output_dir, "test_report.json"), "w") as f:
        json.dump({"lab_mae": float(mae), "palette_classification_report": report}, f, indent=2)

    model.export(os.path.join(args.output_dir, "saved_model"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"image_size": args.image_size}, f, indent=2)

    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
