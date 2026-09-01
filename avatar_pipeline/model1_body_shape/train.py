"""
Model 1 — Body Shape Estimation: training script.

Usage:
    # 1. Generate the synthetic dataset (run once)
    python -m avatar_pipeline.model1_body_shape.synthetic_data

    # 2. Train (fusion = measurements + silhouette image)
    python -m avatar_pipeline.model1_body_shape.train --model_type fusion

    # ...or measurements-only
    python -m avatar_pipeline.model1_body_shape.train --model_type measurement_only
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .architecture import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_fusion_model,
    build_measurement_only_model,
)
from .data_pipeline import (
    ENGINEERED_COLUMNS,
    fit_scaler,
    load_dataframe,
    make_dataset,
    split_dataframe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/model1_body_shape/measurements.csv")
    parser.add_argument("--output_dir", default="saved_models/model1_body_shape")
    parser.add_argument("--model_type", choices=["measurement_only", "fusion"], default="fusion")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=96)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataframe(args.csv)
    train_df, val_df, test_df = split_dataframe(df)

    scaler = fit_scaler(train_df)
    joblib.dump(scaler, os.path.join(args.output_dir, "measurement_scaler.joblib"))

    fusion = args.model_type == "fusion"
    image_size = (args.image_size, args.image_size)

    train_ds = make_dataset(train_df, scaler, image_size, args.batch_size, training=True, fusion=fusion)
    val_ds = make_dataset(val_df, scaler, image_size, args.batch_size, training=False, fusion=fusion)
    test_ds = make_dataset(test_df, scaler, image_size, args.batch_size, training=False, fusion=fusion)

    if fusion:
        model = build_fusion_model(len(ENGINEERED_COLUMNS), image_shape=(args.image_size, args.image_size, 1))
    else:
        model = build_measurement_only_model(len(ENGINEERED_COLUMNS))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=train_df["label_idx"].values
    )
    class_weight_dict = dict(enumerate(class_weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # Reload the best checkpoint before final evaluation/export.
    model = tf.keras.models.load_model(os.path.join(args.output_dir, "best_model.keras"))

    y_true = test_df["label_idx"].values
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    with open(os.path.join(args.output_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))

    # Inference-only SavedModel export (for serving).
    model.export(os.path.join(args.output_dir, "saved_model"))

    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "model_type": args.model_type,
                "image_size": args.image_size,
                "feature_columns": ENGINEERED_COLUMNS,
            },
            f,
            indent=2,
        )

    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
