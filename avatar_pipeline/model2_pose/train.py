"""
Model 2 — Pose Estimation: OPTIONAL fine-tuning of the MobileNetV2 keypoint
regressor on COCO-format data.

Only run this if MoveNet underperforms on your app's photo style. Two-phase
fine-tuning: train the regression head with a frozen backbone, then unfreeze
the backbone at a low learning rate.

Usage:
    python -m avatar_pipeline.model2_pose.train \
        --annotations path/to/person_keypoints_val2017.json \
        --images_dir path/to/val2017
"""

import argparse
import os

import tensorflow as tf

from .architecture import build_pose_regression_model, masked_keypoint_loss, pck_metric
from .data_pipeline import load_coco_annotations, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to person_keypoints_*.json")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", default="saved_models/model2_pose")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = load_coco_annotations(args.annotations)
    split = int(0.9 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"{len(train_samples)} train / {len(val_samples)} val samples")

    train_ds = make_dataset(train_samples, args.images_dir, args.input_size, args.batch_size, training=True)
    val_ds = make_dataset(val_samples, args.images_dir, args.input_size, args.batch_size, training=False)

    model, base = build_pose_regression_model(input_shape=(args.input_size, args.input_size, 3), freeze_backbone=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_keypoint_loss, metrics=[pck_metric(0.1)])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    # Phase 1: train the regression head with the backbone frozen.
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks)

    # Phase 2: unfreeze the backbone and fine-tune at a lower learning rate.
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=masked_keypoint_loss, metrics=[pck_metric(0.1)])
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=args.head_epochs,
        epochs=args.head_epochs + args.finetune_epochs,
        callbacks=callbacks,
    )

    model = tf.keras.models.load_model(
        os.path.join(args.output_dir, "best_model.keras"),
        custom_objects={"masked_keypoint_loss": masked_keypoint_loss, "pck_at_0.1": pck_metric(0.1)},
    )
    model.export(os.path.join(args.output_dir, "saved_model"))
    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
