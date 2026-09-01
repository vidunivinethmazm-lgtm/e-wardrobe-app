"""
Model 3 — Skin Tone Detection: pseudo-label generation.

There is no need for manually-annotated skin-tone labels. Instead, run the
deterministic color-science pipeline (`color_utils.dominant_skin_color` +
`nearest_palette_match`) over a directory of ordinary face photos to produce
(L, a, b) regression targets and a palette-bucket label per image. The CNN in
architecture.py then learns to reproduce this mapping directly from pixels —
which is more robust at inference (handles partial occlusion, motion blur,
unusual lighting) than running the color pipeline on a live photo alone.

Source images: any large face dataset (UTKFace, CelebA, FairFace, or your
own consented user photos). FairFace is recommended if available — it was
explicitly built to balance race/skin-tone representation, which directly
addresses the class-imbalance concern for this task. Whatever source you
use, check the per-bucket counts printed at the end: a bucket with 0-few
samples means your model will not learn that skin tone well, regardless of
training tricks — that's a data collection gap, not a modeling one.

Usage:
    python -m avatar_pipeline.model3_skin_tone.pseudo_labels \
        --images_dir path/to/face_dataset --output_dir data/model3_skin_tone
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image

from .color_utils import (
    MONK_SKIN_TONE_PALETTE,
    dominant_skin_color,
    nearest_palette_match,
    white_balance_scene,
)
from .face_crop import detect_and_crop_face


def generate_pseudo_labels(images_dir, output_dir, face_size=128):
    faces_dir = os.path.join(output_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    rows = []
    for fname in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, fname)
        try:
            image = np.array(Image.open(path).convert("RGB"))
        except (OSError, ValueError):
            continue

        image = white_balance_scene(image)
        face = detect_and_crop_face(image, output_size=face_size)
        if face is None:
            continue

        lab = dominant_skin_color(face)
        match = nearest_palette_match(lab, MONK_SKIN_TONE_PALETTE)

        Image.fromarray(face).save(os.path.join(faces_dir, fname))
        rows.append(
            {
                "file_name": fname,
                "L": float(lab[0]),
                "a": float(lab[1]),
                "b": float(lab[2]),
                "palette_label": match["label"],
            }
        )

    csv_path = os.path.join(output_dir, "pseudo_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "L", "a", "b", "palette_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pseudo-labels to {csv_path} (face crops in {faces_dir})")

    counts = {}
    for row in rows:
        counts[row["palette_label"]] = counts.get(row["palette_label"], 0) + 1
    for name, _ in MONK_SKIN_TONE_PALETTE:
        print(f"  {name}: {counts.get(name, 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", default="data/model3_skin_tone")
    parser.add_argument("--face_size", type=int, default=128)
    args = parser.parse_args()
    generate_pseudo_labels(args.images_dir, args.output_dir, args.face_size)
