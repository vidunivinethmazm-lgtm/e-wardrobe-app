"""
Synthetic dataset generator for Model 1 (Body Shape Estimation).

There is no public dataset that pairs photos/measurements with body-shape
labels (hourglass, pear, apple, rectangle, inverted triangle). The standard
bootstrapping approach is to derive labels from measurements using the
ratio/difference rules below (the same rules used across fashion sizing
guides), then generate matching synthetic measurements + a stylized
silhouette image for each sample.

`classify_body_shape` is the source of truth for labels. If you later get
access to a real anthropometric dataset (e.g. ANSUR II) or real user photos
with measurements, apply this same function to derive labels and fine-tune
the model trained on this synthetic data.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CLASS_NAMES = ["Hourglass", "Pear", "Apple", "Rectangle", "InvertedTriangle"]


def classify_body_shape(bust, waist, hips):
    """Rule-based body shape label from bust/waist/hip measurements (cm)."""
    bust_hip_diff = bust - hips
    waist_def_bust = bust - waist
    waist_def_hip = hips - waist

    if bust_hip_diff >= 9:
        return "InvertedTriangle"
    if -bust_hip_diff >= 9:
        return "Pear"
    if waist_def_bust >= 18 and waist_def_hip >= 18:
        return "Hourglass"
    if waist_def_bust <= 8 and waist_def_hip <= 8:
        return "Apple"
    return "Rectangle"


def _sample_measurements(rng, target_class):
    """Sample (bust, waist, hips) in cm that classify as `target_class`."""
    while True:
        if target_class == "InvertedTriangle":
            bust = rng.uniform(88, 112)
            hips = bust - rng.uniform(9, 18)
            waist = rng.uniform(65, bust - 4)
        elif target_class == "Pear":
            hips = rng.uniform(92, 120)
            bust = hips - rng.uniform(9, 18)
            waist = rng.uniform(62, bust - 2)
        elif target_class == "Hourglass":
            bust = rng.uniform(82, 106)
            hips = bust + rng.uniform(-6, 6)
            waist = rng.uniform(58, min(bust, hips) - 18)
        elif target_class == "Apple":
            bust = rng.uniform(86, 112)
            hips = bust + rng.uniform(-6, 6)
            waist = rng.uniform(max(bust, hips) - 8, max(bust, hips) + 4)
        elif target_class == "Rectangle":
            bust = rng.uniform(80, 106)
            hips = bust + rng.uniform(-6, 6)
            waist = rng.uniform(max(bust, hips) - 17, max(bust, hips) - 9)
        else:
            raise ValueError(target_class)

        if waist <= 0 or bust <= 0 or hips <= 0:
            continue
        if classify_body_shape(bust, waist, hips) == target_class:
            return bust, waist, hips


def draw_silhouette(bust, waist, hips, img_size=96):
    """Render a stylized full-body silhouette encoding the given proportions.

    This is a deliberately simple parametric silhouette (not a photo). At
    inference time, a real user photo should be converted to a comparable
    silhouette mask (e.g. via a person-segmentation model) before being fed
    to the image branch, so the domain matches what the model was trained on.
    """
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    max_measurement = max(bust, waist, hips)
    scale = (img_size * 0.7) / max_measurement

    bust_w = bust * scale
    waist_w = waist * scale
    hip_w = hips * scale

    cx = img_size / 2
    head_r = img_size * 0.06

    head_top = img_size * 0.06
    bust_y = img_size * 0.28
    waist_y = img_size * 0.50
    hip_y = img_size * 0.62
    bottom_y = img_size * 0.95

    draw.ellipse(
        [cx - head_r, head_top, cx + head_r, head_top + 2 * head_r],
        fill=255,
    )

    points = [
        (cx - bust_w / 2, bust_y),
        (cx - waist_w / 2, waist_y),
        (cx - hip_w / 2, hip_y),
        (cx - hip_w / 3, bottom_y),
        (cx + hip_w / 3, bottom_y),
        (cx + hip_w / 2, hip_y),
        (cx + waist_w / 2, waist_y),
        (cx + bust_w / 2, bust_y),
    ]
    draw.polygon(points, fill=255)

    return img


def generate_dataset(output_dir, n_per_class=400, img_size=96, seed=42):
    rng = np.random.default_rng(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    idx = 0
    for body_shape in CLASS_NAMES:
        for _ in range(n_per_class):
            bust, waist, hips = _sample_measurements(rng, body_shape)
            height = float(np.clip(rng.normal(162, 7), 145, 195))

            img = draw_silhouette(bust, waist, hips, img_size)
            image_path = os.path.join(images_dir, f"{idx:05d}.png")
            img.save(image_path)

            rows.append(
                {
                    "bust": round(bust, 1),
                    "waist": round(waist, 1),
                    "hips": round(hips, 1),
                    "height": round(height, 1),
                    "body_shape": body_shape,
                    "image_path": image_path,
                }
            )
            idx += 1

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    csv_path = os.path.join(output_dir, "measurements.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} samples ({n_per_class} per class) to {csv_path}")
    return df


if __name__ == "__main__":
    generate_dataset(output_dir="data/model1_body_shape", n_per_class=400, img_size=96)
