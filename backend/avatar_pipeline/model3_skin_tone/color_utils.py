"""
Model 3 — Skin Tone Detection: color-space utilities, skin segmentation,
dominant-color extraction, and palette mapping.

Two reference palettes are provided:

- MONK_SKIN_TONE_PALETTE: 10-point Monk Skin Tone scale (Google), designed
  specifically to be more inclusive across skin tones than Fitzpatrick.
  The hex values below are close approximations — verify against the
  official swatches at https://skintone.google before shipping.
- FITZPATRICK_PALETTE: 6-category alternative if your team prefers it.

Either palette is just a list of (label, hex) pairs — swap in your own to
use a custom scale.
"""

import cv2
import numpy as np

MONK_SKIN_TONE_PALETTE = [
    ("MST-1", "#f6ede4"),
    ("MST-2", "#f3e7db"),
    ("MST-3", "#f7ead0"),
    ("MST-4", "#eadaba"),
    ("MST-5", "#d7bd96"),
    ("MST-6", "#a07e56"),
    ("MST-7", "#825c43"),
    ("MST-8", "#604134"),
    ("MST-9", "#3a312a"),
    ("MST-10", "#292420"),
]

FITZPATRICK_PALETTE = [
    ("Type-I", "#f6ede4"),
    ("Type-II", "#f1d6c0"),
    ("Type-III", "#e0ac8e"),
    ("Type-IV", "#c68863"),
    ("Type-V", "#8d5a3c"),
    ("Type-VI", "#4a2f23"),
]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb):
    pixel = np.uint8([[rgb]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0]
    return lab.astype(np.float32)


def palette_to_lab(palette):
    return np.array([rgb_to_lab(hex_to_rgb(hex_color)) for _, hex_color in palette])


def white_balance_scene(image_rgb):
    """Gray-world white balance for color-cast correction.

    Apply this to the FULL photo (diverse background/scene content), NOT a
    tight face/skin crop. On a crop dominated by a single skin tone, the
    gray-world assumption ("the average color should be gray") is violated
    and this will desaturate the skin color toward neutral gray, destroying
    the exact signal we're trying to measure.
    """
    img = image_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / (means + 1e-6)
    return np.clip(img * scale, 0, 255).astype(np.uint8)


def segment_skin_pixels(image_rgb):
    """Boolean mask of likely-skin pixels using YCbCr thresholds."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y > 40)


def dominant_skin_color(face_rgb, k=3):
    """Returns the dominant skin color of `face_rgb` as a (L, a, b) array.

    `face_rgb` should be a face crop from a `white_balance_scene`-corrected
    photo. Note: L* (lightness) is intentionally left as-is — it's the
    primary signal that distinguishes skin tones, so it must NOT be
    normalized to a fixed target (that would erase the very thing we're
    measuring). This means photo exposure still affects the result; the
    pseudo-labels and the CNN's training inputs go through this exact same
    pipeline, so the model learns a self-consistent mapping. For best
    results, ask users for a well-lit, natural-light photo.
    """
    mask = segment_skin_pixels(face_rgb)

    pixels = face_rgb[mask]
    if len(pixels) < 50:
        pixels = face_rgb.reshape(-1, 3)

    lab_pixels = (
        cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(lab_pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)]


def nearest_palette_match(lab_color, palette=MONK_SKIN_TONE_PALETTE):
    palette_lab = palette_to_lab(palette)
    dists = np.linalg.norm(palette_lab - np.asarray(lab_color), axis=1)
    idx = int(np.argmin(dists))
    name, hex_color = palette[idx]
    return {
        "label": name,
        "hex": hex_color,
        "lab": np.asarray(lab_color).tolist(),
        "distance": float(dists[idx]),
        "confidence": float(1.0 / (1.0 + dists[idx])),
    }
