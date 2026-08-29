"""Pixel-based dominant garment colour.

The 17-class colour classifier systematically confuses neighbours
(pink -> red, coral -> orange, grey -> beige). Colour is a low-level
property, so we read it straight off the background-removed garment:
classify every interior pixel in HSV, vote, and take the plurality.
Falls back to the model only when the garment has no clear single
colour (patterned / multi-colour).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def _classify_hsv(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-pixel colour label. h in [0,360), s and v in [0,1]. Vectorised."""
    out = np.empty(h.shape, dtype=object)
    out[:] = "Grey"

    warm_neutral = (s < 0.33) & (v > 0.58) & (h >= 18) & (h <= 60)

    # achromatic first ("Silver" from the model, if kept, means light Grey)
    achrom = s < 0.18
    out[achrom & (v > 0.86)] = "White"
    out[achrom & (v <= 0.86) & (v > 0.30)] = "Grey"
    out[achrom & (v <= 0.30)] = "Black"
    out[(~achrom) & (v < 0.16)] = "Black"

    chrom = (~achrom) & (v >= 0.16)

    def paint(cond, label):
        out[chrom & cond] = label

    paint(warm_neutral, "Beige")

    def red_family(cond):
        # dark -> Maroon; vivid + not too light -> Red; otherwise Pink
        paint(cond & (v < 0.42), "Maroon")
        paint(cond & (v >= 0.42) & (s >= 0.60) & (v < 0.80), "Red")
        paint(cond & (v >= 0.42) & ~((s >= 0.60) & (v < 0.80)), "Pink")

    red_family(((h < 15) | (h >= 345)) & ~warm_neutral)

    orange = (h >= 15) & (h < 40) & ~warm_neutral
    paint(orange & (v < 0.50) & (s > 0.35), "Brown")
    paint(orange & ~((v < 0.50) & (s > 0.35)), "Orange")

    yellow = (h >= 40) & (h < 70) & ~warm_neutral
    paint(yellow & (v < 0.5), "Olive")
    paint(yellow & (v >= 0.5) & (h < 52) & (v < 0.85), "Gold")
    paint(yellow & (v >= 0.5) & ~((h < 52) & (v < 0.85)), "Yellow")

    paint((h >= 70) & (h < 90) & (v < 0.55), "Olive")
    paint((h >= 70) & (h < 90) & (v >= 0.55), "Yellow")

    paint((h >= 90) & (h < 200), "Green")

    blue = (h >= 200) & (h < 255)
    paint(blue & (v < 0.40), "Navy Blue")
    paint(blue & (v >= 0.40), "Blue")

    paint((h >= 255) & (h < 330), "Purple")

    red_family((h >= 330) & (h < 345))

    return out


def dominant_color(rgba: Image.Image) -> dict:
    """{'label', 'confidence'} - confidence is the winning colour's pixel share."""
    rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    mask = alpha >= 200
    if mask.sum() < 200:
        return {"label": None, "confidence": 0.0}

    # Sample the garment interior, away from soft matting edges.
    radius = max(1, int(min(mask.shape) * 0.03))
    inner = binary_erosion(mask, structure=np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool))
    if inner.sum() >= 200:
        mask = inner

    hsv = np.asarray(rgba.convert("RGB").convert("HSV"))[mask].astype(np.float32)
    if len(hsv) > 30000:
        idx = np.random.default_rng(0).choice(len(hsv), 30000, replace=False)
        hsv = hsv[idx]

    h = hsv[:, 0] * 360.0 / 255.0
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0

    labels = _classify_hsv(h, s, v)
    names, counts = np.unique(labels, return_counts=True)
    winner = int(counts.argmax())
    return {
        "label": str(names[winner]),
        "confidence": round(float(counts[winner] / counts.sum()), 4),
    }
