"""
Face Skin Tone — simplified Day 3-4 scope: detect a user's skin tone from a
face photo and produce a color value to tint the avatar's skin material.

Deliberately reuses model3_skin_tone's classical color-science pipeline
(face_crop.py + color_utils.py) as-is, skipping the trained CNN in
predict.py entirely — no model loading, no training. Out of scope: face
morph targets, eye/eyebrow/lip libraries, landmark-based reconstruction.
"""

from .model3_skin_tone.color_utils import (
    MONK_SKIN_TONE_PALETTE,
    dominant_skin_color,
    hex_to_rgb,
    nearest_palette_match,
    white_balance_scene,
)
from .model3_skin_tone.face_crop import detect_and_crop_face


def extract_skin_tone(photo_rgb, palette=MONK_SKIN_TONE_PALETTE, face_size=128):
    """photo_rgb: HxWx3 uint8 RGB numpy array (full photo, face visible).

    Returns:
        {
            "label": nearest palette swatch name (e.g. "MST-5"),
            "hex": "#rrggbb",
            "rgb": (r, g, b) ints 0-255,
            "lab": [L, a, b],
            "confidence": float in (0, 1],
        }

    Raises ValueError if no face is detected — callers should catch this and
    prompt the user for a clearer photo (same contract as Model 3's CNN path).
    """
    balanced = white_balance_scene(photo_rgb)
    face = detect_and_crop_face(balanced, output_size=face_size)
    if face is None:
        raise ValueError("No face detected in the input photo.")

    lab = dominant_skin_color(face)
    match = nearest_palette_match(lab, palette)

    return {
        "label": match["label"],
        "hex": match["hex"],
        "rgb": hex_to_rgb(match["hex"]),
        "lab": match["lab"],
        "confidence": match["confidence"],
    }
