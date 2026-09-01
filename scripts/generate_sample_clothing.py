"""
Generates a small demo wardrobe of RGBA clothing PNGs (transparent
background = alpha mask) for local development and testing of the avatar
pipeline, without depending on the recommendation team's real catalog.

Each PNG follows the mobile app's "RGBA-with-alpha-as-mask" convention: the
alpha channel doubles as `ClothingItem.mask`, so `server.app`'s
`/api/avatars/<id>/tryon` endpoint can take a single image upload and split
it into `clothing_rgb` / `clothing_mask` (see pipeline_types.ClothingItem and
model5_tryon.predict.try_on_avatar).

Shapes are simple flat-lay silhouettes roughly matching
`model5_tryon.predict.GARMENT_LANDMARKS` for each category, so the TPS warp
in `try_on_avatar` produces a reasonable fit on the avatar.

Usage:
    python -m scripts.generate_sample_clothing [--output_dir mobile/assets/clothing]
"""

import argparse
import os

from PIL import Image, ImageDraw

IMG_SIZE = 256


def _scaled(points, size=IMG_SIZE):
    return [(x * size, y * size) for x, y in points]


def make_tshirt(color=(214, 64, 64, 255)):
    """upper_body: body + sleeves, roughly matching GARMENT_LANDMARKS["upper_body"]."""
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    points = _scaled([
        (0.38, 0.05), (0.18, 0.05),   # left collar -> left shoulder
        (0.02, 0.22), (0.16, 0.34),   # left sleeve out -> armpit
        (0.12, 0.95), (0.88, 0.95),   # hem
        (0.84, 0.34), (0.98, 0.22),   # right armpit -> sleeve out
        (0.82, 0.05), (0.62, 0.05),   # right shoulder -> right collar
        (0.50, 0.13),                 # collar dip
    ])
    draw.polygon(points, fill=color)

    # Collar ring (slightly darker), purely decorative.
    darker = tuple(max(0, c - 40) for c in color[:3]) + (255,)
    draw.arc(_scaled([(0.40, 0.00), (0.60, 0.16)]), start=0, end=180, fill=darker, width=4)

    return img


def make_jeans(color=(58, 84, 158, 255)):
    """lower_body: waistband + two legs, roughly matching GARMENT_LANDMARKS["lower_body"]."""
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    waistband = _scaled([(0.20, 0.05), (0.80, 0.05), (0.80, 0.20), (0.20, 0.20)])
    left_leg = _scaled([(0.20, 0.18), (0.49, 0.18), (0.46, 0.95), (0.25, 0.95)])
    right_leg = _scaled([(0.51, 0.18), (0.80, 0.18), (0.75, 0.95), (0.54, 0.95)])

    draw.polygon(waistband, fill=color)
    draw.polygon(left_leg, fill=color)
    draw.polygon(right_leg, fill=color)

    # Center seam, purely decorative.
    darker = tuple(max(0, c - 40) for c in color[:3]) + (255,)
    draw.line(_scaled([(0.50, 0.05), (0.50, 0.95)]), fill=darker, width=2)

    return img


def make_dress(color=(58, 150, 102, 255)):
    """dress: sleeveless A-line, roughly matching GARMENT_LANDMARKS["dress"]."""
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    points = _scaled([
        (0.42, 0.03), (0.58, 0.03),   # shoulder straps
        (0.62, 0.15), (0.78, 0.32),   # right strap -> bodice underarm
        (0.85, 0.97), (0.15, 0.97),   # hem
        (0.22, 0.32), (0.38, 0.15),   # left bodice underarm -> strap
    ])
    draw.polygon(points, fill=color)

    # Waist seam, purely decorative.
    darker = tuple(max(0, c - 40) for c in color[:3]) + (255,)
    draw.line(_scaled([(0.30, 0.50), (0.70, 0.50)]), fill=darker, width=2)

    return img


GARMENTS = {
    "tshirt_red.png": (make_tshirt, "upper_body"),
    "jeans_blue.png": (make_jeans, "lower_body"),
    "dress_green.png": (make_dress, "dress"),
}


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename, (make_fn, category) in GARMENTS.items():
        img = make_fn()
        path = os.path.join(output_dir, filename)
        img.save(path)
        print(f"Wrote {path} ({category})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="mobile_app/assets/clothing")
    args = parser.parse_args()
    main(args.output_dir)
