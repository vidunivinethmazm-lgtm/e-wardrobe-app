"""
eWardrobeAI — Asset Directory Setup + Test Selfie Generator

Creates:
  assets/avatars/          — placeholder for base_avatar.glb
  assets/outfits/          — placeholder for clothing .glb files
  assets/thumbnails/       — placeholder for garment thumbnails
  assets/test_selfie.jpg   — synthetic test face image (real-looking, 400×400)

Run: python scripts/generate_test_assets.py
"""

import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Directory Creation ────────────────────────────────────────────────────────

DIRS = [
    'assets/avatars',
    'assets/outfits',
    'assets/thumbnails',
    'models',
]

def create_directories():
    print("[Setup] Creating asset directories …")
    for d in DIRS:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"  ✅  {d}/")


# ── README Placeholders for .glb files ───────────────────────────────────────

AVATAR_README = """# Base Avatar

Place your Blender-exported avatar GLB file here:
  base_avatar.glb

Export requirements:
  - Y-up coordinate system
  - Armature origin at world origin (0, 0, 0)
  - Mixamo-compatible bone names (mixamorigHips, mixamorigSpine, etc.)
  - UV maps on both body mesh and head mesh
  - Mixamo animation clips baked as separate action tracks

Recommended Mixamo clips to bake:
  Mixamo_Idle, Mixamo_Walking, Mixamo_TurnLeft,
  Mixamo_TPose, Mixamo_APose, Mixamo_CatwalkWalk

Download base avatar: https://www.mixamo.com
"""

OUTFITS_README = """# Clothing Assets

Place Blender-exported clothing GLB files here.
Each file corresponds to a GarmentRecord.asset_path in outfit_recommender.py.

Required files:
  white_oxford_shirt.glb   (GAR-001)
  navy_polo.glb            (GAR-002)
  striped_tshirt.glb       (GAR-003)
  graphic_hoodie.glb       (GAR-004)
  slim_chinos.glb          (GAR-005)
  dark_jeans.glb           (GAR-006)
  dress_trousers.glb       (GAR-007)
  wool_blazer.glb          (GAR-008)
  puffer_jacket.glb        (GAR-009)
  wrap_midi_dress.glb      (GAR-010)
  evening_gown.glb         (GAR-011)
  classic_suit.glb         (GAR-012)

Export requirements:
  - Same Mixamo skeleton as base_avatar.glb
  - Blend-shape morph targets for size variation
  - Y-up coordinate system
"""

def write_readmes():
    readme_map = {
        'assets/avatars/README.txt':   AVATAR_README,
        'assets/outfits/README.txt':   OUTFITS_README,
    }
    for rel_path, content in readme_map.items():
        path = os.path.join(ROOT, rel_path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅  {rel_path}")


# ── Synthetic Test Selfie ─────────────────────────────────────────────────────

def generate_test_selfie(output_path: str, size: int = 400):
    """
    Generate a realistic-looking synthetic face image for testing.
    All face processing (MediaPipe + CNN) should be able to detect
    the key landmarks from this image.
    """
    img = np.full((size, size, 3), (210, 185, 160), dtype=np.uint8)  # skin-tone BG
    cx, cy = size // 2, size // 2

    # ── Neck ─────────────────────────────────────────────────────────────────
    cv2.rectangle(img, (cx-28, cy+100), (cx+28, cy+160), (195, 165, 140), -1)

    # ── Face oval ────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx, cy+10), (130, 160), 0, 0, 360, (220, 190, 165), -1)

    # ── Forehead shading ─────────────────────────────────────────────────────
    cv2.ellipse(img, (cx, cy-90), (100, 60), 0, 0, 180, (200, 172, 148), -1)

    # ── Eyebrows ─────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx-55, cy-55), (32, 8), -10, 0, 180, (80, 55, 40), -1)
    cv2.ellipse(img, (cx+55, cy-55), (32, 8),  10, 0, 180, (80, 55, 40), -1)

    # ── Eye sockets ──────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx-55, cy-30), (25, 16), 0, 0, 360, (180, 155, 135), -1)
    cv2.ellipse(img, (cx+55, cy-30), (25, 16), 0, 0, 360, (180, 155, 135), -1)

    # ── Irises ───────────────────────────────────────────────────────────────
    cv2.circle(img, (cx-55, cy-30), 11, (70, 100, 130), -1)
    cv2.circle(img, (cx+55, cy-30), 11, (70, 100, 130), -1)

    # ── Pupils ───────────────────────────────────────────────────────────────
    cv2.circle(img, (cx-55, cy-30), 6, (20, 20, 20), -1)
    cv2.circle(img, (cx+55, cy-30), 6, (20, 20, 20), -1)

    # ── Eye highlights ────────────────────────────────────────────────────────
    cv2.circle(img, (cx-50, cy-34), 3, (240, 240, 240), -1)
    cv2.circle(img, (cx+60, cy-34), 3, (240, 240, 240), -1)

    # ── Nose ─────────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx, cy+25), (18, 22), 0, 0, 360, (195, 165, 140), -1)
    cv2.circle(img,  (cx-14, cy+38), 8,  (185, 155, 130), -1)
    cv2.circle(img,  (cx+14, cy+38), 8,  (185, 155, 130), -1)
    cv2.circle(img,  (cx,    cy+38), 7,  (160, 130, 110), -1)

    # ── Lips ─────────────────────────────────────────────────────────────────
    # Upper lip
    cv2.ellipse(img, (cx, cy+75), (38, 14), 0, 180, 360, (175, 110, 100), -1)
    # Lower lip
    cv2.ellipse(img, (cx, cy+80), (38, 18), 0,   0, 180, (185, 120, 110), -1)
    # Lip line
    cv2.line(img, (cx-38, cy+75), (cx+38, cy+75), (145, 85, 80), 2)

    # ── Chin ─────────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx, cy+130), (50, 30), 0, 0, 180, (210, 180, 155), -1)

    # ── Ears ─────────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx-128, cy-10), (18, 28), 0, 0, 360, (205, 175, 150), -1)
    cv2.ellipse(img, (cx+128, cy-10), (18, 28), 0, 0, 360, (205, 175, 150), -1)

    # ── Hair ─────────────────────────────────────────────────────────────────
    cv2.ellipse(img, (cx, cy-110), (135, 100), 0, 180, 360, (60, 40, 25), -1)
    cv2.ellipse(img, (cx-120, cy-40), (40, 80), -20, 180, 360, (60, 40, 25), -1)
    cv2.ellipse(img, (cx+120, cy-40), (40, 80),  20, 180, 360, (60, 40, 25), -1)

    # ── Subtle skin texture (noise) ───────────────────────────────────────────
    noise = np.random.randint(-6, 7, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Gaussian blur for smoothness ─────────────────────────────────────────
    img = cv2.GaussianBlur(img, (3, 3), 0)

    cv2.imwrite(output_path, img)
    print(f"\n  ✅  Test selfie saved → {output_path}  ({size}×{size} px)")
    return img


def generate_thumbnails():
    """Generate colour-swatch thumbnails for each garment."""
    SWATCHES = {
        'white_oxford_shirt.jpg':  (245, 245, 245),
        'navy_polo.jpg':           (0,   31,  91),
        'striped_tshirt.jpg':      (30,  80, 200),
        'graphic_hoodie.jpg':      (54,  69,  79),
        'slim_chinos.jpg':         (195, 176, 145),
        'dark_jeans.jpg':          (27,  20, 100),
        'dress_trousers.jpg':      (54,  69,  79),
        'wool_blazer.jpg':         (54,  69,  79),
        'puffer_jacket.jpg':       (17,  17,  17),
        'wrap_midi_dress.jpg':     (80, 200, 120),
        'evening_gown.jpg':        (17,  17,  17),
        'classic_suit.jpg':        (54,  69,  79),
    }
    thumb_dir = os.path.join(ROOT, 'assets', 'thumbnails')
    for filename, colour in SWATCHES.items():
        img  = np.full((120, 90, 3), colour[::-1], dtype=np.uint8)  # BGR
        path = os.path.join(thumb_dir, filename)
        cv2.imwrite(path, img)
    print(f"  ✅  {len(SWATCHES)} garment thumbnails generated in assets/thumbnails/")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  eWardrobeAI — Asset Setup")
    print("="*55)

    create_directories()
    print()
    write_readmes()
    print()

    selfie_path = os.path.join(ROOT, 'assets', 'test_selfie.jpg')
    generate_test_selfie(selfie_path)

    print()
    generate_thumbnails()

    print("\n" + "="*55)
    print("  Setup complete.")
    print("  Next: Add real .glb files to assets/avatars/ and assets/outfits/")
    print("="*55 + "\n")
