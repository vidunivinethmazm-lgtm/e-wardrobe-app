"""
Model 5 — Virtual Try-On: one-time preprocessing of a raw VITON / VITON-HD
download into the `.npz` format consumed by data_pipeline.py.

Expected raw layout (standard VITON / VITON-HD download):

    <raw_dir>/
        image/             person photos,     e.g. 00001_00.jpg
        cloth/             in-shop clothing,  e.g. 00001_00.jpg
        cloth-mask/        binary cloth masks,e.g. 00001_00.jpg
        image-parse/       human parsing maps,e.g. 00001_00.png  (LIP 20-class label PNGs)
        openpose_json/     OpenPose keypoints,e.g. 00001_00_keypoints.json
        train_pairs.txt    lines: "<person_img> <cloth_img>"

CAVEAT — VERIFY BEFORE RUNNING AT SCALE: label indices for image-parse and
the OpenPose keypoint layout differ slightly between dataset releases (LIP
vs. ATR label sets; COCO-18 vs. BODY_25 keypoints). The constants below
(CLOTH_LABELS, SKIN_LABELS, OPENPOSE18_TO_JOINT) match the original VITON /
CP-VTON release. Before running on your full download, check ONE sample:
    - `np.unique(np.array(Image.open(parse_path)))` should include the
      CLOTH_LABELS and SKIN_LABELS values below somewhere.
    - `len(json.load(open(pose_path))["people"][0]["pose_keypoints"])` (or
      `pose_keypoints_2d` for BODY_25) should be 18*3=54 for the mapping
      below; if it's 25*3=75 you have BODY_25 and need to adjust
      OPENPOSE18_TO_JOINT to the BODY_25 index layout instead.

Usage:
    python -m avatar_pipeline.model5_tryon.preprocess_viton \
        --raw_dir path/to/viton --output_dir data/model5_tryon \
        --pairs_file train_pairs.txt
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from .pose_repr import build_person_representation
from backend.avatar_pipeline.model4_avatar.condition_utils import JOINT_NAMES

# Default output size — must match architecture.IMG_HEIGHT / IMG_WIDTH.
# Hardcoded (rather than imported from .architecture) so this preprocessing
# script does not require TensorFlow to be installed.
IMG_HEIGHT = 256
IMG_WIDTH = 192

# LIP 20-class parsing labels relevant to try-on (CP-VTON convention).
CLOTH_LABELS = {5, 6, 7}     # upper-clothes, dress, coat
SKIN_LABELS = {13, 14, 15}   # face, left-arm, right-arm

# OpenPose COCO-18 keypoint order -> our JOINT_NAMES (model4_avatar.condition_utils)
OPENPOSE18_TO_JOINT = {
    "left_shoulder": 5, "right_shoulder": 2,
    "left_elbow": 6, "right_elbow": 3,
    "left_wrist": 7, "right_wrist": 4,
    "left_hip": 11, "right_hip": 8,
    "left_knee": 12, "right_knee": 9,
    "left_ankle": 13, "right_ankle": 10,
}


def load_openpose_keypoints(json_path, img_width, img_height):
    """Returns {joint_name: [x, y]} normalized to [0, 1], or zeros if a
    keypoint's confidence is 0 (not detected)."""
    with open(json_path) as f:
        data = json.load(f)
    people = data.get("people", [])
    if not people:
        return {name: [0.5, 0.5] for name in JOINT_NAMES}

    flat = people[0].get("pose_keypoints", people[0].get("pose_keypoints_2d"))
    flat = np.asarray(flat, dtype=np.float32).reshape(-1, 3)

    keypoints = {}
    for name, idx in OPENPOSE18_TO_JOINT.items():
        x, y, c = flat[idx]
        if c <= 0:
            keypoints[name] = [0.5, 0.5]
        else:
            keypoints[name] = [float(x) / img_width, float(y) / img_height]
    return keypoints


def compute_skin_rgb(person_rgb, parse):
    """Average RGB (normalized [0,1]) over pixels labeled as skin (face/arms)."""
    mask = np.isin(parse, list(SKIN_LABELS))
    if mask.sum() < 10:
        return np.array([0.7, 0.55, 0.45], dtype=np.float32)  # fallback mid-tone
    pixels = person_rgb[mask].astype(np.float32) / 255.0
    return pixels.mean(axis=0)


def process_pair(person_path, cloth_path, cloth_mask_path, parse_path, pose_path,
                  img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    person = Image.open(person_path).convert("RGB")
    orig_w, orig_h = person.size

    person_arr = np.array(person.resize((img_width, img_height), Image.BILINEAR))
    cloth_arr = np.array(Image.open(cloth_path).convert("RGB").resize((img_width, img_height), Image.BILINEAR))

    cloth_mask = Image.open(cloth_mask_path).convert("L").resize((img_width, img_height), Image.NEAREST)
    cloth_mask_arr = (np.array(cloth_mask) > 127).astype(np.uint8)[..., None] * 255

    # Parsing map: keep at original resolution for accurate label lookup,
    # then resize the derived masks (nearest, to preserve hard edges).
    parse_full = np.array(Image.open(parse_path))
    person_full = np.array(person)

    skin_rgb = compute_skin_rgb(person_full, parse_full)

    body_mask_full = (parse_full != 0).astype(np.uint8) * 255
    cloth_on_person_mask_full = np.isin(parse_full, list(CLOTH_LABELS)).astype(np.uint8) * 255

    body_mask = np.array(
        Image.fromarray(body_mask_full).resize((img_width, img_height), Image.NEAREST)
    )
    cloth_on_person_mask = np.array(
        Image.fromarray(cloth_on_person_mask_full).resize((img_width, img_height), Image.NEAREST)
    )[..., None]

    cloth_on_person = person_arr * (cloth_on_person_mask > 0).astype(np.uint8)

    keypoints = load_openpose_keypoints(pose_path, orig_w, orig_h)
    person_repr = build_person_representation(body_mask, keypoints, skin_rgb, img_height, img_width)

    return {
        "person": person_arr.astype(np.uint8),
        "cloth": cloth_arr.astype(np.uint8),
        "cloth_mask": cloth_mask_arr.astype(np.uint8),
        "person_repr": person_repr.astype(np.float32),
        "cloth_on_person": cloth_on_person.astype(np.uint8),
        "cloth_on_person_mask": cloth_on_person_mask.astype(np.uint8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", default="data/model5_tryon")
    parser.add_argument("--pairs_file", default="train_pairs.txt")
    parser.add_argument("--img_height", type=int, default=IMG_HEIGHT)
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pairs (debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs_path = os.path.join(args.raw_dir, args.pairs_file)
    with open(pairs_path) as f:
        lines = [line.split() for line in f if line.strip()]
    if args.limit:
        lines = lines[: args.limit]

    written = 0
    for person_name, cloth_name in lines:
        stem = os.path.splitext(person_name)[0]
        cloth_stem = os.path.splitext(cloth_name)[0]
        try:
            sample = process_pair(
                person_path=os.path.join(args.raw_dir, "image", person_name),
                cloth_path=os.path.join(args.raw_dir, "cloth", cloth_name),
                cloth_mask_path=os.path.join(args.raw_dir, "cloth-mask", cloth_name),
                parse_path=os.path.join(args.raw_dir, "image-parse", stem + ".png"),
                pose_path=os.path.join(args.raw_dir, "openpose_json", stem + "_keypoints.json"),
                img_height=args.img_height,
                img_width=args.img_width,
            )
        except FileNotFoundError as e:
            print(f"Skipping {person_name}: {e}")
            continue

        out_path = os.path.join(args.output_dir, f"{stem}__{cloth_stem}.npz")
        np.savez_compressed(out_path, **sample)
        written += 1

    print(f"Wrote {written} preprocessed samples to {args.output_dir}")


if __name__ == "__main__":
    main()
