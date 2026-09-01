"""
Model 2 — Pose Estimation: COCO `person_keypoints` dataset loader (for the
optional fine-tuning path in train.py).

Download (val2017 is enough for experimentation, ~1GB):
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    -> annotations/person_keypoints_val2017.json
"""

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from .keypoint_utils import NUM_KEYPOINTS


def load_coco_annotations(annotation_file):
    with open(annotation_file) as f:
        coco = json.load(f)

    images = {img["id"]: img["file_name"] for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) == 0:
            continue
        samples.append(
            {
                "file_name": images[ann["image_id"]],
                "bbox": ann["bbox"],  # [x, y, w, h]
                "keypoints": ann["keypoints"],  # 51 values: x, y, v per keypoint
            }
        )
    return samples


def _load_and_crop(sample, images_dir, input_size, padding=0.2):
    img = Image.open(os.path.join(images_dir, sample["file_name"])).convert("RGB")
    x, y, w, h = sample["bbox"]
    pad_w, pad_h = w * padding, h * padding
    x0 = max(0.0, x - pad_w)
    y0 = max(0.0, y - pad_h)
    x1 = min(img.width, x + w + pad_w)
    y1 = min(img.height, y + h + pad_h)
    crop_w, crop_h = x1 - x0, y1 - y0

    crop = img.crop((x0, y0, x1, y1)).resize((input_size, input_size))

    kpts = np.array(sample["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
    kpts[:, 0] = (kpts[:, 0] - x0) / crop_w
    kpts[:, 1] = (kpts[:, 1] - y0) / crop_h
    kpts[:, :2] = np.clip(kpts[:, :2], 0.0, 1.0)

    return np.asarray(crop, dtype=np.float32), kpts


def _augment(img, kpts):
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return img, kpts


def make_dataset(samples, images_dir, input_size=224, batch_size=32, training=False):
    def gen():
        for sample in samples:
            try:
                img, kpts = _load_and_crop(sample, images_dir, input_size)
            except (FileNotFoundError, ValueError):
                continue
            yield img, kpts.reshape(-1)

    output_signature = (
        tf.TensorSpec(shape=(input_size, input_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_KEYPOINTS * 3,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if training:
        ds = ds.shuffle(1000)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
