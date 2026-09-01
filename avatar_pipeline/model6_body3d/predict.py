"""
Model 6 — 3D Body Reconstruction: inference.

Given a user photo, their measurements, Model 1's body shape, and Model 2's
pose keypoints, regress `params.PARAM_NAMES` body-mesh parameters
(`predict_body3d_params`) and build the textured-less avatar mesh as a
ready-to-render `.glb` (`predict_body3d`, via `mesh_builder.build_avatar_mesh`
+ `glb_export.mesh_to_glb_bytes`).

`predict_body3d` is what the integration controller (Step 5 of the pipeline)
calls directly — see controller.py / README.md.

CLI usage:
    python -m avatar_pipeline.model6_body3d.predict \
        --photo path/to/user.jpg \
        --bust 92 --waist 70 --hips 98 --height 165 \
        --output avatar_mesh.glb
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from avatar_pipeline.model4_avatar.condition_utils import body_shape_to_onehot, keypoints_to_pose_vector

from .face_features import extract_face_features
from .face_measurements import compute_face_measurements
from .glb_export import mesh_to_glb_bytes
from .mesh_builder import build_avatar_mesh
from .params import MEASUREMENT_SCALE, vector_to_params


def load_artifacts(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return model, config


def preprocess_photo(photo, image_size=128):
    """photo: HxWx3 uint8 RGB numpy array. Returns a (1, image_size,
    image_size, 3) float32 tensor in [0, 255] (architecture.py's photo
    branch applies its own `Rescaling(1/255)`)."""
    img = tf.image.resize(tf.convert_to_tensor(photo, dtype=tf.float32), (image_size, image_size))
    return tf.expand_dims(img, axis=0)


def build_aux_vector(bust, waist, hips, height, body_shape, keypoints_dict):
    """Assembles architecture.py's `AUX_DIM` aux vector: measurements +
    Model 1's body-shape one-hot + Model 2's pose — same order
    `synthetic_data.generate_dataset` uses for training labels."""
    measurements = np.array([bust, waist, hips, height], dtype=np.float32) / MEASUREMENT_SCALE
    return np.concatenate(
        [measurements, body_shape_to_onehot(body_shape), keypoints_to_pose_vector(keypoints_dict)]
    ).astype(np.float32)


def predict_body3d_params(model, photo, bust, waist, hips, height, body_shape, keypoints_dict, image_size=128):
    """Returns a `params.PARAM_NAMES` dict (fractions of height)."""
    photo_input = preprocess_photo(photo, image_size)
    aux_input = np.expand_dims(
        build_aux_vector(bust, waist, hips, height, body_shape, keypoints_dict), axis=0
    )
    vector = model.predict({"photo": photo_input, "aux": aux_input}, verbose=0)[0]
    return vector_to_params(vector, sigmoid_input=True)


def predict_body3d(model, photo, bust, waist, hips, height, body_shape, keypoints_dict, skin_rgb, image_size=128):
    """skin_rgb: (3,) array-like, 0-255 (e.g.
    `model3_skin_tone.color_utils.hex_to_rgb(skin_tone_result["hex"])`).

    Like Model 3's skin tone, the head's face texture and hair color are
    extracted directly from `photo` (`face_features.extract_face_features`)
    rather than predicted by Model 6's CNN — only the body-shape
    `params.PARAM_NAMES` vector is regressed.

    Returns {"params": params.PARAM_NAMES dict, "mesh_glb": bytes, "face":
    face_features.extract_face_features's result} — `mesh_glb` is a complete
    `.glb` file (see glb_export.mesh_to_glb_bytes)."""
    params = predict_body3d_params(model, photo, bust, waist, hips, height, body_shape, keypoints_dict, image_size)

    # Extract face features with landmarks for Delaunay texture warping
    # and face-proportion measurement (used to shape the head geometry).
    face = extract_face_features(photo, estimate_landmarks=True)
    landmarks_2d = face.get("landmarks_2d", None)
    face_width = face.get("face_width")
    face_height = face.get("face_height")

    # Derive face-proportion measurements from the MediaPipe landmarks.
    # These drive the head ellipsoid shape and feature positions in the
    # same way that bust/waist/hips/height drive the body shape.
    face_meas = compute_face_measurements(landmarks_2d)

    # Use detected hair style for procedural hair cap shaping
    _hair_style = face.get("facial_analysis", {}).get("hair_style", "medium")

    mesh = build_avatar_mesh(
        params, height, skin_rgb,
        face_crop=face["face_crop"], hair_rgb=face["hair_rgb"],
        selfie_rgb=photo, landmarks_2d=landmarks_2d,
        blend_mode="feather",
        face_width=face_width,
        face_height=face_height,
        face_measurements=face_meas,
        hair_style=_hair_style,
    )
    return {"params": params, "mesh_glb": mesh_to_glb_bytes(mesh), "face": face}


if __name__ == "__main__":
    from avatar_pipeline.model1_body_shape.synthetic_data import classify_body_shape
    from avatar_pipeline.model3_skin_tone.color_utils import hex_to_rgb
    from avatar_pipeline.model4_avatar.synthetic_avatars import CANONICAL_POSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_models/model6_body3d")
    parser.add_argument("--photo", required=True)
    parser.add_argument("--bust", type=float, required=True)
    parser.add_argument("--waist", type=float, required=True)
    parser.add_argument("--hips", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--skin_hex", default="#d7bd96")
    parser.add_argument("--output", default="avatar_mesh.glb")
    args = parser.parse_args()

    model, config = load_artifacts(args.model_dir)
    image_size = config["image_size"]

    photo = np.array(Image.open(args.photo).convert("RGB").resize((image_size, image_size)))
    body_shape = classify_body_shape(args.bust, args.waist, args.hips)
    keypoints_dict = {name: list(coords) for name, coords in CANONICAL_POSE.items()}
    skin_rgb = hex_to_rgb(args.skin_hex)

    result = predict_body3d(
        model, photo, args.bust, args.waist, args.hips, args.height,
        body_shape, keypoints_dict, skin_rgb, image_size=image_size,
    )

    with open(args.output, "wb") as f:
        f.write(result["mesh_glb"])

    print(json.dumps(result["params"], indent=2))
    print(f"Saved mesh to {args.output}")
