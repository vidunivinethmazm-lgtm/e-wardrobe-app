"""
End-to-end avatar pipeline controller.

Wires Models 1-6 together into two phases (see README.md for the full
ASCII data-flow diagram):

  Phase A — build_avatar()  (Models 1, 2, 3, 4, 6)
    photo + measurements -> body_shape (Model 1)
    photo                -> keypoints_dict (Model 2)
    photo                -> skin_tone_result (Model 3)
    (body_shape, keypoints_dict, skin_tone_result) -> avatar_rgba (Model 4)
    (photo, measurements, body_shape, keypoints_dict, skin_tone_result)
        -> body3d_params, avatar_mesh_glb (Model 6)

  Phase B — dress_avatar()  (Model 5)
    (avatar_rgba, keypoints_dict) + clothing item -> dressed_avatar_rgba

Phase A loads/runs four networks plus a generator and only needs to run
once per user (or whenever their photo / measurements change). Its result
(`AvatarResult`) can be cached — see `save_avatar_result` /
`load_avatar_result` — and reused across many Phase B calls, since Phase B
is a classical TPS warp with no trained weights and runs once per
recommended clothing item.

`run_pipeline()` chains both phases for a single end-to-end call.

CLI usage (run from the project root, after placing trained artifacts under
saved_models/, see README.md):

    python -m avatar_pipeline.controller \
        --photo path/to/user.jpg \
        --bust 92 --waist 70 --hips 98 --height 165 \
        --clothing path/to/clothing.png --clothing_mask path/to/mask.png \
        --category upper_body --output_dir output/
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

from avatar_pipeline.model1_body_shape.predict import (
    load_artifacts as load_body_shape_artifacts,
    predict_body_shape,
)
from avatar_pipeline.model2_pose.keypoint_utils import keypoints_to_avatar_params
from avatar_pipeline.model2_pose.movenet_inference import MoveNetPoseEstimator
from avatar_pipeline.model2_pose.predict import extract_keypoints
from avatar_pipeline.model3_skin_tone.color_utils import hex_to_rgb
from avatar_pipeline.model3_skin_tone.predict import (
    load_model as load_skin_tone_model,
    predict_skin_tone,
)
from avatar_pipeline.model4_avatar.predict import generate_avatar, load_decoder
from avatar_pipeline.model6_body3d.avatar_builder import create_avatar_builder
from avatar_pipeline.model6_body3d.predict import (
    load_artifacts as load_body3d_artifacts,
    predict_body3d,
)
from avatar_pipeline.pipeline_types import (
    AvatarResult,
    ClothingItem,
    dress_avatar,
    load_avatar_result,
    save_avatar_result,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineModels",
    "load_pipeline_models",
    "AvatarResult",
    "ClothingItem",
    "build_avatar",
    "dress_avatar",
    "run_pipeline",
    "save_avatar_result",
    "load_avatar_result",
]


@dataclass
class PipelineModels:
    """Every loaded artifact the pipeline needs. Build once with
    `load_pipeline_models` and reuse across requests."""

    body_shape_model: Any
    body_shape_scaler: Any
    body_shape_config: dict
    skin_tone_model: Any
    skin_tone_config: dict
    avatar_decoder: Any
    body3d_model: Any
    body3d_config: dict
    pose_method: str = "movenet"
    pose_model_dir: Optional[str] = None
    movenet_variant: str = "thunder"
    pose_estimator: Optional[MoveNetPoseEstimator] = None


def load_pipeline_models(saved_models_dir="saved_models", pose_method="movenet", pose_model_dir=None, movenet_variant="thunder"):
    """Loads Models 1, 3, 4, 6 from `saved_models_dir/modelN_xxx/`, plus the
    MoveNet pose estimator if `pose_method == "movenet"` (Model 5 needs no
    trained weights for `try_on_avatar`).

    The MoveNet estimator is instantiated once here (it pulls weights from
    TF Hub) rather than per request, since `extract_keypoints(method="movenet")`
    would otherwise reload it on every call.
    """
    body_shape_model, body_shape_scaler, body_shape_config = load_body_shape_artifacts(
        os.path.join(saved_models_dir, "model1_body_shape")
    )
    skin_tone_model, skin_tone_config = load_skin_tone_model(
        os.path.join(saved_models_dir, "model3_skin_tone")
    )
    avatar_decoder = load_decoder(os.path.join(saved_models_dir, "model4_avatar"))
    body3d_model, body3d_config = load_body3d_artifacts(os.path.join(saved_models_dir, "model6_body3d"))

    pose_estimator = MoveNetPoseEstimator(variant=movenet_variant) if pose_method == "movenet" else None

    return PipelineModels(
        body_shape_model=body_shape_model,
        body_shape_scaler=body_shape_scaler,
        body_shape_config=body_shape_config,
        skin_tone_model=skin_tone_model,
        skin_tone_config=skin_tone_config,
        avatar_decoder=avatar_decoder,
        body3d_model=body3d_model,
        body3d_config=body3d_config,
        pose_method=pose_method,
        pose_model_dir=pose_model_dir,
        movenet_variant=movenet_variant,
        pose_estimator=pose_estimator,
    )


def _extract_keypoints_dict(models, photo):
    if models.pose_method == "movenet":
        keypoints = models.pose_estimator.predict(photo)
    else:
        # "finetuned" loads its model fresh on every call (movenet is the
        # recommended path — see model2_pose/movenet_inference.py).
        keypoints = extract_keypoints(photo, method="finetuned", model_dir=models.pose_model_dir)
    return keypoints_to_avatar_params(keypoints)["keypoints"]


def build_avatar(models, photo, bust, waist, hips, height, silhouette_path=None, seed=None):
    """photo: HxWx3 uint8 RGB numpy array (full-body, face visible — used by
        both Model 2 and Model 3).
    bust/waist/hips/height: cm, from the user's profile.
    silhouette_path: required only if Model 1 was trained with
        config["model_type"] == "fusion".

    Returns an `AvatarResult`. Raises ValueError if Model 3 finds no face in
    `photo` — callers should catch this and prompt the user for a clearer
    photo.
    """
    body_shape_result = predict_body_shape(
        model_dir=None,
        bust=bust, waist=waist, hips=hips, height=height,
        silhouette_path=silhouette_path,
        model=models.body_shape_model, scaler=models.body_shape_scaler, config=models.body_shape_config,
    )

    keypoints_dict = _extract_keypoints_dict(models, photo)

    skin_tone_result = predict_skin_tone(photo, models.skin_tone_model, models.skin_tone_config)

    avatar_rgba = generate_avatar(
        models.avatar_decoder,
        body_shape_result["body_shape"],
        keypoints_dict,
        skin_tone_result,
        seed=seed,
    )

    skin_rgb = hex_to_rgb(skin_tone_result["hex"])
    body3d_result = predict_body3d(
        models.body3d_model, photo, bust, waist, hips, height,
        body_shape_result["body_shape"], keypoints_dict, skin_rgb,
        image_size=models.body3d_config["image_size"],
    )

    mesh_glb = body3d_result["mesh_glb"]
    face = body3d_result["face"]
    facial_analysis = face["facial_analysis"]
    gender = facial_analysis["gender"]

    try:
        avatar_builder = create_avatar_builder(use_realistic=True)
        if avatar_builder is not None:
            realistic_glb = avatar_builder.build_realistic_avatar(
                gender=gender,
                facial_analysis=facial_analysis,
                body3d_params=body3d_result["params"],
                height=height,
                skin_rgb=skin_rgb,
                hair_rgb=face["hair_rgb"],
                face_crop=face["face_crop"],
                selfie_rgb=photo,
                landmarks_2d=face.get("landmarks_2d"),
                blend_mode="feather",
                face_width=face.get("face_width"),
                face_height=face.get("face_height"),
            )
            if realistic_glb is not None:
                mesh_glb = realistic_glb
            else:
                logger.warning("build_realistic_avatar returned None; using Phase 1 mesh")
    except Exception:
        logger.warning("Falling back to Phase 1 procedural mesh", exc_info=True)

    return AvatarResult(
        avatar_rgba=avatar_rgba,
        body_shape=body_shape_result["body_shape"],
        body_shape_confidence=body_shape_result["confidence"],
        keypoints_dict=keypoints_dict,
        skin_tone_result=skin_tone_result,
        avatar_mesh_glb=mesh_glb,
        body3d_params=body3d_result["params"],
        gender=gender,
        facial_analysis=facial_analysis,
    )


def run_pipeline(models, photo, bust, waist, hips, height, clothing_item, silhouette_path=None, seed=None):
    """Runs Models 1->2->3->4->5 in one call. Returns
    (dressed_avatar_rgba, avatar_result) — keep `avatar_result` to call
    `dress_avatar` again for additional clothing items without rerunning
    Models 1-4."""
    avatar_result = build_avatar(models, photo, bust, waist, hips, height, silhouette_path, seed)
    dressed = dress_avatar(avatar_result, clothing_item)
    return dressed, avatar_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_models_dir", default="saved_models")
    parser.add_argument("--photo", required=True, help="Full-body photo, face visible")
    parser.add_argument("--bust", type=float, required=True)
    parser.add_argument("--waist", type=float, required=True)
    parser.add_argument("--hips", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--silhouette", default=None)
    parser.add_argument("--clothing", required=True, help="Clothing product photo")
    parser.add_argument("--clothing_mask", required=True, help="Clothing foreground mask")
    parser.add_argument("--category", default="upper_body", choices=["upper_body", "lower_body", "dress"])
    parser.add_argument("--pose_method", choices=["movenet", "finetuned"], default="movenet")
    parser.add_argument("--movenet_variant", choices=["lightning", "thunder"], default="thunder")
    parser.add_argument("--pose_model_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    models = load_pipeline_models(
        args.saved_models_dir, args.pose_method, args.pose_model_dir, args.movenet_variant
    )

    photo = np.array(Image.open(args.photo).convert("RGB"))
    clothing_rgb = np.array(Image.open(args.clothing).convert("RGB"))
    clothing_mask = np.array(Image.open(args.clothing_mask).convert("L"))
    clothing_item = ClothingItem(rgb=clothing_rgb, mask=clothing_mask, category=args.category)

    dressed, avatar_result = run_pipeline(
        models, photo, args.bust, args.waist, args.hips, args.height,
        clothing_item, silhouette_path=args.silhouette, seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_avatar_result(avatar_result, args.output_dir)
    dressed.save(os.path.join(args.output_dir, "dressed_avatar.png"))

    print(json.dumps({
        "body_shape": avatar_result.body_shape,
        "body_shape_confidence": avatar_result.body_shape_confidence,
        "skin_tone": avatar_result.skin_tone_result["label"],
        "body3d_params": avatar_result.body3d_params,
    }, indent=2))
    print(f"Saved base avatar + 3D mesh + dressed avatar to {args.output_dir}/")
