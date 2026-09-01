# Avatar Pipeline — Integration Guide

End-to-end visual avatar system for the eWardrobe app: turns a user photo +
body measurements into a stylized 2D avatar plus a 3D body mesh, then
dresses the 2D avatar in a clothing item supplied by the recommendation
team.

This document covers the **integration layer** only. Each `modelN_xxx/`
package is independently implemented, trained, and verified — see their
docstrings for model-specific details (architectures, datasets,
known-limitations). The controller that wires them together lives at
[`controller.py`](controller.py), with the TensorFlow-independent shared
types and Phase B logic factored out into
[`pipeline_types.py`](pipeline_types.py) (see §2).

## 1. Data-flow diagram

```
                                    INPUTS (per user)
        +-------------------------------------------------------------+
        | photo: HxWx3 uint8 RGB numpy array (full body, face visible) |
        | measurements: bust, waist, hips, height (cm)                 |
        | [silhouette_path]  (only if Model 1 config.model_type ==     |
        |  "fusion")                                                   |
        +-------------------------------------------------------------+
              |                        |                        |
              v                        v                        v
  +----------------------+  +-----------------------+  +------------------------+
  | MODEL 1 — Body Shape  |  | MODEL 2 — Pose         |  | MODEL 3 — Skin Tone     |
  | predict_body_shape()  |  | extract_keypoints() +  |  | predict_skin_tone()     |
  |                       |  |  keypoints_to_avatar_  |  |                         |
  | in: measurements,     |  |  params()              |  | in: photo               |
  |     [silhouette]      |  | in: photo              |  | out: {label, hex, lab,  |
  | out: body_shape,      |  | out: keypoints_dict    |  |       confidence,       |
  |      confidence,      |  |   {joint_name: [x,y]}  |  |       avatar_render:    |
  |      probabilities    |  |   (17 COCO joints)     |  |        {base_color,...}}|
  +-----------+-----------+  +-----------+------------+  +------------+-----------+
              |                          |                             |
              | body_shape (str)         | keypoints_dict              | skin_tone_result
              |                          | (12 of 17 joints used,      | (dict, must
              |                          |  see condition_utils.       |  contain "hex")
              |                          |  JOINT_NAMES)                |
              +-------------+------------+--------------+--------------+
                             |                           |
                             v                           v
              +--------------------------------------------------------+
              |       MODEL 4 — Avatar Generation (cVAE-GAN decoder)    |
              |                                                          |
              | condition_utils.build_condition_vector(                 |
              |     body_shape_to_onehot(body_shape),       # 5 dims    |
              |     keypoints_to_pose_vector(keypoints_dict),# 24 dims  |
              |     skin_tone_to_rgb(skin_tone_result))      # 3 dims   |
              |   -> condition vector, CONDITION_DIM = 32 float32       |
              |                                                          |
              | generate_avatar(decoder, body_shape, keypoints_dict,    |
              |                 skin_tone_result, seed=None)            |
              | out: avatar_rgba — 128x128 RGBA PIL.Image,              |
              |      transparent background                             |
              +-----------------------------+----------------------------+
                                            |
                                            | avatar_rgba
                                            | == END OF PHASE A ==
                                            | (cache as the user's base avatar)
                                            v
  +------------------------------+   +-----------------------------------------+
  | FROM RECOMMENDATION TEAM      |   | MODEL 5 — Virtual Try-On (TPS warp)      |
  | (per clothing item)           |   |                                           |
  | clothing_rgb:  HcxWcx3 uint8  |-->| try_on_avatar(avatar_rgba, clothing_rgb, |
  |   product photo                |   |     clothing_mask, keypoints_dict,       |
  | clothing_mask: HcxWc {0,255}  |   |     category)                            |
  |   foreground mask              |   |                                           |
  | category: "upper_body" |       |   | out: dressed_avatar_rgba —               |
  |   "lower_body" | "dress"       |   |      128x128 RGBA PIL.Image              |
  +------------------------------+   |      == END OF PHASE B ==                |
                                      +-----------------------------------------+
```

Model 6 runs **in parallel with Model 4**, fed by the same `body_shape` /
`keypoints_dict` / `skin_tone_result` outputs from Models 1-3, plus the raw
`photo` and `measurements` from `INPUTS`. Its outputs join `avatar_rgba` in
the `AvatarResult` returned at the end of Phase A:

```
                  body_shape, keypoints_dict, skin_tone_result
                  (from Models 1/2/3, same fan-in as Model 4)
                                      |
              photo, measurements ---+
                  (from INPUTS)      |
                                      v
              +--------------------------------------------------------+
              |     MODEL 6 — 3D Body Reconstruction (CNN + aux MLP)    |
              |                                                          |
              | predict_body3d(model, photo, bust, waist, hips, height, |
              |     body_shape, keypoints_dict, skin_rgb)               |
              |                                                          |
              | in: photo (128x128x3) + aux vector (measurements /      |
              |     MEASUREMENT_SCALE + body_shape one-hot + pose       |
              |     vector), AUX_DIM = 33                               |
              | out: body3d_params (params.PARAM_NAMES, 13 floats) ->   |
              |      mesh_builder.build_avatar_mesh(...) -> glb_export. |
              |      mesh_to_glb_bytes(...) -> avatar_mesh_glb (bytes)  |
              |                                                          |
              | face_features.extract_face_features(photo) (OpenCV,     |
              |     not the CNN — same pattern as Model 3's face crop)  |
              |     -> face_crop + hair_rgb, also fed into              |
              |     mesh_builder.build_avatar_mesh(...) for the head's  |
              |     texture and hair/eyebrow color                      |
              +-----------------------------+----------------------------+
                                            |
                                            | body3d_params, avatar_mesh_glb
                                            | == ALSO PART OF PHASE A ==
                                            v
                                  (joins avatar_rgba in AvatarResult)
```

## 2. Execution order & phases

The controller (`controller.py`) splits the pipeline into two phases with
very different cost profiles:

**Phase A — `build_avatar()`** runs Models 1, 2, 3, 4, 6 in sequence:

1. **Model 1** (`predict_body_shape`) — measurements (+ optional
   silhouette) → `body_shape` (one of `Hourglass`, `Pear`, `Apple`,
   `Rectangle`, `InvertedTriangle`).
2. **Model 2** (`extract_keypoints` + `keypoints_to_avatar_params`) — photo
   → `keypoints_dict` (17 COCO joints, normalized `[0,1]` `(x, y)`).
3. **Model 3** (`predict_skin_tone`) — photo → `skin_tone_result` (palette
   match + avatar render colors).
4. **Model 4** (`generate_avatar`) — combines the three outputs above into a
   32-dim condition vector and decodes a 128×128 RGBA avatar.
5. **Model 6** (`predict_body3d`) — combines the photo with an aux vector
   (measurements + Model 1's body-shape one-hot + Model 2's pose vector,
   `AUX_DIM=33`) to regress 13 body-mesh parameters (`params.PARAM_NAMES`).
   Independently of the CNN — via OpenCV, the same pattern as Model 3's face
   crop — `face_features.extract_face_features(photo)` crops the user's face
   and samples their hair color. Both are fed into
   `mesh_builder.build_avatar_mesh`, which builds a procedural humanoid mesh
   with a photo-textured head (face crop projected onto the head's UV map)
   and procedural hair/eyes/eyebrows/nose/mouth/ears, then packages it as a
   `.glb` file (`glb_export.mesh_to_glb_bytes`).

This is "expensive" (4 model forward passes + a generator decode) but only
needs to run **once per user**, and again only when their photo or
measurements change. Its result is an `AvatarResult`.

**Phase B — `dress_avatar()`** runs Model 5:

6. **Model 5** (`try_on_avatar`) — classical TPS warp (no trained weights)
   driven by Model 2's `keypoints_dict`, composites the recommendation
   team's clothing image onto the cached avatar.

This is cheap and runs **once per clothing item** the recommendation team
wants visualized — typically many times per user session, reusing the same
`AvatarResult` from Phase A.

`run_pipeline()` runs both phases back-to-back for a single end-to-end call.
`save_avatar_result()` / `load_avatar_result()` persist Phase A's output
(`avatar.png` + `avatar_mesh.glb` + `avatar_meta.json`) so Phase B can run
later, e.g. from a separate API request, without reloading/rerunning Models
1, 2, 3, 4, 6.

`AvatarResult`, `ClothingItem`, `dress_avatar`, `save_avatar_result`, and
`load_avatar_result` live in `pipeline_types.py`, which only depends on
numpy/PIL/json and Model 5 (no TensorFlow). A deployment that only needs
Phase B (e.g. a "try on this item on my saved avatar" endpoint) can import
just `pipeline_types` and skip loading Models 1/3/4/6 entirely.

## 3. Data-passing conventions

| From → To | Value | Type / shape | Notes |
|---|---|---|---|
| caller → Model 1 | `bust, waist, hips, height` | floats (cm) | `silhouette_path` only if `config["model_type"] == "fusion"` |
| Model 1 → Model 4 | `body_shape` | `str`, one of `condition_utils.BODY_SHAPE_NAMES` | one-hot encoded inside `build_condition_vector` |
| caller → Model 2 | `photo` | `HxWx3 uint8 RGB numpy array` | same photo is reused for Model 3 |
| Model 2 → Models 4 & 5 | `keypoints_dict` | `{joint_name: [x, y]}`, 17 COCO joint names, values normalized `[0, 1]` | Model 4 reads only the 12 names in `condition_utils.JOINT_NAMES` (a subset); Model 5 reads a similar subset plus derives `torso_center` / `hip_center` midpoints |
| caller → Model 3 | `photo` | same array as passed to Model 2 | raises `ValueError` if no face is detected — caller should catch and prompt for a clearer photo |
| Model 3 → Model 4 | `skin_tone_result` | `dict` with at least `"hex"` | `condition_utils.skin_tone_to_rgb` extracts normalized RGB from `"hex"` |
| Model 4 → Model 5 | `avatar_rgba` | `128x128x4 uint8` (as `PIL.Image` "RGBA" or `np.array`) | transparent background; Model 5 outputs the same size |
| recommendation team → Model 5 | `clothing_rgb`, `clothing_mask`, `category` | `HcxWcx3 uint8`, `HcxWc {0,255}`, `str` | `category` ∈ `{"upper_body", "lower_body", "dress"}`, must match `model5_tryon.predict.GARMENT_LANDMARKS` |
| Model 5 → caller / app | `dressed_avatar_rgba` | `128x128x4 uint8 PIL.Image "RGBA"` | final output, ready to render in the app UI |
| caller → Model 6 | `photo`, `bust, waist, hips, height` | same `photo` as Models 2/3; measurements as floats (cm) | `predict.build_aux_vector` divides measurements by `params.MEASUREMENT_SCALE`; `photo` is also passed as-is to `face_features.extract_face_features` |
| Models 1/2/3 → Model 6 | `body_shape`, `keypoints_dict`, `skin_tone_result["hex"]` | same values passed to Model 4 | `body_shape` → `condition_utils.body_shape_to_onehot`; `keypoints_dict` → `condition_utils.keypoints_to_pose_vector`; `hex` → `color_utils.hex_to_rgb` → mesh skin color |
| Model 6 → caller / app | `body3d_params`, `avatar_mesh_glb` | `dict` of 13 floats keyed by `params.PARAM_NAMES`; `bytes` (a complete `.glb` file) | `body3d_params` are fractions of `height`; `avatar_mesh_glb` is ready to write to disk or serve as `model/gltf-binary`; the mesh's "face" part is textured with the user's own face crop (or a flat skin-color fallback if no face is detected) and its "hair"/eyebrows use the photo's sampled hair color (or `face_features.DEFAULT_HAIR_RGB`) |

**Caching contract (`AvatarResult`)**: `avatar_rgba` (PIL Image),
`body_shape`, `body_shape_confidence`, `keypoints_dict`, `skin_tone_result`,
`avatar_mesh_glb` (bytes), `body3d_params` (dict). Everything Phase B needs
is in this one object — persist it with `save_avatar_result()` and reload
with `load_avatar_result()`.

## 4. Recommended end-to-end folder structure

> This pipeline is consumed by a Flask API + Expo mobile app — see the
> [project root README](../README.md) for the full-stack layout
> (`server/`, `mobile_app/`, `scripts/`) and how they call into
> `controller.py` / `pipeline_types.py`.

```
New avatar/                          (project root)
├── requirements.txt                 (tensorflow>=2.15, tensorflow-hub, numpy,
│                                      pandas, scikit-learn, matplotlib,
│                                      pillow, opencv-python, joblib)
├── backend/avatar_pipeline/
│   ├── __init__.py
│   ├── controller.py                <- integration entry point (this PR)
│   ├── pipeline_types.py            <- shared types + Phase B (TF-free)
│   ├── README.md                    <- this file
│   ├── model1_body_shape/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── data_pipeline.py
│   │   ├── synthetic_data.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── model2_pose/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── data_pipeline.py
│   │   ├── keypoint_utils.py
│   │   ├── movenet_inference.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── model3_skin_tone/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── color_utils.py
│   │   ├── data_pipeline.py
│   │   ├── face_crop.py
│   │   ├── pseudo_labels.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── model4_avatar/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── condition_utils.py
│   │   ├── data_pipeline.py
│   │   ├── synthetic_avatars.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── model5_tryon/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── data_pipeline.py
│   │   ├── pose_repr.py
│   │   ├── preprocess_viton.py
│   │   ├── tps_utils.py
│   │   ├── train.py
│   │   └── predict.py
│   └── model6_body3d/
│       ├── __init__.py
│       ├── architecture.py
│       ├── params.py                <- PARAM_NAMES, PARAM_RANGES, sigmoid <-> physical
│       ├── face_features.py          <- OpenCV face crop + hair color (no CNN,
│       │                                same pattern as model3's face_crop.py)
│       ├── mesh_builder.py           <- procedural humanoid mesh from params
│       │                                + face_features (textured head, hair)
│       ├── glb_export.py             <- mesh -> .glb bytes (no extra deps)
│       ├── synthetic_data.py
│       ├── train.py
│       └── predict.py
├── saved_models/                    (trained artifacts — produced on Colab/
│   │                                  cloud GPU, downloaded here for inference)
│   ├── model1_body_shape/           best_model.keras, measurement_scaler.joblib,
│   │                                 config.json, class_names.json
│   ├── model2_pose/                 (only needed if pose_method="finetuned";
│   │                                 the recommended movenet path needs nothing
│   │                                 here — weights come from TF Hub)
│   ├── model3_skin_tone/            best_model.keras, config.json
│   ├── model4_avatar/               decoder.keras (+ encoder/discriminator
│   │                                 if present, unused at inference)
│   ├── model5_tryon/                gmm.keras, tom.keras (only for the
│   │                                 try_on_photo real-photo path — not
│   │                                 needed for try_on_avatar)
│   └── model6_body3d/               best_model.keras, config.json
│                                     (image_size, param_names)
├── data/                             (training data per model)
│   ├── model1_body_shape/
│   ├── model2_pose/
│   ├── model3_skin_tone/
│   ├── model4_avatar/
│   ├── model5_tryon/
│   └── model6_body3d/
└── output/                           (controller CLI output: avatar.png,
                                        avatar_mesh.glb, avatar_meta.json,
                                        dressed_avatar.png)
```

`saved_models/` is intentionally excluded from version control in most setups
(large binary artifacts) — see each model's `train.py` for how it's produced.

## 5. Running the pipeline

```bash
python -m backend.avatar_pipeline.controller \
    --photo path/to/user.jpg \
    --bust 92 --waist 70 --hips 98 --height 165 \
    --clothing path/to/clothing.png --clothing_mask path/to/clothing_mask.png \
    --category upper_body \
    --output_dir output/
```

This loads Models 1, 3, 4, 6 + the MoveNet pose estimator once
(`load_pipeline_models`), runs Phase A + Phase B (`run_pipeline`), and writes
`output/avatar.png`, `output/avatar_mesh.glb`, `output/avatar_meta.json`, and
`output/dressed_avatar.png`.

For a long-running app server, call `load_pipeline_models()` once at
startup, then per-request call `build_avatar()` (when the user's
photo/measurements change) and `dress_avatar()` (per recommended item),
using `save_avatar_result()` / `load_avatar_result()` to persist
`AvatarResult` between requests.

This entire pipeline (loading the five trained models + MoveNet, plus
running TF inference) requires TensorFlow and is intended to run on
Colab/cloud GPU per `requirements.txt`, not on a machine without TensorFlow
installed.
