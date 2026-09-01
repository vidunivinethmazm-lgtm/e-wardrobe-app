"""
Flask API for the eWardrobe avatar pipeline — the HTTP layer the mobile app
(`mobile/`) talks to. Wraps `avatar_pipeline.controller` (Phase A: Models
1-4) and `avatar_pipeline.pipeline_types` (Phase B: Model 5).

Two modes, selected by the AVATAR_PIPELINE_MOCK env var (default "1"):

- mock (default): Phase A is computed by `server.mock_pipeline` — TF-free
  rule-based body shape, a canonical standing pose, real face/skin-tone
  color matching, and the programmatic paper-doll renderer. Lets the API and
  the mobile app run end-to-end on a machine without TensorFlow or trained
  model artifacts.
- real (AVATAR_PIPELINE_MOCK=0): Phase A loads the trained Models 1/3/4 +
  MoveNet via `controller.load_pipeline_models()` (requires TensorFlow and
  `saved_models/`, see avatar_pipeline/README.md).

Clothing:
- Catalog garments (legacy, kept for existing items): `GET /api/garments` +
  `POST /api/avatars/<id>/wear` (see `avatar_pipeline.model6_body3d.
  garment_mesh`) — garments are cut from the same MakeHuman base mesh as the
  body and share its morph weights + height, so they track the avatar's
  body shape without any skeleton/rig. No TensorFlow, no external API, no
  mock mode.
- Adaptive garment fitting (Model 7, the primary flow — user-uploaded
  garment photos instead of a fixed catalog): `POST
  /api/avatars/<id>/fit-garment` + `GET
  /api/avatars/<id>/fitted-garment/<fit_id>.glb` (see
  `avatar_pipeline.model7_garment_fitting`). Runs background removal ->
  segmentation -> Unique3D image-to-3D garment mesh generation (see
  `UNIQUE3D_ENABLED`/`UNIQUE3D_ENDPOINT`) -> front/back texture projection
  -> avatar-to-garment region ratios -> Blender-based (or mock, see
  `GARMENT_FIT_MOCK`) region-wise fitting of the *actual generated garment
  mesh* — never a generic template. The response's `is_mock` flag is `true`
  whenever Unique3D/Blender aren't both configured, so the mobile app never
  shows a placeholder as the final result.

Model 5 (`avatar_pipeline/model5_tryon`, 2D TPS warp) and the AI try-on +
image-to-3D pipeline (`/api/ai-tryon/...`, see `server/ai_tryon/` — Gemini 2D
try-on image + a pluggable image-to-3D provider) are both superseded by the
garment endpoints above and unused by the mobile app now, but left in place.

Run:
    python -m server.app                          # mock mode, :5000
    AVATAR_PIPELINE_MOCK=0 python -m server.app    # real models (needs TF)
"""

import base64
import io
import os
import re
import time
from pathlib import Path

# Load server/.env (or repo-root .env) into os.environ BEFORE any of the
# below imports run — several of them (gemini_client, tryon_providers,
# model7_garment_fitting) read provider config (GEMINI_API_KEY,
# REPLICATE_API_TOKEN, VIRTUAL_TRYON_PROVIDER, ...) from os.environ at
# *import* time, so this must come first. No-op (never raises) if
# python-dotenv isn't installed or no .env file exists — env vars set the
# normal way (shell/CI) keep working either way.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image, UnidentifiedImageError

from backend.avatar_pipeline.model6_body3d.face_customization import apply_face_customization
from backend.avatar_pipeline.model6_body3d.garment_mesh import build_garment_glb, get_garment, list_garments
from backend.avatar_pipeline.model6_body3d.garment_texture_paint import (
    CATEGORY_V_BAND, paint_garment_onto_texture,
)
from backend.avatar_pipeline.model6_body3d.makehuman_mesh import (
    TEXTURE_SIZE, extract_base_color_texture, repaint_avatar_texture,
)
from backend.avatar_pipeline.model7_garment_fitting import (
    GARMENT_PIPELINE_MODE, PIPELINE_MODES, GarmentFittingError, run_garment_fitting,
)
from backend.avatar_pipeline.model7_garment_fitting.multiview.pipeline import (
    run_multiview_tryon_fitting, run_multiview_tryon_fitting_top_and_bottom,
)
from backend import garment_fit_storage, storage, users
from backend.ai_tryon import ai_session_storage, gemini_client, image_to_3d
from backend.ai_tryon.gemini_client import generate_tryon_image
from backend.ai_tryon.image_to_3d import get_provider

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# How long POST /api/ai-tryon/<id>/avatar3d will poll a processing
# image-to-3D job before giving up (MockImage23DProvider is always "ready"
# immediately, so this only matters for real providers, which commonly take
# a few minutes - override with IMAGE_TO_3D_TIMEOUT_S if needed).
IMAGE_TO_3D_TIMEOUT_S = int(os.environ.get("IMAGE_TO_3D_TIMEOUT_S", "300"))
IMAGE_TO_3D_POLL_INTERVAL_S = 2

MOCK = os.environ.get("AVATAR_PIPELINE_MOCK", "1") != "0"

if MOCK:
    from backend.mock_pipeline import build_avatar as _build_avatar

    def build_avatar(photo, bust, waist, hips, height):
        return _build_avatar(photo, bust, waist, hips, height)
else:
    from backend.avatar_pipeline.controller import build_avatar as _build_avatar, load_pipeline_models

    _models = load_pipeline_models(os.environ.get("AVATAR_SAVED_MODELS_DIR", "saved_models"))

    def build_avatar(photo, bust, waist, hips, height):
        return _build_avatar(_models, photo, bust, waist, hips, height)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _image_to_data_uri(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _read_image(file_storage, mode):
    try:
        return np.array(Image.open(file_storage.stream).convert(mode))
    except UnidentifiedImageError:
        return None


def _read_image_pil(file_storage):
    try:
        return Image.open(file_storage.stream).convert("RGB")
    except UnidentifiedImageError:
        return None


def _image_to_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _parse_measurements(form):
    values = {}
    for key in ("bust", "waist", "hips", "height"):
        raw = form.get(key)
        if raw is None:
            raise ValueError(f"'{key}' is required")
        try:
            values[key] = float(raw)
        except ValueError:
            raise ValueError(f"'{key}' must be a number")
        if values[key] <= 0:
            raise ValueError(f"'{key}' must be positive")
    return values


_body_shape_artifacts = None


def _get_body_shape_artifacts():
    """Lazily loads Model 1's trained artifacts on first use. Deliberately
    separate from the AVATAR_PIPELINE_MOCK/load_pipeline_models() machinery
    above: that loads Models 1+3+4+MoveNet together and requires all of their
    saved_models/ artifacts to exist, but only Model 1 is actually trained
    today. Importing tensorflow lazily (inside this function, not at module
    load time) preserves mock mode's "runs without TensorFlow installed"
    promise for the rest of this file - this route is the only one that
    requires it, and only once actually called."""
    global _body_shape_artifacts
    if _body_shape_artifacts is None:
        from backend.avatar_pipeline.model1_body_shape.predict import load_artifacts
        _body_shape_artifacts = load_artifacts(
            os.environ.get("BODY_SHAPE_MODEL_DIR", "saved_models/model1_body_shape")
        )
    return _body_shape_artifacts


@app.route("/api/predict-body-shape", methods=["POST"])
def predict_body_shape_route():
    """Runs Model 1's trained measurement-only MLP directly on
    bust/waist/hips/height - independent of AVATAR_PIPELINE_MOCK, and with no
    dependency on Models 3/4/6 (whose artifacts don't exist yet). Body:
    {"bust": ..., "waist": ..., "hips": ..., "height": ...} (JSON, cm).
    """
    from backend.avatar_pipeline.model1_body_shape.predict import predict_body_shape

    body = request.get_json(silent=True) or {}
    try:
        measurements = _parse_measurements(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model, scaler, config = _get_body_shape_artifacts()
    result = predict_body_shape(model_dir=None, model=model, scaler=scaler, config=config, **measurements)
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "mock": MOCK})


@app.route("/api/ai-tryon/config")
def ai_tryon_config():
    """Exposes the AI try-on pipeline's runtime config so the mobile app can
    show *why* it's getting mock output instead of a real Gemini image or
    Meshy avatar. Never exposes GEMINI_API_KEY / IMAGE_TO_3D_API_KEY values -
    only whether they're set.
    """
    return jsonify({
        "ai_tryon_mock": gemini_client.MOCK,
        "gemini_model": gemini_client.GEMINI_MODEL,
        "gemini_api_key_present": gemini_client.GEMINI_API_KEY is not None,
        "image_to_3d_provider": image_to_3d.PROVIDER,
        "image_to_3d_timeout_s": IMAGE_TO_3D_TIMEOUT_S,
        "image_to_3d_api_key_present": image_to_3d.IMAGE_TO_3D_API_KEY is not None,
    })


@app.route("/api/users/email", methods=["POST"])
def submit_email():
    body = request.get_json(silent=True)
    email = (body or {}).get("email")
    if not isinstance(email, str) or not EMAIL_PATTERN.match(email.strip()):
        return jsonify({"error": "'email' must be a valid email address"}), 400

    users.save_email(email.strip())

    return jsonify({"status": "ok"})


@app.route("/api/avatars", methods=["POST"])
def create_avatar():
    photo_file = request.files.get("photo")
    if photo_file is None:
        return jsonify({"error": "'photo' file is required"}), 400

    try:
        measurements = _parse_measurements(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    photo = _read_image(photo_file, "RGB")
    if photo is None:
        return jsonify({"error": "'photo' could not be read as an image"}), 400

    try:
        avatar_result = build_avatar(photo, **measurements)
    except ValueError as exc:
        # e.g. Model 3 found no face in the photo (real mode)
        return jsonify({"error": str(exc)}), 422

    avatar_id = storage.create(avatar_result)

    return jsonify({
        "avatar_id": avatar_id,
        "avatar_image": _image_to_data_uri(avatar_result.avatar_rgba),
        "body_shape": avatar_result.body_shape,
        "body_shape_confidence": avatar_result.body_shape_confidence,
        "skin_tone": avatar_result.skin_tone_result,
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "body3d_params": avatar_result.body3d_params,
        "gender": avatar_result.gender,
        "facial_analysis": avatar_result.facial_analysis,
    })


@app.route("/api/avatars/<avatar_id>", methods=["GET"])
def get_avatar(avatar_id):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    return jsonify({
        "avatar_id": avatar_id,
        "avatar_image": _image_to_data_uri(avatar_result.avatar_rgba),
        "body_shape": avatar_result.body_shape,
        "body_shape_confidence": avatar_result.body_shape_confidence,
        "skin_tone": avatar_result.skin_tone_result,
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "body3d_params": avatar_result.body3d_params,
        "gender": avatar_result.gender,
        "facial_analysis": avatar_result.facial_analysis,
    })


@app.route("/api/avatars/<avatar_id>/customize-face", methods=["POST"])
def customize_face(avatar_id):
    """Single-image or multi-angle face customization.

    Body (single-image mode — existing behaviour):
        { "faceImage": "<base64 front face>", ... }

    Body (multi-angle mode — front + left + right profiles):
        { "faceImage": "<base64 front>", "leftFaceImage": "<base64 left>",
          "rightFaceImage": "<base64 right>", ... }

    When ``leftFaceImage`` AND ``rightFaceImage`` are provided, uses the
    multi-angle projective texture pipeline (``multi_angle_texture``) to
    create a composite UV texture from all 3 angles — producing a 360°-
    natural face texture.  Otherwise falls back to single-image Delaunay
    warping.
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    features = request.get_json(silent=True)
    if not features:
        return jsonify({"error": "JSON body required"}), 400

    from backend.avatar_pipeline.model6_body3d.face_features import estimate_face_landmarks

    # ── Decode front face image ─────────────────────────────────────────
    selfie_rgb = None
    landmarks_2d = None
    face_image_b64 = features.get("faceImage")
    if face_image_b64:
        try:
            raw = base64.b64decode(face_image_b64, validate=True)
            selfie_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            landmarks_2d = estimate_face_landmarks(selfie_rgb)
        except Exception:
            pass

    # ── Decode left/right face images (multi-angle mode) ────────────────
    left_rgb = None
    right_rgb = None
    left_landmarks = None
    right_landmarks = None

    left_b64 = features.get("leftFaceImage")
    if left_b64:
        try:
            raw = base64.b64decode(left_b64, validate=True)
            left_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            left_landmarks = estimate_face_landmarks(left_rgb)
        except Exception:
            pass

    right_b64 = features.get("rightFaceImage")
    if right_b64:
        try:
            raw = base64.b64decode(right_b64, validate=True)
            right_rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            right_landmarks = estimate_face_landmarks(right_rgb)
        except Exception:
            pass

    try:
        gender_override = features.get("gender") if features.get("gender") in ("male", "female") else None
        updated = apply_face_customization(
            avatar_result, features,
            selfie_rgb=selfie_rgb,
            landmarks_2d=landmarks_2d,
            blend_mode="feather",
            gender_override=gender_override,
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_landmarks=left_landmarks,
            right_landmarks=right_landmarks,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    storage.update(avatar_id, updated)

    return jsonify({
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "face_texture_url": f"/api/avatars/{avatar_id}/face-texture.png",
    })


@app.route("/api/avatars/<avatar_id>/face-texture.png", methods=["GET"])
def get_avatar_face_texture(avatar_id):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    try:
        png_bytes = extract_base_color_texture(avatar_result.avatar_mesh_glb)
        if png_bytes is not None:
            return Response(png_bytes, mimetype="image/png")
    except Exception:
        pass
    return jsonify({"error": "no face texture in this avatar"}), 404


@app.route("/api/avatars/<avatar_id>/wear-photo", methods=["POST"])
def wear_photo(avatar_id):
    """Paints a user-supplied clothing/fabric photo directly onto the
    avatar's own body texture (see `garment_texture_paint.py`), instead of
    cutting a separate garment mesh (`garment_mesh.build_garment_glb`) — for
    a real, topologically-complex body mesh (e.g. a Renderpeople scan), a
    band-cut mesh doesn't track the actual surface and floats as an offset
    blob, while a texture decal follows the real geometry exactly since it's
    painted onto that same geometry.

    Body: multipart form with `category` (upper_body | lower_body | dress)
    plus EITHER `garment` (a user-supplied PNG/JPEG file) OR `garment_id`
    (an id from `GET /api/garments`, for the quick-pick catalog swatches).
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    category = request.form.get("category")
    if category not in CATEGORY_V_BAND:
        return jsonify({"error": f"'category' must be one of {sorted(CATEGORY_V_BAND)}"}), 400

    garment_file = request.files.get("garment")
    garment_id = request.form.get("garment_id")
    if garment_file is not None:
        garment_image = _read_image_pil(garment_file)
        if garment_image is None:
            return jsonify({"error": "could not read 'garment' as an image"}), 400
        garment_png = _image_to_png_bytes(garment_image)
    elif garment_id:
        # NOT _resolve_catalog_garment_png: those PNGs are flat garment-
        # silhouette cutouts (collar/sleeve shape included) authored for the
        # old flat front-facing cut-mesh, not for cylindrical wrapping - a
        # shape like that stretched around the body produces a warped block
        # instead of a shirt (see garment_texture_paint.py's module
        # docstring). The catalog names are plain colour choices ("Navy
        # T-Shirt", "Black T-Shirt"), so paint a solid fill of the catalog's
        # own colour_hex instead - solid colour is shape-agnostic and always
        # wraps cleanly regardless of the projection.
        garment_def = get_garment(garment_id)
        if garment_def is None:
            return jsonify({"error": f"unknown garment_id {garment_id!r}"}), 404
        rgb = tuple(int(garment_def.color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        solid = Image.new("RGB", (64, 64), rgb)
        buf = io.BytesIO()
        solid.save(buf, format="PNG")
        garment_png = buf.getvalue()
    else:
        return jsonify({"error": "either 'garment' (file) or 'garment_id' is required"}), 400

    current_texture_png = extract_base_color_texture(avatar_result.avatar_mesh_glb)
    if current_texture_png is None:
        # No baked texture yet (avatar was created but customize-face was
        # never called) - start from a flat skin-tone canvas.
        skin_hex = (avatar_result.skin_tone_result or {}).get("hex", "#c68863")
        skin_rgb = tuple(int(skin_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        flat = Image.new("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), skin_rgb)
        buf = io.BytesIO()
        flat.save(buf, format="PNG")
        current_texture_png = buf.getvalue()

    new_texture_png = paint_garment_onto_texture(current_texture_png, garment_png, category, gender=avatar_result.gender)
    avatar_result.avatar_mesh_glb = repaint_avatar_texture(avatar_result.avatar_mesh_glb, new_texture_png)
    storage.update(avatar_id, avatar_result)

    return jsonify({
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "face_texture_url": f"/api/avatars/{avatar_id}/face-texture.png",
    })


@app.route("/api/avatars/<avatar_id>/remove-photo", methods=["POST"])
def remove_photo(avatar_id):
    """Best-effort undo for `wear_photo`: repaints one category's UV band
    back to a flat skin-tone colour. Not a true undo (any garment
    previously painted there, or geometric seams baked into the source
    body scan itself, aren't recovered) - just clears the painted-on look.

    Body: multipart/form or JSON with `category` (upper_body | lower_body | dress).
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    category = (request.form.get("category") or (request.get_json(silent=True) or {}).get("category"))
    if category not in CATEGORY_V_BAND:
        return jsonify({"error": f"'category' must be one of {sorted(CATEGORY_V_BAND)}"}), 400

    current_texture_png = extract_base_color_texture(avatar_result.avatar_mesh_glb)
    if current_texture_png is None:
        return jsonify({"error": "avatar has no baked texture yet"}), 404

    skin_hex = (avatar_result.skin_tone_result or {}).get("hex", "#c68863")
    skin_rgb = tuple(int(skin_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    flat = Image.new("RGB", (64, 64), skin_rgb)
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    skin_png = buf.getvalue()

    new_texture_png = paint_garment_onto_texture(current_texture_png, skin_png, category, gender=avatar_result.gender)
    avatar_result.avatar_mesh_glb = repaint_avatar_texture(avatar_result.avatar_mesh_glb, new_texture_png)
    storage.update(avatar_id, avatar_result)

    return jsonify({
        "avatar_mesh_url": f"/api/avatars/{avatar_id}/mesh.glb",
        "face_texture_url": f"/api/avatars/{avatar_id}/face-texture.png",
    })


@app.route("/api/avatars/<avatar_id>/mesh.glb", methods=["GET"])
def get_avatar_mesh(avatar_id):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    return Response(avatar_result.avatar_mesh_glb, mimetype="model/gltf-binary")


@app.route("/api/garments", methods=["GET"])
def get_garments():
    """Lists the 3D garment catalog (`garment_mesh._GARMENT_CATALOG`) — each
    item's id/name/category/color, for the mobile wardrobe screen."""
    return jsonify({"garments": list_garments()})


@app.route("/api/avatars/<avatar_id>/wear", methods=["POST"])
def wear_garment(avatar_id):
    """Builds a garment `.glb` shaped to this avatar's body (same morph
    weights + height as `avatar_mesh_glb`, see `garment_mesh.build_garment_glb`)
    and returns a URL to fetch it. Stateless: the garment isn't stored on the
    avatar, it's rebuilt from `avatar_result.body3d_params` on every request/
    GET of the URL.

    Body: {"garment_id": "<id from GET /api/garments>"}
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    body = request.get_json(silent=True) or {}
    garment_id = body.get("garment_id")
    if not garment_id:
        return jsonify({"error": "'garment_id' is required"}), 400

    face_shape = (avatar_result.facial_analysis or {}).get("face_shape", "oval")
    garment_glb = build_garment_glb(
        garment_id, avatar_result.body3d_params, face_shape, gender=avatar_result.gender,
    )
    if garment_glb is None:
        return jsonify({"error": f"unknown garment_id {garment_id!r}"}), 404

    return jsonify({
        "garment_mesh_url": f"/api/avatars/{avatar_id}/garment.glb?garment_id={garment_id}",
    })


@app.route("/api/avatars/<avatar_id>/garment.glb", methods=["GET"])
def get_avatar_garment(avatar_id):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    garment_id = request.args.get("garment_id")
    if not garment_id:
        return jsonify({"error": "'garment_id' query param is required"}), 400

    face_shape = (avatar_result.facial_analysis or {}).get("face_shape", "oval")
    garment_glb = build_garment_glb(
        garment_id, avatar_result.body3d_params, face_shape, gender=avatar_result.gender,
    )
    if garment_glb is None:
        return jsonify({"error": f"unknown garment_id {garment_id!r}"}), 404

    return Response(garment_glb, mimetype="model/gltf-binary")


def _resolve_catalog_garment_png(garment_id: str) -> bytes | None:
    """The catalog garment's texture image bytes for `garment_id`: the
    pre-generated photo in mobile/assets/clothing/<garment_id>.png when
    available, falling back to a flat solid-colour PNG from the garment's
    catalog color_hex. Returns None if `garment_id` isn't in the catalog.
    Shared by `get_avatar_garment_texture` (separate-mesh path) and
    `wear_photo`'s `garment_id` mode (texture-paint path).
    """
    garment_def = get_garment(garment_id)
    if garment_def is None:
        return None

    REPO_ROOT = Path(__file__).resolve().parents[1]
    asset_path = REPO_ROOT / "mobile_app" / "assets" / "clothing" / f"{garment_id}.png"
    if asset_path.exists():
        return asset_path.read_bytes()

    h = garment_def.color_hex.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    solid = Image.new("RGB", (256, 256), rgb)
    buf = io.BytesIO()
    solid.save(buf, format="PNG")
    return buf.getvalue()


@app.route("/api/avatars/<avatar_id>/garment-texture.png", methods=["GET"])
def get_avatar_garment_texture(avatar_id):
    """Serves the catalog garment's texture image PNG for the given garment_id.

    Lets AvatarViewer3D apply a real image texture to the 3D garment mesh
    (TEXCOORD_0 is embedded in the GLB, generated by
    _compute_cylindrical_uvs in garment_mesh.py).
    """
    garment_id = request.args.get("garment_id")
    if not garment_id:
        return jsonify({"error": "'garment_id' query param is required"}), 400

    png_bytes = _resolve_catalog_garment_png(garment_id)
    if png_bytes is None:
        return jsonify({"error": f"unknown garment_id {garment_id!r}"}), 404
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/avatars/<avatar_id>/fit-garment", methods=["POST"])
def fit_garment(avatar_id):
    """Adaptive garment fitting (Model 7, replaces the catalog `/wear` flow
    for new garments): runs `avatar_pipeline.model7_garment_fitting.
    run_garment_fitting` on a user-uploaded front + back garment photo pair,
    fitted to this avatar's `body3d_params`.

    multipart/form-data: garment_front (file), garment_back (file),
    garment_type ("dress" | "upper_body" | "lower_body"), optional
    pipeline_mode ("adaptive_template" | "multiview_tryon", default
    GARMENT_PIPELINE_MODE).

    EXPERIMENTAL: when pipeline_mode=multiview_tryon, also requires
    person_front (a full-body photo of the avatar's own user); person_back
    is optional — if omitted, a back view is auto-generated from
    person_front via Gemini (see `gemini_client.generate_back_view_image`).
    Runs `avatar_pipeline.model7_garment_fitting.multiview.pipeline.
    run_multiview_tryon_fitting` instead — see that module's docstring.

    EXPERIMENTAL, further: also within pipeline_mode=multiview_tryon, an
    optional garment_mode field ("dress" [default] | "top_and_bottom")
    selects between one garment (garment_front/garment_back, any
    garment_type) and a separate top + bottom outfit (top_front/top_back +
    bottom_front/bottom_back, garment_type/garment_front/garment_back
    ignored) — see `run_multiview_tryon_fitting_top_and_bottom`.

    The default (pipeline_mode omitted, or explicitly adaptive_template) is
    completely unchanged from the existing behavior.
    """
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    front_file = request.files.get("garment_front")
    back_file = request.files.get("garment_back")
    garment_type = request.form.get("garment_type")
    pipeline_mode = request.form.get("pipeline_mode") or GARMENT_PIPELINE_MODE

    if pipeline_mode not in PIPELINE_MODES:
        return jsonify({"error": f"'pipeline_mode' must be one of {PIPELINE_MODES}, got {pipeline_mode!r}"}), 400

    if pipeline_mode == "multiview_tryon":
        person_front_file = request.files.get("person_front")
        person_back_file = request.files.get("person_back")
        garment_mode = request.form.get("garment_mode", "dress")

        try:
            if garment_mode == "top_and_bottom":
                multiview_result = run_multiview_tryon_fitting_top_and_bottom(
                    person_front_file, person_back_file,
                    request.files.get("top_front"), request.files.get("top_back"),
                    request.files.get("bottom_front"), request.files.get("bottom_back"),
                )
            elif garment_mode == "dress":
                multiview_result = run_multiview_tryon_fitting(
                    person_front_file, person_back_file, front_file, back_file, garment_type,
                )
            else:
                return jsonify({"error": f"'garment_mode' must be 'dress' or 'top_and_bottom', got {garment_mode!r}"}), 400
        except GarmentFittingError as exc:
            return jsonify({"error": str(exc)}), 400

        garment_fit_storage.save(
            multiview_result.glb_bytes, texture_png=multiview_result.texture_png, fit_id=multiview_result.fit_id,
        )
        garment_fit_storage.save_tryon_previews(
            multiview_result.fit_id, multiview_result.tryon_front_png, multiview_result.tryon_back_png,
        )

        return jsonify({
            "fit_id": multiview_result.fit_id,
            "garment_features": multiview_result.features.as_dict(),
            # PIVOT: no longer computed — this mesh replaces the avatar
            # rather than being fitted onto it (see is_full_avatar_replacement).
            "region_scales": multiview_result.region_scales.as_dict() if multiview_result.region_scales else None,
            # In multiview_tryon mode this URL is a full replacement avatar
            # mesh (Unique3D reconstruction), not a garment overlay.
            "garment_mesh_url": f"/api/avatars/{avatar_id}/fitted-garment/{multiview_result.fit_id}.glb",
            "garment_texture_url": (
                f"/api/avatars/{avatar_id}/fitted-garment/{multiview_result.fit_id}-texture.png"
                if multiview_result.texture_png is not None else None
            ),
            "garment_tryon_front_url":
                f"/api/avatars/{avatar_id}/fitted-garment/{multiview_result.fit_id}-tryon-front.png",
            "garment_tryon_back_url":
                f"/api/avatars/{avatar_id}/fitted-garment/{multiview_result.fit_id}-tryon-back.png",
            "status": multiview_result.status,
            "warnings": multiview_result.warnings,
            "is_mock": multiview_result.is_mock,
            # EXPERIMENTAL pipeline metadata — never claim real AI processing
            # ran when a mock provider was used at any stage.
            "pipeline_mode": "multiview_tryon",
            "virtual_tryon_provider": multiview_result.virtual_tryon_provider,
            "image_to_3d_provider": multiview_result.image_to_3d_provider,
            "texture_provider": multiview_result.texture_provider,
            "is_real_3d_generation": multiview_result.is_real_3d_generation,
            "is_full_avatar_replacement": multiview_result.is_full_avatar_replacement,
        })

    try:
        result = run_garment_fitting(
            front_file, back_file, garment_type,
            avatar_result.body3d_params, gender=avatar_result.gender,
            avatar_mesh_glb=avatar_result.avatar_mesh_glb,
        )
    except GarmentFittingError as exc:
        return jsonify({"error": str(exc)}), 400

    garment_fit_storage.save(result.glb_bytes, texture_png=result.texture_png, fit_id=result.fit_id)

    return jsonify({
        "fit_id": result.fit_id,
        "garment_features": result.features.as_dict(),
        "region_scales": result.region_scales.as_dict(),
        "garment_mesh_url": f"/api/avatars/{avatar_id}/fitted-garment/{result.fit_id}.glb",
        "garment_texture_url": (
            f"/api/avatars/{avatar_id}/fitted-garment/{result.fit_id}-texture.png"
            if result.texture_png is not None else None
        ),
        "status": result.status,
        "warnings": result.warnings,
        # True unless a real Unique3D-generated mesh went through the real
        # Blender fitting backend — see garment_mesh_generation.py /
        # garment_fit_runner.py. The mobile app must show this distinctly,
        # never present a mock result as the final fitted garment.
        "is_mock": result.is_mock,
        "pipeline_mode": "adaptive_template",
    })


@app.route("/api/avatars/<avatar_id>/fitted-garment/<fit_id>.glb", methods=["GET"])
def get_fitted_garment(avatar_id, fit_id):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    glb_bytes = garment_fit_storage.get(fit_id)
    if glb_bytes is None:
        return jsonify({"error": "unknown fit_id"}), 404

    return Response(glb_bytes, mimetype="model/gltf-binary")


@app.route("/api/avatars/<avatar_id>/fitted-garment/<fit_id>-texture.png", methods=["GET"])
def get_fitted_garment_texture(avatar_id, fit_id):
    """Serves the fitted garment's texture atlas (front photo on top half,
    back photo on bottom half — see `blender_runner.build_garment_texture_atlas`)
    out-of-band, for clients (React Native) that can't decode a GLB's
    embedded bufferView image — same pattern as `/face-texture.png`."""
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    texture_png = garment_fit_storage.get_texture(fit_id)
    if texture_png is None:
        return jsonify({"error": "unknown fit_id, or it has no texture"}), 404

    return Response(texture_png, mimetype="image/png")


@app.route("/api/avatars/<avatar_id>/fitted-garment/<fit_id>-tryon-front.png", methods=["GET"])
def get_fitted_garment_tryon_front(avatar_id, fit_id):
    """EXPERIMENTAL — serves the front virtual-try-on preview image produced
    by `pipeline_mode=multiview_tryon` (see `multiview/pipeline.py`)."""
    return _get_tryon_preview(avatar_id, fit_id, "front")


@app.route("/api/avatars/<avatar_id>/fitted-garment/<fit_id>-tryon-back.png", methods=["GET"])
def get_fitted_garment_tryon_back(avatar_id, fit_id):
    """EXPERIMENTAL — serves the back virtual-try-on preview image produced
    by `pipeline_mode=multiview_tryon` (see `multiview/pipeline.py`)."""
    return _get_tryon_preview(avatar_id, fit_id, "back")


def _get_tryon_preview(avatar_id, fit_id, side):
    avatar_result = storage.get(avatar_id)
    if avatar_result is None:
        return jsonify({"error": "unknown avatar_id"}), 404

    preview_png = garment_fit_storage.get_tryon_preview(fit_id, side)
    if preview_png is None:
        return jsonify({"error": "unknown fit_id, or it has no try-on preview"}), 404

    return Response(preview_png, mimetype="image/png")


@app.route("/api/ai-tryon", methods=["POST"])
def create_ai_tryon():
    person_file = request.files.get("person_photo")
    clothing_files = request.files.getlist("clothing_photo")
    if person_file is None:
        return jsonify({"error": "'person_photo' file is required"}), 400
    if not clothing_files:
        return jsonify({"error": "at least one 'clothing_photo' file is required"}), 400

    person_image = _read_image_pil(person_file)
    if person_image is None:
        return jsonify({"error": "'person_photo' could not be read as an image"}), 400

    clothing_images = []
    clothing_names = []
    for index, clothing_file in enumerate(clothing_files):
        clothing_image = _read_image_pil(clothing_file)
        if clothing_image is None:
            return jsonify({"error": f"'clothing_photo' #{index + 1} could not be read as an image"}), 400
        clothing_images.append(_image_to_png_bytes(clothing_image))
        clothing_names.append(clothing_file.filename or f"clothing_{index}.jpg")

    try:
        generated_bytes = generate_tryon_image(
            _image_to_png_bytes(person_image), clothing_images
        )
        generated_image = Image.open(io.BytesIO(generated_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"AI try-on generation failed: {exc}"}), 502

    tryon_id = ai_session_storage.create_session(
        _image_to_png_bytes(generated_image), clothing_photos=clothing_names
    )

    return jsonify({
        "tryon_id": tryon_id,
        "generated_image_url": f"/api/ai-tryon/{tryon_id}/image.png",
    })


@app.route("/api/ai-tryon/<tryon_id>/image.png", methods=["GET"])
def get_ai_tryon_image(tryon_id):
    image_bytes = ai_session_storage.get_generated_image(tryon_id)
    if image_bytes is None:
        return jsonify({"error": "unknown tryon_id"}), 404

    return Response(image_bytes, mimetype="image/png")


@app.route("/api/ai-tryon/<tryon_id>/avatar3d", methods=["POST"])
def create_ai_avatar_3d(tryon_id):
    generated_image = ai_session_storage.get_generated_image(tryon_id)
    if generated_image is None:
        return jsonify({"error": "unknown tryon_id"}), 404

    try:
        provider = get_provider()
        job_id = provider.create_job(generated_image)

        deadline = time.monotonic() + IMAGE_TO_3D_TIMEOUT_S
        status = provider.get_job_status(job_id)
        while status == "processing":
            if time.monotonic() > deadline:
                return jsonify({"error": "image-to-3D generation timed out"}), 504
            time.sleep(IMAGE_TO_3D_POLL_INTERVAL_S)
            status = provider.get_job_status(job_id)

        if status != "ready":
            return jsonify({"error": "image-to-3D generation failed"}), 502

        glb_bytes = provider.get_result_glb(job_id)
    except Exception as exc:
        return jsonify({"error": f"image-to-3D generation failed: {exc}"}), 502

    ai_session_storage.save_avatar_glb(tryon_id, glb_bytes)

    return jsonify({"avatar_mesh_url": f"/api/ai-tryon/{tryon_id}/avatar.glb"})


@app.route("/api/ai-tryon/<tryon_id>/avatar.glb", methods=["GET"])
def get_ai_avatar_glb(tryon_id):
    glb_bytes = ai_session_storage.get_avatar_glb(tryon_id)
    if glb_bytes is None:
        return jsonify({"error": "unknown tryon_id, or its 3D avatar hasn't been generated yet"}), 404

    return Response(glb_bytes, mimetype="model/gltf-binary")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
