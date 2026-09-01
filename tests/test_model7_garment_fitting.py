"""Tests for Model 7 (adaptive garment fitting):
`avatar_pipeline.model7_garment_fitting` — invalid-upload handling,
normalized feature extraction ranges, region-scale calculation, the mock
Blender runner, and the `POST /api/avatars/<id>/fit-garment` /
`GET /api/avatars/<id>/fitted-garment/<fit_id>.glb` endpoints.

Runs with GARMENT_FITTING_MOCK=1 (default) — no Blender install needed.
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.avatar_pipeline.model6_body3d.params import default_params_from_measurements
from backend.avatar_pipeline.model7_garment_fitting.fitting_types import (
    FEATURE_NAMES, GarmentFittingError, RegionScales,
)
from backend.avatar_pipeline.model7_garment_fitting.garment_features import compute_normalized_features
from backend.avatar_pipeline.model7_garment_fitting.garment_fit_runner import MockGarmentFitRunner
from backend.avatar_pipeline.model7_garment_fitting.garment_keypoints import extract_keypoints
from backend.avatar_pipeline.model7_garment_fitting.garment_mesh_generation import (
    GeneratedGarmentMesh, MockGarmentMeshProvider, Unique3DGarmentMeshProvider,
    fuse_front_back_meshes, project_front_back_texture, validate_garment_mesh,
)
from backend.avatar_pipeline.model7_garment_fitting.garment_region_fitting import (
    compute_region_fit_ratios, extract_avatar_region_landmarks,
)
from backend.avatar_pipeline.model7_garment_fitting.garment_segmentation import segment_garment
from backend.avatar_pipeline.model7_garment_fitting.garment_template import REFERENCE_BODY_PARAMS, get_template_features
from backend.avatar_pipeline.model7_garment_fitting.region_scaling import compute_region_scales


# ── Synthetic garment image generator ───────────────────────────────────
#
# Draws a simple t-shirt/dress/pants silhouette (flat gray polygon) on a
# plain white background, mimicking a laid-flat product photo. Good enough
# to exercise the deterministic OpenCV/silhouette pipeline end to end.

def _tshirt_silhouette(w=300, h=360):
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    cx = w / 2
    points = [
        (cx - 0.10 * w, 0.08 * h),   # left neck
        (cx - 0.32 * w, 0.14 * h),   # left shoulder
        (cx - 0.48 * w, 0.30 * h),   # left sleeve end
        (cx - 0.36 * w, 0.34 * h),   # left underarm
        (cx - 0.30 * w, 0.55 * h),   # left waist
        (cx - 0.34 * w, 0.90 * h),   # left hem
        (cx + 0.34 * w, 0.90 * h),   # right hem
        (cx + 0.30 * w, 0.55 * h),   # right waist
        (cx + 0.36 * w, 0.34 * h),   # right underarm
        (cx + 0.48 * w, 0.30 * h),   # right sleeve end
        (cx + 0.32 * w, 0.14 * h),   # right shoulder
        (cx + 0.10 * w, 0.08 * h),   # right neck
    ]
    draw.polygon(points, fill=(60, 90, 140))
    return np.array(img)


def _pants_silhouette(w=260, h=380):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    cx = w / 2
    draw.polygon([
        (cx - 0.34 * w, 0.03 * h), (cx + 0.34 * w, 0.03 * h),
        (cx + 0.38 * w, 0.30 * h), (cx + 0.20 * w, 0.30 * h),
        (cx + 0.22 * w, 0.95 * h), (cx + 0.06 * w, 0.95 * h),
        (cx, 0.32 * h),
        (cx - 0.06 * w, 0.95 * h), (cx - 0.22 * w, 0.95 * h),
        (cx - 0.20 * w, 0.30 * h), (cx - 0.38 * w, 0.30 * h),
    ], fill=(40, 40, 60))
    return np.array(img)


def _png_upload(rgb_array, name="garment.png"):
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    buf.seek(0)
    return buf, name


@pytest.fixture
def body3d_params():
    return default_params_from_measurements("Hourglass", bust=92, waist=70, hips=98, height=165, gender="female")


# ── Segmentation / keypoints / feature-range tests ──────────────────────

def test_segment_garment_rejects_blank_image():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    with pytest.raises(GarmentFittingError):
        segment_garment(blank, "front")


def test_segment_garment_finds_tshirt_silhouette():
    result = segment_garment(_tshirt_silhouette(), "front")
    assert result.mask.dtype == bool
    assert 0.05 < result.mask.mean() < 0.9
    x0, y0, x1, y1 = result.bbox
    assert x1 > x0 and y1 > y0


def test_normalized_feature_extraction_output_ranges(body3d_params):
    front_rgb, back_rgb = _tshirt_silhouette(), _tshirt_silhouette()
    front_seg = segment_garment(front_rgb, "front")
    back_seg = segment_garment(back_rgb, "back")
    front_kp = extract_keypoints(front_seg, "upper_body", "front")
    back_kp = extract_keypoints(back_seg, "upper_body", "back")

    features = compute_normalized_features(
        front_kp, back_kp, front_seg.bbox, back_seg.bbox, "upper_body",
    )

    feature_dict = features.as_dict()
    assert set(feature_dict.keys()) == set(FEATURE_NAMES)
    for name, value in feature_dict.items():
        assert isinstance(value, float), name
        assert 0.0 <= value <= 1.5, f"{name}={value} out of plausible normalized range"
    # A t-shirt has a real silhouette extent — length must be meaningfully > 0.
    assert feature_dict["garment_length"] > 0.1


def test_normalized_feature_extraction_lower_body_has_no_sleeve_or_neck(body3d_params):
    front_rgb, back_rgb = _pants_silhouette(), _pants_silhouette()
    front_seg = segment_garment(front_rgb, "front")
    back_seg = segment_garment(back_rgb, "back")
    front_kp = extract_keypoints(front_seg, "lower_body", "front")
    back_kp = extract_keypoints(back_seg, "lower_body", "back")

    features = compute_normalized_features(
        front_kp, back_kp, front_seg.bbox, back_seg.bbox, "lower_body",
    )
    assert features.sleeve_length == 0.0
    assert features.neck_width == 0.0
    assert features.hip_width > 0.0


# ── Region scaling tests ─────────────────────────────────────────────────

def test_region_scales_are_identity_for_template_garment_and_reference_body():
    template_features = get_template_features("upper_body")
    scales = compute_region_scales(template_features, REFERENCE_BODY_PARAMS, "upper_body")
    for value in scales.as_dict().values():
        assert value == pytest.approx(1.0, abs=1e-6)


def test_region_scales_increase_with_wider_garment_and_body(body3d_params):
    template_features = get_template_features("upper_body")
    baseline = compute_region_scales(template_features, REFERENCE_BODY_PARAMS, "upper_body")

    wider_body = dict(REFERENCE_BODY_PARAMS)
    wider_body["chest_width"] *= 1.3
    scaled = compute_region_scales(template_features, wider_body, "upper_body")

    assert scaled.chest_scale > baseline.chest_scale
    # Unrelated regions shouldn't move — this is region-wise, not global.
    assert scaled.waist_scale == pytest.approx(baseline.waist_scale)


def test_region_scales_are_clamped_to_safe_range():
    from backend.avatar_pipeline.model7_garment_fitting.fitting_types import NormalizedGarmentFeatures

    extreme_features = NormalizedGarmentFeatures(
        shoulder_width=1.5, chest_width=1.5, waist_width=1.5, hip_width=1.5,
        sleeve_length=1.5, garment_length=1.5, neck_width=1.5, hem_width=1.5,
    )
    tiny_body = {name: value * 0.01 for name, value in REFERENCE_BODY_PARAMS.items()}
    scales = compute_region_scales(extreme_features, tiny_body, "upper_body")
    for value in scales.as_dict().values():
        assert 0.6 <= value <= 1.6


# ── Garment mesh generation tests ────────────────────────────────────────

def _positions_and_uvs(glb_bytes):
    from pygltflib import GLTF2

    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = gltf.binary_blob()
    prim = gltf.meshes[0].primitives[0]

    pos_acc = gltf.accessors[prim.attributes.POSITION]
    pos_bv = gltf.bufferViews[pos_acc.bufferView]
    positions = np.frombuffer(
        blob[pos_bv.byteOffset: pos_bv.byteOffset + pos_bv.byteLength], dtype="<f4"
    ).reshape(-1, 3)

    uvs = None
    uv_index = getattr(prim.attributes, "TEXCOORD_0", None)
    if uv_index is not None:
        uv_acc = gltf.accessors[uv_index]
        uv_bv = gltf.bufferViews[uv_acc.bufferView]
        uvs = np.frombuffer(
            blob[uv_bv.byteOffset: uv_bv.byteOffset + uv_bv.byteLength], dtype="<f4"
        ).reshape(-1, 2)

    return positions, uvs


def test_mock_garment_mesh_provider_marks_output_as_mock(body3d_params):
    provider = MockGarmentMeshProvider()
    mesh = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    assert mesh.is_mock is True
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert set(mesh.landmarks.keys()) >= {
        "left_shoulder", "right_shoulder", "left_sleeve_end", "right_sleeve_end", "top_center", "bottom_center",
    }


def test_mock_garment_mesh_provider_rejects_unknown_garment_type():
    provider = MockGarmentMeshProvider()
    with pytest.raises(GarmentFittingError):
        provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "hat")


def test_unique3d_provider_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr("backend.avatar_pipeline.model7_garment_fitting.garment_mesh_generation.UNIQUE3D_ENABLED", False)
    provider = Unique3DGarmentMeshProvider(endpoint="http://example.invalid/generate")
    result = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    assert result is None


def test_unique3d_provider_returns_none_without_endpoint(monkeypatch):
    monkeypatch.setattr("backend.avatar_pipeline.model7_garment_fitting.garment_mesh_generation.UNIQUE3D_ENABLED", True)
    provider = Unique3DGarmentMeshProvider(endpoint=None)
    result = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    assert result is None


def test_project_front_back_texture_preserves_topology_and_colors():
    provider = MockGarmentMeshProvider()
    front = np.full((200, 200, 3), (200, 30, 30), dtype=np.uint8)
    back = np.full((200, 200, 3), (180, 20, 20), dtype=np.uint8)
    mesh = provider.generate(front, back, "upper_body")

    projected = project_front_back_texture(mesh, front, back)

    # Topology (vertex/face count) is untouched — only UVs/texture change.
    assert len(projected.vertices) == len(mesh.vertices)
    assert np.array_equal(projected.faces, mesh.faces)
    assert projected.uvs.shape == (len(mesh.vertices), 2)
    assert projected.texture_png is not None

    atlas = np.array(Image.open(io.BytesIO(projected.texture_png)).convert("RGB"))
    top_half_color = np.median(atlas[: atlas.shape[0] // 2].reshape(-1, 3), axis=0)
    bottom_half_color = np.median(atlas[atlas.shape[0] // 2:].reshape(-1, 3), axis=0)
    np.testing.assert_allclose(top_half_color, [200, 30, 30], atol=5)
    np.testing.assert_allclose(bottom_half_color, [180, 20, 20], atol=5)


def test_validate_garment_mesh_rejects_degenerate_mesh():
    degenerate = GeneratedGarmentMesh(
        vertices=np.zeros((2, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.uint32),
        uvs=np.zeros((2, 2), dtype=np.float32),
        texture_png=None,
        landmarks={name: (0.0, 0.0, 0.0) for name in
                   ("left_shoulder", "right_shoulder", "neck_center", "left_chest", "right_chest",
                    "left_waist", "right_waist", "left_hip", "right_hip", "left_sleeve_end",
                    "right_sleeve_end", "hem_center", "top_center", "bottom_center")},
        garment_type="upper_body", is_mock=True,
    )
    with pytest.raises(GarmentFittingError):
        validate_garment_mesh(degenerate)


def test_validate_garment_mesh_accepts_mock_mesh():
    provider = MockGarmentMeshProvider()
    mesh = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    validate_garment_mesh(mesh)  # should not raise


def test_fuse_front_back_meshes_incorporates_back_geometry():
    """The fused mesh's rear landmarks must come from the *back* mesh, not
    just be a copy of the front mesh's own landmarks — proving the back
    photo actually influenced the result instead of being silently dropped."""
    provider = MockGarmentMeshProvider()
    front_mesh = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")

    # A "back" mesh with the SAME landmarks (so alignment is a no-op: scale
    # 1.0, zero translation) but visibly different vertex positions, so any
    # detected movement at back-facing vertices can only have come from
    # actually incorporating this mesh's own geometry.
    back_mesh = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    rng = np.random.default_rng(0)
    perturbed_vertices = back_mesh.vertices + rng.uniform(0.2, 0.4, size=back_mesh.vertices.shape).astype(np.float32)
    back_mesh = GeneratedGarmentMesh(
        vertices=perturbed_vertices,
        faces=back_mesh.faces, uvs=back_mesh.uvs, texture_png=back_mesh.texture_png,
        landmarks=front_mesh.landmarks, garment_type=back_mesh.garment_type, is_mock=True,
    )

    fused = fuse_front_back_meshes(front_mesh, back_mesh)

    assert len(fused.vertices) == len(front_mesh.vertices)
    assert np.array_equal(fused.faces, front_mesh.faces)
    # Back-facing vertices should have moved away from the unfused front mesh.
    front_facing = front_mesh.vertices[:, 2] >= 0
    back_facing_idx = np.where(~front_facing)[0]
    if len(back_facing_idx) > 0:
        assert not np.allclose(fused.vertices[back_facing_idx], front_mesh.vertices[back_facing_idx])


# ── Avatar-to-garment region fitting tests ───────────────────────────────

def test_extract_avatar_region_landmarks_scales_with_body_params(body3d_params):
    narrow = dict(body3d_params, shoulder_width=body3d_params["shoulder_width"] * 0.8)
    wide = dict(body3d_params, shoulder_width=body3d_params["shoulder_width"] * 1.2)

    narrow_landmarks = extract_avatar_region_landmarks(narrow, height_cm=165)
    wide_landmarks = extract_avatar_region_landmarks(wide, height_cm=165)

    narrow_width = abs(narrow_landmarks.right_shoulder[0] - narrow_landmarks.left_shoulder[0])
    wide_width = abs(wide_landmarks.right_shoulder[0] - wide_landmarks.left_shoulder[0])
    assert wide_width > narrow_width


def test_extract_avatar_region_landmarks_requires_body_params():
    with pytest.raises(ValueError):
        extract_avatar_region_landmarks({}, height_cm=165)


def test_compute_region_fit_ratios_independent_per_region(body3d_params):
    avatar_landmarks = extract_avatar_region_landmarks(body3d_params, height_cm=165)
    provider = MockGarmentMeshProvider()
    mesh = provider.generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")

    ratios = compute_region_fit_ratios(avatar_landmarks, mesh.landmarks)
    for value in ratios.as_dict().values():
        assert 0.6 <= value <= 1.6

    # A wider avatar chest should raise chest_scale without moving waist_scale.
    wider = dict(body3d_params, chest_width=body3d_params["chest_width"] * 1.3)
    wider_landmarks = extract_avatar_region_landmarks(wider, height_cm=165)
    wider_ratios = compute_region_fit_ratios(wider_landmarks, mesh.landmarks)
    assert wider_ratios.chest_scale > ratios.chest_scale
    assert wider_ratios.waist_scale == pytest.approx(ratios.waist_scale)


# ── Garment fit runner tests ─────────────────────────────────────────────

def test_mock_garment_fit_runner_preserves_topology_and_texture(body3d_params):
    provider = MockGarmentMeshProvider()
    front, back = _tshirt_silhouette(), _tshirt_silhouette()
    mesh = provider.generate(front, back, "upper_body")
    mesh = project_front_back_texture(mesh, front, back)

    avatar_landmarks = extract_avatar_region_landmarks(body3d_params, height_cm=165)
    ratios = compute_region_fit_ratios(avatar_landmarks, mesh.landmarks)

    runner = MockGarmentFitRunner()
    glb_bytes, texture_png, used_blender = runner.fit(mesh, b"", avatar_landmarks, ratios)

    assert used_blender is False
    assert texture_png == mesh.texture_png
    positions, uvs = _positions_and_uvs(glb_bytes)
    assert len(positions) == len(mesh.vertices)
    assert uvs is not None and len(uvs) == len(mesh.vertices)


def test_mock_garment_fit_runner_deforms_toward_avatar_region_ratios(body3d_params):
    provider = MockGarmentMeshProvider()
    front, back = _tshirt_silhouette(), _tshirt_silhouette()
    mesh = provider.generate(front, back, "upper_body")

    avatar_landmarks = extract_avatar_region_landmarks(body3d_params, height_cm=165)
    identity_ratios = RegionScales(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    wide_ratios = RegionScales(1.0, 1.5, 1.0, 1.0, 1.0, 1.0)

    runner = MockGarmentFitRunner()
    base_bytes, _, _ = runner.fit(mesh, b"", avatar_landmarks, identity_ratios)
    wide_bytes, _, _ = runner.fit(mesh, b"", avatar_landmarks, wide_ratios)

    base_positions, _ = _positions_and_uvs(base_bytes)
    wide_positions, _ = _positions_and_uvs(wide_bytes)
    assert np.abs(wide_positions[:, [0, 2]]).max() > np.abs(base_positions[:, [0, 2]]).max()


# ── Flask endpoint tests ─────────────────────────────────────────────────

@pytest.fixture
def client():
    from backend.app import app
    return app.test_client()


@pytest.fixture
def avatar_id(client):
    photo = io.BytesIO()
    Image.new("RGB", (96, 96), (200, 180, 160)).save(photo, format="PNG")
    photo.seek(0)
    response = client.post(
        "/api/avatars",
        data={"photo": (photo, "photo.png"), "bust": "92", "waist": "70", "hips": "98", "height": "165"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["avatar_id"]


def test_fit_garment_missing_files_returns_400(client, avatar_id):
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={"garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "garment_front" in response.get_json()["error"]


def test_fit_garment_invalid_garment_type_returns_400(client, avatar_id):
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={
            "garment_front": front, "garment_back": back,
            "garment_type": "hat",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "garment_type" in response.get_json()["error"]


def test_fit_garment_unreadable_image_returns_400(client, avatar_id):
    bad_file = (io.BytesIO(b"not an image"), "bad.png")
    back = _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={"garment_front": bad_file, "garment_back": back, "garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_fit_garment_unknown_avatar_returns_404(client):
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        "/api/avatars/deadbeefdeadbeefdeadbeefdeadbeef/fit-garment",
        data={"garment_front": front, "garment_back": back, "garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 404


def test_fit_garment_end_to_end(client, avatar_id):
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={"garment_front": front, "garment_back": back, "garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()

    assert body["status"] == "ready"
    # UNIQUE3D_ENABLED is unset in tests -> mock mesh source, must be labelled.
    assert body["is_mock"] is True
    assert set(body["garment_features"].keys()) == set(FEATURE_NAMES)
    assert set(body["region_scales"].keys()) == {
        "shoulder_scale", "chest_scale", "waist_scale", "hip_scale", "sleeve_scale", "length_scale",
    }
    fit_id = body["fit_id"]
    assert body["garment_mesh_url"] == f"/api/avatars/{avatar_id}/fitted-garment/{fit_id}.glb"
    assert body["garment_texture_url"] == f"/api/avatars/{avatar_id}/fitted-garment/{fit_id}-texture.png"

    glb_response = client.get(body["garment_mesh_url"])
    assert glb_response.status_code == 200
    assert glb_response.mimetype == "model/gltf-binary"
    assert glb_response.data[:4] == b"glTF"

    texture_response = client.get(body["garment_texture_url"])
    assert texture_response.status_code == 200
    assert texture_response.mimetype == "image/png"
    assert texture_response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_fit_garment_returns_clear_error_when_mesh_generation_unavailable(client, avatar_id, monkeypatch):
    """When the configured provider can't produce a mesh at all (`generate()`
    returns None — e.g. Unique3D enabled but genuinely unreachable), the
    route must return a clear error, never silently substitute a mock."""
    from backend.avatar_pipeline.model7_garment_fitting import pipeline as pipeline_module

    class _UnavailableProvider:
        def generate(self, *args, **kwargs):
            return None

    monkeypatch.setattr(pipeline_module, "get_garment_mesh_provider", lambda: _UnavailableProvider())

    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={"garment_front": front, "garment_back": back, "garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "unavailable" in response.get_json()["error"]


def test_get_fitted_garment_unknown_fit_id_returns_404(client, avatar_id):
    response = client.get(f"/api/avatars/{avatar_id}/fitted-garment/deadbeefdeadbeefdeadbeefdeadbeef.glb")
    assert response.status_code == 404


def test_get_fitted_garment_texture_unknown_fit_id_returns_404(client, avatar_id):
    response = client.get(f"/api/avatars/{avatar_id}/fitted-garment/deadbeefdeadbeefdeadbeefdeadbeef-texture.png")
    assert response.status_code == 404
