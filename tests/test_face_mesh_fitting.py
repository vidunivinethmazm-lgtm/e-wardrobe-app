"""Tests for the optional Unique3D-based face fitting stage
(`avatar_pipeline.model6_body3d.face_mesh_generation` /
`face_mesh_fitting` / `face_fit_runner` / `face_fitting_pipeline`):
scale-ratio calculation, landmark alignment, mock face mesh generation,
fallback when no provider is configured, and preservation of the existing
avatar body parameters / geometry.

Runs with FACE_MESH_PROVIDER=mock and FACE_FIT_MOCK=1 (both defaults) — no
Unique3D endpoint or Blender install needed. Does not exercise or modify
`face_customization.apply_face_customization` (the existing, unrelated face
texture-transfer pipeline) at all.
"""

import copy

import numpy as np
import pytest

from avatar_pipeline.model6_body3d.face_fit_runner import MockFaceFitRunner
from avatar_pipeline.model6_body3d.face_fitting_pipeline import run_face_fitting
from avatar_pipeline.model6_body3d.face_mesh_fitting import (
    FaceLandmarks,
    align_face_mesh,
    compute_scale_ratios,
    extract_avatar_head_landmarks,
    landmarks_from_dict,
)
from avatar_pipeline.model6_body3d.face_mesh_generation import (
    GeneratedFaceMesh,
    LANDMARK_NAMES,
    MockFaceMeshProvider,
    Unique3DFaceMeshProvider,
)
from avatar_pipeline.model6_body3d.params import default_params_from_measurements

HEIGHT_CM = 165.0


@pytest.fixture
def body3d_params():
    return default_params_from_measurements("Hourglass", bust=92, waist=70, hips=98, height=HEIGHT_CM, gender="female")


@pytest.fixture
def avatar_result(body3d_params):
    from server.mock_pipeline import build_avatar

    photo = np.full((96, 96, 3), 200, dtype=np.uint8)
    return build_avatar(photo, bust=92, waist=70, hips=98, height=HEIGHT_CM)


def _uniform_face_landmarks(rx=0.35, ry=0.45, rz=0.30, center=(0.0, 0.0, 0.0)):
    """A synthetic FaceLandmarks with the same relative layout
    MockFaceMeshProvider uses, at an arbitrary scale/center — handy for
    isolating scale/alignment math from the provider itself."""
    cx, cy, cz = center
    return FaceLandmarks(
        eye_left=(cx - 0.40 * rx, cy + 0.05 * ry, cz + 0.85 * rz),
        eye_right=(cx + 0.40 * rx, cy + 0.05 * ry, cz + 0.85 * rz),
        nose_bridge=(cx, cy + 0.0 * ry, cz + 0.90 * rz),
        chin=(cx, cy - 0.85 * ry, cz + 0.55 * rz),
        jaw_left=(cx - 0.55 * rx, cy - 0.55 * ry, cz + 0.55 * rz),
        jaw_right=(cx + 0.55 * rx, cy - 0.55 * ry, cz + 0.55 * rz),
    )


# ── Scale-ratio calculation ──────────────────────────────────────────────

def test_compute_scale_ratios_matches_expected_dimension_ratios():
    avatar_landmarks = _uniform_face_landmarks(rx=0.20, ry=0.20, rz=0.20)
    generated_landmarks = _uniform_face_landmarks(rx=0.10, ry=0.05, rz=0.04)

    scale_x, scale_y, scale_z = compute_scale_ratios(avatar_landmarks, generated_landmarks)

    assert scale_x == pytest.approx(avatar_landmarks.face_width / generated_landmarks.face_width)
    assert scale_y == pytest.approx(avatar_landmarks.face_height / generated_landmarks.face_height)
    assert scale_z == pytest.approx(avatar_landmarks.face_depth / generated_landmarks.face_depth)
    # Different axes scale independently -- not a single uniform factor.
    assert scale_x != pytest.approx(scale_y)
    assert scale_y != pytest.approx(scale_z)


def test_compute_scale_ratios_identity_for_identical_landmarks():
    landmarks = _uniform_face_landmarks()
    scale_x, scale_y, scale_z = compute_scale_ratios(landmarks, landmarks)
    assert scale_x == pytest.approx(1.0)
    assert scale_y == pytest.approx(1.0)
    assert scale_z == pytest.approx(1.0)


def test_compute_scale_ratios_rejects_degenerate_generated_face():
    avatar_landmarks = _uniform_face_landmarks()
    degenerate = FaceLandmarks(
        eye_left=(0, 0, 0), eye_right=(0, 0, 0), nose_bridge=(0, 0, 0),
        chin=(0, 0, 0), jaw_left=(0, 0, 0), jaw_right=(0, 0, 0),
    )
    with pytest.raises(ValueError):
        compute_scale_ratios(avatar_landmarks, degenerate)


def test_extract_avatar_head_landmarks_scales_with_head_radius(body3d_params):
    small = dict(body3d_params, head_radius=0.11)
    large = dict(body3d_params, head_radius=0.15)

    small_landmarks = extract_avatar_head_landmarks(small, HEIGHT_CM)
    large_landmarks = extract_avatar_head_landmarks(large, HEIGHT_CM)

    assert large_landmarks.face_width > small_landmarks.face_width
    assert large_landmarks.eye_distance > small_landmarks.eye_distance


def test_extract_avatar_head_landmarks_requires_head_radius():
    with pytest.raises(ValueError):
        extract_avatar_head_landmarks({}, HEIGHT_CM)


# ── Landmark alignment ───────────────────────────────────────────────────

def test_align_face_mesh_moves_scaled_eye_center_close_to_avatar_eye_center(body3d_params):
    avatar_landmarks = extract_avatar_head_landmarks(body3d_params, HEIGHT_CM)
    generated_landmarks = _uniform_face_landmarks(rx=0.12, ry=0.16, rz=0.09, center=(1.0, 2.0, -3.0))

    vertices = generated_landmarks.as_array().astype(np.float32)  # stand-in "mesh" = its own landmarks
    faces = np.zeros((0, 3), dtype=np.uint32)

    fitted = align_face_mesh(vertices, faces, generated_landmarks, avatar_landmarks)

    avatar_eye_center = np.array(avatar_landmarks.eye_center)
    fitted_eye_center = np.array(fitted.landmarks.eye_center)
    # Eyes have the highest correspondence weight, so they should land much
    # closer to the avatar's eye center than an unweighted/uniform fit would.
    assert np.linalg.norm(fitted_eye_center - avatar_eye_center) < 1e-3


def test_align_face_mesh_preserves_relative_landmark_proportions(body3d_params):
    avatar_landmarks = extract_avatar_head_landmarks(body3d_params, HEIGHT_CM)
    generated_landmarks = _uniform_face_landmarks(rx=0.12, ry=0.16, rz=0.09, center=(1.0, 2.0, -3.0))
    vertices = generated_landmarks.as_array().astype(np.float32)
    faces = np.zeros((0, 3), dtype=np.uint32)

    fitted = align_face_mesh(vertices, faces, generated_landmarks, avatar_landmarks)

    # jaw_width / eye_distance should be roughly preserved by a rigid+scale
    # fit (landmark alignment, not an independent-per-point warp).
    original_ratio = generated_landmarks.jaw_width / generated_landmarks.eye_distance
    fitted_ratio = fitted.landmarks.jaw_width / fitted.landmarks.eye_distance
    assert fitted_ratio == pytest.approx(original_ratio, rel=0.05)


def test_align_face_mesh_identity_when_already_aligned(body3d_params):
    avatar_landmarks = extract_avatar_head_landmarks(body3d_params, HEIGHT_CM)
    vertices = avatar_landmarks.as_array().astype(np.float32)
    faces = np.zeros((0, 3), dtype=np.uint32)

    fitted = align_face_mesh(vertices, faces, avatar_landmarks, avatar_landmarks, scale=(1.0, 1.0, 1.0))

    fitted_pts = fitted.landmarks.as_array()
    avatar_pts = avatar_landmarks.as_array()
    np.testing.assert_allclose(fitted_pts, avatar_pts, atol=1e-5)


# ── Mock face mesh generation ────────────────────────────────────────────

def test_mock_face_mesh_provider_returns_valid_mesh():
    provider = MockFaceMeshProvider()
    front = np.full((200, 150, 3), 180, dtype=np.uint8)

    generated = provider.generate(front)

    assert isinstance(generated, GeneratedFaceMesh)
    assert generated.vertices.shape[1] == 3
    assert generated.faces.shape[1] == 3
    assert generated.faces.max() < len(generated.vertices)
    assert set(generated.landmarks.keys()) == set(LANDMARK_NAMES)
    assert generated.texture_png is not None


def test_mock_face_mesh_provider_width_tracks_photo_aspect_ratio():
    provider = MockFaceMeshProvider()
    narrow = provider.generate(np.full((300, 120, 3), 180, dtype=np.uint8))
    wide = provider.generate(np.full((150, 280, 3), 180, dtype=np.uint8))

    narrow_landmarks = landmarks_from_dict(narrow.landmarks)
    wide_landmarks = landmarks_from_dict(wide.landmarks)
    assert wide_landmarks.eye_distance > narrow_landmarks.eye_distance


def test_generated_face_mesh_requires_all_landmark_keys():
    with pytest.raises(ValueError):
        GeneratedFaceMesh(
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.uint32),
            landmarks={"eye_left": (0, 0, 0)},
        )


# ── Fallback when Unique3D is not configured ─────────────────────────────

def test_unique3d_provider_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("FACE_MESH_ENDPOINT", raising=False)
    provider = Unique3DFaceMeshProvider()
    front = np.full((100, 100, 3), 180, dtype=np.uint8)
    assert provider.generate(front) is None


def test_run_face_fitting_falls_back_when_unique3d_unconfigured(avatar_result):
    front = np.full((100, 100, 3), 180, dtype=np.uint8)
    unconfigured_provider = Unique3DFaceMeshProvider(endpoint=None)

    result = run_face_fitting(avatar_result, front, provider=unconfigured_provider)

    assert result.status == "unavailable"
    assert result.scale is None
    assert result.avatar_result is avatar_result
    assert result.avatar_result.avatar_mesh_glb == avatar_result.avatar_mesh_glb
    assert any("no face mesh provider" in warning for warning in result.warnings)


# ── Preservation of existing avatar body parameters ──────────────────────

def test_run_face_fitting_with_mock_provider_preserves_body_params(avatar_result):
    original_body3d_params = copy.deepcopy(avatar_result.body3d_params)
    original_glb = avatar_result.avatar_mesh_glb
    front = np.full((100, 100, 3), 180, dtype=np.uint8)

    result = run_face_fitting(avatar_result, front, provider=MockFaceMeshProvider())

    # The mock backend never blends geometry in (see MockFaceFitRunner) --
    # confirms the fitting math ran without silently claiming a real edit.
    assert result.status == "fitted_metadata_only"
    assert result.scale is not None
    assert len(result.scale) == 3

    # Body measurements/params and mesh bytes are byte-for-byte unchanged --
    # the existing measurement-based avatar body is untouched either way.
    assert result.avatar_result.body3d_params == original_body3d_params
    assert result.avatar_result.avatar_mesh_glb == original_glb
    assert avatar_result.body3d_params == original_body3d_params  # input not mutated either


def test_mock_face_fit_runner_never_modifies_geometry():
    runner = MockFaceFitRunner()
    original = b"not-really-a-glb-but-treated-as-opaque-bytes"

    result_bytes, geometry_modified = runner.fit(original, fitted_face=None, texture_png=None)

    assert result_bytes == original
    assert geometry_modified is False
