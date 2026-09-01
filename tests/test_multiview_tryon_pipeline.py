"""Tests for the EXPERIMENTAL `multiview_tryon` garment-fitting pipeline
(`avatar_pipeline.model7_garment_fitting.multiview`) and its integration
into `POST /api/avatars/<id>/fit-garment` via `pipeline_mode`.

Covers: pipeline-mode dispatch/validation, fallback to the existing
adaptive_template behavior, provider mock-vs-real status for each of the
three new provider families, the garment-only isolation contract (including
rejecting a full-body reconstruction), and the extended endpoint's
mock-only end-to-end response. Runs entirely under mock providers — no
Blender/network required, same convention as `test_model7_garment_fitting.py`.
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from avatar_pipeline.model7_garment_fitting.fitting_types import (
    GARMENT_PIPELINE_MODE, PIPELINE_MODES, GarmentFittingError,
)
from avatar_pipeline.model7_garment_fitting.garment_mesh_generation import (
    LANDMARK_NAMES, GeneratedGarmentMesh, MockGarmentMeshProvider,
)
from avatar_pipeline.model7_garment_fitting.multiview.avatar3d_providers import (
    MockFullAvatarImageTo3DProvider, Unique3DAvatarProvider, get_full_avatar_provider,
)
from avatar_pipeline.model7_garment_fitting.multiview.garment_isolation import isolate_garment_geometry
from avatar_pipeline.model7_garment_fitting.multiview.mesh3d_providers import (
    Hunyuan3D2MVProvider, MockMultiViewImageTo3DProvider, get_multiview_mesh_provider,
)
from avatar_pipeline.model7_garment_fitting.multiview.texture_providers import (
    Hunyuan3DPaintProvider, MockTextureGenerationProvider, get_texture_provider,
)
from avatar_pipeline.model7_garment_fitting.multiview.tryon_providers import (
    GeminiVirtualTryOnProvider, IDMVTonProvider, MockVirtualTryOnProvider, get_virtual_tryon_provider,
)
from avatar_pipeline.model7_garment_fitting.multiview import (
    avatar3d_providers, tryon_providers, mesh3d_providers, texture_providers,
)


# ── Synthetic images ──────────────────────────────────────────────────────

def _tshirt_silhouette(w=300, h=360):
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    cx = w / 2
    points = [
        (cx - 0.10 * w, 0.08 * h), (cx - 0.32 * w, 0.14 * h), (cx - 0.48 * w, 0.30 * h),
        (cx - 0.36 * w, 0.34 * h), (cx - 0.30 * w, 0.55 * h), (cx - 0.34 * w, 0.90 * h),
        (cx + 0.34 * w, 0.90 * h), (cx + 0.30 * w, 0.55 * h), (cx + 0.36 * w, 0.34 * h),
        (cx + 0.48 * w, 0.30 * h), (cx + 0.32 * w, 0.14 * h), (cx + 0.10 * w, 0.08 * h),
    ]
    draw.polygon(points, fill=(60, 90, 140))
    return np.array(img)


def _person_silhouette(w=240, h=480):
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    cx = w / 2
    draw.ellipse([cx - 0.12 * w, 0.02 * h, cx + 0.12 * w, 0.18 * h], fill=(210, 180, 160))
    draw.polygon([
        (cx - 0.20 * w, 0.18 * h), (cx + 0.20 * w, 0.18 * h),
        (cx + 0.24 * w, 0.55 * h), (cx + 0.10 * w, 0.55 * h),
        (cx + 0.12 * w, 0.95 * h), (cx + 0.04 * w, 0.95 * h),
        (cx, 0.57 * h),
        (cx - 0.04 * w, 0.95 * h), (cx - 0.12 * w, 0.95 * h),
        (cx - 0.10 * w, 0.55 * h), (cx - 0.24 * w, 0.55 * h),
    ], fill=(80, 80, 120))
    return np.array(img)


def _png_upload(rgb_array, name="photo.png"):
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    buf.seek(0)
    return buf, name


# ── Pipeline-mode flag ────────────────────────────────────────────────────

def test_garment_pipeline_mode_defaults_to_adaptive_template():
    assert GARMENT_PIPELINE_MODE == "adaptive_template"
    assert PIPELINE_MODES == ("adaptive_template", "multiview_tryon")


# ── Provider factories: mock-by-default, real-returns-None-when-unconfigured ──

def test_get_virtual_tryon_provider_defaults_to_mock():
    assert isinstance(get_virtual_tryon_provider(), MockVirtualTryOnProvider)


def test_mock_virtual_tryon_provider_returns_person_photos_unchanged():
    front, back = _person_silhouette(), _person_silhouette()
    garment_front, garment_back = _tshirt_silhouette(), _tshirt_silhouette()
    provider = MockVirtualTryOnProvider()
    result = provider.generate(front, back, garment_front, garment_back, "upper_body")
    assert result is not None
    front_png, back_png = result
    assert np.array_equal(np.array(Image.open(io.BytesIO(front_png)).convert("RGB")), front)
    assert np.array_equal(np.array(Image.open(io.BytesIO(back_png)).convert("RGB")), back)


def test_gemini_tryon_provider_returns_none_when_ai_tryon_mock_left_on(monkeypatch):
    """Even with a GEMINI_API_KEY set, gemini_client's own AI_TRYON_MOCK
    defaults to on — this provider must report itself unconfigured rather
    than silently echo the person photos as if a real try-on happened."""
    from server.ai_tryon import gemini_client

    monkeypatch.setattr(gemini_client, "MOCK", True)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "fake-key")

    provider = GeminiVirtualTryOnProvider()
    result = provider.generate(
        _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
    )
    assert result is None


def test_gemini_tryon_provider_returns_none_without_api_key(monkeypatch):
    from server.ai_tryon import gemini_client

    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", None)

    provider = GeminiVirtualTryOnProvider()
    result = provider.generate(
        _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
    )
    assert result is None


def test_gemini_tryon_provider_calls_gemini_client_when_configured(monkeypatch):
    from server.ai_tryon import gemini_client

    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "fake-key")
    calls = []

    def fake_generate(person_image, clothing_images):
        calls.append((person_image, clothing_images))
        return b"fake-png-bytes"

    monkeypatch.setattr(gemini_client, "generate_tryon_image", fake_generate)

    provider = GeminiVirtualTryOnProvider()
    result = provider.generate(
        _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
    )
    assert result == (b"fake-png-bytes", b"fake-png-bytes")
    assert len(calls) == 2


def test_gemini_tryon_provider_raises_on_request_failure(monkeypatch):
    from server.ai_tryon import gemini_client

    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "fake-key")

    def fake_generate(person_image, clothing_images):
        raise RuntimeError("Gemini blocked the prompt")

    monkeypatch.setattr(gemini_client, "generate_tryon_image", fake_generate)

    provider = GeminiVirtualTryOnProvider()
    with pytest.raises(GarmentFittingError):
        provider.generate(
            _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
        )


def test_idm_vton_provider_returns_none_when_fully_unconfigured():
    provider = IDMVTonProvider(endpoint=None, hf_space=None)
    result = provider.generate(
        _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
    )
    assert result is None


def test_idm_vton_provider_defaults_to_public_hf_space():
    """With nothing configured beyond picking idm_vton, the free public
    yisol/IDM-VTON Space is used automatically — no endpoint/payment needed."""
    provider = IDMVTonProvider(endpoint=None)
    assert provider.hf_space == "yisol/IDM-VTON"


def test_idm_vton_provider_raises_on_request_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    provider = IDMVTonProvider(endpoint="http://example.invalid/tryon")
    with pytest.raises(GarmentFittingError):
        provider.generate(
            _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
        )


def test_idm_vton_provider_endpoint_takes_precedence_over_hf_space(monkeypatch):
    """When both are configured, the self-hosted custom endpoint wins —
    the HF Space path must not be attempted (no gradio_client.Client call)."""
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    provider = IDMVTonProvider(endpoint="http://example.invalid/tryon", hf_space="yisol/IDM-VTON")
    with pytest.raises(GarmentFittingError, match="request failed"):
        provider.generate(
            _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
        )


def test_idm_vton_provider_hf_space_raises_when_client_unreachable(monkeypatch):
    class _BrokenClient:
        def __init__(self, *a, **kw):
            raise ConnectionError("space is down")

    monkeypatch.setattr("gradio_client.Client", _BrokenClient)
    provider = IDMVTonProvider(endpoint=None, hf_space="yisol/IDM-VTON")
    with pytest.raises(GarmentFittingError, match="could not connect"):
        provider.generate(
            _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
        )


def test_idm_vton_provider_hf_space_parses_predict_result(monkeypatch, tmp_path):
    output_path = tmp_path / "out.png"
    Image.fromarray(_person_silhouette()).save(output_path)

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def predict(self, **kwargs):
            assert kwargs["api_name"] == "/tryon"
            return (str(output_path), str(output_path))

    monkeypatch.setattr("gradio_client.Client", _FakeClient)
    monkeypatch.setattr("gradio_client.handle_file", lambda path: path)

    provider = IDMVTonProvider(endpoint=None, hf_space="yisol/IDM-VTON")
    result = provider.generate(
        _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
    )
    assert result is not None
    front_png, back_png = result
    assert front_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert back_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_idm_vton_provider_hf_space_raises_on_unexpected_result_shape(monkeypatch):
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def predict(self, **kwargs):
            return None  # not a tuple/list -> not subscriptable

    monkeypatch.setattr("gradio_client.Client", _FakeClient)
    monkeypatch.setattr("gradio_client.handle_file", lambda path: path)

    provider = IDMVTonProvider(endpoint=None, hf_space="yisol/IDM-VTON")
    with pytest.raises(GarmentFittingError, match="unexpected response shape"):
        provider.generate(
            _person_silhouette(), _person_silhouette(), _tshirt_silhouette(), _tshirt_silhouette(), "upper_body",
        )


def test_get_multiview_mesh_provider_defaults_to_mock():
    assert isinstance(get_multiview_mesh_provider(), MockMultiViewImageTo3DProvider)


def test_mock_multiview_mesh_provider_delegates_to_mock_garment_mesh_provider():
    mesh = MockMultiViewImageTo3DProvider().generate(_person_silhouette(), _person_silhouette(), "upper_body")
    assert mesh.is_mock is True
    assert len(mesh.vertices) > 0


def test_hunyuan3d_2mv_provider_returns_none_without_endpoint():
    provider = Hunyuan3D2MVProvider(endpoint=None)
    assert provider.generate(_person_silhouette(), _person_silhouette(), "upper_body") is None


def test_hunyuan3d_2mv_provider_raises_on_request_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    provider = Hunyuan3D2MVProvider(endpoint="http://example.invalid/mesh")
    with pytest.raises(GarmentFittingError):
        provider.generate(_person_silhouette(), _person_silhouette(), "upper_body")


def test_get_texture_provider_defaults_to_mock():
    assert isinstance(get_texture_provider(), MockTextureGenerationProvider)


def test_mock_texture_provider_returns_atlas_from_garment_photos():
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    texture_png = MockTextureGenerationProvider().generate(mesh, _tshirt_silhouette(), _tshirt_silhouette())
    assert texture_png is not None
    assert texture_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_hunyuan3d_paint_provider_returns_none_without_endpoint():
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    provider = Hunyuan3DPaintProvider(endpoint=None)
    assert provider.generate(mesh, _tshirt_silhouette(), _tshirt_silhouette()) is None


def test_hunyuan3d_paint_provider_raises_on_request_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    provider = Hunyuan3DPaintProvider(endpoint="http://example.invalid/paint")
    with pytest.raises(GarmentFittingError):
        provider.generate(mesh, _tshirt_silhouette(), _tshirt_silhouette())


# ── PIVOT: full-avatar reconstruction (Unique3D) — replaces the garment-
# isolation-then-fit design in the pipeline; mesh3d/texture providers above
# stay tested directly (they're just no longer called from pipeline.py). ──

def test_get_full_avatar_provider_defaults_to_mock():
    assert isinstance(get_full_avatar_provider(), MockFullAvatarImageTo3DProvider)


def test_mock_full_avatar_provider_produces_valid_mesh():
    mesh = MockFullAvatarImageTo3DProvider().generate(_person_silhouette(), _person_silhouette())
    assert mesh.is_mock is True
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_unique3d_avatar_provider_returns_none_when_fully_unconfigured():
    provider = Unique3DAvatarProvider(endpoint=None, hf_space=None)
    assert provider.generate(_person_silhouette(), _person_silhouette()) is None


def test_unique3d_avatar_provider_defaults_to_public_hf_space():
    provider = Unique3DAvatarProvider(endpoint=None)
    assert provider.hf_space == "Wuvin/Unique3D"


def test_unique3d_avatar_provider_raises_on_request_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    provider = Unique3DAvatarProvider(endpoint="http://example.invalid/avatar3d")
    with pytest.raises(GarmentFittingError):
        provider.generate(_person_silhouette(), _person_silhouette())


def test_unique3d_avatar_provider_hf_space_raises_when_client_unreachable(monkeypatch):
    class _BrokenClient:
        def __init__(self, *a, **kw):
            raise ConnectionError("space is down")

    monkeypatch.setattr("gradio_client.Client", _BrokenClient)
    provider = Unique3DAvatarProvider(endpoint=None, hf_space="Wuvin/Unique3D")
    with pytest.raises(GarmentFittingError, match="could not connect"):
        provider.generate(_person_silhouette(), _person_silhouette())


def test_unique3d_avatar_provider_hf_space_parses_predict_result(monkeypatch, tmp_path):
    from avatar_pipeline.model7_garment_fitting.glb_writer import write_mesh_glb

    mesh = MockFullAvatarImageTo3DProvider().generate(_person_silhouette(), _person_silhouette())
    glb_bytes = write_mesh_glb(mesh.vertices, mesh.faces, mesh.uvs, mesh.texture_png)
    output_path = tmp_path / "avatar.glb"
    output_path.write_bytes(glb_bytes)

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def predict(self, **kwargs):
            assert kwargs["api_name"] == "/generate3dv2"
            return str(output_path)

    monkeypatch.setattr("gradio_client.Client", _FakeClient)
    monkeypatch.setattr("gradio_client.handle_file", lambda path: path)

    provider = Unique3DAvatarProvider(endpoint=None, hf_space="Wuvin/Unique3D")
    result = provider.generate(_person_silhouette(), _person_silhouette())
    assert result is not None
    assert len(result.vertices) == len(mesh.vertices)
    assert len(result.faces) == len(mesh.faces)
    assert result.is_mock is False


def test_unique3d_avatar_provider_hf_space_raises_on_unexpected_result_shape(monkeypatch):
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def predict(self, **kwargs):
            return 42  # neither str nor tuple/list

    monkeypatch.setattr("gradio_client.Client", _FakeClient)
    monkeypatch.setattr("gradio_client.handle_file", lambda path: path)

    provider = Unique3DAvatarProvider(endpoint=None, hf_space="Wuvin/Unique3D")
    with pytest.raises(GarmentFittingError, match="unexpected response shape"):
        provider.generate(_person_silhouette(), _person_silhouette())


def test_provider_factories_dispatch_on_env_string(monkeypatch):
    monkeypatch.setattr(tryon_providers, "VIRTUAL_TRYON_PROVIDER", "idm_vton")
    assert isinstance(get_virtual_tryon_provider(), IDMVTonProvider)

    monkeypatch.setattr(mesh3d_providers, "IMAGE_TO_3D_MV_PROVIDER", "hunyuan3d_2mv")
    assert isinstance(get_multiview_mesh_provider(), Hunyuan3D2MVProvider)

    monkeypatch.setattr(texture_providers, "TEXTURE_PROVIDER", "hunyuan3d_paint")
    assert isinstance(get_texture_provider(), Hunyuan3DPaintProvider)

    monkeypatch.setattr(avatar3d_providers, "FULL_AVATAR_3D_PROVIDER", "unique3d")
    assert isinstance(get_full_avatar_provider(), Unique3DAvatarProvider)


def test_provider_factories_reject_unknown_provider_name(monkeypatch):
    monkeypatch.setattr(tryon_providers, "VIRTUAL_TRYON_PROVIDER", "not_a_real_provider")
    with pytest.raises(NotImplementedError):
        get_virtual_tryon_provider()


# ── Garment-only isolation contract ───────────────────────────────────────

def test_isolate_garment_geometry_keeps_only_masked_region():
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :] = True  # keep only the upper half of the mesh (rows 0..h/2 -> top of mesh)

    isolated = isolate_garment_geometry(mesh, mask, mask)

    assert 0 < len(isolated.vertices) < len(mesh.vertices)
    assert isolated.faces.max() < len(isolated.vertices)
    assert isolated.landmarks == mesh.landmarks


def test_isolate_garment_geometry_passthrough_when_no_masks_given():
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    isolated = isolate_garment_geometry(mesh, None, None)
    assert len(isolated.vertices) == len(mesh.vertices)


def test_isolate_garment_geometry_rejects_full_body_reconstruction():
    """A mesh shaped like a whole standing figure (tall relative to its
    width) whose mask keeps almost everything must be rejected outright —
    never handed to the avatar-fitting stage as "the garment"."""
    n = 60
    rng = np.random.default_rng(0)
    vertices = np.zeros((n, 3), dtype=np.float32)
    vertices[:, 1] = np.linspace(0.0, 1.7, n)  # full standing-human height
    vertices[:, 0] = rng.uniform(-0.1, 0.1, n)
    vertices[:, 2] = rng.uniform(-0.05, 0.05, n)
    faces = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.uint32)
    landmarks = {name: (0.0, 0.0, 0.0) for name in LANDMARK_NAMES}
    full_body_mesh = GeneratedGarmentMesh(
        vertices=vertices, faces=faces, uvs=np.zeros((n, 2), dtype=np.float32), texture_png=None,
        landmarks=landmarks, garment_type="upper_body", is_mock=True,
    )
    mask = np.ones((10, 10), dtype=bool)  # keeps ~everything

    with pytest.raises(GarmentFittingError):
        isolate_garment_geometry(full_body_mesh, mask, mask)


def test_isolate_garment_geometry_rejects_empty_result():
    mesh = MockGarmentMeshProvider().generate(_tshirt_silhouette(), _tshirt_silhouette(), "upper_body")
    mask = np.zeros((10, 10), dtype=bool)  # keeps nothing
    with pytest.raises(GarmentFittingError):
        isolate_garment_geometry(mesh, mask, mask)


# ── Flask endpoint: pipeline_mode dispatch/fallback ──────────────────────

@pytest.fixture
def client():
    from server.app import app
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


def test_fit_garment_invalid_pipeline_mode_returns_400(client, avatar_id):
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={
            "garment_front": front, "garment_back": back,
            "garment_type": "upper_body", "pipeline_mode": "not_a_real_mode",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "pipeline_mode" in response.get_json()["error"]


def test_fit_garment_omitting_pipeline_mode_uses_adaptive_template(client, avatar_id):
    """Fallback: not sending `pipeline_mode` at all reproduces the existing
    adaptive_template response — the default endpoint behavior is unchanged."""
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={"garment_front": front, "garment_back": back, "garment_type": "upper_body"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    assert body["pipeline_mode"] == "adaptive_template"
    assert body["is_mock"] is True
    assert "garment_tryon_front_url" not in body


def test_fit_garment_multiview_tryon_missing_person_photos_returns_400(client, avatar_id):
    front, back = _png_upload(_tshirt_silhouette()), _png_upload(_tshirt_silhouette())
    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={
            "garment_front": front, "garment_back": back,
            "garment_type": "upper_body", "pipeline_mode": "multiview_tryon",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "person_front" in response.get_json()["error"]


def test_fit_garment_multiview_tryon_end_to_end(client, avatar_id):
    garment_front, garment_back = _png_upload(_tshirt_silhouette(), "gf.png"), _png_upload(_tshirt_silhouette(), "gb.png")
    person_front, person_back = _png_upload(_person_silhouette(), "pf.png"), _png_upload(_person_silhouette(), "pb.png")

    response = client.post(
        f"/api/avatars/{avatar_id}/fit-garment",
        data={
            "garment_front": garment_front, "garment_back": garment_back,
            "person_front": person_front, "person_back": person_back,
            "garment_type": "upper_body", "pipeline_mode": "multiview_tryon",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()

    assert body["pipeline_mode"] == "multiview_tryon"
    assert body["virtual_tryon_provider"] == "mock"
    assert body["image_to_3d_provider"] == "mock"
    # PIVOT: no separate texture stage anymore — the reconstructed avatar
    # mesh carries its own baked texture; region_scales no longer computed
    # (nothing is fitted onto a separate avatar).
    assert body["texture_provider"] is None
    assert body["region_scales"] is None
    assert body["is_real_3d_generation"] is False
    assert body["is_full_avatar_replacement"] is True
    assert body["is_mock"] is True

    fit_id = body["fit_id"]
    assert body["garment_tryon_front_url"] == f"/api/avatars/{avatar_id}/fitted-garment/{fit_id}-tryon-front.png"
    assert body["garment_tryon_back_url"] == f"/api/avatars/{avatar_id}/fitted-garment/{fit_id}-tryon-back.png"

    glb_response = client.get(body["garment_mesh_url"])
    assert glb_response.status_code == 200
    assert glb_response.data[:4] == b"glTF"

    front_preview = client.get(body["garment_tryon_front_url"])
    assert front_preview.status_code == 200
    assert front_preview.mimetype == "image/png"

    back_preview = client.get(body["garment_tryon_back_url"])
    assert back_preview.status_code == 200
    assert back_preview.mimetype == "image/png"
