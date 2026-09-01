"""Tests for `server/ai_tryon/image_to_3d.py` - provider selection and the
Meshy provider's request/response handling. No real network calls; `requests`
is monkeypatched for Meshy tests.
"""

import io

import pytest
from PIL import Image

from backend.ai_tryon import image_to_3d
from backend.app import app


@pytest.fixture
def client():
    return app.test_client()


def _png_file(color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _png_bytes(color=(10, 20, 30)):
    return _png_file(color).getvalue()


# --- get_provider() selection ---------------------------------------------


def test_get_provider_defaults_to_mock(monkeypatch):
    monkeypatch.setattr(image_to_3d, "PROVIDER", "mock")
    provider = image_to_3d.get_provider()
    assert isinstance(provider, image_to_3d.MockImage23DProvider)


def test_get_provider_meshy_requires_api_key(monkeypatch):
    monkeypatch.setattr(image_to_3d, "PROVIDER", "meshy")
    monkeypatch.setattr(image_to_3d, "IMAGE_TO_3D_API_KEY", None)
    with pytest.raises(RuntimeError, match="IMAGE_TO_3D_API_KEY"):
        image_to_3d.get_provider()


def test_get_provider_meshy_with_api_key(monkeypatch):
    monkeypatch.setattr(image_to_3d, "PROVIDER", "meshy")
    monkeypatch.setattr(image_to_3d, "IMAGE_TO_3D_API_KEY", "test-key")
    provider = image_to_3d.get_provider()
    assert isinstance(provider, image_to_3d.MeshyImage23DProvider)


def test_get_provider_unknown_provider_not_implemented(monkeypatch):
    monkeypatch.setattr(image_to_3d, "PROVIDER", "tripo")
    with pytest.raises(NotImplementedError, match="tripo"):
        image_to_3d.get_provider()


# --- MeshyImage23DProvider ---------------------------------------------------


def test_meshy_create_job_sends_data_uri_and_returns_task_id(monkeypatch):
    provider = image_to_3d.MeshyImage23DProvider("test-key")
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "task-123"}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(image_to_3d.requests, "post", fake_post)

    job_id = provider.create_job(_png_bytes())

    assert job_id == "task-123"
    assert captured["url"] == image_to_3d.MESHY_API_BASE
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["image_url"].startswith("data:image/png;base64,")


def test_meshy_create_job_raises_on_http_error(monkeypatch):
    provider = image_to_3d.MeshyImage23DProvider("bad-key")

    class FakeResponse:
        status_code = 401
        text = '{"message": "Invalid API key"}'

        def raise_for_status(self):
            raise image_to_3d.requests.HTTPError("401 Client Error")

    monkeypatch.setattr(image_to_3d.requests, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="Meshy API returned 401"):
        provider.create_job(_png_bytes())


@pytest.mark.parametrize(
    "meshy_status, expected",
    [
        ("PENDING", "processing"),
        ("IN_PROGRESS", "processing"),
        ("SUCCEEDED", "ready"),
        ("FAILED", "error"),
        ("CANCELED", "error"),
    ],
)
def test_meshy_get_job_status_mapping(monkeypatch, meshy_status, expected):
    provider = image_to_3d.MeshyImage23DProvider("test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": meshy_status}

    monkeypatch.setattr(image_to_3d.requests, "get", lambda *a, **k: FakeResponse())

    assert provider.get_job_status("task-123") == expected


def test_meshy_get_result_glb_downloads_model_url(monkeypatch):
    provider = image_to_3d.MeshyImage23DProvider("test-key")

    class FakeTaskResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "SUCCEEDED", "model_urls": {"glb": "https://cdn.example.com/model.glb"}}

    class FakeGlbResponse:
        status_code = 200
        content = b"glTF-fake-bytes"

        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if url == f"{image_to_3d.MESHY_API_BASE}/task-123":
            return FakeTaskResponse()
        return FakeGlbResponse()

    monkeypatch.setattr(image_to_3d.requests, "get", fake_get)

    assert provider.get_result_glb("task-123") == b"glTF-fake-bytes"
    assert "https://cdn.example.com/model.glb" in calls


def test_meshy_get_result_glb_raises_when_no_glb_url(monkeypatch):
    provider = image_to_3d.MeshyImage23DProvider("test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "SUCCEEDED", "model_urls": {}}

    monkeypatch.setattr(image_to_3d.requests, "get", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="no GLB result"):
        provider.get_result_glb("task-123")


# --- POST /api/ai-tryon/<id>/avatar3d integration ---------------------------


def test_avatar3d_returns_clear_error_when_meshy_key_missing(monkeypatch, client):
    monkeypatch.setattr(image_to_3d, "PROVIDER", "meshy")
    monkeypatch.setattr(image_to_3d, "IMAGE_TO_3D_API_KEY", None)

    response = client.post(
        "/api/ai-tryon",
        data={
            "person_photo": (_png_file(), "person.png"),
            "clothing_photo": (_png_file(), "clothing.png"),
        },
        content_type="multipart/form-data",
    )
    tryon_id = response.get_json()["tryon_id"]

    avatar3d_response = client.post(f"/api/ai-tryon/{tryon_id}/avatar3d")
    assert avatar3d_response.status_code == 502
    body = avatar3d_response.get_json()
    assert "IMAGE_TO_3D_API_KEY" in body["error"]
