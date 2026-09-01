"""Smoke tests for the AI try-on + image-to-3D mock endpoints
(`POST /api/ai-tryon`, `GET /api/ai-tryon/<id>/image.png`,
`POST /api/ai-tryon/<id>/avatar3d`, `GET /api/ai-tryon/<id>/avatar.glb`).

Runs with AI_TRYON_MOCK=1 and IMAGE_TO_3D_PROVIDER=mock (both defaults) - no
Gemini or image-to-3D API keys needed.
"""

import io

import pytest
from PIL import Image

from server.app import app


@pytest.fixture
def client():
    return app.test_client()


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_ai_tryon_pipeline(client):
    response = client.post(
        "/api/ai-tryon",
        data={
            "person_photo": (_png_bytes((255, 0, 0)), "person.png"),
            "clothing_photo": (_png_bytes((0, 0, 255)), "clothing.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    tryon_id = body["tryon_id"]
    assert body["generated_image_url"] == f"/api/ai-tryon/{tryon_id}/image.png"

    image_response = client.get(f"/api/ai-tryon/{tryon_id}/image.png")
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/png"
    generated = Image.open(io.BytesIO(image_response.data))
    assert generated.size == (32, 32)

    avatar3d_response = client.post(f"/api/ai-tryon/{tryon_id}/avatar3d")
    assert avatar3d_response.status_code == 200, avatar3d_response.get_data(as_text=True)
    avatar_body = avatar3d_response.get_json()
    assert avatar_body["avatar_mesh_url"] == f"/api/ai-tryon/{tryon_id}/avatar.glb"

    glb_response = client.get(f"/api/ai-tryon/{tryon_id}/avatar.glb")
    assert glb_response.status_code == 200
    assert glb_response.mimetype == "model/gltf-binary"
    assert glb_response.data[:4] == b"glTF"


def test_ai_tryon_missing_files(client):
    response = client.post("/api/ai-tryon", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_ai_tryon_accepts_multiple_clothing_photos(client):
    response = client.post(
        "/api/ai-tryon",
        data={
            "person_photo": (_png_bytes((255, 0, 0)), "person.png"),
            "clothing_photo": [
                (_png_bytes((0, 0, 255)), "top.png"),
                (_png_bytes((0, 255, 0)), "skirt.png"),
            ],
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    image_response = client.get(body["generated_image_url"])
    assert image_response.status_code == 200


def test_ai_tryon_config(client):
    response = client.get("/api/ai-tryon/config")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {
        "ai_tryon_mock",
        "gemini_model",
        "gemini_api_key_present",
        "image_to_3d_provider",
        "image_to_3d_timeout_s",
        "image_to_3d_api_key_present",
    }
    # Defaults (no env vars set): mock mode, mock 3D provider, no API keys.
    assert body["ai_tryon_mock"] is True
    assert body["image_to_3d_provider"] == "mock"
    assert body["gemini_api_key_present"] is False
    assert body["image_to_3d_api_key_present"] is False


def test_ai_tryon_unknown_id(client):
    unknown_id = "0" * 32
    assert client.get(f"/api/ai-tryon/{unknown_id}/image.png").status_code == 404
    assert client.post(f"/api/ai-tryon/{unknown_id}/avatar3d").status_code == 404
    assert client.get(f"/api/ai-tryon/{unknown_id}/avatar.glb").status_code == 404
