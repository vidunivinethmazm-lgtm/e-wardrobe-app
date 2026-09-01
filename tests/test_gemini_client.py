"""Unit tests for `server/ai_tryon/gemini_client.py` - response parsing and
mock/real mode switching. No real network calls are made; `requests.post` is
patched for the "real mode" tests.
"""

import base64
import io

import pytest
from PIL import Image

from backend.ai_tryon import gemini_client


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_mock_mode_returns_person_image_unchanged(monkeypatch):
    monkeypatch.setattr(gemini_client, "MOCK", True)
    person = _png_bytes()
    assert gemini_client.generate_tryon_image(person, [_png_bytes()]) == person


def test_real_mode_requires_api_key(monkeypatch):
    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_client.generate_tryon_image(_png_bytes(), [_png_bytes()])


def test_real_mode_requires_clothing_image(monkeypatch):
    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="clothing image"):
        gemini_client.generate_tryon_image(_png_bytes(), [])


def test_extract_image_returns_inline_data_camel_case():
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("ascii")
    data = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": image_b64}}]},
            }
        ]
    }
    assert gemini_client._extract_image(data) == b"fake-image-bytes"


def test_extract_image_returns_inline_data_snake_case():
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("ascii")
    data = {"candidates": [{"content": {"parts": [{"inline_data": {"data": image_b64}}]}}]}
    assert gemini_client._extract_image(data) == b"fake-image-bytes"


def test_extract_image_raises_on_safety_finish_reason():
    data = {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}
    with pytest.raises(RuntimeError, match="SAFETY"):
        gemini_client._extract_image(data)


def test_extract_image_raises_on_text_only_response():
    data = {
        "candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": "Sorry, I can't do that."}]}}
        ]
    }
    with pytest.raises(RuntimeError, match="text instead of an image"):
        gemini_client._extract_image(data)


def test_extract_image_raises_on_blocked_prompt():
    data = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(RuntimeError, match="blocked the prompt"):
        gemini_client._extract_image(data)


def test_extract_image_raises_on_empty_response():
    with pytest.raises(RuntimeError, match="no candidates"):
        gemini_client._extract_image({})


def test_real_mode_sends_request_and_parses_response(monkeypatch):
    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "test-key")

    image_b64 = base64.b64encode(b"generated-image-bytes").decode("ascii")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": image_b64}}]},
                    }
                ]
            }

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(gemini_client.requests, "post", fake_post)

    result = gemini_client.generate_tryon_image(_png_bytes(), [_png_bytes(), _png_bytes()])

    assert result == b"generated-image-bytes"
    assert captured["url"] == gemini_client.GEMINI_API_URL
    assert "v1beta" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    parts = captured["json"]["contents"][0]["parts"]
    # prompt text + person image + 2 clothing images
    assert len(parts) == 4
    assert parts[0] == {"text": gemini_client.PROMPT}


def test_real_mode_raises_clear_error_on_invalid_key(monkeypatch):
    monkeypatch.setattr(gemini_client, "MOCK", False)
    monkeypatch.setattr(gemini_client, "GEMINI_API_KEY", "bad-key")

    class FakeResponse:
        status_code = 400
        text = '{"error": {"message": "API key not valid"}}'

        def raise_for_status(self):
            raise gemini_client.requests.HTTPError("400 Client Error")

    monkeypatch.setattr(gemini_client.requests, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="Gemini API returned 400"):
        gemini_client.generate_tryon_image(_png_bytes(), [_png_bytes()])
