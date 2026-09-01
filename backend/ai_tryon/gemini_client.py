"""Gemini ("Nano Banana" / Gemini 2.5 Flash Image) client for the AI try-on
step (`POST /api/ai-tryon`, see `server/app.py`).

Two modes, selected by the AI_TRYON_MOCK env var (default "1"):

- mock (default): `generate_tryon_image` returns the person photo unchanged.
  No network call, no GEMINI_API_KEY needed - lets the endpoint and the
  mobile app run end-to-end without any Gemini access.
- real (AI_TRYON_MOCK=0): calls the real Gemini `generateContent` API.
  Requires GEMINI_API_KEY (see https://ai.google.dev/).
"""

import base64
import os

import requests

MOCK = os.environ.get("AI_TRYON_MOCK", "1") != "0"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-image")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

PROMPT = (
    "Using the first image as the reference person and the remaining "
    "image(s) as reference garment(s), generate one photorealistic, "
    "front-facing, full-body image of the same person wearing that garment. "
    "Neutral standing pose, arms relaxed, plain light-gray studio "
    "background, whole head, body, legs and feet visible, no cropping."
)

BACK_VIEW_PROMPT = (
    "Using the given image as the reference person (a front-facing, "
    "full-body photo), generate one photorealistic image of the SAME "
    "person photographed from directly behind (a rear/back view). Preserve "
    "body proportions, height, build, skin tone, hair color/length/style, "
    "and clothing exactly as shown, just rotated 180 degrees so the back of "
    "the head and body is visible instead of the front. Same neutral "
    "standing pose, arms relaxed, same plain light-gray studio background, "
    "whole head, body, legs and feet visible, no cropping."
)


def generate_tryon_image(person_image: bytes, clothing_images: list[bytes]) -> bytes:
    """Returns image bytes (PNG/JPEG) of one generated full-body try-on image.

    `clothing_images` must contain at least one image; additional garments
    are sent as extra reference images in the same prompt.
    """
    if MOCK:
        return person_image

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required when AI_TRYON_MOCK=0")

    if not clothing_images:
        raise RuntimeError("at least one clothing image is required")

    parts = [{"text": PROMPT}, _inline_image_part(person_image)]
    parts.extend(_inline_image_part(image) for image in clothing_images)

    response = requests.post(
        GEMINI_API_URL,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Gemini API returned {response.status_code}: {response.text}") from exc

    return _extract_image(response.json())


def generate_back_view_image(person_front_image: bytes) -> bytes:
    """Returns image bytes (PNG/JPEG) of a generated back/rear view of the
    same person shown in `person_front_image`, so callers that only have a
    front-facing photo (e.g. the mobile app's "Experimental AI 3D fitting"
    flow, which only asks the user for one photo of themselves) can still
    feed a (front, back) pair into a two-sided pipeline.

    Mock mode (`AI_TRYON_MOCK` default "1"): returns the front image
    unchanged (same "identity" convention as `generate_tryon_image`) — no
    network call, no GEMINI_API_KEY needed.
    """
    if MOCK:
        return person_front_image

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required when AI_TRYON_MOCK=0")

    response = requests.post(
        GEMINI_API_URL,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{
                "role": "user",
                "parts": [{"text": BACK_VIEW_PROMPT}, _inline_image_part(person_front_image)],
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Gemini API returned {response.status_code}: {response.text}") from exc

    return _extract_image(response.json())


NORMALIZE_PROMPT = (
    "Using the given image as the reference person, generate one "
    "photorealistic, front-facing, FULL-BODY image of the exact same "
    "person, re-framed and re-lit for a virtual try-on pipeline that needs "
    "a clean pose to work. Preserve the person's identity, face, skin "
    "tone, hair, build, and current clothing exactly. Requirements: whole "
    "head to feet visible with margin above the head and below the feet, "
    "no cropping of any body part, person centered and filling most of "
    "the frame, arms relaxed at the sides (not crossed, not raised, not "
    "occluding the body), simple standing pose facing the camera directly, "
    "plain flat light-gray studio background, even bright lighting, no "
    "other people or animals in frame, roughly 3:4 portrait aspect ratio."
)


def normalize_person_photo(person_image: bytes) -> bytes:
    """Returns image bytes (PNG/JPEG) of `person_image` re-generated as a
    clean, front-facing, full-body studio photo of the SAME person —
    fixing framing/lighting/pose issues (cropped limbs, busy background,
    non-frontal angle, ...) that make real-world phone photos fail a
    virtual-try-on model's own pose-detection step (see
    `tryon_providers.ReplicateVirtualTryOnProvider`, which calls this
    before sending a person photo to Replicate's IDM-VTON).

    Mock mode (`AI_TRYON_MOCK` default "1"): returns the image unchanged
    (same "identity" convention as `generate_tryon_image`) — no network
    call, no GEMINI_API_KEY needed.
    """
    if MOCK:
        return person_image

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required when AI_TRYON_MOCK=0")

    response = requests.post(
        GEMINI_API_URL,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{
                "role": "user",
                "parts": [{"text": NORMALIZE_PROMPT}, _inline_image_part(person_image)],
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        },
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Gemini API returned {response.status_code}: {response.text}") from exc

    return _extract_image(response.json())


def _extract_image(data: dict) -> bytes:
    """Extracts the first inline image from a Gemini `generateContent`
    response body, raising a clear `RuntimeError` if the prompt was blocked,
    generation stopped early, or the response contains no image part.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini blocked the prompt: {block_reason}")
        raise RuntimeError("Gemini response contained no candidates")

    texts = []
    for candidate in candidates:
        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason not in {"STOP", "MAX_TOKENS"}:
            raise RuntimeError(f"Gemini generation stopped with finishReason={finish_reason}")
        for part in candidate.get("content", {}).get("parts", []):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                return base64.b64decode(inline_data["data"])
            if part.get("text"):
                texts.append(part["text"])

    if texts:
        raise RuntimeError(f"Gemini returned text instead of an image: {' '.join(texts)}")
    raise RuntimeError("Gemini response contained no image")


def _inline_image_part(image_bytes: bytes) -> dict:
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }
