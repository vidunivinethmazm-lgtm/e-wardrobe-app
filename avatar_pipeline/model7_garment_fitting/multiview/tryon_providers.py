"""EXPERIMENTAL — virtual try-on step of the `multiview_tryon` pipeline:
turns (person front/back photo, garment front/back photo) into (person
wearing that garment, front photo; same, back photo). Framed strictly as
*virtual try-on image generation*; the 3D reconstruction/fitting happens in
later stages of `multiview/pipeline.py`.

`VIRTUAL_TRYON_PROVIDER` (default "mock"):
- mock: `MockVirtualTryOnProvider` returns the two person photos unchanged
  (no network call needed) — same "identity" mock convention as
  `server/ai_tryon/gemini_client.py`'s `AI_TRYON_MOCK` path.
- idm_vton: `IDMVTonProvider` calls a real IDM-VTON inference backend. Two
  ways to configure it (see each's docstring below):
    - `IDM_VTON_ENDPOINT` — a custom, self-hosted HTTP service using this
      module's own JSON contract (for anyone deploying their own inference
      server, e.g. behind a paid GPU host).
    - `IDM_VTON_HF_SPACE` (default: the free public `yisol/IDM-VTON`
      community Space) — calls that Gradio Space directly via
      `gradio_client`, no self-hosting or payment required. This is the
      default real backend once `VIRTUAL_TRYON_PROVIDER=idm_vton` is set
      with nothing else configured. That Space runs on HuggingFace's
      ZeroGPU, which grants real GPU quota only to authenticated calls —
      set `IDM_VTON_HF_TOKEN` (a free HF account's read token from
      https://huggingface.co/settings/tokens) or anonymous calls will
      routinely fail with "AcceleratorError", and the free daily ZeroGPU
      quota can simply run out from repeated testing.
- gemini: `GeminiVirtualTryOnProvider` reuses the EXISTING `server.ai_tryon.
  gemini_client` (already used by the older `/api/ai-tryon` endpoint) —
  no new integration, just a different entry point into the same Gemini
  ("Nano Banana") image-generation client. A pragmatic alternative when
  IDM-VTON's free Space is unavailable/quota-exhausted. Requires
  `GEMINI_API_KEY` (see https://ai.google.dev/) AND `AI_TRYON_MOCK=0` —
  if `AI_TRYON_MOCK` is left at its default, this provider reports itself
  unconfigured (`None`) rather than silently echoing the person photos.
- replicate: `ReplicateVirtualTryOnProvider` calls the hosted `cuuupid/
  idm-vton` model on Replicate (https://replicate.com/cuuupid/idm-vton),
  a pay-per-use API — requires `REPLICATE_API_TOKEN` from
  https://replicate.com/account/api-tokens. No self-hosting, no free-tier
  GPU queue like the HF Space option, but billed per call.

A real provider returns `None` only when unconfigured — callers must treat
that as "cannot proceed", never as an empty/placeholder image. A
configured-but-failing call raises `GarmentFittingError` instead of
silently falling back to mock output (see `garment_mesh_generation.py`'s
`Unique3DGarmentMeshProvider` for the same convention).
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from ..fitting_types import GarmentFittingError

IDM_VTON_ENDPOINT = os.environ.get("IDM_VTON_ENDPOINT")
IDM_VTON_API_KEY = os.environ.get("IDM_VTON_API_KEY")
IDM_VTON_TIMEOUT_S = float(os.environ.get("IDM_VTON_TIMEOUT_S", "120"))

# Free, public, community-run Gradio Spaces — no API key/payment required.
# The primary Space is set via IDM_VTON_HF_SPACE (default "yisol/IDM-VTON").
# If that Space is down/overloaded, the provider automatically tries each
# fallback Space in IDM_VTON_HF_FALLBACKS (comma-separated) before failing.
# Explicitly set IDM_VTON_HF_SPACE="" to disable all HF Space backends and
# force IDM_VTON_ENDPOINT-only behavior.
IDM_VTON_HF_SPACE = os.environ.get("IDM_VTON_HF_SPACE", "yisol/IDM-VTON")

# ZeroGPU Spaces (yisol/IDM-VTON included) grant real GPU quota only to
# authenticated calls — anonymous gradio_client requests routinely fail with
# "AcceleratorError" (no/near-zero GPU quota). Still 100% free: create a
# free HuggingFace account, generate a read token at
# https://huggingface.co/settings/tokens, and set this env var.
IDM_VTON_HF_TOKEN = os.environ.get("IDM_VTON_HF_TOKEN") or os.environ.get("HF_TOKEN")

_IDM_VTON_HF_FALLBACKS_ENV = os.environ.get("IDM_VTON_HF_FALLBACKS", "")
IDM_VTON_HF_FALLBACKS = [
    s.strip() for s in _IDM_VTON_HF_FALLBACKS_ENV.split(",") if s.strip()
] if _IDM_VTON_HF_FALLBACKS_ENV else [
    # Community-maintained forks that typically track the original API shape.
    # Add more as needed; set IDM_VTON_HF_FALLBACKS env var to override.
]

GARMENT_DESCRIPTION_BY_TYPE = {
    "upper_body": "a shirt",
    "lower_body": "pants",
    "dress": "a dress",
}

# This codebase's internal garment_type values ("dress") vs. Replicate's
# cuuupid/idm-vton `category` input, which only accepts "upper_body" |
# "lower_body" | "dresses" (plural) — mismatch raises a 422 from Replicate's
# own input validation ("category must be one of the following...").
REPLICATE_CATEGORY_BY_GARMENT_TYPE = {
    "upper_body": "upper_body",
    "lower_body": "lower_body",
    "dress": "dresses",
}

# Hosted, pay-per-use Replicate backend — no self-hosting, no ZeroGPU queue
# contention, but requires a Replicate account + billing (small per-call
# cost). https://replicate.com/cuuupid/idm-vton
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_IDM_VTON_MODEL = os.environ.get("REPLICATE_IDM_VTON_MODEL", "cuuupid/idm-vton")
# Pin a specific version hash for reproducibility; unset uses the model's
# latest version automatically.
REPLICATE_IDM_VTON_VERSION = os.environ.get("REPLICATE_IDM_VTON_VERSION")
REPLICATE_TIMEOUT_S = float(os.environ.get("REPLICATE_TIMEOUT_S", "180"))
REPLICATE_POLL_INTERVAL_S = float(os.environ.get("REPLICATE_POLL_INTERVAL_S", "2"))
REPLICATE_API_BASE = "https://api.replicate.com/v1"

VIRTUAL_TRYON_PROVIDER = os.environ.get("VIRTUAL_TRYON_PROVIDER", "mock")


class VirtualTryOnProvider(ABC):
    """Interface for virtual try-on image generation. Never responsible for
    3D reconstruction — see `mesh3d_providers.py` for that."""

    @abstractmethod
    def generate(
        self,
        person_front: np.ndarray,
        person_back: np.ndarray,
        garment_front: np.ndarray,
        garment_back: np.ndarray,
        garment_type: str,
    ) -> tuple[bytes, bytes] | None:
        """Returns (front_tryon_png, back_tryon_png), or `None` only when
        this provider isn't configured/available at all."""


# ── Mock provider (non-production) ───────────────────────────────────────

class MockVirtualTryOnProvider(VirtualTryOnProvider):
    """NON-PRODUCTION / TESTS-DEV-ONLY stand-in: returns the two person
    photos completely unchanged, so the rest of the pipeline (segmentation,
    mesh generation, fitting) can be exercised without an IDM-VTON
    deployment. Never actually depicts the person wearing the uploaded
    garment — callers must treat this pipeline's `is_real_3d_generation`/
    `virtual_tryon_provider` metadata as authoritative, never assume the
    preview images show a real try-on result."""

    def generate(self, person_front, person_back, garment_front, garment_back, garment_type):
        return _encode_png(person_front), _encode_png(person_back)


# ── Real provider: Gemini (reuses existing server.ai_tryon.gemini_client) ──

class GeminiVirtualTryOnProvider(VirtualTryOnProvider):
    """Real virtual try-on provider — reuses the existing `server.ai_tryon.
    gemini_client.generate_tryon_image` (already used by `/api/ai-tryon`)
    rather than a new integration. Calls it once for the front pair
    (person_front + garment_front), once for the back pair.

    Returns `None` unless BOTH `GEMINI_API_KEY` is set AND `AI_TRYON_MOCK=0`
    — `gemini_client` has its own mock flag (default on) that would
    otherwise silently echo the person photos back, which this provider
    must never present as a real try-on result. A configured-but-failing
    call raises `GarmentFittingError`.
    """

    def generate(self, person_front, person_back, garment_front, garment_back, garment_type):
        from server.ai_tryon import gemini_client

        if gemini_client.MOCK or not gemini_client.GEMINI_API_KEY:
            return None

        try:
            front_png = gemini_client.generate_tryon_image(_encode_png(person_front), [_encode_png(garment_front)])
            back_png = gemini_client.generate_tryon_image(_encode_png(person_back), [_encode_png(garment_back)])
        except Exception as exc:
            raise GarmentFittingError(f"Gemini virtual try-on request failed: {exc}") from exc

        return front_png, back_png


# ── Real provider ─────────────────────────────────────────────────────────

class IDMVTonProvider(VirtualTryOnProvider):
    """Real virtual try-on provider. Backends tried in this order:

    1. `IDM_VTON_ENDPOINT` (custom, self-hosted): sends both person photos
       and both garment photos in one HTTP POST using this module's own
       JSON contract, for anyone deploying their own inference server.
    2. `IDM_VTON_HF_SPACE` + `IDM_VTON_HF_FALLBACKS` (HuggingFace Gradio
       Spaces, via `gradio_client`): the primary Space is called once for
       each side (front pair, back pair). If that Space is down/overloaded/
       rate-limited, each fallback Space is tried in turn. Only raises
       `GarmentFittingError` when ALL configured Spaces have been tried and
       failed — see `_try_hf_spaces`.

    Returns `None` only when neither backend is configured. A configured-
    but-failing call raises `GarmentFittingError` — never a silent downgrade
    to the mock identity output.
    """

    _HF_SPACE_UNSET = object()

    def __init__(
        self, endpoint: str | None = None, hf_space: str | None = _HF_SPACE_UNSET,
        hf_fallbacks: list[str] | None = None, timeout_s: float | None = None,
    ):
        self.endpoint = endpoint or IDM_VTON_ENDPOINT
        self.hf_space = IDM_VTON_HF_SPACE if hf_space is self._HF_SPACE_UNSET else hf_space
        self.hf_fallbacks = IDM_VTON_HF_FALLBACKS if hf_fallbacks is None else hf_fallbacks
        self.timeout_s = timeout_s or IDM_VTON_TIMEOUT_S
        # Cached result from the most recent successful HF Space call.
        self._cached_result: tuple[bytes, bytes] | None = None

    def generate(self, person_front, person_back, garment_front, garment_back, garment_type):
        if self.endpoint:
            return self._generate_via_custom_endpoint(
                person_front, person_back, garment_front, garment_back, garment_type,
            )
        # Collect all Spaces to try: primary + fallbacks, skipping empty ones.
        spaces = [s for s in ([self.hf_space] + self.hf_fallbacks) if s]
        if spaces:
            return self._try_hf_spaces(spaces, person_front, person_back, garment_front, garment_back, garment_type)
        return None

    # ── Custom endpoint ─────────────────────────────────────────────────

    def _generate_via_custom_endpoint(self, person_front, person_back, garment_front, garment_back, garment_type):
        import requests

        payload = {
            "garment_type": garment_type,
            "person_front_png_base64": _encode_png(person_front),
            "person_back_png_base64": _encode_png(person_back),
            "garment_front_png_base64": _encode_png(garment_front),
            "garment_back_png_base64": _encode_png(garment_back),
        }
        headers = {"Authorization": f"Bearer {IDM_VTON_API_KEY}"} if IDM_VTON_API_KEY else {}
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise GarmentFittingError(f"IDM-VTON virtual try-on request failed: {exc}") from exc

        try:
            front_png = base64.b64decode(body["front_tryon_png_base64"])
            back_png = base64.b64decode(body["back_tryon_png_base64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GarmentFittingError(f"IDM-VTON response missing/invalid try-on images: {exc}") from exc

        return front_png, back_png

    # ── HuggingFace Spaces with fallback ────────────────────────────────

    def _try_hf_spaces(self, spaces, person_front, person_back, garment_front, garment_back, garment_type):
        """Try each HF Space in order. If one succeeds, return its result.
        If ALL fail, raise `GarmentFittingError` listing every Space that
        was attempted and the first error from each."""
        garment_description = GARMENT_DESCRIPTION_BY_TYPE.get(garment_type, "a garment")
        errors: list[tuple[str, str]] = []

        for space in spaces:
            err = self._try_single_hf_space(space, person_front, person_back, garment_front, garment_back, garment_description)
            if err is None:
                return self._cached_result  # type: ignore[has-type]
            errors.append((space, err))

        # All Spaces failed — build a descriptive error.
        lines = [f"  {s!r}: {e}" for s, e in errors]
        raise GarmentFittingError(
            f"all {len(spaces)} configured IDM-VTON HuggingFace Space(s) failed:\n"
            + "\n".join(lines)
            + "\n\nSet IDM_VTON_ENDPOINT for a self-hosted backend, or use VIRTUAL_TRYON_PROVIDER=mock for local dev."
        )

    def _try_single_hf_space(self, space, person_front, person_back, garment_front, garment_back, garment_description):
        """Try ONE HF Space. Returns `None` on success (result stored in
        `self._cached_result`), or an error string on failure. Never raises
        — the caller (`_try_hf_spaces`) collects errors and decides when to
        fail."""
        try:
            from gradio_client import Client
        except ImportError as exc:
            return f"gradio_client not installed ({exc})"

        try:
            client = Client(space, token=IDM_VTON_HF_TOKEN) if IDM_VTON_HF_TOKEN else Client(space)
        except Exception as exc:
            return f"could not connect: {exc}"

        try:
            front_png = self._call_hf_space(client, space, person_front, garment_front, garment_description)
            back_png = self._call_hf_space(client, space, person_back, garment_back, garment_description)
        except Exception as exc:
            return f"request failed: {exc}"

        self._cached_result = (front_png, back_png)
        return None

    def _call_hf_space(self, client, space_name: str, person_rgb: np.ndarray, garment_rgb: np.ndarray, garment_description: str) -> bytes:
        from gradio_client import handle_file

        with tempfile.TemporaryDirectory() as tmp:
            person_path = Path(tmp) / "person.png"
            garment_path = Path(tmp) / "garment.png"
            Image.fromarray(person_rgb).save(person_path)
            Image.fromarray(garment_rgb).save(garment_path)

            # Verified against the live Space's own `Client.view_api()`
            # output (the authoritative source — parsing app.py source via
            # summarization got parameter names wrong): `dict`/`garm_img`/
            # `garment_des` are correct.
            result = client.predict(
                dict={"background": handle_file(str(person_path)), "layers": [], "composite": None},
                garm_img=handle_file(str(garment_path)),
                garment_des=garment_description,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon",
            )

        try:
            output_path = result[0]
        except (TypeError, IndexError, KeyError) as exc:
            raise GarmentFittingError(
                f"HF Space {space_name!r} returned an unexpected response shape ({exc}) — "
                "this public demo's API can change; consider self-hosting via IDM_VTON_ENDPOINT"
            ) from exc

        return _encode_png(np.array(Image.open(output_path).convert("RGB")))


# ── Real provider: Replicate (hosted `cuuupid/idm-vton`) ────────────────────

class ReplicateVirtualTryOnProvider(VirtualTryOnProvider):
    """Real virtual try-on provider — calls the hosted `cuuupid/idm-vton`
    model on Replicate (https://replicate.com/cuuupid/idm-vton) over its
    REST API. One prediction for the front pair (person_front + garment
    front), one for the back pair.

    Requires `REPLICATE_API_TOKEN` (a free Replicate account's API token
    from https://replicate.com/account/api-tokens; running predictions
    still costs a small per-call fee billed to that account — this is not a
    free-quota backend like the HF Space option). Returns `None` only when
    unconfigured (no token). A configured-but-failing call raises
    `GarmentFittingError`, never falling back to mock output.
    """

    def __init__(
        self, api_token: str | None = None, model: str | None = None,
        version: str | None = None, timeout_s: float | None = None,
    ):
        self.api_token = api_token or REPLICATE_API_TOKEN
        self.model = model or REPLICATE_IDM_VTON_MODEL
        self.version = version if version is not None else REPLICATE_IDM_VTON_VERSION
        self.timeout_s = timeout_s or REPLICATE_TIMEOUT_S
        # Resolved lazily by _resolve_version and cached for the life of
        # this provider instance (one `generate()` call makes two
        # predictions — front + back — no need to re-resolve between them).
        self._resolved_version: str | None = None

    def generate(self, person_front, person_back, garment_front, garment_back, garment_type):
        if not self.api_token:
            return None

        garment_description = GARMENT_DESCRIPTION_BY_TYPE.get(garment_type, "a garment")
        try:
            front_png = self._run(person_front, garment_front, garment_description, garment_type, "front")
            back_png = self._run(person_back, garment_back, garment_description, garment_type, "back")
        except GarmentFittingError:
            raise
        except Exception as exc:
            raise GarmentFittingError(f"Replicate IDM-VTON request failed: {exc}") from exc

        return front_png, back_png

    def _run(self, person_rgb: np.ndarray, garment_rgb: np.ndarray, garment_description: str, garment_type: str, side: str) -> bytes:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # ask Replicate to hold the connection until done, when it can
        }
        payload = {
            "input": {
                "human_img": _encode_data_uri(person_rgb),
                "garm_img": _encode_data_uri(garment_rgb),
                "garment_des": garment_description,
                "category": REPLICATE_CATEGORY_BY_GARMENT_TYPE.get(garment_type, garment_type),
                # cuuupid/idm-vton's own pose/parsing preprocessing throws
                # "list index out of range" (an IndexError deep inside the
                # model) when it can't confidently locate the person —
                # cropped/off-center photos or anything not ~3:4 trigger
                # this. `crop=True` asks the model to auto-detect and crop
                # to the person first, which fixes the vast majority of
                # these failures ("check this if your image is not 3:4",
                # per the model's own input schema).
                "crop": True,
            },
            "version": self._get_version(headers),
        }

        # The `/models/{owner}/{name}/predictions` shortcut only works for
        # models Replicate has deployed with dedicated hardware — a
        # community model like `cuuupid/idm-vton` 404s there. The versioned
        # `/predictions` endpoint works for any public model, so always use
        # that (with an explicitly resolved version id).
        url = f"{REPLICATE_API_BASE}/predictions"

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise GarmentFittingError(f"Replicate prediction request failed ({response.status_code}): {response.text}") from exc

        prediction = self._poll_until_done(response.json(), headers, side)
        return self._extract_image(prediction)

    def _get_version(self, headers: dict) -> str:
        """Returns a pinned version id — either explicitly configured
        (`REPLICATE_IDM_VTON_VERSION`) or resolved once from `self.model`'s
        current `latest_version` via `GET /v1/models/{model}` and cached."""
        import requests

        if self.version:
            return self.version
        if self._resolved_version:
            return self._resolved_version

        response = requests.get(f"{REPLICATE_API_BASE}/models/{self.model}", headers=headers, timeout=30)
        try:
            response.raise_for_status()
            version = response.json()["latest_version"]["id"]
        except Exception as exc:
            raise GarmentFittingError(f"could not resolve Replicate model {self.model!r}'s latest version: {exc}") from exc

        self._resolved_version = version
        return version

    def _poll_until_done(self, prediction: dict, headers: dict, side: str) -> dict:
        import time

        import requests

        get_url = prediction.get("urls", {}).get("get")
        deadline = time.monotonic() + self.timeout_s
        while prediction.get("status") not in ("succeeded", "failed", "canceled"):
            if time.monotonic() > deadline:
                raise GarmentFittingError(f"Replicate prediction ({side}) timed out after {self.timeout_s}s")
            if not get_url:
                raise GarmentFittingError(f"Replicate prediction ({side}) response missing polling URL")
            time.sleep(REPLICATE_POLL_INTERVAL_S)
            poll_response = requests.get(get_url, headers=headers, timeout=30)
            poll_response.raise_for_status()
            prediction = poll_response.json()

        if prediction.get("status") != "succeeded":
            error = prediction.get("error") or prediction.get("status")
            if "list index out of range" in str(error):
                hint = (
                    " This is your auto-generated BACK view (from Gemini) — try uploading your own real back "
                    "photo instead of relying on auto-generation, or check the generated back image looks like "
                    "a normal full-body photo."
                    if side == "back" else
                    " Use a clear, well-lit, front-facing full-body photo (whole head to feet visible, plain "
                    "background, person filling most of the frame) and try again."
                )
                raise GarmentFittingError(
                    f"Replicate's IDM-VTON model could not detect a person in your {side} photo "
                    f"('list index out of range' from its own pose-detection step, even with crop=True).{hint}"
                )
            raise GarmentFittingError(f"Replicate prediction ({side}) did not succeed: {error}")

        return prediction

    def _extract_image(self, prediction: dict) -> bytes:
        import requests

        output = prediction.get("output")
        # cuuupid/idm-vton returns a single image URL string (or a
        # single-element list, depending on API version) — accept either.
        if isinstance(output, list):
            output = output[0] if output else None
        if not output or not isinstance(output, str):
            raise GarmentFittingError(f"Replicate prediction returned an unexpected output shape: {output!r}")

        image_response = requests.get(output, timeout=60)
        image_response.raise_for_status()
        return image_response.content


def _encode_data_uri(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(_encode_png(rgb)).decode("ascii")


def _encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def get_virtual_tryon_provider() -> VirtualTryOnProvider:
    if VIRTUAL_TRYON_PROVIDER == "mock":
        return MockVirtualTryOnProvider()
    if VIRTUAL_TRYON_PROVIDER == "idm_vton":
        return IDMVTonProvider()
    if VIRTUAL_TRYON_PROVIDER == "gemini":
        return GeminiVirtualTryOnProvider()
    if VIRTUAL_TRYON_PROVIDER == "replicate":
        return ReplicateVirtualTryOnProvider()
    raise NotImplementedError(
        f"VIRTUAL_TRYON_PROVIDER={VIRTUAL_TRYON_PROVIDER!r} is not implemented. "
        "Set VIRTUAL_TRYON_PROVIDER=mock, idm_vton, gemini, or replicate."
    )
