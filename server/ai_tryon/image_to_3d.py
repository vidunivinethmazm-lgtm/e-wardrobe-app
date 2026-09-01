"""Pluggable image-to-3D provider for the avatar-3D step
(`POST /api/ai-tryon/<id>/avatar3d`, see `server/app.py`).

Converts a single 2D try-on image into a full-body `.glb` avatar via the
create-job -> poll-status -> fetch-result pattern most image-to-3D APIs use.

IMAGE_TO_3D_PROVIDER (default "mock"):
- mock: `MockImage23DProvider` returns a placeholder GLB immediately - no
  network call, no IMAGE_TO_3D_API_KEY needed.
- meshy: `MeshyImage23DProvider` calls the Meshy Image-to-3D API
  (https://docs.meshy.ai/). Requires IMAGE_TO_3D_API_KEY.
- tripo | threedee: not implemented yet. Set IMAGE_TO_3D_API_KEY and
  implement an `Image23DProvider` subclass for the chosen service, then add
  it to `get_provider` below.
"""

import abc
import base64
import os
import uuid
from pathlib import Path

import requests

PROVIDER = os.environ.get("IMAGE_TO_3D_PROVIDER", "mock")
IMAGE_TO_3D_API_KEY = os.environ.get("IMAGE_TO_3D_API_KEY")

_PLACEHOLDER_GLB = Path(__file__).resolve().parent / "placeholder.glb"

MESHY_API_BASE = "https://api.meshy.ai/openapi/v1/image-to-3d"


class Image23DProvider(abc.ABC):
    """create_job -> (poll) get_job_status -> get_result_glb."""

    @abc.abstractmethod
    def create_job(self, image_bytes: bytes) -> str:
        """Starts a 2D-image-to-3D job for `image_bytes` and returns a job id."""

    @abc.abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """Returns "processing", "ready", or "error"."""

    @abc.abstractmethod
    def get_result_glb(self, job_id: str) -> bytes:
        """Returns the generated `.glb` bytes for a "ready" job."""


class MockImage23DProvider(Image23DProvider):
    """Returns a placeholder full-body GLB immediately, so the mobile +
    backend flow can be built and tested without an image-to-3D API key."""

    def create_job(self, image_bytes: bytes) -> str:
        return uuid.uuid4().hex

    def get_job_status(self, job_id: str) -> str:
        return "ready"

    def get_result_glb(self, job_id: str) -> bytes:
        return _PLACEHOLDER_GLB.read_bytes()


class MeshyImage23DProvider(Image23DProvider):
    """Meshy AI Image-to-3D (https://docs.meshy.ai/openapi/image-to-3d).

    create_job: POSTs the image as a base64 data URI to `MESHY_API_BASE` and
    returns Meshy's task id (`result`).
    get_job_status: GETs `MESHY_API_BASE/<task_id>` and maps Meshy's
    PENDING/IN_PROGRESS -> "processing", SUCCEEDED -> "ready", anything else
    (FAILED/CANCELED/EXPIRED) -> "error".
    get_result_glb: re-fetches the task and downloads `model_urls.glb`.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def create_job(self, image_bytes: bytes) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        response = requests.post(
            MESHY_API_BASE,
            headers=self._headers(),
            json={"image_url": data_uri, "enable_pbr": False},
            timeout=60,
        )
        self._raise_for_status(response)

        task_id = response.json().get("result")
        if not task_id:
            raise RuntimeError("Meshy did not return a task id")
        return task_id

    def get_job_status(self, job_id: str) -> str:
        status = self._get_task(job_id).get("status")
        if status in {"PENDING", "IN_PROGRESS"}:
            return "processing"
        if status == "SUCCEEDED":
            return "ready"
        return "error"

    def get_result_glb(self, job_id: str) -> bytes:
        model_urls = self._get_task(job_id).get("model_urls") or {}
        glb_url = model_urls.get("glb")
        if not glb_url:
            raise RuntimeError("Meshy task has no GLB result")

        response = requests.get(glb_url, timeout=120)
        self._raise_for_status(response, service="Meshy GLB download")
        return response.content

    def _get_task(self, job_id: str) -> dict:
        response = requests.get(f"{MESHY_API_BASE}/{job_id}", headers=self._headers(), timeout=30)
        self._raise_for_status(response)
        return response.json()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _raise_for_status(response, service: str = "Meshy API") -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"{service} returned {response.status_code}: {response.text}") from exc


def get_provider() -> Image23DProvider:
    if PROVIDER == "mock":
        return MockImage23DProvider()
    if PROVIDER == "meshy":
        if not IMAGE_TO_3D_API_KEY:
            raise RuntimeError("IMAGE_TO_3D_API_KEY is required when IMAGE_TO_3D_PROVIDER=meshy")
        return MeshyImage23DProvider(IMAGE_TO_3D_API_KEY)
    raise NotImplementedError(
        f"IMAGE_TO_3D_PROVIDER={PROVIDER!r} is not implemented yet. "
        "Set IMAGE_TO_3D_PROVIDER=mock or meshy."
    )
