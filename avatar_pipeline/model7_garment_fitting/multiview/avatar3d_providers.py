"""EXPERIMENTAL — PIVOT: full-avatar image-to-3D reconstruction step of the
`multiview_tryon` pipeline.

Unlike the original garment-isolation design (`garment_isolation.py`,
`mesh3d_providers.py`, `texture_providers.py` — still present, no longer
called by `multiview/pipeline.py`), this module reconstructs a **full 3D
human avatar** directly from the virtual-try-on image(s) (the person
wearing the uploaded garment) via Unique3D
(https://github.com/AiuniAI/Unique3D), and that mesh becomes the avatar
going forward — it does not get fitted onto the pre-existing MakeHuman
avatar. This intentionally does NOT preserve "existing avatar unchanged";
that tradeoff was explicitly requested and confirmed for this experimental
path only.

`FULL_AVATAR_3D_PROVIDER` (default "mock"):
- mock: `MockFullAvatarImageTo3DProvider` delegates to the existing
  `garment_mesh_generation.MockGarmentMeshProvider` ("dress" category shell,
  the roughest full-body-shaped placeholder already available) — no network
  call, always `is_mock=True`.
- unique3d: `Unique3DAvatarProvider` calls a real Unique3D backend. Two ways
  to configure it, same pattern as `tryon_providers.IDMVTonProvider`:
    - `UNIQUE3D_ENDPOINT` — a custom, self-hosted HTTP service using this
      module's own JSON contract.
    - `UNIQUE3D_HF_SPACE` (default: the public `Wuvin/Unique3D` community
      Space) + `UNIQUE3D_HF_FALLBACKS` — called via `gradio_client`. Like
      `yisol/IDM-VTON`, this runs on HuggingFace's ZeroGPU, so
      `UNIQUE3D_HF_TOKEN` (or `HF_TOKEN`) is required in practice or
      anonymous calls fail with `AcceleratorError`.

Unique3D's public demo reconstructs from a SINGLE reference image — `back_rgb`
is accepted for interface symmetry with the rest of the pipeline but is not
sent to this specific integration (there is no known single-call multi-view
variant of the public Space); only the front try-on image drives generation.

The Gradio call shape for `Wuvin/Unique3D` is verified against that Space's
own source (`gradio_app/gradio_3dgen.py`, the `fullrunv2_btn` handler,
`api_name="generate3dv2"`) — see `_call_hf_space`. A community-run fork
added via `UNIQUE3D_HF_FALLBACKS` may still differ if its owner changed
the interface; that raises a clear error rather than misparsing.

A real provider returns `None` only when unconfigured. A configured-but-
failing call raises `GarmentFittingError` — never a silent downgrade to the
mock placeholder.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..fitting_types import GarmentFittingError
from ..garment_mesh_generation import MockGarmentMeshProvider

UNIQUE3D_ENDPOINT = os.environ.get("UNIQUE3D_AVATAR_ENDPOINT")
UNIQUE3D_API_KEY = os.environ.get("UNIQUE3D_AVATAR_API_KEY")
UNIQUE3D_TIMEOUT_S = float(os.environ.get("UNIQUE3D_AVATAR_TIMEOUT_S", "180"))

UNIQUE3D_HF_SPACE = os.environ.get("UNIQUE3D_HF_SPACE", "Wuvin/Unique3D")
UNIQUE3D_HF_TOKEN = os.environ.get("UNIQUE3D_HF_TOKEN") or os.environ.get("HF_TOKEN")
_UNIQUE3D_HF_FALLBACKS_ENV = os.environ.get("UNIQUE3D_HF_FALLBACKS", "")
UNIQUE3D_HF_FALLBACKS = [s.strip() for s in _UNIQUE3D_HF_FALLBACKS_ENV.split(",") if s.strip()]

FULL_AVATAR_3D_PROVIDER = os.environ.get("FULL_AVATAR_3D_PROVIDER", "mock")


@dataclass(frozen=True)
class GeneratedAvatarMesh:
    """A full reconstructed human avatar mesh — deliberately no landmark
    contract (unlike `garment_mesh_generation.GeneratedGarmentMesh`), since
    this pipeline no longer fits anything onto a separate existing avatar;
    this mesh *is* the avatar."""

    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) uint32
    uvs: np.ndarray | None  # (N, 2) float32
    texture_png: bytes | None
    is_mock: bool


class FullAvatarImageTo3DProvider(ABC):
    """Interface for full-avatar image-to-3D reconstruction from virtual-
    try-on image(s). Never responsible for fitting onto a separate avatar —
    there isn't one in this path."""

    @abstractmethod
    def generate(self, front_rgb: np.ndarray, back_rgb: np.ndarray) -> GeneratedAvatarMesh | None:
        """Returns `None` only when this provider isn't configured/available
        at all. A configured provider that fails to produce a usable mesh
        must raise instead."""


# ── Mock provider (non-production) ───────────────────────────────────────

class MockFullAvatarImageTo3DProvider(FullAvatarImageTo3DProvider):
    """NON-PRODUCTION / TESTS-DEV-ONLY stand-in. Reuses `garment_mesh_
    generation.MockGarmentMeshProvider`'s "dress" category shell (the
    roughest full-body-shaped procedural placeholder already available)
    rather than writing new procedural geometry. Always `is_mock=True`."""

    def generate(self, front_rgb, back_rgb) -> GeneratedAvatarMesh:
        mesh = MockGarmentMeshProvider().generate(front_rgb, back_rgb, "dress")
        return GeneratedAvatarMesh(
            vertices=mesh.vertices, faces=mesh.faces, uvs=mesh.uvs, texture_png=mesh.texture_png, is_mock=True,
        )


# ── Real provider ─────────────────────────────────────────────────────────

class Unique3DAvatarProvider(FullAvatarImageTo3DProvider):
    """Real full-avatar image-to-3D provider. Backends tried in this order:

    1. `UNIQUE3D_ENDPOINT` (custom, self-hosted): POSTs the front try-on
       image using this module's own JSON contract.
    2. `UNIQUE3D_HF_SPACE` + `UNIQUE3D_HF_FALLBACKS` (public Gradio Spaces,
       via `gradio_client`): tried in order; only raises `GarmentFittingError`
       once every configured Space has failed.

    Returns `None` only when neither backend is configured.
    """

    _HF_SPACE_UNSET = object()

    def __init__(
        self, endpoint: str | None = None, hf_space: str | None = _HF_SPACE_UNSET,
        hf_fallbacks: list[str] | None = None, timeout_s: float | None = None,
    ):
        self.endpoint = endpoint or UNIQUE3D_ENDPOINT
        self.hf_space = UNIQUE3D_HF_SPACE if hf_space is self._HF_SPACE_UNSET else hf_space
        self.hf_fallbacks = UNIQUE3D_HF_FALLBACKS if hf_fallbacks is None else hf_fallbacks
        self.timeout_s = timeout_s or UNIQUE3D_TIMEOUT_S
        self._cached_result: GeneratedAvatarMesh | None = None

    def generate(self, front_rgb, back_rgb):
        if self.endpoint:
            return self._generate_via_custom_endpoint(front_rgb, back_rgb)
        spaces = [s for s in ([self.hf_space] + self.hf_fallbacks) if s]
        if spaces:
            return self._try_hf_spaces(spaces, front_rgb)
        return None

    # ── Custom endpoint ─────────────────────────────────────────────────

    def _generate_via_custom_endpoint(self, front_rgb, back_rgb):
        import requests

        payload = {"front_png_base64": _encode_png(front_rgb)}
        headers = {"Authorization": f"Bearer {UNIQUE3D_API_KEY}"} if UNIQUE3D_API_KEY else {}
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise GarmentFittingError(f"Unique3D avatar reconstruction request failed: {exc}") from exc

        try:
            vertices = np.asarray(body["vertices"], dtype=np.float32)
            faces = np.asarray(body["faces"], dtype=np.uint32)
            uvs = np.asarray(body.get("uvs", []), dtype=np.float32)
            if uvs.shape != vertices[:, :2].shape:
                uvs = None
            texture_png = base64.b64decode(body["texture_png_base64"]) if body.get("texture_png_base64") else None
        except (KeyError, TypeError, ValueError) as exc:
            raise GarmentFittingError(f"Unique3D response missing/invalid mesh data: {exc}") from exc

        return GeneratedAvatarMesh(vertices=vertices, faces=faces, uvs=uvs, texture_png=texture_png, is_mock=False)

    # ── HuggingFace Spaces with fallback ────────────────────────────────

    def _try_hf_spaces(self, spaces, front_rgb):
        errors: list[tuple[str, str]] = []
        for space in spaces:
            err = self._try_single_hf_space(space, front_rgb)
            if err is None:
                return self._cached_result
            errors.append((space, err))

        lines = [f"  {s!r}: {e}" for s, e in errors]
        raise GarmentFittingError(
            f"all {len(spaces)} configured Unique3D HuggingFace Space(s) failed:\n"
            + "\n".join(lines)
            + "\n\nSet UNIQUE3D_AVATAR_ENDPOINT for a self-hosted backend, "
            "or use FULL_AVATAR_3D_PROVIDER=mock for local dev."
        )

    def _try_single_hf_space(self, space, front_rgb) -> str | None:
        try:
            from gradio_client import Client
        except ImportError as exc:
            return f"gradio_client not installed ({exc})"

        try:
            client = Client(space, token=UNIQUE3D_HF_TOKEN) if UNIQUE3D_HF_TOKEN else Client(space)
        except Exception as exc:
            return f"could not connect: {exc}"

        try:
            self._cached_result = self._call_hf_space(client, space, front_rgb)
        except Exception as exc:
            return f"request failed: {exc}"

        return None

    def _call_hf_space(self, client, space_name: str, front_rgb: np.ndarray) -> GeneratedAvatarMesh:
        from gradio_client import handle_file

        with tempfile.TemporaryDirectory() as tmp:
            front_path = Path(tmp) / "front.png"
            Image.fromarray(front_rgb).save(front_path)

            # Verified against the live Space's own `Client.view_api()`
            # output (the authoritative source — parsing app.py source via
            # summarization got the parameter name wrong, same lesson as
            # IDM-VTON's `dict`/`garment_des`): the image param is
            # `preview_img`, not `input_image`. Returns (mesh_model_path,
            # preview_video_dict).
            result = client.predict(
                preview_img=handle_file(str(front_path)),
                input_processing=True,
                seed=42,
                render_video=False,
                do_refine=True,
                expansion_weight=0.1,
                init_type="std",
                api_name="/generate3dv2",
            )

        output_path = self._extract_glb_path(result, space_name)
        vertices, faces, uvs, texture_png = _read_glb_mesh(output_path)
        return GeneratedAvatarMesh(vertices=vertices, faces=faces, uvs=uvs, texture_png=texture_png, is_mock=False)

    @staticmethod
    def _extract_glb_path(result, space_name: str) -> str:
        # Best-effort: most single-output Gradio demos return either the
        # path directly or a 1-tuple/list containing it.
        if isinstance(result, (tuple, list)):
            if not result:
                raise GarmentFittingError(
                    f"Unique3D HF Space {space_name!r} returned an empty response — "
                    "this public demo's API can change; consider self-hosting via UNIQUE3D_AVATAR_ENDPOINT"
                )
            return result[0]
        if isinstance(result, str):
            return result
        raise GarmentFittingError(
            f"Unique3D HF Space {space_name!r} returned an unexpected response shape "
            f"({type(result).__name__}) — this public demo's API can change; "
            "consider self-hosting via UNIQUE3D_AVATAR_ENDPOINT"
        )


def _read_glb_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bytes | None]:
    """Reads back a single-mesh GLB (as produced by `Unique3DAvatarProvider`'s
    HF Space call) into (vertices, faces, uvs, texture_png), mirroring what
    `glb_writer.write_mesh_glb` writes."""
    from pygltflib import GLTF2

    try:
        gltf = GLTF2.load(path)
    except Exception as exc:
        raise GarmentFittingError(f"could not read Unique3D's returned mesh file as a GLB: {exc}") from exc

    blob = gltf.binary_blob()
    if blob is None or not gltf.meshes:
        raise GarmentFittingError("Unique3D's returned GLB has no mesh data")

    prim = gltf.meshes[0].primitives[0]

    def _read_accessor(index: int, dtype: str, components: int) -> np.ndarray:
        accessor = gltf.accessors[index]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        offset = buffer_view.byteOffset or 0
        length = buffer_view.byteLength
        return np.frombuffer(blob[offset:offset + length], dtype=dtype).reshape(-1, components)

    try:
        vertices = _read_accessor(prim.attributes.POSITION, "<f4", 3).astype(np.float32)
        idx_accessor = gltf.accessors[prim.indices]
        idx_bv = gltf.bufferViews[idx_accessor.bufferView]
        idx_offset = idx_bv.byteOffset or 0
        idx_dtype = "<u4" if idx_accessor.componentType == 5125 else "<u2"
        faces = np.frombuffer(
            blob[idx_offset:idx_offset + idx_bv.byteLength], dtype=idx_dtype,
        ).astype(np.uint32).reshape(-1, 3)
    except (AttributeError, IndexError, TypeError) as exc:
        raise GarmentFittingError(f"Unique3D's returned GLB is missing expected mesh attributes: {exc}") from exc

    uvs = None
    uv_index = getattr(prim.attributes, "TEXCOORD_0", None)
    if uv_index is not None:
        uvs = _read_accessor(uv_index, "<f4", 2).astype(np.float32)

    texture_png = None
    if gltf.images:
        img = gltf.images[0]
        if img.bufferView is not None:
            img_bv = gltf.bufferViews[img.bufferView]
            img_offset = img_bv.byteOffset or 0
            texture_png = bytes(blob[img_offset:img_offset + img_bv.byteLength])

    return vertices, faces, uvs, texture_png


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_full_avatar_provider() -> FullAvatarImageTo3DProvider:
    if FULL_AVATAR_3D_PROVIDER == "mock":
        return MockFullAvatarImageTo3DProvider()
    if FULL_AVATAR_3D_PROVIDER == "unique3d":
        return Unique3DAvatarProvider()
    raise NotImplementedError(
        f"FULL_AVATAR_3D_PROVIDER={FULL_AVATAR_3D_PROVIDER!r} is not implemented. "
        "Set FULL_AVATAR_3D_PROVIDER=mock or unique3d."
    )
