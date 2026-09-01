"""Disk-backed session storage for the AI try-on pipeline
(`POST /api/ai-tryon`, `POST /api/ai-tryon/<id>/avatar3d`, see
`server/app.py`).

Mirrors `server/storage.py`'s uuid4().hex id + path-traversal-safe regex.
Each session directory (`server/ai_sessions/<id>/`) holds:
  - generated.png   Gemini/mock 2D try-on output
  - avatar.glb      image-to-3D output, once generated
  - meta.json       clothing photo names (a list, so multiple garments can
                     be added later) + avatar_glb status
"""

import json
import re
import uuid
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "ai_sessions"

_VALID_ID = re.compile(r"^[0-9a-f]{32}$")


def _session_dir(session_id):
    if not _VALID_ID.match(session_id):
        return None
    return SESSIONS_DIR / session_id


def create_session(generated_image: bytes, clothing_photos: list[str]) -> str:
    """Stores `generated_image` (PNG bytes) as a new session and returns its id."""
    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "generated.png").write_bytes(generated_image)
    _write_meta(session_dir, {"clothing_photos": clothing_photos, "avatar_glb": None})

    return session_id


def get_generated_image(session_id: str):
    session_dir = _session_dir(session_id)
    if session_dir is None or not session_dir.is_dir():
        return None
    path = session_dir / "generated.png"
    return path.read_bytes() if path.is_file() else None


def save_avatar_glb(session_id: str, glb_bytes: bytes) -> bool:
    session_dir = _session_dir(session_id)
    if session_dir is None or not session_dir.is_dir():
        return False

    (session_dir / "avatar.glb").write_bytes(glb_bytes)

    meta = _read_meta(session_dir)
    meta["avatar_glb"] = "avatar.glb"
    _write_meta(session_dir, meta)
    return True


def get_avatar_glb(session_id: str):
    session_dir = _session_dir(session_id)
    if session_dir is None or not session_dir.is_dir():
        return None
    path = session_dir / "avatar.glb"
    return path.read_bytes() if path.is_file() else None


def _read_meta(session_dir: Path) -> dict:
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_meta(session_dir: Path, meta: dict) -> None:
    with (session_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
