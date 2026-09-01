"""Disk-backed session storage for `AvatarResult`, keyed by an opaque
`avatar_id`. Lets `/api/avatars` (Phase A) and `/api/avatars/<id>/tryon`
(Phase B) be separate requests without rerunning Models 1-4."""

import re
import uuid
from pathlib import Path

from backend.avatar_pipeline.pipeline_types import load_avatar_result, save_avatar_result

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

# avatar_ids are always uuid4().hex (see `create`); reject anything else so a
# crafted `avatar_id` from the URL (e.g. "../../some/other/path") can't be
# used to read files outside SESSIONS_DIR.
_VALID_ID = re.compile(r"^[0-9a-f]{32}$")


def create(avatar_result):
    avatar_id = uuid.uuid4().hex
    save_avatar_result(avatar_result, SESSIONS_DIR / avatar_id)
    return avatar_id


def get(avatar_id):
    if not _VALID_ID.match(avatar_id):
        return None
    session_dir = SESSIONS_DIR / avatar_id
    if not session_dir.is_dir():
        return None
    return load_avatar_result(session_dir)


def update(avatar_id, avatar_result):
    """Overwrites the session for `avatar_id` (e.g. after
    `face_customization.apply_face_customization`) with `avatar_result`.
    `avatar_id` must already exist — see `get`."""
    if not _VALID_ID.match(avatar_id):
        return None
    session_dir = SESSIONS_DIR / avatar_id
    if not session_dir.is_dir():
        return None
    save_avatar_result(avatar_result, session_dir)
    return avatar_id
