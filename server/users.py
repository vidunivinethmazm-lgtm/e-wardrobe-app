"""Lightweight on-disk storage for emails collected during onboarding
(`POST /api/users/email`, see `server/app.py`). Appends one JSON object per
line to `server/data/emails.jsonl`."""

import json
from datetime import datetime, timezone
from pathlib import Path

EMAILS_FILE = Path(__file__).resolve().parent / "data" / "emails.jsonl"


def save_email(email: str) -> None:
    EMAILS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"email": email, "submitted_at": datetime.now(timezone.utc).isoformat()}
    with EMAILS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
