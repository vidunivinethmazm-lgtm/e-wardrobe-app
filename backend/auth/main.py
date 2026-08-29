"""Accounts and profiles - mounted by backend/main.py at /auth.

A small, dependency-free auth layer:

* passwords are stored as PBKDF2-SHA256 hashes with a per-user salt
  (``hashlib`` only, no bcrypt/argon2 build step),
* a successful register / login returns an opaque bearer token that maps
  to a user in the shared store (``backend.core.store``),
* the mobile app sends ``Authorization: Bearer <token>`` on every call.

The store keeps ``users`` and ``sessions`` in MongoDB when configured, or
in memory otherwise - same as the rest of the app.
"""

import hashlib
import re
import secrets
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core import store

app = FastAPI(title="E-Wardrobe - Auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ROUNDS = 200_000


# ── helpers ─────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS
    ).hex()


def _public_user(doc: dict) -> dict:
    """User document without the secret fields."""
    return {
        "user_id": doc["user_id"],
        "email": doc["email"],
        "name": doc.get("name", ""),
        "created_at": doc.get("created_at"),
        "profile": {
            "gender": "", "age": None, "city": "", "style": "", "avatar": "👤", "bio": "",
            **(doc.get("profile") or {}),
        },
    }


def current_user(authorization: str | None = Header(default=None)) -> dict:
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user_id = store.get_session_user_id(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Account no longer exists")
    return user


# ── bodies ──────────────────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


class ProfileBody(BaseModel):
    name: str | None = None
    gender: str | None = None
    age: int | None = None
    city: str | None = None
    style: str | None = None
    avatar: str | None = None
    bio: str | None = None


# ── routes ──────────────────────────────────────────────────────────────────

@app.post("/register")
def register(body: RegisterBody):
    email = body.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    if store.get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    salt = secrets.token_hex(16)
    user = store.create_user({
        "user_id": f"u_{uuid4().hex[:12]}",
        "email": email,
        "name": body.name.strip() or email.split("@")[0],
        "password_hash": _hash_password(body.password, salt),
        "salt": salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": {"avatar": "👤"},
    })

    token = secrets.token_urlsafe(32)
    store.create_session(token, user["user_id"])
    return {"token": token, "user": _public_user(user)}


@app.post("/login")
def login(body: LoginBody):
    user = store.get_user_by_email(body.email)
    if user is None or _hash_password(body.password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Wrong email or password")
    token = secrets.token_urlsafe(32)
    store.create_session(token, user["user_id"])
    return {"token": token, "user": _public_user(user)}


@app.get("/me")
def me(user: dict = Depends(current_user)):
    return {"user": _public_user(user)}


_MAX_AVATAR_CHARS = 1_000_000        # ~700 KB image once base64-encoded


@app.patch("/profile")
def update_profile(body: ProfileBody, user: dict = Depends(current_user)):
    # avatar is either an emoji or an inline data: URI (uploaded photo)
    if body.avatar and body.avatar.startswith("data:") and len(body.avatar) > _MAX_AVATAR_CHARS:
        raise HTTPException(status_code=413, detail="Profile picture is too large")
    updated = store.update_user_profile(user["user_id"], body.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"user": _public_user(updated)}


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        store.delete_session(authorization[7:].strip())
    return {"ok": True}
