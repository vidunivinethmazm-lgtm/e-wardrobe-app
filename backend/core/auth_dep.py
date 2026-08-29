"""FastAPI dependency: resolve the bearer token to a user id.

Every feature that reads or writes account-owned data (wardrobe, schedule,
recommendation history, organization) depends on this so the store is only
ever queried for the signed-in account.
"""

from fastapi import Header, HTTPException

from backend.core import store


def current_user_id(authorization: str | None = Header(default=None)) -> str:
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user_id = store.get_session_user_id(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    return user_id
