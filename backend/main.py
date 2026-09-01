"""E-Wardrobe AI - parent application.

Each team feature is its own FastAPI app, kept in its own package under
``backend/`` and mounted here under a path prefix. A feature's own
``main.py`` is never edited during integration - only a mount entry is
added to ``FEATURES`` below.

Run from the repository root:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Feature routes are then served under their prefix, e.g. the classification
predict endpoint is ``POST /classification/predict``.

Mounting is defensive: if one feature fails to import (e.g. its extra
dependencies aren't installed yet), it is skipped with a logged warning
and the other features still come up. ``GET /`` reports what mounted.
"""

import importlib
import traceback

from fastapi import FastAPI

app = FastAPI(title="E-Wardrobe AI")

# prefix  ->  "package.module:attribute" of that feature's FastAPI app
FEATURES = {
    "/auth":           "backend.auth.main:app",          # accounts + profiles
    "/wardrobe":       "backend.core.api:app",          # shared wardrobe store
    "/classification": "backend.classification.main:app",
    "/recommendation": "backend.recommendation.main:app",
    "/organization":  "backend.organization.main:app",
    "/visualization": "backend.visualization.main:app",   # avatar + garment try-on (Gemini + Replicate)
}

_mounted: list[str] = []
_failed: dict[str, str] = {}

for prefix, target in FEATURES.items():
    module_name, _, attr = target.partition(":")
    try:
        feature_app = getattr(importlib.import_module(module_name), attr or "app")
        app.mount(prefix, feature_app)
        _mounted.append(prefix)
    except Exception as exc:  # noqa: BLE001 - one bad feature must not sink the rest
        _failed[prefix] = f"{type(exc).__name__}: {exc}"
        print(f"[startup] feature {prefix} not mounted -> {_failed[prefix]}")
        traceback.print_exc()


@app.get("/")
def root():
    return {
        "service": "E-Wardrobe AI",
        "mounted": _mounted,
        "failed": _failed,
    }
