"""E-Wardrobe visualization feature - AI avatar + garment try-on.

Mounted by `backend/main.py` at `/visualization` (see the parent app's
`FEATURES` dict), alongside the team's other features (`/auth`,
`/wardrobe`, `/classification`, `/recommendation`, `/organization`).

Unlike every other feature here, this one is a Flask (WSGI) app, not
FastAPI (ASGI) - it wraps six TensorFlow/Keras avatar-pipeline models plus
Gemini (photo normalization + back-view generation) and Replicate
(IDM-VTON virtual try-on) into 3D avatars, and was built and tested as a
standalone Flask service before this mounting scheme existed. Rewriting
~25 routes onto FastAPI to match the other features isn't worth the risk
for this integration, so `WsgiToAsgi` bridges it instead - `backend.main`
treats it exactly like every other mounted sub-app.

All of this feature's own routes (see `backend/app.py`) are unchanged and
still exist under their original paths - only the mount prefix is new,
e.g. `POST /visualization/api/avatars/<id>/fit-garment`.

Run standalone (unchanged from before this integration):

    python -m backend.app                      # Flask dev server, :5000

Run mounted (as part of the full E-Wardrobe AI app):

    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    # -> POST http://localhost:8000/visualization/api/avatars/<id>/fit-garment
"""

from asgiref.wsgi import WsgiToAsgi

from backend.app import app as _flask_app

app = WsgiToAsgi(_flask_app)
