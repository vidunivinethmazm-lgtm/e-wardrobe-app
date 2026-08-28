"""E-Wardrobe AI - parent application.

Each team feature is its own FastAPI app, kept in its own package under
``backend/`` and mounted here under a path prefix. A feature's own
``main.py`` is never edited during integration - only the mount line below
is added.

Run from the repository root:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Feature routes are then served under their prefix, e.g. the classification
predict endpoint is ``POST /classification/predict``.
"""

from fastapi import FastAPI

from backend.classification.main import app as classification_app

app = FastAPI(title="E-Wardrobe AI")


@app.get("/")
def root():
    return {
        "service": "E-Wardrobe AI",
        "features": ["/classification"],
    }


app.mount("/classification", classification_app)

# As the other features are integrated, add them here (no other change):
#
# from backend.recommendation.main import app as recommendation_app
# app.mount("/recommendation", recommendation_app)
#
# from backend.visualization.main import app as visualization_app
# app.mount("/visualization", visualization_app)
#
# from backend.organization.main import app as organization_app
# app.mount("/organization", organization_app)
