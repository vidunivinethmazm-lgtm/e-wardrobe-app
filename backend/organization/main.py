"""Smart Wardrobe Organization - reads the shared wardrobe store.

Items saved from the classification flow are organized here: clustered into
layout groups, assigned physical wardrobe positions, and checked for
over/under-use. Wear and wash events update the same shared store, so the
counts persist across all features.
"""

from types import SimpleNamespace

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core import store
from backend.core.auth_dep import current_user_id
from backend.organization import ml_engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _items(user_id: str):
    """Shared-store items as attribute objects for the ML engine (keyed by item_id)."""
    objs = []
    for d in store.list_items(user_id):
        o = SimpleNamespace(**d)
        o.id = d["item_id"]
        objs.append(o)
    return objs


@app.get("/items/organized")
def get_organized_items(uid: str = Depends(current_user_id)):
    items     = _items(uid)
    clusters  = ml_engine.SmartWardrobeEngine.get_clusters(items)
    positions = ml_engine.SmartWardrobeEngine.assign_wardrobe_positions(items)

    result = []
    for o in items:
        row = dict(o.__dict__)
        row["layout_group"] = clusters.get(o.id, 0)
        row.update(positions.get(o.id, {}))
        result.append(row)
    return result


@app.get("/wardrobe/layout")
def get_wardrobe_layout(uid: str = Depends(current_user_id)):
    items     = _items(uid)
    positions = ml_engine.SmartWardrobeEngine.assign_wardrobe_positions(items)

    section_counts = {k: 0 for k in ml_engine.WARDROBE_SECTIONS}
    for pos in positions.values():
        section_counts[pos["wardrobe_section"]] += 1

    total = len(items)
    return {
        "total_items": total,
        "sections": {
            k: {
                **ml_engine.WARDROBE_SECTIONS[k],
                "item_count":      section_counts[k],
                "utilization_pct": round(section_counts[k] / total * 100) if total else 0,
            }
            for k in ml_engine.WARDROBE_SECTIONS
        },
    }


@app.post("/items/wear/{item_id}")
def wear_item(item_id: str, uid: str = Depends(current_user_id)):
    item = store.record_wear(item_id, uid)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Wear recorded", "status": item["status"]}


@app.post("/items/wash/{item_id}")
def wash_item(item_id: str, uid: str = Depends(current_user_id)):
    item = store.record_wash(item_id, uid)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item marked as washed", "status": item["status"]}


@app.get("/items/insights")
def get_insights(uid: str = Depends(current_user_id)):
    items       = _items(uid)
    anomalies   = ml_engine.SmartWardrobeEngine.detect_anomalies(items)
    dirty_count = sum(1 for o in items if o.status == "Dirty")
    return {
        "dirty_count": dirty_count,
        "underused":   anomalies["underused"],
        "overused":    anomalies["overused"],
    }
