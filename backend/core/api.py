"""Shared wardrobe API - mounted by backend/main.py at /wardrobe.

The mobile app talks to these endpoints instead of Firebase; the
recommendation and organization features read the same store directly
(backend.core.store), so one saved item flows through all three.
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core import store
from backend.core.auth_dep import current_user_id

app = FastAPI(title="E-Wardrobe - Shared Wardrobe")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SaveItem(BaseModel):
    prediction: dict
    images: dict | None = None
    name: str | None = None
    price: float | None = 0.0


class EditItem(BaseModel):
    type: str | None = None
    color: str | None = None
    gender: str | None = None
    season: str | None = None
    material: str | None = None
    name: str | None = None
    category: str | None = None
    occasion: str | None = None
    price: float | None = None
    note: str | None = None            # "" clears it


class ScheduleEvent(BaseModel):
    wardrobe_item_id: str | None = None
    event_name: str
    event_date: str
    event_time: str
    notes: str | None = ""
    clothing_type: str | None = None
    clothing_color: str | None = None
    processed_image_url: str | None = None
    trend_suggestion: dict | None = None


class EditEvent(BaseModel):
    event_name: str | None = None
    event_date: str | None = None
    event_time: str | None = None
    notes: str | None = None


@app.get("/")
def list_wardrobe(uid: str = Depends(current_user_id)):
    return store.list_items(uid)


@app.post("/")
def save_wardrobe_item(body: SaveItem, uid: str = Depends(current_user_id)):
    return store.add_item(body.prediction, uid, body.images, body.name, body.price or 0.0)


@app.get("/count")
def wardrobe_count(uid: str = Depends(current_user_id)):
    return {"count": store.count_items(uid)}


@app.get("/schedule")
def list_schedule(uid: str = Depends(current_user_id)):
    return store.list_events(uid)


@app.post("/schedule")
def create_schedule(body: ScheduleEvent, uid: str = Depends(current_user_id)):
    return store.add_event(body.model_dump(), uid)


@app.patch("/schedule/{event_id}")
def edit_schedule(event_id: str, body: EditEvent, uid: str = Depends(current_user_id)):
    event = store.update_event(event_id, body.model_dump(exclude_none=True), uid)
    if event is None:
        raise HTTPException(404, "Event not found")
    return event


@app.delete("/schedule/{event_id}")
def remove_schedule(event_id: str, uid: str = Depends(current_user_id)):
    if not store.delete_event(event_id, uid):
        raise HTTPException(404, "Event not found")
    return {"deleted": event_id}


@app.get("/{item_id}")
def get_wardrobe_item(item_id: str, uid: str = Depends(current_user_id)):
    item = store.get_item(item_id, uid)
    if item is None:
        raise HTTPException(404, "Item not found")
    return item


@app.patch("/{item_id}")
def edit_wardrobe_item(item_id: str, body: EditItem, uid: str = Depends(current_user_id)):
    item = store.update_item(item_id, body.model_dump(exclude_none=True), uid)
    if item is None:
        raise HTTPException(404, "Item not found")
    return item


@app.delete("/{item_id}")
def delete_wardrobe_item(item_id: str, uid: str = Depends(current_user_id)):
    if not store.delete_item(item_id, uid):
        raise HTTPException(404, "Item not found")
    return {"deleted": item_id}


@app.post("/{item_id}/wear")
def wear_wardrobe_item(item_id: str, uid: str = Depends(current_user_id)):
    item = store.record_wear(item_id, uid)
    if item is None:
        raise HTTPException(404, "Item not found")
    return {"message": "Wear recorded", "status": item["status"], "item": item}


@app.post("/{item_id}/wash")
def wash_wardrobe_item(item_id: str, uid: str = Depends(current_user_id)):
    item = store.record_wash(item_id, uid)
    if item is None:
        raise HTTPException(404, "Item not found")
    return {"message": "Marked as washed", "status": item["status"], "item": item}
