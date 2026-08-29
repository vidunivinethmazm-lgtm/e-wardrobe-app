"""Shared persistence for wardrobe items and dressing-schedule events.

This is the ONLY module that knows how data is stored.

* If ``MONGODB_URI`` is set in the environment, everything is persisted to a
  MongoDB Atlas database (collections ``items`` and ``events``).
* Otherwise it falls back to an in-memory dict that resets on restart, so the
  app still runs with no database configured.

Every function keeps the same signature and returns plain ``dict`` objects
with no Mongo ``_id`` key - nothing else in the codebase changes.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ModuleNotFoundError:
    pass

from backend.core.schema import apply_edits, build_item

_MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
_MONGODB_DB = os.getenv("MONGODB_DB", "ewardrobe").strip() or "ewardrobe"

_lock = RLock()
_items: dict[str, dict] = {}      # item_id -> item   (in-memory fallback)
_events: dict[str, dict] = {}     # event_id -> event (in-memory fallback)

_items_col = None
_events_col = None

if _MONGODB_URI:
    from pymongo import MongoClient, DESCENDING

    _client = MongoClient(_MONGODB_URI, appname="e-wardrobe-ai")
    _db = _client[_MONGODB_DB]
    _items_col = _db["items"]
    _events_col = _db["events"]
    _items_col.create_index("item_id", unique=True)
    _events_col.create_index("event_id", unique=True)
    _items_col.create_index([("created_at", DESCENDING)])
    _events_col.create_index([("created_at", DESCENDING)])


def _use_mongo() -> bool:
    return _items_col is not None


def _clean(doc: dict | None) -> dict | None:
    if doc is not None:
        doc.pop("_id", None)
    return doc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- wardrobe ---

def add_item(prediction: dict, images: dict | None = None,
             name: str | None = None, price: float = 0.0) -> dict:
    item = build_item(prediction, images, name, price)
    if _use_mongo():
        _items_col.insert_one(dict(item))
        return _clean(item)
    with _lock:
        _items[item["item_id"]] = item
    return item


def list_items() -> list[dict]:
    if _use_mongo():
        return [_clean(d) for d in
                _items_col.find().sort("created_at", -1)]
    with _lock:
        return sorted(_items.values(), key=lambda d: d["created_at"], reverse=True)


def get_item(item_id: str) -> dict | None:
    if _use_mongo():
        return _clean(_items_col.find_one({"item_id": item_id}))
    with _lock:
        return _items.get(item_id)


def update_item(item_id: str, changes: dict) -> dict | None:
    if _use_mongo():
        item = _items_col.find_one({"item_id": item_id})
        if item is None:
            return None
        apply_edits(item, changes)
        _items_col.replace_one({"item_id": item_id}, item)
        return _clean(item)
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        apply_edits(item, changes)
        return item


def delete_item(item_id: str) -> bool:
    if _use_mongo():
        return _items_col.delete_one({"item_id": item_id}).deleted_count > 0
    with _lock:
        return _items.pop(item_id, None) is not None


def record_wear(item_id: str) -> dict | None:
    if _use_mongo():
        item = _items_col.find_one({"item_id": item_id})
        if item is None:
            return None
        item["total_wear_count"] += 1
        item["current_cycle_wears"] += 1
        if item["current_cycle_wears"] >= item["max_wears_before_wash"]:
            item["status"] = "Dirty"
        item["updated_at"] = _now()
        _items_col.replace_one({"item_id": item_id}, item)
        return _clean(item)
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        item["total_wear_count"] += 1
        item["current_cycle_wears"] += 1
        if item["current_cycle_wears"] >= item["max_wears_before_wash"]:
            item["status"] = "Dirty"
        item["updated_at"] = _now()
        return item


def record_wash(item_id: str) -> dict | None:
    if _use_mongo():
        item = _items_col.find_one({"item_id": item_id})
        if item is None:
            return None
        item["current_cycle_wears"] = 0
        item["status"] = "Clean"
        item["updated_at"] = _now()
        _items_col.replace_one({"item_id": item_id}, item)
        return _clean(item)
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        item["current_cycle_wears"] = 0
        item["status"] = "Clean"
        item["updated_at"] = _now()
        return item


def count_items() -> int:
    if _use_mongo():
        return _items_col.count_documents({})
    with _lock:
        return len(_items)


# ------------------------------------------------------------ schedule ---

def add_event(data: dict) -> dict:
    event = {
        "event_id": f"e_{uuid4().hex[:12]}",
        "created_at": _now(),
        **data,
    }
    if _use_mongo():
        _events_col.insert_one(dict(event))
        return _clean(event)
    with _lock:
        _events[event["event_id"]] = event
    return event


def list_events() -> list[dict]:
    if _use_mongo():
        return [_clean(d) for d in
                _events_col.find().sort("created_at", -1)]
    with _lock:
        return sorted(_events.values(), key=lambda d: d["created_at"], reverse=True)


_EDITABLE_EVENT_FIELDS = {"event_name", "event_date", "event_time", "notes"}


def update_event(event_id: str, changes: dict) -> dict | None:
    patch = {k: v for k, v in changes.items()
             if k in _EDITABLE_EVENT_FIELDS and v is not None}
    if _use_mongo():
        event = _events_col.find_one({"event_id": event_id})
        if event is None:
            return None
        event.update(patch)
        event["updated_at"] = _now()
        _events_col.replace_one({"event_id": event_id}, event)
        return _clean(event)
    with _lock:
        event = _events.get(event_id)
        if event is None:
            return None
        event.update(patch)
        event["updated_at"] = _now()
        return event


def delete_event(event_id: str) -> bool:
    if _use_mongo():
        return _events_col.delete_one({"event_id": event_id}).deleted_count > 0
    with _lock:
        return _events.pop(event_id, None) is not None
