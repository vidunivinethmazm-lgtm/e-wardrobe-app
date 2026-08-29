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
_recs: dict[str, dict] = {}       # rec_id -> saved recommendation search

_items_col = None
_events_col = None
_recs_col = None

if _MONGODB_URI:
    from pymongo import MongoClient, DESCENDING

    _client = MongoClient(_MONGODB_URI, appname="e-wardrobe-ai")
    _db = _client[_MONGODB_DB]
    _items_col = _db["items"]
    _events_col = _db["events"]
    _recs_col = _db["recommendations"]
    _items_col.create_index("item_id", unique=True)
    _events_col.create_index("event_id", unique=True)
    _recs_col.create_index("rec_id", unique=True)
    _items_col.create_index([("created_at", DESCENDING)])
    _events_col.create_index([("created_at", DESCENDING)])
    _recs_col.create_index([("created_at", DESCENDING)])


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


# ------------------------------------------- recommendation search history ---
#
# One document per "get recommendations" search: the occasion text, the
# resolved location / weather, the ranked outfit list that was shown, and the
# user's like / skip / free-text feedback on that result.

def add_recommendation(entry: dict) -> dict:
    doc = {k: v for k, v in entry.items() if k != "_id"}
    doc["rec_id"] = f"r_{uuid4().hex[:12]}"
    doc.setdefault("created_at", _now())
    doc.setdefault("feedback", {})
    if _use_mongo():
        _recs_col.insert_one(dict(doc))
        return _clean(doc)
    with _lock:
        _recs[doc["rec_id"]] = doc
    return doc


def list_recommendations(limit: int = 50) -> list[dict]:
    limit = max(1, min(limit, 200))
    if _use_mongo():
        return [_clean(d) for d in
                _recs_col.find().sort("created_at", -1).limit(limit)]
    with _lock:
        rows = sorted(_recs.values(), key=lambda d: d["created_at"], reverse=True)
        return rows[:limit]


def _save_rec(doc: dict) -> dict:
    doc["updated_at"] = _now()
    if _use_mongo():
        _recs_col.replace_one({"rec_id": doc["rec_id"]}, doc)
    return _clean(dict(doc))


def _get_rec(rec_id: str) -> dict | None:
    if _use_mongo():
        return _recs_col.find_one({"rec_id": rec_id})
    return _recs.get(rec_id)


def set_recommendation_feedback(rec_id: str, outfit: str, action: str) -> dict | None:
    with _lock:
        doc = _get_rec(rec_id)
        if doc is None:
            return None
        fb = dict(doc.get("feedback") or {})
        if action in ("liked", "skipped"):
            fb[outfit] = action
        else:                                    # "none" / anything else clears
            fb.pop(outfit, None)
        doc["feedback"] = fb
        return _save_rec(doc)


def set_recommendation_feedback_map(rec_id: str, feedback: dict) -> dict | None:
    with _lock:
        doc = _get_rec(rec_id)
        if doc is None:
            return None
        doc["feedback"] = {k: v for k, v in (feedback or {}).items()
                           if v in ("liked", "skipped")}
        return _save_rec(doc)


def set_recommendation_note(rec_id: str, note: str) -> dict | None:
    with _lock:
        doc = _get_rec(rec_id)
        if doc is None:
            return None
        doc["note"] = (note or "").strip() or None
        return _save_rec(doc)


def delete_recommendation(rec_id: str) -> bool:
    if _use_mongo():
        return _recs_col.delete_one({"rec_id": rec_id}).deleted_count > 0
    with _lock:
        return _recs.pop(rec_id, None) is not None


def clear_recommendations() -> int:
    if _use_mongo():
        return _recs_col.delete_many({}).deleted_count
    with _lock:
        n = len(_recs)
        _recs.clear()
        return n
