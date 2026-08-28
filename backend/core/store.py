"""Shared persistence for wardrobe items and dressing-schedule events.

This is the ONLY module that knows how data is stored. Today it is an
in-memory dict (resets on restart). To move to MongoDB Atlas, reimplement
these functions against a collection - nothing else in the codebase changes.
"""

from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from backend.core.schema import apply_edits, build_item

_lock = RLock()
_items: dict[str, dict] = {}      # item_id -> item
_events: dict[str, dict] = {}     # event_id -> event


# ---------------------------------------------------------------- wardrobe ---

def add_item(prediction: dict, images: dict | None = None,
             name: str | None = None, price: float = 0.0) -> dict:
    item = build_item(prediction, images, name, price)
    with _lock:
        _items[item["item_id"]] = item
    return item


def list_items() -> list[dict]:
    with _lock:
        return sorted(_items.values(), key=lambda d: d["created_at"], reverse=True)


def get_item(item_id: str) -> dict | None:
    with _lock:
        return _items.get(item_id)


def update_item(item_id: str, changes: dict) -> dict | None:
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        apply_edits(item, changes)
        return item


def delete_item(item_id: str) -> bool:
    with _lock:
        return _items.pop(item_id, None) is not None


def record_wear(item_id: str) -> dict | None:
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        item["total_wear_count"] += 1
        item["current_cycle_wears"] += 1
        if item["current_cycle_wears"] >= item["max_wears_before_wash"]:
            item["status"] = "Dirty"
        item["updated_at"] = datetime.now(timezone.utc).isoformat()
        return item


def record_wash(item_id: str) -> dict | None:
    with _lock:
        item = _items.get(item_id)
        if item is None:
            return None
        item["current_cycle_wears"] = 0
        item["status"] = "Clean"
        item["updated_at"] = datetime.now(timezone.utc).isoformat()
        return item


def count_items() -> int:
    with _lock:
        return len(_items)


# ------------------------------------------------------------ schedule ---

def add_event(data: dict) -> dict:
    event = {
        "event_id": f"e_{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    with _lock:
        _events[event["event_id"]] = event
    return event


def list_events() -> list[dict]:
    with _lock:
        return sorted(_events.values(), key=lambda d: d["created_at"], reverse=True)


_EDITABLE_EVENT_FIELDS = {"event_name", "event_date", "event_time", "notes"}


def update_event(event_id: str, changes: dict) -> dict | None:
    with _lock:
        event = _events.get(event_id)
        if event is None:
            return None
        for key, value in changes.items():
            if key in _EDITABLE_EVENT_FIELDS and value is not None:
                event[key] = value
        event["updated_at"] = datetime.now(timezone.utc).isoformat()
        return event


def delete_event(event_id: str) -> bool:
    with _lock:
        return _events.pop(event_id, None) is not None
