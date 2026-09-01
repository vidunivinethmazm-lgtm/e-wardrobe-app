"""Canonical wardrobe-item shape shared by every feature.

Classification produces the raw prediction; this module turns it into one
document that also carries the fields the recommendation and organization
features need, so a single saved item flows through all three.
"""

from datetime import datetime, timezone
from uuid import uuid4

# Article type (classification) -> coarse category (recommendation)
TYPE_TO_CATEGORY = {
    # tops / upper
    "Tshirts": "top", "Shirts": "top", "Tops": "top", "Kurtas": "top",
    "Blazers": "top", "Jackets": "top", "Coats": "top", "Sweaters": "top",
    "Cardigans": "top", "Sweatshirts": "top", "Shrugs": "top", "Shawls": "top",
    # bottoms
    "Trousers": "bottom", "Jeans": "bottom", "Shorts": "bottom", "Briefs": "bottom",
    "Skirts": "bottom", "Leggings": "bottom", "Trackpants": "bottom",
    # one-piece / dress family
    "Dresses": "dress", "Gowns": "dress", "Jumpsuits": "dress",
    "Saree": "dress", "Lehenga": "dress", "Salwar": "dress",
}

# Article type -> everyday occasion bucket (organization uses these 5 values)
TYPE_TO_OCCASION = {
    "Tshirts": "Casual", "Tops": "Casual", "Jeans": "Casual", "Shorts": "Home",
    "Shirts": "Office", "Trousers": "Office", "Blazers": "Office",
    "Kurtas": "Religious", "Saree": "Religious", "Lehenga": "Religious", "Salwar": "Religious",
    "Dresses": "Wedding", "Gowns": "Wedding",
    "Jumpsuits": "Casual", "Skirts": "Casual", "Leggings": "Home", "Trackpants": "Home",
    "Jackets": "Casual", "Coats": "Casual", "Sweaters": "Casual",
    "Cardigans": "Casual", "Sweatshirts": "Casual", "Shrugs": "Casual", "Shawls": "Casual",
}

# Rough wash cadence by category
CATEGORY_MAX_WEARS = {"top": 3, "bottom": 5, "dress": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_item(prediction: dict, images: dict | None = None,
               name: str | None = None, price: float = 0.0) -> dict:
    """Assemble a canonical wardrobe document from a classification result."""
    images = images or {}
    article = prediction.get("type") or "Unknown"
    color = prediction.get("color") or ""
    material = prediction.get("material") or ""
    category = TYPE_TO_CATEGORY.get(article, "top")

    return {
        "item_id": f"w_{uuid4().hex[:12]}",
        "created_at": _now(),
        "updated_at": _now(),

        # --- classification attributes ---
        "type": article,
        "color": color,
        "gender": prediction.get("gender") or "",
        "season": prediction.get("season") or "",
        "material": material,
        "type_confidence": prediction.get("type_confidence"),
        "color_confidence": prediction.get("color_confidence"),
        "gender_confidence": prediction.get("gender_confidence"),
        "season_confidence": prediction.get("season_confidence"),
        "material_confidence": prediction.get("material_confidence"),
        "trend_analysis": prediction.get("trend_analysis"),

        # --- images ---
        "original_image_url": images.get("original_image_url"),
        "processed_image_url": images.get("processed_image_url"),
        "back_image_url": images.get("back_image_url"),
        "back_processed_image_url": images.get("back_processed_image_url"),

        # --- recommendation fields ---
        "name": name or " ".join(p for p in (color, material, article) if p) or article,
        "category": category,
        "fabric": material or "Cotton",
        "price": float(price or 0),

        # --- organization fields ---
        "occasion": TYPE_TO_OCCASION.get(article, "Casual"),
        "total_wear_count": 0,
        "current_cycle_wears": 0,
        "max_wears_before_wash": CATEGORY_MAX_WEARS.get(category, 3),
        "status": "Clean",
        "sustainability_score": 0.5,
        "note": "",
    }


# Editable via PATCH /wardrobe/{id}
EDITABLE_FIELDS = {
    "type", "color", "gender", "season", "material", "fabric",
    "name", "category", "price", "occasion",
    "max_wears_before_wash", "sustainability_score", "note",
}


def apply_edits(item: dict, changes: dict) -> dict:
    for key, value in changes.items():
        if key in EDITABLE_FIELDS and value is not None:
            item[key] = value
    # keep fabric / material in sync
    if "material" in changes and "fabric" not in changes:
        item["fabric"] = changes["material"]
    if "fabric" in changes and "material" not in changes:
        item["material"] = changes["fabric"]
    item["updated_at"] = _now()
    return item


def to_recommendation_item(doc: dict, index: int) -> dict:
    """Shape a canonical doc into what the GNN/NLP recommender expects."""
    return {
        "id": index,
        "item_id": doc["item_id"],
        "name": doc.get("name") or doc.get("type") or "Item",
        "fabric": doc.get("fabric") or doc.get("material") or "Cotton",
        "color": (doc.get("color") or "").lower(),
        "category": doc.get("category") or "top",
        "price": doc.get("price") or 0,
        "image_url": doc.get("processed_image_url") or doc.get("original_image_url"),
    }
