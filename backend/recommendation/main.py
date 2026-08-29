import json
import pickle
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.recommendation.gnn_model import (
    StylingGNN,
    build_edges_from_data,
    build_node_features,
    load_graph_data,
)
from backend.core import store
from backend.core.auth_dep import current_user_id
from backend.core.schema import to_recommendation_item

# Data / model files live next to this module; resolve them from here so the
# app works no matter what the current working directory is.
HERE = Path(__file__).resolve().parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Colour harmony scores per event type ─────────────────────────────────────
COLOR_EVENT_SCORES = {
    'Wedding': {
        'white': 0.10, 'gold': 0.10, 'red': 0.08, 'pink': 0.08,
        'multicolor': 0.03, 'olive': -0.08, 'black': -0.03, 'brown': -0.03,
    },
    'Funeral': {
        'white': 0.08, 'black': 0.08, 'navy': 0.06, 'beige': 0.04,
        'red': -0.10, 'pink': -0.08, 'gold': -0.06, 'multicolor': -0.08,
    },
    'Party': {
        'red': 0.10, 'gold': 0.10, 'pink': 0.08, 'blue': 0.06,
        'purple': 0.06, 'teal': 0.06, 'multicolor': 0.08,
        'beige': -0.04, 'brown': -0.04,
    },
    'Formal': {
        'navy': 0.10, 'black': 0.08, 'beige': 0.06, 'white': 0.06,
        'multicolor': -0.08, 'olive': -0.06, 'pink': -0.04,
    },
    'ColdOutdoor': {
        'brown': 0.08, 'beige': 0.06, 'olive': 0.06, 'navy': 0.04,
        'white': -0.04,
    },
}

def load_trained_intelligence():
    try:
        with open(HERE / 'nlp_model.pkl', 'rb') as f: nlp = pickle.load(f)
        with open(HERE / 'vectorizer.pkl', 'rb') as f: vec = pickle.load(f)

        x, edge_index, in_channels = load_graph_data()
        if x is None or edge_index is None:
            raise RuntimeError('Graph data could not be loaded for GNN.')

        gnn = StylingGNN(in_channels=in_channels, hidden_channels=32, out_channels=16)
        gnn.load_state_dict(torch.load(HERE / 'gnn_model.pth', map_location='cpu'))
        gnn.eval()

        return nlp, vec, gnn, x, edge_index
    except Exception as e:
        print(f"❌ Error: Missing trained models! Run your training script first. {e}")
        return None

AI = load_trained_intelligence()

@app.get("/recommend")
async def recommend(
    user_input: str,
    city: str = "Unknown",
    weather: str = "Unknown",
    humidity: int = 0,
    temperature: float = 28.0,
    uid: str = Depends(current_user_id),
):
    if not AI:
        return {"error": "Models not loaded"}
    nlp, vec, gnn, _x0, _e0 = AI

    # Recommend from every item in the user's saved wardrobe (shared store),
    # each carrying its own uploaded / background-removed photo. Only fall back
    # to the bundled demo wardrobe when the user hasn't saved anything yet.
    saved = store.list_items(uid)
    if saved:
        wardrobe = [to_recommendation_item(d, i) for i, d in enumerate(saved)]
        wardrobe_source = "user_wardrobe"
    else:
        with open(HERE / 'wardrobe.json', 'r') as f:
            wardrobe = json.load(f)
        wardrobe_source = "demo_wardrobe"

    # Node features + edges are rebuilt from whatever wardrobe we're using.
    x = build_node_features(wardrobe)
    edge_index = build_edges_from_data(wardrobe)

    def detect_event_type(text: str) -> str:
        text = text.lower()
        if any(k in text for k in ['wedding', 'marriage', 'bride', 'groom', 'reception']):
            return 'Wedding'
        if any(k in text for k in ['funeral', 'mourning', 'memorial', 'wake']):
            return 'Funeral'
        if any(k in text for k in ['party', 'birthday', 'festival', 'celebration', 'dinner']):
            return 'Party'
        if any(k in text for k in ['corporate', 'office', 'meeting', 'interview', 'temple', 'formal']):
            return 'Formal'
        if any(k in text for k in [
            'trip', 'travel', 'tour', 'hike', 'hiking', 'trekking', 'trek',
            'mountain', 'hill', 'nuwara', 'ella', 'horton', 'knuckles',
            'cold', 'cool', 'outdoor', 'nature', 'picnic', 'camp'
        ]):
            return 'ColdOutdoor'
        return 'Casual'

    def item_nlp_suitability(event: str, fabric: str) -> bool:
        text = event + ' ' + fabric
        return nlp.predict(vec.transform([text]))[0] == 'Suitable'

    def event_preference_score(item_name: str, fabric: str, event_type: str) -> float:
        name = item_name.lower()
        fabric = fabric.lower() if fabric else ''
        score = 0.0

        if event_type == 'Wedding':
            if any(k in name for k in ['saree', 'gown', 'blazer', 'silk', 'chiffon', 'georgette']):
                score += 0.35
            elif any(k in name for k in ['dress', 'jumpsuit']):
                score += 0.2
            else:
                score += 0.1
        elif event_type == 'Funeral':
            if any(k in name for k in ['cotton', 'kurti', 'pashmina', 'shawl', 'linen', 'wool']):
                score += 0.35
            elif any(k in name for k in ['silk', 'chiffon', 'georgette']):
                score += 0.15
            else:
                score += 0.05
        elif event_type == 'Party':
            if any(k in name for k in ['gown', 'saree', 'blazer', 'jumpsuit', 'silk', 'chiffon', 'georgette']):
                score += 0.35
            else:
                score += 0.15
        elif event_type == 'Formal':
            if any(k in name for k in ['saree', 'gown', 'blazer', 'silk', 'georgette']):
                score += 0.3
            else:
                score += 0.1
        elif event_type == 'ColdOutdoor':
            if any(k in fabric for k in ['wool', 'linen', 'cotton']):
                score += 0.40
            elif any(k in name for k in ['shawl', 'pashmina', 'suit', 'blazer', 'trouser']):
                score += 0.35
            elif any(k in fabric for k in ['polyester', 'viscose', 'rayon']):
                score += 0.20
            else:
                score += 0.05
            if any(k in fabric for k in ['silk', 'chiffon', 'georgette']):
                score -= 0.15
        else:
            if any(k in name for k in ['jumpsuit', 'kurti', 'linen', 'cotton']):
                score += 0.3
            else:
                score += 0.1

        if event_type == 'Funeral' and any(k in fabric for k in ['linen', 'cotton', 'wool']):
            score += 0.05

        return score

    def colour_score(color: str, event_type: str) -> float:
        c = color.lower() if color else ''
        return COLOR_EVENT_SCORES.get(event_type, {}).get(c, 0.0)

    def suggest_combination(item: dict, wardrobe: list, event_type: str) -> str:
        category = item.get('category', '').lower()
        item_name = item.get('name', '')

        if category in ['saree', 'dress', 'jumpsuit']:
            accessories = [w for w in wardrobe if w.get('category') == 'accessory' and w.get('name') != item_name]
            if accessories:
                return f"Complete the look with {accessories[0]['name']}"
        elif category in ['top', 'blazer']:
            suits = [w for w in wardrobe if w.get('category') == 'suit' and w.get('name') != item_name]
            if suits:
                return f"Pair with {suits[0]['name']} for a polished finish"
        elif category == 'suit':
            return "Pair with a formal blouse and pointed-toe heels"

        fallbacks = {
            'Wedding': 'Add gold jewellery and matching heels to complete the look',
            'Party': 'Accessorise with a statement clutch and strappy heels',
            'Formal': 'Complete with formal shoes and keep accessories minimal',
            'ColdOutdoor': 'Layer with the Woolen Pashmina Shawl for added warmth',
            'Funeral': 'Keep accessories understated and respectful',
            'Casual': 'Style with comfortable flats and a tote bag',
        }
        return fallbacks.get(event_type, 'Style with matching accessories')

    def build_intent_center(embeddings, categories, target_intent):
        indices = [idx for idx, label in enumerate(categories) if label == target_intent]
        if indices:
            return embeddings[indices].mean(dim=0)
        return embeddings.mean(dim=0)

    # Fabric families - covers both the recommender's own vocabulary and the
    # materials the classification model produces (denim, leather, suede, ...).
    WARM_FABRICS  = ('wool', 'polyester', 'fleece', 'leather', 'velvet', 'denim', 'knit', 'suede', 'nylon')
    COOL_FABRICS  = ('linen', 'cotton', 'viscose', 'rayon', 'chambray')
    LIGHT_FABRICS = ('silk', 'chiffon', 'georgette', 'satin', 'organza')
    WET_WEATHER   = ('rain', 'drizzle', 'thunderstorm', 'snow', 'squall')

    def temperature_score(fabric: str) -> float:
        f = fabric.lower() if fabric else ''
        if temperature < 18:
            if any(k in f for k in WARM_FABRICS):  return  0.15
            if any(k in f for k in COOL_FABRICS):  return  0.05
            if any(k in f for k in LIGHT_FABRICS): return -0.10
        elif temperature < 25:
            return 0.05
        else:
            if any(k in f for k in COOL_FABRICS):  return  0.10
            if any(k in f for k in LIGHT_FABRICS): return  0.03
            if any(k in f for k in WARM_FABRICS):  return -0.10
        return 0.0

    def weather_score(fabric: str) -> float:
        """How the fabric copes with the current sky and humidity for `city`."""
        f = fabric.lower() if fabric else ''
        w = weather.lower() if weather else ''
        s = 0.0

        if any(k in w for k in WET_WEATHER):
            if any(k in f for k in ('suede', 'silk', 'satin', 'velvet', 'linen')):
                s -= 0.12                      # water-sensitive
            if any(k in f for k in ('polyester', 'nylon', 'denim', 'wool', 'fleece')):
                s += 0.10                      # quick-dry / water-shedding / durable
        elif 'clear' in w:
            if any(k in f for k in ('linen', 'cotton', 'chambray')):
                s += 0.10                      # breathable under direct sun
            if any(k in f for k in ('wool', 'fleece', 'leather', 'velvet', 'denim')):
                s -= 0.10                      # heat traps

        if humidity >= 75:
            if any(k in f for k in ('linen', 'cotton', 'viscose', 'rayon')):
                s += 0.08
            if any(k in f for k in ('polyester', 'nylon', 'silk', 'leather')):
                s -= 0.08                      # clammy / sticky in humid air
        return s

    def explain(outfit_name: str, fabric: str, color: str, is_suitable: bool, similarity: float, event_type: str) -> str:
        f = fabric.lower() if fabric else ''
        c = color.lower() if color else ''
        name = outfit_name.lower()
        parts = []

        if event_type == 'Wedding':
            if any(k in name or k in f for k in ['saree', 'silk', 'georgette', 'chiffon']):
                parts.append("Traditional choice for Sri Lankan weddings")
            elif any(k in name for k in ['gown', 'blazer']):
                parts.append("Elegant and ceremonially appropriate")
            colour_note = COLOR_EVENT_SCORES.get('Wedding', {}).get(c, 0)
            if colour_note > 0.05:
                parts.append(f"{c.capitalize()} is an auspicious colour for weddings")
            elif colour_note < -0.02:
                parts.append(f"{c.capitalize()} is an unconventional colour for weddings")
        elif event_type == 'Funeral':
            if f in ['cotton', 'wool', 'linen']:
                parts.append("Modest, respectful fabric for solemn occasions")
            if c in ['white', 'black', 'navy', 'beige']:
                parts.append(f"{c.capitalize()} is appropriate for funeral dress")
        elif event_type == 'Party':
            if any(k in name or k in f for k in ['gown', 'saree', 'georgette', 'chiffon']):
                parts.append("Festive and celebratory style")
            if COLOR_EVENT_SCORES.get('Party', {}).get(c, 0) > 0.05:
                parts.append(f"{c.capitalize()} adds vibrancy for a party look")
        elif event_type == 'Formal':
            if any(k in name for k in ['blazer', 'saree', 'gown']):
                parts.append("Professional and formal silhouette")
            if COLOR_EVENT_SCORES.get('Formal', {}).get(c, 0) > 0.05:
                parts.append(f"{c.capitalize()} projects a professional image")
        elif event_type == 'ColdOutdoor':
            if f == 'wool':
                parts.append("Wool retains heat — ideal for hill country")
            elif f in ['linen', 'cotton']:
                parts.append("Breathable and layerable for outdoor comfort")
            elif f in ['silk', 'chiffon', 'georgette']:
                parts.append("Too lightweight for cold outdoor conditions")

        if temperature < 18:
            if any(k in f for k in WARM_FABRICS):
                parts.append(f"Warm fabric suits {temperature:.0f}°C in {city}")
            elif any(k in f for k in LIGHT_FABRICS):
                parts.append(f"Too thin for {temperature:.0f}°C — consider layering")
        elif temperature > 28:
            if any(k in f for k in COOL_FABRICS):
                parts.append(f"Breathable fabric ideal for {temperature:.0f}°C heat in {city}")
            elif any(k in f for k in WARM_FABRICS):
                parts.append(f"Heavy fabric — may be too warm at {temperature:.0f}°C")

        w = weather.lower() if weather else ''
        if any(k in w for k in WET_WEATHER):
            if any(k in f for k in ('suede', 'silk', 'satin', 'velvet')):
                parts.append(f"{weather} in {city} — {fabric} is easily marked by water")
            elif any(k in f for k in ('polyester', 'nylon', 'denim', 'wool')):
                parts.append(f"{weather} in {city} — {fabric} shrugs off wet weather")
        elif 'clear' in w and any(k in f for k in ('linen', 'cotton')):
            parts.append(f"Clear skies in {city} — a breathable weave keeps you cool")
        if humidity >= 75 and any(k in f for k in ('linen', 'cotton', 'viscose', 'rayon')):
            parts.append(f"{humidity}% humidity — this fabric stays comfortable")

        if is_suitable:
            parts.append(f"NLP model: {fabric} suits this event type")
        else:
            parts.append(f"NLP model: {fabric} is a non-typical choice here")

        if similarity > 0.75:
            parts.append("GNN: high style compatibility")
        elif similarity > 0.55:
            parts.append("GNN: moderate style compatibility")
        else:
            parts.append("GNN: low style compatibility")

        return " · ".join(parts) if parts else "Matches your event and climate profile"

    event_type = detect_event_type(user_input)
    predicted_intent = 'Formal' if event_type in ['Wedding', 'Formal'] else 'Casual'
    if event_type == 'ColdOutdoor':
        predicted_intent = 'Casual'

    with torch.no_grad():
        item_embeddings = gnn.encode(x, edge_index)[:len(wardrobe)]

    nlp_labels = ['Suitable' if item_nlp_suitability(event_type, item.get('fabric', ''))
                  else 'Unsuitable' for item in wardrobe]
    intent_center = build_intent_center(item_embeddings, nlp_labels, 'Suitable')

    similarity_scores = torch.nn.functional.cosine_similarity(
        item_embeddings,
        intent_center.unsqueeze(0),
        dim=1
    )
    similarity_scores = ((similarity_scores + 1) / 2).tolist()

    results = []
    for idx, item in enumerate(wardrobe):
        outfit_name = item.get('name', 'Unknown Outfit')
        fabric      = item.get('fabric', '')
        color       = item.get('color', '')
        price       = item.get('price', 0)
        category    = item.get('category', '')

        is_suitable    = nlp_labels[idx] == 'Suitable'
        category_match = 1.0 if is_suitable else 0.7
        similarity     = similarity_scores[idx]
        event_score    = event_preference_score(outfit_name, fabric, event_type)
        temp_score     = temperature_score(fabric)
        weather_s      = weather_score(fabric)
        color_s        = colour_score(color, event_type)

        score = float(
            0.3 * category_match + 0.55 * similarity + 0.15 * event_score
            + temp_score + weather_s + color_s
        )
        confidence  = f"{min(100, max(0, int(score * 100)))}%"
        reason      = explain(outfit_name, fabric, color, is_suitable, similarity, event_type)
        combination = suggest_combination(item, wardrobe, event_type)

        results.append({
            'outfit':      outfit_name,
            'item_id':     item.get('item_id'),
            'fabric':      fabric,
            'color':       color,
            'price':       price,
            'category':    category,
            'image_url':   item.get('image_url'),
            'confidence':  confidence,
            'reason':      reason,
            'combination': combination,
            'score':       score,
        })

    # Every wardrobe item is returned, best fit first - the screen shows the
    # user's whole wardrobe ranked for this occasion, each with its photo.
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    event_class = event_type if event_type in ['Wedding', 'Funeral', 'Party', 'ColdOutdoor'] else predicted_intent
    return {
        'event_class':       event_class,
        'location_detected': city,
        'weather':           weather,
        'temperature':       round(temperature, 1),
        'humidity':          humidity,
        'wardrobe_source':   wardrobe_source,
        'items_considered':  len(wardrobe),
        'logic_summary': (
            f'Ranked all {len(results)} wardrobe items for {event_class} in {city} — '
            f'{weather}, {temperature:.0f}°C, {humidity}% humidity — '
            f'via NLP intent, GNN style match, and fabric/colour fit.'
        ),
        'recommendations':   results,
    }

# ── Recommendation search history (persisted via backend.core.store) ─────────

class SavedSearch(BaseModel):
    occasion: str = ""
    event_class: str = ""
    location: str = ""
    weather: str = ""
    time: str = ""
    data: dict = {}
    feedback: dict = {}
    note: str | None = None


class FeedbackPatch(BaseModel):
    outfit: str | None = None
    action: str = "none"                 # liked | skipped | none
    feedback: dict | None = None         # or replace the whole map at once


class NotePatch(BaseModel):
    note: str = ""


@app.get("/history")
async def get_history(limit: int = 50, uid: str = Depends(current_user_id)):
    return store.list_recommendations(uid, limit)


@app.post("/history")
async def add_history(search: SavedSearch, uid: str = Depends(current_user_id)):
    return store.add_recommendation(search.model_dump(), uid)


@app.patch("/history/{rec_id}/feedback")
async def patch_history_feedback(rec_id: str, patch: FeedbackPatch,
                                 uid: str = Depends(current_user_id)):
    if patch.feedback is not None:
        updated = store.set_recommendation_feedback_map(rec_id, patch.feedback, uid)
    elif patch.outfit is not None:
        updated = store.set_recommendation_feedback(rec_id, patch.outfit, patch.action, uid)
    else:
        raise HTTPException(status_code=422, detail="give 'feedback' map or 'outfit' + 'action'")
    if updated is None:
        raise HTTPException(status_code=404, detail="history entry not found")
    return updated


@app.patch("/history/{rec_id}/note")
async def patch_history_note(rec_id: str, patch: NotePatch,
                             uid: str = Depends(current_user_id)):
    updated = store.set_recommendation_note(rec_id, patch.note, uid)
    if updated is None:
        raise HTTPException(status_code=404, detail="history entry not found")
    return updated


@app.delete("/history/{rec_id}")
async def remove_history(rec_id: str, uid: str = Depends(current_user_id)):
    return {"deleted": store.delete_recommendation(rec_id, uid)}


@app.delete("/history")
async def clear_history(uid: str = Depends(current_user_id)):
    return {"cleared": store.clear_recommendations(uid)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
