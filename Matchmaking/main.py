import json
import pickle
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gnn_model import StylingGNN, load_graph_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE MODEL LOADER ---
# This pulls the NLP and GNN intelligence from your training phase
def load_trained_intelligence():
    try:
        with open('nlp_model.pkl', 'rb') as f: nlp = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f: vec = pickle.load(f)

        x, edge_index, in_channels = load_graph_data()
        if x is None or edge_index is None:
            raise RuntimeError('Graph data could not be loaded for GNN.')

        gnn = StylingGNN(in_channels=in_channels, hidden_channels=32, out_channels=16)
        gnn.load_state_dict(torch.load('gnn_model.pth', map_location='cpu'))
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
):
    if not AI:
        return {"error": "Models not loaded"}
    nlp, vec, gnn, x, edge_index = AI

    with open('wardrobe.json', 'r') as f:
        wardrobe = json.load(f)

    def detect_event_type(text: str) -> str:
        text = text.lower()
        if any(keyword in text for keyword in ['wedding', 'marriage', 'bride', 'groom', 'reception']):
            return 'Wedding'
        if any(keyword in text for keyword in ['funeral', 'mourning', 'memorial', 'wake']):
            return 'Funeral'
        if any(keyword in text for keyword in ['party', 'birthday', 'festival', 'celebration', 'dinner']):
            return 'Party'
        if any(keyword in text for keyword in ['corporate', 'office', 'meeting', 'interview', 'temple', 'formal']):
            return 'Formal'
        if any(keyword in text for keyword in [
            'trip', 'travel', 'tour', 'hike', 'hiking', 'trekking', 'trek',
            'mountain', 'hill', 'nuwara', 'ella', 'horton', 'knuckles',
            'cold', 'cool', 'outdoor', 'nature', 'picnic', 'camp'
        ]):
            return 'ColdOutdoor'
        return 'Casual'

    # NLP model predicts outfit-event suitability (Suitable / Unsuitable)
    # trained on Event + Fabric + Micro_Climate features
    def item_nlp_suitability(event: str, fabric: str) -> bool:
        text = event + ' ' + fabric
        return nlp.predict(vec.transform([text]))[0] == 'Suitable'

    event_type = detect_event_type(user_input)
    predicted_intent = 'Formal' if event_type in ['Wedding', 'Formal'] else 'Casual'
    if event_type == 'ColdOutdoor':
        predicted_intent = 'Casual'

    def event_preference_score(item_name: str, fabric: str, event_type: str) -> float:
        name = item_name.lower()
        fabric = fabric.lower() if fabric else ''
        score = 0.0

        if event_type == 'Wedding':
            if any(keyword in name for keyword in ['saree', 'gown', 'blazer', 'silk', 'chiffon', 'georgette']):
                score += 0.35
            elif any(keyword in name for keyword in ['dress', 'jumpsuit']):
                score += 0.2
            else:
                score += 0.1
        elif event_type == 'Funeral':
            if any(keyword in name for keyword in ['cotton', 'kurti', 'pashmina', 'shawl', 'linen', 'wool']):
                score += 0.35
            elif any(keyword in name for keyword in ['silk', 'chiffon', 'georgette']):
                score += 0.15
            else:
                score += 0.05
        elif event_type == 'Party':
            if any(keyword in name for keyword in ['gown', 'saree', 'blazer', 'jumpsuit', 'silk', 'chiffon', 'georgette']):
                score += 0.35
            else:
                score += 0.15
        elif event_type == 'Formal':
            if any(keyword in name for keyword in ['saree', 'gown', 'blazer', 'silk', 'georgette']):
                score += 0.3
            else:
                score += 0.1
        elif event_type == 'ColdOutdoor':
            # Prioritise warm, layerable fabrics for hill country / outdoor trips
            if any(keyword in fabric for keyword in ['wool', 'linen', 'cotton']):
                score += 0.40
            elif any(keyword in name for keyword in ['shawl', 'pashmina', 'suit', 'blazer', 'trouser']):
                score += 0.35
            elif any(keyword in fabric for keyword in ['polyester', 'viscose', 'rayon']):
                score += 0.20
            else:
                score += 0.05
            # Penalise thin, delicate fabrics — wrong for cold weather
            if any(keyword in fabric for keyword in ['silk', 'chiffon', 'georgette']):
                score -= 0.15
        else:
            if any(keyword in name for keyword in ['jumpsuit', 'kurti', 'linen', 'cotton']):
                score += 0.3
            else:
                score += 0.1

        if event_type == 'Funeral' and any(keyword in fabric for keyword in ['linen', 'cotton', 'wool']):
            score += 0.05

        return score

    def build_intent_center(embeddings, categories, target_intent):
        indices = [idx for idx, label in enumerate(categories) if label == target_intent]
        if indices:
            return embeddings[indices].mean(dim=0)
        return embeddings.mean(dim=0)

    _, _, gnn, x, edge_index = AI
    with torch.no_grad():
        item_embeddings = gnn.encode(x, edge_index)[:len(wardrobe)]

    # NLP suitability label per item — used to build the GNN intent centre
    nlp_labels = ['Suitable' if item_nlp_suitability(event_type, item.get('fabric', ''))
                  else 'Unsuitable' for item in wardrobe]
    intent_center = build_intent_center(item_embeddings, nlp_labels, 'Suitable')

    similarity_scores = torch.nn.functional.cosine_similarity(
        item_embeddings,
        intent_center.unsqueeze(0),
        dim=1
    )
    similarity_scores = ((similarity_scores + 1) / 2).tolist()

    def temperature_score(fabric: str) -> float:
        f = fabric.lower() if fabric else ''
        if temperature < 18:                          # cold — hill country
            if f in ['wool', 'polyester']:   return  0.15
            if f in ['linen', 'cotton']:     return  0.05
            if f in ['silk', 'chiffon', 'georgette', 'rayon', 'viscose']: return -0.10
        elif temperature < 25:                        # mild — comfortable
            return 0.05                               # all fabrics acceptable
        else:                                         # hot — tropical Sri Lanka
            if f in ['linen', 'cotton', 'viscose', 'rayon']: return  0.10
            if f in ['wool', 'polyester']:   return -0.10
        return 0.0

    def explain(outfit_name: str, fabric: str, is_suitable: bool, similarity: float) -> str:
        f = fabric.lower() if fabric else ''
        name = outfit_name.lower()
        parts = []

        # Event-based explanation
        if event_type == 'Wedding':
            if any(k in name or k in f for k in ['saree', 'silk', 'georgette', 'chiffon']):
                parts.append("Traditional choice for Sri Lankan weddings")
            elif any(k in name for k in ['gown', 'blazer']):
                parts.append("Elegant and ceremonially appropriate")
        elif event_type == 'Funeral':
            if f in ['cotton', 'wool', 'linen']:
                parts.append("Modest, respectful fabric for solemn occasions")
        elif event_type == 'Party':
            if any(k in name or k in f for k in ['gown', 'saree', 'georgette', 'chiffon']):
                parts.append("Festive and celebratory style")
        elif event_type == 'Formal':
            if any(k in name for k in ['blazer', 'saree', 'gown']):
                parts.append("Professional and formal silhouette")
        elif event_type == 'ColdOutdoor':
            if f == 'wool':
                parts.append("Wool retains heat — ideal for hill country")
            elif f in ['linen', 'cotton']:
                parts.append("Breathable and layerable for outdoor comfort")
            elif f in ['silk', 'chiffon', 'georgette']:
                parts.append("Too lightweight for cold outdoor conditions")

        # Temperature-based explanation
        if temperature < 18:
            if f in ['wool', 'polyester']:
                parts.append(f"Warm fabric suits {temperature:.0f}°C cool weather")
            elif f in ['silk', 'chiffon', 'rayon']:
                parts.append(f"Too thin for {temperature:.0f}°C — consider layering")
        elif temperature > 28:
            if f in ['linen', 'cotton', 'viscose']:
                parts.append(f"Breathable fabric ideal for {temperature:.0f}°C heat")
            elif f == 'wool':
                parts.append(f"Heavy fabric — may be too warm at {temperature:.0f}°C")

        # NLP model decision
        if is_suitable:
            parts.append(f"NLP model: {fabric} suits this event type")
        else:
            parts.append(f"NLP model: {fabric} is a non-typical choice here")

        # GNN similarity
        if similarity > 0.75:
            parts.append("GNN: high style compatibility")
        elif similarity > 0.55:
            parts.append("GNN: moderate style compatibility")
        else:
            parts.append("GNN: low style compatibility")

        return " · ".join(parts) if parts else "Matches your event and climate profile"

    results = []
    for idx, item in enumerate(wardrobe):
        outfit_name = item.get('name', 'Unknown Outfit')
        is_suitable = nlp_labels[idx] == 'Suitable'
        category_match = 1.0 if is_suitable else 0.7
        similarity = similarity_scores[idx]
        event_score = event_preference_score(outfit_name, item.get('fabric', ''), event_type)
        temp_score  = temperature_score(item.get('fabric', ''))
        score = float(0.3 * category_match + 0.55 * similarity + 0.15 * event_score + temp_score)
        confidence = f"{min(100, max(0, int(score * 100)))}%"
        reason = explain(outfit_name, item.get('fabric', ''), is_suitable, similarity)

        results.append({
            'outfit': outfit_name,
            'fabric': item.get('fabric'),
            'image_url': item.get('image_url'),
            'confidence': confidence,
            'reason': reason,
            'score': score,
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    event_class = event_type if event_type in ['Wedding', 'Funeral', 'Party', 'ColdOutdoor'] else predicted_intent
    return {
        'event_class': event_class,
        'location_detected': city,
        'weather': weather,
        'logic_summary': f'Combined NLP intent and GNN wardrobe ranking for {event_class}.',
        'recommendations': results[:3]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)