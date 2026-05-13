# E-Wardrobe AI — Feature: AI Outfit Recommendation & Matchmaking

Branch: `AI-Outfit-Recommendation-&-Matchmaking`

## Overview

This branch implements the intelligent outfit recommendation engine for the E-Wardrobe application. It combines a **Natural Language Processing (NLP)** model with a **Graph Neural Network (GNN)** to recommend the most suitable outfits from a user's wardrobe based on event type, weather conditions, and temperature — tailored for Sri Lankan climate and cultural context.

---

## Project Structure

```
├── Matchmaking/
│   ├── main.py                 # FastAPI backend — recommendation endpoint
│   ├── gnn_model.py            # GNN architecture (StylingGNN) + graph data builder
│   ├── nlp_model.py            # NLP model training (event + fabric suitability)
│   ├── generate_dataset.py     # Synthetic wardrobe dataset generator
│   ├── wardrobe.json           # User wardrobe data
│   ├── wardrobe_data.csv       # Training data for NLP model
│   ├── gnn_model.pth           # Trained GNN weights
│   ├── nlp_model.pkl           # Trained NLP classifier
│   └── vectorizer.pkl          # TF-IDF vectorizer for NLP
│
└── SmartWardrobeApp/           # Expo React Native mobile app
    ├── app/
    │   ├── _layout.tsx
    │   ├── modal.tsx
    │   └── (tabs)/
    │       ├── index.tsx
    │       └── explore.tsx
    ├── components/
    ├── constants/theme.ts
    ├── hooks/
    ├── package.json
    └── app.json
```

---

## How It Works

### 1. Event Detection (NLP)
The user describes their occasion in natural language (e.g., *"I have a wedding in Kandy"*). A keyword-based parser classifies the event into one of:
- `Wedding`, `Funeral`, `Party`, `Formal`, `ColdOutdoor`, `Casual`

### 2. NLP Suitability Scoring
A trained scikit-learn classifier (Logistic Regression / SVM) predicts whether each wardrobe item's fabric is **Suitable** or **Unsuitable** for the detected event type.

### 3. GNN Style Embedding
A **Graph Convolutional Network (GNN)** (`StylingGNN`) encodes wardrobe items as nodes in a style graph. Cosine similarity between each item's embedding and the "ideal outfit" embedding gives a style compatibility score.

### 4. Final Ranking
The final score is a weighted combination:

```
score = 0.30 × NLP_suitability
      + 0.55 × GNN_similarity
      + 0.15 × event_preference_score
      + temperature_adjustment
```

Top 3 results are returned with confidence percentages and human-readable explanations.

---

## API

Backend runs on `http://localhost:8000`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend` | GET | Get top 3 outfit recommendations |
| `/docs` | GET | Swagger UI |

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_input` | string | Natural language event description |
| `city` | string | User's city |
| `weather` | string | Weather condition |
| `humidity` | int | Humidity percentage |
| `temperature` | float | Temperature in °C |

### Example Request
```
GET /recommend?user_input=I have a wedding tomorrow&city=Colombo&weather=Sunny&temperature=31
```

### Example Response
```json
{
  "event_class": "Wedding",
  "location_detected": "Colombo",
  "recommendations": [
    {
      "outfit": "Blue Silk Saree",
      "fabric": "Silk",
      "confidence": "87%",
      "reason": "Traditional choice for Sri Lankan weddings · NLP model: Silk suits this event type · GNN: high style compatibility"
    }
  ]
}
```

---

## Setup & Run

### Train Models (first time)

```bash
cd Matchmaking
pip install fastapi uvicorn torch torch-geometric scikit-learn pandas numpy
python nlp_model.py       # trains and saves nlp_model.pkl + vectorizer.pkl
python gnn_model.py       # trains and saves gnn_model.pth
```

### Run Backend

```bash
cd Matchmaking
uvicorn main:app --reload --port 8000
```

### Mobile App

```bash
cd SmartWardrobeApp
npm install
npx expo start
```

---

## Tech Stack

- **AI/ML** — PyTorch, PyTorch Geometric (GNN), scikit-learn (NLP), TF-IDF
- **Backend** — FastAPI, Uvicorn
- **Mobile** — React Native (Expo), TypeScript
- **Data** — Synthetic Sri Lankan wardrobe dataset

---

## Team Member

**[Branch Owner]** — AI Outfit Recommendation & Matchmaking
