# E-Wardrobe AI

An AI-powered smart wardrobe management application built with FastAPI backends and a React Native (Expo) mobile app. The system uses deep learning, graph neural networks, and classical machine learning to classify clothing, recommend outfits, enable virtual try-on, and intelligently organize wardrobes.

---

## Branches Overview

| Branch | Feature | Tech |
|--------|---------|------|
| [`feature/ai-clothing-classification`](#-feature-1--ai-clothing-classification) | Classify clothing by type, color, gender & season | MobileNetV2, PyTorch, FastAPI |
| [`AI-Outfit-Recommendation-&-Matchmaking`](#-feature-2--ai-outfit-recommendation--matchmaking) | Recommend outfits using NLP + GNN | scikit-learn, PyTorch Geometric, FastAPI |
| [`AI-visualization`](#-feature-3--ai-visualization--virtual-try-on) | Virtual try-on with 4-stage AI pipeline | PyTorch, OpenCV, FastAPI, WebSocket |
| [`Smart-Wardrobe-Organization-System`](#-feature-4--smart-wardrobe-organization-system) | Organize wardrobe with clustering & anomaly detection | scikit-learn, SQLAlchemy, FastAPI |

---

## Feature 1 — AI Clothing Classification

**Branch:** `feature/ai-clothing-classification`
**Member:** Eshani Perera

### Overview
Trains MobileNetV2-based deep learning models to automatically classify clothing items by **article type**, **color**, **gender**, and **season** from images. Predictions are served through a FastAPI backend consumed by the Expo mobile app.

### Project Structure
```
├── notebooks/
│   ├── 02_create_splits.ipynb          # Train/val/test dataset splitting
│   ├── 03_dataloader_test.ipynb        # Dataloader verification
│   ├── 04_train_mobilenetv2.ipynb      # Article type classifier training
│   ├── 05_evaluate_model.ipynb         # Model evaluation & metrics
│   ├── 07_background_removal.ipynb     # Background removal preprocessing
│   ├── 08_train_color_model.ipynb      # Color classifier training
│   ├── 09_train_gender_model.ipynb     # Gender classifier training
│   └── 10_train_season_model.ipynb     # Season classifier training
├── models/
│   ├── articleType_class_to_idx.json
│   ├── color_class_to_idx.json
│   ├── gender_class_to_idx.json
│   ├── season_class_to_idx.json
│   └── mobilenetv2_training_history.json
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI app & endpoints
│       ├── models_loader.py            # Loads trained model weights
│       └── predictor.py                # Runs inference on uploaded images
├── mobile_app/                         # Expo React Native app
│   ├── app/(tabs)/
│   ├── components/
│   ├── hooks/
│   └── firebaseConfig.ts
├── data/raw/styles.csv                 # Kaggle Fashion dataset metadata
└── outputs/classification_report.txt
```

### Models

| Model | Architecture | Task |
|-------|-------------|------|
| Article Type | MobileNetV2 (fine-tuned) | Multi-class clothing type |
| Color | MobileNetV2 (fine-tuned) | Color category |
| Gender | MobileNetV2 (fine-tuned) | Men / Women / Boys / Girls / Unisex |
| Season | MobileNetV2 (fine-tuned) | Summer / Winter / Fall / Spring |

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload image → get classification results |
| `/docs` | GET | Swagger UI |

**Example Response:**
```json
{
  "article_type": "Shirts",
  "color": "Blue",
  "gender": "Men",
  "season": "Summer"
}
```

### Setup
```bash
# Backend
cd backend
pip install fastapi uvicorn torch torchvision pillow rembg
uvicorn app.main:app --reload

# Mobile App
cd mobile_app
npm install && npx expo start
```

---

## Feature 2 — AI Outfit Recommendation & Matchmaking

**Branch:** `AI-Outfit-Recommendation-&-Matchmaking`

### Overview
Combines an **NLP classifier** with a **Graph Neural Network (GNN)** to recommend the most suitable outfits from a user's wardrobe based on event type, weather, and temperature — tailored for Sri Lankan climate and cultural occasions.

### Project Structure
```
├── Matchmaking/
│   ├── main.py                 # FastAPI backend — /recommend endpoint
│   ├── gnn_model.py            # StylingGNN architecture + graph builder
│   ├── nlp_model.py            # NLP suitability classifier (training)
│   ├── generate_dataset.py     # Synthetic wardrobe dataset generator
│   ├── wardrobe.json           # User wardrobe items
│   ├── gnn_model.pth           # Trained GNN weights
│   ├── nlp_model.pkl           # Trained NLP classifier
│   └── vectorizer.pkl          # TF-IDF vectorizer
└── SmartWardrobeApp/           # Expo React Native mobile app
    ├── app/
    │   ├── _layout.tsx
    │   ├── modal.tsx
    │   └── (tabs)/
    │       ├── _layout.tsx
    │       ├── index.tsx
    │       └── explore.tsx
    ├── assets/images/
    │   ├── android-icon-background.png
    │   ├── android-icon-foreground.png
    │   ├── android-icon-monochrome.png
    │   ├── favicon.png
    │   ├── icon.png
    │   ├── partial-react-logo.png
    │   ├── react-logo.png
    │   ├── react-logo@2x.png
    │   ├── react-logo@3x.png
    │   └── splash-icon.png
    ├── components/
    │   ├── external-link.tsx
    │   ├── haptic-tab.tsx
    │   ├── hello-wave.tsx
    │   ├── parallax-scroll-view.tsx
    │   ├── themed-text.tsx
    │   ├── themed-view.tsx
    │   └── ui/
    │       ├── collapsible.tsx
    │       ├── icon-symbol.tsx
    │       └── icon-symbol.ios.tsx
    ├── constants/theme.ts
    ├── hooks/
    │   ├── use-color-scheme.ts
    │   ├── use-color-scheme.web.ts
    │   └── use-theme-color.ts
    ├── scripts/reset-project.js
    ├── app.json
    ├── eslint.config.js
    ├── package.json
    ├── package-lock.json
    ├── tsconfig.json
    └── README.md
```

### How It Works

1. **Event Detection** — Parses user's natural language input to classify the occasion (Wedding, Funeral, Party, Formal, ColdOutdoor, Casual)
2. **NLP Suitability** — Predicts whether each item's fabric is *Suitable* or *Unsuitable* for the event
3. **GNN Embedding** — Encodes wardrobe items as graph nodes; cosine similarity gives style compatibility score
4. **Final Ranking** — Weighted score:
```
score = 0.30 × NLP_suitability
      + 0.55 × GNN_similarity
      + 0.15 × event_preference
      + temperature_adjustment
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend` | GET | Top 3 outfit recommendations |
| `/docs` | GET | Swagger UI |

**Query Parameters:** `user_input`, `city`, `weather`, `humidity`, `temperature`

**Example Response:**
```json
{
  "event_class": "Wedding",
  "location_detected": "Colombo",
  "recommendations": [
    {
      "outfit": "Blue Silk Saree",
      "fabric": "Silk",
      "confidence": "87%",
      "reason": "Traditional choice for Sri Lankan weddings · NLP: Silk suits this event · GNN: high style compatibility"
    }
  ]
}
```

### Setup
```bash
cd Matchmaking
pip install fastapi uvicorn torch torch-geometric scikit-learn pandas numpy
python nlp_model.py        # train NLP model
python gnn_model.py        # train GNN
uvicorn main:app --reload
```

---

## Feature 3 — AI Visualization & Virtual Try-On

**Branch:** `AI-visualization`

### Overview
A 4-stage deep learning pipeline that generates a realistic avatar wearing recommended outfits. Users take a selfie and submit measurements; the system runs body calibration, face keypoint detection, outfit matching, and avatar rendering — with real-time progress via WebSocket.

### Project Structure
```
├── vineths_wardrobe/
│   ├── app.py                              # FastAPI server (REST + WebSocket)
│   ├── src/
│   │   ├── virtual_tryon_pipeline.py       # 4-stage pipeline orchestrator
│   │   ├── avatar_manager.py               # Avatar generation & rendering
│   │   ├── body_calibration.py             # Body type & size estimation
│   │   ├── face_keypoint_model.py          # Facial landmark detection
│   │   ├── outfit_recommender.py           # Outfit matching logic
│   │   ├── model_registry.py               # Multi-model accuracy reporting
│   │   └── models/
│   │       ├── stage1_body_models.py       # Body classifiers
│   │       ├── stage2_face_models.py       # Face keypoint models
│   │       ├── stage3_outfit_models.py     # Outfit recommendation models
│   │       └── stage4_avatar_models.py     # Avatar generation models
│   ├── frontend/                           # Web frontend (HTML + JS)
│   └── scripts/train_all_models.py
└── ewardrobe-app/                          # React Native mobile app
    └── src/screens/
        ├── SelfieScreen.tsx
        ├── MeasurementsScreen.tsx
        ├── ProcessingScreen.tsx
        └── TryOnScreen.tsx
```

### 4-Stage Pipeline

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Body Calibration | Estimates body type (Hourglass, Pear, Apple…) and standard size (XS–XXL) |
| 2 | Face Processing | Detects facial keypoints for avatar face mapping |
| 3 | Outfit Recommendation | Filters wardrobe by body type, occasion, and season |
| 4 | Avatar Generation | Composites face + outfit onto a body avatar matched to user measurements |

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tryon` | POST | Upload selfie + measurements → full pipeline result |
| `/api/recommend` | POST | Outfit recommendations only |
| `/api/accuracy` | GET | Multi-stage model accuracy report |
| `/ws/tryon/{session_id}` | WebSocket | Real-time pipeline progress |

### Setup
```bash
cd vineths_wardrobe
pip install -r requirements.txt
python scripts/train_all_models.py    # first-time model training
uvicorn app:app --reload --port 8000

# Mobile App
cd ewardrobe-app
npm install && npx expo start
```

---

## Feature 4 — Smart Wardrobe Organization System

**Branch:** `Smart-Wardrobe-Organization-System`

### Overview
Uses machine learning to automatically cluster clothing items for optimal wardrobe layout, detect overused/underused items, track laundry cycles, and provide sustainability insights — all powered by a Sri Lankan wardrobe dataset.

### Project Structure
```
├── backend/
│   ├── main.py                         # FastAPI app & all endpoints
│   ├── database.py                     # SQLAlchemy models (SQLite)
│   ├── crud.py                         # DB read/write operations
│   ├── ml_engine.py                    # K-Means + Isolation Forest engine
│   ├── laundry_logic.py                # Wash cycle tracking
│   ├── train_models.py                 # Model training script
│   └── models/
│       ├── kmeans_storage.pkl          # Wardrobe layout clustering
│       ├── isolation_forest.pkl        # Anomaly detection
│       ├── fabric_encoder.pkl
│       ├── occasion_encoder.pkl
│       └── sustainability_regressor.pkl
├── data/
│   └── sri_lanka_smart_wardrobe_dataset.csv
└── frontend/                           # React Native app
    └── src/
        ├── api/wardrobeApi.js
        └── screens/
```

### ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Wardrobe Layout | K-Means Clustering | Group items for smart physical arrangement |
| Anomaly Detection | Isolation Forest | Flag overused / underused clothing |
| Sustainability Score | Regression | Predict eco-score for new items |

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/items/organized` | GET | All items with layout cluster assignments |
| `/items/wear/{item_id}` | POST | Record a wear event |
| `/items/wash/{item_id}` | POST | Mark item as washed |
| `/items/insights` | GET | Anomaly report (dirty count, overused, underused) |

**Example — Insights Response:**
```json
{
  "dirty_count": 5,
  "underused": ["W014", "W021"],
  "overused": ["W003"]
}
```

### Dataset

**Sri Lanka Smart Wardrobe Dataset** — columns include `item_id`, `category`, `color`, `fabric`, `occasion`, `total_wear_count`, `current_cycle_wears`, `max_wears_before_wash`, `status`, `sustainability_score`.

### Setup
```bash
cd backend
pip install -r requirements.txt
python train_models.py        # first-time model training
uvicorn main:app --reload     # auto-loads dataset on first run

# Frontend
cd frontend
npm install && npx expo start
```

---

## Tech Stack Summary

| Layer | Technologies |
|-------|-------------|
| **AI / ML** | PyTorch, PyTorch Geometric, scikit-learn, OpenCV, torchvision |
| **Backend** | FastAPI, SQLAlchemy, Uvicorn, WebSocket |
| **Mobile** | React Native (Expo), TypeScript |
| **Data** | Kaggle Fashion Dataset, Sri Lankan Wardrobe Dataset |

---

## Repository Structure

```
e-wardrobe-app/
├── main                                    ← This file (project overview)
├── feature/ai-clothing-classification      ← Eshani Perera
├── AI-Outfit-Recommendation-&-Matchmaking  ← Outfit recommendation
├── AI-visualization                        ← Virtual try-on
└── Smart-Wardrobe-Organization-System      ← Wardrobe organization
```
