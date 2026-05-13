# E-Wardrobe AI — Feature: Smart Wardrobe Organization System

Branch: `Smart-Wardrobe-Organization-System`

## Overview

This branch implements the **smart wardrobe organization and management system** for the E-Wardrobe application. It uses machine learning to automatically cluster clothing items for optimal wardrobe layout, detect anomalies (overused or underused items), track laundry cycles, and provide sustainability insights — all backed by a real Sri Lankan wardrobe dataset.

---

## Project Structure

```
├── backend/
│   ├── main.py                         # FastAPI application & all endpoints
│   ├── database.py                     # SQLAlchemy database models & setup
│   ├── crud.py                         # Database read/write operations
│   ├── schemas.py                      # Pydantic request/response models
│   ├── ml_engine.py                    # ML engine (K-Means + Isolation Forest)
│   ├── laundry_logic.py                # Laundry cycle & wash tracking logic
│   ├── train_models.py                 # Model training script
│   ├── generate_dataset.py             # Dataset generation utility
│   ├── requirements.txt
│   └── models/
│       ├── kmeans_storage.pkl          # Trained K-Means (storage layout)
│       ├── kmeans_scaler.pkl           # Feature scaler for K-Means
│       ├── isolation_forest.pkl        # Trained Isolation Forest (anomaly detection)
│       ├── fabric_encoder.pkl          # Label encoder for fabric types
│       ├── occasion_encoder.pkl        # Label encoder for occasions
│       └── sustainability_regressor.pkl # Sustainability score regressor
│
├── data/
│   └── sri_lanka_smart_wardrobe_dataset.csv   # Sri Lankan wardrobe dataset
│
└── frontend/
    ├── App.js                          # Main React Native app
    ├── app.json
    ├── babel.config.js
    ├── package.json
    └── src/
        ├── api/
        │   ├── config.js               # API base URL config
        │   └── wardrobeApi.js          # API call functions
        ├── store/
        │   └── index.js                # State management
        └── theme/
            └── colors.js              # App color theme
```

---

## Features

### Smart Wardrobe Layout (K-Means Clustering)
Items are automatically grouped into layout clusters based on features like fabric type, occasion, color, and wear frequency. Clusters tell the app how to physically arrange clothes (e.g., casual cluster vs. formal cluster) for easy access.

### Anomaly Detection (Isolation Forest)
Detects items that are:
- **Overused** — worn too frequently without washing (risk of damage)
- **Underused** — rarely worn items that could be donated or rotated

### Laundry & Wear Tracking
- Tracks `current_cycle_wears` vs `max_wears_before_wash` per item
- Automatically marks items as **Dirty** when the wash threshold is exceeded
- Marks items as **Clean** after a wash is recorded

### Sustainability Scoring
Each item has a sustainability score derived from fabric type, wear frequency, and laundry cycles. A regression model predicts sustainability scores for new items.

---

## API

Backend runs on `http://localhost:8000`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/items/organized` | GET | Get all wardrobe items with ML layout clusters |
| `/items/wear/{item_id}` | POST | Record a wear event for an item |
| `/items/wash/{item_id}` | POST | Mark item as washed (reset cycle count) |
| `/items/insights` | GET | Get anomaly report (dirty count, overused, underused) |
| `/docs` | GET | Swagger UI |

### Example — Get Organized Wardrobe
```
GET /items/organized
```
```json
[
  {
    "item_id_str": "W001",
    "category": "Saree",
    "color": "Red",
    "fabric": "Silk",
    "occasion": "Wedding",
    "status": "Clean",
    "sustainability_score": 8.4,
    "layout_group": 2
  }
]
```

### Example — Get Insights
```
GET /items/insights
```
```json
{
  "dirty_count": 5,
  "underused": ["W014", "W021"],
  "overused": ["W003"]
}
```

---

## Setup & Run

### Backend

```bash
cd backend
pip install -r requirements.txt
python train_models.py        # train ML models (first time only)
uvicorn main:app --reload --port 8000
```

The backend auto-loads the Sri Lankan dataset from `data/sri_lanka_smart_wardrobe_dataset.csv` on first startup.

### Mobile App (Frontend)

```bash
cd frontend
npm install
npx expo start
```

---

## Dataset

**Sri Lanka Smart Wardrobe Dataset** (`sri_lanka_smart_wardrobe_dataset.csv`)

Columns include:
| Column | Description |
|--------|-------------|
| `item_id` | Unique item identifier |
| `category` | Clothing category (Saree, Shirt, etc.) |
| `color` | Color of the item |
| `fabric` | Fabric type (Cotton, Silk, Linen, etc.) |
| `occasion` | Suitable occasion |
| `total_wear_count` | Total number of times worn |
| `current_cycle_wears` | Wears since last wash |
| `max_wears_before_wash` | Wash threshold |
| `status` | Clean / Dirty |
| `sustainability_score` | Eco-sustainability rating (0–10) |

---

## Tech Stack

- **AI/ML** — scikit-learn (K-Means, Isolation Forest, regression), pandas
- **Backend** — FastAPI, SQLAlchemy (SQLite), Uvicorn
- **Mobile** — React Native (Expo), JavaScript
- **Data** — Sri Lankan smart wardrobe dataset (custom)

---

## Team Member

**[Branch Owner]** — Smart Wardrobe Organization System
