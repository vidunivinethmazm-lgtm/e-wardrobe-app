# E-Wardrobe AI — Feature: AI Visualization & Virtual Try-On

Branch: `AI-visualization`

## Overview

This branch implements the **virtual try-on and AI visualization pipeline** for the E-Wardrobe application. Users take a selfie, submit body measurements, and the system generates a realistic avatar wearing recommended outfits. The backend runs a 4-stage deep learning pipeline covering body calibration, face keypoint detection, outfit recommendation, and avatar rendering.

---

## Project Structure

```
├── vineths_wardrobe/                       # Python backend
│   ├── app.py                              # FastAPI server (REST + WebSocket)
│   ├── config.py                           # App configuration
│   ├── mobile_app.py                       # Mobile-specific API layer
│   ├── requirements.txt
│   │
│   ├── src/
│   │   ├── virtual_tryon_pipeline.py       # Main 4-stage pipeline orchestrator
│   │   ├── outfit_recommender.py           # Outfit matching logic
│   │   ├── avatar_manager.py               # Avatar generation & rendering
│   │   ├── body_calibration.py             # Body type & size estimation
│   │   ├── face_keypoint_model.py          # Face landmark detection
│   │   ├── face_processor.py               # Face image processing
│   │   ├── wardrobe_database.py            # Wardrobe item storage
│   │   ├── terminal_evaluator.py           # CLI accuracy evaluation
│   │   ├── model_registry.py               # Multi-model accuracy reporting
│   │   │
│   │   ├── models/                         # Stage model implementations
│   │   │   ├── stage1_body_models.py       # Body type + size classifiers
│   │   │   ├── stage2_face_models.py       # Face keypoint models
│   │   │   ├── stage3_outfit_models.py     # Outfit recommendation models
│   │   │   └── stage4_avatar_models.py     # Avatar generation models
│   │   │
│   │   └── mobile/                         # Lightweight mobile model variants
│   │       ├── body_models.py
│   │       ├── face_models.py
│   │       ├── trainer.py
│   │       └── visualiser.py
│   │
│   ├── frontend/                           # Web frontend
│   │   ├── index.html
│   │   ├── accuracy.html
│   │   └── js/
│   │       ├── app.js
│   │       ├── selfie_capture.js
│   │       └── tryon_renderer.js
│   │
│   ├── scripts/
│   │   ├── train_all_models.py             # Full training pipeline
│   │   ├── generate_test_assets.py
│   │   ├── run_swagger_tests.py
│   │   └── check_accuracy.py
│   │
│   └── tests/
│       └── test_api.py
│
└── ewardrobe-app/                          # React Native mobile app
    ├── App.tsx
    ├── src/
    │   ├── api/client.ts                   # API client
    │   ├── components/
    │   │   ├── MetricBar.tsx               # Accuracy metric display
    │   │   ├── OutfitCard.tsx              # Outfit result card
    │   │   └── StepNav.tsx                 # Step navigation
    │   ├── constants/theme.ts
    │   └── screens/
    │       ├── SelfieScreen.tsx            # Camera capture
    │       ├── MeasurementsScreen.tsx      # Body measurements input
    │       ├── ProcessingScreen.tsx        # Real-time pipeline progress
    │       └── TryOnScreen.tsx             # Final avatar + outfit result
    ├── package.json
    └── tsconfig.json
```

---

## 4-Stage Pipeline

### Stage 1 — Body Calibration
Estimates the user's **body type** (Hourglass, Pear, Apple, Rectangle, Inverted Triangle) and **standard size** (XS–XXL) from a selfie and optional measurements. Multiple classifiers are evaluated and the best-performing model is selected automatically.

### Stage 2 — Face Processing
Detects **facial keypoints** and extracts face region data to enable realistic avatar face mapping. Uses a CNN-based landmark detection model.

### Stage 3 — Outfit Recommendation
Recommends outfits from the wardrobe database filtered by body type, occasion, season, and style preference. Evaluated across 6 real-world scenarios with top-5 accuracy.

### Stage 4 — Avatar Generation
Composites the detected face and recommended outfit onto a body avatar matched to the user's measurements. Outputs a rendered try-on image.

---

## API

Backend runs on `http://localhost:8000`. Full Swagger docs at `/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tryon` | POST | Upload selfie + measurements, run full pipeline |
| `/api/recommend` | POST | Get outfit recommendations only |
| `/api/accuracy` | GET | Get multi-stage model accuracy report |
| `/ws/tryon/{session_id}` | WebSocket | Real-time pipeline progress streaming |

### Example Try-On Request
```
POST /api/tryon
Content-Type: multipart/form-data

selfie: <image file>
height: 165
weight: 58
occasion: "Party"
```

---

## Setup & Run

### Backend

```bash
cd vineths_wardrobe
pip install -r requirements.txt
python scripts/train_all_models.py    # train all 4 stages (first time only)
uvicorn app:app --reload --port 8000
```

### Mobile App

```bash
cd ewardrobe-app
npm install
npx expo start
```

### Accuracy Report

```bash
cd vineths_wardrobe
python run_accuracy.py
```

---

## Tech Stack

- **AI/ML** — PyTorch, OpenCV, NumPy, scikit-learn
- **Backend** — FastAPI, WebSocket (real-time), Uvicorn
- **Mobile** — React Native (Expo), TypeScript
- **Web** — Vanilla JS frontend with Swagger/ReDoc

---

## Team Member

**[Branch Owner]** — AI Visualization & Virtual Try-On
