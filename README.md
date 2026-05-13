# E-Wardrobe AI — Feature: AI Clothing Classification

Branch: `feature/ai-clothing-classification`

## Overview

This branch implements the AI-powered clothing classification engine for the E-Wardrobe application. It trains MobileNetV2-based deep learning models to automatically classify clothing items by **article type**, **color**, **gender**, and **season** from images, and exposes predictions through a FastAPI backend consumed by the React Native mobile app.

---

## Project Structure

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
│
├── models/
│   ├── articleType_class_to_idx.json   # Article type label mapping
│   ├── color_class_to_idx.json         # Color label mapping
│   ├── gender_class_to_idx.json        # Gender label mapping
│   ├── season_class_to_idx.json        # Season label mapping
│   ├── mobilenetv2_training_history.json
│   └── mobilenetv2_color_training_history.json
│
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI app & endpoints
│       ├── models_loader.py            # Loads trained model weights
│       ├── predictor.py                # Runs inference on uploaded images
│       └── __init__.py
│
├── mobile_app/                         # Expo React Native app
│   ├── app/
│   │   ├── _layout.tsx
│   │   ├── modal.tsx
│   │   └── (tabs)/
│   │       ├── index.tsx               # Home screen
│   │       └── explore.tsx
│   ├── components/                     # Reusable UI components
│   ├── constants/theme.ts
│   ├── hooks/
│   ├── firebaseConfig.ts
│   └── package.json
│
├── data/
│   └── raw/styles.csv                  # Kaggle Fashion dataset metadata
│
└── outputs/
    └── classification_report.txt       # Model evaluation report
```

---

## Models

| Model | Architecture | Task |
|-------|-------------|------|
| Article Type | MobileNetV2 (fine-tuned) | Multi-class clothing type classification |
| Color | MobileNetV2 (fine-tuned) | Color category classification |
| Gender | MobileNetV2 (fine-tuned) | Gender classification (Men/Women/Boys/Girls/Unisex) |
| Season | MobileNetV2 (fine-tuned) | Season classification (Summer/Winter/Fall/Spring) |

All models are trained on the [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

---

## Backend API

The FastAPI backend runs on `http://localhost:8000`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload a clothing image and get classification results |
| `/docs` | GET | Swagger UI |

### Example Response
```json
{
  "article_type": "Shirts",
  "color": "Blue",
  "gender": "Men",
  "season": "Summer"
}
```

---

## Setup & Run

### Backend

```bash
cd backend
pip install fastapi uvicorn torch torchvision pillow rembg
uvicorn app.main:app --reload
```

### Mobile App

```bash
cd mobile_app
npm install
npx expo start
```

### Training (Notebooks)

Run the notebooks in order:
1. `02_create_splits.ipynb` — split dataset
2. `04_train_mobilenetv2.ipynb` — train article type model
3. `08_train_color_model.ipynb` — train color model
4. `09_train_gender_model.ipynb` — train gender model
5. `10_train_season_model.ipynb` — train season model

---

## Tech Stack

- **Python** — PyTorch, FastAPI, torchvision, rembg
- **Mobile** — React Native (Expo), TypeScript, Firebase
- **Data** — Kaggle Fashion Product Images Dataset (~44,000 images)
- **Models** — MobileNetV2 (transfer learning)

---

## Team Member

**Eshani Perera** — AI Clothing Classification
