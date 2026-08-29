from pathlib import Path
from datetime import datetime
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.classification.predictor import predict_clothing, process_clothing_image
from backend.classification.models_loader import device
from backend.classification.trends import (
    analyze_wardrobe_items,
    get_fashion_trends,
    get_trend_matches,
    recommend_for_event,
)


BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TrendRequest(BaseModel):
    items: list[dict] = []


class ScheduleSuggestionRequest(BaseModel):
    items: list[dict] = []
    event: dict = {}


app = FastAPI(
    title="E-Wardrobe AI Backend",
    description="AI clothing classification backend with background removal and multi-attribute prediction",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)


@app.get("/")
def home():
    return {
        "message": "E-Wardrobe AI Backend Running",
        "device": str(device)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device)
    }


@app.get("/trends")
def trends(clothing_type: str = "", color: str = "", season: str = "", gender: str = ""):
    return get_trend_matches(clothing_type, color, season, gender)


@app.get("/fashion-trends")
def fashion_trends(audience: str = "general"):
    return get_fashion_trends(audience)


@app.post("/wardrobe/trend-analysis")
def wardrobe_trend_analysis(request: TrendRequest):
    return analyze_wardrobe_items(request.items)


@app.post("/schedule/suggestion")
def schedule_suggestion(request: ScheduleSuggestionRequest):
    return recommend_for_event(request.items, request.event)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    back_file: UploadFile | None = File(None),
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        return {
            "error": "Invalid file type. Please upload JPG, JPEG, PNG, or WEBP image."
        }

    saved_filename = f"{timestamp}_{file.filename}"
    saved_path = UPLOAD_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    back_url = None
    if back_file is not None and back_file.filename:
        back_extension = Path(back_file.filename).suffix.lower()
        if back_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
            return {
                "error": "Invalid back image type. Please upload JPG, JPEG, PNG, or WEBP image."
            }

        back_filename = f"{timestamp}_back_{Path(back_file.filename).name}"
        back_path = UPLOAD_DIR / back_filename
        with back_path.open("wb") as buffer:
            shutil.copyfileobj(back_file.file, buffer)
        back_url = f"/uploads/{back_filename}"

    prediction = predict_clothing(str(saved_path))

    back_processed_url = None
    if back_file is not None and back_file.filename:
        back_processed_path = Path(process_clothing_image(
            str(back_path),
            prediction["type"],
        ))
        back_processed_url = f"/uploads/processed/{back_processed_path.name}"

    processed_path = Path(prediction["processed_image_path"])

    processed_url = f"/uploads/processed/{processed_path.name}"
    original_url = f"/uploads/{saved_filename}"
    trend_analysis = get_trend_matches(
        prediction["type"],
        prediction["color"],
        prediction["season"],
        prediction["gender"]
    )

    return {
        "status": "success",
        "original_image_url": original_url,
        "back_image_url": back_url,
        "back_processed_image_url": back_processed_url,
        "processed_image_url": processed_url,
        "prediction": {
            "type": prediction["type"],
            "type_confidence": prediction["type_confidence"],
            "color": prediction["color"],
            "color_confidence": prediction["color_confidence"],
            "gender": prediction["gender"],
            "gender_confidence": prediction["gender_confidence"],
            "season": prediction["season"],
            "season_confidence": prediction["season_confidence"],
            "material": prediction["material"],
            "material_confidence": prediction["material_confidence"],
        },
        "trend_analysis": trend_analysis
    }
