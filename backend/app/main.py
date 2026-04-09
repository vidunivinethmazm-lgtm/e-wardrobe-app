from pathlib import Path
from datetime import datetime
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.predictor import predict_clothing
from backend.app.models_loader import device


BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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

    prediction = predict_clothing(str(saved_path))

    processed_path = Path(prediction["processed_image_path"])

    processed_url = f"/uploads/processed/{processed_path.name}"
    original_url = f"/uploads/{saved_filename}"

    return {
        "status": "success",
        "original_image_url": original_url,
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
        }
    }