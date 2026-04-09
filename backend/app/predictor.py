from pathlib import Path

import torch
from PIL import Image
from rembg import remove
from torchvision import transforms

from backend.app.models_loader import (
    device,
    type_model,
    type_idx_to_class,
    color_model,
    color_idx_to_class,
    gender_model,
    gender_idx_to_class,
    season_model,
    season_idx_to_class,
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_attribute(model, idx_to_class, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    return {
        "label": idx_to_class[prediction.item()],
        "confidence": round(confidence.item(), 4)
    }


def predict_clothing(image_path: str):
    input_image = Image.open(image_path).convert("RGBA")

    bg_removed = remove(input_image)
    rgb_image = bg_removed.convert("RGB")

    processed_dir = Path(image_path).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_path = processed_dir / f"{Path(image_path).stem}_no_bg.png"
    rgb_image.save(processed_path)

    image_tensor = transform(rgb_image).unsqueeze(0).to(device)

    type_result = predict_attribute(type_model, type_idx_to_class, image_tensor)
    color_result = predict_attribute(color_model, color_idx_to_class, image_tensor)
    gender_result = predict_attribute(gender_model, gender_idx_to_class, image_tensor)
    season_result = predict_attribute(season_model, season_idx_to_class, image_tensor)

    return {
        "type": type_result["label"],
        "type_confidence": type_result["confidence"],

        "color": color_result["label"],
        "color_confidence": color_result["confidence"],

        "gender": gender_result["label"],
        "gender_confidence": gender_result["confidence"],

        "season": season_result["label"],
        "season_confidence": season_result["confidence"],

        "processed_image_path": str(processed_path)
    }