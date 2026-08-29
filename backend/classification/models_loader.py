import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, class_map_path):
    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {
        int(v): k
        for k, v in class_to_idx.items()
    }

    num_classes = len(class_to_idx)

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model, idx_to_class


type_model, type_idx_to_class = load_model(
    MODEL_DIR / "mobilenetv2_articleType_best.pth",
    MODEL_DIR / "articleType_class_to_idx.json"
)

color_model, color_idx_to_class = load_model(
    MODEL_DIR / "mobilenetv2_color_best.pth",
    MODEL_DIR / "color_class_to_idx.json"
)

gender_model, gender_idx_to_class = load_model(
    MODEL_DIR / "mobilenetv2_gender_best.pth",
    MODEL_DIR / "gender_class_to_idx.json"
)

season_model, season_idx_to_class = load_model(
    MODEL_DIR / "mobilenetv2_season_best.pth",
    MODEL_DIR / "season_class_to_idx.json"
)