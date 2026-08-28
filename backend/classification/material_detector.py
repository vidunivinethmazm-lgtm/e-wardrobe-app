"""Pretrained zero-shot clothing material detection.

The model is loaded lazily so starting the API does not add any work to the
existing four custom classifiers. The weights are downloaded and cached by
Hugging Face the first time material prediction is used.
"""

import os
from threading import Lock

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = os.getenv("MATERIAL_MODEL_NAME", "openai/clip-vit-base-patch32")
MATERIALS = [
    "cotton",
    "denim",
    "leather",
    "wool",
    "silk",
    "linen",
    "polyester",
    "nylon",
    "velvet",
    "knit",
    "fleece",
    "suede",
    "satin",
    "chiffon",
]

_model = None
_processor = None
_load_lock = Lock()


def _load_model():
    global _model, _processor

    if _model is None or _processor is None:
        with _load_lock:
            if _model is None or _processor is None:
                _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                _model = CLIPModel.from_pretrained(MODEL_NAME)
                _model.eval()

    return _model, _processor


def predict_material(image: Image.Image) -> dict:
    """Return the most likely material and its zero-shot probability."""
    model, processor = _load_model()
    prompts = [f"a close-up photo of {material} clothing fabric" for material in MATERIALS]
    inputs = processor(
        text=prompts,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )

    model_device = next(model.parameters()).device
    inputs = {name: value.to(model_device) for name, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits_per_image
        probabilities = logits.softmax(dim=1)[0]
        confidence, prediction = probabilities.max(dim=0)

    return {
        "label": MATERIALS[prediction.item()].title(),
        "confidence": round(confidence.item(), 4),
    }
