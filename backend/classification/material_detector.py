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


# ── Zero-shot garment type ──────────────────────────────────────────────────
#
# The custom article-type classifier only knows 23 classes and has no
# Dresses / Skirt / Saree / Gown / Jacket / etc. This CLIP pass covers the
# full spread of everyday garments so a dress is never forced into "Tshirts".
# Each (phrase -> canonical label) pair; several phrases can share a label.

GARMENT_TYPES: list[tuple[str, str]] = [
    ("a t-shirt", "Tshirts"),
    ("a polo shirt", "Tshirts"),
    ("a tank top", "Tops"),
    ("a casual top", "Tops"),
    ("a blouse", "Tops"),
    ("a tunic top", "Tops"),
    ("a crop top", "Tops"),
    ("a formal shirt", "Shirts"),
    ("a kurta", "Kurtas"),
    ("a kurti", "Kurtas"),
    ("a dress", "Dresses"),
    ("a frock", "Dresses"),
    ("a sundress", "Dresses"),
    ("a maxi dress", "Dresses"),
    ("a midi dress", "Dresses"),
    ("a bodycon dress", "Dresses"),
    ("a party dress", "Dresses"),
    ("a wedding gown", "Gowns"),
    ("an evening gown", "Gowns"),
    ("a ball gown", "Gowns"),
    ("a saree", "Saree"),
    ("a sari", "Saree"),
    ("a half saree", "Saree"),
    ("a lehenga", "Lehenga"),
    ("a lehenga choli", "Lehenga"),
    ("a salwar kameez", "Salwar"),
    ("a shalwar suit", "Salwar"),
    ("an anarkali suit", "Salwar"),
    ("a jumpsuit", "Jumpsuits"),
    ("a romper", "Jumpsuits"),
    ("a playsuit", "Jumpsuits"),
    ("a skirt", "Skirts"),
    ("a mini skirt", "Skirts"),
    ("a maxi skirt", "Skirts"),
    ("a pleated skirt", "Skirts"),
    ("a denim skirt", "Skirts"),
    ("trousers", "Trousers"),
    ("formal pants", "Trousers"),
    ("chinos", "Trousers"),
    ("jeans", "Jeans"),
    ("denim jeans", "Jeans"),
    ("shorts", "Shorts"),
    ("denim shorts", "Shorts"),
    ("leggings", "Leggings"),
    ("track pants", "Trackpants"),
    ("a jacket", "Jackets"),
    ("a denim jacket", "Jackets"),
    ("a bomber jacket", "Jackets"),
    ("a blazer", "Blazers"),
    ("a suit blazer", "Blazers"),
    ("an overcoat", "Coats"),
    ("a winter coat", "Coats"),
    ("a trench coat", "Coats"),
    ("a sweater", "Sweaters"),
    ("a pullover", "Sweaters"),
    ("a jumper", "Sweaters"),
    ("a cardigan", "Cardigans"),
    ("a hoodie", "Sweatshirts"),
    ("a sweatshirt", "Sweatshirts"),
    ("a shrug", "Shrugs"),
    ("a shawl", "Shawls"),
    ("a pashmina shawl", "Shawls"),
]

# Canonical labels the custom classifier already produces well - CLIP is only
# trusted to *override* into a label OUTSIDE this set.
_NATIVE_TYPES = {"Tshirts", "Tops", "Shirts", "Kurtas", "Trousers", "Jeans", "Shorts"}


def predict_garment_type(image: Image.Image) -> dict:
    """Zero-shot garment type over the full everyday-wear vocabulary."""
    model, processor = _load_model()
    prompts = [f"a photo of {phrase}" for phrase, _ in GARMENT_TYPES]
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
        probs = logits.softmax(dim=1)[0]

    # Aggregate probability per canonical label (phrases share labels).
    scores: dict[str, float] = {}
    for (_, label), p in zip(GARMENT_TYPES, probs.tolist()):
        scores[label] = scores.get(label, 0.0) + p

    label = max(scores, key=scores.get)
    return {
        "label": label,
        "confidence": round(scores[label], 4),
        "is_native": label in _NATIVE_TYPES,
    }
