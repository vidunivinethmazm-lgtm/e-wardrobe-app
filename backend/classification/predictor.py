import os
from pathlib import Path
from threading import Lock

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label as label_components,
)

# Keep downloaded segmentation weights on the project drive instead of the
# space-constrained user profile on C:.
PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("U2NET_HOME", str(PROJECT_DIR / "models" / "rembg"))

from rembg import new_session, remove
from torchvision import transforms

from backend.classification.color_analyzer import dominant_color
from backend.classification.material_detector import predict_garment_type, predict_material

from backend.classification.models_loader import (
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


_cloth_session = None
_cloth_session_lock = Lock()

_subject_session = None
_subject_session_lock = Lock()

# Bound the working resolution so alpha matting stays fast and memory-light.
MAX_WORK_SIZE = 1024

# Below this the cloth-parser mask is torn / hole-riddled garbage and is
# ignored; a real garment shape fills much more of its own bounding box.
MIN_PARSE_FILL_RATIO = 0.28

# Below this the cloth parser effectively found nothing usable.
MIN_PARSE_COVERAGE = 0.015


def get_cloth_session():
    """Load the clothing parser once, on the first prediction request."""
    global _cloth_session
    if _cloth_session is None:
        with _cloth_session_lock:
            if _cloth_session is None:
                _cloth_session = new_session("u2net_cloth_seg")
    return _cloth_session


def get_subject_session():
    """General-purpose subject cut-out model (weights already cached on disk)."""
    global _subject_session
    if _subject_session is None:
        with _subject_session_lock:
            if _subject_session is None:
                _subject_session = new_session("u2net")
    return _subject_session


def _limit_size(image: Image.Image) -> Image.Image:
    """Downscale very large photos before segmentation / matting."""
    longest = max(image.size)
    if longest <= MAX_WORK_SIZE:
        return image
    scale = MAX_WORK_SIZE / longest
    return image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.LANCZOS,
    )


def clean_and_crop_cloth(image: Image.Image) -> Image.Image:
    """Keep one solid garment shape, drop specks / spikes, feather the edge."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    solid = alpha >= 128

    if solid.any():
        # Fill interior gaps the model punched through prints, buttons, or
        # thin fabric, so the garment reads as one piece.
        solid = binary_fill_holes(solid)

        # Erode-then-dilate: removes thin protrusions (hanger hooks, shadow
        # slivers, ragged fringe) without eating into the garment body.
        radius = max(2, int(min(alpha.shape) * 0.012))
        structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
        solid = binary_opening(solid, structure=structure)

        # Keep only the largest remaining blob.
        components, component_count = label_components(solid)
        if component_count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            solid = components == sizes.argmax()

        cleaned = np.where(solid, alpha, 0).astype(np.float32)

        # Feather the boundary so the cut edge is smooth, not stair-stepped.
        cleaned = gaussian_filter(cleaned, sigma=1.2)
        cleaned_alpha = np.clip(cleaned, 0, 255).astype(np.uint8)
        rgba.putalpha(Image.fromarray(cleaned_alpha, mode="L"))

    bounds = rgba.getchannel("A").getbbox()
    if bounds:
        left, top, right, bottom = bounds
        padding = max(8, int(max(right - left, bottom - top) * 0.04))
        bounds = (
            max(0, left - padding),
            max(0, top - padding),
            min(rgba.width, right + padding),
            min(rgba.height, bottom + padding),
        )
        rgba = rgba.crop(bounds)

    return rgba


def _bbox_fill_ratio(alpha: np.ndarray) -> float:
    """How much of its own bounding box a mask's solid pixels occupy."""
    solid = alpha >= 128
    if not solid.any():
        return 0.0
    ys, xs = np.where(solid)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(solid.sum()) / float(bbox_area) if bbox_area else 0.0


def _cloth_region(parsed_alpha: np.ndarray):
    """Soft 0..1 mask of the garment area from the cloth parser.

    Returns None when the parser's output is too torn or too empty to trust.
    This mask is what excludes worn-on skin, hair and hands - the general
    subject cut-out cannot tell those apart from the garment.
    """
    if _bbox_fill_ratio(parsed_alpha) < MIN_PARSE_FILL_RATIO:
        return None

    solid = parsed_alpha >= 128
    if solid.mean() < MIN_PARSE_COVERAGE:
        return None

    region = binary_fill_holes(solid)

    smooth_r = max(2, int(min(parsed_alpha.shape) * 0.01))
    region = binary_closing(
        region,
        structure=np.ones((smooth_r * 2 + 1, smooth_r * 2 + 1), dtype=bool),
    )

    components, component_count = label_components(region)
    if component_count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        region = components == sizes.argmax()

    # Pull the boundary slightly inward so skin right next to a collar or
    # cuff falls outside the garment region instead of being kept.
    erode_r = max(1, int(min(parsed_alpha.shape) * 0.006))
    region = binary_erosion(
        region,
        structure=np.ones((erode_r * 2 + 1, erode_r * 2 + 1), dtype=bool),
    )
    if not region.any():
        return None

    soft = gaussian_filter(region.astype(np.float32), sigma=2.0)
    return np.clip(soft, 0.0, 1.0)


def extract_cloth_only(input_image: Image.Image, predicted_type: str) -> Image.Image:
    """Cut out just the garment on a transparent background.

    Two models are combined: the alpha-matted general subject cut-out gives
    smooth, artefact-free edges, and the cloth parser decides which of those
    pixels are actually garment - so worn-on skin, hair and hands are
    dropped. If the cloth parser fails, the matted cut-out is used as-is.
    """
    lower_types = {"Briefs", "Jeans", "Shorts", "Trousers"}
    upper_types = {"Kurtas", "Shirts", "Tops", "Tshirts"}

    if predicted_type in lower_types:
        cloth_category = "lower"
    elif predicted_type in upper_types:
        cloth_category = "upper"
    else:
        cloth_category = None

    work = _limit_size(input_image.convert("RGBA"))

    # Alpha matting gives the smoothest edge but needs a ~1.9 GB transient
    # allocation per image - enough to tip this memory-constrained box over
    # (it always fell back anyway). Set EWARDROBE_ALPHA_MATTING=1 to opt in.
    use_matting = os.getenv("EWARDROBE_ALPHA_MATTING") == "1"
    matting_opts = dict(
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    ) if use_matting else {}
    try:
        matted = remove(
            work, session=get_subject_session(),
            post_process_mask=True, **matting_opts,
        ).convert("RGBA")
    except Exception as exc:
        print(f"Subject removal fallback ({type(exc).__name__}: {exc})")
        matted = remove(work, session=get_subject_session()).convert("RGBA")

    # The cloth parser is only a useful region gate when we can tell it which
    # half of the body to look at. Without a category (dresses, gowns, sarees,
    # jumpsuits, ...) it returns a stacked 3-panel mask that can't be used, so
    # fall back to the matted full-subject cut-out.
    region = None
    if cloth_category:
        try:
            parsed = remove(
                work,
                session=get_cloth_session(),
                post_process_mask=True,
                cloth_category=cloth_category,
            ).convert("RGBA")
            region = _cloth_region(np.asarray(parsed.getchannel("A")))
        except Exception as exc:
            print(f"Cloth parsing skipped: {exc}")

    matted_alpha = np.asarray(matted.getchannel("A"))
    if region is not None and region.shape == matted_alpha.shape:
        garment_alpha = np.clip(matted_alpha.astype(np.float32) * region, 0, 255).astype(np.uint8)
        result = matted.copy()
        result.putalpha(Image.fromarray(garment_alpha, mode="L"))
    else:
        result = matted

    return clean_and_crop_cloth(result)


def process_clothing_image(image_path: str, predicted_type: str) -> str:
    """Remove everything except the garment and save a transparent PNG."""
    input_image = Image.open(image_path).convert("RGBA")
    cloth_only = extract_cloth_only(input_image, predicted_type)
    processed_dir = Path(image_path).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{Path(image_path).stem}_no_bg.png"
    cloth_only.save(processed_path)
    return str(processed_path)


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

    # Material is a separate pretrained zero-shot prediction. The existing
    # background-removal and four custom prediction paths below are unchanged.
    try:
        material_result = predict_material(input_image)
    except Exception as exc:
        # An optional material-model download/API failure must not block the
        # established front-image prediction pipeline.
        print(f"Material prediction unavailable: {exc}")
        material_result = {"label": "Unavailable", "confidence": 0.0}

    # Preserve the established classifier input: the default rembg model
    # removes only the scene background and keeps the photographed subject.
    # Reuse the cached u2net session - a bare remove() rebuilds a 176 MB ONNX
    # session on every request and eventually exhausts memory ("bad allocation").
    subject_only = remove(input_image, session=get_subject_session()).convert("RGBA")
    rgb_image = Image.new("RGB", subject_only.size, "white")
    rgb_image.paste(subject_only, mask=subject_only.getchannel("A"))

    image_tensor = transform(rgb_image).unsqueeze(0).to(device)

    type_result = predict_attribute(type_model, type_idx_to_class, image_tensor)
    color_result = predict_attribute(color_model, color_idx_to_class, image_tensor)
    gender_result = predict_attribute(gender_model, gender_idx_to_class, image_tensor)
    season_result = predict_attribute(season_model, season_idx_to_class, image_tensor)

    # The 23-class article-type model has no Dress / Skirt / Saree / Gown /
    # Jacket / etc. A zero-shot CLIP pass over the full garment vocabulary
    # overrides it whenever it confidently sees one of those missing types.
    try:
        garment = predict_garment_type(input_image)
        if not garment["is_native"] and garment["confidence"] >= 0.30:
            type_result = {"label": garment["label"], "confidence": garment["confidence"]}
    except Exception as exc:
        print(f"Garment-type refinement unavailable: {exc}")

    predicted_type = type_result["label"]
    cloth_only = extract_cloth_only(input_image, predicted_type)

    # The 17-class colour model confuses neighbours (pink->red, coral->orange).
    # Read the dominant colour straight off the isolated garment; keep the
    # model's answer only when the garment has no clear single colour.
    try:
        px_color = dominant_color(cloth_only)
        if px_color["label"] and px_color["confidence"] >= 0.35:
            color_result = {"label": px_color["label"], "confidence": px_color["confidence"]}
    except Exception as exc:
        print(f"Colour analysis unavailable: {exc}")

    processed_dir = Path(image_path).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{Path(image_path).stem}_no_bg.png"
    cloth_only.save(processed_path)

    return {
        "type": type_result["label"],
        "type_confidence": type_result["confidence"],

        "color": color_result["label"],
        "color_confidence": color_result["confidence"],

        "gender": gender_result["label"],
        "gender_confidence": gender_result["confidence"],

        "season": season_result["label"],
        "season_confidence": season_result["confidence"],

        "material": material_result["label"],
        "material_confidence": material_result["confidence"],

        "processed_image_path": str(processed_path)
    }
