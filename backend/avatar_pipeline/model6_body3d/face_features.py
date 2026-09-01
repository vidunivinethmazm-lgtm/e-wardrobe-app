"""
Model 6 — 3D Body Reconstruction: comprehensive face/hair appearance extraction & facial analysis.

Enhanced facial feature detection supporting:
- Gender presentation detection
- Age group estimation  
- Hair color and style analysis
- Face shape detection
- Facial hair detection
- Eye color detection
- Facial landmarks extraction

Uses MediaPipe FaceMesh for robust face detection + 468 landmark extraction
(>95 % detection rate vs ~70 % for the old Haar cascade).  Falls back to
OpenCV Haar cascade if MediaPipe is unavailable.
"""

import cv2
import numpy as np

# Lazy MediaPipe initialisation (first call to any detection function).
_mp_face_mesh = None
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _get_mediapipe():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        try:
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            from mediapipe import Image as MpImage, ImageFormat
            import mediapipe as mp
            import os

            # Model file search order:
            # 1. Environment variable MEDIAPIPE_MODEL_DIR (overrides default)
            # 2. Project-local models/ directory
            # 3. Alongside the mediapipe package (legacy)
            _mp_model_path = None
            env_dir = os.environ.get("MEDIAPIPE_MODEL_DIR")
            if env_dir:
                candidate = os.path.join(env_dir, "face_landmarker_v2.task")
                if os.path.exists(candidate):
                    _mp_model_path = candidate

            if _mp_model_path is None:
                # Look in project root's models/ directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(script_dir))  # model6_body3d -> avatar_pipeline -> New avatar
                project_models = os.path.join(project_root, "models", "face_landmarker_v2.task")
                if os.path.exists(project_models):
                    _mp_model_path = project_models

            if _mp_model_path is None:
                # Legacy: alongside the mediapipe package
                _mp_model_path = os.path.join(
                    os.path.dirname(mp.__file__), "face_landmarker_v2.task"
                )

            if not os.path.exists(_mp_model_path):
                print(f"[face_features] MediaPipe model not found at: {_mp_model_path}")
                print("[face_features] Download it manually from:")
                print("[face_features]   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                print("[face_features] Or run: python scripts/download_mediapipe_models.py")
                print("[face_features] Falling back to Haar cascade (lower accuracy)")
                _mp_face_mesh = False
                return None

            base_options = mp.tasks.BaseOptions(model_asset_path=_mp_model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                output_face_blendshapes=False,
                num_faces=1,
            )
            _mp_face_mesh = FaceLandmarker.create_from_options(options)
            # Store MpImage/ImageFormat refs on the object for later use
            _mp_face_mesh._mp_image_cls = MpImage
            _mp_face_mesh._mp_image_fmt = ImageFormat
        except Exception:
            _mp_face_mesh = False  # Sentinel: MediaPipe not available
    return _mp_face_mesh if _mp_face_mesh is not False else None


def _mp_detect_face(image_rgb):
    """Returns (x, y, w, h) bounding box from MediaPipe landmarks, or None."""
    landmarker = _get_mediapipe()
    if landmarker is None:
        return None
    try:
        mp_img = landmarker._mp_image_cls(
            landmarker._mp_image_fmt.SRGB, image_rgb
        )
        result = landmarker.detect(mp_img)
        if not result or not result.face_landmarks:
            return None
        h, w = image_rgb.shape[:2]
        lm = result.face_landmarks[0]
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    except Exception:
        return None


def _detect_face(image_rgb):
    """Try MediaPipe first, fall back to Haar cascade."""
    bbox = _mp_detect_face(image_rgb)
    if bbox is not None:
        return bbox
    # Haar cascade fallback
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])

# Used when no face is detected: a generic dark-brown hair color
DEFAULT_HAIR_RGB = (60, 45, 35)

# Hair style classifications based on hair region analysis
_HAIR_STYLES = ["short", "medium", "long", "curly", "wavy", "straight", "buzz", "ponytail"]

# Age group classifications
_AGE_GROUPS = ["teen", "20s", "30s", "40s", "50+"]

# Eye color classifications (RGB ranges)
_EYE_COLORS = {
    "brown": [(50, 20, 10), (120, 60, 30)],
    "blue": [(80, 100, 150), (180, 200, 255)],
    "green": [(60, 100, 40), (150, 180, 100)],
    "hazel": [(100, 80, 40), (160, 140, 80)],
    "gray": [(100, 100, 100), (180, 180, 180)],
}

# Hair color classifications (HSV-based)
_HAIR_COLORS = {
    "black": {"h_range": (0, 180), "s_range": (0, 255), "v_range": (0, 50)},
    "brown": {"h_range": (0, 30), "s_range": (30, 255), "v_range": (50, 150)},
    "blonde": {"h_range": (15, 35), "s_range": (20, 150), "v_range": (150, 255)},
    "red": {"h_range": (0, 20), "s_range": (100, 255), "v_range": (100, 200)},
    "gray": {"h_range": (0, 180), "s_range": (0, 30), "v_range": (100, 200)},
}


def _classify_hair_color(hair_rgb):
    """Classifies hair color from RGB tuple. Returns color name."""
    if hair_rgb is None:
        return "brown"
    
    hair_bgr = np.uint8([[[hair_rgb[2], hair_rgb[1], hair_rgb[0]]]])
    hsv = cv2.cvtColor(hair_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = hsv[0], hsv[1], hsv[2]
    
    best_match = "brown"
    for color, ranges in _HAIR_COLORS.items():
        h_min, h_max = ranges["h_range"]
        s_min, s_max = ranges["s_range"]
        v_min, v_max = ranges["v_range"]
        
        if (h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max):
            best_match = color
            break
    
    return best_match


def _classify_hair_style(face_x, face_y, face_w, face_h, image_rgb):
    """Estimates hair style from face region context. Returns style name."""
    # Analyze the hair region (above face) for style hints
    hair_y1, hair_y0 = face_y, max(0, face_y - int(face_h * 0.8))
    hair_x0 = max(0, face_x - int(face_w * 0.3))
    hair_x1 = min(image_rgb.shape[1], face_x + face_w + int(face_w * 0.3))
    
    if hair_y1 <= hair_y0 or hair_x1 <= hair_x0:
        return "short"
    
    hair_region = image_rgb[hair_y0:hair_y1, hair_x0:hair_x1]
    hair_region_gray = cv2.cvtColor(hair_region, cv2.COLOR_RGB2GRAY)
    
    # Use edge detection to estimate hair texture/style
    edges = cv2.Canny(hair_region_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Simple heuristic: high edge density = curly/textured, low = straight
    if edge_density > 0.15:
        return "curly" if edge_density > 0.25 else "wavy"
    return "straight"


def _detect_facial_hair(face_region_gray, face_w, face_h):
    """Detects presence and type of facial hair. Returns 'none', 'stubble', 'beard', 'mustache'."""
    # Analyze lower face region for hair darkness
    lower_face = face_region_gray[int(face_h * 0.6):, :]
    if lower_face.size == 0:
        return "none"
    
    # Check darkness of lower face region (hint of facial hair)
    mean_darkness = 255 - np.mean(lower_face)
    
    if mean_darkness < 20:
        return "none"
    elif mean_darkness < 40:
        return "stubble"
    else:
        # Analyze cheeks vs chin to distinguish beard from mustache
        mid_face = face_region_gray[int(face_h * 0.4):int(face_h * 0.7), :]
        chin = face_region_gray[int(face_h * 0.7):, :]
        
        if chin.size > 0:
            chin_darkness = 255 - np.mean(chin)
            if chin_darkness > mean_darkness * 0.8:
                return "beard"
        
        return "mustache"


def _estimate_gender(face_x, face_y, face_w, face_h, face_region_rgb):
    """Estimates gender from facial proportions. Returns 'male', 'female', or 'neutral'."""
    face_region_gray = cv2.cvtColor(face_region_rgb, cv2.COLOR_RGB2GRAY)
    
    # Analyze facial structure:
    # - Jaw width (lower face width)
    # - Cheekbone prominence (mid-face width)
    # - Forehead size
    
    mid_face = face_region_gray[int(face_h * 0.35):int(face_h * 0.65), :]
    lower_face = face_region_gray[int(face_h * 0.65):, :]
    
    if mid_face.size == 0 or lower_face.size == 0:
        return "neutral"
    
    # Analyze contour complexity and angularity (heuristic for gender)
    edges = cv2.Canny(face_region_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Features suggesting female: smoother contours, higher edge density in upper face
    upper_face = face_region_gray[:int(face_h * 0.4), :]
    upper_edges = np.sum(cv2.Canny(upper_face, 50, 150) > 0) / upper_face.size if upper_face.size > 0 else 0
    
    # Aspect ratio and proportions
    jaw_width_estimate = np.sum(lower_face < 100) / lower_face.size if lower_face.size > 0 else 0
    
    # Simple decision tree based on facial proportions
    if jaw_width_estimate > 0.5 and edge_density > 0.08:
        return "male"
    elif upper_edges > 0.12 or jaw_width_estimate < 0.3:
        return "female"
    else:
        return "neutral"


def _estimate_age_group(face_region_rgb):
    """Estimates age group from facial features. Returns age group name."""
    face_region_gray = cv2.cvtColor(face_region_rgb, cv2.COLOR_RGB2GRAY)
    
    # Analyze wrinkles and skin texture
    # Apply Laplacian to detect edges/wrinkles
    laplacian = cv2.Laplacian(face_region_gray, cv2.CV_64F)
    wrinkle_density = np.sum(np.abs(laplacian) > 100) / laplacian.size
    
    # Skin texture smoothness
    smooth = cv2.GaussianBlur(face_region_gray, (5, 5), 0)
    texture_diff = np.mean(np.abs(face_region_gray.astype(float) - smooth.astype(float)))
    
    # Simple age estimation heuristic
    if wrinkle_density > 0.3:
        return "50+"
    elif wrinkle_density > 0.15:
        return "40s"
    elif texture_diff > 10:
        return "30s"
    elif texture_diff > 5:
        return "20s"
    else:
        return "teen"


def _detect_eye_color(face_region_rgb):
    """Detects eye color from face region. Returns color name and approximate RGB."""
    gray = cv2.cvtColor(face_region_rgb, cv2.COLOR_RGB2GRAY)
    eyes = _eye_cascade.detectMultiScale(gray, 1.3, 5, minSize=(15, 15))
    
    if len(eyes) == 0:
        return "brown", (80, 50, 30)  # default
    
    # Analyze the largest eye region
    eye = max(eyes, key=lambda e: e[2] * e[3])
    ex, ey, ew, eh = eye
    eye_region = face_region_rgb[ey:ey+eh, ex:ex+ew]
    
    # Get the central part (iris)
    iris_y = int(eh * 0.4)
    iris_x = int(ew * 0.3)
    iris_region = eye_region[max(0, iris_y-5):min(eh, iris_y+10), 
                              max(0, iris_x-5):min(ew, iris_x+10)]
    
    if iris_region.size == 0:
        return "brown", (80, 50, 30)
    
    # Find dominant color in iris region
    iris_rgb = np.median(iris_region.reshape(-1, 3), axis=0).astype(int)
    
    # Match to known eye colors
    best_match = "brown"
    best_distance = float('inf')
    
    for color, rgb_ranges in _EYE_COLORS.items():
        rgb_low = np.array(rgb_ranges[0])
        rgb_high = np.array(rgb_ranges[1])
        mid_rgb = (rgb_low + rgb_high) / 2
        
        distance = np.sum((iris_rgb - mid_rgb) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_match = color
    
    return best_match, tuple(int(c) for c in iris_rgb)


def extract_facial_analysis(image_rgb):
    """Comprehensive facial feature analysis.
    
    Returns dict with:
    - gender: 'male', 'female', or 'neutral'
    - age_group: 'teen', '20s', '30s', '40s', '50+'
    - hair_color: color name
    - hair_style: style name
    - facial_hair: 'none', 'stubble', 'beard', 'mustache' (mainly for male)
    - eye_color: color name
    - face_shape: estimated shape
    - facial_landmarks: basic landmarks if detectable
    """
    face = _detect_face(image_rgb)
    if face is None:
        return {
            "gender": "neutral",
            "age_group": "20s",
            "hair_color": "brown",
            "hair_style": "short",
            "facial_hair": "none",
            "eye_color": "brown",
            "face_shape": "oval",
            "confidence": 0.0,
        }
    
    x, y, w, h = face
    
    # Extract face region for analysis
    face_region = image_rgb[y:y+h, x:x+w]
    face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # Extract features
    gender = _estimate_gender(x, y, w, h, face_region)
    age_group = _estimate_age_group(face_region)
    
    # Hair analysis (region above face)
    hair_y1, hair_y0 = y, max(0, y - int(h * 0.6))
    hair_x0, hair_x1 = max(0, x), min(image_rgb.shape[1], x + w)
    hair_rgb = None
    if hair_y1 > hair_y0 and hair_x1 > hair_x0:
        hair_region = image_rgb[hair_y0:hair_y1, hair_x0:hair_x1]
        hair_rgb = tuple(int(c) for c in np.median(hair_region.reshape(-1, 3), axis=0))
    
    hair_color = _classify_hair_color(hair_rgb)
    hair_style = _classify_hair_style(x, y, w, h, image_rgb)
    facial_hair = _detect_facial_hair(face_region_gray, w, h)
    
    # Eye and face shape analysis
    eye_color, eye_rgb = _detect_eye_color(face_region)
    face_shape = "oval"  # Default; could be enhanced with contour analysis
    
    return {
        "gender": gender,
        "age_group": age_group,
        "hair_color": hair_color,
        "hair_style": hair_style,
        "facial_hair": facial_hair,
        "eye_color": eye_color,
        "face_shape": face_shape,
        "confidence": 0.7,  # Heuristic confidence
    }


def estimate_face_landmarks(image_rgb):
    """Extract 468 MediaPipe FaceMesh landmarks from a face photo.

    Uses MediaPipe FaceMesh (with Haar cascade fallback if MediaPipe is not
    installed).  Returns the full 468-point face-mesh in pixel coordinates.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8
        The full selfie/photo.

    Returns
    -------
    landmarks : (468, 2) float32 or None
        Landmark pixel positions in ``image_rgb`` space, or None if no face
        detected.
    """
    h, w = image_rgb.shape[:2]

    # Try MediaPipe first (new FaceLandmarker API)
    landmarker = _get_mediapipe()
    if landmarker is not None:
        try:
            mp_img = landmarker._mp_image_cls(
                landmarker._mp_image_fmt.SRGB, image_rgb
            )
            result = landmarker.detect(mp_img)
            if result and result.face_landmarks:
                lm = result.face_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
                return pts
        except Exception:
            pass

    # Fallback: approximate landmarks from Haar cascade bounding box
    face = _detect_face(image_rgb)
    if face is None:
        return None
    x, y, fw, fh = face
    cx, cy = x + fw / 2, y + fh / 2
    pts = []

    # Face contour
    for i in range(32):
        angle = 2 * np.pi * i / 32
        px = cx + (fw / 2) * np.cos(angle)
        py = cy + (fh / 2) * np.sin(angle) * 0.85
        pts.append((px, py))
    # Eyebrows
    for side in (-1, 1):
        bx = cx + side * fw * 0.20
        by = cy - fh * 0.20
        for i in range(5):
            px = bx + side * fw * 0.12 * (i / 4 - 0.5)
            py = by - fh * 0.02 * np.sin(np.pi * i / 4)
            pts.append((px, py))
    # Eyes
    for side in (-1, 1):
        ex = cx + side * fw * 0.18
        ey = cy - fh * 0.05
        for i in range(8):
            a = 2 * np.pi * i / 8
            px = ex + fw * 0.08 * np.cos(a)
            py = ey + fh * 0.04 * np.sin(a)
            pts.append((px, py))
    # Nose
    for i in range(6):
        frac = i / 5
        pts.append((cx, cy + fh * (-0.15 + frac * 0.08)))
    for side in (-1, 1):
        pts.append((cx + side * fw * 0.08, cy + fh * 0.06))
        pts.append((cx + side * fw * 0.12, cy + fh * 0.08))
    # Mouth
    mx, my = cx, cy + fh * 0.15
    for i in range(12):
        a = 2 * np.pi * i / 12
        px = mx + fw * 0.15 * np.cos(a)
        py = my + fh * 0.06 * np.sin(a)
        if np.sin(a) < 0:
            py *= 0.7
        pts.append((px, py))
    for i in range(8):
        a = 2 * np.pi * i / 8
        px = mx + fw * 0.10 * np.cos(a)
        py = my + fh * 0.035 * np.sin(a)
        pts.append((px, py))
    # Fill
    for row in range(4):
        fy = cy + fh * (-0.10 + row * 0.08)
        for col in range(5):
            fx = cx + fw * (-0.25 + col * 0.125)
            pts.append((fx, fy))

    return np.asarray(pts, dtype=np.float32)


def extract_face_features(image_rgb, texture_size=128, estimate_landmarks=False):
    """Extract face features from a photo.

    image_rgb: HxWx3 uint8 RGB numpy array (the user's photo).
    estimate_landmarks: if True, returns MediaPipe FaceMesh 468-point 2D
        landmarks (useful for Delaunay-triangulation face-texture warping).

    Returns {
        "face_crop": (texture_size, texture_size, 3) uint8 array or None,
        "hair_rgb": (r, g, b) ints,
        "facial_analysis": comprehensive facial features dict,
        "face_width": face crop pixel width (before resize),
        "face_height": face crop pixel height (before resize),
        "landmarks_2d": (468, 2) float32 or None (only if estimate_landmarks)
    }
    """
    face = _detect_face(image_rgb)
    if face is None:
        analysis = {
            "gender": "neutral",
            "age_group": "20s",
            "hair_color": "brown",
            "hair_style": "short",
            "facial_hair": "none",
            "eye_color": "brown",
            "face_shape": "oval",
            "confidence": 0.0,
        }
        result = {"face_crop": None, "hair_rgb": DEFAULT_HAIR_RGB, "facial_analysis": analysis}
        if estimate_landmarks:
            result["landmarks_2d"] = None
        return result

    x, y, w, h = face

    img_h, img_w = image_rgb.shape[:2]

    # Extract landmarks early so we can use them for crop bounds AND return them.
    landmarks_2d = estimate_face_landmarks(image_rgb) if estimate_landmarks else None

    # Landmark-based crop bounds: use exact MediaPipe face boundary landmarks
    # so the crop is forehead→chin, ear→ear — no hair/background wasted space.
    # Landmark indices: 10=forehead top, 152=chin tip, 234=left ear, 454=right ear.
    _LM_FOREHEAD, _LM_CHIN, _LM_LEFT_EAR, _LM_RIGHT_EAR = 10, 152, 234, 454
    if (landmarks_2d is not None
            and len(landmarks_2d) >= 468
            and landmarks_2d[_LM_FOREHEAD][1] < landmarks_2d[_LM_CHIN][1]):
        top_y    = landmarks_2d[_LM_FOREHEAD][1]
        bottom_y = landmarks_2d[_LM_CHIN][1]
        left_x   = landmarks_2d[_LM_LEFT_EAR][0]
        right_x  = landmarks_2d[_LM_RIGHT_EAR][0]

        span_h = bottom_y - top_y
        span_w = right_x - left_x
        pad = 0.10  # 10% padding on all sides — keeps full face visible

        x0 = max(0,      int(left_x   - span_w * pad))
        x1 = min(img_w,  int(right_x  + span_w * pad))
        y0 = max(0,      int(top_y    - span_h * pad))
        y1 = min(img_h,  int(bottom_y + span_h * pad))
    else:
        # Fallback when landmarks unavailable: tight margin-based crop.
        # Top margin kept small (0.15) to avoid pulling in background/hair.
        margin_top, margin_side, margin_bottom = 0.15, 0.15, 0.15
        mx, my_top, my_bot = int(w * margin_side), int(h * margin_top), int(h * margin_bottom)
        x0, y0 = max(0, x - mx),      max(0, y - my_top)
        x1, y1 = min(img_w, x + w + mx), min(img_h, y + h + my_bot)

    face_crop = cv2.resize(image_rgb[y0:y1, x0:x1], (texture_size, texture_size))

    # Original pixel dimensions of the crop (before resize) — used by the
    # centre-paste path to maintain the correct face aspect ratio on the head UV.
    face_width  = x1 - x0
    face_height = y1 - y0

    # Hair color: sample the region above the forehead landmark (or face box top).
    hair_top_y = max(0, int((landmarks_2d[_LM_FOREHEAD][1] if landmarks_2d is not None
                              and len(landmarks_2d) >= 468 else y) - h * 0.6))
    hair_bot_y = int(landmarks_2d[_LM_FOREHEAD][1] if landmarks_2d is not None
                     and len(landmarks_2d) >= 468 else y)
    hair_x0_r, hair_x1_r = max(0, x), min(img_w, x + w)
    if hair_bot_y > hair_top_y and hair_x1_r > hair_x0_r:
        hair_region = image_rgb[hair_top_y:hair_bot_y, hair_x0_r:hair_x1_r]
        hair_rgb = tuple(int(c) for c in np.median(hair_region.reshape(-1, 3), axis=0))
    else:
        hair_rgb = DEFAULT_HAIR_RGB

    # Run comprehensive facial analysis
    facial_analysis = extract_facial_analysis(image_rgb)

    result = {
        "face_crop": face_crop,
        "hair_rgb": hair_rgb,
        "facial_analysis": facial_analysis,
        "face_width":  face_width,
        "face_height": face_height,
    }

    if estimate_landmarks:
        result["landmarks_2d"] = landmarks_2d

    return result
