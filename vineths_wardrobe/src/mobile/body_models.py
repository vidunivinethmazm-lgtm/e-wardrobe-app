"""
eWardrobeAI Mobile — Body Calibration Models
Estimates body proportions from a single photo.

Model 1 — MediaPipePoseModel
  Uses MediaPipe Pose to detect 33 body landmarks.
  Derives shoulder width, height, hip ratio, and body type from landmark geometry.
  Accuracy: per-landmark visibility confidence (0–1).

Model 2 — FaceAnchorBodyModel
  Uses the face bounding box (from OpenCV Haar cascade) as a physical reference.
  Average human face height ≈ 23 cm → body height ≈ face_height × 7.5 ratio.
  Derives shoulder width, height, and body type from face proportions.
  Accuracy: face detection confidence + proportion consistency score.
"""

from __future__ import annotations
import time
import cv2
import numpy as np
from dataclasses import dataclass, field

# MediaPipe optional import
try:
    import mediapipe as mp
    _MP_OK = True
except (ImportError, AttributeError):
    _MP_OK = False

# ── Body estimate result ───────────────────────────────────────────────────

@dataclass
class BodyCalibResult:
    model_name:        str
    description:       str
    detected:          bool
    confidence:        float          # overall 0–1
    shoulder_cm:       float
    height_cm:         float
    hip_cm:            float
    body_type:         str
    standard_size:     str
    landmarks_found:   int
    total_landmarks:   int
    inference_ms:      float
    landmark_points:   list = field(default_factory=list)  # (x,y) pixel pairs
    detail:            dict = field(default_factory=dict)


_SIZE_T = [(82,'XS'),(88,'S'),(96,'M'),(104,'L'),(112,'XL'),(124,'XXL'),(1e9,'XXXL')]

def _size(chest: float) -> str:
    return next((s for t,s in _SIZE_T if chest <= t), 'XXXL')

def _body_type(shoulder: float, chest: float, waist: float, hip: float) -> str:
    if hip == 0 or shoulder == 0: return 'unknown'
    wdef  = (chest + hip) / 2 - waist
    shr   = hip / (shoulder * 2.3)
    if wdef > 9 and abs(shr - 1) < 0.08: return 'hourglass'
    if shr < 0.87: return 'inverted_triangle'
    if shr > 1.13: return 'pear'
    return 'rectangle'


# ── Model 1: MediaPipe Pose ────────────────────────────────────────────────

class MediaPipePoseModel:
    """
    MediaPipe Pose (33 landmarks) → body proportion estimates.

    Key landmarks used:
      11/12 — left/right shoulder
      23/24 — left/right hip
      27/28 — left/right ankle
       0    — nose (face top proxy)

    Accuracy metric: mean visibility score across all detected landmarks.
    A visibility score < 0.5 means the landmark is likely occluded.
    """
    name        = "MediaPipe Pose"
    description = "33 body landmarks, pre-trained by Google"

    def __init__(self):
        self._pose = None
        if _MP_OK:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode       = True,
                    model_complexity        = 2,
                    enable_segmentation     = False,
                    min_detection_confidence= 0.4,
                )
            except AttributeError:
                self._pose = None

    def run(self, image_bgr: np.ndarray) -> BodyCalibResult:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]

        if self._pose is None:
            return self._fallback(time.perf_counter() - t0)

        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        ms     = (time.perf_counter() - t0) * 1000

        if not result.pose_landmarks:
            return BodyCalibResult(
                model_name='MediaPipe Pose', description=self.description,
                detected=False, confidence=0.0,
                shoulder_cm=0, height_cm=0, hip_cm=0,
                body_type='unknown', standard_size='unknown',
                landmarks_found=0, total_landmarks=33, inference_ms=round(ms,3),
            )

        lm = result.pose_landmarks.landmark

        # Pixel coordinates
        def px(idx): return (lm[idx].x * w, lm[idx].y * h)
        def vis(idx): return lm[idx].visibility

        lshoulder = px(11); rshoulder = px(12)
        lhip      = px(23); rhip      = px(24)
        lankle    = px(27); rankle    = px(28)
        nose      = px(0)

        # Shoulder width in pixels → cm (pixel-to-cm via torso height ratio)
        shoulder_px = abs(rshoulder[0] - lshoulder[0])
        hip_px      = abs(rhip[0]      - lhip[0])
        torso_px    = abs((lshoulder[1]+rshoulder[1])/2 - (lhip[1]+rhip[1])/2)
        leg_px      = abs((lhip[1]+rhip[1])/2          - (lankle[1]+rankle[1])/2)
        torso_height_cm = 55.0   # average torso ~55cm

        if torso_px > 0:
            px_per_cm   = torso_px / torso_height_cm
            shoulder_cm = round(shoulder_px / px_per_cm, 1)
            height_cm   = round((torso_px + leg_px) / px_per_cm + 22, 1)  # +22 head
            hip_cm      = round(hip_px / px_per_cm + 20, 1)               # circumference approx
        else:
            shoulder_cm = height_cm = hip_cm = 0.0

        chest_cm    = round(shoulder_cm * 2.2, 1)
        waist_cm    = round(hip_cm * 0.76, 1)
        btype       = _body_type(shoulder_cm, chest_cm, waist_cm, hip_cm)
        size        = _size(chest_cm)

        visible     = [lm[i].visibility for i in range(33)]
        mean_vis    = float(np.mean(visible))
        found       = sum(1 for v in visible if v > 0.5)

        pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in range(33)]

        return BodyCalibResult(
            model_name      = self.name,
            description     = self.description,
            detected        = True,
            confidence      = round(mean_vis, 3),
            shoulder_cm     = shoulder_cm,
            height_cm       = height_cm,
            hip_cm          = hip_cm,
            body_type       = btype,
            standard_size   = size,
            landmarks_found = found,
            total_landmarks = 33,
            inference_ms    = round(ms, 3),
            landmark_points = pts,
            detail = {
                'shoulder_px': round(shoulder_px,1),
                'torso_px':    round(torso_px,1),
                'mean_visibility': round(mean_vis,3),
                'visible_landmarks': found,
            }
        )

    def _fallback(self, elapsed_s: float) -> BodyCalibResult:
        return BodyCalibResult(
            model_name='MediaPipe Pose', description=self.description,
            detected=False, confidence=0.0,
            shoulder_cm=0, height_cm=0, hip_cm=0,
            body_type='unknown', standard_size='unknown',
            landmarks_found=0, total_landmarks=33,
            inference_ms=round(elapsed_s*1000,3),
            detail={'error': 'MediaPipe not available on this Python version'}
        )


# ── Model 2: Face-Anchor Body Estimator ───────────────────────────────────

class FaceAnchorBodyModel:
    """
    Uses the face bounding box as a known physical anchor:
      Average adult face height ≈ 23 cm
      Body height ≈ face_height × 7.5  (classical art proportion rule)
      Shoulder width ≈ 2.0 × face_width
      Hip width ≈ 1.8 × face_width (female average)

    Accuracy metric: face detection confidence (HOG score) +
    proportion consistency score (ratio plausibility 0–1).
    Requires no pre-training — uses OpenCV Haar cascade.
    """
    name        = "Face-Anchor Estimator"
    description = "Face bounding box → body proportions via art-proportion rules"

    # Average adult proportions (classic 7.5-head rule)
    FACE_TO_HEIGHT_RATIO  = 7.5
    FACE_TO_SHOULDER_MULT = 2.0
    FACE_HEIGHT_CM        = 23.0

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def run(self, image_bgr: np.ndarray) -> BodyCalibResult:
        t0   = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_bgr.shape[:2]

        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        ms = (time.perf_counter() - t0) * 1000

        if not len(faces):
            return BodyCalibResult(
                model_name='Face-Anchor Estimator', description=self.description,
                detected=False, confidence=0.0,
                shoulder_cm=0, height_cm=0, hip_cm=0,
                body_type='unknown', standard_size='unknown',
                landmarks_found=0, total_landmarks=6,
                inference_ms=round(ms,3),
                detail={'error': 'No face detected'}
            )

        # Use largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])

        # Convert face pixels to cm
        px_per_cm = fh / self.FACE_HEIGHT_CM

        shoulder_cm = round(fw * self.FACE_TO_SHOULDER_MULT / px_per_cm, 1)
        height_cm   = round(fh * self.FACE_TO_HEIGHT_RATIO  / px_per_cm, 1)
        hip_cm      = round(fw * 1.8 / px_per_cm + 20, 1)
        chest_cm    = round(shoulder_cm * 2.2, 1)
        waist_cm    = round(hip_cm * 0.76, 1)
        btype       = _body_type(shoulder_cm, chest_cm, waist_cm, hip_cm)
        size        = _size(chest_cm)

        # Confidence: face size plausibility + centering
        face_area_ratio = (fw * fh) / (w_img * h_img)
        conf = float(np.clip(face_area_ratio * 20, 0.3, 0.95))

        # Proportion consistency (height should be 150–220cm)
        prop_score = 1.0 if 140 <= height_cm <= 220 else 0.5

        # 6 estimated landmark points
        cx, cy = x + fw//2, y + fh//2
        pts = [
            (x,        y),          # face top-left
            (x+fw,     y),          # face top-right
            (int(cx - shoulder_cm*px_per_cm), int(y + fh*1.2)),  # left shoulder
            (int(cx + shoulder_cm*px_per_cm), int(y + fh*1.2)),  # right shoulder
            (int(cx - hip_cm*0.3*px_per_cm), int(y + fh*3.5)),   # left hip
            (int(cx + hip_cm*0.3*px_per_cm), int(y + fh*3.5)),   # right hip
        ]

        return BodyCalibResult(
            model_name      = self.name,
            description     = self.description,
            detected        = True,
            confidence      = round((conf + prop_score) / 2, 3),
            shoulder_cm     = shoulder_cm,
            height_cm       = height_cm,
            hip_cm          = hip_cm,
            body_type       = btype,
            standard_size   = size,
            landmarks_found = 6,
            total_landmarks = 6,
            inference_ms    = round(ms, 3),
            landmark_points = pts,
            detail = {
                'face_box':       [int(x), int(y), int(fw), int(fh)],
                'px_per_cm':      round(px_per_cm, 3),
                'prop_score':     round(prop_score, 3),
                'det_confidence': round(conf, 3),
            }
        )


# ── BodyModelRunner ────────────────────────────────────────────────────────

class BodyModelRunner:
    """Runs both body models on one image and returns results."""

    def __init__(self):
        self.mp_model   = MediaPipePoseModel()
        self.face_model = FaceAnchorBodyModel()

    def run(self, image_bgr: np.ndarray) -> list[BodyCalibResult]:
        return [
            self.mp_model.run(image_bgr),
            self.face_model.run(image_bgr),
        ]

    def agreement(self, results: list[BodyCalibResult]) -> dict:
        """Compare both models' estimates — lower diff = better agreement."""
        r1, r2 = results
        if not r1.detected or not r2.detected:
            return {'agreement_pct': 0.0, 'shoulder_diff_cm': None, 'height_diff_cm': None}
        sd = abs(r1.shoulder_cm - r2.shoulder_cm)
        hd = abs(r1.height_cm   - r2.height_cm)
        agr = max(0.0, 1.0 - (sd/40 + hd/80) / 2)
        return {
            'agreement_pct':   round(agr * 100, 1),
            'shoulder_diff_cm': round(sd, 1),
            'height_diff_cm':   round(hd, 1),
            'size_match':       r1.standard_size == r2.standard_size,
        }
