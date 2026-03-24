"""
eWardrobeAI — Stage 2: AI-Driven Face Processing Module
Research Component

Two-layer face processing pipeline:
  Layer 1 — MediaPipe FaceMesh (468 landmark points, real-time, device-side)
  Layer 2 — eWardrobeAI CNN (15 keypoints, trained on facial keypoints dataset)

The two layers are complementary:
  - MediaPipe provides dense, real-time landmark detection for live camera input
  - The CNN provides high-accuracy keypoint regression for uploaded still images
    and generates facial geometry data for texture mapping onto the 3D avatar head

Output: FaceProfile dataclass consumed by the virtual try-on pipeline
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# MediaPipe (optional soft import — gracefully degrades if not installed)
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("MediaPipe not installed. Dense landmark layer disabled.")

# Internal CNN module
from src.face_keypoint_model import (
    load_trained_model,
    predict_single_image,
    IMG_SIZE,
)

# ── Constants ────────────────────────────────────────────────────────────────
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

_LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
_NOSE_TIP_INDEX    = 4
_MOUTH_INDICES     = [61, 291, 13, 14]

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'face_keypoint_cnn.keras'
)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class FaceLandmark:
    """Single 2D landmark with normalised and pixel coordinates."""
    name: str
    x_norm: float      # [0, 1] relative to image width
    y_norm: float      # [0, 1] relative to image height
    x_px: float = 0.0  # absolute pixel coordinate
    y_px: float = 0.0  # absolute pixel coordinate
    z: float = 0.0     # depth estimate (MediaPipe only)


@dataclass
class FaceProfile:
    """
    Complete facial analysis output consumed downstream by:
      - AvatarManager  (texture mapping, head scaling)
      - BodyCalibrator (face-to-body proportion checks)

    Attributes
    ----------
    landmarks_468 : MediaPipe dense landmarks (available when live camera used)
    landmarks_15  : CNN sparse landmarks (always populated for uploaded images)
    face_texture  : Cropped + normalised face patch for UV texture mapping
    inter_eye_dist: Inter-ocular distance in pixels (avatar head scaling proxy)
    face_width_px : Bounding-box width of detected face in pixels
    face_height_px: Bounding-box height of detected face in pixels
    yaw_deg       : Estimated head yaw angle (left–right rotation)
    pitch_deg     : Estimated head pitch angle (up–down rotation)
    source_image  : Original BGR image for downstream processing
    """
    landmarks_468: list[FaceLandmark] = field(default_factory=list)
    landmarks_15:  dict[str, tuple[float, float]] = field(default_factory=dict)
    face_texture:  Optional[np.ndarray] = None
    inter_eye_dist: float = 0.0
    face_width_px:  float = 0.0
    face_height_px: float = 0.0
    yaw_deg:        float = 0.0
    pitch_deg:      float = 0.0
    source_image:   Optional[np.ndarray] = None


# ── FaceProcessor ─────────────────────────────────────────────────────────────

class FaceProcessor:
    """
    Orchestrates both landmark layers and produces a FaceProfile.

    Usage
    -----
    >>> processor = FaceProcessor()
    >>> profile = processor.process(image_bgr)
    >>> print(profile.inter_eye_dist)
    """

    def __init__(self,
                 use_mediapipe: bool = True,
                 use_cnn: bool = True,
                 cnn_model_path: str = MODEL_PATH):

        self._use_mp  = use_mediapipe and _MP_AVAILABLE
        self._use_cnn = use_cnn

        self._mp_face_mesh = None
        self._cnn_model    = None

        if self._use_mp:
            self._init_mediapipe()

        if self._use_cnn:
            self._init_cnn(cnn_model_path)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        """Initialise MediaPipe FaceMesh with 468-landmark model."""
        try:
            mp_face = mp.solutions.face_mesh
            self._mp_face_mesh = mp_face.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("[FaceProcessor] MediaPipe FaceMesh initialised (468 pts).")
        except AttributeError:
            logger.warning(
                "[FaceProcessor] MediaPipe solutions API unavailable on this "
                "Python version. Falling back to OpenCV Haar cascade only."
            )
            self._use_mp = False
            self._mp_face_mesh = None

    def _init_cnn(self, model_path: str):
        """Load the trained facial keypoint CNN."""
        try:
            self._cnn_model = load_trained_model(model_path)
            logger.info("[FaceProcessor] CNN keypoint model loaded.")
        except FileNotFoundError:
            logger.warning(
                "[FaceProcessor] CNN model not found. "
                "Run `python -m src.face_keypoint_model` to train first."
            )
            self._use_cnn = False

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, image_bgr: np.ndarray) -> FaceProfile:
        """
        Run full face processing pipeline on a BGR image.

        Steps
        -----
        1. Detect face bounding box via MediaPipe or OpenCV Haar cascade
        2. Crop + pre-process face region
        3. Run MediaPipe FaceMesh → 468 dense landmarks
        4. Run CNN → 15 sparse keypoints (research-grade regression)
        5. Extract face texture patch for avatar UV mapping
        6. Compute geometric metrics (inter-eye distance, head pose)

        Returns
        -------
        FaceProfile
        """
        profile = FaceProfile(source_image=image_bgr.copy())

        face_roi, offset = self._detect_and_crop_face(image_bgr)
        if face_roi is None:
            logger.error("[FaceProcessor] No face detected in image.")
            return profile

        h, w = face_roi.shape[:2]
        profile.face_width_px  = float(w)
        profile.face_height_px = float(h)

        # Layer 1 — MediaPipe dense landmarks
        if self._use_mp:
            profile.landmarks_468 = self._run_mediapipe(face_roi, w, h, offset)

        # Layer 2 — CNN sparse keypoints
        if self._use_cnn and self._cnn_model is not None:
            profile.landmarks_15 = self._run_cnn(face_roi, w, h, offset)

        # Derived geometry
        profile.inter_eye_dist = self._inter_eye_distance(profile)
        profile.yaw_deg, profile.pitch_deg = self._estimate_head_pose(profile)

        # Face texture patch for avatar UV mapping
        profile.face_texture = self._extract_face_texture(face_roi)

        return profile

    # ── Face Detection ────────────────────────────────────────────────────────

    def _detect_and_crop_face(
        self, image_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], tuple[int, int]]:
        """
        Attempt MediaPipe-based face crop first; fall back to OpenCV
        Haar cascade if MediaPipe is unavailable.

        Returns (cropped_bgr, (x_offset, y_offset)) or (None, (0, 0)).
        """
        if self._use_mp and self._mp_face_mesh:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                h_img, w_img = image_bgr.shape[:2]
                xs = [int(p.x * w_img) for p in lm]
                ys = [int(p.y * h_img) for p in lm]
                x1 = max(0, min(xs) - 20)
                y1 = max(0, min(ys) - 20)
                x2 = min(w_img, max(xs) + 20)
                y2 = min(h_img, max(ys) + 20)
                return image_bgr[y1:y2, x1:x2].copy(), (x1, y1)

        # Haar cascade fallback
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = faces[0]
            return image_bgr[y:y+h, x:x+w].copy(), (x, y)

        return None, (0, 0)

    # ── MediaPipe Layer ───────────────────────────────────────────────────────

    def _run_mediapipe(
        self,
        face_roi: np.ndarray,
        w: int,
        h: int,
        offset: tuple[int, int],
    ) -> list[FaceLandmark]:
        """
        Run MediaPipe FaceMesh on the cropped face ROI.
        Returns a list of 468 FaceLandmark objects with global pixel coords.
        """
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        result = self._mp_face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return []

        lm_list: list[FaceLandmark] = []
        ox, oy = offset
        for idx, lm in enumerate(result.multi_face_landmarks[0].landmark):
            lm_list.append(FaceLandmark(
                name=f'mp_{idx}',
                x_norm=lm.x,
                y_norm=lm.y,
                x_px=lm.x * w + ox,
                y_px=lm.y * h + oy,
                z=lm.z,
            ))

        logger.debug(f"[FaceProcessor] MediaPipe detected {len(lm_list)} landmarks.")
        return lm_list

    # ── CNN Layer ─────────────────────────────────────────────────────────────

    def _run_cnn(
        self,
        face_roi: np.ndarray,
        w: int,
        h: int,
        offset: tuple[int, int],
    ) -> dict[str, tuple[float, float]]:
        """
        Pre-process the face ROI to 96×96 grayscale and run the CNN.
        Returns {keypoint_name: (x_global_px, y_global_px)}.
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        raw_preds = predict_single_image(self._cnn_model, resized)

        # Scale CNN predictions from [0,96] back to the original face ROI size
        ox, oy = offset
        scaled: dict[str, tuple[float, float]] = {}
        for name, (cx, cy) in raw_preds.items():
            global_x = (cx / IMG_SIZE) * w + ox
            global_y = (cy / IMG_SIZE) * h + oy
            scaled[name] = (global_x, global_y)

        logger.debug(f"[FaceProcessor] CNN detected {len(scaled)} keypoints.")
        return scaled

    # ── Geometry Helpers ──────────────────────────────────────────────────────

    def _inter_eye_distance(self, profile: FaceProfile) -> float:
        """
        Inter-ocular distance from CNN keypoints (preferred) or MediaPipe.
        Used to estimate head scale for 3D avatar sizing.
        """
        kp = profile.landmarks_15
        if 'left_eye_center' in kp and 'right_eye_center' in kp:
            lx, ly = kp['left_eye_center']
            rx, ry = kp['right_eye_center']
            return float(np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))

        # Fallback: MediaPipe eye landmarks
        if profile.landmarks_468:
            lm = profile.landmarks_468
            if len(lm) > max(_LEFT_EYE_INDICES + _RIGHT_EYE_INDICES):
                lx = np.mean([lm[i].x_px for i in _LEFT_EYE_INDICES])
                ly = np.mean([lm[i].y_px for i in _LEFT_EYE_INDICES])
                rx = np.mean([lm[i].x_px for i in _RIGHT_EYE_INDICES])
                ry = np.mean([lm[i].y_px for i in _RIGHT_EYE_INDICES])
                return float(np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))

        return 0.0

    def _estimate_head_pose(
        self, profile: FaceProfile
    ) -> tuple[float, float]:
        """
        Estimate yaw and pitch from the symmetry of eye positions.
        A full 3D pose-estimation (solvePnP) would use the 468-landmark set;
        this lightweight version gives a reliable approximation from 4 keypoints.

        Returns (yaw_deg, pitch_deg).
        """
        kp = profile.landmarks_15
        if 'left_eye_center' not in kp or 'right_eye_center' not in kp:
            return 0.0, 0.0

        lx, ly = kp['left_eye_center']
        rx, ry = kp['right_eye_center']
        dx = rx - lx
        dy = ry - ly

        # Yaw: asymmetry in horizontal eye separation relative to face width
        yaw = 0.0
        if profile.face_width_px > 0:
            expected = profile.face_width_px * 0.32   # typical ratio
            actual   = abs(dx)
            yaw = float(np.degrees(np.arcsin(
                np.clip((expected - actual) / expected, -1, 1)
            )))

        # Pitch: vertical tilt of eye-to-eye line
        pitch = float(np.degrees(np.arctan2(dy, max(abs(dx), 1e-6))))

        return round(yaw, 2), round(pitch, 2)

    # ── Texture Extraction ────────────────────────────────────────────────────

    def _extract_face_texture(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Produce a UV-ready face texture patch:
          1. Resize to 512×512 (standard UV map resolution)
          2. Apply CLAHE for perceptual normalisation
          3. Convert to RGB for Three.js texture loading

        The resulting patch is applied to the avatar head mesh as a
        THREE.Texture in the frontend renderer.
        """
        resized = cv2.resize(face_roi, (512, 512))
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # ── Utility ───────────────────────────────────────────────────────────────

    def draw_landmarks(
        self, image_bgr: np.ndarray, profile: FaceProfile
    ) -> np.ndarray:
        """Render both landmark layers onto the source image for debugging."""
        canvas = image_bgr.copy()

        # MediaPipe: small cyan dots
        for lm in profile.landmarks_468:
            cv2.circle(canvas, (int(lm.x_px), int(lm.y_px)),
                       1, (255, 255, 0), -1)

        # CNN: larger coloured circles with labels
        colours = {
            'left_eye_center':         (0, 255, 0),
            'right_eye_center':        (0, 255, 0),
            'nose_tip':                (0, 128, 255),
            'mouth_left_corner':       (255, 0, 255),
            'mouth_right_corner':      (255, 0, 255),
            'mouth_center_top_lip':    (200, 0, 200),
            'mouth_center_bottom_lip': (200, 0, 200),
        }
        for name, (x, y) in profile.landmarks_15.items():
            colour = colours.get(name, (0, 0, 255))
            cv2.circle(canvas, (int(x), int(y)), 4, colour, -1)
            cv2.putText(canvas, name.split('_')[0],
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)

        return canvas

    def release(self):
        """Free MediaPipe resources."""
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
