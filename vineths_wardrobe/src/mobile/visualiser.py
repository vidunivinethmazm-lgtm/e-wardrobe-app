"""
eWardrobeAI Mobile — Image Visualiser
Draws face keypoints and body landmarks on the uploaded photo.
Returns annotated images as numpy RGB arrays for display in Streamlit.
"""

from __future__ import annotations
import cv2
import numpy as np
from src.mobile.face_models  import FaceAccuracyResult, KP_NAMES, IMG_SIZE
from src.mobile.body_models  import BodyCalibResult

# Colour palette (BGR for cv2)
DEEP_CLR  = (124, 111, 255)   # purple  — DeepFaceCNN
LIGHT_CLR = (255, 107, 157)   # pink    — LightFaceCNN
BODY_CLR1 = (78,  205, 196)   # teal    — MediaPipe Pose
BODY_CLR2 = (255, 180,  50)   # orange  — Face-Anchor
LABEL_CLR = (255, 255, 255)   # white


# ── Face Keypoints ─────────────────────────────────────────────────────────

def draw_face_keypoints(
    image_bgr: np.ndarray,
    face_results: list[FaceAccuracyResult],
    original_w: int,
    original_h: int,
) -> np.ndarray:
    """
    Draw predicted keypoints from both face models on the image.
    Keypoints are stored in normalised [0, 96] space → rescale to original image.
    """
    canvas = image_bgr.copy()
    colours = [DEEP_CLR, LIGHT_CLR]
    labels  = [r.model_name for r in face_results]

    for res, colour in zip(face_results, colours):
        if not res.keypoints:
            continue
        for i, (name, (x96, y96)) in enumerate(res.keypoints.items()):
            # Scale from 96×96 CNN space → original image size
            px = int(x96 / IMG_SIZE * original_w)
            py = int(y96 / IMG_SIZE * original_h)
            px = max(0, min(original_w - 1, px))
            py = max(0, min(original_h - 1, py))

            cv2.circle(canvas, (px, py), 4, colour, -1)
            cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1)   # black outline

    # Legend
    for i, (label, colour) in enumerate(zip(labels, colours)):
        y = 24 + i * 22
        cv2.rectangle(canvas, (8, y-14), (22, y+2), colour, -1)
        cv2.putText(canvas, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, LABEL_CLR, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_face_comparison(
    image_bgr: np.ndarray,
    face_results: list[FaceAccuracyResult],
    original_w: int,
    original_h: int,
) -> np.ndarray:
    """
    Side-by-side: left half shows DeepFaceCNN, right half shows LightFaceCNN.
    """
    h, w = image_bgr.shape[:2]
    left  = image_bgr.copy()
    right = image_bgr.copy()
    colours = [DEEP_CLR, LIGHT_CLR]

    panels = [(left, face_results[0], colours[0]) if len(face_results) > 0 else None,
              (right, face_results[1], colours[1]) if len(face_results) > 1 else None]

    for panel, res, colour in [p for p in panels if p]:
        if not res.keypoints: continue
        for name, (x96, y96) in res.keypoints.items():
            px = int(x96 / IMG_SIZE * original_w)
            py = int(y96 / IMG_SIZE * original_h)
            px = max(0, min(w-1, px)); py = max(0, min(h-1, py))
            cv2.circle(panel, (px, py), 4, colour, -1)
            cv2.circle(panel, (px, py), 5, (0,0,0), 1)
        # Label top-left
        cv2.putText(panel, res.model_name, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if res.mae_px:
            cv2.putText(panel, f"MAE: {res.mae_px:.2f}px", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    combined = np.concatenate([left, right], axis=1)
    # Divider line
    cv2.line(combined, (w, 0), (w, h), (80, 80, 80), 2)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


# ── Body Landmarks ─────────────────────────────────────────────────────────

def draw_body_landmarks(
    image_bgr: np.ndarray,
    body_results: list[BodyCalibResult],
) -> np.ndarray:
    """Draw body landmark points from both body models."""
    canvas  = image_bgr.copy()
    colours = [BODY_CLR1, BODY_CLR2]

    # MediaPipe Pose — connect skeleton lines
    POSE_CONNECTIONS = [
        (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
        (11,23),(12,24),(23,24),                   # torso
        (23,25),(25,27),(24,26),(26,28),            # legs
    ]

    for res, colour in zip(body_results, colours):
        if not res.detected or not res.landmark_points:
            continue
        pts = res.landmark_points

        # Draw skeleton connections (MediaPipe only)
        if res.model_name == 'MediaPipe Pose' and len(pts) == 33:
            for i, j in POSE_CONNECTIONS:
                if i < len(pts) and j < len(pts):
                    cv2.line(canvas, pts[i], pts[j], colour, 2, cv2.LINE_AA)

        # Draw points
        for pt in pts:
            cv2.circle(canvas, pt, 4, colour, -1)
            cv2.circle(canvas, pt, 5, (0,0,0), 1)

    # Legend + stats
    for i, (res, colour) in enumerate(zip(body_results, colours)):
        y = 24 + i * 22
        cv2.rectangle(canvas, (8, y-14), (22, y+2), colour, -1)
        label = f"{res.model_name}  conf={res.confidence:.0%}"
        cv2.putText(canvas, label, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_CLR, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ── Combined Output ────────────────────────────────────────────────────────

def draw_all(
    image_bgr: np.ndarray,
    face_results: list[FaceAccuracyResult],
    body_results: list[BodyCalibResult],
) -> np.ndarray:
    """Single image with both face keypoints and body landmarks overlaid."""
    h, w = image_bgr.shape[:2]
    canvas = image_bgr.copy()
    colours_face = [DEEP_CLR, LIGHT_CLR]
    colours_body = [BODY_CLR1, BODY_CLR2]

    # Face keypoints (small circles)
    for res, colour in zip(face_results, colours_face):
        for name, (x96, y96) in res.keypoints.items():
            px = int(x96 / IMG_SIZE * w)
            py = int(y96 / IMG_SIZE * h)
            px = max(0, min(w-1, px)); py = max(0, min(h-1, py))
            cv2.circle(canvas, (px, py), 3, colour, -1)

    # Body landmarks (larger circles)
    for res, colour in zip(body_results, colours_body):
        if res.detected:
            for pt in res.landmark_points:
                cv2.circle(canvas, pt, 5, colour, -1)
                cv2.circle(canvas, pt, 6, (0,0,0), 1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
