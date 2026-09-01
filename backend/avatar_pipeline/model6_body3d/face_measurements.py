"""
Model 6 — 3D Body Reconstruction: face measurement extraction.

Mirrors what body measurements (bust/waist/hips/height) do for the body —
here we extract normalised face-proportion measurements from the 468
MediaPipe FaceMesh landmarks and convert them to head-geometry parameters
that drive the 3D head ellipsoid and facial-feature positions in
mesh_builder.py.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Key MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------------------------
_LM = {
    "forehead":          10,   # top centre of forehead
    "chin":             152,   # bottom of chin
    "left_cheek":       234,   # widest point left cheek
    "right_cheek":      454,   # widest point right cheek
    "left_eye_outer":    33,   # left eye outer corner
    "left_eye_inner":   133,   # left eye inner corner
    "right_eye_inner":  362,   # right eye inner corner
    "right_eye_outer":  263,   # right eye outer corner
    "left_eye_top":     159,   # top of left upper eyelid
    "left_eye_bottom":  145,   # bottom of left lower eyelid
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    "nose_tip":           1,   # nose tip
    "nose_left":        129,   # left nostril ala
    "nose_right":       358,   # right nostril ala
    "mouth_left":        61,   # left mouth corner
    "mouth_right":      291,   # right mouth corner
    "left_jaw":         127,   # left jaw angle
    "right_jaw":        356,   # right jaw angle
    "brow_center":        9,   # glabella (between brows)
}

# Neutral / average human face proportions — used as fallback when no
# landmarks are available.
NEUTRAL_MEASUREMENTS: dict[str, float] = {
    "width_height_ratio": 0.82,   # face width / face height
    "eye_spacing_ratio":  0.34,   # inner-corner gap / face width
    "eye_openness":       0.28,   # eye height / eye width
    "nose_width_ratio":   0.31,   # nose ala width / face width
    "nose_height_ratio":  0.28,   # brow-to-tip / face height
    "mouth_width_ratio":  0.47,   # mouth width / face width
    "jaw_taper":          0.82,   # jaw width / face width
    "forehead_ratio":     0.34,   # forehead height / face height
}


def _d(lm: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(lm[a] - lm[b]))


def compute_face_measurements(landmarks_2d) -> dict[str, float]:
    """Compute normalised face-proportion measurements from 468 MediaPipe landmarks.

    Parameters
    ----------
    landmarks_2d : (N, 2) array-like
        MediaPipe FaceMesh landmark pixel positions.  N must be >= 468.

    Returns
    -------
    dict with float values (all normalised / dimensionless):
        width_height_ratio  – face width / face height  (>1 = wide, <1 = long)
        eye_spacing_ratio   – inner-corner gap / face width
        eye_openness        – eye height / eye width  (larger = more open)
        nose_width_ratio    – ala-to-ala / face width
        nose_height_ratio   – glabella-to-nose-tip / face height
        mouth_width_ratio   – mouth corner / face width
        jaw_taper           – jaw width / face width  (<1 = pointed jaw)
        forehead_ratio      – glabella-to-forehead / face height
    """
    if landmarks_2d is None:
        return dict(NEUTRAL_MEASUREMENTS)

    lm = np.asarray(landmarks_2d, dtype=np.float32)
    if lm.shape[0] < 468:
        return dict(NEUTRAL_MEASUREMENTS)

    face_h = _d(lm, _LM["forehead"], _LM["chin"])
    face_w = _d(lm, _LM["left_cheek"], _LM["right_cheek"])
    if face_h < 1.0 or face_w < 1.0:
        return dict(NEUTRAL_MEASUREMENTS)

    l_eye_h = _d(lm, _LM["left_eye_top"], _LM["left_eye_bottom"])
    l_eye_w = _d(lm, _LM["left_eye_outer"], _LM["left_eye_inner"])
    eye_openness = l_eye_h / max(l_eye_w, 1.0)

    nose_h = _d(lm, _LM["brow_center"], _LM["nose_tip"])
    forehead_h = _d(lm, _LM["forehead"], _LM["brow_center"])

    return {
        "width_height_ratio": float(np.clip(face_w / face_h,                                         0.55, 1.40)),
        "eye_spacing_ratio":  float(np.clip(_d(lm, _LM["left_eye_inner"], _LM["right_eye_inner"]) / face_w, 0.18, 0.60)),
        "eye_openness":       float(np.clip(eye_openness,                                             0.12, 0.55)),
        "nose_width_ratio":   float(np.clip(_d(lm, _LM["nose_left"], _LM["nose_right"]) / face_w,    0.15, 0.55)),
        "nose_height_ratio":  float(np.clip(nose_h / face_h,                                         0.15, 0.45)),
        "mouth_width_ratio":  float(np.clip(_d(lm, _LM["mouth_left"], _LM["mouth_right"]) / face_w,  0.28, 0.65)),
        "jaw_taper":          float(np.clip(_d(lm, _LM["left_jaw"], _LM["right_jaw"]) / face_w,      0.55, 1.10)),
        "forehead_ratio":     float(np.clip(forehead_h / face_h,                                     0.18, 0.50)),
    }


def measurements_to_head_params(measurements: dict[str, float], head_radius: float) -> dict:
    """Convert face measurements to head-geometry parameters for mesh_builder.

    Body measurements → body shape.  Face measurements → head shape.
    This function is the head-side equivalent of params.py's measurement
    regression: it converts selfie-derived proportions to the geometry knobs
    that control the 3D head ellipsoid and facial-feature positions.

    Parameters
    ----------
    measurements : dict
        Output of compute_face_measurements().
    head_radius : float
        Base head half-height in metres (from body params — sets overall
        head size; we only change the *shape*, not the size).

    Returns
    -------
    dict with keys:
        head_rx, head_ry, head_rz  – ellipsoid semi-axes (metres)
        eye_x      – eye-centre X as fraction of head_rx (left-right)
        eye_y      – eye-centre Y offset as fraction of head_ry (vertical)
        eye_z      – eye-centre Z as fraction of head_rz (front-back, fixed)
        eye_radius, iris_radius  – fractions of head_radius
        nose_radii  – (rx, ry, rz) fractions of head_radius
        nose_y, nose_z  – nose-centre fractions (of head_ry / head_rz)
        mouth_radii – (rx, ry, rz) fractions of head_radius
        mouth_y, mouth_z  – mouth-centre fractions
        ear_x       – ear-centre X fraction of head_rx
    """
    whr = measurements["width_height_ratio"]   # face width / height
    esr = measurements["eye_spacing_ratio"]    # inner-corner gap / face_width
    eo  = measurements["eye_openness"]          # eye height / eye width
    nwr = measurements["nose_width_ratio"]     # nose ala / face_width
    nhr = measurements["nose_height_ratio"]    # brow-to-tip / face_height
    mwr = measurements["mouth_width_ratio"]    # mouth / face_width
    jt  = measurements["jaw_taper"]            # jaw_width / face_width
    fr  = measurements["forehead_ratio"]       # forehead / face_height

    # ── Head ellipsoid ──────────────────────────────────────────────────────
    # head_ry (vertical) = head_radius — body params set the overall head
    # height; we only reshape left-right (rx) and front-back (rz).
    # A wider face (high whr) → wider rx; depth is ~80 % of width for humans.
    head_rx = head_radius * whr * 0.93
    head_ry = head_radius
    head_rz = head_radius * whr * 0.78

    # ── Eye X position ───────────────────────────────────────────────────────
    # esr = inner_gap / face_width.  Eye centre X ≈ (esr/2 + ~eye_radius/face_width).
    # Calibrated so that esr=0.34, whr=0.82 → eye_x ≈ 0.40  (the old _EYE_X constant).
    eye_x = float(np.clip((esr / 2.0 + 0.065) / (whr * 0.93), 0.22, 0.56))

    # ── Eye Y position (vertical offset above equator) ───────────────────────
    # A smaller forehead pushes the eyes higher; larger forehead → eyes lower.
    # fr=0.34 (neutral) → eye_y ≈ 0.05  (the old _EYE_Y constant).
    eye_y = float(np.clip(0.05 + (0.34 - fr) * 0.25, -0.08, 0.20))

    # ── Eye size ─────────────────────────────────────────────────────────────
    # eo=0.28 (neutral) → eye_radius ≈ 0.13  (old _EYE_RADIUS constant).
    eye_radius = float(np.clip(0.13 + (eo - 0.28) * 0.25, 0.08, 0.20))
    iris_radius = eye_radius * 0.54   # iris ≈ 54 % of eye-white radius

    # ── Nose ─────────────────────────────────────────────────────────────────
    # nwr=0.31 (neutral) → nose_rx ≈ 0.13  (old _NOSE_RADII[0] constant).
    nose_rx = float(np.clip(nwr * 0.42, 0.07, 0.24))
    nose_ry = float(np.clip(nhr * 0.70, 0.13, 0.32))
    nose_rz = float(np.clip(nwr * 0.70, 0.13, 0.28))
    # nose_y: nhr=0.28 (neutral) → nose_y ≈ -0.05  (old _NOSE_Y constant).
    nose_y = float(np.clip(-0.05 - (nhr - 0.28) * 0.40, -0.22, 0.04))

    # ── Mouth ────────────────────────────────────────────────────────────────
    # mwr=0.47 (neutral) → mouth_rx ≈ 0.22  (old _MOUTH_RADII[0] constant).
    mouth_rx = float(np.clip(mwr * 0.47, 0.14, 0.32))
    # mouth_y: fr=0.34 (neutral) → mouth_y ≈ -0.35  (old _MOUTH_Y constant).
    mouth_y = float(np.clip(-0.35 - (fr - 0.34) * 0.30, -0.52, -0.18))

    # ── Ear X position ───────────────────────────────────────────────────────
    # jaw_taper drives how far the ears sit from centre.
    # jt=0.82 (neutral) → ear_x ≈ 0.95  (old _EAR_X constant).
    ear_x = float(np.clip(jt * 1.16, 0.70, 1.10))

    return {
        "head_rx":     head_rx,
        "head_ry":     head_ry,
        "head_rz":     head_rz,
        "eye_x":       eye_x,
        "eye_y":       eye_y,
        "eye_z":       0.85,        # always on the front hemisphere
        "eye_radius":  eye_radius,
        "iris_radius": iris_radius,
        "nose_radii":  (nose_rx, nose_ry, nose_rz),
        "nose_y":      nose_y,
        "nose_z":      0.92,
        "mouth_radii": (mouth_rx, 0.07, 0.08),
        "mouth_y":     mouth_y,
        "mouth_z":     0.88,
        "ear_x":       ear_x,
    }
