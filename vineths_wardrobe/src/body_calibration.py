"""
eWardrobeAI — Stage 1: User Input & Body Calibration Module

Responsibilities
----------------
1. Accept raw body measurements (manual entry or QR/barcode scan)
2. Validate each measurement against anatomical feasibility bounds
3. Standardise units (cm only internally)
4. Compute derived sizing metrics: BMI-proxy, proportionality ratios
5. Produce AvatarScaleParams — the transformation matrix fed into the
   Blender avatar in the renderer

Avatar Scaling Model
--------------------
The Blender base avatar is rigged at a neutral pose with these reference dims:
  Shoulder : 40 cm  |  Chest : 90 cm  |  Waist : 70 cm  |  Height : 170 cm

Scale factors are computed per axis and clamped to prevent mesh distortion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Reference (base avatar) dimensions ───────────────────────────────────────
_BASE_SHOULDER_CM = 40.0
_BASE_CHEST_CM    = 90.0
_BASE_WAIST_CM    = 70.0
_BASE_HEIGHT_CM   = 170.0
_BASE_HIP_CM      = 95.0
_BASE_INSEAM_CM   = 80.0

# ── Anatomical feasibility bounds (cm) ───────────────────────────────────────
_BOUNDS: dict[str, tuple[float, float]] = {
    'shoulder_width_cm': (30.0, 65.0),
    'chest_cm':          (60.0, 160.0),
    'waist_cm':          (50.0, 150.0),
    'height_cm':         (120.0, 230.0),
    'hip_cm':            (60.0, 175.0),
    'inseam_cm':         (60.0, 110.0),
    'weight_kg':         (30.0, 250.0),
}

# ── Standard clothing size thresholds (chest measurement) ────────────────────
_SIZE_THRESHOLDS: list[tuple[float, str]] = [
    (82.0,  'XS'),
    (88.0,  'S'),
    (96.0,  'M'),
    (104.0, 'L'),
    (112.0, 'XL'),
    (124.0, 'XXL'),
    (float('inf'), 'XXXL'),
]


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class BodyMeasurements:
    """
    Raw measurements provided by the user.

    All measurements in centimetres.
    Optional fields default to anatomically-estimated values when omitted.
    """
    shoulder_width_cm: float
    chest_cm:          float
    waist_cm:          float
    height_cm:         float
    hip_cm:            Optional[float] = None
    inseam_cm:         Optional[float] = None
    weight_kg:         Optional[float] = None

    def __post_init__(self):
        """Estimate missing optional measurements from provided core values."""
        if self.hip_cm is None:
            # Hip ≈ waist + 25 cm (typical female/male average offset)
            self.hip_cm = round(self.waist_cm + 25.0, 1)
        if self.inseam_cm is None:
            # Inseam ≈ 47 % of standing height (anthropometric average)
            self.inseam_cm = round(self.height_cm * 0.47, 1)


@dataclass
class ValidationResult:
    """Outcome of measurement validation."""
    is_valid:  bool
    errors:    list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)


@dataclass
class SizingProfile:
    """
    Derived sizing metadata produced after validation.

    Used by the NisfaMatchmaking outfit recommender to filter
    clothing items to the correct size category.
    """
    standard_size:    str            # XS / S / M / L / XL / XXL / XXXL
    shoulder_ratio:   float          # shoulder / height
    waist_hip_ratio:  float          # WHR — body shape proxy
    torso_length_cm:  float          # height - inseam (approx)
    body_type:        str            # 'hourglass' | 'rectangle' | 'pear' | 'inverted_triangle'


@dataclass
class AvatarScaleParams:
    """
    Per-bone / per-axis scale factors applied to the Blender avatar rig.

    These values are sent to the Three.js renderer as a JSON payload
    and applied using avatar_mesh.scale.set(x, y, z) per bone group.
    """
    # Global scale (height-driven)
    global_scale_y: float      # vertical / height axis

    # Upper body
    shoulder_scale_x: float    # shoulder width
    chest_scale_x:    float    # chest circumference
    waist_scale_x:    float    # waist circumference

    # Lower body
    hip_scale_x:      float
    leg_scale_y:      float    # inseam / leg length

    # Head (driven by face processor inter-eye distance)
    head_scale:       float = 1.0

    def to_dict(self) -> dict:
        return {
            'globalY':   round(self.global_scale_y, 4),
            'shoulderX': round(self.shoulder_scale_x, 4),
            'chestX':    round(self.chest_scale_x, 4),
            'waistX':    round(self.waist_scale_x, 4),
            'hipX':      round(self.hip_scale_x, 4),
            'legY':      round(self.leg_scale_y, 4),
            'headScale': round(self.head_scale, 4),
        }


# ── BodyCalibrator ────────────────────────────────────────────────────────────

class BodyCalibrator:
    """
    Validates and standardises user measurements, then computes
    the avatar scale parameters for the 3D renderer.

    Example
    -------
    >>> cal = BodyCalibrator()
    >>> meas = BodyMeasurements(42, 92, 72, 168)
    >>> result = cal.validate(meas)
    >>> if result.is_valid:
    ...     params = cal.compute_avatar_scale(meas)
    ...     profile = cal.compute_sizing_profile(meas)
    """

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def validate(measurements: BodyMeasurements) -> ValidationResult:
        """
        Validate all measurements against anatomical bounds and
        cross-field consistency rules.

        Returns ValidationResult with accumulated errors and warnings.
        """
        errors:   list[str] = []
        warnings: list[str] = []

        m = measurements
        fields_to_check = {
            'shoulder_width_cm': m.shoulder_width_cm,
            'chest_cm':          m.chest_cm,
            'waist_cm':          m.waist_cm,
            'height_cm':         m.height_cm,
            'hip_cm':            m.hip_cm,
            'inseam_cm':         m.inseam_cm,
        }
        if m.weight_kg is not None:
            fields_to_check['weight_kg'] = m.weight_kg

        # ── Range checks ─────────────────────────────────────────────────────
        for field_name, value in fields_to_check.items():
            if value is None:
                continue
            lo, hi = _BOUNDS[field_name]
            if value < lo or value > hi:
                errors.append(
                    f"{field_name} = {value:.1f} cm is outside the feasible "
                    f"range [{lo:.0f}, {hi:.0f}]."
                )

        # ── Cross-field consistency checks ───────────────────────────────────
        if m.waist_cm > m.chest_cm:
            warnings.append(
                f"Waist ({m.waist_cm} cm) exceeds chest ({m.chest_cm} cm) — "
                "verify measurement orientation."
            )

        if m.shoulder_width_cm > m.chest_cm * 0.55:
            warnings.append(
                "Shoulder width is unusually wide relative to chest. "
                "Please re-verify shoulder measurement."
            )

        if m.inseam_cm is not None and m.inseam_cm > m.height_cm * 0.60:
            warnings.append(
                f"Inseam ({m.inseam_cm} cm) > 60 % of height ({m.height_cm} cm). "
                "This may produce leg-length distortion on the avatar."
            )

        # ── Proportionality ratio checks ─────────────────────────────────────
        waist_height_ratio = m.waist_cm / m.height_cm
        if waist_height_ratio > 0.63:
            warnings.append(
                f"Waist-to-height ratio {waist_height_ratio:.2f} is elevated. "
                "Avatar may display as stockier than expected."
            )

        is_valid = len(errors) == 0
        if errors:
            for e in errors:
                logger.error(f"[BodyCalibrator] Validation ERROR: {e}")
        if warnings:
            for w in warnings:
                logger.warning(f"[BodyCalibrator] Validation WARNING: {w}")

        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    # ── Sizing Profile ────────────────────────────────────────────────────────

    @staticmethod
    def compute_sizing_profile(m: BodyMeasurements) -> SizingProfile:
        """
        Derive standardised size label and body shape classification
        used by NisfaMatchmaking for outfit filtering.
        """
        # Standard size from chest circumference
        size = 'XXXL'
        for threshold, label in _SIZE_THRESHOLDS:
            if m.chest_cm <= threshold:
                size = label
                break

        shoulder_ratio  = m.shoulder_width_cm / m.height_cm
        waist_hip_ratio = m.waist_cm / m.hip_cm if m.hip_cm else 0.0
        torso_length    = m.height_cm - (m.inseam_cm or (m.height_cm * 0.47))

        # Body shape heuristic
        s_to_h = m.shoulder_width_cm / m.hip_cm if m.hip_cm else 1.0
        body_type = _classify_body_type(m.chest_cm, m.waist_cm,
                                         m.hip_cm or m.waist_cm + 25,
                                         m.shoulder_width_cm)

        return SizingProfile(
            standard_size   = size,
            shoulder_ratio  = round(shoulder_ratio, 4),
            waist_hip_ratio = round(waist_hip_ratio, 4),
            torso_length_cm = round(torso_length, 2),
            body_type       = body_type,
        )

    # ── Avatar Scale ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_avatar_scale(
        m: BodyMeasurements,
        inter_eye_dist_px: float = 0.0,
        reference_ied_px: float  = 42.0,
    ) -> AvatarScaleParams:
        """
        Compute per-bone scale factors relative to the base Blender avatar.

        Scale factor = user_measurement / base_avatar_measurement
        All factors are clamped to [0.70, 1.40] to prevent mesh degeneracy.

        Parameters
        ----------
        m                  : validated BodyMeasurements
        inter_eye_dist_px  : inter-ocular distance from FaceProcessor
        reference_ied_px   : expected IED for the base avatar at default size
        """
        def _clamp(v: float, lo: float = 0.70, hi: float = 1.40) -> float:
            return max(lo, min(hi, v))

        height_scale   = _clamp(m.height_cm   / _BASE_HEIGHT_CM)
        shoulder_scale = _clamp(m.shoulder_width_cm / _BASE_SHOULDER_CM)
        chest_scale    = _clamp(m.chest_cm    / _BASE_CHEST_CM)
        waist_scale    = _clamp(m.waist_cm    / _BASE_WAIST_CM)
        hip_scale      = _clamp((m.hip_cm or m.waist_cm + 25) / _BASE_HIP_CM)

        inseam = m.inseam_cm or (m.height_cm * 0.47)
        leg_scale = _clamp(inseam / _BASE_INSEAM_CM)

        # Head scale: driven by face inter-ocular distance
        head_scale = 1.0
        if inter_eye_dist_px > 0 and reference_ied_px > 0:
            head_scale = _clamp(
                inter_eye_dist_px / reference_ied_px,
                lo=0.85, hi=1.15
            )

        params = AvatarScaleParams(
            global_scale_y   = height_scale,
            shoulder_scale_x = shoulder_scale,
            chest_scale_x    = chest_scale,
            waist_scale_x    = waist_scale,
            hip_scale_x      = hip_scale,
            leg_scale_y      = leg_scale,
            head_scale       = head_scale,
        )

        logger.info(f"[BodyCalibrator] Scale params: {params.to_dict()}")
        return params

    # ── Public Convenience ────────────────────────────────────────────────────

    def calibrate(
        self,
        shoulder_width_cm: float,
        chest_cm: float,
        waist_cm: float,
        height_cm: float,
        hip_cm: Optional[float]    = None,
        inseam_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
        inter_eye_dist_px: float   = 0.0,
    ) -> tuple[ValidationResult, Optional[AvatarScaleParams], Optional[SizingProfile]]:
        """
        All-in-one convenience method called by the FastAPI endpoint.

        Returns (ValidationResult, AvatarScaleParams | None, SizingProfile | None).
        AvatarScaleParams and SizingProfile are None when validation fails.
        """
        m = BodyMeasurements(
            shoulder_width_cm=shoulder_width_cm,
            chest_cm=chest_cm,
            waist_cm=waist_cm,
            height_cm=height_cm,
            hip_cm=hip_cm,
            inseam_cm=inseam_cm,
            weight_kg=weight_kg,
        )
        result = self.validate(m)
        if not result.is_valid:
            return result, None, None

        scale   = self.compute_avatar_scale(m, inter_eye_dist_px)
        profile = self.compute_sizing_profile(m)
        return result, scale, profile


# ── Body Type Classifier ──────────────────────────────────────────────────────

def _classify_body_type(
    chest_cm: float,
    waist_cm: float,
    hip_cm: float,
    shoulder_cm: float,
) -> str:
    """
    Rule-based body shape classification based on measurements.

    Categories:
      hourglass          — balanced shoulder/hip, narrow waist
      inverted_triangle  — broad shoulders relative to hips
      pear               — hips wider than shoulders
      rectangle          — similar chest, waist, hip measurements
    """
    waist_def   = (chest_cm + hip_cm) / 2.0 - waist_cm   # definition
    s_h_diff    = abs(shoulder_cm * 2.3 - hip_cm)         # shoulder × 2.3 ≈ chest
    hip_vs_shldr = hip_cm / (shoulder_cm * 2.3)

    if waist_def > 9 and s_h_diff < 8:
        return 'hourglass'
    if hip_vs_shldr < 0.87:
        return 'inverted_triangle'
    if hip_vs_shldr > 1.13:
        return 'pear'
    return 'rectangle'
