"""
eWardrobeAI — Main Virtual Try-On Pipeline Orchestrator

This module is the central coordinator for all four workflow stages:

  Stage 1 ── BodyCalibrator
               Validates + standardises user measurements
               Outputs AvatarScaleParams + SizingProfile

  Stage 2 ── FaceProcessor
               MediaPipe 468-landmark detection (live)
               CNN 15-keypoint regression (uploaded selfie)
               Outputs FaceProfile (landmarks + texture + geometry)

  Stage 3 ── RaveehaOrganisationalDB + NisfaMatchmaking
               Queries wardrobe for Clean + owned garments
               Generates ranked outfit bundles
               Outputs List[OutfitRecommendation]

  Stage 4 ── AvatarManager
               Builds AvatarRenderPayload from all stage outputs
               JSON payload consumed by Three.js renderer

Data Flow
---------
    UserInput (selfie + measurements + preferences)
         │
         ▼
    [Stage 1] BodyCalibrator ──→ AvatarScaleParams, SizingProfile
         │
         ▼
    [Stage 2] FaceProcessor ──→ FaceProfile
         │
         ▼
    [Stage 3] RaveehaDB → NisfaMatchmaking ──→ List[OutfitRecommendation]
         │
         ▼
    [Stage 4] AvatarManager ──→ AvatarRenderPayload (JSON → Three.js)
"""

from __future__ import annotations

import time
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.body_calibration   import BodyCalibrator, BodyMeasurements, AvatarScaleParams, SizingProfile
from src.face_processor     import FaceProcessor, FaceProfile
from src.outfit_recommender import (
    RaveehaOrganisationalDB,
    NisfaMatchmaking,
    OutfitRecommendation,
    Style,
    CleaningStatus,
)
from src.wardrobe_database import SQLiteWardrobeDB
from src.avatar_manager     import AvatarManager, AvatarRenderPayload

logger = logging.getLogger(__name__)


# ── Request / Response Models ─────────────────────────────────────────────────

@dataclass
class TryOnRequest:
    """
    All inputs provided by the user to initiate a virtual try-on session.

    image_bytes      : Raw bytes of the selfie JPEG/PNG uploaded via the app
    shoulder_width_cm: Shoulder width entered or scanned by user
    chest_cm         : Chest circumference
    waist_cm         : Waist circumference
    height_cm        : Standing height
    hip_cm           : Hip circumference (optional — estimated if absent)
    inseam_cm        : Inseam length    (optional — estimated if absent)
    weight_kg        : Body weight      (optional — used for BMI proxy only)
    preferred_styles : Ordered list of style preferences (e.g. ['smart','casual'])
    occasion         : Occasion tag     (e.g. 'office', 'casual', 'evening')
    animation_key    : Which Mixamo animation to play ('idle','walk','catwalk',…)
    top_k            : Max outfit recommendations to return
    """
    image_bytes:       bytes
    shoulder_width_cm: float = 42.0
    chest_cm:          float = 92.0
    waist_cm:          float = 72.0
    height_cm:         float = 168.0
    hip_cm:            Optional[float] = None
    inseam_cm:         Optional[float] = None
    weight_kg:         Optional[float] = None
    preferred_styles:  list[str]       = field(default_factory=lambda: ['smart_casual', 'casual'])
    occasion:          str             = 'casual'
    animation_key:     str             = 'idle'
    top_k:             int             = 3


@dataclass
class TryOnResult:
    """
    Complete pipeline output for a single try-on session.

    Contains both the primary render payload (for the selected/top outfit)
    and all alternative recommendations (for the outfit carousel).
    """
    success:              bool
    error_message:        Optional[str]          = None
    warnings:             list[str]              = field(default_factory=list)

    # Stage outputs
    sizing_profile:       Optional[SizingProfile]          = None
    scale_params:         Optional[AvatarScaleParams]      = None
    face_profile:         Optional[FaceProfile]            = None
    recommendations:      list[OutfitRecommendation]       = field(default_factory=list)

    # Render payload (primary recommendation, ready for Three.js)
    render_payload:       Optional[AvatarRenderPayload]    = None

    # Alternative payloads for outfit carousel
    alternate_payloads:   list[AvatarRenderPayload]        = field(default_factory=list)

    # Timing
    processing_time_ms:   float = 0.0


# ── VirtualTryOnPipeline ──────────────────────────────────────────────────────

class VirtualTryOnPipeline:
    """
    Orchestrates all four pipeline stages for a single user try-on session.

    Initialise once per server start; process() is thread-safe for
    concurrent user requests (each call is stateless at the pipeline level).

    Example
    -------
    >>> pipeline = VirtualTryOnPipeline()
    >>> result = pipeline.process(request)
    >>> renderer_json = result.render_payload.to_dict()
    """

    def __init__(
        self,
        use_mediapipe: bool = True,
        use_cnn: bool       = True,
    ):
        logger.info("[Pipeline] Initialising eWardrobeAI Virtual Try-On Pipeline …")

        self._calibrator = BodyCalibrator()
        self._face_proc  = FaceProcessor(
            use_mediapipe=use_mediapipe,
            use_cnn=use_cnn,
        )
        self._wardrobe_db = SQLiteWardrobeDB()          # persists to wardrobe.db
        self._matchmaker  = NisfaMatchmaking(self._wardrobe_db)
        self._avatar_mgr  = AvatarManager()

        logger.info("[Pipeline] All subsystems initialised. Ready.")

    # ── Public Entry Point ────────────────────────────────────────────────────

    def process(self, request: TryOnRequest) -> TryOnResult:
        """
        Execute all four pipeline stages and return TryOnResult.

        Failures in individual stages are caught and reported in
        TryOnResult.error_message / warnings without crashing the pipeline.
        """
        t_start = time.perf_counter()

        # ── Stage 1: Body Calibration ─────────────────────────────────────────
        logger.info("[Pipeline] Stage 1 — Body Calibration")
        val_result, scale_params, sizing_profile = self._run_stage_1(request)

        if not val_result.is_valid:
            return TryOnResult(
                success=False,
                error_message='; '.join(val_result.errors),
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
            )

        # ── Stage 2: Face Processing ──────────────────────────────────────────
        logger.info("[Pipeline] Stage 2 — Face Processing")
        face_profile, face_warning = self._run_stage_2(request.image_bytes)
        warnings = val_result.warnings.copy()
        if face_warning:
            warnings.append(face_warning)

        # Update head scale using inter-eye distance from face analysis
        if face_profile.inter_eye_dist > 0:
            scale_params = BodyCalibrator.compute_avatar_scale(
                m=BodyMeasurements(
                    shoulder_width_cm=request.shoulder_width_cm,
                    chest_cm=request.chest_cm,
                    waist_cm=request.waist_cm,
                    height_cm=request.height_cm,
                    hip_cm=request.hip_cm,
                    inseam_cm=request.inseam_cm,
                ),
                inter_eye_dist_px=face_profile.inter_eye_dist,
            )

        # ── Stage 3: Outfit Recommendation ───────────────────────────────────
        logger.info("[Pipeline] Stage 3 — Outfit Recommendation")
        recommendations = self._run_stage_3(request, sizing_profile)

        if not recommendations:
            return TryOnResult(
                success=False,
                error_message=(
                    "No available outfits found for the given size and style. "
                    "Check that garments are not all marked as Dirty or In Laundry."
                ),
                warnings=warnings,
                sizing_profile=sizing_profile,
                scale_params=scale_params,
                face_profile=face_profile,
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
            )

        # ── Stage 4: Avatar Rendering ─────────────────────────────────────────
        logger.info("[Pipeline] Stage 4 — Avatar Render Payload Assembly")
        primary_payload, alternate_payloads = self._run_stage_4(
            scale_params, face_profile, recommendations, request.animation_key
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Pipeline] Complete. {elapsed_ms:.1f} ms  "
                    f"{len(recommendations)} outfit(s) prepared.")

        return TryOnResult(
            success             = True,
            warnings            = warnings,
            sizing_profile      = sizing_profile,
            scale_params        = scale_params,
            face_profile        = face_profile,
            recommendations     = recommendations,
            render_payload      = primary_payload,
            alternate_payloads  = alternate_payloads,
            processing_time_ms  = elapsed_ms,
        )

    # ── Stage Implementations ─────────────────────────────────────────────────

    def _run_stage_1(self, request: TryOnRequest):
        return self._calibrator.calibrate(
            shoulder_width_cm  = request.shoulder_width_cm,
            chest_cm           = request.chest_cm,
            waist_cm           = request.waist_cm,
            height_cm          = request.height_cm,
            hip_cm             = request.hip_cm,
            inseam_cm          = request.inseam_cm,
            weight_kg          = request.weight_kg,
        )

    def _run_stage_2(
        self, image_bytes: bytes
    ) -> tuple[FaceProfile, Optional[str]]:
        """Decode image bytes → BGR array → FaceProcessor."""
        warning = None
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("cv2.imdecode returned None — invalid image data.")
            profile = self._face_proc.process(image_bgr)
        except Exception as e:
            logger.warning(f"[Pipeline] Face processing failed: {e}")
            profile = FaceProfile()
            warning = f"Face analysis unavailable: {e}. Avatar head will use default geometry."
        return profile, warning

    def _run_stage_3(
        self,
        request: TryOnRequest,
        sizing_profile: SizingProfile,
    ) -> list[OutfitRecommendation]:
        """Convert string style preferences → Style enums → matchmaker."""
        style_map = {
            'casual':       Style.CASUAL,
            'formal':       Style.FORMAL,
            'smart_casual': Style.SMART,
            'smart':        Style.SMART,
            'sporty':       Style.SPORTY,
            'evening':      Style.EVENING,
        }
        styles = [
            style_map[s.lower()]
            for s in request.preferred_styles
            if s.lower() in style_map
        ]
        if not styles:
            styles = [Style.CASUAL]

        return self._matchmaker.recommend(
            size             = sizing_profile.standard_size,
            body_type        = sizing_profile.body_type,
            preferred_styles = styles,
            occasion         = request.occasion,
            top_k            = request.top_k,
        )

    def _run_stage_4(
        self,
        scale_params:    AvatarScaleParams,
        face_profile:    FaceProfile,
        recommendations: list[OutfitRecommendation],
        animation_key:   str,
    ) -> tuple[AvatarRenderPayload, list[AvatarRenderPayload]]:
        """Build render payloads for primary + alternate outfit recommendations."""
        primary_payload = self._avatar_mgr.build_render_payload(
            scale_params  = scale_params,
            face_profile  = face_profile,
            outfit        = recommendations[0],
            animation_key = animation_key,
        )

        alternate_payloads: list[AvatarRenderPayload] = []
        for rec in recommendations[1:]:
            alt = self._avatar_mgr.build_render_payload(
                scale_params  = scale_params,
                face_profile  = face_profile,
                outfit        = rec,
                animation_key = 'idle',
            )
            alternate_payloads.append(alt)

        return primary_payload, alternate_payloads

    # ── Wardrobe Management ───────────────────────────────────────────────────

    def mark_garment_worn(self, garment_id: str):
        """Mark a garment as Dirty after the user has worn it."""
        self._wardrobe_db.update_cleaning_status(
            garment_id, CleaningStatus.DIRTY
        )

    def mark_garment_laundering(self, garment_id: str):
        """Move a garment from Dirty to In Laundry."""
        self._wardrobe_db.update_cleaning_status(
            garment_id, CleaningStatus.IN_LAUNDRY
        )

    def mark_garment_clean(self, garment_id: str):
        """Mark a garment as Clean and back in wardrobe."""
        self._wardrobe_db.update_cleaning_status(
            garment_id, CleaningStatus.CLEAN
        )

    def get_wardrobe_summary(self) -> dict:
        return self._wardrobe_db.wardrobe_summary()

    def teardown(self):
        """Release held resources (MediaPipe handles etc.)."""
        self._face_proc.release()
