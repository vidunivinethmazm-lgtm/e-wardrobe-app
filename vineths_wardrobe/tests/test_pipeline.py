"""
Tests — Full Pipeline Integration
Covers: all 4 stages end-to-end using a synthetic face image
No selfie upload required — uses cv2-generated test image.
"""

import pytest
import cv2
import numpy as np

from src.virtual_tryon_pipeline import VirtualTryOnPipeline, TryOnRequest, TryOnResult
from src.body_calibration        import AvatarScaleParams, SizingProfile
from src.face_processor          import FaceProfile
from src.outfit_recommender      import OutfitRecommendation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    p = VirtualTryOnPipeline(use_mediapipe=False, use_cnn=False)
    yield p
    p.teardown()


def _synthetic_selfie() -> bytes:
    """96×96 synthetic face image encoded as JPEG bytes."""
    img = np.full((200, 200), 200, dtype=np.uint8)
    cv2.ellipse(img, (100, 105), (75, 90), 0, 0, 360, 230, -1)
    cv2.circle(img, (75, 85),  14, 100, -1)
    cv2.circle(img, (125, 85), 14, 100, -1)
    cv2.circle(img, (75, 85),  8,  40, -1)
    cv2.circle(img, (125, 85), 8,  40, -1)
    cv2.ellipse(img, (100, 130), (15, 10), 0, 0, 360, 130, -1)
    cv2.ellipse(img, (100, 155), (28, 12), 0, 0, 180,  60,  2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def _default_request(**overrides) -> TryOnRequest:
    base = dict(
        image_bytes       = _synthetic_selfie(),
        shoulder_width_cm = 42.0,
        chest_cm          = 92.0,
        waist_cm          = 72.0,
        height_cm         = 168.0,
        preferred_styles  = ['smart_casual', 'casual'],
        occasion          = 'office',
        animation_key     = 'idle',
        top_k             = 3,
    )
    base.update(overrides)
    return TryOnRequest(**base)


# ── Stage 1: Body Calibration ─────────────────────────────────────────────────

class TestStage1BodyCalibration:

    def test_valid_request_succeeds(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.success is True

    def test_sizing_profile_returned(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.sizing_profile is not None
        assert isinstance(result.sizing_profile, SizingProfile)

    def test_sizing_profile_standard_size(self, pipeline):
        result = pipeline.process(_default_request(chest_cm=92))
        assert result.sizing_profile.standard_size == 'M'

    def test_scale_params_returned(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.scale_params is not None
        assert isinstance(result.scale_params, AvatarScaleParams)

    def test_invalid_measurements_fail_gracefully(self, pipeline):
        result = pipeline.process(_default_request(chest_cm=250))
        assert result.success       is False
        assert result.error_message is not None
        assert result.render_payload is None

    def test_body_type_in_valid_set(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.sizing_profile.body_type in (
            'hourglass', 'inverted_triangle', 'pear', 'rectangle'
        )


# ── Stage 2: Face Processing ──────────────────────────────────────────────────

class TestStage2FaceProcessing:

    def test_face_profile_returned(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.face_profile is not None
        assert isinstance(result.face_profile, FaceProfile)

    def test_pipeline_continues_without_cnn_model(self, pipeline):
        # Pipeline was initialised with use_cnn=False — should still succeed
        result = pipeline.process(_default_request())
        assert result.success is True

    def test_face_profile_has_no_crash_on_blank_image(self, pipeline):
        blank = np.zeros((96, 96), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', blank)
        result = pipeline.process(_default_request(image_bytes=buf.tobytes()))
        assert result.success is True   # pipeline degrades gracefully


# ── Stage 3: Outfit Recommendation ───────────────────────────────────────────

class TestStage3OutfitRecommendation:

    def test_recommendations_returned(self, pipeline):
        result = pipeline.process(_default_request())
        assert len(result.recommendations) > 0

    def test_all_recommendation_items_wearable(self, pipeline):
        result = pipeline.process(_default_request())
        for rec in result.recommendations:
            for item in rec.items:
                assert item.is_wearable, (
                    f"{item.garment_id} ({item.cleaning_status.value}) "
                    "should not appear in recommendations"
                )

    def test_top_k_limit_respected(self, pipeline):
        result = pipeline.process(_default_request(top_k=2))
        assert len(result.recommendations) <= 2

    def test_recommendations_sorted_by_score(self, pipeline):
        result = pipeline.process(_default_request(top_k=5))
        scores = [r.score for r in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_no_dirty_garments_in_recommendations(self, pipeline):
        EXCLUDED = {'GAR-003', 'GAR-004', 'GAR-011'}
        result   = pipeline.process(_default_request(top_k=10))
        for rec in result.recommendations:
            for item in rec.items:
                assert item.garment_id not in EXCLUDED


# ── Stage 4: Avatar Render Payload ────────────────────────────────────────────

class TestStage4RenderPayload:

    def test_render_payload_present(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.render_payload is not None

    def test_render_payload_serialisable(self, pipeline):
        result  = pipeline.process(_default_request())
        payload = result.render_payload.to_dict()
        assert isinstance(payload, dict)
        assert 'scaleParams'    in payload
        assert 'clothingAssets' in payload
        assert 'animation'      in payload
        assert 'scene'          in payload

    def test_scale_params_in_payload(self, pipeline):
        result  = pipeline.process(_default_request())
        sp      = result.render_payload.to_dict()['scaleParams']
        assert all(k in sp for k in
                   ['globalY','shoulderX','chestX','waistX','hipX','legY','headScale'])
        for v in sp.values():
            assert 0.70 <= v <= 1.40

    def test_alternate_payloads_returned(self, pipeline):
        result = pipeline.process(_default_request(top_k=3))
        assert isinstance(result.alternate_payloads, list)

    def test_animation_key_reflected_in_payload(self, pipeline):
        result  = pipeline.process(_default_request(animation_key='walk'))
        clip    = result.render_payload.to_dict()['animation']['clipName']
        assert 'Walk' in clip or 'walk' in clip.lower()

    def test_outfit_name_present(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.render_payload.outfit_name != ''

    def test_processing_time_recorded(self, pipeline):
        result = pipeline.process(_default_request())
        assert result.processing_time_ms > 0


# ── Wardrobe Management ───────────────────────────────────────────────────────

class TestWardrobeManagement:

    def test_mark_worn_excludes_from_next_recommendation(self, pipeline):
        # Get initial recommendations
        result1 = pipeline.process(_default_request(top_k=10))
        ids_before = {
            item.garment_id
            for rec in result1.recommendations
            for item in rec.items
        }

        # Mark all recommended items as worn (Dirty)
        for gid in ids_before:
            pipeline.mark_garment_worn(gid)

        # Run again — previously worn items should not reappear
        result2 = pipeline.process(_default_request(top_k=10))
        ids_after = {
            item.garment_id
            for rec in result2.recommendations
            for item in rec.items
        }
        assert ids_before.isdisjoint(ids_after)

        # Restore
        for gid in ids_before:
            pipeline.mark_garment_clean(gid)

    def test_wardrobe_summary_totals(self, pipeline):
        summary = pipeline.get_wardrobe_summary()
        assert sum(summary.values()) == 12  # total seeded garments
