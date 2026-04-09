"""
Tests — FastAPI Endpoints
Covers all REST endpoints using httpx TestClient (no running server needed).
"""

import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# ── Synthetic image helper ────────────────────────────────────────────────────

def _synthetic_selfie_bytes() -> bytes:
    img = np.full((200, 200), 200, dtype=np.uint8)
    cv2.ellipse(img, (100, 105), (75, 90), 0, 0, 360, 230, -1)
    cv2.circle(img, (75,  85),  12, 80, -1)
    cv2.circle(img, (125, 85),  12, 80, -1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


# ── Health & System ───────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_ok(self):
        r = client.get('/api/health')
        assert r.status_code == 200
        data = r.json()
        assert data['status']        == 'ok'
        assert data['pipelineReady'] is True

    def test_root_returns_html(self):
        r = client.get('/')
        assert r.status_code == 200
        assert 'text/html' in r.headers['content-type']


# ── Wardrobe Endpoints ────────────────────────────────────────────────────────

class TestWardrobeEndpoints:

    def test_summary_returns_three_keys(self):
        r = client.get('/api/wardrobe/summary')
        assert r.status_code == 200
        data = r.json()
        assert 'Clean'      in data
        assert 'Dirty'      in data
        assert 'In Laundry' in data

    def test_summary_total_is_12(self):
        r    = client.get('/api/wardrobe/summary')
        data = r.json()
        assert sum(data.values()) == 12

    def test_items_returns_all_garments(self):
        r    = client.get('/api/wardrobe/items')
        assert r.status_code == 200
        data = r.json()
        assert data['total'] == 12
        assert len(data['items']) == 12

    def test_items_contain_required_fields(self):
        r    = client.get('/api/wardrobe/items')
        item = r.json()['items'][0]
        for field in ['garmentId', 'name', 'category', 'cleaningStatus',
                      'isWearable', 'assetPath']:
            assert field in item

    def test_dirty_items_not_wearable(self):
        r     = client.get('/api/wardrobe/items')
        items = r.json()['items']
        dirty = [i for i in items if i['cleaningStatus'] in ('Dirty', 'In Laundry')]
        for item in dirty:
            assert item['isWearable'] is False

    def test_update_status_to_dirty(self):
        r = client.patch('/api/wardrobe/GAR-002/status',
                         json={'status': 'Dirty'})
        assert r.status_code == 200
        data = r.json()
        assert data['newStatus'] == 'Dirty'
        assert data['isWearable'] is False

    def test_update_status_back_to_clean(self):
        client.patch('/api/wardrobe/GAR-002/status', json={'status': 'Dirty'})
        r = client.patch('/api/wardrobe/GAR-002/status', json={'status': 'Clean'})
        assert r.status_code == 200
        assert r.json()['isWearable'] is True

    def test_update_status_invalid_value(self):
        r = client.patch('/api/wardrobe/GAR-001/status',
                         json={'status': 'Washed'})
        assert r.status_code == 400

    def test_update_status_nonexistent_garment(self):
        r = client.patch('/api/wardrobe/GAR-999/status',
                         json={'status': 'Clean'})
        assert r.status_code == 404


# ── Avatar & Animation Endpoints ──────────────────────────────────────────────

class TestAnimationEndpoint:

    def test_animations_returns_all_keys(self):
        r    = client.get('/api/animations')
        assert r.status_code == 200
        data = r.json()
        assert 'animations' in data
        for key in ['idle', 'walk', 'rotate', 'pose_t', 'pose_a', 'catwalk']:
            assert key in data['animations']


# ── Body Calibration Endpoint ─────────────────────────────────────────────────

class TestSizingValidateEndpoint:

    def test_valid_default_values(self):
        r = client.post('/api/sizing/validate', json={
            'shoulder_width_cm': 42,
            'chest_cm': 92, 'waist_cm': 72, 'height_cm': 168,
        })
        assert r.status_code == 200
        data = r.json()
        assert data['valid']                         is True
        assert data['sizingProfile']['standardSize'] == 'M'
        assert 'avatarScaleParams'                   in data

    def test_invalid_chest_returns_422(self):
        r = client.post('/api/sizing/validate', json={
            'shoulder_width_cm': 42,
            'chest_cm': 250, 'waist_cm': 72, 'height_cm': 168,
        })
        assert r.status_code == 422

    def test_scale_params_all_keys_present(self):
        r    = client.post('/api/sizing/validate', json={
            'shoulder_width_cm': 42,
            'chest_cm': 92, 'waist_cm': 72, 'height_cm': 168,
        })
        sp   = r.json()['avatarScaleParams']
        for key in ['globalY','shoulderX','chestX','waistX','hipX','legY','headScale']:
            assert key in sp

    @pytest.mark.parametrize("chest,expected", [
        (78, 'XS'), (85, 'S'), (92, 'M'),
        (100, 'L'), (108, 'XL'), (118, 'XXL'), (130, 'XXXL'),
    ])
    def test_size_labels(self, chest, expected):
        r = client.post('/api/sizing/validate', json={
            'shoulder_width_cm': 40,
            'chest_cm': chest, 'waist_cm': 70, 'height_cm': 170,
        })
        assert r.json()['sizingProfile']['standardSize'] == expected


# ── Demo Try-On Endpoint ──────────────────────────────────────────────────────

class TestDemoTryOnEndpoint:

    def test_demo_tryon_succeeds(self):
        r = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
        })
        assert r.status_code == 200
        data = r.json()
        assert data['success']             is True
        assert len(data['recommendations']) > 0
        assert data['renderPayload']       is not None

    def test_demo_tryon_has_sizing_profile(self):
        r    = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
        })
        data = r.json()
        assert data['sizingProfile'] is not None
        assert data['sizingProfile']['standardSize'] == 'M'

    def test_demo_tryon_recommendations_wearable_only(self):
        r    = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168, 'top_k': 10,
        })
        recs = r.json()['recommendations']
        for rec in recs:
            for item in rec['items']:
                assert item['status'] == 'Clean'

    def test_demo_tryon_scores_valid_range(self):
        r    = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
        })
        for rec in r.json()['recommendations']:
            assert 0.0 <= rec['score'] <= 1.0

    def test_demo_tryon_render_payload_keys(self):
        r       = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
        })
        payload = r.json()['renderPayload']
        for key in ['scaleParams', 'clothingAssets', 'animation', 'scene']:
            assert key in payload

    def test_demo_tryon_invalid_measurements(self):
        r = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 250,
            'waist_cm': 72, 'height_cm': 168,
        })
        assert r.status_code == 422

    def test_demo_tryon_walk_animation(self):
        r = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
            'animation_key': 'walk',
        })
        clip = r.json()['renderPayload']['animation']['clipName']
        assert 'Walk' in clip

    def test_demo_tryon_formal_occasion(self):
        r    = client.post('/api/demo/tryon', json={
            'shoulder_width_cm': 42, 'chest_cm': 92,
            'waist_cm': 72, 'height_cm': 168,
            'styles': 'formal', 'occasion': 'interview',
        })
        data = r.json()
        assert data['success'] is True


# ── Full Try-On with Selfie ───────────────────────────────────────────────────

class TestFullTryOnEndpoint:

    def test_tryon_with_synthetic_selfie(self):
        img_bytes = _synthetic_selfie_bytes()
        r = client.post('/api/tryon', data={
            'shoulder_width_cm': '42',
            'chest_cm': '92', 'waist_cm': '72', 'height_cm': '168',
            'styles': 'smart_casual,casual', 'occasion': 'casual',
            'animation_key': 'idle', 'top_k': '3',
        }, files={'selfie': ('selfie.jpg', img_bytes, 'image/jpeg')})
        assert r.status_code == 200
        assert r.json()['success'] is True

    def test_tryon_processing_time_recorded(self):
        img_bytes = _synthetic_selfie_bytes()
        r = client.post('/api/tryon', data={
            'shoulder_width_cm': '42',
            'chest_cm': '92', 'waist_cm': '72', 'height_cm': '168',
        }, files={'selfie': ('selfie.jpg', img_bytes, 'image/jpeg')})
        assert r.json()['processingTimeMs'] > 0


# ── Model Status Endpoint ─────────────────────────────────────────────────────

class TestModelStatusEndpoint:

    def test_model_status_returns_path(self):
        r    = client.get('/api/model/status')
        assert r.status_code == 200
        data = r.json()
        assert 'modelPath'   in data
        assert 'exists'      in data
        assert 'fileSizeMB'  in data
