"""
Tests — Stage 1: Body Calibration
Covers: range validation, cross-field checks, sizing profile, avatar scale params
"""

import pytest
from src.body_calibration import (
    BodyCalibrator, BodyMeasurements,
    AvatarScaleParams, SizingProfile,
    _classify_body_type,
)

cal = BodyCalibrator()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_measurements(**overrides) -> BodyMeasurements:
    base = dict(shoulder_width_cm=42, chest_cm=92, waist_cm=72, height_cm=168)
    base.update(overrides)
    return BodyMeasurements(**base)


# ── Valid measurement sets ────────────────────────────────────────────────────

class TestValidMeasurements:

    def test_default_demo_values_pass(self):
        m = _valid_measurements()
        r = cal.validate(m)
        assert r.is_valid

    def test_minimum_bounds_pass(self):
        m = _valid_measurements(
            shoulder_width_cm=30, chest_cm=60, waist_cm=50, height_cm=120
        )
        r = cal.validate(m)
        assert r.is_valid

    def test_maximum_bounds_pass(self):
        m = _valid_measurements(
            shoulder_width_cm=65, chest_cm=160, waist_cm=150, height_cm=230
        )
        r = cal.validate(m)
        assert r.is_valid

    def test_optional_fields_estimated(self):
        m = BodyMeasurements(shoulder_width_cm=42, chest_cm=92,
                             waist_cm=72, height_cm=168)
        assert m.hip_cm    == pytest.approx(72 + 25, abs=0.1)
        assert m.inseam_cm == pytest.approx(168 * 0.47, abs=0.1)

    def test_optional_fields_respected_when_provided(self):
        m = BodyMeasurements(shoulder_width_cm=42, chest_cm=92,
                             waist_cm=72, height_cm=168,
                             hip_cm=98, inseam_cm=80)
        assert m.hip_cm    == 98
        assert m.inseam_cm == 80


# ── Invalid measurements → hard errors ───────────────────────────────────────

class TestInvalidMeasurements:

    @pytest.mark.parametrize("field,value", [
        ("shoulder_width_cm", 20),   # too narrow
        ("shoulder_width_cm", 70),   # too wide
        ("chest_cm",          50),   # too small
        ("chest_cm",          200),  # too large
        ("waist_cm",          30),   # too small
        ("waist_cm",          200),  # too large
        ("height_cm",         90),   # too short
        ("height_cm",         250),  # too tall
    ])
    def test_out_of_range_produces_error(self, field, value):
        m = _valid_measurements(**{field: value})
        r = cal.validate(m)
        assert not r.is_valid
        assert any(field in e for e in r.errors)

    def test_multiple_invalid_fields_all_reported(self):
        m = _valid_measurements(chest_cm=200, height_cm=90)
        r = cal.validate(m)
        assert not r.is_valid
        assert len(r.errors) >= 2


# ── Warnings (valid but unusual) ──────────────────────────────────────────────

class TestWarnings:

    def test_waist_exceeds_chest_produces_warning(self):
        m = _valid_measurements(waist_cm=95, chest_cm=92)
        r = cal.validate(m)
        assert r.is_valid          # not an error, just a warning
        assert len(r.warnings) > 0

    def test_high_waist_height_ratio_warns(self):
        # waist/height > 0.63 should warn
        m = _valid_measurements(waist_cm=110, height_cm=168)
        r = cal.validate(m)
        assert r.is_valid
        assert any("ratio" in w.lower() for w in r.warnings)


# ── Sizing Profile ────────────────────────────────────────────────────────────

class TestSizingProfile:

    @pytest.mark.parametrize("chest,expected_size", [
        (78,  "XS"),
        (85,  "S"),
        (92,  "M"),
        (100, "L"),
        (108, "XL"),
        (118, "XXL"),
        (130, "XXXL"),
    ])
    def test_standard_size_labels(self, chest, expected_size):
        m = _valid_measurements(chest_cm=chest)
        p = cal.compute_sizing_profile(m)
        assert p.standard_size == expected_size

    def test_torso_length_computed(self):
        m = BodyMeasurements(42, 92, 72, 168, hip_cm=97, inseam_cm=79)
        p = cal.compute_sizing_profile(m)
        assert p.torso_length_cm == pytest.approx(168 - 79, abs=0.1)

    def test_waist_hip_ratio(self):
        m = BodyMeasurements(42, 92, 72, 168, hip_cm=90)
        p = cal.compute_sizing_profile(m)
        assert p.waist_hip_ratio == pytest.approx(72 / 90, rel=0.01)


# ── Body Type Classification ──────────────────────────────────────────────────

class TestBodyTypeClassification:

    def test_hourglass(self):
        # Large waist definition (chest+hip)/2 - waist > 9, balanced S/H
        result = _classify_body_type(96, 72, 96, 40)
        assert result == "hourglass"

    def test_rectangle(self):
        result = _classify_body_type(90, 85, 92, 40)
        assert result == "rectangle"

    def test_pear(self):
        # hip >> shoulder proxy
        result = _classify_body_type(88, 80, 115, 36)
        assert result == "pear"

    def test_inverted_triangle(self):
        # shoulder proxy >> hip
        result = _classify_body_type(100, 80, 80, 50)
        assert result == "inverted_triangle"


# ── Avatar Scale Parameters ───────────────────────────────────────────────────

class TestAvatarScaleParams:

    def test_scale_factors_between_clamp_bounds(self):
        m = _valid_measurements()
        s = cal.compute_avatar_scale(m)
        for val in s.to_dict().values():
            assert 0.70 <= val <= 1.40

    def test_base_avatar_dimensions_give_scale_1(self):
        # Base avatar dims: shoulder=40, chest=90, waist=70, height=170
        m = BodyMeasurements(40, 90, 70, 170)
        s = cal.compute_avatar_scale(m)
        assert s.global_scale_y   == pytest.approx(1.0, abs=0.01)
        assert s.chest_scale_x    == pytest.approx(1.0, abs=0.01)
        assert s.waist_scale_x    == pytest.approx(1.0, abs=0.01)
        assert s.shoulder_scale_x == pytest.approx(1.0, abs=0.01)

    def test_extreme_values_clamped(self):
        # Very tall person — scale should be clamped at 1.40
        m = BodyMeasurements(40, 90, 70, 250)
        s = cal.compute_avatar_scale(m)
        assert s.global_scale_y == pytest.approx(1.40, abs=0.01)

    def test_head_scale_from_inter_eye_distance(self):
        m = _valid_measurements()
        s = cal.compute_avatar_scale(m, inter_eye_dist_px=50, reference_ied_px=42)
        assert s.head_scale > 1.0

    def test_to_dict_has_all_keys(self):
        m = _valid_measurements()
        s = cal.compute_avatar_scale(m)
        d = s.to_dict()
        assert all(k in d for k in
                   ['globalY','shoulderX','chestX','waistX','hipX','legY','headScale'])

    def test_calibrate_convenience_method(self):
        val, scale, profile = cal.calibrate(42, 92, 72, 168)
        assert val.is_valid
        assert scale is not None
        assert profile is not None
        assert profile.standard_size == "M"

    def test_calibrate_returns_none_on_invalid(self):
        val, scale, profile = cal.calibrate(42, 200, 72, 168)  # chest=200 invalid
        assert not val.is_valid
        assert scale   is None
        assert profile is None
