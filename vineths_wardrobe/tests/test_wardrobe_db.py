"""
Tests — Stage 3: RaveehaOrganisationalDB + NisfaMatchmaking
Covers: garment exclusion, status lifecycle, scoring, recommendation output
"""

import pytest
from src.outfit_recommender import (
    RaveehaOrganisationalDB, NisfaMatchmaking,
    CleaningStatus, Availability, GarmentCategory, Style,
    GarmentRecord, OutfitRecommendation,
)


@pytest.fixture
def db():
    return RaveehaOrganisationalDB()


@pytest.fixture
def matchmaker(db):
    return NisfaMatchmaking(db)


# ── Garment Wearability ───────────────────────────────────────────────────────

class TestGarmentWearability:

    def test_clean_owned_is_wearable(self, db):
        rec = db.get('GAR-001')
        assert rec is not None
        assert rec.cleaning_status == CleaningStatus.CLEAN
        assert rec.availability    == Availability.OWNED
        assert rec.is_wearable     is True

    def test_dirty_garment_not_wearable(self, db):
        # GAR-003 Striped T-Shirt is seeded as Dirty
        rec = db.get('GAR-003')
        assert rec.cleaning_status == CleaningStatus.DIRTY
        assert rec.is_wearable     is False

    def test_in_laundry_not_wearable(self, db):
        # GAR-004 Graphic Hoodie is seeded as In Laundry
        rec = db.get('GAR-004')
        assert rec.cleaning_status == CleaningStatus.IN_LAUNDRY
        assert rec.is_wearable     is False

    def test_evening_gown_dirty_excluded(self, db):
        # GAR-011 Evening Gown is seeded as Dirty
        rec = db.get('GAR-011')
        assert rec.cleaning_status == CleaningStatus.DIRTY
        assert rec.is_wearable     is False


# ── Status Lifecycle ──────────────────────────────────────────────────────────

class TestStatusLifecycle:

    def test_clean_to_dirty(self, db):
        db.update_cleaning_status('GAR-001', CleaningStatus.DIRTY)
        assert db.get('GAR-001').cleaning_status == CleaningStatus.DIRTY
        assert db.get('GAR-001').is_wearable     is False

    def test_dirty_to_in_laundry(self, db):
        db.update_cleaning_status('GAR-003', CleaningStatus.IN_LAUNDRY)
        assert db.get('GAR-003').cleaning_status == CleaningStatus.IN_LAUNDRY

    def test_in_laundry_to_clean(self, db):
        db.update_cleaning_status('GAR-004', CleaningStatus.CLEAN)
        assert db.get('GAR-004').cleaning_status == CleaningStatus.CLEAN
        assert db.get('GAR-004').is_wearable     is True

    def test_update_nonexistent_returns_false(self, db):
        result = db.update_cleaning_status('GAR-999', CleaningStatus.CLEAN)
        assert result is False


# ── Query Available ───────────────────────────────────────────────────────────

class TestQueryAvailable:

    def test_query_excludes_dirty(self, db):
        results = db.query_available()
        garment_ids = [g.garment_id for g in results]
        assert 'GAR-003' not in garment_ids   # Dirty
        assert 'GAR-004' not in garment_ids   # In Laundry
        assert 'GAR-011' not in garment_ids   # Dirty

    def test_query_includes_clean(self, db):
        results = db.query_available()
        garment_ids = [g.garment_id for g in results]
        assert 'GAR-001' in garment_ids
        assert 'GAR-005' in garment_ids
        assert 'GAR-008' in garment_ids

    def test_query_by_style(self, db):
        results = db.query_available(style=Style.FORMAL)
        assert all(g.style == Style.FORMAL for g in results)

    def test_query_by_category(self, db):
        results = db.query_available(category=GarmentCategory.TOP)
        assert all(g.category == GarmentCategory.TOP for g in results)

    def test_query_by_size(self, db):
        results = db.query_available(size='M')
        for g in results:
            assert 'M' in g.sizes or 'all' in g.sizes

    def test_query_by_body_type(self, db):
        results = db.query_available(body_type='hourglass')
        for g in results:
            assert 'hourglass' in g.body_types or 'all' in g.body_types

    def test_wardrobe_summary_keys(self, db):
        summary = db.wardrobe_summary()
        assert 'Clean'      in summary
        assert 'Dirty'      in summary
        assert 'In Laundry' in summary

    def test_wardrobe_summary_total_matches_catalogue(self, db):
        summary = db.wardrobe_summary()
        total = sum(summary.values())
        assert total == len(db._records)


# ── NisfaMatchmaking ──────────────────────────────────────────────────────────

class TestNisfaMatchmaking:

    def test_returns_list_of_recommendations(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle', [Style.SMART], 'office')
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_all_items_in_recommendations_are_wearable(self, matchmaker, db):
        recs = matchmaker.recommend('M', 'hourglass', [Style.CASUAL, Style.SMART], 'casual')
        for rec in recs:
            for item in rec.items:
                assert item.is_wearable, (
                    f"Non-wearable item {item.garment_id} "
                    f"({item.cleaning_status.value}) in recommendation"
                )

    def test_scores_in_valid_range(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle', [Style.CASUAL], 'casual')
        for rec in recs:
            assert 0.0 <= rec.score <= 1.0

    def test_sorted_by_score_descending(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle',
                                    [Style.SMART, Style.CASUAL], 'office', top_k=5)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle',
                                    [Style.SMART, Style.CASUAL], 'casual', top_k=2)
        assert len(recs) <= 2

    def test_dirty_items_never_recommended(self, matchmaker):
        recs = matchmaker.recommend('S', 'rectangle',
                                    [Style.CASUAL], 'casual', top_k=10)
        for rec in recs:
            for item in rec.items:
                assert item.garment_id not in ('GAR-003', 'GAR-004', 'GAR-011')

    def test_formal_occasion_returns_formal_style(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle',
                                    [Style.FORMAL], 'interview', top_k=5)
        # At least one recommendation should have a formal item
        all_styles = [item.style for rec in recs for item in rec.items]
        assert Style.FORMAL in all_styles

    def test_recommendation_has_asset_paths(self, matchmaker):
        recs = matchmaker.recommend('M', 'rectangle', [Style.SMART], 'office')
        assert len(recs) > 0
        for rec in recs:
            assert len(rec.asset_paths) == len(rec.items)
            for path in rec.asset_paths:
                assert path.endswith('.glb') or path.endswith('.fbx')

    def test_no_recommendations_when_all_dirty(self, matchmaker, db):
        # Mark all clean garments as dirty
        for gid, rec in db._records.items():
            if rec.is_wearable:
                db.update_cleaning_status(gid, CleaningStatus.DIRTY)
        recs = matchmaker.recommend('M', 'rectangle', [Style.CASUAL], 'casual')
        assert recs == []
