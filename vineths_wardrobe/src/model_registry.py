"""
eWardrobeAI — Central Model Registry

Single entry point for:
  - Running accuracy checks for any/all pipeline stages
  - Switching the active model per stage
  - Loading persisted model weights
  - Returning structured accuracy reports for the API

Usage
-----
    from src.model_registry import ModelRegistry
    reg    = ModelRegistry()
    report = reg.evaluate_stage(2)          # evaluate Stage 2 CNNs
    report = reg.evaluate_all()             # all 4 stages
    result = reg.predict_stage1(42, 92, 72, 168)  # predict with both models
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class ModelRegistry:
    """
    Lazy-loads accuracy checkers per stage.
    Each checker owns both models for that stage.
    """

    def __init__(self):
        self._s1: Optional[object] = None
        self._s2: Optional[object] = None
        self._s3: Optional[object] = None
        self._s4: Optional[object] = None
        self._reports: dict[int, dict] = {}

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def _get_s1(self):
        if self._s1 is None:
            from src.models.stage1_body_models import Stage1AccuracyChecker
            self._s1 = Stage1AccuracyChecker()
            self._s1.load_models()
        return self._s1

    def evaluate_stage1(self, n_samples: int = 4000) -> dict:
        report = self._get_s1().run(n_samples)
        self._reports[1] = report
        return report

    def predict_stage1(self, shoulder, chest, waist, height,
                        hip=None, inseam=None) -> dict:
        return self._get_s1().predict_all(shoulder, chest, waist, height, hip, inseam)

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def _get_s2(self):
        if self._s2 is None:
            from src.models.stage2_face_models import Stage2AccuracyChecker
            self._s2 = Stage2AccuracyChecker()
            self._s2.load_models()
        return self._s2

    def evaluate_stage2(self, retrain_light: bool = False) -> dict:
        report = self._get_s2().run(retrain_light=retrain_light)
        self._reports[2] = report
        return report

    def predict_stage2(self, image_96x96) -> dict:
        return self._get_s2().predict_both(image_96x96)

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def _get_s3(self):
        if self._s3 is None:
            from src.models.stage3_outfit_models import Stage3AccuracyChecker
            self._s3 = Stage3AccuracyChecker()
        return self._s3

    def evaluate_stage3(self, top_k: int = 5) -> dict:
        report = self._get_s3().run(top_k)
        self._reports[3] = report
        return report

    def predict_stage3(self, size, body_type, styles, occasion, top_k=3) -> dict:
        return self._get_s3().compare_single(size, body_type, styles, occasion, top_k)

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def _get_s4(self):
        if self._s4 is None:
            from src.models.stage4_avatar_models import Stage4AccuracyChecker
            self._s4 = Stage4AccuracyChecker()
            self._s4.load_models()
        return self._s4

    def evaluate_stage4(self, n_samples: int = 5000) -> dict:
        report = self._get_s4().run(n_samples)
        self._reports[4] = report
        return report

    def predict_stage4(self, shoulder, chest, waist, height,
                        hip=None, inseam=None) -> dict:
        return self._get_s4().predict_all(shoulder, chest, waist, height, hip, inseam)

    # ── Evaluate All ──────────────────────────────────────────────────────────

    def evaluate_all(self) -> dict:
        """Run accuracy checks for all 4 stages. Returns combined report."""
        logger.info("[ModelRegistry] Running full accuracy evaluation…")
        results = {}
        for stage, fn in [
            (1, self.evaluate_stage1),
            (2, lambda: self.evaluate_stage2(retrain_light=False)),
            (3, self.evaluate_stage3),
            (4, self.evaluate_stage4),
        ]:
            try:
                results[f'stage{stage}'] = fn()
                logger.info(f"[ModelRegistry] Stage {stage} evaluation complete.")
            except Exception as e:
                logger.error(f"[ModelRegistry] Stage {stage} failed: {e}")
                results[f'stage{stage}'] = {'error': str(e), 'stage': stage}
        return results

    def get_cached_report(self, stage: int) -> Optional[dict]:
        return self._reports.get(stage)

    def get_model_catalogue(self) -> dict:
        """Returns a static catalogue of all models per stage (no evaluation)."""
        return {
            'stage1': {
                'task': 'Body Type Classification + Size Prediction',
                'models': [
                    {'name': 'Rule-Based Classifier',        'type': 'deterministic', 'params': 0,     'requiresTraining': False},
                    {'name': 'Random Forest (n=200)',         'type': 'sklearn-RF',   'params': 200,   'requiresTraining': True},
                    {'name': 'Gradient Boosting (n=150)',     'type': 'sklearn-GB',   'params': 150,   'requiresTraining': True},
                ],
                'metrics': ['accuracy', 'f1_weighted', 'cv5_accuracy'],
            },
            'stage2': {
                'task': 'Facial Keypoint Regression (15 landmarks, 96×96 images)',
                'models': [
                    {'name': 'DeepFaceCNN (5-block)',    'type': 'pytorch-CNN', 'params': '~4.5M', 'requiresTraining': True},
                    {'name': 'LightFaceCNN (3-block)',   'type': 'pytorch-CNN', 'params': '~1.2M', 'requiresTraining': True},
                ],
                'metrics': ['mae_px', 'rmse_px', 'ms_per_image', 'per_keypoint_mae'],
            },
            'stage3': {
                'task': 'Outfit Recommendation',
                'models': [
                    {'name': 'Heuristic Recommender',           'type': 'rule-based',   'params': 0,  'requiresTraining': False},
                    {'name': 'Content-Based TF-IDF Recommender','type': 'sklearn-TFIDF','params': '~',  'requiresTraining': True},
                ],
                'metrics': ['precision_at_k', 'coverage', 'avg_score', 'diversity', 'response_ms'],
            },
            'stage4': {
                'task': 'Avatar Scale Parameter Regression (7 dimensions)',
                'models': [
                    {'name': 'Linear Ratio Scaler',             'type': 'deterministic',       'params': 0, 'requiresTraining': False},
                    {'name': 'Ridge Regression (poly degree=2)','type': 'sklearn-Ridge',       'params': '~800', 'requiresTraining': True},
                    {'name': 'Lasso Regression (poly degree=2)','type': 'sklearn-Lasso-L1',    'params': '~800', 'requiresTraining': True},
                ],
                'metrics': ['rmse', 'mae', 'per_dim_mae', 'ms_per_sample'],
            },
        }
