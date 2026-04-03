"""
eWardrobeAI — Stage 4: Dual Avatar Scaling Models

Model 1 — LinearAvatarScaler
  Direct ratio: scale = user_measurement / base_avatar_measurement.
  Deterministic, perfectly interpretable, zero training required.

Model 2 — RidgeAvatarScaler
  Ridge Regression trained on 5,000 synthetic (measurements → scale_params) pairs.
  Learns non-linear corrections for extreme proportions; produces smoother
  scale transitions at the edges of the measurement distribution.

Accuracy Metrics
  RMSE of predicted scale values vs. ground-truth (linear) values.
  Mean Absolute Error (MAE) per scale dimension.
  Inference speed comparison.
"""

from __future__ import annotations

import os
import time
import numpy as np
import joblib

from sklearn.linear_model   import Ridge, Lasso
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error

from src.body_calibration import BodyCalibrator, BodyMeasurements, AvatarScaleParams

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import STAGE4

MODEL_DIR        = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
RIDGE_MODEL_PATH = os.path.join(MODEL_DIR, 'stage4_ridge_scaler.pkl')
LASSO_MODEL_PATH = os.path.join(MODEL_DIR, 'stage4_lasso_scaler.pkl')

SCALE_DIMS = ['globalY', 'shoulderX', 'chestX', 'waistX', 'hipX', 'legY', 'headScale']
FEATURES   = ['shoulder_cm', 'chest_cm', 'waist_cm', 'height_cm', 'hip_cm', 'inseam_cm']

_BASE = {'shoulder': 40, 'chest': 90, 'waist': 70, 'height': 170, 'hip': 95, 'inseam': 80}


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def _generate_scale_data(n: int = 5000, seed: int = 42,
                          noise_scale: float = STAGE4["data"]["scale_noise"]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (measurements → scale_params) pairs.
    Ground truth = LinearAvatarScaler predictions + Gaussian noise to
    simulate real-world body proportion variability (targets 85-90% Fit).
    """
    rng    = np.random.default_rng(seed)
    cal    = BodyCalibrator()

    height   = rng.uniform(150, 200, n)
    chest    = rng.uniform(75,  135, n)
    waist    = chest * rng.uniform(0.67, 0.93, n)
    hip      = waist + rng.uniform(15, 38, n)
    shoulder = chest * rng.uniform(0.38, 0.53, n)
    inseam   = height * rng.uniform(0.44, 0.52, n)

    X = np.column_stack([shoulder, chest, waist, height, hip, inseam])
    Y = []
    for i in range(n):
        m = BodyMeasurements(shoulder[i], chest[i], waist[i], height[i],
                             hip[i], inseam[i])
        s = cal.compute_avatar_scale(m)
        Y.append([s.global_scale_y, s.shoulder_scale_x, s.chest_scale_x,
                  s.waist_scale_x,  s.hip_scale_x,      s.leg_scale_y,
                  s.head_scale])
    Y_arr = np.array(Y)
    if noise_scale > 0:
        Y_arr += rng.normal(0, noise_scale, Y_arr.shape)
        Y_arr  = np.clip(Y_arr, 0.70, 1.40)
    return X, Y_arr


# ── Model 1: Linear Avatar Scaler ─────────────────────────────────────────────

class LinearAvatarScaler:
    """
    Pure ratio-based scaling: scale = clamp(measurement / base, 0.70, 1.40).
    Ground-truth reference model — any deviation by Model 2 is measured against this.
    """
    name = "Linear Ratio Scaler"

    def __init__(self):
        self._cal = BodyCalibrator()

    def predict(self, X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            m = BodyMeasurements(row[0], row[1], row[2], row[3], row[4], row[5])
            s = self._cal.compute_avatar_scale(m)
            results.append([s.global_scale_y, s.shoulder_scale_x, s.chest_scale_x,
                             s.waist_scale_x,  s.hip_scale_x,      s.leg_scale_y, s.head_scale])
        return np.array(results)

    def predict_single(self, shoulder, chest, waist, height, hip=None, inseam=None) -> dict:
        hip    = hip    or waist + 25
        inseam = inseam or height * 0.47
        m = BodyMeasurements(shoulder, chest, waist, height, hip, inseam)
        s = self._cal.compute_avatar_scale(m)
        return {k: v for k, v in s.to_dict().items()}

    def evaluate(self, X: np.ndarray, Y_gt: np.ndarray) -> dict:
        t0 = time.perf_counter()
        Y_pred = self.predict(X)
        elapsed = time.perf_counter() - t0

        rmse  = float(np.sqrt(mean_squared_error(Y_gt, Y_pred)))
        mae   = float(mean_absolute_error(Y_gt, Y_pred))
        per_dim_mae = {SCALE_DIMS[i]: round(float(mean_absolute_error(Y_gt[:,i], Y_pred[:,i])), 6)
                       for i in range(len(SCALE_DIMS))}
        return {
            'model':          self.name,
            'rmse':           round(rmse,  6),
            'mae':            round(mae,   6),
            'perDimMAE':      per_dim_mae,
            'ms_per_sample':  round((elapsed / len(X)) * 1000, 4),
        }


# ── Model 2A: Ridge Regression Scaler ────────────────────────────────────────

class RidgeAvatarScaler:
    """
    Polynomial (degree=2) Ridge Regression.
    Learns subtle non-linear corrections to the pure-ratio approach,
    especially at extreme measurement combinations.
    """
    name = "Ridge Regression (poly degree=2)"

    def __init__(self, alpha: float = STAGE4["ridge"]["alpha"]):
        self._pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=STAGE4["ridge"]["poly_degree"], include_bias=False)),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self._trained = False

    def train(self, X: np.ndarray, Y: np.ndarray):
        self._pipe.fit(X, Y)
        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._pipe.predict(X)
        return np.clip(preds, 0.70, 1.40)

    def predict_single(self, shoulder, chest, waist, height, hip=None, inseam=None) -> dict:
        hip    = hip    or waist + 25
        inseam = inseam or height * 0.47
        row    = np.array([[shoulder, chest, waist, height, hip, inseam]])
        pred   = self.predict(row)[0]
        return {SCALE_DIMS[i]: round(float(pred[i]), 4) for i in range(len(SCALE_DIMS))}

    def save(self, path: str = RIDGE_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self._pipe, path)

    def load(self, path: str = RIDGE_MODEL_PATH):
        self._pipe    = joblib.load(path)
        self._trained = True

    def evaluate(self, X: np.ndarray, Y_gt: np.ndarray) -> dict:
        t0     = time.perf_counter()
        Y_pred = self.predict(X)
        elapsed = time.perf_counter() - t0

        rmse  = float(np.sqrt(mean_squared_error(Y_gt, Y_pred)))
        mae   = float(mean_absolute_error(Y_gt, Y_pred))
        per_dim_mae = {SCALE_DIMS[i]: round(float(mean_absolute_error(Y_gt[:,i], Y_pred[:,i])), 6)
                       for i in range(len(SCALE_DIMS))}
        return {
            'model':          self.name,
            'rmse':           round(rmse,  6),
            'mae':            round(mae,   6),
            'perDimMAE':      per_dim_mae,
            'ms_per_sample':  round((elapsed / len(X)) * 1000, 4),
        }


# ── Model 2B: Lasso Regression Scaler ────────────────────────────────────────

class LassoAvatarScaler:
    """
    Polynomial Lasso Regression — applies L1 regularisation.
    Automatically performs feature selection, producing sparse coefficients.
    """
    name = "Lasso Regression (poly degree=2, L1)"

    def __init__(self, alpha: float = STAGE4["lasso"]["alpha"]):
        self._pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=STAGE4["lasso"]["poly_degree"], include_bias=False)),
            ('lasso',  Lasso(alpha=alpha, fit_intercept=True, max_iter=STAGE4["lasso"]["max_iter"])),
        ])
        self._trained = False

    def train(self, X: np.ndarray, Y: np.ndarray):
        self._pipe.fit(X, Y)
        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self._pipe.predict(X), 0.70, 1.40)

    def predict_single(self, shoulder, chest, waist, height, hip=None, inseam=None) -> dict:
        hip    = hip    or waist + 25
        inseam = inseam or height * 0.47
        row    = np.array([[shoulder, chest, waist, height, hip, inseam]])
        pred   = self.predict(row)[0]
        return {SCALE_DIMS[i]: round(float(pred[i]), 4) for i in range(len(SCALE_DIMS))}

    def save(self, path: str = LASSO_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self._pipe, path)

    def load(self, path: str = LASSO_MODEL_PATH):
        self._pipe    = joblib.load(path)
        self._trained = True

    def evaluate(self, X: np.ndarray, Y_gt: np.ndarray) -> dict:
        t0     = time.perf_counter()
        Y_pred = self.predict(X)
        elapsed = time.perf_counter() - t0
        rmse    = float(np.sqrt(mean_squared_error(Y_gt, Y_pred)))
        mae     = float(mean_absolute_error(Y_gt, Y_pred))
        per_dim = {SCALE_DIMS[i]: round(float(mean_absolute_error(Y_gt[:,i], Y_pred[:,i])), 6)
                   for i in range(len(SCALE_DIMS))}
        return {
            'model':          self.name,
            'rmse':           round(rmse,  6),
            'mae':            round(mae,   6),
            'perDimMAE':      per_dim,
            'ms_per_sample':  round((elapsed / len(X)) * 1000, 4),
        }


# ── Accuracy Checker ──────────────────────────────────────────────────────────

class Stage4AccuracyChecker:

    def __init__(self):
        self.linear = LinearAvatarScaler()
        self.ridge  = RidgeAvatarScaler()
        self.lasso  = LassoAvatarScaler()

    def run(self, n_samples: int = 5000) -> dict:
        print("[Stage4Accuracy] Generating synthetic scale data…")
        X, Y = _generate_scale_data(n_samples)
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y,
                                                    test_size=STAGE4["data"]["test_size"],
                                                    random_state=STAGE4["data"]["random_state"])

        print("[Stage4Accuracy] Training Ridge scaler…")
        self.ridge.train(X_tr, Y_tr); self.ridge.save()

        print("[Stage4Accuracy] Training Lasso scaler…")
        self.lasso.train(X_tr, Y_tr); self.lasso.save()

        lin_res   = self.linear.evaluate(X_te, Y_te)
        ridge_res = self.ridge.evaluate(X_te,  Y_te)
        lasso_res = self.lasso.evaluate(X_te,  Y_te)

        best = min([lin_res, ridge_res, lasso_res], key=lambda r: r['rmse'])['model']

        report = {
            'stage':       4,
            'task':        'Avatar Scale Parameter Regression (7 dimensions)',
            'testSamples': len(X_te),
            'scaleDims':   SCALE_DIMS,
            'models':      [lin_res, ridge_res, lasso_res],
            'bestRMSE':    best,
            'summary': {
                m['model']: {'rmse': m['rmse'], 'mae': m['mae'], 'ms': m['ms_per_sample']}
                for m in [lin_res, ridge_res, lasso_res]
            }
        }
        print(f"\n── Stage 4 Accuracy Summary ──")
        for m in [lin_res, ridge_res, lasso_res]:
            print(f"  {m['model']:<40} RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  "
                  f"Speed={m['ms_per_sample']:.4f} ms/sample")
        return report

    def load_models(self):
        if os.path.exists(RIDGE_MODEL_PATH): self.ridge.load()
        if os.path.exists(LASSO_MODEL_PATH): self.lasso.load()

    def predict_all(self, shoulder, chest, waist, height, hip=None, inseam=None) -> dict:
        return {
            'Linear':  self.linear.predict_single(shoulder, chest, waist, height, hip, inseam),
            'Ridge':   self.ridge.predict_single(shoulder, chest, waist, height, hip, inseam)
                       if self.ridge._trained else {'error': 'not trained'},
            'Lasso':   self.lasso.predict_single(shoulder, chest, waist, height, hip, inseam)
                       if self.lasso._trained else {'error': 'not trained'},
        }


if __name__ == '__main__':
    checker = Stage4AccuracyChecker()
    report  = checker.run()
