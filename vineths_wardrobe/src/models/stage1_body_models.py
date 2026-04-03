"""
eWardrobeAI — Stage 1: Dual Body Calibration Models

Model 1 — RuleBasedBodyModel
  Pure deterministic validator + classifier using anatomical bounds and
  rule-based proportionality ratios. No training required.
  Metrics: validation accuracy, error rate on boundary cases.

Model 2 — MLBodyTypeModel
  Random Forest classifier trained on synthetically generated body
  measurement data. Predicts body type (4-class) and standard size (7-class)
  from [shoulder, chest, waist, height, hip, inseam] features.
  Metrics: classification accuracy, F1-score (weighted), confusion matrix.

Accuracy Checker
  Generates 2,000 synthetic samples with known ground-truth labels,
  evaluates both models and returns a side-by-side comparison report.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import STAGE1

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
RF_MODEL_PATH  = os.path.join(MODEL_DIR, 'stage1_rf_bodytype.pkl')
GB_MODEL_PATH  = os.path.join(MODEL_DIR, 'stage1_gb_bodytype.pkl')
LE_PATH        = os.path.join(MODEL_DIR, 'stage1_label_encoder.pkl')

BODY_TYPES     = ['hourglass', 'inverted_triangle', 'pear', 'rectangle']
SIZE_LABELS    = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']
SIZE_THRESHOLDS = [82, 88, 96, 104, 112, 124, float('inf')]

FEATURES = ['shoulder_cm', 'chest_cm', 'waist_cm', 'height_cm', 'hip_cm', 'inseam_cm']


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def _generate_synthetic_data(n: int = STAGE1["data"]["n_samples"],
                              seed: int = STAGE1["data"]["random_state"],
                              noise_rate: float = STAGE1["data"]["label_noise"]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic body measurement data with deterministic ground-truth labels.
    Used to train and evaluate the ML models.

    Returns (X, y_bodytype, y_size) where:
      X           : (n, 6) float array of measurements
      y_bodytype  : (n,) int array of body type labels [0-3]
      y_size      : (n,) int array of size labels [0-6]
    """
    rng = np.random.default_rng(seed)

    height    = rng.uniform(150, 195, n)
    chest     = rng.uniform(75, 130, n)
    waist     = chest * rng.uniform(0.68, 0.92, n)
    hip       = waist + rng.uniform(15, 38, n)
    shoulder  = chest * rng.uniform(0.38, 0.52, n)
    inseam    = height * rng.uniform(0.44, 0.52, n)

    X = np.column_stack([shoulder, chest, waist, height, hip, inseam])

    # Ground-truth body type from the same rules as _classify_body_type()
    y_bt = []
    for i in range(n):
        wDef = (chest[i] + hip[i]) / 2 - waist[i]
        sHR  = hip[i] / (shoulder[i] * 2.3)
        if wDef > 9 and abs(sHR - 1) < 0.08:
            bt = 'hourglass'
        elif sHR < 0.87:
            bt = 'inverted_triangle'
        elif sHR > 1.13:
            bt = 'pear'
        else:
            bt = 'rectangle'
        y_bt.append(BODY_TYPES.index(bt))
    y_bt = np.array(y_bt)

    # Ground-truth size from chest
    y_sz = np.array([
        next(i for i, t in enumerate(SIZE_THRESHOLDS) if chest[j] <= t)
        for j in range(n)
    ])

    # Add label noise to simulate real-world measurement ambiguity
    if noise_rate > 0:
        noise_mask = rng.random(n) < noise_rate
        y_bt[noise_mask] = rng.integers(0, len(BODY_TYPES), int(noise_mask.sum()))
        sz_noise_mask = rng.random(n) < noise_rate
        y_sz[sz_noise_mask] = rng.integers(0, len(SIZE_LABELS), int(sz_noise_mask.sum()))

    return X, y_bt, y_sz


# ── Model 1: Rule-Based Body Model ────────────────────────────────────────────

class RuleBasedBodyModel:
    """
    Deterministic rule-based classifier.
    No training required — uses anatomical proportion rules directly.

    Accuracy is evaluated by counting how often the rules agree with
    the deterministic synthetic ground-truth (should be ~100 % since
    both use the same logic — serves as a correctness sanity check).
    """
    name = "Rule-Based Classifier"

    def predict_body_type(self, shoulder: float, chest: float,
                           waist: float, hip: float) -> str:
        wDef = (chest + hip) / 2 - waist
        sHR  = hip / (shoulder * 2.3)
        if wDef > 9 and abs(sHR - 1) < 0.08: return 'hourglass'
        if sHR < 0.87:  return 'inverted_triangle'
        if sHR > 1.13:  return 'pear'
        return 'rectangle'

    def predict_size(self, chest: float) -> str:
        for threshold, label in zip(SIZE_THRESHOLDS, SIZE_LABELS):
            if chest <= threshold:
                return label
        return 'XXXL'

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bt_preds = np.array([
            BODY_TYPES.index(self.predict_body_type(*row[:3], row[4]))
            for row in X
        ])
        sz_preds = np.array([self.predict_size(row[1]) for row in X])
        sz_enc   = np.array([SIZE_LABELS.index(s) for s in sz_preds])
        return bt_preds, sz_enc

    def evaluate(self, X: np.ndarray, y_bt: np.ndarray, y_sz: np.ndarray) -> dict:
        bt_pred, sz_pred = self.predict_batch(X)
        return {
            'model':              self.name,
            'bodyType': {
                'accuracy':       round(accuracy_score(y_bt, bt_pred), 4),
                'f1_weighted':    round(f1_score(y_bt, bt_pred, average='weighted'), 4),
                'report':         classification_report(y_bt, bt_pred,
                                    target_names=BODY_TYPES, output_dict=True),
            },
            'standardSize': {
                'accuracy':       round(accuracy_score(y_sz, sz_pred), 4),
                'f1_weighted':    round(f1_score(y_sz, sz_pred, average='weighted'), 4),
            },
        }


# ── Model 2A: Random Forest Body Model ───────────────────────────────────────

class RandomForestBodyModel:
    """
    Random Forest classifier for body type prediction.
    Trained on 4,000 synthetic body measurement samples.
    Uses 6 features: [shoulder, chest, waist, height, hip, inseam].
    """
    name = "Random Forest (n_estimators=200)"

    def __init__(self):
        cfg_bt = STAGE1["random_forest"]["body_type"]
        cfg_sz = STAGE1["random_forest"]["size"]
        self.rf_bt = RandomForestClassifier(**cfg_bt)
        self.rf_sz = RandomForestClassifier(**cfg_sz)
        self._trained = False

    def train(self, X: np.ndarray, y_bt: np.ndarray, y_sz: np.ndarray):
        self.rf_bt.fit(X, y_bt)
        self.rf_sz.fit(X, y_sz)
        self._trained = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.rf_bt.predict(X), self.rf_sz.predict(X)

    def predict_single(self, shoulder, chest, waist, height, hip, inseam) -> dict:
        row   = np.array([[shoulder, chest, waist, height, hip, inseam]])
        bt, sz = self.predict(row)
        proba  = self.rf_bt.predict_proba(row)[0]
        return {
            'bodyType':         BODY_TYPES[bt[0]],
            'bodyTypeProba':    {BODY_TYPES[i]: round(float(p), 3)
                                 for i, p in enumerate(proba)},
            'standardSize':     SIZE_LABELS[sz[0]],
        }

    def save(self, path: str = RF_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({'bt': self.rf_bt, 'sz': self.rf_sz}, path)

    def load(self, path: str = RF_MODEL_PATH):
        d = joblib.load(path)
        self.rf_bt, self.rf_sz = d['bt'], d['sz']
        self._trained = True

    def evaluate(self, X: np.ndarray, y_bt: np.ndarray, y_sz: np.ndarray) -> dict:
        bt_pred, sz_pred = self.predict(X)
        cv_scores = cross_val_score(self.rf_bt, X, y_bt, cv=5, scoring='accuracy')
        fi = dict(zip(FEATURES, self.rf_bt.feature_importances_.tolist()))
        return {
            'model':              self.name,
            'bodyType': {
                'accuracy':       round(accuracy_score(y_bt, bt_pred), 4),
                'f1_weighted':    round(f1_score(y_bt, bt_pred, average='weighted'), 4),
                'cv5_mean':       round(float(cv_scores.mean()), 4),
                'cv5_std':        round(float(cv_scores.std()),  4),
                'report':         classification_report(y_bt, bt_pred,
                                    target_names=BODY_TYPES, output_dict=True),
                'featureImportance': {k: round(v, 4) for k, v in fi.items()},
            },
            'standardSize': {
                'accuracy':       round(accuracy_score(y_sz, sz_pred), 4),
                'f1_weighted':    round(f1_score(y_sz, sz_pred, average='weighted'), 4),
            },
        }


# ── Model 2B: Gradient Boosting Body Model ────────────────────────────────────

class GradientBoostingBodyModel:
    """
    Gradient Boosting classifier for body type prediction.
    Typically achieves higher accuracy than Random Forest on tabular data.
    """
    name = "Gradient Boosting (n_estimators=150)"

    def __init__(self):
        cfg_bt = STAGE1["gradient_boosting"]["body_type"]
        cfg_sz = STAGE1["gradient_boosting"]["size"]
        self.gb_bt = GradientBoostingClassifier(**cfg_bt)
        self.gb_sz = GradientBoostingClassifier(**cfg_sz)
        self._trained = False

    def train(self, X: np.ndarray, y_bt: np.ndarray, y_sz: np.ndarray):
        self.gb_bt.fit(X, y_bt)
        self.gb_sz.fit(X, y_sz)
        self._trained = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gb_bt.predict(X), self.gb_sz.predict(X)

    def predict_single(self, shoulder, chest, waist, height, hip, inseam) -> dict:
        row   = np.array([[shoulder, chest, waist, height, hip, inseam]])
        bt, sz = self.predict(row)
        proba  = self.gb_bt.predict_proba(row)[0]
        return {
            'bodyType':         BODY_TYPES[bt[0]],
            'bodyTypeProba':    {BODY_TYPES[i]: round(float(p), 3)
                                 for i, p in enumerate(proba)},
            'standardSize':     SIZE_LABELS[sz[0]],
        }

    def save(self, path: str = GB_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({'bt': self.gb_bt, 'sz': self.gb_sz}, path)

    def load(self, path: str = GB_MODEL_PATH):
        d = joblib.load(path)
        self.gb_bt, self.gb_sz = d['bt'], d['sz']
        self._trained = True

    def evaluate(self, X: np.ndarray, y_bt: np.ndarray, y_sz: np.ndarray) -> dict:
        bt_pred, sz_pred = self.predict(X)
        cv_scores = cross_val_score(self.gb_bt, X, y_bt, cv=5, scoring='accuracy')
        fi = dict(zip(FEATURES, self.gb_bt.feature_importances_.tolist()))
        return {
            'model':              self.name,
            'bodyType': {
                'accuracy':       round(accuracy_score(y_bt, bt_pred), 4),
                'f1_weighted':    round(f1_score(y_bt, bt_pred, average='weighted'), 4),
                'cv5_mean':       round(float(cv_scores.mean()), 4),
                'cv5_std':        round(float(cv_scores.std()),  4),
                'report':         classification_report(y_bt, bt_pred,
                                    target_names=BODY_TYPES, output_dict=True),
                'featureImportance': {k: round(v, 4) for k, v in fi.items()},
            },
            'standardSize': {
                'accuracy':       round(accuracy_score(y_sz, sz_pred), 4),
                'f1_weighted':    round(f1_score(y_sz, sz_pred, average='weighted'), 4),
            },
        }


# ── Accuracy Checker ──────────────────────────────────────────────────────────

class Stage1AccuracyChecker:
    """
    Trains both ML models and evaluates all three models (rule-based + 2 ML)
    on a held-out test set. Returns a structured comparison report.
    """

    def __init__(self):
        self.rule_model = RuleBasedBodyModel()
        self.rf_model   = RandomForestBodyModel()
        self.gb_model   = GradientBoostingBodyModel()

    def run(self, n_samples: int = 4000) -> dict:
        print("[Stage1Accuracy] Generating synthetic data…")
        X, y_bt, y_sz = _generate_synthetic_data(n_samples)
        X_train, X_test, ybt_train, ybt_test, ysz_train, ysz_test = \
            train_test_split(X, y_bt, y_sz,
                             test_size=STAGE1["data"]["test_size"],
                             random_state=STAGE1["data"]["random_state"])

        print("[Stage1Accuracy] Training Random Forest…")
        self.rf_model.train(X_train, ybt_train, ysz_train)
        self.rf_model.save()

        print("[Stage1Accuracy] Training Gradient Boosting…")
        self.gb_model.train(X_train, ybt_train, ysz_train)
        self.gb_model.save()

        print("[Stage1Accuracy] Evaluating all three models…")
        rule_res = self.rule_model.evaluate(X_test, ybt_test, ysz_test)
        rf_res   = self.rf_model.evaluate(X_test, ybt_test, ysz_test)
        gb_res   = self.gb_model.evaluate(X_test, ybt_test, ysz_test)

        best_bt   = max([rule_res, rf_res, gb_res],
                        key=lambda r: r['bodyType']['accuracy'])['model']
        best_size = max([rule_res, rf_res, gb_res],
                        key=lambda r: r['standardSize']['accuracy'])['model']

        report = {
            'stage':        1,
            'task':         'Body Type Classification + Size Prediction',
            'testSamples':  len(X_test),
            'models':       [rule_res, rf_res, gb_res],
            'bestBodyType': best_bt,
            'bestSize':     best_size,
            'summary': {
                m['model']: {
                    'bodyTypeAccuracy': m['bodyType']['accuracy'],
                    'sizeAccuracy':     m['standardSize']['accuracy'],
                    'bodyTypeF1':       m['bodyType']['f1_weighted'],
                }
                for m in [rule_res, rf_res, gb_res]
            }
        }
        print(f"[Stage1Accuracy] Best body-type model: {best_bt}")
        print(f"[Stage1Accuracy] Best size model     : {best_size}")
        return report

    def load_models(self):
        if os.path.exists(RF_MODEL_PATH): self.rf_model.load()
        if os.path.exists(GB_MODEL_PATH): self.gb_model.load()

    def predict_all(self, shoulder, chest, waist, height, hip=None, inseam=None) -> dict:
        hip    = hip    or waist + 25
        inseam = inseam or height * 0.47
        row    = np.array([[shoulder, chest, waist, height, hip, inseam]])

        rule_bt = self.rule_model.predict_body_type(shoulder, chest, waist, hip)
        rule_sz = self.rule_model.predict_size(chest)

        results = {
            'Rule-Based':       {'bodyType': rule_bt, 'size': rule_sz},
            'Random Forest':    self.rf_model.predict_single(shoulder, chest, waist, height, hip, inseam)
                                if self.rf_model._trained else {'error': 'not trained'},
            'Gradient Boosting': self.gb_model.predict_single(shoulder, chest, waist, height, hip, inseam)
                                if self.gb_model._trained else {'error': 'not trained'},
        }
        # Agreement: all models agree on body type?
        bts = [v.get('bodyType', v.get('error')) for v in results.values()]
        results['agreement'] = len(set(bts)) == 1
        return results


if __name__ == '__main__':
    checker = Stage1AccuracyChecker()
    report  = checker.run()
    print("\n── Stage 1 Accuracy Summary ──")
    for m in report['models']:
        print(f"  {m['model']:<40} "
              f"BodyType Acc={m['bodyType']['accuracy']:.4f}  "
              f"Size Acc={m['standardSize']['accuracy']:.4f}")
    print(f"\n  Best Body Type Model : {report['bestBodyType']}")
    print(f"  Best Size Model     : {report['bestSize']}")
