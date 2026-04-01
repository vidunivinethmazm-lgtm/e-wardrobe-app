"""
eWardrobeAI Mobile — Model Trainer with Terminal Accuracy Output

Trains both face CNN models on training.csv and prints a detailed
accuracy comparison table to the terminal.

Run: python -m src.mobile.trainer
"""

import os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)

from src.mobile.face_models import (
    FaceModelRunner, DeepFaceCNN, LightFaceCNN,
    load_csv_data, DEEP_PATH, LIGHT_PATH, MODEL_DIR, IMG_SIZE,
    KP_NAMES, DEVICE
)

# ── Terminal Formatting ────────────────────────────────────────────────────

W = 64   # line width

def bar(label='', char='─'): print(f"  {char*W}")
def sep(): bar()
def title(t): print(f"\n  {'═'*W}\n  {t}\n  {'═'*W}")
def row(label, val, width=38):
    print(f"  {label:<{width}} {val}")
def blank(): print()


def accuracy_table(results):
    """Print a formatted accuracy comparison table."""
    blank()
    bar('', '─')
    print(f"  {'Model':<22} {'MAE(px)':>8} {'RMSE(px)':>9} {'Params':>10} "
          f"{'Speed':>9}  {'Rating'}")
    bar('', '─')

    best_mae = min(r.mae_px for r in results)
    for r in results:
        tag     = '← Best accuracy' if r.mae_px == best_mae else ''
        rating  = _star_rating(r.mae_px)
        print(f"  {r.model_name:<22} {r.mae_px:>7.3f}  {r.rmse_px:>8.3f}  "
              f"{r.param_count:>9,}  {r.inference_ms:>6.2f}ms  {rating}  {tag}")
    bar('', '─')
    blank()


def per_keypoint_table(results):
    """Print per-keypoint MAE for both models side by side."""
    blank()
    print(f"  Per-Keypoint MAE (pixels)")
    bar('', '─')
    print(f"  {'Keypoint':<30} {results[0].model_name:>12} {results[1].model_name:>14}")
    bar('', '─')
    for kp in KP_NAMES:
        v1 = results[0].per_kp_mae.get(kp, 0)
        v2 = results[1].per_kp_mae.get(kp, 0)
        winner = '◀' if v1 < v2 else ('▶' if v2 < v1 else '=')
        print(f"  {kp.replace('_',' '):<30} {v1:>11.3f}  {v2:>12.3f}  {winner}")
    bar('', '─')


def body_results_table(results):
    """Print body calibration model results."""
    blank()
    bar('', '─')
    print(f"  {'Model':<26} {'Conf':>6} {'Landmarks':>10} {'Speed':>9}  {'Detected'}")
    bar('', '─')
    for r in results:
        det  = '✓ Yes' if r.detected else '✗ No'
        lm   = f"{r.landmarks_found}/{r.total_landmarks}"
        print(f"  {r.model_name:<26} {r.confidence:>5.1%}  {lm:>10}  "
              f"{r.inference_ms:>6.2f}ms  {det}")
    bar('', '─')
    blank()

    # Body proportion comparison
    detected = [r for r in results if r.detected]
    if len(detected) >= 2:
        print(f"  Body Proportion Estimates")
        bar('', '─')
        print(f"  {'Measurement':<22} {detected[0].model_name:>18} "
              f"{detected[1].model_name:>20}")
        bar('', '─')
        fields = [
            ('Shoulder width', 'shoulder_cm', 'cm'),
            ('Height estimate','height_cm',   'cm'),
            ('Hip estimate',   'hip_cm',      'cm'),
            ('Body type',      'body_type',   ''),
            ('Standard size',  'standard_size',''),
        ]
        for label, attr, unit in fields:
            v1 = getattr(detected[0], attr)
            v2 = getattr(detected[1], attr)
            s1 = f"{v1}{unit}" if unit else str(v1)
            s2 = f"{v2}{unit}" if unit else str(v2)
            print(f"  {label:<22} {s1:>18}  {s2:>20}")
        bar('', '─')


def _star_rating(mae_px: float) -> str:
    if mae_px <= 3.5: return '★★★★★'
    if mae_px <= 5.0: return '★★★★☆'
    if mae_px <= 7.0: return '★★★☆☆'
    if mae_px <= 9.0: return '★★☆☆☆'
    return '★☆☆☆☆'


# ── Main Training Entry Point ──────────────────────────────────────────────

def train_and_evaluate(epochs_deep=80, epochs_light=50):
    os.makedirs(MODEL_DIR, exist_ok=True)

    title("eWardrobeAI Mobile — Model Training & Accuracy Evaluation")
    print(f"  Dataset    : training.csv (Facial Keypoints Detection)")
    print(f"  Device     : {DEVICE}")
    print(f"  Models     : DeepFaceCNN (5-block) + LightFaceCNN (3-block)")
    blank()

    runner = FaceModelRunner()

    # ── Load data ──────────────────────────────────────────────────────────
    bar('', '─')
    print("  Loading training.csv …")
    t0 = time.time()
    X, y = load_csv_data()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    runner._val_X, runner._val_y = X_val, y_val
    print(f"  Samples    : {len(X):,} total  |  {len(X_tr):,} train  |  {len(X_val):,} val")
    print(f"  Image size : {IMG_SIZE}×{IMG_SIZE} grayscale")
    print(f"  Keypoints  : 15 landmarks (30 coordinates)")
    bar('', '─')

    # ── Train DeepFaceCNN ──────────────────────────────────────────────────
    blank()
    print(f"  ┌─ Model 1: DeepFaceCNN ({'exists — loading' if os.path.exists(DEEP_PATH) else f'training {epochs_deep} epochs'})")
    print(f"  │  5 conv blocks · BatchNorm · Dropout(0.4/0.3) · Sigmoid output")
    print(f"  │  Params: {DeepFaceCNN().param_count:,}")
    sep()

    if os.path.exists(DEEP_PATH):
        runner.deep.load_state_dict(__import__('torch').load(DEEP_PATH, map_location=DEVICE))
        print(f"  Loaded existing weights from {DEEP_PATH}")
    else:
        t1 = time.time()
        from src.mobile.face_models import _train
        _train(runner.deep, DEEP_PATH, X_tr, y_tr, X_val, y_val, epochs_deep)
        print(f"  Training complete in {time.time()-t1:.1f}s")

    # ── Train LightFaceCNN ─────────────────────────────────────────────────
    blank()
    print(f"  ┌─ Model 2: LightFaceCNN ({'exists — loading' if os.path.exists(LIGHT_PATH) else f'training {epochs_light} epochs'})")
    print(f"  │  3 conv blocks · BatchNorm · Dropout(0.4/0.3) · Sigmoid output")
    print(f"  │  Params: {LightFaceCNN().param_count:,}")
    sep()

    if os.path.exists(LIGHT_PATH):
        runner.light.load_state_dict(__import__('torch').load(LIGHT_PATH, map_location=DEVICE))
        print(f"  Loaded existing weights from {LIGHT_PATH}")
    else:
        t1 = time.time()
        from src.mobile.face_models import _train
        _train(runner.light, LIGHT_PATH, X_tr, y_tr, X_val, y_val, epochs_light)
        print(f"  Training complete in {time.time()-t1:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────
    blank()
    bar('', '═')
    print("  ACCURACY RESULTS — Face Detection (Facial Keypoint Regression)")
    bar('', '═')

    print("  Evaluating on validation split …")
    results = runner.evaluate()

    # Main accuracy table
    accuracy_table(results)

    # Key findings
    best = min(results, key=lambda r: r.mae_px)
    fast = min(results, key=lambda r: r.inference_ms)
    print(f"  🏆  Best Accuracy : {best.model_name}  (MAE = {best.mae_px:.3f} px)")
    print(f"  ⚡  Fastest       : {fast.model_name}  ({fast.inference_ms:.2f} ms/image)")
    blank()

    # Interpretation
    bar('', '─')
    print("  Accuracy Interpretation (96×96 image):")
    for r in results:
        pct = round(r.mae_px / IMG_SIZE * 100, 2)
        print(f"  {r.model_name:<22} MAE={r.mae_px:.3f}px = {pct:.2f}% of image width  {_star_rating(r.mae_px)}")
    bar('', '─')

    # Per-keypoint breakdown
    blank()
    bar('', '═')
    print("  PER-KEYPOINT MAE BREAKDOWN")
    bar('', '═')
    per_keypoint_table(results)

    # Agreement between models
    blank()
    bar('', '─')
    agree_count = sum(
        1 for kp in KP_NAMES
        if abs(results[0].per_kp_mae.get(kp,0) - results[1].per_kp_mae.get(kp,0)) < 2.0
    )
    print(f"  Model Agreement: {agree_count}/{len(KP_NAMES)} keypoints within 2px of each other  "
          f"({agree_count/len(KP_NAMES)*100:.0f}%)")
    bar('', '─')

    title("TRAINING & EVALUATION COMPLETE")
    print(f"  Total time  : {time.time()-t0:.1f}s")
    print(f"  Model files : {DEEP_PATH}")
    print(f"              : {LIGHT_PATH}")
    print(f"\n  Next: python -m streamlit run mobile_app.py")
    blank()

    return results


if __name__ == '__main__':
    train_and_evaluate()
