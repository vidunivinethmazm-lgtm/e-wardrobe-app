"""
eWardrobeAI — Train All Models
Trains every ML model across all 4 pipeline stages in sequence.

Run: python scripts/train_all_models.py

Output files:
  models/face_keypoint_cnn.pth        Stage 2 — DeepFaceCNN
  models/face_keypoint_light.pth      Stage 2 — LightFaceCNN
  models/stage1_rf_bodytype.pkl       Stage 1 — Random Forest
  models/stage1_gb_bodytype.pkl       Stage 1 — Gradient Boosting
  models/stage4_ridge_scaler.pkl      Stage 4 — Ridge Regression
  models/stage4_lasso_scaler.pkl      Stage 4 — Lasso Regression
  models/training_curves.png          Stage 2 — Loss plots
  models/keypoint_predictions.png     Stage 2 — Prediction overlays
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)

RESULTS = {}

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def acc_tag(val_str):
    """Return a terminal tag based on accuracy value string like '87.50%'."""
    try:
        v = float(val_str.replace('%','').strip())
        if v >= 95:   return '🟢 EXCELLENT (>95%)'
        if v >= 85:   return '🟢 TARGET ✓ (85–95%)'
        if v >= 75:   return '🟡 WARN (75–85%)'
        return '🔴 BELOW TARGET (<75%)'
    except Exception:
        return ''

def done(stage, name, elapsed, metric_label, metric_val):
    tag = acc_tag(metric_val) if '%' in str(metric_val) else ''
    RESULTS[f'Stage {stage} — {name}'] = {
        'time_s': round(elapsed, 1),
        'metric': f'{metric_label} = {metric_val}'
    }
    print(f"  ✅  {name} trained in {elapsed:.1f}s  ({metric_label} = {metric_val})  {tag}")


# ── Stage 1: Body Calibration Models ─────────────────────────────────────────

section("STAGE 1 — Body Calibration: Random Forest + Gradient Boosting")
print("  Generating 4,000 synthetic body measurement samples…")

t0 = time.time()
from src.models.stage1_body_models import (
    Stage1AccuracyChecker, _generate_synthetic_data
)
from sklearn.model_selection import train_test_split

checker1 = Stage1AccuracyChecker()
X, y_bt, y_sz = _generate_synthetic_data(4000)
X_train, X_test, ybt_train, ybt_test, ysz_train, ysz_test = \
    train_test_split(X, y_bt, y_sz, test_size=0.25, random_state=42)

# Random Forest
print("\n  Training Random Forest (n_estimators=200)…")
t1 = time.time()
checker1.rf_model.train(X_train, ybt_train, ysz_train)
checker1.rf_model.save()
rf_res = checker1.rf_model.evaluate(X_test, ybt_test, ysz_test)
done(1, "Random Forest", time.time()-t1,
     "Body Type Accuracy", f"{rf_res['bodyType']['accuracy']*100:.2f}%")

# Gradient Boosting
print("\n  Training Gradient Boosting (n_estimators=150)…")
t1 = time.time()
checker1.gb_model.train(X_train, ybt_train, ysz_train)
checker1.gb_model.save()
gb_res = checker1.gb_model.evaluate(X_test, ybt_test, ysz_test)
done(1, "Gradient Boosting", time.time()-t1,
     "Body Type Accuracy", f"{gb_res['bodyType']['accuracy']*100:.2f}%")

# Rule-based (no training — just evaluate for comparison)
rule_res = checker1.rule_model.evaluate(X_test, ybt_test, ysz_test)
print(f"  ℹ️   Rule-Based (no training):  "
      f"Body Type Accuracy = {rule_res['bodyType']['accuracy']*100:.2f}%")

print(f"\n  Stage 1 Summary:")
print(f"    {'Model':<35} {'Body Type Acc':>15} {'Size Acc':>10}  {'Range'}")
print(f"    {'-'*75}")
for name, res in [('Rule-Based', rule_res), ('Random Forest', rf_res), ('Gradient Boosting', gb_res)]:
    bt_acc = res['bodyType']['accuracy'] * 100
    print(f"    {name:<35} {bt_acc:>14.2f}%  "
          f"{res['standardSize']['accuracy']*100:>9.2f}%  "
          f"{acc_tag(f'{bt_acc:.2f}%')}")


# ── Stage 2: Face Keypoint CNNs ───────────────────────────────────────────────

section("STAGE 2 — Face Keypoints: DeepFaceCNN + LightFaceCNN")

from src.models.stage2_face_models import (
    Stage2AccuracyChecker, _load_data, DEEP_MODEL_PATH, LIGHT_MODEL_PATH,
    EPOCHS_LIGHT, _train_model, FaceDataset
)
from sklearn.model_selection import train_test_split as tts2
from torch.utils.data import DataLoader
import torch

print("  Loading training.csv…")
X2, y2 = _load_data()
X2_tr, X2_val, y2_tr, y2_val = tts2(X2, y2, test_size=0.15, random_state=42)
print(f"  Train={len(X2_tr):,}  Val={len(X2_val):,}")

checker2 = Stage2AccuracyChecker()

# DeepFaceCNN
if os.path.exists(DEEP_MODEL_PATH):
    print(f"\n  DeepFaceCNN: loading existing weights from {DEEP_MODEL_PATH}")
    checker2.deep.load_state_dict(
        torch.load(DEEP_MODEL_PATH, map_location=checker2.deep.parameters().__next__().device
                   if hasattr(checker2.deep, 'parameters') else 'cpu')
    )
else:
    print("\n  Training DeepFaceCNN (5-block CNN)…  [this takes ~5-25 min]")
    from src.face_keypoint_model import train as train_deep
    t1 = time.time()
    train_deep(epochs=100, batch_size=64)
    elapsed = time.time() - t1
    checker2.deep.load_state_dict(
        torch.load(DEEP_MODEL_PATH, map_location='cpu')
    )
    print(f"  DeepFaceCNN trained in {elapsed:.1f}s")

# LightFaceCNN
print("\n  Training LightFaceCNN (3-block, lightweight)…")
t1 = time.time()
_train_model(checker2.light, LIGHT_MODEL_PATH, X2_tr, y2_tr, X2_val, y2_val,
             epochs=EPOCHS_LIGHT)
elapsed = time.time() - t1

# Evaluate both
deep_res  = checker2._evaluate(checker2.deep,  X2_val, y2_val, "DeepFaceCNN")
light_res = checker2._evaluate(checker2.light, X2_val, y2_val, "LightFaceCNN")

done(2, "LightFaceCNN",  elapsed,       "Val MAE", f"{light_res['mae_px']:.3f} px")
print(f"  ℹ️   DeepFaceCNN (loaded):  MAE = {deep_res['mae_px']:.3f} px")

print(f"\n  Stage 2 Summary:")
print(f"    {'Model':<35} {'MAE (px)':>10} {'RMSE (px)':>10} {'Params':>10} {'Speed':>10}")
print(f"    {'-'*77}")
for m in [deep_res, light_res]:
    print(f"    {m['model']:<35} {m['mae_px']:>10.3f} {m['rmse_px']:>10.3f} "
          f"{m['params']:>10,} {m['ms_per_img']:>9.2f}ms")


# ── Stage 3: Outfit Recommendation ───────────────────────────────────────────

section("STAGE 3 — Outfit Recommendation: Heuristic + TF-IDF Content-Based")
print("  Fitting TF-IDF vectoriser on wardrobe catalogue…")

t1 = time.time()
from src.models.stage3_outfit_models import Stage3AccuracyChecker
checker3 = Stage3AccuracyChecker()
# TF-IDF auto-fits in __init__ — evaluate both
report3  = checker3.run(top_k=5)
elapsed  = time.time() - t1

for m in report3['models']:
    done(3, m['model'], elapsed / 2,
         "Precision@5", f"{m['precision_at_k']*100:.1f}%")

print(f"\n  Stage 3 Summary:")
print(f"    {'Model':<40} {'Precision@5':>12} {'Coverage':>10} {'Avg Score':>10}  {'Range'}")
print(f"    {'-'*90}")
for m in report3['models']:
    p = m['precision_at_k'] * 100
    print(f"    {m['model']:<40} {p:>11.1f}%  "
          f"{m['coverage']*100:>9.1f}%  {m['avg_score']:>9.3f}  "
          f"{acc_tag(f'{p:.1f}%')}")


# ── Stage 4: Avatar Scale Models ─────────────────────────────────────────────

section("STAGE 4 — Avatar Scaling: Linear + Ridge + Lasso Regression")
print("  Generating 5,000 synthetic (measurement → scale) pairs…")

from src.models.stage4_avatar_models import (
    Stage4AccuracyChecker, _generate_scale_data
)
from sklearn.model_selection import train_test_split as tts4

checker4 = Stage4AccuracyChecker()
X4, Y4   = _generate_scale_data(5000)
X4_tr, X4_te, Y4_tr, Y4_te = tts4(X4, Y4, test_size=0.2, random_state=42)

# Ridge
print("\n  Training Ridge Regression (polynomial degree=2)…")
t1 = time.time()
checker4.ridge.train(X4_tr, Y4_tr)
checker4.ridge.save()
ridge_res = checker4.ridge.evaluate(X4_te, Y4_te)
done(4, "Ridge Regression", time.time()-t1, "RMSE", f"{ridge_res['rmse']:.6f}")

# Lasso
print("\n  Training Lasso Regression (polynomial degree=2, L1)…")
t1 = time.time()
checker4.lasso.train(X4_tr, Y4_tr)
checker4.lasso.save()
lasso_res = checker4.lasso.evaluate(X4_te, Y4_te)
done(4, "Lasso Regression",  time.time()-t1, "RMSE", f"{lasso_res['rmse']:.6f}")

# Linear baseline
linear_res = checker4.linear.evaluate(X4_te, Y4_te)
print(f"  ℹ️   Linear Scaler (no training):  RMSE = {linear_res['rmse']:.6f}")

print(f"\n  Stage 4 Summary:")
print(f"    {'Model':<42} {'RMSE':>12} {'MAE':>12}")
print(f"    {'-'*68}")
for name, res in [('Linear Scaler', linear_res), ('Ridge Regression', ridge_res), ('Lasso Regression', lasso_res)]:
    print(f"    {name:<42} {res['rmse']:>12.6f} {res['mae']:>12.6f}")


# ── Final Report ──────────────────────────────────────────────────────────────

section("TRAINING COMPLETE — All Models Summary")
total_time = time.time() - t0
print(f"\n  Total training time: {total_time:.1f}s  ({total_time/60:.1f} min)\n")
print(f"  {'Component':<40} {'Result'}")
print(f"  {'-'*70}")
for name, info in RESULTS.items():
    print(f"  {name:<40} {info['metric']}  ({info['time_s']}s)")

print(f"\n  Saved model files:")
model_files = [
    'stage1_rf_bodytype.pkl',
    'stage1_gb_bodytype.pkl',
    'face_keypoint_cnn.pth',
    'face_keypoint_light.pth',
    'stage4_ridge_scaler.pkl',
    'stage4_lasso_scaler.pkl',
]
models_dir = os.path.join(ROOT, 'models')
for f in model_files:
    path   = os.path.join(models_dir, f)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) / 1024 if exists else 0
    print(f"  {'✅' if exists else '❌'}  {f:<40} {size:.0f} KB")

print(f"\n  Next: open http://localhost:8000/accuracy to see live comparisons\n")
