"""
eWardrobeAI — Terminal Accuracy Checker
Trains Stage 1 models and compares accuracy in the terminal.
Highlights models in the 85–90% target range and picks the best.

Run: python scripts/check_accuracy.py
"""

import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.model_selection import train_test_split
from src.models.stage1_body_models import (
    Stage1AccuracyChecker, _generate_synthetic_data
)

W = 66

def line(c='─'): print('  ' + c * W)
def header(t):
    print()
    line('═')
    print(f"  {t}")
    line('═')

def acc_label(pct):
    if pct > 90:        return '▲ HIGH  (>90%)'
    if pct >= 85:       return '✓ TARGET (85–90%)'
    if pct >= 75:       return '~ WARN  (75–85%)'
    return              '✗ LOW   (<75%)'

def acc_bar(pct, width=30):
    filled = int(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)


# ── Train & Evaluate ──────────────────────────────────────────────────────────

header("eWardrobeAI  —  Accuracy Comparison  (Terminal Mode)")
print(f"  Target range : 85–90%  |  Best model auto-detected")
print(f"  Stage        : Stage 1 — Body Type Classification")
line()

print("  Generating 4,000 synthetic samples…")
X, y_bt, y_sz = _generate_synthetic_data(4000)
X_tr, X_te, ybt_tr, ybt_te, ysz_tr, ysz_te = train_test_split(
    X, y_bt, y_sz, test_size=0.25, random_state=42
)

checker = Stage1AccuracyChecker()

print("  Training Random Forest      (n_estimators=200)…", end=' ', flush=True)
t0 = time.time()
checker.rf_model.train(X_tr, ybt_tr, ysz_tr)
checker.rf_model.save()
print(f"done in {time.time()-t0:.1f}s")

print("  Training Gradient Boosting  (n_estimators=150)…", end=' ', flush=True)
t0 = time.time()
checker.gb_model.train(X_tr, ybt_tr, ysz_tr)
checker.gb_model.save()
print(f"done in {time.time()-t0:.1f}s")

print("  Evaluating all 3 models on held-out test set…")

rule_res = checker.rule_model.evaluate(X_te, ybt_te, ysz_te)
rf_res   = checker.rf_model.evaluate(X_te, ybt_te, ysz_te)
gb_res   = checker.gb_model.evaluate(X_te, ybt_te, ysz_te)

models = [
    ('Rule-Based',        rule_res),
    ('Random Forest',     rf_res),
    ('Gradient Boosting', gb_res),
]

# ── Results Table ─────────────────────────────────────────────────────────────

header("ACCURACY RESULTS — Body Type Classification")
print(f"  {'Model':<24} {'Body Acc':>9} {'Size Acc':>9}  {'Bar (Body Type)':32}  Status")
line()

scores = {}
for name, res in models:
    bt  = res['bodyType']['accuracy'] * 100
    sz  = res['standardSize']['accuracy'] * 100
    bar = acc_bar(bt)
    lbl = acc_label(bt)
    scores[name] = bt
    print(f"  {name:<24} {bt:>8.2f}%  {sz:>8.2f}%  [{bar}]  {lbl}")

line()

# ── Best Model Detection ──────────────────────────────────────────────────────

best_name = max(scores, key=scores.get)
best_acc  = scores[best_name]

print()
print(f"  {'─'*W}")
print(f"  🏆  HIGHEST ACCURACY : {best_name}")
print(f"       Body Type Acc   : {best_acc:.2f}%   {acc_label(best_acc)}")
print(f"  {'─'*W}")

# Per-keypoint F1 for best ML model
best_res = rf_res if 'Random' in best_name else (
           gb_res if 'Gradient' in best_name else rule_res)

print()
print(f"  Per-Class F1  ({best_name})")
line('─')
report = best_res['bodyType'].get('report', {})
for cls in ['hourglass', 'inverted_triangle', 'pear', 'rectangle']:
    f1  = report.get(cls, {}).get('f1-score', 0) * 100
    bar = acc_bar(f1, width=20)
    print(f"  {cls:<22} [{bar}]  {f1:>6.2f}%")
line('─')

# ── Range Summary ─────────────────────────────────────────────────────────────

header("RANGE SUMMARY  (85–90% = Target)")
in_target   = [n for n, s in scores.items() if 85 <= s <= 90]
above_target= [n for n, s in scores.items() if s > 90]
below_target= [n for n, s in scores.items() if s < 85]

def fmt_list(lst): return ', '.join(lst) if lst else 'none'

print(f"  ▲ Above target (>90%) : {fmt_list(above_target)}")
print(f"  ✓ In target (85–90%)  : {fmt_list(in_target)}")
print(f"  ✗ Below target (<85%) : {fmt_list(below_target)}")
print()
print(f"  Recommended model : {best_name}  ({best_acc:.2f}%)")
line('═')
print()
