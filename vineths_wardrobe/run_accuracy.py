"""Print a clean accuracy report for all pipeline stages."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

from src.model_registry import ModelRegistry

W = 66
reg = ModelRegistry()

def sep(c='='):  print(c * W)
def hdr(t):      sep(); print(f'  {t}'); sep()

print()
hdr('eWardrobeAI  --  FULL PIPELINE ACCURACY REPORT')

# ── STAGE 1 ──────────────────────────────────────────────────────────────────
print('\n  STAGE 1: Body Calibration  (4,000 synthetic samples)')
sep('-')
r1 = reg.evaluate_stage1(4000)
print(f"  {'Model':<38} {'BodyType Acc':>12} {'Size Acc':>9} {'F1':>7}")
sep('-')
for m in r1['models']:
    bt   = m['bodyType']['accuracy']
    sz   = m['standardSize']['accuracy']
    f1   = m['bodyType']['f1_weighted']
    best = '  <-- BEST' if m['model'] == r1['bestBodyType'] else ''
    print(f"  {m['model']:<38} {bt:>11.1%} {sz:>8.1%} {f1:>6.3f}{best}")
sep('-')
print()

# ── STAGE 3 ──────────────────────────────────────────────────────────────────
print('  STAGE 3: Outfit Recommendation  (6 scenarios, top-5)')
sep('-')
r3 = reg.evaluate_stage3()
print(f"  {'Model':<38} {'Prec@5':>7} {'Coverage':>9} {'AvgScore':>9} {'Speed':>8}")
sep('-')
for m in r3['models']:
    best = '  <-- BEST' if 'Heuristic' in m['model'] and r3['bestPrecision'] == 'Heuristic' else ''
    print(f"  {m['model']:<38} {m['precision_at_k']:>6.3f}  {m['coverage']:>8.3f}  {m['avg_score']:>8.3f}  {m['response_time_ms']:>5.1f}ms{best}")
sep('-')
print()

# ── STAGE 4 ──────────────────────────────────────────────────────────────────
print('  STAGE 4: Avatar Scale Regression  (1,000 samples, 7 dims)')
sep('-')
r4 = reg.evaluate_stage4()
print(f"  {'Model':<42} {'RMSE':>8} {'MAE':>8} {'Speed':>10}")
sep('-')
for m in r4['models']:
    best = '  <-- BEST' if m['model'] == r4['bestRMSE'] else ''
    print(f"  {m['model']:<42} {m['rmse']:>8.6f} {m['mae']:>8.6f} {m['ms_per_sample']:>7.4f}ms{best}")
sep('-')
print()

# ── SUMMARY ──────────────────────────────────────────────────────────────────
hdr('SUMMARY  --  Best Models Per Stage')
print(f"  Stage 1  Best body-type model  :  {r1['bestBodyType']}")
print(f"  Stage 1  Best size model       :  {r1['bestSize']}")
print(f"  Stage 3  Best recommender      :  {r3['bestPrecision']} Recommender")
print(f"  Stage 4  Best scaler           :  {r4['bestRMSE']}")
print()
print('  Stage 2 (Face CNN) requires a trained model.')
print('  Run: python scripts/train_all_models.py  to train it first.')
sep()
print()
