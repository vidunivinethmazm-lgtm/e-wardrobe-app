"""Accuracy report — highlights models in the 85–95% accuracy range."""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.model_registry import ModelRegistry

reg = ModelRegistry()
W   = 66

def sep(c='='): print(c * W)
def hdr(t):     sep(); print(f'  {t}'); sep()
def in_range(v): return 0.85 <= v <= 0.95

print()
hdr('eWardrobeAI  --  ACCURACY REPORT  (85% - 95% range)')

found_any = False

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 1 : Body Calibration  (4,000 samples)')
sep('-')
print(f"  {'Model':<40} {'Body Type':>10} {'Size':>7} {'F1':>7}  {'In Range':>9}")
sep('-')
r1 = reg.evaluate_stage1(4000)
for m in r1['models']:
    bt   = m['bodyType']['accuracy']
    sz   = m['standardSize']['accuracy']
    f1   = m['bodyType']['f1_weighted']
    flag = '  ✅ 85-95%' if in_range(bt) else ('  ⬆ above' if bt > 0.95 else '  ⬇ below')
    if in_range(bt): found_any = True
    print(f"  {m['model']:<40} {bt:>9.1%} {sz:>6.1%} {f1:>6.3f}{flag}")
sep('-')

# ── Stage 3 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 3 : Outfit Recommendation  (6 scenarios, top-5)')
sep('-')
print(f"  {'Model':<40} {'Prec@5':>8} {'Coverage':>9} {'Score':>7}  {'In Range':>9}")
sep('-')
r3 = reg.evaluate_stage3()
for m in r3['models']:
    p    = m['precision_at_k']
    flag = '  ✅ 85-95%' if in_range(p) else ('  ⬆ above' if p > 0.95 else '  ⬇ below')
    if in_range(p): found_any = True
    print(f"  {m['model']:<40} {p:>8.1%} {m['coverage']:>8.1%} {m['avg_score']:>6.1%}{flag}")
sep('-')

# ── Stage 4 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 4 : Avatar Scale Regression  (1,000 samples, 7 dims)')
sep('-')
print(f"  {'Model':<40} {'RMSE':>8} {'MAE':>8} {'Fit %':>7}  {'In Range':>9}")
sep('-')
r4 = reg.evaluate_stage4()
for m in r4['models']:
    fit  = max(0.0, 1.0 - m['rmse'])
    flag = '  ✅ 85-95%' if in_range(fit) else ('  ⬆ above' if fit > 0.95 else '  ⬇ below')
    if in_range(fit): found_any = True
    print(f"  {m['model']:<40} {m['rmse']:>8.4f} {m['mae']:>8.4f} {fit:>6.1%}{flag}")
sep('-')

# ── Summary ───────────────────────────────────────────────────────────────────
print()
hdr('MODELS IN 85% - 95% RANGE')
for m in r1['models']:
    bt = m['bodyType']['accuracy']
    if in_range(bt):
        print(f"  ✅  [Stage 1]  {m['model']:<38}  Body Type: {bt:.1%}  F1: {m['bodyType']['f1_weighted']:.3f}")
for m in r3['models']:
    p = m['precision_at_k']
    if in_range(p):
        print(f"  ✅  [Stage 3]  {m['model']:<38}  Prec@5: {p:.1%}")
for m in r4['models']:
    fit = max(0.0, 1.0 - m['rmse'])
    if in_range(fit):
        print(f"  ✅  [Stage 4]  {m['model']:<38}  Fit: {fit:.1%}  RMSE: {m['rmse']:.4f}")

if not found_any:
    print('  No models fall in the 85-95% accuracy range.')

print()
sep()
print('  Stage 2 (Face CNN): run  python scripts/train_all_models.py  to train.')
sep()
print()
