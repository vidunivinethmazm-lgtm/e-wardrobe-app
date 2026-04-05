"""Fetch and print accuracy report from the running app endpoints."""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import requests

BASE = 'http://localhost:8000'
W    = 66

def sep(c='='): print(c * W)
def hdr(t):     sep(); print(f'  {t}'); sep()

print()
hdr('eWardrobeAI  --  ACCURACY REPORT  (live from app)')

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 1: Body Calibration')
sep('-')
r1 = requests.post(f'{BASE}/api/accuracy/stage/1').json()
print(f"  {'Model':<40} {'BodyType':>9} {'Size':>7} {'F1':>7}")
sep('-')
for m in r1['models']:
    best = '  <-- BEST' if m['model'] == r1['bestBodyType'] else ''
    bt = m['bodyType']['accuracy']
    sz = m['standardSize']['accuracy']
    f1 = m['bodyType']['f1_weighted']
    print(f"  {m['model']:<40} {bt:>8.1%} {sz:>6.1%} {f1:>6.3f}{best}")
sep('-')

# ── Stage 3 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 3: Outfit Recommendation')
sep('-')
r3 = requests.post(f'{BASE}/api/accuracy/stage/3').json()
print(f"  {'Model':<38} {'Prec@5':>7} {'Cover':>7} {'Score':>7} {'Speed':>8}")
sep('-')
for m in r3['models']:
    best = '  <-- BEST' if 'Heuristic' in m['model'] else ''
    print(f"  {m['model']:<38} {m['precision_at_k']:>6.3f} {m['coverage']:>6.3f} {m['avg_score']:>6.3f} {m['response_time_ms']:>5.1f}ms{best}")
sep('-')

# ── Stage 4 ───────────────────────────────────────────────────────────────────
print('\n  STAGE 4: Avatar Scale Regression')
sep('-')
r4 = requests.post(f'{BASE}/api/accuracy/stage/4').json()
print(f"  {'Model':<42} {'RMSE':>9} {'MAE':>9} {'Speed':>9}")
sep('-')
for m in r4['models']:
    best = '  <-- BEST' if m['model'] == r4['bestRMSE'] else ''
    print(f"  {m['model']:<42} {m['rmse']:>9.6f} {m['mae']:>9.6f} {m['ms_per_sample']:>6.4f}ms{best}")
sep('-')

# ── Summary ───────────────────────────────────────────────────────────────────
print()
hdr('BEST MODELS PER STAGE')
print(f"  Stage 1  Best body-type  :  {r1['bestBodyType']}")
print(f"  Stage 1  Best size       :  {r1['bestSize']}")
print(f"  Stage 3  Best recommender:  {r3['bestPrecision']} Recommender")
print(f"  Stage 4  Best scaler     :  {r4['bestRMSE']}")
print()
print('  Stage 2 (Face CNN): requires trained model.')
print('  Run: python scripts/train_all_models.py  to enable it.')
sep()
print()
