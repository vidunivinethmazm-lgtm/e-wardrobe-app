"""
eWardrobeAI — Automated Swagger / API Test Runner
Runs against a live server at http://localhost:8000
Execute: python scripts/run_swagger_tests.py
"""

import sys
import json
import time
import cv2
import numpy as np

try:
    import requests
except ImportError:
    print("[ERROR] Install requests: python -m pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg    = f"  {status}  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append((label, condition))


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def synthetic_selfie() -> bytes:
    img = np.full((200, 200), 200, dtype=np.uint8)
    cv2.ellipse(img, (100, 105), (75, 90), 0, 0, 360, 230, -1)
    cv2.circle(img, (75, 85),  12, 80, -1)
    cv2.circle(img, (125, 85), 12, 80, -1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def run_all_tests():
    print("\n" + "═"*60)
    print("  eWardrobeAI — Swagger API Test Suite")
    print("  Target:", BASE)
    print("═"*60)

    # ── Test 1: Health Check ─────────────────────────────────────────────────
    section("1 / 11  —  GET /api/health")
    try:
        r    = requests.get(f"{BASE}/api/health", timeout=5)
        data = r.json()
        check("Status 200",            r.status_code == 200)
        check("pipelineReady = true",  data.get('pipelineReady') is True)
        check("status = ok",           data.get('status') == 'ok')
    except Exception as e:
        check("Server reachable", False, str(e))
        print("\n[FATAL] Server not running. Start with: python app.py")
        sys.exit(1)

    # ── Test 2: Wardrobe Summary ─────────────────────────────────────────────
    section("2 / 11  —  GET /api/wardrobe/summary")
    r    = requests.get(f"{BASE}/api/wardrobe/summary")
    data = r.json()
    check("Status 200",                r.status_code == 200)
    check("Has 'Clean' key",           'Clean'      in data)
    check("Has 'Dirty' key",           'Dirty'      in data)
    check("Has 'In Laundry' key",      'In Laundry' in data)
    check("Total = 12 garments",       sum(data.values()) == 12,
          f"Got {sum(data.values())}")
    check("8 Clean garments",          data.get('Clean') == 8,
          f"Got {data.get('Clean')}")

    # ── Test 3: Wardrobe Items ────────────────────────────────────────────────
    section("3 / 11  —  GET /api/wardrobe/items")
    r    = requests.get(f"{BASE}/api/wardrobe/items")
    data = r.json()
    check("Status 200",                r.status_code == 200)
    check("total = 12",                data.get('total') == 12)
    dirty_wearable = [i for i in data.get('items', [])
                      if i['cleaningStatus'] in ('Dirty','In Laundry')
                      and i['isWearable']]
    check("No Dirty/Laundry items are wearable", len(dirty_wearable) == 0,
          f"{len(dirty_wearable)} violations found")

    # ── Test 4: Sizing Validate (valid) ──────────────────────────────────────
    section("4 / 11  —  POST /api/sizing/validate  (valid input)")
    r = requests.post(f"{BASE}/api/sizing/validate", json={
        'shoulder_width_cm': 42, 'chest_cm': 92,
        'waist_cm': 72, 'height_cm': 168
    })
    data = r.json()
    check("Status 200",                    r.status_code == 200)
    check("valid = true",                  data.get('valid') is True)
    check("standardSize = M",             data.get('sizingProfile',{}).get('standardSize') == 'M')
    check("avatarScaleParams present",    'avatarScaleParams' in data)
    sp = data.get('avatarScaleParams', {})
    all_in_range = all(0.70 <= v <= 1.40 for v in sp.values())
    check("All scale params in [0.70, 1.40]", all_in_range)

    # ── Test 5: Sizing Validate (invalid) ────────────────────────────────────
    section("5 / 11  —  POST /api/sizing/validate  (invalid input)")
    r = requests.post(f"{BASE}/api/sizing/validate", json={
        'shoulder_width_cm': 42, 'chest_cm': 250,
        'waist_cm': 72, 'height_cm': 168
    })
    check("Status 422 for chest=250",  r.status_code == 422)
    check("Errors field present",      'errors' in r.json().get('detail',{}))

    # ── Test 6: Demo Try-On (default) ────────────────────────────────────────
    section("6 / 11  —  POST /api/demo/tryon  (default values)")
    r = requests.post(f"{BASE}/api/demo/tryon", json={
        'shoulder_width_cm': 42, 'chest_cm': 92,
        'waist_cm': 72, 'height_cm': 168,
        'styles': 'smart_casual,casual', 'occasion': 'office',
        'animation_key': 'walk', 'top_k': 3,
    })
    data = r.json()
    check("Status 200",                    r.status_code == 200)
    check("success = true",                data.get('success') is True)
    check("recommendations > 0",          len(data.get('recommendations',[])) > 0)
    check("renderPayload present",         data.get('renderPayload') is not None)
    check("processingTimeMs > 0",         data.get('processingTimeMs', 0) > 0)
    recs = data.get('recommendations', [])
    scores_valid = all(0.0 <= r['score'] <= 1.0 for r in recs)
    check("All scores in [0, 1]",         scores_valid)
    all_clean = all(
        item['status'] == 'Clean'
        for rec in recs for item in rec.get('items', [])
    )
    check("All recommended items are Clean", all_clean)

    # ── Test 7: Demo Try-On (formal occasion) ────────────────────────────────
    section("7 / 11  —  POST /api/demo/tryon  (formal occasion)")
    r = requests.post(f"{BASE}/api/demo/tryon", json={
        'shoulder_width_cm': 42, 'chest_cm': 92,
        'waist_cm': 72, 'height_cm': 168,
        'styles': 'formal', 'occasion': 'interview',
    })
    data = r.json()
    check("Status 200",     r.status_code == 200)
    check("success = true", data.get('success') is True)

    # ── Test 8: Garment Status Update ────────────────────────────────────────
    section("8 / 11  —  PATCH /api/wardrobe/{id}/status")
    # Mark dirty
    r = requests.patch(f"{BASE}/api/wardrobe/GAR-001/status",
                       json={'status': 'Dirty'})
    check("Mark Dirty — status 200",      r.status_code == 200)
    check("isWearable = false after Dirty", r.json().get('isWearable') is False)

    # Confirm excluded from recommendation
    r2 = requests.post(f"{BASE}/api/demo/tryon", json={
        'shoulder_width_cm': 42, 'chest_cm': 92,
        'waist_cm': 72, 'height_cm': 168, 'top_k': 10,
    })
    ids = [item['garmentId']
           for rec in r2.json().get('recommendations',[])
           for item in rec['items']]
    check("GAR-001 excluded after marking Dirty", 'GAR-001' not in ids)

    # Restore to Clean
    r = requests.patch(f"{BASE}/api/wardrobe/GAR-001/status",
                       json={'status': 'Clean'})
    check("Restore to Clean — status 200",  r.status_code == 200)
    check("isWearable = true after Clean",  r.json().get('isWearable') is True)

    # Invalid status
    r = requests.patch(f"{BASE}/api/wardrobe/GAR-001/status",
                       json={'status': 'Washed'})
    check("Invalid status → 400",          r.status_code == 400)

    # Non-existent garment
    r = requests.patch(f"{BASE}/api/wardrobe/GAR-999/status",
                       json={'status': 'Clean'})
    check("Non-existent garment → 404",    r.status_code == 404)

    # ── Test 9: Full Try-On with Selfie ──────────────────────────────────────
    section("9 / 11  —  POST /api/tryon  (with synthetic selfie)")
    img_bytes = synthetic_selfie()
    r = requests.post(f"{BASE}/api/tryon",
        data={
            'shoulder_width_cm': '42', 'chest_cm': '92',
            'waist_cm': '72',          'height_cm': '168',
            'styles': 'smart_casual',  'occasion': 'casual',
            'animation_key': 'idle',   'top_k': '3',
        },
        files={'selfie': ('selfie.jpg', img_bytes, 'image/jpeg')},
    )
    data = r.json()
    check("Status 200",                  r.status_code == 200)
    check("success = true",              data.get('success') is True)
    check("faceAnalysis present",        data.get('faceAnalysis') is not None)
    check("renderPayload present",       data.get('renderPayload') is not None)

    # ── Test 10: Model Status ─────────────────────────────────────────────────
    section("10 / 11  —  GET /api/model/status")
    r    = requests.get(f"{BASE}/api/model/status")
    data = r.json()
    check("Status 200",          r.status_code == 200)
    check("modelPath present",   'modelPath'  in data)
    check("exists key present",  'exists'     in data)
    check("fileSizeMB present",  'fileSizeMB' in data)
    if data.get('exists'):
        check("Model file > 0 MB",   data['fileSizeMB'] > 0,
              f"{data['fileSizeMB']} MB")
    else:
        print(f"  {WARN}  Model not yet trained — run: python -m src.face_keypoint_model")

    # ── Test 11: Animations ───────────────────────────────────────────────────
    section("11 / 11  —  GET /api/animations")
    r    = requests.get(f"{BASE}/api/animations")
    data = r.json()
    check("Status 200",           r.status_code == 200)
    for key in ['idle','walk','rotate','pose_t','pose_a','catwalk']:
        check(f"Key '{key}' present", key in data.get('animations', {}))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"  RESULTS:  {passed}/{total} passed  |  {failed} failed")
    print("═"*60 + "\n")

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
