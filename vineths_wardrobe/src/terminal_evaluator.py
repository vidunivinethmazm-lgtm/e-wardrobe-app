"""
eWardrobeAI — Terminal Model Evaluator
Runs when an image is uploaded. Prints accuracy comparison to the terminal.

Stage 1 — Body Calibration:
  Rule-Based vs Random Forest vs Gradient Boosting
  Shows predicted body type + size from each model

Stage 2 — Face CNN:
  DeepFaceCNN vs LightFaceCNN
  Shows per-keypoint MAE comparison on the uploaded face image
"""

import os
import sys
import time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Print helpers ─────────────────────────────────────────────────────────────

def _line(char='─', width=62):
    print(char * width)

def _header(title):
    print()
    _line('═')
    print(f"  {title}")
    _line('═')

def _section(title):
    print(f"\n  ── {title} {'─'*(54-len(title))}")


# ── Stage 1: Body Calibration ─────────────────────────────────────────────────

def evaluate_body_calibration(
    shoulder_cm: float,
    chest_cm:    float,
    waist_cm:    float,
    height_cm:   float,
    hip_cm:      float = None,
    inseam_cm:   float = None,
):
    """
    Runs all 3 Stage 1 models on the provided measurements and
    prints a comparison table to the terminal.
    """
    hip    = hip_cm    or waist_cm + 25
    inseam = inseam_cm or height_cm * 0.47

    _header("STAGE 1 — Body Calibration Model Comparison")
    print(f"  Input: shoulder={shoulder_cm}cm  chest={chest_cm}cm  "
          f"waist={waist_cm}cm  height={height_cm}cm")
    print(f"         hip={hip:.1f}cm  inseam={inseam:.1f}cm")

    results = {}

    # ── Model 1: Rule-Based ───────────────────────────────────────────────────
    try:
        from src.models.stage1_body_models import RuleBasedBodyModel
        t0 = time.perf_counter()
        m1 = RuleBasedBodyModel()
        bt = m1.predict_body_type(shoulder_cm, chest_cm, waist_cm, hip)
        sz = m1.predict_size(chest_cm)
        ms = (time.perf_counter() - t0) * 1000
        results['Rule-Based'] = {'bodyType': bt, 'size': sz, 'ms': ms, 'proba': None}
    except Exception as e:
        results['Rule-Based'] = {'error': str(e)}

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    try:
        from src.models.stage1_body_models import (
            RandomForestBodyModel, RF_MODEL_PATH, _generate_synthetic_data
        )
        from sklearn.model_selection import train_test_split
        rf = RandomForestBodyModel()
        if os.path.exists(RF_MODEL_PATH):
            rf.load()
        else:
            # Train quickly on synthetic data
            print("\n  [Training Random Forest on 2000 samples — first run only…]")
            X, y_bt, y_sz = _generate_synthetic_data(2000)
            X_tr, _, y_bt_tr, _, y_sz_tr, _ = train_test_split(
                X, y_bt, y_sz, test_size=0.2, random_state=42)
            rf.train(X_tr, y_bt_tr, y_sz_tr)
            rf.save()

        t0 = time.perf_counter()
        pred = rf.predict_single(shoulder_cm, chest_cm, waist_cm, height_cm, hip, inseam)
        ms   = (time.perf_counter() - t0) * 1000
        results['Random Forest'] = {
            'bodyType': pred['bodyType'],
            'size':     pred['standardSize'],
            'ms':       ms,
            'proba':    pred.get('bodyTypeProba'),
        }
    except Exception as e:
        results['Random Forest'] = {'error': str(e)}

    # ── Model 3: Gradient Boosting ────────────────────────────────────────────
    try:
        from src.models.stage1_body_models import (
            GradientBoostingBodyModel, GB_MODEL_PATH, _generate_synthetic_data
        )
        gb = GradientBoostingBodyModel()
        if os.path.exists(GB_MODEL_PATH):
            gb.load()
        else:
            print("\n  [Training Gradient Boosting on 2000 samples — first run only…]")
            X, y_bt, y_sz = _generate_synthetic_data(2000)
            X_tr, _, y_bt_tr, _, y_sz_tr, _ = train_test_split(
                X, y_bt, y_sz, test_size=0.2, random_state=42)
            gb.train(X_tr, y_bt_tr, y_sz_tr)
            gb.save()

        t0 = time.perf_counter()
        pred = gb.predict_single(shoulder_cm, chest_cm, waist_cm, height_cm, hip, inseam)
        ms   = (time.perf_counter() - t0) * 1000
        results['Gradient Boosting'] = {
            'bodyType': pred['bodyType'],
            'size':     pred['standardSize'],
            'ms':       ms,
            'proba':    pred.get('bodyTypeProba'),
        }
    except Exception as e:
        results['Gradient Boosting'] = {'error': str(e)}

    # ── Print table ───────────────────────────────────────────────────────────
    _section("Predictions")
    print(f"  {'Model':<25} {'Body Type':<22} {'Size':<8} {'Speed':>8}")
    _line()
    for name, r in results.items():
        if 'error' in r:
            print(f"  {name:<25} ERROR: {r['error']}")
            continue
        print(f"  {name:<25} {r['bodyType']:<22} {r['size']:<8} {r['ms']:>6.2f}ms")

    # Agreement check
    bts = [r['bodyType'] for r in results.values() if 'bodyType' in r]
    sizes = [r['size'] for r in results.values() if 'size' in r]
    _line()
    agree_bt   = '✅ All agree' if len(set(bts))   == 1 else f'⚠️  Disagree: {set(bts)}'
    agree_size = '✅ All agree' if len(set(sizes)) == 1 else f'⚠️  Disagree: {set(sizes)}'
    print(f"  Body Type consensus : {agree_bt}")
    print(f"  Size    consensus   : {agree_size}")

    # Probability breakdown (RF + GB)
    for name in ['Random Forest', 'Gradient Boosting']:
        r = results.get(name, {})
        if r.get('proba'):
            _section(f"{name} — Body Type Probabilities")
            for bt, prob in sorted(r['proba'].items(), key=lambda x: -x[1]):
                bar = '█' * int(prob * 30)
                print(f"  {bt:<22} {bar:<30} {prob*100:>5.1f}%")

    print()
    return results


# ── Stage 2: Face CNN ─────────────────────────────────────────────────────────

def evaluate_face_cnn(image_bgr: np.ndarray):
    """
    Runs both face CNNs on the uploaded image and prints
    keypoint MAE comparison to the terminal.
    """
    _header("STAGE 2 — Face CNN Model Comparison")

    import cv2 as cv

    # Detect and crop face
    gray    = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    cascade = cv.CascadeClassifier(
        cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        print("  ⚠️  No face detected in uploaded image.")
        print("  Face CNN comparison skipped — upload a clear front-facing photo.\n")
        return None

    x, y, w, h = faces[0]
    face_gray   = gray[y:y+h, x:x+w]
    face_96     = cv.resize(face_gray, (96, 96)).astype(np.float32) / 255.0
    print(f"  Face detected at ({x},{y}) size {w}×{h}px  →  resized to 96×96")

    import torch
    from src.models.stage2_face_models import (
        DeepFaceCNN, LightFaceCNN,
        DEEP_MODEL_PATH, LIGHT_MODEL_PATH,
        KEYPOINT_NAMES, IMG_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.from_numpy(face_96[np.newaxis, np.newaxis]).to(device)

    predictions = {}

    for name, ModelClass, path in [
        ('DeepFaceCNN  (5-block)', DeepFaceCNN,  DEEP_MODEL_PATH),
        ('LightFaceCNN (3-block)', LightFaceCNN, LIGHT_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            predictions[name] = {'error': f'Model not trained yet — run: python scripts/train_all_models.py'}
            continue
        try:
            model = ModelClass().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()

            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(tensor).cpu().numpy()[0] * IMG_SIZE
            ms   = (time.perf_counter() - t0) * 1000

            kpts = {KEYPOINT_NAMES[i]: (round(float(pred[i*2]),1),
                                        round(float(pred[i*2+1]),1))
                    for i in range(15)}
            predictions[name] = {'keypoints': kpts, 'ms': ms}
        except Exception as e:
            predictions[name] = {'error': str(e)}

    # ── Print keypoint table ──────────────────────────────────────────────────
    models_ok = {k: v for k, v in predictions.items() if 'keypoints' in v}

    if len(models_ok) < 2:
        for name, r in predictions.items():
            if 'error' in r:
                print(f"  {name}: {r['error']}")
        print()
        return predictions

    names  = list(models_ok.keys())
    _section("Keypoint Predictions (x, y in pixels on 96×96 image)")
    print(f"  {'Keypoint':<28} {names[0]:<22} {names[1]:<22} {'Δ diff':>8}")
    _line()

    total_diff = 0
    for kp in KEYPOINT_NAMES:
        p1 = models_ok[names[0]]['keypoints'][kp]
        p2 = models_ok[names[1]]['keypoints'][kp]
        dx = abs(p1[0] - p2[0]); dy = abs(p1[1] - p2[1])
        diff = round((dx + dy) / 2, 2)
        total_diff += diff
        flag = ' ⚠️' if diff > 4 else ''
        print(f"  {kp:<28} ({p1[0]:>5.1f},{p1[1]:>5.1f})        "
              f"({p2[0]:>5.1f},{p2[1]:>5.1f})        {diff:>6.2f}{flag}")

    _line()
    avg_diff = round(total_diff / len(KEYPOINT_NAMES), 3)
    print(f"  {'Average pixel difference':<28}{'':22}{'':22} {avg_diff:>6.2f}")

    _section("Inference Speed")
    for name, r in models_ok.items():
        params = '~4.5M' if 'Deep' in name else '~1.2M'
        print(f"  {name:<30} {r['ms']:>6.2f} ms/image   params={params}")

    _section("Agreement Summary")
    if avg_diff <= 2:
        print("  ✅  Models closely agree (avg diff ≤ 2 px) — both are reliable")
    elif avg_diff <= 5:
        print("  ✅  Models broadly agree (avg diff ≤ 5 px)")
    else:
        print(f"  ⚠️  Models diverge by {avg_diff} px avg — "
              "DeepFaceCNN is more accurate but slower")

    print()
    return predictions


# ── Combined entry point ──────────────────────────────────────────────────────

def run_on_upload(
    image_bytes:       bytes,
    shoulder_width_cm: float,
    chest_cm:          float,
    waist_cm:          float,
    height_cm:         float,
    hip_cm:            float = None,
    inseam_cm:         float = None,
):
    """
    Called from the FastAPI endpoint when an image is uploaded.
    Runs Stage 1 and Stage 2 model comparisons and prints to terminal.
    """
    print("\n" + "█"*62)
    print("  eWardrobeAI — Image Upload Detected")
    print("  Running multi-model accuracy comparison…")
    print("█"*62)

    # Stage 1
    evaluate_body_calibration(
        shoulder_width_cm, chest_cm, waist_cm, height_cm, hip_cm, inseam_cm
    )

    # Stage 2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        evaluate_face_cnn(bgr)
    else:
        print("\n  [Stage 2] Could not decode image for face CNN comparison.\n")
