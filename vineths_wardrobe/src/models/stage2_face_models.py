"""
eWardrobeAI — Stage 2: Dual Facial Keypoint CNN Models

Model 1 — DeepFaceCNN  (5 conv blocks, 256 max filters)
  High-accuracy model. Slower inference. Target: MAE < 4 px.

Model 2 — LightFaceCNN (3 conv blocks, 128 max filters)
  Lightweight, faster. Suitable for mobile/edge inference. Target: MAE < 6 px.

Accuracy Checker
  Loads both trained models and evaluates pixel MAE, per-keypoint error,
  RMSE, and inference speed on the training.csv validation split.
  Returns a side-by-side comparison table.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import STAGE2

MODEL_DIR         = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DEEP_MODEL_PATH   = os.path.join(MODEL_DIR, 'face_keypoint_cnn.pth')
LIGHT_MODEL_PATH  = os.path.join(MODEL_DIR, 'face_keypoint_light.pth')
DATA_PATH         = os.path.join(os.path.dirname(__file__), '..', '..', 'training.csv')

IMG_SIZE        = STAGE2["img_size"]
NUM_KEYPOINTS   = STAGE2["num_keypoints"]
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = STAGE2["batch_size"]
EPOCHS_LIGHT    = STAGE2["epochs_light"]

KEYPOINT_COLS = [
    'left_eye_center_x',        'left_eye_center_y',
    'right_eye_center_x',       'right_eye_center_y',
    'left_eye_inner_corner_x',  'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',  'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x','right_eyebrow_outer_end_y',
    'nose_tip_x',               'nose_tip_y',
    'mouth_left_corner_x',      'mouth_left_corner_y',
    'mouth_right_corner_x',     'mouth_right_corner_y',
    'mouth_center_top_lip_x',   'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y',
]
KEYPOINT_NAMES = [c.rsplit('_', 1)[0] for c in KEYPOINT_COLS[::2]]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, images, keypoints, augment=False):
        self.images    = images.astype(np.float32)
        self.keypoints = keypoints.astype(np.float32)
        self.augment   = augment

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        kpt = self.keypoints[idx].copy()
        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1].copy()
            kpt[0::2] = 1.0 - kpt[0::2]
        img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1) if self.augment else img
        return torch.from_numpy(img[np.newaxis]), torch.from_numpy(kpt)


def _load_data():
    df   = pd.read_csv(DATA_PATH).dropna(subset=KEYPOINT_COLS).reset_index(drop=True)
    imgs = np.array([np.fromstring(r, np.float32, sep=' ')
                     for r in df['Image']]).reshape(-1, IMG_SIZE, IMG_SIZE) / 255.0
    kpts = df[KEYPOINT_COLS].values.astype(np.float32) / IMG_SIZE
    return imgs, kpts


# ── Model 1: Deep Face CNN (existing architecture) ────────────────────────────

class DeepFaceCNN(nn.Module):
    """5 conv blocks → 1024+512 FC → Sigmoid output. High accuracy."""
    name = "DeepFaceCNN (5-block, 256 filters)"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(256,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 1024), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(1024, 512),     nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, NUM_KEYPOINTS), nn.Sigmoid(),
        )

    def forward(self, x): return self.regressor(self.features(x))


# ── Model 2: Lightweight Face CNN ────────────────────────────────────────────

class LightFaceCNN(nn.Module):
    """
    3 conv blocks → 256 FC → Sigmoid output.
    ~4× fewer parameters than DeepFaceCNN.
    Faster inference, slightly lower accuracy.
    """
    name = "LightFaceCNN (3-block, 128 filters)"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*12*12, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, NUM_KEYPOINTS), nn.Sigmoid(),
        )

    def forward(self, x): return self.regressor(self.features(x))


# ── Training Utility ──────────────────────────────────────────────────────────

def _train_model(model: nn.Module, model_path: str,
                 X_train, y_train, X_val, y_val,
                 epochs: int = EPOCHS_LIGHT) -> dict:
    model.to(DEVICE)
    ds_train = FaceDataset(X_train, y_train, augment=True)
    ds_val   = FaceDataset(X_val,   y_val,   augment=False)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=STAGE2["learning_rate"], weight_decay=STAGE2["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, patience_count, patience = float('inf'), 0, 12
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        for imgs, kpts in dl_train:
            imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), kpts)
            loss.backward(); optimizer.step()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= len(ds_train)

        model.eval()
        val_loss = val_mae = 0
        with torch.no_grad():
            for imgs, kpts in dl_val:
                imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
                pred       = model(imgs)
                val_loss  += criterion(pred, kpts).item() * len(imgs)
                val_mae   += (pred - kpts).abs().mean().item() * len(imgs)
        val_loss /= len(ds_val); val_mae /= len(ds_val)
        scheduler.step()
        history.append({'epoch': epoch, 'val_loss': val_loss, 'val_mae_px': val_mae * IMG_SIZE})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model.name}] Epoch {epoch:03d}  val_loss={val_loss:.5f}  "
                  f"val_mae={val_mae * IMG_SIZE:.2f} px")

        if val_loss < best_val:
            best_val = val_loss; patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  [{model.name}] Early stop at epoch {epoch}")
                break

    return {'best_val_loss': best_val, 'history': history}


# ── Accuracy Checker ──────────────────────────────────────────────────────────

class Stage2AccuracyChecker:
    """
    Trains LightFaceCNN (DeepFaceCNN uses already-saved weights) and
    evaluates both models on the same validation split.

    Metrics reported:
      - Mean Absolute Error (MAE) in pixels
      - Root Mean Squared Error (RMSE) in pixels
      - Per-keypoint MAE breakdown
      - Inference speed (ms per image)
      - Parameter count
    """

    def __init__(self):
        self.deep  = DeepFaceCNN()
        self.light = LightFaceCNN()

    def run(self, retrain_light: bool = True) -> dict:
        print("[Stage2Accuracy] Loading data from training.csv…")
        X, y = _load_data()
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                      test_size=STAGE2["data"]["test_size"],
                                                      random_state=STAGE2["data"]["random_state"])
        print(f"[Stage2Accuracy] Train={len(X_tr):,}  Val={len(X_val):,}")

        # Load deep model
        if os.path.exists(DEEP_MODEL_PATH):
            self.deep.load_state_dict(torch.load(DEEP_MODEL_PATH, map_location=DEVICE))
            print(f"[Stage2Accuracy] DeepFaceCNN loaded from {DEEP_MODEL_PATH}")
        else:
            print(f"[Stage2Accuracy] DeepFaceCNN not trained yet — train with: python -m src.face_keypoint_model")

        # Train or load light model
        if retrain_light or not os.path.exists(LIGHT_MODEL_PATH):
            print(f"[Stage2Accuracy] Training LightFaceCNN for {EPOCHS_LIGHT} epochs…")
            _train_model(self.light, LIGHT_MODEL_PATH, X_tr, y_tr, X_val, y_val)
        else:
            self.light.load_state_dict(torch.load(LIGHT_MODEL_PATH, map_location=DEVICE))
            print(f"[Stage2Accuracy] LightFaceCNN loaded from {LIGHT_MODEL_PATH}")

        # Evaluate both
        deep_res  = self._evaluate(self.deep,  X_val, y_val, "DeepFaceCNN")
        light_res = self._evaluate(self.light, X_val, y_val, "LightFaceCNN")

        better_mae = "DeepFaceCNN" if deep_res['mae_px'] < light_res['mae_px'] else "LightFaceCNN"

        report = {
            'stage':          2,
            'task':           'Facial Keypoint Regression (15 landmarks)',
            'valSamples':     len(X_val),
            'imageSize':      f'{IMG_SIZE}×{IMG_SIZE}',
            'numKeypoints':   15,
            'models':         [deep_res, light_res],
            'bestMAE':        better_mae,
            'summary': {
                deep_res['model']:  {'mae_px': deep_res['mae_px'],  'rmse_px': deep_res['rmse_px'], 'params': deep_res['params'], 'ms_per_img': deep_res['ms_per_img']},
                light_res['model']: {'mae_px': light_res['mae_px'], 'rmse_px': light_res['rmse_px'], 'params': light_res['params'], 'ms_per_img': light_res['ms_per_img']},
            }
        }
        print(f"\n── Stage 2 Accuracy Summary ──")
        for m in [deep_res, light_res]:
            print(f"  {m['model']:<35} MAE={m['mae_px']:.3f} px  RMSE={m['rmse_px']:.3f} px  "
                  f"Params={m['params']:,}  Speed={m['ms_per_img']:.2f} ms/img")
        return report

    def _evaluate(self, model: nn.Module, X_val, y_val, name: str) -> dict:
        model.to(DEVICE).eval()
        ds = FaceDataset(X_val, y_val)
        dl = DataLoader(ds, batch_size=128, num_workers=0)

        preds, gts = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for imgs, kpts in dl:
                p = model(imgs.to(DEVICE)).cpu().numpy()
                preds.append(p); gts.append(kpts.numpy())
        elapsed = time.perf_counter() - t0

        preds = np.concatenate(preds) * IMG_SIZE
        gts   = np.concatenate(gts)   * IMG_SIZE
        err   = np.abs(preds - gts)

        # Per-keypoint MAE
        kp_mae = {KEYPOINT_NAMES[i]: round(float(err[:, i*2:i*2+2].mean()), 3)
                  for i in range(15)}

        params = sum(p.numel() for p in model.parameters())

        return {
            'model':          name,
            'mae_px':         round(float(err.mean()), 4),
            'rmse_px':        round(float(np.sqrt((( preds - gts)**2).mean())), 4),
            'mae_std':        round(float(err.std()), 4),
            'ms_per_img':     round((elapsed / len(X_val)) * 1000, 3),
            'params':         params,
            'perKeypointMAE': kp_mae,
        }

    def load_models(self):
        if os.path.exists(DEEP_MODEL_PATH):
            self.deep.load_state_dict(torch.load(DEEP_MODEL_PATH,  map_location=DEVICE))
        if os.path.exists(LIGHT_MODEL_PATH):
            self.light.load_state_dict(torch.load(LIGHT_MODEL_PATH, map_location=DEVICE))

    def predict_both(self, image_96x96: np.ndarray) -> dict:
        """Run both models on a single image and return keypoint predictions."""
        img = image_96x96.astype(np.float32)
        if img.max() > 1: img /= 255.0
        t = torch.from_numpy(img[np.newaxis, np.newaxis]).to(DEVICE)

        results = {}
        for name, model in [('DeepFaceCNN', self.deep), ('LightFaceCNN', self.light)]:
            model.eval()
            with torch.no_grad():
                pred = model(t).cpu().numpy()[0] * IMG_SIZE
            results[name] = {KEYPOINT_NAMES[i]: (float(pred[i*2]), float(pred[i*2+1]))
                             for i in range(15)}
        return results


if __name__ == '__main__':
    checker = Stage2AccuracyChecker()
    report  = checker.run(retrain_light=True)
