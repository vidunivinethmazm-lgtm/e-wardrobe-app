"""
eWardrobeAI Mobile — Face Detection Models
Trained on training.csv (Facial Keypoints Detection dataset)

Model 1 — DeepFaceCNN  : 5 conv blocks, ~4.5M params  → high accuracy
Model 2 — LightFaceCNN : 3 conv blocks, ~1.2M params  → fast inference

Both detect 15 facial landmarks (30 x,y coordinates) from a 96×96 grayscale image.
Accuracy is measured as Mean Absolute Error (MAE) in pixels on the validation split.
"""

from __future__ import annotations
import os, time, csv
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_PATH     = os.path.join(ROOT, 'training.csv')
MODEL_DIR     = os.path.join(ROOT, 'models')
DEEP_PATH     = os.path.join(MODEL_DIR, 'mobile_deep_face.pth')
LIGHT_PATH    = os.path.join(MODEL_DIR, 'mobile_light_face.pth')
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.insert(0, ROOT)
from config import MOBILE

IMG_SIZE      = MOBILE["img_size"]
NUM_KP        = MOBILE["num_keypoints"]
BATCH         = MOBILE["batch_size"]

KEYPOINT_COLS = [
    'left_eye_center_x','left_eye_center_y',
    'right_eye_center_x','right_eye_center_y',
    'left_eye_inner_corner_x','left_eye_inner_corner_y',
    'left_eye_outer_corner_x','left_eye_outer_corner_y',
    'right_eye_inner_corner_x','right_eye_inner_corner_y',
    'right_eye_outer_corner_x','right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x','left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x','right_eyebrow_outer_end_y',
    'nose_tip_x','nose_tip_y',
    'mouth_left_corner_x','mouth_left_corner_y',
    'mouth_right_corner_x','mouth_right_corner_y',
    'mouth_center_top_lip_x','mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y',
]
KP_NAMES = [c.rsplit('_',1)[0] for c in KEYPOINT_COLS[::2]]


# ── Dataset ────────────────────────────────────────────────────────────────

class FaceKPDataset(Dataset):
    def __init__(self, images, keypoints, augment=False):
        self.X = images.astype(np.float32)
        self.y = keypoints.astype(np.float32)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        img = self.X[i].copy()
        kpt = self.y[i].copy()
        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1].copy()
            kpt[0::2] = 1.0 - kpt[0::2]
        img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1) if self.augment else img
        return torch.from_numpy(img[np.newaxis]), torch.from_numpy(kpt)


def load_csv_data():
    """Read training.csv using built-in csv module — no pandas needed."""
    imgs_list, kpts_list = [], []
    with open(DATA_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with any missing keypoint
            try:
                kpt_vals = [float(row[col]) for col in KEYPOINT_COLS]
            except (ValueError, KeyError):
                continue
            imgs_list.append(np.fromstring(row['Image'], dtype=np.float32, sep=' '))
            kpts_list.append(kpt_vals)

    imgs = np.array(imgs_list).reshape(-1, IMG_SIZE, IMG_SIZE) / 255.0
    kpts = np.array(kpts_list, dtype=np.float32) / IMG_SIZE
    return imgs, kpts


# ── Model 1: Deep Face CNN ─────────────────────────────────────────────────

class DeepFaceCNN(nn.Module):
    """5 conv blocks → 1024+512 FC → Sigmoid. High accuracy."""
    name = "DeepFaceCNN"
    description = "5-block CNN, ~4.5M params — high accuracy"

    def __init__(self):
        super().__init__()
        def block(ci, co, pool=True):
            layers = [nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(True)]
            if pool: layers.append(nn.MaxPool2d(2))
            return layers

        self.features = nn.Sequential(
            *block(1,  32),   # 48×48
            *block(32, 64),   # 24×24
            *block(64, 128),  # 12×12
            *block(128,256),  #  6×6
            *block(256,256, pool=False),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 1024), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(1024, 512),     nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, NUM_KP),   nn.Sigmoid(),
        )

    def forward(self, x): return self.head(self.features(x))

    @property
    def param_count(self): return sum(p.numel() for p in self.parameters())


# ── Model 2: Light Face CNN ────────────────────────────────────────────────

class LightFaceCNN(nn.Module):
    """3 conv blocks → 512+256 FC → Sigmoid. Fast inference."""
    name = "LightFaceCNN"
    description = "3-block CNN, ~1.2M params — fast inference"

    def __init__(self):
        super().__init__()
        def block(ci, co):
            return [nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co),
                    nn.ReLU(True), nn.MaxPool2d(2)]

        self.features = nn.Sequential(
            *block(1,  32),   # 48×48
            *block(32, 64),   # 24×24
            *block(64, 128),  # 12×12
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*12*12, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, NUM_KP),    nn.Sigmoid(),
        )

    def forward(self, x): return self.head(self.features(x))

    @property
    def param_count(self): return sum(p.numel() for p in self.parameters())


# ── Training ───────────────────────────────────────────────────────────────

def _train(model, save_path, X_tr, y_tr, X_val, y_val, epochs=60):
    model.to(DEVICE)
    dl_tr  = DataLoader(FaceKPDataset(X_tr, y_tr, augment=True),
                        batch_size=BATCH, shuffle=True,  num_workers=0)
    dl_val = DataLoader(FaceKPDataset(X_val, y_val),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    opt   = optim.Adam(model.parameters(), lr=MOBILE["learning_rate"], weight_decay=MOBILE["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()
    best  = float('inf'); patience = 0; PATIENCE = 12

    for ep in range(1, epochs+1):
        model.train()
        for imgs, kpts in dl_tr:
            imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
            opt.zero_grad(); crit(model(imgs), kpts).backward(); opt.step()
        sched.step()

        model.eval(); vl = vm = 0
        with torch.no_grad():
            for imgs, kpts in dl_val:
                imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
                p  = model(imgs)
                vl += crit(p, kpts).item() * len(imgs)
                vm += (p - kpts).abs().mean().item() * len(imgs)
        vl /= len(X_val); vm /= len(X_val)

        if ep % 10 == 0 or ep == 1:
            print(f"    [{model.name}] Epoch {ep:03d}/{epochs}  "
                  f"val_loss={vl:.5f}  mae={vm*IMG_SIZE:.2f}px")

        if vl < best:
            best = vl; patience = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"    [{model.name}] Early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model


# ── Accuracy Result ────────────────────────────────────────────────────────

@dataclass
class FaceAccuracyResult:
    model_name:    str
    description:   str
    param_count:   int
    mae_px:        float
    rmse_px:       float
    mae_std:       float
    per_kp_mae:    dict
    inference_ms:  float
    keypoints:     dict = field(default_factory=dict)  # on the uploaded image


# ── FaceModelRunner ────────────────────────────────────────────────────────

class FaceModelRunner:
    """
    Loads (or trains) both face CNN models.
    Runs inference on an uploaded image.
    Reports accuracy on the training.csv validation split.
    """

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.deep  = DeepFaceCNN().to(DEVICE)
        self.light = LightFaceCNN().to(DEVICE)
        self._val_X = self._val_y = None

    # ── Train ──────────────────────────────────────────────────────────────

    def train(self, epochs_deep=MOBILE["epochs_deep"], epochs_light=MOBILE["epochs_light"]):
        print("\n  Loading training.csv …")
        X, y = load_csv_data()
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=MOBILE["data"]["test_size"], random_state=MOBILE["data"]["random_state"])
        self._val_X, self._val_y = X_val, y_val
        print(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Device={DEVICE}")

        print(f"\n  ── Training DeepFaceCNN ({self.deep.param_count:,} params) ──")
        _train(self.deep,  DEEP_PATH,  X_tr, y_tr, X_val, y_val, epochs_deep)

        print(f"\n  ── Training LightFaceCNN ({self.light.param_count:,} params) ──")
        _train(self.light, LIGHT_PATH, X_tr, y_tr, X_val, y_val, epochs_light)

        self._val_X, self._val_y = X_val, y_val

    def load(self):
        if os.path.exists(DEEP_PATH):
            self.deep.load_state_dict(torch.load(DEEP_PATH, map_location=DEVICE))
        if os.path.exists(LIGHT_PATH):
            self.light.load_state_dict(torch.load(LIGHT_PATH, map_location=DEVICE))

    # ── Evaluate on validation split ──────────────────────────────────────

    def evaluate(self) -> list[FaceAccuracyResult]:
        if self._val_X is None:
            X, y = load_csv_data()
            _, self._val_X, _, self._val_y = train_test_split(
                X, y, test_size=MOBILE["data"]["test_size"], random_state=MOBILE["data"]["random_state"])
        results = []
        for model, path, ep_d, ep_l in [
            (self.deep,  DEEP_PATH,  None, None),
            (self.light, LIGHT_PATH, None, None),
        ]:
            results.append(self._eval_model(model))
        return results

    def _eval_model(self, model: nn.Module) -> FaceAccuracyResult:
        model.eval()
        ds  = FaceKPDataset(self._val_X, self._val_y)
        dl  = DataLoader(ds, batch_size=128, num_workers=0)
        all_pred, all_gt = [], []

        t0 = time.perf_counter()
        with torch.no_grad():
            for imgs, kpts in dl:
                all_pred.append(model(imgs.to(DEVICE)).cpu().numpy())
                all_gt.append(kpts.numpy())
        elapsed = time.perf_counter() - t0

        pred = np.concatenate(all_pred) * IMG_SIZE
        gt   = np.concatenate(all_gt)   * IMG_SIZE
        err  = np.abs(pred - gt)
        per_kp = {KP_NAMES[i]: round(float(err[:, i*2:i*2+2].mean()), 3)
                  for i in range(15)}

        return FaceAccuracyResult(
            model_name   = model.name,
            description  = model.description,
            param_count  = model.param_count,
            mae_px       = round(float(err.mean()), 3),
            rmse_px      = round(float(np.sqrt((pred-gt)**2).mean()), 3),
            mae_std      = round(float(err.std()), 3),
            per_kp_mae   = per_kp,
            inference_ms = round((elapsed / len(self._val_X)) * 1000, 3),
        )

    # ── Predict on single image ────────────────────────────────────────────

    def predict(self, image_96x96_gray: np.ndarray) -> list[FaceAccuracyResult]:
        """Run both models on one image, return keypoint predictions."""
        img = image_96x96_gray.astype(np.float32)
        if img.max() > 1.0: img /= 255.0
        t   = torch.from_numpy(img[np.newaxis, np.newaxis]).to(DEVICE)

        results = []
        for model in (self.deep, self.light):
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(t).cpu().numpy()[0] * IMG_SIZE
            ms = (time.perf_counter() - t0) * 1000
            kp = {KP_NAMES[i]: (float(pred[i*2]), float(pred[i*2+1]))
                  for i in range(15)}
            res = FaceAccuracyResult(
                model_name   = model.name,
                description  = model.description,
                param_count  = model.param_count,
                mae_px       = 0.0, rmse_px=0.0, mae_std=0.0,
                per_kp_mae   = {},
                inference_ms = round(ms, 3),
                keypoints    = kp,
            )
            results.append(res)
        return results

    def models_trained(self) -> bool:
        return os.path.exists(DEEP_PATH) and os.path.exists(LIGHT_PATH)
