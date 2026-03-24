"""
eWardrobeAI — Stage 2: AI-Driven Facial Keypoint Detection Model
Research Component: CNN trained on the Facial Keypoints Detection dataset.
Framework: PyTorch

Architecture Overview:
  Input  : 96×96 grayscale face image
  Target : 15 facial landmark coordinates (30 values: x,y pairs)
  Model  : Multi-scale CNN with BatchNorm + Dropout
  Output : Normalised (x,y) coordinates in [0,1] space, rescaled to [0,96]

Keypoints detected (15 landmarks):
  left_eye_center, right_eye_center,
  left_eye_inner_corner, left_eye_outer_corner,
  right_eye_inner_corner, right_eye_outer_corner,
  left_eyebrow_inner_end, left_eyebrow_outer_end,
  right_eyebrow_inner_end, right_eyebrow_outer_end,
  nose_tip,
  mouth_left_corner, mouth_right_corner,
  mouth_center_top_lip, mouth_center_bottom_lip
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STAGE2

IMG_SIZE        = STAGE2["img_size"]
NUM_KEYPOINTS   = STAGE2["num_keypoints"]
BATCH_SIZE      = STAGE2["batch_size"]
EPOCHS          = STAGE2["epochs_deep"]
LEARNING_RATE   = STAGE2["learning_rate"]
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'training.csv')
MODEL_SAVE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'face_keypoint_cnn.pth')
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

KEYPOINT_NAMES = [col.rsplit('_', 1)[0] for col in KEYPOINT_COLS[::2]]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceKeypointDataset(Dataset):
    def __init__(self, images: np.ndarray, keypoints: np.ndarray, augment: bool = False):
        self.images    = images.astype(np.float32)
        self.keypoints = keypoints.astype(np.float32)
        self.augment   = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()    # (96, 96)
        kpt = self.keypoints[idx].copy() # (30,)

        if self.augment:
            # Horizontal flip (50 % probability)
            if np.random.rand() > 0.5:
                img = img[:, ::-1].copy()
                kpt[0::2] = 1.0 - kpt[0::2]  # mirror x-coords

            # Brightness jitter
            img *= np.random.uniform(0.85, 1.15)
            img  = np.clip(img, 0.0, 1.0)

        # Shape: (1, 96, 96) — channel-first for PyTorch
        img_tensor = torch.from_numpy(img[np.newaxis, :, :])
        kpt_tensor = torch.from_numpy(kpt)
        return img_tensor, kpt_tensor


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str = DATA_PATH):
    """
    Parse training.csv and return normalised (X_images, y_keypoints).
    Rows with any missing keypoint are dropped.
    """
    import pandas as pd
    print("[DataLoader] Reading CSV …")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=KEYPOINT_COLS).reset_index(drop=True)
    print(f"[DataLoader] {len(df):,} complete samples after NaN drop.")

    images = np.array(
        [np.fromstring(row, dtype=np.float32, sep=' ')
         for row in df['Image']],
        dtype=np.float32,
    ).reshape(-1, IMG_SIZE, IMG_SIZE) / 255.0

    keypoints = df[KEYPOINT_COLS].values.astype(np.float32) / IMG_SIZE

    print(f"[DataLoader] Images shape   : {images.shape}")
    print(f"[DataLoader] Keypoints shape: {keypoints.shape}")
    return images, keypoints


# ── Model Architecture ────────────────────────────────────────────────────────

class FaceKeypointCNN(nn.Module):
    """
    Multi-scale CNN for facial landmark regression.

    Feature extractor : 5 conv blocks with progressive filter widening
    Regression head   : Two FC layers with BatchNorm + Dropout
    Output activation : Sigmoid → maps naturally to [0, 1] normalised coords
    """

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 96×96 → 48×48
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 2: 48×48 → 24×24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 3: 24×24 → 12×12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 4: 12×12 → 6×6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 5: 6×6 (no pool — extra depth)
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(1024, 512),          nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_keypoints),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str = DATA_PATH,
          model_path: str = MODEL_SAVE_PATH,
          epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE):
    """
    End-to-end training routine:
      1. Load + preprocess data
      2. Train/validation split (85 / 15)
      3. Build model
      4. Train with cosine annealing LR, early stopping
      5. Save best model weights
      6. Plot training curves
    """
    from sklearn.model_selection import train_test_split
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"[Train] Device: {DEVICE}")

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y = load_data(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=STAGE2["data"]["test_size"], random_state=STAGE2["data"]["random_state"]
    )
    print(f"[Train] Train={len(X_train):,}  Val={len(X_val):,}")

    train_ds = FaceKeypointDataset(X_train, y_train, augment=True)
    val_ds   = FaceKeypointDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FaceKeypointCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {total_params:,}")

    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, kpts in train_loader:
            imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, kpts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for imgs, kpts in val_loader:
                imgs, kpts = imgs.to(DEVICE), kpts.to(DEVICE)
                preds     = model(imgs)
                val_loss += criterion(preds, kpts).item() * len(imgs)
                val_mae  += (preds - kpts).abs().mean().item() * len(imgs)
        val_loss /= len(val_ds)
        val_mae  /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        scheduler.step()

        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  "
              f"val_mae={val_mae:.5f}  "
              f"(~{val_mae * IMG_SIZE:.2f} px)")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model → {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    print(f"\n[Train] Best val_loss: {best_val_loss:.6f}")
    _plot_training_curves(history)
    return model, history


# ── Evaluation & Visualisation ────────────────────────────────────────────────

def _plot_training_curves(history: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('MSE Loss over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
    axes[0].legend()

    axes[1].plot([v * IMG_SIZE for v in history['val_mae']], label='Val MAE (px)')
    axes[1].set_title('Validation MAE (pixel units)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE (px)')
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(MODEL_SAVE_DIR, 'training_curves.png')
    plt.savefig(out, dpi=150)
    print(f"[Plot] Training curves saved → {out}")
    plt.close()


def evaluate_predictions(model, X_val: np.ndarray, y_val: np.ndarray,
                          n_samples: int = 8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    model.eval()
    imgs_t = torch.from_numpy(
        X_val[:n_samples, np.newaxis, :, :].astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():
        preds = model(imgs_t).cpu().numpy() * IMG_SIZE

    gt_px = y_val[:n_samples] * IMG_SIZE
    mae   = np.mean(np.abs(preds - gt_px), axis=1)
    print(f"\n[Eval] Pixel MAE — mean: {mae.mean():.3f} px  std: {mae.std():.3f} px")

    fig, axes = plt.subplots(2, n_samples // 2, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(X_val[i], cmap='gray')
        axes[i].scatter(gt_px[i, 0::2],   gt_px[i, 1::2],   s=14, c='lime', marker='o')
        axes[i].scatter(preds[i, 0::2],   preds[i, 1::2],   s=14, c='red',  marker='x')
        axes[i].set_title(f'MAE: {mae[i]:.2f} px', fontsize=9)
        axes[i].axis('off')

    gt_patch   = mpatches.Patch(color='lime', label='Ground Truth')
    pred_patch = mpatches.Patch(color='red',  label='Predicted')
    fig.legend(handles=[gt_patch, pred_patch], loc='lower right', fontsize=11)
    fig.suptitle('Facial Keypoint Detection — Validation Samples', fontsize=14)
    plt.tight_layout()

    out = os.path.join(MODEL_SAVE_DIR, 'keypoint_predictions.png')
    plt.savefig(out, dpi=150)
    print(f"[Plot] Prediction overlay saved → {out}")
    plt.close()


# ── Inference Helpers ─────────────────────────────────────────────────────────

def predict_single_image(model, image_array: np.ndarray) -> dict:
    """
    Given a 96×96 grayscale image (uint8 or float32),
    returns {keypoint_name: (x_px, y_px)}.
    """
    model.eval()
    img = image_array.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    img_t = torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).to(DEVICE)

    with torch.no_grad():
        pred = model(img_t).cpu().numpy()[0] * IMG_SIZE

    result = {}
    for idx, name in enumerate(KEYPOINT_NAMES):
        result[name] = (float(pred[idx * 2]), float(pred[idx * 2 + 1]))
    return result


def load_trained_model(model_path: str = MODEL_SAVE_PATH) -> FaceKeypointCNN:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run train() first: python -m src.face_keypoint_model"
        )
    print(f"[Model] Loading from {model_path} …")
    model = FaceKeypointCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    model, history = train()

    X, y = load_data(DATA_PATH)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    evaluate_predictions(model, X_val, y_val)
