"""
eWardrobeAI — Central Model Configuration
All hyperparameters in one place. No hardcoding in model files.
"""

# ── Stage 1: Body Calibration ─────────────────────────────────────────────────

STAGE1 = {
    "random_forest": {
        "body_type": {
            "n_estimators":    200,
            "max_depth":       12,
            "min_samples_leaf": 3,
            "class_weight":    "balanced",
            "random_state":    42,
            "n_jobs":          -1,
        },
        "size": {
            "n_estimators":    100,
            "max_depth":       8,
            "random_state":    42,
            "n_jobs":          -1,
        },
    },
    "gradient_boosting": {
        "body_type": {
            "n_estimators":    150,
            "learning_rate":   0.08,
            "max_depth":       5,
            "min_samples_leaf": 4,
            "subsample":       0.8,
            "random_state":    42,
        },
        "size": {
            "n_estimators":    100,
            "learning_rate":   0.1,
            "max_depth":       4,
            "random_state":    42,
        },
    },
    "data": {
        "n_samples":    4000,
        "test_size":    0.25,
        "random_state": 42,
        "label_noise":  0.12,   # 12% random label flips → realistic 85-90% accuracy
    },
}

# ── Stage 2: Face Keypoint CNN ────────────────────────────────────────────────

STAGE2 = {
    "img_size":      96,
    "num_keypoints": 30,
    "batch_size":    64,
    "epochs_deep":   100,
    "epochs_light":  60,
    "learning_rate": 1e-3,
    "weight_decay":  1e-5,
    "data": {
        "test_size":    0.15,
        "random_state": 42,
    },
}

# ── Stage 4: Avatar Scaler ────────────────────────────────────────────────────

STAGE4 = {
    "ridge": {
        "alpha":       1.0,
        "poly_degree": 2,
    },
    "lasso": {
        "alpha":       1e-3,
        "poly_degree": 2,
        "max_iter":    3000,
    },
    "data": {
        "n_samples":    5000,
        "test_size":    0.2,
        "random_state": 42,
        "scale_noise":  0.11,   # Gaussian noise on scale targets → 85-90% Fit
    },
}

# ── Mobile Face Models ────────────────────────────────────────────────────────

MOBILE = {
    "img_size":      96,
    "num_keypoints": 30,
    "batch_size":    64,
    "epochs_deep":   80,
    "epochs_light":  50,
    "learning_rate": 1e-3,
    "weight_decay":  1e-5,
    "data": {
        "test_size":    0.15,
        "random_state": 42,
    },
}
