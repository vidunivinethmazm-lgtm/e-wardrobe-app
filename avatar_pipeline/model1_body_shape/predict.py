"""
Model 1 — Body Shape Estimation: inference.

CLI usage:
    python -m avatar_pipeline.model1_body_shape.predict \
        --bust 92 --waist 70 --hips 98 --height 165 \
        --silhouette path/to/silhouette.png

`predict_body_shape` is also the function the integration controller
(Step 4 of the pipeline) imports directly.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .architecture import CLASS_NAMES
from .data_pipeline import ENGINEERED_COLUMNS, add_engineered_features


def load_artifacts(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, "best_model.keras"))
    scaler = joblib.load(os.path.join(model_dir, "measurement_scaler.joblib"))
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return model, scaler, config


def preprocess_measurements(bust, waist, hips, height, scaler):
    df = pd.DataFrame([{"bust": bust, "waist": waist, "hips": hips, "height": height}])
    df = add_engineered_features(df)
    return scaler.transform(df[ENGINEERED_COLUMNS].values).astype("float32")


def preprocess_silhouette(image_path, image_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (image_size, image_size))
    return tf.expand_dims(img, axis=0)


def predict_body_shape(model_dir, bust, waist, hips, height, silhouette_path=None, model=None, scaler=None, config=None):
    if model is None or scaler is None or config is None:
        model, scaler, config = load_artifacts(model_dir)

    meas = preprocess_measurements(bust, waist, hips, height, scaler)

    if config["model_type"] == "fusion":
        if silhouette_path is None:
            raise ValueError("This model requires a silhouette image (config.model_type == 'fusion').")
        img = preprocess_silhouette(silhouette_path, config["image_size"])
        probs = model.predict({"measurements": meas, "silhouette": img}, verbose=0)[0]
    else:
        probs = model.predict(meas, verbose=0)[0]

    idx = int(np.argmax(probs))
    return {
        "body_shape": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_models/model1_body_shape")
    parser.add_argument("--bust", type=float, required=True)
    parser.add_argument("--waist", type=float, required=True)
    parser.add_argument("--hips", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--silhouette", default=None)
    args = parser.parse_args()

    result = predict_body_shape(
        args.model_dir, args.bust, args.waist, args.hips, args.height, args.silhouette
    )
    print(json.dumps(result, indent=2))
