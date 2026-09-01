"""
Download the MediaPipe FaceLandmarker model file required by the avatar
pipeline's face-feature extraction (avatar_pipeline/model6_body3d/face_features.py).

The model is ~3.7 MB and is placed in the project root's `models/` directory,
where face_features.py will find it automatically.

Usage:
    python scripts/download_mediapipe_models.py

If you have a proxy / firewall, set HTTP_PROXY / HTTPS_PROXY env vars, or
download the file manually from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
and save it as:
    <project_root>/models/face_landmarker_v2.task
"""

import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker_v2.task"
# ---------------------------------------------------------------------------


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # scripts/ is one level down
    models_dir = os.path.join(project_root, "models")
    dest_path = os.path.join(models_dir, MODEL_FILENAME)

    if os.path.exists(dest_path):
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"[✓] Model already exists: {dest_path} ({size_kb:.0f} KB)")
        print("    Delete the file first if you want to re-download.")
        return

    os.makedirs(models_dir, exist_ok=True)

    print(f"Downloading MediaPipe FaceLandmarker model...")
    print(f"  From: {MODEL_URL}")
    print(f"  To:   {dest_path}")

    try:
        urllib.request.urlretrieve(MODEL_URL, dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"\n[✓] Done! ({size_kb:.0f} KB saved)")
    except Exception as e:
        print(f"\n[✗] Download failed: {e}")
        print("\nTry downloading manually from:")
        print(f"  {MODEL_URL}")
        print(f"And save as: {dest_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
