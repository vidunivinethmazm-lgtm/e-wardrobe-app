"""Lightweight face detection + crop using OpenCV's bundled Haar cascade.

A heavier/more accurate detector (e.g. MediaPipe Face Detection or the
BlazeFace model behind Model 2's MoveNet) can be swapped in here without
changing any other Model 3 code — `detect_and_crop_face` is the only
integration point.
"""

import cv2

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_crop_face(image_rgb, output_size=128, margin=0.2):
    """Returns an `output_size`x`output_size` RGB face crop, or None if no
    face is detected."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(image_rgb.shape[1], x + w + mx), min(image_rgb.shape[0], y + h + my)

    crop = image_rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (output_size, output_size))
