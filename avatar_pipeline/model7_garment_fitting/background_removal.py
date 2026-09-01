"""
Model 7 — background removal for a garment product photo.

Phase 1 implementation: deterministic OpenCV GrabCut seeded from a
border-vs-center heuristic (product photos are near-universally shot on a
plain/light backdrop with the garment centered and filling most of the
frame). No ML model, no network call, fully offline.

Interface is intentionally narrow (`remove_background(rgb) -> mask`) so a
learned matting model (e.g. rembg/U^2-Net) can replace the implementation
later without touching any downstream stage — `garment_segmentation.py`
only depends on this function's signature.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_background(rgb: np.ndarray) -> np.ndarray:
    """Returns a boolean HxW mask, True where `rgb` (HxWx3 uint8) is judged to
    be the foreground garment rather than background.

    Uses GrabCut initialized with a rectangle covering the central ~86% of
    the frame (product photos rarely crop the garment right at the edge),
    refined with a probable-background border strip. Falls back to a plain
    Otsu threshold against the estimated background color if GrabCut fails to
    converge to a non-trivial mask (e.g. a near-uniform image).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {rgb.shape}")

    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        raise ValueError(f"image too small to segment ({w}x{h})")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    margin_x, margin_y = int(w * 0.07), int(h * 0.07)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
        fg_mask = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    except cv2.error:
        fg_mask = np.zeros((h, w), dtype=bool)

    coverage = fg_mask.mean()
    if coverage < 0.01 or coverage > 0.99:
        fg_mask = _fallback_threshold_mask(bgr)

    fg_mask = _clean_mask(fg_mask)
    return fg_mask


def _fallback_threshold_mask(bgr: np.ndarray) -> np.ndarray:
    """Otsu threshold on grayscale distance from the estimated (border-
    sampled) background color — used only if GrabCut degenerates."""
    h, w = bgr.shape[:2]
    border = np.concatenate([
        bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :],
    ], axis=0)
    bg_color = np.median(border, axis=0)

    dist = np.linalg.norm(bgr.astype(np.float32) - bg_color, axis=2)
    dist_u8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh.astype(bool)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological close + largest-connected-component filter, so stray
    background specks/holes don't corrupt downstream keypoint extraction."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = labels == largest_label

    filled = cleaned.astype(np.uint8) * 255
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(filled)
        cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cleaned = filled.astype(bool)

    return cleaned
