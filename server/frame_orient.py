"""Client-frame orientation fix.

Android sends already-upright portrait frames (the native encoder rotates by
sensorOrientation). iOS sends the raw camera buffer, which is landscape and
needs a 90° rotation to become the upright portrait the classifier was
trained on -- doing it in the iOS CoreImage layer proved unreliable, so we
normalise here instead.

Heuristic: only landscape frames (width > height) are rotated, so Android's
portrait frames pass through untouched. Direction is env-tunable so it can be
flipped without a code change (LANDSCAPE_ROTATE = cw | ccw | off).
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def orient_frame(bgr: np.ndarray) -> np.ndarray:
    if bgr is None:
        return bgr
    h, w = bgr.shape[:2]
    if w <= h:
        return bgr  # already portrait (Android) -> leave as-is
    mode = os.getenv("LANDSCAPE_ROTATE", "cw").lower()
    if mode == "off":
        return bgr
    if mode == "ccw":
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
