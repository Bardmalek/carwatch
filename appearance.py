"""Dominant paint colour of a vehicle, as a plain word. Used only to label boxes."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

# BGR swatch used for the small colour chip next to the label
SWATCH = {"red": (40, 40, 220), "orange": (0, 140, 255), "yellow": (0, 220, 240),
          "green": (60, 180, 60), "blue": (200, 90, 30), "purple": (170, 60, 140),
          "brown": (40, 70, 110), "white": (245, 245, 245), "silver": (190, 190, 190),
          "gray": (120, 120, 120), "black": (20, 20, 20)}


def _classify(hsv: np.ndarray) -> np.ndarray:
    """hsv (N,3) OpenCV scale -> array of colour names."""
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    out = np.full(len(h), "gray", dtype=object)
    chroma = s >= 60
    out[chroma & ((h < 8) | (h >= 165))] = "red"
    out[chroma & (h >= 8) & (h < 20)] = "orange"
    out[chroma & (h >= 8) & (h < 20) & (v < 130)] = "brown"
    out[chroma & (h >= 20) & (h < 35)] = "yellow"
    out[chroma & (h >= 35) & (h < 85)] = "green"
    out[chroma & (h >= 85) & (h < 135)] = "blue"
    out[chroma & (h >= 135) & (h < 165)] = "purple"
    grey = ~chroma
    out[grey & (v < 55)] = "black"
    out[grey & (v >= 55) & (v < 135)] = "gray"
    out[grey & (v >= 135) & (v < 185)] = "silver"
    out[grey & (v >= 185)] = "white"
    out[(v < 40)] = "black"                       # very dark is black whatever the hue
    return out


def color_name(frame: np.ndarray, xyxy) -> str | None:
    """Mode of per-pixel colour names over the inner part of the box (skips road,
    edges and most glass). Majority over pixels resists glare and shadows."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 8:
        return None
    cx1, cx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    cy1, cy2 = y1 + int(0.3 * h), y2 - int(0.15 * h)
    fh, fw = frame.shape[:2]
    crop = frame[max(cy1, 0):min(cy2, fh), max(cx1, 0):min(cx2, fw)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return Counter(_classify(hsv)).most_common(1)[0][0]
