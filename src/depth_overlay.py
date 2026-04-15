"""
Depth Overlay

Takes a raw image + YOLO detections + depth map and produces a single
visualization that combines both:

  - Left panel  : original image with YOLO boxes + labels
  - Right panel : depth heatmap with the same boxes drawn on top,
                  each box labelled with its proximity (CLOSE/MEDIUM/FAR)
                  and depth score

Color coding on boxes (both panels):
  CLOSE  → red
  MEDIUM → orange
  FAR    → green
  (uses the depth_score from depth_module, not the YOLO risk_group)
"""

import numpy as np
import cv2
from PIL import Image


# Proximity thresholds (must match depth_module.py)
THRESHOLDS = {"CLOSE": 0.35, "MEDIUM": 0.65}

PROXIMITY_COLORS = {
    "CLOSE":  (0,   0,   255),   # red   (BGR)
    "MEDIUM": (0,   140, 255),   # orange
    "FAR":    (0,   200, 80),    # green
}


def _proximity(depth_score: float) -> str:
    if depth_score <= THRESHOLDS["CLOSE"]:
        return "CLOSE"
    if depth_score <= THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "FAR"


def make_overlay(
    image: Image.Image,
    detections: list[dict],
    depth_map: np.ndarray,
) -> np.ndarray:
    """
    Build a side-by-side BGR image: [YOLO boxes | depth heatmap + boxes].

    Args:
        image:      Original PIL image (RGB).
        detections: List of dicts with keys:
                      box         [x1, y1, x2, y2]
                      label       str
                      score       float
                      depth_score float  (added by depth_module)
        depth_map:  (H, W) float32 array, raw depth values.

    Returns:
        BGR numpy array ready for cv2.imwrite.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # Normalize depth to uint8 and apply colormap
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-6:
        depth_u8 = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth_map, dtype=np.uint8)
    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_colored = cv2.resize(depth_colored, (W, H))

    left  = img_bgr.copy()
    right = depth_colored.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        depth_score = det.get("depth_score", 0.5)
        prox        = det.get("proximity_label") or _proximity(depth_score)
        color       = PROXIMITY_COLORS.get(prox, (200, 200, 200))

        label = det.get("label", "?")
        score = det.get("score", 0)

        # ── Left: YOLO box + label + proximity ───────────────────────────────
        cv2.rectangle(left, (x1, y1), (x2, y2), color, 2)
        text_left = f"{label} {score:.2f} | {prox}"
        cv2.rectangle(left, (x1, y1 - 16), (x1 + len(text_left) * 8, y1), color, -1)
        cv2.putText(left, text_left, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Right: same box on depth heatmap + depth score ───────────────────
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 2)
        text_right = f"{prox} {depth_score:.2f}"
        cv2.rectangle(right, (x1, y1 - 16), (x1 + len(text_right) * 8, y1), color, -1)
        cv2.putText(right, text_right, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Divider line
    divider = np.full((H, 4, 3), 60, dtype=np.uint8)
    return np.hstack([left, divider, right])
