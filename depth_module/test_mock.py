"""
test_mock.py — tests the full depth module pipeline without a real checkpoint.

Replaces model.infer_image() with a synthetic depth map so we can verify
bbox cropping, normalization, proximity_label, and path_zone all work correctly.

Run from the repo root:
    python depth_module/test_mock.py
"""

import sys
import os
import numpy as np
import cv2

# Make sure depth_module is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_module import DepthModule, _normalize_depth, _proximity_label, _path_zone

# ---------------------------------------------------------------------------
# Build a synthetic 480x640 BGR image (a simple gradient + colored boxes)
# ---------------------------------------------------------------------------
H, W = 480, 640
image_bgr = np.zeros((H, W, 3), dtype=np.uint8)
# Background: dark grey
image_bgr[:] = (60, 60, 60)
# Draw a "person" box near center (red)
cv2.rectangle(image_bgr, (280, 150), (360, 400), (0, 0, 200), -1)
# Draw a "forklift" box at left edge (yellow)
cv2.rectangle(image_bgr, (30, 200), (150, 420), (0, 200, 200), -1)
# Draw a "box" in the far right (blue)
cv2.rectangle(image_bgr, (520, 300), (620, 440), (200, 100, 0), -1)

cv2.imwrite("depth_module/mock_image.png", image_bgr)
print("Saved mock_image.png")

# ---------------------------------------------------------------------------
# Synthetic depth map: gradient — left=close (low), right=far (high)
# This mimics a scene where objects on the left are closer.
# ---------------------------------------------------------------------------
depth_map = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32), (H, 1))
# Make center region slightly closer to exercise CLOSE threshold
depth_map[:, 280:360] = 0.08   # person bbox  → will be CLOSE
depth_map[:, 30:150]  = 0.25   # forklift bbox → will be CLOSE
depth_map[:, 520:640] = 0.80   # box bbox      → will be FAR

# ---------------------------------------------------------------------------
# Mock detections (as the Filtering stage would output)
# ---------------------------------------------------------------------------
detections = [
    {"image_id": 1, "category_id": 1, "bbox": [280, 150, 80, 250],  "score": 0.91, "label": "person"},
    {"image_id": 1, "category_id": 4, "bbox": [30,  200, 120, 220], "score": 0.85, "label": "forklift"},
    {"image_id": 1, "category_id": 7, "bbox": [520, 300, 100, 140], "score": 0.72, "label": "box"},
]

# ---------------------------------------------------------------------------
# Patch DepthModule to skip model loading and inject our synthetic depth map
# ---------------------------------------------------------------------------
dm = DepthModule.__new__(DepthModule)
dm.model = object()   # non-None sentinel so process() doesn't complain
dm.infer_depth = lambda img: depth_map   # inject synthetic depth

results = dm.process(image_bgr, detections)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'label':<12} {'depth_score':>12} {'proximity_label':>16} {'path_zone':>12}")
print("-" * 58)
for r in results:
    print(
        f"{r['label']:<12} "
        f"{r['depth_score']:>12.4f} "
        f"{r['proximity_label']:>16} "
        f"{r['path_zone']:>12}"
    )

# ---------------------------------------------------------------------------
# Assertions — verify expected behaviour
# ---------------------------------------------------------------------------
by_label = {r["label"]: r for r in results}

assert by_label["person"]["proximity_label"]   == "CLOSE",      "person should be CLOSE"
assert by_label["person"]["path_zone"]         == "CENTER",     "person should be CENTER"
assert by_label["forklift"]["proximity_label"] == "CLOSE",      "forklift should be CLOSE"
assert by_label["forklift"]["path_zone"]       == "PERIPHERAL", "forklift should be PERIPHERAL"
assert by_label["box"]["proximity_label"]      == "FAR",        "box should be FAR"
assert by_label["box"]["path_zone"]            == "PERIPHERAL", "box should be PERIPHERAL"

print("\nAll assertions passed.")

# Show what the decision module would see:
print("\nDecision-relevant summary:")
for r in results:
    risk_signal = f"{r['proximity_label']} + {r['path_zone']}"
    if r["proximity_label"] == "CLOSE" and r["path_zone"] == "CENTER":
        action = "=> STOP"
    elif r["proximity_label"] in ("CLOSE", "MEDIUM"):
        action = "=> SLOW"
    else:
        action = "=> CONTINUE"
    print(f"  {r['label']:<12} {risk_signal:<26} {action}")
