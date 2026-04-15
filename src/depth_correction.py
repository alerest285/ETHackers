"""
Depth Correction via Bounding Box Size Heuristics.

Depth Anything V2 produces relative depth, which can drift in scenes with
unusual perspective or lighting.  This module corrects the normalized depth
map by blending each detected object's bbox region toward a depth value
implied by how large that bbox is relative to the full image.

Physical basis
--------------
In perspective projection a rigid object's projected area scales as 1/d²
(d = distance to camera), so:

    sqrt(relative_area) ∝ 1/d ∝ proximity

That gives us a geometry-grounded estimate of how close each detected
object should be, independent of pixel intensities.

Correction model
----------------
For each detection bbox:

    corrected_roi = (1 − α) · model_roi  +  α · implied_depth

where implied_depth ∈ [0, 1]  (1 = very close, 0 = very far)
and α (correction strength) scales with:
  • How extreme the relative area is  (very large / very small → high confidence)
  • How much the model reading contradicts the size heuristic (big gap → correct more)

Overlapping bboxes are applied largest-first so that smaller foreground
objects (processed last) refine on top of background corrections.
"""

import numpy as np

# ── Calibration constants ──────────────────────────────────────────────────────
# An object that occupies 25 % of the image is treated as "fully close" (score 1.0).
# An object that occupies 0.1 % of the image maps to score 0.0 (very far).
_REF_CLOSE_AREA = 0.25
_REF_FAR_AREA   = 0.001

# Bounding box size thresholds that define "extreme" (high-confidence) cases
_LARGE_AREA_THRESH = 0.12   # > 12 % of image → strong close signal
_SMALL_AREA_THRESH = 0.005  # < 0.5 % of image → strong far signal

# Hard cap: never let the heuristic fully override the depth model
_MAX_ALPHA = 0.55


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_bbox_implied_depth(relative_area: float) -> float:
    """
    Map a detection's relative bounding box area to an implied proximity
    score in [0, 1], where 1 = very close and 0 = very far.

    Uses a linear interpolation in sqrt-space, grounded in the perspective
    projection relationship  area ∝ 1/d².
    """
    sqrt_area  = np.sqrt(np.clip(relative_area, 0.0, 1.0))
    sqrt_close = np.sqrt(_REF_CLOSE_AREA)   # ≈ 0.500
    sqrt_far   = np.sqrt(_REF_FAR_AREA)     # ≈ 0.032

    t = (sqrt_area - sqrt_far) / (sqrt_close - sqrt_far + 1e-9)
    return float(np.clip(t, 0.0, 1.0))


def correct_depth_map(
    norm_depth:  np.ndarray,   # (H, W) float32, [0,1], 1 = close / 0 = far
    detections:  list[dict],   # YOLO dets with 'box' [x1,y1,x2,y2] + 'relative_area'
    image_h:     int,
    image_w:     int,
) -> np.ndarray:
    """
    Return a corrected copy of *norm_depth*.

    Each detection's bbox region is blended toward the depth implied by that
    bbox's size relative to the full image.  Regions without any detection
    are left unchanged.

    Parameters
    ----------
    norm_depth  : Normalized depth map from Depth Anything V2.
                  Convention: 1.0 = closest pixel, 0.0 = farthest pixel.
    detections  : List of detection dicts; must contain 'box' [x1,y1,x2,y2].
                  If 'relative_area' is present it is reused; otherwise
                  recomputed from the box and image dimensions.
    image_h, image_w : Image dimensions in pixels.

    Returns
    -------
    corrected : (H, W) float32 array, same range as input.
    """
    corrected  = norm_depth.copy()
    image_area = float(image_h * image_w)

    # Largest bboxes first → smaller (foreground) detections refine on top
    sorted_dets = sorted(
        detections,
        key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
        reverse=True,
    )

    for det in sorted_dets:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(image_w, x2);  y2 = min(image_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        rel_area = det.get(
            "relative_area",
            ((x2 - x1) * (y2 - y1)) / image_area,
        )

        implied       = compute_bbox_implied_depth(rel_area)
        roi           = corrected[y1:y2, x1:x2]
        model_median  = float(np.median(roi))
        contradiction = abs(model_median - implied)   # 0..1

        alpha = _compute_alpha(rel_area, contradiction)
        corrected[y1:y2, x1:x2] = (1.0 - alpha) * roi + alpha * implied

    return corrected


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_alpha(relative_area: float, contradiction: float) -> float:
    """
    Blend weight for the bbox-size heuristic correction.

    Base weight is higher when the bbox size is extreme (very large or very
    small), and the base is further scaled by how much the model disagrees
    with the geometric expectation.
    """
    if relative_area >= _LARGE_AREA_THRESH:
        base = 0.40   # large object → strong confidence it is close
    elif relative_area <= _SMALL_AREA_THRESH:
        base = 0.30   # tiny object  → moderate confidence it is far
    else:
        # Smooth U-shape in log-area space:
        # low base (≈ 0.05) for ambiguous mid-sized objects,
        # rising toward 0.20 as area approaches either threshold.
        log_area  = np.log(max(relative_area, 1e-12))
        log_large = np.log(_LARGE_AREA_THRESH)
        log_small = np.log(_SMALL_AREA_THRESH)
        t    = (log_area - log_small) / (log_large - log_small)  # 0..1
        base = 0.05 + 0.15 * (1.0 - 4.0 * (t - 0.5) ** 2)       # peak 0.05, edges 0.20

    # Additional weight proportional to how wrong the model looks
    alpha = base + 0.20 * contradiction
    return min(alpha, _MAX_ALPHA)
