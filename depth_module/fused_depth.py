"""
Phase 2: Triple-Fused Depth

Computes three independent depth estimates per detected object, then fuses
them into a single depth_score.

  Input 1 — DepthAnything V2      → depth_da
      Median normalized depth inside the detection bbox.

  Input 2 — Real-world size ref   → depth_rw
      Pinhole-camera estimate: distance_m = (ref_height * focal_px) / bbox_h_px
      Log-normalized to [0, 1].  Falls back to None for unknown object types.

  Input 3 — Relative bbox area    → depth_area
      depth_area = 1 - clamp(bbox_area / frame_area / 0.25, 0, 1)
      Large bbox → low depth_area (close); small bbox → high depth_area (far).

  Fusion:
      depth_score = W_DA * depth_da + W_RW * depth_rw + W_AREA * depth_area
      Weights default to 0.6 / 0.3 / 0.1 and are tunable via env vars:
        DEPTH_W_DA, DEPTH_W_RW, DEPTH_W_AREA

Convention throughout: depth ∈ [0, 1], where 0 = closest, 1 = farthest.

Proximity thresholds:
  CLOSE   depth_score <= 0.35
  MEDIUM  depth_score <= 0.65
  FAR     depth_score >  0.65

Path zone:
  CENTER      bbox x-centre falls in the middle third of the image
  PERIPHERAL  otherwise
"""

import os
import numpy as np

# ── Real-world reference heights (metres) ─────────────────────────────────────
# Keyed by risk_group from label_ontology.
REFERENCE_HEIGHTS_M: dict[str, float] = {
    "HUMAN":         1.70,   # average standing person / worker
    "VEHICLE":       2.50,   # forklift / truck / bus; car uses height not length
    "OBSTACLE":      0.90,   # barrel / crate / box average
    "SAFETY_MARKER": 0.75,   # traffic cone / sign post
}

# Log-scale anchor points for depth_rw normalisation
_D_NEAR_M = 0.5    # ≤ 0.5 m  →  depth_rw = 0.0  (very close)
_D_FAR_M  = 20.0   # ≥ 20 m   →  depth_rw = 1.0  (very far)

# Area threshold: bbox covering this fraction of the frame = "fully close"
_MAX_AREA_RATIO = 0.25

# Area at which geometric signals are considered fully confident for adaptive weighting
# (lower than _MAX_AREA_RATIO — a 12% bbox is already "clearly notable")
_GEOM_CONF_AREA = 0.12

# Default fusion weights (four signals, sum to 1.0)
# Tunable via DEPTH_W_DA / DEPTH_W_RW / DEPTH_W_AREA / DEPTH_W_LLM env vars.
_W_DA_DEFAULT   = 0.40   # DepthAnything V2 monocular depth
_W_RW_DEFAULT   = 0.20   # real-world size + focal estimate
_W_AREA_DEFAULT = 0.10   # relative bbox area proxy
_W_LLM_DEFAULT  = 0.30   # LLM holistic scene reasoning

# Proximity thresholds (match depth_overlay.py)
_PROX = {"CLOSE": 0.35, "MEDIUM": 0.65}


# ── Weight loading ─────────────────────────────────────────────────────────────

def _load_weights() -> tuple[float, float, float, float]:
    """
    Load fusion weights from env vars:
        DEPTH_W_DA   DEPTH_W_RW   DEPTH_W_AREA   DEPTH_W_LLM
    Always re-normalises so the four weights sum to 1.
    When DEPTH_W_LLM=0 the LLM signal is excluded and the other three
    weights are renormalised — useful for ablations or offline runs.
    """
    w_da   = float(os.environ.get("DEPTH_W_DA",   _W_DA_DEFAULT))
    w_rw   = float(os.environ.get("DEPTH_W_RW",   _W_RW_DEFAULT))
    w_area = float(os.environ.get("DEPTH_W_AREA", _W_AREA_DEFAULT))
    w_llm  = float(os.environ.get("DEPTH_W_LLM",  _W_LLM_DEFAULT))
    total  = w_da + w_rw + w_area + w_llm
    if total < 1e-9:
        return _W_DA_DEFAULT, _W_RW_DEFAULT, _W_AREA_DEFAULT, _W_LLM_DEFAULT
    return w_da/total, w_rw/total, w_area/total, w_llm/total


# ── Individual depth signals ───────────────────────────────────────────────────

def depth_from_depthmap(
    norm_depth: np.ndarray,   # (H, W) float32, 0=close  1=far
    x1: int, y1: int, x2: int, y2: int,
) -> float:
    """
    Input 1 — DepthAnything V2.
    Returns the median normalised depth value inside the bbox region.
    Falls back to image median for degenerate boxes.
    """
    if x2 > x1 and y2 > y1:
        roi = norm_depth[y1:y2, x1:x2]
        return float(np.median(roi))
    return float(np.median(norm_depth))


def depth_from_real_size(
    risk_group: str,
    bbox_h_px:  int,
    image_w:    int,
    image_h:    int,
) -> float | None:
    """
    Input 2 — Real-world size reference.

    Estimates metric distance using the pinhole camera model:
        distance_m = (ref_height_m * focal_px) / bbox_h_px

    Focal length is estimated without calibration as:
        focal_px = max(W, H) * 0.8   (~60° FOV assumption)

    The resulting distance is log-normalised to [0, 1].

    Returns None when the risk group has no reference height, so the caller
    can redistribute its weight to the other two signals.
    """
    ref_h = REFERENCE_HEIGHTS_M.get(risk_group)
    if ref_h is None or bbox_h_px <= 0:
        return None

    focal_px   = max(image_w, image_h) * 0.8
    distance_m = (ref_h * focal_px) / bbox_h_px

    log_near = np.log(_D_NEAR_M)
    log_far  = np.log(_D_FAR_M)
    log_dist = np.log(max(distance_m, _D_NEAR_M))
    depth_rw = (log_dist - log_near) / (log_far - log_near + 1e-9)
    return float(np.clip(depth_rw, 0.0, 1.0))


def depth_from_area(relative_area: float) -> float:
    """
    Input 3 — Relative bbox area proxy.

    depth_area = 1 - clamp(area_ratio / MAX_AREA_RATIO, 0, 1)

    Large bbox (relative_area → 0.25)  →  depth_area → 0.0  (close)
    Small bbox (relative_area → 0)     →  depth_area → 1.0  (far)
    """
    return float(np.clip(1.0 - relative_area / _MAX_AREA_RATIO, 0.0, 1.0))


# ── Fusion ─────────────────────────────────────────────────────────────────────

def _adaptive_weights(
    depth_da:   float,
    depth_rw:   float,
    depth_area: float,
    rel_area:   float,
    w_da:   float,
    w_rw:   float,
    w_area: float,
) -> tuple[float, float, float]:
    """
    Reduce DepthAnything's weight when it is a clear outlier relative to the
    two geometry-based signals (which agree with each other).

    Problem this solves: DA is a monocular model and can badly misjudge depth
    when a foreground object is silhouetted against sky or a uniform background.
    In those cases the bbox area and real-world size both independently say
    CLOSE while DA says FAR — and with the default 0.6 weight DA wins.

    Logic:
      1. Compute geometric consensus = weighted mean(depth_rw, depth_area)
      2. Measure how much DA deviates from that consensus
      3. Measure how much the two geometric signals agree with each other
         AND how large the bbox is (a large bbox → high geometric confidence)
      4. When deviation > 0.25 and geometric confidence > 0.4:
           demote DA weight toward 5 % of its original value as
           disagreement grows from 0.25 → 0.40

    Weight conservation: the reduction is redistributed to depth_rw and
    depth_area proportionally, so weights still sum to 1.
    """
    geom_total     = w_rw + w_area
    geom_consensus = (w_rw * depth_rw + w_area * depth_area) / geom_total
    da_outlier_mag = abs(depth_da - geom_consensus)

    # Geometric confidence: both signals must agree AND the bbox must be notable.
    # Use _GEOM_CONF_AREA (12 %) as the saturation point — a bbox that large is
    # already clearly notable; _MAX_AREA_RATIO (25 %) is too high a bar here.
    geom_agreement  = 1.0 - min(abs(depth_rw - depth_area), 1.0)
    area_confidence = min(rel_area / _GEOM_CONF_AREA, 1.0)
    geom_strength   = geom_agreement * area_confidence

    if da_outlier_mag > 0.25 and geom_strength > 0.4:
        # t: 0 at disagreement=0.25, saturates at 1.0 when disagreement≥0.40
        t = min((da_outlier_mag - 0.25) / 0.15, 1.0)
        effective_da_w = max(w_da * (1.0 - t * geom_strength), w_da * 0.05)
        redistributed  = w_da - effective_da_w
        w_rw_new   = w_rw   + redistributed * (w_rw   / geom_total)
        w_area_new = w_area + redistributed * (w_area / geom_total)
        return effective_da_w, w_rw_new, w_area_new

    return w_da, w_rw, w_area


def fuse_depth(
    depth_da:   float,
    depth_rw:   float | None,
    depth_area: float,
    rel_area:   float,
    w_da:   float,
    w_rw:   float,
    w_area: float,
    depth_llm: float | None = None,
    w_llm:     float = 0.0,
) -> float:
    """
    Weighted fusion of up to four depth signals with adaptive outlier correction.

    Signals
    -------
    depth_da   : DepthAnything V2 monocular depth
    depth_rw   : real-world size + focal estimate (None → weight redistributed)
    depth_area : relative bbox area proxy
    depth_llm  : LLM holistic scene reasoning (None → weight redistributed)

    When depth_rw is None, its weight is split proportionally among the rest.
    When depth_llm is None (e.g. offline run), its weight is similarly folded in.
    Adaptive weight correction runs on the DA-vs-geometric-consensus axis.
    """
    # Handle absent signals: redistribute their weights proportionally
    signals: list[tuple[float, float]] = []   # (signal, weight) pairs

    # Always present
    signals.append((depth_da,   w_da))
    signals.append((depth_area, w_area))
    if depth_rw  is not None: signals.append((depth_rw,  w_rw))
    if depth_llm is not None: signals.append((depth_llm, w_llm))

    total_w = sum(w for _, w in signals)
    if total_w < 1e-9:
        return depth_da

    # Renormalise
    signals = [(s, w / total_w) for s, w in signals]

    # Adaptive DA outlier correction (only when we have rw as a reference)
    if depth_rw is not None:
        # Rebuild effective weights for the three geometric signals
        eff_w = {s: w for s, w in signals}
        eff_da   = next(w for s, w in signals if s == depth_da)
        eff_rw   = next((w for s, w in signals if s == depth_rw),  0.0)
        eff_area = next((w for s, w in signals if s == depth_area), 0.0)
        eff_da_new, eff_rw_new, eff_area_new = _adaptive_weights(
            depth_da, depth_rw, depth_area, rel_area,
            eff_da, eff_rw, eff_area,
        )
        # Rebuild the full signal list with adapted DA/RW/area weights
        adapted: list[tuple[float, float]] = []
        for s, w in signals:
            if   s == depth_da:   adapted.append((s, eff_da_new))
            elif s == depth_rw:   adapted.append((s, eff_rw_new))
            elif s == depth_area: adapted.append((s, eff_area_new))
            else:                 adapted.append((s, w))   # LLM weight unchanged
        signals = adapted

    return sum(s * w for s, w in signals)


# ── Proximity / path-zone helpers ──────────────────────────────────────────────

def proximity_label(depth_score: float) -> str:
    if depth_score <= _PROX["CLOSE"]:
        return "CLOSE"
    if depth_score <= _PROX["MEDIUM"]:
        return "MEDIUM"
    return "FAR"


def path_zone(bbox_cx: float, image_w: int) -> str:
    """Middle third of the image = robot's forward path corridor."""
    return "CENTER" if image_w / 3 <= bbox_cx <= 2 * image_w / 3 else "PERIPHERAL"


# ── Public API ─────────────────────────────────────────────────────────────────

def enrich_detections(
    detections: list[dict],
    norm_depth:  np.ndarray,   # (H, W) float32, 0=close  1=far
    image_h:     int,
    image_w:     int,
) -> tuple[list[dict], np.ndarray]:
    """
    Annotate each detection with triple-fused depth information and return a
    corrected depth map for overlay visualisation.

    Each detection dict gains:
        depth_da        float        DepthAnything V2 signal
        depth_rw        float|None   Real-world size signal (None if unknown type)
        depth_area      float        Bbox area signal
        depth_score     float        Fused score in [0,1], 0=closest
        proximity_label str          CLOSE | MEDIUM | FAR
        path_zone       str          CENTER | PERIPHERAL

    The returned corrected_norm map shifts each bbox region toward its fused
    depth_score so the overlay heatmap reflects the same values as the JSON.

    Parameters
    ----------
    detections  : YOLO/GroundingDINO detections with 'box' [x1,y1,x2,y2],
                  'risk_group', and optionally 'relative_area'.
    norm_depth  : Normalised depth map (0=close, 1=far) from DepthAnything V2.
    image_h, image_w : Image dimensions in pixels.
    """
    w_da, w_rw, w_area, w_llm = _load_weights()
    image_area = float(image_h * image_w)

    corrected = norm_depth.copy()
    enriched  = []

    for det in detections:
        d = dict(det)

        # Clamp bbox to image bounds
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(image_w, x2);  y2 = min(image_h, y2)

        # ── Signal 1: DepthAnything V2 ────────────────────────────────────────
        da = depth_from_depthmap(norm_depth, x1, y1, x2, y2)

        # ── Signal 2: Real-world size reference ───────────────────────────────
        rw = depth_from_real_size(
            risk_group=d.get("risk_group", ""),
            bbox_h_px=y2 - y1,
            image_w=image_w,
            image_h=image_h,
        )

        # ── Signal 3: Relative bbox area ──────────────────────────────────────
        rel_area = d.get(
            "relative_area",
            ((x2 - x1) * (y2 - y1)) / image_area,
        )
        # Use LLM-rationalized override for edge-clipped objects when available.
        # A clipped bbox is close to the camera, not far — override the naive
        # area signal that would otherwise score it as distant.
        if "depth_area_override" in d:
            area_sig = float(d["depth_area_override"])
        else:
            area_sig = depth_from_area(rel_area)

        # ── Signal 4: LLM holistic depth reasoning ────────────────────────────
        llm_sig = d.get("depth_llm")   # set by src/llm_depth.get_llm_depth_signals()

        # ── Fusion (with adaptive outlier correction) ─────────────────────────
        fused = fuse_depth(
            da, rw, area_sig, rel_area,
            w_da, w_rw, w_area,
            depth_llm=llm_sig, w_llm=w_llm,
        )

        # ── Update corrected map for visualization ────────────────────────────
        # Blend the bbox region toward the fused score; correction strength
        # scales with the disagreement between raw model and fused estimate.
        if x2 > x1 and y2 > y1:
            roi          = corrected[y1:y2, x1:x2]
            disagreement = abs(float(np.median(roi)) - fused)
            alpha        = float(np.clip(0.45 + 0.35 * disagreement, 0.0, 0.70))
            corrected[y1:y2, x1:x2] = (1.0 - alpha) * roi + alpha * fused

        # ── Annotate detection ────────────────────────────────────────────────
        bbox_cx = (x1 + x2) / 2.0
        d["depth_da"]        = round(da,    4)
        d["depth_rw"]        = round(rw,    4) if rw is not None else None
        d["depth_area"]      = round(area_sig, 4)
        d["depth_score"]     = round(fused, 4)
        d["proximity_label"] = proximity_label(fused)
        d["path_zone"]       = path_zone(bbox_cx, image_w)
        enriched.append(d)

    return enriched, corrected
