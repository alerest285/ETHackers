"""
Graph Classifier — scene-graph feature vector → STOP / SLOW / CONTINUE

Takes the outputs already produced by the pipeline:
  - enriched detections  (list[dict], from depth_module/fused_depth.py)
  - scene graph          (SceneGraph, from 3d-module/lift_3d.py)
  - segmented image      (PIL Image, SAM2 mask overlay or original)
  - DINO boxes           (list[dict], raw Grounding DINO detections)

and returns:
  {
    "action":     "STOP" | "SLOW" | "CONTINUE",
    "confidence": float,          # calibrated probability in (0, 1)
    "probabilities": {            # full class distribution
        "STOP":     float,
        "SLOW":     float,
        "CONTINUE": float,
    },
    "top_features": dict[str, float],   # the most discriminative features
  }

Architecture
------------
Feature extraction → 43-dimensional float vector
                  → XGBoost / sklearn GradientBoosting classifier
                  → softmax probabilities over {STOP, SLOW, CONTINUE}

The classifier is trained with `train()` and persisted to a .joblib file.
Before training data is available, `predict()` falls back to the rule engine
(SAFETY_RULES.md heuristics) returning well-calibrated rule-based probabilities.

Usage
-----
    from action_module.graph_classifier import GraphClassifier

    clf = GraphClassifier()                      # loads weights if available
    result = clf.predict(detections, graph, image)

    # Training (after you have labelled samples):
    clf.train(X, y)                              # X: list of dicts or np arrays
    clf.save()
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass   # SceneGraph imported lazily to avoid circular import at module level

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
WEIGHTS_PATH = _HERE / "graph_classifier.joblib"

# ── Class ordering ─────────────────────────────────────────────────────────────
CLASSES = ["STOP", "SLOW", "CONTINUE"]
_CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── Proximity / zone constants (must match fused_depth.py) ─────────────────────
_CLOSE_THRESH  = 0.35
_MEDIUM_THRESH = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val, default=0.0):
    """Return val if it is a finite number, otherwise default."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def extract_features(
    detections: list[dict],
    graph,                      # SceneGraph | None
    image: Image.Image | None,
) -> np.ndarray:
    """
    Build a fixed 43-dimensional feature vector from pipeline outputs.

    Feature groups
    --------------
    [0-4]   Global counts: total, n_humans, n_vehicles, n_obstacles, n_markers
    [5-9]   Closest object per group: min depth (0=closest) for H/V/O/M + global
    [10-14] Center-zone counts per group + total center objects
    [15-19] Close-zone counts per group + total close objects
    [20-24] High-risk combo flags: human_close, vehicle_close_center,
            human+vehicle_both_close, multi_human (>=2), extreme_close (depth<0.15)
    [25-29] Detection quality: mean score, mean mask_score, min score, min mask_score,
            fraction with score > 0.5
    [30-34] Scene coverage: human frame coverage, vehicle frame coverage,
            total risk coverage, mean depth_da, mean depth_area
    [35-39] 3D scene graph stats: n_nodes, n_edges, n_blocking_edges,
            mean centroid Z, min nearest_z
    [40-42] Image stats: aspect ratio, mean brightness (SAM overlay), std brightness
    """
    H_img, W_img = (image.height, image.width) if image is not None else (1, 1)
    img_area = float(H_img * W_img)

    # ── Partition detections by group ─────────────────────────────────────────
    groups: dict[str, list[dict]] = {
        "HUMAN": [], "VEHICLE": [], "OBSTACLE": [], "SAFETY_MARKER": [],
    }
    for d in detections:
        g = d.get("risk_group", "BACKGROUND")
        if g in groups:
            groups[g].append(d)

    # Helper: depth_score of a detection (None → treat as FAR=0.9)
    def ds(d: dict) -> float:
        return _safe(d.get("depth_score"), 0.9)

    def is_close(d: dict) -> bool:
        return ds(d) <= _CLOSE_THRESH

    def is_center(d: dict) -> bool:
        return d.get("path_zone", "PERIPHERAL") == "CENTER"

    def bbox_coverage(d: dict) -> float:
        box = d.get("box", [0, 0, 0, 0])
        if len(box) < 4:
            return 0.0
        x1, y1, x2, y2 = box
        return max(0.0, (x2 - x1) * (y2 - y1)) / (img_area + 1e-6)

    # ── [0-4] Counts ──────────────────────────────────────────────────────────
    n_total    = len(detections)
    n_human    = len(groups["HUMAN"])
    n_vehicle  = len(groups["VEHICLE"])
    n_obstacle = len(groups["OBSTACLE"])
    n_marker   = len(groups["SAFETY_MARKER"])

    # ── [5-9] Min depth per group ─────────────────────────────────────────────
    def min_depth(dets: list[dict]) -> float:
        return min((ds(d) for d in dets), default=1.0)

    min_depth_human    = min_depth(groups["HUMAN"])
    min_depth_vehicle  = min_depth(groups["VEHICLE"])
    min_depth_obstacle = min_depth(groups["OBSTACLE"])
    min_depth_marker   = min_depth(groups["SAFETY_MARKER"])
    min_depth_global   = min((ds(d) for d in detections), default=1.0)

    # ── [10-14] Center zone counts ────────────────────────────────────────────
    center_human    = sum(1 for d in groups["HUMAN"]         if is_center(d))
    center_vehicle  = sum(1 for d in groups["VEHICLE"]       if is_center(d))
    center_obstacle = sum(1 for d in groups["OBSTACLE"]      if is_center(d))
    center_marker   = sum(1 for d in groups["SAFETY_MARKER"] if is_center(d))
    center_total    = sum(1 for d in detections              if is_center(d))

    # ── [15-19] Close zone counts ─────────────────────────────────────────────
    close_human    = sum(1 for d in groups["HUMAN"]         if is_close(d))
    close_vehicle  = sum(1 for d in groups["VEHICLE"]       if is_close(d))
    close_obstacle = sum(1 for d in groups["OBSTACLE"]      if is_close(d))
    close_marker   = sum(1 for d in groups["SAFETY_MARKER"] if is_close(d))
    close_total    = sum(1 for d in detections              if is_close(d))

    # ── [20-24] High-risk combo flags ─────────────────────────────────────────
    flag_human_close           = float(close_human >= 1)
    flag_vehicle_close_center  = float(
        any(is_close(d) and is_center(d) for d in groups["VEHICLE"])
    )
    flag_human_vehicle_both_close = float(close_human >= 1 and close_vehicle >= 1)
    flag_multi_human              = float(n_human >= 2)
    flag_extreme_close            = float(
        any(ds(d) < 0.15 for d in detections)
    )

    # ── [25-29] Detection quality ─────────────────────────────────────────────
    all_scores      = [_safe(d.get("score"),      0.0) for d in detections]
    all_mask_scores = [_safe(d.get("mask_score"), 0.0) for d in detections]
    mean_score       = float(np.mean(all_scores))      if all_scores else 0.0
    mean_mask_score  = float(np.mean(all_mask_scores)) if all_mask_scores else 0.0
    min_score        = float(np.min(all_scores))       if all_scores else 0.0
    min_mask_score   = float(np.min(all_mask_scores))  if all_mask_scores else 0.0
    frac_high_conf   = float(np.mean([s > 0.5 for s in all_scores])) if all_scores else 0.0

    # ── [30-34] Scene coverage ────────────────────────────────────────────────
    cov_human   = sum(bbox_coverage(d) for d in groups["HUMAN"])
    cov_vehicle = sum(bbox_coverage(d) for d in groups["VEHICLE"])
    cov_risk    = sum(bbox_coverage(d) for d in detections
                      if d.get("risk_group") in ("HUMAN", "VEHICLE"))
    mean_da     = float(np.mean([_safe(d.get("depth_da"),   0.5) for d in detections])) \
                  if detections else 0.5
    mean_area   = float(np.mean([_safe(d.get("depth_area"), 0.5) for d in detections])) \
                  if detections else 0.5

    # ── [35-39] 3D scene graph stats ──────────────────────────────────────────
    n_nodes, n_edges, n_blocking = 0, 0, 0
    mean_centroid_z, min_nearest_z = 0.5, 1.0

    if graph is not None:
        nodes = getattr(graph, "nodes", [])
        edges = getattr(graph, "edges", [])
        n_nodes   = len(nodes)
        n_edges   = len(edges)
        n_blocking = sum(1 for e in edges if getattr(e, "blocking", False))

        centroid_zs  = [getattr(n, "centroid_3d", [0, 0, 0.5])[2] for n in nodes]
        nearest_zs   = [getattr(n, "nearest_z", 1.0) for n in nodes]
        mean_centroid_z = float(np.mean(centroid_zs)) if centroid_zs else 0.5
        min_nearest_z   = float(np.min(nearest_zs))   if nearest_zs  else 1.0

    # ── [40-42] Image stats ───────────────────────────────────────────────────
    aspect_ratio    = W_img / (H_img + 1e-6)
    mean_brightness = 0.5
    std_brightness  = 0.0
    if image is not None:
        try:
            gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
            mean_brightness = float(np.mean(gray))
            std_brightness  = float(np.std(gray))
        except Exception:
            pass

    vec = np.array([
        # [0-4]
        n_total, n_human, n_vehicle, n_obstacle, n_marker,
        # [5-9]
        min_depth_human, min_depth_vehicle, min_depth_obstacle,
        min_depth_marker, min_depth_global,
        # [10-14]
        center_human, center_vehicle, center_obstacle, center_marker, center_total,
        # [15-19]
        close_human, close_vehicle, close_obstacle, close_marker, close_total,
        # [20-24]
        flag_human_close, flag_vehicle_close_center,
        flag_human_vehicle_both_close, flag_multi_human, flag_extreme_close,
        # [25-29]
        mean_score, mean_mask_score, min_score, min_mask_score, frac_high_conf,
        # [30-34]
        cov_human, cov_vehicle, cov_risk, mean_da, mean_area,
        # [35-39]
        n_nodes, n_edges, n_blocking, mean_centroid_z, min_nearest_z,
        # [40-42]
        aspect_ratio, mean_brightness, std_brightness,
    ], dtype=np.float32)

    return vec


# Feature names (same order as the vector above, useful for SHAP / importances)
FEATURE_NAMES = [
    "n_total", "n_human", "n_vehicle", "n_obstacle", "n_marker",
    "min_depth_human", "min_depth_vehicle", "min_depth_obstacle",
    "min_depth_marker", "min_depth_global",
    "center_human", "center_vehicle", "center_obstacle", "center_marker", "center_total",
    "close_human", "close_vehicle", "close_obstacle", "close_marker", "close_total",
    "flag_human_close", "flag_vehicle_close_center",
    "flag_human_vehicle_both_close", "flag_multi_human", "flag_extreme_close",
    "mean_score", "mean_mask_score", "min_score", "min_mask_score", "frac_high_conf",
    "cov_human", "cov_vehicle", "cov_risk", "mean_da", "mean_area",
    "n_nodes", "n_edges", "n_blocking", "mean_centroid_z", "min_nearest_z",
    "aspect_ratio", "mean_brightness", "std_brightness",
]

assert len(FEATURE_NAMES) == 43


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based fallback  (used before training data is available)
# ══════════════════════════════════════════════════════════════════════════════

def _rule_probabilities(vec: np.ndarray) -> np.ndarray:
    """
    Translate the feature vector into STOP/SLOW/CONTINUE probabilities using
    SAFETY_RULES.md heuristics.  Returns a (3,) array summing to 1.

    Maps rule confidence scores → soft probabilities via a temperature-scaled
    softmax so the output is a proper distribution rather than a hard label.
    """
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    # Extract key scalars
    flag_human_close         = vec[idx["flag_human_close"]]
    flag_extreme              = vec[idx["flag_extreme_close"]]
    flag_vc_center            = vec[idx["flag_vehicle_close_center"]]
    flag_multi_human          = vec[idx["flag_multi_human"]]
    flag_hv_both_close        = vec[idx["flag_human_vehicle_both_close"]]
    min_depth_global          = vec[idx["min_depth_global"]]
    n_human                   = vec[idx["n_human"]]
    n_vehicle                 = vec[idx["n_vehicle"]]
    n_obstacle                = vec[idx["n_obstacle"]]
    n_marker                  = vec[idx["n_marker"]]
    n_total                   = vec[idx["n_total"]]
    min_depth_human           = vec[idx["min_depth_human"]]
    min_depth_vehicle         = vec[idx["min_depth_vehicle"]]
    center_human              = vec[idx["center_human"]]
    close_vehicle             = vec[idx["close_vehicle"]]
    center_obstacle           = vec[idx["center_obstacle"]]
    close_obstacle            = vec[idx["close_obstacle"]]
    center_marker             = vec[idx["center_marker"]]

    # ── Compute a raw STOP / SLOW / CONTINUE score ────────────────────────────
    stop_score = 0.0
    slow_score = 0.0
    cont_score = 0.0

    # STOP rules
    if flag_extreme:                                    stop_score = max(stop_score, 0.97)
    if flag_human_close:                                stop_score = max(stop_score, 0.95)
    if flag_hv_both_close:                              stop_score = max(stop_score, 0.95)
    if flag_vc_center:                                  stop_score = max(stop_score, 0.92)
    if center_human >= 1:                               stop_score = max(stop_score, 0.90)
    if flag_multi_human:                                stop_score = max(stop_score, 0.88)

    # SLOW rules
    if n_human >= 1 and min_depth_human <= _MEDIUM_THRESH:
        slow_score = max(slow_score, 0.78)
    if center_human >= 1 and min_depth_human > _MEDIUM_THRESH:
        slow_score = max(slow_score, 0.70)
    if close_vehicle >= 1 and not flag_vc_center:
        slow_score = max(slow_score, 0.75)
    if n_vehicle >= 1 and min_depth_vehicle <= _MEDIUM_THRESH:
        slow_score = max(slow_score, 0.72)
    if center_obstacle >= 1 and close_obstacle >= 1:
        slow_score = max(slow_score, 0.68)
    if center_marker >= 1:
        slow_score = max(slow_score, 0.62)
    if n_vehicle >= 1:
        slow_score = max(slow_score, 0.65)
    if n_obstacle >= 3:
        slow_score = max(slow_score, 0.65)

    # CONTINUE rules
    if n_total == 0:
        cont_score = 0.82
    elif n_human == 0 and n_vehicle == 0:
        if min_depth_global > _MEDIUM_THRESH:
            cont_score = max(cont_score, 0.75)
        else:
            cont_score = max(cont_score, 0.60)

    # Convert scores to a probability distribution via softmax
    raw = np.array([stop_score, slow_score, cont_score], dtype=np.float64)
    if raw.sum() < 1e-9:
        # nothing fired — default to cautious SLOW
        raw = np.array([0.15, 0.55, 0.30])
    else:
        # temperature-scaled softmax (T=0.3 → sharpens the distribution)
        T = 0.3
        raw = np.exp(raw / T)
        raw /= raw.sum()

    return raw.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# GraphClassifier
# ══════════════════════════════════════════════════════════════════════════════

class GraphClassifier:
    """
    Wraps feature extraction + classifier.

    Before `train()` / `load()` is called the classifier falls back to the
    rule-based probability estimator.  Once trained it uses a gradient-boosted
    tree model (scikit-learn GradientBoostingClassifier) whose `predict_proba`
    output is used directly as the confidence.

    Parameters
    ----------
    weights_path : Path  — where to load/save the trained model.
    """

    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.weights_path = weights_path
        self._model = None   # None → use rule fallback

        if weights_path.exists():
            self.load(weights_path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        detections: list[dict],
        graph=None,
        image: Image.Image | None = None,
    ) -> dict:
        """
        Classify the scene.

        Parameters
        ----------
        detections : enriched detection dicts (from fused_depth.enrich_detections)
        graph      : SceneGraph from lift_3d.SceneGraphBuilder.process (or None)
        image      : PIL image (original or SAM overlay) for image-level features

        Returns
        -------
        {
          "action":        str,              — STOP | SLOW | CONTINUE
          "confidence":    float,            — probability of the predicted class
          "probabilities": dict[str, float], — full class distribution
          "top_features":  dict[str, float], — top-5 most discriminative features
          "source":        str,              — "classifier" | "rules"
        }
        """
        vec  = extract_features(detections, graph, image)
        probs = self._classify(vec)

        best_idx    = int(np.argmax(probs))
        best_class  = CLASSES[best_idx]
        confidence  = float(probs[best_idx])

        top_features = self._top_features(vec)

        return {
            "action":        best_class,
            "confidence":    round(confidence, 4),
            "probabilities": {
                "STOP":     round(float(probs[0]), 4),
                "SLOW":     round(float(probs[1]), 4),
                "CONTINUE": round(float(probs[2]), 4),
            },
            "top_features": top_features,
            "source": "classifier" if self._model is not None else "rules",
        }

    def _classify(self, vec: np.ndarray) -> np.ndarray:
        """Return (3,) probability array."""
        if self._model is not None:
            try:
                probs = self._model.predict_proba(vec.reshape(1, -1))[0]
                # Re-order to STOP/SLOW/CONTINUE; assign 0 to any missing class
                # (handles classifiers trained on a subset of CLASSES, e.g. when
                # the training set didn't contain CONTINUE samples).
                model_classes = list(self._model.classes_)
                reordered = np.array([
                    probs[model_classes.index(c)] if c in model_classes else 0.0
                    for c in CLASSES
                ], dtype=np.float32)
                # Re-normalise so probabilities still sum to 1
                total = float(reordered.sum())
                if total > 0:
                    reordered = reordered / total
                else:
                    return _rule_probabilities(vec)
                return reordered
            except Exception as e:
                warnings.warn(f"Classifier predict_proba failed ({e}), using rule fallback.")
        return _rule_probabilities(vec)

    def _top_features(self, vec: np.ndarray, k: int = 5) -> dict[str, float]:
        """Return the k features with highest absolute values (after z-scoring)."""
        # Use feature magnitude as a simple importance proxy
        feat_vals = {FEATURE_NAMES[i]: float(vec[i]) for i in range(len(FEATURE_NAMES))}
        # For human-readable top features, sort by deviation from neutral value
        sorted_feats = sorted(feat_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {k: round(v, 4) for k, v in sorted_feats[:k]}

    # ── Training ──────────────────────────────────────────────────────────────

    def build_feature_matrix(
        self,
        samples: list[dict],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build (X, y) from a list of labelled sample dicts.

        Each sample dict must have:
            "detections" : list[dict]
            "label"      : "STOP" | "SLOW" | "CONTINUE"
        Optional:
            "graph"      : SceneGraph
            "image"      : PIL Image

        Returns (X: np.ndarray shape (N,43), labels: list[str])
        """
        X, y = [], []
        for s in samples:
            vec = extract_features(
                s["detections"],
                s.get("graph"),
                s.get("image"),
            )
            X.append(vec)
            y.append(s["label"])
        return np.stack(X), y

    def train(
        self,
        X: np.ndarray,
        y: list[str],
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        pca_variance: float = 0.95,
    ) -> "GraphClassifier":
        """
        Fit StandardScaler → PCA → GradientBoostingClassifier on (X, y).

        X            : (N, 43) feature matrix from build_feature_matrix()
        y            : list of "STOP" | "SLOW" | "CONTINUE" strings
        pca_variance : fraction of variance to retain (default 0.95).
                       With 43 dimensions this typically reduces to ~8-15 components,
                       which prevents the tree from over-fitting to correlated features
                       (e.g. close_human and flag_human_close are nearly co-linear).
                       Set to 1.0 to disable PCA.

        Returns self for chaining.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.decomposition import PCA
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for training: pip install scikit-learn")

        n_samples, n_features = X.shape
        # PCA n_components must be <= min(n_samples, n_features).
        # With very small datasets, cap at n_samples - 1 to avoid SVD errors.
        max_components = min(n_samples - 1, n_features)
        pca_n = pca_variance if pca_variance < 1.0 else max_components

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_n, random_state=42)),
            ("clf",    GradientBoostingClassifier(
                n_estimators  = n_estimators,
                max_depth     = max_depth,
                learning_rate = learning_rate,
                subsample     = 0.8,
                random_state  = 42,
            )),
        ])
        pipeline.fit(X, y)
        self._model = pipeline

        pca_step = pipeline.named_steps["pca"]
        n_components = pca_step.n_components_
        var_explained = pca_step.explained_variance_ratio_.sum()
        print(
            f"Trained on {len(y)} samples  |  "
            f"PCA: {n_features}d -> {n_components}d "
            f"({var_explained:.1%} variance retained)  |  "
            f"Classes: {list(pipeline.classes_)}"
        )
        return self

    def pseudo_label_from_rules(
        self,
        detections_dir: Path,
        graphs_dir:     Path | None = None,
        conf_threshold: float = 0.85,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate pseudo-labels by running the rule engine over every detections
        JSON in detections_dir and keeping only high-confidence predictions.

        This lets you bootstrap the classifier without any manual annotation:
        the rule engine acts as a noisy but reliable labeller for unambiguous
        scenes (e.g. human at close range → STOP with conf 0.95 is trustworthy).

        Parameters
        ----------
        detections_dir  : directory of enriched detection JSONs (pipeline output)
        graphs_dir      : optional scene-graph JSON directory
        conf_threshold  : minimum rule confidence to accept as a pseudo-label
                          (0.85 keeps only the clearest cases)

        Returns
        -------
        X : (N, 43) feature matrix
        y : list[str] pseudo-labels
        """
        X_rows, y_rows = [], []
        skipped = 0

        for det_path in sorted(detections_dir.glob("*.json")):
            try:
                with open(det_path) as f:
                    dets = json.load(f).get("detections", [])
            except Exception:
                continue

            graph = None
            if graphs_dir is not None:
                gpath = graphs_dir / det_path.name
                if gpath.exists():
                    graph = _load_graph_json(gpath)

            vec   = extract_features(dets, graph, None)
            probs = _rule_probabilities(vec)
            conf  = float(probs.max())
            label = CLASSES[int(probs.argmax())]

            if conf < conf_threshold:
                skipped += 1
                continue

            X_rows.append(vec)
            y_rows.append(label)

        if not X_rows:
            raise ValueError(
                f"No samples passed conf_threshold={conf_threshold}. "
                "Lower the threshold or check detections_dir."
            )

        from collections import Counter
        dist = Counter(y_rows)
        print(
            f"Pseudo-labels: {len(y_rows)} accepted, {skipped} skipped "
            f"(threshold={conf_threshold})  |  {dict(dist)}"
        )
        return np.stack(X_rows), y_rows

    def train_from_pseudo_labels(
        self,
        detections_dir: Path,
        graphs_dir:     Path | None = None,
        conf_threshold: float = 0.85,
        extra_labels:   dict[str, str] | None = None,
        pca_variance:   float = 0.95,
        **train_kwargs,
    ) -> "GraphClassifier":
        """
        Bootstrap training without manual annotation:
          1. Run rule engine over detections_dir, keep conf >= conf_threshold.
          2. Optionally merge in manually-annotated labels (extra_labels dict).
          3. Train with PCA + GradientBoosting.

        extra_labels overrides pseudo-labels for any matching stem, giving
        manually-labelled examples priority over the rule engine.

        Parameters
        ----------
        detections_dir  : path to pipeline_output/detections/
        graphs_dir      : path to pipeline_output/scene_graphs/ (optional)
        conf_threshold  : rule engine confidence cutoff for pseudo-labels
        extra_labels    : {stem: label} dict of manually-annotated samples
        pca_variance    : passed through to train()
        **train_kwargs  : extra kwargs forwarded to train()
        """
        X_pseudo, y_pseudo = self.pseudo_label_from_rules(
            detections_dir, graphs_dir, conf_threshold
        )

        # Merge in manual labels if provided
        if extra_labels:
            extra_X, extra_y = [], []
            for stem, label in extra_labels.items():
                det_path = detections_dir / f"{stem}.json"
                if not det_path.exists():
                    warnings.warn(f"Missing detections for manual label {stem}, skipping.")
                    continue
                with open(det_path) as f:
                    dets = json.load(f).get("detections", [])
                graph = None
                if graphs_dir is not None:
                    gpath = graphs_dir / f"{stem}.json"
                    if gpath.exists():
                        graph = _load_graph_json(gpath)
                extra_X.append(extract_features(dets, graph, None))
                extra_y.append(label)

            if extra_X:
                X_pseudo = np.vstack([X_pseudo, np.stack(extra_X)])
                y_pseudo  = y_pseudo + extra_y
                print(f"Merged {len(extra_y)} manual labels into training set.")

        return self.train(X_pseudo, y_pseudo, pca_variance=pca_variance, **train_kwargs)

    def train_from_detections_dir(
        self,
        detections_dir: Path,
        labels: dict[str, str],   # stem → "STOP" | "SLOW" | "CONTINUE"
        graphs_dir: Path | None = None,
        pca_variance: float = 0.95,
    ) -> "GraphClassifier":
        """
        Train from explicit labels only (no pseudo-labelling).

        detections_dir : path to pipeline_output/detections/
        labels         : {image_stem: "STOP"|"SLOW"|"CONTINUE"}
        graphs_dir     : path to pipeline_output/scene_graphs/ (optional)
        pca_variance   : passed through to train()
        """
        samples = []
        for stem, label in labels.items():
            det_path = detections_dir / f"{stem}.json"
            if not det_path.exists():
                warnings.warn(f"Missing detections for {stem}, skipping.")
                continue
            with open(det_path) as f:
                dets = json.load(f).get("detections", [])

            graph = None
            if graphs_dir is not None:
                gpath = graphs_dir / f"{stem}.json"
                if gpath.exists():
                    graph = _load_graph_json(gpath)

            samples.append({"detections": dets, "graph": graph, "label": label})

        X, y = self.build_feature_matrix(samples)
        return self.train(X, y, pca_variance=pca_variance)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Persist the trained model to disk."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")
        if self._model is None:
            raise RuntimeError("No trained model to save. Call train() first.")
        target = path or self.weights_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, target)
        print(f"Model saved to {target}")

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved model from disk."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")
        target = path or self.weights_path
        self._model = joblib.load(target)
        print(f"Graph classifier loaded from {target}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def feature_importances(self) -> dict[str, float] | None:
        """
        Return original-space feature importances by projecting tree importances
        back through PCA loadings.

        Each original feature's importance = sum over PCA components of
        (tree importance of that component × |loading of that feature in that component|).

        This gives a meaningful importance score in the original 43-d space even
        though the tree only sees PCA components.
        """
        if self._model is None:
            return None
        try:
            pca = self._model.named_steps["pca"]
            clf = self._model.named_steps["clf"]
            # tree_imp: (n_components,)
            tree_imp = clf.feature_importances_
            # components_: (n_components, n_features)
            loadings = np.abs(pca.components_)           # (n_components, 43)
            # project: (43,) — weighted sum of |loadings| by tree importance
            original_imp = (tree_imp[:, None] * loadings).sum(axis=0)
            original_imp /= original_imp.sum() + 1e-9   # re-normalise to sum=1
            return {
                FEATURE_NAMES[i]: round(float(v), 6)
                for i, v in enumerate(original_imp)
            }
        except Exception:
            return None

    def pca_summary(self) -> dict | None:
        """Return PCA diagnostics: n_components, variance per component, total."""
        if self._model is None:
            return None
        try:
            pca = self._model.named_steps["pca"]
            ratios = pca.explained_variance_ratio_
            return {
                "n_components":      int(pca.n_components_),
                "total_variance":    round(float(ratios.sum()), 4),
                "per_component":     [round(float(r), 4) for r in ratios],
            }
        except Exception:
            return None

    def evaluate(self, X: np.ndarray, y: list[str]) -> dict:
        """Quick accuracy report on a held-out set."""
        try:
            from sklearn.metrics import classification_report, accuracy_score
        except ImportError:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        y_pred = [self.predict_label(x) for x in X]
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "report":   classification_report(y, y_pred, target_names=CLASSES),
        }

    def predict_label(self, vec: np.ndarray) -> str:
        probs = self._classify(vec)
        return CLASSES[int(np.argmax(probs))]


# ══════════════════════════════════════════════════════════════════════════════
# Minimal scene-graph JSON loader  (avoids importing lift_3d at module level)
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleGraph:
    """Lightweight stand-in for SceneGraph when lift_3d is not importable."""
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class _SimpleNode:
    def __init__(self, d: dict):
        self.centroid_3d = d.get("centroid_3d", [0, 0, 0.5])
        self.nearest_z   = d.get("nearest_z",   1.0)
        self.blocking    = False


class _SimpleEdge:
    def __init__(self, d: dict):
        self.blocking = d.get("blocking", False)


def _load_graph_json(path: Path) -> _SimpleGraph:
    with open(path) as f:
        data = json.load(f)
    nodes = [_SimpleNode(n) for n in data.get("nodes", [])]
    edges = [_SimpleEdge(e) for e in data.get("edges", [])]
    return _SimpleGraph(nodes, edges)
