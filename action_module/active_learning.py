"""
Active Learning — uncertainty sampling + interactive labelling loop.

After a benchmark or pipeline run, this module:
  1. Collects every prediction into an UncertaintyBuffer.
  2. Ranks samples by prediction entropy (most uncertain first).
  3. Presents them to you one at a time: shows the image, the current
     prediction, the probability distribution, and the top features.
  4. You type S / W / C (or STOP / SLOW / CONTINUE) to label, or Enter to skip.
  5. Labels are appended to action_module/labels.json.
  6. After labelling, calls GraphClassifier.train() on all accumulated labels
     and saves the updated model.

Works in two modes:
  - Interactive (local):  opens images with PIL Image.show()
  - Headless (Northflank): saves a review bundle to --review-dir; you download
    it, run  python action_module/active_learning.py --label --review-dir ...
    locally to annotate, then re-upload labels.json and retrain.

Usage
-----
  # Stand-alone review after a benchmark run:
  python action_module/active_learning.py \\
      --buffer    data/pipeline_output/uncertainty_buffer.json \\
      --det-dir   data/pipeline_output/detections \\
      --img-dir   data/challenge/data/images/val \\
      --top-k     50

  # Export a review bundle without an interactive session (headless):
  python action_module/active_learning.py \\
      --buffer    data/pipeline_output/uncertainty_buffer.json \\
      --export    data/review_bundle \\
      --top-k     50

  # Retrain after labels.json has been updated:
  python action_module/active_learning.py --retrain \\
      --det-dir   data/pipeline_output/detections
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
LABELS_PATH = _HERE / "labels.json"

CLASSES = ["STOP", "SLOW", "CONTINUE"]
_SHORTCUT = {"s": "STOP", "w": "SLOW", "c": "CONTINUE",
             "stop": "STOP", "slow": "SLOW", "continue": "CONTINUE"}


# ══════════════════════════════════════════════════════════════════════════════
# UncertaintyBuffer — lightweight collector used during pipeline / benchmark
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyBuffer:
    """
    Collects prediction metadata during a pipeline run.

    Attach one instance at the start of benchmark.py / pipeline.py, call
    `record()` after each image, then `save()` at the end.

    Example
    -------
        buf = UncertaintyBuffer()
        ...
        result = clf.predict(enriched, graph, image)
        buf.record(stem, img_path, result)
        ...
        buf.save(Path("data/pipeline_output/uncertainty_buffer.json"))
    """

    def __init__(self):
        self._entries: list[dict] = []

    def record(
        self,
        stem:       str,
        img_path:   Path | str,
        clf_result: dict,
    ) -> None:
        """
        Add one prediction to the buffer.

        clf_result must have keys: action, confidence, probabilities,
        top_features (as returned by GraphClassifier.predict).
        """
        probs = clf_result.get("probabilities", {})
        entropy = _entropy(list(probs.values()))
        self._entries.append({
            "stem":         stem,
            "img_path":     str(img_path),
            "action":       clf_result.get("action"),
            "confidence":   clf_result.get("confidence"),
            "entropy":      round(entropy, 6),
            "probabilities": probs,
            "top_features": clf_result.get("top_features", {}),
            "source":       clf_result.get("source", "unknown"),
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._entries, f, indent=2)
        print(f"Uncertainty buffer saved ({len(self._entries)} entries) → {path}")

    @staticmethod
    def load(path: Path) -> "UncertaintyBuffer":
        buf = UncertaintyBuffer()
        with open(path) as f:
            buf._entries = json.load(f)
        return buf

    def __len__(self) -> int:
        return len(self._entries)

    def most_uncertain(self, k: int, strategy: str = "entropy") -> list[dict]:
        """
        Return the top-k most uncertain entries.

        strategy:
          "entropy"  — highest Shannon entropy over class probabilities (default)
          "margin"   — smallest gap between top-2 probabilities
          "least_conf" — lowest confidence (probability of predicted class)
        """
        already_labelled = {e["stem"] for e in _load_labels()}

        candidates = [e for e in self._entries if e["stem"] not in already_labelled]

        if strategy == "entropy":
            key = lambda e: -e["entropy"]           # most uncertain first
        elif strategy == "margin":
            key = lambda e: _margin(e["probabilities"])
        else:   # least_conf
            key = lambda e: e["confidence"]

        return sorted(candidates, key=key)[:k]


# ══════════════════════════════════════════════════════════════════════════════
# Label persistence
# ══════════════════════════════════════════════════════════════════════════════

def _load_labels(path: Path = LABELS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_labels(labels: list[dict], path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def append_label(stem: str, label: str, path: Path = LABELS_PATH) -> None:
    """Add or update a single label entry."""
    labels = _load_labels(path)
    # Update existing entry if present
    for entry in labels:
        if entry["stem"] == stem:
            entry["label"] = label
            _save_labels(labels, path)
            return
    labels.append({"stem": stem, "label": label})
    _save_labels(labels, path)


def labels_as_dict(path: Path = LABELS_PATH) -> dict[str, str]:
    """Return {stem: label} dict for all saved labels."""
    return {e["stem"]: e["label"] for e in _load_labels(path)}


# ══════════════════════════════════════════════════════════════════════════════
# Interactive labelling loop
# ══════════════════════════════════════════════════════════════════════════════

def _entropy(probs: list[float]) -> float:
    p = np.array(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _margin(probs: dict) -> float:
    vals = sorted(probs.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    return float(vals[0] - vals[1])


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def _render_entry(entry: dict, idx: int, total: int) -> None:
    """Print a rich summary of one uncertain sample."""
    print("\n" + "═" * 60)
    print(f"  Sample {idx}/{total}  ·  stem: {entry['stem']}")
    print(f"  Prediction : {entry['action']}  (conf {entry['confidence']:.3f},"
          f" entropy {entry['entropy']:.3f}, source: {entry['source']})")
    print()

    probs = entry.get("probabilities", {})
    for cls in CLASSES:
        p = probs.get(cls, 0.0)
        marker = " ◄" if cls == entry["action"] else ""
        print(f"  {cls:10s}  {_bar(p)}  {p:.3f}{marker}")

    feats = entry.get("top_features", {})
    if feats:
        print()
        print("  Top features:")
        for name, val in list(feats.items())[:5]:
            print(f"    {name:<30s}  {val:.4f}")

    print()
    print(f"  Image: {entry['img_path']}")
    print("═" * 60)


def interactive_label(
    samples:     list[dict],
    open_images: bool = True,
) -> int:
    """
    Present samples one at a time and collect labels.

    Returns the number of labels collected in this session.
    """
    if not samples:
        print("No uncertain samples to label.")
        return 0

    print(f"\n{'='*60}")
    print(f"  Active learning: {len(samples)} samples to label")
    print(f"  Commands: S=STOP  W=SLOW  C=CONTINUE  Enter=skip  Q=quit")
    print(f"{'='*60}")

    collected = 0

    for i, entry in enumerate(samples, 1):
        _render_entry(entry, i, len(samples))

        # Try to open the image
        if open_images:
            try:
                from PIL import Image
                img_path = Path(entry["img_path"])
                if img_path.exists():
                    Image.open(img_path).show()
            except Exception:
                pass   # headless or file missing — continue without

        # Prompt loop
        while True:
            try:
                raw = input("  Label [S/W/C / Enter=skip / Q=quit]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nLabelling interrupted.")
                return collected

            if raw == "q":
                print(f"\nSession ended. Labelled {collected} samples this session.")
                return collected

            if raw == "":
                print("  Skipped.")
                break

            label = _SHORTCUT.get(raw)
            if label is None:
                print("  Unrecognised input. Use S, W, C, or Enter to skip.")
                continue

            append_label(entry["stem"], label)
            print(f"  ✓ Saved: {entry['stem']} → {label}")
            collected += 1
            break

    print(f"\nDone. Labelled {collected} new samples this session.")
    return collected


# ══════════════════════════════════════════════════════════════════════════════
# Headless export (for Northflank / remote sessions)
# ══════════════════════════════════════════════════════════════════════════════

def export_review_bundle(
    samples:    list[dict],
    export_dir: Path,
    img_dirs:   list[Path],
) -> None:
    """
    Copy the top-k uncertain images + a metadata JSON into export_dir.

    Download this folder locally, run the label command, then re-upload
    labels.json and retrain.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    meta = []

    for entry in samples:
        # Find the image in any of the provided search dirs
        found = None
        stem = entry["stem"]
        for d in img_dirs:
            for ext in (".jpg", ".jpeg", ".png"):
                p = d / (stem + ext)
                if p.exists():
                    found = p
                    break
            if found:
                break

        dest_img = None
        if found:
            dest_img = export_dir / found.name
            shutil.copy2(found, dest_img)

        meta.append({
            **entry,
            "local_image": str(dest_img.name) if dest_img else None,
        })

    meta_path = export_dir / "review_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Review bundle exported to {export_dir}")
    print(f"  {len(samples)} images + review_metadata.json")
    print(f"  Run locally:  python action_module/active_learning.py --label"
          f" --review-dir {export_dir}")


def interactive_label_from_bundle(review_dir: Path) -> int:
    """Label from a previously exported review bundle (works locally)."""
    meta_path = review_dir / "review_metadata.json"
    if not meta_path.exists():
        print(f"No review_metadata.json found in {review_dir}")
        return 0

    with open(meta_path) as f:
        samples = json.load(f)

    # Patch img_path to point to the local bundle copies
    for s in samples:
        local = s.get("local_image")
        if local:
            candidate = review_dir / local
            if candidate.exists():
                s["img_path"] = str(candidate)

    return interactive_label(samples, open_images=True)


# ══════════════════════════════════════════════════════════════════════════════
# Retrain helper
# ══════════════════════════════════════════════════════════════════════════════

def retrain(
    det_dir:          Path,
    graphs_dir:       Path | None = None,
    labels_path:      Path = LABELS_PATH,
    pseudo_threshold: float = 0.85,
) -> None:
    """
    Train the classifier using:
      1. Pseudo-labels from high-confidence rule engine predictions
         (all detections in det_dir with rule conf >= pseudo_threshold).
      2. Manual labels from labels.json, which override pseudo-labels.

    The pipeline is: StandardScaler → PCA (95% variance) → GradientBoosting.
    PCA reduces the 43-d feature vector to ~8-15 components before the tree sees it.
    """
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))
    from action_module.graph_classifier import GraphClassifier

    manual_labels = labels_as_dict(labels_path)
    print(f"Manual labels: {len(manual_labels)}  |  pseudo threshold: {pseudo_threshold}")

    clf = GraphClassifier()
    clf.train_from_pseudo_labels(
        detections_dir  = det_dir,
        graphs_dir      = graphs_dir,
        conf_threshold  = pseudo_threshold,
        extra_labels    = manual_labels if manual_labels else None,
    )
    clf.save()

    pca_info = clf.pca_summary()
    if pca_info:
        print(f"\nPCA: 43d → {pca_info['n_components']}d  "
              f"({pca_info['total_variance']:.1%} variance retained)")

    importances = clf.feature_importances()
    if importances:
        top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop-10 original-space feature importances (projected through PCA):")
        for name, imp in top:
            print(f"  {name:<35s}  {imp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Integration helper for pipeline.py / benchmark.py
# ══════════════════════════════════════════════════════════════════════════════

def maybe_run_active_learning(
    buffer:       UncertaintyBuffer,
    buffer_path:  Path,
    det_dir:      Path,
    graphs_dir:   Path | None = None,
    top_k:        int  = 30,
    strategy:     str  = "entropy",
    headless:     bool = False,
    export_dir:   Path | None = None,
    img_dirs:     list[Path] | None = None,
) -> None:
    """
    Call this at the end of benchmark.py / pipeline.py to trigger the active
    learning session automatically.

    headless=True  → export review bundle, skip interactive prompt
    headless=False → open images interactively, collect labels, retrain
    """
    buffer.save(buffer_path)

    samples = buffer.most_uncertain(top_k, strategy=strategy)
    if not samples:
        print("Active learning: no uncertain samples found.")
        return

    if headless:
        if export_dir is None:
            export_dir = buffer_path.parent / "review_bundle"
        export_review_bundle(samples, export_dir, img_dirs or [])
        return

    n_labelled = interactive_label(samples, open_images=True)

    if n_labelled > 0:
        print(f"\nRetraining classifier with {len(labels_as_dict())} total labels ...")
        retrain(det_dir, graphs_dir)
    else:
        print("No new labels — skipping retrain.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Active learning for graph classifier")
    p.add_argument("--buffer",     type=Path, default=None,
                   help="Path to uncertainty_buffer.json")
    p.add_argument("--det-dir",    type=Path,
                   default=Path("data/pipeline_output/detections"),
                   help="Detections directory (for retrain)")
    p.add_argument("--graphs-dir", type=Path, default=None,
                   help="Scene graphs directory (optional)")
    p.add_argument("--img-dir",    type=Path, action="append", dest="img_dirs",
                   default=None,
                   help="Image search directory (repeat for multiple)")
    p.add_argument("--top-k",      type=int, default=30,
                   help="Number of uncertain samples to present")
    p.add_argument("--strategy",   choices=["entropy", "margin", "least_conf"],
                   default="entropy")
    p.add_argument("--export",     type=Path, default=None,
                   help="Export review bundle to this directory (headless mode)")
    p.add_argument("--review-dir", type=Path, default=None,
                   help="Label from an exported review bundle")
    p.add_argument("--retrain",    action="store_true",
                   help="Retrain immediately from saved labels.json")
    p.add_argument("--pseudo-threshold", type=float, default=0.85,
                   help="Min rule confidence to accept as pseudo-label (default 0.85)")
    p.add_argument("--label",      action="store_true",
                   help="Run interactive labelling from --review-dir bundle")
    return p.parse_args()


def main():
    args = _parse_args()
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))

    # ── Retrain only ──────────────────────────────────────────────────────────
    if args.retrain:
        retrain(args.det_dir, args.graphs_dir,
                pseudo_threshold=args.pseudo_threshold)
        return

    # ── Label from a downloaded bundle ────────────────────────────────────────
    if args.label:
        review_dir = args.review_dir
        if review_dir is None:
            print("Provide --review-dir when using --label.")
            sys.exit(1)
        n = interactive_label_from_bundle(review_dir)
        if n > 0:
            print(f"\nRetrain? ({len(labels_as_dict())} total labels)")
            ans = input("  Retrain now? [Y/n]: ").strip().lower()
            if ans in ("", "y"):
                retrain(args.det_dir, args.graphs_dir)
        return

    # ── Load buffer and run ───────────────────────────────────────────────────
    if args.buffer is None:
        default = ROOT / "data" / "pipeline_output" / "uncertainty_buffer.json"
        if default.exists():
            args.buffer = default
        else:
            print("Provide --buffer path to uncertainty_buffer.json.")
            sys.exit(1)

    buf     = UncertaintyBuffer.load(args.buffer)
    samples = buf.most_uncertain(args.top_k, strategy=args.strategy)

    print(f"Loaded {len(buf)} predictions. Top-{args.top_k} most uncertain selected.")

    if args.export:
        export_review_bundle(samples, args.export, args.img_dirs or [ROOT / "data"])
        return

    n = interactive_label(samples, open_images=True)
    if n > 0:
        retrain(args.det_dir, args.graphs_dir)


if __name__ == "__main__":
    main()
