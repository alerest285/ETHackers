"""
Train the GraphClassifier on pre-computed pipeline JSONs (CPU only).

Reads enriched detection JSONs (+ optional scene graph JSONs) that were
produced by pipeline.py or benchmark.py, pseudo-labels them via the rule
engine, and fits the GradientBoosting classifier.

No torch, no transformers, no GPU required.

Usage (local):
    python scripts/train_action_model.py

Usage (Northflank Job — JSONs mounted at /data):
    python scripts/train_action_model.py \
        --det-dir /data/detections \
        --graph-dir /data/scene_graphs \
        --output-dir /data/output

Environment variables:
    PSEUDO_THRESHOLD — rule-engine confidence cutoff (default 0.85).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from action_module.graph_classifier import GraphClassifier


def train(
    det_dir:          Path,
    graph_dir:        Path | None,
    output_dir:       Path,
    pseudo_threshold: float = 0.85,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "graph_classifier.joblib"

    # Load optional manual labels from action_module/labels.json
    labels_path = ROOT / "action_module" / "labels.json"
    extra_labels: dict[str, str] | None = None
    if labels_path.exists():
        with open(labels_path) as f:
            extra_labels = json.load(f)
        print(f"Loaded {len(extra_labels)} manual labels from {labels_path}")

    clf = GraphClassifier(weights_path=weights_path)

    print("Training classifier ...\n")
    clf.train_from_pseudo_labels(
        detections_dir = det_dir,
        graphs_dir     = graph_dir,
        conf_threshold = pseudo_threshold,
        extra_labels   = extra_labels,
    )

    clf.save(weights_path)
    print(f"\nModel saved to {weights_path}")

    # Copy to canonical repo location so benchmark.py picks it up directly
    canonical = ROOT / "action_module" / "graph_classifier.joblib"
    if weights_path.resolve() != canonical.resolve():
        import shutil
        shutil.copy2(weights_path, canonical)
        print(f"Copied to   {canonical}")

    # Diagnostics
    importances = clf.feature_importances()
    if importances:
        print("\nTop-10 features:")
        for name, imp in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {name:<30s} {imp:.4f}")

    pca = clf.pca_summary()
    if pca:
        print(f"\nPCA: {pca['n_components']} components, "
              f"{pca['total_variance']:.1%} variance retained")


def main():
    default_det   = ROOT / "data" / "pipeline_output" / "detections"
    default_graph = ROOT / "data" / "pipeline_output" / "scene_graphs"

    parser = argparse.ArgumentParser(
        description="Train the action classifier on pre-computed pipeline JSONs (CPU only)",
    )
    parser.add_argument("--det-dir",   type=Path, default=default_det,
                        help="Directory of enriched detection JSONs")
    parser.add_argument("--graph-dir", type=Path, default=default_graph,
                        help="Directory of scene graph JSONs (optional)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "training_output",
                        help="Where to save the trained .joblib weights")
    parser.add_argument("--pseudo-threshold", type=float,
                        default=float(os.environ.get("PSEUDO_THRESHOLD", "0.85")),
                        help="Rule-engine confidence cutoff for pseudo-labels")
    args = parser.parse_args()

    graph_dir = args.graph_dir if args.graph_dir.exists() else None

    n_det = len(list(args.det_dir.glob("*.json"))) if args.det_dir.exists() else 0
    n_graph = len(list(args.graph_dir.glob("*.json"))) if graph_dir else 0

    print("=" * 60)
    print("  ETHackers — Action Model Training (CPU)")
    print("=" * 60)
    print(f"  Detections:      {args.det_dir} ({n_det} files)")
    print(f"  Scene graphs:    {graph_dir or 'none'} ({n_graph} files)")
    print(f"  Output:          {args.output_dir}")
    print(f"  Pseudo thresh:   {args.pseudo_threshold}")
    print("=" * 60 + "\n")

    if n_det == 0:
        print("ERROR: no detection JSONs found. Run pipeline.py or benchmark.py first "
              "to generate them, then point --det-dir at the output.")
        sys.exit(1)

    train(
        det_dir          = args.det_dir,
        graph_dir        = graph_dir,
        output_dir       = args.output_dir,
        pseudo_threshold = args.pseudo_threshold,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
