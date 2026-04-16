"""
GPU training pipeline — process images through the full 7-step pipeline,
then train the GraphClassifier on the resulting enriched detections.

Two phases:
  Phase 1 (GPU): iterate over training images → run detection, segmentation,
                  depth estimation, scene graph → save enriched JSONs.
  Phase 2 (CPU): load JSONs → pseudo-label via rule engine → train the
                  GradientBoosting classifier → save weights.

Usage (Northflank Job — data mounted at /data, output at /output):
    python scripts/train_action_model.py \
        --data-dir /data --output-dir /output --fast

Usage (local):
    python scripts/train_action_model.py --n 200 --fast

Environment variables:
    OPENAI_API_KEY   — required for LLM prompt generation + LLM decisions.
    PSEUDO_THRESHOLD — rule-engine confidence cutoff (default 0.85).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── GPU setup ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3d-module"))

# ── Local imports ────────────────────────────────────────────────────────────
from segment_module.llm_objects    import load_client as load_llm_client, get_detection_prompt
from segment_module.grounding_dino import load_model  as load_gdino,      detect
from segment_module.sam2           import load_model  as load_sam2,       segment
from depth_module.fused_depth      import enrich_detections
from src.filter_module             import is_interesting
from src.edge_rationalization      import rationalize_edge_detections
from src.llm_depth                 import get_llm_depth_signals
from lift_3d                       import SceneGraphBuilder
from action_module.graph_classifier import GraphClassifier

# ── Model IDs ────────────────────────────────────────────────────────────────
DEPTH_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
SAM2_BASE   = "facebook/sam2-hiera-base-plus"
SAM2_LARGE  = "facebook/sam2-hiera-large"


def _norm_depth(raw: np.ndarray) -> np.ndarray:
    """Invert DepthAnything disparity -> normalised depth (0=close, 1=far)."""
    d_min, d_max = raw.min(), raw.max()
    return (1.0 - (raw - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)


def _serialise(detections: list[dict]) -> list[dict]:
    """Return a JSON-safe copy of detections (strips numpy arrays)."""
    out = []
    for d in detections:
        entry = {k: v for k, v in d.items() if k != "mask"}
        for k, v in entry.items():
            if isinstance(v, np.floating):
                entry[k] = float(v)
            elif isinstance(v, np.integer):
                entry[k] = int(v)
        out.append(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: generate enriched detections + scene graphs for all images
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_data(
    image_dir:  Path,
    det_dir:    Path,
    graph_dir:  Path,
    n:          int | None = None,
    conf:       float = 0.25,
    fast:       bool = True,
) -> int:
    """
    Run the 7-step pipeline on every image in `image_dir`.
    Saves enriched detection JSONs to `det_dir` and scene graph JSONs/txt to
    `graph_dir`.  Returns the number of images successfully processed.

    Resumable: images whose detection JSON already exists are skipped.
    """
    det_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    all_images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in image_dir.glob(ext)
    )
    if not all_images:
        print(f"No images found in {image_dir}")
        return 0
    if n is not None:
        all_images = all_images[:n]

    # Skip already-processed images (resume support)
    todo = [p for p in all_images if not (det_dir / f"{p.stem}.json").exists()]
    print(f"Images: {len(all_images)} total, {len(all_images) - len(todo)} already done, "
          f"{len(todo)} to process.\n")
    if not todo:
        return len(all_images)

    # ── Load models ──────────────────────────────────────────────────────────
    use_bf16 = fast and torch.cuda.is_available()
    th_dtype = torch.bfloat16 if use_bf16 else torch.float32
    depth_model = DEPTH_LARGE if fast else DEPTH_SMALL
    sam2_model  = SAM2_LARGE  if fast else SAM2_BASE

    print("Loading models ...")
    llm_client    = load_llm_client()
    gdino         = load_gdino()
    sam2          = load_sam2(sam2_model)

    depth_kwargs: dict = dict(task="depth-estimation", model=depth_model)
    if use_bf16:
        depth_kwargs["torch_dtype"] = th_dtype
    if torch.cuda.is_available():
        depth_kwargs["device"] = 0
    depth_pipe = hf_pipeline(**depth_kwargs)

    # torch.compile for throughput on B200
    if fast and torch.cuda.is_available():
        try:
            depth_pipe.model = torch.compile(depth_pipe.model, mode="reduce-overhead")
            print("  torch.compile applied to depth model.")
        except Exception as e:
            print(f"  torch.compile skipped ({e})")

    graph_builder = SceneGraphBuilder(point_cloud_step=4)
    print("All models loaded.\n")

    # ── Process images ───────────────────────────────────────────────────────
    processed = len(all_images) - len(todo)
    for img_path in tqdm(todo, desc="Phase 1 — pipeline"):
        stem = img_path.stem
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  [{stem}] cannot open ({e}) — skipping")
            continue

        W, H = image.size

        # Step 1 — LLM detection prompt
        try:
            prompt = get_detection_prompt(llm_client, image)
        except Exception as e:
            tqdm.write(f"  [{stem}] prompt failed ({e}) — default prompt")
            prompt = "person . vehicle . forklift . barrel . cone . box . ladder ."

        # Step 2 — Grounding DINO
        try:
            detections = detect(gdino, image, prompt, score_threshold=conf)
        except Exception as e:
            tqdm.write(f"  [{stem}] GDINO failed ({e}) — skipping")
            continue

        if not detections:
            # Save empty detection file so we don't re-process on resume.
            with open(det_dir / f"{stem}.json", "w") as f:
                json.dump({"image": img_path.name, "detections": []}, f, indent=2)
            continue

        # Step 3 — SAM2 segmentation
        try:
            segmented = segment(sam2, image, detections)
        except Exception as e:
            tqdm.write(f"  [{stem}] SAM2 failed ({e}) — masks zeroed")
            segmented = [{**d, "mask": np.zeros((H, W), dtype=bool), "mask_score": 0.0}
                         for d in detections]

        # Step 4 — DepthAnything V2
        try:
            raw_depth  = np.array(depth_pipe(image)["depth"], dtype=np.float32)
            norm_depth = _norm_depth(raw_depth)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth failed ({e}) — uniform")
            norm_depth = np.full((H, W), 0.5, dtype=np.float32)

        # Step 4a/4b — LLM depth + edge rationalization (skipped in --fast)
        if not fast:
            try:
                segmented = get_llm_depth_signals(image, segmented, llm_client)
            except Exception:
                pass
            try:
                segmented = rationalize_edge_detections(image, segmented, llm_client)
            except Exception:
                pass

        # Step 4c — Fused depth enrichment
        try:
            enriched, corrected_depth = enrich_detections(segmented, norm_depth, H, W)
        except Exception as e:
            tqdm.write(f"  [{stem}] fused_depth failed ({e})")
            enriched, corrected_depth = segmented, norm_depth

        # Step 5 — Scene graph
        masks = [d.get("mask", np.zeros((H, W), dtype=bool)) for d in segmented]
        try:
            graph = graph_builder.process(
                depth_map  = corrected_depth,
                detections = enriched,
                img_w      = W,
                img_h      = H,
                image_id   = stem,
                masks      = masks,
            )
            graph_builder.save(graph, graph_dir)
        except Exception as e:
            tqdm.write(f"  [{stem}] scene graph failed ({e})")

        # Save enriched detections
        with open(det_dir / f"{stem}.json", "w") as f:
            json.dump({
                "image":      img_path.name,
                "detections": _serialise(enriched),
            }, f, indent=2)

        processed += 1

    print(f"\nPhase 1 complete. {processed} images processed total.\n")
    return processed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: train the classifier on the generated JSONs
# ══════════════════════════════════════════════════════════════════════════════

def train_classifier(
    det_dir:    Path,
    graph_dir:  Path,
    output_dir: Path,
    pseudo_threshold: float = 0.85,
) -> None:
    """
    Train GraphClassifier on enriched detections via pseudo-labelling.

    1. Rule engine labels every detection JSON with confidence >= threshold.
    2. GradientBoosting classifier fits on the resulting feature matrix.
    3. Saves weights to output_dir/graph_classifier.joblib.
    """
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

    print("Phase 2 — training classifier ...\n")
    clf.train_from_pseudo_labels(
        detections_dir = det_dir,
        graphs_dir     = graph_dir if graph_dir.exists() else None,
        conf_threshold = pseudo_threshold,
        extra_labels   = extra_labels,
    )

    clf.save(weights_path)
    print(f"\nModel saved to {weights_path}")

    # Also copy to canonical repo location for direct use in benchmark.py
    canonical = ROOT / "action_module" / "graph_classifier.joblib"
    if weights_path != canonical:
        import shutil
        shutil.copy2(weights_path, canonical)
        print(f"Copied to   {canonical}")

    # Print diagnostics
    importances = clf.feature_importances()
    if importances:
        print("\nTop-10 features:")
        for name, imp in sorted(importances.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {name:<30s} {imp:.4f}")

    pca = clf.pca_summary()
    if pca:
        print(f"\nPCA: {pca['n_components']} components, "
              f"{pca['total_variance']:.1%} variance retained")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline + classifier training (designed for Northflank B200)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data" / "challenge" / "data" / "images" / "train",
        help="Directory containing training images (default: repo train set)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "training_output",
        help="Root output dir for detections, graphs, and model weights",
    )

    # Processing options
    parser.add_argument("--n",    type=int,   default=None,  help="Limit to first N images (default: all)")
    parser.add_argument("--conf", type=float, default=0.25,  help="Detection confidence threshold")
    parser.add_argument("--fast", action="store_true",       help="B200 mode: larger models, BF16, skip intermediate LLM steps")
    parser.add_argument(
        "--pseudo-threshold", type=float,
        default=float(os.environ.get("PSEUDO_THRESHOLD", "0.85")),
        help="Rule-engine confidence cutoff for pseudo-labels (default: 0.85)",
    )

    # Phase control
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip Phase 1 (assume JSONs already exist)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Phase 2 (only generate pipeline JSONs)")

    args = parser.parse_args()

    det_dir   = args.output_dir / "detections"
    graph_dir = args.output_dir / "scene_graphs"

    print("=" * 70)
    print("  ETHackers — Action Model Training Pipeline")
    print("=" * 70)
    print(f"  GPU:             {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Images:          {args.data_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  Fast mode:       {args.fast}")
    print(f"  Pseudo thresh:   {args.pseudo_threshold}")
    print(f"  Skip pipeline:   {args.skip_pipeline}")
    print(f"  Skip training:   {args.skip_training}")
    print("=" * 70 + "\n")

    # Phase 1 — generate enriched detections + scene graphs
    if not args.skip_pipeline:
        generate_training_data(
            image_dir = args.data_dir,
            det_dir   = det_dir,
            graph_dir = graph_dir,
            n         = args.n,
            conf      = args.conf,
            fast      = args.fast,
        )

    # Phase 2 — train the classifier
    if not args.skip_training:
        train_classifier(
            det_dir          = det_dir,
            graph_dir        = graph_dir,
            output_dir       = args.output_dir,
            pseudo_threshold = args.pseudo_threshold,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
