"""
Generate training data — run the 7-step perception pipeline on images,
save enriched detection + scene graph JSONs, and export a review bundle
of the most uncertain predictions for human labelling.

Designed to run as a Northflank / CoreWeave GPU Job.

Usage (cloud — images mounted at /data, output at /output):
    python scripts/generate_training_data.py \
        --image-dir /data/images/train \
        --output-dir /output \
        --n 500 --fast --export-top-k 100

Usage (local):
    python scripts/generate_training_data.py --n 50 --fast --export-top-k 30

After the Job finishes:
  1. Download /output/review_bundle/ to your laptop.
  2. Label:  python action_module/active_learning.py --label --review-dir review_bundle/
  3. Train:  python scripts/train_action_model.py --det-dir /output/detections

Environment variables:
    OPENAI_API_KEY — required for LLM prompt generation.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3d-module"))

from segment_module.llm_objects    import load_client as load_llm_client, get_detection_prompt
from segment_module.grounding_dino import load_model  as load_gdino,      detect
from segment_module.sam2           import load_model  as load_sam2,       segment
from depth_module.fused_depth      import enrich_detections
from src.edge_rationalization      import rationalize_edge_detections
from src.llm_depth                 import get_llm_depth_signals
from lift_3d                       import SceneGraphBuilder
from action_module.graph_classifier import GraphClassifier
from action_module.active_learning  import UncertaintyBuffer, export_review_bundle

DEPTH_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
SAM2_BASE   = "facebook/sam2-hiera-base-plus"
SAM2_LARGE  = "facebook/sam2-hiera-large"


def _norm_depth(raw: np.ndarray) -> np.ndarray:
    d_min, d_max = raw.min(), raw.max()
    return (1.0 - (raw - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)


def _serialise(detections: list[dict]) -> list[dict]:
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


def run(
    image_dir:    Path,
    output_dir:   Path,
    n:            int | None,
    conf:         float,
    fast:         bool,
    seed:         int | None,
    export_top_k: int,
):
    det_dir   = output_dir / "detections"
    graph_dir = output_dir / "scene_graphs"
    det_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images ───────────────────────────────────────────────────────
    all_images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in image_dir.glob(ext)
    )
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    if seed is not None:
        random.seed(seed)
        random.shuffle(all_images)
    if n is not None:
        all_images = all_images[:n]

    # Skip already-processed (resume support)
    todo = [p for p in all_images if not (det_dir / f"{p.stem}.json").exists()]
    print(f"Images: {len(all_images)} total, {len(all_images) - len(todo)} already done, "
          f"{len(todo)} to process.\n")

    # ── Load models ──────────────────────────────────────────────────────────
    use_bf16     = fast and torch.cuda.is_available()
    th_dtype     = torch.bfloat16 if use_bf16 else torch.float32
    depth_model  = DEPTH_LARGE if fast else DEPTH_SMALL
    sam2_model   = SAM2_LARGE  if fast else SAM2_BASE

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

    if fast and torch.cuda.is_available():
        try:
            depth_pipe.model = torch.compile(depth_pipe.model, mode="reduce-overhead")
            print("  torch.compile applied to depth model.")
        except Exception as e:
            print(f"  torch.compile skipped ({e})")

    graph_builder = SceneGraphBuilder(point_cloud_step=4)
    safety_rules  = (ROOT / "action_module" / "SAFETY_RULES.md").read_text()
    clf           = GraphClassifier()
    unc_buffer    = UncertaintyBuffer()
    print("All models loaded.\n")

    # ── Process images ───────────────────────────────────────────────────────
    for img_path in tqdm(todo, desc="Processing"):
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
            tqdm.write(f"  [{stem}] prompt failed ({e}) — default")
            prompt = "person . vehicle . forklift . barrel . cone . box . ladder ."

        # Step 2 — Grounding DINO
        try:
            detections = detect(gdino, image, prompt, score_threshold=conf)
        except Exception as e:
            tqdm.write(f"  [{stem}] GDINO failed ({e})")
            detections = []

        if not detections:
            with open(det_dir / f"{stem}.json", "w") as f:
                json.dump({"image": img_path.name, "detections": []}, f, indent=2)
            unc_buffer.record(stem, img_path, {
                "action": "CONTINUE", "confidence": 0.80,
                "probabilities": {"STOP": 0.05, "SLOW": 0.15, "CONTINUE": 0.80},
                "top_features": {}, "source": "rules",
            })
            continue

        # Step 3 — SAM2
        try:
            segmented = segment(sam2, image, detections)
        except Exception as e:
            tqdm.write(f"  [{stem}] SAM2 failed ({e})")
            segmented = [{**d, "mask": np.zeros((H, W), dtype=bool), "mask_score": 0.0}
                         for d in detections]

        # Step 4 — Depth
        try:
            raw_depth  = np.array(depth_pipe(image)["depth"], dtype=np.float32)
            norm_depth = _norm_depth(raw_depth)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth failed ({e})")
            norm_depth = np.full((H, W), 0.5, dtype=np.float32)

        if not fast:
            try:
                segmented = get_llm_depth_signals(image, segmented, llm_client)
            except Exception:
                pass
            try:
                segmented = rationalize_edge_detections(image, segmented, llm_client)
            except Exception:
                pass

        # Step 4c — Fused depth
        try:
            enriched, corrected_depth = enrich_detections(segmented, norm_depth, H, W)
        except Exception:
            enriched, corrected_depth = segmented, norm_depth

        # Step 5 — Scene graph
        masks = [d.get("mask", np.zeros((H, W), dtype=bool)) for d in segmented]
        try:
            graph = graph_builder.process(corrected_depth, enriched, W, H, stem, masks)
            graph_builder.save(graph, graph_dir)
        except Exception as e:
            tqdm.write(f"  [{stem}] scene graph failed ({e})")
            graph = None

        # Classify (rule engine or trained model) — for uncertainty ranking
        clf_result = clf.predict(enriched, graph, image)
        unc_buffer.record(stem, img_path, clf_result)

        tqdm.write(f"  [{stem}] {clf_result['action']} ({clf_result['confidence']:.2f})")

        # Save enriched detections
        with open(det_dir / f"{stem}.json", "w") as f:
            json.dump({"image": img_path.name, "detections": _serialise(enriched)}, f, indent=2)

    # ── Save uncertainty buffer ──────────────────────────────────────────────
    buffer_path = output_dir / "uncertainty_buffer.json"
    unc_buffer.save(buffer_path)

    # ── Export review bundle for labelling ────────────────────────────────────
    if export_top_k > 0:
        samples = unc_buffer.most_uncertain(export_top_k, strategy="entropy")
        if samples:
            bundle_dir = output_dir / "review_bundle"
            export_review_bundle(samples, bundle_dir, [image_dir])
            print(f"\nReview bundle: {bundle_dir} ({len(samples)} samples)")
            print("Download this folder, then label locally:")
            print(f"  python action_module/active_learning.py --label --review-dir {bundle_dir}")

    n_det = len(list(det_dir.glob("*.json")))
    n_graph = len(list(graph_dir.glob("*.json")))
    print(f"\nDone. {n_det} detections, {n_graph} scene graphs saved to {output_dir}")


def main():
    default_images = ROOT / "data" / "challenge" / "data" / "images" / "train"

    parser = argparse.ArgumentParser(
        description="Generate training data: pipeline + uncertainty ranking + review bundle",
    )
    parser.add_argument("--image-dir",    type=Path, default=default_images)
    parser.add_argument("--output-dir",   type=Path, default=ROOT / "data" / "training_output")
    parser.add_argument("--n",            type=int,  default=None,  help="Limit to N images (default: all)")
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--fast",         action="store_true", help="Larger models, BF16, skip LLM depth")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--export-top-k", type=int,  default=100,
                        help="Export N most uncertain images for labelling (0 to skip)")
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print("  ETHackers — Training Data Generation")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Images:          {args.image_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  Fast mode:       {args.fast}")
    print(f"  Export top-k:    {args.export_top_k}")
    print("=" * 60 + "\n")

    run(
        image_dir    = args.image_dir,
        output_dir   = args.output_dir,
        n            = args.n,
        conf         = args.conf,
        fast         = args.fast,
        seed         = args.seed,
        export_top_k = args.export_top_k,
    )


if __name__ == "__main__":
    main()
