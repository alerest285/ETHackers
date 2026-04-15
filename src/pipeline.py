"""
Main pre-processing pipeline (Phase 1–5).

Steps:
  1. Sample N random images from the training set.
  2. Run YOLO detection (segment_module) → labeled boxes per image.
  3. Filter with filter_module → discard images with no relevant objects.
  4. Run DepthAnything (depth_module) on remaining images.
  5. Combine YOLO boxes + depth info into a single overlay image (depth_overlay).
  6. Run Qwen2.5-VL (llm_module) on each kept image → scene analysis text.

Outputs saved to data/pipeline_output/:
  detections/<stem>.json          — YOLO detections enriched with depth info
  overlays/<stem>_overlay.png     — side-by-side: YOLO boxes | depth heatmap
  llm/<stem>_analysis.txt         — Qwen scene description + navigation decision
  discarded/<stem>.json           — detections for discarded images (for reference)

Usage:
  python src/pipeline.py                  # 5 images
  python src/pipeline.py --n 20           # 20 images
  python src/pipeline.py --seed 7         # different random sample
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from segment_module.segment  import load_model as load_yolo, detect_image
from src.label_ontology      import map_detections
from src.filter_module       import filter_images
from src.depth_overlay       import make_overlay
from llm_module.llm          import get_client as get_llm_client, process_one as llm_process

# ── paths ─────────────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("data/challenge/data/images/train")
OUT_ROOT    = Path("data/pipeline_output")
OUT_DET     = OUT_ROOT / "detections"
OUT_OVERLAY = OUT_ROOT / "overlays"
OUT_DISC    = OUT_ROOT / "discarded"
OUT_LLM     = OUT_ROOT / "llm"

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ── depth helper ──────────────────────────────────────────────────────────────

def get_depth_map(depth_pipe, image: Image.Image) -> np.ndarray:
    """Return a (H, W) float32 raw depth array for the image."""
    result = depth_pipe(image)
    return np.array(result["depth"], dtype=np.float32)


def enrich_with_depth(detections: list[dict], depth_map: np.ndarray) -> list[dict]:
    """
    Add depth_score and proximity_label to each detection.
    Reuses the logic from depth_module.DepthModule.process() without
    requiring a local checkpoint.
    """
    from src.depth_overlay import _proximity

    H, W    = depth_map.shape
    d_min   = depth_map.min()
    d_max   = depth_map.max()
    # DepthAnything outputs disparity: higher value = closer object.
    # Invert so that norm=0 → far, norm=1 → close, matching proximity thresholds.
    norm    = 1.0 - (depth_map - d_min) / (d_max - d_min + 1e-6)

    enriched = []
    for det in detections:
        d    = dict(det)
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)

        if x2 > x1 and y2 > y1:
            roi          = norm[y1:y2, x1:x2]
            depth_score  = float(np.median(roi))
        else:
            depth_score  = float(np.median(norm))

        d["depth_score"]      = round(depth_score, 4)
        d["proximity_label"]  = _proximity(depth_score)
        enriched.append(d)

    return enriched


# ── main ──────────────────────────────────────────────────────────────────────

def run(n: int = 5, seed: int = 42, conf: float = 0.25) -> None:
    random.seed(seed if seed is not None else random.randint(0, 2**32))

    # ── 1. Sample images ──────────────────────────────────────────────────────
    all_images = (
        sorted(TRAIN_DIR.glob("*.jpg")) +
        sorted(TRAIN_DIR.glob("*.jpeg")) +
        sorted(TRAIN_DIR.glob("*.png"))
    )
    if not all_images:
        print(f"No images found in {TRAIN_DIR}")
        return

    sample = random.sample(all_images, min(n, len(all_images)))
    print(f"Sampled {len(sample)} images from {len(all_images)} total.\n")

    # ── 2. YOLO detection ─────────────────────────────────────────────────────
    print("Loading YOLO ...")
    yolo = load_yolo()

    image_detections: dict[str, list[dict]] = {}
    images_cache:     dict[str, Image.Image] = {}

    print("Running YOLO detection ...")
    for img_path in tqdm(sample, desc="Detection"):
        image = Image.open(img_path).convert("RGB")
        dets  = detect_image(yolo, image, score_threshold=conf)
        dets  = map_detections(dets)          # add risk_group + risk_score
        image_detections[img_path.name] = dets
        images_cache[img_path.name]     = image

    # ── 3. Filter ─────────────────────────────────────────────────────────────
    kept, discarded = filter_images(image_detections, conf_threshold=conf)
    print(f"\nFilter results: {len(kept)} kept, {len(discarded)} discarded.")
    if discarded:
        print(f"  Discarded: {discarded}")

    if not kept:
        print("All images were discarded — nothing relevant detected. Try a larger sample.")
        return

    # Save discarded detections for reference
    OUT_DISC.mkdir(parents=True, exist_ok=True)
    for name in discarded:
        with open(OUT_DISC / f"{Path(name).stem}.json", "w") as f:
            json.dump({"image": name, "detections": image_detections[name]}, f, indent=2)

    # ── 4. Depth estimation ───────────────────────────────────────────────────
    print("\nLoading DepthAnything V2 ...")
    depth_pipe = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL)

    OUT_DET.mkdir(parents=True, exist_ok=True)
    OUT_OVERLAY.mkdir(parents=True, exist_ok=True)

    print("Running depth + building overlays ...")
    for name in tqdm(kept, desc="Depth + Overlay"):
        image = images_cache[name]
        dets  = image_detections[name]

        # Get depth map
        depth_map = get_depth_map(depth_pipe, image)

        # ── 5. Enrich detections + build overlay ─────────────────────────────
        dets_enriched = enrich_with_depth(dets, depth_map)

        # Save enriched detections JSON
        with open(OUT_DET / f"{Path(name).stem}.json", "w") as f:
            json.dump({"image": name, "detections": dets_enriched}, f, indent=2)

        # Save overlay image
        overlay_bgr = make_overlay(image, dets_enriched, depth_map)
        cv2.imwrite(str(OUT_OVERLAY / f"{Path(name).stem}_overlay.png"), overlay_bgr)

    # ── 6. LLM scene analysis ─────────────────────────────────────────────────
    print("\nConnecting to Qwen API ...")
    llm_client = get_llm_client()

    print("Running LLM scene analysis ...")
    for name in tqdm(kept, desc="LLM"):
        stem         = Path(name).stem
        overlay_path = OUT_OVERLAY / f"{stem}_overlay.png"
        json_path    = OUT_DET    / f"{stem}.json"

        original_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = TRAIN_DIR / (stem + ext)
            if candidate.exists():
                original_path = candidate
                break

        if original_path is None:
            print(f"  [SKIP] Original image not found for {stem}")
            continue

        llm_process(llm_client, overlay_path, original_path, json_path, OUT_LLM)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Detections  → {OUT_DET}")
    print(f"  Overlays    → {OUT_OVERLAY}  ← open these to check the pipeline")
    print(f"  LLM outputs → {OUT_LLM}      ← scene analysis + navigation decisions")
    print(f"  Discarded   → {OUT_DISC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pre-processing pipeline")
    parser.add_argument("--n",    type=int,   default=5,    help="Number of images to sample")
    parser.add_argument("--seed", type=int,   default=None, help="Random seed (omit for true random)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    run(n=args.n, seed=args.seed, conf=args.conf)
