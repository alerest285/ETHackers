"""
run_sample.py — pick 100 random train images, run detection + depth, save all results.

Outputs
-------
data/sample/detections/<stem>.json          — YOLOS detections (label, box, score, risk_group)
data/sample/visualizations/<stem>_viz.png   — boxes drawn on original image
data/sample/depth/<stem>_depth.png          — Depth Anything V2 heatmap (original | depth)

Usage
-----
python run_sample.py                        # 100 images, default paths
python run_sample.py --n 10                 # quick test with 10 images
python run_sample.py --seed 99              # different random sample
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

# ── local modules ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from segment_module.segment   import load_model, detect_image
from segment_module.visualize import draw_detections

# ── paths ────────────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("data/challenge/data/images/train")
OUT_DET     = Path("data/sample/detections")
OUT_VIZ     = Path("data/sample/visualizations")
OUT_DEPTH   = Path("data/sample/depth")

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
COLORMAP    = cv2.COLORMAP_INFERNO


# ── depth helper ─────────────────────────────────────────────────────────────

def run_depth(depth_pipe, image_path: Path, out_dir: Path) -> Path:
    """Run depth estimation and save a side-by-side heatmap PNG."""
    image    = Image.open(image_path).convert("RGB")
    result   = depth_pipe(image)
    depth_np = np.array(result["depth"])

    d_min, d_max = depth_np.min(), depth_np.max()
    depth_u8 = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    heatmap      = cv2.applyColorMap(depth_u8, COLORMAP)
    original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_bgr = cv2.resize(original_bgr, (depth_u8.shape[1], depth_u8.shape[0]))
    combined     = np.hstack([original_bgr, heatmap])

    out_path = out_dir / f"{image_path.stem}_depth.png"
    cv2.imwrite(str(out_path), combined)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main(n: int = 100, seed: int = 42, score_threshold: float = 0.5) -> None:
    random.seed(seed)

    # Gather all train images
    all_images = (
        sorted(TRAIN_DIR.glob("*.jpg")) +
        sorted(TRAIN_DIR.glob("*.jpeg")) +
        sorted(TRAIN_DIR.glob("*.png"))
    )
    if not all_images:
        print(f"No images found in {TRAIN_DIR}. Check that the data is extracted.")
        return

    sample = random.sample(all_images, min(n, len(all_images)))
    print(f"Selected {len(sample)} images from {len(all_images)} total train images.")

    # Create output dirs
    for d in [OUT_SEG, OUT_VIZ, OUT_DEPTH]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load models once ──────────────────────────────────────────────────────
    print("\nLoading YOLO ...")
    det_model = load_model()

    print("Loading Depth Anything V2 ...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
    )
    print("Models ready.\n")

    # Create output dirs
    for d in [OUT_DET, OUT_VIZ, OUT_DEPTH]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────────────
    det_errors   = 0
    depth_errors = 0

    for img_path in tqdm(sample, desc="Processing"):
        # Detection
        try:
            image = Image.open(img_path).convert("RGB")
            dets  = detect_image(det_model, image, score_threshold)

            json_path = OUT_DET / f"{img_path.stem}.json"
            with open(json_path, "w") as f:
                json.dump({"image": img_path.name, "detections": dets}, f, indent=2)

            viz_path = OUT_VIZ / f"{img_path.stem}_viz.png"
            draw_detections(img_path, json_path, viz_path)
        except Exception as e:
            print(f"\n[DET ERROR] {img_path.name}: {e}")
            det_errors += 1

        # Depth
        try:
            run_depth(depth_pipe, img_path, OUT_DEPTH)
        except Exception as e:
            print(f"\n[DEPTH ERROR] {img_path.name}: {e}")
            depth_errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Detections    → {OUT_DET}  ({det_errors} errors)")
    print(f"  Visualizations→ {OUT_VIZ}")
    print(f"  Depth         → {OUT_DEPTH}  ({depth_errors} errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=100, help="Number of images to sample")
    parser.add_argument("--seed",      type=int,   default=42,  help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="SAM2 score threshold")
    args = parser.parse_args()

    main(n=args.n, seed=args.seed, score_threshold=args.threshold)
