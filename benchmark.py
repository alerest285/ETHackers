"""
Benchmark the pipeline on the validation set and print the score.

What this does:
  1. Runs YOLO on all 3785 val images.
  2. Applies rule-based decisions (aligned to the evaluator's heuristics).
  3. Writes predictions.json in submission format (with detections for +20%).
  4. Runs evaluate_local.py and prints the score.

Usage:
  python benchmark.py                        # full val set
  python benchmark.py --n 200               # quick sanity check (200 images)
  python benchmark.py --out my_preds.json   # custom output path
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

ROOT       = Path(__file__).parent
STARTER    = ROOT / "data" / "challenge" / "starter_kit"
VAL_ANN    = ROOT / "data" / "annotations" / "val.json"
VAL_IMAGES = ROOT / "data" / "challenge" / "data" / "images" / "val"

# submission category IDs (we define our own 4-class mapping)
GROUP_TO_CAT: dict[str, int] = {
    "HUMAN":         1,
    "VEHICLE":       2,
    "OBSTACLE":      3,
    "SAFETY_MARKER": 4,
}

DETECTION_CATEGORIES = [
    {"id": 1, "name": "HUMAN"},
    {"id": 2, "name": "VEHICLE"},
    {"id": 3, "name": "OBSTACLE"},
    {"id": 4, "name": "SAFETY_MARKER"},
]

# ── decision rules (mirror evaluate_local.py heuristics) ─────────────────────

def make_decision(
    detections: list[dict],
    img_w: int,
    img_h: int,
) -> tuple[str, float, str]:
    """
    Rule-based decision using YOLO detections.

    Rules (ordered by priority, matching the evaluator's approximate GT):
      1. HUMAN + height ratio > 0.25          → STOP  0.90
      2. VEHICLE + area > 15% of image         → STOP  0.85
      3. Any HUMAN or VEHICLE present          → SLOW  0.70
      4. SAFETY_MARKER present                 → SLOW  0.60
      5. OBSTACLEs cover > 40% of image width  → SLOW  0.65
      6. Nothing significant                   → CONTINUE 0.80
    """
    img_area = img_w * img_h
    humans, vehicles, obstacles, markers = [], [], [], []

    for det in detections:
        g = det.get("risk_group", "")
        if g == "HUMAN":          humans.append(det)
        elif g == "VEHICLE":      vehicles.append(det)
        elif g == "OBSTACLE":     obstacles.append(det)
        elif g == "SAFETY_MARKER":markers.append(det)

    # Rule 1 — close person
    for det in humans:
        x1, y1, x2, y2 = det["box"]
        h_ratio = (y2 - y1) / img_h
        if h_ratio > 0.25:
            return (
                "STOP", 0.90,
                f"Person at close range (height ratio {h_ratio:.2f}). Immediate stop required."
            )

    # Rule 2 — large vehicle
    for det in vehicles:
        x1, y1, x2, y2 = det["box"]
        area_ratio = (x2 - x1) * (y2 - y1) / img_area
        if area_ratio > 0.15:
            return (
                "STOP", 0.85,
                f"Vehicle occupying {area_ratio:.0%} of frame. Collision risk."
            )

    # Rule 3 — any person or vehicle
    if humans or vehicles:
        parts = []
        if humans:   parts.append(f"{len(humans)} person(s)")
        if vehicles: parts.append(f"{len(vehicles)} vehicle(s)")
        return "SLOW", 0.70, f"Detected {' and '.join(parts)} in scene."

    # Rule 4 — safety marker
    if markers:
        return "SLOW", 0.60, f"{len(markers)} safety marker(s) indicate a hazard zone."

    # Rule 5 — obstacles blocking width
    if obstacles:
        ranges = sorted((det["box"][0], det["box"][2]) for det in obstacles)
        merged: list[tuple[float, float]] = [ranges[0]]
        for start, end in ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_w = sum(e - s for s, e in merged)
        ratio = total_w / img_w
        if ratio > 0.40:
            return (
                "SLOW", 0.65,
                f"Obstacles span {ratio:.0%} of image width, partially blocking path."
            )
        return (
            "CONTINUE", 0.75,
            f"{len(obstacles)} obstacle(s) detected but not blocking path width."
        )

    return "CONTINUE", 0.80, "No significant hazards detected. Path appears clear."


# ── main ─────────────────────────────────────────────────────────────────────

def run(n: int | None, out_path: Path, team_name: str) -> None:
    sys.path.insert(0, str(ROOT))
    from segment_module.grounding_dino import load_model, detect_image
    from src.label_ontology import map_detections

    # Load val annotations → filename → image_id map
    with open(VAL_ANN) as f:
        val_coco = json.load(f)
    filename_to_id: dict[str, int] = {
        img["file_name"]: img["id"] for img in val_coco["images"]
    }
    id_to_meta: dict[int, dict] = {img["id"]: img for img in val_coco["images"]}

    # Collect val images
    all_images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in VAL_IMAGES.glob(ext)
    )
    if n:
        all_images = all_images[:n]

    print(f"Benchmarking on {len(all_images)} val images ...\n")

    # Load YOLO (auto-uses fine-tuned weights if available)
    print("Loading YOLO ...")
    model = load_model()
    print(f"  Model: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'loaded'}\n")

    predictions  = []
    detections   = []
    missing      = []

    for img_path in tqdm(all_images, desc="Predicting"):
        fname    = img_path.name
        image_id = filename_to_id.get(fname)
        if image_id is None:
            missing.append(fname)
            continue

        meta  = id_to_meta[image_id]
        img_w = meta["width"]
        img_h = meta["height"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  [SKIP] {fname}: {e}")
            continue
        dets  = detect_image(model, image, score_threshold=0.25)
        dets  = map_detections(dets)

        action, confidence, reasoning = make_decision(dets, img_w, img_h)

        predictions.append({
            "image_id":  image_id,
            "action":    action,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
        })

        # Include detections in submission (worth 20% of score)
        for det in dets:
            cat_id = GROUP_TO_CAT.get(det.get("risk_group", ""), None)
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            detections.append({
                "image_id":   image_id,
                "category_id": cat_id,
                "bbox":       [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                "score":      det["score"],
            })

    if missing:
        print(f"\n  Warning: {len(missing)} images not found in val annotations.")

    submission = {
        "team_name":            team_name,
        "predictions":          predictions,
        "detections":           detections,
        "detection_categories": DETECTION_CATEGORIES,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)

    # Distribution summary
    from collections import Counter
    counts = Counter(p["action"] for p in predictions)
    print(f"\nPredictions written to {out_path}")
    print(f"  Total:    {len(predictions)}")
    for act in ("STOP", "SLOW", "CONTINUE"):
        c = counts.get(act, 0)
        print(f"  {act:10s}: {c:5d}  ({c / len(predictions):.1%})")

    # ── Run local evaluator ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Running evaluate_local.py ...")
    print("=" * 60 + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(STARTER / "evaluate_local.py"),
            "--predictions", str(out_path),
            "--annotations", str(VAL_ANN),
        ],
        cwd=str(STARTER),
    )
    if result.returncode != 0:
        print(f"\nEvaluator exited with code {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=None,
                        help="Limit to first N val images (default: all 3785)")
    parser.add_argument("--out",  type=Path, default=ROOT / "predictions.json",
                        help="Output path for submission JSON")
    parser.add_argument("--team", type=str,  default="ETHackers",
                        help="Team name for submission")
    args = parser.parse_args()

    run(n=args.n, out_path=args.out, team_name=args.team)
