"""
Ultralytics YOLO-based object detection for industrial scene understanding.

What YOLO returns per detection:
  - label         : str               — COCO class name (e.g. "person", "car", "truck")
  - box           : [x1, y1, x2, y2] — bounding box in pixels (top-left origin)
  - score         : float [0, 1]      — detection confidence
  - risk_group    : str               — mapped risk group for the pipeline
  - relative_area : float             — fraction of image covered (proxy for proximity)
"""

import json
from pathlib import Path
from typing import Literal

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "yolo11n.pt"   # auto-downloaded on first run; swap for yolo11s/m/l/x for more accuracy

# ---------------------------------------------------------------------------
# COCO label → risk group
# ---------------------------------------------------------------------------

COCO_TO_GROUP: dict[str, str] = {
    "person":        "person",
    "car":           "vehicle",
    "truck":         "vehicle",
    "bus":           "vehicle",
    "train":         "vehicle",
    "airplane":      "vehicle",
    "boat":          "vehicle",
    "bicycle":       "bicycle",
    "motorcycle":    "bicycle",
    "suitcase":      "box",
    "backpack":      "box",
    "handbag":       "box",
    "stop sign":     "cone",
    "parking meter": "cone",
    "bird":          "animal",
    "cat":           "animal",
    "dog":           "animal",
    "horse":         "animal",
    "sheep":         "animal",
    "cow":           "animal",
    "elephant":      "animal",
    "bear":          "animal",
    "zebra":         "animal",
    "giraffe":       "animal",
}

GROUP_RISK: dict[str, float] = {
    "person":  1.0,
    "vehicle": 0.8,
    "bicycle": 0.6,
    "animal":  0.5,
    "cone":    0.4,
    "box":     0.3,
    "other":   0.2,
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = MODEL_ID) -> YOLO:
    """Load a YOLO model (downloads weights automatically on first run)."""
    print(f"Loading YOLO model '{model_id}'...")
    return YOLO(model_id)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_image(
    model: YOLO,
    image: Image.Image,
    score_threshold: float = 0.25,
) -> list[dict]:
    """
    Run YOLO on a single PIL image.

    Returns a list of detections sorted by descending score:
        {
            "label":         str,            — COCO class name
            "risk_group":    str,            — mapped risk group
            "box":           [x1,y1,x2,y2], — pixels, top-left origin
            "score":         float,
            "relative_area": float,          — box area / image area
        }
    """
    W, H   = image.size
    results = model.predict(image, conf=score_threshold, verbose=False)
    boxes   = results[0].boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
        score = round(float(box.conf[0]), 4)
        label = model.names[int(box.cls[0])]
        rel_area = round((x2 - x1) * (y2 - y1) / (W * H), 6)

        detections.append({
            "label":         label,
            "risk_group":    COCO_TO_GROUP.get(label, "other"),
            "box":           [x1, y1, x2, y2],
            "score":         score,
            "relative_area": rel_area,
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def detect_split(
    data_dir:        str | Path,
    split:           Literal["train", "val", "test"] = "test",
    output_dir:      str | Path | None = None,
    score_threshold: float = 0.25,
    max_images:      int | None = None,
) -> None:
    """
    Run YOLO detection on all images in a dataset split.

    Output: one JSON per image at <output_dir>/<stem>.json
    """
    data_dir   = Path(data_dir)
    images_dir = data_dir / "images" / split

    if output_dir is None:
        output_dir = data_dir / "detections" / split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        sorted(images_dir.glob("*.jpg")) +
        sorted(images_dir.glob("*.jpeg")) +
        sorted(images_dir.glob("*.png"))
    )
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Found {len(image_paths)} images in {images_dir}")
    if not image_paths:
        print("No images found — check the data folder is extracted correctly.")
        return

    model = load_model()

    for img_path in tqdm(image_paths, desc=f"Detecting [{split}]"):
        image = Image.open(img_path).convert("RGB")
        dets  = detect_image(model, image, score_threshold)

        with open(output_dir / f"{img_path.stem}.json", "w") as f:
            json.dump({"image": img_path.name, "detections": dets}, f, indent=2)

    print(f"Done. Results saved to {output_dir}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect objects with YOLO (Ultralytics)")
    parser.add_argument("--data-dir",   default="../data")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold",  default=0.25,  type=float)
    parser.add_argument("--max-images", default=None,  type=int)
    args = parser.parse_args()

    detect_split(
        data_dir        = args.data_dir,
        split           = args.split,
        output_dir      = args.output_dir,
        score_threshold = args.threshold,
        max_images      = args.max_images,
    )
