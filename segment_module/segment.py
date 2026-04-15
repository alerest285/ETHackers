"""
SAM2-based image segmentation using HuggingFace Transformers.

SAM2 does NOT use text prompts. It segments objects visually.

There are two modes:
  1. AUTO (default, used on test set)
     A grid of points is spread across the image. SAM2 segments every object
     it can find. No labels are assigned — you get raw masks.

  2. BOX (used on train/val where annotations exist)
     Bounding boxes from COCO annotations are fed as prompts. SAM2 produces
     a precise pixel mask for each box. The category label comes from the annotation.

What SAM2 returns per detection:
  - mask          : (H, W) bool array  — exact pixel outline of the object
  - box           : [x1, y1, x2, y2]  — bounding box in pixels
  - iou_score     : float [0, 1]       — model's confidence that the mask is accurate
  - mask_area_px  : int                — number of pixels inside the mask
  - relative_area : float              — fraction of the full image covered (useful for
                                         spatial risk: large = close/blocking, small = far)
  - label         : str                — category name (BOX mode only, "unknown" in AUTO mode)

Note: to assign semantic labels in AUTO mode, pipe the cropped masks through a
classifier (e.g. CLIP) in a downstream step.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline, Sam2Processor, Sam2Model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "facebook/sam2.1-hiera-base-plus"

# AUTO mode: denser grid = more segments found, but slower
POINTS_PER_BATCH = 64

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = MODEL_ID):
    """Load SAM2 processor and model."""
    print(f"Loading SAM2 from '{model_id}'...")
    processor = Sam2Processor.from_pretrained(model_id)
    model     = Sam2Model.from_pretrained(model_id)
    model.eval()

    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Running on Apple MPS.")
    elif torch.cuda.is_available():
        model = model.to("cuda")
        print("Running on CUDA.")
    else:
        print("Running on CPU.")

    return processor, model


def load_pipeline(model_id: str = MODEL_ID):
    """Load SAM2 as a mask-generation pipeline (used for AUTO mode)."""
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"Loading SAM2 mask-generation pipeline on {device}...")
    return pipeline("mask-generation", model=model_id, device=device)

# ---------------------------------------------------------------------------
# AUTO mode — no prompts, segments everything
# ---------------------------------------------------------------------------

def segment_auto(
    pipe,
    image: Image.Image,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Segment all objects in an image without any prompts.

    Returns one detection per object found:
        {
            "label":         "unknown",
            "mask":          np.ndarray bool (H, W),
            "box":           [x1, y1, x2, y2],
            "iou_score":     float,
            "mask_area_px":  int,
            "relative_area": float,
        }
    """
    outputs = pipe(image, points_per_batch=POINTS_PER_BATCH)

    # Pipeline returns either a list of dicts or {"masks": ..., "scores": ...}
    if isinstance(outputs, dict):
        masks_raw  = outputs.get("masks", [])
        scores_raw = outputs.get("scores", [])
    else:
        # list of dicts: [{"mask": ..., "score": ...}, ...]
        masks_raw  = [o["mask"]  if isinstance(o, dict) else o for o in outputs]
        scores_raw = [o.get("score", 1.0) if isinstance(o, dict) else 1.0 for o in outputs]

    # Normalise scores to a plain list of floats
    if hasattr(scores_raw, "tolist"):
        scores_raw = scores_raw.tolist()

    detections = []
    for i, mask_raw in enumerate(masks_raw):
        score = float(scores_raw[i]) if i < len(scores_raw) else 1.0
        if score < score_threshold:
            continue

        # Accept tensor, PIL image, or numpy array
        if hasattr(mask_raw, "numpy"):
            mask_np = mask_raw.cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask_raw, dtype=bool)

        if mask_np.sum() == 0:
            continue

        # Derive bounding box from the mask itself
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y1 = int(np.argmax(rows))
        y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        x1 = int(np.argmax(cols))
        x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))

        detections.append({
            "label":         "unknown",
            "mask":          mask_np,
            "box":           [x1, y1, x2, y2],
            "iou_score":     round(score, 4),
            "mask_area_px":  int(mask_np.sum()),
            "relative_area": round(float(mask_np.mean()), 6),
        })

    return detections

# ---------------------------------------------------------------------------
# BOX mode — use COCO bounding boxes as prompts
# ---------------------------------------------------------------------------

def segment_with_boxes(
    processor: Sam2Processor,
    model:     Sam2Model,
    image:     Image.Image,
    boxes:     list[list[float]],   # [[x1,y1,x2,y2], ...]
    labels:    list[str],           # one label string per box
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Segment objects using bounding box prompts from COCO annotations.

    Args:
        boxes:  List of [x1, y1, x2, y2] boxes in pixel coordinates.
        labels: Category name for each box (same order as boxes).

    Returns same detection dict format as segment_auto, but with real labels.
    """
    H, W = image.size[1], image.size[0]

    # SAM2 processor expects [batch, boxes, 4] — one image at a time here
    input_boxes = [[[b[0], b[1], b[2], b[3]] for b in boxes]]

    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Post-process: upsample low-res masks back to original image size
    masks_np = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]  # shape: (num_boxes, 1, H, W)

    iou_scores = outputs.iou_scores[0, :, 0].cpu().tolist()  # (num_boxes,)

    detections = []
    for i, (mask_tensor, score) in enumerate(zip(masks_np, iou_scores)):
        if score < score_threshold:
            continue
        mask_np = mask_tensor[0].cpu().numpy().astype(bool)
        box     = boxes[i]
        detections.append({
            "label":         labels[i] if i < len(labels) else "unknown",
            "mask":          mask_np,
            "box":           [round(v, 2) for v in box],
            "iou_score":     round(float(score), 4),
            "mask_area_px":  int(mask_np.sum()),
            "relative_area": round(float(mask_np.mean()), 6),
        })

    return detections

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_detections(detections: list[dict], image_name: str, output_dir: Path) -> None:
    """Save masks as .npy and metadata as .json for one image."""
    masks_dir    = output_dir / Path(image_name).stem
    json_records = []

    if detections:
        masks_dir.mkdir(exist_ok=True)

    for idx, det in enumerate(detections):
        mask_filename = f"{det['label']}_{idx}.npy"
        np.save(masks_dir / mask_filename, det["mask"])

        json_records.append({
            "label":         det["label"],
            "box":           det["box"],
            "iou_score":     det["iou_score"],
            "mask_file":     f"{masks_dir.name}/{mask_filename}",
            "mask_area_px":  det["mask_area_px"],
            "relative_area": det["relative_area"],
        })

    out_json = output_dir / f"{Path(image_name).stem}.json"
    with open(out_json, "w") as f:
        json.dump({"image": image_name, "detections": json_records}, f, indent=2)

# ---------------------------------------------------------------------------
# Main entry: segment a full dataset split
# ---------------------------------------------------------------------------

def segment_split(
    data_dir:        str | Path,
    split:           Literal["train", "val", "test"] = "test",
    output_dir:      str | Path | None = None,
    score_threshold: float = 0.5,
    max_images:      int | None = None,
) -> None:
    """
    Segment all images in a dataset split.

    - test  : AUTO mode (no annotations available)
    - train/val : BOX mode if annotations exist, AUTO otherwise

    Output layout:
        <output_dir>/
            <image_stem>.json      # boxes, scores, labels (no raw masks)
            <image_stem>/
                <label>_<idx>.npy  # one binary mask per detection
    """
    data_dir   = Path(data_dir)
    images_dir = data_dir / "images" / split

    if output_dir is None:
        output_dir = data_dir / "segmentation" / split
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

    # Check if COCO annotations exist for BOX mode
    ann_path = data_dir / "annotations" / f"{split}.json"
    use_box_mode = ann_path.exists() and split != "test"

    if use_box_mode:
        print(f"Annotations found — using BOX mode for [{split}]")
        with open(ann_path) as f:
            coco = json.load(f)
        cat_map   = {c["id"]: c["name"] for c in coco["categories"]}
        img_map   = {img["id"]: img["file_name"] for img in coco["images"]}
        ann_by_fn: dict[str, list] = {}
        for ann in coco["annotations"]:
            fn = img_map[ann["image_id"]]
            ann_by_fn.setdefault(fn, []).append(ann)

        processor, model = load_model()
    else:
        print(f"No annotations — using AUTO mode for [{split}]")
        pipe = load_pipeline()

    for img_path in tqdm(image_paths, desc=f"Segmenting [{split}]"):
        image = Image.open(img_path).convert("RGB")

        if use_box_mode:
            anns   = ann_by_fn.get(img_path.name, [])
            # COCO bbox is [x, y, w, h] — convert to [x1, y1, x2, y2]
            boxes  = [[a["bbox"][0], a["bbox"][1],
                       a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3]] for a in anns]
            labels = [cat_map.get(a["category_id"], "unknown") for a in anns]
            dets   = segment_with_boxes(processor, model, image, boxes, labels,
                                        score_threshold=score_threshold)
        else:
            dets = segment_auto(pipe, image, score_threshold=score_threshold)

        _save_detections(dets, img_path.name, output_dir)

    print(f"Done. Results saved to {output_dir}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment images with SAM2")
    parser.add_argument("--data-dir",   default="../data", help="Root data directory")
    parser.add_argument("--split",      default="test",    choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold",  default=0.5,       type=float)
    parser.add_argument("--max-images", default=None,      type=int)
    args = parser.parse_args()

    segment_split(
        data_dir        = args.data_dir,
        split           = args.split,
        output_dir      = args.output_dir,
        score_threshold = args.threshold,
        max_images      = args.max_images,
    )
