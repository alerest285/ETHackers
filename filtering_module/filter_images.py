"""
filter_images.py
Filters out images that have no detections from the 27-category ontology.

An image is REMOVED if:
  - No objects detected with confidence > CONF_THRESHOLD, OR
  - All detected objects map to BACKGROUND group

This cleans the dataset before expensive downstream processing (depth analysis,
VLM queries) by discarding irrelevant images (e.g., a photo of a pizza).

Usage:
    # Using pre-computed detections from detection_pipeline.py:
    python src/filter_images.py --detections outputs/detections/detections_test.json \
                                 --coco-json data/annotations/test.json \
                                 --output-dir outputs/filtered

    # Re-run detection inline (slower, but single-step):
    python src/filter_images.py --split test --data-dir data --output-dir outputs/filtered \
                                 --run-detection

Outputs:
    outputs/filtered/filtered_image_ids.json  — image IDs to keep
    outputs/filtered/removed_image_ids.json   — image IDs to drop
    outputs/filtered/filter_stats.json        — summary statistics
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def filter_from_detections(
    detections_path: str,
    coco_json_path: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    require_non_background: bool = True,
) -> dict:
    """
    Filter images based on pre-computed detections JSON.

    Args:
        detections_path:      Path to detections JSON from detection_pipeline.py
        coco_json_path:       Path to COCO JSON (to get the full image ID list)
        output_dir:           Where to write filtered/removed ID lists
        conf_threshold:       Minimum confidence to count a detection
        require_non_background: If True, images with only BACKGROUND detections are removed

    Returns:
        stats dict
    """
    with open(detections_path) as f:
        detections = json.load(f)

    with open(coco_json_path) as f:
        coco = json.load(f)

    all_image_ids: set[int] = {img["id"] for img in coco["images"]}

    # Build per-image detection summary
    image_detections: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        if det["score"] >= conf_threshold:
            image_detections[det["image_id"]].append(det)

    filtered_ids: list[int] = []
    removed_ids: list[int] = []
    removal_reasons: dict[int, str] = {}

    group_counts: dict[str, int] = defaultdict(int)

    for img_id in sorted(all_image_ids):
        dets = image_detections.get(img_id, [])

        if not dets:
            removed_ids.append(img_id)
            removal_reasons[img_id] = "no_detections"
            continue

        if require_non_background:
            non_bg = [d for d in dets if d.get("risk_group", "BACKGROUND") != "BACKGROUND"]
            if not non_bg:
                removed_ids.append(img_id)
                removal_reasons[img_id] = "background_only"
                continue

        filtered_ids.append(img_id)
        for det in dets:
            group_counts[det.get("risk_group", "BACKGROUND")] += 1

    # Write outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "filtered_image_ids.json", "w") as f:
        json.dump(filtered_ids, f)

    with open(output_path / "removed_image_ids.json", "w") as f:
        json.dump(removed_ids, f)

    stats = {
        "total_images": len(all_image_ids),
        "kept_images": len(filtered_ids),
        "removed_images": len(removed_ids),
        "removal_rate_pct": round(len(removed_ids) / max(len(all_image_ids), 1) * 100, 1),
        "removal_reasons": {
            "no_detections": sum(1 for r in removal_reasons.values() if r == "no_detections"),
            "background_only": sum(1 for r in removal_reasons.values() if r == "background_only"),
        },
        "detections_by_group_in_kept": dict(group_counts),
    }

    with open(output_path / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Image Filtering Results ===")
    print(f"Total images:    {stats['total_images']}")
    print(f"Kept:            {stats['kept_images']}  ({100 - stats['removal_rate_pct']:.1f}%)")
    print(f"Removed:         {stats['removed_images']} ({stats['removal_rate_pct']:.1f}%)")
    print(f"  No detections: {stats['removal_reasons']['no_detections']}")
    print(f"  Background only:{stats['removal_reasons']['background_only']}")
    print(f"\nDetections in kept images (by group):")
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"  {group:15s}: {count}")
    print(f"\nOutput → {output_path}")

    return stats


def filter_with_inline_detection(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    weights_path: str,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Run YOLOv8 detection inline and filter in a single pass.
    Useful when you haven't pre-computed detections.
    """
    import sys

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        sys.exit(1)

    from pathlib import Path

    if not Path(weights_path).exists():
        print(f"ERROR: Weights not found: {weights_path}")
        sys.exit(1)

    with open(coco_json_path) as f:
        coco = json.load(f)

    from label_ontology import CLASS_ID_TO_GROUP

    model = YOLO(weights_path)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filtered_ids: list[int] = []
    removed_ids: list[int] = []
    all_detections: list[dict] = []

    for img_info in tqdm(coco["images"], desc="Filtering images"):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = images_path / file_name

        if not img_path.exists():
            removed_ids.append(img_id)
            continue

        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)

        has_relevant = False
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                risk_group = CLASS_ID_TO_GROUP.get(class_id, "BACKGROUND")
                if risk_group != "BACKGROUND":
                    has_relevant = True
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append({
                    "image_id": img_id,
                    "category_id": class_id,
                    "risk_group": risk_group,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": round(conf, 4),
                })

        if has_relevant:
            filtered_ids.append(img_id)
        else:
            removed_ids.append(img_id)

    with open(output_path / "filtered_image_ids.json", "w") as f:
        json.dump(filtered_ids, f)

    with open(output_path / "removed_image_ids.json", "w") as f:
        json.dump(removed_ids, f)

    # Also save the detections for downstream use
    det_path = output_path.parent / "detections" / f"detections_{Path(coco_json_path).stem}.json"
    det_path.parent.mkdir(parents=True, exist_ok=True)
    with open(det_path, "w") as f:
        json.dump(all_detections, f)

    total = len(coco["images"])
    stats = {
        "total_images": total,
        "kept_images": len(filtered_ids),
        "removed_images": len(removed_ids),
        "removal_rate_pct": round(len(removed_ids) / max(total, 1) * 100, 1),
    }

    print(f"\nKept: {len(filtered_ids)}, Removed: {len(removed_ids)}")
    print(f"Output → {output_path}")
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Filter images with no relevant detections")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/filtered")
    parser.add_argument(
        "--detections",
        default="",
        help="Path to pre-computed detections JSON (skip if using --run-detection)",
    )
    parser.add_argument(
        "--run-detection",
        action="store_true",
        help="Run YOLO detection inline instead of using pre-computed detections",
    )
    parser.add_argument(
        "--weights",
        default="models/yolov8s_finetuned.pt",
        help="YOLO weights (only needed with --run-detection)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    coco_json = data_dir / "annotations" / f"{args.split}.json"

    if args.run_detection or not args.detections:
        filter_with_inline_detection(
            coco_json_path=str(coco_json),
            images_dir=str(data_dir / "images" / args.split),
            output_dir=args.output_dir,
            weights_path=args.weights,
            conf_threshold=args.conf,
        )
    else:
        filter_from_detections(
            detections_path=args.detections,
            coco_json_path=str(coco_json),
            output_dir=args.output_dir,
            conf_threshold=args.conf,
        )


if __name__ == "__main__":
    main()
