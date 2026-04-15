"""
run_depth.py — runs the Depth Anything V2 module over a set of test images.

Reads:
  --filtered-detections  : JSON produced by the Filtering stage
                           Format: list of COCO-style dicts with
                           {image_id, bbox, category_id, score}
  --images-dir           : directory containing the images
  --image-list           : path to test.json (COCO images array used to
                           resolve image_id → file_name)

Writes:
  --output               : JSON file, same format as input but each entry
                           additionally contains:
                             depth_score, proximity_label, raw_depth_median

Example
-------
python depth_module/run_depth.py \
    --filtered-detections outputs/filtered_detections.json \
    --images-dir          data/images/test \
    --image-list          data/annotations/test.json \
    --checkpoint          depth_module/checkpoints/depth_anything_v2_vitl.pth \
    --output              outputs/depth_detections.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# Allow running from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from depth_module.depth_module import DepthModule  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Depth Anything V2 inference over filtered detections")
    p.add_argument("--filtered-detections", required=True,
                   help="Path to JSON output from the Filtering stage")
    p.add_argument("--images-dir", required=True,
                   help="Directory containing raw images")
    p.add_argument("--image-list", required=True,
                   help="Path to COCO-format JSON with an 'images' array (e.g. test.json)")
    p.add_argument("--checkpoint", default="depth_module/checkpoints/depth_anything_v2_vitl.pth",
                   help="Path to the ViT-L checkpoint (.pth)")
    p.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"],
                   help="Which encoder variant matches your checkpoint")
    p.add_argument("--output", default="outputs/depth_detections.json",
                   help="Where to write the output JSON")
    p.add_argument("--max-images", type=int, default=None,
                   help="Process at most N images (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ load
    print("Loading filtered detections …")
    with open(args.filtered_detections) as f:
        filtered: list[dict] = json.load(f)

    print("Loading image list …")
    with open(args.image_list) as f:
        image_list_data = json.load(f)
    # Build image_id → file_name mapping
    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"]
        for img in image_list_data["images"]
    }

    # Group detections by image_id for batch processing
    by_image: dict[int, list[dict]] = defaultdict(list)
    for det in filtered:
        by_image[det["image_id"]].append(det)

    # ------------------------------------------------------------------ model
    print(f"Loading Depth Anything V2 ({args.encoder}) from {args.checkpoint} …")
    dm = DepthModule(checkpoint_path=args.checkpoint, encoder=args.encoder)
    dm.load_model()
    print("Model ready.\n")

    # ------------------------------------------------------------------ infer
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    image_ids = sorted(by_image.keys())
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    all_results: list[dict] = []
    n = len(image_ids)

    for i, image_id in enumerate(image_ids, 1):
        filename = id_to_filename.get(image_id)
        if filename is None:
            print(f"  [WARN] image_id={image_id} not found in image list — skipping")
            continue

        image_path = os.path.join(args.images_dir, filename)
        if not os.path.exists(image_path):
            print(f"  [WARN] image file not found: {image_path} — skipping")
            continue

        dets = by_image[image_id]
        try:
            enriched = dm.process_from_path(image_path, dets)
            all_results.extend(enriched)
        except Exception as exc:
            print(f"  [ERROR] image_id={image_id} ({filename}): {exc}")
            # Pass through the original detections without depth info so
            # downstream stages don't lose these detections entirely.
            all_results.extend(dets)

        if i % 100 == 0 or i == n:
            print(f"  {i}/{n} images processed …")

    # ------------------------------------------------------------------ save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} detections written to {args.output}")
    _print_proximity_summary(all_results)


def _print_proximity_summary(results: list[dict]) -> None:
    from collections import Counter
    counts = Counter(r.get("proximity_label", "UNKNOWN") for r in results)
    total = len(results)
    print("\nProximity summary:")
    for label in ("CLOSE", "MEDIUM", "FAR", "UNKNOWN"):
        n = counts[label]
        pct = 100 * n / total if total else 0
        print(f"  {label:8s}: {n:6d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
