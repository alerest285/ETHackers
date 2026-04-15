"""
Visualize SAM2 segmentation results.

Loads a .json + .npy masks produced by segment.py and draws
colored mask overlays + bounding boxes on the original image.

Usage:
    python visualize.py --image ../data/images/test/download.jpeg
    python visualize.py --split test          # visualize all images in a split
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# One distinct color per detection (RGBA, semi-transparent)
PALETTE = [
    (255,  64,  64, 120),   # red
    ( 64, 160, 255, 120),   # blue
    ( 64, 220,  64, 120),   # green
    (255, 200,   0, 120),   # yellow
    (200,  64, 255, 120),   # purple
    (  0, 220, 200, 120),   # teal
    (255, 140,   0, 120),   # orange
    (220, 220, 220, 120),   # grey
]


def color_for(idx: int) -> tuple:
    return PALETTE[idx % len(PALETTE)]


def overlay_masks(
    image_path: Path,
    json_path:  Path,
    seg_dir:    Path,
    output_path: Path,
) -> Path:
    """
    Draw masks + boxes on the original image and save as PNG.

    Args:
        image_path:  Original image.
        json_path:   Detection JSON produced by segment.py.
        seg_dir:     Root of the segmentation output (where mask .npy files live).
        output_path: Where to save the visualized image.

    Returns the output path.
    """
    image = Image.open(image_path).convert("RGBA")
    W, H  = image.size

    with open(json_path) as f:
        data = json.load(f)

    detections = data.get("detections", [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not detections:
        print(f"No detections for {image_path.name}, saving original.")
        image.convert("RGB").save(output_path)
        return output_path

    # Composite layer for transparent mask fills
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for idx, det in enumerate(detections):
        color = color_for(idx)

        # --- Mask fill ---
        mask_file = seg_dir / det["mask_file"]
        if mask_file.exists():
            mask_np   = np.load(mask_file).astype(bool)
            mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            mask_rgba[mask_np] = color
            mask_img  = Image.fromarray(mask_rgba, mode="RGBA")
            overlay   = Image.alpha_composite(overlay, mask_img)
            draw      = ImageDraw.Draw(overlay)

        # --- Bounding box ---
        x1, y1, x2, y2 = det["box"]
        box_color = color[:3] + (255,)   # fully opaque outline
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # --- Label ---
        label = det["label"]
        score = det["iou_score"]
        text  = f"{label} {score:.2f}"
        draw.rectangle([x1, y1 - 14, x1 + len(text) * 7, y1], fill=box_color)
        draw.text((x1 + 2, y1 - 13), text, fill=(0, 0, 0, 255))

    # Merge overlay onto original
    result = Image.alpha_composite(image, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


def visualize_image(
    image_path: str | Path,
    data_dir:   str | Path = "../data",
    split:      str = "test",
    output_dir: str | Path | None = None,
) -> Path:
    """Visualize a single image by path."""
    image_path = Path(image_path)
    data_dir   = Path(data_dir)
    seg_dir    = data_dir / "segmentation" / split
    json_path  = seg_dir / f"{image_path.stem}.json"

    if output_dir is None:
        output_dir = data_dir / "visualizations" / split
    output_path = Path(output_dir) / f"{image_path.stem}_viz.png"

    return overlay_masks(image_path, json_path, seg_dir, output_path)


def visualize_split(
    data_dir:   str | Path = "../data",
    split:      str = "test",
    output_dir: str | Path | None = None,
    max_images: int | None = None,
) -> None:
    """Visualize all segmented images in a split."""
    data_dir  = Path(data_dir)
    seg_dir   = data_dir / "segmentation" / split
    images_dir = data_dir / "images" / split

    if output_dir is None:
        output_dir = data_dir / "visualizations" / split

    json_files = sorted(seg_dir.glob("*.json"))
    if max_images:
        json_files = json_files[:max_images]

    for json_path in json_files:
        # Find original image (try jpg, jpeg, png)
        for ext in [".jpg", ".jpeg", ".png"]:
            image_path = images_dir / (json_path.stem + ext)
            if image_path.exists():
                break
        else:
            print(f"Original image not found for {json_path.stem}, skipping.")
            continue

        output_path = Path(output_dir) / f"{json_path.stem}_viz.png"
        overlay_masks(image_path, json_path, seg_dir, output_path)

    print(f"Done. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize SAM2 segmentation results")
    parser.add_argument("--image",      default=None,    help="Path to a single image")
    parser.add_argument("--split",      default="test",  choices=["train", "val", "test"])
    parser.add_argument("--data-dir",   default="../data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", default=None,    type=int)
    args = parser.parse_args()

    if args.image:
        visualize_image(
            image_path = args.image,
            data_dir   = args.data_dir,
            split      = args.split,
            output_dir = args.output_dir,
        )
    else:
        visualize_split(
            data_dir   = args.data_dir,
            split      = args.split,
            output_dir = args.output_dir,
            max_images = args.max_images,
        )
