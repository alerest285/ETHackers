"""
Visualize YOLOS detection results.

Draws bounding boxes + labels on the original image.

Usage:
    python visualize.py --image ../data/images/test/foo.jpg
    python visualize.py --split test
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

# One color per risk group
GROUP_COLORS: dict[str, tuple] = {
    "person":  (255,  50,  50),   # red
    "vehicle": (255, 160,   0),   # orange
    "bicycle": (255, 220,   0),   # yellow
    "animal":  (100, 200, 100),   # green
    "cone":    (  0, 180, 255),   # blue
    "box":     (180, 100, 255),   # purple
    "other":   (180, 180, 180),   # grey
}


def draw_detections(
    image_path:  Path,
    json_path:   Path,
    output_path: Path,
) -> Path:
    """Draw boxes + labels on the original image and save as PNG."""
    image = Image.open(image_path).convert("RGB")
    draw  = ImageDraw.Draw(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    detections = data.get("detections", [])
    if not detections:
        image.save(output_path)
        print(f"No detections — saved original: {output_path}")
        return output_path

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        group  = det.get("risk_group", "other")
        color  = GROUP_COLORS.get(group, GROUP_COLORS["other"])
        label  = det["label"]
        score  = det["score"]

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label background + text
        text     = f"{label} {score:.2f}"
        text_w   = len(text) * 7
        draw.rectangle([x1, y1 - 15, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - 14), text, fill=(0, 0, 0))

    image.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


def visualize_image(
    image_path: str | Path,
    data_dir:   str | Path = "../data",
    split:      str = "test",
    output_dir: str | Path | None = None,
) -> Path:
    image_path = Path(image_path)
    data_dir   = Path(data_dir)
    det_dir    = data_dir / "detections" / split
    json_path  = det_dir / f"{image_path.stem}.json"

    if output_dir is None:
        output_dir = data_dir / "visualizations" / split
    output_path = Path(output_dir) / f"{image_path.stem}_viz.png"

    return draw_detections(image_path, json_path, output_path)


def visualize_split(
    data_dir:   str | Path = "../data",
    split:      str = "test",
    output_dir: str | Path | None = None,
    max_images: int | None = None,
) -> None:
    data_dir   = Path(data_dir)
    det_dir    = data_dir / "detections" / split
    images_dir = data_dir / "images" / split

    if output_dir is None:
        output_dir = data_dir / "visualizations" / split

    json_files = sorted(det_dir.glob("*.json"))
    if max_images:
        json_files = json_files[:max_images]

    for json_path in json_files:
        for ext in [".jpg", ".jpeg", ".png"]:
            image_path = images_dir / (json_path.stem + ext)
            if image_path.exists():
                break
        else:
            print(f"Image not found for {json_path.stem}, skipping.")
            continue

        output_path = Path(output_dir) / f"{json_path.stem}_viz.png"
        draw_detections(image_path, json_path, output_path)

    print(f"Done. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize YOLOS detections")
    parser.add_argument("--image",      default=None)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir",   default="../data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", default=None, type=int)
    args = parser.parse_args()

    if args.image:
        visualize_image(args.image, args.data_dir, args.split, args.output_dir)
    else:
        visualize_split(args.data_dir, args.split, args.output_dir, args.max_images)
