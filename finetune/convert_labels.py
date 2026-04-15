"""
Convert COCO annotations → YOLO format with 5 risk-group classes.

27 raw categories are collapsed to:
  0  HUMAN          (person, head, hat, helmet)
  1  VEHICLE        (forklift, car, truck, bus, motorcycle, bicycle, train)
  2  OBSTACLE       (barrel, crate, box, container, suitcase, handcart, ladder, chair)
  3  SAFETY_MARKER  (cone, traffic sign, stop sign, traffic light)
  4  BACKGROUND     (anything else — skipped, not written to label files)

YOLO label format per line:
  <class_id> <cx> <cy> <w> <h>   (all normalised 0–1)

Usage:
  python finetune/convert_labels.py
"""

import json
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
ANN_DIR     = ROOT / "data" / "annotations"
OUT_ROOT    = ROOT / "data" / "yolo_labels"

SPLITS = {
    "train": ANN_DIR / "train.json",
    "val":   ANN_DIR / "val.json",
}

# ── ontology ──────────────────────────────────────────────────────────────────
# Maps lowercase-stripped raw label → class index (0–4)
# BACKGROUND (4) annotations are skipped entirely — no point training on noise.

LABEL_MAP: dict[str, int] = {
    # HUMAN = 0
    "person":   0, "head": 0, "hat": 0, "helmet": 0,
    # VEHICLE = 1
    "forklift": 1, "car": 1, "truck": 1, "bus": 1,
    "motorcycle": 1, "bicycle": 1, "train": 1,
    # OBSTACLE = 2
    "barrel":   2, "crate": 2, "box": 2, "container": 2,
    "suitcase": 2, "handcart": 2, "ladder": 2, "chair": 2,
    # SAFETY_MARKER = 3
    "cone":         3,
    "traffic sign": 3,
    "stop sign":    3,
    "traffic light":3,
}

CLASS_NAMES = ["HUMAN", "VEHICLE", "OBSTACLE", "SAFETY_MARKER", "BACKGROUND"]


def convert_split(split: str, ann_path: Path) -> None:
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        coco = json.load(f)

    # id → raw name
    cat_name: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}

    # image_id → {id, file_name, width, height}
    img_meta: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # group annotations by image
    ann_by_image: dict[int, list] = {img["id"]: [] for img in coco["images"]}
    skipped_bg = 0
    for ann in coco["annotations"]:
        raw    = cat_name.get(ann["category_id"], "").lower().strip()
        cls_id = LABEL_MAP.get(raw)
        if cls_id is None:
            skipped_bg += 1
            continue
        ann_by_image[ann["image_id"]].append((cls_id, ann["bbox"]))

    written = 0
    for img_id, img in img_meta.items():
        anns = ann_by_image[img_id]
        stem = Path(img["file_name"]).stem
        out_path = out_dir / f"{stem}.txt"

        if not anns:
            # Write empty file so YOLO knows the image has no objects
            out_path.write_text("")
            continue

        W, H = img["width"], img["height"]
        lines = []
        for cls_id, (x, y, w, h) in anns:
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_path.write_text("\n".join(lines))
        written += 1

    print(f"  [{split}] {written:>6} label files written  "
          f"({skipped_bg} BACKGROUND annotations skipped)")


def main() -> None:
    print("Converting COCO annotations → YOLO 5-class format ...\n")
    for split, ann_path in SPLITS.items():
        if not ann_path.exists():
            print(f"  [{split}] annotation file not found: {ann_path} — skipping")
            continue
        convert_split(split, ann_path)

    # Write dataset.yaml next to this script's parent
    yaml_path = ROOT / "finetune" / "dataset.yaml"
    yaml_path.write_text(f"""\
# THEKER challenge — 5 risk-group classes
path: {ROOT}
train: data/challenge/data/images/train
val:   data/challenge/data/images/val

nc: 5
names: {CLASS_NAMES}

# Label directories (auto-detected by Ultralytics when images/ → labels/ sibling swap)
# Ultralytics looks for labels in the path obtained by replacing
# 'images' with 'labels' in the image path.
# We symlink / copy so that structure holds:
#   data/challenge/data/labels/train/<stem>.txt
""")
    print(f"\nWrote {yaml_path}")

    # Create symlink so Ultralytics finds labels alongside images
    for split in ("train", "val"):
        link = ROOT / "data" / "challenge" / "data" / "labels" / split
        target = ROOT / "data" / "yolo_labels" / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve())
        print(f"Symlinked {link} → {target.resolve()}")

    print("\nDone. Run: python finetune/train.py")


if __name__ == "__main__":
    main()
