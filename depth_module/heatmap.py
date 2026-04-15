"""
heatmap.py — run Depth Anything V2 on an image and save a colorized heatmap.

Uses the HuggingFace transformers pipeline (auto-downloads weights).

Usage:
    python depth_module/heatmap.py depth_module/strawberry.jpg
    python depth_module/heatmap.py path/to/any/image.jpg --colormap magma
"""

import argparse
import os
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma":   cv2.COLORMAP_MAGMA,
    "plasma":  cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet":     cv2.COLORMAP_JET,
}


def run(image_path: str, colormap: str = "inferno") -> str:
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Loading Depth Anything V2 (downloading weights if needed) ...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",  # Small = fast, good quality
    )

    print("Running depth inference ...")
    result = pipe(image)
    depth = np.array(result["depth"])  # PIL Image → numpy, values are relative depth

    # Normalize to uint8 for colormap application
    d_min, d_max = depth.min(), depth.max()
    depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    # Apply colormap
    cmap_code = COLORMAPS.get(colormap, cv2.COLORMAP_INFERNO)
    heatmap = cv2.applyColorMap(depth_u8, cmap_code)

    # Side-by-side: original | heatmap
    original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_bgr = cv2.resize(original_bgr, (depth_u8.shape[1], depth_u8.shape[0]))
    combined = np.hstack([original_bgr, heatmap])

    # Save
    stem = os.path.splitext(image_path)[0]
    out_path = f"{stem}_depth_{colormap}.png"
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to input image")
    p.add_argument("--colormap", default="inferno",
                   choices=list(COLORMAPS.keys()),
                   help="Colormap for the heatmap (default: inferno)")
    args = p.parse_args()
    run(args.image, args.colormap)
