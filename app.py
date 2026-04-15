"""
app.py — FastAPI server for the ETHackers pipeline web UI.

Run:
    python app.py

Then open http://localhost:7860
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Depth pipeline (lazy-loaded on first request) ─────────────────────────────
_depth_pipe = None

def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import pipeline as hf_pipeline
        print("Loading Depth Anything V2 …")
        _depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        print("Model ready.")
    return _depth_pipe


COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma":   cv2.COLORMAP_MAGMA,
    "plasma":  cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet":     cv2.COLORMAP_JET,
}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/run")
async def run_pipeline(
    image:    UploadFile = File(...),
    stages:   str        = Form(...),   # comma-separated, e.g. "depth"
    colormap: str        = Form("inferno"),
):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    selected = [s.strip() for s in stages.split(",")]
    result   = {}

    # ── Depth ──────────────────────────────────────────────────────────────────
    if "depth" in selected:
        pipe  = get_depth_pipe()
        depth = np.array(pipe(pil_image)["depth"])   # HxW float

        # Normalize → uint8
        d_min, d_max = depth.min(), depth.max()
        depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Colormap
        cmap    = COLORMAPS.get(colormap, cv2.COLORMAP_INFERNO)
        heatmap = cv2.applyColorMap(depth_u8, cmap)

        # Resize original to match heatmap dims
        orig_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        orig_bgr = cv2.resize(orig_bgr, (heatmap.shape[1], heatmap.shape[0]))

        result["depth"] = {
            "original": _to_b64(orig_bgr),
            "heatmap":  _to_b64(heatmap),
        }

    return JSONResponse(result)


def _to_b64(bgr: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
