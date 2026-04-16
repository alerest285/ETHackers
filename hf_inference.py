"""
hf_inference.py — Hugging Face Inference API wrappers for faster inference.

Uses `huggingface_hub.InferenceClient` to run Grounding DINO and Depth
Anything V2 on HF's hosted infrastructure (free serverless + optional
dedicated endpoints), skipping local model downloads and CPU inference.

Enable by setting environment variables:
    HF_TOKEN=hf_...        # required for most inference endpoints
    USE_HF_API=1           # toggle app.py between HF API and local models

Optional:
    HF_GDINO_MODEL=IDEA-Research/grounding-dino-base
    HF_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf

Same public surface as the local modules:
    detect_hf(image, text_prompt, score_threshold) -> list[dict]
    depth_hf(image)                                -> np.ndarray (H, W) float32
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.label_ontology import get_risk_group as _get_risk_group


# ── config ────────────────────────────────────────────────────────────────────

GDINO_MODEL = os.environ.get("HF_GDINO_MODEL", "IDEA-Research/grounding-dino-base")
DEPTH_MODEL = os.environ.get("HF_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
HF_TOKEN    = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


_clients: dict = {}

def _client(model_id: str):
    """Lazy singleton InferenceClient per model."""
    if model_id not in _clients:
        from huggingface_hub import InferenceClient
        _clients[model_id] = InferenceClient(model=model_id, token=HF_TOKEN, timeout=120)
    return _clients[model_id]


def _pil_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    save_img = image if image.mode == "RGB" else image.convert("RGB")
    save_img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


# ── Grounding DINO via HF ─────────────────────────────────────────────────────

def detect_hf(
    image:           Image.Image,
    text_prompt:     str,
    score_threshold: float = 0.30,
) -> list[dict]:
    """
    Open-vocabulary object detection via the HF Inference API.

    Accepts the same dot-separated prompt format used by the local detector:
        "person . forklift . safety cone ."

    Returns detections in the exact same shape as the local module.
    """
    client = _client(GDINO_MODEL)
    W, H   = image.size

    # HF wants a list of candidate labels, not a dot-string
    labels = [p.strip() for p in text_prompt.split(".") if p.strip()]
    if not labels:
        return []

    img_bytes = _pil_bytes(image)

    try:
        raw = client.zero_shot_object_detection(
            image=img_bytes,
            candidate_labels=labels,
        )
    except TypeError:
        # Older huggingface_hub: arg was named `labels`
        raw = client.zero_shot_object_detection(image=img_bytes, labels=labels)

    detections = []
    for det in raw:
        score = float(det.get("score", 0))
        if score < score_threshold:
            continue
        b = det.get("box") or det.get("bbox") or {}
        x1 = float(b.get("xmin", b.get("x1", 0)))
        y1 = float(b.get("ymin", b.get("y1", 0)))
        x2 = float(b.get("xmax", b.get("x2", 0)))
        y2 = float(b.get("ymax", b.get("y2", 0)))

        label_s  = str(det.get("label", "")).strip().lower()
        resolved = _get_risk_group(label_s)
        rel_area = round((x2 - x1) * (y2 - y1) / (W * H), 6)

        d = {
            "label":         label_s,
            "risk_group":    resolved["group"],
            "risk_score":    resolved["risk_score"],
            "box":           [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "score":         round(score, 4),
            "relative_area": rel_area,
        }
        if resolved.get("unmapped"):
            d["unmapped"] = True
        detections.append(d)

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ── DepthAnything via HF ──────────────────────────────────────────────────────

def depth_hf(image: Image.Image) -> np.ndarray:
    """
    Run depth estimation on the HF Inference API.

    Returns a (H, W) float32 array of RAW disparity (same convention as the
    local hf_pipeline output — invert downstream to get 0=close, 1=far).
    """
    client = _client(DEPTH_MODEL)
    img_bytes = _pil_bytes(image)

    # Newer huggingface_hub has a typed helper; older versions need .post().
    try:
        result = client.depth_estimation(image=img_bytes)
        # Typed result is a PIL image (grayscale) OR a dict {"depth": PIL.Image}
        if isinstance(result, dict) and "depth" in result:
            depth_pil = result["depth"]
        else:
            depth_pil = result
        arr = np.array(depth_pil, dtype=np.float32)
    except (AttributeError, TypeError):
        # Fallback: raw post, expecting image bytes back
        resp = client.post(data=img_bytes, task="depth-estimation")
        depth_pil = Image.open(io.BytesIO(resp))
        arr = np.array(depth_pil, dtype=np.float32)

    # HF often returns a single-channel grayscale image; make sure we're 2D
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


# ── warmup (optional, call once at startup) ───────────────────────────────────

def warmup(image_size: tuple[int, int] = (640, 480)) -> None:
    """Send a tiny probe request so the HF endpoints spin up warm for the user."""
    try:
        dummy = Image.new("RGB", image_size, (128, 128, 128))
        detect_hf(dummy, "person .", score_threshold=0.99)
    except Exception as e:
        print(f"[hf_inference] warmup detect skipped: {e}")
    try:
        dummy = Image.new("RGB", image_size, (128, 128, 128))
        depth_hf(dummy)
    except Exception as e:
        print(f"[hf_inference] warmup depth skipped: {e}")
