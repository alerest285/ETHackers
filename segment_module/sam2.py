"""
SAM2 — precise instance segmentation from Grounding DINO bounding boxes.

Takes the output of grounding_dino.detect() (bounding boxes + labels) and
produces per-object binary masks at full image resolution.

SAM2 is used in box-prompted mode: each Grounding DINO box is passed as a
spatial prompt, and SAM2 returns the best mask within that box.

Usage:
    from segment_module.grounding_dino import load_model as load_gdino, detect
    from segment_module.sam2           import load_model as load_sam2, segment

    gdino      = load_gdino()
    sam        = load_sam2()
    detections = detect(gdino, image, prompt)
    results    = segment(sam, image, detections)

Each result dict extends the Grounding DINO detection with:
    {
        ...all grounding_dino fields...,
        "mask":       np.ndarray bool (H, W),  — True = object pixel
        "mask_score": float,                    — SAM2 confidence for this mask
    }
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "facebook/sam2-hiera-base-plus"


# ── model handle ──────────────────────────────────────────────────────────────

class SAM2Model:
    def __init__(self, processor, model, device: str):
        self.processor = processor
        self.model     = model
        self.device    = device


# ── public API ────────────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID) -> SAM2Model:
    """Load SAM2. Weights (~380 MB) auto-download from HuggingFace on first run."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading SAM2 on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return SAM2Model(processor, model, device)


def segment(
    model:      SAM2Model,
    image:      Image.Image,
    detections: list[dict],
) -> list[dict]:
    """
    Run SAM2 on all bounding boxes from Grounding DINO.

    Args:
        model:      Loaded SAM2Model.
        image:      Same PIL image passed to Grounding DINO.
        detections: Output of grounding_dino.detect().

    Returns:
        A copy of each detection dict extended with "mask" (bool H×W array)
        and "mask_score" (float). Detections with no box are passed through
        unchanged (no mask added).
    """
    if not detections:
        return []

    W, H = image.size
    boxes = [[d["box"] for d in detections]]   # shape: (1, N, 4) — one image

    inputs = model.processor(
        images=image,
        input_boxes=boxes,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.model(**inputs)

    # Upsample masks manually — avoids processor API version differences.
    # outputs.pred_masks: (1, N, num_candidates, H_low, W_low)
    # We upsample each to (H, W) using bilinear interpolation.
    import torch.nn.functional as F

    pred = outputs.pred_masks.squeeze(0).float().contiguous()  # (N, C, H_low, W_low)
    N, C, h_low, w_low = pred.shape
    # reshape to (N*C, 1, H_low, W_low), upsample, reshape back
    upsampled = F.interpolate(
        pred.reshape(N * C, 1, h_low, w_low),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).reshape(N, C, H, W)  # (N, num_candidates, H, W)

    # Convert to list of tensors, one per box (each tensor: (num_candidates, H, W))
    masks_per_box = [upsampled[i] for i in range(N)]

    # iou_scores: guard against missing field
    iou_scores = outputs.iou_scores[0] if hasattr(outputs, "iou_scores") else torch.ones(N, C)

    results = []
    for i, det in enumerate(detections):
        result = dict(det)

        if i < len(masks_per_box):
            candidates   = masks_per_box[i]          # (num_candidates, H, W)
            scores       = iou_scores[i]              # (num_candidates,)
            best_idx     = int(scores.argmax())
            best_mask    = (candidates[best_idx].cpu().numpy() > 0.0).astype(bool)
            best_score   = round(float(scores[best_idx]), 4)

            result["mask"]       = best_mask
            result["mask_score"] = best_score
        else:
            result["mask"]       = np.zeros((H, W), dtype=bool)
            result["mask_score"] = 0.0

        results.append(result)

    return results


def masks_to_overlay(
    image:   Image.Image,
    results: list[dict],
    alpha:   float = 0.45,
) -> Image.Image:
    """
    Render coloured semi-transparent masks over the original image.

    Useful for visual inspection. Returns a PIL RGBA image.
    """
    COLORS = {
        "HUMAN":         (220, 50,  50),
        "VEHICLE":       (50,  50,  220),
        "OBSTACLE":      (220, 140, 50),
        "SAFETY_MARKER": (50,  200, 50),
        "BACKGROUND":    (160, 160, 160),
    }

    overlay = image.convert("RGBA").copy()
    W, H    = image.size

    for res in results:
        mask  = res.get("mask")
        if mask is None or not mask.any():
            continue
        color = COLORS.get(res.get("risk_group", "BACKGROUND"), (160, 160, 160))
        layer = np.zeros((H, W, 4), dtype=np.uint8)
        layer[mask] = (*color, int(alpha * 255))
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer, "RGBA"))

    return overlay
