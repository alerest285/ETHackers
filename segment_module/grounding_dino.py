"""
Grounding DINO — open-vocabulary object detection.

Takes a PIL image and a dot-separated text prompt (from llm_objects.py)
and returns bounding boxes with labels.

The prompt is generated per-image by GPT-4o-mini (llm_objects.py), so
Grounding DINO always searches for exactly what is visible in the scene.

Usage:
    from segment_module.llm_objects   import load_client, get_detection_prompt
    from segment_module.grounding_dino import load_model, detect

    client     = load_client()
    gdino      = load_model()
    prompt     = get_detection_prompt(client, image)
    detections = detect(gdino, image, prompt)

Each detection:
    {
        "label":         str,            — matched phrase from the prompt
        "risk_group":    str,            — HUMAN/VEHICLE/OBSTACLE/SAFETY_MARKER/BACKGROUND
        "box":           [x1,y1,x2,y2], — pixels, top-left origin
        "score":         float,
        "relative_area": float,
    }
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── repo root on path (needed when segment_module is imported directly) ────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.label_ontology import get_risk_group as _get_risk_group  # single source of truth

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "IDEA-Research/grounding-dino-base"


# ── model handle ──────────────────────────────────────────────────────────────

class GroundingDINOModel:
    def __init__(self, processor, model, device: str):
        self.processor  = processor
        self.model      = model
        self.device     = device
        self.ckpt_path  = MODEL_ID


# ── public API ────────────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID) -> GroundingDINOModel:
    """Load Grounding DINO. Weights (~680 MB) auto-download on first run."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Grounding DINO on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return GroundingDINOModel(processor, model, device)


def detect(
    model:           GroundingDINOModel,
    image:           Image.Image,
    text_prompt:     str,
    score_threshold: float = 0.30,
) -> list[dict]:
    """
    Run Grounding DINO on a single image with the given text prompt.

    Args:
        model:           Loaded GroundingDINOModel.
        image:           PIL image.
        text_prompt:     Dot-separated phrases, e.g. "person . forklift . cone ."
        score_threshold: Minimum detection confidence.

    Returns:
        List of detection dicts sorted by descending score.
    """
    W, H = image.size

    inputs = model.processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.model(**inputs)

    results = model.processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=score_threshold,
        text_threshold=score_threshold * 0.8,
        target_sizes=[(H, W)],
    )[0]

    detections = []
    # Use text_labels (string names) — "labels" returns integer IDs in transformers ≥ 4.51
    label_key = "text_labels" if "text_labels" in results else "labels"
    for box, score, label in zip(results["boxes"], results["scores"], results[label_key]):
        x1, y1, x2, y2 = [round(float(v), 2) for v in box]
        score_f  = round(float(score), 4)
        label_s  = label.strip().lower()
        group    = _get_risk_group(label_s)["group"]
        rel_area = round((x2 - x1) * (y2 - y1) / (W * H), 6)

        detections.append({
            "label":         label_s,
            "risk_group":    group,
            "box":           [x1, y1, x2, y2],
            "score":         score_f,
            "relative_area": rel_area,
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections
