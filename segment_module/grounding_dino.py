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

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── model ─────────────────────────────────────────────────────────────────────

MODEL_ID = "IDEA-Research/grounding-dino-base"

# ── label → risk group ────────────────────────────────────────────────────────
# Maps common Grounding DINO output phrases to the 5 risk groups.
# The LLM prompt may introduce novel phrases — unknown labels fall back to BACKGROUND.

_KEYWORDS: dict[str, str] = {
    # HUMAN
    "person": "HUMAN", "worker": "HUMAN", "pedestrian": "HUMAN",
    "human": "HUMAN", "head": "HUMAN", "helmet": "HUMAN",
    "hard hat": "HUMAN", "hat": "HUMAN", "operator": "HUMAN",
    # VEHICLE
    "forklift": "VEHICLE", "pallet jack": "VEHICLE", "car": "VEHICLE",
    "truck": "VEHICLE", "bus": "VEHICLE", "motorcycle": "VEHICLE",
    "bicycle": "VEHICLE", "train": "VEHICLE", "vehicle": "VEHICLE",
    "cart": "VEHICLE", "tow truck": "VEHICLE",
    # OBSTACLE
    "barrel": "OBSTACLE", "drum": "OBSTACLE", "crate": "OBSTACLE",
    "box": "OBSTACLE", "cardboard box": "OBSTACLE", "container": "OBSTACLE",
    "suitcase": "OBSTACLE", "handcart": "OBSTACLE", "ladder": "OBSTACLE",
    "chair": "OBSTACLE", "pallet": "OBSTACLE", "shelf": "OBSTACLE",
    "rack": "OBSTACLE", "pipe": "OBSTACLE",
    # SAFETY_MARKER
    "cone": "SAFETY_MARKER", "safety cone": "SAFETY_MARKER",
    "traffic cone": "SAFETY_MARKER", "traffic sign": "SAFETY_MARKER",
    "stop sign": "SAFETY_MARKER", "traffic light": "SAFETY_MARKER",
    "warning sign": "SAFETY_MARKER", "wet floor sign": "SAFETY_MARKER",
    "barrier": "SAFETY_MARKER", "caution tape": "SAFETY_MARKER",
}


def _label_to_group(label: str) -> str:
    label_l = label.strip().lower()
    # exact match first
    if label_l in _KEYWORDS:
        return _KEYWORDS[label_l]
    # keyword scan for partial matches (e.g. "yellow safety cone" → SAFETY_MARKER)
    for kw, group in _KEYWORDS.items():
        if kw in label_l:
            return group
    return "BACKGROUND"


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
        group    = _label_to_group(label_s)
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
