"""
LLM Depth Rationalization - 4th Fusion Signal

Sends the full annotated image to GPT-4o-mini and asks it to estimate
proximity (CLOSE / MEDIUM / FAR) for every detected object.

This is a holistic reasoning pass: the LLM can exploit perspective cues,
occlusion, shadows, known object sizes, and scene context - things that
DepthAnything V2, real-world size references, and bbox area cannot capture.

Output: a numeric `depth_llm` signal per detection in [0, 1] (0 = closest),
incorporated into the fused depth formula as a 4th weighted signal.
"""

import base64
import io
import json

import cv2
import numpy as np
from PIL import Image

# Numeric depth value for each LLM proximity label (0=close, 1=far)
_PROX_TO_DEPTH: dict[str, float] = {
    "CLOSE":  0.18,   # centre of CLOSE  band [0.00, 0.35]
    "MEDIUM": 0.50,   # centre of MEDIUM band [0.35, 0.65]
    "FAR":    0.78,   # centre of FAR    band [0.65, 1.00]
}
_DEFAULT_DEPTH = 0.50   # neutral fallback when LLM omits a detection


# ── Image helpers ──────────────────────────────────────────────────────────────

def _annotate_all(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Return a copy of the image with every detection numbered in white."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(bgr, str(i), (max(x1 + 4, 4), max(y1 + 22, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _encode_pil(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Prompt ─────────────────────────────────────────────────────────────────────

def _build_prompt(detections: list[dict], image_w: int, image_h: int) -> str:
    image_area = image_w * image_h
    lines = []
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        rel_area = det.get(
            "relative_area",
            ((x2 - x1) * (y2 - y1)) / image_area,
        )
        lines.append(
            f"  [{i}] {det['label']} "
            f"(risk: {det.get('risk_group', '?')}, "
            f"bbox covers {rel_area * 100:.1f}% of image)"
        )

    return f"""You are a depth-perception module for an industrial robot's camera.

The image has numbered white bounding boxes around every detected object:
{chr(10).join(lines)}

CRITICAL RULE - bbox coverage is your most reliable depth cue:
  > 25% of image  = almost certainly CLOSE  (object is right in front of camera)
  10-25%          = likely CLOSE to MEDIUM
  3-10%           = likely MEDIUM
  < 3%            = likely MEDIUM to FAR

Additional depth cues to use:
  • Perspective and vanishing-point geometry
  • Occlusion - objects in front partially cover objects behind
  • Sharpness vs. blur (close objects are sharper)
  • Shadows and ground-contact points
  • Scene layout and typical industrial environment structure

Do NOT reason about whether the object "looks normal-sized". A person filling
half the frame is CLOSE regardless of whether they look like a typical human.

Return a JSON object in this exact format:
{{
  "detections": [
    {{
      "id": 1,
      "label": "...",
      "proximity": "CLOSE" | "MEDIUM" | "FAR",
      "confidence": 0.0-1.0,
      "reasoning": "one concise sentence"
    }}
  ]
}}

Return ONLY the JSON object, no other text."""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> list[dict]:
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    for key in ("detections", "results", "objects"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm_depth_signals(
    image:      Image.Image,
    detections: list[dict],
    client,                    # OpenAI client
) -> list[dict]:
    """
    Ask the LLM to estimate proximity for every detection in the image.

    Adds to each detection dict:
        depth_llm         float   depth signal in [0,1], 0=closest
        depth_llm_label   str     CLOSE | MEDIUM | FAR
        depth_llm_conf    float   LLM confidence [0,1]
        depth_llm_reason  str     one-sentence explanation

    Falls back to depth_llm=0.50 (neutral) per detection on any error,
    so the rest of the fusion pipeline is unaffected.

    Parameters
    ----------
    image       : Original PIL image (RGB).
    detections  : Detection dicts with at least 'box' and 'label'.
    client      : Initialised OpenAI client.
    """
    if not detections:
        return detections

    print(f"  [llm_depth] Rationalizing {len(detections)} detection(s) ...")

    try:
        W, H      = image.size
        annotated = _annotate_all(image, detections)
        img_data  = _encode_pil(annotated)
        prompt    = _build_prompt(detections, W, H)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img_data},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1800,
        )

        results  = _parse_response(response.content[0].text.strip())
        enriched = [dict(d) for d in detections]

        for res in results:
            idx = int(res.get("id", 0)) - 1
            if not (0 <= idx < len(enriched)):
                continue
            prox  = res.get("proximity", "MEDIUM")
            conf  = float(res.get("confidence", 0.5))
            depth = _PROX_TO_DEPTH.get(prox, _DEFAULT_DEPTH)
            enriched[idx]["depth_llm"]        = depth
            enriched[idx]["depth_llm_label"]  = prox
            enriched[idx]["depth_llm_conf"]   = conf
            enriched[idx]["depth_llm_reason"] = res.get("reasoning", "")
            print(f"    [{idx+1}] {enriched[idx]['label']:<12} = {prox:<6} "
                  f"(conf={conf:.2f})  depth_llm={depth:.2f}  {res.get('reasoning','')[:60]}")

        # Fill any detections the LLM skipped with a neutral value
        for d in enriched:
            d.setdefault("depth_llm",       _DEFAULT_DEPTH)
            d.setdefault("depth_llm_label", "MEDIUM")
            d.setdefault("depth_llm_conf",  0.0)
            d.setdefault("depth_llm_reason", "")

        return enriched

    except Exception as exc:
        print(f"  [llm_depth] Failed ({exc}), using neutral fallback.")
        enriched = [dict(d) for d in detections]
        for d in enriched:
            d.setdefault("depth_llm",       _DEFAULT_DEPTH)
            d.setdefault("depth_llm_label", "MEDIUM")
            d.setdefault("depth_llm_conf",  0.0)
            d.setdefault("depth_llm_reason", "")
        return enriched
