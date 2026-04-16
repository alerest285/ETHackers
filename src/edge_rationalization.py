"""
Edge-Clip Rationalization

Solves the bbox-size ambiguity at image boundaries:

  A small bounding box can mean either:
    (A) the object is far away, OR
    (B) the object is so close it is cut off by the image edge

  Case B breaks the depth_area heuristic (small bbox → assumed far).
  This module uses a vision LLM to disambiguate.

How it works
------------
1. Identify detections whose bounding boxes touch or nearly touch the image edge.
2. Draw numbered annotations on a copy of the image highlighting those detections.
3. Send the annotated image + a structured prompt to GPT-4o-mini.
4. Parse the JSON response to determine whether each edge detection is clipped.
5. If clipped → attach a `depth_area_override` (low value = close) to the detection
   dict, overriding the naive area-based depth signal in fused_depth.py.

Non-edge detections are returned unchanged.
Falls back gracefully (returns original detections) if the LLM call fails.
"""

import base64
import io
import json

import cv2
import numpy as np
from PIL import Image

# Fraction of image dimension within which a bbox edge is considered "at the boundary"
EDGE_MARGIN_FRAC = 0.03   # 3 % of image width/height

# depth_area_override values for each proximity outcome
_OVERRIDE = {"CLOSE": 0.05, "MEDIUM": 0.28}   # FAR → no override (keep original)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _touched_edges(box: list, image_w: int, image_h: int) -> list[str]:
    """Return which image edges the bbox touches (within margin). Empty = not clipped."""
    x1, y1, x2, y2 = box
    mx = image_w * EDGE_MARGIN_FRAC
    my = image_h * EDGE_MARGIN_FRAC
    edges = []
    if x1 <= mx:              edges.append("left")
    if y1 <= my:              edges.append("top")
    if x2 >= image_w - mx:   edges.append("right")
    if y2 >= image_h - my:   edges.append("bottom")
    return edges


# ── Image encoding ─────────────────────────────────────────────────────────────

def _annotate_image(image: Image.Image, edge_dets: list[dict]) -> Image.Image:
    """
    Return a copy of the image with numbered yellow boxes on edge-clipped detections.
    The numbers match the 1-based index sent in the LLM prompt.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, det in enumerate(edge_dets, start=1):
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(img_bgr, str(i), (max(x1 + 4, 4), max(y1 + 22, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 220), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _encode_pil(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Prompt ─────────────────────────────────────────────────────────────────────

def _build_prompt(edge_dets: list[dict], touched: list[list[str]], W: int, H: int) -> str:
    lines = []
    for i, (det, edges) in enumerate(zip(edge_dets, touched), start=1):
        lines.append(
            f"  [{i}] {det['label']} "
            f"(bbox=[{','.join(str(int(v)) for v in det['box'])}], "
            f"image={W}x{H}, touching: {', '.join(edges)})"
        )

    return f"""You are a depth-perception assistant for an industrial robot.

The image shows detections highlighted with numbered yellow boxes.
These detections have bounding boxes that touch the image boundary.

Detections:
{chr(10).join(lines)}

Your task: for each detection, decide whether the small or partial bounding box is because:
  (A) the object is CUT OFF by the image frame — it extends beyond the edge,
      meaning it is VERY CLOSE to the camera (closer than the bbox size implies)
  (B) the object is simply positioned near the border but fully visible and far away

Clues for CUT OFF: the object appears large relative to the image; body parts or
vehicle parts are visibly truncated; the visible portion fills its region densely.
Clues for NOT CUT OFF: the object is small and complete; there is open space around it.

Respond with a JSON object in this exact format:
{{
  "detections": [
    {{
      "id": 1,
      "label": "...",
      "is_clipped": true,
      "estimated_proximity": "CLOSE",
      "confidence": 0.85,
      "reasoning": "one sentence"
    }}
  ]
}}

Return ONLY the JSON object, no other text."""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(client, annotated_image: Image.Image, prompt: str) -> list[dict]:
    """Send the annotated image + prompt to Claude Haiku, return parsed detections list."""
    img_data = _encode_pil(annotated_image)
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
        max_tokens=600,
    )
    raw = response.content[0].text.strip()
    parsed = json.loads(raw)
    # Unwrap either {"detections": [...]} or a bare list
    if isinstance(parsed, dict):
        for key in ("detections", "results", "objects"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Fallback: first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    if isinstance(parsed, list):
        return parsed
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def rationalize_edge_detections(
    image:      Image.Image,
    detections: list[dict],
    client,                       # OpenAI client from llm_module.llm.get_client()
) -> list[dict]:
    """
    Inspect edge-clipped detections with a vision LLM and attach
    `depth_area_override` to those whose small bbox is due to proximity,
    not distance.

    Fields added to clipped detections:
        depth_area_override   float        replacement depth_area value [0,1]
        edge_clip_label       str          CLOSE | MEDIUM | NOT_CLIPPED
        edge_clip_confidence  float        LLM confidence
        edge_clip_reasoning   str          one-sentence explanation

    Parameters
    ----------
    image       : Original PIL image (RGB).
    detections  : Detection dicts with 'box' [x1,y1,x2,y2].
    client      : Initialised OpenAI client.

    Returns
    -------
    Same list (shallow copies) with edge-clip fields attached where relevant.
    Returns original list unchanged on any error.
    """
    W, H = image.size

    # Find edge-clipped detections
    touched_per_det = [_touched_edges(det["box"], W, H) for det in detections]
    edge_indices    = [i for i, edges in enumerate(touched_per_det) if edges]

    if not edge_indices:
        return detections

    edge_dets   = [detections[i] for i in edge_indices]
    edge_touches = [touched_per_det[i] for i in edge_indices]

    print(f"  [edge_rationalization] {len(edge_indices)} edge-clipped detection(s) → querying LLM ...")

    try:
        annotated = _annotate_image(image, edge_dets)
        prompt    = _build_prompt(edge_dets, edge_touches, W, H)
        results   = _call_llm(client, annotated, prompt)
    except Exception as exc:
        print(f"  [edge_rationalization] LLM call failed ({exc}), skipping overrides.")
        return detections

    # Apply results back
    enriched = [dict(d) for d in detections]
    for j, res in enumerate(results):
        if j >= len(edge_indices):
            break
        idx  = edge_indices[j]
        prox = res.get("estimated_proximity", "FAR")
        conf = float(res.get("confidence", 0.5))

        if res.get("is_clipped") and prox in _OVERRIDE:
            enriched[idx]["depth_area_override"]  = _OVERRIDE[prox]
            enriched[idx]["edge_clip_label"]       = prox
            enriched[idx]["edge_clip_confidence"]  = conf
            enriched[idx]["edge_clip_reasoning"]   = res.get("reasoning", "")
            print(f"    [{j+1}] {enriched[idx]['label']} → CLIPPED {prox} "
                  f"(conf={conf:.2f}) override={_OVERRIDE[prox]:.2f}")
        else:
            enriched[idx]["edge_clip_label"] = "NOT_CLIPPED"

    return enriched
