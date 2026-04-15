"""
LLM-guided object prompt generation for Grounding DINO.

Sends the image to GPT-4o-mini with a PhD-level prompt engineering request.
Returns a dot-separated string of precise object phrases optimised for
Grounding DINO's open-vocabulary detector.

Usage:
    from segment_module.llm_objects import load_client, get_detection_prompt

    client = load_client()
    prompt = get_detection_prompt(client, image)
    # → "person . forklift . safety cone . cardboard box . hard hat ."
"""

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────

API_KEY_ENV = "OPENAI_API_KEY"
API_KEY     = "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"
MODEL       = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a computer vision expert specialising in open-vocabulary object detection \
with Grounding DINO for an industrial ground robot.

Your task: given a scene image, list every object the robot must be aware of \
for safe navigation as a dot-separated phrase list.

Output rules:
- ONLY a dot-separated noun phrase list, e.g.: person . forklift . cone . floor .
- Each phrase 1–4 words, concrete, visually unambiguous.
- No explanations, numbering, or extra text. End with a final dot.

Use EXACTLY these canonical terms (preferred over synonyms):

  Navigable surfaces — always include when visible (robot drives ON them, not INTO them):
    "floor"          any indoor floor, concrete or tiled surface
    "ground"         outdoor ground, gravel, dirt
    "road"           road / asphalt / tarmac
    "parking lot"    parking area or parking space
    "pavement"       paved outdoor surface
    "floor marking"  painted lines, lane markings, floor stripes

  People (all map to the same high-risk class — use the single best term):
    "person"         any visible human — worker, operator, pedestrian
    "hard hat"       any head protection worn by a person
    "safety vest"    hi-vis or reflective vest/jacket
    "head"           when only a head is visible, not the full body

  Vehicles:
    "forklift"       all forklifts, reach trucks, pallet jacks
    "truck"          lorries, delivery trucks, heavy vehicles
    "car"            passenger cars, vans
    "cart"           motorized or electric carts

  Safety markers:
    "cone"           traffic cones, delineators
    "barrier"        jersey barriers, safety barriers
    "caution tape"   warning / hazard tape
    "wet floor sign" wet-floor or caution signs

  Obstacles:
    "barrel"         drums, chemical barrels
    "box"            cardboard boxes, cartons, crates
    "pallet"         wooden or plastic pallets
    "rack"           storage racks and shelving

Cover every risk-relevant object. Prefer canonical terms above over synonyms."""

USER_PROMPT = """\
Examine this industrial scene carefully. \
List every object relevant to robot navigation safety as a Grounding DINO prompt."""


# ── classifier: maps a novel label to one of the 6 canonical groups ───────────

CLASSIFY_SYSTEM = """\
You classify detection labels for an industrial ground robot into risk groups.

Pick EXACTLY one group name from this list:
  SURFACE        navigable ground — floor, road, parking lot, sidewalk, pavement
  BACKGROUND     irrelevant structures — walls, ceiling, sky, windows, columns
  SAFETY_MARKER  hazard indicators — cones, signs, barriers, caution tape, bollards
  OBSTACLE       static physical obstructions — boxes, pallets, machines, pipes, furniture
  VEHICLE        motorized or large mobile objects — cars, forklifts, trucks, carts, AGVs
  HUMAN          any person, body part, or protective gear worn by a person

Rules:
- Output ONLY the group name in uppercase. No punctuation, no explanation.
- When genuinely uncertain, choose OBSTACLE — that keeps the robot safe.
- Protective equipment (helmet, vest, hard hat) -> HUMAN (indicates a person).
- Painted lines / floor markings -> SURFACE (robot drives on them)."""


def classify_label(client: OpenAI, label: str) -> str:
    """Ask GPT-4o-mini to classify a single label into one of the 6 groups.

    Returns the group name. Falls back to "OBSTACLE" (conservative) on any
    malformed response or API failure.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user",   "content": f"Label: {label!r}"},
            ],
            max_tokens=8,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip().upper()
    except Exception:
        raw = ""

    valid = {"SURFACE", "BACKGROUND", "SAFETY_MARKER", "OBSTACLE", "VEHICLE", "HUMAN"}
    return raw if raw in valid else "OBSTACLE"


def classify_and_learn(client: OpenAI, detections: list[dict]) -> list[dict]:
    """Classify every detection flagged `unmapped`, persist to the ontology,
    and rewrite its risk_group / risk_score in-place.

    Batched per unique label so each new phrase costs at most one LLM call.
    Safe to call on an already-resolved list (no-op if no unmapped entries).
    """
    # Imported here to avoid a circular import at module load time.
    from src.label_ontology import register_learned_alias, get_risk_group

    unique_unmapped: dict[str, str] = {}   # label -> chosen group
    for det in detections:
        if not det.get("unmapped"):
            continue
        lbl = (det.get("label") or "").strip().lower()
        if lbl and lbl not in unique_unmapped:
            unique_unmapped[lbl] = classify_label(client, lbl)

    for lbl, grp in unique_unmapped.items():
        register_learned_alias(lbl, grp)

    # Re-resolve every detection; unmapped flag clears for learned labels.
    for det in detections:
        r = get_risk_group(det.get("label", ""))
        det["risk_group"] = r["group"]
        det["risk_score"] = r["risk_score"]
        if r.get("unmapped"):
            det["unmapped"] = True
        else:
            det.pop("unmapped", None)

    return detections


# ── client ────────────────────────────────────────────────────────────────────

def load_client() -> OpenAI:
    key = API_KEY or os.environ.get(API_KEY_ENV, "")
    if not key:
        raise EnvironmentError(f"Set {API_KEY_ENV} or hardcode API_KEY in llm_objects.py")
    return OpenAI(api_key=key)


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image: Image.Image) -> str:
    """Base64-encode a PIL image as JPEG for the API."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── public API ────────────────────────────────────────────────────────────────

def get_detection_prompt(
    client:    OpenAI,
    image:     Image.Image,
    max_tokens: int = 128,
) -> str:
    """
    Ask GPT-4o-mini to generate an optimised Grounding DINO prompt for the image.

    Returns:
        A dot-separated phrase string ready to pass to grounding_dino.detect().
        e.g. "person . forklift . safety cone . cardboard box ."
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image)}"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Normalise: ensure it ends with a dot and has no stray whitespace
    if not raw.endswith("."):
        raw += " ."
    return raw
