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
You are a computer vision expert and prompt engineer specialising in open-vocabulary \
object detection with Grounding DINO.

Your task: given an industrial or warehouse scene image, produce a precise, \
exhaustive list of all physically present objects that a ground robot must be \
aware of for safe navigation.

Rules:
- Output ONLY a dot-separated list of short noun phrases, e.g.:
      person . forklift . safety cone . cardboard box . hard hat .
- Each phrase must be 1–4 words, concrete, and visually unambiguous.
- Use the most specific term Grounding DINO would recognise \
  (e.g. "hard hat" not "protective equipment", "pallet jack" not "machine").
- Cover every risk-relevant object: humans, vehicles, obstacles, \
  safety markers, and spatial features (doorway, narrow corridor, wet floor sign).
- Do NOT include abstract concepts, verbs, or adjectives alone.
- Do NOT add explanations, numbering, or any text other than the phrase list.
- End the list with a final dot."""

USER_PROMPT = """\
Examine this industrial scene carefully. \
List every object relevant to robot navigation safety as a Grounding DINO prompt."""


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
