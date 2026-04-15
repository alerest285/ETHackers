"""
LLM Module — GPT-4o scene understanding via OpenAI API.

No local weights — uses the OpenAI cloud API.

Set your API key in one of two ways:
  1. Environment variable (recommended):
       export OPENAI_API_KEY=sk-...
  2. Hardcode in this file (see API_KEY below) — fine for local hackathon use.

Images passed to the model:
  1. Original image  — clean scene
  2. Overlay image   — YOLO boxes + depth heatmap side-by-side

Output: SCENE / RISKS / DECISION saved as .txt per image.

Usage:
  python llm_module/llm.py --run-all
  python llm_module/llm.py --overlay data/pipeline_output/overlays/foo_overlay.png \\
                           --original data/challenge/data/images/train/foo.jpg \\
                           --json     data/pipeline_output/detections/foo.json
"""

import argparse
import base64
import json
import os
from pathlib import Path

from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────

MODEL       = "gpt-4o-mini"
BASE_URL    = None   # use default OpenAI endpoint
API_KEY_ENV = "OPENAI_API_KEY"

# Paste your key here or set the environment variable above
API_KEY     = "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"   # e.g. "sk-..."

OUT_ROOT    = Path("data/pipeline_output/llm")
OVERLAY_DIR = Path("data/pipeline_output/overlays")
DET_DIR     = Path("data/pipeline_output/detections")
TRAIN_DIR   = Path("data/challenge/data/images/train")

# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an autonomous industrial robot's perception system. "
    "You receive visual data and structured sensor outputs and must reason "
    "about scene safety to help the robot decide whether to STOP, SLOW DOWN, "
    "or CONTINUE moving."
)


def build_user_prompt(detections: list[dict]) -> str:
    if not detections:
        obj_summary = "No objects were detected."
    else:
        lines = []
        for d in detections:
            lines.append(
                f"  - {d['label']} (risk={d.get('risk_group','?')}, "
                f"confidence={d.get('score',0):.2f}, "
                f"proximity={d.get('proximity_label','?')}, "
                f"depth_score={d.get('depth_score','?')})"
            )
        obj_summary = "\n".join(lines)

    return f"""You are given two images:
  Image 1: The original scene photo.
  Image 2: A side-by-side analysis — LEFT shows detected objects with bounding boxes \
colored by proximity (red=CLOSE, orange=MEDIUM, green=FAR); \
RIGHT shows a depth heatmap (bright=close, dark=far) with the same boxes.

Structured detection data from the perception pipeline:
{obj_summary}

Please provide:
1. A concise description of what is happening in the scene.
2. Which detected objects pose the most risk to the robot and why.
3. A navigation recommendation: STOP, SLOW, or CONTINUE — with a confidence (0.0–1.0) and a one-sentence justification.

Format your answer as:
SCENE: <description>
RISKS: <risk analysis>
DECISION: <STOP|SLOW|CONTINUE> (<confidence>) — <justification>"""


# ── client ────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = API_KEY or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Either set API_KEY in llm.py or run: export {API_KEY_ENV}=sk-..."
        )
    kwargs = {"api_key": api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def _encode_image(path: Path) -> str:
    """Base64-encode an image file for the API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_message(path: Path) -> dict:
    ext    = path.suffix.lower().lstrip(".")
    mime   = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    data   = _encode_image(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }


# ── inference ─────────────────────────────────────────────────────────────────

def analyse(
    client:         OpenAI,
    original_path:  Path,
    overlay_path:   Path,
    detections:     list[dict],
    max_tokens:     int = 512,
) -> str:
    """Call the Qwen API and return the model's scene analysis."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    _image_message(original_path),
                    _image_message(overlay_path),
                    {"type": "text", "text": build_user_prompt(detections)},
                ],
            },
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── file-level helpers ────────────────────────────────────────────────────────

def process_one(
    client:        OpenAI,
    overlay_path:  Path,
    original_path: Path,
    json_path:     Path,
    output_dir:    Path,
) -> Path:
    """Run analysis for one image and save the result as .txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    detections = data.get("detections", [])

    print(f"  Analysing {original_path.name} ({len(detections)} detections) ...")
    response = analyse(client, original_path, overlay_path, detections)

    stem     = overlay_path.stem.replace("_overlay", "")
    out_path = output_dir / f"{stem}_analysis.txt"
    out_path.write_text(response)
    print(f"  Saved: {out_path}")
    return out_path


def run_all(client: OpenAI, output_dir: Path = OUT_ROOT) -> None:
    """Process every overlay that has a matching detections JSON."""
    overlays = sorted(OVERLAY_DIR.glob("*_overlay.png"))
    if not overlays:
        print(f"No overlays found in {OVERLAY_DIR}. Run src/pipeline.py first.")
        return

    for overlay_path in overlays:
        stem      = overlay_path.stem.replace("_overlay", "")
        json_path = DET_DIR / f"{stem}.json"

        original_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = TRAIN_DIR / (stem + ext)
            if candidate.exists():
                original_path = candidate
                break

        if not json_path.exists():
            print(f"  [SKIP] No detections JSON for {stem}")
            continue
        if original_path is None:
            print(f"  [SKIP] Original image not found for {stem}")
            continue

        process_one(client, overlay_path, original_path, json_path, output_dir)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL scene analysis via API")
    parser.add_argument("--overlay",    default=None)
    parser.add_argument("--original",   default=None)
    parser.add_argument("--json",       default=None)
    parser.add_argument("--run-all",    action="store_true")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    client  = get_client()
    out_dir = Path(args.output_dir)

    if args.run_all:
        run_all(client, out_dir)
    elif args.overlay and args.original and args.json:
        process_one(client, Path(args.overlay), Path(args.original), Path(args.json), out_dir)
    else:
        parser.error("Provide --overlay, --original, --json  OR  use --run-all")
