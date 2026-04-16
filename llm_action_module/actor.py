"""
LLM Actor — asks the LLM directly for action probabilities over STOP / SLOW / CONTINUE.

Inputs sent to the API
-----------------------
  - SAFETY_RULES.md          injected as system context
  - Scene graph text          structured description of detected objects + spatial relations
  - Depth map stats           min/max/mean/std from the numpy array (0=close, 1=far)
  - Point cloud stats         bounding box of the 3D point cloud
  - Segmented image (PNG)     SAM2 overlay showing detected objects
  - Depth heatmap (PNG)       colourised depth map

Output
------
  {
    "action":        "STOP" | "SLOW" | "CONTINUE",
    "confidence":    float,                          # probability of the chosen action
    "reasoning":     str,                            # ≤20 words, names specific hazard
    "probabilities": {"STOP": p, "SLOW": p, "CONTINUE": p},  # must sum to 1
    "entropy":       float,                          # Shannon entropy (bits); 0=certain, log2(3)≈1.585=uniform
  }

Entropy tracking
----------------
  Every call appends {"stem", "entropy", "action", "probabilities"} to an
  uncertainty buffer JSON file.  High-entropy frames are the best candidates
  for human labelling (active learning).

Public API
----------
  from llm_action_module.actor import LLMActor

  actor = LLMActor()
  result = actor.query(
      scene_graph_text = scene_text,
      seg_img_path     = Path("data/pipeline_output/overlays/frame_001_overlay.png"),
      depth_img_path   = Path("data/pipeline_output/depth_maps/frame_001_depth.png"),
      depth_arr        = corrected_depth,       # (H, W) float32 numpy array
      point_cloud      = point_cloud_arr,       # (N, 3) float32 numpy array, or None
      stem             = "frame_001",           # used for the entropy buffer
  )
"""

from __future__ import annotations

import base64
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from llm_module.llm import get_client, MODEL

# ── constants ─────────────────────────────────────────────────────────────────
SAFETY_RULES_PATH  = ROOT / "action_module" / "SAFETY_RULES.md"
ENTROPY_BUFFER     = ROOT / "data" / "pipeline_output" / "llm_actor_entropy.json"
CLASSES            = ["STOP", "SLOW", "CONTINUE"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _entropy(probs: dict[str, float]) -> float:
    """Shannon entropy in bits.  0 = certain; log2(3) ≈ 1.585 = uniform."""
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log2(p)
    return round(h, 6)


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    """Force probabilities to sum to 1; fill missing classes with 0."""
    vals = {c: max(0.0, float(raw.get(c, 0.0))) for c in CLASSES}
    total = sum(vals.values())
    if total <= 0:
        return {c: round(1 / 3, 6) for c in CLASSES}
    return {c: round(vals[c] / total, 6) for c in CLASSES}


# ══════════════════════════════════════════════════════════════════════════════
# LLMActor
# ══════════════════════════════════════════════════════════════════════════════

class LLMActor:
    """
    Sends scene data to GPT-4o-mini and receives action probabilities.

    Parameters
    ----------
    model       : OpenAI model name (default: same as llm_module.llm.MODEL)
    temperature : sampling temperature (low = more deterministic)
    """

    def __init__(self, model: str = MODEL, temperature: float = 0.1):
        self.client       = get_client()
        self.model        = model
        self.temperature  = temperature
        self.safety_rules = SAFETY_RULES_PATH.read_text()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _img_part(path: Optional[Path]) -> Optional[dict]:
        """Encode an image file as an OpenAI vision content part."""
        if path is None or not Path(path).exists():
            return None
        p    = Path(path)
        ext  = p.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        data = base64.b64encode(p.read_bytes()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}

    @staticmethod
    def _depth_stats(depth_arr: Optional[np.ndarray]) -> str:
        if depth_arr is None:
            return ""
        d = depth_arr
        return (
            f"## Depth Map Statistics (0 = closest, 1 = farthest)\n"
            f"min={d.min():.3f}  max={d.max():.3f}  "
            f"mean={d.mean():.3f}  std={d.std():.3f}"
        )

    @staticmethod
    def _pc_stats(point_cloud: Optional[np.ndarray]) -> str:
        if point_cloud is None or len(point_cloud) == 0:
            return ""
        xyz = point_cloud[:, :3]
        return (
            f"## Point Cloud ({len(point_cloud):,} points)\n"
            f"x = [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
            f"y = [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
            f"z = [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]"
        )

    def _build_context(
        self,
        scene_graph_text: str,
        depth_arr:        Optional[np.ndarray],
        point_cloud:      Optional[np.ndarray],
    ) -> str:
        parts = [f"## Scene Graph\n{scene_graph_text}"]
        depth_s = self._depth_stats(depth_arr)
        pc_s    = self._pc_stats(point_cloud)
        if depth_s:
            parts.append(depth_s)
        if pc_s:
            parts.append(pc_s)
        return "\n\n".join(parts)

    @staticmethod
    def _append_entropy_buffer(entry: dict) -> None:
        """Append one record to the per-run entropy buffer (JSON-lines style)."""
        ENTROPY_BUFFER.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if ENTROPY_BUFFER.exists():
            try:
                existing = json.loads(ENTROPY_BUFFER.read_text())
            except Exception:
                existing = []
        existing.append(entry)
        ENTROPY_BUFFER.write_text(json.dumps(existing, indent=2))

    # ── public API ────────────────────────────────────────────────────────────

    def query(
        self,
        scene_graph_text: str,
        seg_img_path:     Optional[Path] = None,
        depth_img_path:   Optional[Path] = None,
        depth_arr:        Optional[np.ndarray] = None,
        point_cloud:      Optional[np.ndarray] = None,
        stem:             str = "",
    ) -> dict:
        """
        Query the LLM for action probabilities.

        Parameters
        ----------
        scene_graph_text : structured scene description
        seg_img_path     : path to SAM2 overlay PNG
        depth_img_path   : path to depth heatmap PNG
        depth_arr        : (H, W) float32 depth array for textual stats
        point_cloud      : (N, 3) float32 point cloud for textual stats
        stem             : image stem used for entropy buffer logging

        Returns
        -------
        {action, confidence, reasoning, probabilities, entropy}
        """
        system = (
            "You are the action-decision system of an autonomous industrial robot.\n"
            "Apply the following safety rules strictly to decide the robot's next action.\n\n"
            "## Safety Rules\n\n"
            + self.safety_rules
            + "\n\n"
            "You will receive:\n"
            "  • One or two images (segmentation overlay, depth heatmap)\n"
            "  • A structured scene graph describing detected objects\n"
            "  • Depth map statistics\n"
            "  • Point cloud bounding-box statistics\n\n"
            "Respond ONLY with a single JSON object — no markdown, no extra text:\n"
            '{"probabilities": {"STOP": <0–1>, "SLOW": <0–1>, "CONTINUE": <0–1>}, '
            '"action": "STOP"|"SLOW"|"CONTINUE", '
            '"confidence": <float 0–1>, '
            '"reasoning": "<one sentence, ≤20 words, name the specific hazard>"}\n\n'
            "Rules:\n"
            "  - The three probabilities MUST sum to 1.0.\n"
            "  - 'action' MUST equal the class with the highest probability.\n"
            "  - 'confidence' MUST equal the probability of the chosen action.\n"
            "  - Reasoning must be concrete: name the object, its zone, the reason."
        )

        context = self._build_context(scene_graph_text, depth_arr, point_cloud)

        user_parts: list[dict] = []
        img_labels: list[str] = []
        for path, label in [
            (seg_img_path,   "segmentation overlay (objects coloured by risk group)"),
            (depth_img_path, "depth heatmap (bright = close, dark = far)"),
        ]:
            part = self._img_part(path)
            if part:
                user_parts.append(part)
                img_labels.append(label)

        img_note = (
            "  ".join(f"Image {i+1}: {l}." for i, l in enumerate(img_labels))
            if img_labels else "No images provided."
        )

        user_parts.append({
            "type": "text",
            "text": f"{img_note}\n\n{context}\n\nApply the safety rules and return your JSON decision.",
        })

        # ── API call ──────────────────────────────────────────────────────────
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_parts},
                ],
                max_tokens=256,
                temperature=self.temperature,
            )
            raw = resp.choices[0].message.content.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                parsed = json.loads(m.group()) if m else {}

            probs      = _normalise(parsed.get("probabilities", {}))
            action     = parsed.get("action", "").upper()
            if action not in CLASSES:
                action = max(probs, key=probs.get)
            confidence = round(float(parsed.get("confidence", probs[action])), 4)
            reasoning  = str(parsed.get("reasoning", "")).strip()

        except Exception as e:
            # Uniform fallback — maximum entropy
            probs      = {c: round(1 / 3, 6) for c in CLASSES}
            action     = "CONTINUE"
            confidence = round(1 / 3, 4)
            reasoning  = f"LLM unavailable: {e}"

        ent = _entropy(probs)

        # ── entropy buffer ────────────────────────────────────────────────────
        if stem:
            try:
                self._append_entropy_buffer({
                    "stem":          stem,
                    "entropy":       ent,
                    "action":        action,
                    "probabilities": probs,
                })
            except Exception:
                pass   # never crash the pipeline over logging

        return {
            "action":        action,
            "confidence":    confidence,
            "reasoning":     reasoning,
            "probabilities": probs,
            "entropy":       ent,
        }
