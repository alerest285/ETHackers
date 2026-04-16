"""
Decision Module — final action arbitration with LLM rationale.

Takes three signals:
  1. NeuralPredictor distribution   {STOP: p, SLOW: p, CONTINUE: p}
  2. Safety-LLM result              output of analyse_with_scene_graph() from
                                    llm_module/llm.py — GPT-4o-mini prompted
                                    with SAFETY_RULES.md + scene graph
                                    {action, confidence, reasoning}
  3. Visual context                 bbox image path + depth image path

Calls GPT-4o-mini (same API key as llm_module/llm.py) and asks it to weigh
both signals against the images and return a single decided action + one crisp
line of reasoning (≤ 20 words, naming the specific hazard).

Output
------
  {
    "action":       "STOP" | "SLOW" | "CONTINUE",
    "confidence":   float,
    "reasoning":    str,          # ≤ 20 words, sharp and specific
    "nn_action":    str,          # what the neural predictor said
    "safety_action": str,         # what the safety-rules LLM said
    "agreement":    bool,         # did they agree?
  }

Public API
----------
  from decision_module.decider import Decider
  from llm_module.llm import get_client, analyse_with_scene_graph

  safety_rules = Path("action_module/SAFETY_RULES.md").read_text()
  client       = get_client()
  safety_result = analyse_with_scene_graph(client, img_path, scene_graph_text, safety_rules)

  decider = Decider()
  result  = decider.decide(
      nn_probs      = predictor.predict(embedding)["probabilities"],
      safety_result = safety_result,
      bbox_img_path  = Path("data/pipeline_output/bbox_images/frame_001.png"),
      depth_img_path = Path("data/pipeline_output/depth_maps/frame_001.png"),
  )
  # result["reasoning"] → "Worker stepping into lane at 1.5 m; emergency stop."
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_module.llm import get_client, MODEL

# ── constants ─────────────────────────────────────────────────────────────────
CLASSES      = ["STOP", "SLOW", "CONTINUE"]
_CONSERV_ORD = {"STOP": 0, "SLOW": 1, "CONTINUE": 2}   # lower = more conservative


class Decider:
    """
    Final arbitration layer.

    Receives the NeuralPredictor distribution and the safety-LLM result
    (analyse_with_scene_graph output), then makes one more GPT-4o-mini call
    that sees both signals plus the bbox and depth images and produces the
    final action + a single sharp line of reasoning.
    """

    def __init__(self, model: str = MODEL, temperature: float = 0.2):
        self.client      = get_client()
        self.model       = model
        self.temperature = temperature

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _img_part(path: Optional[Path]) -> Optional[dict]:
        if path is None or not Path(path).exists():
            return None
        p    = Path(path)
        ext  = p.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        data = base64.b64encode(p.read_bytes()).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"}}

    @staticmethod
    def _nn_bar(probs: dict[str, float]) -> str:
        lines = []
        for cls in CLASSES:
            p   = probs.get(cls, 0.0)
            bar = "█" * round(p * 20) + "░" * (20 - round(p * 20))
            lines.append(f"  {cls:10s} {bar} {p:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _embed_summary(embedding) -> str:
        """Summarise a (D,) scene embedding as human-readable stats for the LLM."""
        try:
            arr  = np.asarray(embedding, dtype=float).ravel()
            norm = float(np.linalg.norm(arr))
            mean = float(arr.mean())
            std  = float(arr.std())
            top5 = np.argsort(np.abs(arr))[-5:][::-1]
            top5_str = ", ".join(f"dim{i}={arr[i]:+.3f}" for i in top5)
            return (
                f"L2-norm={norm:.3f}, mean={mean:.4f}, std={std:.4f}; "
                f"top-5 activated dims: [{top5_str}]"
            )
        except Exception:
            return "embedding unavailable"

    @staticmethod
    def _conservative_merge(nn_action: str, safety_action: str) -> str:
        """Return the more conservative of the two actions."""
        return nn_action if _CONSERV_ORD[nn_action] <= _CONSERV_ORD[safety_action] \
               else safety_action

    # ── core ──────────────────────────────────────────────────────────────────

    def decide(
        self,
        nn_probs:       dict[str, float],
        safety_result:  dict,
        bbox_img_path:  Optional[Path] = None,
        depth_img_path: Optional[Path] = None,
        embedding=None,
    ) -> dict:
        """
        Parameters
        ----------
        nn_probs       : {"STOP": float, "SLOW": float, "CONTINUE": float}
                         from NeuralPredictor.predict()["probabilities"]
        safety_result  : {"action": str, "confidence": float, "reasoning": str}
                         from llm_module.llm.analyse_with_scene_graph()
        bbox_img_path  : path to bounding-box annotated image (RGB PNG/JPG)
        depth_img_path : path to depth heatmap image (PNG)
        embedding      : (D,) array-like — MultimodalFusion scene embedding
                         (graph + visual + depth + point-cloud fused, default D=256)

        Returns
        -------
        {action, confidence, reasoning, nn_action, safety_action, agreement}
        """
        nn_action     = max(nn_probs, key=nn_probs.get)
        safety_action = safety_result.get("action", "CONTINUE").upper()
        agreement     = (nn_action == safety_action)

        user_parts: list[dict] = []

        # Images first so the model grounds its reasoning visually
        img_labels: list[str] = []
        for path, label in (
            (bbox_img_path,  "bounding-box detections (boxes colored by proximity)"),
            (depth_img_path, "depth heatmap (bright = close, dark = far)"),
        ):
            part = self._img_part(path)
            if part:
                user_parts.append(part)
                img_labels.append(label)

        img_note = ("  ".join(f"Image {i+1}: {l}." for i, l in enumerate(img_labels))
                    if img_labels else "No images available.")

        embed_line = (
            f"Scene embedding stats: {self._embed_summary(embedding)}"
            if embedding is not None
            else "Scene embedding: not provided."
        )

        text_block = (
            f"{img_note}\n\n"
            f"Neural predictor distribution:\n"
            f"{self._nn_bar(nn_probs)}\n"
            f"  → neural verdict: {nn_action}\n\n"
            f"Safety-rules LLM verdict: {safety_action} "
            f"(confidence {safety_result.get('confidence', 0.0):.2f})\n"
            f"Safety-rules reasoning: {safety_result.get('reasoning', 'N/A')}\n\n"
            f"{embed_line}\n"
            f"The two signals {'AGREE' if agreement else 'DISAGREE'}.\n\n"
            "Decide the final robot action. "
            "Give a single sentence (≤ 20 words) naming the specific hazard, "
            "its position, and why that action was chosen.\n\n"
            "Respond ONLY with JSON — no markdown:\n"
            '{"action": "STOP"|"SLOW"|"CONTINUE", '
            '"confidence": <float 0-1>, '
            '"reasoning": "<≤20 words>"}'
        )

        user_parts.append({"type": "text", "text": text_block})

        system = (
            "You are the final arbitration layer of an autonomous industrial robot. "
            "You receive two independent action predictions — one from a neural network "
            "trained via LLM distillation, one from a safety-rules LLM that applied "
            "explicit heuristics to the scene graph. "
            "When the two signals disagree, choose the MORE CONSERVATIVE action "
            "(STOP > SLOW > CONTINUE) unless the visual evidence clearly overrides it. "
            "Your reasoning must be concrete: name the object, its zone, and why "
            "that action was chosen. No generic phrases. Maximum 20 words."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_parts},
                ],
                max_tokens=128,
                temperature=self.temperature,
            )
            raw = resp.choices[0].message.content.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                parsed = json.loads(m.group()) if m else {}

            action = parsed.get("action", "").upper()
            if action not in _CONSERV_ORD:
                raise ValueError(f"invalid action: {action!r}")

            return {
                "action":        action,
                "confidence":    round(float(parsed.get("confidence", 0.5)), 4),
                "reasoning":     str(parsed.get("reasoning", "")).strip(),
                "nn_action":     nn_action,
                "safety_action": safety_action,
                "agreement":     agreement,
            }

        except Exception as e:
            # Hard fallback — never crash the pipeline
            fallback = self._conservative_merge(nn_action, safety_action)
            return {
                "action":        fallback,
                "confidence":    round(min(
                    nn_probs.get(fallback, 0.5),
                    safety_result.get("confidence", 0.5),
                ), 4),
                "reasoning":     safety_result.get(
                    "reasoning", "Conservative fallback: signals merged."),
                "nn_action":     nn_action,
                "safety_action": safety_action,
                "agreement":     agreement,
                "_fallback":     str(e),
            }
