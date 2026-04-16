"""
Neural Predictor — scene embedding → action probabilities, trained via LLM soft labels.

Architecture
------------
  NeuralPredictor : MLP (sigmoid hidden activations) that maps a scene embedding
                    (embed_dim,) → (3,) soft probabilities over [STOP, SLOW, CONTINUE].

Training loop
-------------
  1. For every sample in pipeline_output, load (or compute) its scene embedding.
  2. Call LLMTeacher: give GPT-4o-mini the scene graph, depth map path, point cloud
     summary, bbox image, segmented image, and SAFETY_RULES.  It returns a soft
     distribution over the three classes.
  3. Compute cross-entropy( predictor_output , llm_distribution ) and backpropagate
     through the neural predictor only.
  4. Record the LLM distribution entropy per sample to an UncertaintyBuffer.

After training
--------------
  Active learning: the 500 highest-entropy LLM predictions are surfaced for human
  labelling via active_learning.py.  Human labels are then used to:
    (a) fine-tune the neural predictor with hard cross-entropy targets,
    (b) call suggest_rule_updates() which asks the LLM to propose edits to
        action_module/SAFETY_RULES.md based on the disagreement patterns.

Public API
----------
  from action_module.neural_predictor import NeuralPredictor, LLMTeacher, train, fine_tune_from_labels

  predictor = NeuralPredictor(embed_dim=256)
  teacher   = LLMTeacher()
  train(predictor, teacher, pipeline_dir="data/pipeline_output", epochs=50)
  predictor.save("data/neural_predictor.pt")

  # After active labelling:
  fine_tune_from_labels(predictor, labels_path="action_module/labels.json",
                        emb_dir="data/pipeline_output/embeddings")

CLI
---
  python action_module/neural_predictor.py --train
  python action_module/neural_predictor.py --fine-tune
  python action_module/neural_predictor.py --suggest-rules
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
sys.path.insert(0, str(_ROOT))

SAFETY_RULES_PATH = _HERE / "SAFETY_RULES.md"
LABELS_PATH       = _HERE / "labels.json"

CLASSES     = ["STOP", "SLOW", "CONTINUE"]
_CLS_IDX    = {c: i for i, c in enumerate(CLASSES)}

# Smoothing applied to the LLM one-hot to form a soft label.
# The predicted class gets (1 - SMOOTH), rest is split over the others.
_LABEL_SMOOTH = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# Neural Predictor
# ══════════════════════════════════════════════════════════════════════════════

class NeuralPredictor(nn.Module):
    """
    MLP scene classifier.

    Input  : (embed_dim,)   — scene embedding from MultimodalFusion
    Output : (3,)           — softmax probabilities over [STOP, SLOW, CONTINUE]

    Hidden activations are sigmoid as specified; output uses softmax so the
    three values form a proper probability simplex.
    """

    def __init__(self, embed_dim: int = 256, hidden_dims: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        dims = [embed_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(dims[-1], len(CLASSES)))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (embed_dim,) or (B, embed_dim) → (3,) or (B, 3) probabilities"""
        logits = self.net(emb)
        return F.softmax(logits, dim=-1)

    def predict(self, emb: torch.Tensor) -> dict:
        """Convenience wrapper returning a result dict compatible with the pipeline."""
        self.eval()
        with torch.no_grad():
            probs = self.forward(emb.unsqueeze(0) if emb.dim() == 1 else emb)
            probs = probs.squeeze(0)
        probs_np = probs.cpu().numpy()
        idx      = int(probs_np.argmax())
        return {
            "action":       CLASSES[idx],
            "confidence":   float(probs_np[idx]),
            "probabilities": {c: float(probs_np[i]) for i, c in enumerate(CLASSES)},
            "source":       "neural_predictor",
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  self.state_dict(),
            "embed_dim":   self.net[0].in_features,
            "hidden_dims": tuple(
                layer.out_features for layer in self.net
                if isinstance(layer, nn.Linear)
            )[:-1],
        }, path)
        print(f"NeuralPredictor saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "NeuralPredictor":
        dev = torch.device(device or "cpu")
        ckpt = torch.load(path, map_location=dev)
        model = cls(embed_dim=ckpt["embed_dim"], hidden_dims=ckpt["hidden_dims"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev)
        return model


# ══════════════════════════════════════════════════════════════════════════════
# LLM Teacher
# ══════════════════════════════════════════════════════════════════════════════

class LLMTeacher:
    """
    Wraps GPT-4o-mini to produce soft probability labels for a scene.

    The LLM receives:
      - The segmented image (overlay PNG)
      - The bbox image
      - A text summary of the scene graph + depth + point cloud
      - The full SAFETY_RULES.md injected as a system prompt

    It responds with JSON: {"action": ..., "confidence": ..., "reasoning": ...}

    The confidence is used to construct a smoothed soft distribution:
      predicted class → confidence × (1 - smooth)   +  smooth / 3
      other classes   → (1 - confidence × (1-smooth)) / 2   + smooth / 3
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        from llm_module.llm import get_client
        self.client      = get_client()
        self.model       = model
        self.temperature = temperature
        self.safety_rules = SAFETY_RULES_PATH.read_text()

    def _build_context_text(
        self,
        scene_graph_text: str,
        pc_path:          Optional[Path],
        depth_path:       Optional[Path],
    ) -> str:
        parts = ["Scene graph:\n" + scene_graph_text]

        if depth_path and depth_path.exists():
            try:
                d = np.load(depth_path)
                parts.append(
                    f"Depth map stats: min={d.min():.3f} max={d.max():.3f} "
                    f"mean={d.mean():.3f} std={d.std():.3f} (0=close, 1=far)"
                )
            except Exception:
                pass

        if pc_path and pc_path.exists():
            try:
                pc = np.load(pc_path)
                n_pts = pc.shape[0]
                xyz   = pc[:, :3]
                parts.append(
                    f"Point cloud: {n_pts} points, "
                    f"x=[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] "
                    f"y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] "
                    f"z=[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]"
                )
            except Exception:
                pass

        return "\n\n".join(parts)

    def query(
        self,
        stem:             str,
        scene_graph_text: str,
        seg_img_path:     Optional[Path] = None,
        bbox_img_path:    Optional[Path] = None,
        pc_path:          Optional[Path] = None,
        depth_path:       Optional[Path] = None,
    ) -> dict:
        """
        Query the LLM teacher and return:
          {
            "action": str,
            "confidence": float,
            "reasoning": str,
            "soft_label": [p_stop, p_slow, p_continue],   # smoothed distribution
            "entropy": float,
          }
        Returns a rule-fallback distribution on failure.
        """
        import base64

        def _img_msg(path: Path) -> dict | None:
            if path is None or not path.exists():
                return None
            ext  = path.suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            data = base64.b64encode(path.read_bytes()).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }

        system = (
            "You are an autonomous industrial robot's perception system. "
            "Apply the following safety rules strictly to decide the robot action.\n\n"
            "## Safety Rules\n\n"
            + self.safety_rules
            + "\n\nRespond ONLY with a JSON object — no markdown, no extra text:\n"
            '{"action": "STOP"|"SLOW"|"CONTINUE", "confidence": <float 0-1>, '
            '"reasoning": "<one sentence>"}'
        )

        context = self._build_context_text(scene_graph_text, pc_path, depth_path)
        user_parts: list[dict] = []
        for p in [seg_img_path, bbox_img_path]:
            msg = _img_msg(p) if p else None
            if msg:
                user_parts.append(msg)
        user_parts.append({"type": "text", "text": context
                           + "\n\nApply the safety rules and return your JSON decision."})

        try:
            from openai import OpenAI
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
                m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                parsed = json.loads(m.group()) if m else {}

            action = parsed.get("action", "").upper()
            if action not in _CLS_IDX:
                raise ValueError(f"bad action: {action}")
            conf = float(parsed.get("confidence", 0.5))
        except Exception as e:
            # fallback: uniform distribution
            return {
                "action": "CONTINUE", "confidence": 1/3, "reasoning": f"fallback: {e}",
                "soft_label": [1/3, 1/3, 1/3], "entropy": math.log2(3),
            }

        soft = _soft_label_from_action(action, conf)
        return {
            "action":     action,
            "confidence": round(conf, 4),
            "reasoning":  str(parsed.get("reasoning", "")),
            "soft_label": soft,
            "entropy":    _entropy(soft),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _soft_label_from_action(action: str, confidence: float,
                             smooth: float = _LABEL_SMOOTH) -> list[float]:
    """
    Convert a (action, confidence) pair to a smoothed 3-class distribution.
    confidence=0.9, action=STOP → [0.81+ε, ε, ε]  (sum=1, labels smoothed)
    """
    idx  = _CLS_IDX[action]
    base = smooth / len(CLASSES)
    dist = [base] * len(CLASSES)
    dist[idx] += (1.0 - smooth) * confidence
    # distribute remaining mass uniformly to other classes
    remaining = (1.0 - smooth) * (1.0 - confidence)
    others    = len(CLASSES) - 1
    for i in range(len(CLASSES)):
        if i != idx:
            dist[i] += remaining / others
    # normalise to handle floating-point drift
    total = sum(dist)
    return [v / total for v in dist]


def _entropy(probs: list[float]) -> float:
    p = np.array(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ══════════════════════════════════════════════════════════════════════════════
# Embedding I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_or_compute_embedding(
    stem:         str,
    pipeline_dir: Path,
    fusion_model: Optional["MultimodalFusion"] = None,  # type: ignore[name-defined]
    device:       torch.device = torch.device("cpu"),
) -> Optional[torch.Tensor]:
    """
    Load a pre-saved embedding npy if it exists, otherwise compute it via
    MultimodalFusion (requires fusion_model to be provided).

    Embeddings are saved to <pipeline_dir>/embeddings/<stem>.npy.
    """
    emb_path = pipeline_dir / "embeddings" / f"{stem}.npy"
    if emb_path.exists():
        arr = np.load(emb_path).astype(np.float32)
        return torch.from_numpy(arr).to(device)

    if fusion_model is None:
        return None

    # Compute from raw artefacts
    try:
        from action_module.multimodal_fusion import (
            pil_to_tensor, depth_array_to_tensor, point_cloud_to_tensor,
            detections_to_graph_data,
        )
        from PIL import Image

        def _try_img(path: Path) -> Optional[torch.Tensor]:
            if path.exists():
                try:
                    return pil_to_tensor(Image.open(path).convert("RGB"))
                except Exception:
                    pass
            return None

        seg_img  = _try_img(pipeline_dir / "overlays"    / f"{stem}_overlay.png")
        bbox_img = _try_img(pipeline_dir / "bbox_images" / f"{stem}.png")

        depth_t = None
        dp = pipeline_dir / "depth_maps" / f"{stem}.npy"
        if dp.exists():
            depth_t = depth_array_to_tensor(np.load(dp).astype(np.float32))

        pc_t = None
        pp = pipeline_dir / "point_clouds" / f"{stem}.npy"
        if pp.exists():
            pc_t = point_cloud_to_tensor(np.load(pp).astype(np.float32))

        graph_data = None
        det_p  = pipeline_dir / "detections"   / f"{stem}.json"
        sg_p   = pipeline_dir / "scene_graphs" / f"{stem}.json"
        if det_p.exists() and sg_p.exists():
            with open(det_p) as f:
                dets = json.load(f)
            with open(sg_p) as f:
                sg = json.load(f)
            graph_data = detections_to_graph_data(dets, sg)

        boxes = []
        if det_p.exists():
            with open(det_p) as f:
                dets = json.load(f)
            boxes = [d["box"] for d in dets if "box" in d]

        fusion_model.eval()
        with torch.no_grad():
            emb = fusion_model(
                graph_data  = graph_data,
                seg_image   = seg_img.to(device)   if seg_img  is not None else None,
                bbox_image  = bbox_img.to(device)  if bbox_img is not None else None,
                bbox_boxes  = boxes,
                depth_map   = depth_t.to(device)   if depth_t  is not None else None,
                point_cloud = pc_t.to(device)      if pc_t     is not None else None,
            )

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb.cpu().numpy())
        return emb

    except Exception as e:
        tqdm.write(f"  [emb] {stem}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Training — LLM soft-label distillation
# ══════════════════════════════════════════════════════════════════════════════

def train(
    predictor:    NeuralPredictor,
    teacher:      LLMTeacher,
    pipeline_dir: str | Path,
    epochs:       int   = 50,
    lr:           float = 3e-4,
    weight_decay: float = 0.01,
    save_path:    str | Path = "data/neural_predictor.pt",
    buffer_path:  str | Path = "data/pipeline_output/uncertainty_buffer.json",
    fusion_model: Optional[object] = None,
    device:       str | None = None,
) -> None:
    """
    Train the NeuralPredictor using LLM soft labels (knowledge distillation).

    For each sample:
      1. Load/compute scene embedding.
      2. Ask LLMTeacher for a soft label distribution.
      3. Cross-entropy loss between predictor output and LLM distribution.
      4. Record LLM entropy to UncertaintyBuffer for active learning.

    The teacher API is called once per sample per epoch (or once total and
    cached when use_cache=True — default True to limit API costs).
    """
    from action_module.active_learning import UncertaintyBuffer

    dev = torch.device(
        device or
        ("cuda" if torch.cuda.is_available() else
         ("mps"  if torch.backends.mps.is_available() else "cpu"))
    )
    predictor.to(dev)
    pipeline_dir = Path(pipeline_dir)

    stems = sorted(p.stem for p in (pipeline_dir / "detections").glob("*.json"))
    if not stems:
        print("No samples found in pipeline_output/detections. Run pipeline.py first.")
        return

    print(f"Training NeuralPredictor on {len(stems)} samples, {epochs} epochs.")

    # ── load or build LLM soft label cache ───────────────────────────────────
    label_cache_path = pipeline_dir / "llm_soft_labels.json"
    label_cache: dict[str, dict] = {}
    if label_cache_path.exists():
        with open(label_cache_path) as f:
            label_cache = json.load(f)
        print(f"  Loaded {len(label_cache)} cached LLM labels.")

    buffer = UncertaintyBuffer()

    # Fill cache for stems that haven't been queried yet
    uncached = [s for s in stems if s not in label_cache]
    if uncached:
        print(f"  Querying LLM teacher for {len(uncached)} new samples ...")
        for stem in tqdm(uncached, desc="LLM teacher"):
            sg_path  = pipeline_dir / "scene_graphs" / f"{stem}.txt"
            sg_text  = sg_path.read_text() if sg_path.exists() else "No scene graph."
            result   = teacher.query(
                stem             = stem,
                scene_graph_text = sg_text,
                seg_img_path     = pipeline_dir / "overlays"     / f"{stem}_overlay.png",
                bbox_img_path    = pipeline_dir / "bbox_images"  / f"{stem}.png",
                pc_path          = pipeline_dir / "point_clouds" / f"{stem}.npy",
                depth_path       = pipeline_dir / "depth_maps"   / f"{stem}.npy",
            )
            label_cache[stem] = result

        label_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_cache_path, "w") as f:
            json.dump(label_cache, f, indent=2)
        print(f"  LLM label cache saved → {label_cache_path}")

    # Record all LLM entropies to uncertainty buffer now
    for stem, result in label_cache.items():
        img_candidates = [
            pipeline_dir / "overlays" / f"{stem}_overlay.png",
        ]
        img_path = next((p for p in img_candidates if p.exists()), Path(stem))
        buffer.record(
            stem      = stem,
            img_path  = img_path,
            clf_result = {
                "action":        result["action"],
                "confidence":    result["confidence"],
                "probabilities": {c: result["soft_label"][i]
                                  for i, c in enumerate(CLASSES)},
                "source":        "llm_teacher",
            },
        )

    buffer.save(Path(buffer_path))

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=lr, weight_decay=weight_decay
    )
    n_steps   = epochs * len(stems)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # ── training loop ──────────────────────────────────────────────────────────
    predictor.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count      = 0
        random_stems = stems[:]  # could shuffle each epoch
        import random as _random
        _random.shuffle(random_stems)

        for stem in random_stems:
            if stem not in label_cache:
                continue
            emb = load_or_compute_embedding(
                stem, pipeline_dir, fusion_model, dev)
            if emb is None:
                continue

            soft = label_cache[stem]["soft_label"]
            target = torch.tensor(soft, dtype=torch.float32, device=dev)

            probs = predictor(emb.to(dev))                    # (3,)
            # cross-entropy between two distributions = -sum(target * log(pred))
            loss  = -(target * torch.log(probs.clamp(min=1e-8))).sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            count      += 1

        avg = total_loss / max(count, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:3d}/{epochs}]  loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  samples={count}")

    predictor.save(save_path)
    print(f"\nTraining complete. Model saved → {save_path}")
    print(f"Uncertainty buffer (LLM entropy) saved → {buffer_path}")
    print(f"  Run active learning to label top-500 most uncertain predictions:")
    print(f"  python action_module/active_learning.py "
          f"--buffer {buffer_path} --top-k 500 --mode neural")


# ══════════════════════════════════════════════════════════════════════════════
# Fine-tuning on human labels
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_from_labels(
    predictor:    NeuralPredictor,
    labels_path:  str | Path = LABELS_PATH,
    emb_dir:      str | Path = "data/pipeline_output/embeddings",
    pipeline_dir: str | Path = "data/pipeline_output",
    epochs:       int   = 30,
    lr:           float = 1e-4,
    save_path:    str | Path = "data/neural_predictor.pt",
    device:       str | None = None,
) -> None:
    """
    Fine-tune the neural predictor on hard human labels from labels.json.
    Uses standard one-hot cross-entropy (hard targets, not soft LLM labels).
    """
    labels_path  = Path(labels_path)
    emb_dir      = Path(emb_dir)
    pipeline_dir = Path(pipeline_dir)

    if not labels_path.exists():
        print("No labels.json found. Run active learning labelling first.")
        return

    with open(labels_path) as f:
        raw_labels = json.load(f)

    labelled = [(e["stem"], e["label"]) for e in raw_labels
                if e.get("label") in _CLS_IDX]
    if not labelled:
        print("No valid labels in labels.json (need STOP / SLOW / CONTINUE).")
        return

    dev = torch.device(
        device or
        ("cuda" if torch.cuda.is_available() else
         ("mps"  if torch.backends.mps.is_available() else "cpu"))
    )
    predictor.to(dev).train()

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)
    print(f"Fine-tuning on {len(labelled)} human labels for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count      = 0
        for stem, label in labelled:
            emb = load_or_compute_embedding(stem, pipeline_dir, None, dev)
            if emb is None:
                # Try loading from emb_dir directly
                p = emb_dir / f"{stem}.npy"
                if p.exists():
                    emb = torch.from_numpy(
                        np.load(p).astype(np.float32)).to(dev)
                else:
                    continue

            target = torch.tensor(_CLS_IDX[label], dtype=torch.long, device=dev)
            probs  = predictor(emb.to(dev))
            # hard cross-entropy via log-loss on one-hot target
            loss   = F.cross_entropy(probs.unsqueeze(0).log(),
                                     target.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count      += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [FT Epoch {epoch:3d}/{epochs}]  loss={total_loss/max(count,1):.4f}")

    predictor.save(save_path)
    print(f"Fine-tuning complete. Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Rule suggestion from disagreement patterns
# ══════════════════════════════════════════════════════════════════════════════

def suggest_rule_updates(
    labels_path:       str | Path = LABELS_PATH,
    llm_cache_path:    str | Path = "data/pipeline_output/llm_soft_labels.json",
    safety_rules_path: str | Path = SAFETY_RULES_PATH,
    output_path:       str | Path = _HERE / "SAFETY_RULES_SUGGESTIONS.md",
) -> str:
    """
    Analyse disagreements between human labels and LLM predictions, then
    ask GPT-4o-mini to propose concrete amendments to SAFETY_RULES.md.

    Returns the suggestion text (also written to output_path).
    """
    labels_path    = Path(labels_path)
    llm_cache_path = Path(llm_cache_path)

    if not labels_path.exists():
        return "No labels.json found."
    with open(labels_path) as f:
        human_labels = {e["stem"]: e["label"] for e in json.load(f)
                        if e.get("label") in _CLS_IDX}

    llm_preds: dict[str, str] = {}
    if llm_cache_path.exists():
        with open(llm_cache_path) as f:
            cache = json.load(f)
        llm_preds = {stem: v["action"] for stem, v in cache.items()}

    disagreements: list[dict] = []
    for stem, human in human_labels.items():
        llm = llm_preds.get(stem)
        if llm and llm != human:
            disagreements.append({
                "stem":        stem,
                "human_label": human,
                "llm_label":   llm,
            })

    if not disagreements:
        msg = "No disagreements between human labels and LLM predictions."
        print(msg)
        return msg

    print(f"Found {len(disagreements)} disagreements between human labels and LLM.")

    summary_lines = [
        f"- stem={d['stem']}: human={d['human_label']}, LLM={d['llm_label']}"
        for d in disagreements[:50]   # cap at 50 to keep prompt manageable
    ]
    summary = "\n".join(summary_lines)

    safety_rules = Path(safety_rules_path).read_text()

    prompt = f"""
You are a safety rules editor for an autonomous robot system.

Below are the current safety rules:

{safety_rules}

A human annotator disagreed with the LLM's action prediction on {len(disagreements)} images.
Here are up to 50 disagreements (human label vs LLM label):

{summary}

Based on these disagreements, propose specific, minimal amendments to the safety rules above.
Each amendment should:
  1. Identify which rule(s) caused the disagreement.
  2. Propose a concrete change (new rule, modified threshold, or clarification).
  3. Justify the change in one sentence.

Format each amendment as:
## Amendment N
**Rule**: <rule ID or NEW>
**Change**: <exact text change or new rule>
**Rationale**: <one sentence>
"""

    try:
        from llm_module.llm import get_client
        client = get_client()
        from openai import OpenAI  # noqa: F401
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                    "You are a precise safety-rules editor. "
                    "Propose minimal, justified amendments only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        suggestions = resp.choices[0].message.content.strip()
    except Exception as e:
        suggestions = f"LLM call failed: {e}\n\nRaw disagreements:\n{summary}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# Safety Rules Suggestions\n\n"
        f"Generated from {len(disagreements)} human-vs-LLM disagreements "
        f"out of {len(human_labels)} total human labels.\n\n"
        + suggestions
    )
    print(f"Suggestions written → {output_path}")
    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuralPredictor training / fine-tuning")
    parser.add_argument("--train",         action="store_true",
                        help="Train predictor via LLM soft labels")
    parser.add_argument("--fine-tune",     action="store_true",
                        help="Fine-tune on human labels from labels.json")
    parser.add_argument("--suggest-rules", action="store_true",
                        help="Generate SAFETY_RULES amendment suggestions")
    parser.add_argument("--pipeline-dir",  type=Path,
                        default=Path("data/pipeline_output"))
    parser.add_argument("--save-path",     type=Path,
                        default=Path("data/neural_predictor.pt"))
    parser.add_argument("--buffer-path",   type=Path,
                        default=Path("data/pipeline_output/uncertainty_buffer.json"))
    parser.add_argument("--labels-path",   type=Path, default=LABELS_PATH)
    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--embed-dim",     type=int, default=256)
    parser.add_argument("--device",        type=str, default=None)
    args = parser.parse_args()

    if args.train:
        predictor = NeuralPredictor(embed_dim=args.embed_dim)
        teacher   = LLMTeacher()
        train(
            predictor    = predictor,
            teacher      = teacher,
            pipeline_dir = args.pipeline_dir,
            epochs       = args.epochs,
            save_path    = args.save_path,
            buffer_path  = args.buffer_path,
            device       = args.device,
        )

    elif args.fine_tune:
        if not args.save_path.exists():
            print(f"No checkpoint at {args.save_path}. Run --train first.")
            sys.exit(1)
        predictor = NeuralPredictor.load(args.save_path, device=args.device)
        fine_tune_from_labels(
            predictor    = predictor,
            labels_path  = args.labels_path,
            pipeline_dir = args.pipeline_dir,
            epochs       = 30,
            save_path    = args.save_path,
            device       = args.device,
        )

    elif args.suggest_rules:
        suggest_rule_updates(labels_path=args.labels_path)

    else:
        parser.print_help()
