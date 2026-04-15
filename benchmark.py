"""
Benchmark — full 7-step pipeline on the validation set.

Steps per image:
  1. GPT-4o-mini generates a Grounding DINO object prompt.
  2. Grounding DINO detects objects (open-vocabulary).
  3. SAM2 segments each detected object.
  4. DepthAnything V2 produces a monocular depth map.
     4a. LLM holistic depth signals.      [skipped with --fast]
     4b. Edge-clipping rationalization.   [skipped with --fast]
     4c. Triple-fused depth enrichment.
  5. 3D lift + scene graph (SceneGraphBuilder).
  6. GPT-4o-mini reads scene graph + SAFETY_RULES → STOP/SLOW/CONTINUE.
  7. Write predictions.json (submission format) and run evaluate_local.py.

Usage:
  python benchmark.py                         # full val set (3785 images)
  python benchmark.py --fast                  # B100-optimised: larger models, BF16, skip intermediate LLM steps
  python benchmark.py --n 50                  # quick sanity check
  python benchmark.py --out my.json           # custom output path
  python benchmark.py --no-eval               # skip evaluate_local.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── GPU setup ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
STARTER    = ROOT / "data" / "challenge" / "starter_kit"
VAL_ANN    = ROOT / "data" / "annotations" / "val.json"
VAL_IMAGES = ROOT / "data" / "challenge" / "data" / "images" / "val"
SAFETY_RULES_PATH = ROOT / "action_module" / "SAFETY_RULES.md"

# Output dirs (saved alongside submission JSON for inspection)
OUT_ROOT  = ROOT / "data" / "benchmark_output"
OUT_GRAPH = OUT_ROOT / "scene_graphs"

DEPTH_MODEL_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
SAM2_MODEL_BASE   = "facebook/sam2-hiera-base-plus"
SAM2_MODEL_LARGE  = "facebook/sam2-hiera-large"

# ── sys.path: repo root + hyphenated 3d-module folder ─────────────────────────
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3d-module"))

# ── imports (after sys.path fixup) ────────────────────────────────────────────
from segment_module.llm_objects    import load_client as load_llm_client, get_detection_prompt
from segment_module.grounding_dino import load_model  as load_gdino,      detect
from segment_module.sam2           import load_model  as load_sam2,       segment
from depth_module.fused_depth      import enrich_detections
from src.filter_module             import is_interesting
from src.edge_rationalization      import rationalize_edge_detections
from src.llm_depth                 import get_llm_depth_signals
from llm_module.llm                import get_client  as get_llm_client,  analyse_with_scene_graph
from lift_3d                       import SceneGraphBuilder

# ── submission schema ─────────────────────────────────────────────────────────
GROUP_TO_CAT: dict[str, int] = {
    "HUMAN":         1,
    "VEHICLE":       2,
    "OBSTACLE":      3,
    "SAFETY_MARKER": 4,
}

DETECTION_CATEGORIES = [
    {"id": 1, "name": "HUMAN"},
    {"id": 2, "name": "VEHICLE"},
    {"id": 3, "name": "OBSTACLE"},
    {"id": 4, "name": "SAFETY_MARKER"},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_depth(raw: np.ndarray) -> np.ndarray:
    """Invert DepthAnything disparity → normalised depth (0=close, 1=far)."""
    d_min, d_max = raw.min(), raw.max()
    return (1.0 - (raw - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)


def _detection_summary(detections: list[dict]) -> str:
    """Minimal fallback scene text when the scene graph builder fails."""
    if not detections:
        return "Scene: no objects detected."
    lines = ["Scene (detection summary — no scene graph available):"]
    for d in detections:
        lines.append(
            f"  - {d.get('label','?')} ({d.get('risk_group','?')}) "
            f"proximity={d.get('proximity_label','?')} "
            f"zone={d.get('path_zone','?')} "
            f"depth={d.get('depth_score','?')}"
        )
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def run(n: int | None, out_path: Path, team_name: str, run_eval: bool = True,
        conf: float = 0.25, fast: bool = False) -> None:

    for d in (OUT_GRAPH,):
        d.mkdir(parents=True, exist_ok=True)

    # ── GPU dtype ─────────────────────────────────────────────────────────────
    use_bf16  = fast and torch.cuda.is_available()
    th_dtype  = torch.bfloat16 if use_bf16 else torch.float32
    depth_model_id = DEPTH_MODEL_LARGE if fast else DEPTH_MODEL_SMALL
    sam2_model_id  = SAM2_MODEL_LARGE  if fast else SAM2_MODEL_BASE

    if fast:
        print("Fast (B100-optimised) mode:")
        print(f"  depth model : {depth_model_id}")
        print(f"  SAM2 model  : {sam2_model_id}")
        print(f"  precision   : bfloat16" if use_bf16 else "  precision   : float32")
        print("  skipping    : llm_depth + edge_rationalization\n")

    # ── Load val annotations ──────────────────────────────────────────────────
    with open(VAL_ANN) as f:
        val_coco = json.load(f)
    filename_to_id: dict[str, int] = {
        img["file_name"]: img["id"] for img in val_coco["images"]
    }
    id_to_meta: dict[int, dict] = {img["id"]: img for img in val_coco["images"]}

    # ── Collect val images ────────────────────────────────────────────────────
    all_images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in VAL_IMAGES.glob(ext)
    )
    if n:
        all_images = all_images[:n]
    print(f"Benchmarking on {len(all_images)} val images.\n")

    # ── Load all models once ──────────────────────────────────────────────────
    print("Loading models ...")
    llm_client    = load_llm_client()
    gdino         = load_gdino()
    sam2_model    = load_sam2(sam2_model_id)

    depth_kwargs: dict = dict(task="depth-estimation", model=depth_model_id)
    if use_bf16:
        depth_kwargs["torch_dtype"] = th_dtype
    if torch.cuda.is_available():
        depth_kwargs["device"] = 0
    depth_pipe    = hf_pipeline(**depth_kwargs)

    # torch.compile on depth pipe's model for ~20-30% throughput gain on B100
    if fast and torch.cuda.is_available():
        try:
            depth_pipe.model = torch.compile(depth_pipe.model, mode="reduce-overhead")
            print("  torch.compile applied to depth model.")
        except Exception as e:
            print(f"  torch.compile skipped ({e})")

    graph_builder = SceneGraphBuilder(point_cloud_step=4)
    safety_rules  = SAFETY_RULES_PATH.read_text()
    print("All models loaded.\n")

    predictions: list[dict] = []
    det_records: list[dict] = []
    missing: list[str]      = []

    for img_path in tqdm(all_images, desc="Benchmark"):
        fname    = img_path.name
        stem     = img_path.stem
        image_id = filename_to_id.get(fname)
        if image_id is None:
            missing.append(fname)
            continue

        meta  = id_to_meta[image_id]
        img_w = meta["width"]
        img_h = meta["height"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  [{stem}] cannot open ({e}) — skipping")
            continue

        W, H = image.size

        # Step 1 — LLM object prompt
        try:
            prompt = get_detection_prompt(llm_client, image)
        except Exception as e:
            tqdm.write(f"  [{stem}] prompt failed ({e}) — using default")
            prompt = "person . vehicle . forklift . barrel . cone . box . ladder ."

        # Step 2 — Grounding DINO
        try:
            detections = detect(gdino, image, prompt, score_threshold=conf)
        except Exception as e:
            tqdm.write(f"  [{stem}] Grounding DINO failed ({e}) — no detections")
            detections = []

        if not detections:
            predictions.append({
                "image_id":   image_id,
                "action":     "CONTINUE",
                "confidence": 0.80,
                "reasoning":  "No objects detected.",
            })
            continue

        # Step 3 — SAM2
        try:
            segmented = segment(sam2_model, image, detections)
        except Exception as e:
            tqdm.write(f"  [{stem}] SAM2 failed ({e}) — masks zeroed")
            segmented = [{**d, "mask": np.zeros((H, W), dtype=bool), "mask_score": 0.0}
                         for d in detections]

        # Step 4 — DepthAnything V2
        try:
            raw_depth  = np.array(depth_pipe(image)["depth"], dtype=np.float32)
            norm_depth = _norm_depth(raw_depth)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth failed ({e}) — uniform depth")
            norm_depth = np.full((H, W), 0.5, dtype=np.float32)

        # Step 4a — LLM holistic depth signals  (skipped in --fast mode)
        if not fast:
            try:
                segmented = get_llm_depth_signals(image, segmented, llm_client)
            except Exception as e:
                tqdm.write(f"  [{stem}] llm_depth skipped ({e})")

        # Step 4b — Edge rationalization  (skipped in --fast mode)
        if not fast:
            try:
                segmented = rationalize_edge_detections(image, segmented, llm_client)
            except Exception as e:
                tqdm.write(f"  [{stem}] edge_rationalization skipped ({e})")

        # Step 4c — Triple-fused depth enrichment
        try:
            enriched, corrected_depth = enrich_detections(segmented, norm_depth, H, W)
        except Exception as e:
            tqdm.write(f"  [{stem}] fused_depth failed ({e}) — raw depth")
            enriched, corrected_depth = segmented, norm_depth

        # Step 5 — Scene graph
        masks = [d["mask"] for d in segmented]
        try:
            graph = graph_builder.process(
                depth_map  = corrected_depth,
                detections = enriched,
                img_w      = W,
                img_h      = H,
                image_id   = stem,
                masks      = masks,
            )
            graph_builder.save(graph, OUT_GRAPH)
            scene_text = graph.text
        except Exception as e:
            tqdm.write(f"  [{stem}] scene graph failed ({e}) — using detection summary")
            scene_text = _detection_summary(enriched)

        # Step 6 — LLM action decision
        result = analyse_with_scene_graph(
            client           = llm_client,
            original_path    = img_path,
            scene_graph_text = scene_text,
            safety_rules     = safety_rules,
        )

        predictions.append({
            "image_id":   image_id,
            "action":     result["action"],
            "confidence": round(result["confidence"], 4),
            "reasoning":  result["reasoning"],
        })

        tqdm.write(
            f"  [{stem}] {result['action']} ({result['confidence']:.2f})"
        )

        # Detection records (20% of score)
        for det in enriched:
            cat_id = GROUP_TO_CAT.get(det.get("risk_group", ""), None)
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            det_records.append({
                "image_id":    image_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                "score":       det.get("score", 0.0),
            })

    if missing:
        print(f"\nWarning: {len(missing)} images not in val annotations — skipped.")

    # ── Write submission JSON ─────────────────────────────────────────────────
    submission = {
        "team_name":            team_name,
        "predictions":          predictions,
        "detections":           det_records,
        "detection_categories": DETECTION_CATEGORIES,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)

    counts = Counter(p["action"] for p in predictions)
    print(f"\nPredictions written to {out_path}")
    print(f"  Total:    {len(predictions)}")
    for act in ("STOP", "SLOW", "CONTINUE"):
        c = counts.get(act, 0)
        n_total = len(predictions) or 1
        print(f"  {act:10s}: {c:5d}  ({c / n_total:.1%})")

    if not run_eval:
        return

    # ── Run local evaluator ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Running evaluate_local.py ...")
    print("=" * 60 + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(STARTER / "evaluate_local.py"),
            "--predictions", str(out_path),
            "--annotations", str(VAL_ANN),
        ],
        cwd=str(STARTER),
    )
    if result.returncode != 0:
        print(f"\nEvaluator exited with code {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-pipeline benchmark on val set")
    parser.add_argument("--n",       type=int,   default=None,
                        help="Limit to first N val images (default: all 3785)")
    parser.add_argument("--out",     type=Path,  default=ROOT / "predictions.json",
                        help="Output path for submission JSON")
    parser.add_argument("--team",    type=str,   default="ETHackers",
                        help="Team name in submission")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluate_local.py (just write predictions.json)")
    parser.add_argument("--fast",    action="store_true",
                        help="B100-optimised: larger models, BF16, skip intermediate LLM steps")
    args = parser.parse_args()

    run(
        n         = args.n,
        out_path  = args.out,
        team_name = args.team,
        run_eval  = not args.no_eval,
        conf      = args.conf,
        fast      = args.fast,
    )
