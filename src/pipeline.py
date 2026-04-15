"""
Main pipeline — 7-step perception-to-decision.

Steps per image:
  1. GPT-4o-mini generates an optimised Grounding DINO object prompt.
  2. Grounding DINO detects objects (open-vocabulary, no fixed classes).
  3. SAM2 segments each detected object to pixel-level masks.
  4. DepthAnything V2 produces a monocular depth map.
     4a. LLM holistic depth signals enrich per-object depth estimates.
     4b. Edge-clipping rationalization corrects truncated detections.
     4c. Triple-fused depth (DA + real-world size + bbox area) per object.
  5. 3D lift + scene graph: project depth → point cloud → spatial relations.
  6. GPT-4o-mini reads the scene graph + SAFETY_RULES → STOP/SLOW/CONTINUE.

Usage:
  python src/pipeline.py                  # 5 random images
  python src/pipeline.py --n 20
  python src/pipeline.py --n 3 --seed 7   # reproducible sample
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── repo root on sys.path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3d-module"))   # hyphenated folder — import via path

# ── local imports ─────────────────────────────────────────────────────────────
from segment_module.llm_objects    import load_client as load_llm_client, get_detection_prompt
from segment_module.grounding_dino import load_model  as load_gdino,      detect
from segment_module.sam2           import load_model  as load_sam2,       segment, masks_to_overlay
from depth_module.fused_depth      import enrich_detections
from src.filter_module             import is_interesting
from src.edge_rationalization      import rationalize_edge_detections
from src.llm_depth                 import get_llm_depth_signals
from llm_module.llm                import get_client  as get_llm_client,  analyse_with_scene_graph
from lift_3d                       import SceneGraphBuilder, visualize_3d

# ── paths ─────────────────────────────────────────────────────────────────────
TRAIN_DIR    = ROOT / "data" / "challenge" / "data" / "images" / "train"
OUT_ROOT     = ROOT / "data" / "pipeline_output"
OUT_DET      = OUT_ROOT / "detections"
OUT_OVERLAY  = OUT_ROOT / "overlays"
OUT_DEPTH    = OUT_ROOT / "depth_maps"
OUT_GRAPH    = OUT_ROOT / "scene_graphs"
OUT_CLOUD    = OUT_ROOT / "point_clouds"
OUT_LLM      = OUT_ROOT / "llm"
OUT_DISC     = OUT_ROOT / "discarded"
SAFETY_RULES = ROOT / "action_module" / "SAFETY_RULES.md"

DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Small-hf"


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_depth(raw: np.ndarray) -> np.ndarray:
    """Invert DepthAnything disparity → normalised depth (0=close, 1=far)."""
    d_min, d_max = raw.min(), raw.max()
    return (1.0 - (raw - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)


def _save_depth(norm_depth: np.ndarray, stem: str) -> None:
    """Save normalised depth map as a colourised PNG (inferno colormap)."""
    import cv2
    d8 = (norm_depth * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(OUT_DEPTH / f"{stem}_depth.png"), colored)


def _save_overlay(image: Image.Image, segmented: list[dict], stem: str) -> None:
    """Save SAM2 mask overlay as PNG."""
    overlay_rgba = masks_to_overlay(image, segmented)
    bg = Image.new("RGBA", overlay_rgba.size, (255, 255, 255, 255))
    bg.paste(overlay_rgba, mask=overlay_rgba.split()[3])
    bg.convert("RGB").save(str(OUT_OVERLAY / f"{stem}_overlay.png"))


def _detection_summary(detections: list[dict]) -> str:
    """Minimal scene text fallback when lift_3d fails."""
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


def _serialise(detections: list[dict]) -> list[dict]:
    """Return a JSON-safe copy of detections (strips numpy arrays like masks)."""
    out = []
    for d in detections:
        entry = {k: v for k, v in d.items() if k != "mask"}
        for k, v in entry.items():
            if isinstance(v, np.floating):
                entry[k] = float(v)
            elif isinstance(v, np.integer):
                entry[k] = int(v)
        out.append(entry)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def run(n: int = 5, seed: int | None = None, conf: float = 0.25) -> None:
    random.seed(seed if seed is not None else random.randint(0, 2 ** 32))

    for d in (OUT_DET, OUT_OVERLAY, OUT_DEPTH, OUT_GRAPH, OUT_CLOUD, OUT_LLM, OUT_DISC):
        d.mkdir(parents=True, exist_ok=True)

    # ── Sample images ─────────────────────────────────────────────────────────
    all_images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in TRAIN_DIR.glob(ext)
    )
    if not all_images:
        print(f"No images found in {TRAIN_DIR}")
        return
    sample = random.sample(all_images, min(n, len(all_images)))
    print(f"Sampled {len(sample)} images from {len(all_images)} total.\n")

    # ── Load all models once ──────────────────────────────────────────────────
    print("Loading models ...")
    llm_client    = load_llm_client()
    gdino         = load_gdino()
    sam2_model    = load_sam2()
    depth_pipe    = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL)
    graph_builder = SceneGraphBuilder(point_cloud_step=4)
    safety_rules  = SAFETY_RULES.read_text()
    print("All models loaded.\n")

    kept_count = 0
    disc_count = 0

    for img_path in tqdm(sample, desc="Pipeline"):
        stem = img_path.stem

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  [{stem}] cannot open image ({e}) — skipping")
            disc_count += 1
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
            tqdm.write(f"  [{stem}] Grounding DINO failed ({e}) — skipping")
            disc_count += 1
            continue

        # Early-out: no detections at all
        if not detections:
            tqdm.write(f"  [{stem}] discarded — Grounding DINO found nothing")
            with open(OUT_DISC / f"{stem}.json", "w") as f:
                json.dump({"image": img_path.name, "detections": []}, f, indent=2)
            disc_count += 1
            continue

        # Step 3 — SAM2 segmentation
        try:
            segmented = segment(sam2_model, image, detections)
        except Exception as e:
            tqdm.write(f"  [{stem}] SAM2 failed ({e}) — using detections without masks")
            segmented = [{**d, "mask": np.zeros((H, W), dtype=bool), "mask_score": 0.0}
                         for d in detections]
        masks = [d["mask"] for d in segmented]

        # Filter — discard if no risk-relevant objects
        if not is_interesting(segmented, conf_threshold=conf):
            tqdm.write(f"  [{stem}] discarded — no relevant objects")
            with open(OUT_DISC / f"{stem}.json", "w") as f:
                json.dump({"image": img_path.name, "detections": _serialise(segmented)}, f, indent=2)
            disc_count += 1
            continue

        kept_count += 1

        # Step 4 — DepthAnything V2
        try:
            raw_depth  = np.array(depth_pipe(image)["depth"], dtype=np.float32)
            norm_depth = _norm_depth(raw_depth)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth failed ({e}) — using uniform depth")
            norm_depth = np.full((H, W), 0.5, dtype=np.float32)

        # Step 4a — LLM holistic depth signals
        try:
            segmented = get_llm_depth_signals(image, segmented, llm_client)
        except Exception as e:
            tqdm.write(f"  [{stem}] llm_depth skipped ({e})")

        # Step 4b — Edge rationalization
        try:
            segmented = rationalize_edge_detections(image, segmented, llm_client)
        except Exception as e:
            tqdm.write(f"  [{stem}] edge_rationalization skipped ({e})")

        # Step 4c — Triple-fused depth enrichment
        try:
            enriched, corrected_depth = enrich_detections(segmented, norm_depth, H, W)
        except Exception as e:
            tqdm.write(f"  [{stem}] fused_depth failed ({e}) — using raw depth")
            enriched, corrected_depth = segmented, norm_depth

        # Step 5 — 3D lift + scene graph
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

        # Save depth map PNG
        try:
            _save_depth(corrected_depth, stem)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth map save failed ({e})")

        # Save point cloud visualisation PNG
        try:
            visualize_3d(
                graph,
                corrected_depth,
                save_path = OUT_CLOUD / f"{stem}_pointcloud.png",
                show      = False,
            )
        except Exception as e:
            tqdm.write(f"  [{stem}] point cloud viz failed ({e})")

        # Step 6 — LLM action decision
        result = analyse_with_scene_graph(
            client           = llm_client,
            original_path    = img_path,
            scene_graph_text = scene_text,
            safety_rules     = safety_rules,
        )

        # Save outputs
        with open(OUT_DET / f"{stem}.json", "w") as f:
            json.dump({"image": img_path.name, "detections": _serialise(enriched)}, f, indent=2)

        try:
            _save_overlay(image, segmented, stem)
        except Exception as e:
            tqdm.write(f"  [{stem}] overlay failed ({e})")

        with open(OUT_LLM / f"{stem}_analysis.json", "w") as f:
            json.dump(result, f, indent=2)

        tqdm.write(
            f"  [{stem}] {result['action']} ({result['confidence']:.2f}) — {result['reasoning'][:80]}"
        )

    print(f"\nDone.  Kept: {kept_count}  Discarded: {disc_count}")
    print(f"  Detections   → {OUT_DET}")
    print(f"  Overlays     → {OUT_OVERLAY}")
    print(f"  Depth maps   → {OUT_DEPTH}")
    print(f"  Scene graphs → {OUT_GRAPH}")
    print(f"  Point clouds → {OUT_CLOUD}")
    print(f"  Decisions    → {OUT_LLM}")
    print(f"  Discarded    → {OUT_DISC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perception-to-decision pipeline")
    parser.add_argument("--n",    type=int,   default=5,    help="Images to sample")
    parser.add_argument("--seed", type=int,   default=None, help="Random seed")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    run(n=args.n, seed=args.seed, conf=args.conf)
