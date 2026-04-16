"""
LLM-Actor pipeline — same perception stack as pipeline.py, but steps 7–12
are handled entirely by the LLMActor instead of MultimodalFusion + NeuralPredictor.

Steps per image:
  1. GPT-4o-mini generates an optimised Grounding DINO object prompt.
  2. Grounding DINO detects objects (open-vocabulary, no fixed classes).
  3. SAM2 segments each detected object to pixel-level masks.
  4. DepthAnything V2 produces a monocular depth map.
     4a. LLM holistic depth signals enrich per-object depth estimates.
     4b. Edge-clipping rationalization corrects truncated detections.
     4c. Triple-fused depth (DA + real-world size + bbox area) per object.
  5. 3D lift: depth map → point cloud.
  6. Scene graph from 3D point cloud + detections.
  7. LLMActor: send scene graph + depth stats + point cloud stats +
               segmentation image + depth image → STOP/SLOW/CONTINUE
               probabilities, confidence, reasoning.
  8. Track decision entropy to data/pipeline_output/llm_actor_entropy.json
     for future active learning (high-entropy frames = best labelling targets).
  9. Save action + confidence + reasoning + full probability distribution
     to one JSON file per image.

Usage:
  python src/llmactor_pipeline.py                  # 5 random images
  python src/llmactor_pipeline.py --n 20
  python src/llmactor_pipeline.py --n 3 --seed 7   # reproducible sample
"""

import argparse
import datetime
import json
import math
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
sys.path.insert(0, str(ROOT / "3d-module"))

# ── local imports ─────────────────────────────────────────────────────────────
from segment_module.llm_objects    import load_client as load_llm_client, get_detection_prompt
from segment_module.grounding_dino import load_model  as load_gdino,      detect
from segment_module.sam2           import load_model  as load_sam2,       segment, masks_to_overlay
from depth_module.fused_depth      import enrich_detections
from src.filter_module             import is_interesting
from src.edge_rationalization      import rationalize_edge_detections
from src.llm_depth                 import get_llm_depth_signals
from lift_3d                       import SceneGraphBuilder, visualize_3d, lift_to_3d
from llm_action_module.actor        import LLMActor
from llm_action_module.rule_updater import update_rules

# ── paths ─────────────────────────────────────────────────────────────────────
TRAIN_DIR  = ROOT / "data" / "challenge" / "data" / "images" / "train"
OUT_ROOT   = ROOT / "data" / "pipeline_output"
OUT_DET    = OUT_ROOT / "detections"
OUT_OVERLAY= OUT_ROOT / "overlays"
OUT_DEPTH  = OUT_ROOT / "depth_maps"
OUT_GRAPH  = OUT_ROOT / "scene_graphs"
OUT_CLOUD  = OUT_ROOT / "point_clouds"
OUT_ACTION = OUT_ROOT / "llm_actions"   # separate folder from the NN pipeline
OUT_DISC          = OUT_ROOT / "discarded"
CORRECTIONS_PATH  = OUT_ROOT / "human_corrections.json"

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_depth(raw: np.ndarray) -> np.ndarray:
    d_min, d_max = raw.min(), raw.max()
    return (1.0 - (raw - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)


def _save_depth(norm_depth: np.ndarray, stem: str) -> None:
    import cv2
    d8      = (norm_depth * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(OUT_DEPTH / f"{stem}_depth.png"), colored)


def _save_overlay(image: Image.Image, segmented: list[dict], stem: str) -> None:
    overlay_rgba = masks_to_overlay(image, segmented)
    bg = Image.new("RGBA", overlay_rgba.size, (255, 255, 255, 255))
    bg.paste(overlay_rgba, mask=overlay_rgba.split()[3])
    bg.convert("RGB").save(str(OUT_OVERLAY / f"{stem}_overlay.png"))


def _detection_summary(detections: list[dict]) -> str:
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


# ── active learning ───────────────────────────────────────────────────────────

def _load_corrections() -> list[dict]:
    if CORRECTIONS_PATH.exists():
        try:
            return json.loads(CORRECTIONS_PATH.read_text())
        except Exception:
            return []
    return []


def _save_correction(entry: dict) -> None:
    existing = _load_corrections()
    existing.append(entry)
    CORRECTIONS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"  Correction saved → {CORRECTIONS_PATH}")


def _active_learning_prompt(candidate: dict, idx: int, total: int) -> None:
    """
    Show one high-entropy prediction and ask the user to verify / correct it.
    If they disagree, the correction is appended to human_corrections.json and
    the user is offered an immediate rule update.
    """
    VALID = {"STOP", "SLOW", "CONTINUE"}
    probs = candidate["probabilities"]

    print("\n" + "═" * 60)
    print(f"  ACTIVE LEARNING — sample {idx} of {total}  "
          f"(ranked by entropy, highest first)")
    print("═" * 60)
    print(f"  Image    : {candidate['image']}")
    print(f"  Entropy  : {candidate['entropy']:.4f}  "
          f"(max={math.log2(3):.4f} = fully uncertain)")
    print(f"  Decision : {candidate['action']}  "
          f"(confidence {candidate['confidence']:.2f})")
    print(f"  Reasoning: {candidate['reasoning']}")
    print(f"  Probs    : " +
          "  ".join(f"{k}={v:.0%}" for k, v in probs.items()))
    if candidate.get("overlay_path"):
        print(f"  Overlay  : {candidate['overlay_path']}")
    print()

    # Ask for the human's action directly — richer signal than agree/disagree
    print(f"  What action would YOU have taken for this scene?")
    while True:
        try:
            answer = input("  [STOP / SLOW / CONTINUE]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n  (skipped)")
            return
        if answer in VALID:
            break
        print(f"  Please enter one of: {', '.join(VALID)}")

    agreed = (answer == candidate["action"])
    kind   = "confirmation" if agreed else "correction"
    signal = "reinforcement" if agreed else "correction"
    print(f"  {'✓ Matches prediction' if agreed else '✗ Differs from prediction'} "
          f"— saving as {signal}.")

    try:
        note = input("  Optional note for the rule updater (Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        note = ""

    entry = {
        "type":                    kind,
        "stem":                    candidate["stem"],
        "image":                   candidate["image"],
        "predicted_action":        candidate["action"],
        "correct_action":          answer,
        "predicted_confidence":    candidate["confidence"],
        "predicted_probabilities": probs,
        "entropy":                 candidate["entropy"],
        "reasoning":               candidate["reasoning"],
        "scene_summary":           candidate.get("scene_summary", ""),
        "user_note":               note,
        "timestamp":               datetime.datetime.now().isoformat(),
    }
    _save_correction(entry)

    verb = "Reinforce" if agreed else "Fix"
    try:
        do_update = input(f"\n  {verb} SAFETY_RULES.md now? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        do_update = "n"

    if do_update == "y":
        print()
        update_rules()
    else:
        print("  Run later with:  python llm_action_module/rule_updater.py")

    print("═" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    n:               int = 5,
    seed:            int | None = None,
    conf:            float = 0.25,
    learn:           str = "off",
    active_learning: bool = False,
    n_correct:       int = 1,
) -> None:
    if learn not in ("off", "auto", "interactive"):
        raise ValueError(f"learn must be 'off', 'auto', or 'interactive'; got {learn!r}")
    random.seed(seed if seed is not None else random.randint(0, 2 ** 32))

    for d in (OUT_DET, OUT_OVERLAY, OUT_DEPTH, OUT_GRAPH, OUT_CLOUD, OUT_ACTION, OUT_DISC):
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

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models ...")
    llm_client    = load_llm_client()
    gdino         = load_gdino()
    sam2_model    = load_sam2()
    depth_pipe    = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL)
    graph_builder = SceneGraphBuilder(point_cloud_step=4)
    actor         = LLMActor()
    print("All models loaded.\n")

    kept_count  = 0
    disc_count  = 0
    run_results = []   # collects per-image results for the active-learning prompt

    for img_path in tqdm(sample, desc="LLMActor Pipeline"):
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
        detect_kwargs = {"score_threshold": conf}
        if learn == "auto":
            detect_kwargs["learn_client"] = llm_client
        elif learn == "interactive":
            detect_kwargs["interactive"] = True

        try:
            detections = detect(gdino, image, prompt, **detect_kwargs)
        except Exception as e:
            tqdm.write(f"  [{stem}] Grounding DINO failed ({e}) — skipping")
            disc_count += 1
            continue

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

        # Step 5 — 3D lift: depth map → point cloud
        try:
            point_cloud_arr = lift_to_3d(corrected_depth, W, H, step=8)
        except Exception as e:
            tqdm.write(f"  [{stem}] lift_to_3d failed ({e}) — no point cloud")
            point_cloud_arr = None

        # Step 6 — Scene graph from 3D lift + detections
        graph = None
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

        # Save depth map PNG (used by actor as visual input)
        try:
            _save_depth(corrected_depth, stem)
        except Exception as e:
            tqdm.write(f"  [{stem}] depth map save failed ({e})")

        # Save SAM2 overlay PNG (used by actor as visual input)
        try:
            _save_overlay(image, segmented, stem)
        except Exception as e:
            tqdm.write(f"  [{stem}] overlay failed ({e})")

        # Save point cloud visualisation
        if graph is not None:
            try:
                visualize_3d(
                    graph,
                    corrected_depth,
                    save_path = OUT_CLOUD / f"{stem}_pointcloud.png",
                    show      = False,
                )
            except Exception as e:
                tqdm.write(f"  [{stem}] point cloud viz failed ({e})")

        # Save detection JSON
        with open(OUT_DET / f"{stem}.json", "w") as f:
            json.dump({"image": img_path.name, "detections": _serialise(enriched)}, f, indent=2)

        # Step 7 — LLMActor: scene graph + depth + point cloud + images → probabilities
        # Step 8 — entropy is logged inside actor.query() to llm_actor_entropy.json
        result = actor.query(
            scene_graph_text = scene_text,
            seg_img_path     = OUT_OVERLAY / f"{stem}_overlay.png",
            depth_img_path   = OUT_DEPTH   / f"{stem}_depth.png",
            depth_arr        = corrected_depth,
            point_cloud      = point_cloud_arr,
            stem             = stem,
        )

        # Step 9 — Save one JSON: action + confidence + reasoning + full distribution
        output = {
            "image":         img_path.name,
            "action":        result["action"],
            "confidence":    result["confidence"],
            "reasoning":     result["reasoning"],
            "probabilities": result["probabilities"],
            "entropy":       result["entropy"],
        }
        with open(OUT_ACTION / f"{stem}_action.json", "w") as f:
            json.dump(output, f, indent=2)

        # Track for active-learning prompt at end of run
        run_results.append({
            "stem":                   stem,
            "image":                  img_path.name,
            "action":                 result["action"],
            "confidence":             result["confidence"],
            "reasoning":              result["reasoning"],
            "probabilities":          result["probabilities"],
            "entropy":                result["entropy"],
            "scene_summary":          scene_text[:400],   # for rule updater context
            "overlay_path":           str(OUT_OVERLAY / f"{stem}_overlay.png"),
        })

        tqdm.write(
            f"  [{stem}] {result['action']} "
            f"(conf={result['confidence']:.2f} entropy={result['entropy']:.3f}) "
            f"— {result['reasoning'][:70]}"
        )

    # ── Active-learning correction ────────────────────────────────────────────
    if active_learning and run_results:
        ranked   = sorted(run_results, key=lambda r: r["entropy"], reverse=True)
        to_label = ranked[:min(n_correct, len(ranked))]
        for i, candidate in enumerate(to_label, start=1):
            _active_learning_prompt(candidate, idx=i, total=len(to_label))

    print(f"\nDone.  Kept: {kept_count}  Discarded: {disc_count}")
    print(f"  Detections   → {OUT_DET}")
    print(f"  Overlays     → {OUT_OVERLAY}")
    print(f"  Depth maps   → {OUT_DEPTH}")
    print(f"  Scene graphs → {OUT_GRAPH}")
    print(f"  Point clouds → {OUT_CLOUD}")
    print(f"  Decisions    → {OUT_ACTION}")
    print(f"  Entropy log  → {ROOT / 'data' / 'pipeline_output' / 'llm_actor_entropy.json'}")
    print(f"  Discarded    → {OUT_DISC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Actor perception-to-decision pipeline")
    parser.add_argument("--n",    type=int,   default=5,    help="Images to sample")
    parser.add_argument("--seed", type=int,   default=None, help="Random seed")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--learn",
        choices=("off", "auto", "interactive"),
        default="off",
        help="Handle UNKNOWN labels: off / auto (GPT-4o-mini) / interactive (stdin)",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        default=False,
        help="After the run, prompt for corrections on the highest-entropy predictions",
    )
    parser.add_argument(
        "--correct",
        type=int,
        default=1,
        metavar="N",
        help="Number of high-entropy samples to correct in active-learning mode (default: 1)",
    )
    args = parser.parse_args()

    run(
        n               = args.n,
        seed            = args.seed,
        conf            = args.conf,
        learn           = args.learn,
        active_learning = args.active_learning,
        n_correct       = args.correct,
    )
