# NavSense

**Deep, multi-stage robotic scene understanding — detection, segmentation, monocular depth, 3D lift, and safety decision, streamed live to a browser.**

NavSense is an end-to-end perception pipeline built for robots operating in dynamic, hazard-rich environments. Given a single RGB image, it produces a full scene graph, a fused depth map, a sparse 3D point cloud, and a `STOP / SLOW / CONTINUE` verdict — all driven by a layered combination of vision models and LLM reasoning.

The system is designed to be robust: three independent decision pathways cross-check each other, an LLM sanity pass catches physically implausible depths, and a human-correction loop continuously refines the safety policy.

---

## Table of contents

1. [How it works — the 12-stage pipeline](#how-it-works--the-12-stage-pipeline)
2. [Decision system](#decision-system)
3. [Install](#install)
4. [API keys](#api-keys)
5. [Run the web UI](#run-the-web-ui)
6. [Run batch mode on a dataset](#run-batch-mode-on-a-dataset)
7. [GPU acceleration & HF Inference API](#gpu-acceleration--hf-inference-api)
8. [Deploy to a GPU server](#deploy-to-a-gpu-server)
9. [Output layout](#output-layout)
10. [Troubleshooting](#troubleshooting)

---

## How it works — the 12-stage pipeline

NavSense processes every image through a fixed sequence of stages. Each stage enriches the data before passing it downstream.

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | **Ingest** | — | Load the raw image. |
| 2 | **LLM prompt generation** | `segment_module/llm_objects.py` | GPT-4o-mini reads the scene and writes a Grounding DINO prompt (e.g. `"person . forklift . hard hat . cardboard box ."`). |
| 3 | **Open-vocabulary detection** | `segment_module/grounding_dino.py` | Grounding DINO finds all objects matching the prompt. Unknown labels are auto-classified and persisted to `data/learned_aliases.json`. |
| 4 | **Instance segmentation** | `segment_module/sam2.py` | SAM 2 generates pixel-accurate masks for each detection box. |
| 5 | **Hazard filter** | `src/filter_module.py` | `is_interesting()` — does the scene contain any hazard-class object? Non-hazard scenes are early-exited and logged to `discarded/`. |
| 6 | **Monocular depth** | Depth Anything V2 via `transformers.pipeline` | Full-image depth map. Inverted so `0 = close`, `1 = far`. |
| 7 | **LLM depth reasoning** | `src/llm_depth.py` | GPT rates per-object proximity using spatial context and relative sizes. |
| 8 | **Edge rationalization** | `src/edge_rationalization.py` | LLM disambiguates objects clipped by the image border — cut-off vs. small and far is a critical distinction for navigation. |
| 9 | **Fused depth** | `depth_module/fused_depth.py` | Four signals merged per object: Depth Anything output, real-world size priors, bounding-box area, and LLM estimate. Weights are adaptive per object class. |
| 10 | **Part binding** | `app.py` | Deterministic spatial binding — helmets snap to their human, wheels to their vehicle (bbox containment ≥ 50%). |
| 11 | **LLM consistency check** | `app.py` | A second LLM pass catches physically implausible depth assignments and pushes background objects back. |
| 12 | **3D lift + Scene graph** | `3d-module/lift_3d.py` | Pinhole unprojection produces a sparse point cloud. A node/edge scene graph is built with blocking-relation detection. |

---

## Decision system

The final `STOP / SLOW / CONTINUE` verdict is produced by **three independent sources** that are then aggregated:

| Source | Module | How it reasons |
|---|---|---|
| **Scene graph LLM** | `llm_module/llm.py` — `analyse_with_scene_graph` | Passes the full scene graph + `SAFETY_RULES.md` to GPT-4o-mini and asks for a verdict with reasoning. |
| **LLM Actor** | `llm_action_module/actor.py` — `LLMActor` | Produces a full action-probability distribution; Shannon entropy flags uncertain scenes. |
| **Graph Classifier** | `action_module/` | 43-dimensional feature vector fed to a GradientBoosting model (rule-based fallback when untrained). |

The UI shows live probability bars and entropy for each source, plus a consensus view. When a human overrides a verdict in the UI, the correction is saved to `data/pipeline_output/human_corrections.json` and can be used to update `SAFETY_RULES.md`:

```bash
# Preview what the updated rules would look like
python -m llm_action_module.rule_updater --dry-run

# Rewrite the rules from accumulated corrections
python -m llm_action_module.rule_updater
```

This closes the loop: UI corrections → correction log → rule updater → refined policy → better decisions next run.

---

## Install

```bash
git clone https://github.com/alerest285/ETHackers.git
cd ETHackers

# Python 3.10+
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install PyTorch for your platform first — https://pytorch.org
#   Mac (MPS):   pip install torch torchvision
#   Linux/CUDA:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   CPU only:    pip install torch torchvision

pip install -r requirements.txt

# Web server extras (needed for the UI)
pip install fastapi "uvicorn[standard]" python-multipart
```

On first run, model weights are downloaded automatically from HuggingFace Hub:

| Model | Size |
|---|---|
| Depth Anything V2 Small | ~100 MB |
| Grounding DINO Base | ~680 MB |
| SAM 2 Hiera Base+ | ~380 MB |

Total: **~1.2 GB** cached under `~/.cache/huggingface/`.

---

## API keys

NavSense uses hosted LLMs for prompt generation, depth reasoning, consistency checking, and final decisions. Set these before running:

```bash
export OPENAI_API_KEY="sk-..."         # GPT-4o-mini — used throughout the pipeline
export ANTHROPIC_API_KEY="sk-ant-..."  # LLMActor and scene-graph LLM
```

If both are missing, NavSense falls back to rule-based decisions (`_rule_fallback` in `llm_module/llm.py`). The system still runs, but reasoning quality degrades significantly.

---

## Run the web UI

```bash
python app.py
```

Open **http://localhost:7860**.

### What the UI does

- **Upload one or many images** — drag-and-drop or click. Multiple images run in parallel with a live queue strip.
- **Every stage streams in real time** — DINO boxes, SAM masks, depth heatmap, and fused depth cross-fade into the center compositor as they complete.
- **When 3D lift finishes**, the compositor becomes an interactive Three.js point cloud — drag to orbit, scroll to zoom, click an object to see its depth, risk group, and 3D position.
- **Right sidebar** has three panels: Entities, Scene graph, Understanding. Each expands on hover to its full scrollable content.
- **Understanding panel** typewrites the LLM's reasoning, then shows consensus across all three decision sources with live probability bars, entropy, and top contributing features.
- **Result modal** pops up on decision — verdict + confidence + reasoning + correction buttons (`STOP` / `SLOW` / `CONTINUE` + optional note).
- **`⋯ outputs`** in the top bar opens a drawer of raw NDJSON event payloads per stage.

### Other run modes

```bash
# Different port
uvicorn app:app --host 0.0.0.0 --port 8080

# Live reload during development
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

---

## Run batch mode on a dataset

```bash
# Process 5 random images from data/challenge/data/images/train
python src/pipeline.py

# 50 images, fixed seed (reproducible)
python src/pipeline.py --n 50 --seed 42

# Raise detection threshold
python src/pipeline.py --n 100 --conf 0.30

# Auto-classify unknown labels via GPT and persist them
python src/pipeline.py --n 100 --learn auto
```

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 5 | Number of images to sample. |
| `--seed` | None | Random seed for reproducibility. |
| `--conf` | 0.25 | Grounding DINO confidence threshold. |
| `--learn` | off | Unknown label handling: `off`, `auto` (GPT classifies + persists), `interactive` (stdin prompt). |

To point at a different image directory:

```bash
ln -s /path/to/your/images data/challenge/data/images/train
python src/pipeline.py --n 1000
```

### LLM-Actor-only pipeline

A leaner variant that skips multimodal fusion and goes straight to the probability-based LLMActor:

```bash
python src/llmactor_pipeline.py --n 20
```

---

## GPU acceleration & HF Inference API

### Local GPU

If `torch.cuda.is_available()` is `True`, all models move to CUDA automatically. Expected latency:

| Stage | CPU | CUDA (B200) |
|---|---|---|
| Grounding DINO | ~25 s | ~1.2 s |
| SAM 2 (5 boxes) | ~18 s | ~0.8 s |
| Depth Anything V2 Small | ~8 s | ~0.3 s |
| **End-to-end** | **~60–90 s** | **~3–6 s** |

### HF Inference API (no local GPU)

```bash
export USE_HF_API=1
export HF_TOKEN="hf_..."
python app.py
```

Routes Grounding DINO and Depth Anything to HuggingFace's hosted endpoints (~2–4 s per image). SAM 2 stays local.

Optional model overrides:

```bash
export HF_GDINO_MODEL=IDEA-Research/grounding-dino-tiny
export HF_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf
```

---

## Deploy to a GPU server

The repo ships a production Dockerfile that serves the UI on port 7860 on a CUDA base image.

**Recommended spec (Northflank / CoreWeave):** 4 vCPU, 16 GB RAM, 1× B200, 20 GB ephemeral storage, 2 GB SHM.

1. New service → Combined from GitHub (`main` branch), build mode **Dockerfile**.
2. Expose port **7860** (HTTP, public).
3. Set runtime env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
4. CMD defaults to `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1`.

To tunnel privately without exposing the service:

```bash
npm i -g @northflank/cli
northflank login
northflank forward service --projectId hackathon --serviceId <your-service> --port 7860
```

---

## Output layout

```
data/pipeline_output/
├── detections/              # per-image enriched detection JSON (fused depth included)
├── overlays/                # SAM mask overlay PNGs
├── depth_maps/              # inferno-colored depth PNGs
├── point_clouds/            # top-down + perspective 3D renders
├── scene_graphs/            # nodes/edges JSON + LLM-ready text summary
├── llm/                     # final decision JSON {action, confidence, reasoning}
├── discarded/               # images rejected by the hazard filter
├── llm_actor_entropy.json   # {stem, entropy, action, probabilities} per image
└── human_corrections.json   # corrections submitted via the UI
```

---

## Troubleshooting

**`transformers has no attribute zero_shot_object_detection`**  
`huggingface_hub` is too old. Upgrade: `pip install -U huggingface_hub`.

**`TypeError: Object of type ndarray is not JSON serializable`**  
A SAM mask is leaking into an event payload. The `_SafeEncoder` in `app.py` handles this — confirm you're on the latest commit.

**`depth_score` is None downstream**  
Some fused-depth calls return `None` for objects with no usable signals. `app.py` sanitizes these to `0.5` before `lift_3d`. If you call `SceneGraphBuilder.process` directly, do the same.

**Grounding DINO / SAM fail on startup**  
Newer `transformers` versions renamed some kwargs. Both modules handle this; if you pinned an old version, upgrade: `pip install -U "transformers>=4.46"`.

**Container OOMs on cold start**  
The three models total ~1.2 GB on first download. Use at least **16 GB RAM + 20 GB ephemeral storage**. On a B200 they fit easily in VRAM; OOMs are host RAM, not VRAM.

**Port 7860 already in use**  
`uvicorn app:app --port 8080` or find the conflicting process: `lsof -i :7860`.

**No CUDA detected despite having a GPU**  
Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`. If `False`, reinstall PyTorch with the correct CUDA index URL (see Install).

---

## License

Research use. See the original THEKER Robotics challenge materials for dataset terms.
