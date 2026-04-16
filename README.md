# Navsense

**End-to-end robot vision — detection, segmentation, depth estimation, 3D lift, and safety decision, streamed live to a browser.**

Navsense runs a 12-stage perception pipeline on any image and outputs a
`STOP / SLOW / CONTINUE` verdict plus a full scene graph, interactive 3D
point cloud, and per-object reasoning. It ships two entry points:

- **`python src/pipeline.py`** — batch mode, runs over a whole dataset and writes results to disk.
- **`python app.py`** — an interactive web UI at `http://localhost:7860` that streams every stage live and lets you inspect the 3D scene, compare three independent decision sources, and submit human corrections.

Everything in this guide is written for someone who has just cloned the
repo and wants it running end-to-end in ten minutes.

---

## Table of contents

1. [The 12-stage pipeline](#the-12-stage-pipeline)
2. [Install](#install)
3. [API keys](#api-keys)
4. [Run the UI on localhost](#run-the-ui-on-localhost)
5. [Run the pipeline on a dataset](#run-the-pipeline-on-a-dataset)
6. [Speeding it up — GPU / HF Inference API](#speeding-it-up--gpu--hf-inference-api)
7. [Deploy on a GPU (Northflank / CoreWeave B200)](#deploy-on-a-gpu-northflank--coreweave-b200)
8. [Output layout](#output-layout)
9. [Troubleshooting](#troubleshooting)

---

## The 12-stage pipeline

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | **Raw** | — | Ingest the image. |
| 2 | **LLM → prompt** | `segment_module/llm_objects.py` | GPT-4o-mini writes the Grounding DINO prompt (e.g. `"person . forklift . hard hat . cardboard box ."`). |
| 3 | **Grounding DINO** | `segment_module/grounding_dino.py` | Open-vocabulary detection from that prompt. Unknown labels auto-classify via `classify_and_learn` and persist to `data/learned_aliases.json`. |
| 4 | **SAM 2** | `segment_module/sam2.py` | Per-detection pixel masks. |
| 5 | **Filter** | `src/filter_module.py` | `is_interesting()` — does the scene contain any hazard-group object? |
| 6 | **Depth** | Depth Anything V2 via `transformers.pipeline` | Monocular depth map, inverted so `0 = close`, `1 = far`. |
| 7 | **LLM depth** | `src/llm_depth.py` | GPT rates per-object proximity using scene context. |
| 8 | **Edge rationalization** | `src/edge_rationalization.py` | LLM disambiguates clipped-bbox objects (cut off vs. small and far). |
| 9 | **Fused depth** | `depth_module/fused_depth.py` | Four-signal fusion: DA + real-world size + bbox area + LLM signal. Per-object adaptive weights. |
| 10 | **Binding** | `app.py::_snap_parts_to_owners` | Deterministic: helmets → humans, wheels → vehicles (bbox-containment ≥ 50%). |
| 11 | **Sanity** | `app.py::_llm_consistency_check` | Second LLM pass catches physically implausible depths and pushes background objects back. |
| 12 | **3D lift + Scene graph** | `3d-module/lift_3d.py` | Pinhole unprojection → sparse point cloud + node/edge scene graph with blocking detection. |

The final **Decision** aggregates three independent sources:

- `llm_module.llm.analyse_with_scene_graph` — scene graph + `SAFETY_RULES.md` through GPT-4o-mini
- `llm_action_module.actor.LLMActor` — full action-probability distribution + Shannon entropy
- `action_module.graph_classifier.GraphClassifier` — 43-dimensional feature vector → GradientBoosting (or rule fallback when untrained)

---

## Install

```bash
git clone https://github.com/alerest285/ETHackers.git
cd ETHackers

# Create a venv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch for your platform first — see https://pytorch.org
# Examples:
#   Mac (MPS):    pip install torch torchvision
#   Linux/CUDA:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   CPU only:     pip install torch torchvision

# Then the rest
pip install -r requirements.txt

# Web-server extras (needed for the UI)
pip install fastapi "uvicorn[standard]" python-multipart
```

On first run, each model downloads its weights from HuggingFace Hub:

| Model | Size | Where |
|---|---|---|
| Depth Anything V2 Small | ~100 MB | `depth-anything/Depth-Anything-V2-Small-hf` |
| Grounding DINO Base | ~680 MB | `IDEA-Research/grounding-dino-base` |
| SAM 2 Hiera Base+ | ~380 MB | `facebook/sam2-hiera-base-plus` |

Total: **~1.2 GB** cached under `~/.cache/huggingface/`.

---

## API keys

The pipeline calls hosted LLMs for the prompt, depth reasoning,
consistency check, and decision. Set both keys in your shell
(or a `.env` file):

```bash
export OPENAI_API_KEY="sk-..."        # GPT-4o-mini, SAM/GDINO prompt, consistency, decision
export ANTHROPIC_API_KEY="sk-ant-..." # LLMActor (upstream llm_module.llm)
```

The keys are loaded by:

- `segment_module/llm_objects.load_client()` — primary client used across the app
- `llm_module/llm.get_client()` — used by `analyse_with_scene_graph` and the LLMActor

If both are missing, the pipeline degrades to rule-based decisions (`_rule_fallback` in `llm_module/llm.py`) but you'll lose most of the reasoning quality.

---

## Run the UI on localhost

The interactive web UI is the easiest way to experiment.

```bash
python app.py
```

Open **http://localhost:7860**.

### What you can do in the UI

- **Upload one or many images** (drag-and-drop or click).
- **Multiple images run in parallel** — a queue strip appears above the main canvas with live progress on each.
- **Every stage streams live**: YOLO/GDINO boxes, SAM masks, depth heatmap, fused heatmap, all cross-fade into the center compositor.
- **When the 3D lift finishes**, the compositor morphs into an interactive Three.js point cloud. Drag to orbit, scroll to zoom, click an object to open a popover with its depth, risk group and 3D position.
- **The right sidebar** has three hover-expanding panels: Entities, Scene graph, Understanding. Each stays compact by default and expands to its scrollable content on hover.
- **The Understanding panel** typewrites the LLM's reasoning and then shows a consensus view across all three decision sources with live probability bars, Shannon entropy, and top contributing features.
- **The Result modal** pops up when the decision completes — verdict + confidence + reasoning + entities + human-correction buttons (`STOP` / `SLOW` / `CONTINUE` + optional note). Corrections are saved to `data/pipeline_output/human_corrections.json`.
- **`⋯ outputs`** in the top bar opens a drawer of raw NDJSON event payloads per stage.

### Stop it

`Ctrl-C` in the terminal.

### Run on a different port

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Run with live reload during development

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

---

## Run the pipeline on a dataset

For batch processing, use the canonical CLI:

```bash
# Process 5 random images from data/challenge/data/images/train
python src/pipeline.py

# Process 50 images with a fixed seed (reproducible)
python src/pipeline.py --n 50 --seed 42

# Raise the detection threshold
python src/pipeline.py --n 100 --conf 0.30

# Auto-classify unknown labels via GPT and persist them
python src/pipeline.py --n 100 --learn auto
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 5 | Number of images to sample. |
| `--seed` | None | Random seed for reproducibility. |
| `--conf` | 0.25 | Grounding DINO confidence threshold. |
| `--learn` | off | How to handle UNKNOWN labels: `off`, `auto` (GPT classifies + persists), `interactive` (prompt on stdin). |

### Point at a different image directory

The pipeline reads from `data/challenge/data/images/train/` by default.
To process a different set, symlink your images into that location:

```bash
mkdir -p data/challenge/data/images
ln -s /path/to/your/images data/challenge/data/images/train
python src/pipeline.py --n 1000
```

### Outputs

Results land under `data/pipeline_output/`:

```
data/pipeline_output/
  detections/{stem}.json          # enriched detections w/ fused depth
  overlays/{stem}_overlay.png     # SAM mask overlay on the source
  depth_maps/{stem}_depth.png     # inferno-colored depth
  point_clouds/{stem}_pointcloud.png  # top-down + perspective 3D viz
  scene_graphs/{stem}.json        # nodes + edges JSON
  scene_graphs/{stem}.txt         # LLM-ready scene description
  llm/{stem}_analysis.json        # {action, confidence, reasoning}
  discarded/{stem}.json           # images the filter rejected
  llm_actor_entropy.json          # Shannon-entropy log per image
  human_corrections.json          # populated by the UI's feedback buttons
```

### Alternative: the LLM-Actor-only pipeline

```bash
python src/llmactor_pipeline.py --n 20
```

Same inputs, different final stage — skips `MultimodalFusion` and goes
straight to the probability-based `LLMActor`. Faster, simpler, and the
entropy log it produces is the input for the **rule updater**:

```bash
# Preview what the updated SAFETY_RULES.md would look like
python -m llm_action_module.rule_updater --dry-run

# Actually rewrite the rules from accumulated corrections
python -m llm_action_module.rule_updater
```

This closes the loop: UI corrections → `human_corrections.json` → rule
updater → new `SAFETY_RULES.md` → next run uses the refined policy.

---

## Speeding it up — GPU / HF Inference API

### Use a local GPU

If `torch.cuda.is_available()` returns `True`, every model moves to
CUDA automatically:

- **Grounding DINO** and **SAM 2** check `torch.cuda.is_available()` at
  `load_model()` and place the weights on `cuda`.
- **Depth Anything V2** moves to `device=0` via the HF pipeline call in
  `app.py::_get_depth_pipe()`.
- **YOLO** (when used) also respects CUDA via Ultralytics defaults.

Startup log confirms the device:

```
[app] CUDA available — 1× NVIDIA B200
[app] DepthAnything loaded on device=0
Loading Grounding DINO on cuda ...
Loading SAM2 on cuda ...
```

Expected latency on a B200:

| Stage | CPU | CUDA |
|---|---|---|
| YOLO | ~0.8 s | ~0.05 s |
| Grounding DINO | ~25 s | ~1.2 s |
| SAM 2 (5 boxes) | ~18 s | ~0.8 s |
| DepthAnything V2 Small | ~8 s | ~0.3 s |
| **End-to-end** | **~60–90 s** | **~3–6 s** |

### Offload to HF Inference API

If you don't have a GPU locally, you can route the two heaviest models
to Hugging Face's hosted endpoints:

```bash
export USE_HF_API=1
export HF_TOKEN="hf_..."              # https://huggingface.co/settings/tokens
python app.py
```

The wrapper in `hf_inference.py` replaces the local Grounding DINO and
Depth Anything calls with `huggingface_hub.InferenceClient` calls —
about 2–4 s per image on HF's shared GPUs. SAM 2 stays local because
the bbox-prompted API path isn't clean on the serverless side.

Optional overrides:

```bash
export HF_GDINO_MODEL=IDEA-Research/grounding-dino-tiny   # smaller/faster
export HF_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf  # higher quality
```

---

## Deploy on a GPU (Northflank / CoreWeave B200)

The repo ships a production Dockerfile that serves the web UI on port
7860 on a CUDA base image.

### One-shot deploy

1. **New service → Combined** from GitHub (`main` branch), build mode
   **Dockerfile**.
2. **Resources**: 4 vCPU, 16 GB RAM, 1× B200, 20 GB ephemeral storage,
   2 GB SHM.
3. **Ports**: expose **7860** (HTTP, public). Northflank hands you a
   `https://<service>.<project>.code.run` URL.
4. **Runtime env vars**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
5. **CMD**: leave empty — the Dockerfile defaults to
   `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1`.

Once the pod boots, open the public URL and you'll see the same UI that
runs on `localhost:7860`, now backed by a B200.

### Tunnel through Northflank CLI

Don't want to expose the service publicly? Port-forward instead:

```bash
npm i -g @northflank/cli
northflank login
northflank forward service \
  --projectId hackathon \
  --serviceId <your-service> \
  --port 7860
```

Then browse to http://localhost:7860 — your laptop is tunneled to the
B200 pod over SSH.

---

## Output layout

After a run (UI or CLI), check these directories:

```
data/pipeline_output/
├── detections/           # per-image enriched detection JSON
├── overlays/             # SAM mask overlay PNGs
├── depth_maps/           # colorized depth PNGs
├── point_clouds/         # matplotlib 3D renders
├── scene_graphs/         # nodes/edges JSON + text summary
├── llm/                  # final decision JSON (action/confidence/reasoning)
├── discarded/            # images the filter rejected
├── llm_actor_entropy.json # {stem, entropy, action, probabilities}[]
└── human_corrections.json # feedback submitted via the UI
```

The UI reads `llm_actor_entropy.json` through `GET /entropy_log` to
surface the most-uncertain frames.

---

## Troubleshooting

**"transformers has no attribute `zero_shot_object_detection`"**  
Your `huggingface_hub` is too old for the HF Inference API route. Upgrade: `pip install -U huggingface_hub`.

**`TypeError: Object of type ndarray is not JSON serializable`**  
A detection dict still has a SAM mask when an event is emitted. The custom `_SafeEncoder` in `app.py` handles this; if you see it again, confirm you're on the latest commit.

**`depth_score` is None downstream**  
Some fused-depth calls can set `depth_score=None` for a detection with no usable signals. `app.py` sanitizes these to 0.5 before handing them to `lift_3d`; if you bypass `app.py` and call `SceneGraphBuilder.process` directly, do the same.

**Grounding DINO / SAM fail on startup**  
Newer `transformers` versions rename kwargs (`box_threshold` → `threshold`, `reshaped_input_sizes` removed). Both `segment_module/grounding_dino.py` and `segment_module/sam2.py` now handle this. If you pinned an old version, upgrade: `pip install -U "transformers>=4.46"`.

**Container OOMs on Northflank cold start**  
The three models cache ~1.2 GB of weights on first request. Bump the service to **16 GB RAM + 20 GB ephemeral storage**. On a B200 the models fit comfortably in GPU memory; the OOM you're seeing is host RAM, not VRAM.

**Port 7860 already in use**  
Run on a different port: `uvicorn app:app --port 8080` or kill the other process with `lsof -i :7860`.

**No accelerator detected but I have a GPU**  
PyTorch doesn't always pick up CUDA in every venv. Verify:
```python
import torch; print(torch.cuda.is_available(), torch.cuda.device_count())
```
If `False`, reinstall torch with the correct CUDA index URL (see the Install section).

---

## License

Research / hackathon use. See the original THEKER Robotics challenge
materials for the dataset's terms.
