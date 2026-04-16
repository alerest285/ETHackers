# NavSense

Three ETH Zürich robotics undergrads. 20 hours. No sleep. No GPUs.

We still built something that kinda works.

---

## The short version

NavSense takes a single RGB image and spits out: a fused depth map, a sparse 3D point cloud, a spatial scene graph, and a safety verdict (`STOP / SLOW / CONTINUE`). The verdict comes from three completely independent reasoning systems that cross-check each other.

Built for the THEKER Robotics challenge. Goal: help a robot make safe navigation calls in busy industrial scenes — forklifts, people, spills, hard hats, barrels, that sort of chaos.

Important caveat: we had 20 hours and literally zero GPU access. Everything is implemented. Not everything is pretrained. Below we explain what works, what's sketchy, and how to run it.

---

## What's inside

- [High level — what NavSense actually does](#high-level--what-navsense-actually-does)
- [The 12-stage pipeline](#the-12-stage-pipeline)
- [Fused depth — 4-signal adaptive fusion](#fused-depth--4-signal-adaptive-fusion)
- [Scene graph & 3D lift](#scene-graph--3d-lift)
- [The multimodal encoder we designed (and couldn't fully train)](#the-multimodal-encoder-we-designed-and-couldnt-fully-train)
- [Three independent decision legs](#three-independent-decision-legs)
- [Active learning loop](#active-learning-loop)
- [Install](#install)
- [API keys](#api-keys)
- [Run the web UI](#run-the-web-ui)
- [Run batch mode](#run-batch-mode)
- [GPU / HF Inference API](#gpu--hf-inference-api)
- [Deploy to a GPU server](#deploy-to-a-gpu-server)
- [Output layout](#output-layout)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## High level — what NavSense actually does

1. **Detects things** — open-vocabulary detection (Grounding DINO) + pixel-accurate masks (SAM 2). Labels get mapped to risk tiers: HUMAN, VEHICLE, OBSTACLE, SAFETY_MARKER, BACKGROUND. Unknown labels auto-classify via GPT and persist for next time.
2. **Estimates where things are** — monocular depth (Depth Anything V2) fused with real-world size priors, bounding box geometry, and LLM proximity estimates. Four signals, not one.
3. **Builds a 3D model of the scene** — pinhole unprojection into a sparse point cloud, then a node-edge scene graph with blocking relations, proximity labels, and path zone classification.
4. **Makes a safety decision** — three independent systems each look at the scene differently, then we pick the safest output.
5. **Learns from mistakes** — human corrections flow into a rule updater that rewrites the safety policy for next time.

All from a single RGB image. No LiDAR, no stereo, no depth sensor.

---

## The 12-stage pipeline

| # | Stage | Where | What happens |
|---|---|---|---|
| 1 | **Ingest** | — | Load the raw image |
| 2 | **Prompt gen** | `segment_module/llm_objects.py` | GPT-4o-mini writes a Grounding DINO prompt: `"person . forklift . hard hat . barrel ."` |
| 3 | **Open-vocab detection** | `segment_module/grounding_dino.py` | Grounding DINO finds everything matching the prompt. Unknown labels get auto-classified and saved |
| 4 | **Instance masks** | `segment_module/sam2.py` | SAM 2 makes pixel-accurate masks per detection box |
| 5 | **Hazard filter** | `src/filter_module.py` | No hazard-tier objects? Exit early, log to `discarded/`, skip the expensive stuff |
| 6 | **Monocular depth** | Depth Anything V2 | Full-image depth map. Inverted so `0 = close`, `1 = far` |
| 7 | **LLM depth** | `src/llm_depth.py` | GPT estimates per-object proximity from the image — catches things the depth model gets wrong |
| 8 | **Edge rationalization** | `src/edge_rationalization.py` | Clipped-edge objects are ambiguous: far away, or very close and cut off? Looks identical to an area heuristic. LLM disambiguates |
| 9 | **Fused depth** | `depth_module/fused_depth.py` | Four signals merged per object with adaptive outlier correction |
| 10 | **Part binding** | `app.py` | Deterministic snap: helmet → person, wheels → vehicle (bbox containment ≥ 50%) |
| 11 | **LLM consistency check** | `app.py` | Second LLM pass. Catches physically implausible assignments — person marked FAR when they're clearly in the foreground, that kind of thing |
| 12 | **3D lift + scene graph** | `3d-module/lift_3d.py` | Pinhole unprojection → sparse point cloud → node-edge scene graph with blocking detection |

Final verdict = conservative merge of three independent decision sources.

---

## Fused depth — the practical hack

Monocular depth from a single image is genuinely hard and Depth Anything V2 makes real mistakes, especially on industrial objects it hasn't seen much of. We don't trust one signal.

For each detected object we compute a `depth_score` from four independent signals:

| Signal | Weight | How |
|---|---|---|
| **Depth Anything V2** | 0.40 | Median depth value inside the object's mask region |
| **Real-world size prior** | 0.20 | Pinhole model: `distance = known_height × focal_px / bbox_height_px`. Reference heights per risk tier (human = 1.7m, vehicle = 2.5m, etc.) |
| **BBox area heuristic** | 0.10 | Big bbox = probably close. Overridden by edge rationalization when the object is clipped |
| **LLM holistic estimate** | 0.30 | GPT estimates proximity from the full image. Handles occlusion, perspective, shadows |

When Depth Anything strongly disagrees with the geometry-based signals (deviation > 0.25) and those signals agree with each other (confidence > 0.4), we automatically demote its weight down toward 0.05 and trust geometry more. Missing signals get skipped, weights renormalize.

```
depth_score = Σ(signal_i × weight_i) / Σ(weight_i)
```

Proximity buckets: `CLOSE` ≤ 0.35 / `MEDIUM` 0.35–0.65 / `FAR` > 0.65

Path zone: `CENTER` if the object's bbox center sits in the middle third of the frame (the robot's forward corridor). `PERIPHERAL` otherwise.

---

## Scene graph & 3D lift

We unproject with a basic pinhole model and an estimated focal length:

```
fx = fy = max(W, H) × 0.8    # roughly 60° FOV
cx = W/2,  cy = H/2

Z = depth_map[y, x]
X = (x - cx) × Z / fx
Y = (y - cy) × Z / fy
```

Sample every 4th pixel for the full-frame cloud. For each detection we compute centroid 3D, nearest Z (closest point inside the object — used for blocking), and 3D extent. SAM mask when available, bbox ROI otherwise.

**Nodes** carry: risk group, risk score, proximity label, path zone, depth score, detection confidence, 2D and 3D spatial info.

**Edges** carry: 3D Euclidean distance, relative position, blocking flag.

Relative position logic:
- If depth difference between centroids > 8% of total scene depth range → `in front of` / `behind`
- Otherwise → `to the left of` / `to the right of`

Blocking:
```python
blocking = (
    node_a.nearest_z < node_b.centroid_z - 0.05   # A is closer to camera
    and bboxes_overlap_in_2d(a, b)                  # their 2D projections overlap
)
```

We serialize the graph to compact readable text and feed it directly into the LLM reasoning stages:

```
Scene (3 objects detected):
  [1] HUMAN 'person' (conf=0.95) — CLOSE, CENTER, depth=0.12
  [2] VEHICLE 'forklift' (conf=0.88) — MEDIUM, PERIPHERAL, depth=0.42
  [3] OBSTACLE 'barrel' (conf=0.72) — FAR, PERIPHERAL, depth=0.78

Spatial relationships:
  [1] person is 0.30 units in front of [2] forklift
  [1] person is blocking [2] forklift (closer and overlapping in 2D)
```

---

## The multimodal encoder we designed (and couldn't fully train)

This is the part we're most proud of architecturally and most frustrated we couldn't fully ship.

The idea was to replace the LLM-based decision legs with a proper trained policy. We designed a 5-modality encoder (`action_module/multimodal_fusion.py`):

**1. GAT over the scene graph**
3-layer Graph Attention Network with edge-feature-augmented attention. Node features (15-dim): risk group one-hot + proximity label one-hot + path zone one-hot + numeric (depth score, bbox area, 3D distance). Edge features (6-dim): relative position encoding + distance + blocking flag. Multi-head attention, scatter-softmax per destination node. The GAT was specifically designed to reason about *relational* structure — not just "there's a person" but "there's a person blocking a forklift in the robot's path corridor." That distinction matters.

**2. ViT-S/16 on the segmented image overlay**
MAE-pretrained via `timm`. 224×224 SAM2 overlay showing objects colored by risk group. Global average pool → linear projection.

**3. ResNet-50 + RoI Align on detection crops**
Per-object feature extraction with RoI Align. Aggregated via adaptive pooling. Falls back to the segmented image if no crop available.

**4. Dilated ResNet-34 on the depth map**
Dilated convolutions in layer3/4 for multi-scale depth patterns. UNet-style decoder available for pretraining.

**5. PointNet++ on the point cloud**
3-level set abstraction with progressive downsampling. Per-point risk group labels as additional features.

All five (embed_dim,) embeddings + learnable modality positional embeddings → 2-layer TransformerEncoder → global mean pool → final scene embedding → action policy.

We also wrote a full self-supervised pretraining pipeline with 7 objectives:
1. MAE on segmented image (75% masked patches)
2. MAE on bbox-annotated image
3. Scale-invariant depth loss
4. Point-MAE (75% masked tokens + Chamfer decoder)
5. Masked graph reconstruction (BERT-style node/edge masking)
6. InfoNCE grounding loss (align DINO text-grounded regions with image patches, τ=0.07)
7. Geometric consistency (L2 between depth encoder and point cloud encoder)

It all runs. We just had no GPU time to train it to anything meaningful. So in production, NavSense uses the LLM decision legs instead. The whole encoder is sitting there initialized but untrained — that's the first thing we fix next.

---

## Three independent decision legs

**1. Scene graph LLM** (`llm_module/llm.py`)

Serialized scene graph + `SAFETY_RULES.md` → GPT-4o-mini → `{action, confidence, reasoning}`. Pure symbolic reasoning over the graph structure.

**2. LLMActor** (`llm_action_module/actor.py`)

Richer context: scene graph text + depth stats + point cloud bounding box + SAM2 overlay PNG + depth heatmap PNG → GPT → full probability distribution:

```json
{
  "action": "STOP",
  "confidence": 0.87,
  "probabilities": {"STOP": 0.87, "SLOW": 0.11, "CONTINUE": 0.02},
  "entropy": 0.52,
  "reasoning": "Worker in CENTER path at CLOSE range, forklift approaching"
}
```

Shannon entropy (`H = -Σ p_i log₂ p_i`, range 0–1.585 bits) flags uncertain frames for human review. High-entropy outputs get appended to `llm_actor_entropy.json`.

**3. Graph classifier** (`action_module/`)

43-dimensional feature vector from the scene graph → GradientBoosting classifier. Falls back to rule-based when untrained (which is currently always — see above re: no GPU/data).

**Arbitration** (`decision_module/decider.py`): safest option wins. `STOP > SLOW > CONTINUE`. When legs disagree, we don't average — we pick the most conservative. The web UI shows live probability bars and entropy per source.

---

## Active learning loop

```
Image → Pipeline → Verdict
  → Human clicks "wrong" in the UI
  → Correction saved to human_corrections.json
  → python -m llm_action_module.rule_updater
  → GPT proposes edits to SAFETY_RULES.md (backup kept)
  → Next run uses the updated policy
```

```bash
python -m llm_action_module.rule_updater --dry-run   # preview first
python -m llm_action_module.rule_updater             # apply
```

---

## Install

```bash
git clone https://github.com/alerest285/ETHackers.git
cd ETHackers

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# PyTorch first — https://pytorch.org
# Mac (MPS):   pip install torch torchvision
# Linux/CUDA:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU only:    pip install torch torchvision

pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart
```

Models auto-download on first run (~1.2 GB total in `~/.cache/huggingface/`):

| Model | Size |
|---|---|
| Depth Anything V2 Small | ~100 MB |
| Grounding DINO Base | ~680 MB |
| SAM 2 Hiera Base+ | ~380 MB |

---

## API keys

```bash
export OPENAI_API_KEY="sk-..."         # GPT-4o-mini — prompt gen, depth, consistency, decision
export ANTHROPIC_API_KEY="sk-ant-..."  # Claude Haiku — LLMActor + edge rationalization
```

No keys → system falls back to rule-based decisions. It runs, reasoning quality tanks.

---

## Run the web UI

```bash
python app.py
# open http://localhost:7860
```

- Upload one or many images. Multiple images run in parallel with a live queue strip.
- Every stage streams in real time — DINO boxes, SAM masks, depth heatmap, fused depth, all cross-fading as they complete.
- When 3D lift finishes the compositor becomes a Three.js point cloud. Drag to orbit, scroll to zoom, click an object to see its depth, risk group, and 3D position.
- Right sidebar: Entities, Scene graph, Understanding — each expands on hover.
- Understanding panel typewrites the LLM's reasoning then shows probability bars + entropy across all three legs.
- Result modal: verdict + confidence + reasoning + correction buttons. Corrections go to `human_corrections.json`.

```bash
uvicorn app:app --host 0.0.0.0 --port 8080          # different port
uvicorn app:app --host 0.0.0.0 --port 7860 --reload  # dev mode
```

---

## Run batch mode

```bash
python src/pipeline.py                        # 5 random images
python src/pipeline.py --n 50 --seed 42       # reproducible
python src/pipeline.py --n 100 --conf 0.30    # tighter detection
python src/pipeline.py --n 100 --learn auto   # auto-classify unknowns via GPT
```

| Flag | Default | What it does |
|---|---|---|
| `--n` | 5 | Images to process |
| `--seed` | None | Random seed |
| `--conf` | 0.25 | Grounding DINO confidence threshold |
| `--learn` | off | Unknown labels: `off` / `auto` (GPT) / `interactive` (stdin) |

LLMActor-only pipeline (faster, skips multimodal fusion):
```bash
python src/llmactor_pipeline.py --n 20
```

---

## GPU / HF Inference API

### Local GPU

All models move to CUDA automatically if available. Rough numbers:

| Stage | CPU | GPU (B200) |
|---|---|---|
| Grounding DINO | ~25s | ~1.2s |
| SAM 2 (5 boxes) | ~18s | ~0.8s |
| Depth Anything V2 Small | ~8s | ~0.3s |
| **End-to-end** | **~60–90s** | **~3–6s** |

### HF Inference API (no GPU)

```bash
export USE_HF_API=1
export HF_TOKEN="hf_..."
python app.py
```

Routes Grounding DINO and Depth Anything to HF hosted endpoints. SAM 2 stays local.

```bash
export HF_GDINO_MODEL=IDEA-Research/grounding-dino-tiny   # smaller/faster
export HF_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf  # better quality
```

---

## Deploy to a GPU server

The repo ships a Dockerfile. Recommended: 4 vCPU, 16 GB RAM, 1× B200, 20 GB ephemeral storage.

1. New service from GitHub (`main`), build mode Dockerfile
2. Expose port **7860**
3. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
4. CMD defaults to `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1`

Private tunnel via Northflank CLI:
```bash
northflank forward service --projectId hackathon --serviceId <your-service> --port 7860
```

---

## Output layout

```
data/pipeline_output/
├── detections/              # per-image detection JSON with fused depth
├── overlays/                # SAM mask overlay PNGs
├── depth_maps/              # colorized depth PNGs
├── point_clouds/            # 3D renders (top-down + perspective)
├── scene_graphs/            # nodes/edges JSON + LLM-readable text
├── llm/                     # final decision {action, confidence, reasoning}
├── discarded/               # rejected by hazard filter
├── llm_actor_entropy.json   # {stem, entropy, action, probabilities} per image
└── human_corrections.json   # UI feedback for the rule updater
```

---

## Roadmap

The architecture is done. The hard part — getting all of this designed and wired together in 20 hours — is behind us. Here's what we'd actually do with more time:

**Next 2 weeks**
- Train the GAT encoder on the THEKER dataset. The architecture is done, we just need GPU time. Relational reasoning over the scene graph is exactly where we'd expect the biggest lift.
- Run the full MultimodalFusion SSL pretraining. 7 objectives, ~24h on a decent GPU. This is what replaces the LLM decision legs with a proper trained policy.
- Fine-tune Depth Anything V2 on warehouse/industrial scenes. The base model is trained on general images; shiny floors and forklifts are a distribution shift. Even a few thousand labeled frames should noticeably help.

**Next 2 months**
- Replace GPT-4o-mini with a local fine-tuned smaller model (Mistral 7B or similar) for the reasoning stages. Hosted LLMs are fine for a hackathon but the latency and cost make them non-viable for real deployment.
- Add temporal linking for video input. The scene graph structure already supports it; we'd need to track objects across frames and propagate depth estimates.
- Proper evaluation + ablation study. What does each fusion signal actually contribute? How much does the consistency check help? How fast does entropy drop with human corrections? We have the logging infrastructure, we just haven't run the numbers.
- Train the graph classifier on real labeled data from the correction loop. Right now it's rule-based because we have no training examples.

**If we keep going**
- Deploy on an actual mobile robot and test in a real lab environment.
- Release pretrained encoder weights so others can use NavSense as a foundation.
- Stereo depth as a fifth fusion signal — the architecture already supports dropping it in.

---

## What's actually flaky / what to expect

- **Depth Anything makes mistakes on weird industrial objects.** The 4-signal fusion helps, but expect occasional wrong proximity estimates.
- **The multimodal encoder is untrained.** LLMs are doing the heavy lifting for decisions right now.
- **CPU is slow.** ~60–90s per image end-to-end. On a GPU it's seconds.
- **This is baseline performance with zero fine-tuning.** We used every pretrained model as-is. What you're seeing is the floor, not the ceiling.

---

## Troubleshooting

**`transformers has no attribute zero_shot_object_detection`**
`huggingface_hub` is out of date: `pip install -U huggingface_hub`

**`TypeError: Object of type ndarray is not JSON serializable`**
A SAM mask leaked into an event payload. The `_SafeEncoder` in `app.py` handles this — make sure you're on the latest commit.

**`depth_score` is None downstream**
`app.py` sanitizes to `0.5` before `lift_3d`. If you call `SceneGraphBuilder.process` directly, do the same.

**Grounding DINO / SAM fail on startup**
Newer `transformers` renamed some kwargs. Upgrade: `pip install -U "transformers>=4.46"`

**Container OOMs on cold start**
~1.2 GB downloads on first run. Use at least 16 GB RAM + 20 GB ephemeral storage. OOMs are host RAM, not VRAM.

**Port conflict**
`uvicorn app:app --port 8080` or `lsof -i :7860`

**CUDA not detected**
`python -c "import torch; print(torch.cuda.is_available())"` — if False, reinstall PyTorch with the right index URL from pytorch.org.

---

*Built in Barcelona (with love), April 2026.*
