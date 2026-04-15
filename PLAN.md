# THEKER Hackathon: Perception-to-Decision Pipeline

## Context

**Challenge:** Build an autonomous robot navigation system that classifies industrial scenes as STOP / SLOW / CONTINUE, with confidence scores, reasoning, and bounding-box detections.

**Scoring:** 80% decision quality (50% accuracy, 20% confidence calibration, 10% reasoning) + 20% detection F1.

**Team:** 3 people (ML-experienced). Deployment target: **Northflank** (single Docker container, GPU job).

**Key constraints:**
- No action labels exist — decision logic must be designed from scratch
- Test images have no annotations — must detect objects ourselves
- 27 raw, messy category labels that need cleaning/grouping
- Images need to be downloaded to Northflank volume

---

## Architecture

```
Raw Image
    │
    ▼
┌─────────────────────────────────┐
│ Phase 1: Grounding DINO + SAM   │  Text prompts: 5 semantic groups
│  → bbox per object + pixel mask │  GroundingDINO → boxes → SAM2 → masks
└────────────┬────────────────────┘
             │ {risk_group, bbox, mask, score}
             ▼
┌─────────────────────────────────┐
│ Phase 2: Triple-Fused Depth     │  Per segmented object:
│  Input 1: DepthAnything V2      │  → depth_da   (monocular depth map, median in mask)
│  Input 2: Real-world size ref   │  → depth_rw   (known object height + focal estimate)
│  Input 3: Relative bbox area    │  → depth_area (bbox_area / frame_area, inverted)
│  → weighted fusion → depth_score│  Weights: 0.6 / 0.3 / 0.1 (env-var tunable)
└────────────┬────────────────────┘
             │ {depth_score, proximity_label, path_zone}
             ▼
┌─────────────────────────────────┐
│ Phase 3: 3D Lift → Scene Graph  │  Unproject depth map → point cloud (pinhole model)
│  Per-object 3D cluster          │  Extract 3D centroid + extent per mask
│  → spatial relations            │  Edges: distance, relative position, blocking
│  → serialised scene graph text  │  Output: structured text for LLM
└────────────┬────────────────────┘
             │ scene graph (text) + detections
             ▼
┌─────────────────────────────────┐
│ Phase 4: LLM Decision Engine    │  Rule engine (deterministic, always runs first)
│  Rule engine + LLM backend      │  + pluggable LLM (Qwen2.5-VL / LLaMA / GPT-4o)
│  → STOP / SLOW / CONTINUE       │  Ensemble: agree → avg conf; disagree → conservative
└────────────┬────────────────────┘
             ▼
      Submission JSON
```

---

## 5 Semantic Risk Groups (Ontology)

Maps the 27 raw categories → 5 groups with risk scores:

| Group | Risk Level | Score | Raw Categories |
|-------|-----------|-------|----------------|
| HUMAN | Critical | 5 | person, head, hat, helmet |
| VEHICLE | High | 4 | forklift, car, truck, bus, motorcycle, bicycle, train |
| OBSTACLE | Medium | 3 | barrel, Barrel, crate, box, Box, Container, suitcase, Suitcase, handcart, ladder, Ladder, chair |
| SAFETY_MARKER | Low | 2 | cone, Traffic sign, Stop sign, Traffic light |
| BACKGROUND | None | 1 | undetected / nothing matched |

**Note on duplicates:** box/Box, barrel/Barrel, ladder/Ladder, suitcase/Suitcase are the same object
with inconsistent casing — normalised at ontology load time (lowercase + strip).

---

## Phase 0: Setup

### 0.1 Install dependencies
```bash
pip install -r requirements.txt
```

### 0.2 Download dataset
```
data/
  challenge/data/
    images/train/    # ~17,500 images
    images/val/      # ~3,800 images
    images/test/     # ~3,800 images
    annotations/train.json   ✅ present
    annotations/val.json     ✅ present
    annotations/test.json    ✅ present
```

### 0.3 Project structure
```
ETHackers/
  segment_module/
    grounding_sam.py        # Phase 1: Grounding DINO + SAM2 segmentation  ← NEW
    visualize.py            # Bounding box / mask visualisation (reused)
  depth_module/
    depth_module.py         # DepthAnything V2 wrapper (reused)
    fused_depth.py          # Phase 2: Triple-fused depth per object        ← NEW
    lift_3d.py              # Phase 3: 3D lift → point cloud → scene graph  ← NEW
    run_depth.py            # CLI runner (reused)
    heatmap.py              # Depth heatmap visualisation (reused)
  filtering_module/
    filter_images.py        # Drop images with no relevant detections (reused)
  llm_module/
    llm_backend.py          # Abstract LLM backend interface                ← NEW
    decision_engine.py      # Phase 4: Rule engine + LLM ensemble (updated)
  data/
    challenge/data/         # annotations + images
  outputs/
    detections/             # Grounding DINO + SAM results
    filtered/               # filtered/removed image ID lists
    scene_graphs/           # Per-image scene graph JSON + text
    decisions.json          # Final STOP/SLOW/CONTINUE per image
  run_pipeline.py           # End-to-end orchestrator                       ← NEW
  requirements.txt
  Dockerfile
```

---

## Phase 1: Grounding DINO + SAM Segmentation

**File:** `segment_module/grounding_sam.py`

**Models:**
- **GroundingDINO:** `IDEA-Research/grounding-dino-tiny` (~700MB, HuggingFace)
- **SAM2:** `facebook/sam2-hiera-small` (~350MB)

**How it works:**
1. Prompt GroundingDINO with the 5 semantic group descriptions:
   ```
   "person . vehicle . forklift . obstacle . cone . safety marker"
   ```
2. GroundingDINO returns bounding boxes + matched text phrases
3. Map matched phrases → risk group via keyword lookup
4. Feed boxes to SAM2 (box-prompted mode) → pixel-level mask per object
5. Apply NMS to remove overlapping detections (IoU threshold 0.5)

**Output per detection:**
```python
{
    "risk_group":  str,          # HUMAN | VEHICLE | OBSTACLE | SAFETY_MARKER | BACKGROUND
    "label":       str,          # matched text phrase from GroundingDINO
    "bbox":        [x, y, w, h], # pixels, top-left origin
    "mask":        np.ndarray,   # (H, W) bool array
    "score":       float,        # GroundingDINO confidence
}
```

**Key detail:** Use score threshold 0.15–0.20 (lower than YOLO's 0.25) — open-vocab models tend to be underconfident on industrial objects.

---

## Phase 2: Triple-Fused Depth

**File:** `depth_module/fused_depth.py`

Computes three independent depth estimates per object, then fuses them into one depth score.

### Input 1 — DepthAnything V2
- Run `DepthModule.infer_depth(image_bgr)` → raw depth map (H×W)
- Extract median depth value inside the SAM **mask** (more accurate than bbox rectangle)
- Min-max normalise across all objects in the image → `depth_da ∈ [0,1]`

### Input 2 — Real-world size reference
```python
REFERENCE_HEIGHTS_M = {
    "HUMAN":         1.70,   # average person
    "VEHICLE":       2.50,   # forklift / truck
    "OBSTACLE":      0.90,   # barrel / crate average
    "SAFETY_MARKER": 0.75,   # cone / sign
}
# Focal length estimate (no calibration available):
focal_px = max(img_W, img_H) * 0.8

# Metric distance estimate:
distance_m = (REFERENCE_HEIGHTS_M[group] * focal_px) / bbox_height_px

# Normalise across all objects in image → depth_rw ∈ [0,1]
```
Falls back to `depth_da` if the group has no reference height.

### Input 3 — Relative bbox area proxy
```python
area_ratio = (bbox_w * bbox_h) / (img_W * img_H)
MAX_AREA_RATIO = 0.25   # object covering 25% of frame = very close
depth_area = 1.0 - min(area_ratio / MAX_AREA_RATIO, 1.0)
# Large bbox → small depth_area (close); small bbox → large depth_area (far)
```

### Fusion
```python
# Default weights (tunable via env vars DEPTH_W_DA / DEPTH_W_RW / DEPTH_W_AREA):
depth_score = 0.6 * depth_da + 0.3 * depth_rw + 0.1 * depth_area
```

### Proximity + path zone assignment
```python
# Proximity:
CLOSE  if depth_score <= 0.35
MEDIUM if depth_score <= 0.65
FAR    otherwise

# Path zone (middle third of image = robot's forward path):
CENTER     if img_W/3 <= bbox_cx <= 2*img_W/3
PERIPHERAL otherwise
```

**Output per object adds:**
```python
{
    "depth_score":     float,   # fused [0,1], 0=closest
    "depth_da":        float,   # DepthAnything contribution
    "depth_rw":        float,   # real-world size contribution
    "depth_area":      float,   # bbox area contribution
    "proximity_label": str,     # CLOSE | MEDIUM | FAR
    "path_zone":       str,     # CENTER | PERIPHERAL
}
```

---

## Phase 3: 3D Lift → Scene Graph

**File:** `depth_module/lift_3d.py`

### Step 1 — 3D Lift (pinhole camera model)
```python
# Estimated intrinsics (no calibration available):
fx = fy = max(W, H) * 0.8
cx, cy  = W / 2, H / 2

# Unproject each pixel (u, v) with depth value d:
X = (u - cx) * d / fx   # horizontal
Y = (v - cy) * d / fy   # vertical
Z = d                    # forward (depth axis)
```
Sparse-sampled every 4th pixel for efficiency. Output: `(N, 3)` point cloud array.

### Step 2 — Object clusters
For each detected object, extract 3D points inside its SAM mask and compute:
- `centroid_3d`: mean (X, Y, Z) of points in mask
- `nearest_z`: minimum Z (closest point to camera)
- `extent_3d`: 3D bounding box dimensions

### Step 3 — Scene Graph (nodes + edges)
**Nodes** — one per object:
```python
{
    "id":              int,
    "risk_group":      str,
    "label":           str,
    "score":           float,
    "centroid_3d":     [x, y, z],   # metres (approximate)
    "nearest_z":       float,
    "depth_score":     float,
    "proximity_label": str,
    "path_zone":       str,
}
```

**Edges** — for every pair (A, B):
```python
{
    "from":              int,    # node id A
    "to":                int,    # node id B
    "distance_3d":       float,  # Euclidean between centroids
    "relative_position": str,    # "A in front of B" | "A behind B" | "A left of B" | "A right of B"
    "blocking":          bool,   # True if A.Z < B.Z and their bboxes overlap in 2D
}
```

### Step 4 — Text serialisation (fed to LLM)
```
Scene (3 objects):
  [1] HUMAN 'person'        — CLOSE,  CENTER path, depth=0.12, at (0.1, 0.0, 1.8m)
  [2] VEHICLE 'forklift'    — MEDIUM, PERIPHERAL,  depth=0.41, at (1.2, 0.0, 3.5m)
  [3] SAFETY_MARKER 'cone'  — FAR,    CENTER path,  depth=0.72, at (-0.1, 0.0, 5.2m)

Spatial relationships:
  [1] person is 1.9m in front of [2] forklift
  [1] person is blocking the robot's view of [3] cone
  [2] forklift is 1.7m to the right of [1] person
```

**Saved to:** `outputs/scene_graphs/{image_id}.json` (machine-readable) and passed as text to the LLM.

---

## Phase 4: LLM Decision Engine

**Files:** `llm_module/llm_backend.py`, `llm_module/decision_engine.py`

### Rule Engine (always runs — deterministic safety net)

Priority-ordered rules applied before any LLM call:

```
STOP    — HUMAN at CLOSE proximity OR in CENTER path zone
STOP    — VEHICLE at CLOSE AND in CENTER path zone
SLOW    — Any HUMAN present (not triggering STOP)
SLOW    — Any VEHICLE present (not triggering STOP)
SLOW    — Any OBSTACLE in CENTER path zone
SLOW    — Any SAFETY_MARKER present
SLOW    — Any OBSTACLE off-path
CONTINUE — Scene empty or only BACKGROUND
```

### LLM Backend (pluggable)

Abstract interface so any model can be swapped in without changing the engine:

```python
class LLMBackend(ABC):
    def query(self, scene_graph_text: str, image_path: str | None) -> dict | None:
        """Returns {action, confidence, reasoning} or None on failure."""
```

Available backends (select via `--backend` flag or `LLM_BACKEND` env var):

| Backend | Model | Input | Notes |
|---------|-------|-------|-------|
| `qwen_vl` | Qwen2.5-VL-7B-Instruct | Image + scene graph text | Multimodal, local GPU |
| `llama` | LLaMA 3 8B Instruct | Scene graph text only | Text-only, local GPU |
| `gpt` | GPT-4o | Scene graph text only | API, json_object mode |
| `rules` | — | — | Rule engine only, no LLM |

### Ensemble
```
Agreement  → average confidence, use LLM reasoning
Disagreement → conservative action wins (STOP > SLOW > CONTINUE),
               confidence reduced 10%, disagreement noted in reasoning
LLM failure → rule engine result used silently (rules_fallback)
```

### LLM Prompt (scene graph → structured JSON)
The LLM receives:
1. System prompt with ontology + decision rules + confidence calibration guide
2. The scene graph text (Phase 3 output)
3. Optionally: the annotated image (multimodal backends)

Output enforced as JSON: `{"action": "STOP"|"SLOW"|"CONTINUE", "confidence": float, "reasoning": str}`

---

## End-to-End Runner

**File:** `run_pipeline.py`

```bash
python run_pipeline.py \
    --split      test \
    --data-dir   data/challenge/data \
    --output-dir outputs/ \
    --backend    qwen_vl \
    --max-images 100        # omit for full run
```

Runs all 4 phases in sequence, writes `outputs/decisions.json`.

---

## Dependencies

```
# Already in requirements.txt:
transformers, huggingface_hub, ultralytics, pillow, opencv-python,
numpy, scipy, tqdm, psutil, openai

# New additions:
groundingdino-py          # Grounding DINO (or via HF transformers AutoModel)
sam2                      # pip install git+https://github.com/facebookresearch/sam2
qwen-vl-utils             # Qwen2.5-VL image pre-processing
```

---

## Northflank Deployment

**Single container:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV LLM_BACKEND=qwen_vl
ENV DEPTH_W_DA=0.6
ENV DEPTH_W_RW=0.3
ENV DEPTH_W_AREA=0.1
CMD ["python", "run_pipeline.py", "--split", "test", "--data-dir", "data/challenge/data"]
```

**VRAM strategy (T4 = 16GB):**
- Load models sequentially, offload each after use
- GroundingDINO-tiny (~700MB) + SAM2-small (~350MB) + DepthAnything-small (~400MB) = ~1.5GB
- LLM: Qwen2.5-VL-7B in 4-bit quant ≈ 4–5GB → total ≈ 6–7GB active at once
- A100 variant: load all simultaneously

**Env vars:**
- `LLM_BACKEND` — qwen_vl | llama | gpt | rules
- `OPENAI_API_KEY` — required for gpt backend
- `DEPTH_W_DA`, `DEPTH_W_RW`, `DEPTH_W_AREA` — depth fusion weights
- `CONF_THRESHOLD` — detection confidence threshold (default 0.15)

---

## Team of 3 — Work Split

| Person | Responsibility |
|--------|---------------|
| Person 1 | `grounding_sam.py` (Phase 1) + `visualize.py` updates |
| Person 2 | `fused_depth.py` + `lift_3d.py` (Phases 2 & 3) |
| Person 3 | `llm_backend.py` + `decision_engine.py` + `run_pipeline.py` (Phase 4 + integration) |

Converge on end-to-end test with 20 images before full run.

---

## Verification Checkpoints

1. **Segmentation:** `grounding_sam.py` on 10 images → each returns masks with correct `risk_group` labels
2. **Fused depth:** `fused_depth.py` on 10 images → print all 3 depth inputs + fused score; CLOSE objects have `depth_score < 0.35`
3. **Scene graph:** `lift_3d.py` on 3 images → print scene graph text; spatial relations match visual inspection
4. **Decision engine:** `--backend rules` on 20 images → reasonable STOP/SLOW/CONTINUE distribution; swap LLM backend and compare
5. **End-to-end:** `run_pipeline.py --max-images 50 --backend rules` → `outputs/decisions.json` has one entry per image with valid `action`/`confidence`/`reasoning`
6. **Submission:** Run `data/challenge/starter_kit/validate_submission.py` → zero errors

---

## Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Grounding DINO misses industrial objects with semantic group prompts | Medium | Enrich prompts with synonyms: "person . worker . vehicle . forklift . machinery . barrel . cone" |
| Camera intrinsics unknown → 3D lift approximate | Low | Spatial *relations* are directionally correct; metric precision not needed for STOP/SLOW/CONTINUE |
| VRAM tight on T4 (16GB) with all models | High | Load sequentially + offload; use small/tiny variants; 4-bit LLM quant |
| Real-world size depth meaningless for unknown object types | Low | Falls back to DepthAnything-only (`depth_rw = depth_da`) |
| bbox area proxy unreliable for large distant objects | Low | Weight = 0.1 only; dominated by DepthAnything |
| LLM hallucinating invalid JSON | Medium | Robust JSON extractor (`_extract_json`); rule engine always provides valid fallback |
