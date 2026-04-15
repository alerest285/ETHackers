# THEKER Robotics Hackathon Challenge

## Industrial Robot Navigation: From Perception to Decision

---

## Overview

Autonomous robots in warehouses, factories, and logistics hubs must make safety decisions in real time. They fail not because they cannot detect objects, but because they cannot **interpret what those objects mean, combine signals, and make robust decisions under uncertainty**.

Your challenge: **build a complete pipeline that perceives a scene and decides what a robot should do.**

| | |
|---|---|
| **Input** | Images of industrial environments. For train/val, object annotations are provided. For test, you must detect objects yourself. |
| **Output** | For each test image: a navigation decision (STOP / SLOW / CONTINUE), a confidence score, a reasoning explanation, and your object detections. |

There are no action labels in the dataset. You must reason about risk from the scene content alone.

---

## What You Receive

### Dataset Structure

```
data/
  images/
    train/           # ~17,500 images
    val/             # ~3,800 images
    test/            # ~3,800 images
  annotations/
    train.json       # Bounding boxes + labels (27 raw categories)
    val.json         # Bounding boxes + labels (27 raw categories)
    test.json        # Images only -- NO annotations
  metadata/
    dataset_info.json
```

### Train and Val Annotations

The `train.json` and `val.json` files follow the COCO JSON format:

```json
{
  "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, ...],
  "images": [{"id": 1, "file_name": "xxx.jpg", "width": 1280, "height": 960}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}]
}
```

Each annotation has: `image_id`, `category_id`, and `bbox` in `[x, y, width, height]` format (pixels, top-left origin).

### Test Annotations

**The test set has NO annotations.** `test.json` contains the image list but the `annotations` array is empty. You must run your own object detection on test images.

### About the Labels

The dataset is assembled from **multiple public sources**. Labels are **raw and heterogeneous** -- they have not been cleaned or unified. For example:

| What you see | What it means |
|---|---|
| `"box"` and `"Box"` | Same concept, different casing from different sources |
| `"barrel"` and `"Barrel"` | Same object type, inconsistent naming |
| `"hat"` and `"helmet"` and `"head"` | Different labels from different PPE datasets |
| `"cone"` and `"Traffic sign"` and `"Stop sign"` | Different safety indicators |
| `"car"`, `"truck"`, `"forklift"`, `"bus"` | Individual vehicle types, not grouped |

There are **27 raw categories** in total. Preprocessing and grouping these labels into meaningful categories is **part of the challenge**. You decide the ontology.

---

## The Challenge

This is a **systems challenge**, not a model benchmark. You must:

1. **Understand the data** -- Explore the 27 raw labels. Decide which ones represent similar concepts. Clean and group them into a working ontology.
2. **Build a detector** -- For the test set, you must detect objects yourself. Use pretrained models, fine-tune on the training data, or combine approaches.
3. **Reason about risk** -- Given the objects in a scene, their positions, sizes, and relationships, determine the risk level.
4. **Output a decision** -- For each test image: STOP, SLOW, or CONTINUE.

---

## Action Space

| Action | Meaning | Example Triggers |
|---|---|---|
| **STOP** | Halt immediately. High risk. | Person at close range, vehicle on collision course, full path blockage |
| **SLOW** | Reduce speed. Moderate risk. | Person nearby but not in path, safety markers present, partial obstruction |
| **CONTINUE** | Proceed normally. Low risk. | Clear path, objects far away, no hazards |

---

## Scoring

Your submission is scored on **two dimensions**:

### 1. Decision Quality (80% of total score)

| Component | Weight | Description |
|---|---|---|
| Decision accuracy | 50% | STOP/SLOW/CONTINUE compared to hidden ground truth. Exact match = 1.0, one level off = 0.3, two levels off = 0.0. |
| Confidence calibration | 20% | Correct predictions should have high confidence; incorrect ones should have low confidence. |
| Reasoning quality | 10% | Is the reasoning specific, substantive, and scene-relevant? |

### 2. Detection Quality (20% of total score)

| Component | Weight | Description |
|---|---|---|
| Detection F1 | 20% | Your detections on the test set are compared to hidden ground-truth annotations using IoU >= 0.5. Precision, recall, and F1 are computed per object group. |

Participants who do not submit detections receive 0 for this component but can still score up to 80%.

---

## Submission Format

Your submission is a JSON file with three sections:

```json
{
  "team_name": "your_team",
  "predictions": [
    {
      "image_id": 123,
      "action": "STOP",
      "confidence": 0.92,
      "reasoning": "Person detected at close range near active forklift."
    }
  ],
  "detections": [
    {
      "image_id": 123,
      "category_id": 1,
      "bbox": [100, 200, 50, 120],
      "score": 0.95
    }
  ]
}
```

- **`predictions`** (required): One entry per test image with action, confidence, reasoning.
- **`detections`** (optional but scored): COCO-format bounding box detections on test images.

See `SUBMISSION_FORMAT.md` for the complete specification.

---

## Rules

1. **No action labels exist.** Your system must reason from visual content and annotations.
2. **Test set has no annotations.** You must detect objects yourself.
3. **Labels are messy by design.** Data preprocessing is part of the challenge.
4. **One decision per test image.** Every `image_id` in `test.json` must appear exactly once in `predictions`.
5. **Valid actions only.** `"STOP"`, `"SLOW"`, or `"CONTINUE"` (case-sensitive, uppercase).
6. **Confidence in [0.0, 1.0].** Values outside this range invalidate the entry.
7. **Reasoning is required.** Empty strings are penalized.
8. **No manual labeling.** Automated pipelines only.
9. **External data is allowed.** You may collect, download, or use additional public datasets to train your detector or improve your pipeline. Pretrained models (YOLO, Faster R-CNN, CLIP, VLMs, etc.) are also allowed. However, you may NOT manually label or annotate the test images.

---

## Starter Kit

```bash
cd starter_kit/
pip install -r requirements.txt

# Run the baseline on the validation set
python predict.py --data-dir ../data --split val --output predictions.json

# Self-evaluate locally (approximate, NOT the official scorer)
python evaluate_local.py --predictions predictions.json --annotations ../data/annotations/val.json
```

The baseline is intentionally simple. It scores ~0.59. Strong solutions will preprocess labels, use pretrained detectors, combine multiple AI tools, and reason about spatial context.

---

## Tips for Strong Solutions

- **Start with data exploration.** Print the 27 category names. Look at how labels overlap. Build your own grouping before writing any decision logic.
- **Combine multiple approaches.** Detection model + VLM scene understanding + rule-based safety constraints is a strong pattern.
- **For test detection:** Fine-tune a pretrained detector (YOLO, Faster R-CNN) on the train split, or use a zero-shot VLM.
- **Bring your own data.** You are free to use additional public datasets (COCO, Open Images, Roboflow, etc.) to train or augment your detector. This is encouraged -- the provided training data is intentionally heterogeneous, and supplementing it with cleaner or domain-specific data is a valid strategy.
- **Spatial reasoning matters.** A person far away is different from a person 2 meters ahead. Use bbox size relative to image dimensions.
- **Calibrate confidence.** Honest uncertainty scores better than always outputting 0.99.
- **Write specific reasoning.** "Dangerous" scores poorly. "Forklift detected at close range with worker in adjacent aisle" scores well.
- **Validate before submitting.** A malformed file scores zero.

---

## What Success Looks Like

A strong solution:

- Groups the 27 raw labels into meaningful categories
- Detects objects in test images with reasonable precision
- Reasons about the spatial relationships between detected objects
- Combines multiple signals to make robust decisions
- Handles ambiguity and edge cases gracefully
- Produces structured, specific justifications

This is a **multi-tool AI systems challenge**. The best teams will build pipelines, not single models.

---

Good luck. Build something that keeps robots -- and the people around them -- safe.
