# Safety Rules — Action Decision Engine

Rules for the deterministic rule engine in `decision_engine.py`.  
These run **before** any LLM call and always provide a valid fallback.

---

## Definitions

**Proximity labels** (from fused depth score, 0 = closest):

| Label  | Depth Score    | Approx. distance |
|--------|----------------|------------------|
| CLOSE  | ≤ 0.35         | < ~3 m           |
| MEDIUM | 0.35 – 0.65    | ~3 – 8 m         |
| FAR    | > 0.65         | > ~8 m           |

**Path zones** (horizontal position in frame):

| Zone        | Condition                              |
|-------------|----------------------------------------|
| CENTER      | bbox centre x ∈ [img_W/3, 2·img_W/3] |
| PERIPHERAL  | anything outside center third          |

**Risk groups** (highest → lowest):  
`HUMAN (5) > VEHICLE (4) > OBSTACLE (3) > SAFETY_MARKER (2) > BACKGROUND (1)`

---

## Rules (evaluated top-to-bottom, first match wins)

### STOP

| # | Condition | Confidence | Reasoning hint |
|---|-----------|------------|----------------|
| S1 | HUMAN detected at **CLOSE** proximity | 0.95 | Person within collision range — immediate stop |
| S2 | HUMAN in **CENTER** path zone at any proximity | 0.90 | Person is directly in the robot's forward path |
| S3 | VEHICLE at **CLOSE** proximity AND in **CENTER** zone | 0.92 | Large moving object in immediate path |
| S4 | Multiple HUMANs (≥ 2) anywhere in frame | 0.88 | Crowded scene, unpredictable movement |
| S5 | HUMAN + VEHICLE both present and both **CLOSE** | 0.95 | Maximum hazard — human and vehicle collision risk |
| S6 | Any object with depth_score < 0.15 (extreme proximity) | 0.97 | Object essentially at robot contact distance |

---

### SLOW

| # | Condition | Confidence | Reasoning hint |
|---|-----------|------------|----------------|
| W1 | HUMAN at **MEDIUM** proximity | 0.78 | Person at moderate range, robot must proceed cautiously |
| W2 | HUMAN at **FAR** proximity but in **CENTER** zone | 0.70 | Person ahead, may move closer |
| W3 | VEHICLE at **CLOSE** proximity but **PERIPHERAL** | 0.75 | Vehicle nearby but not directly in path |
| W4 | VEHICLE at **MEDIUM** proximity (any zone) | 0.72 | Vehicle within awareness range |
| W5 | OBSTACLE in **CENTER** zone at **CLOSE** or **MEDIUM** proximity | 0.68 | Physical obstruction in forward path |
| W6 | Multiple OBSTACLEs (≥ 3) spanning > 40% of frame width | 0.65 | Path significantly narrowed by obstacles |
| W7 | SAFETY_MARKER in **CENTER** zone | 0.62 | Active hazard zone indicated — reduce speed |
| W8 | SAFETY_MARKER at **CLOSE** or **MEDIUM** proximity | 0.60 | Safety marker signals nearby hazard |
| W9 | OBSTACLE at **CLOSE** proximity in **PERIPHERAL** zone | 0.58 | Lateral obstacle, risk of collision during turns |
| W10 | Any VEHICLE present (not already triggering STOP) | 0.65 | Vehicle in scene demands reduced speed |

---

### CONTINUE

| # | Condition | Confidence | Reasoning hint |
|---|-----------|------------|----------------|
| C1 | No detections above threshold | 0.82 | Clear scene, no hazards detected |
| C2 | Only BACKGROUND detections | 0.80 | No risk-relevant objects present |
| C3 | Objects only at **FAR** proximity in **PERIPHERAL** zones | 0.75 | Hazards present but distant and off-path |
| C4 | Only SAFETY_MARKERs at **FAR** proximity | 0.70 | Distant markers, informational only |
| C5 | Only OBSTACLEs at **FAR** proximity, **PERIPHERAL** | 0.72 | Obstacles exist but do not threaten current path |

---

## Ensemble with LLM

When the LLM is available, its decision is compared to the rule engine output:

| Outcome | Handling |
|---------|----------|
| **Agreement** | Average the two confidence scores; use LLM reasoning |
| **Disagreement** | Take the **more conservative** action (STOP > SLOW > CONTINUE); reduce confidence by 10%; note disagreement in reasoning |
| **LLM failure** | Use rule engine result silently (`rules_fallback`) |

---

## Confidence Calibration Guidelines

- Never output confidence **1.0** — perfect certainty is epistemically wrong for vision-only systems
- Reduce confidence by **0.05–0.10** when:
  - Detection score for the triggering object is < 0.50
  - Image is low-light, blurry, or heavily occluded
  - Scene graph has fewer than 2 objects (poor context)
- Increase confidence by **0.05** when:
  - Multiple independent rules fire for the same action
  - SAM2 mask quality is high (mask_score > 0.85)
  - Both rule engine and LLM agree

---

## Edge Cases

| Scenario | Rule | Rationale |
|----------|------|-----------|
| Detection with score < 0.25 | Ignore for rule purposes | Too uncertain to act on |
| Object bbox covers > 60% of frame | Treat as CLOSE regardless of depth | Likely a calibration artefact — be conservative |
| HUMAN detected but depth unavailable | Default to SLOW | Fail safe — assume worst case |
| VEHICLE at FAR + PERIPHERAL | CONTINUE (C3) | Distant parked vehicle is not a threat |
| Overlapping HUMAN + VEHICLE bboxes | Apply most severe rule (STOP if either triggers it) | Co-location is inherently high risk |
| Empty scene after filtering | CONTINUE (C1, 0.82) | Nothing to react to |
| All objects at depth_score > 0.80 | CONTINUE — reduce confidence to 0.65 | Scene may be poorly estimated |
