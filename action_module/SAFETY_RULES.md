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

| Risk | Groups |
|------|--------|
| 5    | HUMAN |
| 4    | VEHICLE, ANIMAL |
| 3    | SAFETY_MARKER, UNKNOWN |
| 2    | OBSTACLE, FOOD |
| 1    | BACKGROUND |
| 0    | SURFACE |

**Group notes:**
- **ANIMAL** (risk=4) — living non-human creatures. Same tier as VEHICLE because
  they move unpredictably; apply VEHICLE-tier rules (slow or stop on CLOSE/CENTER).
- **FOOD** (risk=2) — edible items / food debris (apple, banana, wrapper).
  Same tier as OBSTACLE; apply OBSTACLE-tier rules.
- **Puddles / spills / wet floors** are OBSTACLE (risk=2) — physical hazards
  the robot should drive around. The matching *signage* (`wet floor sign`) is
  still SAFETY_MARKER (risk=3).
- **UNKNOWN** (risk=3) is the fail-safe fallback for labels the ontology has
  never seen. It carries SAFETY_MARKER-tier risk — conservative until the
  interactive or LLM classifier promotes it to a concrete group.

**SURFACE** objects (floor, road, parking lot, floor markings, ground) are what the robot
navigates ON. Their `proximity_label` is always `NAVIGABLE` — this is not a hazard signal.
SURFACE detections must **never** trigger STOP or SLOW under any condition.

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

---

### SURFACE (risk=0) — Navigable Ground

SURFACE objects are never hazards. The robot drives on them.

| Rule | Condition | Action |
|------|-----------|--------|
| SRF1 | Any SURFACE detection, any zone | Ignore — no change to action |
| SRF2 | Only SURFACE + BACKGROUND in frame | CONTINUE (same as C2) |

- Floor markings, painted lines, and parking outlines are SURFACE — never treat as OBSTACLE even in CENTER zone.
- `proximity_label = NAVIGABLE` means the detection is a surface, not a threat.

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
| SURFACE detected at any proximity | CONTINUE (SRF1) | Robot is on it — expected |
| Floor marking in CENTER zone | CONTINUE (SRF1) | Painted lines are navigable |
| proximity_label = NAVIGABLE | Skip depth rules entirely | Not a positional hazard |
| risk_group = UNKNOWN | Apply SAFETY_MARKER-tier rules (W7, W8) | Fail-safe: unseen label, conservative until classified |
| UNKNOWN + CENTER + CLOSE | SLOW (confidence 0.60) | Conservative default until classifier resolves it |
| risk_group = ANIMAL | Apply VEHICLE-tier rules (W3, W4, W10) | Unpredictable movement, same hazard profile as vehicles |
| risk_group = FOOD | Apply OBSTACLE-tier rules (W5, W9) | Small floor items, avoid like general debris |
| Puddle / spill in CENTER zone | SLOW (W5 as OBSTACLE) | Slip hazard treated as a physical obstruction |
