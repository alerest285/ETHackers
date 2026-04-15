"""
Phase 1: Label Ontology

Maps the 27 raw, heterogeneous dataset categories → 5 semantic risk groups.

Groups (ordered by risk, highest first):
  HUMAN          risk=5  person, head, hat, helmet
  VEHICLE        risk=4  forklift, car, truck, bus, motorcycle, bicycle, train
  OBSTACLE       risk=3  barrel, crate, box, container, suitcase, handcart, ladder, chair
  SAFETY_MARKER  risk=2  cone, traffic sign, stop sign, traffic light
  BACKGROUND     risk=1  fallback — nothing matched

Normalization: all lookups are case-insensitive and whitespace-stripped,
so "Box" / "box" / " BOX " all resolve correctly.
"""

# ---------------------------------------------------------------------------
# Raw → group mapping (27 labels from train.json)
# ---------------------------------------------------------------------------

ONTOLOGY: dict[str, str] = {
    # HUMAN
    "person":       "HUMAN",
    "head":         "HUMAN",
    "hat":          "HUMAN",
    "helmet":       "HUMAN",

    # VEHICLE
    "forklift":     "VEHICLE",
    "car":          "VEHICLE",
    "truck":        "VEHICLE",
    "bus":          "VEHICLE",
    "motorcycle":   "VEHICLE",
    "bicycle":      "VEHICLE",
    "train":        "VEHICLE",

    # OBSTACLE
    "barrel":       "OBSTACLE",
    "crate":        "OBSTACLE",
    "box":          "OBSTACLE",
    "Box":          "OBSTACLE",
    "Barrel":          "OBSTACLE",
    "Suitcase":          "OBSTACLE",
    "container":    "OBSTACLE",
    "suitcase":     "OBSTACLE",
    "handcart":     "OBSTACLE",
    "ladder":       "OBSTACLE",
    "chair":        "OBSTACLE",

    # SAFETY_MARKER
    "cone":          "SAFETY_MARKER",
    "traffic sign":  "SAFETY_MARKER",
    "stop sign":     "SAFETY_MARKER",
    "traffic light": "SAFETY_MARKER",
}

# Risk score per group
GROUP_RISK: dict[str, int] = {
    "HUMAN":         5,
    "VEHICLE":       4,
    "OBSTACLE":      3,
    "SAFETY_MARKER": 2,
    "BACKGROUND":    1,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Lowercase + strip so 'Box', 'BOX', ' box ' all match."""
    return name.lower().strip()


def get_risk_group(category_name: str) -> dict:
    """
    Resolve a raw category name to its risk group and score.

    Returns:
        {
            "group":      str,  — e.g. "HUMAN"
            "risk_score": int,  — 1-5
        }

    Unknown labels fall back to BACKGROUND (risk=1).

    Examples:
        get_risk_group("person")   → {"group": "HUMAN",    "risk_score": 5}
        get_risk_group("Box")      → {"group": "OBSTACLE",  "risk_score": 3}
        get_risk_group("unknown")  → {"group": "BACKGROUND","risk_score": 1}
    """
    group = ONTOLOGY.get(normalize(category_name), "BACKGROUND")
    return {"group": group, "risk_score": GROUP_RISK[group]}


def map_detections(detections: list[dict]) -> list[dict]:
    """
    Enrich a list of detection dicts with risk_group and risk_score.

    Each detection must have a "label" key (raw category name).
    Returns copies with "risk_group" and "risk_score" added.
    """
    enriched = []
    for det in detections:
        d = dict(det)
        result = get_risk_group(d.get("label", ""))
        d["risk_group"]  = result["group"]
        d["risk_score"]  = result["risk_score"]
        enriched.append(d)
    return enriched


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    print("=== Ontology verification ===\n")

    # Check all 27 raw categories resolve correctly
    expected = list(ONTOLOGY.keys())
    for raw in expected:
        r = get_risk_group(raw)
        print(f"  {raw:<20s} → {r['group']:<15s} (risk={r['risk_score']})")

    # Check normalization edge cases
    print("\n=== Normalization checks ===\n")
    for test in ["Box", "BOX", " barrel ", "Barrel", "Ladder", "HELMET", "unknown_label"]:
        r = get_risk_group(test)
        print(f"  {test!r:<20s} → {r['group']}")

    # Optionally validate against actual train.json
    ann_path = Path("data/annotations/train.json")
    if ann_path.exists():
        with open(ann_path) as f:
            coco = json.load(f)
        raw_cats = {c["name"] for c in coco["categories"]}
        missing  = [c for c in raw_cats if normalize(c) not in ONTOLOGY]
        print(f"\n=== train.json categories ({len(raw_cats)} total) ===")
        print(f"  Unmapped (→ BACKGROUND): {missing if missing else 'none'}")
    else:
        print("\n(train.json not found — skipping annotation check)")
