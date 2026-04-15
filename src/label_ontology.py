"""
Unified Label Ontology — single source of truth for all label -> risk group mappings.

Risk levels (0–5):
  0  SURFACE        navigable floor — robot drives ON it, never a hazard
  1  BACKGROUND     irrelevant background (sky, wall, generic objects)
  2  SAFETY_MARKER  hazard indicators (cone, sign, barrier, caution tape)
  3  OBSTACLE       static physical obstruction (box, barrel, crate, pallet)
  4  VEHICLE        motorized or large mobile objects (forklift, car, truck)
  5  HUMAN          any person or their presence indicator

  Note: head / helmet / hat / safety vest all map to HUMAN (risk=5) — they
  indicate human proximity and must not form a separate lower-risk class.

Matching strategy (get_risk_group):
  1. Exact match on normalized label
  2. Word-boundary substring scan — longest alias wins (guards "car"≠"cardboard")
  3. Fallback -> BACKGROUND (risk=1)
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Canonical risk scores per group
# ---------------------------------------------------------------------------

GROUP_RISK: dict[str, int] = {
    "SURFACE":       0,
    "BACKGROUND":    1,
    "SAFETY_MARKER": 2,
    "OBSTACLE":      3,
    "VEHICLE":       4,
    "HUMAN":         5,
}

# ---------------------------------------------------------------------------
# Alias groups
# Each group maps to a list of alias buckets.
# Within a bucket, the FIRST entry is the canonical/preferred term.
# All entries in a bucket (and all buckets in a group) share the same risk class.
# ---------------------------------------------------------------------------

ALIAS_GROUPS: dict[str, list[list[str]]] = {

    # ── SURFACE (risk 0) — robot drives ON these ──────────────────────────────
    "SURFACE": [
        ["floor", "indoor floor", "ground floor", "concrete floor", "tiled floor"],
        ["ground", "ground level", "outdoor ground", "dirt ground", "gravel"],
        ["road", "roadway", "road surface", "driving lane"],
        ["pavement", "paved surface", "paving", "sidewalk"],
        ["asphalt", "tarmac", "blacktop"],
        ["parking lot", "car park", "parking area", "parking space", "parking spot", "parking zone"],
        ["path", "walkway", "footpath", "pedestrian path", "corridor floor"],
        ["tile", "floor tile"],
        ["floor marking", "floor line", "painted line", "lane marking",
         "painted marking", "floor stripe", "painted stripe"],
        ["lane", "traffic lane"],
        ["carpet", "floor carpet"],
        ["dirt path", "dirt road"],
    ],

    # ── BACKGROUND (risk 1) — irrelevant structures ───────────────────────────
    "BACKGROUND": [
        ["wall", "concrete wall", "brick wall", "partition"],
        ["ceiling"],
        ["sky"],
        ["background"],
        ["window", "glass window"],
        ["column", "support column"],
    ],

    # ── SAFETY_MARKER (risk 2) — hazard indicators ────────────────────────────
    "SAFETY_MARKER": [
        ["cone", "traffic cone", "safety cone", "orange cone", "road cone", "delineator"],
        ["traffic sign", "road sign", "warning sign", "hazard sign", "safety sign"],
        ["stop sign", "stop board"],
        ["traffic light", "traffic signal", "stoplight"],
        ["wet floor sign", "caution wet floor", "wet floor warning", "caution sign"],
        ["barrier", "safety barrier", "jersey barrier", "water barrier"],
        ["caution tape", "warning tape", "hazard tape", "barrier tape",
         "yellow tape", "danger tape", "safety tape"],
        ["bollard", "safety bollard"],
        ["fire extinguisher", "extinguisher"],
        ["speed bump", "speed hump", "road bump"],
        ["emergency exit sign", "exit sign", "fire exit"],
        ["safety line", "painted safety line", "yellow safety line"],
    ],

    # ── OBSTACLE (risk 3) — static physical obstructions ─────────────────────
    "OBSTACLE": [
        ["barrel", "oil barrel", "metal barrel", "drum barrel"],
        ["drum", "chemical drum", "plastic drum", "oil drum"],
        ["crate", "wooden crate", "plastic crate", "storage crate"],
        ["box", "cardboard box", "shipping box", "storage box", "carton", "package"],
        ["container", "shipping container", "storage container"],
        ["suitcase", "luggage", "travel bag"],
        ["handcart", "hand cart", "hand truck", "dolly", "trolley"],
        ["pallet", "wooden pallet", "plastic pallet", "euro pallet", "pallet stack"],
        ["shelf", "shelving unit", "storage shelf"],
        ["rack", "storage rack", "metal rack", "shelving rack"],
        ["ladder", "step ladder", "extension ladder", "scaffold"],
        ["pipe", "metal pipe", "plastic pipe", "hose"],
        ["chair", "office chair", "folding chair"],
        ["bag", "sandbag", "garbage bag", "large bag"],
        ["machine", "industrial machine"],
        ["cable", "wire", "cable bundle"],
        ["table", "work table", "desk"],
    ],

    # ── VEHICLE (risk 4) — motorized or large mobile objects ─────────────────
    "VEHICLE": [
        ["forklift", "fork lift", "fork-lift", "forklift truck", "reach truck", "stacker"],
        ["pallet jack", "pallet truck", "electric pallet jack", "powered pallet truck"],
        ["car", "passenger car", "automobile", "sedan", "hatchback"],
        ["truck", "lorry", "delivery truck", "heavy truck", "semi truck", "flatbed truck"],
        ["van", "cargo van", "delivery van", "minivan"],
        ["bus", "minibus", "shuttle bus", "transit bus", "coach"],
        ["motorcycle", "motorbike"],
        ["scooter", "moped", "electric scooter"],
        ["bicycle", "bike", "e-bike", "electric bike", "pushbike"],
        ["train", "locomotive", "freight train", "rail vehicle", "tram"],
        ["tow truck", "recovery truck", "wrecker"],
        ["cart", "motorized cart", "electric cart", "golf cart"],
        ["aerial lift", "scissor lift", "boom lift", "cherry picker"],
        ["agv", "automated guided vehicle", "autonomous vehicle"],
        ["floor sweeper", "street sweeper", "cleaning vehicle"],
        ["vehicle"],   # generic fallback within VEHICLE group
    ],

    # ── HUMAN (risk 5) — persons AND proximity indicators ────────────────────
    "HUMAN": [
        ["person", "people", "human", "individual"],
        ["worker", "factory worker", "warehouse worker", "employee", "staff"],
        ["pedestrian", "bystander", "passerby"],
        ["operator", "machine operator", "equipment operator"],
        ["technician", "maintenance technician", "engineer"],
        ["driver", "truck driver", "forklift driver", "vehicle operator"],
        ["guard", "security guard", "security officer"],
        ["child", "kid"],
        # Body parts -> someone is present
        ["head", "face"],
        # Protective gear -> worn by a person -> treat as HUMAN proximity indicator
        ["hard hat", "hardhat", "hard-hat", "safety helmet", "construction helmet"],
        ["helmet", "bike helmet", "motorcycle helmet", "protective helmet"],
        ["hat", "cap", "safety cap"],
        ["safety vest", "reflective vest", "hi-vis vest", "high-visibility vest",
         "high vis vest", "orange vest", "hi vis vest"],
        ["safety jacket", "hi-vis jacket", "reflective jacket", "high-vis jacket"],
    ],
}

# ---------------------------------------------------------------------------
# Build fast lookup structures (computed once at import time)
# ---------------------------------------------------------------------------

# _EXACT: normalized alias -> group
_EXACT: dict[str, str] = {}

# _CANONICAL: normalized alias -> preferred term (first in its bucket)
_CANONICAL: dict[str, str] = {}

for _group, _buckets in ALIAS_GROUPS.items():
    for _bucket in _buckets:
        _canon = _bucket[0].lower().strip()
        for _alias in _bucket:
            _key = _alias.lower().strip()
            _EXACT[_key] = _group
            _CANONICAL[_key] = _canon

# Sorted by descending phrase length so longer (more specific) matches win.
# Example: "pallet jack" (11 chars) is tried before "pallet" (6 chars).
_SUBSTR_SORTED: list[tuple[str, str]] = sorted(
    _EXACT.items(), key=lambda kv: len(kv[0]), reverse=True
)


def _word_boundary_match(phrase: str, text: str) -> bool:
    """True if `phrase` appears in `text` as a complete word sequence.

    Uses negative lookaround to reject partial token matches:
      "car" in "cardboard box" -> False  (followed by 'd')
      "car" in "old car here"  -> True
    """
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Lowercase + strip so 'Box', 'BOX', ' box ' all match."""
    return name.lower().strip()


def get_risk_group(category_name: str) -> dict:
    """
    Resolve any raw detection label to its risk group and score.

    Resolution order:
      1. Exact match (normalized)
      2. Word-boundary substring scan — longest alias tested first
      3. Fallback -> BACKGROUND (risk=1)

    Returns:
        {"group": str, "risk_score": int}

    Examples:
        get_risk_group("person")       -> {"group": "HUMAN",         "risk_score": 5}
        get_risk_group("hard hat")     -> {"group": "HUMAN",         "risk_score": 5}
        get_risk_group("helmet")       -> {"group": "HUMAN",         "risk_score": 5}
        get_risk_group("parking lot")  -> {"group": "SURFACE",       "risk_score": 0}
        get_risk_group("floor")        -> {"group": "SURFACE",       "risk_score": 0}
        get_risk_group("cardboard box")-> {"group": "OBSTACLE",      "risk_score": 3}
        get_risk_group("pallet jack")  -> {"group": "VEHICLE",       "risk_score": 4}
        get_risk_group("unknown")      -> {"group": "BACKGROUND",    "risk_score": 1}
    """
    norm = normalize(category_name)

    # Step 1 — exact
    if norm in _EXACT:
        group = _EXACT[norm]
        return {"group": group, "risk_score": GROUP_RISK[group]}

    # Step 2 — word-boundary substring (longest alias wins)
    for phrase, group in _SUBSTR_SORTED:
        if _word_boundary_match(phrase, norm):
            return {"group": group, "risk_score": GROUP_RISK[group]}

    # Step 3 — fallback
    return {"group": "BACKGROUND", "risk_score": GROUP_RISK["BACKGROUND"]}


def get_canonical(category_name: str) -> str:
    """Return the canonical/preferred term for a label (best-effort).

    Example: "hard-hat" -> "hard hat", "motorcycle helmet" -> "helmet"
    """
    norm = normalize(category_name)
    if norm in _CANONICAL:
        return _CANONICAL[norm]
    for phrase, _ in _SUBSTR_SORTED:
        if _word_boundary_match(phrase, norm):
            return _CANONICAL.get(phrase, norm)
    return norm


def map_detections(detections: list[dict]) -> list[dict]:
    """Enrich a list of detection dicts with risk_group and risk_score.

    Each detection must have a "label" key.
    Returns new dicts with "risk_group" and "risk_score" added.
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
    print("=== Risk group resolution ===\n")
    tests = [
        ("person", "HUMAN", 5),
        ("worker", "HUMAN", 5),
        ("hard hat", "HUMAN", 5),
        ("helmet", "HUMAN", 5),
        ("hat", "HUMAN", 5),
        ("safety vest", "HUMAN", 5),
        ("head", "HUMAN", 5),
        ("forklift", "VEHICLE", 4),
        ("pallet jack", "VEHICLE", 4),
        ("pallet", "OBSTACLE", 3),        # shorter -> OBSTACLE, not VEHICLE
        ("car", "VEHICLE", 4),
        ("cardboard box", "OBSTACLE", 3), # "car" must NOT match here
        ("cone", "SAFETY_MARKER", 2),
        ("traffic cone", "SAFETY_MARKER", 2),
        ("floor", "SURFACE", 0),
        ("parking lot", "SURFACE", 0),
        ("floor marking", "SURFACE", 0),
        ("ground", "SURFACE", 0),
        ("unknown_label", "BACKGROUND", 1),
    ]
    all_pass = True
    for label, exp_group, exp_risk in tests:
        r = get_risk_group(label)
        ok = r["group"] == exp_group and r["risk_score"] == exp_risk
        mark = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{mark}] {label!r:<25s} -> {r['group']:<15s} risk={r['risk_score']}  "
              f"(expected {exp_group}, {exp_risk})")

    print(f"\n=== get_canonical ===\n")
    for raw in ["hard-hat", "motorcycle helmet", "hi-vis vest", "cardboard box"]:
        print(f"  {raw!r:<25s} -> {get_canonical(raw)!r}")

    print(f"\n{'All tests passed.' if all_pass else 'FAILURES detected — check above.'}")
