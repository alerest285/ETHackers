"""
Unified Label Ontology — single source of truth for all label -> risk group mappings.

Risk levels (0–5):
  0  SURFACE        navigable floor — robot drives ON it, never a hazard
  1  BACKGROUND     irrelevant background (sky, wall, generic objects)
  2  SAFETY_MARKER  hazard indicators (cone, sign, barrier, caution tape)
  3  OBSTACLE       static physical obstruction (box, barrel, crate, pallet)
  3  UNKNOWN        label not (yet) in any alias bucket — conservative medium risk
  4  VEHICLE        motorized or large mobile objects (forklift, car, truck)
  5  HUMAN          any person or their presence indicator

  Note: head / helmet / hat / safety vest all map to HUMAN (risk=5) — they
  indicate human proximity and must not form a separate lower-risk class.

Matching strategy (get_risk_group):
  1. Exact match on normalized label (includes persisted learned aliases)
  2. Word-boundary substring scan — longest alias wins (guards "car"≠"cardboard")
  3. Fallback -> UNKNOWN (risk=3, flagged `unmapped=True`) — fail-safe

Auto-learning:
  Labels that hit the UNKNOWN fallback can be classified into one of the 6
  real groups via `segment_module.llm_objects.classify_and_learn`, which calls
  `register_learned_alias(label, group)`. The mapping is persisted to
  `data/learned_aliases.json` and auto-loaded on next import, so the ontology
  grows as the LLM produces new phrasings.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical risk scores per group
# ---------------------------------------------------------------------------

GROUP_RISK: dict[str, int] = {
    "SURFACE":       0,
    "BACKGROUND":    1,
    "SAFETY_MARKER": 2,
    "OBSTACLE":      3,
    "UNKNOWN":       3,   # fail-safe fallback for unseen labels — same risk as OBSTACLE
    "VEHICLE":       4,
    "HUMAN":         5,
}

# Groups the LLM classifier is allowed to pick when learning a new alias.
# UNKNOWN is excluded — it must resolve to a concrete class.
_CLASSIFIABLE_GROUPS: frozenset[str] = frozenset({
    "SURFACE", "BACKGROUND", "SAFETY_MARKER", "OBSTACLE", "VEHICLE", "HUMAN",
})

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
        # Spills / puddles — slip hazards, not obstacles
        ["spill", "liquid spill", "oil spill", "chemical spill"],
        ["puddle", "water puddle", "wet patch", "wet floor"],
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
        # Animals — living but classed as physical obstructions within the 6 groups.
        # If you want stricter handling (unpredictable movement), override via
        # the interactive classifier to promote them to HUMAN- or VEHICLE-tier.
        ["dog", "puppy", "stray dog"],
        ["cat", "kitten", "stray cat"],
        ["bird", "pigeon", "chicken", "duck"],
        ["animal", "livestock", "cow", "horse", "sheep", "goat", "pig"],
        ["rodent", "rat", "mouse"],
        # Food / floor debris — small items to avoid driving over
        ["bottle", "water bottle", "plastic bottle", "glass bottle"],
        ["can", "soda can", "tin can", "aluminium can", "aluminum can"],
        ["cup", "paper cup", "coffee cup", "disposable cup"],
        ["trash", "litter", "rubbish", "debris", "waste"],
        ["food", "food item", "apple", "banana", "sandwich", "bread", "food wrapper"],
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

    # ── UNKNOWN (risk 3) — starts empty; auto-populated at runtime ────────────
    # New labels are inserted here via register_learned_alias() before being
    # moved to one of the 6 real groups by the LLM classifier.
    "UNKNOWN": [],
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
        if not _bucket:
            continue
        _canon = _bucket[0].lower().strip()
        for _alias in _bucket:
            _key = _alias.lower().strip()
            _EXACT[_key] = _group
            _CANONICAL[_key] = _canon

# ---------------------------------------------------------------------------
# Learned-alias persistence
# Learned entries are stored in data/learned_aliases.json so the ontology
# grows across runs as the LLM classifier resolves new labels.
# ---------------------------------------------------------------------------

_LEARNED_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "learned_aliases.json"


def _load_learned() -> dict[str, str]:
    if not _LEARNED_PATH.exists():
        return {}
    try:
        data = json.loads(_LEARNED_PATH.read_text(encoding="utf-8"))
        # Filter out anything the current schema no longer recognizes.
        return {k: v for k, v in data.items()
                if isinstance(k, str) and v in _CLASSIFIABLE_GROUPS}
    except (json.JSONDecodeError, OSError):
        return {}


_LEARNED: dict[str, str] = _load_learned()

# Merge learned aliases into the in-memory lookup (exact match only — we do
# NOT add them to the substring index because arbitrary LLM phrasings may be
# overly generic and cause spurious matches).
for _lbl, _grp in _LEARNED.items():
    _key = _lbl.lower().strip()
    _EXACT[_key] = _grp
    _CANONICAL.setdefault(_key, _key)

# Sorted by descending phrase length so longer (more specific) matches win.
# Example: "pallet jack" (11 chars) is tried before "pallet" (6 chars).
# Only the curated ALIAS_GROUPS entries participate — learned aliases are
# exact-match-only (see note above).
_SUBSTR_SORTED: list[tuple[str, str]] = sorted(
    ((k, v) for k, v in _EXACT.items() if k not in _LEARNED),
    key=lambda kv: len(kv[0]),
    reverse=True,
)


def register_learned_alias(label: str, group: str) -> None:
    """Record a classified label so future detections resolve instantly.

    - Updates the in-memory exact-match table.
    - Appends to the UNKNOWN-bucket-turned-learned bookkeeping list.
    - Persists to data/learned_aliases.json (crash-safe: written every call).

    Raises ValueError if `group` is not one of the 6 canonical groups.
    """
    if group not in _CLASSIFIABLE_GROUPS:
        raise ValueError(
            f"Cannot register alias into {group!r}; "
            f"must be one of {sorted(_CLASSIFIABLE_GROUPS)}"
        )
    key = label.lower().strip()
    if not key:
        return

    _EXACT[key] = group
    _CANONICAL.setdefault(key, key)
    _LEARNED[key] = group

    _LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEARNED_PATH.write_text(
        json.dumps(_LEARNED, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
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
      1. Exact match (normalized) — includes persisted learned aliases
      2. Word-boundary substring scan — longest alias tested first
      3. Fallback -> UNKNOWN (risk=3, `unmapped=True`) — fail-safe, not BACKGROUND

    Returns:
        {"group": str, "risk_score": int, "unmapped": bool}
        `unmapped` is True only when step 3 fires — consumers can then pass
        the label to `classify_and_learn` to promote it into a real group.

    Examples:
        get_risk_group("person")        -> {"group": "HUMAN",    "risk_score": 5, "unmapped": False}
        get_risk_group("parking lot")   -> {"group": "SURFACE",  "risk_score": 0, "unmapped": False}
        get_risk_group("cardboard box") -> {"group": "OBSTACLE", "risk_score": 3, "unmapped": False}
        get_risk_group("some new thing")-> {"group": "UNKNOWN",  "risk_score": 3, "unmapped": True}
    """
    norm = normalize(category_name)

    # Step 1 — exact
    if norm in _EXACT:
        group = _EXACT[norm]
        return {"group": group, "risk_score": GROUP_RISK[group], "unmapped": False}

    # Step 2 — word-boundary substring (longest alias wins)
    for phrase, group in _SUBSTR_SORTED:
        if _word_boundary_match(phrase, norm):
            return {"group": group, "risk_score": GROUP_RISK[group], "unmapped": False}

    # Step 3 — fallback: UNKNOWN (medium risk, flagged for learning)
    return {"group": "UNKNOWN", "risk_score": GROUP_RISK["UNKNOWN"], "unmapped": True}


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


# ---------------------------------------------------------------------------
# Interactive classifier — human-in-the-loop
# ---------------------------------------------------------------------------

_INTERACTIVE_MENU: list[tuple[str, str, str]] = [
    ("1", "SURFACE",       "navigable ground (floor, road, parking lot)"),
    ("2", "BACKGROUND",    "walls, ceiling, sky, windows"),
    ("3", "SAFETY_MARKER", "cones, signs, barriers, caution tape, spills"),
    ("4", "OBSTACLE",      "static objects: boxes, pallets, animals, debris"),
    ("5", "VEHICLE",       "forklifts, cars, trucks, AGVs"),
    ("6", "HUMAN",         "persons, body parts, protective gear"),
]


def classify_label_interactive(label: str) -> str:
    """Prompt the user on stdin to classify `label` into one of the 6 groups.

    Returns the chosen group name, or "" if the user skips (label stays UNKNOWN).
    Loops until valid input is given. Accepts:
      - The menu number ("1"–"6")
      - The group name, case-insensitive ("obstacle", "HUMAN")
      - "s" / "skip" to leave the label as UNKNOWN
    """
    print(f"\n[label_ontology] New label needs classification: {label!r}")
    print("  Pick a group:")
    for num, name, hint in _INTERACTIVE_MENU:
        print(f"    {num}) {name:<15s} - {hint}")
    print("    s) skip (leave as UNKNOWN, risk=3)")

    by_num  = {num: name for num, name, _ in _INTERACTIVE_MENU}
    by_name = {name for _, name, _ in _INTERACTIVE_MENU}

    while True:
        try:
            choice = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("  (input closed — skipping)")
            return ""
        if not choice:
            continue
        if choice in by_num:
            return by_num[choice]
        if choice.upper() in by_name:
            return choice.upper()
        if choice.lower() in ("s", "skip"):
            return ""
        print("  invalid — enter 1–6, a group name, or 's' to skip")


def classify_and_learn_interactive(detections: list[dict]) -> list[dict]:
    """Human-in-the-loop counterpart to `llm_objects.classify_and_learn`.

    For every detection flagged `unmapped=True`, prompt the user once per
    unique label, persist the answer via `register_learned_alias`, and
    rewrite `risk_group` / `risk_score` in place.

    Skipped labels stay UNKNOWN (risk=3) — safe default. On the next run the
    user is prompted again unless they classify it.
    """
    unique_unmapped: dict[str, str] = {}
    for det in detections:
        if not det.get("unmapped"):
            continue
        lbl = (det.get("label") or "").strip().lower()
        if lbl and lbl not in unique_unmapped:
            unique_unmapped[lbl] = classify_label_interactive(lbl)

    for lbl, grp in unique_unmapped.items():
        if grp:  # empty string = user skipped
            register_learned_alias(lbl, grp)

    # Re-resolve every detection so learned entries update in place.
    for det in detections:
        r = get_risk_group(det.get("label", ""))
        det["risk_group"] = r["group"]
        det["risk_score"] = r["risk_score"]
        if r.get("unmapped"):
            det["unmapped"] = True
        else:
            det.pop("unmapped", None)
    return detections


def map_detections(detections: list[dict]) -> list[dict]:
    """Enrich a list of detection dicts with risk_group and risk_score.

    Each detection must have a "label" key.
    Returns new dicts with "risk_group" and "risk_score" added.
    """
    enriched = []
    for det in detections:
        d = dict(det)
        result = get_risk_group(d.get("label", ""))
        d["risk_group"] = result["group"]
        d["risk_score"] = result["risk_score"]
        if result.get("unmapped"):
            d["unmapped"] = True
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
        ("dog", "OBSTACLE", 3),
        ("cat", "OBSTACLE", 3),
        ("bird", "OBSTACLE", 3),
        ("cow", "OBSTACLE", 3),
        ("bottle", "OBSTACLE", 3),
        ("apple", "OBSTACLE", 3),
        ("trash", "OBSTACLE", 3),
        ("spill", "SAFETY_MARKER", 2),
        ("puddle", "SAFETY_MARKER", 2),
        ("unknown_label", "UNKNOWN", 3),   # fail-safe fallback, flagged for learning
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

    print(f"\n=== Unmapped flag ===\n")
    r = get_risk_group("robot arm")
    print(f"  'robot arm' -> group={r['group']} risk={r['risk_score']} unmapped={r['unmapped']}")
    assert r["unmapped"] is True and r["group"] == "UNKNOWN"

    print(f"\n=== register_learned_alias (in-memory only, skipping disk write) ===\n")
    # Simulate classifier output without touching the JSON file.
    _EXACT["robot arm test"] = "OBSTACLE"
    r = get_risk_group("robot arm test")
    print(f"  'robot arm test' -> group={r['group']} risk={r['risk_score']} unmapped={r['unmapped']}")
    assert r["group"] == "OBSTACLE" and r["unmapped"] is False
    del _EXACT["robot arm test"]

    print(f"\n{'All tests passed.' if all_pass else 'FAILURES detected — check above.'}")
