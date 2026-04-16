"""
Rule Updater — reads human corrections and rewrites SAFETY_RULES.md.

For each correction where the LLM was wrong, it feeds the LLM:
  - The current SAFETY_RULES.md
  - The full list of human corrections (scene context + predicted vs. correct action)

The LLM proposes targeted edits to the rules so future runs handle these
cases correctly.  A timestamped backup is saved before every rewrite.

Public API
----------
  from llm_action_module.rule_updater import update_rules

  update_rules()                        # reads corrections from default path
  update_rules(corrections_path=...)    # custom path
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from llm_module.llm import get_client, MODEL

SAFETY_RULES_PATH  = ROOT / "action_module" / "SAFETY_RULES.md"
CORRECTIONS_PATH   = ROOT / "data" / "pipeline_output" / "human_corrections.json"
BACKUP_DIR         = ROOT / "action_module" / "rules_backup"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_feedback(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (confirmations, corrections) from the feedback file."""
    if not path.exists():
        raise FileNotFoundError(f"No corrections file found at {path}")
    data = json.loads(path.read_text())
    confirmations = [c for c in data if c.get("type") == "confirmation"
                     or c.get("predicted_action") == c.get("correct_action")]
    corrections   = [c for c in data if c.get("type") == "correction"
                     or c.get("predicted_action") != c.get("correct_action")]
    return confirmations, corrections


def _format_entry(c: dict, idx: int, kind: str) -> str:
    label = "Confirmation" if kind == "confirmation" else "Correction"
    lines = [
        f"### {label} {idx}",
        f"Image: {c.get('image', c.get('stem', '?'))}",
        f"LLM predicted: {c['predicted_action']} "
        f"(confidence {c.get('predicted_confidence', 0):.2f}, "
        f"entropy {c.get('entropy', 0):.3f})",
    ]
    if kind == "correction":
        lines.append(f"Human corrected to: {c['correct_action']}")
    else:
        lines.append("Human confirmed: prediction was CORRECT")
    if c.get("user_note"):
        lines.append(f"Human note: {c['user_note']}")
    if c.get("scene_summary"):
        lines.append(f"Scene: {c['scene_summary']}")
    probs = c.get("predicted_probabilities", {})
    if probs:
        lines.append("Predicted distribution: " + "  ".join(f"{k}={v:.0%}" for k, v in probs.items()))
    return "\n".join(lines)


def _backup_rules() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"SAFETY_RULES_{ts}.md"
    shutil.copy2(SAFETY_RULES_PATH, dest)
    return dest


# ── core ──────────────────────────────────────────────────────────────────────

def update_rules(
    corrections_path: Optional[Path] = None,
    dry_run: bool = False,
) -> str:
    """
    Read human corrections, call the LLM to propose rule updates, and write
    the new SAFETY_RULES.md (backing up the old one first).

    Parameters
    ----------
    corrections_path : path to human_corrections.json (default: standard location)
    dry_run          : if True, print the proposed rules but do not overwrite

    Returns
    -------
    The updated rules text.
    """
    path = Path(corrections_path) if corrections_path else CORRECTIONS_PATH
    confirmations, corrections = _load_feedback(path)

    if not confirmations and not corrections:
        print("No feedback found — rules unchanged.")
        return SAFETY_RULES_PATH.read_text()

    current_rules = SAFETY_RULES_PATH.read_text()
    client        = get_client()

    confirm_text = "\n\n".join(
        _format_entry(c, i + 1, "confirmation") for i, c in enumerate(confirmations)
    ) or "None."
    correct_text = "\n\n".join(
        _format_entry(c, i + 1, "correction") for i, c in enumerate(corrections)
    ) or "None."

    system = (
        "You are a safety rules editor for an autonomous industrial robot.\n"
        "You will receive the current SAFETY_RULES.md and two sets of human feedback:\n\n"
        "  REINFORCEMENTS — scenes where the LLM predicted correctly and the human "
        "confirmed it.  These rules are working; make them clearer and more explicit "
        "so similar future scenes are handled with higher confidence.\n\n"
        "  CORRECTIONS — scenes where the LLM was wrong and the human provided the "
        "right action.  Identify the rule gap or misfire and fix it with a targeted "
        "edit.  Do not over-tighten unrelated rules.\n\n"
        "Think of this as a reward signal: reinforcements mean 'do more of this', "
        "corrections mean 'do less of this'.\n\n"
        "Output rules:\n"
        "  - Keep the same markdown structure.\n"
        "  - Mark every changed or new line with <!-- reinforced --> or <!-- corrected -->.\n"
        "  - Be specific — name the object type, zone, or proximity that changed.\n"
        "  - Output ONLY the markdown — no explanation, no code fences."
    )

    user = (
        f"## Current SAFETY_RULES.md\n\n{current_rules}\n\n"
        f"---\n\n"
        f"## Reinforcements (prediction was correct — strengthen these patterns)\n\n"
        f"{confirm_text}\n\n"
        f"---\n\n"
        f"## Corrections (prediction was wrong — fix these patterns)\n\n"
        f"{correct_text}\n\n"
        f"---\n\n"
        f"Rewrite SAFETY_RULES.md applying both signals."
    )

    print(f"Sending {len(corrections)} correction(s) to LLM for rule update ...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    new_rules = resp.choices[0].message.content.strip()

    # Strip accidental code fences
    if new_rules.startswith("```"):
        lines = new_rules.splitlines()
        new_rules = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    if dry_run:
        print("\n── Proposed SAFETY_RULES.md ─────────────────────────────────────\n")
        print(new_rules)
        print("\n── (dry_run=True — file not changed) ────────────────────────────")
        return new_rules

    backup = _backup_rules()
    SAFETY_RULES_PATH.write_text(new_rules)
    print(f"Rules updated.  Backup saved → {backup}")
    print(f"New rules      → {SAFETY_RULES_PATH}")
    return new_rules


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update SAFETY_RULES.md from human corrections")
    parser.add_argument("--corrections", type=str, default=None,
                        help="Path to human_corrections.json (default: data/pipeline_output/human_corrections.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print proposed rules without overwriting the file")
    args = parser.parse_args()

    update_rules(
        corrections_path = Path(args.corrections) if args.corrections else None,
        dry_run          = args.dry_run,
    )
