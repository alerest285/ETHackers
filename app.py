"""
app.py — FastAPI server for the ETHackers pipeline web UI.

Streams every pipeline stage as NDJSON:
  raw → yolo → llm_prompt → gdino → sam2 → depth → fused → scene_data
  → decision → done

Each visual stage emits a full-frame composited PNG so the frontend can
cross-fade them into a single image. The `scene_data` event ships the
pixel-level depth + color data needed to build an interactive 3D point
cloud on the client with Three.js.

Run:
    python app.py
Then open http://localhost:7860
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MAX_W = 960         # cap width of the stage images streamed to browser
POINTCLOUD_W = 300  # target width of the point cloud grid
FOCAL_SCALE = 0.8   # matches lift_3d.py

# Toggle: route Grounding DINO + DepthAnything to HF Inference API for speed.
# Set USE_HF_API=1 in the environment and HF_TOKEN=hf_... to enable.
import os as _os
USE_HF_API = _os.environ.get("USE_HF_API", "").lower() in ("1", "true", "yes")


# ── lazy model cache ──────────────────────────────────────────────────────────

_cache: dict = {}


def _get_yolo():
    if "yolo" not in _cache:
        from segment_module.segment import load_model
        _cache["yolo"] = load_model()
    return _cache["yolo"]


def _get_depth_pipe():
    if "depth" not in _cache:
        from transformers import pipeline as hf_pipeline
        import torch as _torch
        # Use GPU (cuda:0) when available, else MPS (Apple), else CPU.
        if _torch.cuda.is_available():
            device = 0
        elif getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
        _cache["depth"] = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        print(f"[app] DepthAnything loaded on device={device}")
    return _cache["depth"]


def _get_gdino():
    if "gdino" not in _cache:
        from segment_module.grounding_dino import load_model
        _cache["gdino"] = load_model()
    return _cache["gdino"]


def _get_sam2():
    if "sam2" not in _cache:
        from segment_module.sam2 import load_model
        _cache["sam2"] = load_model()
    return _cache["sam2"]


def _get_scene_builder():
    if "scene" not in _cache:
        spec = importlib.util.spec_from_file_location(
            "lift_3d", ROOT / "3d-module" / "lift_3d.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache["scene"] = mod
    return _cache["scene"]


# ── drawing helpers ───────────────────────────────────────────────────────────

GROUP_COLORS = {
    "HUMAN":         (80,  80, 255),
    "VEHICLE":       (80, 170, 255),
    "OBSTACLE":      (255, 180, 80),
    "SAFETY_MARKER": (80, 255, 130),
    "BACKGROUND":    (180, 180, 180),
    "person":        (80,  80, 255),
    "vehicle":       (80, 170, 255),
    "bicycle":       (80, 220, 255),
    "animal":        (120, 255, 120),
    "cone":          (80, 255, 100),
    "box":           (255, 170,  80),
    "other":         (180, 180, 180),
}

MASK_PALETTE = [
    (255, 80, 80),  (80, 140, 255), (80, 255, 140), (255, 200, 80),
    (200, 80, 255), (80, 255, 255), (255, 100, 180), (160, 255, 80),
]


def _resize_pil(pil: Image.Image, max_w: int = MAX_W) -> Image.Image:
    w, h = pil.size
    if w <= max_w:
        return pil
    new_h = int(h * max_w / w)
    return pil.resize((max_w, new_h), Image.LANCZOS)


def _bgr_to_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _pil_to_b64(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _draw_detections_on(base_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    img = base_bgr.copy()
    overlay = img.copy()
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = GROUP_COLORS.get(d.get("risk_group", ""), (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = GROUP_COLORS.get(d.get("risk_group", ""), (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"[{i}] {d.get('label','?')} {d.get('score',0):.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, y_label), (x1 + tw + 8, y_label + th + 6), color, -1)
        cv2.putText(img, text, (x1 + 4, y_label + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _overlay_masks_on(base_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    img = base_bgr.copy().astype(np.float32)
    for i, r in enumerate(results):
        m = r.get("mask")
        if m is None or not m.any():
            continue
        color = np.array(MASK_PALETTE[i % len(MASK_PALETTE)][::-1], dtype=np.float32)  # RGB→BGR
        mask_f = m.astype(np.float32)[..., None]
        img = img * (1 - 0.45 * mask_f) + color[None, None, :] * 0.45 * mask_f

        # outline
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_u8 = img.astype(np.uint8)
        cv2.drawContours(img_u8, contours, -1, tuple(int(c) for c in color), 2)
        img = img_u8.astype(np.float32)
    return img.astype(np.uint8)


def _depth_blend_on(base_bgr: np.ndarray, depth_norm: np.ndarray,
                    cmap=cv2.COLORMAP_INFERNO, alpha: float = 0.65) -> np.ndarray:
    vis = (1.0 - np.clip(depth_norm, 0, 1)) * 255  # close = bright
    heat = cv2.applyColorMap(vis.astype(np.uint8), cmap)
    if heat.shape[:2] != base_bgr.shape[:2]:
        heat = cv2.resize(heat, (base_bgr.shape[1], base_bgr.shape[0]))
    return cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)


def _encode_gray_png(arr_u8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr_u8)
    return base64.b64encode(buf.tobytes()).decode()


# ── decision heuristic ────────────────────────────────────────────────────────

def _get_openai_client():
    """Shared OpenAI client (same singleton segment_module.llm_objects uses)."""
    if "openai" not in _cache:
        from segment_module.llm_objects import load_client
        _cache["openai"] = load_client()
    return _cache["openai"]


WEARABLE_HINTS = (
    "helmet", "hard hat", "hardhat", "cap", "hat", "beanie", "headband",
    "vest", "hi-vis", "hi vis", "safety vest", "reflective vest",
    "gloves", "glove", "boots", "boot", "shoes", "sneakers", "footwear",
    "mask", "face mask", "goggles", "glasses", "sunglasses",
    "bag", "backpack", "handbag",
    "head", "face", "hand", "arm", "leg", "foot", "torso", "body",
    "jacket", "shirt", "coat", "uniform", "hair", "hairnet", "ear",
    "safety harness", "harness",
)

VEHICLE_PART_HINTS = (
    "wheel", "tire", "tyre", "headlight", "taillight", "tail light",
    "mirror", "side mirror", "rearview mirror",
    "windshield", "window", "bumper", "door", "license plate",
    "fender", "grille", "roof rack", "exhaust",
)


def _bbox_containment(outer, inner) -> float:
    """Fraction of inner bbox area that lies inside outer bbox."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    x1 = max(ox1, ix1); y1 = max(oy1, iy1)
    x2 = min(ox2, ix2); y2 = min(oy2, iy2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    inner_area = max(1e-6, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


def _snap_parts_to_owners(detections: list[dict]) -> list[dict]:
    """
    Deterministic part→anchor binding.

    For every detection whose label matches a known WEARABLE or VEHICLE_PART,
    find the HUMAN / VEHICLE detection whose 2D bbox most contains it
    (≥ 50% of the part's area inside the anchor's bbox) and snap the part's
    depth_score (and proximity_label) to the anchor's.

    Returns the list of snaps applied (for the UI).
    """
    from depth_module.fused_depth import proximity_label as _prox

    humans   = [(i, d) for i, d in enumerate(detections)
                if d.get("risk_group") == "HUMAN"]
    vehicles = [(i, d) for i, d in enumerate(detections)
                if d.get("risk_group") == "VEHICLE"]

    snaps: list[dict] = []
    for i, d in enumerate(detections):
        label = (d.get("label") or "").lower()
        grp   = d.get("risk_group", "")
        if grp in ("HUMAN", "VEHICLE"):
            continue

        candidates = None
        if any(h in label for h in WEARABLE_HINTS):
            candidates = humans
        elif any(h in label for h in VEHICLE_PART_HINTS):
            candidates = vehicles
        if not candidates:
            continue

        best_j, best_a, best_frac = None, None, 0.0
        for j, a in candidates:
            if j == i:
                continue
            frac = _bbox_containment(a["box"], d["box"])
            if frac > best_frac:
                best_frac, best_j, best_a = frac, j, a
        if best_a is None or best_frac < 0.5:
            continue

        anchor_depth = best_a.get("depth_score")
        if anchor_depth is None:
            continue
        old_depth = d.get("depth_score", 0.5)
        d["depth_score_original"] = old_depth
        d["depth_score"]          = anchor_depth
        d["snapped_to"]           = best_a.get("label")
        d["snapped_overlap"]      = round(best_frac, 3)
        try:
            d["proximity_label"] = _prox(anchor_depth)
        except Exception:
            pass
        snaps.append({
            "id": i, "label": d.get("label"),
            "anchor_id": best_j, "anchor_label": best_a.get("label"),
            "overlap": round(best_frac, 2),
            "old_depth": round(float(old_depth or 0.5), 3),
            "new_depth": round(float(anchor_depth), 3),
        })
    return snaps


def _llm_consistency_check(client, enriched: list[dict]) -> list[dict]:
    """
    Ask GPT-4o-mini whether any objects' depth estimates are inconsistent
    with real-world relationships (helmet/person, tire/car, etc.) and
    return a list of corrections: [{"id": i, "new_depth": 0-1, "because": ...}]
    """
    import json as _json
    obj_lines = []
    for i, d in enumerate(enriched):
        ds = d.get("depth_score")
        if ds is None:
            ds = 0.5
        box = d.get("box") or []
        obj_lines.append(
            f"[{i}] {d.get('label','?')} "
            f"(group={d.get('risk_group','?')}, "
            f"depth={float(ds):.2f}, "
            f"prox={d.get('proximity_label','?')}, "
            f"box={[int(v) for v in box]})"
        )

    prompt = (
        "You are reviewing a robot perception pipeline's per-object proximity "
        "estimates. Each object has a proximity score in [0, 1] where 0 = closest "
        "to the camera and 1 = farthest.\n\n"
        "Objects:\n" + "\n".join(obj_lines) + "\n\n"
        "Apply these real-world spatial priors:\n\n"
        "GROUP SNAPPING (same physical entity → same depth):\n"
        "  • Wearables (helmet, hard hat, hi-vis vest) → snap to the person wearing them.\n"
        "  • Vehicle parts (tire, wheel, headlight, mirror) → snap to the vehicle.\n"
        "  • Body parts (head, hand, torso) → snap to the person.\n"
        "  • Held tools / carried objects → snap to the worker holding them.\n"
        "  • Objects overlapping in 2D with very different depths → probably one entity.\n\n"
        "BACKGROUND / ENVIRONMENT SEPARATION:\n"
        "  • BACKGROUND and SURFACE objects (walls, floor, shelves, racks, scenery) "
        "should be pushed FARTHER from camera (higher depth, 0.65–0.90) unless they "
        "are obviously right in front of the camera (bbox > 25% of image).\n"
        "  • Small-bbox background items (< 5% of image) at CLOSE depth are almost "
        "certainly wrong — push them to at least 0.60.\n"
        "  • Scenery / infrastructure (ceiling, pillars, distant shelving) should be "
        "at FAR depth (0.70+).\n"
        "  • Do NOT pull background objects forward — only push them farther.\n\n"
        "RELATIVE ORDERING:\n"
        "  • Large foreground humans/vehicles should always be CLOSER than small "
        "background objects behind them.\n"
        "  • If a small object is in front of a large object (by 2D overlap) but "
        "scored farther, correct it.\n\n"
        "For each detection whose depth is clearly wrong, output a correction.\n\n"
        'Return ONLY a JSON object:\n'
        '{"corrections": [\n'
        '  {"id": <int>, "new_depth": <float 0-1>, "anchor_id": <int or null>, '
        '"because": "<short reason>"}\n'
        ']}\n\n'
        "If everything looks consistent, return {\"corrections\": []}. "
        "Only flag cases with HIGH confidence — do not guess."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = _json.loads(resp.choices[0].message.content.strip())
        corrections = parsed.get("corrections", [])
        # Basic sanitisation
        clean = []
        for c in corrections:
            try:
                idx = int(c.get("id", -1))
                nd  = float(c.get("new_depth", 0.5))
                nd  = max(0.0, min(1.0, nd))
                if 0 <= idx < len(enriched):
                    clean.append({
                        "id": idx,
                        "new_depth": nd,
                        "anchor_id": c.get("anchor_id"),
                        "because": str(c.get("because", "")),
                    })
            except Exception:
                continue
        return clean
    except Exception as e:
        print(f"  [consistency_check] failed: {e}")
        return []


def _detection_summary(detections: list[dict]) -> str:
    """Text fallback when the scene graph isn't available."""
    if not detections:
        return "Scene: no objects detected."
    lines = ["Scene (detection summary):"]
    for d in detections:
        lines.append(
            f"  - {d.get('label','?')} ({d.get('risk_group','?')}) "
            f"proximity={d.get('proximity_label','?')} "
            f"zone={d.get('path_zone','?')} "
            f"depth={d.get('depth_score','?')}"
        )
    return "\n".join(lines)


def _make_decision(enriched: list[dict]) -> dict:
    hits = []
    reasons = []
    for d in enriched:
        prox = d.get("proximity_label")
        zone = d.get("path_zone")
        grp  = d.get("risk_group", "")
        high_risk = grp in ("HUMAN", "person", "VEHICLE", "vehicle", "bicycle")

        if prox == "CLOSE" and zone == "CENTER" and high_risk:
            hits.append(("STOP", d))
            reasons.append(f"{d.get('label','object')} CLOSE in CENTER path")
        elif prox == "CLOSE" and zone == "CENTER":
            hits.append(("SLOW", d))
            reasons.append(f"{d.get('label','object')} close in path")
        elif prox == "CLOSE" and high_risk:
            hits.append(("SLOW", d))
            reasons.append(f"{d.get('label','object')} close (peripheral)")

    if any(h[0] == "STOP" for h in hits):
        return {"label": "STOP", "color": "#ef4444",
                "reasons": reasons[:4] or ["High-risk object in path"]}
    if any(h[0] == "SLOW" for h in hits):
        return {"label": "SLOW", "color": "#f59e0b",
                "reasons": reasons[:4] or ["Obstacles nearby"]}
    return {"label": "CONTINUE", "color": "#22c55e",
            "reasons": ["No close obstacles in the forward path"]}


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media",  StaticFiles(directory="ui_media"), name="media")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── Human feedback endpoint ───────────────────────────────────────────────────
# Records a user correction to data/pipeline_output/human_corrections.json
# so the rule-updater can later re-learn from them.
CORRECTIONS_PATH = ROOT / "data" / "pipeline_output" / "human_corrections.json"


from fastapi import Request

@app.post("/correction")
async def record_correction(request: Request):
    import time as _time
    data = await request.json()
    entry = {
        "type":               "correction" if data.get("predicted_action") != data.get("correct_action") else "reinforcement",
        "stem":               str(data.get("stem", "")),
        "predicted_action":   str(data.get("predicted_action", "")).upper(),
        "correct_action":     str(data.get("correct_action", "")).upper(),
        "predicted_confidence": float(data.get("predicted_confidence", 0.0) or 0.0),
        "entropy":            float(data.get("entropy", 0.0) or 0.0),
        "reasoning":          str(data.get("reasoning", "")),
        "user_note":          str(data.get("user_note", "")),
        "scene_summary":      str(data.get("scene_summary", "")),
        "timestamp":          _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if CORRECTIONS_PATH.exists():
        try:
            existing = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(entry)
    CORRECTIONS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "total": len(existing), "entry": entry})


@app.get("/entropy_log")
def entropy_log():
    """Top uncertain frames, for an optional 'most ambiguous' dashboard."""
    p = ROOT / "data" / "pipeline_output" / "llm_actor_entropy.json"
    if not p.exists():
        return JSONResponse({"entries": []})
    try:
        entries = json.loads(p.read_text(encoding="utf-8"))
        entries.sort(key=lambda x: x.get("entropy", 0), reverse=True)
        return JSONResponse({"entries": entries[:20]})
    except Exception as e:
        return JSONResponse({"entries": [], "error": str(e)})


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return None
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return super().default(o)


def _event(stage: str, status: str, **kw) -> bytes:
    return (json.dumps({"stage": stage, "status": status, **kw},
                       cls=_SafeEncoder) + "\n").encode()


# Serialize heavy model inference across concurrent requests so parallel
# multi-image runs don't corrupt global model state. Requests queue, the UI
# still displays them all as "in progress" via their own streams.
import asyncio as _asyncio
_INFERENCE_LOCK = _asyncio.Semaphore(1)


@app.post("/run_full")
async def run_full(
    image: UploadFile = File(...),
    use_gdino: str = Form("true"),
    use_sam: str = Form("true"),
):
    contents = await image.read()
    pil_orig = Image.open(io.BytesIO(contents)).convert("RGB")
    pil = _resize_pil(pil_orig, MAX_W)
    W, H = pil.size
    base_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    want_gdino = use_gdino.lower() == "true"
    want_sam = use_sam.lower() == "true"

    def gen():
        # 1. Raw ---------------------------------------------------------------
        yield _event("raw", "ok",
                     title="Raw image",
                     image=_pil_to_b64(pil),
                     meta={"width": W, "height": H})

        # Share a single OpenAI client across all LLM sub-steps
        llm_client_shared = None
        try:
            llm_client_shared = _get_openai_client()
        except Exception as e:
            traceback.print_exc()

        # 2. LLM → object prompt ----------------------------------------------
        llm_prompt = None
        yield _event("llm_prompt", "running", title="LLM → object prompt",
                     message="Asking GPT-4o-mini to enumerate visible objects...")
        try:
            from segment_module.llm_objects import load_client, get_detection_prompt
            prompt_client = load_client()
            llm_prompt = get_detection_prompt(prompt_client, pil)
            yield _event("llm_prompt", "ok", title="LLM → object prompt",
                         text=llm_prompt)
        except Exception as e:
            traceback.print_exc()
            # Fallback — generic industrial-scene prompt
            llm_prompt = "person . vehicle . forklift . barrel . cone . box . ladder . hard hat ."
            yield _event("llm_prompt", "error", title="LLM → object prompt",
                         message=f"{type(e).__name__}: {e}; using fallback prompt",
                         text=llm_prompt)

        # 3. Grounding DINO ----------------------------------------------------
        gdino_dets: list[dict] = []
        if want_gdino:
            source = "HF API" if USE_HF_API else "local"
            yield _event("gdino", "running", title="Grounding DINO",
                         message=f"Open-vocabulary detection ({source})...")
            try:
                if USE_HF_API:
                    from hf_inference import detect_hf
                    gdino_dets = detect_hf(pil, llm_prompt, score_threshold=0.30)
                else:
                    from segment_module.grounding_dino import detect as gdino_detect
                    gdino = _get_gdino()
                    gdino_dets = gdino_detect(
                        gdino, pil, llm_prompt,
                        score_threshold=0.30,
                        learn_client=llm_client_shared,
                    )
                img = _draw_detections_on(base_bgr, gdino_dets)
                yield _event("gdino", "ok", title="Grounding DINO",
                             image=_bgr_to_b64(img),
                             meta={"count": len(gdino_dets),
                                   "detections": [
                                       {"label": d["label"], "score": d["score"],
                                        "risk_group": d.get("risk_group")}
                                       for d in gdino_dets]})
            except Exception as e:
                traceback.print_exc()
                yield _event("gdino", "error", title="Grounding DINO",
                             message=f"{type(e).__name__}: {e}")
        else:
            yield _event("gdino", "error", title="Grounding DINO",
                         message="Grounding DINO disabled — skipping")

        dets_for_depth = list(gdino_dets)
        if not dets_for_depth:
            yield _event("filter", "error", title="Filter",
                         message="No detections — nothing to process downstream")

        # 4. SAM 2 -------------------------------------------------------------
        sam_results: list[dict] = []
        if want_sam and dets_for_depth:
            yield _event("sam2", "running", title="SAM 2",
                         message="Segmenting each detection...")
            try:
                from segment_module.sam2 import segment
                sam = _get_sam2()
                sam_results = segment(sam, pil, dets_for_depth)
                img = _overlay_masks_on(base_bgr, sam_results)
                yield _event("sam2", "ok", title="SAM 2 segmentation",
                             image=_bgr_to_b64(img),
                             meta={"count": sum(1 for r in sam_results
                                                if r.get("mask") is not None
                                                and r["mask"].any())})
                # sam_results carry the masks + all fields from gdino_dets →
                # promote to the downstream detection list
                dets_for_depth = sam_results
            except Exception as e:
                traceback.print_exc()
                yield _event("sam2", "error", title="SAM 2",
                             message=f"{type(e).__name__}: {e}")

        # 5. Filter — is this scene worth analysing? ---------------------------
        if dets_for_depth:
            yield _event("filter", "running", title="Scene filter",
                         message="Checking for risk-relevant objects...")
            try:
                from src.filter_module import is_interesting
                interesting = is_interesting(dets_for_depth, conf_threshold=0.25)
                if interesting:
                    hazards = [d for d in dets_for_depth
                               if d.get("risk_group") not in ("BACKGROUND", "SURFACE")]
                    yield _event("filter", "ok", title="Scene filter",
                                 text=f"Interesting — {len(hazards)} hazard-group detection(s)",
                                 meta={"interesting": True,
                                       "hazard_count": len(hazards)})
                else:
                    yield _event("filter", "ok", title="Scene filter",
                                 text="Scene has no risk-relevant objects "
                                      "(only SURFACE/BACKGROUND); continuing for visualisation",
                                 meta={"interesting": False})
            except Exception as e:
                traceback.print_exc()
                yield _event("filter", "error", title="Scene filter",
                             message=f"{type(e).__name__}: {e}")

        # 6. DepthAnything -----------------------------------------------------
        source = "HF API" if USE_HF_API else "local"
        yield _event("depth", "running", title="Depth Anything V2",
                     message=f"Running depth estimation ({source})...")
        depth_norm = None
        try:
            if USE_HF_API:
                from hf_inference import depth_hf
                raw_depth = depth_hf(pil)
            else:
                pipe = _get_depth_pipe()
                raw_depth = np.array(pipe(pil)["depth"], dtype=np.float32)
            if raw_depth.shape != (H, W):
                raw_depth = cv2.resize(raw_depth, (W, H))
            d_min, d_max = raw_depth.min(), raw_depth.max()
            # invert disparity: 0=close, 1=far
            depth_norm = 1.0 - (raw_depth - d_min) / (d_max - d_min + 1e-6)
            img = _depth_blend_on(base_bgr, depth_norm)
            yield _event("depth", "ok", title="Depth Anything V2",
                         image=_bgr_to_b64(img))
        except Exception as e:
            traceback.print_exc()
            yield _event("depth", "error", title="Depth Anything V2",
                         message=f"{type(e).__name__}: {e}")

        # 7. LLM holistic depth signals ---------------------------------------
        if depth_norm is not None and dets_for_depth and llm_client_shared is not None:
            yield _event("llm_depth", "running", title="LLM depth reasoning",
                         message="GPT scores per-object proximity from scene context...")
            try:
                from src.llm_depth import get_llm_depth_signals
                dets_for_depth = get_llm_depth_signals(pil, dets_for_depth, llm_client_shared)
                hits = [d for d in dets_for_depth
                        if d.get("depth_llm_label") in ("CLOSE", "MEDIUM", "FAR")]
                yield _event("llm_depth", "ok", title="LLM depth reasoning",
                             text=f"Per-object proximity estimated for {len(hits)} object(s)",
                             meta={"detections": [
                                 {"label": d.get("label"),
                                  "risk_group": d.get("risk_group"),
                                  "depth_llm_label": d.get("depth_llm_label"),
                                  "depth_llm_conf": d.get("depth_llm_conf"),
                                  "depth_llm_reason": d.get("depth_llm_reason"),
                                  }
                                 for d in dets_for_depth]})
            except Exception as e:
                traceback.print_exc()
                yield _event("llm_depth", "error", title="LLM depth reasoning",
                             message=f"{type(e).__name__}: {e}")

        # 8. Edge-clip rationalization ----------------------------------------
        if depth_norm is not None and dets_for_depth and llm_client_shared is not None:
            yield _event("edge_rat", "running", title="Edge rationalization",
                         message="Disambiguating edge-clipped detections...")
            try:
                from src.edge_rationalization import rationalize_edge_detections
                dets_for_depth = rationalize_edge_detections(
                    pil, dets_for_depth, llm_client_shared
                )
                clipped = sum(1 for d in dets_for_depth
                              if d.get("edge_clip_label") in ("CLOSE", "MEDIUM"))
                yield _event("edge_rat", "ok", title="Edge rationalization",
                             text=f"{clipped} detection(s) flagged as edge-clipped (override applied)",
                             meta={"clipped": clipped})
            except Exception as e:
                traceback.print_exc()
                yield _event("edge_rat", "error", title="Edge rationalization",
                             message=f"{type(e).__name__}: {e}")

        # 9. Triple-fused depth -----------------------------------------------
        enriched: list[dict] = []
        corrected = depth_norm
        if depth_norm is not None and dets_for_depth:
            yield _event("fused", "running", title="Fused depth (DA + size + area + LLM)",
                         message="Fusing four depth signals per object...")
            try:
                from depth_module.fused_depth import enrich_detections
                enriched, corrected = enrich_detections(
                    dets_for_depth, depth_norm, H, W
                )
                img = _depth_blend_on(base_bgr, corrected, alpha=0.55)
                # re-draw boxes with proximity label
                for i, d in enumerate(enriched, 1):
                    x1, y1, x2, y2 = [int(v) for v in d["box"]]
                    prox = d.get("proximity_label", "")
                    col = {"CLOSE": (80, 80, 255), "MEDIUM": (80, 180, 255),
                           "FAR":   (80, 255, 180)}.get(prox, (220, 220, 220))
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                    tag = f"[{i}] {d.get('label','?')} {prox}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    y_label = max(0, y1 - th - 6)
                    cv2.rectangle(img, (x1, y_label), (x1 + tw + 8, y_label + th + 6), col, -1)
                    cv2.putText(img, tag, (x1 + 4, y_label + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                yield _event("fused", "ok", title="Fused depth + heuristics",
                             image=_bgr_to_b64(img),
                             meta={"detections": [
                                 {"label": d["label"],
                                  "risk_group": d.get("risk_group"),
                                  "depth_score": d.get("depth_score"),
                                  "proximity_label": d.get("proximity_label"),
                                  "path_zone": d.get("path_zone")}
                                 for d in enriched]})
            except Exception as e:
                traceback.print_exc()
                yield _event("fused", "error", title="Fused depth",
                             message=f"{type(e).__name__}: {e}")

        # 9.5 Deterministic part → owner binding ------------------------------
        # Snap helmets/vests/hats/body parts to their containing HUMAN; snap
        # wheels/mirrors/headlights to their containing VEHICLE. Runs BEFORE
        # the LLM sanity pass so the easy cases never burn an LLM call.
        if enriched:
            yield _event("binding", "running", title="Part → owner binding",
                         message="Snapping wearables and parts to their anchors...")
            try:
                snaps = _snap_parts_to_owners(enriched)
                yield _event("binding", "ok", title="Part → owner binding",
                             text=(f"{len(snaps)} part(s) snapped to owners"
                                   if snaps else "No part/owner pairs found"),
                             meta={"snaps": snaps})
            except Exception as e:
                traceback.print_exc()
                yield _event("binding", "error", title="Part → owner binding",
                             message=f"{type(e).__name__}: {e}")

        # 10. LLM consistency loop --------------------------------------------
        # Ask the LLM to flag physically-inconsistent depth estimates
        # (helmet vs person, tire vs car, etc.) and snap them together.
        if enriched and llm_client_shared is not None:
            yield _event("consistency", "running", title="3D consistency check",
                         message="LLM reviewing depths for real-world plausibility...")
            try:
                corrections = _llm_consistency_check(llm_client_shared, enriched)
                # Apply corrections, re-derive proximity labels
                from depth_module.fused_depth import proximity_label as _prox
                applied = []
                for c in corrections:
                    idx = c["id"]
                    old = enriched[idx].get("depth_score", 0.5)
                    enriched[idx]["depth_score_original"] = old
                    enriched[idx]["depth_score"] = c["new_depth"]
                    enriched[idx]["consistency_reason"] = c["because"]
                    enriched[idx]["consistency_anchor"] = c.get("anchor_id")
                    try:
                        enriched[idx]["proximity_label"] = _prox(c["new_depth"])
                    except Exception:
                        pass
                    applied.append({
                        "id": idx,
                        "label": enriched[idx].get("label"),
                        "old_depth": old, "new_depth": c["new_depth"],
                        "because": c["because"],
                    })
                text = (f"{len(applied)} correction(s) applied"
                        if applied else "No inconsistencies found")
                yield _event("consistency", "ok", title="3D consistency check",
                             text=text,
                             meta={"corrections": applied})
            except Exception as e:
                traceback.print_exc()
                yield _event("consistency", "error", title="3D consistency check",
                             message=f"{type(e).__name__}: {e}")

        # 11. scene_data (for 3D rendering) -----------------------------------
        scene_graph_payload = None
        if corrected is not None:
            yield _event("scene", "running", title="3D lift + scene graph",
                         message="Building scene graph...")
            try:
                scene_mod = _get_scene_builder()
                masks = None
                if sam_results:
                    masks = []
                    for r in sam_results:
                        m = r.get("mask")
                        masks.append(m if m is not None else np.zeros((H, W), dtype=bool))
                # Sanitize detections: lift_3d uses det.get("depth_score", ...)
                # which returns None if the key is present with value None.
                # Replace any None numeric fields with safe defaults.
                sanitized = []
                for d in (enriched or dets_for_depth):
                    dd = dict(d)
                    if dd.get("depth_score") is None:
                        dd["depth_score"] = 0.5
                    if dd.get("score") is None:
                        dd["score"] = 0.0
                    if dd.get("risk_score") is None:
                        dd["risk_score"] = 1
                    sanitized.append(dd)

                builder = scene_mod.SceneGraphBuilder(point_cloud_step=6)
                graph = builder.process(
                    depth_map=corrected,
                    detections=sanitized,
                    img_w=W, img_h=H,
                    image_id="webui",
                    masks=masks,
                )

                # Downsample depth + original for a fast client-side point cloud
                pc_w = min(POINTCLOUD_W, W)
                pc_h = int(H * pc_w / W)
                depth_small = cv2.resize(corrected, (pc_w, pc_h))
                orig_small = cv2.resize(np.array(pil), (pc_w, pc_h))
                depth_u8 = (np.clip(depth_small, 0, 1) * 255).astype(np.uint8)

                # Segmentation tint map at cloud resolution.
                # RGBA per pixel — A>0 means the pixel belongs to an object mask.
                tint_full = np.zeros((H, W, 4), dtype=np.uint8)
                group_rgb = {
                    "HUMAN":         (255,  80,  80),
                    "VEHICLE":       (255, 170,  80),
                    "OBSTACLE":      ( 80, 170, 255),
                    "SAFETY_MARKER": ( 80, 255, 130),
                    "BACKGROUND":    (180, 180, 180),
                    "person":        (255,  80,  80),
                    "vehicle":       (255, 170,  80),
                    "bicycle":       ( 80, 220, 255),
                    "animal":        (120, 255, 120),
                    "cone":          ( 80, 255, 100),
                    "box":           (255, 170,  80),
                }
                for r in (sam_results or []):
                    m = r.get("mask")
                    if m is None or not m.any():
                        continue
                    grp = r.get("risk_group", "")
                    rgb = group_rgb.get(grp, (180, 180, 180))
                    tint_full[m] = [rgb[0], rgb[1], rgb[2], 200]
                # also fill bbox regions when there is no SAM mask
                if not sam_results:
                    for d in enriched or dets_for_depth:
                        x1, y1, x2, y2 = [int(v) for v in d["box"]]
                        rgb = group_rgb.get(d.get("risk_group", ""), (180, 180, 180))
                        tint_full[y1:y2, x1:x2] = [rgb[0], rgb[1], rgb[2], 140]
                tint_small = cv2.resize(tint_full, (pc_w, pc_h),
                                         interpolation=cv2.INTER_NEAREST)

                # Per-pixel instance index map (0 = background, i+1 = object index).
                # Lets the client isolate/highlight any single detected object in 3D.
                src_list = sam_results if sam_results else (enriched or dets_for_depth)
                seg_idx_full = np.zeros((H, W), dtype=np.uint8)
                for i, r in enumerate(src_list[:250]):
                    val = i + 1
                    if isinstance(r, dict) and r.get("mask") is not None and r["mask"].any():
                        seg_idx_full[r["mask"]] = val
                    elif isinstance(r, dict) and "box" in r:
                        x1, y1, x2, y2 = [int(v) for v in r["box"]]
                        seg_idx_full[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = val
                seg_small = cv2.resize(seg_idx_full, (pc_w, pc_h),
                                        interpolation=cv2.INTER_NEAREST)

                # send as PNGs
                ok1, buf1 = cv2.imencode(".png", depth_u8)
                ok2, buf2 = cv2.imencode(".png", cv2.cvtColor(orig_small, cv2.COLOR_RGB2BGR))
                tint_bgra = cv2.cvtColor(tint_small, cv2.COLOR_RGBA2BGRA)
                ok3, buf3 = cv2.imencode(".png", tint_bgra)
                ok4, buf4 = cv2.imencode(".png", seg_small)

                fx = max(W, H) * FOCAL_SCALE

                nodes_payload = [{
                    "id": n.id, "label": n.label,
                    "risk_group": n.risk_group,
                    "proximity_label": n.proximity_label,
                    "centroid_3d": n.centroid_3d,
                    "box_2d": n.box_2d,
                    "depth_score": n.depth_score,
                } for n in graph.nodes]

                edges_payload = [{
                    "from_id": e.from_id, "to_id": e.to_id,
                    "distance_3d": e.distance_3d,
                    "relative_position": e.relative_position,
                    "blocking": e.blocking,
                } for e in graph.edges]

                scene_graph_payload = {
                    "nodes": nodes_payload, "edges": edges_payload,
                    "text": graph.text,
                    "n_points": int(graph.n_points),
                }

                yield _event("scene_data", "ok", title="Scene data",
                             depth_png=base64.b64encode(buf1.tobytes()).decode(),
                             color_png=base64.b64encode(buf2.tobytes()).decode(),
                             tint_png=base64.b64encode(buf3.tobytes()).decode(),
                             seg_png=base64.b64encode(buf4.tobytes()).decode(),
                             W=W, H=H, pc_w=pc_w, pc_h=pc_h,
                             fx=fx, cx=W / 2.0, cy=H / 2.0,
                             nodes=nodes_payload, edges=edges_payload,
                             text=graph.text,
                             meta={"nodes": len(nodes_payload),
                                   "edges": len(edges_payload),
                                   "points": int(graph.n_points)})
            except Exception as e:
                traceback.print_exc()
                yield _event("scene", "error", title="Scene graph",
                             message=f"{type(e).__name__}: {e}")

        # 11. LLM action decision (canonical — reads SAFETY_RULES.md) -------
        yield _event("decision", "running", title="Navigation decision",
                     message="GPT-4o-mini applies SAFETY_RULES.md to the scene graph...")

        scene_text = (scene_graph_payload or {}).get("text") \
                     or _detection_summary(enriched or [])

        action_result = None
        safety_rules_path = ROOT / "action_module" / "SAFETY_RULES.md"
        if llm_client_shared is not None and safety_rules_path.exists():
            import tempfile, os as _os
            tmp_path = None
            try:
                from llm_module.llm import analyse_with_scene_graph
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil.save(tmp_path, format="PNG")
                action_result = analyse_with_scene_graph(
                    client=llm_client_shared,
                    original_path=Path(tmp_path),
                    scene_graph_text=scene_text,
                    safety_rules=safety_rules_path.read_text(encoding="utf-8"),
                )
            except Exception as e:
                traceback.print_exc()
                action_result = None
            finally:
                if tmp_path:
                    try:
                        _os.unlink(tmp_path)
                    except OSError:
                        pass

        if action_result is None:
            # Fallback to heuristic critic
            fallback = _make_decision(enriched or [])
            action_result = {
                "action": fallback.get("label", "CONTINUE"),
                "confidence": 0.55,
                "reasoning": "; ".join(fallback.get("reasons", [])) or "Heuristic fallback.",
            }

        action = action_result.get("action", "CONTINUE").upper()
        conf = float(action_result.get("confidence", 0.5))
        reasoning = action_result.get("reasoning", "")
        color_map = {"STOP": "#ef4444", "SLOW": "#f59e0b", "CONTINUE": "#10b981"}

        # ── Secondary decision sources: LLMActor + GraphClassifier ─────────
        # Run in parallel (both are independent of each other and of the above)
        # so we can present a consensus view in the UI.
        sources = {
            "llm_rules": {
                "label": "LLM + SAFETY_RULES.md",
                "action": action,
                "confidence": conf,
                "reasoning": reasoning,
                "probabilities": None,
            }
        }

        # LLMActor — gives us action probabilities + entropy
        stem_for_logs = f"webui_{int(time.time()*1000)}" if False else "webui"
        try:
            import time as _time
            stem_for_logs = f"webui_{int(_time.time())}"
        except Exception:
            pass

        try:
            from llm_action_module.actor import LLMActor
            actor = _cache.get("llm_actor")
            if actor is None:
                actor = LLMActor()
                _cache["llm_actor"] = actor

            # LLMActor expects optional image paths + depth/pointcloud arrays.
            import tempfile as _tempfile
            seg_tmp = None
            depth_tmp = None
            try:
                if sam_results:
                    seg_img = Image.fromarray(_overlay_masks_on(base_bgr, sam_results)
                                              [..., ::-1])  # BGR→RGB
                    seg_tmp = _tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    seg_img.save(seg_tmp)
                if corrected is not None:
                    heat_bgr = _depth_blend_on(base_bgr, corrected)
                    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
                    depth_tmp = _tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    Image.fromarray(heat_rgb).save(depth_tmp)

                actor_result = actor.query(
                    scene_graph_text=scene_text,
                    seg_img_path=Path(seg_tmp) if seg_tmp else None,
                    depth_img_path=Path(depth_tmp) if depth_tmp else None,
                    depth_arr=corrected,
                    point_cloud=None,
                    stem=stem_for_logs,
                )
                sources["llm_actor"] = {
                    "label": "LLM Actor",
                    "action": actor_result.get("action"),
                    "confidence": float(actor_result.get("confidence", 0.5)),
                    "reasoning": actor_result.get("reasoning", ""),
                    "probabilities": actor_result.get("probabilities"),
                    "entropy": float(actor_result.get("entropy", 0.0)),
                }
            finally:
                for p in (seg_tmp, depth_tmp):
                    if p:
                        try: _os.unlink(p)
                        except Exception: pass
        except Exception as e:
            traceback.print_exc()
            sources["llm_actor"] = {"label": "LLM Actor", "error": str(e)}

        # GraphClassifier — 43-dim feature vector → rule or ML probabilities
        try:
            from action_module.graph_classifier import GraphClassifier
            clf = _cache.get("graph_classifier")
            if clf is None:
                clf = GraphClassifier()
                _cache["graph_classifier"] = clf
            clf_result = clf.predict(enriched or [], None, pil)
            sources["graph_classifier"] = {
                "label": "Graph Classifier",
                "action": clf_result.get("action"),
                "confidence": float(clf_result.get("confidence", 0.5)),
                "probabilities": clf_result.get("probabilities"),
                "top_features": clf_result.get("top_features"),
                "classifier_source": clf_result.get("source"),
            }
        except Exception as e:
            traceback.print_exc()
            sources["graph_classifier"] = {"label": "Graph Classifier", "error": str(e)}

        # Consensus: do the sources that returned an action agree?
        actions_seen = [s.get("action") for s in sources.values() if s.get("action")]
        consensus = (len(set(actions_seen)) == 1) if actions_seen else False

        yield _event("decision", "ok", title="Navigation decision",
                     label=action,
                     color=color_map.get(action, "#10b981"),
                     reasons=[reasoning] if reasoning else ["No reasoning provided."],
                     confidence=conf,
                     sources=sources,
                     consensus=consensus,
                     stem=stem_for_logs,
                     meta={"action": action, "confidence": conf,
                           "reasoning": reasoning,
                           "sources": sources,
                           "consensus": consensus,
                           "stem": stem_for_logs,
                           "enriched": [
                               {k: v for k, v in d.items()
                                if not isinstance(v, np.ndarray)}
                               for d in enriched],
                           "scene_graph": scene_graph_payload})

        yield _event("done", "ok")

    async def serialized_gen():
        async with _INFERENCE_LOCK:
            for chunk in gen():
                yield chunk

    return StreamingResponse(serialized_gen(), media_type="application/x-ndjson")


def _log_gpu_status():
    try:
        import torch as _t
        if _t.cuda.is_available():
            n = _t.cuda.device_count()
            names = [_t.cuda.get_device_name(i) for i in range(n)]
            print(f"[app] CUDA available — {n}× {', '.join(names)}")
        elif getattr(_t.backends, "mps", None) and _t.backends.mps.is_available():
            print("[app] Apple MPS backend available")
        else:
            print("[app] No accelerator detected — running on CPU")
    except Exception as e:
        print(f"[app] GPU probe failed: {e}")

_log_gpu_status()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
