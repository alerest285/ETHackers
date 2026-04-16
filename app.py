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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MAX_W = 960         # cap width of the stage images streamed to browser
POINTCLOUD_W = 300  # target width of the point cloud grid
FOCAL_SCALE = 0.8   # matches lift_3d.py


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
        _cache["depth"] = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
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


@app.get("/")
def index():
    return FileResponse("static/index.html")


def _event(stage: str, status: str, **kw) -> bytes:
    return (json.dumps({"stage": stage, "status": status, **kw}) + "\n").encode()


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

        # 2. YOLO --------------------------------------------------------------
        yield _event("yolo", "running", title="YOLO detection",
                     message="Running YOLO...")
        yolo_dets: list[dict] = []
        try:
            from src.label_ontology import map_detections
            from segment_module.segment import detect_image
            yolo = _get_yolo()
            raw_dets = detect_image(yolo, pil, score_threshold=0.25)
            yolo_dets = map_detections(raw_dets)
            img = _draw_detections_on(base_bgr, yolo_dets)
            yield _event("yolo", "ok", title="YOLO detection",
                         image=_bgr_to_b64(img),
                         meta={"count": len(yolo_dets),
                               "detections": [
                                   {"label": d["label"], "score": d["score"],
                                    "risk_group": d.get("risk_group")}
                                   for d in yolo_dets]})
        except Exception as e:
            traceback.print_exc()
            yield _event("yolo", "error", title="YOLO detection",
                         message=f"{type(e).__name__}: {e}")

        # 3. LLM prompt --------------------------------------------------------
        llm_prompt = None
        if want_gdino:
            yield _event("llm_prompt", "running", title="LLM → object prompt",
                         message="Asking GPT to enumerate scene objects...")
            try:
                from segment_module.llm_objects import load_client, get_detection_prompt
                client = load_client()
                llm_prompt = get_detection_prompt(client, pil)
                yield _event("llm_prompt", "ok", title="LLM → object prompt",
                             text=llm_prompt)
            except Exception as e:
                yield _event("llm_prompt", "error", title="LLM → object prompt",
                             message=f"{type(e).__name__}: {e}")

        # 4. Grounding DINO ----------------------------------------------------
        gdino_dets: list[dict] = []
        if want_gdino and llm_prompt:
            yield _event("gdino", "running", title="Grounding DINO",
                         message="Loading Grounding DINO...")
            try:
                from segment_module.grounding_dino import detect as gdino_detect
                gdino = _get_gdino()
                gdino_dets = gdino_detect(gdino, pil, llm_prompt, score_threshold=0.30)
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

        dets_for_depth = gdino_dets if gdino_dets else yolo_dets

        # 5. SAM 2 -------------------------------------------------------------
        sam_results: list[dict] = []
        if want_sam and dets_for_depth:
            yield _event("sam2", "running", title="SAM 2",
                         message="Loading SAM 2...")
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
            except Exception as e:
                traceback.print_exc()
                yield _event("sam2", "error", title="SAM 2",
                             message=f"{type(e).__name__}: {e}")

        # 6. DepthAnything -----------------------------------------------------
        yield _event("depth", "running", title="Depth Anything V2",
                     message="Running depth estimation...")
        depth_norm = None
        try:
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

        # 7. Fused depth -------------------------------------------------------
        enriched: list[dict] = []
        corrected = depth_norm
        if depth_norm is not None and dets_for_depth:
            yield _event("fused", "running", title="Fused depth heuristics",
                         message="Fusing depth signals...")
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

        # 8. scene_data (for 3D rendering) ------------------------------------
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
                builder = scene_mod.SceneGraphBuilder(point_cloud_step=6)
                graph = builder.process(
                    depth_map=corrected,
                    detections=enriched or dets_for_depth,
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

                # send as PNGs
                ok1, buf1 = cv2.imencode(".png", depth_u8)
                ok2, buf2 = cv2.imencode(".png", cv2.cvtColor(orig_small, cv2.COLOR_RGB2BGR))
                # RGBA for tint — cv2 expects BGRA
                tint_bgra = cv2.cvtColor(tint_small, cv2.COLOR_RGBA2BGRA)
                ok3, buf3 = cv2.imencode(".png", tint_bgra)

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

        # 9. Decision ----------------------------------------------------------
        decision = _make_decision(enriched or [])
        yield _event("decision", "ok", title="Navigation decision",
                     **decision,
                     meta={"enriched": enriched,
                           "scene_graph": scene_graph_payload})

        yield _event("done", "ok")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
