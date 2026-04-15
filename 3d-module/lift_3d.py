"""
3D Lift — Point Cloud + Scene Graph

Takes a normalized depth map and a list of detections and:
  1. Unprojects the depth map to a sparse 3D point cloud (pinhole camera model).
  2. Extracts per-object 3D clusters (centroid, nearest point, extent).
  3. Builds a scene graph — nodes (objects) + edges (spatial relationships).
  4. Serializes the scene graph to structured text for LLM consumption.

Depth convention (input):
  depth_map must be normalized to [0, 1] where 0 = closest, 1 = farthest.
  This matches the depth_score produced by depth_module.py and src/pipeline.py.
  If your depth map is in "disparity" convention (higher = closer), invert it first:
      depth_map_norm = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)

Camera model (estimated — no calibration required):
  fx = fy = max(W, H) * 0.8
  cx = W / 2,  cy = H / 2

Output units are in "normalized depth units" (NDU), not metric.
Spatial relationships (closer/farther, left/right) are directionally correct.

Usage:
  from depth_module.lift_3d import SceneGraphBuilder
  builder = SceneGraphBuilder()
  result  = builder.process(depth_map, detections, img_w=W, img_h=H, image_id="frame_001")
  print(result["text"])

  # Save to JSON
  builder.save(result, output_dir=Path("outputs/scene_graphs"))

CLI:
  python depth_module/lift_3d.py \\
      --depth-npy  outputs/depth/frame_001.npy \\
      --detections outputs/detections/frame_001.json \\
      --image-id   frame_001 \\
      --output-dir outputs/scene_graphs
"""

import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Focal-length multiplier for the estimated pinhole model
FOCAL_SCALE = 0.8

# Sparse point cloud: sample every Nth pixel (lower = denser but slower)
DEFAULT_STEP = 4

# Thresholds for classifying relative position as "in front of" vs "beside"
DEPTH_DOMINATES_THRESHOLD = 0.08   # if |dZ| > this * scene_depth_range, use front/behind
BLOCKING_Z_MARGIN         = 0.05   # A blocks B if A.nearest_z < B.centroid_z + this margin

# Proximity thresholds (must stay in sync with depth_module.py / src/depth_overlay.py)
PROXIMITY_THRESHOLDS = {"CLOSE": 0.35, "MEDIUM": 0.65}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node3D:
    """One detected object, enriched with 3D spatial information."""
    id:              int
    label:           str
    risk_group:      str
    risk_score:      int
    score:           float          # detection confidence
    depth_score:     float          # 0 = closest, 1 = farthest
    proximity_label: str
    path_zone:       str            # CENTER | PERIPHERAL
    box_2d:          list           # [x1, y1, x2, y2]
    centroid_3d:     list           # [X, Y, Z] in NDU
    nearest_z:       float          # closest Z within the object's region
    extent_3d:       list           # [width_3d, height_3d, depth_3d] in NDU


@dataclass
class Edge3D:
    """Directed spatial relationship from source to target node."""
    from_id:           int
    to_id:             int
    distance_3d:       float        # Euclidean distance between centroids (NDU)
    relative_position: str          # "in front of" | "behind" | "to the left of" | "to the right of"
    blocking:          bool         # True if source is between camera and target in 2D projection


@dataclass
class SceneGraph:
    image_id:    str
    img_w:       int
    img_h:       int
    nodes:       list = field(default_factory=list)
    edges:       list = field(default_factory=list)
    text:        str  = ""
    n_points:    int  = 0           # total point-cloud points sampled


# ---------------------------------------------------------------------------
# Step 1 — Pinhole unproject
# ---------------------------------------------------------------------------

def lift_to_3d(
    depth_map: np.ndarray,
    img_w: int,
    img_h: int,
    step: int = DEFAULT_STEP,
) -> np.ndarray:
    """
    Unproject a (H, W) normalized depth map to a sparse (N, 3) point cloud.

    Parameters
    ----------
    depth_map : np.ndarray (H, W), float32, values in [0, 1]
        Normalized depth map, 0 = closest, 1 = farthest.
    img_w, img_h : int
        Image dimensions in pixels.
    step : int
        Stride for sparse sampling (every `step`-th pixel in each direction).

    Returns
    -------
    points : np.ndarray (N, 3), float32
        Each row is (X, Y, Z) in normalized depth units.
    """
    H, W = depth_map.shape
    fx = fy = max(W, H) * FOCAL_SCALE
    cx, cy = W / 2.0, H / 2.0

    # Build pixel-coordinate grids (sampled)
    us = np.arange(0, W, step, dtype=np.float32)
    vs = np.arange(0, H, step, dtype=np.float32)
    ug, vg = np.meshgrid(us, vs)          # both shape (Hs, Ws)

    # Sample depth at those pixel locations
    u_idx = ug.astype(np.int32).ravel()
    v_idx = vg.astype(np.int32).ravel()
    # Clip to valid range
    u_idx = np.clip(u_idx, 0, W - 1)
    v_idx = np.clip(v_idx, 0, H - 1)
    Z = depth_map[v_idx, u_idx]

    X = (u_idx - cx) * Z / fx
    Y = (v_idx - cy) * Z / fy

    return np.stack([X, Y, Z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 — Per-object 3D cluster
# ---------------------------------------------------------------------------

def _to_xyxy(det: dict) -> tuple:
    """Return (x1, y1, x2, y2) regardless of bbox format in the detection dict."""
    if "box" in det:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
    elif "bbox" in det:
        x, y, w, h = [int(v) for v in det["bbox"]]
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        raise KeyError("Detection dict must have 'box' [x1,y1,x2,y2] or 'bbox' [x,y,w,h]")
    return x1, y1, x2, y2


def get_object_cluster(
    depth_map: np.ndarray,
    det: dict,
    img_w: int,
    img_h: int,
    mask: np.ndarray | None = None,
) -> dict:
    """
    Extract the 3D cluster for a single detected object.

    Parameters
    ----------
    depth_map : (H, W) float32, 0 = closest
    det       : detection dict (must have 'box' or 'bbox')
    mask      : optional (H, W) bool array — pixel mask from SAM; falls back to bbox ROI
    img_w, img_h : image dimensions

    Returns
    -------
    dict with keys: centroid_3d, nearest_z, extent_3d, box_2d
    """
    H, W = depth_map.shape
    fx = fy = max(W, H) * FOCAL_SCALE
    cx, cy = W / 2.0, H / 2.0

    x1, y1, x2, y2 = _to_xyxy(det)
    x1 = max(x1, 0);  y1 = max(y1, 0)
    x2 = min(x2, W);  y2 = min(y2, H)

    if x2 <= x1 or y2 <= y1:
        # Degenerate box — fall back to image-centre point with median depth
        z_val = float(np.median(depth_map))
        return {
            "centroid_3d": [0.0, 0.0, z_val],
            "nearest_z":   z_val,
            "extent_3d":   [0.0, 0.0, 0.0],
            "box_2d":      [x1, y1, x2, y2],
        }

    # Build pixel grids for the object region
    us = np.arange(x1, x2, dtype=np.float32)
    vs = np.arange(y1, y2, dtype=np.float32)
    ug, vg = np.meshgrid(us, vs)
    u_idx = ug.astype(np.int32).ravel()
    v_idx = vg.astype(np.int32).ravel()

    if mask is not None:
        # Filter to pixels inside the SAM mask
        inside = mask[v_idx, u_idx]
        if inside.sum() > 0:
            u_idx = u_idx[inside]
            v_idx = v_idx[inside]

    Z = depth_map[v_idx, u_idx]
    X = (u_idx.astype(np.float32) - cx) * Z / fx
    Y = (v_idx.astype(np.float32) - cy) * Z / fy

    centroid = [float(X.mean()), float(Y.mean()), float(Z.mean())]
    nearest_z = float(Z.min())
    extent = [
        float(X.max() - X.min()),
        float(Y.max() - Y.min()),
        float(Z.max() - Z.min()),
    ]

    return {
        "centroid_3d": centroid,
        "nearest_z":   nearest_z,
        "extent_3d":   extent,
        "box_2d":      [x1, y1, x2, y2],
    }


# ---------------------------------------------------------------------------
# Step 3 — Scene graph (nodes + edges)
# ---------------------------------------------------------------------------

def _proximity_label(depth_score: float) -> str:
    if depth_score <= PROXIMITY_THRESHOLDS["CLOSE"]:
        return "CLOSE"
    if depth_score <= PROXIMITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "FAR"


def _path_zone(box_2d: list, img_w: int) -> str:
    """CENTER if the bbox x-centre falls in the middle third."""
    x1, _, x2, _ = box_2d
    cx = (x1 + x2) / 2.0
    return "CENTER" if img_w / 3.0 <= cx <= 2.0 * img_w / 3.0 else "PERIPHERAL"


def _bboxes_overlap_2d(a: list, b: list) -> bool:
    """Return True if two [x1,y1,x2,y2] boxes have any intersection."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _relative_position(a: Node3D, b: Node3D, scene_depth_range: float) -> str:
    """
    Compute how A is positioned relative to B from the camera's perspective.

    Priority: if the depth difference dominates, say "in front of" / "behind".
    Otherwise, use the lateral (X) offset: "to the left of" / "to the right of".
    """
    dZ = a.centroid_3d[2] - b.centroid_3d[2]
    dX = a.centroid_3d[0] - b.centroid_3d[0]

    threshold = DEPTH_DOMINATES_THRESHOLD * max(scene_depth_range, 0.1)

    if abs(dZ) >= threshold:
        return "in front of" if dZ < 0 else "behind"
    else:
        return "to the left of" if dX < 0 else "to the right of"


def build_scene_graph(
    detections: list[dict],
    depth_map: np.ndarray,
    img_w: int,
    img_h: int,
    masks: list | None = None,
) -> tuple[list[Node3D], list[Edge3D]]:
    """
    Build nodes and edges from a list of enriched detections + depth map.

    Parameters
    ----------
    detections : list of dicts (each must have label, risk_group, risk_score,
                 score, depth_score, proximity_label, and 'box' or 'bbox')
    depth_map  : (H, W) float32, normalized 0=closest
    img_w, img_h : image dimensions
    masks      : optional list of (H, W) bool arrays, one per detection (from SAM)

    Returns
    -------
    (nodes, edges)
    """
    nodes: list[Node3D] = []

    # --- Build nodes ---
    for i, det in enumerate(detections):
        mask = masks[i] if masks is not None and i < len(masks) else None
        cluster = get_object_cluster(depth_map, det, img_w, img_h, mask=mask)

        depth_score = float(det.get("depth_score", cluster["centroid_3d"][2]))
        prox = det.get("proximity_label") or _proximity_label(depth_score)
        zone = det.get("path_zone") or _path_zone(cluster["box_2d"], img_w)

        node = Node3D(
            id              = i + 1,
            label           = det.get("label", "unknown"),
            risk_group      = det.get("risk_group", "BACKGROUND"),
            risk_score      = int(det.get("risk_score", 1)),
            score           = float(det.get("score", 0.0)),
            depth_score     = depth_score,
            proximity_label = prox,
            path_zone       = zone,
            box_2d          = cluster["box_2d"],
            centroid_3d     = cluster["centroid_3d"],
            nearest_z       = cluster["nearest_z"],
            extent_3d       = cluster["extent_3d"],
        )
        nodes.append(node)

    # Z-range for relative-position thresholding
    all_z = [n.centroid_3d[2] for n in nodes]
    scene_depth_range = (max(all_z) - min(all_z)) if len(all_z) > 1 else 1.0

    # --- Build edges (all ordered pairs) ---
    edges: list[Edge3D] = []
    for a in nodes:
        for b in nodes:
            if a.id == b.id:
                continue

            dX = a.centroid_3d[0] - b.centroid_3d[0]
            dY = a.centroid_3d[1] - b.centroid_3d[1]
            dZ = a.centroid_3d[2] - b.centroid_3d[2]
            dist_3d = math.sqrt(dX**2 + dY**2 + dZ**2)

            rel_pos = _relative_position(a, b, scene_depth_range)

            # A blocks B: A is closer to camera AND their 2D projections overlap
            blocking = (
                a.nearest_z < b.centroid_3d[2] - BLOCKING_Z_MARGIN
                and _bboxes_overlap_2d(a.box_2d, b.box_2d)
            )

            edges.append(Edge3D(
                from_id           = a.id,
                to_id             = b.id,
                distance_3d       = round(dist_3d, 4),
                relative_position = rel_pos,
                blocking          = blocking,
            ))

    return nodes, edges


# ---------------------------------------------------------------------------
# Step 4 — Text serialization
# ---------------------------------------------------------------------------

def serialize_scene_graph(nodes: list[Node3D], edges: list[Edge3D]) -> str:
    """
    Produce a compact, human-readable scene description for the LLM.

    Format:
      Scene (N objects detected):
        [1] HUMAN 'person' — CLOSE, CENTER path, depth=0.12, pos=(0.10, 0.00, 0.12)
        ...

      Spatial relationships:
        [1] person is 0.30 units in front of [2] forklift
        [1] person is blocking [3] cone
        ...
    """
    lines = [f"Scene ({len(nodes)} object{'s' if len(nodes) != 1 else ''} detected):"]

    for n in sorted(nodes, key=lambda x: x.risk_score, reverse=True):
        cx, cy, cz = [round(v, 3) for v in n.centroid_3d]
        lines.append(
            f"  [{n.id}] {n.risk_group} '{n.label}' "
            f"(conf={n.score:.2f}) — {n.proximity_label}, {n.path_zone} path, "
            f"depth={n.depth_score:.3f}, pos=({cx}, {cy}, {cz} NDU)"
        )

    lines.append("")
    lines.append("Spatial relationships:")

    # Include only blocking edges + one representative directional edge per pair
    seen_pairs: set[tuple] = set()
    rel_lines = []

    for e in edges:
        if e.blocking:
            a_node = next(n for n in nodes if n.id == e.from_id)
            b_node = next(n for n in nodes if n.id == e.to_id)
            rel_lines.append(
                f"  [{e.from_id}] {a_node.label} is blocking [{e.to_id}] {b_node.label} "
                f"(closer and overlapping in 2D)"
            )

        pair = (min(e.from_id, e.to_id), max(e.from_id, e.to_id))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            a_node = next(n for n in nodes if n.id == e.from_id)
            b_node = next(n for n in nodes if n.id == e.to_id)
            rel_lines.append(
                f"  [{e.from_id}] {a_node.label} is {e.distance_3d:.3f} units "
                f"{e.relative_position} [{e.to_id}] {b_node.label}"
            )

    if rel_lines:
        lines.extend(rel_lines)
    else:
        lines.append("  (no objects detected — scene is clear)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5 — 3D visualizer
# ---------------------------------------------------------------------------

# Colors per risk group (used for node markers)
_GROUP_COLORS = {
    "HUMAN":         "#e74c3c",   # red
    "VEHICLE":       "#e67e22",   # orange
    "OBSTACLE":      "#3498db",   # blue
    "SAFETY_MARKER": "#2ecc71",   # green
    "BACKGROUND":    "#95a5a6",   # gray
}

# Outline color per proximity (ring around node marker)
_PROX_EDGE_COLOR = {
    "CLOSE":  "#ff0000",
    "MEDIUM": "#ff8800",
    "FAR":    "#00cc44",
}


def visualize_3d(
    graph: SceneGraph,
    depth_map: np.ndarray,
    step: int = DEFAULT_STEP,
    max_cloud_points: int = 8000,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    Open a separate window with two panels:
      LEFT  — 3D perspective view: full point cloud + labelled object nodes + edges
      RIGHT — top-down (bird's-eye) view: same data projected onto the X-Z plane,
              with a shaded CENTER corridor showing the robot's forward path

    Point cloud is colored by depth (plasma colormap: bright = close, dark = far).
    Object nodes are colored by risk group; their edge-ring color shows proximity.
    Blocking edges are drawn as thick dashed red lines; other edges as thin grey lines.

    Parameters
    ----------
    graph           : SceneGraph returned by SceneGraphBuilder.process()
    depth_map       : same (H, W) float32 array used to build the graph
    step            : sampling stride for the point cloud (default = DEFAULT_STEP)
    max_cloud_points: cap on scatter points for render performance
    save_path       : if set, also save figure to this path (PNG/PDF)
    show            : call plt.show() — set False for headless/testing
    """
    # ── 1. Build point cloud ────────────────────────────────────────────────
    cloud = lift_to_3d(depth_map, graph.img_w, graph.img_h, step=step)

    # Subsample for render performance
    if len(cloud) > max_cloud_points:
        idx   = np.random.choice(len(cloud), max_cloud_points, replace=False)
        cloud = cloud[idx]

    # cloud columns: X (right+), Y (down+, image coords), Z (far+)
    cx_cloud = cloud[:, 0]
    cy_cloud = cloud[:, 1]
    cz_cloud = cloud[:, 2]

    # ── 2. Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8), facecolor="#1a1a2e")
    fig.suptitle(
        f"3D Scene Map  —  {graph.image_id}  ({len(graph.nodes)} objects)",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    for ax in (ax3d, ax2d):
        ax.set_facecolor("#0f0f23")

    # ── 3. 3D perspective ───────────────────────────────────────────────────
    # Axes: X = left-right,  Z = depth (mapped to y-axis),  -Y = up (mapped to z-axis)
    sc = ax3d.scatter(
        cx_cloud, cz_cloud, -cy_cloud,
        c=cz_cloud, cmap="plasma_r",   # bright = close (low Z)
        s=1, alpha=0.25, linewidths=0,
    )

    # Draw nodes
    for node in graph.nodes:
        nx, ny, nz = node.centroid_3d
        col = _GROUP_COLORS.get(node.risk_group, "#ffffff")
        ec  = _PROX_EDGE_COLOR.get(node.proximity_label, "#ffffff")

        ax3d.scatter(
            [nx], [nz], [-ny],
            c=col, s=220, edgecolors=ec, linewidths=2.5, zorder=5,
        )
        ax3d.text(
            nx, nz, -ny + 0.03,
            f"[{node.id}] {node.label}\n{node.proximity_label}",
            color="white", fontsize=7, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="#00000099", ec="none"),
        )

    # Draw edges
    node_by_id = {n.id: n for n in graph.nodes}
    drawn_pairs: set[tuple] = set()
    for e in graph.edges:
        pair = (min(e.from_id, e.to_id), max(e.from_id, e.to_id))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        a = node_by_id[e.from_id]
        b = node_by_id[e.to_id]
        ax3d.plot(
            [a.centroid_3d[0], b.centroid_3d[0]],
            [a.centroid_3d[2], b.centroid_3d[2]],
            [-a.centroid_3d[1], -b.centroid_3d[1]],
            color="#ff4444" if e.blocking else "#555577",
            linewidth=2.0 if e.blocking else 0.8,
            linestyle="--" if e.blocking else "-",
            alpha=0.9 if e.blocking else 0.5,
        )

    ax3d.set_xlabel("X (right +)", color="#aaaacc", labelpad=6)
    ax3d.set_ylabel("Z (depth)", color="#aaaacc", labelpad=6)
    ax3d.set_zlabel("Up", color="#aaaacc", labelpad=6)
    ax3d.tick_params(colors="#666688", labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333355")
    ax3d.yaxis.pane.set_edgecolor("#333355")
    ax3d.zaxis.pane.set_edgecolor("#333355")
    ax3d.view_init(elev=25, azim=-65)
    ax3d.set_title("3D Perspective", color="#ccccee", fontsize=10, pad=8)

    # ── 4. Top-down (bird's-eye) view ────────────────────────────────────────
    # X = left-right,  Z = depth going upward
    ax2d.scatter(
        cx_cloud, cz_cloud,
        c=cz_cloud, cmap="plasma_r",
        s=1, alpha=0.20, linewidths=0,
    )

    # Robot camera position
    ax2d.scatter([0], [0], marker="^", s=180, c="#00ffff",
                 zorder=10, label="Camera")
    ax2d.text(0.01, 0.01, "ROBOT", color="#00ffff", fontsize=8,
              transform=ax2d.transAxes)

    # CENTER path corridor: middle third of image → X range at depth Z
    # At depth Z, the centre-third image columns map to X = ±(W/6)*Z/fx
    # Since fx = max(W,H)*0.8 ≈ W*0.8 for landscape, corridor half-width ≈ Z/(6*0.8)
    z_vals = np.linspace(0, 1.0, 100)
    fw = graph.img_w
    fh = graph.img_h
    fx_est = max(fw, fh) * FOCAL_SCALE
    half_w = (fw / 6.0) * z_vals / fx_est
    ax2d.fill_betweenx(
        z_vals, -half_w, half_w,
        color="#00ff8844", linewidth=0, label="CENTER path",
    )
    ax2d.axvline(0, color="#444466", linewidth=0.5, linestyle=":")

    # Nodes
    for node in graph.nodes:
        nx, _, nz = node.centroid_3d
        col = _GROUP_COLORS.get(node.risk_group, "#ffffff")
        ec  = _PROX_EDGE_COLOR.get(node.proximity_label, "#ffffff")
        ax2d.scatter([nx], [nz], c=col, s=200, edgecolors=ec,
                     linewidths=2.5, zorder=5)
        ax2d.annotate(
            f"[{node.id}] {node.label}\n{node.proximity_label}",
            xy=(nx, nz), xytext=(6, 6), textcoords="offset points",
            color="white", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="#00000099", ec="none"),
        )

    # Edges (top-down)
    drawn_pairs2: set[tuple] = set()
    for e in graph.edges:
        pair = (min(e.from_id, e.to_id), max(e.from_id, e.to_id))
        if pair in drawn_pairs2:
            continue
        drawn_pairs2.add(pair)
        a = node_by_id[e.from_id]
        b = node_by_id[e.to_id]
        ax2d.plot(
            [a.centroid_3d[0], b.centroid_3d[0]],
            [a.centroid_3d[2], b.centroid_3d[2]],
            color="#ff4444" if e.blocking else "#555577",
            linewidth=2.0 if e.blocking else 0.8,
            linestyle="--" if e.blocking else "-",
            alpha=0.9 if e.blocking else 0.5,
        )

    ax2d.set_xlabel("X  (left  ←  0  →  right)", color="#aaaacc")
    ax2d.set_ylabel("Z  (depth — camera at bottom)", color="#aaaacc")
    ax2d.tick_params(colors="#666688", labelsize=8)
    ax2d.spines[:].set_color("#333355")
    ax2d.set_title("Top-Down View", color="#ccccee", fontsize=10)

    # ── 5. Shared legend ────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=c, label=g)
        for g, c in _GROUP_COLORS.items() if g != "BACKGROUND"
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="#ff4444", lw=2, linestyle="--", label="Blocking edge"),
        plt.Line2D([0], [0], color="#555577", lw=1, label="Spatial edge"),
        mpatches.Patch(color="#00ff8844", label="CENTER path"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        fontsize=8, framealpha=0.2,
        labelcolor="white", facecolor="#1a1a2e",
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved viz: {save_path}")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# SceneGraphBuilder — main public class
# ---------------------------------------------------------------------------

class SceneGraphBuilder:
    """
    Orchestrates the full 3D lift → scene graph pipeline for one image.

    Example
    -------
    builder = SceneGraphBuilder(point_cloud_step=4)
    result  = builder.process(depth_map, detections, img_w=1280, img_h=720,
                               image_id="frame_001")
    builder.save(result, output_dir=Path("outputs/scene_graphs"))
    """

    def __init__(self, point_cloud_step: int = DEFAULT_STEP):
        self.step = point_cloud_step

    def process(
        self,
        depth_map:   np.ndarray,
        detections:  list[dict],
        img_w:       int,
        img_h:       int,
        image_id:    str = "unknown",
        masks:       list | None = None,
    ) -> SceneGraph:
        """
        Full pipeline: depth map + detections → SceneGraph.

        Parameters
        ----------
        depth_map   : (H, W) float32, normalized [0, 1], 0 = closest.
        detections  : list of enriched detection dicts (see module docstring).
        img_w, img_h: image dimensions in pixels.
        image_id    : identifier string used for output file naming.
        masks       : optional list of (H, W) bool arrays from SAM2.

        Returns
        -------
        SceneGraph dataclass with nodes, edges, text, n_points.
        """
        # 1. Lift entire frame to 3D (sparse point cloud)
        point_cloud = lift_to_3d(depth_map, img_w, img_h, step=self.step)

        # 2-3. Build nodes + edges
        nodes, edges = build_scene_graph(detections, depth_map, img_w, img_h, masks=masks)

        # 4. Serialize to text
        text = serialize_scene_graph(nodes, edges)

        graph = SceneGraph(
            image_id  = image_id,
            img_w     = img_w,
            img_h     = img_h,
            nodes     = nodes,
            edges     = edges,
            text      = text,
            n_points  = len(point_cloud),
        )
        return graph

    # ------------------------------------------------------------------
    def save(self, graph: SceneGraph, output_dir: Path) -> Path:
        """
        Save the scene graph to JSON + a .txt sidecar for the LLM.

        Files created:
          {output_dir}/{image_id}.json   — machine-readable graph
          {output_dir}/{image_id}.txt    — LLM-ready text (same as graph.text)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _node_to_dict(n: Node3D) -> dict:
            return {
                "id":              n.id,
                "label":           n.label,
                "risk_group":      n.risk_group,
                "risk_score":      n.risk_score,
                "score":           n.score,
                "depth_score":     n.depth_score,
                "proximity_label": n.proximity_label,
                "path_zone":       n.path_zone,
                "box_2d":          n.box_2d,
                "centroid_3d":     [round(v, 5) for v in n.centroid_3d],
                "nearest_z":       round(n.nearest_z, 5),
                "extent_3d":       [round(v, 5) for v in n.extent_3d],
            }

        def _edge_to_dict(e: Edge3D) -> dict:
            return {
                "from_id":           e.from_id,
                "to_id":             e.to_id,
                "distance_3d":       e.distance_3d,
                "relative_position": e.relative_position,
                "blocking":          e.blocking,
            }

        payload = {
            "image_id":   graph.image_id,
            "img_w":      graph.img_w,
            "img_h":      graph.img_h,
            "n_objects":  len(graph.nodes),
            "n_points":   graph.n_points,
            "nodes":      [_node_to_dict(n) for n in graph.nodes],
            "edges":      [_edge_to_dict(e) for e in graph.edges],
        }

        json_path = output_dir / f"{graph.image_id}.json"
        txt_path  = output_dir / f"{graph.image_id}.txt"

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        txt_path.write_text(graph.text)

        return json_path


# ---------------------------------------------------------------------------
# CLI — quick test / standalone use
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D lift + scene graph from a depth map + detections JSON"
    )
    p.add_argument("--depth-npy",    required=False,
                   help="Path to (H, W) float32 numpy depth array (.npy). "
                        "Must already be normalized [0, 1], 0 = closest.")
    p.add_argument("--detections",   required=False,
                   help="Path to detections JSON (pipeline output format).")
    p.add_argument("--image-id",     default="test_frame",
                   help="Identifier used for output filenames.")
    p.add_argument("--output-dir",   default="outputs/scene_graphs",
                   help="Directory to save the scene graph JSON + text.")
    p.add_argument("--step",         type=int, default=DEFAULT_STEP,
                   help="Point cloud sampling stride (default=4).")
    p.add_argument("--img-w",        type=int, default=0,
                   help="Image width in pixels (inferred from depth map if 0).")
    p.add_argument("--img-h",        type=int, default=0,
                   help="Image height in pixels (inferred from depth map if 0).")
    p.add_argument("--visualize",    action="store_true",
                   help="Open a 3D visualization window after building the scene graph.")
    p.add_argument("--save-viz",     default=None,
                   help="Save the visualization to this path (e.g. outputs/viz.png).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --- Load or generate a synthetic depth map ---
    if args.depth_npy and Path(args.depth_npy).exists():
        depth_map = np.load(args.depth_npy).astype(np.float32)
        H, W = depth_map.shape
    else:
        print("[INFO] No --depth-npy provided; using a synthetic 720×1280 gradient depth map.")
        H, W = 720, 1280
        # Gradient: left side close (0), right side far (1)
        depth_map = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))

    img_w = args.img_w if args.img_w > 0 else W
    img_h = args.img_h if args.img_h > 0 else H

    # --- Load or generate synthetic detections ---
    if args.detections and Path(args.detections).exists():
        with open(args.detections) as f:
            raw = json.load(f)
        # Support both {"detections": [...]} wrapper and plain list
        detections = raw.get("detections", raw) if isinstance(raw, dict) else raw
    else:
        print("[INFO] No --detections provided; using synthetic detections.")
        detections = [
            {
                "label": "person", "risk_group": "HUMAN", "risk_score": 5,
                "score": 0.92, "depth_score": 0.12, "proximity_label": "CLOSE",
                "path_zone": "CENTER",
                "box": [int(W * 0.38), int(H * 0.2), int(W * 0.62), int(H * 0.9)],
            },
            {
                "label": "forklift", "risk_group": "VEHICLE", "risk_score": 4,
                "score": 0.81, "depth_score": 0.45, "proximity_label": "MEDIUM",
                "path_zone": "PERIPHERAL",
                "box": [int(W * 0.65), int(H * 0.15), int(W * 0.95), int(H * 0.85)],
            },
            {
                "label": "cone", "risk_group": "SAFETY_MARKER", "risk_score": 2,
                "score": 0.73, "depth_score": 0.78, "proximity_label": "FAR",
                "path_zone": "CENTER",
                "box": [int(W * 0.43), int(H * 0.55), int(W * 0.57), int(H * 0.80)],
            },
        ]

    # --- Run pipeline ---
    builder = SceneGraphBuilder(point_cloud_step=args.step)
    graph   = builder.process(depth_map, detections, img_w, img_h, image_id=args.image_id)

    print("=" * 60)
    print(graph.text)
    print("=" * 60)
    print(f"\nPoint cloud: {graph.n_points:,} points sampled (step={args.step})")

    out_path = builder.save(graph, output_dir=Path(args.output_dir))
    print(f"Saved: {out_path}")
    print(f"Text:  {out_path.with_suffix('.txt')}")

    if args.visualize or args.save_viz:
        save_path = Path(args.save_viz) if args.save_viz else None
        visualize_3d(graph, depth_map, step=args.step,
                     save_path=save_path, show=args.visualize)
