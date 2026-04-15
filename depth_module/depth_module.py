"""
Depth Anything V2 module for the ETHackers pipeline.

Pipeline position:
  Raw Image → Detection → Labeling+Segmentation → Filtering → [THIS MODULE] → Decision

Input:
  - image path (or BGR numpy array)
  - filtered detections: list of dicts with 'bbox' [x, y, w, h], 'category_id', 'score', 'image_id'

Output:
  - same detections, each annotated with:
      'depth_score'     : float in [0.0, 1.0], 0 = closest, 1 = farthest (normalized per image)
      'proximity_label' : str, one of 'CLOSE' | 'MEDIUM' | 'FAR'
      'raw_depth_median': float, raw (unnormalized) median depth in the bbox region
      'path_zone'       : str, one of 'CENTER' | 'PERIPHERAL'
                          CENTER = bbox x-center falls in the middle third of the image
                          (objects in CENTER + CLOSE → candidate STOP; PERIPHERAL + CLOSE → SLOW)
"""

import sys
import os
import numpy as np
import cv2
import torch

# Depth Anything V2 lives in the sibling directory cloned from GitHub
_HERE = os.path.dirname(os.path.abspath(__file__))
_DA2_ROOT = os.path.join(_HERE, "Depth-Anything-V2")
if _DA2_ROOT not in sys.path:
    sys.path.insert(0, _DA2_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Model config table — mirrors the official README exactly
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,   96,   192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,   192,  384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256,  512,  1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# Proximity thresholds on the *normalized* depth score (0 = closest, 1 = farthest)
# Tune these once you have a feel for the dataset.
PROXIMITY_THRESHOLDS = {
    "CLOSE":  0.35,   # depth_score <= 0.35  → CLOSE
    "MEDIUM": 0.65,   # depth_score <= 0.65  → MEDIUM
    # else              → FAR
}


class DepthModule:
    """
    Wraps Depth Anything V2 and enriches filtered detections with depth estimates.

    Usage
    -----
    dm = DepthModule(checkpoint_path="depth_module/checkpoints/depth_anything_v2_vitl.pth")
    results = dm.process(image_bgr, detections)
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder: str = "vitl",
        device: str | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DepthAnythingV2 | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load weights from checkpoint.  Call once before any inference."""
        cfg = MODEL_CONFIGS[self.encoder]
        self.model = DepthAnythingV2(**cfg)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    def infer_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run the model on a single BGR image (as returned by cv2.imread).

        Returns
        -------
        depth : np.ndarray, shape (H, W), dtype float32
            Raw relative depth map.  Larger values = farther away.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_depth().")
        return self.model.infer_image(image_bgr)  # official API

    def process(
        self,
        image_bgr: np.ndarray,
        detections: list[dict],
    ) -> list[dict]:
        """
        Annotate each detection with depth information.

        Parameters
        ----------
        image_bgr   : BGR numpy array (H, W, 3)
        detections  : list of dicts, each must have 'bbox': [x, y, w, h]

        Returns
        -------
        Enriched copies of the input dicts (originals are not mutated).
        """
        depth_map = self.infer_depth(image_bgr)
        depth_norm = _normalize_depth(depth_map)

        results = []
        for det in detections:
            det = dict(det)  # shallow copy — don't mutate caller's data
            x, y, w, h = [int(v) for v in det["bbox"]]

            # Clamp to image bounds
            H, W = depth_map.shape
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, W), min(y + h, H)

            if x2 <= x1 or y2 <= y1:
                # Degenerate bbox — assign neutral depth
                raw_median  = float(np.median(depth_map))
                norm_median = float(np.median(depth_norm))
            else:
                roi_raw  = depth_map[y1:y2, x1:x2]
                roi_norm = depth_norm[y1:y2, x1:x2]
                raw_median  = float(np.median(roi_raw))
                norm_median = float(np.median(roi_norm))

            # Path-zone: is this object in the robot's forward path?
            img_W = depth_map.shape[1]
            bbox_cx = x + w / 2.0

            det["raw_depth_median"] = raw_median
            det["depth_score"]      = norm_median
            det["proximity_label"]  = _proximity_label(norm_median)
            det["path_zone"]        = _path_zone(bbox_cx, img_W)
            results.append(det)

        return results

    def process_from_path(
        self,
        image_path: str,
        detections: list[dict],
    ) -> list[dict]:
        """Convenience wrapper that loads the image then calls process()."""
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.process(image_bgr, detections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Min-max normalize a depth map to [0, 1] per image."""
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map, dtype=np.float32)
    return ((depth_map - d_min) / (d_max - d_min)).astype(np.float32)


def _proximity_label(depth_score: float) -> str:
    if depth_score <= PROXIMITY_THRESHOLDS["CLOSE"]:
        return "CLOSE"
    if depth_score <= PROXIMITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "FAR"


def _path_zone(bbox_cx: float, img_width: int) -> str:
    """
    Is the object in the robot's forward path?

    The middle third of the image is treated as the path center.
    This is a heuristic — the robot's camera is assumed to face forward
    and the path is roughly the central vertical corridor.
    """
    left  = img_width / 3.0
    right = 2.0 * img_width / 3.0
    return "CENTER" if left <= bbox_cx <= right else "PERIPHERAL"
