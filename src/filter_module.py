"""
Phase 3: Image Filtering

Discards images where every detection maps to a non-hazard group.
An image is "interesting" if it has at least one HUMAN, VEHICLE, OBSTACLE,
or SAFETY_MARKER detection above the confidence threshold.

SURFACE (risk=0) is excluded along with BACKGROUND — the robot drives ON
surfaces, so their presence alone does not make a scene actionable.
"""

from src.label_ontology import get_risk_group

# Groups that carry no safety action on their own
_NON_HAZARD = {"BACKGROUND", "SURFACE"}


def is_interesting(detections: list[dict], conf_threshold: float = 0.25) -> bool:
    """
    Return True if the image contains at least one actionable hazard object.

    Args:
        detections:     List of dicts with "label" and "score" keys.
        conf_threshold: Minimum confidence to count a detection.
    """
    for det in detections:
        if det.get("score", 0) < conf_threshold:
            continue
        if get_risk_group(det["label"])["group"] not in _NON_HAZARD:
            return True
    return False


def filter_images(
    image_detections: dict[str, list[dict]],
    conf_threshold: float = 0.25,
) -> tuple[list[str], list[str]]:
    """
    Split image names into kept and discarded lists.

    Args:
        image_detections: {image_name: [detections]} mapping.
        conf_threshold:   Min confidence to count a detection.

    Returns:
        (kept, discarded) — lists of image names.
    """
    kept      = []
    discarded = []

    for name, dets in image_detections.items():
        if is_interesting(dets, conf_threshold):
            kept.append(name)
        else:
            discarded.append(name)

    return kept, discarded
