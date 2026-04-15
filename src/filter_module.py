"""
Phase 3: Image Filtering

Discards images where every detection maps to BACKGROUND (risk=1).
An image is "interesting" if it has at least one HUMAN, VEHICLE, OBSTACLE,
or SAFETY_MARKER detection above the confidence threshold.
"""

from src.label_ontology import get_risk_group


def is_interesting(detections: list[dict], conf_threshold: float = 0.25) -> bool:
    """
    Return True if the image contains at least one relevant object.

    Args:
        detections:     List of dicts with "label" and "score" keys.
        conf_threshold: Minimum confidence to count a detection.
    """
    for det in detections:
        if det.get("score", 0) < conf_threshold:
            continue
        if get_risk_group(det["label"])["group"] != "BACKGROUND":
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
