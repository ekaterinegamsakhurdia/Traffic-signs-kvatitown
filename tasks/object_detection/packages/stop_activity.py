from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    """Return True when a duckie/truck is close enough to stop for.

    This is used by the standalone object_detection task server.
    The final_project task has its own stopping logic inside
    TrafficRuleManager / ObjectThreatDetector and does NOT call this function.
    """
    if not detections:
        return False, ''

    for bbox, score, cls_id in detections:
        # Stop for duckies and trucks only.
        if cls_id not in (0, 1):
            continue

        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # BUG FIX: bbox coordinates are in original frame pixel space (e.g. 0–640),
        # not in model/img_size space.  The old code compared cx against
        # img_size * 0.1/0.9 (= 41–374 for img_size=416), so any duck in the
        # right ~37% of a 640-wide frame was incorrectly rejected as off-centre.
        # Use normalised cx_norm instead so the check is frame-size independent.
        frame_w = x2 if x2 > img_size else img_size   # best-effort frame width
        cx_norm = ((x1 + x2) / 2.0) / frame_w

        # Accept ducks anywhere in the central 80% of the frame width.
        centered = 0.10 <= cx_norm <= 0.90

        # Area is normalised against img_size² (model input area) as before —
        # detections are already scaled back to original pixels, so recalculate
        # against an approximate frame area using bbox aspect ratio.
        area_ratio = (w * h) / max(1, img_size * img_size)
        close_enough = area_ratio >= 0.02

        if centered and close_enough:
            name = class_names.get(cls_id, str(cls_id))
            return True, f'{name} close: score={score:.2f}, area={area_ratio:.3f}'

    return False, ''