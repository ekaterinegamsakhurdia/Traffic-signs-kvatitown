from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

# Horizontal: accept ducks/trucks in central 80% of frame width (cx 0.10–0.90).
# Widen if robot veers and ducks appear near edges.
CENTERED_MIN = 0.10
CENTERED_MAX = 0.90

# Vertical: bottom edge of bbox must be at or below this fraction of frame height.
# From logs, duck on road peaks at cy_bottom ~0.46 before passing under camera.
# 0.35 catches it with reaction time. Lower toward 0.25 for earlier trigger.
# Raise toward 0.50 only if getting false positives from distant objects.
LOWER_ZONE_THRESHOLD = 0.6

# Minimum bbox area relative to frame area.
# Small Duckietown duckies at trigger distance are ~0.001–0.004.
# Raise if stopping for far-away/tiny false detections.
MIN_AREA_FRACTION = 0.0005


def should_stop(
    detections: List[Detection],
    frame_w: int = 640,
    frame_h: int = 480,
) -> Tuple[bool, str]:
    """Return True when a duckie or truck is close enough to stop for.

    Called by VideoStream with the old (detections, img_size) signature —
    img_size lands in frame_w, frame_h defaults to 480. The cy_bottom gate
    is normalised so this approximation has negligible effect.

    Used by the standalone object_detection task only.
    The final_project uses ObjectThreatDetector and does NOT call this function.
    """
    if not detections:
        return False, ''

    frame_area = max(1, frame_w * frame_h)

    for bbox, score, cls_id in detections:
        if cls_id not in (0, 1):
            continue

        x1, y1, x2, y2 = bbox
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)

        cx_norm   = ((x1 + x2) / 2.0) / frame_w
        cy_bottom = y2 / frame_h
        area_frac = (bw * bh) / frame_area

        centered     = CENTERED_MIN <= cx_norm <= CENTERED_MAX
        in_lower     = cy_bottom >= LOWER_ZONE_THRESHOLD
        close_enough = area_frac >= MIN_AREA_FRACTION

        if centered and in_lower and close_enough:
            name = class_names.get(cls_id, str(cls_id))
            return True, (
                f'{name} in lower zone: score={score:.2f}, '
                f'area={area_frac:.4f}, cy_bottom={cy_bottom:.2f}'
            )

    return False, ''