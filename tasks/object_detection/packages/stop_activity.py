from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    """Return True when a duckie/truck is close enough to stop for.

    This is used by the object_detection task. The final_project task has
    similar logic inside TrafficRuleManager, but keeping this implemented
    prevents TODO crashes if the original object detection server is used.
    """
    if not detections:
        return False, ''

    image_area = max(1, img_size * img_size)
    for bbox, score, cls_id in detections:
        # Stop for duckies and trucks. Do not stop just because a sign is visible.
        if cls_id not in (0, 1):
            continue

        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area_ratio = (w * h) / image_area
        cx = (x1 + x2) / 2

        centered = img_size * 0.15 <= cx <= img_size * 0.85
        close_enough = area_ratio >= 0.035

        if centered and close_enough:
            name = class_names.get(cls_id, str(cls_id))
            return True, f'{name} close: score={score:.2f}, area={area_ratio:.3f}'

    return False, ''
