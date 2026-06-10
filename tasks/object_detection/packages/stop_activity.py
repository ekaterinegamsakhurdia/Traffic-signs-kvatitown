from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:

    """Return (True, reason) to stop the bot, (False, '') to keep moving."""
    stop_y = img_size * 0.55
    for (x1, y1, x2, y2), score, cls_id in detections:
        if y2 > stop_y:
            return True, class_names.get(cls_id, str(cls_id)) + ' detected close ahead'
    return False, ''