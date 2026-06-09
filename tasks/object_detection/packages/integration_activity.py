from typing import Tuple
MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    return 1


def filter_by_classes(pred_class: int) -> bool:
    # 0 = duckie, 1 = truck, 2 = sign
    return pred_class in (0, 1, 2)


def filter_by_scores(score: float) -> bool:
    # Keep only confident detections. Tune from the web UI threshold too.
    return score >= 0.1


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    # Drop invalid/tiny boxes. bbox = (xmin, ymin, xmax, ymax) in pixels.
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= 8 and (y2 - y1) >= 8
