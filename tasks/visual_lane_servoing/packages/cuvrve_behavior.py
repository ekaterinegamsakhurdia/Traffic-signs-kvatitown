from typing import List, Tuple


def detect_curve(
    yellow_xs: List[int],
    white_xs: List[int],
    curve_threshold: int = 350,
) -> Tuple[bool, int]:
    # Curve-specific speed/boost is disabled.
    # Steering is handled smoothly inside agent.py.
    return False, 0