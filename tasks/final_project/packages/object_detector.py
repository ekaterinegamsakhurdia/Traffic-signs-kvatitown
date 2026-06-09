from typing import List, Optional, Tuple
 
# Minimum fraction of frame area the bbox must occupy to be considered close.
MIN_AREA_FRACTION   = 0.012   # ~1.2 % of frame
 
# Horizontal corridor: ±30% from centre = 20%–80% of frame width.
FRONTAL_CORRIDOR_HW = 0.30
 
# Objects above this y-fraction are too far away (near horizon).
FRONTAL_LOWER_GATE  = 0.35   # must be in lower 65% of frame
 
CLASS_NAMES = {0: "DUCK", 1: "VEHICLE", 2: "SIGN"}
 
 
class ThreatObject:
    """Represents a single detected collision threat."""
 
    def __init__(
        self,
        cls_id:     int,
        bbox:       Tuple[float, float, float, float],
        score:      float,
        cx_norm:    float,
        cy_norm:    float,
        area_frac:  float,
        side:       Optional[str] = None,
    ):
        self.cls_id    = cls_id
        self.label     = CLASS_NAMES.get(cls_id, str(cls_id))
        self.bbox      = bbox
        self.score     = score
        self.cx_norm   = cx_norm
        self.cy_norm   = cy_norm
        self.area_frac = area_frac
        self.side      = side
 
    def __repr__(self):
        return (
            f"ThreatObject(label={self.label} score={self.score:.2f} "
            f"cx={self.cx_norm:.2f} cy={self.cy_norm:.2f} "
            f"area={self.area_frac:.4f} side={self.side})"
        )
 
 
class ObjectThreatDetector:
    """
    Analyses raw YOLO detections and flags collision threats.
 
    Usage
    -----
        detector = ObjectThreatDetector()
        threats  = detector.evaluate(frame_shape, detections)
    """
 
    def evaluate(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
        verbose:     bool = True,
    ) -> List[ThreatObject]:
        h, w = frame_shape[:2]
        frame_area = max(1, h * w)
 
        threats: List[ThreatObject] = []
        relevant_count = 0
 
        for bbox, score, cls_id in detections:
            if cls_id not in (0, 1):
                continue
 
            relevant_count += 1
            x1, y1, x2, y2 = [float(v) for v in bbox]
 
            bw        = x2 - x1
            bh        = y2 - y1
            area_frac = (bw * bh) / frame_area
            cx_norm   = ((x1 + x2) / 2.0) / w
            cy_norm   = ((y1 + y2) / 2.0) / h
 
            label = CLASS_NAMES.get(cls_id, str(cls_id))
 
            # ── size gate ────────────────────────────────────────────────────
            if area_frac < MIN_AREA_FRACTION:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"area={area_frac:.4f} -> too small, ignored"
                    )
                continue
 
            # ── vertical gate ────────────────────────────────────────────────
            if cy_norm < FRONTAL_LOWER_GATE:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cy={cy_norm:.2f} -> too high in frame (far), ignored"
                    )
                continue
 
            # ── horizontal corridor ──────────────────────────────────────────
            deviation = abs(cx_norm - 0.5)
            if deviation > FRONTAL_CORRIDOR_HW:
                side_str = "left" if cx_norm < 0.5 else "right"
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cx={cx_norm:.2f} deviation={deviation:.2f} "
                        f"-> out of frontal corridor ({side_str}), not a frontal threat"
                    )
                continue
 
            # ── all gates passed → threat ────────────────────────────────────
            side = (
                "left"   if cx_norm < 0.42 else
                "right"  if cx_norm > 0.58 else
                "centre"
            )
 
            threat = ThreatObject(
                cls_id    = cls_id,
                bbox      = (x1, y1, x2, y2),
                score     = score,
                cx_norm   = cx_norm,
                cy_norm   = cy_norm,
                area_frac = area_frac,
                side      = side,
            )
            threats.append(threat)
 
            if verbose:
                print(
                    f"[OBJ_DETECT] *** THREAT *** {label} score={score:.2f} "
                    f"cx={cx_norm:.2f} cy={cy_norm:.2f} "
                    f"area={area_frac:.4f} side={side}"
                )
 
        if verbose and not threats and relevant_count > 0:
            print(
                f"[OBJ_DETECT] {relevant_count} duck/vehicle detection(s) present "
                f"but none qualify as frontal threats"
            )
 
        return threats
 
    def has_duck_threat(self, threats: List[ThreatObject]) -> bool:
        return any(t.cls_id == 0 for t in threats)
 
    def has_vehicle_threat(self, threats: List[ThreatObject]) -> bool:
        return any(t.cls_id == 1 for t in threats)
 
    def closest_threat(self, threats: List[ThreatObject]) -> Optional[ThreatObject]:
        """Returns the threat with the largest bounding-box area (nearest)."""
        if not threats:
            return None
        return max(threats, key=lambda t: t.area_frac)
 