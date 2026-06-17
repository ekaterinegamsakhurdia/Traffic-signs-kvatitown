from typing import List, Optional, Tuple

# Minimum fraction of real frame area the bbox must occupy to be a threat.
# 0.004 suits small Duckietown duckies at close range.
# Raise toward 0.02 to ignore farther/smaller detections.
MIN_AREA_FRACTION = 0.0

# Horizontal corridor half-width from centre (normalised).
# 0.44 → accepts cx in 0.06–0.94 (covers ducks at white/yellow lane lines).
# Narrow toward 0.20 if you only want strictly frontal threats.
FRONTAL_CORRIDOR_HW = 0.2

# cy_bottom threshold per class.
# Duck sits low on the road — camera sees it at cy_bottom ~0.35–0.46 max before passing under.
# Truck is taller and visible from further away, so can afford a stricter threshold.
# Lower toward 0.25 for earlier trigger; raise toward 0.50 to require closer approach.
LOWER_ZONE_DUCK    = 0.35
LOWER_ZONE_TRUCK   = 0.7

CLASS_NAMES = {0: "DUCK", 1: "VEHICLE", 2: "SIGN"}


class ThreatObject:
    def __init__(
        self,
        cls_id:    int,
        bbox:      Tuple[float, float, float, float],
        score:     float,
        cx_norm:   float,
        cy_norm:   float,
        area_frac: float,
        side:      Optional[str] = None,
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
    Analyses YOLO detections (already filtered by integration_activity)
    and flags collision threats using geometric gates.

    Usage:
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

            bw           = x2 - x1
            bh           = y2 - y1
            area_frac    = (bw * bh) / frame_area
            cx_norm      = ((x1 + x2) / 2.0) / w
            cy_norm      = ((y1 + y2) / 2.0) / h
            cy_bottom    = y2 / h

            label = CLASS_NAMES.get(cls_id, str(cls_id))

            # ── size gate ────────────────────────────────────────────────────
            if area_frac < MIN_AREA_FRACTION:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"area={area_frac:.4f} -> too small, ignored" 
                    )
                continue
            lower_zone = LOWER_ZONE_DUCK if cls_id == 0 else LOWER_ZONE_TRUCK
            if cy_bottom < lower_zone:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cy_bottom={cy_bottom:.2f} -> not in lower zone, ignored"
                    )
                    print(MIN_AREA_FRACTION )
                    print(FRONTAL_CORRIDOR_HW)
                continue

            # ── horizontal corridor ──────────────────────────────────────────
            deviation = abs(cx_norm - 0.5)
            if deviation > FRONTAL_CORRIDOR_HW:
                side_str = "left" if cx_norm < 0.5 else "right"
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cx={cx_norm:.2f} -> out of corridor ({side_str}), ignored"
                    )
                continue
            print(cx_norm)
            print(deviation)
            # ── all gates passed → threat ────────────────────────────────────
            side = (
                "left"   if cx_norm < 0.4 else
                "right"  if cx_norm > 0.6 else
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
                    f"cx={cx_norm:.2f} cy_bottom={cy_bottom:.2f} "
                    f"area={area_frac:.4f} side={side}"
                )

        if verbose and not threats and relevant_count > 0:
            print(
                f"[OBJ_DETECT] {relevant_count} duck/vehicle detection(s) found "
                f"but none passed all gates"
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