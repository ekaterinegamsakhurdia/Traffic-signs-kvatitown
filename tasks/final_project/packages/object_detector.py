from collections import deque
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tunable threat gates — DUCK (cls_id = 0)
# ---------------------------------------------------------------------------
# Keep these at your current working duck values.  Truck tuning below will
# not change duck detection.

DUCK_MIN_CONFIDENCE      = 0.50
DUCK_MIN_AREA_FRACTION   = 0.008
DUCK_FRONTAL_CORRIDOR_HW = 0.30
DUCK_LOWER_ZONE          = 0.60
DUCK_ZONE_CRITICAL_AREA  = 0.040
DUCK_ZONE_DANGER_AREA    = 0.012

# ---------------------------------------------------------------------------
# Tunable threat gates — TRUCK / VEHICLE (cls_id = 1)
# ---------------------------------------------------------------------------
# These start deliberately stricter than the duck gates:
#   • higher confidence      = ignores weaker YOLO truck predictions
#   • larger bbox required   = ignores farther/smaller trucks
#   • narrower corridor      = ignores trucks beside our lane
#   • lower screen gate      = truck must be visibly closer before it blocks us
#
# Increase MIN_AREA / LOWER_ZONE / MIN_CONFIDENCE to make truck detection
# even stricter.  Decrease them if the robot begins missing a real truck.

TRUCK_MIN_CONFIDENCE      = 0.70
# TRUCK_MIN_AREA_FRACTION   = 0.012
TRUCK_MIN_AREA_FRACTION   = 0.005
TRUCK_FRONTAL_CORRIDOR_HW = 0.60
TRUCK_LOWER_ZONE          = 0.60
TRUCK_ZONE_CRITICAL_AREA  = 0.070
TRUCK_ZONE_DANGER_AREA    = 0.025

# Motion tracking: how many frames of cx/area history to keep per vehicle.
# Used by traffic_rule_manager to decide stationary vs moving.
VEHICLE_HISTORY_LEN = 6

# Movement threshold: sum of |Δcx| + |Δarea| over the history window.
# Below this the vehicle is declared stationary.
VEHICLE_STATIONARY_THRESHOLD = 0.01

CLASS_NAMES = {0: "DUCK", 1: "VEHICLE", 2: "SIGN"}


class ThreatObject:
    """
    A detection that passed all geometric gates and is considered a threat.

    Attributes used by traffic_rule_manager
    ────────────────────────────────────────
    is_duck          bool    True for cls_id == 0
    is_vehicle       bool    True for cls_id == 1
    side             str     "left" | "centre" | "right"
    proximity_zone   str     "CRITICAL" | "DANGER" | "FAR"
    bottom_norm      float   cy_bottom (y2/h), used as proximity proxy
    area_frac        float   bbox area / frame area
    cx_norm          float   normalised centre-x
    cy_norm          float   normalised centre-y
    is_stationary    bool    True if vehicle hasn't moved for VEHICLE_HISTORY_LEN frames
                             (always False for ducks — they don't move)
    """

    def __init__(
        self,
        cls_id:       int,
        bbox:         Tuple[float, float, float, float],
        score:        float,
        cx_norm:      float,
        cy_norm:      float,
        cy_bottom:    float,
        area_frac:    float,
        side:         str,
        is_stationary: bool = False,
    ):
        self.cls_id        = cls_id
        self.label         = CLASS_NAMES.get(cls_id, str(cls_id))
        self.bbox          = bbox
        self.score         = score
        self.cx_norm       = cx_norm
        self.cy_norm       = cy_norm
        self.bottom_norm   = cy_bottom   
        self.area_frac     = area_frac
        self.side          = side
        self.is_duck       = cls_id == 0
        self.is_vehicle    = cls_id == 1
        self.is_stationary = is_stationary

        if cls_id == 0:
            critical_area = DUCK_ZONE_CRITICAL_AREA
            danger_area   = DUCK_ZONE_DANGER_AREA
        else:
            critical_area = TRUCK_ZONE_CRITICAL_AREA
            danger_area   = TRUCK_ZONE_DANGER_AREA

        if area_frac >= critical_area:
            self.proximity_zone = "CRITICAL"
        elif area_frac >= danger_area:
            self.proximity_zone = "DANGER"
        else:
            self.proximity_zone = "FAR"

    def __repr__(self):
        motion = ("stationary" if self.is_stationary else "moving") if self.is_vehicle else "n/a"
        return (
            f"ThreatObject(label={self.label} score={self.score:.2f} "
            f"cx={self.cx_norm:.2f} cy={self.cy_norm:.2f} "
            f"area={self.area_frac:.4f} side={self.side} "
            f"zone={self.proximity_zone} motion={motion})"
        )


class ObjectThreatDetector:
    """
    Analyses YOLO detections and flags collision threats using geometric gates.

    Motion tracking
    ───────────────
    For vehicle detections (cls_id == 1) the detector maintains a short
    history of (cx_norm, area_frac) per approximate screen position.
    After VEHICLE_HISTORY_LEN frames it decides whether the vehicle is
    stationary or moving and sets ThreatObject.is_stationary accordingly.

    Usage:
        detector = ObjectThreatDetector()
        threats  = detector.evaluate(frame_shape, detections)
    """

    def __init__(self):
        # Circular buffer: maps a bucket-key to deque of (cx, area) tuples.
        # Bucket key = round(cx_norm, 1) so nearby detections share history.
        self._vehicle_history: Dict[str, deque] = {}

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
            cy_bottom = y2 / h
            label     = CLASS_NAMES.get(cls_id, str(cls_id))

            if cls_id == 0:
                min_confidence = DUCK_MIN_CONFIDENCE
                min_area        = DUCK_MIN_AREA_FRACTION
                lower_zone      = DUCK_LOWER_ZONE
                corridor_hw     = DUCK_FRONTAL_CORRIDOR_HW
            else:
                min_confidence = TRUCK_MIN_CONFIDENCE
                min_area        = TRUCK_MIN_AREA_FRACTION
                lower_zone      = TRUCK_LOWER_ZONE
                corridor_hw     = TRUCK_FRONTAL_CORRIDOR_HW

            # ── confidence gate ───────────────────────────────────────────────
            if score < min_confidence:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"< {min_confidence:.2f} → low confidence, ignored"
                    )
                continue

            # ── size gate ────────────────────────────────────────────────────
            if area_frac < min_area:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"area={area_frac:.4f} < {min_area:.4f} → too small, ignored"
                    )
                continue

            # ── vertical gate ─────────────────────────────────────────────────
            if cy_bottom < lower_zone:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cy_bottom={cy_bottom:.2f} < {lower_zone:.2f} "
                        f"→ not close enough, ignored"
                    )
                continue

            # ── horizontal corridor gate ──────────────────────────────────────
            # Uses a dead-band around centre to classify side cleanly.
            deviation = abs(cx_norm - 0.5)
            if deviation > corridor_hw:
                side_str = "left" if cx_norm < 0.5 else "right"
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cx={cx_norm:.2f} deviation={deviation:.2f} "
                        f"> {corridor_hw:.2f} → out of corridor ({side_str}), ignored"
                    )
                continue

            # ── side classification (dead-band avoids oscillation) ─────────────
            if cx_norm < 0.40:
                side = "left"
            elif cx_norm > 0.60:
                side = "right"
            else:
                side = "centre"

            # ── motion tracking for vehicles ──────────────────────────────────
            is_stationary = False
            if cls_id == 1:
                is_stationary = self._update_vehicle_history(cx_norm, area_frac, verbose)

            # ── all gates passed → threat ─────────────────────────────────────
            threat = ThreatObject(
                cls_id       = cls_id,
                bbox         = (x1, y1, x2, y2),
                score        = score,
                cx_norm      = cx_norm,
                cy_norm      = cy_norm,
                cy_bottom    = cy_bottom,
                area_frac    = area_frac,
                side         = side,
                is_stationary = is_stationary,
            )
            threats.append(threat)

            if verbose:
                motion_str = f" motion={'stationary' if is_stationary else 'moving'}" if cls_id == 1 else ""
                print(
                    f"[OBJ_DETECT] *** THREAT *** {label} score={score:.2f} "
                    f"cx={cx_norm:.2f} cy_bottom={cy_bottom:.2f} "
                    f"area={area_frac:.4f} side={side} zone={threat.proximity_zone}"
                    f"{motion_str}"
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

    # ------------------------------------------------------------------
    # Internal motion tracking
    # ------------------------------------------------------------------

    def _update_vehicle_history(
        self, cx_norm: float, area_frac: float, verbose: bool
    ) -> bool:
        """
        Append (cx_norm, area_frac) to the history bucket for this vehicle
        and return True if the vehicle is considered stationary.

        Bucket key is the cx rounded to 1 decimal place so that a vehicle
        drifting slightly in cx doesn't spawn a new history.  Stale buckets
        (whose key differs from the current cx by > 0.2) are pruned each call
        to prevent unbounded growth.
        """
        bucket = f"{round(cx_norm, 1):.1f}"

        # Prune buckets that no longer correspond to any live detection.
        stale = [k for k in self._vehicle_history if abs(float(k) - cx_norm) > 0.2]
        for k in stale:
            del self._vehicle_history[k]

        if bucket not in self._vehicle_history:
            self._vehicle_history[bucket] = deque(maxlen=VEHICLE_HISTORY_LEN)

        hist = self._vehicle_history[bucket]
        hist.append((cx_norm, area_frac))

        if len(hist) < VEHICLE_HISTORY_LEN:
            # Not enough history yet — treat as moving (safe default).
            if verbose:
                print(
                    f"[MOTION] vehicle cx={cx_norm:.2f} area={area_frac:.4f} "
                    f"history {len(hist)}/{VEHICLE_HISTORY_LEN} — insufficient, assuming moving"
                )
            return False

        # Total variation in cx and area over the history window.
        cx_var   = sum(abs(hist[i][0] - hist[i-1][0]) for i in range(1, len(hist)))
        area_var = sum(abs(hist[i][1] - hist[i-1][1]) for i in range(1, len(hist)))
        movement = cx_var + area_var

        stationary = movement < VEHICLE_STATIONARY_THRESHOLD
        if verbose:
            print(
                f"[MOTION] vehicle cx={cx_norm:.2f} area={area_frac:.4f} "
                f"Δcx={cx_var:.4f} Δarea={area_var:.4f} total={movement:.4f} "
                f"threshold={VEHICLE_STATIONARY_THRESHOLD} "
                f"→ {'STATIONARY' if stationary else 'MOVING'}"
            )
        return stationary