from collections import deque
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tunable gates
# ---------------------------------------------------------------------------

# Minimum fraction of real frame area the bbox must occupy to be a threat.
# 0.008 catches small Duckietown duckies at close range; raise toward 0.02
# to ignore farther/smaller detections.
MIN_AREA_FRACTION = 0.008

# Horizontal corridor half-width from centre (normalised).
# 0.30 → accepts cx in 0.20–0.80 (strictly frontal zone).
# Widen toward 0.45 if lane-edge ducks are being missed.
FRONTAL_CORRIDOR_HW = 0.30

# cy_bottom thresholds per class.
# Duck sits low — camera sees it at cy_bottom ~0.35–0.46 before passing under.
# Truck is taller and visible from further away.
LOWER_ZONE_DUCK  = 0.6
LOWER_ZONE_TRUCK = 0.3

# Proximity zones based on area fraction (larger area = closer = more danger).
# Tune these to match your actual camera / robot scale.
ZONE_CRITICAL_AREA = 0.04   # very close — immediate emergency stop
ZONE_DANGER_AREA   = 0.012  # close — priority stop

# Motion tracking: how many frames of cx/area history to keep per vehicle.
# Used by traffic_rule_manager to decide stationary vs moving.
VEHICLE_HISTORY_LEN = 6

# Movement threshold: sum of |Δcx| + |Δarea| over the history window.
# Below this the vehicle is declared stationary.
VEHICLE_STATIONARY_THRESHOLD = 0.03

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
        self.bottom_norm   = cy_bottom   # alias used by traffic_rule_manager
        self.area_frac     = area_frac
        self.side          = side
        self.is_duck       = cls_id == 0
        self.is_vehicle    = cls_id == 1
        self.is_stationary = is_stationary

        if area_frac >= ZONE_CRITICAL_AREA:
            self.proximity_zone = "CRITICAL"
        elif area_frac >= ZONE_DANGER_AREA:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

            # ── size gate ────────────────────────────────────────────────────
            if area_frac < MIN_AREA_FRACTION:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"area={area_frac:.4f} < {MIN_AREA_FRACTION} → too small, ignored"
                    )
                continue

            # ── vertical gate ─────────────────────────────────────────────────
            lower_zone = LOWER_ZONE_DUCK if cls_id == 0 else LOWER_ZONE_TRUCK
            if cy_bottom < lower_zone:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cy_bottom={cy_bottom:.2f} < {lower_zone} → not in lower zone, ignored"
                    )
                continue

            # ── horizontal corridor gate ──────────────────────────────────────
            # Uses a dead-band around centre to classify side cleanly.
            deviation = abs(cx_norm - 0.5)
            if deviation > FRONTAL_CORRIDOR_HW:
                side_str = "left" if cx_norm < 0.5 else "right"
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"cx={cx_norm:.2f} deviation={deviation:.2f} "
                        f"→ out of corridor ({side_str}), ignored"
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