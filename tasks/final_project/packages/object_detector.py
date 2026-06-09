from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Minimum fraction of frame area the bbox must occupy to be considered a
# threat.  Lowered to catch clipped edge-duckies whose visible area is
# smaller than their physical size.
MIN_AREA_FRACTION = 0.0002

# Edge objects are partially outside the frame so YOLO sees only part of them.
# Apply a more lenient area threshold when the object centre is near an edge.
# Criticism 2: narrowed from 0.25 → 0.15 (the old value made 50 % of the
# image "edge", which is not an edge — it's half the lane).
EDGE_ZONE          = 0.15
EDGE_AREA_FRACTION = 0.0001

# Horizontal lane-overlap check.  An object is a threat if its bbox overlaps
# the danger corridor — i.e. right_norm > DANGER_LEFT and left_norm < DANGER_RIGHT.
# Criticism 3: tightened from 0.05/0.95 → 0.10/0.90 to reduce false positives
# from objects barely visible at the extreme image borders.
DANGER_LEFT  = 0.10
DANGER_RIGHT = 0.90

# Bottom-of-bbox vertical gate.  y2/h is a far better proximity indicator
# than cy_norm because it tracks where the object's feet are on the ground.
BOTTOM_GATE = 0.45

# ---------------------------------------------------------------------------
# Per-class confidence thresholds
# ---------------------------------------------------------------------------
MIN_CONF = {
    0: 0.35,   # DUCK    — small, hard to detect; be lenient
    1: 0.50,   # VEHICLE — false positive aborts intersection; be strict
}

# ---------------------------------------------------------------------------
# Distance-zone thresholds  (area_frac is a reliable distance proxy because
# all objects of the same class are the same physical size)
# ---------------------------------------------------------------------------
DUCK_DANGER_AREA    = 0.0015
DUCK_CRITICAL_AREA  = 0.0040
TRUCK_DANGER_AREA   = 0.008
TRUCK_CRITICAL_AREA = 0.020

# ---------------------------------------------------------------------------
# Temporal filter  (Criticism 5)
# Keep a rolling window of the last N frames.  A detection is only promoted
# to a confirmed threat when it appears in at least MIN_HITS_IN_WINDOW of
# those frames.  This eliminates single-frame YOLO glitches caused by motion
# blur, shadows, and edge-clipping on a physical Duckiebot camera.
# ---------------------------------------------------------------------------
TEMPORAL_WINDOW   = 5   # frames of history to keep per class
TEMPORAL_MIN_HITS = 3   # detections required within that window

CLASS_NAMES = {0: "DUCK", 1: "VEHICLE", 2: "SIGN"}


class ThreatObject:
    """Represents a single confirmed collision threat."""

    def __init__(
        self,
        cls_id:      int,
        bbox:        Tuple[float, float, float, float],
        score:       float,
        cx_norm:     float,
        cy_norm:     float,
        bottom_norm: float,
        left_norm:   float,
        right_norm:  float,
        area_frac:   float,
        side:        Optional[str] = None,
    ):
        self.cls_id      = cls_id
        self.label       = CLASS_NAMES.get(cls_id, str(cls_id))
        self.bbox        = bbox
        self.score       = score
        self.cx_norm     = cx_norm
        self.cy_norm     = cy_norm
        self.bottom_norm = bottom_norm
        self.left_norm   = left_norm
        self.right_norm  = right_norm
        self.area_frac   = area_frac
        self.side        = side

        # Criticism 6: explicit type flags for behaviour-specific downstream logic.
        self.is_duck:    bool = cls_id == 0   # static obstacle
        self.is_vehicle: bool = cls_id == 1   # moving obstacle — different rules apply

        # Criticism 1: danger_score was bottom_norm-dominated because the two
        # terms live on completely different scales (area ~0.001–0.05 vs
        # bottom ~0.45–1.0).  Use bottom_norm alone as the ranking key; it is
        # already the best single-value proximity indicator for ground objects
        # of fixed physical size.  area_frac is still used for the proximity_zone
        # classification below where its absolute value is meaningful.
        self.danger_score: float = bottom_norm   # rank by feet-position in frame

        # Proximity zone: classify by area_frac because fixed-size objects
        # make area a reliable distance proxy.
        if cls_id == 0:   # DUCK
            critical_thresh = DUCK_CRITICAL_AREA
            danger_thresh   = DUCK_DANGER_AREA
        else:             # VEHICLE
            critical_thresh = TRUCK_CRITICAL_AREA
            danger_thresh   = TRUCK_DANGER_AREA

        if area_frac >= critical_thresh:
            self.proximity_zone: str = "CRITICAL"
        elif area_frac >= danger_thresh:
            self.proximity_zone = "DANGER"
        else:
            self.proximity_zone = "FAR"

    def __repr__(self):
        kind = "DUCK(static)" if self.is_duck else "VEHICLE(moving)"
        return (
            f"ThreatObject({kind} score={self.score:.2f} "
            f"cx={self.cx_norm:.2f} bottom={self.bottom_norm:.2f} "
            f"area={self.area_frac:.4f} zone={self.proximity_zone} "
            f"side={self.side})"
        )


class ObjectThreatDetector:
    """
    Analyses raw YOLO detections and flags confirmed collision threats.

    Pipeline per frame
    ------------------
    1. Confidence gate   — discard low-score detections per class.
    2. Size gate         — discard detections too small to be real (edge-aware).
    3. Vertical gate     — discard objects whose feet are above BOTTOM_GATE
                           (too far away to matter this frame).
    4. Horizontal gate   — discard objects entirely outside the lane corridor.
    5. Temporal filter   — only promote detections seen in >= TEMPORAL_MIN_HITS
                           of the last TEMPORAL_WINDOW frames (eliminates flicker).

    Usage
    -----
        detector = ObjectThreatDetector()
        threats  = detector.evaluate(frame_shape, detections)
    """

    def __init__(self):
        # Per-class rolling history: cls_id → deque of booleans (True = seen).
        # One entry is appended per frame per class regardless of detection count,
        # so the window length is always exactly TEMPORAL_WINDOW frames.
        self._history: Dict[int, Deque[bool]] = {
            0: deque(maxlen=TEMPORAL_WINDOW),   # DUCK
            1: deque(maxlen=TEMPORAL_WINDOW),   # VEHICLE
        }

    def evaluate(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
        verbose:     bool = True,
    ) -> List[ThreatObject]:
        h, w = frame_shape[:2]
        frame_area = max(1, h * w)

        # ── per-frame candidate extraction ───────────────────────────────────
        # Build the best candidate ThreatObject per class that passes all
        # single-frame gates.  "Best" = highest bottom_norm (closest feet).
        candidates: Dict[int, Optional[ThreatObject]] = {0: None, 1: None}
        relevant_count = 0

        for bbox, score, cls_id in detections:
            if cls_id not in (0, 1):
                continue

            relevant_count += 1
            x1, y1, x2, y2 = [float(v) for v in bbox]

            bw          = x2 - x1
            bh          = y2 - y1
            area_frac   = (bw * bh) / frame_area
            cx_norm     = ((x1 + x2) / 2.0) / w
            cy_norm     = ((y1 + y2) / 2.0) / h
            bottom_norm = y2 / h
            left_norm   = x1 / w
            right_norm  = x2 / w

            label = CLASS_NAMES.get(cls_id, str(cls_id))

            # ── confidence gate ──────────────────────────────────────────────
            min_conf = MIN_CONF.get(cls_id, 0.4)
            if score < min_conf:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} < {min_conf:.2f} "
                        f"-> low confidence, ignored"
                    )
                continue

            # ── size gate (edge-aware) ───────────────────────────────────────
            is_edge_object = cx_norm < EDGE_ZONE or cx_norm > (1.0 - EDGE_ZONE)
            min_area = EDGE_AREA_FRACTION if is_edge_object else MIN_AREA_FRACTION
            if area_frac < min_area:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"area={area_frac:.4f} min={min_area:.4f} "
                        f"edge={is_edge_object} -> too small, ignored"
                    )
                continue

            # ── vertical gate ────────────────────────────────────────────────
            if bottom_norm < BOTTOM_GATE:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"bottom={bottom_norm:.2f} -> too high in frame (far), ignored"
                    )
                continue

            # ── horizontal lane-overlap gate ─────────────────────────────────
            if right_norm < DANGER_LEFT:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"right_norm={right_norm:.2f} -> entirely left of lane, ignored"
                    )
                continue
            if left_norm > DANGER_RIGHT:
                if verbose:
                    print(
                        f"[OBJ_DETECT] {label} score={score:.2f} "
                        f"left_norm={left_norm:.2f} -> entirely right of lane, ignored"
                    )
                continue

            # ── all single-frame gates passed ────────────────────────────────
            side = (
                "left"   if cx_norm < EDGE_ZONE        else
                "right"  if cx_norm > 1.0 - EDGE_ZONE  else
                "centre"
            )

            candidate = ThreatObject(
                cls_id      = cls_id,
                bbox        = (x1, y1, x2, y2),
                score       = score,
                cx_norm     = cx_norm,
                cy_norm     = cy_norm,
                bottom_norm = bottom_norm,
                left_norm   = left_norm,
                right_norm  = right_norm,
                area_frac   = area_frac,
                side        = side,
            )

            # Keep the closest candidate per class this frame.
            existing = candidates[cls_id]
            if existing is None or candidate.bottom_norm > existing.bottom_norm:
                candidates[cls_id] = candidate

        # ── temporal filter (Criticism 5) ─────────────────────────────────────
        # Append one boolean per class to the rolling history, then only emit
        # a threat when the class was seen in enough recent frames.
        threats: List[ThreatObject] = []

        for cls_id, candidate in candidates.items():
            seen_this_frame = candidate is not None
            self._history[cls_id].append(seen_this_frame)

            hit_count = sum(self._history[cls_id])
            confirmed = hit_count >= TEMPORAL_MIN_HITS
            label     = CLASS_NAMES.get(cls_id, str(cls_id))

            if verbose and seen_this_frame:
                print(
                    f"[OBJ_DETECT] {label} candidate "
                    f"bottom={candidate.bottom_norm:.2f} "
                    f"area={candidate.area_frac:.4f} "
                    f"zone={candidate.proximity_zone} "
                    f"hits={hit_count}/{TEMPORAL_WINDOW} "
                    f"confirmed={confirmed}"
                )

            if confirmed and candidate is not None:
                threats.append(candidate)
                if verbose:
                    print(
                        f"[OBJ_DETECT] *** THREAT CONFIRMED *** {label} "
                        f"score={candidate.score:.2f} "
                        f"cx={candidate.cx_norm:.2f} bottom={candidate.bottom_norm:.2f} "
                        f"area={candidate.area_frac:.4f} zone={candidate.proximity_zone} "
                        f"side={candidate.side} is_duck={candidate.is_duck}"
                    )
            elif confirmed and candidate is None:
                # Was confirmed but not seen this frame — clear the history so
                # re-detection requires a fresh run of TEMPORAL_MIN_HITS.
                self._history[cls_id].clear()

        if verbose and not threats and relevant_count > 0:
            print(
                f"[OBJ_DETECT] {relevant_count} duck/vehicle detection(s) present "
                f"but none confirmed as threats yet (temporal filter pending)"
            )

        return threats

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_duck_threat(self, threats: List[ThreatObject]) -> bool:
        return any(t.is_duck for t in threats)

    def has_vehicle_threat(self, threats: List[ThreatObject]) -> bool:
        return any(t.is_vehicle for t in threats)

    def has_edge_threat(self, threats: List[ThreatObject]) -> bool:
        """True if any threat is near the camera / lane edge."""
        return any(t.side in ("left", "right") for t in threats)

    def closest_threat(self, threats: List[ThreatObject]) -> Optional[ThreatObject]:
        """Returns the threat whose feet are lowest in frame (bottom_norm = nearest)."""
        if not threats:
            return None
        return max(threats, key=lambda t: t.bottom_norm)

    def critical_threats(self, threats: List[ThreatObject]) -> List[ThreatObject]:
        """Threats in the CRITICAL proximity zone — caller should emergency-stop."""
        return [t for t in threats if t.proximity_zone == "CRITICAL"]