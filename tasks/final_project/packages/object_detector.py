"""
Corridor-based obstacle detector for Duckietown (no YOLO required).

Architecture — third generation
─────────────────────────────────
v1: single rectangular corridor (global left/right x).
v2: per-scanline corridor that follows road curvature + dynamic duck
    exclusion zone + temporal hysteresis.
v3: model-based lane axis with full obstacle isolation.

What changed in v3
------------------
The core problem with v2 was that lane geometry was built directly from
raw per-scanline pixel observations.  A duck (or any obstacle) sitting on
a few scanlines could corrupt yellow_x / white_x at those rows, which in
turn shifted the corridor boundary and the duck-candidate zone — a feedback
loop that made detection unreliable on curves.

v3 introduces a strict separation:

  LANE GEOMETRY  →  ROAD MODEL  →  OBSTACLE DETECTION
        ↑                                   ↓
   (pixels only)              (never feeds back into model)

The road model has four layers of robustness:

  FIX 1 — Scanline outlier rejection (spatial)
      Per-scanline centerline values are compared against their local
      neighbours.  Any scanline whose center deviates more than
      `axis_outlier_px` from the local median is marked as an outlier
      and excluded from model fitting.  A duck can corrupt at most a
      handful of scanlines; the rest vote it down.

  FIX 2 — Polynomial axis fitting
      The surviving inlier centerlines are fitted with a 2nd-degree
      polynomial  center(y) = a·y² + b·y + c  (numpy.polyfit).
      This collapses the noisy per-row zig-zag into one smooth road
      curve.  The polynomial is evaluated at every sampled row to
      produce the smoothed corridor used for obstacle measurement.
      Fallback to raw smoothed values when fewer than 4 inliers exist.

  FIX 3 — Temporal EMA smoothing (frame-to-frame)
      Polynomial coefficients are blended with their previous-frame
      values:  coeff_new = α·coeff_raw + (1-α)·coeff_prev
      Default α = 0.35 (i.e. ~65 % inertia).  This kills per-frame
      jitter from lighting flicker and sensor noise.

  FIX 4 — Lane consistency score
      std(center_values) across detection-zone scanlines is computed
      after outlier removal.  High std → geometry is unreliable
      (obstacle interference or bad lighting).  When consistency_std
      exceeds `axis_freeze_std_px` the polynomial update is frozen
      and the previous coefficients are kept unchanged.

  FIX 5 — Obstacle detection runs on the fitted model
      Obstacle measurement uses left_x / right_x derived from the
      fitted polynomial center ± half the smoothed lane width.
      Raw pixel observations never feed back into the geometry.

  FIX 6 — Midline / normalised-center space
      Internally everything is computed in normalised coordinates:
        offset   = (right_x - left_x) / w          (lane width)
        center_n = (left_x + right_x/2) / w         (midline)
      This makes thresholds resolution-independent.

How it works (step by step)
----------------------------
Step 1 — Raw scanline positions
    Same as v2: N rows sampled across the ROI.  Yellow_x (rightmost
    yellow edge) and white_x (leftmost white edge) are estimated at
    each row.  A 3-point median filter removes single-pixel spikes.

Step 2 — Build road model
    a) Compute raw center[i] = (yellow_x[i] + white_x[i]) / 2 for
       rows where both boundaries are known.
    b) Reject outliers: |center[i] - median(center[i-2..i+2])| >
       axis_outlier_px.
    c) Fit 2nd-degree polynomial to inlier (y, center) pairs.
    d) Blend polynomial coefficients with previous frame (EMA).
    e) Freeze update if consistency_std > axis_freeze_std_px.
    f) Evaluate fitted poly at every sampled row → smooth left_x,
       right_x using the same lane-width estimate.

Step 3 — Obstacle detection (lower corridor only)
    Uses ONLY the model-derived boundaries, never raw pixel positions.
    Duck (yellow) and truck (blue) logic unchanged from v2.

Step 4 — Temporal confirmation / hysteresis
    Unchanged from v2.
"""

import cv2
import numpy as np

# ── Module-level constants (may be patched by agent.py for backward compat) ───
MIN_AREA_FRACTION   = 0.012
FRONTAL_CORRIDOR_HW = 0.30
FRONTAL_LOWER_GATE  = 0.5

CLASS_NAMES = {0: "DUCK", 1: "VEHICLE", 2: "SIGN"}

# ── HSV colour windows ────────────────────────────────────────────────────────
_YELLOW_LO = np.array([ 18,  80, 100], dtype=np.uint8)
_YELLOW_HI = np.array([ 38, 255, 255], dtype=np.uint8)

_WHITE_LO  = np.array([  0,   0, 180], dtype=np.uint8)
_WHITE_HI  = np.array([180,  40, 255], dtype=np.uint8)

_BLUE_LO   = np.array([100,  80,  50], dtype=np.uint8)
_BLUE_HI   = np.array([130, 255, 255], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# ThreatObject
# ─────────────────────────────────────────────────────────────────────────────

class ThreatObject(object):
    """
    A single confirmed collision threat.

    All fields expected by traffic_rules.py are present, including the
    extended ones accessed via getattr() (bottom_norm, proximity_zone,
    danger_score, is_duck, is_vehicle).
    """

    __slots__ = (
        "cls_id", "label", "bbox", "score",
        "cx_norm", "cy_norm", "area_frac", "side",
        "bottom_norm", "proximity_zone", "danger_score",
    )

    def __init__(
        self,
        cls_id,
        bbox,
        score,
        cx_norm,
        cy_norm,
        area_frac,
        side=None,
        bottom_norm=0.0,
        proximity_zone="DANGER",
        danger_score=0.0,
    ):
        self.cls_id         = cls_id
        self.label          = CLASS_NAMES.get(cls_id, str(cls_id))
        self.bbox           = bbox
        self.score          = score
        self.cx_norm        = cx_norm
        self.cy_norm        = cy_norm
        self.area_frac      = area_frac
        self.side           = side
        self.bottom_norm    = bottom_norm
        self.proximity_zone = proximity_zone
        self.danger_score   = danger_score

    @property
    def is_duck(self):
        return self.cls_id == 0

    @property
    def is_vehicle(self):
        return self.cls_id == 1

    def __repr__(self):
        return (
            "ThreatObject(label={} score={:.2f} "
            "cx={:.2f} cy={:.2f} "
            "area={:.4f} side={} "
            "zone={} bot={:.2f})".format(
                self.label, self.score,
                self.cx_norm, self.cy_norm,
                self.area_frac, self.side,
                self.proximity_zone, self.bottom_norm,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# CorridorObstacleDetector
# ─────────────────────────────────────────────────────────────────────────────

class CorridorObstacleDetector(object):
    """
    Model-based, curved-corridor obstacle detector (v3).

    Parameters
    ----------
    roi_y_start
        Top of the scanline estimation window (fraction of frame height).
    detection_y_start
        Top of the obstacle detection window (fraction of frame height).
        Must be >= roi_y_start.
    num_scanlines
        Rows sampled between roi_y_start and the frame bottom.  30 is a
        good balance between coverage and speed.
    corridor_left_frac / corridor_right_frac
        Fallback corridor boundaries used when the road model cannot be
        initialised (first few frames, or completely dark scene).
    min_lane_width_px
        Minimum corridor width in pixels; narrower rows are skipped.
    exclusion_half_px
        Minimum pixel half-width of the yellow lane exclusion zone.
    exclusion_corridor_frac
        Exclusion half-width as a fraction of corridor width per row.
        Effective half-width = max(exclusion_half_px, cw * this).
    duck_min_area_frac
        Yellow-pixel fraction threshold for duck detection.
    duck_min_active_rows
        Minimum number of scanlines that must individually show duck pixels.
        Suppresses single-row reflections.
    truck_min_area_frac
        Blue-pixel fraction threshold for truck detection.
    side_vehicle_min_area_frac
        Blue blob minimum area fraction for detect_side_vehicle().

    -- Road model parameters (v3) --

    axis_poly_deg
        Degree of the polynomial fitted to the lane centerline.  2 handles
        typical Duckietown curves; raise to 3 only for very tight S-bends.
    axis_outlier_px
        A scanline's center is rejected as an outlier if it deviates more
        than this many pixels from its local-window median.  Keeps ducks
        from bending the polynomial.
    axis_outlier_window
        Half-size of the local median window used for outlier detection
        (in scanline index units, not pixels).
    axis_ema_alpha
        EMA blend weight for polynomial coefficients.  Lower = more
        temporal inertia.  Range 0..1, default 0.35.
    axis_freeze_std_px
        If the std of the inlier centerlines across the detection zone
        exceeds this value the polynomial update is frozen for this frame.
        Prevents a high-disturbance frame from corrupting the road model.
    axis_min_inliers
        Minimum inlier count required to attempt polynomial fitting.
        Below this the previous model is kept unchanged.

    confirm_frames / clear_frames
        Temporal hysteresis frame counts.
    critical_bottom_thresh / danger_bottom_thresh
        bottom_norm thresholds for proximity zone labels.

    -- Legacy / renamed parameters --
    obstacle_y_start, corridor_inset_frac, min_area_fraction,
    frontal_lower_gate  (kept for backward compatibility)
    """

    def __init__(
        self,
        roi_y_start=0.40,
        detection_y_start=0.55,
        num_scanlines=30,
        corridor_left_frac=0.18,
        corridor_right_frac=0.82,
        min_lane_width_px=40,
        exclusion_half_px=15,
        exclusion_corridor_frac=0.20,
        duck_min_area_frac=0.035,
        duck_min_active_rows=3,
        truck_min_area_frac=0.025,
        side_vehicle_min_area_frac=0.010,
        # Road model (v3)
        axis_poly_deg=2,
        axis_outlier_px=20,
        axis_outlier_window=2,
        axis_ema_alpha=0.35,
        axis_freeze_std_px=25.0,
        axis_min_inliers=4,
        # Temporal hysteresis
        confirm_frames=3,
        clear_frames=6,
        critical_bottom_thresh=0.85,
        danger_bottom_thresh=0.65,
        # Legacy / renamed parameters
        obstacle_y_start=None,
        corridor_inset_frac=None,
        min_area_fraction=None,
        frontal_lower_gate=None,
    ):
        # Resolve legacy aliases
        if obstacle_y_start is not None:
            detection_y_start = obstacle_y_start
        if min_area_fraction is not None:
            duck_min_area_frac  = min_area_fraction
            truck_min_area_frac = min_area_fraction * 0.8
        if frontal_lower_gate is not None:
            roi_y_start       = frontal_lower_gate
            detection_y_start = min(frontal_lower_gate + 0.10, 0.80)

        self.roi_y_start                = roi_y_start
        self.detection_y_start          = detection_y_start
        self.num_scanlines              = num_scanlines
        self.corridor_left_frac         = corridor_left_frac
        self.corridor_right_frac        = corridor_right_frac
        self.min_lane_width_px          = min_lane_width_px
        self.exclusion_half_px          = exclusion_half_px
        self.exclusion_corridor_frac    = exclusion_corridor_frac
        self.duck_min_area_frac         = duck_min_area_frac
        self.duck_min_active_rows       = duck_min_active_rows
        self.truck_min_area_frac        = truck_min_area_frac
        self.side_vehicle_min_area_frac = side_vehicle_min_area_frac
        self.axis_poly_deg              = axis_poly_deg
        self.axis_outlier_px            = axis_outlier_px
        self.axis_outlier_window        = axis_outlier_window
        self.axis_ema_alpha             = axis_ema_alpha
        self.axis_freeze_std_px         = axis_freeze_std_px
        self.axis_min_inliers           = axis_min_inliers
        self.confirm_frames             = confirm_frames
        self.clear_frames               = clear_frames
        self.critical_bottom_thresh     = critical_bottom_thresh
        self.danger_bottom_thresh       = danger_bottom_thresh

        # ── Temporal detection state ──────────────────────────────────────────
        self._duck_consec  = 0
        self._duck_absent  = 0
        self._duck_active  = False

        self._truck_consec = 0
        self._truck_absent = 0
        self._truck_active = False

        self._duck_snap  = dict(cx=0.5, cy=0.75, bot=0.85, area=0.0)
        self._truck_snap = dict(cx=0.5, cy=0.75, bot=0.85, area=0.0)

        # ── Road model state (v3) ─────────────────────────────────────────────
        # Polynomial coefficients for center(y) = poly[0]*y^2 + poly[1]*y + poly[2]
        # None until the first frame with enough inliers.
        self._axis_poly      = None   # np.ndarray of shape (deg+1,)
        self._axis_width_px  = None   # smoothed lane half-width in pixels
        self._axis_frozen    = False  # True when last frame was frozen

        # ── Visualization ─────────────────────────────────────────────────────
        self._corridor_poly  = []     # [(y, left_x, right_x), ...] sorted by y

    # ── Visualization properties ──────────────────────────────────────────────

    @property
    def last_corridor_poly(self):
        """
        Per-row corridor geometry: [(y, left_x, right_x), ...] sorted by y.
        Suitable for drawing with cv2.polylines in the server overlay.
        """
        return self._corridor_poly

    @property
    def last_corridor(self):
        """
        Backward-compatible 2-tuple (min_left_x, max_right_x).
        """
        if not self._corridor_poly:
            return (0, 0)
        left_xs  = [lx for _, lx, _  in self._corridor_poly]
        right_xs = [rx for _, _,  rx in self._corridor_poly]
        return (min(left_xs), max(right_xs))

    @property
    def last_axis_frozen(self):
        """True when the road model update was frozen on the last frame."""
        return self._axis_frozen

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        frame_rgb,
        frame_shape=None,
        detections=None,
        lane_info=None,
        verbose=True,
    ):
        """
        Analyse *frame_rgb* and return a list of currently ACTIVE threats.

        Parameters
        ----------
        frame_rgb
            RGB frame (numpy array, uint8, RGB channel order).
            A bare shape tuple is accepted for backward compatibility
            (returns [] immediately).
        lane_info
            Optional debug dict from LaneServoingAgent.get_debug_info().
            Pre-computed scanline data is used when available so both
            systems share identical lane geometry.
        """
        if isinstance(frame_rgb, tuple):
            return []

        h, w = frame_rgb.shape[:2]
        bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv  = cv2.cvtColor(bgr,       cv2.COLOR_BGR2HSV)

        # ── Colour masks (computed once, shared across all steps) ─────────────
        yellow_mask = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)
        white_mask  = cv2.inRange(hsv, _WHITE_LO,  _WHITE_HI)
        blue_mask   = cv2.inRange(hsv, _BLUE_LO,   _BLUE_HI)

        # ── Step 1: raw per-scanline positions ────────────────────────────────
        raw_scanlines = self._try_lane_info(lane_info, h, w)
        src = "lane_agent"
        if raw_scanlines is None:
            raw_scanlines = self._compute_scanlines(yellow_mask, white_mask, h, w)
            src = "HSV"

        # ── Step 2: build / update road model ────────────────────────────────
        # This is the key v3 addition.  The model is updated from raw pixel
        # geometry, then used to produce clean model-derived boundaries.
        # Obstacle pixels NEVER touch this step.
        model_scanlines = self._update_road_model(raw_scanlines, h, w, verbose)

        # Update corridor polygon for visualisation (uses model boundaries)
        self._corridor_poly = sorted(
            [(y, lx, rx) for y, lx, rx in model_scanlines],
            key=lambda t: t[0],
        )

        if verbose:
            n_y   = sum(1 for _, (yx, _) in raw_scanlines.items() if yx is not None)
            n_w   = sum(1 for _, (_, wx) in raw_scanlines.items() if wx is not None)
            n_det = sum(
                1 for y in raw_scanlines
                if y >= int(h * self.detection_y_start)
            )
            print(
                "[CORRIDOR] src={} rows={} det_rows={} "
                "yellow={} white={} frozen={}".format(
                    src, len(raw_scanlines), n_det,
                    n_y, n_w, self._axis_frozen,
                )
            )

        # ── Step 3: measure duck / truck occupancy on model boundaries ────────
        duck_frac, duck_active_rows, truck_frac = self._measure_corridor(
            yellow_mask, blue_mask, model_scanlines, h, w, verbose
        )

        # ── Step 4: temporal state ────────────────────────────────────────────
        duck_now  = (duck_frac  >= self.duck_min_area_frac and
                     duck_active_rows >= self.duck_min_active_rows)
        truck_now = truck_frac >= self.truck_min_area_frac

        self._duck_active  = self._tick(
            "DUCK",  duck_now,  self._duck_active,  duck_frac,  verbose,
            "_duck_consec",  "_duck_absent",
        )
        self._truck_active = self._tick(
            "TRUCK", truck_now, self._truck_active, truck_frac, verbose,
            "_truck_consec", "_truck_absent",
        )

        # ── Step 5: build result list ─────────────────────────────────────────
        threats = []
        if self._duck_active:
            threats.append(self._make_threat(0, self._duck_snap))
        if self._truck_active:
            threats.append(self._make_threat(1, self._truck_snap))
        return threats

    def detect_side_vehicle(self, frame_rgb, verbose=True):
        """
        Full-frame blue scan for intersection peek / yield logic.

        Returns (offset_from_centre, side, cx_norm) for the largest qualifying
        blue connected component, or None.  Intentionally stateless.
        """
        h, w = frame_rgb.shape[:2]
        bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv  = cv2.cvtColor(bgr,       cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
        min_px    = int(self.side_vehicle_min_area_frac * h * w)

        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            blue_mask, connectivity=8
        )

        best_area, best_cx = 0, None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cy   = float(centroids[i][1]) / h
            if area < min_px or cy < 0.25:
                continue
            if area > best_area:
                best_area = area
                best_cx   = float(centroids[i][0]) / w

        if best_cx is None:
            return None

        offset = abs(best_cx - 0.5)
        side   = "left" if best_cx < 0.5 else "right"
        if verbose:
            print(
                "[SIDE_VEH] blue vehicle cx={:.2f} "
                "side={} area={}px".format(best_cx, side, best_area)
            )
        return (offset, side, best_cx)

    # ── Convenience helpers (backward compat) ─────────────────────────────────

    def has_duck_threat(self, threats):
        return any(t.cls_id == 0 for t in threats)

    def has_vehicle_threat(self, threats):
        return any(t.cls_id == 1 for t in threats)

    def closest_threat(self, threats):
        return max(threats, key=lambda t: t.area_frac) if threats else None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — scanline extraction  (Step 1, unchanged from v2)
    # ──────────────────────────────────────────────────────────────────────────

    def _try_lane_info(self, lane_info, h, w):
        """
        Extract {row_y: (yellow_x, white_x)} from the lane agent debug dict.
        Returns None if the dict is absent or unrecognised.

        Recognised patterns (tried in order):
          A  scanline_data  — list of (y, yellow_x, white_x)
          B  yellow_scanlines / white_scanlines  — {y: x} dicts
             (also yellow_positions / white_positions)
          C  Single global yellow_x / white_x or yellow_position /
             white_position  → broadcast to all sampled rows
        """
        if not lane_info or not isinstance(lane_info, dict):
            return None

        # Pattern A
        sd = lane_info.get("scanline_data")
        if isinstance(sd, (list, tuple)) and len(sd) >= 3:
            result = {}
            for item in sd:
                try:
                    item = list(item)
                    while len(item) < 3:
                        item.append(None)
                    y, yx, wx = item[:3]
                    if y is not None:
                        result[int(y)] = (
                            int(yx) if yx is not None else None,
                            int(wx) if wx is not None else None,
                        )
                except (TypeError, ValueError):
                    continue
            if len(result) >= 3:
                return result

        # Pattern B
        ydict = lane_info.get("yellow_scanlines") or lane_info.get("yellow_positions")
        wdict = lane_info.get("white_scanlines")  or lane_info.get("white_positions")
        if isinstance(ydict, dict) or isinstance(wdict, dict):
            all_ys = set()
            if isinstance(ydict, dict): all_ys.update(ydict.keys())
            if isinstance(wdict, dict): all_ys.update(wdict.keys())
            if len(all_ys) >= 3:
                return {
                    int(y): (
                        int(ydict[y]) if isinstance(ydict, dict) and y in ydict else None,
                        int(wdict[y]) if isinstance(wdict, dict) and y in wdict else None,
                    )
                    for y in all_ys
                }

        # Pattern C
        yx_g = lane_info.get("yellow_x")
        wx_g = lane_info.get("white_x")
        if yx_g is None and "yellow_position" in lane_info:
            yx_g = int(lane_info["yellow_position"] * w)
        if wx_g is None and "white_position" in lane_info:
            wx_g = int(lane_info["white_position"] * w)
        if yx_g is not None or wx_g is not None:
            roi_y1 = int(h * self.roi_y_start)
            rows = np.unique(
                np.round(np.linspace(roi_y1, h - 1, self.num_scanlines)).astype(int)
            )
            return {int(y): (yx_g, wx_g) for y in rows}

        return None

    def _compute_scanlines(self, yellow_mask, white_mask, h, w):
        """
        Per-row HSV estimation with 3-point median spike filter.

        yellow_x = median of rightmost 1/3 of yellow columns (right edge
                   of the centre-line).
        white_x  = median of leftmost 1/3 of white columns (left edge of
                   the right boundary).
        """
        roi_y1 = int(h * self.roi_y_start)
        rows = np.unique(
            np.round(np.linspace(roi_y1, h - 1, self.num_scanlines)).astype(int)
        )
        raw = {}
        for row_y in rows:
            ycols = np.where(yellow_mask[row_y] > 0)[0]
            wcols = np.where(white_mask[row_y]  > 0)[0]
            raw[int(row_y)] = (
                self._lane_x(ycols, "right"),
                self._lane_x(wcols, "left"),
            )
        return self._smooth_scanlines(raw)

    @staticmethod
    def _smooth_scanlines(scanlines):
        """
        3-point median filter over neighbouring scanlines to remove single-row
        spikes (stray pixels that misplace one boundary by 100+ px).
        Only non-None values participate; rows with no neighbours keep their
        original value.
        """
        rows  = sorted(scanlines.keys())
        n     = len(rows)
        out   = {}
        for i, y in enumerate(rows):
            yx_v, wx_v = [], []
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    yx, wx = scanlines[rows[j]]
                    if yx is not None: yx_v.append(yx)
                    if wx is not None: wx_v.append(wx)
            out[y] = (
                int(np.median(yx_v)) if yx_v else None,
                int(np.median(wx_v)) if wx_v else None,
            )
        return out

    @staticmethod
    def _lane_x(cols, side):
        """
        Representative x for one lane boundary at one row.
        side='right' → median of rightmost third  (yellow left boundary).
        side='left'  → median of leftmost  third  (white right boundary).
        """
        if len(cols) == 0:
            return None
        k = max(1, len(cols) // 3)
        if side == "right":
            return int(np.median(np.sort(cols)[-k:]))
        return int(np.median(np.sort(cols)[:k]))

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — road model  (v3, Steps 2a-f)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_road_model(self, scanlines, h, w, verbose):
        """
        Build a smooth, obstacle-isolated road model from raw scanlines and
        return model-derived corridor boundaries as a list of
        (y, left_x, right_x) tuples.

        Pipeline
        --------
        a) Extract rows where BOTH boundaries are known → compute center & width.
        b) FIX 1 — Reject per-scanline outliers via local median test.
        c) FIX 4 — Check consistency (std); freeze update if too noisy.
        d) FIX 2 — Fit polynomial to inlier centers.
        e) FIX 3 — EMA blend with previous polynomial.
        f) FIX 5 — Evaluate polynomial at all rows → model boundaries.
           Obstacles never touch a, b, c, d, or e.
        """
        default_l   = int(w * self.corridor_left_frac)
        default_r   = int(w * self.corridor_right_frac)
        default_hw  = (default_r - default_l) / 2.0

        rows_sorted = sorted(scanlines.keys())

        # ── a) Collect rows with both boundaries ──────────────────────────────
        valid_y      = []
        valid_center = []
        valid_hw     = []   # half-width per row

        for y in rows_sorted:
            yx, wx = scanlines[y]
            if yx is None or wx is None:
                continue
            cw = wx - yx
            if cw < self.min_lane_width_px:
                continue
            valid_y.append(y)
            valid_center.append((yx + wx) / 2.0)
            valid_hw.append(cw / 2.0)

        # ── b) FIX 1 — Outlier rejection via local median ─────────────────────
        inlier_y      = []
        inlier_center = []
        inlier_hw     = []
        win           = self.axis_outlier_window

        for i in range(len(valid_y)):
            lo  = max(0, i - win)
            hi  = min(len(valid_y), i + win + 1)
            neighbors = valid_center[lo:hi]
            local_med = float(np.median(neighbors))
            if abs(valid_center[i] - local_med) <= self.axis_outlier_px:
                inlier_y.append(valid_y[i])
                inlier_center.append(valid_center[i])
                inlier_hw.append(valid_hw[i])

        # ── c) FIX 4 — Consistency check; freeze if too noisy ────────────────
        det_y1       = int(h * self.detection_y_start)
        det_centers  = [
            c for y, c in zip(inlier_y, inlier_center) if y >= det_y1
        ]
        consistency_std = float(np.std(det_centers)) if len(det_centers) >= 2 else 0.0
        freeze = (
            consistency_std > self.axis_freeze_std_px or
            len(inlier_y) < self.axis_min_inliers
        )
        self._axis_frozen = freeze

        if verbose and (freeze or consistency_std > 5.0):
            print(
                "[ROAD_MODEL] inliers={}/{} consistency_std={:.1f}px frozen={}".format(
                    len(inlier_y), len(valid_y), consistency_std, freeze
                )
            )

        # ── d) FIX 2 — Polynomial fit ─────────────────────────────────────────
        if not freeze and len(inlier_y) >= self.axis_min_inliers:
            y_arr = np.array(inlier_y,      dtype=float)
            c_arr = np.array(inlier_center, dtype=float)
            # Normalise y to [0, 1] for numerical stability
            y_norm   = y_arr / h
            new_poly = np.polyfit(y_norm, c_arr, self.axis_poly_deg)

            # ── e) FIX 3 — EMA blend with previous frame ──────────────────────
            if self._axis_poly is None or len(self._axis_poly) != len(new_poly):
                self._axis_poly = new_poly
            else:
                alpha = self.axis_ema_alpha
                self._axis_poly = alpha * new_poly + (1.0 - alpha) * self._axis_poly

            # Smooth half-width estimate
            hw_new = float(np.median(inlier_hw)) if inlier_hw else default_hw
            if self._axis_width_px is None:
                self._axis_width_px = hw_new
            else:
                self._axis_width_px = (
                    alpha * hw_new + (1.0 - alpha) * self._axis_width_px
                )

        # ── f) FIX 5 — Evaluate polynomial → model boundaries ─────────────────
        # Obstacle detection will ONLY see these values, never raw pixels.
        hw = self._axis_width_px if self._axis_width_px is not None else default_hw

        model = []
        for y in rows_sorted:
            if self._axis_poly is not None:
                center = float(np.polyval(self._axis_poly, y / float(h)))
            else:
                # No model yet — fall back to raw smoothed value if available
                yx, wx = scanlines[y]
                if yx is not None and wx is not None:
                    center = (yx + wx) / 2.0
                elif yx is not None:
                    center = yx + hw
                elif wx is not None:
                    center = wx - hw
                else:
                    center = (default_l + default_r) / 2.0

            left_x  = int(max(0,     center - hw))
            right_x = int(min(w - 1, center + hw))
            model.append((y, left_x, right_x))

        return model

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — occupancy measurement  (Step 3, uses model boundaries)
    # ──────────────────────────────────────────────────────────────────────────

    def _measure_corridor(self, yellow_mask, blue_mask, model_scanlines, h, w, verbose):
        """
        Count duck (yellow) and truck (blue) pixels in the lower corridor,
        using MODEL-DERIVED boundaries exclusively.

        model_scanlines : list of (y, left_x, right_x) from _update_road_model.

        Returns
        -------
        duck_frac        : yellow pixels / duck candidate area  (0-1)
        duck_active_rows : number of rows individually above duck threshold
        truck_frac       : blue pixels   / truck candidate area (0-1)
        """
        det_y1 = int(h * self.detection_y_start)

        duck_pix  = duck_area  = 0
        truck_pix = truck_area = 0

        duck_cx_sum  = duck_cy_sum  = duck_w  = 0.0
        truck_cx_sum = truck_cy_sum = truck_w = 0.0
        n_det_rows       = 0
        det_y_max        = det_y1
        duck_active_rows = 0

        # 1 % of candidate width must be yellow to count as an active duck row
        DUCK_ROW_PIX_THRESH_FRAC = 0.01

        for (row_y, left, right) in model_scanlines:
            if row_y < det_y1:
                continue
            n_det_rows += 1
            det_y_max   = max(det_y_max, row_y)

            cw = right - left
            if cw < self.min_lane_width_px:
                continue

            # ── Truck: full model corridor ────────────────────────────────────
            b_row = blue_mask[row_y, left:right]
            b_pix = int(np.count_nonzero(b_row))
            truck_pix  += b_pix
            truck_area += cw
            if b_pix > 0:
                bxs = np.where(b_row > 0)[0] + left
                truck_cx_sum += float(np.mean(bxs)) * b_pix
                truck_cy_sum += float(row_y)         * b_pix
                truck_w      += b_pix

            # ── Duck: corridor minus dynamic exclusion zone ───────────────────
            # The exclusion zone is measured from the MODEL left boundary, not
            # from yellow_x directly — this is the v3 key change.  Because the
            # model boundary is already smoothed and outlier-free, the exclusion
            # zone is stable even when a duck sits near the yellow line.
            excl = max(self.exclusion_half_px, int(cw * self.exclusion_corridor_frac))
            dx1  = min(right, left + excl)   # start of duck-candidate zone
            dx2  = right

            if dx2 <= dx1:
                continue

            y_row = yellow_mask[row_y, dx1:dx2]
            y_pix = int(np.count_nonzero(y_row))
            duck_pix  += y_pix
            duck_area += (dx2 - dx1)

            if y_pix >= max(1, int((dx2 - dx1) * DUCK_ROW_PIX_THRESH_FRAC)):
                duck_active_rows += 1

            if y_pix > 0:
                yxs = np.where(y_row > 0)[0] + dx1
                duck_cx_sum += float(np.mean(yxs)) * y_pix
                duck_cy_sum += float(row_y)         * y_pix
                duck_w      += y_pix

        duck_frac  = duck_pix  / max(1, duck_area)
        truck_frac = truck_pix / max(1, truck_area)

        # Update position snapshots
        bot_norm = det_y_max / float(h)
        if duck_w > 0:
            self._duck_snap = dict(
                cx   = (duck_cx_sum / duck_w) / w,
                cy   = (duck_cy_sum / duck_w) / h,
                bot  = bot_norm,
                area = duck_frac,
            )
        if truck_w > 0:
            self._truck_snap = dict(
                cx   = (truck_cx_sum / truck_w) / w,
                cy   = (truck_cy_sum / truck_w) / h,
                bot  = bot_norm,
                area = truck_frac,
            )

        if verbose and n_det_rows > 0 and (duck_frac > 0.005 or truck_frac > 0.005):
            print(
                "[CORRIDOR/MEASURE] det_rows={} "
                "duck_pix={}/{}={:.4f} active_rows={} "
                "truck_pix={}/{}={:.4f}".format(
                    n_det_rows,
                    duck_pix, duck_area, duck_frac, duck_active_rows,
                    truck_pix, truck_area, truck_frac,
                )
            )

        return duck_frac, duck_active_rows, truck_frac

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — temporal state
    # ──────────────────────────────────────────────────────────────────────────

    def _tick(self, label, seen_now, active, frac, verbose, consec_attr, absent_attr):
        consec = getattr(self, consec_attr)
        absent = getattr(self, absent_attr)

        if seen_now:
            consec += 1
            absent  = 0
            if verbose:
                print(
                    "[CORRIDOR/{}] frac={:.4f} "
                    "({}/{} confirm)".format(label, frac, consec, self.confirm_frames)
                )
        else:
            consec = 0
            if active:
                absent += 1
                if verbose:
                    print(
                        "[CORRIDOR/{}] absent "
                        "({}/{} clear)".format(label, absent, self.clear_frames)
                    )

        setattr(self, consec_attr, consec)
        setattr(self, absent_attr, absent)

        if not active and consec >= self.confirm_frames:
            print("[CORRIDOR/{}] *** THREAT CONFIRMED *** frac={:.4f}".format(label, frac))
            return True
        if active and absent >= self.clear_frames:
            print("[CORRIDOR/{}] *** THREAT CLEARED ***".format(label))
            return False
        return active

    def _make_threat(self, cls_id, snap):
        cx, cy, bot, area = snap["cx"], snap["cy"], snap["bot"], snap["area"]
        side = (
            "left"   if cx < 0.42 else
            "right"  if cx > 0.58 else
            "centre"
        )
        zone = (
            "CRITICAL" if bot >= self.critical_bottom_thresh else
            "DANGER"   if bot >= self.danger_bottom_thresh    else
            "FAR"
        )
        return ThreatObject(
            cls_id         = cls_id,
            bbox           = (0.0, 0.0, 0.0, 0.0),
            score          = 1.0,
            cx_norm        = cx,
            cy_norm        = cy,
            area_frac      = area,
            side           = side,
            bottom_norm    = bot,
            proximity_zone = zone,
            danger_score   = area * (1.0 + bot),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ─────────────────────────────────────────────────────────────────────────────

ObjectThreatDetector = CorridorObstacleDetector