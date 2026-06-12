from typing import List, Tuple
import os
import yaml
import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.final_project.packages.apriltag_activity import AprilTagDetector
from tasks.final_project.packages.traffic_rules import TrafficRuleManager, TrafficDecision
from tasks.final_project.packages.object_detector import CorridorObstacleDetector

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'final_project_config.yaml'
))


def _load_config() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class FinalProjectAgent:
    def __init__(self):
        cfg = _load_config()

        self.lane_agent   = LaneServoingAgent()
        self.tag_detector = AprilTagDetector()
        self.rules        = TrafficRuleManager()

        # ── Red-line detection thresholds ─────────────────────────────────────
        self.red_line_threshold = cfg.get('red_line_ratio_threshold', 0.025)
        self.red_line_roi_y     = cfg.get('red_line_roi_y_start',     0.58)

        # ── Corridor obstacle detector ────────────────────────────────────────
        #
        # Obstacle detection is purely colour-based (HSV corridor masks).
        # No external model or GPU required.
        #
        # On every frame the detector receives the lane agent's debug dict,
        # which it uses to build the per-scanline corridor directly from the
        # lane follower's own lane estimates rather than running a second HSV
        # segmentation pass.  If the debug dict does not contain scanline data
        # in a recognised format, the detector falls back to its own HSV
        # estimation automatically.
        #
        # Tunable in final_project_config.yaml:
        #   roi_y_start              (float) top of scanline estimation window
        #   detection_y_start        (float) top of obstacle detection window
        #   num_scanlines            (int)   rows sampled for lane estimation
        #   exclusion_half_px        (int)   min dynamic exclusion half-width
        #   exclusion_corridor_frac  (float) exclusion as fraction of corridor w
        #   duck_min_area_frac       (float) yellow occupancy threshold
        #   truck_min_area_frac      (float) blue occupancy threshold
        #   obstacle_confirm_frames  (int)   frames before threat is ACTIVE
        #   obstacle_clear_frames    (int)   frames before threat is CLEARED
        _min_area   = cfg.get('min_object_area_fraction', 0.012)
        _lower_gate = cfg.get('frontal_lower_gate',       0.40)

        self.obj_detector = CorridorObstacleDetector(
            roi_y_start             = cfg.get('roi_y_start',             _lower_gate),
            detection_y_start       = cfg.get('detection_y_start',       max(_lower_gate + 0.10, 0.55)),
            num_scanlines           = cfg.get('num_scanlines',           20),
            corridor_left_frac      = cfg.get('corridor_left_frac',      0.18),
            corridor_right_frac     = cfg.get('corridor_right_frac',     0.82),
            exclusion_half_px       = cfg.get('exclusion_half_px',       15),
            exclusion_corridor_frac = cfg.get('exclusion_corridor_frac', 0.15),
            duck_min_area_frac      = cfg.get('duck_min_area_frac',      max(0.025, _min_area * 2.0)),
            truck_min_area_frac     = cfg.get('truck_min_area_frac',     max(0.020, _min_area * 1.5)),
            confirm_frames          = cfg.get('obstacle_confirm_frames', 3),
            clear_frames            = cfg.get('obstacle_clear_frames',   6),
        )

        print(
            f"[FinalProjectAgent] "
            f"red_line_threshold={self.red_line_threshold:.3f}  "
            f"red_line_roi_y={self.red_line_roi_y:.2f}  "
            f"roi_y={self.obj_detector.roi_y_start:.2f}  "
            f"det_y={self.obj_detector.detection_y_start:.2f}  "
            f"scanlines={self.obj_detector.num_scanlines}  "
            f"excl_px={self.obj_detector.exclusion_half_px}  "
            f"excl_frac={self.obj_detector.exclusion_corridor_frac:.2f}  "
            f"duck_thr={self.obj_detector.duck_min_area_frac:.3f}  "
            f"truck_thr={self.obj_detector.truck_min_area_frac:.3f}"
        )
        print(
            f"[FinalProjectAgent] lane: "
            f"speed={self.lane_agent.base_speed:.3f}  "
            f"detection_threshold={self.lane_agent.detection_threshold}  "
            f"p={self.lane_agent.p_gain:.3f}  "
            f"d={self.lane_agent.d_gain:.3f}  "
            f"max_steer={self.lane_agent.max_steer:.3f}"
        )

        # ── Per-frame state snapshots ──────────────────────────────────────────
        self.last_tags:          List[dict]             = []
        self.last_decision:      TrafficDecision | None = None
        self.last_red_line_seen: bool                   = False
        self.last_threats:       list                   = []

        # Cached lane debug info: populated inside compute_commands() so that
        # (a) the obstacle detector can consume it without a second call, and
        # (b) get_debug_info() can return consistent data without recomputing.
        self._lane_debug_cache:   dict | None = None
        # Guard: print the lane debug structure exactly once to help identify
        # the correct key names for _try_lane_info.
        self._lane_debug_printed: bool        = False

        # Red-line debounce
        self._red_line_consec:  int = 0
        self._RED_LINE_CONFIRM: int = 3

        self._prev_state: str | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Red-line detection (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_red_line(self, frame_rgb) -> bool:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]
        roi_y1, roi_y2 = int(h * self.red_line_roi_y), int(h * 0.88)
        roi_x1, roi_x2 = int(w * 0.18), int(w * 0.82)
        roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]

        lower_red1 = np.array([0,   90,  80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170,  90,  80])
        upper_red2 = np.array([179, 255, 255])

        mask = cv2.bitwise_or(
            cv2.inRange(roi, lower_red1, upper_red1),
            cv2.inRange(roi, lower_red2, upper_red2),
        )

        red_pixels = int(np.count_nonzero(mask))
        roi_area   = mask.shape[0] * mask.shape[1]
        red_ratio  = red_pixels / max(1, roi_area)
        raw_seen   = red_ratio > self.red_line_threshold

        if raw_seen:
            self._red_line_consec += 1
            print(
                f"[RED_LINE] candidate ratio={red_ratio:.4f} "
                f"pixels={red_pixels}/{roi_area} "
                f"({self._red_line_consec}/{self._RED_LINE_CONFIRM} confirm frames)"
            )
        else:
            if self._red_line_consec > 0:
                print(
                    f"[RED_LINE] cleared after {self._red_line_consec} "
                    f"frame(s) — likely false positive at curve"
                )
            self._red_line_consec = 0

        confirmed = self._red_line_consec >= self._RED_LINE_CONFIRM
        if confirmed and self._red_line_consec == self._RED_LINE_CONFIRM:
            print(
                f"[RED_LINE] *** RED LINE CONFIRMED *** "
                f"ratio={red_ratio:.4f} pixels={red_pixels}/{roi_area}"
            )
        return confirmed

    # ──────────────────────────────────────────────────────────────────────────
    # Main compute loop
    # ──────────────────────────────────────────────────────────────────────────

    def compute_commands(self, frame_rgb) -> Tuple[float, float]:
        """
        Compute wheel commands for the current frame.

        Perception priority (highest → lowest):
          1. Obstacle in corridor  (duck / truck)  → OBSTACLE_STOP
          2. Red stop line                          → crossroad sequence
          3. AprilTag navigation                   → turn / stop / yield
          4. Lane following                        → lane-servo commands

        The lane agent's debug dict is cached after each compute call so the
        obstacle detector can consume its per-scanline lane geometry without
        triggering a second HSV segmentation pass.
        """
        # ── 1. Lane servoing baseline ─────────────────────────────────────────
        lane_left, lane_right = self.lane_agent.compute_commands(frame_rgb)

        # ── 2. Cache lane agent debug info ────────────────────────────────────
        # Called once here; used by both the obstacle detector (lane geometry)
        # and get_debug_info() (overlay rendering).  A shallow copy is stored so
        # the obstacle detector cannot accidentally mutate the agent's state.
        try:
            raw_debug = self.lane_agent.get_debug_info(frame_rgb)
            self._lane_debug_cache = dict(raw_debug) if raw_debug else {}
        except Exception:
            self._lane_debug_cache = {}

        # One-time introspection: print the lane debug dict structure so we can
        # identify the correct key names for _try_lane_info in object_detector.py.
        # This block runs exactly once and then silences itself.
        if not self._lane_debug_printed and self._lane_debug_cache:
            self._lane_debug_printed = True
            print("[LANE_DEBUG] Lane agent debug dict — keys and value types:")
            for k, v in self._lane_debug_cache.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    print(f"  {k!r:30s} list/tuple len={len(v)}  first={v[0]!r}")
                elif isinstance(v, dict):
                    inner = list(v.items())[:3]
                    print(f"  {k!r:30s} dict  len={len(v)}  sample={inner}")
                elif hasattr(v, 'shape'):   # numpy array
                    print(f"  {k!r:30s} ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  {k!r:30s} {type(v).__name__} = {v!r}"[:120])
            print("[LANE_DEBUG] End of lane debug structure")

        # ── 3. AprilTag detection — full-frame (no YOLO sign crops) ──────────
        self.last_tags = self.tag_detector.detect_combined(frame_rgb, [])

        # ── 4. Red-line detection ─────────────────────────────────────────────
        self.last_red_line_seen = self._detect_red_line(frame_rgb)

        # ── 5. Corridor obstacle detection ────────────────────────────────────
        # Pass the cached lane debug so the detector can use the lane agent's
        # own per-scanline geometry instead of re-running HSV estimation.
        self.last_threats = self.obj_detector.evaluate(
            frame_rgb,
            lane_info = self._lane_debug_cache,
            verbose   = False,
        )

        # ── 6. Side-vehicle scan for intersection peek / yield ────────────────
        side_vehicle = self.obj_detector.detect_side_vehicle(frame_rgb, verbose=False)

        # ── Logging ───────────────────────────────────────────────────────────
        if self.last_tags:
            tag_summary = [(t["id"], f"{t.get('area', 0):.0f}px²") for t in self.last_tags]
            print(f"[PERCEPTION] AprilTags: {tag_summary}")

        if self.last_threats:
            threat_summary = [
                f"{t.label}@({t.cx_norm:.2f},{t.cy_norm:.2f}) {t.proximity_zone}"
                for t in self.last_threats
            ]
            print(f"[PERCEPTION] Threats: {threat_summary}")

        if side_vehicle is not None:
            offset, side, cx = side_vehicle
            print(f"[PERCEPTION] Side vehicle: cx={cx:.2f} side={side}")

        # ── 7. Traffic-rule decision ──────────────────────────────────────────
        decision = self.rules.update(
            lane_left     = lane_left,
            lane_right    = lane_right,
            frame_shape   = frame_rgb.shape,
            tags          = self.last_tags,
            detections    = [],           # YOLO removed; no raw detections
            red_line_seen = self.last_red_line_seen,
            threats       = self.last_threats,
            side_vehicle  = side_vehicle,
        )

        self.last_decision = decision

        current_state = decision.state.value
        if current_state != self._prev_state:
            print(f"[STATE_CHANGE] {self._prev_state} → {current_state}")
            self._prev_state = current_state

        plain_lane_follow = (
            current_state == "LANE_FOLLOW"
            and not self.last_tags
            and not self.last_red_line_seen
            and not self.last_threats
        )
        if not plain_lane_follow:
            print(
                f"[DECISION] state={current_state} "
                f"L={decision.left:.3f} R={decision.right:.3f} "
                f"reason='{decision.reason}'"
                + (f" tag={decision.active_tag_id}" if decision.active_tag_id is not None else "")
                + (f" turn={decision.chosen_turn}"  if decision.chosen_turn  is not None else "")
            )

        return decision.left, decision.right

    # ──────────────────────────────────────────────────────────────────────────
    # Debug info for visualiser / overlay
    # ──────────────────────────────────────────────────────────────────────────

    def get_debug_info(self, image) -> dict:
        # Use the cached lane debug produced in the most recent compute_commands()
        # call rather than re-running the lane agent's analysis.  Falls back to a
        # live call if the cache is empty (e.g. on the very first server render).
        base = self._lane_debug_cache
        if not base:
            try:
                base = self.lane_agent.get_debug_info(image)
            except Exception:
                base = {}

        info = dict(base)   # shallow copy — safe to augment

        info["apriltag_ready"]   = self.tag_detector.ready
        info["apriltag_backend"] = self.tag_detector.backend
        info["tags"]             = self.last_tags
        info["red_line_seen"]    = self.last_red_line_seen
        info["corridor_poly"]    = self.obj_detector.last_corridor_poly
        info["threats"]          = [
            {
                "label":    t.label,
                "side":     t.side,
                "area":     t.area_frac,
                "cx":       t.cx_norm,
                "cy":       t.cy_norm,
                "zone":     t.proximity_zone,
            }
            for t in self.last_threats
        ]

        if self.last_decision:
            info["behavior_state"]  = self.last_decision.state.value
            info["behavior_reason"] = self.last_decision.reason
            info["active_tag_id"]   = self.last_decision.active_tag_id
            info["chosen_turn"]     = self.last_decision.chosen_turn

        return info