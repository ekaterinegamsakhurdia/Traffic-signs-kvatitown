from typing import List, Tuple
import os
import yaml
import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.final_project.packages.apriltag_activity import AprilTagDetector
from tasks.final_project.packages.traffic_rules import TrafficRuleManager, TrafficDecision
from tasks.final_project.packages.object_detector import ObjectThreatDetector

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

        # Configurable detection thresholds — set higher values in
        # final_project_config.yaml on the physical robot so the robot
        # gets closer before reacting. Simulation uses the defaults.
        self.red_line_threshold = cfg.get('red_line_ratio_threshold', 0.025)
        self.red_line_roi_y     = cfg.get('red_line_roi_y_start',     0.58)

        _min_area   = cfg.get('min_object_area_fraction', 0.0005)
        _lower_gate = cfg.get('frontal_lower_gate',       0.4)

        # Patch module-level constants first — works for the old object_detector.py
        # that uses bare globals instead of constructor arguments.
        from tasks.final_project.packages import object_detector as _od_mod
        if hasattr(_od_mod, 'MIN_AREA_FRACTION'):
            _od_mod.MIN_AREA_FRACTION  = _min_area
        if hasattr(_od_mod, 'FRONTAL_LOWER_GATE'):
            _od_mod.FRONTAL_LOWER_GATE = _lower_gate

        # Try new-style constructor (our updated object_detector.py).
        # Fall back to bare constructor for the old version.
        try:
            self.obj_detector = ObjectThreatDetector(
                min_area_fraction  = _min_area,
                frontal_lower_gate = _lower_gate,
            )
        except TypeError:
            self.obj_detector = ObjectThreatDetector()

        print(
            f"[FinalProjectAgent] "
            f"red_line_threshold={self.red_line_threshold:.3f}  "
            f"red_line_roi_y={self.red_line_roi_y:.2f}  "
            f"min_object_area={_min_area:.3f}  "
            f"frontal_lower_gate={_lower_gate:.2f}"
        )
        print(
            f"[FinalProjectAgent] lane: "
            f"speed={self.lane_agent.base_speed:.3f}  "
            f"detection_threshold={self.lane_agent.detection_threshold}  "
            f"p={self.lane_agent.p_gain:.3f}  "
            f"d={self.lane_agent.d_gain:.3f}  "
            f"max_steer={self.lane_agent.max_steer:.3f}"
        )

        self.last_tags:           List[dict]           = []
        self.last_decision:       TrafficDecision | None = None
        self.last_red_line_seen:  bool                 = False
        self.last_threats:        list                 = []

        # Red-line debounce: require this many consecutive frames above the
        # threshold before reporting True. Prevents single-frame glints from
        # curved tiles or environmental red from triggering the crossroad sequence.
        self._red_line_consec:  int = 0
        self._RED_LINE_CONFIRM: int = 3

        # Tracks the previous decision state for change-detection logging.
        self._prev_state: str | None = None

    # ------------------------------------------------------------------
    # Red-line detection
    # ------------------------------------------------------------------

    def _detect_red_line(self, frame_rgb) -> bool:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]

        # ROI: lower road area directly in front of the robot.
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

        # Debounce: require _RED_LINE_CONFIRM consecutive frames above threshold.
        # A real stop line is stable for many frames as the robot approaches.
        # A false positive from a curved tile or environmental colour is usually
        # 1-2 frames — the counter resets immediately on any clear frame.
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

    # ------------------------------------------------------------------
    # Main compute loop
    # ------------------------------------------------------------------

    def compute_commands(self, frame_rgb, detections=None) -> Tuple[float, float]:
        detections = detections or []

        # 1. Lane servoing baseline
        lane_left, lane_right = self.lane_agent.compute_commands(frame_rgb)

        # 2. Perception — AprilTags
        self.last_tags          = self.tag_detector.detect_combined(frame_rgb, detections)
        self.last_red_line_seen = self._detect_red_line(frame_rgb)

        # 3. Threat detection — ducks and vehicles in front of the robot
        self.last_threats = self.obj_detector.evaluate(frame_rgb.shape, detections)

        # Log perception summary (only when something noteworthy is present).
        if self.last_tags:
            tag_summary = [
                (t["id"], f"{t.get('area', 0):.0f}px²", t.get("source", "?"))
                for t in self.last_tags
            ]
            print(f"[PERCEPTION] AprilTags seen: {tag_summary}")

        vehicle_count = sum(1 for _, _, cls_id in detections if cls_id == 1)
        duck_count    = sum(1 for _, _, cls_id in detections if cls_id == 0)
        sign_count    = sum(1 for _, _, cls_id in detections if cls_id == 2)
        if detections:
            print(
                f"[PERCEPTION] YOLO detections — "
                f"vehicles={vehicle_count} ducks={duck_count} signs={sign_count}"
            )

        if self.last_threats:
            threat_summary = [
                f"{t.label}@({t.cx_norm:.2f},{t.cy_norm:.2f}) area={t.area_frac:.4f}"
                for t in self.last_threats
            ]
            print(f"[PERCEPTION] Collision threats: {threat_summary}")

        # 4. Traffic-rule decision
        decision = self.rules.update(
            lane_left      = lane_left,
            lane_right     = lane_right,
            frame_shape    = frame_rgb.shape,
            tags           = self.last_tags,
            detections     = detections,
            red_line_seen  = self.last_red_line_seen,
            threats        = self.last_threats,
        )

        self.last_decision = decision

        # Always log state transitions — this fires even during lane follow
        # so the cause of any unexpected behaviour is immediately visible.
        current_state = decision.state.value
        if current_state != self._prev_state:
            print(f"[STATE_CHANGE] {self._prev_state} → {current_state}")
            self._prev_state = current_state

        # Only log the full DECISION line when something interesting is happening —
        # suppress the per-frame spam when the robot is plainly following the lane.
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

    # ------------------------------------------------------------------
    # Debug info for visualiser / overlay
    # ------------------------------------------------------------------

    def get_debug_info(self, image) -> dict:
        info = self.lane_agent.get_debug_info(image)

        info["apriltag_ready"]   = self.tag_detector.ready
        info["apriltag_backend"] = self.tag_detector.backend
        info["tags"]             = self.last_tags
        info["red_line_seen"]    = self.last_red_line_seen
        info["threats"]          = [
            {"label": t.label, "side": t.side, "area": t.area_frac}
            for t in self.last_threats
        ]

        if self.last_decision:
            info["behavior_state"]  = self.last_decision.state.value
            info["behavior_reason"] = self.last_decision.reason
            info["active_tag_id"]   = self.last_decision.active_tag_id
            info["chosen_turn"]     = self.last_decision.chosen_turn

        return info