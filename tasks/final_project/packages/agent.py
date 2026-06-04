from typing import List, Tuple
import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.final_project.packages.apriltag_activity import AprilTagDetector
from tasks.final_project.packages.traffic_rules import TrafficRuleManager, TrafficDecision
from tasks.final_project.packages.object_detector import ObjectThreatDetector


class FinalProjectAgent:
    def __init__(self):
        self.lane_agent    = LaneServoingAgent()
        self.tag_detector  = AprilTagDetector()
        self.rules         = TrafficRuleManager()
        self.obj_detector  = ObjectThreatDetector()

        self.last_tags:           List[dict]           = []
        self.last_decision:       TrafficDecision | None = None
        self.last_red_line_seen:  bool                 = False
        self.last_threats:        list                 = []

    # ------------------------------------------------------------------
    # Red-line detection
    # ------------------------------------------------------------------

    def _detect_red_line(self, frame_rgb) -> bool:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]

        # ROI: lower road area directly in front of the robot.
        roi_y1, roi_y2 = int(h * 0.58), int(h * 0.88)
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
        red_seen   = red_ratio > 0.025

        if red_seen:
            print(f"[RED_LINE] *** RED LINE DETECTED *** ratio={red_ratio:.4f} "
                  f"pixels={red_pixels}/{roi_area}")

        return red_seen

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

        print(
            # f"[DECISION] state={decision.state.value} "
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