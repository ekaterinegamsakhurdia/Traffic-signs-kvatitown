from typing import List, Tuple
import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.final_project.packages.apriltag_activity import AprilTagDetector
from tasks.final_project.packages.traffic_rules import TrafficRuleManager, TrafficDecision


class FinalProjectAgent:
    def __init__(self):
        self.lane_agent = LaneServoingAgent()
        self.tag_detector = AprilTagDetector()
        self.rules = TrafficRuleManager()

        self.last_tags: List[dict] = []
        self.last_decision: TrafficDecision | None = None
        self.last_red_line_seen = False

    def _detect_red_line(self, frame_rgb):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]

        # Look only at lower road area directly in front.
        roi_y1 = int(h * 0.58)
        roi_y2 = int(h * 0.88)
        roi_x1 = int(w * 0.18)
        roi_x2 = int(w * 0.82)

        roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]

        lower_red1 = np.array([0, 90, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 90, 80])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(roi, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = int(np.count_nonzero(red_mask))
        roi_area = red_mask.shape[0] * red_mask.shape[1]
        red_ratio = red_pixels / max(1, roi_area)

        # Tune if needed.
        red_seen = red_ratio > 0.025

        print(f"[RED_LINE] seen={red_seen} ratio={red_ratio:.4f}")

        return red_seen

    def compute_commands(self, frame_rgb, detections=None) -> Tuple[float, float]:
        detections = detections or []

        lane_left, lane_right = self.lane_agent.compute_commands(frame_rgb)

        self.last_tags = self.tag_detector.detect_combined(frame_rgb, detections)
        self.last_red_line_seen = self._detect_red_line(frame_rgb)

        decision = self.rules.update(
            lane_left=lane_left,
            lane_right=lane_right,
            frame_shape=frame_rgb.shape,
            tags=self.last_tags,
            detections=detections,
            red_line_seen=self.last_red_line_seen,
        )

        self.last_decision = decision
        return decision.left, decision.right

    def get_debug_info(self, image):
        info = self.lane_agent.get_debug_info(image)

        info["apriltag_ready"] = self.tag_detector.ready
        info["apriltag_backend"] = self.tag_detector.backend
        info["tags"] = self.last_tags
        info["red_line_seen"] = self.last_red_line_seen

        if self.last_decision:
            info["behavior_state"] = self.last_decision.state.value
            info["behavior_reason"] = self.last_decision.reason
            info["active_tag_id"] = self.last_decision.active_tag_id
            info["chosen_turn"] = self.last_decision.chosen_turn

        return info