import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tasks.final_project.packages.behavior_state import BehaviorState


STOP_TAGS = {1}

INTERSECTION_OPTIONS = {
    2: ["left"],
    3: ["right"],
    4: ["straight"],
}

TURN_COMMANDS = {
    "left": (0.02, 0.18, 1.15),
    "right": (0.18, 0.02, 1.15),
    "straight": (0.16, 0.16, 0.85),
}

PEEK_COMMANDS = {
    "left": (0.02, 0.16, 0.45),
    "right": (0.16, 0.02, 0.45),
}


@dataclass
class TrafficDecision:
    left: float
    right: float
    state: BehaviorState
    reason: str = ""
    active_tag_id: Optional[int] = None
    chosen_turn: Optional[str] = None


@dataclass
class TrafficRuleManager:
    stop_wait_s: float = 1.0
    sign_cooldown_s: float = 5.0

    state: BehaviorState = BehaviorState.LANE_FOLLOW
    state_until: float = 0.0
    turn_until: float = 0.0

    active_tag_id: Optional[int] = None
    chosen_turn: Optional[str] = None

    last_tag_seen_at: Dict[int, float] = field(default_factory=dict)

    pending_tag_id: Optional[int] = None
    pending_tag_area_ratio: float = 0.0
    pending_tag_seen_at: float = 0.0

    observed_vehicle_offsets: list = field(default_factory=list)
    observed_vehicle_until: float = 0.0

    crossroad_active: bool = False

    def update(
        self,
        lane_left: float,
        lane_right: float,
        frame_shape: Tuple[int, int, int],
        tags: List[dict],
        detections: List[tuple],
        red_line_seen: bool = False,
    ) -> TrafficDecision:
        now = time.time()

        # 1. Enter crossroad state when red line is reached.
        if red_line_seen and self.state == BehaviorState.LANE_FOLLOW:
            self.crossroad_active = True
            self.state = BehaviorState.CROSSROAD_STOP
            self.state_until = now + self.stop_wait_s
            print("[CROSSROAD] red line reached -> stop")
            return TrafficDecision(0.0, 0.0, self.state, "crossroad red line stop")

        # 2. Crossroad stop.
        if self.state == BehaviorState.CROSSROAD_STOP:
            if now < self.state_until:
                return TrafficDecision(0.0, 0.0, self.state, "stopping at red line")

            self.state = BehaviorState.CROSSROAD_PEEK_LEFT
            self.state_until = now + PEEK_COMMANDS["left"][2]
            print("[CROSSROAD] peek left")
            return TrafficDecision(
                PEEK_COMMANDS["left"][0],
                PEEK_COMMANDS["left"][1],
                self.state,
                "peek left",
            )

        # 3. Peek left.
        if self.state == BehaviorState.CROSSROAD_PEEK_LEFT:
            if now < self.state_until:
                return TrafficDecision(
                    PEEK_COMMANDS["left"][0],
                    PEEK_COMMANDS["left"][1],
                    self.state,
                    "peeking left",
                )

            self.state = BehaviorState.CROSSROAD_PEEK_RIGHT
            self.state_until = now + PEEK_COMMANDS["right"][2]
            print("[CROSSROAD] peek right")
            return TrafficDecision(
                PEEK_COMMANDS["right"][0],
                PEEK_COMMANDS["right"][1],
                self.state,
                "peek right",
            )

        # 4. Peek right.
        if self.state == BehaviorState.CROSSROAD_PEEK_RIGHT:
            if now < self.state_until:
                return TrafficDecision(
                    PEEK_COMMANDS["right"][0],
                    PEEK_COMMANDS["right"][1],
                    self.state,
                    "peeking right",
                )

            vehicle_offset = self._vehicle_offset(frame_shape, detections)

            if vehicle_offset is not None:
                self.observed_vehicle_offsets = [vehicle_offset]
                self.observed_vehicle_until = now + 0.7
                self.state = BehaviorState.CROSSROAD_WAIT_VEHICLE
                print(f"[CROSSROAD] vehicle seen offset={vehicle_offset:.3f}")
                return TrafficDecision(0.0, 0.0, self.state, "observing vehicle")

            self.state = BehaviorState.INTERSECTION_DECIDE
            print("[CROSSROAD] no vehicle -> decide from sign")

        # 5. Observe vehicle: if offset decreases, vehicle approaches.
        if self.state == BehaviorState.CROSSROAD_WAIT_VEHICLE:
            vehicle_offset = self._vehicle_offset(frame_shape, detections)

            if vehicle_offset is not None:
                self.observed_vehicle_offsets.append(vehicle_offset)

            if now < self.observed_vehicle_until:
                return TrafficDecision(0.0, 0.0, self.state, "observing vehicle motion")

            approaching = self._vehicle_is_approaching()

            if approaching:
                print("[CROSSROAD] vehicle approaching -> wait")
                self.observed_vehicle_offsets = []
                self.observed_vehicle_until = now + 0.7
                return TrafficDecision(0.0, 0.0, self.state, "vehicle approaching, waiting")

            print("[CROSSROAD] vehicle not approaching -> decide")
            self.state = BehaviorState.INTERSECTION_DECIDE

        # 6. Normal stop sign from AprilTag.
        if self.state == BehaviorState.STOP_WAIT:
            if now < self.state_until:
                return TrafficDecision(0.0, 0.0, self.state, "full stop", self.active_tag_id)
            self._reset_state()

        # 7. Turning.
        if self.state == BehaviorState.TURNING:
            if now < self.turn_until:
                left, right, _ = TURN_COMMANDS[self.chosen_turn]
                return TrafficDecision(left, right, self.state, f"turning {self.chosen_turn}", self.active_tag_id, self.chosen_turn)

            self._reset_state()
            return TrafficDecision(lane_left, lane_right, BehaviorState.LANE_FOLLOW, "finished crossroad")

        # 8. Read/remember AprilTags.
        visible_tag = self._remember_visible_close_tag(frame_shape, tags)

        if self.state == BehaviorState.INTERSECTION_DECIDE:
            tag_id = self._best_current_tag_id(frame_shape, tags)

            if tag_id is None:
                print("[CROSSROAD] waiting for direction tag")
                return TrafficDecision(0.0, 0.0, self.state, "waiting for sign")

            return self._execute_tag_action(tag_id, now)

        # Outside crossroad, use previous disappear-trigger behavior.
        if visible_tag is not None:
            return TrafficDecision(lane_left, lane_right, BehaviorState.LANE_FOLLOW, "seen tag, waiting disappear")

        disappeared_tag_id = self._consume_disappeared_tag(now)

        if disappeared_tag_id is None:
            return TrafficDecision(lane_left, lane_right, BehaviorState.LANE_FOLLOW, "lane follow")

        return self._execute_tag_action(disappeared_tag_id, now)

    def _execute_tag_action(self, tag_id: int, now: float) -> TrafficDecision:
        if now - self.last_tag_seen_at.get(tag_id, 0.0) < self.sign_cooldown_s:
            return TrafficDecision(0.0, 0.0, BehaviorState.INTERSECTION_DECIDE, "tag cooldown")

        self.last_tag_seen_at[tag_id] = now

        if tag_id in STOP_TAGS:
            self.state = BehaviorState.STOP_WAIT
            self.active_tag_id = tag_id
            self.state_until = now + self.stop_wait_s
            print("[TRAFFIC] STOP")
            return TrafficDecision(0.0, 0.0, self.state, "stop tag", tag_id)

        if tag_id in INTERSECTION_OPTIONS:
            chosen = random.choice(INTERSECTION_OPTIONS[tag_id])
            left, right, duration = TURN_COMMANDS[chosen]

            self.state = BehaviorState.TURNING
            self.active_tag_id = tag_id
            self.chosen_turn = chosen
            self.turn_until = now + duration

            print(f"[TRAFFIC] TURN {chosen}")
            return TrafficDecision(left, right, self.state, f"turn {chosen}", tag_id, chosen)

        return TrafficDecision(0.0, 0.0, BehaviorState.INTERSECTION_DECIDE, "unknown tag")

    def _vehicle_offset(self, frame_shape, detections):
        h, w = frame_shape[:2]
        best = None

        for bbox, score, cls_id in detections:
            # class 1 = truck/vehicle. Ignore duck here.
            if cls_id != 1:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = ((x1 + x2) / 2.0) / w
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, h * w)

            if area_ratio < 0.004:
                continue

            offset = abs(cx - 0.5)

            if best is None or offset < best:
                best = offset

        return best

    def _vehicle_is_approaching(self):
        if len(self.observed_vehicle_offsets) < 2:
            return False

        first = self.observed_vehicle_offsets[0]
        last = self.observed_vehicle_offsets[-1]

        print(f"[VEHICLE_OBSERVE] first={first:.3f} last={last:.3f}")

        return last < first - 0.03

    def _best_current_tag_id(self, frame_shape, tags):
        h, w = frame_shape[:2]
        image_area = h * w

        best_id = None
        best_area = 0.0

        for tag in tags:
            tag_id = int(tag.get("id", -1))
            if tag_id not in STOP_TAGS and tag_id not in INTERSECTION_OPTIONS:
                continue

            area_ratio = float(tag.get("area", 0.0)) / max(1, image_area)

            if tag_id == 1 and area_ratio < 0.012:
                continue

            if tag_id in INTERSECTION_OPTIONS and area_ratio < 0.0012:
                continue

            if area_ratio > best_area:
                best_area = area_ratio
                best_id = tag_id

        return best_id

    def _remember_visible_close_tag(self, frame_shape, tags):
        if not tags:
            return None

        h, w = frame_shape[:2]
        image_area = h * w

        best_tag = None
        best_area_ratio = 0.0

        for tag in tags:
            tag_id = int(tag.get("id", -1))
            if tag_id not in STOP_TAGS and tag_id not in INTERSECTION_OPTIONS:
                continue

            area = float(tag.get("area", 0.0))
            area_ratio = area / max(1, image_area)
            cx, _ = tag.get("center", (0.0, 0.0))
            cx_ratio = cx / w

            if not (0.15 <= cx_ratio <= 0.95):
                continue

            if tag_id == 1 and area_ratio < 0.020:
                continue

            if tag_id in INTERSECTION_OPTIONS and area_ratio < 0.0015:
                continue

            if area_ratio > best_area_ratio:
                best_area_ratio = area_ratio
                best_tag = tag

        if best_tag is None:
            return None

        self.pending_tag_id = int(best_tag["id"])
        self.pending_tag_area_ratio = best_area_ratio
        self.pending_tag_seen_at = time.time()

        print(f"[TAG_REMEMBERED] id={self.pending_tag_id} area={best_area_ratio:.4f}")
        return best_tag

    def _consume_disappeared_tag(self, now):
        if self.pending_tag_id is None:
            return None

        if now - self.pending_tag_seen_at < 0.15:
            return None

        tag_id = self.pending_tag_id
        area_ratio = self.pending_tag_area_ratio

        if tag_id == 1 and area_ratio < 0.028:
            self._clear_pending_tag()
            return None

        if tag_id in INTERSECTION_OPTIONS and area_ratio < 0.0020:
            self._clear_pending_tag()
            return None

        print(f"[TAG_DISAPPEARED_TRIGGER] id={tag_id}")
        self._clear_pending_tag()
        return tag_id

    def _clear_pending_tag(self):
        self.pending_tag_id = None
        self.pending_tag_area_ratio = 0.0
        self.pending_tag_seen_at = 0.0

    def _reset_state(self):
        self.state = BehaviorState.LANE_FOLLOW
        self.state_until = 0.0
        self.turn_until = 0.0
        self.active_tag_id = None
        self.chosen_turn = None
        self.crossroad_active = False
        self.observed_vehicle_offsets = []
        self._clear_pending_tag()