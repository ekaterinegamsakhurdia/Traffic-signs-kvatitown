import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tasks.final_project.packages.behavior_state import BehaviorState


# ---------------------------------------------------------------------------
# AprilTag ID → behaviour
# ---------------------------------------------------------------------------

# Stop: robot must stop for 5 seconds
STOP_TAGS  = {20, 24, 25, 26}

# Yield: slow down a few frames then go straight
YIELD_TAGS = {39}

# Turn decision tags
LEFT_RIGHT_TAGS   = {11}   # turn left OR right (random)
LEFT_FORWARD_TAGS = {10}   # turn left OR go forward (random)
RIGHT_FORWARD_TAGS = {9}   # turn right OR go forward (random)

# All known tags combined
ALL_KNOWN_TAGS = STOP_TAGS | YIELD_TAGS | LEFT_RIGHT_TAGS | LEFT_FORWARD_TAGS | RIGHT_FORWARD_TAGS

TAG_NAMES = {
    20: "STOP",
    24: "STOP",
    25: "STOP",
    26: "STOP",
    39: "YIELD",
    9:  "RIGHT_OR_FORWARD",
    10: "LEFT_OR_FORWARD",
    11: "LEFT_OR_RIGHT",
}


# ---------------------------------------------------------------------------
# Motion commands  (left_wheel, right_wheel)
# ---------------------------------------------------------------------------
DRIVE_SPEED  = 0.3
CREEP_SPEED  = 0.05   # slow for yield
TURN_LEFT    = (0.16, 0.02)
TURN_RIGHT   = (0.02, 0.16)
PEEK_L       = (0.14, 0.02)
PEEK_R       = (0.02, 0.14)

# How long to drive past the red line before executing turn
CROSS_LINE_S = 0.3

# Turn durations (seconds)
TURN_DURATION = {
    "left":     2.0,
    "right":    2.0,
    "straight": 1.0,
}

# ---------------------------------------------------------------------------
# Obstacle-stop parameters
# ---------------------------------------------------------------------------
OBSTACLE_CLEAR_FRAMES = 4   # consecutive clear frames before resuming

# ---------------------------------------------------------------------------
# Peek parameters — 3 frames each side
# ---------------------------------------------------------------------------
PEEK_FRAMES_L1  = 3    # frames rotating left
PEEK_HOLD_1_S   = 2.0  # hold & scan LEFT
PEEK_FRAMES_R   = 3    # frames rotating right   ← changed to 3
PEEK_HOLD_2_S   = 2.0  # hold & scan RIGHT
PEEK_FRAMES_L2  = 4    # re-align frames

# ---------------------------------------------------------------------------
# Misc timing constants
# ---------------------------------------------------------------------------
RED_LINE_COOLDOWN_S        = 10.0
STOP_SIGN_WAIT_S           = 5.0
YIELD_CREEP_FRAMES         = 8     # frames to creep slowly for yield
VEHICLE_OBSERVE_WINDOW_S   = 1.2
APPROACH_THRESHOLD         = 0.03
STATIONARY_FRAME_THRESHOLD = 0.01
STATIONARY_FRAMES_IGNORE   = 5
CROSSROAD_STOP_S           = 1.0   # brief pause at red line before peek

# ---------------------------------------------------------------------------
# Peek sub-states
# ---------------------------------------------------------------------------
PH_L1    = "L1"
PH_HOLD1 = "HOLD1"
PH_R     = "R"
PH_HOLD2 = "HOLD2"
PH_L2    = "L2"
PH_DONE  = "DONE"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrafficDecision:
    left:          float
    right:         float
    state:         BehaviorState
    reason:        str           = ""
    active_tag_id: Optional[int] = None
    chosen_turn:   Optional[str] = None


@dataclass
class TrafficRuleManager:
    # ── core state ──────────────────────────────────────────────────────────
    state:       BehaviorState = BehaviorState.LANE_FOLLOW
    state_until: float         = 0.0
    turn_until:  float         = 0.0

    active_tag_id: Optional[int] = None
    chosen_turn:   Optional[str] = None

    # ── obstacle stop ────────────────────────────────────────────────────────
    obstacle_clear_frames: int = 0

    # ── peek sub-state ───────────────────────────────────────────────────────
    peek_phase:       str   = PH_L1
    peek_frame_count: int   = 0
    peek_hold_until:  float = 0.0
    peek_vehicles:    list  = field(default_factory=list)

    # ── vehicle observation ──────────────────────────────────────────────────
    observed_vehicle_offsets: list = field(default_factory=list)
    observed_vehicle_sides:   list = field(default_factory=list)
    observed_vehicle_until:   float = 0.0
    vehicle_stationary_frames: int = 0

    # ── sign read at red-line approach ───────────────────────────────────────
    approach_tag_id: Optional[int] = None
    peek_for_stop:   bool          = False

    # ── yield creep counter ──────────────────────────────────────────────────
    yield_creep_frames_done: int = 0

    # ── crossroad timing ─────────────────────────────────────────────────────
    last_crossroad_at: float = 0.0

    # =========================================================================
    # Main update — called every frame
    # =========================================================================

    def update(
        self,
        lane_left:     float,
        lane_right:    float,
        frame_shape:   Tuple[int, int, int],
        tags:          List[dict],
        detections:    List[tuple],
        red_line_seen: bool = False,
        threats=None,
    ) -> TrafficDecision:
        now     = time.time()
        threats = threats or []

        # ── 0. OBSTACLE STOP ─────────────────────────────────────────────────
        # Only interrupts LANE_FOLLOW so crossroad sequences are never aborted.
        if self.state == BehaviorState.LANE_FOLLOW and threats:
            closest = max(threats, key=lambda t: t.area_frac)
            print(
                f"[OBSTACLE] *** COLLISION RISK *** "
                f"label={closest.label} area={closest.area_frac:.4f} "
                f"cx={closest.cx_norm:.2f} cy={closest.cy_norm:.2f} side={closest.side} "
                f"→ STOPPING"
            )
            self.obstacle_clear_frames = 0
            self.state = BehaviorState.OBSTACLE_STOP
            return TrafficDecision(0.0, 0.0, BehaviorState.OBSTACLE_STOP,
                                   f"obstacle: {closest.label} in path")

        if self.state == BehaviorState.OBSTACLE_STOP:
            if threats:
                closest = max(threats, key=lambda t: t.area_frac)
                self.obstacle_clear_frames = 0
                print(
                    f"[OBSTACLE] still blocked — {closest.label} "
                    f"cx={closest.cx_norm:.2f} area={closest.area_frac:.4f}"
                )
                return TrafficDecision(0.0, 0.0, BehaviorState.OBSTACLE_STOP,
                                       "obstacle: waiting for path to clear")
            else:
                self.obstacle_clear_frames += 1
                print(
                    f"[OBSTACLE] path clear? "
                    f"({self.obstacle_clear_frames}/{OBSTACLE_CLEAR_FRAMES} frames)"
                )
                if self.obstacle_clear_frames >= OBSTACLE_CLEAR_FRAMES:
                    print("[OBSTACLE] *** path confirmed clear *** resuming LANE_FOLLOW")
                    self.obstacle_clear_frames = 0
                    self.state = BehaviorState.LANE_FOLLOW
                    return TrafficDecision(lane_left, lane_right,
                                          BehaviorState.LANE_FOLLOW,
                                          "obstacle cleared: lane follow")
                return TrafficDecision(0.0, 0.0, BehaviorState.OBSTACLE_STOP,
                                       "obstacle: confirming clear")

        # ── 1. RED LINE → enter crossroad sequence ───────────────────────────
        if red_line_seen and self.state == BehaviorState.LANE_FOLLOW:
            elapsed = now - self.last_crossroad_at
            if elapsed < RED_LINE_COOLDOWN_S:
                print(
                    f"[RED_LINE] cooldown active "
                    f"({elapsed:.1f}s / {RED_LINE_COOLDOWN_S}s) — ignoring"
                )
            else:
                tag_id   = self._best_visible_tag(tags)
                tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NO TAG"
                print(
                    f"[RED_LINE] *** CROSSROAD DETECTED *** "
                    f"tag={tag_name} (id={tag_id})"
                )
                print(
                    f"[RED_LINE] entering CROSSROAD_STOP for {CROSSROAD_STOP_S}s"
                )
                self.approach_tag_id   = tag_id
                self.last_crossroad_at = now
                self.state             = BehaviorState.CROSSROAD_STOP
                self.state_until       = now + CROSSROAD_STOP_S
                self.peek_for_stop     = tag_id in STOP_TAGS
                return TrafficDecision(0.0, 0.0, BehaviorState.CROSSROAD_STOP,
                                       f"red line: stopping (tag={tag_name})", tag_id)

        # ── 2. CROSSROAD_STOP — brief halt, improve tag reading ──────────────
        if self.state == BehaviorState.CROSSROAD_STOP:
            better_tag = self._best_visible_tag(tags)
            if better_tag is not None and better_tag != self.approach_tag_id:
                old = TAG_NAMES.get(self.approach_tag_id, str(self.approach_tag_id))
                new = TAG_NAMES.get(better_tag, str(better_tag))
                print(
                    f"[CROSSROAD_STOP] tag updated {old} → {new} (id={better_tag})"
                )
                self.approach_tag_id = better_tag
                self.peek_for_stop   = better_tag in STOP_TAGS

            if now < self.state_until:
                remaining = self.state_until - now
                tag_name  = TAG_NAMES.get(self.approach_tag_id, "?")
                print(
                    f"[CROSSROAD_STOP] stopped at red line "
                    f"{remaining:.2f}s remaining tag={tag_name} (id={self.approach_tag_id})"
                )
                return TrafficDecision(0.0, 0.0, BehaviorState.CROSSROAD_STOP,
                                       f"crossroad stop ({tag_name})",
                                       self.approach_tag_id)

            # Brief stop over → start peek
            tag_name = TAG_NAMES.get(self.approach_tag_id, "UNKNOWN")
            print(
                f"[CROSSROAD_STOP] brief stop done — starting peek (L→R) "
                f"for tag={tag_name}"
            )
            self._reset_peek()
            self.state = BehaviorState.CROSSROAD_PEEK_LEFT
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 3. PEEK sequence ─────────────────────────────────────────────────
        if self.state in (BehaviorState.CROSSROAD_PEEK_LEFT,
                          BehaviorState.CROSSROAD_PEEK_RIGHT):
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 4. STOP_WAIT — full 5-second stop (stop sign) ────────────────────
        if self.state == BehaviorState.STOP_WAIT:
            if now < self.state_until:
                remaining = self.state_until - now
                print(f"[STOP_WAIT] full stop, {remaining:.2f}s remaining")
                return TrafficDecision(0.0, 0.0, self.state, "full stop",
                                       self.active_tag_id or self.approach_tag_id)

            v = self._vehicle_offset_and_side(frame_shape, detections)
            if v is not None:
                offset, side = v
                print(
                    f"[STOP_WAIT] vehicle present "
                    f"(offset={offset:.3f} side={side}) — holding"
                )
                return TrafficDecision(0.0, 0.0, self.state, "stop: vehicle present",
                                       self.active_tag_id or self.approach_tag_id)

            print("[STOP_WAIT] clear — proceeding based on tag")
            return self._proceed_after_crossroad(now)

        # ── 5. CROSSROAD_WAIT_VEHICLE ─────────────────────────────────────────
        if self.state == BehaviorState.CROSSROAD_WAIT_VEHICLE:
            v = self._vehicle_offset_and_side(frame_shape, detections)

            if v is not None:
                offset, side = v
                self.observed_vehicle_offsets.append(offset)
                self.observed_vehicle_sides.append(side)

                if len(self.observed_vehicle_offsets) >= 2:
                    prev = self.observed_vehicle_offsets[-2]
                    curr = self.observed_vehicle_offsets[-1]
                    if abs(curr - prev) < STATIONARY_FRAME_THRESHOLD:
                        self.vehicle_stationary_frames += 1
                    else:
                        self.vehicle_stationary_frames = 0

                print(
                    f"[VEHICLE_OBSERVE] vehicle seen "
                    f"offset={offset:.3f} side={side} "
                    f"stationary_frames={self.vehicle_stationary_frames}"
                )

                if self.vehicle_stationary_frames >= STATIONARY_FRAMES_IGNORE:
                    print(
                        f"[VEHICLE_OBSERVE] vehicle parked for "
                        f"{STATIONARY_FRAMES_IGNORE} frames → ignoring, proceeding"
                    )
                    return self._proceed_after_crossroad(now)
            else:
                self.vehicle_stationary_frames = 0
                print("[VEHICLE_OBSERVE] no vehicle this frame")

            if now < self.observed_vehicle_until:
                return TrafficDecision(0.0, 0.0, self.state, "observing vehicle")

            approaching, approach_side = self._vehicle_is_approaching()

            if approaching:
                if approach_side == "left":
                    print(
                        "[VEHICLE] *** approaching from LEFT — "
                        "yielding, extending observation window ***"
                    )
                    self.observed_vehicle_offsets  = []
                    self.observed_vehicle_sides    = []
                    self.vehicle_stationary_frames = 0
                    self.observed_vehicle_until    = now + VEHICLE_OBSERVE_WINDOW_S
                    return TrafficDecision(0.0, 0.0, self.state,
                                          "vehicle from left: yielding")
                else:
                    print(
                        "[VEHICLE] *** approaching from RIGHT — "
                        "we have priority, proceeding ***"
                    )
                    return self._proceed_after_crossroad(now)
            else:
                print("[VEHICLE] not approaching — proceeding")
                return self._proceed_after_crossroad(now)

        # ── 6. INTERSECTION_DECIDE — cross line, then turn ───────────────────
        if self.state == BehaviorState.INTERSECTION_DECIDE:
            if now < self.state_until:
                remaining = self.state_until - now
                print(
                    f"[INTERSECTION_DECIDE] crossing red line, "
                    f"{remaining:.2f}s remaining"
                )
                return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                       self.state, "crossing line")

            tag_id    = self.approach_tag_id
            tag_name  = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
            direction = self.chosen_turn or "straight"
            print(
                f"[INTERSECTION_DECIDE] crossed line — "
                f"tag={tag_name} executing turn={direction}"
            )
            self._begin_turn(direction, now, tag_id)
            l, r = self._wheel_speeds_for(direction)
            return TrafficDecision(l, r, BehaviorState.TURNING,
                                   f"turn {direction}", tag_id, direction)

        # ── 7. YIELD creep ───────────────────────────────────────────────────
        if self.state == BehaviorState.YIELD_WAIT:
            self.yield_creep_frames_done += 1
            remaining = YIELD_CREEP_FRAMES - self.yield_creep_frames_done
            if remaining > 0:
                print(
                    f"[YIELD] creeping slowly "
                    f"({self.yield_creep_frames_done}/{YIELD_CREEP_FRAMES} frames)"
                )
                return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                       self.state, "yield: creeping",
                                       self.approach_tag_id)
            print("[YIELD] creep done → going straight")
            self.state         = BehaviorState.TURNING
            self.chosen_turn   = "straight"
            self.turn_until    = now + TURN_DURATION["straight"]
            self.active_tag_id = self.approach_tag_id
            return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                   BehaviorState.TURNING, "yield done: straight",
                                   self.active_tag_id, "straight")

        # ── 8. TURNING ────────────────────────────────────────────────────────
        if self.state == BehaviorState.TURNING:
            if now < self.turn_until:
                l, r = self._wheel_speeds_for(self.chosen_turn)
                print(
                    f"[TURNING] '{self.chosen_turn}' "
                    f"{self.turn_until - now:.2f}s remaining"
                )
                return TrafficDecision(l, r, self.state,
                                       f"turning {self.chosen_turn}",
                                       self.active_tag_id, self.chosen_turn)

            print(
                f"[TURNING] '{self.chosen_turn}' complete "
                f"→ snapping back to LANE_FOLLOW"
            )
            self._reset_state()
            return TrafficDecision(lane_left, lane_right,
                                   BehaviorState.LANE_FOLLOW, "turn done: lane follow")

        # ── LANE_FOLLOW fallthrough ───────────────────────────────────────────
        return TrafficDecision(lane_left, lane_right,
                               BehaviorState.LANE_FOLLOW, "lane follow")

    # =========================================================================
    # Proceed after crossroad — interprets the AprilTag
    # =========================================================================

    def _proceed_after_crossroad(self, now: float) -> TrafficDecision:
        tag_id   = self.approach_tag_id
        tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"

        print(
            f"[CROSSROAD_GO] deciding based on tag={tag_name} (id={tag_id})"
        )

        # ── STOP: 5-second wait ──────────────────────────────────────────────
        if tag_id in STOP_TAGS:
            print(
                f"[CROSSROAD_GO] STOP sign → STOP_WAIT for {STOP_SIGN_WAIT_S}s"
            )
            self.state         = BehaviorState.STOP_WAIT
            self.state_until   = now + STOP_SIGN_WAIT_S
            self.active_tag_id = tag_id
            return TrafficDecision(0.0, 0.0, BehaviorState.STOP_WAIT,
                                   f"stop sign: {STOP_SIGN_WAIT_S}s wait", tag_id)

        # ── YIELD: creep forward ─────────────────────────────────────────────
        if tag_id in YIELD_TAGS:
            print("[CROSSROAD_GO] YIELD → creeping forward slowly")
            self.state                  = BehaviorState.YIELD_WAIT
            self.yield_creep_frames_done = 0
            self.active_tag_id          = tag_id
            return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                   BehaviorState.YIELD_WAIT, "yield: creeping",
                                   tag_id)

        # ── LEFT_OR_RIGHT: random choice ─────────────────────────────────────
        if tag_id in LEFT_RIGHT_TAGS:
            direction = random.choice(["left", "right"])
            print(
                f"[CROSSROAD_GO] LEFT_OR_RIGHT → random choice: {direction}"
            )
            return self._enter_intersection_decide(direction, now, tag_id)

        # ── LEFT_OR_FORWARD: random choice ───────────────────────────────────
        if tag_id in LEFT_FORWARD_TAGS:
            direction = random.choice(["left", "straight"])
            print(
                f"[CROSSROAD_GO] LEFT_OR_FORWARD → random choice: {direction}"
            )
            return self._enter_intersection_decide(direction, now, tag_id)

        # ── RIGHT_OR_FORWARD: random choice ──────────────────────────────────
        if tag_id in RIGHT_FORWARD_TAGS:
            direction = random.choice(["right", "straight"])
            print(
                f"[CROSSROAD_GO] RIGHT_OR_FORWARD → random choice: {direction}"
            )
            return self._enter_intersection_decide(direction, now, tag_id)

        # ── Unknown / no tag → go straight ───────────────────────────────────
        print(
            f"[CROSSROAD_GO] unknown tag (id={tag_id}) → defaulting to straight"
        )
        return self._enter_intersection_decide("straight", now, tag_id)

    def _enter_intersection_decide(
        self, direction: str, now: float, tag_id: Optional[int]
    ) -> TrafficDecision:
        """Cross the red line first, then execute the chosen turn."""
        self.state         = BehaviorState.INTERSECTION_DECIDE
        self.state_until   = now + CROSS_LINE_S
        self.chosen_turn   = direction
        self.active_tag_id = tag_id
        tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
        print(
            f"[INTERSECTION_DECIDE] crossing line for {CROSS_LINE_S}s "
            f"then turning {direction} (tag={tag_name})"
        )
        return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                               BehaviorState.INTERSECTION_DECIDE,
                               f"crossing line before {direction}", tag_id, direction)

    # =========================================================================
    # Peek sub-state machine
    # =========================================================================

    def _reset_peek(self):
        self.peek_phase        = PH_L1
        self.peek_frame_count  = 0
        self.peek_hold_until   = 0.0
        self.peek_vehicles     = []

    def _do_peek(self, frame_shape, tags, detections, now) -> TrafficDecision:
        """
        Peek sequence:
          PH_L1    → PEEK_FRAMES_L1 (3) frames rotating left
          PH_HOLD1 → hold PEEK_HOLD_1_S seconds  (scan LEFT)
          PH_R     → PEEK_FRAMES_R  (3) frames rotating right
          PH_HOLD2 → hold PEEK_HOLD_2_S seconds  (scan RIGHT)
          PH_L2    → PEEK_FRAMES_L2 frames re-aligning to centre
          PH_DONE  → evaluate & transition
        """
        scan_side = (
            "LEFT"  if self.peek_phase in (PH_L1, PH_HOLD1) else
            "RIGHT" if self.peek_phase in (PH_R,  PH_HOLD2) else
            "CENTRE"
        )

        # Collect any vehicle detections every frame
        v = self._vehicle_offset_and_side(frame_shape, detections)
        if v is not None:
            offset, side = v
            self.peek_vehicles.append((offset, side))
            print(
                f"[PEEK/{self.peek_phase}] *** VEHICLE SEEN *** "
                f"offset={offset:.3f} side={side} scanning={scan_side}"
            )
        else:
            print(
                f"[PEEK/{self.peek_phase}] no vehicle  scanning={scan_side}"
            )

        # Log any tags visible during peek
        for tag in tags:
            tid   = int(tag.get("id", -1))
            tname = TAG_NAMES.get(tid, f"id={tid}")
            print(
                f"[PEEK/{self.peek_phase}] tag visible: {tname} "
                f"area={tag.get('area', 0):.0f}px² scanning={scan_side}"
            )
            # Update approach tag if we now have a better reading
            if tid in ALL_KNOWN_TAGS and self.approach_tag_id is None:
                self.approach_tag_id = tid
                self.peek_for_stop   = tid in STOP_TAGS
                print(
                    f"[PEEK] approach_tag_id set to {tname} during peek"
                )

        # ── PH_L1 ─────────────────────────────────────────────────────────────
        if self.peek_phase == PH_L1:
            self.peek_frame_count += 1
            print(
                f"[PEEK/L1] frame {self.peek_frame_count}/{PEEK_FRAMES_L1} "
                f"rotating left"
            )
            if self.peek_frame_count >= PEEK_FRAMES_L1:
                self.peek_phase       = PH_HOLD1
                self.peek_hold_until  = now + PEEK_HOLD_1_S
                self.peek_frame_count = 0
                print(
                    f"[PEEK] L1 complete → HOLD1 "
                    f"scanning LEFT for {PEEK_HOLD_1_S}s"
                )
            return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT,
                                   "peek left frames")

        # ── PH_HOLD1 ──────────────────────────────────────────────────────────
        if self.peek_phase == PH_HOLD1:
            if now < self.peek_hold_until:
                remaining = self.peek_hold_until - now
                print(
                    f"[PEEK/HOLD1] scanning LEFT  {remaining:.2f}s remaining"
                )
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_LEFT,
                                       "peek hold1: scanning LEFT")
            self.peek_phase       = PH_R
            self.peek_frame_count = 0
            print(
                f"[PEEK] HOLD1 done → PH_R "
                f"({PEEK_FRAMES_R} frames rotating right)"
            )
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_RIGHT,
                                   "peek right start")

        # ── PH_R ──────────────────────────────────────────────────────────────
        if self.peek_phase == PH_R:
            self.peek_frame_count += 1
            print(
                f"[PEEK/R] frame {self.peek_frame_count}/{PEEK_FRAMES_R} "
                f"rotating right"
            )
            if self.peek_frame_count >= PEEK_FRAMES_R:
                self.peek_phase       = PH_HOLD2
                self.peek_hold_until  = now + PEEK_HOLD_2_S
                self.peek_frame_count = 0
                print(
                    f"[PEEK] R complete → HOLD2 "
                    f"scanning RIGHT for {PEEK_HOLD_2_S}s"
                )
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_RIGHT,
                                   "peek right frames")

        # ── PH_HOLD2 ──────────────────────────────────────────────────────────
        if self.peek_phase == PH_HOLD2:
            if now < self.peek_hold_until:
                remaining = self.peek_hold_until - now
                print(
                    f"[PEEK/HOLD2] scanning RIGHT  {remaining:.2f}s remaining"
                )
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_RIGHT,
                                       "peek hold2: scanning RIGHT")
            self.peek_phase       = PH_L2
            self.peek_frame_count = 0
            print(
                f"[PEEK] HOLD2 done → PH_L2 "
                f"({PEEK_FRAMES_L2} frames re-aligning)"
            )
            return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT,
                                   "peek re-align start")

        # ── PH_L2 ─────────────────────────────────────────────────────────────
        if self.peek_phase == PH_L2:
            self.peek_frame_count += 1
            print(
                f"[PEEK/L2] re-aligning frame "
                f"{self.peek_frame_count}/{PEEK_FRAMES_L2}"
            )
            if self.peek_frame_count >= PEEK_FRAMES_L2:
                self.peek_phase = PH_DONE
                print("[PEEK] L2 done — re-aligned to centre → evaluating")
            else:
                return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                       BehaviorState.CROSSROAD_PEEK_LEFT,
                                       "peek re-align")

        # ── PH_DONE — evaluate what we saw ───────────────────────────────────
        vehicles_seen = len(self.peek_vehicles)
        tag_name = TAG_NAMES.get(self.approach_tag_id, str(self.approach_tag_id))
        print(
            f"[PEEK] *** PEEK COMPLETE *** "
            f"{vehicles_seen} vehicle sample(s) collected "
            f"approach_tag={tag_name} (id={self.approach_tag_id})"
        )

        if self.peek_vehicles:
            best_offset, best_side = min(self.peek_vehicles, key=lambda x: x[0])
            print(
                f"[PEEK] vehicle detected — "
                f"best_offset={best_offset:.3f} side={best_side} "
                f"→ entering CROSSROAD_WAIT_VEHICLE to observe"
            )
            self.state                    = BehaviorState.CROSSROAD_WAIT_VEHICLE
            self.observed_vehicle_offsets = [best_offset]
            self.observed_vehicle_sides   = [best_side]
            self.vehicle_stationary_frames = 0
            self.observed_vehicle_until   = now + VEHICLE_OBSERVE_WINDOW_S
            return TrafficDecision(0.0, 0.0, self.state, "vehicle seen: observing")

        print(
            f"[PEEK] no vehicle seen — proceeding based on tag={tag_name}"
        )
        return self._proceed_after_crossroad(now)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _best_visible_tag(self, tags) -> Optional[int]:
        best_id, best_area = None, 0.0
        for tag in tags:
            tid  = int(tag.get("id", -1))
            area = float(tag.get("area", 0.0))
            if tid not in ALL_KNOWN_TAGS:
                continue
            if area > best_area:
                best_area, best_id = area, tid
        if best_id is not None:
            print(
                f"[TAG_SCAN] best tag: {TAG_NAMES.get(best_id, best_id)} "
                f"(id={best_id}) area={best_area:.0f}px²"
            )
        else:
            print("[TAG_SCAN] no known tag visible")
        return best_id

    def _vehicle_offset_and_side(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
    ) -> Optional[Tuple[float, str]]:
        h, w = frame_shape[:2]
        best: Optional[Tuple[float, str]] = None

        for bbox, score, cls_id in detections:
            if cls_id != 1:   # class 1 = vehicle/robot
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
            if area_frac < 0.004:
                continue

            cx_norm = (x1 + x2) / 2.0 / w
            offset  = abs(cx_norm - 0.5)
            side    = "left" if cx_norm < 0.5 else "right"

            if best is None or offset < best[0]:
                best = (offset, side)

        return best

    def _vehicle_is_approaching(self) -> Tuple[bool, Optional[str]]:
        s     = self.observed_vehicle_offsets
        sides = self.observed_vehicle_sides

        if len(s) < 2:
            print("[VEHICLE] not enough samples — assuming not approaching")
            return False, None

        delta       = s[0] - s[-1]   # positive = offset shrinking = approaching
        approaching = delta > APPROACH_THRESHOLD

        left_count    = sides.count("left")
        right_count   = sides.count("right")
        dominant_side = "left" if left_count >= right_count else "right"

        print(
            f"[VEHICLE] offset first={s[0]:.3f} last={s[-1]:.3f} "
            f"delta={delta:+.3f} approaching={approaching} "
            f"dominant_side={dominant_side} "
            f"(L={left_count} R={right_count})"
        )
        return approaching, dominant_side

    def _wheel_speeds_for(self, direction: Optional[str]) -> Tuple[float, float]:
        if direction == "left":
            return TURN_LEFT
        if direction == "right":
            return TURN_RIGHT
        return (DRIVE_SPEED, DRIVE_SPEED)   # straight

    def _begin_turn(self, direction: str, now: float, tag_id: Optional[int] = None):
        self.state         = BehaviorState.TURNING
        self.chosen_turn   = direction
        self.turn_until    = now + TURN_DURATION.get(direction, 1.0)
        self.active_tag_id = tag_id
        print(
            f"[TURN] beginning '{direction}' "
            f"for {TURN_DURATION.get(direction, 1.0):.2f}s"
        )

    def _reset_state(self):
        print(f"[STATE_RESET] {self.state.value} → LANE_FOLLOW")
        self.state                    = BehaviorState.LANE_FOLLOW
        self.state_until              = 0.0
        self.turn_until               = 0.0
        self.active_tag_id            = None
        self.chosen_turn              = None
        self.approach_tag_id          = None
        self.peek_for_stop            = False
        self.observed_vehicle_offsets = []
        self.observed_vehicle_sides   = []
        self.vehicle_stationary_frames = 0
        self.peek_vehicles            = []
        self.yield_creep_frames_done  = 0
        self.obstacle_clear_frames    = 0
        self._reset_peek()