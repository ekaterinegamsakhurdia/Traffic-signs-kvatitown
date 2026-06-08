import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tasks.final_project.packages.behavior_state import BehaviorState


# ---------------------------------------------------------------------------
# AprilTag ID → behaviour
# ---------------------------------------------------------------------------
STOP_TAGS          = {20, 24, 25, 26}   # stop for 5 seconds
YIELD_TAGS         = {39}               # creep slowly then go straight
LEFT_RIGHT_TAGS    = {11}              # turn left OR right (random)
LEFT_FORWARD_TAGS  = {10}             # turn left OR go forward (random)
RIGHT_FORWARD_TAGS = {9}              # turn right OR go forward (random)

ALL_KNOWN_TAGS = (
    STOP_TAGS | YIELD_TAGS |
    LEFT_RIGHT_TAGS | LEFT_FORWARD_TAGS | RIGHT_FORWARD_TAGS
)

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
DRIVE_SPEED = 0.3
CREEP_SPEED = 0.05   # slow for yield

# Negative inner wheel gives a tighter ~in-place pivot for 90° turns
TURN_RIGHT  = (0.3, -0.15)
TURN_LEFT = (-0.15, 0.3)

PEEK_L = (0.02, 0.14)
PEEK_R = (0.14, 0.02)


# ---------------------------------------------------------------------------
# Intersection timing / frame tuning
# ---------------------------------------------------------------------------

# Seconds to drive straight immediately after the red line (clears the line)
CROSS_LINE_S = 0.3

# Frames to drive straight BEFORE rotating.
# Left needs more: robot is in the right lane and must reach the intersection
# centre before swinging left. Right needs fewer: already near the right side.
PRE_TURN_FRAMES = {
    "left":     25,  # tune me
    "right":    12,   # tune me
    "straight": 0,
}

TURN_DURATION = {
    "left":     0.27,   # tune me
    "right":    0.27,   # tune me
    "straight": 1.0,
}

# Frames to drive straight AFTER rotating before handing back to lane follow.
POST_TURN_FRAMES = {
    "left":     15, 
    "right":    15,    
    "straight": 10,
}


# ---------------------------------------------------------------------------
# Obstacle-stop parameters
# ---------------------------------------------------------------------------
OBSTACLE_CLEAR_FRAMES = 4 

OBSTACLE_INTERRUPTIBLE_STATES = frozenset({
    BehaviorState.LANE_FOLLOW,
    BehaviorState.CROSSROAD_PRE_TURN,   
    BehaviorState.YIELD_WAIT,           
})


# ---------------------------------------------------------------------------
# Peek parameters — 3 frames each side
# ---------------------------------------------------------------------------
PEEK_FRAMES_L1 = 5    # frames rotating left
PEEK_HOLD_1_S  = 1.0  # hold & scan LEFT
PEEK_FRAMES_R  = 10    # frames rotating right
PEEK_HOLD_2_S  = 1.0  # hold & scan RIGHT
PEEK_FRAMES_L2 = 7    # re-align frames back to centre
                       # NOTE: L1=3 left, R=6 right, L2=3 left → net 0 rotation ✓


# ---------------------------------------------------------------------------
# Misc timing constants
# ---------------------------------------------------------------------------
RED_LINE_COOLDOWN_S        = 13.0
STOP_SIGN_WAIT_S           = 4.0
YIELD_CREEP_S              = 4
VEHICLE_OBSERVE_WINDOW_S   = 2
APPROACH_THRESHOLD         = 0.03
STATIONARY_FRAME_THRESHOLD = 0.01
STATIONARY_FRAMES_IGNORE   = 5
CROSSROAD_STOP_S           = 1.0   

PEEK_MIN_AREA_FRACTION = 0.008


# ---------------------------------------------------------------------------
# Peek sub-state labels
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
    # ── core state ───────────────────────────────────────────────────────────
    state:       BehaviorState = BehaviorState.LANE_FOLLOW
    state_until: float         = 0.0
    turn_until:  float         = 0.0

    active_tag_id: Optional[int] = None
    chosen_turn:   Optional[str] = None

    # ── obstacle stop ─────────────────────────────────────────────────────────
    obstacle_clear_frames: int = 0

    # ── peek sub-state ────────────────────────────────────────────────────────
    peek_phase:       str   = PH_L1
    peek_frame_count: int   = 0
    peek_hold_until:  float = 0.0
    peek_vehicles:    list  = field(default_factory=list)

    # ── vehicle observation ───────────────────────────────────────────────────
    observed_vehicle_offsets:  list  = field(default_factory=list)
    observed_vehicle_sides:    list  = field(default_factory=list)
    observed_vehicle_until:    float = 0.0
    vehicle_stationary_frames: int   = 0

    # ── sign read at red-line approach ────────────────────────────────────────
    approach_tag_id: Optional[int] = None
    # When True, the active tag is a stop sign. After peek the robot skips the
    # CROSSROAD_WAIT_VEHICLE observation phase because the mandatory 5-second
    # STOP_WAIT already provides the yield window for any crossing traffic.
    peek_for_stop:   bool          = False

    # ── yield creep timing ────────────────────────────────────────────────────
    yield_until: float = 0.0

    # ── pre / post-turn straight-drive frame counters ─────────────────────────
    pre_turn_frames_done:  int = 0
    post_turn_frames_done: int = 0

    # ── crossroad cooldown ────────────────────────────────────────────────────
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

        # ── 0. OBSTACLE STOP ──────────────────────────────────────────────────
        # Interrupts LANE_FOLLOW, CROSSROAD_PRE_TURN, and YIELD_WAIT.
        # Peek and turning states are intentionally excluded — see
        # OBSTACLE_INTERRUPTIBLE_STATES definition for the rationale.
        if self.state in OBSTACLE_INTERRUPTIBLE_STATES and threats:
            closest = max(threats, key=lambda t: t.area_frac)
            print(
                f"[OBSTACLE] *** COLLISION RISK *** "
                f"label={closest.label} area={closest.area_frac:.4f} "
                f"cx={closest.cx_norm:.2f} cy={closest.cy_norm:.2f} "
                f"side={closest.side} → STOPPING "
                f"(interrupted from {self.state.value})"
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

        # ── 1. RED LINE → enter crossroad sequence ────────────────────────────
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
                self.approach_tag_id   = tag_id
                self.last_crossroad_at = now
                self.state             = BehaviorState.CROSSROAD_STOP
                self.state_until       = now + CROSSROAD_STOP_S
                self.peek_for_stop     = tag_id in STOP_TAGS
                return TrafficDecision(0.0, 0.0, BehaviorState.CROSSROAD_STOP,
                                       f"red line: stopping (tag={tag_name})", tag_id)

        # ── 2. CROSSROAD_STOP — brief halt, improve tag reading ───────────────
        if self.state == BehaviorState.CROSSROAD_STOP:
            better_tag = self._best_visible_tag(tags)
            if better_tag is not None and better_tag != self.approach_tag_id:
                old = TAG_NAMES.get(self.approach_tag_id, str(self.approach_tag_id))
                new = TAG_NAMES.get(better_tag, str(better_tag))
                print(f"[CROSSROAD_STOP] tag updated {old} → {new} (id={better_tag})")
                self.approach_tag_id = better_tag
                self.peek_for_stop   = better_tag in STOP_TAGS

            if now < self.state_until:
                tag_name = TAG_NAMES.get(self.approach_tag_id, "?")
                print(
                    f"[CROSSROAD_STOP] stopped at red line "
                    f"{self.state_until - now:.2f}s remaining "
                    f"tag={tag_name} (id={self.approach_tag_id}) "
                    f"peek_for_stop={self.peek_for_stop}"
                )
                return TrafficDecision(0.0, 0.0, BehaviorState.CROSSROAD_STOP,
                                       f"crossroad stop ({tag_name})",
                                       self.approach_tag_id)

            tag_name = TAG_NAMES.get(self.approach_tag_id, "UNKNOWN")
            print(
                f"[CROSSROAD_STOP] brief stop done — starting peek (L→R) "
                f"for tag={tag_name} peek_for_stop={self.peek_for_stop}"
            )
            self._reset_peek()
            self.state = BehaviorState.CROSSROAD_PEEK_LEFT
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 3. PEEK sequence ──────────────────────────────────────────────────
        if self.state in (BehaviorState.CROSSROAD_PEEK_LEFT,
                          BehaviorState.CROSSROAD_PEEK_RIGHT):
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 4. STOP_WAIT — full 5-second stop (stop sign) ─────────────────────
        # Note: stop signs always exit straight. The mandatory 5-second wait
        # is what yields to crossing traffic; no separate direction choice is made.
        if self.state == BehaviorState.STOP_WAIT:
            if now < self.state_until:
                print(f"[STOP_WAIT] full stop, {self.state_until - now:.2f}s remaining")
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

            print("[STOP_WAIT] clear — 5s wait done, crossing straight")
            return self._enter_crossroad_pre_turn("straight", now, self.active_tag_id)

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
                    f"[VEHICLE_OBSERVE] vehicle seen offset={offset:.3f} side={side} "
                    f"stationary_frames={self.vehicle_stationary_frames}"
                )

                if self.vehicle_stationary_frames >= STATIONARY_FRAMES_IGNORE:
                    print(
                        f"[VEHICLE_OBSERVE] parked for {STATIONARY_FRAMES_IGNORE} "
                        f"frames → ignoring, proceeding"
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

        # ── 6. CROSSROAD_PRE_TURN ─────────────────────────────────────────────
        # Phase 1: time-based initial line cross
        # Phase 2: frame-counted straight drive before rotating
        if self.state == BehaviorState.CROSSROAD_PRE_TURN:
            direction = self.chosen_turn or "straight"

            if now < self.state_until:
                print(
                    f"[CROSSROAD_PRE_TURN] phase1 crossing red line "
                    f"{self.state_until - now:.2f}s remaining  direction={direction}"
                )
                return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                       self.state, "crossing line")

            needed_frames = PRE_TURN_FRAMES.get(direction, 0)
            if self.pre_turn_frames_done < needed_frames:
                self.pre_turn_frames_done += 1
                print(
                    f"[CROSSROAD_PRE_TURN] phase2 pre-turn straight "
                    f"frame {self.pre_turn_frames_done}/{needed_frames} "
                    f"direction={direction}"
                )
                return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                       self.state, f"pre-turn straight ({direction})")

            tag_id   = self.approach_tag_id
            tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
            print(
                f"[CROSSROAD_PRE_TURN] pre-turn done ({needed_frames} frames) — "
                f"tag={tag_name} NOW rotating {direction}"
            )
            self.pre_turn_frames_done = 0
            self._begin_turn(direction, now, tag_id)
            l, r = self._wheel_speeds_for(direction)
            return TrafficDecision(l, r, BehaviorState.CROSSROAD_TURNING,
                                   f"turn {direction}", tag_id, direction)

        # ── 7. YIELD creep ────────────────────────────────────────────────────
        if self.state == BehaviorState.YIELD_WAIT:
            if now < self.yield_until:
                print(
                    f"[YIELD] creeping at speed={CREEP_SPEED} "
                    f"{self.yield_until - now:.2f}s remaining"
                )
                return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                       self.state, "yield: creeping",
                                       self.approach_tag_id)
            print("[YIELD] creep done → going straight")
            # FIX: use _begin_turn instead of manually setting state fields,
            # ensuring post_turn_frames_done is reset consistently.
            self.active_tag_id = self.approach_tag_id
            self._begin_turn("straight", now, self.approach_tag_id)
            l, r = self._wheel_speeds_for("straight")
            return TrafficDecision(l, r,
                                   BehaviorState.CROSSROAD_TURNING, "yield done: straight",
                                   self.active_tag_id, "straight")

        # ── 8. CROSSROAD_TURNING ─────────────────────────────────────────────
        if self.state == BehaviorState.CROSSROAD_TURNING:
            if now < self.turn_until:
                l, r = self._wheel_speeds_for(self.chosen_turn)
                print(
                    f"[CROSSROAD_TURNING] '{self.chosen_turn}' "
                    f"{self.turn_until - now:.2f}s remaining"
                )
                return TrafficDecision(l, r, self.state,
                                       f"turning {self.chosen_turn}",
                                       self.active_tag_id, self.chosen_turn)

            # Rotation done — drive straight for POST_TURN_FRAMES
            needed = POST_TURN_FRAMES.get(self.chosen_turn, 0)
            if self.post_turn_frames_done < needed:
                self.post_turn_frames_done += 1
                print(
                    f"[CROSSROAD_TURNING] '{self.chosen_turn}' rotation done — "
                    f"post-turn straight frame "
                    f"{self.post_turn_frames_done}/{needed}"
                )
                return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED, self.state,
                                       f"post-turn straight ({self.chosen_turn})",
                                       self.active_tag_id, self.chosen_turn)

            print(
                f"[CROSSROAD_TURNING] '{self.chosen_turn}' post-turn straight done "
                f"→ snapping back to LANE_FOLLOW"
            )
            self._reset_state()
            return TrafficDecision(lane_left, lane_right,
                                   BehaviorState.LANE_FOLLOW, "turn done: lane follow")

        # ── LANE_FOLLOW fallthrough ────────────────────────────────────────────
        return TrafficDecision(lane_left, lane_right,
                               BehaviorState.LANE_FOLLOW, "lane follow")

    # =========================================================================
    # Proceed after crossroad — interprets the AprilTag
    # =========================================================================

    def _proceed_after_crossroad(self, now: float) -> TrafficDecision:
        tag_id   = self.approach_tag_id
        tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
        print(f"[CROSSROAD_GO] deciding based on tag={tag_name} (id={tag_id})")

        # STOP: 5-second mandatory wait (replaces vehicle-observation phase)
        if tag_id in STOP_TAGS:
            print(f"[CROSSROAD_GO] STOP sign → STOP_WAIT for {STOP_SIGN_WAIT_S}s")
            self.state         = BehaviorState.STOP_WAIT
            self.state_until   = now + STOP_SIGN_WAIT_S
            self.active_tag_id = tag_id
            return TrafficDecision(0.0, 0.0, BehaviorState.STOP_WAIT,
                                   f"stop sign: {STOP_SIGN_WAIT_S}s wait", tag_id)

        # YIELD: creep for YIELD_CREEP_S seconds
        if tag_id in YIELD_TAGS:
            print(f"[CROSSROAD_GO] YIELD → creeping for {YIELD_CREEP_S}s")
            self.state         = BehaviorState.YIELD_WAIT
            self.yield_until   = now + YIELD_CREEP_S
            self.active_tag_id = tag_id
            return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                   BehaviorState.YIELD_WAIT, "yield: creeping", tag_id)

        # LEFT_OR_RIGHT: random choice
        if tag_id in LEFT_RIGHT_TAGS:
            direction = random.choice(["left", "right"])
            print(f"[CROSSROAD_GO] LEFT_OR_RIGHT → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)

        # LEFT_OR_FORWARD: random choice
        if tag_id in LEFT_FORWARD_TAGS:
            direction = random.choice(["left", "straight"])
            print(f"[CROSSROAD_GO] LEFT_OR_FORWARD → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)

        # RIGHT_OR_FORWARD: random choice
        if tag_id in RIGHT_FORWARD_TAGS:
            direction = random.choice(["right", "straight"])
            print(f"[CROSSROAD_GO] RIGHT_OR_FORWARD → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)

        # Unknown / no tag → go straight
        print(f"[CROSSROAD_GO] unknown tag (id={tag_id}) → defaulting to straight")
        return self._enter_crossroad_pre_turn("straight", now, tag_id)

    def _enter_crossroad_pre_turn(
        self, direction: str, now: float, tag_id: Optional[int]
    ) -> TrafficDecision:
        """Cross the red line first (timed), then count pre-turn frames, then rotate."""
        self.state                = BehaviorState.CROSSROAD_PRE_TURN
        self.state_until          = now + CROSS_LINE_S
        self.chosen_turn          = direction
        self.active_tag_id        = tag_id
        self.pre_turn_frames_done = 0   # always reset on fresh entry
        tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
        print(
            f"[CROSSROAD_PRE_TURN] crossing line for {CROSS_LINE_S}s "
            f"then {PRE_TURN_FRAMES.get(direction, 0)} pre-turn frames "
            f"then rotating {direction} (tag={tag_name})"
        )
        return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                               BehaviorState.CROSSROAD_PRE_TURN,
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
          PH_L1    → PEEK_FRAMES_L1 frames rotating left
          PH_HOLD1 → hold PEEK_HOLD_1_S seconds  (scan LEFT)
          PH_R     → PEEK_FRAMES_R  frames rotating right
          PH_HOLD2 → hold PEEK_HOLD_2_S seconds  (scan RIGHT)
          PH_L2    → PEEK_FRAMES_L2 frames re-aligning to centre
          PH_DONE  → evaluate & transition
        """
        scan_side = (
            "LEFT"   if self.peek_phase in (PH_L1, PH_HOLD1) else
            "RIGHT"  if self.peek_phase in (PH_R,  PH_HOLD2) else
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
            print(f"[PEEK/{self.peek_phase}] no vehicle  scanning={scan_side}")

        # Log any tags visible during peek and update approach_tag_id if needed
        for tag in tags:
            tid   = int(tag.get("id", -1))
            tname = TAG_NAMES.get(tid, f"id={tid}")
            print(
                f"[PEEK/{self.peek_phase}] tag visible: {tname} "
                f"area={tag.get('area', 0):.0f}px² scanning={scan_side}"
            )
            if tid in ALL_KNOWN_TAGS and self.approach_tag_id is None:
                self.approach_tag_id = tid
                self.peek_for_stop   = tid in STOP_TAGS
                print(f"[PEEK] approach_tag_id set to {tname} during peek")

        # ── PH_L1 ─────────────────────────────────────────────────────────────
        if self.peek_phase == PH_L1:
            self.peek_frame_count += 1
            print(
                f"[PEEK/L1] frame {self.peek_frame_count}/{PEEK_FRAMES_L1} rotating left"
            )
            if self.peek_frame_count >= PEEK_FRAMES_L1:
                self.peek_phase       = PH_HOLD1
                self.peek_hold_until  = now + PEEK_HOLD_1_S
                self.peek_frame_count = 0
                print(f"[PEEK] L1 complete → HOLD1 scanning LEFT for {PEEK_HOLD_1_S}s")
            return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT, "peek left frames")

        # ── PH_HOLD1 ──────────────────────────────────────────────────────────
        if self.peek_phase == PH_HOLD1:
            if now < self.peek_hold_until:
                print(
                    f"[PEEK/HOLD1] scanning LEFT  "
                    f"{self.peek_hold_until - now:.2f}s remaining"
                )
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_LEFT,
                                       "peek hold1: scanning LEFT")
            self.peek_phase       = PH_R
            self.peek_frame_count = 0
            print(f"[PEEK] HOLD1 done → PH_R ({PEEK_FRAMES_R} frames rotating right)")
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_RIGHT, "peek right start")

        # ── PH_R ──────────────────────────────────────────────────────────────
        if self.peek_phase == PH_R:
            self.peek_frame_count += 1
            print(
                f"[PEEK/R] frame {self.peek_frame_count}/{PEEK_FRAMES_R} rotating right"
            )
            if self.peek_frame_count >= PEEK_FRAMES_R:
                self.peek_phase       = PH_HOLD2
                self.peek_hold_until  = now + PEEK_HOLD_2_S
                self.peek_frame_count = 0
                print(f"[PEEK] R complete → HOLD2 scanning RIGHT for {PEEK_HOLD_2_S}s")
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_RIGHT, "peek right frames")

        # ── PH_HOLD2 ──────────────────────────────────────────────────────────
        if self.peek_phase == PH_HOLD2:
            if now < self.peek_hold_until:
                print(
                    f"[PEEK/HOLD2] scanning RIGHT  "
                    f"{self.peek_hold_until - now:.2f}s remaining"
                )
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_RIGHT,
                                       "peek hold2: scanning RIGHT")
            self.peek_phase       = PH_L2
            self.peek_frame_count = 0
            print(f"[PEEK] HOLD2 done → PH_L2 ({PEEK_FRAMES_L2} frames re-aligning)")
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

        # ── PH_DONE — evaluate what we saw ────────────────────────────────────
        vehicles_seen = len(self.peek_vehicles)
        tag_name = TAG_NAMES.get(self.approach_tag_id, str(self.approach_tag_id))
        print(
            f"[PEEK] *** PEEK COMPLETE *** "
            f"{vehicles_seen} vehicle sample(s)  "
            f"approach_tag={tag_name} (id={self.approach_tag_id}) "
            f"peek_for_stop={self.peek_for_stop}"
        )

        if self.peek_vehicles and not self.peek_for_stop:
            # FIX: use the most recent (last) detection as the seed for approach
            # tracking. It is the freshest position sample, and its side is the
            # most reliable indicator of where the vehicle currently is.
            # (Previously used min-offset which incorrectly favoured centre-frame
            # detections over meaningful side detections.)
            seed_offset, seed_side = self.peek_vehicles[-1]
            print(
                f"[PEEK] vehicle detected — "
                f"seed_offset={seed_offset:.3f} seed_side={seed_side} "
                f"→ CROSSROAD_WAIT_VEHICLE"
            )
            self.state                     = BehaviorState.CROSSROAD_WAIT_VEHICLE
            self.observed_vehicle_offsets  = [seed_offset]
            self.observed_vehicle_sides    = [seed_side]
            self.vehicle_stationary_frames = 0
            self.observed_vehicle_until    = now + VEHICLE_OBSERVE_WINDOW_S
            return TrafficDecision(0.0, 0.0, self.state, "vehicle seen: observing")

        if self.peek_vehicles and self.peek_for_stop:
            # Stop sign: skip vehicle-observation phase. STOP_WAIT (5s mandatory
            # halt + final vehicle presence check) is the yield mechanism.
            print(
                "[PEEK] vehicle seen but stop sign — "
                "skipping vehicle observation, STOP_WAIT handles yield"
            )
        else:
            print(f"[PEEK] no vehicle seen — proceeding based on tag={tag_name}")

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
        """
        Returns (offset_from_centre, side) for the most prominent vehicle
        in view, or None if none qualify.

        Uses PEEK_MIN_AREA_FRACTION (0.008) rather than the frontal-threat
        threshold (0.012) so partially-visible side-approaching vehicles are
        still caught during the peek phase.

        A vertical gate (cy_norm >= 0.25) rejects far-away horizon detections
        that would otherwise inflate peek_vehicles with noise.
        """
        h, w = frame_shape[:2]
        best: Optional[Tuple[float, str]] = None
        for bbox, score, cls_id in detections:
            if cls_id != 1:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
            if area_frac < PEEK_MIN_AREA_FRACTION:
                continue
            cy_norm = (y1 + y2) / 2.0 / h
            if cy_norm < 0.25:   # near horizon → too far away, ignore
                continue
            cx_norm = (x1 + x2) / 2.0 / w
            offset  = abs(cx_norm - 0.5)
            side    = "left" if cx_norm < 0.5 else "right"
            if best is None or offset < best[0]:
                best = (offset, side)
        return best

    def _vehicle_is_approaching(self) -> Tuple[bool, Optional[str]]:
        """
        Determines whether the observed vehicle is moving towards our robot.

        Uses a first-half vs second-half mean comparison over all accumulated
        samples, which is more robust against single-frame outliers than a
        simple first-vs-last delta.

        An additional sanity check requires both of the last two samples to be
        below the first sample, guarding against a single high outlier at the
        start of the window inflating the first-half mean.
        """
        s     = self.observed_vehicle_offsets
        sides = self.observed_vehicle_sides
        if len(s) < 2:
            print("[VEHICLE] not enough samples — assuming not approaching")
            return False, None

        mid              = max(1, len(s) // 2)
        first_half_mean  = sum(s[:mid]) / mid
        second_half_mean = sum(s[mid:]) / max(1, len(s) - mid)
        delta            = first_half_mean - second_half_mean  # positive = approaching

        # With ≥4 samples, require the last two to both be below the first
        # reading to filter out transient noise pushing the first-half mean up.
        if len(s) >= 4:
            last_two_ok = (s[-1] < s[0]) and (s[-2] < s[0])
        else:
            last_two_ok = True

        approaching   = delta > APPROACH_THRESHOLD and last_two_ok
        left_count    = sides.count("left")
        right_count   = sides.count("right")
        dominant_side = "left" if left_count >= right_count else "right"
        print(
            f"[VEHICLE] samples={len(s)} "
            f"first_mean={first_half_mean:.3f} second_mean={second_half_mean:.3f} "
            f"delta={delta:+.3f} last_two_ok={last_two_ok} "
            f"approaching={approaching} dominant_side={dominant_side} "
            f"(L={left_count} R={right_count})"
        )
        return approaching, dominant_side

    def _wheel_speeds_for(self, direction: Optional[str]) -> Tuple[float, float]:
        if direction == "left":
            return TURN_LEFT
        if direction == "right":
            return TURN_RIGHT
        return (DRIVE_SPEED, DRIVE_SPEED)

    def _begin_turn(self, direction: str, now: float, tag_id: Optional[int] = None):
        self.state                 = BehaviorState.CROSSROAD_TURNING
        self.chosen_turn           = direction
        self.turn_until            = now + TURN_DURATION.get(direction, 1.0)
        self.active_tag_id         = tag_id
        self.post_turn_frames_done = 0
        print(
            f"[TURN] beginning '{direction}' "
            f"for {TURN_DURATION.get(direction, 1.0):.2f}s"
        )

    def _reset_state(self):
        print(f"[STATE_RESET] {self.state.value} → LANE_FOLLOW")
        self.state                     = BehaviorState.LANE_FOLLOW
        self.state_until               = 0.0
        self.turn_until                = 0.0
        self.active_tag_id             = None
        self.chosen_turn               = None
        self.approach_tag_id           = None
        self.peek_for_stop             = False
        self.observed_vehicle_offsets  = []
        self.observed_vehicle_sides    = []
        self.vehicle_stationary_frames = 0
        self.peek_vehicles             = []
        self.yield_until               = 0.0
        self.pre_turn_frames_done      = 0
        self.post_turn_frames_done     = 0
        self.obstacle_clear_frames     = 0
        self._reset_peek()