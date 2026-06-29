import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
 
from tasks.final_project.packages.behavior_state import BehaviorState
 
STOP_TAGS          = {20, 24, 25, 26}  
YIELD_TAGS         = {39}              
LEFT_RIGHT_TAGS    = {11}
LEFT_FORWARD_TAGS  = {10}            
RIGHT_FORWARD_TAGS = {9}              
 
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
 
# DRIVE_SPEED = 0.35

# V1
DRIVE_SPEED = 0.4

CREEP_SPEED = 0.05
 
# TURN_RIGHT  = (0.9, 0.1)
# TURN_LEFT = (0.2, 0.8)
 
# -------------------------------ordis---------------------------------------------
TURN_RIGHT  = (1, 0.1)
TURN_LEFT = (0.2, 0.8)

PEEK_L = (-0.02, 0.5)
PEEK_R = (0.5, -0.02)
PEEK_SETTLE_S = 0.35
 
CROSS_LINE_S = 0
 
# -------------------------------ordis---------------------------------------------
# PRE_TURN_FRAMES = {
#     "left":     22,  
#     "right":    18,  
#     "straight": 10,
# }
 
#   -------------------------------------V1------------------
PRE_TURN_FRAMES = {
    "left":     19,  
    "right":    12,  
    "straight": 10,
}
 
# -------------------------------ordis---------------------------------------------
TURN_DURATION = {
    "left":     1.2,
    "right":    0.4,
    "straight": 1.0,
}

#   -------------------------------------V1------------------
# TURN_DURATION = {
#     "left":     1,
#     "right":    0.5,
#     "straight": 1.0,
# }

POST_TURN_FRAMES = {
    "left":     3,
    "right":    3,    
    "straight": 10,
}
 
OBSTACLE_CLEAR_FRAMES = 8
 
OBSTACLE_INTERRUPTIBLE_STATES = frozenset({
    BehaviorState.LANE_FOLLOW,
    BehaviorState.CROSSROAD_STOP,
    BehaviorState.CROSSROAD_WAIT_VEHICLE,
    BehaviorState.STOP_WAIT,
    BehaviorState.CROSSROAD_PRE_TURN,
    BehaviorState.YIELD_WAIT,
    BehaviorState.CROSSROAD_TURNING,
})
 
 
# -------------------------------ordis---------------------------------------------
PEEK_FRAMES_L1 = 2    # frames rotating left
PEEK_HOLD_1_S  = 2  # hold & scan LEFT
PEEK_FRAMES_R  = 4    # frames rotating right
PEEK_HOLD_2_S  = 2  # hold & scan RIGHT
PEEK_FRAMES_L2 = 2    # re-align frames  (L1=3 left, R=6 right, L2=3 left → net 0 ✓)
 
#  -------------------------------V1---------------------------------

# PEEK_FRAMES_L1 = 5    # frames rotating left
# PEEK_HOLD_1_S  = 2  # hold & scan LEFT
# PEEK_FRAMES_R  = 5    # frames rotating right
# PEEK_HOLD_2_S  = 2  # hold & scan RIGHT
# PEEK_FRAMES_L2 = 5    # re-align frames  (L1=3 left, R=6 right, L2=3 left → net 0 ✓)
 

# ---------------------------------------------------------------------------
# Misc timing constants
# ---------------------------------------------------------------------------

RED_LINE_COOLDOWN_S  = 15.0
SIGN_MEMORY_S        = 7.0  # remember a tag seen shortly before the red line
STOP_SIGN_WAIT_S     = 2.0
YIELD_CREEP_S        = 3
CROSSROAD_STOP_S     = 0.5
 
# Area threshold for vehicle detection during peek / observation.
# A vertical gate (cy_norm >= 0.25) separately filters horizon-level noise.
PEEK_MIN_AREA_FRACTION = 0.0015 
PEEK_MIN_CY_NORM = 0.15
PEEK_MIN_DETECTIONS_FOR_SIDE = 3
 
#   1) Peek left/right.
#   2) If a bot is seen, decide whether it moved during the peek samples.
#   3) Moving bot  -> stop and wait until it disappears.
#   4) Stationary bot -> ignore unless object_detector marks it as a frontal path threat.
VEHICLE_OBSERVE_FRAMES       = 8
VEHICLE_MOVING_MIN_DX        = 0.035  # normalized bbox centre movement
VEHICLE_MOVING_MIN_DAREA     = 0.003  # normalized bbox area change
VEHICLE_MOVING_MIN_DBOTTOM   = 0.040  # normalized bbox bottom movement
VEHICLE_MOVING_VOTES         = 2
VEHICLE_GONE_CONFIRM_FRAMES  = 6

PH_L1    = "L1"
PH_HOLD1 = "HOLD1"
PH_R     = "R"
PH_HOLD2 = "HOLD2"
PH_L2    = "L2"
PH_DONE  = "DONE"
 
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
    state:       BehaviorState = BehaviorState.LANE_FOLLOW
    state_until: float         = 0.0
    turn_until:  float         = 0.0
 
    active_tag_id: Optional[int] = None
    chosen_turn:   Optional[str] = None
 
    obstacle_clear_frames: int             = 0
    # State to return to once the duck/obstacle clears.  None means fall back
    resume_state:          Optional[BehaviorState] = None
 
    peek_phase:       str   = PH_L1
    peek_frame_count: int   = 0
    peek_hold_until:  float = 0.0
    peek_sample_after: float = 0.0
    peek_vehicles:    list  = field(default_factory=list)
 
    #   vehicle seen during peek -> classify as moving/stationary from bbox samples.
    #   moving vehicle -> wait here until it disappears from the triggering side.
    #   stationary vehicle -> proceed, unless it is also a frontal obstacle threat.
    vehicle_clear_frames:   int   = 0     # consecutive frames with no vehicle on triggering side
    waiting_vehicle_side:   Optional[str] = None  # "LEFT" or "RIGHT" — which peek side triggered wait
 
    waiting_vehicle_last_cx:   float         = 0.0  # cx_norm of the vehicle when wait began
    waiting_vehicle_last_area: float         = 0.0  # area_frac when wait began; used for moving-away check
 
    stop_completed:  bool          = False  # True once STOP_WAIT has been served for this intersection
    approach_tag_id: Optional[int] = None
    peek_for_stop:   bool          = False
 
    # Most recent known sign seen while approaching the intersection.
    # This survives temporary detector loss and is copied into
    # approach_tag_id when the red line is reached.
    last_seen_tag_id: Optional[int] = None
    last_seen_tag_at: float         = 0.0
 
    yield_until: float = 0.0
 
    pre_turn_frames_done:  int = 0
    post_turn_frames_done: int = 0
 
    last_crossroad_at: float = 0.0

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
 
        # Read and remember the sign continuously, not only on the exact
        # frame where the red line is detected. Once approach_tag_id is
        # locked for the current intersection, later scans cannot replace it.
        visible_tag_id = self._best_visible_tag(tags)
        if visible_tag_id is not None and self.approach_tag_id is None:
            self.last_seen_tag_id = visible_tag_id
            self.last_seen_tag_at = now
            print(
                f"[TAG_MEMORY] remembered {TAG_NAMES.get(visible_tag_id, visible_tag_id)} "
                f"(id={visible_tag_id})"
            )
 
        # Design principles:
        #   • DUCK threats interrupt EVERY state with no exceptions — a duck in
        #     the path is always stopped, even mid-turn or mid-yield.
        #   • VEHICLE threats do NOT interrupt crossroad states (PEEK, WAIT,
        #     PRE_TURN, TURNING, STOP_WAIT, YIELD_WAIT) because those states
        #     already handle vehicles via the yield / priority logic.  Letting a
        #     side-approaching vehicle also fire OBSTACLE_STOP would cause the
        #     robot to abort the intersection sequence and lose all crossroad state.
        #   • When a duck interrupts a crossroad state, resume_state is saved so
        #     the robot returns to exactly where it was once the duck clears,
        #     rather than restarting from LANE_FOLLOW and re-approaching the line.
        #   • OBSTACLE_STOP itself is NOT in OBSTACLE_INTERRUPTIBLE_STATES — it
        #     is handled separately below so a new duck while already stopped
        #     correctly resets the clear-frame counter.
 
        duck_threats    = [t for t in threats if getattr(t, 'is_duck',    t.cls_id == 0)]
        vehicle_threats = [t for t in threats if getattr(t, 'is_vehicle', t.cls_id == 1)]
 
        # Vehicles only interrupt during plain lane follow — not during any
        _crossroad_states = frozenset({
            BehaviorState.CROSSROAD_STOP,
            BehaviorState.CROSSROAD_PEEK_LEFT,
            BehaviorState.CROSSROAD_PEEK_RIGHT,
            BehaviorState.CROSSROAD_WAIT_VEHICLE,
            BehaviorState.STOP_WAIT,
            BehaviorState.CROSSROAD_PRE_TURN,
            BehaviorState.YIELD_WAIT,
            BehaviorState.CROSSROAD_TURNING,
        })
        in_crossroad = self.state in _crossroad_states
 
        # Ducks always block.
        # Vehicles also block if object_detector says they are in our frontal path.
        # This keeps stationary trucks safe: if they block the lane/intersection,
        # we stop until the path clears. Side vehicles are handled by peek logic.
        blocking_threats = duck_threats + vehicle_threats
 
        # Rank by bottom_norm (feet position = best single proximity proxy).
        def _rank(t):
            return getattr(t, 'bottom_norm', getattr(t, 'danger_score', t.area_frac))
 
        if self.state in OBSTACLE_INTERRUPTIBLE_STATES and blocking_threats:
            closest = max(blocking_threats, key=_rank)
            zone    = getattr(closest, 'proximity_zone', 'DANGER')
            is_duck = getattr(closest, 'is_duck', closest.cls_id == 0)
            saved = self.state
            self.resume_state          = saved
            self.obstacle_clear_frames = 0
            self.state                 = BehaviorState.OBSTACLE_STOP
            kind = "DUCK" if is_duck else "VEHICLE"
            print(
                f"[OBSTACLE] *** {'EMERGENCY' if zone == 'CRITICAL' else 'PRIORITY'} "
                f"STOP *** {kind} zone={zone} "
                f"bottom={getattr(closest, 'bottom_norm', '?'):.2f} "
                f"area={closest.area_frac:.4f} "
                f"cx={closest.cx_norm:.2f} side={closest.side} "
                f"interrupted={saved.value} resume_to={saved.value}"
            )
            return TrafficDecision(
                0.0, 0.0, BehaviorState.OBSTACLE_STOP,
                f"obstacle: {kind} [{zone}] in path (interrupted {saved.value})",
            )
 
        if self.state == BehaviorState.OBSTACLE_STOP:
            if blocking_threats:
                closest = max(blocking_threats, key=_rank)
                zone    = getattr(closest, 'proximity_zone', '?')
                is_duck = getattr(closest, 'is_duck', closest.cls_id == 0)
 
                # CRITICAL zone: reset counter — do not inch toward resuming
                # while the object is dangerously close.
                # DANGER/FAR zone: allow the counter to advance so the robot
                # doesn't wait forever if the threat is moving away.
                if zone == "CRITICAL":
                    self.obstacle_clear_frames = 0
                    print(
                        f"[OBSTACLE] CRITICAL — "
                        f"{'duck' if is_duck else 'vehicle'} "
                        f"cx={closest.cx_norm:.2f} area={closest.area_frac:.4f} "
                        f"bottom={getattr(closest, 'bottom_norm', '?'):.2f} "
                        f"— holding, counter reset"
                    )
                else:
                    # Still present but moving away — let counter tick so we
                    # don't freeze indefinitely behind a slow-moving truck.
                    print(
                        f"[OBSTACLE] {zone} zone — "
                        f"{'duck' if is_duck else 'vehicle'} still present "
                        f"cx={closest.cx_norm:.2f} area={closest.area_frac:.4f} "
                        f"— holding (counter NOT reset)"
                    )
                return TrafficDecision(0.0, 0.0, BehaviorState.OBSTACLE_STOP,
                                       f"obstacle [{zone}]: waiting for path to clear")
 
            self.obstacle_clear_frames += 1
            resume = self.resume_state or BehaviorState.LANE_FOLLOW
            print(
                f"[OBSTACLE] path clear? "
                f"({self.obstacle_clear_frames}/{OBSTACLE_CLEAR_FRAMES} frames) "
                f"resume_to={resume.value}"
            )
            if self.obstacle_clear_frames >= OBSTACLE_CLEAR_FRAMES:
                self.obstacle_clear_frames = 0
                self.resume_state          = None
 
                if resume == BehaviorState.LANE_FOLLOW:
                    print("[OBSTACLE] *** path clear *** resuming LANE_FOLLOW")
                    self.state = BehaviorState.LANE_FOLLOW
                    return TrafficDecision(lane_left, lane_right,
                                           BehaviorState.LANE_FOLLOW,
                                           "obstacle cleared: lane follow")
                else:
                    # Resume mid-crossroad: restore the interrupted state and let
                    # its handler re-evaluate timers on the next frame.
                    print(
                        f"[OBSTACLE] *** path clear *** resuming crossroad "
                        f"state={resume.value}"
                    )
                    self.state = resume
                    return TrafficDecision(0.0, 0.0, resume,
                                           f"obstacle cleared: resuming {resume.value}")
 
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
                tag_id = visible_tag_id
 
                # The sign may disappear from view just before the red line.
                # In that case, use the most recently remembered tag.
                memory_age = now - self.last_seen_tag_at
                if (
                    tag_id is None
                    and self.last_seen_tag_id is not None
                    and memory_age <= SIGN_MEMORY_S
                ):
                    tag_id = self.last_seen_tag_id
                    print(
                        f"[RED_LINE] using remembered tag "
                        f"{TAG_NAMES.get(tag_id, tag_id)} (id={tag_id}), "
                        f"last seen {memory_age:.2f}s ago"
                    )
 
                tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id is not None else "NO TAG"
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
            if self.approach_tag_id is None and visible_tag_id is not None:
                self.approach_tag_id = visible_tag_id
                self.peek_for_stop   = visible_tag_id in STOP_TAGS
                self.last_seen_tag_id = visible_tag_id
                self.last_seen_tag_at = now
                print(
                    f"[CROSSROAD_STOP] locked late-visible tag "
                    f"{TAG_NAMES.get(visible_tag_id, visible_tag_id)} "
                    f"(id={visible_tag_id})"
                )
 
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
 
        if self.state == BehaviorState.STOP_WAIT:
            if now < self.state_until:
                print(f"[STOP_WAIT] full stop, {self.state_until - now:.2f}s remaining")
                return TrafficDecision(0.0, 0.0, self.state, "full stop",
                                       self.active_tag_id or self.approach_tag_id)
 
            self.stop_completed = True
            print("[STOP_WAIT] timer done — stop satisfied, crossing straight")
            return self._enter_crossroad_pre_turn("straight", now, self.active_tag_id)
        if self.state == BehaviorState.CROSSROAD_WAIT_VEHICLE:
            snap = self._vehicle_on_side(
                frame_shape, detections, self.waiting_vehicle_side
            )
 
            if snap is not None:
                cur_cx, cur_area = snap
                # "Moving away" heuristic: the truck's bbox is meaningfully
                # smaller than when we first stopped.  It has crossed and is
                # receding, so we can safely proceed even while it is still
                # technically visible on the same side.
                area_shrunk = (
                    self.waiting_vehicle_last_area > 0 and
                    cur_area < self.waiting_vehicle_last_area * 0.6
                )
                if area_shrunk:
                    print(
                        f"[WAIT_MOVING_VEHICLE] vehicle moving away "
                        f"(area {cur_area:.4f} < 60% of {self.waiting_vehicle_last_area:.4f}) "
                        f"— treating as cleared"
                    )
                    self.vehicle_clear_frames = VEHICLE_GONE_CONFIRM_FRAMES  # fast-track confirm
                else:
                    self.vehicle_clear_frames = 0
                    print(
                        f"[WAIT_MOVING_VEHICLE] bot still on {self.waiting_vehicle_side} side "
                        f"cx={cur_cx:.2f} area={cur_area:.4f} — waiting"
                    )
                    return TrafficDecision(0.0, 0.0, self.state,
                                           "moving vehicle: waiting until gone",
                                           self.active_tag_id or self.approach_tag_id)
 
            self.vehicle_clear_frames += 1
            print(
                f"[WAIT_MOVING_VEHICLE] bot gone from {self.waiting_vehicle_side} side? "
                f"({self.vehicle_clear_frames}/{VEHICLE_GONE_CONFIRM_FRAMES} frames)"
            )
 
            if self.vehicle_clear_frames >= VEHICLE_GONE_CONFIRM_FRAMES:
                print("[WAIT_MOVING_VEHICLE] confirmed gone — proceeding by sign")
                self.vehicle_clear_frames        = 0
                self.waiting_vehicle_side        = None
                self.waiting_vehicle_last_cx     = 0.0
                self.waiting_vehicle_last_area   = 0.0
                return self._proceed_after_crossroad(now)
 
            return TrafficDecision(0.0, 0.0, self.state,
                                   "moving vehicle: confirming clear",
                                   self.active_tag_id or self.approach_tag_id)
 
        # ── 6. CROSSROAD_PRE_TURN ─────────────────────────────────────────────
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
 
        if tag_id in STOP_TAGS and not self.stop_completed:
            print(f"[CROSSROAD_GO] STOP sign → STOP_WAIT for {STOP_SIGN_WAIT_S}s")
            self.state         = BehaviorState.STOP_WAIT
            self.state_until   = now + STOP_SIGN_WAIT_S
            self.active_tag_id = tag_id
            return TrafficDecision(0.0, 0.0, BehaviorState.STOP_WAIT,
                                   f"stop sign: {STOP_SIGN_WAIT_S}s wait", tag_id)
 
        if tag_id in YIELD_TAGS:
            print(f"[CROSSROAD_GO] YIELD → creeping for {YIELD_CREEP_S}s")
            self.state         = BehaviorState.YIELD_WAIT
            self.yield_until   = now + YIELD_CREEP_S
            self.active_tag_id = tag_id
            return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                   BehaviorState.YIELD_WAIT, "yield: creeping", tag_id)
 
        if tag_id in LEFT_RIGHT_TAGS:
            direction = random.choice(["left", "right"])
            print(f"[CROSSROAD_GO] LEFT_OR_RIGHT → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)
 
        if tag_id in LEFT_FORWARD_TAGS:
            direction = random.choice(["left", "straight"])
            print(f"[CROSSROAD_GO] LEFT_OR_FORWARD → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)
 
        if tag_id in RIGHT_FORWARD_TAGS:
            direction = random.choice(["right", "straight"])
            print(f"[CROSSROAD_GO] RIGHT_OR_FORWARD → random choice: {direction}")
            return self._enter_crossroad_pre_turn(direction, now, tag_id)
 
        print(f"[CROSSROAD_GO] unknown tag (id={tag_id}) → defaulting to straight")
        return self._enter_crossroad_pre_turn("straight", now, tag_id)
 
    def _enter_crossroad_pre_turn(
        self, direction: str, now: float, tag_id: Optional[int]
    ) -> TrafficDecision:
        self.state                = BehaviorState.CROSSROAD_PRE_TURN
        self.state_until          = now + CROSS_LINE_S
        self.chosen_turn          = direction
        self.active_tag_id        = tag_id
        self.pre_turn_frames_done = 0
        tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
        print(
            f"[CROSSROAD_PRE_TURN] crossing line for {CROSS_LINE_S}s "
            f"then {PRE_TURN_FRAMES.get(direction, 0)} pre-turn frames "
            f"then rotating {direction} (tag={tag_name})"
        )
        return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                               BehaviorState.CROSSROAD_PRE_TURN,
                               f"crossing line before {direction}", tag_id, direction)
 
    def _reset_peek(self):
        self.peek_phase        = PH_L1
        self.peek_frame_count  = 0
        self.peek_hold_until   = 0.0
        self.peek_sample_after = 0.0
        self.peek_vehicles     = []
 
    def _do_peek(self, frame_shape, tags, detections, now) -> TrafficDecision:
        """
        Peek sequence:
          PH_L1    → PEEK_FRAMES_L1 frames rotating LEFT
          PH_HOLD1 → hold PEEK_HOLD_1_S seconds  (scan LEFT)
          PH_R     → PEEK_FRAMES_R  frames rotating RIGHT
          PH_HOLD2 → hold PEEK_HOLD_2_S seconds  (scan RIGHT)
          PH_L2    → PEEK_FRAMES_L2 frames re-aligning to centre
          PH_DONE  → evaluate & transition
 
        Movement rule:
          • Moving vehicle seen during LEFT or RIGHT scan  → CROSSROAD_WAIT_VEHICLE
          • Stationary vehicle seen                        → proceed based on tag
          • No vehicle seen                                → proceed based on tag
        """
        scan_side = (
            "LEFT"   if self.peek_phase in (PH_L1, PH_HOLD1) else
            "RIGHT"  if self.peek_phase in (PH_R,  PH_HOLD2) else
            "CENTRE"
        )

        snap = self._vehicle_snapshot(frame_shape, detections)
        collecting_scan = self.peek_phase in (PH_HOLD1, PH_HOLD2)
        ready_to_sample = collecting_scan and now >= self.peek_sample_after

        if snap is not None:
            cx, area_frac, bottom_norm, side = snap

            if ready_to_sample:
                self.peek_vehicles.append((cx, area_frac, bottom_norm, scan_side))
                print(
                    f"[PEEK/{self.peek_phase}] *** VEHICLE SAMPLE *** "
                    f"cx={cx:.2f} side={side} area={area_frac:.4f} "
                    f"bottom={bottom_norm:.2f} scan_side={scan_side}"
                )

            elif collecting_scan:
                remaining = max(0.0, self.peek_sample_after - now)
                print(
                    f"[PEEK/{self.peek_phase}] settling after rotation "
                    f"({remaining:.2f}s left) — truck ignored for motion"
                )

            else:
                print(
                    f"[PEEK/{self.peek_phase}] vehicle visible while rotating "
                    f"(ignored for motion) cx={cx:.2f} side={side}"
                )
        else:
            print(f"[PEEK/{self.peek_phase}] no vehicle  scanning={scan_side}")
 
        if (
            self.approach_tag_id is None
            and self.last_seen_tag_id is not None
            and now - self.last_seen_tag_at <= SIGN_MEMORY_S
        ):
            self.approach_tag_id = self.last_seen_tag_id
            self.peek_for_stop   = self.approach_tag_id in STOP_TAGS
            print(
                f"[PEEK] restored remembered approach tag "
                f"{TAG_NAMES.get(self.approach_tag_id, self.approach_tag_id)} "
                f"(id={self.approach_tag_id})"
            )
 
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
                self.peek_sample_after = now + PEEK_SETTLE_S
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
                self.peek_sample_after = now + PEEK_SETTLE_S
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
 
        tag_name = TAG_NAMES.get(self.approach_tag_id, str(self.approach_tag_id))
 
        left_samples  = [v for v in self.peek_vehicles if v[3] == "LEFT"]
        right_samples = [v for v in self.peek_vehicles if v[3] == "RIGHT"]
 
        left_seen   = len(left_samples)  >= PEEK_MIN_DETECTIONS_FOR_SIDE
        right_seen  = len(right_samples) >= PEEK_MIN_DETECTIONS_FOR_SIDE
        left_moving  = self._samples_show_motion(left_samples)
        right_moving = self._samples_show_motion(right_samples)
 
        print(
            f"[PEEK] *** PEEK COMPLETE ***  "
            f"left_seen={left_seen} samples={len(left_samples)} moving={left_moving}  "
            f"right_seen={right_seen} samples={len(right_samples)} moving={right_moving}  "
            f"total_detections={len(self.peek_vehicles)}  "
            f"approach_tag={tag_name} (id={self.approach_tag_id})"
        )
 
        if left_seen or right_seen:
            seen_sides = []

            if left_seen:
                seen_sides.append("LEFT")

            if right_seen:
                seen_sides.append("RIGHT")

            print(
                f"[PEEK] truck seen on {seen_sides} — "
                f"restarting full peek instead of waiting"
            )

            self._reset_peek()
            self.state = BehaviorState.CROSSROAD_PEEK_LEFT

            return TrafficDecision(
                0.0,
                0.0,
                BehaviorState.CROSSROAD_PEEK_LEFT,
                f"truck seen on {seen_sides}: re-peeking",
                self.approach_tag_id,
            )
 
        if left_seen or right_seen:
            stationary_sides = []
            if left_seen:
                stationary_sides.append("LEFT")
            if right_seen:
                stationary_sides.append("RIGHT")
            print(
                f"[PEEK] stationary bot(s) seen on {stationary_sides} — "
                f"not yielding; proceeding by sign unless path is blocked"
            )
            return self._proceed_after_crossroad(now)
 
        print(f"[PEEK] no vehicle seen — proceeding based on tag={tag_name}")
        return self._proceed_after_crossroad(now)

 
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
 
 
    def _vehicle_on_side(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
        side:        Optional[str],
    ) -> Optional[Tuple[float, float]]:
        """
        Returns (cx_norm, area_frac) of the largest qualifying vehicle on the
        given side, or None if no vehicle is present on that side.
 
        side="LEFT"  -> only vehicles with cx_norm < 0.5
        side="RIGHT" -> only vehicles with cx_norm >= 0.5
 
        Used by CROSSROAD_WAIT_VEHICLE so that a parked truck on the other
        side can never block progress, and so that a truck moving away
        (shrinking bbox) is not treated the same as one still approaching.
        """
        if side is None:
            return None
        h, w = frame_shape[:2]
        best_area = 0.0
        best: Optional[Tuple[float, float]] = None
        for bbox, score, cls_id in detections:
            if cls_id != 1:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
            if area_frac < PEEK_MIN_AREA_FRACTION:
                continue
            cy_norm = (y1 + y2) / 2.0 / h
            if cy_norm < PEEK_MIN_CY_NORM :
                continue
            cx_norm = (x1 + x2) / 2.0 / w
            on_side = (
                (side == "LEFT"  and cx_norm < 0.5) or
                (side == "RIGHT" and cx_norm >= 0.5)
            )
            if on_side and area_frac > best_area:
                best_area = area_frac
                best = (cx_norm, area_frac)
        return best
 
    def _vehicle_snapshot(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
    ) -> Optional[Tuple[float, float, float, str]]:
        """
        Returns (cx_norm, area_frac, bottom_norm, side) for the largest
        visible vehicle/truck bbox, or None if no vehicle qualifies.
 
        Used for collecting movement samples during the peek sequence.
        (CROSSROAD_WAIT_VEHICLE uses _vehicle_on_side instead.)
        """
        h, w = frame_shape[:2]
        best_area = 0.0
        best: Optional[Tuple[float, float, float, str]] = None
 
        for bbox, score, cls_id in detections:
            if cls_id != 1:
                continue
 
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
            if area_frac < PEEK_MIN_AREA_FRACTION:
                continue
 
            cy_norm = (y1 + y2) / 2.0 / h
            if cy_norm < PEEK_MIN_CY_NORM :
                continue
 
            cx_norm = (x1 + x2) / 2.0 / w
            bottom_norm = y2 / h
            side = "left" if cx_norm < 0.5 else "right"
 
            if area_frac > best_area:
                best_area = area_frac
                best = (cx_norm, area_frac, bottom_norm, side)
 
        return best
 
    def _samples_show_motion(
        self,
        samples: List[Tuple[float, float, float, str]],
    ) -> bool:
        """
        Decide if a peeked bot is moving by comparing bbox samples taken while
        looking in the same direction.
 
        samples item = (cx_norm, area_frac, bottom_norm, scan_side)
        """
        if len(samples) < max(2, PEEK_MIN_DETECTIONS_FOR_SIDE):
            return False
 
        first_cx, first_area, first_bottom, _ = samples[0]
        votes = 0
 
        for cx, area_frac, bottom_norm, _ in samples[1:]:
            dx = abs(cx - first_cx)
            da = abs(area_frac - first_area)
            db = abs(bottom_norm - first_bottom)
 
            if (
                dx >= VEHICLE_MOVING_MIN_DX
                or da >= VEHICLE_MOVING_MIN_DAREA
                or db >= VEHICLE_MOVING_MIN_DBOTTOM
            ):
                votes += 1
 
        print(
            f"[VEHICLE_MOTION] samples={len(samples)} votes={votes} "
            f"need={VEHICLE_MOVING_VOTES}"
        )
        return votes >= VEHICLE_MOVING_VOTES
 
    def _vehicle_offset_and_side(
        self,
        frame_shape: Tuple[int, int, int],
        detections:  List[tuple],
    ) -> Optional[Tuple[float, str, float]]:
        """
        Returns (offset_from_centre, side, cx_norm) for the most prominent
        vehicle in view (largest bbox area), or None if none qualify.
 
        "Most prominent" = largest area, NOT most centred.  During a left peek
        the approaching truck is near the left edge of the frame; sorting by
        smallest offset would prefer centred noise over the actual threat.
 
        Uses PEEK_MIN_AREA_FRACTION so partially-visible side vehicles are caught.
        A vertical gate (cy_norm >= 0.25) filters far-away horizon detections.
        """
        h, w = frame_shape[:2]
        best_area = 0.0
        best: Optional[Tuple[float, str, float]] = None
 
        for bbox, score, cls_id in detections:
            if cls_id != 1:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
            if area_frac < PEEK_MIN_AREA_FRACTION:
                continue
            cy_norm = (y1 + y2) / 2.0 / h
            if cy_norm < 0.25:
                continue
            cx_norm = (x1 + x2) / 2.0 / w
            if area_frac > best_area:
                best_area = area_frac
                offset = abs(cx_norm - 0.5)
                side   = "left" if cx_norm < 0.5 else "right"
                best   = (offset, side, cx_norm)
 
        return best
 
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
        self.state                = BehaviorState.LANE_FOLLOW
        self.state_until          = 0.0
        self.turn_until           = 0.0
        self.active_tag_id        = None
        self.chosen_turn          = None
        self.approach_tag_id      = None
        self.peek_for_stop        = False
        self.last_seen_tag_id     = None
        self.last_seen_tag_at     = 0.0
        self.yield_until          = 0.0
        self.pre_turn_frames_done  = 0
        self.post_turn_frames_done = 0
        self.obstacle_clear_frames = 0
        self.resume_state          = None
        self.vehicle_clear_frames  = 0
        self.waiting_vehicle_side  = None
        self.waiting_vehicle_last_cx    = 0.0
        self.waiting_vehicle_last_area  = 0.0
        self.stop_completed          = False
        self._reset_peek()