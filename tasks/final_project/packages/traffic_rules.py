import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tasks.final_project.packages.behavior_state import BehaviorState
# ---------------------------------------------------------------------------
# Tag ID → meaning
# ---------------------------------------------------------------------------
STOP_TAGS  = {1}
YIELD_TAGS = {4}

INTERSECTION_OPTIONS = {
    2: "left",
    3: "right",
}

TAG_NAMES = {
    25: "STOP",
    9: "LEFT_ONLY",
    3: "RIGHT_ONLY",
    4: "YIELD",
}

# ---------------------------------------------------------------------------
# Motion commands  (left_wheel, right_wheel)
# ---------------------------------------------------------------------------
DRIVE_SPEED   = 0.16   # normal straight speed
CREEP_SPEED   = 0.03   # slow creep while decelerating toward line
TURN_LEFT     = (0.16, 0.02)
TURN_RIGHT    = (0.02, 0.16)
PEEK_L        = (0.16, 0.02)   # rotate left
PEEK_R        = (0.02, 0.16)   # rotate right

# How long to drive forward past the red line before turning (left/right signs).
CROSS_LINE_S  = 0.6

# Turn durations.
TURN_DURATION = {
    "left":  2,
    "right": 2,
    "straight": 1,
}

# ---------------------------------------------------------------------------
# Peek parameters
# ---------------------------------------------------------------------------
# Sequence: 3 frames LEFT → hold 1s → 6 frames RIGHT → hold 1s → 3 frames LEFT
PEEK_FRAMES_L1   = 3     # first left rotation
PEEK_HOLD_1_S    = 1.0   # hold after left
PEEK_FRAMES_R    = 6     # right rotation
PEEK_HOLD_2_S    = 1.0   # hold after right
PEEK_FRAMES_L2   = 4     # re-align left

# ---------------------------------------------------------------------------
# Misc constants
# ---------------------------------------------------------------------------
RED_LINE_COOLDOWN_S        = 10.0
STOP_SIGN_WAIT_S           = 10.0
VEHICLE_OBSERVE_WINDOW_S   = 1.0
APPROACH_THRESHOLD         = 0.03
# If vehicle offset doesn't change more than this across 5 consecutive frames,
# treat it as parked and ignore it.
STATIONARY_FRAME_THRESHOLD = 0.01
STATIONARY_FRAMES_IGNORE   = 5
INTERSECTION_DECIDE_TIMEOUT_S = 4.0


# ---------------------------------------------------------------------------
# Peek state machine sub-states (stored as strings in peek_phase)
# ---------------------------------------------------------------------------
PH_L1    = "L1"      # rotating left, PEEK_FRAMES_L1 frames
PH_HOLD1 = "HOLD1"   # holding, 1 second
PH_R     = "R"       # rotating right, PEEK_FRAMES_R frames
PH_HOLD2 = "HOLD2"   # holding, 1 second
PH_L2    = "L2"      # re-aligning left, PEEK_FRAMES_L2 frames
PH_DONE  = "DONE"


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
    # ── core state ──────────────────────────────────────────────────────────
    state:       BehaviorState = BehaviorState.LANE_FOLLOW
    state_until: float = 0.0
    turn_until:  float = 0.0

    active_tag_id: Optional[int] = None
    chosen_turn:   Optional[str] = None

    # ── peek sub-state ───────────────────────────────────────────────────────
    peek_phase:        str   = PH_L1
    peek_frame_count:  int   = 0
    peek_hold_until:   float = 0.0
    peek_vehicles:     list  = field(default_factory=list)  # offsets collected across all peeks

    # ── vehicle observation (after peeks finish) ─────────────────────────────
    observed_vehicle_offsets: list  = field(default_factory=list)
    observed_vehicle_until:   float = 0.0
    vehicle_stationary_frames: int  = 0   # counts frames where offset barely changes

    # ── sign read at red-line approach ───────────────────────────────────────
    approach_tag_id: Optional[int] = None   # tag seen when red line first detected
    peek_for_stop:   bool          = False  # True when peek is part of a STOP sign sequence

    # ── crossroad timing ─────────────────────────────────────────────────────
    last_crossroad_at:         float = 0.0
    intersection_decide_since: float = 0.0

    # =========================================================================
    # Main update
    # =========================================================================

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

        # ── 1. RED LINE seen while lane-following ────────────────────────────
        if red_line_seen and self.state == BehaviorState.LANE_FOLLOW:
            elapsed = now - self.last_crossroad_at
            if elapsed < RED_LINE_COOLDOWN_S:
                print(f"[RED_LINE] cooldown active ({elapsed:.1f}s / {RED_LINE_COOLDOWN_S}s)")
            else:
                tag_id = self._best_visible_tag(tags)
                self.approach_tag_id  = tag_id
                self.last_crossroad_at = now
                tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NO TAG"
                print(f"[RED_LINE] *** detected *** tag={tag_name} (id={tag_id})")

                if tag_id in STOP_TAGS:
                    # Decelerate to the line, peek both ways, THEN stop 5 seconds.
                    self.state       = BehaviorState.YIELD_WAIT
                    self.state_until = now + 0.8
                    self.peek_for_stop = True
                    print("[STOP] decelerating to red line — will peek then stop 5s")
                    return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                           self.state, "stop: decelerating to line", tag_id)

                elif tag_id in YIELD_TAGS:
                    # Decelerate and stop just before the line, then peek.
                    self.state       = BehaviorState.YIELD_WAIT
                    self.state_until = now + 0.8
                    self.peek_for_stop = False
                    print("[YIELD] decelerating to red line")
                    return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                           self.state, "yield: decelerating", tag_id)

                elif tag_id in INTERSECTION_OPTIONS:
                    # Cross the line first, then turn.
                    self.state       = BehaviorState.INTERSECTION_DECIDE
                    self.state_until = now + CROSS_LINE_S
                    self.intersection_decide_since = now
                    print(f"[TURN] crossing red line before turning ({CROSS_LINE_S}s)")
                    return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                           self.state, "crossing red line", tag_id)

                else:
                    # Unknown or no tag — just cross and go straight.
                    self.state       = BehaviorState.INTERSECTION_DECIDE
                    self.state_until = now + CROSS_LINE_S
                    self.intersection_decide_since = now
                    print("[RED_LINE] no known tag — crossing and going straight")
                    return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                           self.state, "no tag: crossing", tag_id)

        # ── 2. STOP_WAIT ─────────────────────────────────────────────────────
        if self.state == BehaviorState.STOP_WAIT:
            if now < self.state_until:
                print(f"[STOP_WAIT] {self.state_until - now:.2f}s remaining")
                return TrafficDecision(0.0, 0.0, self.state, "full stop",
                                       self.active_tag_id or self.approach_tag_id)

            # Stop time over — check for traffic.
            v = self._vehicle_offset(frame_shape, detections)
            if v is not None:
                print(f"[STOP_WAIT] vehicle present (offset={v:.3f}) — holding")
                return TrafficDecision(0.0, 0.0, self.state, "stop: vehicle present",
                                       self.active_tag_id or self.approach_tag_id)

            print("[STOP_WAIT] clear — going straight")
            # Use a longer forward drive so slow frame rates don't skip it entirely.
            self.state        = BehaviorState.TURNING
            self.chosen_turn  = "straight"
            self.turn_until   = now + 2.0   # 2 seconds forward after stop sign
            self.active_tag_id = self.approach_tag_id
            print(f"[STOP_WAIT] entering TURNING straight for 2.0s "
                  f"turn_until={self.turn_until:.2f} now={now:.2f}")
            return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                   BehaviorState.TURNING, "stop done: straight",
                                   self.active_tag_id, "straight")

        # ── 3. YIELD_WAIT (decelerate to stop just before red line) ──────────
        if self.state == BehaviorState.YIELD_WAIT:
            if now < self.state_until:
                remaining = self.state_until - now
                print(f"[YIELD_WAIT] creeping to line, {remaining:.2f}s remaining")
                return TrafficDecision(CREEP_SPEED, CREEP_SPEED,
                                       self.state, "yield: creeping",
                                       self.approach_tag_id)

            # Stopped at line — begin peek sequence.
            print("[YIELD] stopped at line — starting peek sequence")
            self._reset_peek()
            self.state = BehaviorState.CROSSROAD_PEEK_LEFT
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 4. PEEK sequence (CROSSROAD_PEEK_LEFT used as the peek state) ────
        if self.state == BehaviorState.CROSSROAD_PEEK_LEFT:
            return self._do_peek(frame_shape, tags, detections, now)

        # ── 5. CROSSROAD_WAIT_VEHICLE — vehicle seen after peeks ─────────────
        if self.state == BehaviorState.CROSSROAD_WAIT_VEHICLE:
            v = self._vehicle_offset(frame_shape, detections)

            if v is not None:
                self.observed_vehicle_offsets.append(v)

                # Check if offset has barely moved across last N frames.
                if len(self.observed_vehicle_offsets) >= 2:
                    prev = self.observed_vehicle_offsets[-2]
                    curr = self.observed_vehicle_offsets[-1]
                    if abs(curr - prev) < STATIONARY_FRAME_THRESHOLD:
                        self.vehicle_stationary_frames += 1
                    else:
                        self.vehicle_stationary_frames = 0  # reset if it moved

                print(f"[VEHICLE_OBSERVE] offset={v:.3f} "
                      f"stationary_frames={self.vehicle_stationary_frames} "
                      f"(n={len(self.observed_vehicle_offsets)})")

                # Vehicle hasn't moved for 5 frames — treat as parked, ignore it.
                if self.vehicle_stationary_frames >= STATIONARY_FRAMES_IGNORE:
                    print("[VEHICLE] stationary for "
                          f"{STATIONARY_FRAMES_IGNORE} frames — ignoring, going straight")
                    self.state        = BehaviorState.TURNING
                    self.chosen_turn  = "straight"
                    self.turn_until   = now + 2.0
                    self.active_tag_id = self.approach_tag_id
                    return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                           BehaviorState.TURNING, "parked vehicle: ignored")
            else:
                self.vehicle_stationary_frames = 0
                print("[VEHICLE_OBSERVE] no vehicle this frame")

            if now < self.observed_vehicle_until:
                return TrafficDecision(0.0, 0.0, self.state, "observing vehicle")

            if self._vehicle_is_approaching():
                print("[VEHICLE] approaching — extending wait")
                self.observed_vehicle_offsets  = []
                self.vehicle_stationary_frames = 0
                self.observed_vehicle_until    = now + VEHICLE_OBSERVE_WINDOW_S
                return TrafficDecision(0.0, 0.0, self.state, "vehicle approaching")

            print("[VEHICLE] not approaching — going straight")
            self.state        = BehaviorState.TURNING
            self.chosen_turn  = "straight"
            self.turn_until   = now + 2.0
            self.active_tag_id = self.approach_tag_id
            return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                   BehaviorState.TURNING, "vehicle not approaching: straight")

        # ── 6. INTERSECTION_DECIDE — for left/right: cross line then turn ────
        if self.state == BehaviorState.INTERSECTION_DECIDE:
            if now < self.state_until:
                # Still crossing the line.
                remaining = self.state_until - now
                print(f"[CROSSING] crossing red line, {remaining:.2f}s remaining")
                return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                                       self.state, "crossing line")

            # Crossed — now turn based on the tag we read at approach.
            tag_id   = self.approach_tag_id
            tag_name = TAG_NAMES.get(tag_id, "UNKNOWN") if tag_id else "NONE"
            direction = INTERSECTION_OPTIONS.get(tag_id, "straight")
            print(f"[INTERSECTION] crossed line, tag={tag_name} -> turning {direction}")
            self._begin_turn(direction, now, tag_id)
            l, r = (TURN_LEFT if direction == "left" else
                    TURN_RIGHT if direction == "right" else
                    (DRIVE_SPEED, DRIVE_SPEED))
            return TrafficDecision(l, r, BehaviorState.TURNING,
                                   f"turn {direction}", tag_id, direction)

        # ── 7. TURNING ────────────────────────────────────────────────────────
        if self.state == BehaviorState.TURNING:
            if now < self.turn_until:
                l, r = (TURN_LEFT  if self.chosen_turn == "left"  else
                        TURN_RIGHT if self.chosen_turn == "right" else
                        (DRIVE_SPEED, DRIVE_SPEED))
                print(f"[TURNING] '{self.chosen_turn}' {self.turn_until - now:.2f}s remaining")
                return TrafficDecision(l, r, self.state,
                                       f"turning {self.chosen_turn}",
                                       self.active_tag_id, self.chosen_turn)

            print(f"[TURNING] '{self.chosen_turn}' done -> LANE_FOLLOW")
            self._reset_state()
            return TrafficDecision(lane_left, lane_right,
                                   BehaviorState.LANE_FOLLOW, "turn done: lane follow")

        # ── 8. LANE_FOLLOW — pass through lane servoing, ignore signs ────────
        return TrafficDecision(lane_left, lane_right,
                               BehaviorState.LANE_FOLLOW, "lane follow")

    # =========================================================================
    # Peek sub-state machine
    # =========================================================================

    def _reset_peek(self):
        self.peek_phase       = PH_L1
        self.peek_frame_count = 0
        self.peek_hold_until  = 0.0
        self.peek_vehicles    = []

    def _do_peek(self, frame_shape, tags, detections, now) -> TrafficDecision:
        """
        Peek sequence:
          PH_L1    → 3 frames rotating left
          PH_HOLD1 → hold 1 second (collect vehicles, log)
          PH_R     → 6 frames rotating right
          PH_HOLD2 → hold 1 second (collect vehicles, log)
          PH_L2    → 3 frames rotating left (re-align)
          PH_DONE  → evaluate and transition
        """
        # Collect vehicle offset every frame regardless of phase.
        v = self._vehicle_offset(frame_shape, detections)
        if v is not None:
            self.peek_vehicles.append(v)

        # Log tags every frame.
        for tag in tags:
            tid = int(tag.get("id", -1))
            print(f"[PEEK/{self.peek_phase}] tag={TAG_NAMES.get(tid, tid)} "
                  f"area={tag.get('area', 0):.0f}px²")
        if v is not None:
            print(f"[PEEK/{self.peek_phase}] vehicle offset={v:.3f}")
        else:
            print(f"[PEEK/{self.peek_phase}] no vehicle")

        # ── PH_L1: rotate left for 3 frames ──────────────────────────────────
        if self.peek_phase == PH_L1:
            self.peek_frame_count += 1
            if self.peek_frame_count >= PEEK_FRAMES_L1:
                self.peek_phase       = PH_HOLD1
                self.peek_hold_until  = now + PEEK_HOLD_1_S
                self.peek_frame_count = 0
                print(f"[PEEK] L1 done -> HOLD1 ({PEEK_HOLD_1_S}s)")
            return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT, "peek left frames")

        # ── PH_HOLD1: hold for 1 second ──────────────────────────────────────
        if self.peek_phase == PH_HOLD1:
            if now < self.peek_hold_until:
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_LEFT, "peek hold 1")
            self.peek_phase       = PH_R
            self.peek_frame_count = 0
            print(f"[PEEK] HOLD1 done -> R ({PEEK_FRAMES_R} frames)")
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT, "peek right start")

        # ── PH_R: rotate right for 6 frames ──────────────────────────────────
        if self.peek_phase == PH_R:
            self.peek_frame_count += 1
            if self.peek_frame_count >= PEEK_FRAMES_R:
                self.peek_phase       = PH_HOLD2
                self.peek_hold_until  = now + PEEK_HOLD_2_S
                self.peek_frame_count = 0
                print(f"[PEEK] R done -> HOLD2 ({PEEK_HOLD_2_S}s)")
            return TrafficDecision(PEEK_R[0], PEEK_R[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT, "peek right frames")

        # ── PH_HOLD2: hold for 1 second ──────────────────────────────────────
        if self.peek_phase == PH_HOLD2:
            if now < self.peek_hold_until:
                return TrafficDecision(0.0, 0.0,
                                       BehaviorState.CROSSROAD_PEEK_LEFT, "peek hold 2")
            self.peek_phase       = PH_L2
            self.peek_frame_count = 0
            print(f"[PEEK] HOLD2 done -> L2 ({PEEK_FRAMES_L2} frames re-align)")
            return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                   BehaviorState.CROSSROAD_PEEK_LEFT, "peek re-align start")

        # ── PH_L2: re-align left for 3 frames ────────────────────────────────
        if self.peek_phase == PH_L2:
            self.peek_frame_count += 1
            if self.peek_frame_count >= PEEK_FRAMES_L2:
                self.peek_phase = PH_DONE
                print("[PEEK] L2 done -> DONE")
            else:
                return TrafficDecision(PEEK_L[0], PEEK_L[1],
                                       BehaviorState.CROSSROAD_PEEK_LEFT, "peek re-align")

        # ── PH_DONE: evaluate everything collected across all peeks ──────────
        print(f"[PEEK] complete — {len(self.peek_vehicles)} vehicle samples collected")

        if self.peek_vehicles:
            best = min(self.peek_vehicles)
            print(f"[PEEK] vehicle detected (best offset={best:.3f}) -> WAIT_VEHICLE")
            self.state                     = BehaviorState.CROSSROAD_WAIT_VEHICLE
            self.observed_vehicle_offsets  = [best]
            self.vehicle_stationary_frames = 0
            self.observed_vehicle_until    = now + VEHICLE_OBSERVE_WINDOW_S
            return TrafficDecision(0.0, 0.0, self.state, "vehicle seen: observing")

        # No vehicle seen during peeks.
        if self.peek_for_stop:
            # STOP sign sequence — now do the 5 second full stop.
            print(f"[PEEK] no vehicle, STOP sign — stopping for {STOP_SIGN_WAIT_S}s")
            self.state       = BehaviorState.STOP_WAIT
            self.state_until = now + STOP_SIGN_WAIT_S
            return TrafficDecision(0.0, 0.0, BehaviorState.STOP_WAIT,
                                   "stop sign: full stop after peek")

        print("[PEEK] no vehicle, YIELD — going straight")
        self.state        = BehaviorState.TURNING
        self.chosen_turn  = "straight"
        self.turn_until   = now + 2.0
        self.active_tag_id = self.approach_tag_id
        return TrafficDecision(DRIVE_SPEED, DRIVE_SPEED,
                               BehaviorState.TURNING, "yield peeks clear: straight")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _best_visible_tag(self, tags) -> Optional[int]:
        """Largest known tag currently visible. No area threshold."""
        best_id, best_area = None, 0.0
        for tag in tags:
            tid  = int(tag.get("id", -1))
            area = float(tag.get("area", 0.0))
            if tid not in STOP_TAGS and tid not in YIELD_TAGS and tid not in INTERSECTION_OPTIONS:
                continue
            if area > best_area:
                best_area, best_id = area, tid
        if best_id is not None:
            print(f"[TAG_SCAN] {TAG_NAMES.get(best_id, best_id)} (id={best_id}) "
                  f"area={best_area:.0f}px²")
        return best_id

    def _vehicle_offset(self, frame_shape, detections) -> Optional[float]:
        h, w = frame_shape[:2]
        best: Optional[float] = None
        for bbox, score, cls_id in detections:
            if cls_id != 1:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if ((x2 - x1) * (y2 - y1)) / max(1, h * w) < 0.004:
                continue
            offset = abs(((x1 + x2) / 2.0) / w - 0.5)
            if best is None or offset < best:
                best = offset
        return best

    def _vehicle_is_approaching(self) -> bool:
        s = self.observed_vehicle_offsets
        if len(s) < 2:
            print("[VEHICLE] not enough samples — assuming stationary")
            return False
        delta = s[0] - s[-1]
        result = delta > APPROACH_THRESHOLD
        print(f"[VEHICLE] first={s[0]:.3f} last={s[-1]:.3f} "
              f"delta={delta:+.3f} -> approaching={result}")
        return result

    def _begin_turn(self, direction: str, now: float, tag_id: Optional[int] = None):
        self.state         = BehaviorState.TURNING
        self.chosen_turn   = direction
        self.turn_until    = now + TURN_DURATION.get(direction, 0.85)
        self.active_tag_id = tag_id

    def _reset_state(self):
        print(f"[STATE_RESET] {self.state.value} -> LANE_FOLLOW")
        self.state                     = BehaviorState.LANE_FOLLOW
        self.state_until               = 0.0
        self.turn_until                = 0.0
        self.active_tag_id             = None
        self.chosen_turn               = None
        self.approach_tag_id           = None
        self.peek_for_stop             = False
        self.observed_vehicle_offsets  = []
        self.vehicle_stationary_frames = 0
        self.peek_vehicles             = []
        self.intersection_decide_since = 0.0
        self._reset_peek()
        # last_crossroad_at kept intentionally for cooldown