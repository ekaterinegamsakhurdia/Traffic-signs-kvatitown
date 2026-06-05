from enum import Enum


class BehaviorState(str, Enum):
    # ── Normal driving ────────────────────────────────────────────────────────
    LANE_FOLLOW   = "LANE_FOLLOW"
    OBSTACLE_STOP = "OBSTACLE_STOP"

    # ── Crossroads sequence (in order of execution) ───────────────────────────
    CROSSROAD_STOP         = "CROSSROAD_STOP"          # brief halt at the red line
    CROSSROAD_PEEK_LEFT    = "CROSSROAD_PEEK_LEFT"     # rotating left to scan
    CROSSROAD_PEEK_RIGHT   = "CROSSROAD_PEEK_RIGHT"    # rotating right to scan
    CROSSROAD_WAIT_VEHICLE = "CROSSROAD_WAIT_VEHICLE"  # observing & yielding to incoming vehicle
    STOP_WAIT              = "STOP_WAIT"               # mandatory 5-second stop (stop sign)
    YIELD_WAIT             = "YIELD_WAIT"              # slow creep-through (yield sign)
    CROSSROAD_PRE_TURN     = "CROSSROAD_PRE_TURN"      # cross the line + pre-turn straight drive
    CROSSROAD_TURNING      = "CROSSROAD_TURNING"       # execute rotation + post-turn straight drive