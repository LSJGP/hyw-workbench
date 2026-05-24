"""Universal "trajectory -> PlanCommand" adapter.

Most open-source planners output a *trajectory* (sequence of waypoints over a
planning horizon). Our bicycle vehicle, however, consumes a single tick's
`(target_acceleration, steering_angle)`. This module bridges the two:

  - lateral  : pure-pursuit toward a waypoint at a velocity-scaled lookahead
  - longitud.: a = (v_ref - v_ego) / control_horizon  (with bounds left to the
               vehicle's max_accel / max_decel saturation)

So a new planner integration only needs to:
  1. produce a `List[Waypoint]` per tick (any open-source planner does this);
  2. call `trajectory_to_command(ego, traj, ...)` to convert.

If the planner instead emits a (accel, steer) pair natively, just build a
`PlanCommand` directly and skip this helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from ..vehicle import VehicleState
from .base import PlanCommand


@dataclass
class Waypoint:
    """One sample on a planner's output trajectory.

    `t` is seconds from "now" — used to pick which waypoint's speed becomes
    the reference for IDM-style longitudinal tracking.
    """

    x: float
    y: float
    yaw: float = 0.0
    speed: float = 0.0
    t: float = 0.0


def trajectory_to_command(
    ego: VehicleState,
    trajectory: List[Waypoint],
    desired_speed_mps: float,
    wheelbase: float = 2.7,
    control_horizon: float = 0.5,
    lookahead_min: float = 4.0,
    lookahead_k: float = 0.6,
    lookahead_max: float = 25.0,
    fallback_decel: float = 2.0,
) -> PlanCommand:
    """Convert a waypoint trajectory into a single-tick `PlanCommand`."""
    if not trajectory:
        return PlanCommand(
            target_acceleration=-abs(fallback_decel),
            steering_angle=0.0,
            desired_speed_mps=desired_speed_mps,
        )

    # ---- Lateral: pure pursuit ----
    ld = max(lookahead_min, min(lookahead_max, ego.speed * lookahead_k + lookahead_min))
    target = trajectory[-1]
    accum = 0.0
    px, py = ego.x, ego.y
    for wp in trajectory:
        d = math.hypot(wp.x - px, wp.y - py)
        accum += d
        px, py = wp.x, wp.y
        if accum >= ld:
            target = wp
            break

    dx = target.x - ego.x
    dy = target.y - ego.y
    c = math.cos(ego.heading)
    s = math.sin(ego.heading)
    fx = c * dx + s * dy
    fy = -s * dx + c * dy
    L = math.hypot(fx, fy)
    if L < 1e-6:
        steer = 0.0
    else:
        alpha = math.atan2(fy, fx)
        steer = math.atan2(2.0 * wheelbase * math.sin(alpha), L)

    # ---- Longitudinal: track reference speed at `control_horizon` ahead ----
    ref_speed = trajectory[0].speed
    for wp in trajectory:
        if wp.t >= control_horizon:
            ref_speed = wp.speed
            break
    accel = (ref_speed - ego.speed) / max(0.1, control_horizon)

    return PlanCommand(
        target_acceleration=accel,
        steering_angle=steer,
        desired_speed_mps=desired_speed_mps,
        debug={"target_wp": (target.x, target.y), "ref_speed": ref_speed},
    )
