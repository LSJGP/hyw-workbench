"""Closed-loop simulation world.

Owns the ego vehicle + scenario, ticks at fixed dt, replays Waymo NPC tracks at
their native rate, and detects regulatory-aware collisions each frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import OBB, obb_overlap
from .planners.base import NPCSnapshot, PlanCommand, Planner
from .scenario import Scenario, Track
from .vehicle import BicycleVehicle, VehicleParams, VehicleState


# Regulatory exemption thresholds
EXEMPT_REAR_END_EGO_SPEED_MAX = 5.0   # m/s — ego is "slow" below this
EXEMPT_REAR_END_REL_SPEED_MIN = 2.0   # m/s — NPC must close at least this fast
EXEMPT_CUT_IN_LAT_VEL_MIN = 0.5       # m/s — NPC lateral velocity into ego
EXEMPT_HEAD_ON_ANGLE_DEG = 135.0      # >= this -> head-on


@dataclass
class CollisionInfo:
    collided: bool = False
    other_id: int = 0
    kind: str = ""
    ego_at_fault: bool = True
    exempt: bool = False
    exempt_reason: str = ""
    relative_speed_mps: float = 0.0
    ego_speed_mps: float = 0.0
    approach_angle_deg: float = 0.0


@dataclass
class FrameRecord:
    frame_id: int
    timestamp_us: int
    ego: VehicleState
    command: PlanCommand
    collision: CollisionInfo
    num_npcs: int


@dataclass
class WorldConfig:
    dt: float = 0.1                  # control / sim timestep in seconds
    max_seconds: float = 0.0         # 0 -> use scenario duration
    stop_on_first_collision: bool = False
    interpolate_npcs: bool = True
    vehicle_params: VehicleParams = field(default_factory=VehicleParams)


class World:
    def __init__(self, scenario: Scenario, ego: BicycleVehicle, config: Optional[WorldConfig] = None):
        self.scenario = scenario
        self.ego = ego
        self.config = config or WorldConfig()
        self._scenario_dt = _scenario_dt(scenario)

    def run(self, planner: Planner, hooks: Optional[List["FrameHook"]] = None) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        cfg = self.config
        hooks = hooks or []
        if cfg.max_seconds > 0:
            total_seconds = cfg.max_seconds
        else:
            total_seconds = (
                self.scenario.timestamps_seconds[-1] - self.scenario.timestamps_seconds[0]
                if len(self.scenario.timestamps_seconds) >= 2
                else 5.0
            )
        n_steps = max(1, int(round(total_seconds / cfg.dt)))

        t0 = self.scenario.timestamps_seconds[0] if self.scenario.timestamps_seconds else 0.0

        for step in range(n_steps):
            t = step * cfg.dt
            scenario_time = t0 + t
            npcs = self._npcs_at_time(scenario_time)
            cmd = planner.plan(self.ego.state, npcs, t, cfg.dt)

            self.ego.step(cmd.target_acceleration, cmd.steering_angle, cfg.dt)

            collision = self._detect_collision(npcs)

            rec = FrameRecord(
                frame_id=step,
                timestamp_us=int(round(scenario_time * 1e6)),
                ego=VehicleState(**self.ego.state.__dict__),
                command=cmd,
                collision=collision,
                num_npcs=len(npcs),
            )
            records.append(rec)
            for h in hooks:
                h.on_frame(rec)

            if collision.collided and cfg.stop_on_first_collision:
                break

        for h in hooks:
            h.on_finish(records)
        return records

    # ----- NPC sampling -----

    def _npcs_at_time(self, scenario_time: float) -> List[NPCSnapshot]:
        ts = self.scenario.timestamps_seconds
        if not ts:
            return []
        if scenario_time <= ts[0]:
            return self._npcs_at_index(0)
        if scenario_time >= ts[-1]:
            return self._npcs_at_index(len(ts) - 1)
        # find bracketing indices
        lo, hi = 0, len(ts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if ts[mid] <= scenario_time:
                lo = mid
            else:
                hi = mid
        if not self.config.interpolate_npcs or hi == lo:
            return self._npcs_at_index(lo)
        seg_len = ts[hi] - ts[lo]
        a = 0.0 if seg_len < 1e-9 else (scenario_time - ts[lo]) / seg_len
        return self._interp_npcs(lo, hi, a)

    def _npcs_at_index(self, idx: int) -> List[NPCSnapshot]:
        out: List[NPCSnapshot] = []
        for tr in self.scenario.tracks:
            if tr.is_sdc:
                continue
            if idx >= len(tr.states):
                continue
            st = tr.states[idx]
            if not st.valid:
                continue
            out.append(_track_to_snapshot(tr, st))
        return out

    def _interp_npcs(self, lo: int, hi: int, a: float) -> List[NPCSnapshot]:
        out: List[NPCSnapshot] = []
        for tr in self.scenario.tracks:
            if tr.is_sdc:
                continue
            if lo >= len(tr.states) or hi >= len(tr.states):
                continue
            s0, s1 = tr.states[lo], tr.states[hi]
            if s0.valid and s1.valid:
                yaw0, yaw1 = s0.yaw, s1.yaw
                dy = ((yaw1 - yaw0 + math.pi) % (2 * math.pi)) - math.pi
                yaw = yaw0 + a * dy
                out.append(
                    NPCSnapshot(
                        id=tr.id,
                        object_type=tr.object_type,
                        x=s0.x + a * (s1.x - s0.x),
                        y=s0.y + a * (s1.y - s0.y),
                        z=s0.z + a * (s1.z - s0.z),
                        heading=yaw,
                        vx=s0.vx + a * (s1.vx - s0.vx),
                        vy=s0.vy + a * (s1.vy - s0.vy),
                        length=s1.length,
                        width=s1.width,
                        height=s1.height,
                    )
                )
            elif s0.valid:
                out.append(_track_to_snapshot(tr, s0))
            elif s1.valid:
                out.append(_track_to_snapshot(tr, s1))
        return out

    # ----- Collision detection -----

    def _detect_collision(self, npcs: List[NPCSnapshot]) -> CollisionInfo:
        ego_box = self.ego.bbox()
        worst: Optional[CollisionInfo] = None
        for n in npcs:
            half_l = max(0.5, 0.5 * n.length)
            half_w = max(0.3, 0.5 * n.width)
            n_box = OBB(cx=n.x, cy=n.y, heading=n.heading, half_length=half_l, half_width=half_w)
            if not obb_overlap(ego_box, n_box):
                continue
            info = _classify_collision(self.ego.state, n)
            if worst is None:
                worst = info
            else:
                # priority: non-exempt > exempt; tie-break by relative speed
                if (not info.exempt and worst.exempt) or (
                    info.exempt == worst.exempt and info.relative_speed_mps > worst.relative_speed_mps
                ):
                    worst = info
        return worst if worst is not None else CollisionInfo(collided=False)


# ----- Internal helpers -----

class FrameHook:
    """Optional callback for streaming output."""

    def on_frame(self, rec: FrameRecord) -> None:  # noqa: D401
        pass

    def on_finish(self, records: List[FrameRecord]) -> None:  # noqa: D401
        pass


def _track_to_snapshot(tr: Track, st) -> NPCSnapshot:
    return NPCSnapshot(
        id=tr.id,
        object_type=tr.object_type,
        x=st.x,
        y=st.y,
        z=st.z,
        heading=st.yaw,
        vx=st.vx,
        vy=st.vy,
        length=st.length,
        width=st.width,
        height=st.height,
    )


def _scenario_dt(scenario: Scenario) -> float:
    ts = scenario.timestamps_seconds
    if len(ts) < 2:
        return 0.1
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
    return float(sum(diffs) / len(diffs)) if diffs else 0.1


def _classify_collision(ego: VehicleState, n: NPCSnapshot) -> CollisionInfo:
    """Determine kind + regulatory exemption of an OBB-overlap event."""
    dx = n.x - ego.x
    dy = n.y - ego.y
    c = math.cos(-ego.heading)
    s = math.sin(-ego.heading)
    fx = c * dx - s * dy
    fy = s * dx + c * dy
    bearing = math.atan2(fy, fx)  # 0 = front, +/-pi = rear, +pi/2 = left

    ego_speed = ego.speed
    npc_speed = math.hypot(n.vx, n.vy)
    rvx = n.vx - ego.speed * math.cos(ego.heading)
    rvy = n.vy - ego.speed * math.sin(ego.heading)
    rel_speed = math.hypot(rvx, rvy)

    if npc_speed > 0.5:
        ndx = n.vx / npc_speed
        ndy = n.vy / npc_speed
        edx = math.cos(ego.heading)
        edy = math.sin(ego.heading)
        cosang = max(-1.0, min(1.0, edx * ndx + edy * ndy))
        approach_angle = math.degrees(math.acos(cosang))
    else:
        approach_angle = 0.0

    abs_b = abs(bearing)
    if abs_b < math.pi / 4:
        zone = "front"
        kind = "ego_front_into_npc"
    elif abs_b > 3 * math.pi / 4:
        zone = "rear"
        kind = "npc_rear_into_ego"
    else:
        zone = "side"
        kind = "side_collision"

    exempt = False
    exempt_reason = ""
    ego_at_fault = True

    if zone == "rear":
        if (
            ego_speed < EXEMPT_REAR_END_EGO_SPEED_MAX
            and (npc_speed - ego_speed) > EXEMPT_REAR_END_REL_SPEED_MIN
        ):
            exempt = True
            ego_at_fault = False
            exempt_reason = "rear_end_on_slow_ego"
    elif zone == "side":
        # NPC lateral velocity in ego frame
        sx = -math.sin(ego.heading)
        sy = math.cos(ego.heading)
        lat_v_npc = sx * n.vx + sy * n.vy
        if (bearing > 0 and lat_v_npc < -EXEMPT_CUT_IN_LAT_VEL_MIN) or (
            bearing < 0 and lat_v_npc > EXEMPT_CUT_IN_LAT_VEL_MIN
        ):
            exempt = True
            ego_at_fault = False
            exempt_reason = "forced_cut_in"
    else:  # front
        if approach_angle > EXEMPT_HEAD_ON_ANGLE_DEG:
            exempt = True
            ego_at_fault = False
            exempt_reason = "wrong_way_head_on"

    return CollisionInfo(
        collided=True,
        other_id=n.id,
        kind=kind,
        ego_at_fault=ego_at_fault,
        exempt=exempt,
        exempt_reason=exempt_reason,
        relative_speed_mps=rel_speed,
        ego_speed_mps=ego_speed,
        approach_angle_deg=approach_angle,
    )
