"""IDM longitudinal control + Pure Pursuit lateral control.

Two textbook open-source algorithms composed into a usable lane-following AV
baseline. References:
  - Treiber, Hennecke, Helbing 2000  (IDM car-following)
  - Coulter 1992                      (Pure-pursuit path tracking)

Inputs:
  - reference_path : densely sampled (x, y, z) polyline along the SDC route
  - speed_limit_mps: legal speed limit on the route (used as IDM v0)

Behaviour:
  - Nearest in-corridor NPC (lateral offset under `leader_lateral_tol`) becomes
    the IDM "leader". Pure pursuit picks a lookahead point ahead of the ego
    along the path and outputs a steering angle. The combined command is fed
    to the bicycle model.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..vehicle import VehicleState
from .base import NPCSnapshot, PlanCommand, Planner


@dataclass
class IDMParams:
    desired_speed: float = 13.9      # m/s — clipped to lane speed limit
    time_headway: float = 1.5
    min_gap: float = 2.5
    max_accel: float = 1.5
    comfort_decel: float = 2.0
    delta: float = 4.0
    detection_radius: float = 80.0
    leader_lateral_tol: float = 1.5  # meters off centerline to count as leader


@dataclass
class PurePursuitParams:
    k_lookahead: float = 0.6
    min_lookahead: float = 4.0
    max_lookahead: float = 25.0
    wheelbase: float = 2.7


class IDMPurePursuitPlanner(Planner):
    name = "idm_pure_pursuit"

    def __init__(
        self,
        reference_path: List[Tuple[float, float, float]],
        speed_limit_mps: float = 13.9,
        idm: Optional[IDMParams] = None,
        pp: Optional[PurePursuitParams] = None,
    ):
        if len(reference_path) < 2:
            raise ValueError("reference_path must have at least 2 points")
        self.path: List[Tuple[float, float, float]] = list(reference_path)
        self.idm = idm or IDMParams()
        self.idm.desired_speed = min(self.idm.desired_speed, speed_limit_mps)
        self.pp = pp or PurePursuitParams()

        # Cumulative arc length along the path
        self._arc: List[float] = [0.0]
        for i in range(len(self.path) - 1):
            x1, y1, _ = self.path[i]
            x2, y2, _ = self.path[i + 1]
            self._arc.append(self._arc[-1] + math.hypot(x2 - x1, y2 - y1))
        self._total_arc = self._arc[-1]

    def plan(
        self,
        ego_state: VehicleState,
        npcs: List[NPCSnapshot],
        t: float,
        dt: float,
    ) -> PlanCommand:
        seg, t_seg, _, _ = self._project(ego_state.x, ego_state.y)
        ego_arc = self._arc_at(seg, t_seg)

        # ----- Pure Pursuit lateral -----
        ld = max(
            self.pp.min_lookahead,
            min(self.pp.max_lookahead, self.pp.k_lookahead * ego_state.speed + self.pp.min_lookahead),
        )
        lx, ly = self._lookahead(ego_arc, ld)
        dx = lx - ego_state.x
        dy = ly - ego_state.y
        c = math.cos(ego_state.heading)
        s = math.sin(ego_state.heading)
        fx = c * dx + s * dy
        fy = -s * dx + c * dy
        L = math.hypot(fx, fy)
        if L < 1e-6:
            steer = 0.0
        else:
            alpha = math.atan2(fy, fx)
            steer = math.atan2(2.0 * self.pp.wheelbase * math.sin(alpha), L)

        # ----- IDM longitudinal -----
        leader = self._find_leader(ego_state, npcs, ego_arc)
        if leader is None:
            leader_id, gap, leader_v = None, float("inf"), self.idm.desired_speed
        else:
            leader_id, gap, leader_v = leader
        accel = self._idm(ego_state.speed, gap, leader_v)

        # End-of-route brake: smoothly stop at goal
        remaining = self._total_arc - ego_arc
        if remaining < max(2.0, ego_state.speed * 1.5):
            stop_decel = self._stop_decel(ego_state.speed, max(0.5, remaining))
            accel = min(accel, -stop_decel)

        return PlanCommand(
            target_acceleration=accel,
            steering_angle=steer,
            desired_speed_mps=self.idm.desired_speed,
            leader_id=leader_id,
            leader_gap=gap,
            debug={"ego_arc": ego_arc, "lookahead": (lx, ly), "remaining": remaining},
        )

    # ----- helpers -----

    def _project(self, x: float, y: float) -> Tuple[int, float, float, float]:
        best = (0, 0.0, x, y, float("inf"))
        for i in range(len(self.path) - 1):
            x1, y1, _ = self.path[i]
            x2, y2, _ = self.path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
            px = x1 + t * dx
            py = y1 + t * dy
            d = math.hypot(x - px, y - py)
            if d < best[4]:
                best = (i, t, px, py, d)
        return best[0], best[1], best[2], best[3]

    def _arc_at(self, seg: int, t: float) -> float:
        if seg + 1 >= len(self._arc):
            return self._total_arc
        return self._arc[seg] + t * (self._arc[seg + 1] - self._arc[seg])

    def _lookahead(self, ego_arc: float, ld: float) -> Tuple[float, float]:
        target = ego_arc + ld
        if target >= self._total_arc:
            return self.path[-1][0], self.path[-1][1]
        idx = bisect.bisect_left(self._arc, target) - 1
        idx = max(0, min(idx, len(self.path) - 2))
        seg_len = self._arc[idx + 1] - self._arc[idx]
        if seg_len < 1e-9:
            return self.path[idx][0], self.path[idx][1]
        t = (target - self._arc[idx]) / seg_len
        x1, y1, _ = self.path[idx]
        x2, y2, _ = self.path[idx + 1]
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    def _find_leader(
        self, ego: VehicleState, npcs: List[NPCSnapshot], ego_arc: float
    ) -> Optional[Tuple[int, float, float]]:
        best_id, best_gap, best_v = None, float("inf"), 0.0
        for n in npcs:
            dx = n.x - ego.x
            dy = n.y - ego.y
            if dx * dx + dy * dy > self.idm.detection_radius ** 2:
                continue
            seg, t_seg, px, py = self._project(n.x, n.y)
            lat = math.hypot(n.x - px, n.y - py)
            if lat > self.idm.leader_lateral_tol + 0.5 * n.width:
                continue
            n_arc = self._arc_at(seg, t_seg)
            # subtract approximate half-lengths so the gap is bumper-to-bumper
            gap = n_arc - ego_arc - 2.25 - 0.5 * n.length
            if gap <= 0.05:
                continue
            # NPC longitudinal speed projected on local path tangent
            seg_idx = max(0, min(seg, len(self.path) - 2))
            x1, y1, _ = self.path[seg_idx]
            x2, y2, _ = self.path[seg_idx + 1]
            seg_h = math.atan2(y2 - y1, x2 - x1)
            v_along = n.vx * math.cos(seg_h) + n.vy * math.sin(seg_h)
            if gap < best_gap:
                best_id, best_gap, best_v = n.id, gap, v_along
        if best_id is None:
            return None
        return best_id, best_gap, best_v

    def _idm(self, v: float, gap: float, leader_v: float) -> float:
        v0 = max(0.1, self.idm.desired_speed)
        T = self.idm.time_headway
        s0 = self.idm.min_gap
        a = self.idm.max_accel
        b = self.idm.comfort_decel
        delta = self.idm.delta
        free = 1.0 - (v / v0) ** delta
        if not math.isfinite(gap) or gap > 1e6:
            return a * free
        s_star = s0 + max(0.0, v * T + v * (v - leader_v) / (2.0 * math.sqrt(a * b)))
        interaction = (s_star / max(gap, 1e-3)) ** 2
        return a * (free - interaction)

    @staticmethod
    def _stop_decel(v: float, dist: float) -> float:
        # required constant deceleration to stop within `dist`
        if dist <= 0.01:
            return 6.0
        return min(6.0, max(0.0, v * v / (2.0 * dist)))
