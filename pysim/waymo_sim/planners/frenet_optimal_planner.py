"""
Planner adapter: PythonRobotics-style Frenet optimal trajectory on the route centerline.

Uses vendored Werling Frenet sampling (`python_robotics_vendor.frenet_werling_core`).
Obstacles are NPC centers with a circular envelope (`robot_radius`).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..vehicle import VehicleState
from .base import NPCSnapshot, PlanCommand, Planner
from .python_robotics_vendor.cartesian_frenet_converter import CartesianFrenetConverter
from .python_robotics_vendor.frenet_werling_core import (
    FrenetConfig,
    FrenetPath,
    frenet_optimal_planning,
    make_cubic_spline,
    nearest_arclength,
)


def _vehicle_robot_radius(length: float, width: float) -> float:
    """Radius for Frenet vs. point obstacles (NPC centers).

    PythonRobotics treats obstacles as points; a huge circle makes every
    candidate trajectory fail in dense traffic. We use ~body half-width with a
    floor/ceiling so sampling stays feasible in this sim.
    """

    hw = 0.5 * max(width, 0.1)
    return float(min(max(hw * 1.4, 0.8), 2.0))


@dataclass
class FrenetPlannerParams:
    """Tuning for Frenet sampling (see FrenetConfig)."""

    frenet_dt: float = 0.2
    min_t: float = 4.0
    max_t: float = 5.0
    max_road_width: float = 7.0


class FrenetOptimalPlanner(Planner):
    """Sample-based Frenet optimal trajectory along `reference_path` (x,y,z polyline)."""

    name = "frenet_optimal"

    def __init__(
        self,
        reference_path: List[Tuple[float, float, float]],
        speed_limit_mps: float = 13.9,
        wheelbase: float = 2.7,
        ego_length: float = 4.5,
        ego_width: float = 1.85,
        desired_speed_mps: Optional[float] = None,
        params: Optional[FrenetPlannerParams] = None,
    ):
        if len(reference_path) < 2:
            raise ValueError("reference_path must have at least 2 points")
        self._path = list(reference_path)
        self._wheelbase = float(wheelbase)
        self._speed_limit = float(speed_limit_mps)
        self._desired = float(desired_speed_mps if desired_speed_mps is not None else speed_limit_mps)
        self._desired = min(self._desired, self._speed_limit)
        pp = params or FrenetPlannerParams()

        wx = [p[0] for p in self._path]
        wy = [p[1] for p in self._path]
        self._csp = make_cubic_spline(wx, wy)
        rr = _vehicle_robot_radius(ego_length, ego_width)

        self._cfg = FrenetConfig(
            dt=float(pp.frenet_dt),
            min_t=float(pp.min_t),
            max_t=float(pp.max_t),
            max_road_width=float(pp.max_road_width),
            max_speed_mps=max(self._speed_limit, self._desired),
            robot_radius=max(rr, 0.5),
        )
        self._frenet_initialized = False
        self._last_path: Optional[FrenetPath] = None

    def _update_frenet_state(self, path: FrenetPath) -> None:
        if len(path.s) < 2:
            return
        self._s0 = path.s[1]
        self._c_d = path.d[1]
        self._c_d_d = path.d_d[1]
        self._c_d_dd = path.d_dd[1]
        self._c_s_d = path.s_d[1]
        self._c_s_dd = path.s_dd[1]

    def _init_frenet_from_ego(self, ego: VehicleState) -> None:
        s0 = nearest_arclength(self._csp, ego.x, ego.y)
        ix, iy = self._csp.calc_position(s0)
        if ix is None or iy is None:
            ix, iy = ego.x, ego.y
        ryaw = self._csp.calc_yaw(s0)
        rk = self._csp.calc_curvature(s0)
        rdk = self._csp.calc_curvature_rate(s0)
        kappa_ego = math.tan(ego.steer) / max(self._wheelbase, 0.5)
        s_cond, d_cond = CartesianFrenetConverter.cartesian_to_frenet(
            s0,
            ix,
            iy,
            ryaw,
            rk,
            rdk,
            ego.x,
            ego.y,
            ego.speed,
            ego.acceleration,
            ego.heading,
            kappa_ego,
        )
        self._s0 = float(s_cond[0])
        self._c_s_d = float(s_cond[1])
        self._c_s_dd = float(s_cond[2])
        self._c_d = float(d_cond[0])
        self._c_d_d = float(d_cond[1])
        self._c_d_dd = float(d_cond[2])

    def plan(
        self,
        ego_state: VehicleState,
        npcs: List[NPCSnapshot],
        t: float,
        dt: float,
    ) -> PlanCommand:
        if not self._frenet_initialized:
            self._init_frenet_from_ego(ego_state)
            self._frenet_initialized = True

        if npcs:
            ob = np.array([[n.x, n.y] for n in npcs], dtype=float)
        else:
            ob = np.zeros((0, 2), dtype=float)

        best, _ = frenet_optimal_planning(
            self._csp,
            self._s0,
            self._c_s_d,
            self._c_s_dd,
            self._c_d,
            self._c_d_d,
            self._c_d_dd,
            ob,
            self._cfg,
            self._desired,
        )

        path = best
        if path is None:
            if self._last_path is not None:
                path = copy.deepcopy(self._last_path)
                path.pop_front()
            else:
                return PlanCommand(
                    target_acceleration=-1.0,
                    steering_angle=0.0,
                    desired_speed_mps=self._desired,
                )

        if not path.x or len(path.x) <= 1:
            return PlanCommand(
                target_acceleration=-1.0,
                steering_angle=0.0,
                desired_speed_mps=self._desired,
            )

        self._last_path = path
        self._update_frenet_state(path)

        accel = float(path.a[1]) if len(path.a) > 1 else float(path.a[0])
        if len(path.c) > 1:
            steer = math.atan(self._wheelbase * float(path.c[1]))
        else:
            steer = 0.0

        max_steer = math.radians(35.0)
        steer = max(-max_steer, min(max_steer, steer))
        accel = max(-6.0, min(2.5, accel))

        return PlanCommand(
            target_acceleration=accel,
            steering_angle=steer,
            desired_speed_mps=self._desired,
        )
