"""
Frenet optimal trajectory (Werling-style), adapted from PythonRobotics.

Source: Atsushi Sakai, PythonRobotics — PathPlanning/FrenetOptimalTrajectory/
frenet_optimal_trajectory.py (MIT License).

Modifications for this repo:
- No matplotlib / no script entrypoint
- Parameters passed via FrenetConfig (no module-level toggles)
- High-speed lateral + velocity-keeping longitudinal only
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cartesian_frenet_converter import CartesianFrenetConverter
from .cubic_spline_planner import CubicSpline2D
from .quintic_polynomial import QuinticPolynomial


@dataclass
class FrenetConfig:
    """Defaults aligned with PythonRobotics high-speed example."""

    max_speed_mps: float = 50.0 / 3.6
    max_accel: float = 5.0
    max_curvature: float = 1.0
    dt: float = 0.2
    max_t: float = 5.0
    min_t: float = 4.0
    n_s_sample: int = 1
    max_road_width: float = 7.0
    d_road_w: float = 1.0
    d_target_speed: float = 5.0 / 3.6
    k_j: float = 0.1
    k_t: float = 0.1
    k_d: float = 1.0
    k_s_dot: float = 1.0
    k_s: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0
    robot_radius: float = 2.0


class QuarticPolynomial:
    def __init__(self, xs: float, vxs: float, axs: float, vxe: float, axe: float, time: float):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def calc_first_derivative(self, t: float) -> float:
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def calc_second_derivative(self, t: float) -> float:
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def calc_third_derivative(self, t: float) -> float:
        return 6 * self.a3 + 24 * self.a4 * t


class FrenetPath:
    __slots__ = (
        "t",
        "d",
        "d_d",
        "d_dd",
        "d_ddd",
        "s",
        "s_d",
        "s_dd",
        "s_ddd",
        "cf",
        "x",
        "y",
        "yaw",
        "v",
        "a",
        "c",
    )

    def __init__(self) -> None:
        self.t: List[float] = []
        self.d: List[float] = []
        self.d_d: List[float] = []
        self.d_dd: List[float] = []
        self.d_ddd: List[float] = []
        self.s: List[float] = []
        self.s_d: List[float] = []
        self.s_dd: List[float] = []
        self.s_ddd: List[float] = []
        self.cf: float = 0.0
        self.x: List[float] = []
        self.y: List[float] = []
        self.yaw: List[float] = []
        self.v: List[float] = []
        self.a: List[float] = []
        self.c: List[float] = []

    def pop_front(self) -> None:
        for name in (
            "x",
            "y",
            "yaw",
            "v",
            "a",
            "s",
            "s_d",
            "s_dd",
            "s_ddd",
            "d",
            "d_d",
            "d_dd",
            "d_ddd",
        ):
            getattr(self, name).pop(0)


class _HighSpeedLateral:
    def calc_lateral_trajectory(
        self, fp: FrenetPath, di: float, c_d: float, c_d_d: float, c_d_dd: float, Ti: float
    ) -> FrenetPath:
        tp = copy.deepcopy(fp)
        s0_d = fp.s_d[0]
        s0_dd = fp.s_dd[0]
        lat_qp = QuinticPolynomial(
            c_d, c_d_d * s0_d, c_d_dd * s0_d**2 + c_d_d * s0_dd, di, 0.0, 0.0, Ti
        )
        tp.d = []
        tp.d_d = []
        tp.d_dd = []
        tp.d_ddd = []
        for i in range(len(fp.t)):
            t = fp.t[i]
            s_d = fp.s_d[i]
            s_dd = fp.s_dd[i]
            s_d_inv = 1.0 / (s_d + 1e-6) + 1e-6
            s_d_inv_sq = s_d_inv * s_d_inv
            d = lat_qp.calc_point(t)
            d_d = lat_qp.calc_first_derivative(t)
            d_dd = lat_qp.calc_second_derivative(t)
            d_ddd = lat_qp.calc_third_derivative(t)
            tp.d.append(d)
            tp.d_d.append(d_d * s_d_inv)
            tp.d_dd.append((d_dd - tp.d_d[i] * s_dd) * s_d_inv_sq)
            tp.d_ddd.append(d_ddd)
        return tp

    def calc_cartesian_parameters(self, fp: FrenetPath, csp: CubicSpline2D) -> FrenetPath:
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            i_kappa = csp.calc_curvature(fp.s[i])
            i_dkappa = csp.calc_curvature_rate(fp.s[i])
            s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
            d_condition = [fp.d[i], fp.d_d[i], fp.d_dd[i]]
            x, y, theta, kappa, v, a = CartesianFrenetConverter.frenet_to_cartesian(
                fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa, s_condition, d_condition
            )
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(theta)
            fp.c.append(kappa)
            fp.v.append(v)
            fp.a.append(a)
        return fp


class _VelocityKeepingLongitudinal:
    def __init__(self, cfg: FrenetConfig, target_speed_mps: float) -> None:
        self.cfg = cfg
        self.target_speed_mps = float(target_speed_mps)

    def calc_longitudinal_trajectory(
        self, c_speed: float, c_accel: float, Ti: float, s0: float
    ) -> List[FrenetPath]:
        fplist: List[FrenetPath] = []
        d_ts = self.cfg.d_target_speed
        n = self.cfg.n_s_sample
        lo = self.target_speed_mps - d_ts * n
        hi = self.target_speed_mps + d_ts * n
        if hi < lo:
            lo, hi = hi, lo
        step = max(float(d_ts), 0.1)
        speeds = list(np.arange(lo, hi + 0.5 * step, step))
        if not speeds:
            speeds = [float(self.target_speed_mps)]
        for tv in speeds:
            fp = FrenetPath()
            lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, self.cfg.dt)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def get_d_arrange(self, s0: float) -> np.ndarray:
        return np.arange(-self.cfg.max_road_width, self.cfg.max_road_width, self.cfg.d_road_w)

    def calc_destination_cost(self, fp: FrenetPath) -> float:
        ds = (self.target_speed_mps - fp.s_d[-1]) ** 2
        return self.cfg.k_s_dot * ds


def _calc_frenet_paths(
    cfg: FrenetConfig,
    lon: _VelocityKeepingLongitudinal,
    c_s_d: float,
    c_s_dd: float,
    c_d: float,
    c_d_d: float,
    c_d_dd: float,
    s0: float,
) -> List[FrenetPath]:
    lateral = _HighSpeedLateral()
    frenet_paths: List[FrenetPath] = []
    for Ti in np.arange(cfg.min_t, cfg.max_t, cfg.dt):
        lon_paths = lon.calc_longitudinal_trajectory(c_s_d, c_s_dd, Ti, s0)
        for fp in lon_paths:
            for di in lon.get_d_arrange(s0):
                tp = lateral.calc_lateral_trajectory(fp, di, c_d, c_d_d, c_d_dd, Ti)
                Jp = sum(np.power(tp.d_ddd, 2))
                Js = sum(np.power(tp.s_ddd, 2))
                lat_cost = cfg.k_j * Jp + cfg.k_t * Ti + cfg.k_d * tp.d[-1] ** 2
                lon_cost = cfg.k_j * Js + cfg.k_t * Ti + lon.calc_destination_cost(tp)
                tp.cf = cfg.k_lat * lat_cost + cfg.k_lon * lon_cost
                frenet_paths.append(tp)
    return frenet_paths


def _calc_global_paths(fplist: List[FrenetPath], csp: CubicSpline2D) -> List[FrenetPath]:
    lateral = _HighSpeedLateral()
    return [lateral.calc_cartesian_parameters(fp, csp) for fp in fplist]


def _check_collision(fp: FrenetPath, ob: np.ndarray, robot_radius: float) -> bool:
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for (ix, iy) in zip(fp.x, fp.y)]
        if any([di <= robot_radius**2 for di in d]):
            return False
    return True


def check_paths(
    fplist: List[FrenetPath], ob: np.ndarray, cfg: FrenetConfig
) -> Dict[str, Any]:
    path_dict: Dict[str, Any] = {
        "max_speed_error": [],
        "max_accel_error": [],
        "max_curvature_error": [],
        "collision_error": [],
        "ok": [],
    }
    for i, _ in enumerate(fplist):
        if any([v > cfg.max_speed_mps for v in fplist[i].v]):
            path_dict["max_speed_error"].append(fplist[i])
        elif any([abs(a) > cfg.max_accel for a in fplist[i].a]):
            path_dict["max_accel_error"].append(fplist[i])
        elif any([abs(c) > cfg.max_curvature for c in fplist[i].c]):
            path_dict["max_curvature_error"].append(fplist[i])
        elif not _check_collision(fplist[i], ob, cfg.robot_radius):
            path_dict["collision_error"].append(fplist[i])
        else:
            path_dict["ok"].append(fplist[i])
    return path_dict


def frenet_optimal_planning(
    csp: CubicSpline2D,
    s0: float,
    c_s_d: float,
    c_s_dd: float,
    c_d: float,
    c_d_d: float,
    c_d_dd: float,
    obstacles_xy: np.ndarray,
    cfg: FrenetConfig,
    target_speed_mps: float,
) -> Tuple[Optional[FrenetPath], Dict[str, Any]]:
    lon = _VelocityKeepingLongitudinal(cfg, target_speed_mps=target_speed_mps)
    fplist = _calc_frenet_paths(cfg, lon, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0)
    fplist = _calc_global_paths(fplist, csp)
    if obstacles_xy.size == 0:
        ob = np.zeros((0, 2))
    else:
        ob = obstacles_xy
    fpdict = check_paths(fplist, ob, cfg)
    best: Optional[FrenetPath] = None
    min_cost = float("inf")
    for fp in fpdict["ok"]:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best = fp
    return best, fpdict


def nearest_arclength(csp: CubicSpline2D, x: float, y: float, ds: float = 0.25) -> float:
    """Project (x,y) onto the spline by grid search on s (fast enough for sim)."""
    s_end = float(csp.s[-1])
    best_s = 0.0
    best_d2 = float("inf")
    s = 0.0
    while s <= s_end:
        ix, iy = csp.calc_position(s)
        if ix is not None and iy is not None:
            d2 = (ix - x) ** 2 + (iy - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = s
        s += ds
    return best_s


def make_cubic_spline(wx: List[float], wy: List[float]) -> CubicSpline2D:
    return CubicSpline2D(wx, wy)
