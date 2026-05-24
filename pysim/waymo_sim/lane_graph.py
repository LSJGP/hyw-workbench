"""Lane graph + routing for the Python sim.

The graph file is dumped by tools/waymo_to_lanelet2.py and contains the lane
type, speed limit, centerline polyline, and successor lane IDs. We use BFS to
build a route (sequence of lane IDs) from the SDC's start pose to its goal
pose, then concatenate centerlines into a single dense reference path that the
planner follows.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Lane:
    id: int
    type: str
    speed_limit_kmh: float
    centerline: List[Tuple[float, float, float]]
    entry_lanes: List[int] = field(default_factory=list)
    exit_lanes: List[int] = field(default_factory=list)


class LaneGraph:
    def __init__(self, lanes: List[Lane]):
        self.lanes: Dict[int, Lane] = {l.id: l for l in lanes}

    @classmethod
    def load(cls, path: Path) -> "LaneGraph":
        with open(path) as f:
            doc = json.load(f)
        lanes_raw = doc.get("lanes", [])
        lanes: List[Lane] = []
        for l in lanes_raw:
            cl = [
                (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
                for p in l.get("centerline", [])
            ]
            lanes.append(
                Lane(
                    id=int(l["id"]),
                    type=str(l.get("type", "UNDEFINED")),
                    speed_limit_kmh=float(l.get("speed_limit_kmh", 50.0)),
                    centerline=cl,
                    entry_lanes=[int(x) for x in l.get("entry_lanes", [])],
                    exit_lanes=[int(x) for x in l.get("exit_lanes", [])],
                )
            )
        return cls(lanes)

    def closest_lane(
        self, x: float, y: float, heading: Optional[float] = None,
        only_drivable: bool = True, max_dh: float = math.pi / 2,
    ) -> Optional[Lane]:
        best: Optional[Lane] = None
        best_d = float("inf")
        for lane in self.lanes.values():
            if only_drivable and lane.type == "BIKE_LANE":
                continue
            if len(lane.centerline) < 2:
                continue
            for i in range(len(lane.centerline) - 1):
                p1 = lane.centerline[i]
                p2 = lane.centerline[i + 1]
                d = _point_to_segment_dist(x, y, p1[0], p1[1], p2[0], p2[1])
                if heading is not None:
                    seg_h = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    if abs(_wrap_pi(seg_h - heading)) > max_dh:
                        continue
                if d < best_d:
                    best_d = d
                    best = lane
        return best

    def shortest_path(self, start_id: int, goal_id: int) -> List[int]:
        if start_id == goal_id:
            return [start_id]
        prev: Dict[int, int] = {start_id: -1}
        q = deque([start_id])
        while q:
            cur = q.popleft()
            if cur == goal_id:
                path = []
                while cur != -1:
                    path.append(cur)
                    cur = prev[cur]
                return list(reversed(path))
            cur_lane = self.lanes.get(cur)
            if not cur_lane:
                continue
            for nxt in cur_lane.exit_lanes:
                if nxt in self.lanes and nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return []

    def route_centerline(self, lane_ids: List[int]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        for lid in lane_ids:
            lane = self.lanes.get(lid)
            if not lane or len(lane.centerline) < 2:
                continue
            if not out:
                out.extend(lane.centerline)
                continue
            last = out[-1]
            first = lane.centerline[0]
            if math.hypot(first[0] - last[0], first[1] - last[1]) < 0.5:
                out.extend(lane.centerline[1:])
            else:
                out.extend(lane.centerline)
        return out

    def speed_limit_mps(self, lane_ids: List[int], default_kmh: float = 50.0) -> float:
        limits = [self.lanes[i].speed_limit_kmh for i in lane_ids if i in self.lanes]
        if not limits:
            return default_kmh / 3.6
        return min(limits) / 3.6


def resample_polyline(
    pts: List[Tuple[float, float, float]], step: float = 1.0
) -> List[Tuple[float, float, float]]:
    """Resample a polyline at uniform arc-length intervals (XY-based)."""
    if len(pts) < 2:
        return list(pts)
    out: List[Tuple[float, float, float]] = [pts[0]]
    pending = step
    for i in range(len(pts) - 1):
        x1, y1, z1 = pts[i]
        x2, y2, z2 = pts[i + 1]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 1e-9:
            continue
        consumed = 0.0
        while consumed + pending <= seg:
            consumed += pending
            t = consumed / seg
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)))
            pending = step
        pending -= (seg - consumed)
        if pending < 0:
            pending = step
    if math.hypot(out[-1][0] - pts[-1][0], out[-1][1] - pts[-1][1]) > 1e-3:
        out.append(pts[-1])
    return out


def fallback_path_to_goal(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    step: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """Straight line from start to goal — last-resort reference if no route is found."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < step:
        return [start, goal]
    n = max(2, int(math.ceil(dist / step)))
    out: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        t = i / n
        out.append((start[0] + t * dx, start[1] + t * dy, start[2] + t * (goal[2] - start[2])))
    return out


def _point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    qx, qy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - qx, py - qy)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
