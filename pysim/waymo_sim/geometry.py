"""Oriented-bounding-box geometry helpers.

Used by both the simulator (for collision detection) and any planner that wants
volume-aware obstacle inflation. Pure stdlib so it runs anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class OBB:
    """Oriented bounding box on the XY plane.

    `half_length` is along the heading axis; `half_width` is perpendicular.
    """

    cx: float
    cy: float
    heading: float
    half_length: float
    half_width: float

    def corners(self) -> List[Tuple[float, float]]:
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        l = self.half_length
        w = self.half_width
        out: List[Tuple[float, float]] = []
        for lx, ly in ((l, w), (l, -w), (-l, -w), (-l, w)):
            out.append((self.cx + c * lx - s * ly, self.cy + s * lx + c * ly))
        return out

    def axes(self) -> List[Tuple[float, float]]:
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return [(c, s), (-s, c)]


def _project(corners: Iterable[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    vals = [px * axis[0] + py * axis[1] for px, py in corners]
    return min(vals), max(vals)


def obb_overlap(a: OBB, b: OBB) -> bool:
    """Separating Axis Theorem: True iff the two OBBs overlap on XY."""
    ca = a.corners()
    cb = b.corners()
    for ax in a.axes() + b.axes():
        a_min, a_max = _project(ca, ax)
        b_min, b_max = _project(cb, ax)
        if a_max < b_min or b_max < a_min:
            return False
    return True


def obb_overlap_with_inflation(a: OBB, b: OBB, inflate: float) -> bool:
    """Inflate `b` by `inflate` on both half-extents before testing.

    Useful for swept / safety-margin checks without rebuilding the box.
    """
    inflated = OBB(b.cx, b.cy, b.heading, b.half_length + inflate, b.half_width + inflate)
    return obb_overlap(a, inflated)


def aabb_of(boxes: Iterable[OBB]) -> Tuple[float, float, float, float]:
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for box in boxes:
        for x, y in box.corners():
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
    return xmin, ymin, xmax, ymax
