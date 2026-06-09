#!/usr/bin/env python3
"""Visualize road edges from lane_graph (JSON or pb).

Highlights road edge polylines, optional lane context, and ego clearance
when a sim log is provided (matches grading MinDistToRoadEdges / corner checks).

Requires: pip install matplotlib

Example:
  python3 tools/viz_road_edge.py --scenario-dir scenarios/waymo_scenario_0

  python3 tools/viz_road_edge.py \\
    --scenario-dir scenarios/waymo_scenario_0 \\
    --sim-log output/log/waymo_scenario_0_sim_log.json \\
    --interactive

  python3 tools/viz_road_edge.py --map scenarios/waymo_scenario_0/lane_graph.json \\
    --output output/viz/waymo_scenario_0_road_edges.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH_ROOT))
sys.path.insert(0, str(WORKBENCH_ROOT / "pysim"))
GEN_DIR = WORKBENCH_ROOT / "tools" / "gen"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from hyw_paths import OUTPUT_DIR  # noqa: E402

from waymo_sim.geometry import OBB  # noqa: E402
from waymo_sim.lane_graph import LaneGraph  # noqa: E402
from waymo_sim.scenario import load_scenario  # noqa: E402

from viz_sim import (  # noqa: E402
    StaticMapDraw,
    _polyline_xy,
    ego_obb,
    load_sim_log,
    load_static_map,
)


def _point_to_segment_dist(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    qx = x1 + t * dx
    qy = y1 + t * dy
    return math.hypot(px - qx, py - qy)


def min_dist_to_road_edges(
    road_edges: Sequence[dict], x: float, y: float
) -> float:
    best = float("inf")
    for edge in road_edges:
        pts = edge.get("polyline", [])
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            best = min(
                best,
                _point_to_segment_dist(
                    x, y, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
                ),
            )
    return best


def min_corner_road_edge_dist(
    road_edges: Sequence[dict], box: OBB
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Return (min corner dist, [(cx, cy, dist), ...])."""
    corner_dists: List[Tuple[float, float, float]] = []
    best = float("inf")
    for cx, cy in box.corners():
        d = min_dist_to_road_edges(road_edges, cx, cy)
        corner_dists.append((cx, cy, d))
        best = min(best, d)
    return best, corner_dists


def _map_bounds(
    static_map: StaticMapDraw, lane_graph: Optional[LaneGraph], margin: float
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for edge in static_map.road_edges:
        for x, y in _polyline_xy(edge.get("polyline", [])):
            xs.append(x)
            ys.append(y)
    if lane_graph is not None:
        for lane in lane_graph.lanes.values():
            for p in lane.centerline:
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return -100.0, -100.0, 100.0, 100.0
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _edge_color(edge_type: str) -> str:
    if "MEDIAN" in edge_type:
        return "#d93025"
    if "CURB" in edge_type:
        return "#e37400"
    return "#1a73e8"


def _draw_lanes_faint(ax, lane_graph: Optional[LaneGraph]) -> None:
    if lane_graph is None:
        return
    for lane in lane_graph.lanes.values():
        if len(lane.centerline) < 2:
            continue
        xs = [p[0] for p in lane.centerline]
        ys = [p[1] for p in lane.centerline]
        ax.plot(xs, ys, color="#dadce0", linewidth=0.45, alpha=0.55, zorder=0)


def _draw_road_edges(
    ax,
    static_map: StaticMapDraw,
    *,
    show_ids: bool,
    linewidth: float,
) -> None:
    for edge in static_map.road_edges:
        pts = _polyline_xy(edge.get("polyline", []))
        if len(pts) < 2:
            continue
        edge_type = str(edge.get("type", "TYPE_ROAD_EDGE_BOUNDARY"))
        color = _edge_color(edge_type)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9, zorder=2)
        if show_ids and pts:
            mid = pts[len(pts) // 2]
            ax.annotate(
                str(edge.get("id", "")),
                mid,
                fontsize=6,
                color=color,
                alpha=0.85,
                ha="center",
                zorder=3,
            )


def _draw_ego(
    ax,
    fr: dict,
    road_edges: Sequence[dict],
    *,
    clearance_m: float,
    ego_length: float,
    ego_width: float,
    ego_rear: float,
) -> None:
    vs = fr.get("vehicle_state", fr.get("ego", {}))
    ex, ey = float(vs.get("x", 0)), float(vs.get("y", 0))
    eh = float(vs.get("heading", 0))
    ev = fr.get("ego_vehicle", {})
    elen = float(ev.get("length", ego_length))
    ewid = float(ev.get("width", ego_width))
    erear = float(ev.get("rear_overhang", ego_rear))
    box = ego_obb(ex, ey, eh, elen, ewid, erear)

    center_dist = min_dist_to_road_edges(road_edges, ex, ey)
    corner_min, corner_dists = min_corner_road_edge_dist(road_edges, box)
    violated = corner_min < clearance_m

    corners = box.corners()
    xs = [c[0] for c in corners] + [corners[0][0]]
    ys = [c[1] for c in corners] + [corners[0][1]]
    face = "#fce8e6" if violated else "#e6f4ea"
    edge = "#d93025" if violated else "#137333"
    ax.fill(xs, ys, facecolor=face, edgecolor=edge, linewidth=2.0, alpha=0.45, zorder=5)
    ax.plot(xs, ys, color=edge, linewidth=2.0, zorder=6)
    ax.plot(ex, ey, "ko", markersize=4, zorder=7)

    import matplotlib.pyplot as plt

    for cx, cy, d in corner_dists:
        circle_color = "#d93025" if d < clearance_m else "#34a853"
        circle = plt.Circle(
            (cx, cy),
            clearance_m,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            color=circle_color,
            alpha=0.55,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.annotate(
            f"{d:.2f}m",
            (cx, cy),
            fontsize=6,
            color=circle_color,
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            zorder=8,
        )

    rc = fr.get("road_context", {})
    sim_edge = rc.get("dist_to_road_edge_m")
    sim_txt = f"  sim_edge={sim_edge:.2f}m" if sim_edge is not None else ""
    t_us = int(fr.get("timestamp_us", 0))
    ax.set_title(
        f"road edge clearance  center={center_dist:.2f}m  corner_min={corner_min:.2f}m"
        f"  threshold={clearance_m:.2f}m  t={t_us / 1e6:.2f}s{sim_txt}",
        fontsize=10,
    )


def _draw_trail(ax, frames: Sequence[dict], upto: int) -> None:
    trail: List[Tuple[float, float]] = []
    for i in range(upto + 1):
        vs = frames[i].get("vehicle_state", frames[i].get("ego", {}))
        trail.append((float(vs.get("x", 0)), float(vs.get("y", 0))))
    if len(trail) < 2:
        return
    xs, ys = zip(*trail)
    ax.plot(xs, ys, color="#5f6368", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)


def _draw_frame(
    ax,
    static_map: StaticMapDraw,
    lane_graph: Optional[LaneGraph],
    frames: Optional[Sequence[dict]],
    frame_idx: int,
    args: argparse.Namespace,
) -> None:
    ax.clear()
    xmin, ymin, xmax, ymax = _map_bounds(static_map, lane_graph, args.margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    if not args.no_lanes:
        _draw_lanes_faint(ax, lane_graph)
    _draw_road_edges(ax, static_map, show_ids=args.show_ids, linewidth=args.linewidth)

    n_edges = len(static_map.road_edges)
    if frames:
        _draw_trail(ax, frames, frame_idx)
        _draw_ego(
            ax,
            frames[frame_idx],
            static_map.road_edges,
            clearance_m=args.clearance,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            ego_rear=args.ego_rear_overhang,
        )
    else:
        types = sorted({str(e.get("type", "")) for e in static_map.road_edges})
        ax.set_title(f"road edges ({n_edges})  types={', '.join(types) or 'n/a'}")

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#1a73e8", linewidth=2, label="road edge (boundary)"),
        Line2D([0], [0], color="#e37400", linewidth=2, label="road edge (curb)"),
        Line2D([0], [0], color="#d93025", linewidth=2, label="road edge (median)"),
    ]
    if frames:
        legend_items.append(
            Line2D([0], [0], color="#137333", linewidth=2, label="ego OBB (ok)")
        )
        legend_items.append(
            Line2D([0], [0], color="#d93025", linewidth=2, label="ego OBB (violation)")
        )
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.9)


def _resolve_map_path(args: argparse.Namespace) -> Path:
    if args.map is not None:
        return args.map.expanduser().resolve()
    if args.scenario_dir is None:
        raise SystemExit("provide --scenario-dir or --map")
    scenario = load_scenario(args.scenario_dir.expanduser().resolve(), input_format=args.input_format)
    return scenario.lane_graph_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize road edges from lane_graph")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenario-dir", type=Path, help="scenario folder with lane_graph.json")
    src.add_argument("--map", type=Path, help="lane_graph.json or lane_graph.pb")
    p.add_argument("--sim-log", type=Path, default=None, help="optional sim log for ego overlay")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--clearance", type=float, default=0.35, help="grading road edge clearance (m)")
    p.add_argument("--margin", type=float, default=10.0, help="plot margin around map bounds (m)")
    p.add_argument("--linewidth", type=float, default=2.0)
    p.add_argument("--show-ids", action="store_true", help="label each road edge with feature id")
    p.add_argument("--no-lanes", action="store_true", help="hide faint lane centerlines")
    p.add_argument("--ego-length", type=float, default=4.5)
    p.add_argument("--ego-width", type=float, default=1.85)
    p.add_argument("--ego-rear-overhang", type=float, default=0.95)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--input-format",
        choices=("auto", "json", "proto"),
        default="auto",
        help="scenario read format when using --scenario-dir",
    )
    return p.parse_args()


def main() -> int:
    global plt
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    args = _parse_args()
    map_path = _resolve_map_path(args)
    if not map_path.is_file():
        print(f"map not found: {map_path}", file=sys.stderr)
        return 2

    static_map = load_static_map(map_path)
    lane_graph: Optional[LaneGraph] = None
    if not args.no_lanes:
        try:
            lane_graph = LaneGraph.load(map_path)
        except Exception:
            lane_graph = None

    frames: Optional[List[dict]] = None
    if args.sim_log is not None:
        sim_log = args.sim_log.expanduser().resolve()
        if not sim_log.is_file():
            print(f"sim log not found: {sim_log}", file=sys.stderr)
            return 2
        frames = load_sim_log(sim_log)
        if not frames:
            print("sim log has no frames", file=sys.stderr)
            return 2

    print(
        f"[viz_road_edge] map={map_path.name} road_edges={len(static_map.road_edges)} "
        f"lanes={len(lane_graph.lanes) if lane_graph else 0} "
        f"sim_frames={len(frames) if frames else 0}"
    )

    if args.interactive and frames:
        fig, ax = plt.subplots(figsize=(12, 10))
        from matplotlib.widgets import Slider

        def update(idx: int) -> None:
            _draw_frame(ax, static_map, lane_graph, frames, idx, args)
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        slider = Slider(ax_slider, "frame", 0, len(frames) - 1, valinit=0, valstep=1)
        slider.on_changed(lambda val: update(int(val)))
        update(0)
        plt.show()
        return 0

    fig, ax = plt.subplots(figsize=(12, 10))
    frame_idx = 0
    if frames:
        frame_idx = len(frames) - 1 if args.frame < 0 else max(0, min(args.frame, len(frames) - 1))
    _draw_frame(ax, static_map, lane_graph, frames, frame_idx, args)

    if args.interactive:
        plt.show()
        return 0

    stem = map_path.parent.name if map_path.parent.name else map_path.stem
    out = args.output or (OUTPUT_DIR / "viz" / f"{stem}_road_edges.png")
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[viz_road_edge] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
