#!/usr/bin/env python3
"""2D visualization: full static map, reference route, sim ego trail, NPC/ego OBBs.

Requires: pip install matplotlib

Example:
  python3 tools/viz_sim.py \\
    --scenario-dir scenarios/waymo_scenario_244 \\
    --sim-log output/log/waymo_scenario_244_sim_log.json \\
    --animate --output output/viz/waymo_scenario_244_sim.gif --fps 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH_ROOT))
sys.path.insert(0, str(WORKBENCH_ROOT / "pysim"))
from hyw_paths import OUTPUT_DIR, WORKBENCH_ROOT  # noqa: E402,F401

from waymo_sim.geometry import OBB  # noqa: E402
from waymo_sim.lane_graph import LaneGraph, resample_polyline  # noqa: E402
from waymo_sim.scenario import Pose2D, Scenario, TrackState, load_scenario  # noqa: E402


@dataclass
class NpcView:
    id: int
    object_type: str
    x: float
    y: float
    heading: float
    length: float
    width: float


@dataclass
class StaticMapDraw:
    lanes: List[dict]
    road_lines: List[dict]
    road_edges: List[dict]
    crosswalks: List[dict]


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def load_static_map(path: Path) -> StaticMapDraw:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return StaticMapDraw(
        lanes=doc.get("lanes", []),
        road_lines=doc.get("road_lines", []),
        road_edges=doc.get("road_edges", []),
        crosswalks=doc.get("crosswalks", []),
    )


def npcs_at_time(scenario: Scenario, scenario_time: float) -> List[NpcView]:
    ts = scenario.timestamps_seconds
    if not ts:
        return []
    if scenario_time <= ts[0]:
        return _npcs_at_index(scenario, 0)
    if scenario_time >= ts[-1]:
        return _npcs_at_index(scenario, len(ts) - 1)
    lo, hi = 0, len(ts) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ts[mid] <= scenario_time:
            lo = mid
        else:
            hi = mid
    seg = ts[hi] - ts[lo]
    a = 0.0 if seg < 1e-9 else (scenario_time - ts[lo]) / seg
    return _interp_npcs(scenario, lo, hi, a)


def _npcs_at_index(scenario: Scenario, idx: int) -> List[NpcView]:
    out: List[NpcView] = []
    for tr in scenario.tracks:
        if tr.is_sdc or idx < 0 or idx >= len(tr.states):
            continue
        st = tr.states[idx]
        if not st.valid:
            continue
        out.append(_npc_from_state(tr.id, tr.object_type, st))
    return out


def _npc_from_state(tid: int, otype: str, st: TrackState) -> NpcView:
    return NpcView(
        id=tid,
        object_type=otype,
        x=st.x,
        y=st.y,
        heading=st.yaw,
        length=st.length,
        width=st.width,
    )


def _interp_npcs(scenario: Scenario, lo: int, hi: int, a: float) -> List[NpcView]:
    out: List[NpcView] = []
    for tr in scenario.tracks:
        if tr.is_sdc:
            continue
        if lo >= len(tr.states) or hi >= len(tr.states):
            continue
        s0, s1 = tr.states[lo], tr.states[hi]
        if s0.valid and s1.valid:
            dy = _wrap_pi(s1.yaw - s0.yaw)
            yaw = _wrap_pi(s0.yaw + a * dy)
            out.append(
                NpcView(
                    id=tr.id,
                    object_type=tr.object_type,
                    x=s0.x + a * (s1.x - s0.x),
                    y=s0.y + a * (s1.y - s0.y),
                    heading=yaw,
                    length=s1.length,
                    width=s1.width,
                )
            )
        elif s0.valid:
            out.append(_npc_from_state(tr.id, tr.object_type, s0))
        elif s1.valid:
            out.append(_npc_from_state(tr.id, tr.object_type, s1))
    return out


def npcs_from_frame(fr: dict) -> List[NpcView]:
    out: List[NpcView] = []
    for n in fr.get("npcs", []):
        out.append(
            NpcView(
                id=int(n.get("id", 0)),
                object_type=str(n.get("object_type", "VEHICLE")),
                x=float(n.get("x", 0)),
                y=float(n.get("y", 0)),
                heading=float(n.get("heading", 0)),
                length=float(n.get("length", 4.0)),
                width=float(n.get("width", 1.8)),
            )
        )
    return out


def build_map_route(
    scenario: Scenario, lane_graph: LaneGraph, reference_step: float
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    if scenario.init_pose is None or scenario.goal_pose is None:
        return [], []
    init, goal = scenario.init_pose, scenario.goal_pose
    start = lane_graph.closest_lane(init.x, init.y, heading=init.yaw)
    goal_lane = lane_graph.closest_lane(goal.x, goal.y, heading=goal.yaw)
    if start is None or goal_lane is None:
        return [], []
    route = lane_graph.shortest_path(start.id, goal_lane.id)
    if not route:
        return [], []
    raw = lane_graph.route_centerline(route)
    if len(raw) < 2:
        return [], []
    if math.hypot(raw[0][0] - init.x, raw[0][1] - init.y) > 0.5:
        raw = [(init.x, init.y, raw[0][2])] + raw
    if math.hypot(raw[-1][0] - goal.x, raw[-1][1] - goal.y) > 0.5:
        raw = raw + [(goal.x, goal.y, raw[-1][2])]
    return resample_polyline(raw, step=reference_step), route


def ego_obb(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    rear_overhang: float,
) -> OBB:
    d = length / 2.0 - rear_overhang
    cx = x + d * math.cos(heading)
    cy = y + d * math.sin(heading)
    return OBB(cx=cx, cy=cy, heading=heading, half_length=length / 2.0, half_width=width / 2.0)


def npc_obb(n: NpcView) -> OBB:
    return OBB(
        cx=n.x,
        cy=n.y,
        heading=n.heading,
        half_length=max(0.5, n.length * 0.5),
        half_width=max(0.3, n.width * 0.5),
    )


def load_sim_log(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("frames", [])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D sim visualization (full map + OBBs)")
    p.add_argument("--scenario-dir", required=True, type=Path)
    p.add_argument("--sim-log", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--reference-step", type=float, default=1.0)
    p.add_argument("--ego-length", type=float, default=4.5)
    p.add_argument("--ego-width", type=float, default=1.85)
    p.add_argument("--ego-rear-overhang", type=float, default=0.95)
    p.add_argument("--show-sdc-track", action="store_true")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def _setup_axes(fig, ax, scenario: Scenario, margin: float = 15.0):
    meta = scenario.meta.get("bbox", {})
    if meta:
        xmin = float(meta.get("xmin", -100))
        ymin = float(meta.get("ymin", -100))
        xmax = float(meta.get("xmax", 100))
        ymax = float(meta.get("ymax", 100))
    else:
        xmin = ymin = -100
        xmax = ymax = 100
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linewidth=0.5)


def _polyline_xy(poly: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]


def _draw_full_map(ax, static_map: StaticMapDraw, lane_graph: LaneGraph) -> None:
    for cw in static_map.crosswalks:
        poly = _polyline_xy(cw.get("polygon", []))
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, ys, color="#e8eaed", alpha=0.45, zorder=0)

    for edge in static_map.road_edges:
        pts = _polyline_xy(edge.get("polyline", []))
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#5f6368", linewidth=1.4, alpha=0.75, zorder=1)

    for line in static_map.road_lines:
        pts = _polyline_xy(line.get("polyline", []))
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#fbbc04", linewidth=0.9, alpha=0.7, linestyle="--", zorder=1)

    for lane in lane_graph.lanes.values():
        if len(lane.centerline) < 2:
            continue
        xs = [p[0] for p in lane.centerline]
        ys = [p[1] for p in lane.centerline]
        color = "#9aa0a6" if lane.type != "BIKE_LANE" else "#c4c7c5"
        lw = 0.55 if lane.type != "FREEWAY" else 0.85
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.6, zorder=2)


def _draw_polyline_xy(
    ax, pts: Sequence[Tuple[float, float]], *, color: str, lw: float, label: str, ls: str = "-", zorder: int = 3
):
    if len(pts) < 2:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=label, zorder=zorder)


def _draw_obb(ax, box: OBB, *, edge: str, face: str, lw: float, label: Optional[str] = None, zorder: int = 5):
    corners = box.corners()
    xs = [c[0] for c in corners] + [corners[0][0]]
    ys = [c[1] for c in corners] + [corners[0][1]]
    ax.fill(xs, ys, facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.35, zorder=zorder, label=label)
    ax.plot(xs, ys, color=edge, linewidth=lw, zorder=zorder + 1)


def _draw_frame(
    ax,
    scenario: Scenario,
    static_map: StaticMapDraw,
    lane_graph: LaneGraph,
    route_xy: List[Tuple[float, float]],
    frames: List[dict],
    frame_idx: int,
    args: argparse.Namespace,
    sdc_xy: Optional[List[Tuple[float, float]]],
):
    ax.clear()
    _setup_axes(ax.figure, ax, scenario)
    _draw_full_map(ax, static_map, lane_graph)

    if not args.no_reference and route_xy:
        _draw_polyline_xy(ax, route_xy, color="#1a73e8", lw=2.2, label="map reference", zorder=3)

    if args.show_sdc_track and sdc_xy:
        _draw_polyline_xy(ax, sdc_xy, color="#5f6368", lw=1.2, ls="--", label="SDC recorded", zorder=3)

    trail = []
    for i in range(frame_idx + 1):
        vs = frames[i].get("vehicle_state", frames[i].get("ego", {}))
        trail.append((float(vs.get("x", 0)), float(vs.get("y", 0))))
    if len(trail) >= 2:
        _draw_polyline_xy(ax, trail, color="#34a853", lw=2.0, label="sim ego trail", zorder=4)

    if scenario.init_pose:
        ax.plot(scenario.init_pose.x, scenario.init_pose.y, "o", color="#1a73e8", markersize=7, label="init", zorder=6)
    if scenario.goal_pose:
        ax.plot(scenario.goal_pose.x, scenario.goal_pose.y, "*", color="#ea4335", markersize=12, label="goal", zorder=6)

    fr = frames[frame_idx]
    t_us = int(fr.get("timestamp_us", 0))
    t_sec = t_us / 1e6

    npcs = npcs_from_frame(fr)
    if not npcs:
        npcs = npcs_at_time(scenario, t_sec)
    for n in npcs:
        _draw_obb(ax, npc_obb(n), edge="#f9ab00", face="#f9ab00", lw=1.0, zorder=5)

    vs = fr.get("vehicle_state", fr.get("ego", {}))
    ex, ey = float(vs.get("x", 0)), float(vs.get("y", 0))
    eh = float(vs.get("heading", 0))
    esp = float(vs.get("speed", 0))
    ev = fr.get("ego_vehicle", {})
    elen = float(ev.get("length", args.ego_length))
    ewid = float(ev.get("width", args.ego_width))
    erear = float(ev.get("rear_overhang", args.ego_rear_overhang))
    ebox = ego_obb(ex, ey, eh, elen, ewid, erear)
    _draw_obb(ax, ebox, edge="#137333", face="#137333", lw=2.0, label="sim ego", zorder=7)
    ax.plot(ex, ey, "ko", markersize=3, zorder=8)

    rc = fr.get("road_context", {})
    rc_txt = ""
    if rc:
        rc_txt = f"  edge={rc.get('dist_to_road_edge_m', 0):.1f}m"
    ax.set_title(
        f"frame {frame_idx}  t={t_sec:.2f}s  speed={esp:.2f} m/s  npcs={len(npcs)}{rc_txt}"
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=7, framealpha=0.9)


def _sdc_recorded_xy(scenario: Scenario) -> List[Tuple[float, float]]:
    tr = scenario.sdc_track()
    if tr is None:
        return []
    return [(st.x, st.y) for st in tr.states if st.valid]


def main() -> int:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    args = _parse_args()
    scenario_dir = args.scenario_dir.expanduser().resolve()
    sim_log = args.sim_log.expanduser().resolve()
    if not sim_log.is_file():
        print(f"sim log not found: {sim_log}", file=sys.stderr)
        return 2

    scenario = load_scenario(scenario_dir)
    static_map = load_static_map(scenario.lane_graph_path)
    lane_graph = LaneGraph.load(scenario.lane_graph_path)
    route_pts, route_ids = build_map_route(scenario, lane_graph, args.reference_step)
    route_xy = [(p[0], p[1]) for p in route_pts]
    frames = load_sim_log(sim_log)
    if not frames:
        print("sim log has no frames", file=sys.stderr)
        return 2

    print(
        f"[viz] scenario={scenario.scenario_id} lanes={len(lane_graph.lanes)} "
        f"road_lines={len(static_map.road_lines)} road_edges={len(static_map.road_edges)} "
        f"crosswalks={len(static_map.crosswalks)} route_lanes={len(route_ids)} "
        f"ref_pts={len(route_xy)} sim_frames={len(frames)}"
    )
    sdc_xy = _sdc_recorded_xy(scenario) if args.show_sdc_track else None

    if args.interactive:
        fig, ax = plt.subplots(figsize=(12, 10))
        from matplotlib.widgets import Slider

        def update(idx: int):
            _draw_frame(ax, scenario, static_map, lane_graph, route_xy, frames, idx, args, sdc_xy)
            fig.canvas.draw_idle()

        ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        slider = Slider(ax_slider, "frame", 0, len(frames) - 1, valinit=0, valstep=1)
        slider.on_changed(lambda val: update(int(val)))
        update(0)
        plt.show()
        return 0

    if args.animate:
        out = args.output or (OUTPUT_DIR / "viz" / "sim_anim.gif")
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 10))

        def anim_fn(i: int):
            _draw_frame(ax, scenario, static_map, lane_graph, route_xy, frames, i, args, sdc_xy)

        ani = FuncAnimation(fig, anim_fn, frames=len(frames), interval=1000 // max(1, args.fps))
        ani.save(str(out), writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        print(f"[viz] wrote {out}")
        plt.close(fig)
        return 0

    frame_idx = len(frames) - 1 if args.frame < 0 else max(0, min(args.frame, len(frames) - 1))
    fig, ax = plt.subplots(figsize=(12, 10))
    _draw_frame(ax, scenario, static_map, lane_graph, route_xy, frames, frame_idx, args, sdc_xy)
    out = args.output or (OUTPUT_DIR / "viz" / f"frame_{frame_idx:04d}.png")
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[viz] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
