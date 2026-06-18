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
GEN_DIR = WORKBENCH_ROOT / "tools" / "gen"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))
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
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class StaticMapDraw:
    lanes: List[dict]
    road_lines: List[dict]
    road_edges: List[dict]
    crosswalks: List[dict]
    driveways: List[dict]


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def load_static_map(path: Path) -> StaticMapDraw:
    if str(path).endswith(".pb"):
        from proto.sim import map_pb2  # type: ignore

        sm = map_pb2.StaticMap()
        sm.ParseFromString(path.read_bytes())
        return StaticMapDraw(
            lanes=[
                {
                    "id": int(l.id),
                    "centerline": [[float(p.x), float(p.y), float(p.z)] for p in l.centerline],
                }
                for l in sm.lanes
            ],
            road_lines=[
                {
                    "id": int(r.id),
                    "polyline": [[float(p.x), float(p.y), float(p.z)] for p in r.polyline],
                }
                for r in sm.road_lines
            ],
            road_edges=[
                {
                    "id": int(r.id),
                    "polyline": [[float(p.x), float(p.y), float(p.z)] for p in r.polyline],
                }
                for r in sm.road_edges
            ],
            crosswalks=[
                {
                    "id": int(c.id),
                    "polygon": [[float(p.x), float(p.y), float(p.z)] for p in c.polygon],
                }
                for c in sm.crosswalks
            ],
            driveways=[
                {
                    "id": int(d.id),
                    "polygon": [[float(p.x), float(p.y), float(p.z)] for p in d.polygon],
                }
                for d in sm.driveways
            ],
        )

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return StaticMapDraw(
        lanes=doc.get("lanes", []),
        road_lines=doc.get("road_lines", []),
        road_edges=doc.get("road_edges", []),
        crosswalks=doc.get("crosswalks", []),
        driveways=doc.get("driveways", []),
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
        raw_id = n.get("id", 0)
        try:
            npc_id = int(raw_id)
        except (TypeError, ValueError):
            npc_id = 0
        out.append(
            NpcView(
                id=npc_id,
                object_type=str(n.get("object_type", "VEHICLE")),
                x=float(n.get("x", 0)),
                y=float(n.get("y", 0)),
                heading=float(n.get("heading", 0)),
                length=float(n.get("length", 4.0)),
                width=float(n.get("width", 1.8)),
                vx=float(n.get("vx", 0.0)),
                vy=float(n.get("vy", 0.0)),
            )
        )
    return out


def npcs_for_frame(fr: dict, scenario: Scenario) -> List[NpcView]:
    npcs = npcs_from_frame(fr)
    if not npcs:
        t_us = int(fr.get("timestamp_us", 0))
        npcs = npcs_at_time(scenario, t_us / 1e6)
    return npcs


def _ego_motion(fr: dict) -> Dict[str, float]:
    vs = fr.get("vehicle_state", fr.get("ego", {}))
    return {
        "speed": float(vs.get("speed", 0.0)),
        "heading": float(vs.get("heading", 0.0)),
    }


def _npc_to_dict(n: NpcView, _ego: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "id": n.id,
        "object_type": n.object_type,
        "x": n.x,
        "y": n.y,
        "heading": n.heading,
        "length": n.length,
        "width": n.width,
        "vx": n.vx,
        "vy": n.vy,
    }
    return doc


def compute_plot_rect(fig, ax, dpi: int) -> Dict[str, float]:
    """Axes bounding box in saved image pixels (origin top-left)."""
    fig.canvas.draw()
    bbox = ax.get_position()
    fig_w = fig.get_size_inches()[0] * dpi
    fig_h = fig.get_size_inches()[1] * dpi
    left = bbox.x0 * fig_w
    width = bbox.width * fig_w
    bottom = bbox.y0 * fig_h
    height = bbox.height * fig_h
    top = fig_h - bottom - height
    return {"left": left, "top": top, "width": width, "height": height}


def _matplotlib_frame_layout(
    scenario: Scenario,
    static_map: StaticMapDraw,
    lane_graph: LaneGraph,
    route_xy: List[Tuple[float, float]],
    frames: List[dict],
    args: argparse.Namespace,
    sdc_xy: Optional[List[Tuple[float, float]]],
    *,
    dpi: int = 100,
    figsize: Tuple[float, float] = (12.0, 10.0),
    frame_idx: int = 0,
) -> Tuple[int, int, Dict[str, float], Dict[str, float]]:
    """Match saved GIF layout: image size, axes plot_rect, world limits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    _draw_frame(
        ax, scenario, static_map, lane_graph, route_xy, frames, frame_idx, args, sdc_xy
    )
    plot_rect = compute_plot_rect(fig, ax, dpi)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    img_w = int(round(figsize[0] * dpi))
    img_h = int(round(figsize[1] * dpi))
    plt.close(fig)
    world = {"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax)}
    return img_w, img_h, plot_rect, world


def build_overlay_doc(
    scenario: Scenario,
    frames: List[dict],
    *,
    dpi: int = 100,
    figsize: Tuple[float, float] = (12.0, 10.0),
    static_map: Optional[StaticMapDraw] = None,
    lane_graph: Optional[LaneGraph] = None,
    route_xy: Optional[List[Tuple[float, float]]] = None,
    args: Optional[argparse.Namespace] = None,
    sdc_xy: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Build hover-overlay metadata aligned with matplotlib GIF rendering."""
    if args is None:
        args = argparse.Namespace(
            no_reference=False,
            show_sdc_track=False,
            ego_length=4.5,
            ego_width=1.85,
            ego_rear_overhang=0.95,
        )
    if static_map is None:
        static_map = load_static_map(scenario.lane_graph_path)
    if lane_graph is None:
        lane_graph = LaneGraph.load(scenario.lane_graph_path)
    if route_xy is None:
        route_pts, _ = build_map_route(scenario, lane_graph, 1.0)
        route_xy = [(p[0], p[1]) for p in route_pts]
    if sdc_xy is None and args.show_sdc_track:
        sdc_xy = _sdc_recorded_xy(scenario)

    img_w, img_h, plot_rect, world = _matplotlib_frame_layout(
        scenario,
        static_map,
        lane_graph,
        route_xy,
        frames,
        args,
        sdc_xy,
        dpi=dpi,
        figsize=figsize,
        frame_idx=0,
    )

    overlay_frames: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        npcs = npcs_for_frame(fr, scenario)
        ego_motion = _ego_motion(fr)
        overlay_frames.append(
            {
                "index": i,
                "ego": ego_motion,
                "npcs": [_npc_to_dict(n, ego_motion) for n in npcs],
            }
        )

    return {
        "version": 1,
        "width": img_w,
        "height": img_h,
        "world": world,
        "plot_rect": plot_rect,
        "frames": overlay_frames,
    }


def write_overlay_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


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


def _fix_gif_frame_duration(path: Path, fps: int) -> None:
    """PillowWriter at high fps often writes duration=0; patch per-frame timing."""
    try:
        from PIL import Image
    except ImportError:
        return
    # GIF/PIL 最小约 10ms；120fps 目标 8.3ms 无法写入，取 10ms ≈ 100fps。
    duration_ms = max(10, int(round(1000 / max(1, fps))))
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(im.copy())
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[duration_ms] * len(frames),
        loop=0,
    )


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
    p.add_argument(
        "--input-format",
        choices=("auto", "json", "proto"),
        default="auto",
        help="scenario_meta/lane_graph/dynamic_objects read format",
    )
    return p.parse_args()


def _scenario_bbox(scenario: Scenario) -> Optional[Tuple[float, float, float, float]]:
    meta = scenario.meta.get("bbox") or {}
    try:
        xmin = float(meta.get("xmin", 0.0))
        ymin = float(meta.get("ymin", 0.0))
        xmax = float(meta.get("xmax", 0.0))
        ymax = float(meta.get("ymax", 0.0))
    except (TypeError, ValueError):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _setup_axes(fig, ax, scenario: Scenario, margin: float = 15.0):
    bbox = _scenario_bbox(scenario)
    if bbox:
        xmin, ymin, xmax, ymax = bbox
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
    for dw in static_map.driveways:
        poly = _polyline_xy(dw.get("polygon", []))
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.fill(xs, ys, color="#d0d0d0", alpha=0.35, zorder=0)

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
    ax,
    pts: Sequence[Tuple[float, float]],
    *,
    color: str,
    lw: float,
    label: Optional[str] = None,
    ls: str = "-",
    zorder: int = 3,
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


GOAL_STAR_MARKERSIZE = 8.4  # original 12, ~30% smaller


def _legend_proxies(
    scenario: Scenario,
    route_xy: List[Tuple[float, float]],
    args: argparse.Namespace,
    sdc_xy: Optional[List[Tuple[float, float]]],
) -> List[Any]:
    from matplotlib.lines import Line2D

    proxies: List[Any] = []
    if not args.no_reference and route_xy:
        proxies.append(Line2D([0], [0], color="#1a73e8", lw=2.2, label="map reference"))
    if args.show_sdc_track and sdc_xy:
        proxies.append(
            Line2D([0], [0], color="#5f6368", lw=1.2, ls="--", label="SDC recorded")
        )
    proxies.append(Line2D([0], [0], color="#34a853", lw=2.0, label="sim ego trail"))
    if scenario.init_pose:
        proxies.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#1a73e8",
                linestyle="None",
                markersize=4,
                label="init",
            )
        )
    if scenario.goal_pose:
        proxies.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="#ea4335",
                linestyle="None",
                markersize=GOAL_STAR_MARKERSIZE,
                label="goal",
            )
        )
    proxies.append(Line2D([0], [0], color="#f9ab00", lw=1.0, label="NPC"))
    proxies.append(Line2D([0], [0], color="#137333", lw=2.0, label="sim ego"))
    return proxies


def _draw_static_scene(
    ax,
    scenario: Scenario,
    static_map: StaticMapDraw,
    lane_graph: LaneGraph,
    route_xy: List[Tuple[float, float]],
    args: argparse.Namespace,
    sdc_xy: Optional[List[Tuple[float, float]]],
) -> None:
    """Map + route + markers + legend (CPU rasterized once for GIF export)."""
    _setup_axes(ax.figure, ax, scenario)
    _draw_full_map(ax, static_map, lane_graph)

    if not args.no_reference and route_xy:
        _draw_polyline_xy(ax, route_xy, color="#1a73e8", lw=2.2, zorder=3)

    if args.show_sdc_track and sdc_xy:
        _draw_polyline_xy(ax, sdc_xy, color="#5f6368", lw=1.2, ls="--", zorder=3)

    if scenario.init_pose:
        ax.plot(
            scenario.init_pose.x,
            scenario.init_pose.y,
            "o",
            color="#1a73e8",
            markersize=4,
            zorder=6,
        )
    if scenario.goal_pose:
        ax.plot(
            scenario.goal_pose.x,
            scenario.goal_pose.y,
            "*",
            color="#ea4335",
            markersize=GOAL_STAR_MARKERSIZE,
            zorder=9,
        )

    proxies = _legend_proxies(scenario, route_xy, args, sdc_xy)
    if proxies:
        ax.legend(handles=proxies, loc="upper right", fontsize=7, framealpha=0.9)


def _draw_obb_artists(
    ax, box: OBB, *, edge: str, face: str, lw: float, zorder: int
) -> List[Any]:
    corners = box.corners()
    xs = [c[0] for c in corners] + [corners[0][0]]
    ys = [c[1] for c in corners] + [corners[0][1]]
    from matplotlib.collections import PolyCollection

    fill = PolyCollection(
        [[(c[0], c[1]) for c in corners]],
        facecolors=face,
        edgecolors=edge,
        linewidths=lw,
        alpha=0.35,
        zorder=zorder,
    )
    ax.add_collection(fill)
    outline, = ax.plot(xs, ys, color=edge, linewidth=lw, zorder=zorder + 1)
    return [fill, outline]


def _draw_dynamic_scene(
    ax,
    scenario: Scenario,
    frames: List[dict],
    frame_idx: int,
    args: argparse.Namespace,
) -> List[Any]:
    """Per-frame ego / NPC / trail only."""
    artists: List[Any] = []

    trail = []
    for i in range(frame_idx + 1):
        vs = frames[i].get("vehicle_state", frames[i].get("ego", {}))
        trail.append((float(vs.get("x", 0)), float(vs.get("y", 0))))
    if len(trail) >= 2:
        xs, ys = zip(*trail)
        ln, = ax.plot(xs, ys, color="#34a853", linewidth=2.0, zorder=4)
        artists.append(ln)

    fr = frames[frame_idx]
    t_us = int(fr.get("timestamp_us", 0))
    t_sec = t_us / 1e6

    npcs = npcs_for_frame(fr, scenario)
    for n in npcs:
        artists.extend(
            _draw_obb_artists(
                ax, npc_obb(n), edge="#f9ab00", face="#f9ab00", lw=1.0, zorder=5
            )
        )

    vs = fr.get("vehicle_state", fr.get("ego", {}))
    ex, ey = float(vs.get("x", 0)), float(vs.get("y", 0))
    eh = float(vs.get("heading", 0))
    esp = float(vs.get("speed", 0))
    ev = fr.get("ego_vehicle", {})
    elen = float(ev.get("length", args.ego_length))
    ewid = float(ev.get("width", args.ego_width))
    erear = float(ev.get("rear_overhang", args.ego_rear_overhang))
    ebox = ego_obb(ex, ey, eh, elen, ewid, erear)
    artists.extend(
        _draw_obb_artists(ax, ebox, edge="#137333", face="#137333", lw=2.0, zorder=7)
    )
    pt, = ax.plot(ex, ey, "ko", markersize=3, zorder=8)
    artists.append(pt)

    rc = fr.get("road_context", {})
    rc_txt = ""
    if rc:
        rc_txt = f"  edge={rc.get('dist_to_road_edge_m', 0):.1f}m"
    ax.set_title(
        f"frame {frame_idx}  t={t_sec:.2f}s  speed={esp:.2f} m/s  npcs={len(npcs)}{rc_txt}"
    )
    return artists


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
    _draw_static_scene(ax, scenario, static_map, lane_graph, route_xy, args, sdc_xy)
    _draw_dynamic_scene(ax, scenario, frames, frame_idx, args)


def _sdc_recorded_xy(scenario: Scenario) -> List[Tuple[float, float]]:
    tr = scenario.sdc_track()
    if tr is None:
        return []
    return [(st.x, st.y) for st in tr.states if st.valid]


def _world_bounds(scenario: Scenario, margin: float = 15.0) -> Dict[str, float]:
    bbox = _scenario_bbox(scenario)
    if bbox:
        xmin, ymin, xmax, ymax = bbox
    else:
        xmin = ymin = -100
        xmax = ymax = 100
    return {
        "xmin": xmin - margin,
        "ymin": ymin - margin,
        "xmax": xmax + margin,
        "ymax": ymax + margin,
    }


def _polyline_to_xy_list(poly: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(p[0]), float(p[1])] for p in poly if len(p) >= 2]


def _frame_to_viz_dict(
    fr: dict, scenario: Scenario, args: argparse.Namespace
) -> Dict[str, Any]:
    vs = fr.get("vehicle_state", fr.get("ego", {}))
    ev = fr.get("ego_vehicle", {})
    npcs = npcs_for_frame(fr, scenario)
    ego_motion = {
        "speed": float(vs.get("speed", 0)),
        "heading": float(vs.get("heading", 0)),
    }
    return {
        "index": int(fr.get("frame_id", 0)),
        "timestamp_us": int(fr.get("timestamp_us", 0)),
        "ego": {
            "x": float(vs.get("x", 0)),
            "y": float(vs.get("y", 0)),
            "heading": float(vs.get("heading", 0)),
            "speed": float(vs.get("speed", 0)),
            "length": float(ev.get("length", args.ego_length)),
            "width": float(ev.get("width", args.ego_width)),
            "rear_overhang": float(ev.get("rear_overhang", args.ego_rear_overhang)),
        },
        "npcs": [_npc_to_dict(n, ego_motion) for n in npcs],
    }


def build_viz_scene_doc(
    scenario: Scenario,
    frames: List[dict],
    *,
    dpi: int = 100,
    figsize: Tuple[float, float] = (12.0, 10.0),
    static_map: Optional[StaticMapDraw] = None,
    lane_graph: Optional[LaneGraph] = None,
    route_xy: Optional[List[Tuple[float, float]]] = None,
    args: Optional[argparse.Namespace] = None,
    sdc_xy: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """JSON scene for WebGL viewer (map once + per-frame ego/npcs)."""
    if args is None:
        args = argparse.Namespace(
            no_reference=False,
            show_sdc_track=False,
            ego_length=4.5,
            ego_width=1.85,
            ego_rear_overhang=0.95,
        )
    if static_map is None:
        static_map = load_static_map(scenario.lane_graph_path)
    if lane_graph is None:
        lane_graph = LaneGraph.load(scenario.lane_graph_path)
    if route_xy is None and not args.no_reference:
        route_pts, _ = build_map_route(scenario, lane_graph, 1.0)
        route_xy = [(p[0], p[1]) for p in route_pts]
    elif route_xy is None:
        route_xy = []
    if sdc_xy is None and args.show_sdc_track:
        sdc_xy = _sdc_recorded_xy(scenario)

    img_w, img_h, plot_rect, world = _matplotlib_frame_layout(
        scenario,
        static_map,
        lane_graph,
        route_xy,
        frames,
        args,
        sdc_xy,
        dpi=dpi,
        figsize=figsize,
        frame_idx=0,
    )

    lane_centerlines: List[Dict[str, Any]] = []
    for lane in lane_graph.lanes.values():
        if len(lane.centerline) < 2:
            continue
        lane_centerlines.append(
            {
                "xy": [[p[0], p[1]] for p in lane.centerline],
                "color": "#c4c7c5" if lane.type == "BIKE_LANE" else "#9aa0a6",
                "width": 0.85 if lane.type == "FREEWAY" else 0.55,
            }
        )

    doc: Dict[str, Any] = {
        "version": 1,
        "renderer": "webgl",
        "scenario_id": scenario.scenario_id,
        "width": img_w,
        "height": img_h,
        "dpi": dpi,
        "world": world,
        "plot_rect": plot_rect,
        "background": "#ffffff",
        "map": {
            "driveways": [
                _polyline_to_xy_list(dw.get("polygon", []))
                for dw in static_map.driveways
            ],
            "crosswalks": [
                _polyline_to_xy_list(cw.get("polygon", []))
                for cw in static_map.crosswalks
            ],
            "road_edges": [
                _polyline_to_xy_list(e.get("polyline", []))
                for e in static_map.road_edges
            ],
            "road_lines": [
                _polyline_to_xy_list(r.get("polyline", []))
                for r in static_map.road_lines
            ],
            "lane_centerlines": lane_centerlines,
        },
        "route_xy": [[x, y] for x, y in (route_xy or [])],
        "sdc_xy": [[x, y] for x, y in (sdc_xy or [])],
        "init": None,
        "goal": None,
        "frames": [_frame_to_viz_dict(fr, scenario, args) for fr in frames],
    }
    if scenario.init_pose:
        doc["init"] = {"x": scenario.init_pose.x, "y": scenario.init_pose.y}
    if scenario.goal_pose:
        doc["goal"] = {"x": scenario.goal_pose.x, "y": scenario.goal_pose.y}
    return doc


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
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

    scenario = load_scenario(scenario_dir, input_format=args.input_format)
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
        f"crosswalks={len(static_map.crosswalks)} driveways={len(static_map.driveways)} "
        f"route_lanes={len(route_ids)} "
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
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.set_dpi(args.dpi)
        _draw_static_scene(ax, scenario, static_map, lane_graph, route_xy, args, sdc_xy)
        ax.set_title("")
        fig.canvas.draw()
        static_bg = fig.canvas.copy_from_bbox(ax.bbox)
        dynamic_artists: List[Any] = []

        def anim_fn(i: int):
            fig.canvas.restore_region(static_bg)
            for art in dynamic_artists:
                try:
                    art.remove()
                except (NotImplementedError, ValueError):
                    pass
            dynamic_artists.clear()
            dynamic_artists.extend(
                _draw_dynamic_scene(ax, scenario, frames, i, args)
            )
            for art in dynamic_artists:
                ax.draw_artist(art)
            fig.canvas.draw()

        ani = FuncAnimation(
            fig,
            anim_fn,
            frames=len(frames),
            interval=1000 // max(1, args.fps),
            blit=False,
        )
        print(
            f"[viz] GIF export: matplotlib CPU (static map cached once; no GPU). "
            f"frames={len(frames)} dpi={args.dpi}"
        )
        ani.save(str(out), writer=PillowWriter(fps=args.fps))
        _fix_gif_frame_duration(out, args.fps)
        overlay_path = out.with_name(out.stem + "_overlay.json")
        overlay_doc = build_overlay_doc(
            scenario,
            frames,
            dpi=args.dpi,
            static_map=static_map,
            lane_graph=lane_graph,
            route_xy=route_xy,
            args=args,
            sdc_xy=sdc_xy,
        )
        write_overlay_json(overlay_path, overlay_doc)
        print(f"[viz] wrote {out}")
        print(f"[viz] wrote {overlay_path}")
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
