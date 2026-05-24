#!/usr/bin/env python3
"""Closed-loop sim: open-source planner on Waymo NPC backgrounds.

Reads a scenario directory produced by tools/waymo_to_lanelet2.py, builds a
route through the lane graph, runs the chosen planner against the replayed NPC
tracks (with OBB-based collision detection), and writes a SimLog JSON that the
existing C++ grading_main can re-score offline. Optionally an in-process Python
grader mirrors legacy metrics (`--no-python-grader` to disable; use C++ only).

Example:
  python run_sim.py \\
      --scenario-dir ../scenarios/waymo_scenario_5 \\
      --planner idm_pure_pursuit \\
      --dt 0.1

  (Default SimLog: <repo>/output/log/sim_log.json; grading report: <repo>/output/report/grading_report.json.)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_LOG_DIR = REPO_ROOT / "output" / "log"
DEFAULT_REPORT_DIR = REPO_ROOT / "output" / "report"
DEFAULT_SIMLOG_PATH = DEFAULT_LOG_DIR / "sim_log.json"
DEFAULT_GRADING_REPORT_PATH = DEFAULT_REPORT_DIR / "grading_report.json"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from waymo_sim.grading_stream import CppOnlineGrader, OnlineGrader, SimLogWriter
from waymo_sim.lane_graph import LaneGraph, fallback_path_to_goal, resample_polyline
from waymo_sim.planners import PLANNERS
from waymo_sim.scenario import Pose2D, load_scenario
from waymo_sim.vehicle import BicycleVehicle, VehicleParams, VehicleState
from waymo_sim.world import World, WorldConfig


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario-dir", required=True, help="Directory with scenario_meta/dynamic_objects/lane_graph json")
    p.add_argument("--planner", default="idm_pure_pursuit", choices=sorted(PLANNERS.keys()))
    p.add_argument(
        "--output",
        default=str(DEFAULT_SIMLOG_PATH),
        help=f"SimLog JSON output path (default: {DEFAULT_SIMLOG_PATH})",
    )
    p.add_argument("--source-tag", default="waymo_sim", help="`source` field embedded into SimLog")
    p.add_argument("--dt", type=float, default=0.1, help="Sim / control timestep in seconds")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Override scenario duration (0 = use full duration)")
    p.add_argument("--stop-on-collision", action="store_true", help="Halt simulation on the first non-exempt collision")
    p.add_argument("--no-interpolate-npcs", action="store_true", help="Use nearest-neighbor NPC sampling instead of linear interpolation")

    # ego model knobs (defaults match a typical sedan)
    p.add_argument("--ego-length", type=float, default=4.5)
    p.add_argument("--ego-width", type=float, default=1.85)
    p.add_argument("--ego-wheelbase", type=float, default=2.7)
    p.add_argument("--ego-rear-overhang", type=float, default=0.95)
    p.add_argument("--ego-max-speed", type=float, default=33.3)

    # planner knobs
    p.add_argument("--desired-speed", type=float, default=13.9, help="IDM v0; clipped to lane speed limit")
    p.add_argument("--reference-step", type=float, default=1.0, help="Resample step for reference path (m)")

    # C++ scorer
    p.add_argument(
        "--grading-bin",
        default="",
        help="Path to grading_main binary; if set, also runs the C++ scorer",
    )
    p.add_argument(
        "--grading-report",
        default="",
        help="Where to write the C++ grading_main JSON report (default: <repo>/output/report/grading_report.json)",
    )
    p.add_argument(
        "--cpp-mode",
        choices=("online", "offline", "both", "off"),
        default="online",
        help=(
            "How to drive the C++ scorer when --grading-bin is given:\n"
            "  online  : pipe frames to `grading_main --stream` live (default)\n"
            "  offline : run `grading_main <SimLog>` once after sim finishes (batch)\n"
            "  both    : online during sim AND offline batch at the end\n"
            "  off     : skip the C++ scorer (Python grader if enabled, else SimLog only)"
        ),
    )
    p.add_argument(
        "--no-python-grader",
        action="store_true",
        help=(
            "Do not run the in-process Python OnlineGrader. "
            "Use with --grading-bin so Pass/Fail comes only from grading_main "
            "(avoids duplicating new C++ metrics in Python)."
        ),
    )

    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="Reserved for future stochastic planners")

    return p.parse_args(argv)


def _build_route(scenario, lane_graph: LaneGraph, init: Pose2D, goal: Pose2D, step: float):
    """Return (reference_path, speed_limit_mps, route_lane_ids, used_fallback)."""
    start_lane = lane_graph.closest_lane(init.x, init.y, heading=init.yaw)
    goal_lane = lane_graph.closest_lane(goal.x, goal.y, heading=goal.yaw)
    if start_lane is None or goal_lane is None:
        path = fallback_path_to_goal((init.x, init.y, 0.0), (goal.x, goal.y, 0.0), step=step)
        return path, 13.9, [], True

    route = lane_graph.shortest_path(start_lane.id, goal_lane.id)
    if not route:
        path = fallback_path_to_goal((init.x, init.y, 0.0), (goal.x, goal.y, 0.0), step=step)
        return path, lane_graph.speed_limit_mps([start_lane.id]), [], True

    raw = lane_graph.route_centerline(route)
    if len(raw) < 2:
        path = fallback_path_to_goal((init.x, init.y, 0.0), (goal.x, goal.y, 0.0), step=step)
        return path, lane_graph.speed_limit_mps(route), route, True

    # Prepend the actual ego start so the planner doesn't have to chase to the centerline
    if math.hypot(raw[0][0] - init.x, raw[0][1] - init.y) > 0.5:
        raw = [(init.x, init.y, raw[0][2])] + raw
    if math.hypot(raw[-1][0] - goal.x, raw[-1][1] - goal.y) > 0.5:
        raw = raw + [(goal.x, goal.y, raw[-1][2])]

    return resample_polyline(raw, step=step), lane_graph.speed_limit_mps(route), route, False


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    scenario_dir = Path(args.scenario_dir)
    scenario = load_scenario(scenario_dir)
    if scenario.init_pose is None or scenario.goal_pose is None:
        print(f"[sim] scenario_meta.json missing init or goal pose: {scenario_dir}", file=sys.stderr)
        return 2

    lane_graph = LaneGraph.load(scenario.lane_graph_path)
    print(
        f"[sim] scenario {scenario.scenario_id} loaded: "
        f"{len(scenario.tracks)} tracks, {len(lane_graph.lanes)} lanes, "
        f"{scenario.num_steps} timestamps"
    )

    ref_path, speed_limit_mps, route_ids, fallback = _build_route(
        scenario, lane_graph, scenario.init_pose, scenario.goal_pose, args.reference_step
    )
    if fallback:
        print(f"[sim] WARNING: route via lane graph failed -> using straight-line fallback ({len(ref_path)} pts)")
    else:
        print(f"[sim] route: {len(route_ids)} lanes, {len(ref_path)} points, limit={speed_limit_mps*3.6:.1f} km/h")

    # ego construction
    ego_params = VehicleParams(
        length=args.ego_length,
        width=args.ego_width,
        wheelbase=args.ego_wheelbase,
        rear_overhang=args.ego_rear_overhang,
        max_speed=args.ego_max_speed,
    )
    init_state = VehicleState(x=scenario.init_pose.x, y=scenario.init_pose.y, heading=scenario.init_pose.yaw)
    ego = BicycleVehicle(state=init_state, params=ego_params)

    planner_cls = PLANNERS[args.planner]
    planner_kwargs = {}
    if args.planner == "idm_pure_pursuit":
        from waymo_sim.planners.idm_pure_pursuit import IDMParams, PurePursuitParams

        planner_kwargs.update(
            idm=IDMParams(desired_speed=args.desired_speed),
            pp=PurePursuitParams(wheelbase=args.ego_wheelbase),
        )
    elif args.planner == "frenet_optimal":
        planner_kwargs.update(
            wheelbase=args.ego_wheelbase,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            desired_speed_mps=min(args.desired_speed, speed_limit_mps),
        )
    planner = planner_cls(reference_path=ref_path, speed_limit_mps=speed_limit_mps, **planner_kwargs)
    print(f"[sim] planner = {planner.name}")

    cfg = WorldConfig(
        dt=args.dt,
        max_seconds=args.max_seconds,
        stop_on_first_collision=args.stop_on_collision,
        interpolate_npcs=not args.no_interpolate_npcs,
        vehicle_params=ego_params,
    )
    world = World(scenario=scenario, ego=ego, config=cfg)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = SimLogWriter(output_path=output_path, source=args.source_tag)
    grader: Optional[OnlineGrader] = None
    hooks = []
    if not args.no_python_grader:
        grader = OnlineGrader(
            max_speed_mps=args.ego_max_speed,
            max_desired_speed_mps=args.ego_max_speed,
            print_every=args.print_every,
        )
        hooks.append(grader)
    hooks.append(writer)

    if args.no_python_grader and not args.grading_bin:
        print(
            "[sim] note: --no-python-grader with no --grading-bin: only SimLog will be written.",
            file=sys.stderr,
        )

    cpp_online: Optional[CppOnlineGrader] = None
    bin_path: Optional[Path] = None
    report_path: Optional[Path] = None
    if args.grading_bin and args.cpp_mode != "off":
        bin_path = Path(args.grading_bin).expanduser().resolve()
        if not bin_path.is_file():
            print(f"[sim] grading_main not found: {bin_path}", file=sys.stderr)
            return 3
        report_path = Path(args.grading_report).expanduser().resolve() if args.grading_report else (
            DEFAULT_GRADING_REPORT_PATH
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if args.cpp_mode in ("online", "both"):
            print(f"[sim] starting C++ scorer in stream mode: {bin_path} -> {report_path}")
            cpp_online = CppOnlineGrader(binary_path=bin_path, report_path=report_path)
            hooks.append(cpp_online)

    records = world.run(planner=planner, hooks=hooks)
    if cpp_online is not None:
        cpp_online.close()
    print(f"[sim] wrote {output_path} ({len(records)} frames)")

    if bin_path is not None and args.cpp_mode in ("offline", "both"):
        # In "both" mode the offline run overwrites the same report path so the
        # final artefact is the deterministic batch result.
        print(f"[sim] running C++ scorer in batch mode: {bin_path} {output_path} -> {report_path}")
        proc = subprocess.run(
            [str(bin_path), str(output_path), str(report_path)],
            check=False,
        )
        if proc.returncode != 0:
            print(f"[sim] grading_main exited with code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    if args.no_python_grader:
        return 0
    assert grader is not None
    return 0 if grader.overall_passed else 10


if __name__ == "__main__":
    sys.exit(main())
