"""Load a Waymo-derived scenario directory produced by tools/waymo_to_lanelet2.py.

Required files inside `scenario_dir`:
  JSON mode:
    scenario_meta.json       init / goal / world_offset
    dynamic_objects.json     per-frame NPC tracks
    lane_graph.json          routable lane topology
  Proto mode:
    scenario_meta.pb
    dynamic_objects.pb
    lane_graph.pb
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackState:
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.5
    width: float = 1.85
    height: float = 1.6


@dataclass
class Track:
    track_index: int
    id: int
    object_type: str  # VEHICLE / PEDESTRIAN / CYCLIST / OTHER
    is_sdc: bool
    states: List[TrackState] = field(default_factory=list)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Scenario:
    scenario_id: str
    world_offset: Tuple[float, float, float]
    init_pose: Optional[Pose2D]
    goal_pose: Optional[Pose2D]
    timestamps_seconds: List[float]
    current_time_index: int
    sdc_track_index: int
    tracks: List[Track]
    lane_graph_path: Path
    meta: Dict
    dynamic_objects_path: Path

    @property
    def num_steps(self) -> int:
        return len(self.timestamps_seconds)

    def npcs_at(self, t_idx: int) -> List[Track]:
        out: List[Track] = []
        for tr in self.tracks:
            if tr.is_sdc:
                continue
            if t_idx < 0 or t_idx >= len(tr.states):
                continue
            if not tr.states[t_idx].valid:
                continue
            out.append(tr)
        return out

    def sdc_track(self) -> Optional[Track]:
        for tr in self.tracks:
            if tr.is_sdc:
                return tr
        return None


def _pose(d: Optional[Dict]) -> Optional[Pose2D]:
    if not d:
        return None
    return Pose2D(x=float(d["x"]), y=float(d["y"]), yaw=float(d.get("yaw", 0.0)))


def _protobuf_runtime_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("google.protobuf") is not None
    except ModuleNotFoundError:
        return False


def _resolve_scenario_file(
    scenario_dir: Path, stem: str, input_format: str = "auto"
) -> Path:
    pb = scenario_dir / f"{stem}.pb"
    js = scenario_dir / f"{stem}.json"
    if input_format == "proto":
        return pb
    if input_format == "json":
        return js
    # auto: prefer protobuf when runtime + files exist, else json
    if pb.is_file() and _protobuf_runtime_available():
        return pb
    return js if js.is_file() else pb


def _load_scenario_from_json(meta_path: Path, objs_path: Path, graph_path: Path) -> Scenario:
    with open(meta_path) as f:
        meta = json.load(f)
    with open(objs_path) as f:
        objs = json.load(f)

    tracks_raw = objs.get("tracks", [])
    tracks: List[Track] = []
    for tr in tracks_raw:
        states = []
        for st in tr.get("states", []):
            if not st.get("valid", False):
                states.append(TrackState(valid=False))
            else:
                states.append(
                    TrackState(
                        valid=True,
                        x=float(st["x"]),
                        y=float(st["y"]),
                        z=float(st.get("z", 0.0)),
                        yaw=float(st.get("yaw", 0.0)),
                        vx=float(st.get("vx", 0.0)),
                        vy=float(st.get("vy", 0.0)),
                        length=float(st.get("length", 4.5)),
                        width=float(st.get("width", 1.85)),
                        height=float(st.get("height", 1.6)),
                    )
                )
        tracks.append(
            Track(
                track_index=int(tr["track_index"]),
                id=int(tr["id"]),
                object_type=str(tr.get("object_type", "OTHER")),
                is_sdc=bool(tr.get("is_sdc", False)),
                states=states,
            )
        )

    wo = meta.get("world_offset", {})
    return Scenario(
        scenario_id=str(meta.get("scenario_id", "")),
        world_offset=(float(wo.get("x", 0.0)), float(wo.get("y", 0.0)), float(wo.get("z", 0.0))),
        init_pose=_pose(meta.get("init_pose")),
        goal_pose=_pose(meta.get("goal_pose")),
        timestamps_seconds=list(map(float, objs.get("timestamps_seconds", []))),
        current_time_index=int(objs.get("current_time_index", 0)),
        sdc_track_index=int(objs.get("sdc_track_index", -1)),
        tracks=tracks,
        lane_graph_path=graph_path,
        meta=meta,
        dynamic_objects_path=objs_path,
    )


def _load_scenario_from_proto(meta_path: Path, objs_path: Path, graph_path: Path) -> Scenario:
    from proto.sim import scenario_pb2  # type: ignore

    meta_pb = scenario_pb2.ScenarioMeta()
    dyn_pb = scenario_pb2.DynamicObjects()
    meta_pb.ParseFromString(meta_path.read_bytes())
    dyn_pb.ParseFromString(objs_path.read_bytes())

    tracks: List[Track] = []
    for tr in dyn_pb.tracks:
        states: List[TrackState] = []
        for st in tr.states:
            if not st.valid:
                states.append(TrackState(valid=False))
                continue
            states.append(
                TrackState(
                    valid=True,
                    x=float(st.x),
                    y=float(st.y),
                    z=float(st.z),
                    yaw=float(st.yaw),
                    vx=float(st.vx),
                    vy=float(st.vy),
                    length=float(st.length) if st.length > 0 else 4.5,
                    width=float(st.width) if st.width > 0 else 1.85,
                    height=float(st.height) if st.height > 0 else 1.6,
                )
            )
        tracks.append(
            Track(
                track_index=int(tr.track_index),
                id=int(tr.id),
                object_type=str(tr.object_type or "OTHER"),
                is_sdc=bool(tr.is_sdc),
                states=states,
            )
        )

    return Scenario(
        scenario_id=str(meta_pb.scenario_id),
        world_offset=(
            float(meta_pb.world_offset.x),
            float(meta_pb.world_offset.y),
            float(meta_pb.world_offset.z),
        ),
        init_pose=Pose2D(
            x=float(meta_pb.init_pose.x),
            y=float(meta_pb.init_pose.y),
            yaw=float(meta_pb.init_pose.yaw),
        ),
        goal_pose=Pose2D(
            x=float(meta_pb.goal_pose.x),
            y=float(meta_pb.goal_pose.y),
            yaw=float(meta_pb.goal_pose.yaw),
        ),
        timestamps_seconds=[float(x) for x in dyn_pb.timestamps_seconds],
        current_time_index=int(dyn_pb.current_time_index),
        sdc_track_index=int(dyn_pb.sdc_track_index),
        tracks=tracks,
        lane_graph_path=graph_path,
        meta={
            "source": meta_pb.source,
            "scenario_id": meta_pb.scenario_id,
            "scenario_index": int(meta_pb.scenario_index),
        },
        dynamic_objects_path=objs_path,
    )


def load_scenario(scenario_dir: Path, input_format: str = "auto") -> Scenario:
    scenario_dir = Path(scenario_dir).expanduser().resolve()
    meta_path = _resolve_scenario_file(scenario_dir, "scenario_meta", input_format)
    objs_path = _resolve_scenario_file(scenario_dir, "dynamic_objects", input_format)
    graph_path = _resolve_scenario_file(scenario_dir, "lane_graph", input_format)
    for p in (meta_path, objs_path, graph_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")
    if meta_path.suffix == ".pb" or objs_path.suffix == ".pb":
        return _load_scenario_from_proto(meta_path, objs_path, graph_path)
    return _load_scenario_from_json(meta_path, objs_path, graph_path)


def scenario_dt(scenario: Scenario) -> float:
    """Native scenario timestep (seconds). Defaults to 0.1 if unknown."""
    ts = scenario.timestamps_seconds
    if len(ts) < 2:
        return 0.1
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
    if not diffs:
        return 0.1
    return float(sum(diffs) / len(diffs))


_SCENARIO_TRIPLET_STEMS = ("scenario_meta", "dynamic_objects", "lane_graph")


def _has_scenario_triplet(scenario_dir: Path, fmt: str) -> bool:
    ext = "pb" if fmt == "proto" else "json"
    return all((scenario_dir / f"{stem}.{ext}").is_file() for stem in _SCENARIO_TRIPLET_STEMS)


def _map_route_available(scenario_dir: Path, fmt: str) -> Optional[bool]:
    """Return whether init→goal lane routing works for this on-disk triplet."""
    if not _has_scenario_triplet(scenario_dir, fmt):
        return None
    if fmt == "proto" and not _protobuf_runtime_available():
        return None
    try:
        from waymo_sim.lane_graph import LaneGraph

        scenario = load_scenario(scenario_dir, input_format=fmt)
        if scenario.init_pose is None or scenario.goal_pose is None:
            return False
        lane_graph = LaneGraph.load(scenario.lane_graph_path)
        start = lane_graph.closest_lane(
            scenario.init_pose.x,
            scenario.init_pose.y,
            heading=scenario.init_pose.yaw,
        )
        goal = lane_graph.closest_lane(
            scenario.goal_pose.x,
            scenario.goal_pose.y,
            heading=scenario.goal_pose.yaw,
        )
        if start is None or goal is None:
            return False
        return bool(lane_graph.shortest_path(start.id, goal.id))
    except Exception:
        return False


def resolve_scenario_input_format(
    scenario_dir: Path,
    input_format: str = "auto",
    *,
    reference_source: str = "map",
) -> str:
    """Pick json/proto triplet for sim + viz when ``auto`` would otherwise fail routing."""
    scenario_dir = Path(scenario_dir).expanduser().resolve()
    if input_format in ("json", "proto"):
        return input_format

    has_json = _has_scenario_triplet(scenario_dir, "json")
    has_pb = _has_scenario_triplet(scenario_dir, "proto")
    if has_json and not has_pb:
        return "json"
    if has_pb and not has_json:
        return "proto"
    if not has_json and not has_pb:
        return "auto"

    if reference_source != "map":
        return "proto" if has_pb else "json"

    json_ok = _map_route_available(scenario_dir, "json")
    pb_ok = _map_route_available(scenario_dir, "proto")
    if pb_ok:
        return "proto"
    if json_ok:
        return "json"
    return "json" if has_json else "proto"
