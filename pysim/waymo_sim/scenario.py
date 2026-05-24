"""Load a Waymo-derived scenario directory produced by tools/waymo_to_lanelet2.py.

Required files inside `scenario_dir`:
  scenario_meta.json       init / goal / world_offset
  dynamic_objects.json     per-frame NPC tracks
  lane_graph.json          routable lane topology
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


def load_scenario(scenario_dir: Path) -> Scenario:
    scenario_dir = Path(scenario_dir).expanduser().resolve()
    meta_path = scenario_dir / "scenario_meta.json"
    objs_path = scenario_dir / "dynamic_objects.json"
    graph_path = scenario_dir / "lane_graph.json"
    for p in (meta_path, objs_path, graph_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

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


def scenario_dt(scenario: Scenario) -> float:
    """Native scenario timestep (seconds). Defaults to 0.1 if unknown."""
    ts = scenario.timestamps_seconds
    if len(ts) < 2:
        return 0.1
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
    if not diffs:
        return 0.1
    return float(sum(diffs) / len(diffs))
