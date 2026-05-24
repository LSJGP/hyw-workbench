#!/usr/bin/env python3
"""Convert one Waymo Motion Scenario into the 3 JSONs the hywgrading sim eats.

Outputs (under --out-dir):
  scenario_meta.json     init / goal / world_offset / scenario_id / 统计信息
  dynamic_objects.json   每个 track 的逐帧状态（车 / 人 / 自行车 / SDC）
  lane_graph.json        静态地图全集（lanes / road_lines / road_edges /
                                       crosswalks / stop_signs / driveways /
                                       speed_bumps；坐标已减 world_offset）

By default the SDC's first valid pose is anchored at (0, 0, 0); all map
features and all NPC tracks are translated by the same world_offset, so the
sim works in a local SDC-centric frame.

需要在 conda 环境里跑（含 tensorflow + waymo-open-dataset），见 run_converter.sh。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_SPEED_LIMIT_KMH = 50.0


# ---------------- Waymo enum mappings ----------------

WAYMO_OBJECT_TYPE = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER",
}

WAYMO_LANE_TYPE = {
    0: "UNDEFINED",
    1: "FREEWAY",
    2: "SURFACE_STREET",
    3: "BIKE_LANE",
}

WAYMO_ROAD_LINE_TYPE = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}

WAYMO_ROAD_EDGE_TYPE = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    2: "TYPE_ROAD_EDGE_MEDIAN",
}


# ---------------- helpers ----------------

def _polyline_to_xyz(polyline) -> List[Tuple[float, float, float]]:
    return [(float(p.x), float(p.y), float(getattr(p, "z", 0.0))) for p in polyline]


def _offset_xyz(x: float, y: float, z: float, ox: float, oy: float, oz: float) -> List[float]:
    return [x - ox, y - oy, z - oz]


def _offset_polyline(
    poly: List[Tuple[float, float, float]], ox: float, oy: float, oz: float
) -> List[List[float]]:
    return [_offset_xyz(x, y, z, ox, oy, oz) for x, y, z in poly]


def _offset_point3(p, ox: float, oy: float, oz: float) -> Dict[str, float]:
    return {
        "x": float(p.x) - ox,
        "y": float(p.y) - oy,
        "z": float(getattr(p, "z", 0.0)) - oz,
    }


def _boundary_segments(segments) -> List[Dict]:
    out: List[Dict] = []
    for b in segments:
        out.append(
            {
                "lane_start_index": int(b.lane_start_index),
                "lane_end_index": int(b.lane_end_index),
                "boundary_feature_id": int(b.boundary_feature_id),
                "boundary_type": WAYMO_ROAD_LINE_TYPE.get(
                    int(b.boundary_type), "TYPE_UNKNOWN"
                ),
            }
        )
    return out


def _lane_neighbors(neighbors) -> List[Dict]:
    out: List[Dict] = []
    for n in neighbors:
        out.append(
            {
                "feature_id": int(n.feature_id),
                "self_start_index": int(n.self_start_index),
                "self_end_index": int(n.self_end_index),
                "neighbor_start_index": int(n.neighbor_start_index),
                "neighbor_end_index": int(n.neighbor_end_index),
                "boundaries": _boundary_segments(n.boundaries),
            }
        )
    return out


def _lane_speed_kmh(ln) -> float:
    mph = float(getattr(ln, "speed_limit_mph", 0.0) or 0.0)
    if mph > 0.0:
        return mph * 1.609344
    return DEFAULT_SPEED_LIMIT_KMH


# ---------------- core dataclass ----------------

@dataclass
class ConvertedScene:
    scenario_id: str
    world_offset: Tuple[float, float, float]
    init_pose: Optional[Tuple[float, float, float]]
    goal_pose: Optional[Tuple[float, float, float]]
    timestamps_seconds: List[float]
    current_time_index: int
    sdc_track_index: int
    tracks: List[Dict]
    track_type_counts: Dict[str, int]
    static_map: Dict[str, List[Dict]]
    bbox: Tuple[float, float, float, float]
    map_feature_counts: Dict[str, int]


# ---------------- extraction ----------------

def _extract_tracks(scenario, ox: float, oy: float, oz: float, sdc_idx: int):
    """Per-track 逐帧状态（应用 world_offset）。返回 (tracks, type_counts)."""
    tracks: List[Dict] = []
    counts: Dict[str, int] = {}
    for ti, tr in enumerate(scenario.tracks):
        type_name = WAYMO_OBJECT_TYPE.get(int(tr.object_type), "OTHER")
        states = []
        for st in tr.states:
            if not st.valid:
                states.append({"valid": False})
                continue
            states.append(
                {
                    "valid": True,
                    "x": float(st.center_x) - ox,
                    "y": float(st.center_y) - oy,
                    "z": float(getattr(st, "center_z", 0.0)) - oz,
                    "yaw": float(st.heading),
                    "vx": float(st.velocity_x),
                    "vy": float(st.velocity_y),
                    "length": float(st.length),
                    "width": float(st.width),
                    "height": float(st.height),
                }
            )
        is_sdc = ti == sdc_idx
        tracks.append(
            {
                "track_index": ti,
                "id": int(tr.id),
                "object_type": type_name,
                "is_sdc": is_sdc,
                "states": states,
            }
        )
        if not is_sdc:
            counts[type_name] = counts.get(type_name, 0) + 1
    return tracks, counts


def _extract_static_map(scenario, ox: float, oy: float, oz: float) -> Dict[str, List[Dict]]:
    """导出 Waymo 静态地图全部 7 类 map feature。"""
    out: Dict[str, List[Dict]] = {
        "lanes": [],
        "road_lines": [],
        "road_edges": [],
        "crosswalks": [],
        "stop_signs": [],
        "driveways": [],
        "speed_bumps": [],
    }
    for mf in scenario.map_features:
        which = mf.WhichOneof("feature_data")
        fid = int(mf.id)
        if which == "lane":
            ln = mf.lane
            poly = _polyline_to_xyz(ln.polyline)
            if len(poly) < 2:
                continue
            out["lanes"].append(
                {
                    "id": fid,
                    "type": WAYMO_LANE_TYPE.get(int(ln.type), "UNDEFINED"),
                    "speed_limit_kmh": _lane_speed_kmh(ln),
                    "interpolating": bool(ln.interpolating),
                    "centerline": _offset_polyline(poly, ox, oy, oz),
                    "entry_lanes": [int(x) for x in ln.entry_lanes],
                    "exit_lanes": [int(x) for x in ln.exit_lanes],
                    "left_boundaries": _boundary_segments(ln.left_boundaries),
                    "right_boundaries": _boundary_segments(ln.right_boundaries),
                    "left_neighbors": _lane_neighbors(ln.left_neighbors),
                    "right_neighbors": _lane_neighbors(ln.right_neighbors),
                }
            )
        elif which == "road_line":
            rl = mf.road_line
            poly = _polyline_to_xyz(rl.polyline)
            if len(poly) < 2:
                continue
            out["road_lines"].append(
                {
                    "id": fid,
                    "type": WAYMO_ROAD_LINE_TYPE.get(int(rl.type), "TYPE_UNKNOWN"),
                    "polyline": _offset_polyline(poly, ox, oy, oz),
                }
            )
        elif which == "road_edge":
            re = mf.road_edge
            poly = _polyline_to_xyz(re.polyline)
            if len(poly) < 2:
                continue
            out["road_edges"].append(
                {
                    "id": fid,
                    "type": WAYMO_ROAD_EDGE_TYPE.get(int(re.type), "TYPE_UNKNOWN"),
                    "polyline": _offset_polyline(poly, ox, oy, oz),
                }
            )
        elif which == "crosswalk":
            poly = _polyline_to_xyz(mf.crosswalk.polygon)
            if len(poly) < 3:
                continue
            out["crosswalks"].append(
                {"id": fid, "polygon": _offset_polyline(poly, ox, oy, oz)}
            )
        elif which == "stop_sign":
            ss = mf.stop_sign
            out["stop_signs"].append(
                {
                    "id": fid,
                    "position": _offset_point3(ss.position, ox, oy, oz),
                    "lanes": [int(x) for x in ss.lane],
                }
            )
        elif which == "driveway":
            poly = _polyline_to_xyz(mf.driveway.polygon)
            if len(poly) < 3:
                continue
            out["driveways"].append(
                {"id": fid, "polygon": _offset_polyline(poly, ox, oy, oz)}
            )
        elif which == "speed_bump":
            poly = _polyline_to_xyz(mf.speed_bump.polygon)
            if len(poly) < 3:
                continue
            out["speed_bumps"].append(
                {"id": fid, "polygon": _offset_polyline(poly, ox, oy, oz)}
            )
    return out


_MAP_KIND_KEYS = {
    "lane": "lanes",
    "road_line": "road_lines",
    "road_edge": "road_edges",
    "crosswalk": "crosswalks",
    "stop_sign": "stop_signs",
    "driveway": "driveways",
    "speed_bump": "speed_bumps",
}


def _scene_stats(scenario) -> Tuple[Dict[str, int], Tuple[float, float, float, float]]:
    """统计各 map feature 数量 + 整张图 XY bbox（局部坐标前，用全局坐标算 bbox）。"""
    counts: Dict[str, int] = {v: 0 for v in _MAP_KIND_KEYS.values()}
    xs: List[float] = []
    ys: List[float] = []
    for mf in scenario.map_features:
        which = mf.WhichOneof("feature_data")
        key = _MAP_KIND_KEYS.get(which or "")
        if not key:
            continue
        counts[key] += 1
        poly: List[Tuple[float, float, float]] = []
        if which == "lane":
            poly = _polyline_to_xyz(mf.lane.polyline)
        elif which == "road_line":
            poly = _polyline_to_xyz(mf.road_line.polyline)
        elif which == "road_edge":
            poly = _polyline_to_xyz(mf.road_edge.polyline)
        elif which in ("crosswalk", "driveway", "speed_bump"):
            poly = _polyline_to_xyz(getattr(mf, which).polygon)
        elif which == "stop_sign":
            p = mf.stop_sign.position
            xs.append(float(p.x))
            ys.append(float(p.y))
            continue
        for x, y, _ in poly:
            xs.append(x)
            ys.append(y)
    if not xs:
        bbox = (0.0, 0.0, 0.0, 0.0)
    else:
        bbox = (min(xs), min(ys), max(xs), max(ys))
    return counts, bbox


def convert_scenario(scenario, center_on_sdc: bool = True) -> ConvertedScene:
    """Top-level: 解一个 Waymo Scenario,  返回 ConvertedScene."""
    init_pose: Optional[Tuple[float, float, float]] = None
    goal_pose: Optional[Tuple[float, float, float]] = None
    ox = oy = oz = 0.0
    sdc_idx = -1

    try:
        sdc_idx = int(scenario.sdc_track_index)
        sdc = scenario.tracks[sdc_idx]
        valid = [s for s in sdc.states if s.valid]
        if valid:
            s0, s1 = valid[0], valid[-1]
            init_pose = (float(s0.center_x), float(s0.center_y), float(s0.heading))
            goal_pose = (float(s1.center_x), float(s1.center_y), float(s1.heading))
            if center_on_sdc:
                ox, oy = init_pose[0], init_pose[1]
                oz = float(getattr(s0, "center_z", 0.0))
    except (AttributeError, IndexError, ValueError):
        pass

    if init_pose is not None:
        init_pose = (init_pose[0] - ox, init_pose[1] - oy, init_pose[2])
    if goal_pose is not None:
        goal_pose = (goal_pose[0] - ox, goal_pose[1] - oy, goal_pose[2])

    map_counts, raw_bbox = _scene_stats(scenario)
    bbox = (raw_bbox[0] - ox, raw_bbox[1] - oy, raw_bbox[2] - ox, raw_bbox[3] - oy)

    tracks, counts = _extract_tracks(scenario, ox, oy, oz, sdc_idx)
    static_map = _extract_static_map(scenario, ox, oy, oz)

    sid = scenario.scenario_id if scenario.HasField("scenario_id") else "unknown"
    timestamps = list(scenario.timestamps_seconds)
    cti = int(scenario.current_time_index) if scenario.HasField("current_time_index") else 0

    return ConvertedScene(
        scenario_id=str(sid),
        world_offset=(ox, oy, oz),
        init_pose=init_pose,
        goal_pose=goal_pose,
        timestamps_seconds=timestamps,
        current_time_index=cti,
        sdc_track_index=sdc_idx,
        tracks=tracks,
        track_type_counts=counts,
        static_map=static_map,
        bbox=bbox,
        map_feature_counts=map_counts,
    )


# ---------------- writers ----------------

def _pose(p: Optional[Tuple[float, float, float]]):
    if p is None:
        return None
    return {"x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])}


def write_meta(scene: ConvertedScene, path: Path, source: str, scenario_index: int) -> None:
    duration = (
        float(scene.timestamps_seconds[-1] - scene.timestamps_seconds[0])
        if len(scene.timestamps_seconds) >= 2
        else 0.0
    )
    doc = {
        "source": source,
        "scenario_id": scene.scenario_id,
        "scenario_index": scenario_index,
        "world_offset": {
            "x": scene.world_offset[0],
            "y": scene.world_offset[1],
            "z": scene.world_offset[2],
        },
        "init_pose": _pose(scene.init_pose),
        "goal_pose": _pose(scene.goal_pose),
        "bbox": {
            "xmin": scene.bbox[0],
            "ymin": scene.bbox[1],
            "xmax": scene.bbox[2],
            "ymax": scene.bbox[3],
        },
        "stats": {
            **dict(scene.map_feature_counts),
            "timestamps": len(scene.timestamps_seconds),
            "duration_s": duration,
            "current_time_index": scene.current_time_index,
            "tracks_total": len(scene.tracks),
            "tracks_non_sdc_by_type": dict(scene.track_type_counts),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def write_dynamic_objects(scene: ConvertedScene, path: Path, source: str) -> None:
    doc = {
        "source": source,
        "scenario_id": scene.scenario_id,
        "world_offset": {
            "x": scene.world_offset[0],
            "y": scene.world_offset[1],
            "z": scene.world_offset[2],
        },
        "timestamps_seconds": scene.timestamps_seconds,
        "current_time_index": scene.current_time_index,
        "sdc_track_index": scene.sdc_track_index,
        "tracks": scene.tracks,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
        f.write("\n")


def write_lane_graph(scene: ConvertedScene, path: Path, source: str) -> None:
    doc = {
        "source": source,
        "scenario_id": scene.scenario_id,
        "world_offset": {
            "x": scene.world_offset[0],
            "y": scene.world_offset[1],
            "z": scene.world_offset[2],
        },
        "map_feature_counts": dict(scene.map_feature_counts),
        **scene.static_map,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
        f.write("\n")


# ---------------- driver ----------------

def _resolve_data_dir(explicit: str) -> str:
    explicit = (explicit or "").strip()
    if explicit and Path(explicit).is_dir():
        return str(Path(explicit).resolve())
    env = (os.environ.get("HYW_DATA_DIR") or os.environ.get("GRADING_DATA_DIR") or "").strip()
    if env and Path(env).is_dir():
        return str(Path(env).resolve())
    here = Path(__file__).resolve().parent.parent
    cand = here / "data"
    if cand.is_dir() and any(cand.glob("*.tfrecord*")):
        return str(cand.resolve())
    return ""


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tfrecord", default="", help="显式 TFRecord 路径")
    p.add_argument("--data-dir", default="", help="Waymo 数据目录（或设 HYW_DATA_DIR）")
    p.add_argument("--scenario-index", type=int, default=0, help="shard 内第几个 scenario")
    p.add_argument("--out-dir", required=True, help="输出目录（3 个 JSON 都写到这里）")
    p.add_argument(
        "--no-center", dest="center_on_sdc", action="store_false",
        help="保留 Waymo 原始全局坐标（默认是把 SDC 起点平移到原点）",
    )
    p.set_defaults(center_on_sdc=True)
    args = p.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as e:
        print(
            "需要 tensorflow + waymo-open-dataset。请用 run_converter.sh 在 conda 环境里跑。\n"
            f"原始错误: {e}",
            file=sys.stderr,
        )
        return 2

    if args.tfrecord:
        tf_path = Path(args.tfrecord).expanduser().resolve()
        if not tf_path.is_file():
            print(f"--tfrecord 不存在: {tf_path}", file=sys.stderr)
            return 2
    else:
        ddir = _resolve_data_dir(args.data_dir)
        if not ddir:
            print("找不到数据目录。请用 --data-dir 或设 HYW_DATA_DIR。", file=sys.stderr)
            return 2
        records = sorted(Path(ddir).glob("*.tfrecord*"))
        if not records:
            print(f"{ddir} 下没有 *.tfrecord*", file=sys.stderr)
            return 2
        tf_path = records[0]

    print(f"[converter] tfrecord = {tf_path}")
    print(f"[converter] scenario index = {args.scenario_index}")

    import tensorflow as tf  # type: ignore
    ds = tf.data.TFRecordDataset(str(tf_path), compression_type="")
    scenario = None
    for i, raw in enumerate(ds):
        if i < args.scenario_index:
            continue
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(raw.numpy()))
        break
    if scenario is None:
        print(f"scenario_index={args.scenario_index} 越界", file=sys.stderr)
        return 2

    scene = convert_scenario(scenario, center_on_sdc=args.center_on_sdc)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "scenario_meta.json"
    objs_path = out_dir / "dynamic_objects.json"
    graph_path = out_dir / "lane_graph.json"

    write_meta(scene, meta_path, source=str(tf_path), scenario_index=args.scenario_index)
    write_dynamic_objects(scene, objs_path, source=str(tf_path))
    write_lane_graph(scene, graph_path, source=str(tf_path))

    mc = scene.map_feature_counts
    print(
        f"[converter] static_map: lanes={mc.get('lanes', 0)} "
        f"road_lines={mc.get('road_lines', 0)} road_edges={mc.get('road_edges', 0)} "
        f"crosswalks={mc.get('crosswalks', 0)} stop_signs={mc.get('stop_signs', 0)} "
        f"driveways={mc.get('driveways', 0)} speed_bumps={mc.get('speed_bumps', 0)}"
    )
    print(f"[converter] tracks: total={len(scene.tracks)} "
          f"non_sdc_by_type={dict(scene.track_type_counts)} "
          f"timestamps={len(scene.timestamps_seconds)}")
    print(f"[converter] lane_graph.json: lanes exported={len(scene.static_map.get('lanes', []))}")
    print(f"[converter] init_pose = {scene.init_pose}")
    print(f"[converter] goal_pose = {scene.goal_pose}")
    print(f"[converter] wrote {meta_path}")
    print(f"[converter] wrote {objs_path}")
    print(f"[converter] wrote {graph_path}")


if __name__ == "__main__":
    sys.exit(main())
