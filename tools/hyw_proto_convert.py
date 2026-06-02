"""Convert ConvertedScene dicts to hyw_sim protobuf messages.

This module is reused by:
- tools/waymo_to_scenario.py (--write-proto / --split-dynamic-frames)
- tools/split_existing_dynamic_objects.py (JSON/.pb -> stream layout)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

_GEN = Path(__file__).resolve().parent / "gen"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

from proto.sim import common_pb2, map_pb2, scenario_pb2  # noqa: E402


def _vec3_from_list(pt: List[float]) -> common_pb2.Vec3:
    v = common_pb2.Vec3()
    v.x = float(pt[0]) if len(pt) > 0 else 0.0
    v.y = float(pt[1]) if len(pt) > 1 else 0.0
    v.z = float(pt[2]) if len(pt) > 2 else 0.0
    return v


def _fill_boundary_segments(segs: List[Dict[str, Any]], out) -> None:
    for b in segs:
        seg = out.add()
        seg.lane_start_index = int(b.get("lane_start_index", 0))
        seg.lane_end_index = int(b.get("lane_end_index", 0))
        seg.boundary_feature_id = int(b.get("boundary_feature_id", 0))
        seg.boundary_type = str(b.get("boundary_type", "TYPE_UNKNOWN"))


def _fill_lane_neighbors(neighbors: List[Dict[str, Any]], out) -> None:
    for n in neighbors:
        ln = out.add()
        ln.feature_id = int(n.get("feature_id", 0))
        ln.self_start_index = int(n.get("self_start_index", 0))
        ln.self_end_index = int(n.get("self_end_index", 0))
        ln.neighbor_start_index = int(n.get("neighbor_start_index", 0))
        ln.neighbor_end_index = int(n.get("neighbor_end_index", 0))
        _fill_boundary_segments(n.get("boundaries", []), ln.boundaries)


def _track_state_from_dict(st: Dict[str, Any]) -> scenario_pb2.TrackState:
    ts = scenario_pb2.TrackState()
    if not st.get("valid", False):
        ts.valid = False
        return ts
    ts.valid = True
    ts.x = float(st.get("x", 0.0))
    ts.y = float(st.get("y", 0.0))
    ts.z = float(st.get("z", 0.0))
    ts.yaw = float(st.get("yaw", 0.0))
    ts.vx = float(st.get("vx", 0.0))
    ts.vy = float(st.get("vy", 0.0))
    ts.length = float(st.get("length", 4.5))
    ts.width = float(st.get("width", 1.85))
    ts.height = float(st.get("height", 1.6))
    return ts


def scene_to_scenario_meta(
    scene: Any, source: str, scenario_index: int
) -> scenario_pb2.ScenarioMeta:
    duration = (
        float(scene.timestamps_seconds[-1] - scene.timestamps_seconds[0])
        if len(scene.timestamps_seconds) >= 2
        else 0.0
    )

    meta = scenario_pb2.ScenarioMeta()
    meta.source = source
    meta.scenario_id = scene.scenario_id
    meta.scenario_index = int(scenario_index)
    meta.world_offset.x = float(scene.world_offset[0])
    meta.world_offset.y = float(scene.world_offset[1])
    meta.world_offset.z = float(scene.world_offset[2])

    if scene.init_pose is not None:
        meta.init_pose.x = float(scene.init_pose[0])
        meta.init_pose.y = float(scene.init_pose[1])
        meta.init_pose.yaw = float(scene.init_pose[2])
    if scene.goal_pose is not None:
        meta.goal_pose.x = float(scene.goal_pose[0])
        meta.goal_pose.y = float(scene.goal_pose[1])
        meta.goal_pose.yaw = float(scene.goal_pose[2])

    meta.bbox.xmin = float(scene.bbox[0])
    meta.bbox.ymin = float(scene.bbox[1])
    meta.bbox.xmax = float(scene.bbox[2])
    meta.bbox.ymax = float(scene.bbox[3])

    stats = meta.stats
    for k, v in scene.map_feature_counts.items():
        if hasattr(stats, k):
            setattr(stats, k, int(v))
    stats.timestamps = len(scene.timestamps_seconds)
    stats.duration_s = duration
    stats.current_time_index = int(scene.current_time_index)
    stats.tracks_total = len(scene.tracks)
    for t, c in scene.track_type_counts.items():
        stats.tracks_non_sdc_by_type[t] = int(c)
    return meta


def scene_to_dynamic_objects(scene: Any, source: str) -> scenario_pb2.DynamicObjects:
    dyn = scenario_pb2.DynamicObjects()
    dyn.source = source
    dyn.scenario_id = scene.scenario_id
    dyn.world_offset.x = float(scene.world_offset[0])
    dyn.world_offset.y = float(scene.world_offset[1])
    dyn.world_offset.z = float(scene.world_offset[2])

    dyn.timestamps_seconds.extend(float(t) for t in scene.timestamps_seconds)
    dyn.current_time_index = int(scene.current_time_index)
    dyn.sdc_track_index = int(scene.sdc_track_index)

    for tr in scene.tracks:
        track = dyn.tracks.add()
        track.track_index = int(tr["track_index"])
        track.id = int(tr["id"])
        track.object_type = str(tr.get("object_type", "OTHER"))
        track.is_sdc = bool(tr.get("is_sdc", False))
        for st in tr.get("states", []):
            track.states.append(_track_state_from_dict(st))
    return dyn


def scene_to_static_map(scene: Any, source: str) -> map_pb2.StaticMap:
    sm = map_pb2.StaticMap()
    sm.source = source
    sm.scenario_id = scene.scenario_id
    sm.world_offset.x = float(scene.world_offset[0])
    sm.world_offset.y = float(scene.world_offset[1])
    sm.world_offset.z = float(scene.world_offset[2])

    counts = sm.map_feature_counts
    for k, v in scene.map_feature_counts.items():
        if hasattr(counts, k):
            setattr(counts, k, int(v))

    static = scene.static_map

    for ln in static.get("lanes", []):
        lane = sm.lanes.add()
        lane.id = int(ln["id"])
        lane.type = str(ln.get("type", "UNDEFINED"))
        lane.speed_limit_kmh = float(ln.get("speed_limit_kmh", 50.0))
        lane.interpolating = bool(ln.get("interpolating", False))
        for pt in ln.get("centerline", []):
            lane.centerline.append(_vec3_from_list(pt))
        lane.entry_lanes.extend(int(x) for x in ln.get("entry_lanes", []))
        lane.exit_lanes.extend(int(x) for x in ln.get("exit_lanes", []))
        _fill_boundary_segments(ln.get("left_boundaries", []), lane.left_boundaries)
        _fill_boundary_segments(ln.get("right_boundaries", []), lane.right_boundaries)
        _fill_lane_neighbors(ln.get("left_neighbors", []), lane.left_neighbors)
        _fill_lane_neighbors(ln.get("right_neighbors", []), lane.right_neighbors)

    for rl in static.get("road_lines", []):
        r = sm.road_lines.add()
        r.id = int(rl["id"])
        r.type = str(rl.get("type", "TYPE_UNKNOWN"))
        for pt in rl.get("polyline", []):
            r.polyline.append(_vec3_from_list(pt))

    for re in static.get("road_edges", []):
        r = sm.road_edges.add()
        r.id = int(re["id"])
        r.type = str(re.get("type", "TYPE_UNKNOWN"))
        for pt in re.get("polyline", []):
            r.polyline.append(_vec3_from_list(pt))

    for cw in static.get("crosswalks", []):
        c = sm.crosswalks.add()
        c.id = int(cw["id"])
        for pt in cw.get("polygon", []):
            c.polygon.append(_vec3_from_list(pt))

    for ss in static.get("stop_signs", []):
        s = sm.stop_signs.add()
        s.id = int(ss["id"])
        pos = ss.get("position", {})
        s.position.x = float(pos.get("x", 0.0))
        s.position.y = float(pos.get("y", 0.0))
        s.position.z = float(pos.get("z", 0.0))
        s.lanes.extend(int(x) for x in ss.get("lanes", []))

    for dw in static.get("driveways", []):
        d = sm.driveways.add()
        d.id = int(dw["id"])
        for pt in dw.get("polygon", []):
            d.polygon.append(_vec3_from_list(pt))

    for sb in static.get("speed_bumps", []):
        b = sm.speed_bumps.add()
        b.id = int(sb["id"])
        for pt in sb.get("polygon", []):
            b.polygon.append(_vec3_from_list(pt))

    return sm


def write_message_pb(msg, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(msg.SerializeToString())


def dynamic_objects_from_json_doc(doc: Dict[str, Any]) -> scenario_pb2.DynamicObjects:
    dyn = scenario_pb2.DynamicObjects()
    dyn.source = str(doc.get("source", ""))
    dyn.scenario_id = str(doc.get("scenario_id", ""))
    wo = doc.get("world_offset", {})
    dyn.world_offset.x = float(wo.get("x", 0.0))
    dyn.world_offset.y = float(wo.get("y", 0.0))
    dyn.world_offset.z = float(wo.get("z", 0.0))

    dyn.timestamps_seconds.extend(float(t) for t in doc.get("timestamps_seconds", []))
    dyn.current_time_index = int(doc.get("current_time_index", 0))
    dyn.sdc_track_index = int(doc.get("sdc_track_index", -1))

    for tr in doc.get("tracks", []):
        track = dyn.tracks.add()
        track.track_index = int(tr["track_index"])
        track.id = int(tr["id"])
        track.object_type = str(tr.get("object_type", "OTHER"))
        track.is_sdc = bool(tr.get("is_sdc", False))
        for st in tr.get("states", []):
            track.states.append(_track_state_from_dict(st))
    return dyn


def track_from_json_doc(doc: Dict[str, Any]) -> scenario_pb2.Track:
    tr = scenario_pb2.Track()
    tr.track_index = int(doc.get("track_index", 0))
    tr.id = int(doc.get("id", 0))
    tr.object_type = str(doc.get("object_type", "OTHER"))
    tr.is_sdc = bool(doc.get("is_sdc", False))
    for st in doc.get("states", []):
        tr.states.append(_track_state_from_dict(st))
    return tr


def build_stream_header(doc: Dict[str, Any]) -> scenario_pb2.StreamDynamicHeader:
    hdr = scenario_pb2.StreamDynamicHeader()
    hdr.source = str(doc.get("source", ""))
    hdr.scenario_id = str(doc.get("scenario_id", ""))
    wo = doc.get("world_offset", {})
    hdr.world_offset.x = float(wo.get("x", 0.0))
    hdr.world_offset.y = float(wo.get("y", 0.0))
    hdr.world_offset.z = float(wo.get("z", 0.0))
    hdr.timestamps_seconds.extend(float(t) for t in doc.get("timestamps_seconds", []))
    hdr.current_time_index = int(doc.get("current_time_index", 0))
    hdr.sdc_track_index = int(doc.get("sdc_track_index", -1))

    for tm in doc.get("tracks", []):
        m = hdr.tracks.add()
        m.track_index = int(tm["track_index"])
        m.id = int(tm["id"])
        m.object_type = str(tm.get("object_type", "OTHER"))
        m.is_sdc = bool(tm.get("is_sdc", False))
    return hdr


def build_dynamic_frame(frame_doc: Dict[str, Any]) -> scenario_pb2.DynamicFrame:
    fr = scenario_pb2.DynamicFrame()
    fr.frame_index = int(frame_doc.get("frame_index", 0))
    fr.timestamp = float(frame_doc.get("timestamp", 0.0))

    for entry in frame_doc.get("states", []):
        fn = fr.states.add()
        fn.track_index = int(entry.get("track_index", 0))
        fn.id = int(entry.get("id", 0))
        fn.object_type = str(entry.get("object_type", "OTHER"))
        if entry.get("valid", False):
            fn.state.CopyFrom(_track_state_from_dict(entry))
        else:
            fn.state.valid = False
    return fr

