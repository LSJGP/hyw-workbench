#!/usr/bin/env python3
"""Split dynamic_objects.json / dynamic_objects.pb into stream layout.

Output under <scenario_dir>/dynamic_objects/:
  header.json / header.pb
  sdc_states.json / sdc_states.pb (optional)
  frames/NNNNN.json / frames/NNNNN.pb (non-SDC only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def _load_dynamic_doc(scenario_dir: Path) -> Dict[str, Any]:
    pb = scenario_dir / "dynamic_objects.pb"
    js = scenario_dir / "dynamic_objects.json"

    if pb.is_file():
        from proto.sim import scenario_pb2

        dyn = scenario_pb2.DynamicObjects()
        dyn.ParseFromString(pb.read_bytes())
        return {
            "source": dyn.source,
            "scenario_id": dyn.scenario_id,
            "world_offset": {"x": dyn.world_offset.x, "y": dyn.world_offset.y, "z": dyn.world_offset.z},
            "timestamps_seconds": list(dyn.timestamps_seconds),
            "current_time_index": int(dyn.current_time_index),
            "sdc_track_index": int(dyn.sdc_track_index),
            "tracks": [
                {
                    "track_index": int(tr.track_index),
                    "id": int(tr.id),
                    "object_type": tr.object_type,
                    "is_sdc": bool(tr.is_sdc),
                    "states": [
                        {
                            "valid": st.valid,
                            "x": st.x,
                            "y": st.y,
                            "z": st.z,
                            "yaw": st.yaw,
                            "vx": st.vx,
                            "vy": st.vy,
                            "length": st.length,
                            "width": st.width,
                            "height": st.height,
                        }
                        for st in tr.states
                    ],
                }
                for tr in dyn.tracks
            ],
        }

    if js.is_file():
        with js.open(encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"missing dynamic_objects.{js.suffix} or {pb.suffix} in {scenario_dir}")


def _write_json_frames(
    out_base: Path,
    header: Dict[str, Any],
    sdc_track: Optional[Dict[str, Any]],
    timestamps: List[float],
    tracks: List[Dict[str, Any]],
    sdc_idx: int,
) -> None:
    frames_dir = out_base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with (out_base / "header.json").open("w", encoding="utf-8") as f:
        json.dump(header, f, indent=2)
        f.write("\n")

    if sdc_track is not None:
        with (out_base / "sdc_states.json").open("w", encoding="utf-8") as f:
            json.dump(sdc_track, f)
            f.write("\n")

    for fi, t in enumerate(timestamps):
        states_out: List[Dict[str, Any]] = []
        for tr in tracks:
            if tr.get("is_sdc") or int(tr.get("track_index", -1)) == sdc_idx:
                continue
            states = tr.get("states", [])
            if fi >= len(states):
                continue
            st = states[fi]
            entry: Dict[str, Any] = {
                "track_index": int(tr["track_index"]),
                "id": int(tr["id"]),
                "object_type": str(tr.get("object_type", "OTHER")),
            }
            if st.get("valid", False):
                entry.update(
                    {
                        "valid": True,
                        "x": float(st["x"]),
                        "y": float(st["y"]),
                        "z": float(st.get("z", 0.0)),
                        "yaw": float(st.get("yaw", 0.0)),
                        "vx": float(st.get("vx", 0.0)),
                        "vy": float(st.get("vy", 0.0)),
                        "length": float(st.get("length", 4.5)),
                        "width": float(st.get("width", 1.85)),
                        "height": float(st.get("height", 1.6)),
                    }
                )
            else:
                entry["valid"] = False
            states_out.append(entry)

        frame_doc = {"frame_index": fi, "timestamp": t, "states": states_out}
        with (frames_dir / f"{fi:05d}.json").open("w", encoding="utf-8") as f:
            json.dump(frame_doc, f)
            f.write("\n")


def _write_proto_frames(
    out_base: Path,
    header: Dict[str, Any],
    sdc_track: Optional[Dict[str, Any]],
    timestamps: List[float],
    tracks: List[Dict[str, Any]],
    sdc_idx: int,
) -> None:
    from hyw_proto_convert import (
        build_dynamic_frame,
        build_stream_header,
        track_from_json_doc,
        write_message_pb,
    )

    frames_dir = out_base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    write_message_pb(build_stream_header(header), out_base / "header.pb")

    if sdc_track is not None:
        write_message_pb(track_from_json_doc(sdc_track), out_base / "sdc_states.pb")

    for fi, t in enumerate(timestamps):
        states_out: List[Dict[str, Any]] = []
        for tr in tracks:
            if tr.get("is_sdc") or int(tr.get("track_index", -1)) == sdc_idx:
                continue
            states = tr.get("states", [])
            if fi >= len(states):
                continue
            st = states[fi]
            entry: Dict[str, Any] = {
                "track_index": int(tr["track_index"]),
                "id": int(tr["id"]),
                "object_type": str(tr.get("object_type", "OTHER")),
            }
            if st.get("valid", False):
                entry.update(st)
            else:
                entry["valid"] = False
            states_out.append(entry)

        frame_doc = {"frame_index": fi, "timestamp": t, "states": states_out}
        write_message_pb(build_dynamic_frame(frame_doc), frames_dir / f"{fi:05d}.pb")


def split_dynamic_objects(scenario_dir: Path, fmt: str = "json") -> None:
    scenario_dir = scenario_dir.expanduser().resolve()
    doc = _load_dynamic_doc(scenario_dir)

    timestamps: List[float] = list(map(float, doc.get("timestamps_seconds", [])))
    if not timestamps:
        raise ValueError("timestamps_seconds is empty")

    tracks: List[Dict[str, Any]] = doc.get("tracks", [])
    sdc_idx = int(doc.get("sdc_track_index", -1))

    out_base = scenario_dir / "dynamic_objects"
    out_base.mkdir(parents=True, exist_ok=True)

    track_meta: List[Dict[str, Any]] = []
    sdc_track: Optional[Dict[str, Any]] = None

    for tr in tracks:
        meta = {
            "track_index": int(tr["track_index"]),
            "id": int(tr["id"]),
            "object_type": str(tr.get("object_type", "OTHER")),
            "is_sdc": bool(tr.get("is_sdc", False)),
        }
        track_meta.append(meta)
        if meta["is_sdc"] or meta["track_index"] == sdc_idx:
            sdc_track = {
                "track_index": meta["track_index"],
                "id": meta["id"],
                "object_type": meta["object_type"],
                "is_sdc": True,
                "states": tr.get("states", []),
            }

    header = {
        "source": doc.get("source", ""),
        "scenario_id": doc.get("scenario_id", ""),
        "world_offset": doc.get("world_offset", {"x": 0, "y": 0, "z": 0}),
        "timestamps_seconds": timestamps,
        "current_time_index": int(doc.get("current_time_index", 0)),
        "sdc_track_index": sdc_idx,
        "tracks": track_meta,
    }

    if fmt in ("json", "both"):
        _write_json_frames(out_base, header, sdc_track, timestamps, tracks, sdc_idx)
    if fmt in ("proto", "both"):
        try:
            import hyw_proto_convert  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "hyw_proto_convert import failed; run: bash tools/gen_sim_protos.sh"
            ) from None
        _write_proto_frames(out_base, header, sdc_track, timestamps, tracks, sdc_idx)

    print(
        f"[split] {scenario_dir.name}: {len(timestamps)} frames, {len(track_meta)} tracks -> {out_base} (format={fmt})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "scenario_dirs",
        nargs="+",
        help="Scenario directories containing dynamic_objects.json or .pb",
    )
    p.add_argument(
        "--format",
        choices=("json", "proto", "both"),
        default="json",
        help="Output format for stream layout (default: json)",
    )
    args = p.parse_args()

    for d in args.scenario_dirs:
        try:
            split_dynamic_objects(Path(d), fmt=args.format)
        except (OSError, ValueError, json.JSONDecodeError, ImportError) as e:
            print(f"[split] failed {d}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

