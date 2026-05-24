#!/usr/bin/env python3
"""Parse the first N Waymo scenarios from ``data/*.tfrecord*`` into ``scenarios/``.

Uses ``waymo_to_scenario.py`` (conda env ``waymo_env`` by default).

Example:
  ./tools/parse_data_scenarios.py --limit 10
  ./tools/parse_data_scenarios.py --limit 10 --list-only
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKBENCH_ROOT))
from hyw_paths import SCENARIOS_DIR, WORKBENCH_ROOT  # noqa: E402

CONVERTER = WORKBENCH_ROOT / "tools" / "waymo_to_scenario.py"
DEFAULT_CONDA_PYTHON = Path.home() / "miniconda3/envs/waymo_env/bin/python"


def _resolve_data_dir(explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"data-dir not found: {p}")
        return p
    env = (os.environ.get("HYW_DATA_DIR") or os.environ.get("GRADING_DATA_DIR") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    d = WORKBENCH_ROOT / "data"
    if d.is_dir():
        return d
    raise FileNotFoundError("No data directory; pass --data-dir")


def _resolve_tfrecord(data_dir: Path, explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"tfrecord not found: {p}")
        return p
    records = sorted(data_dir.glob("*.tfrecord*"))
    if not records:
        raise FileNotFoundError(f"No *.tfrecord* under {data_dir}")
    return records[0]


def _pick_python(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_CONDA_PYTHON.is_file():
        return DEFAULT_CONDA_PYTHON
    return Path(sys.executable)


def list_scenarios(tf_path: Path, limit: int) -> list[dict]:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    rows: list[dict] = []
    ds = tf.data.TFRecordDataset(str(tf_path), compression_type="")
    for i, raw in enumerate(ds):
        if i >= limit:
            break
        s = scenario_pb2.Scenario()
        s.ParseFromString(bytes(raw.numpy()))
        sid = s.scenario_id if s.HasField("scenario_id") else "?"
        n_lanes = sum(1 for mf in s.map_features if mf.WhichOneof("feature_data") == "lane")
        try:
            sdc = s.tracks[int(s.sdc_track_index)]
            valid = [st for st in sdc.states if st.valid]
            n_frames = len(valid)
            travel = 0.0
            for a, b in zip(valid, valid[1:]):
                travel += math.hypot(b.center_x - a.center_x, b.center_y - a.center_y)
        except Exception:
            n_frames = 0
            travel = 0.0
        rows.append(
            {
                "index": i,
                "scenario_id": sid,
                "frames": n_frames,
                "travel_m": travel,
                "lanes": n_lanes,
            }
        )
    return rows


def convert_one(python: Path, tf_path: Path, scenario_index: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        str(CONVERTER),
        "--tfrecord",
        str(tf_path),
        "--scenario-index",
        str(scenario_index),
        "--out-dir",
        str(out_dir),
    ]
    print(f"[parse] idx={scenario_index} -> {out_dir}")
    subprocess.run(cmd, check=True, cwd=str(WORKBENCH_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="", help="Directory with *.tfrecord* (default: repo data/)")
    p.add_argument("--tfrecord", default="", help="Explicit TFRecord path")
    p.add_argument("--limit", type=int, default=10, help="Number of scenarios from shard start")
    p.add_argument("--start-index", type=int, default=0, help="First scenario index in shard")
    p.add_argument(
        "--out-prefix",
        default="waymo_scenario",
        help="Output dir: scenarios/<prefix>_<index>/",
    )
    p.add_argument("--python", default="", help="Python with tensorflow + waymo-open-dataset")
    p.add_argument("--list-only", action="store_true", help="Only list, do not convert")
    args = p.parse_args()

    try:
        import tensorflow  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    except ImportError as e:
        print(f"Need waymo_env (tensorflow + waymo-open-dataset): {e}", file=sys.stderr)
        return 2

    python = _pick_python(args.python)
    data_dir = _resolve_data_dir(args.data_dir)
    tf_path = _resolve_tfrecord(data_dir, args.tfrecord)

    end = args.start_index + args.limit
    rows = list_scenarios(tf_path, end)
    rows = [r for r in rows if r["index"] >= args.start_index]

    print(f"# tfrecord: {tf_path}")
    print(f"{'idx':>4} {'scenario_id':>20} {'frames':>6} {'travel_m':>9} {'lanes':>5}  out_dir")
    scenarios_root = SCENARIOS_DIR
    for r in rows:
        out_name = f"{args.out_prefix}_{r['index']}"
        out_dir = scenarios_root / out_name
        print(
            f"{r['index']:>4} {r['scenario_id']:>20} {r['frames']:>6} "
            f"{r['travel_m']:>9.1f} {r['lanes']:>5}  {out_dir.name}"
        )

    if args.list_only:
        return 0

    for r in rows:
        out_dir = scenarios_root / f"{args.out_prefix}_{r['index']}"
        convert_one(python, tf_path, r["index"], out_dir)

    print(f"\n[parse] done: {len(rows)} scenarios under {scenarios_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
