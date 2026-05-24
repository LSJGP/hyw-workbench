#!/usr/bin/env python3
"""列出 Waymo TFRecord shard 里前 N 个 scenario，附带 SDC 行驶距离 / 帧数 / 车道数。

挑场景用：travel_m 太小（比如 < 5 米）的 scenario 做闭环没什么意义，会一开始
就算"到目标点"。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfrecord", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as e:
        print(f"需要 conda 环境 (TensorFlow + waymo-open-dataset): {e}", file=sys.stderr)
        return 2

    if args.tfrecord:
        tf_path = Path(args.tfrecord).expanduser().resolve()
    else:
        d = _resolve_data_dir(args.data_dir)
        if not d:
            print("没找到数据目录；用 --tfrecord 或 --data-dir 显式指定", file=sys.stderr)
            return 2
        records = sorted(Path(d).glob("*.tfrecord*"))
        if not records:
            print(f"{d} 下没有 *.tfrecord*", file=sys.stderr)
            return 2
        tf_path = records[0]

    print(f"# tfrecord: {tf_path}")
    print(f"{'idx':>4} {'scenario_id':>20} {'frames':>6} {'travel_m':>9} {'lanes':>5}")

    import tensorflow as tf  # type: ignore
    ds = tf.data.TFRecordDataset(str(tf_path), compression_type="")
    for i, raw in enumerate(ds):
        if i >= args.limit:
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
        print(f"{i:>4} {sid:>20} {n_frames:>6} {travel:>9.1f} {n_lanes:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
