#!/usr/bin/env python3
"""Scan Waymo TFRecord shards and save only complex scenarios to ``scenarios/``.

Complexity is defined by interpretable rules (not ranking):
  - SDC travel distance above a minimum (simulation-worthy)
  - AND at least one of: long travel, passes intersection, turn at intersection

Uses ``waymo_to_scenario.py`` for conversion (conda env ``waymo_env`` by default).

Example:
  ./tools/filter_complex_scenarios.py --dry-run --limit 200
  ./tools/filter_complex_scenarios.py --require-intersection --limit 500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKBENCH_ROOT))
from hyw_paths import OUTPUT_DIR, SCENARIOS_DIR, WORKBENCH_ROOT  # noqa: E402

CONVERTER = WORKBENCH_ROOT / "tools" / "waymo_to_scenario.py"
DEFAULT_CONDA_PYTHON = Path.home() / "miniconda3/envs/waymo_env/bin/python"

Point2D = Tuple[float, float]
Polyline2D = List[Point2D]


@dataclass
class ScenarioMetrics:
    index: int
    scenario_id: str
    frames: int
    travel_m: float
    intersection_lane_count: int
    passes_intersection: bool
    turn_at_intersection: bool
    max_turn_deg: float
    min_intersection_dist_m: float


@dataclass
class FilterConfig:
    min_travel_m: float = 30.0
    long_travel_m: float = 80.0
    intersection_dist_m: float = 8.0
    turn_deg: float = 25.0
    turn_near_intersection_m: float = 15.0
    require_intersection: bool = False
    require_turn: bool = False


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


def _polyline_to_xy(polyline) -> Polyline2D:
    return [(float(p.x), float(p.y)) for p in polyline]


def _is_intersection_lane(lane) -> bool:
    n_entry = len(lane.entry_lanes)
    n_exit = len(lane.exit_lanes)
    return n_entry >= 2 or n_exit >= 2 or (n_entry >= 1 and n_exit >= 1)


def _extract_intersection_polylines(scenario) -> List[Polyline2D]:
    polylines: List[Polyline2D] = []
    for mf in scenario.map_features:
        if mf.WhichOneof("feature_data") != "lane":
            continue
        ln = mf.lane
        if not _is_intersection_lane(ln):
            continue
        poly = _polyline_to_xy(ln.polyline)
        if len(poly) >= 2:
            polylines.append(poly)
    return polylines


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _point_polyline_distance(px: float, py: float, polyline: Polyline2D) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in zip(polyline, polyline[1:]):
        best = min(best, _point_segment_distance(px, py, ax, ay, bx, by))
    return best


def _min_distance_to_polylines(px: float, py: float, polylines: Sequence[Polyline2D]) -> float:
    if not polylines:
        return float("inf")
    return min(_point_polyline_distance(px, py, poly) for poly in polylines)


def _normalize_angle_rad(delta: float) -> float:
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _sdc_valid_states(scenario) -> List[Tuple[float, float, float]]:
    """Return list of (x, y, heading_rad) for valid SDC states."""
    try:
        sdc = scenario.tracks[int(scenario.sdc_track_index)]
    except (IndexError, ValueError):
        return []
    out: List[Tuple[float, float, float]] = []
    for st in sdc.states:
        if not st.valid:
            continue
        out.append((float(st.center_x), float(st.center_y), float(st.heading)))
    return out


def _compute_travel_m(states: Sequence[Tuple[float, float, float]]) -> float:
    travel = 0.0
    for (x0, y0, _), (x1, y1, _) in zip(states, states[1:]):
        travel += math.hypot(x1 - x0, y1 - y0)
    return travel


def _compute_intersection_metrics(
    states: Sequence[Tuple[float, float, float]],
    polylines: Sequence[Polyline2D],
    intersection_dist_m: float,
    turn_deg: float,
    turn_near_intersection_m: float,
) -> Tuple[bool, bool, float, float]:
    if not states or not polylines:
        return False, False, 0.0, float("inf")

    turn_threshold_rad = math.radians(turn_deg)
    min_dist = float("inf")
    passes_intersection = False
    max_turn_deg = 0.0
    turn_at_intersection = False

    frame_dists: List[float] = []
    for x, y, _ in states:
        d = _min_distance_to_polylines(x, y, polylines)
        frame_dists.append(d)
        min_dist = min(min_dist, d)
        if d <= intersection_dist_m:
            passes_intersection = True

    for i in range(1, len(states)):
        _, _, h0 = states[i - 1]
        _, _, h1 = states[i]
        turn_rad = abs(_normalize_angle_rad(h1 - h0))
        turn_single_deg = math.degrees(turn_rad)
        max_turn_deg = max(max_turn_deg, turn_single_deg)
        if turn_rad < turn_threshold_rad:
            continue
        near_prev = frame_dists[i - 1] <= turn_near_intersection_m
        near_curr = frame_dists[i] <= turn_near_intersection_m
        if near_prev or near_curr:
            turn_at_intersection = True

    return passes_intersection, turn_at_intersection, max_turn_deg, min_dist


def analyze_scenario(
    scenario,
    index: int,
    cfg: FilterConfig,
) -> ScenarioMetrics:
    scenario_id = scenario.scenario_id if scenario.HasField("scenario_id") else "?"
    polylines = _extract_intersection_polylines(scenario)
    states = _sdc_valid_states(scenario)
    travel_m = _compute_travel_m(states)
    passes_ix, turn_at_ix, max_turn_deg, min_ix_dist = _compute_intersection_metrics(
        states,
        polylines,
        cfg.intersection_dist_m,
        cfg.turn_deg,
        cfg.turn_near_intersection_m,
    )
    return ScenarioMetrics(
        index=index,
        scenario_id=scenario_id,
        frames=len(states),
        travel_m=travel_m,
        intersection_lane_count=len(polylines),
        passes_intersection=passes_ix,
        turn_at_intersection=turn_at_ix,
        max_turn_deg=max_turn_deg,
        min_intersection_dist_m=min_ix_dist,
    )


def is_complex(metrics: ScenarioMetrics, cfg: FilterConfig) -> bool:
    if metrics.travel_m < cfg.min_travel_m:
        return False

    has_complex_behavior = (
        metrics.travel_m >= cfg.long_travel_m
        or metrics.passes_intersection
        or metrics.turn_at_intersection
    )
    if not has_complex_behavior:
        return False

    if cfg.require_intersection and not metrics.passes_intersection:
        return False
    if cfg.require_turn and not metrics.turn_at_intersection:
        return False
    return True


def _format_saved_line(metrics: ScenarioMetrics, out_name: str) -> str:
    ix = "yes" if metrics.passes_intersection else "no"
    turn = "yes" if metrics.turn_at_intersection else "no"
    return (
        f"idx={metrics.index} travel={metrics.travel_m:.1f}m "
        f"intersection={ix} turn@ix={turn} -> {out_name}"
    )


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
        "--proto-only",
        "--split-dynamic-frames",
    ]
    print(f"[convert] idx={scenario_index} -> {out_dir}")
    subprocess.run(cmd, check=True, cwd=str(WORKBENCH_ROOT))


def _scenario_out_dir(out_prefix: str, index: int) -> Path:
    return SCENARIOS_DIR / f"{out_prefix}_{index}"


def _out_dir_complete(out_dir: Path) -> bool:
    return (
        (out_dir / "scenario_meta.pb").is_file()
        and (out_dir / "lane_graph.pb").is_file()
        and (out_dir / "dynamic_objects" / "header.pb").is_file()
    )


def scan_and_filter(
    tf_path: Path,
    start_index: int,
    limit: int,
    cfg: FilterConfig,
) -> Iterable[ScenarioMetrics]:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    ds = tf.data.TFRecordDataset(str(tf_path), compression_type="")
    for i, raw in enumerate(ds):
        if i < start_index:
            continue
        if limit > 0 and i >= start_index + limit:
            break
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(raw.numpy()))
        try:
            yield analyze_scenario(scenario, i, cfg)
        except Exception as exc:
            print(f"[warn] idx={i} analyze failed: {exc}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="", help="Directory with *.tfrecord* (default: repo data/)")
    p.add_argument("--tfrecord", default="", help="Explicit TFRecord path")
    p.add_argument("--start-index", type=int, default=0, help="First scenario index in shard")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of scenarios to scan from start-index (0 = scan to shard end)",
    )
    p.add_argument(
        "--out-prefix",
        default="waymo_complex",
        help="Output dir: scenarios/<prefix>_<index>/",
    )
    p.add_argument("--python", default="", help="Python with tensorflow + waymo-open-dataset")
    p.add_argument("--dry-run", action="store_true", help="Only print matches, do not convert")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip conversion if output dir already exists (default: on)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Convert even when output dir exists",
    )
    p.add_argument("--overwrite", action="store_true", help="Alias for --no-skip-existing")
    p.add_argument(
        "--report-json",
        default="",
        help="Write saved scenario metrics to JSON (default: none)",
    )
    p.add_argument("--min-travel-m", type=float, default=30.0)
    p.add_argument("--long-travel-m", type=float, default=80.0)
    p.add_argument("--intersection-dist-m", type=float, default=8.0)
    p.add_argument("--turn-deg", type=float, default=25.0)
    p.add_argument("--turn-near-intersection-m", type=float, default=15.0)
    p.add_argument(
        "--require-intersection",
        action="store_true",
        help="Saved scenarios must pass through an intersection",
    )
    p.add_argument(
        "--require-turn",
        action="store_true",
        help="Saved scenarios must turn near an intersection",
    )
    args = p.parse_args()

    try:
        import tensorflow  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    except ImportError as e:
        print(f"Need waymo_env (tensorflow + waymo-open-dataset): {e}", file=sys.stderr)
        return 2

    cfg = FilterConfig(
        min_travel_m=args.min_travel_m,
        long_travel_m=args.long_travel_m,
        intersection_dist_m=args.intersection_dist_m,
        turn_deg=args.turn_deg,
        turn_near_intersection_m=args.turn_near_intersection_m,
        require_intersection=args.require_intersection,
        require_turn=args.require_turn,
    )

    skip_existing = args.skip_existing and not args.overwrite
    python = _pick_python(args.python)
    data_dir = _resolve_data_dir(args.data_dir)
    tf_path = _resolve_tfrecord(data_dir, args.tfrecord)

    print(f"# tfrecord: {tf_path}")
    print(
        f"# rules: min_travel={cfg.min_travel_m}m long_travel={cfg.long_travel_m}m "
        f"ix_dist={cfg.intersection_dist_m}m turn={cfg.turn_deg}deg "
        f"turn_near_ix={cfg.turn_near_intersection_m}m"
    )

    scanned = 0
    saved = 0
    skipped_existing = 0
    saved_rows: List[dict] = []

    for metrics in scan_and_filter(tf_path, args.start_index, args.limit, cfg):
        scanned += 1
        if not is_complex(metrics, cfg):
            continue

        out_dir = _scenario_out_dir(args.out_prefix, metrics.index)
        out_name = out_dir.name
        print(_format_saved_line(metrics, out_name))

        if skip_existing and _out_dir_complete(out_dir):
            print(f"  [skip-existing] {out_dir}")
            skipped_existing += 1
            saved += 1
            saved_rows.append(asdict(metrics) | {"out_dir": str(out_dir), "converted": False})
            continue

        saved += 1
        converted = False
        if not args.dry_run:
            convert_one(python, tf_path, metrics.index, out_dir)
            converted = True
        saved_rows.append(asdict(metrics) | {"out_dir": str(out_dir), "converted": converted})

    print(f"# scanned={scanned} saved={saved} skipped={scanned - saved} skipped_existing={skipped_existing}")
    if args.dry_run:
        print("# dry-run: no scenarios converted")

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        if not report_path.is_absolute():
            report_path = (OUTPUT_DIR / report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tfrecord": str(tf_path),
            "filter_config": asdict(cfg),
            "scanned": scanned,
            "saved": saved,
            "scenarios": saved_rows,
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"# report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
