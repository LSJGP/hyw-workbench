#!/usr/bin/env python3
"""Batch sim + grading + optional GIF for scenarios under ``scenarios/``.

Called by ``web/server.py`` or CLI. Uses ``sim/run_sim.py`` and ``tools/viz_sim.py``.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH_ROOT))
from hyw_paths import (  # noqa: E402
    DEFAULT_GRADING_BIN,
    DEFAULT_METRICS,
    HYW_SIM,
    OUTPUT_DIR,
    RUN_SIM,
    SCENARIOS_DIR,
    VIZ_SIM,
    WORKBENCH_ROOT,
)

DEFAULT_SIM_RUNNER_HINT = (
    "hyw-sim/bazel-bin/cpp/sim_runner "
    "(build: cd hyw-sim && bazel build //cpp:sim_runner)"
)

PLANNERS = ["local_dwa", "reference_tracker", "goal_seek"]
LOG_LEVELS = ["trace", "debug", "info", "warn", "error", "off"]
CPP_MODES = ["online", "offline", "both", "off"]
REFERENCE_SOURCES = ["map", "sdc"]

# Keys must match REGISTER_METRIC(..., "name") in grading_mini.
METRIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "planning_limit_checker": {
        "paramsJson": '{"maxDesiredSpeedMps": 33.3}',
    },
    "speed_checker": {
        "paramsJson": '{"maxSpeedThreshold": 33.3}',
    },
    "regulatory_collision_checker": {
        "paramsJson": None,
    },
    "lane_departure_checker": {
        "paramsJson": '{"minRoadEdgeClearanceM": 0.35, "minLaneBoundaryClearanceM": 0.0}',
    },
    "drivable_area_checker": {
        "paramsJson": '{"minClearanceM": 0.35, "checkCenterOnly": false}',
    },
    "solid_line_crossing_checker": {
        "paramsJson": '{"minClearanceM": 0.25}',
    },
}


@dataclass
class BatchConfig:
    scenario_names: List[str]
    planner: str = "local_dwa"
    metrics: List[str] = field(default_factory=lambda: list(METRIC_CATALOG.keys()))
    reference_source: str = "map"
    reference_step: float = 1.0
    dt: float = 0.1
    max_seconds: float = 0.0
    desired_speed: float = 13.9
    cpp_mode: str = "both"
    run_grading: bool = True
    grading_bin: str = ""
    log_level: str = "info"
    log_dir: str = ""
    source_tag: str = "batch_ui"
    make_gif: bool = True
    gif_fps: int = 120
    gif_dpi: int = 100
    gif_reference_step: float = 1.0
    output_log_dir: str = ""
    output_report_dir: str = ""
    output_viz_dir: str = ""


def list_scenarios() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not SCENARIOS_DIR.is_dir():
        return out
    for d in sorted(SCENARIOS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "scenario_meta.json"
        dyn = d / "dynamic_objects.json"
        lg = d / "lane_graph.json"
        if not (meta.is_file() and dyn.is_file() and lg.is_file()):
            continue
        info: Dict[str, Any] = {"name": d.name, "path": str(d)}
        try:
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            info["scenario_id"] = m.get("scenario_id", "")
            info["duration_s"] = m.get("duration_s")
        except OSError:
            pass
        out.append(info)
    return out


def build_metrics_config(
    metric_names: List[str],
    spdlog_level: str = "info",
    simple_planner_max_speed: float = 33.3,
) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []
    for name in metric_names:
        if name not in METRIC_CATALOG:
            continue
        entry: Dict[str, Any] = {"name": name}
        pj = METRIC_CATALOG[name].get("paramsJson")
        if pj:
            entry["paramsJson"] = pj
        metrics.append(entry)
    return {
        "simplePlannerMaxSpeedMps": simple_planner_max_speed,
        "spdlogLevel": spdlog_level,
        "metrics": metrics,
    }


def write_metrics_config(path: Path, cfg: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    line = msg.rstrip("\n")
    if cb:
        cb(line)
    else:
        print(line, flush=True)


def _run(cmd: List[str], cwd: Path, log: Optional[Callable[[str], None]]) -> int:
    _log(log, "$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _log(log, line.rstrip("\n"))
    return proc.wait()


def run_one_scenario(
    scenario_name: str,
    cfg: BatchConfig,
    metrics_path: Path,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    scenario_dir = SCENARIOS_DIR / scenario_name
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario not found: {scenario_dir}")

    log_dir = Path(cfg.output_log_dir or (OUTPUT_DIR / "log"))
    report_dir = Path(cfg.output_report_dir or (OUTPUT_DIR / "report"))
    viz_dir = Path(cfg.output_viz_dir or (OUTPUT_DIR / "viz"))
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    sim_log = log_dir / f"{scenario_name}_sim_log.json"
    report = report_dir / f"{scenario_name}_grading_report.json"
    gif_path = viz_dir / f"{scenario_name}_sim.gif"

    result: Dict[str, Any] = {
        "scenario": scenario_name,
        "sim_log": str(sim_log),
        "grading_report": str(report) if cfg.run_grading else None,
        "gif": str(gif_path) if cfg.make_gif else None,
        "sim_rc": None,
        "viz_rc": None,
        "passed": None,
    }

    sim_cmd = [
        sys.executable,
        str(RUN_SIM),
        "--scenario-dir",
        str(scenario_dir),
        "--planner",
        cfg.planner,
        "--reference-source",
        cfg.reference_source,
        "--reference-step",
        str(cfg.reference_step),
        "--output",
        str(sim_log),
        "--source-tag",
        cfg.source_tag,
        "--dt",
        str(cfg.dt),
        "--max-seconds",
        str(cfg.max_seconds),
        "--desired-speed",
        str(cfg.desired_speed),
        "--cpp-mode",
        cfg.cpp_mode if cfg.run_grading else "off",
        "--log-level",
        cfg.log_level,
    ]
    if cfg.log_dir:
        sim_cmd.extend(["--log-dir", str(Path(cfg.log_dir).expanduser().resolve())])
    elif cfg.log_level not in ("info", "off"):
        sim_cmd.extend(["--log-dir", str(log_dir)])

    if cfg.run_grading:
        grading_bin = Path(cfg.grading_bin or DEFAULT_GRADING_BIN).expanduser().resolve()
        if not grading_bin.is_file():
            raise FileNotFoundError(
                f"grading_main not found: {grading_bin}\n"
                "Build: cd hyw-grading && bazel build //src/entry:grading_main"
            )
        sim_cmd.extend(
            [
                "--grading-bin",
                str(grading_bin),
                "--grading-report",
                str(report),
                "--metrics-config",
                str(metrics_path),
            ]
        )

    _log(log, f"\n=== [{scenario_name}] simulation ===")
    t0 = time.time()
    result["sim_rc"] = _run(sim_cmd, HYW_SIM, log)
    result["sim_seconds"] = round(time.time() - t0, 2)

    if cfg.run_grading and report.is_file():
        try:
            with open(report, encoding="utf-8") as f:
                rep = json.load(f)
            result["passed"] = rep.get("overallPassed")
            result["summaries"] = rep.get("summaries", [])
        except (OSError, json.JSONDecodeError):
            result["passed"] = None

    if cfg.make_gif and result["sim_rc"] == 0 and sim_log.is_file():
        _log(log, f"\n=== [{scenario_name}] visualization ===")
        viz_cmd = [
            sys.executable,
            str(VIZ_SIM),
            "--scenario-dir",
            str(scenario_dir),
            "--sim-log",
            str(sim_log),
            "--animate",
            "--fps",
            str(cfg.gif_fps),
            "--dpi",
            str(cfg.gif_dpi),
            "--reference-step",
            str(cfg.gif_reference_step),
            "--output",
            str(gif_path),
        ]
        result["viz_rc"] = _run(viz_cmd, WORKBENCH_ROOT, log)

    return result


def run_batch(
    cfg: BatchConfig,
    log: Optional[Callable[[str], None]] = None,
    metrics_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not cfg.scenario_names:
        raise ValueError("no scenarios selected")

    mpath = metrics_config_path or (
        OUTPUT_DIR / "batch" / f"metrics_{int(time.time())}.json"
    )
    if cfg.run_grading:
        write_metrics_config(
            mpath,
            build_metrics_config(cfg.metrics, spdlog_level=cfg.log_level),
        )
        _log(log, f"[batch] metrics config: {mpath}")

    summary: Dict[str, Any] = {
        "planner": cfg.planner,
        "metrics": cfg.metrics,
        "scenarios": [],
        "started_at": time.time(),
    }
    for name in cfg.scenario_names:
        try:
            one = run_one_scenario(name, cfg, mpath, log=log)
        except Exception as e:
            one = {"scenario": name, "error": str(e), "sim_rc": -1}
        summary["scenarios"].append(one)
    summary["finished_at"] = time.time()
    summary["elapsed_s"] = round(summary["finished_at"] - summary["started_at"], 2)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", required=True, help="scenario folder names")
    p.add_argument("--planner", default="local_dwa", choices=PLANNERS)
    p.add_argument("--metrics", nargs="*", default=list(METRIC_CATALOG.keys()))
    p.add_argument("--no-grading", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--cpp-mode", default="both", choices=CPP_MODES)
    p.add_argument("--log-level", default="info", choices=LOG_LEVELS)
    p.add_argument("--gif-fps", type=int, default=120)
    args = p.parse_args()

    cfg = BatchConfig(
        scenario_names=args.scenarios,
        planner=args.planner,
        metrics=list(args.metrics),
        run_grading=not args.no_grading,
        make_gif=not args.no_gif,
        cpp_mode=args.cpp_mode,
        log_level=args.log_level,
        gif_fps=args.gif_fps,
    )
    summary = run_batch(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = [
        s
        for s in summary["scenarios"]
        if s.get("sim_rc", 0) != 0 or s.get("passed") is False
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
