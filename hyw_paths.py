"""Shared path layout for the ~/hyw multi-repo workbench."""
from __future__ import annotations

import os
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parent
HYW_ROOT = Path(os.environ.get("HYW_ROOT", str(WORKBENCH_ROOT.parent))).resolve()

HYW_PROTO = HYW_ROOT / "hyw-proto"
HYW_GRADING = HYW_ROOT / "hyw-grading"
HYW_SIM = HYW_ROOT / "hyw-sim"
HYW_PLANNER = HYW_ROOT / "hyw-planner"

SCENARIOS_DIR = WORKBENCH_ROOT / "scenarios"
OUTPUT_DIR = WORKBENCH_ROOT / "output"
PYSIM_DIR = WORKBENCH_ROOT / "pysim"
TOOLS_DIR = WORKBENCH_ROOT / "tools"

DEFAULT_GRADING_BIN = (
    HYW_GRADING / "bazel-bin" / "src" / "entry" / "grading_main"
)
DEFAULT_PLANNER_BIN = HYW_PLANNER / "bazel-bin" / "cpp" / "planner_server"
DEFAULT_PLANNER_ADDRESS = "localhost:50051"
DEFAULT_METRICS = HYW_GRADING / "config" / "metrics_default.json"
RUN_SIM = HYW_SIM / "run_sim.py"
VIZ_SIM = TOOLS_DIR / "viz_sim.py"
