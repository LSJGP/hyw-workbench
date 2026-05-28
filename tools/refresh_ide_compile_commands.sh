#!/usr/bin/env bash
# Regenerate compile_commands.json for clangd (hyw-sim + hyw-planner + hyw-grading).
set -euo pipefail
WORKBENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HYW_ROOT="$(cd "${WORKBENCH_ROOT}/.." && pwd)"

echo "[ide] hyw-sim: build protos + refresh compile_commands"
cd "$HYW_ROOT/hyw-sim"
bazel build @hyw_proto//proto/sim:runtime_cc_proto //cpp:sim_runner
bazel run //:refresh_compile_commands

echo "[ide] hyw-planner: build + refresh compile_commands"
cd "$HYW_ROOT/hyw-planner"
bazel build @hyw_proto//proto/planner:planner_cc_grpc //cpp:planner_server
bazel run //:refresh_compile_commands

echo "[ide] hyw-grading: refresh compile_commands"
cd "$HYW_ROOT/hyw-grading"
bazel build //src/entry:grading_main
bazel run //:refresh_compile_commands

echo "[ide] merge compile_commands.json at workbench root"
python3 "$WORKBENCH_ROOT/tools/merge_compile_commands.py"

echo "[ide] done. Restart clangd: Command Palette -> clangd: Restart language server"
