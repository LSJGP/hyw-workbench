#!/usr/bin/env bash
# Start the Hyw batch-sim web dashboard (stdlib HTTP server).
#
# Usage:
#   ./start_dashboard.sh
#   ./start_dashboard.sh --port 8765
#   HOST=0.0.0.0 ./start_dashboard.sh --port 9000

set -euo pipefail

WORKBENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
HYW_ROOT="$(cd "${WORKBENCH_ROOT}/.." && pwd)"
cd "$WORKBENCH_ROOT"

export HYW_ROOT
export HYW_WORKBENCH="${WORKBENCH_ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

mkdir -p output/batch output/log output/report output/viz

GRADING_BIN="${HYW_ROOT}/hyw-grading/bazel-bin/src/entry/grading_main"
if [[ ! -x "${GRADING_BIN}" ]]; then
  echo "[dashboard] note: grading_main not built yet."
  echo "[dashboard]   cd ${HYW_ROOT}/hyw-grading && bazel build //src/entry:grading_main"
fi

SIM_RUNNER="${HYW_ROOT}/hyw-sim/bazel-bin/cpp/sim_runner"
if [[ ! -x "${SIM_RUNNER}" ]]; then
  echo "[dashboard] note: sim_runner not built yet."
  echo "[dashboard]   cd ${HYW_ROOT}/hyw-sim && bazel build //cpp:sim_runner"
fi

echo "[dashboard] hyw root: ${HYW_ROOT}"
echo "[dashboard] workbench: ${WORKBENCH_ROOT}"
echo "[dashboard] url:  http://${HOST}:${PORT}/"
echo "[dashboard] stop: Ctrl+C"
echo ""

args=(--host "${HOST}" --port "${PORT}")
if [[ $# -gt 0 ]]; then
  args=("$@")
fi
exec python3 "${WORKBENCH_ROOT}/web/server.py" "${args[@]}"
