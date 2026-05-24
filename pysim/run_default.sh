#!/usr/bin/env bash
# 使用 pysim/run_sim.py 的默认参数跑 Python 闭环仿真；仅固定场景目录为仓库内示例（可覆盖）。
# 用法:
#   ./run_default.sh
#   SCENARIO_DIR=/path/to/scenario ./run_default.sh
#   ./run_default.sh --planner frenet_optimal --dt 0.05

set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_DIR="${SCENARIO_DIR:-$SIM_ROOT/../scenarios/waymo_scenario_5}"

if [[ ! -d "$SCENARIO_DIR" ]]; then
  echo "错误: 场景目录不存在: $SCENARIO_DIR" >&2
  echo "请设置 SCENARIO_DIR 指向含 scenario_meta / dynamic_objects / lane_graph 的目录，或先按 tools 流程生成场景。" >&2
  exit 1
fi

cd "$SIM_ROOT"
exec python3 run_sim.py --scenario-dir "$SCENARIO_DIR" "$@"
