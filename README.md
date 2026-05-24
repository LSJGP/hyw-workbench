# hyw-workbench

集成仓：批跑脚本、Web Dashboard、示例场景、Python 可视化依赖（pysim）。

## 目录

| 路径 | 说明 |
|------|------|
| `tools/` | `batch_run_scenarios.py`、`viz_sim.py`、Waymo 转换等 |
| `web/` | 本地 Dashboard |
| `scenarios/` | 示例 JSON 场景 |
| `pysim/` | `viz_sim` 使用的 `waymo_sim` 包 |
| `data/` | 本地 Waymo tfrecord（不进 git） |
| `output/` | 仿真/评分/可视化输出（运行时生成） |
| `hyw_paths.py` | 四仓路径约定 |

## 使用

```bash
# 可选
source hyw.env
# 没别的问题直接开前端就行
./start_dashboard.sh
# 或 CLI
python3 tools/batch_run_scenarios.py --scenarios waymo_scenario_5
```
