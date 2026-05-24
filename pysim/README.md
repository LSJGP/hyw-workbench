# pysim — Python 闭环仿真模块

`pysim` 是从 `sim` 中拆分出的 Python 仿真实现，包含：

- `run_sim.py`：Python 仿真 CLI 入口
- `waymo_sim/`：场景解析、车辆模型、碰撞与 Python planners
- `requirements.txt`：Python 侧可选依赖说明
- `run_default.sh`：使用示例场景的一键运行脚本

快速运行：

```bash
cd pysim
python3 run_sim.py --scenario-dir ../scenarios/waymo_scenario_5
# scenarios live in hyw-workbench/scenarios/
```
