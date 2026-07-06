# 焓差室制冷仿真 V3

目录和用法与 `simu/temperature` 保持一致：

```text
continuous_cooling_model.joblib  默认模型
continuous_model.py              在线特征和热负荷诊断状态
simulator.py                     reset/step/simulate 环境
example_usage.py                 最小在线示例
tools/                           数据、训练和评估脚本
output/                          评估结果
```

## 最小在线接口

```python
from simu.room.simulator import ContinuousEnthalpyRoomEnv

env = ContinuousEnthalpyRoomEnv()
env.reset(T_out=35.0, T_in=30.0, T_out_coil=37.0, T_in_coil=24.0)

for _ in range(720):
    observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)

print(observation["T_in"])
```

与 `simu/temperature` 一样，也支持：

```python
trajectory = env.simulate(
    initial_state=[35.0, 30.0, 37.0, 24.0],
    controls=controls,
)
```

使用完整初始观测可获得训练时一致的输入：

```python
from simu.room.tools.data_pipeline import load_status_csv

frame = load_status_csv("data/dataset/X1/status_data_20260519103131.csv")
env.reset(initial_observation=frame.iloc[0])
```

## 运行

```powershell
python -m simu.room.example_usage
python -m simu.room.tools.train_continuous
python -m simu.room.tools.evaluate_all
python -m simu.room.tools.evaluate_all --data-dir data/dataset_eval --output-dir simu/room/output/dataset_eval_v3
```

## 当前结果

- 12 条完整留出测试：MAE 0.3422 degC，RMSE 0.4467 degC；
- `dataset_eval` 6 条独立实验：MAE 0.2010 degC，RMSE 0.2925 degC；
- 最大单步温变不超过 0.0417 degC/5s。

热负荷跟踪状态用于诊断，当前不直接进入温度估计。连续性机理见
`CONTINUITY_FIX.md`。
