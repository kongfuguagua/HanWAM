# 房间制冷仿真 V4

目录和 V3 的 `simu/room` 保持相似，但 V4 独立放在 `simu/roomv4`，避免覆盖 V3：

```text
room_v4_model.pt              默认 V4 Torch 模型
simulator.py                  reset/step/simulate 在线环境
following_simulator.py        V4.1 跟随型环境
model.py                      V4 动态模型
data.py                       dataset_full 数据发现和切分
train.py / evaluate.py        训练与评估脚本
plot_curves.py                温度曲线绘图脚本
example_usage.py              最小在线示例
output/                       评估和绘图结果
```

## 最小在线接口

```python
from simu.roomv4.simulator import HybridRoomV4Env

env = HybridRoomV4Env()
env.reset(T_out=35.0, T_in=30.0, T_out_coil=37.0, T_in_coil=24.0)

for _ in range(720):
    observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)

print(observation["T_in"])
```

初始化时可传完整观测；目标温度、PID 输出、室内风机等目标/推理字段不会作为 V4
运行时输入。初始化后 `step()` 只接收三个控制量：

- `freq`
- `eev`
- `fan_out`

V4.1 跟随型接口：

```python
from simu.roomv4.following_simulator import HybridRoomV4FollowingEnv

env = HybridRoomV4FollowingEnv(fast_weight=0.5)
env.reset(initial_observation=frame.iloc[0])
observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)
```

## 运行

```powershell
python -m simu.roomv4.example_usage
python -m simu.roomv4.train --data-dir data/dataset_full
python -m simu.roomv4.evaluate --data-dir data/dataset_full
python -m simu.roomv4.plot_curves --data-dir data/dataset_full --output-dir simu/roomv4/output/temperature_curves_dataset_full_following --following-weight 0.5 --overview-only --quiet --jobs 4
```

更多设计说明和当前结果见 [V4_PLAN.md](V4_PLAN.md)。
