# room_hybrid

`room_hybrid` 是目前室内环境温度 `T_in` 开环估计最好的部署方案整理版。
该目录是自包含的：运行时不再引用 `simu/roomv5` 或 `simu/room_world`
目录下的代码/模型文件。

它保持 `room`/`roomv5` 的在线用法：

- `reset(first_row)`：只用数据第一行初始化；
- `step(freq, eev, fan_out)`：之后每 5 秒只输入压缩机频率、EEV、外风机转速；
- `simulate(first_row, actions)`：批量开环滚动完整动作序列。

## 模型组成

- 室温 `T_in`：`models/room_v5_model.pt` + `models/room_v5_residual_tcn.pt`
- 辅助温度：默认 `models/room_world_aux_v6_context.pt`

输出中的 `T_in` 是 hybrid 室温。为了诊断，也会保留：

- `T_in_slow`：V5 slow 主干；
- `T_in_residual`：Residual TCN 修正量；
- `T_in_aux_backbone`：辅助 world model 自己预测的室温；
- `T_in_coil`、`T_out_coil`、`T_out_discharge`：辅助 world model 输出。

## 示例

```python
from simu.room_hybrid import HybridRoomEnv
from simu.room_hybrid.api import ACTION_COLUMNS
from simu.room_hybrid.data import load_status_csv

frame = load_status_csv("data/dataset_own/1_data_202607120204.csv")
env = HybridRoomEnv()
actions = frame[ACTION_COLUMNS].to_numpy("float32")[:-1]
result = env.simulate(frame.iloc[0], actions)
print(result[["T_in", "T_in_slow", "T_in_residual"]].tail())
```

## 评估

```powershell
C:/Users/liuyx/miniconda3/envs/torch/python.exe -m simu.room_hybrid.evaluate --data-dir data/dataset_eval
```

如果直接执行遇到顶层 `simu` 包导入副作用，可使用项目里现有的 `runpy` 包装方式运行。
