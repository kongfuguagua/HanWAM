# HanWAM 控制器

HanWAM 是用于空调制冷控制的两阶段 latent world model + MPPI/MPC 控制器。当前干净提交保留 E065 代码、配置和服务集成，不提交真实 CSV、checkpoint 或评估输出。

## 数据接口

标准 schema 定义在 `type.py`。

观察量：

```text
freq, fan_out, fan_in, eev,
T_out_coil, T_in_coil, T_out_discharge,
T_in, T_out, energy_cum, T_set, mode
```

控制动作：

```text
freq_target, eev, fan_out
```

hard-mechanism prober 输出：

```text
T_in_delta, electric_kwh_delta
```

planner 使用累计温度增量和非负电量增量评分候选动作序列。

## 训练约定

Stage 1 只训练 latent world model。Stage 2 载入 Stage 1 checkpoint，冻结 world model，只训练物理 prober 或 hard-mechanism prober。E065 的 Stage 2 仍复用 E062 h4/f8 Stage 1，实际 checkpoint 不在仓库内。

主要外部 checkpoint 路径：

```text
control/HanWAM/checkpoints/hanwam_e062_h4_f8_lagged_energy_v1_mode1_stage1.pt
control/HanWAM/checkpoints/hanwam_e065_positive_eev_energy_v1_mode1_stage2_epoch0040.pt
control/HanWAM/checkpoints/hanwam_e065_lowfreq10_positive_eev_energy_v1_mode1.pt
```

## E065 配置

E065 修正 E064 的 EEV 能耗项：在相同压缩机频率下，EEV 越大只增加非负附加功率，旧的 signed-linear 模式仍保留用于复现实验。

```text
config/hanwam_e065_positive_eev_energy_v1.yml
config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml
```

`hanwam_e065_lowfreq10_positive_eev_energy_v1.yml` 是当前服务默认候选，允许 `10Hz` 连续低频运行，并将低频 anchors 显式绑定到低 EEV。

## 规划约定

MPPI 在未来动作块上采样，调用 HanWAM rollout，按 deadline 达温、舒适带约束、能耗和动作平滑评分，再下发第一个 receding-horizon 动作。

当前生产风格配置：

```text
step_seconds = 5
frames_per_block = 12
history_blocks = 4
future_blocks = 8
horizon_steps = 96
```

## 代码入口

```text
model.py       world model 与 hard-mechanism controller model
planner.py     MPPI planner 和 E060/E065 cost
train.py       两阶段训练入口
eval.py        标准场景评估入口
utils.py       deadline 和共用工具
```
