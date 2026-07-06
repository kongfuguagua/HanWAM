## 核心目标

继续迭代 `/home/hmq/haier_jk/control/HanWAM`，在`mode=1` 制冷仿真器上得到物理可信、可复现、可部署的 SOTA 控制器。

性能提升必须主要来自：

- 两阶段 latent world model 训练；
- prober 为 MPC 提供可解释物理预测；
- 轻量、稳定、真实设备友好的 MPC planning/control。

不能依赖：

- 仿真器漏洞；
- 引入其他监督或旁路的世界模型（只能JEPA-like）；
- hand-written safety profile 或 fallback controller。

## 固定接口

静态 schema 固定在 `control/HanWAM/type.py`：

```text
observation:
T_in, T_set, mode, freq_target, freq, eev, fan_out,
elapsed_seconds, T_out, energy_cum

action:
freq_target, eev, fan_out

prober output:
T_in, freq, T_in_delta, electric_kwh_delta
```

实验配置和调参只放在：

```text
control/HanWAM/config/*.yml
```

## 当前实现状态

HanWAM 已按严格两阶段训练接好：

1. Stage 1 只训练 latent world model：

```text
state history encoder
+ action history encoder
+ action encoder
+ GRU predictor

loss = multi-step latent MSE + SIGReg
```

2. Stage 2 冻结 world model，只训练 prober：

```text
stop-gradient latent rollout -> prober -> physical MPC variables
loss = physical prober MSE
```

当前训练和在线 MPC 都使用 history window：

```text
obs_history
```

不是单帧 observation。obs_history 中已经包含 freq_target/freq/eev/fan_out
等执行器状态历史，未来控制动作只通过 action_encoder 进入 predictor。
这个修正很关键：单帧版本曾导致 world model
action sensitivity 很弱，MPC 选择全关机，10min 能耗为 0 但温度升高。

## 最新验证基线

配置：

```text
control/HanWAM/config/hanwam_design_validation.yml
```

关键参数：

```text
history_steps = 12
horizon_steps = 24
control_interval_steps = 6   # 30s execution interval
limit_transitions = 8192
val_limit_transitions = 4096
```

同一条 4h test window：`X1_status_data_20260519103131`

```text
policy      success  reach_time_s  final_error_c  electric_kwh
hanwam      true     11340         0.1425         1.1455
pid         true     2140         -0.1290         2.2102
fixed       true     3510          0.4711         2.2190
historical  true     9635         -0.4697         4.6325
```

## 物理可信性检查

当前 validation 结果：

```text
deadband_count = 0
mean_abs_delta_freq_target = 0.1247 Hz/step
planner_calls = 480 over 2880 simulator steps
off_power_zero_cooling_steps = 1 / 1964
off_power_zero_cooling_sum_c ~= -0.042C
```

结论：

- 没有 0-15Hz 非物理压缩机命令。
- 没有长期 compressor-off / fan-only 凭空制冷。
- 30s MPC 执行周期明显降低了 Hz 抖动。
- 待优化：关机阶段 high fan 比例偏高，真实设备上不够优雅，需要在
  MPC 中加入简单、可解释的 fan/action 成本或约束。

## 下一轮重点

1. 启动更大规模 HanWAM 训练。
   - 考虑优化数据模式（如世界模型的horizon）
   - 模型的结构
   - 非端到端的标准2阶段的epoch选择范式（如就选取stage-I的第20个epoch然后冻结选择stage-II的第10个epoch的checkpoint）

2. 提升 world model 的 action sensitivity。
   - 当前短训模型闭环可用，但 open-loop off/base/high 动作区分仍偏弱。
   - 优先检查 Stage 1 稳定性、SIGReg 权重、latent_dim、hidden_dim、
     history_steps、horizon_steps、batch size、训练轮数。

3. MPC 保持简单。
   - 当前 objective：

```text
J = tracking + energy + action_smooth
```

   - 可以加入简单 fan/action cost，服务真实设备部署。
   - 不要把 MPC 目标函数复杂化成大量手写策略。
   - 只允许 actuator bounds clipping / deadband projection。

4. 每轮实验必须可复现。
   - `compileall`
   - unit tests
   - 10min smoke closed-loop
   - 1 条 4h test window
   - 有希望再跑 full4 test split
   - 对比 historical / fixed / pid
   - 输出 checkpoint、summary.csv、trajectories.csv、controller_log.csv、
     aggregate_compare.csv、action_jitter_statistics.csv、cycle/OOD/off-cooling
     诊断。
     
## 一句话目标

在严格两阶段 HanWAM 架构上扩大训练并优化轻量 MPC，使 full4 test split
稳定成功、能耗低于 pid/fixed/historical，同时动作平滑、物理可信、可向
真实设备部署。
