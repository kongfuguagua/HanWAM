# 训练与测试流程

一份 YAML 配置描述一次训练或评估。HanWAM 主配置在 `control/HanWAM/config/`，PID/fixed 等基线配置在 `control/config/experiments/`。

## 数据流

闭环评估按真实控制协议运行：

```text
controller 输出 [freq_target, eev, fan_out]
  -> AirConditionerSimulator 生成实际频率、功率和电量
  -> EnvironmentSimulator 或 room-hybrid simulator 生成温度轨迹
  -> metrics / trajectories / plots / SwanLab
```

默认控制周期为 `5s`，标准实验时长为 `4h`。

## 运行

```bash
python -m control.MiniController.main \
  --config control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml \
  --stage eval

python -m control.MiniController.main \
  --config control/config/experiments/pid.yml \
  --stage eval
```

HanWAM 也可直接使用模块入口：

```bash
python -m control.HanWAM.train --config control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml
python -m control.HanWAM.eval --config control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml
```

## 输出

输出目录由 `experiment.output_root` 和 `experiment.name` 控制，通常位于 `control/outputs/<experiment>/...`。这些运行产物不提交到仓库。

常见输出包括：

- `summary.csv`
- `trajectories.csv`
- `controller_log.csv`
- `controller_history.csv`
- `*_debug.png`

## 提交约束

仓库只提交代码、配置、文档和 mock 测试。真实数据、模型权重、图片、PDF 和运行输出保持外部化。
