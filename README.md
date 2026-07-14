# HanWAM

这是 HanWAM 空调制冷控制器的干净代码快照，当前重点是 E065 world-model + MPPI 控制实验。

## 内容

- `control/HanWAM/`：HanWAM 两阶段 world model、hard-mechanism prober、MPPI planner、E065 配置。
- `control/MiniController/`：统一实验配置、闭环评估、指标和可视化框架。
- `servicer/`：对外控制 API 服务，以及 HanWAM WM+MPPI adapter。
- `simu/`：执行器、温度、房间和 room-hybrid 仿真代码。
- `tests/` 和 `servicer/tests/`：不依赖真实 CSV 或 checkpoint 的单元测试与 mock 测试。

## 不包含的文件

本仓库不提交真实数据、训练产物和模型权重，包括 `.csv`、`.pt`、`.pth`、`.pkl`、`.joblib`、`.pdf`、`.png` 等。E065 配置中的 checkpoint 路径保留为运行约定，实际权重请在部署或训练环境中外部挂载。

默认外部路径约定：

```text
data/dataset/X*/status_data_*.csv
data/split_manifest.csv
control/HanWAM/checkpoints/*.pt
control/outputs/
```

## E065 入口

主要配置：

```text
control/HanWAM/config/hanwam_e065_positive_eev_energy_v1.yml
control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml
```

API 服务默认使用低频 10Hz E065 候选：

```text
servicer/config/api_service_hanwam_docker.yml
```

## 常用命令

```bash
python -m compileall control simu servicer tests
python -m pytest tests servicer/tests
python -m control.HanWAM.eval --config control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml
```

训练、评估和数据约定见 `docs/README.md`。HanWAM 架构和 E065 说明见 `control/HanWAM/README.md`。
