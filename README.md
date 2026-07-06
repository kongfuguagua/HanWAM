# Haier JK AC Energy-Saving Control

本项目围绕空调节能控制组织为三条主线：

```text
data/       整理后的工况数据、原始数据和评估结果
simu/       温度、频率、能耗仿真/预测模型
control/    PID、HanWAM 等控制器方法与 MiniController 实验框架
docs/       环境、训练测试流程和迁移说明
```

## 当前能力

- `data/dataset/`：默认训练/评估入口，按 `X1..X27/unclassified` 工况目录保存 `status_data_*.csv`。
- `data/split_manifest.csv`：整理版数据到 raw 来源的映射，用于恢复 train/val/test。
- `simu/air_conditioner.py`：空调执行器仿真器，输入 `[freq_target, eev, fan_out]`。
- `simu/environment.py`：环境仿真器，输出温度轨迹和热量代理。
- `control/MiniController/`：统一配置、控制器接口、闭环实验、指标、可视化和 SwanLab 追踪。
- `control/PID/`：PID 控制器方法。
- `control/HanWAM/`：两阶段 SkyJEPA-style latent world model + prober + MPC 控制器。
- `control/config/experiments/`：实验配置目录；一份 YAML 描述一次实验和一个方法。

控制实验默认按真实实验协议运行：控制周期 `5s`，单次实验 `4h`（`14400s`）。

## 常用命令

```bash
source /home/hmq/miniconda3/etc/profile.d/conda.sh
conda activate haier-jk

python -m simu.temperature.example_usage
python -m simu.temperature.tools.evaluate_online
python -m simu.frequency.freq_response_model
python -m simu.energy.pinn

CONFIG=control/HanWAM/config/hanwam_smoke.yml SWANLAB=0 scripts/train_hanwam.sh
python -m control.HanWAM.eval --config control/HanWAM/config/hanwam_smoke.yml
python -m control.MiniController.main --config control/config/experiments/pid.yml --stage eval
```

推荐用 HanWAM shell 入口统一训练配置：

```bash
CONFIG=control/HanWAM/config/hanwam.yml SWANLAB=1 scripts/train_hanwam.sh
```

详细训练、测试、可视化和 SwanLab 流程见：

```text
docs/TRAIN_TEST_FLOW.md
```

新控制方法请按方法目录放入 `control/<METHOD>/`，并接入 `control/MiniController/` 的配置、实验流程和轨迹可视化。
