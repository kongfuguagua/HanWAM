# 训练与测试流程

本文档说明如何用一份 YAML 配置开启一次控制实验。一个配置只描述一个方法，例如 WAM、PID、fixed 或 historical。

## 1. 数据与流程

默认数据入口是整理版：

```text
data/dataset/
  X1/
    status_data_*.csv
  X2/
  ...
  X27/
  unclassified/
data/split_manifest.csv
```

`data/dataset` 按工况目录组织；`split_manifest.csv` 记录每个整理后 CSV 的 raw 来源。MiniController 通过 manifest 中的 `1轮/2轮/3轮` 恢复 train/val/test：

- train：`1轮` 和 `0414~0424`
- val：`2轮`
- test：`3轮`

闭环测试的数据流：

```text
controller 输出 [freq_target, eev, fan_out]
  -> AirConditionerSimulator 输出 actual freq、power_w、electric_kwh
  -> EnvironmentSimulator 输出 T_in、thermal_power_w、thermal_kwh
  -> metrics / trajectories / plots / SwanLab
```

真实实验协议固定为：

- 控制周期：`5s`，即每 5 秒调用一次 controller。
- 单次实验时长：`4h`，即 `14400s`、`2880` 个控制步。

## 2. 配置文件

实验配置目录：

```text
control/HanWAM/config/
  hanwam.yml
  hanwam_smoke.yml
  hanwam_design_validation.yml

control/config/experiments/
  pid.yml
  fixed.yml
  historical.yml
```

一级目录固定为：

```text
experiment   实验名、seed、输出目录、mode、stages、SwanLab
data         dataset 根目录、manifest、工况 include/exclude、split、采样、字段列
simulator    空调和环境仿真器参数
train        学习方法训练参数
method       当前实验唯一方法及其参数
eval         测评场景、4h horizon、目标温度、舒适带
visualization 可视化和 CSV 输出开关
```

没有顶层 `schema` 或 `constraints`：

- 字段列属于 `data.columns`。
- PID/WAM 动作边界属于 `method.action_space_by_mode`。
- 频率上限等物理仿真限制属于 `simulator.air_conditioner`。

`mode` 支持 `1`、`3`、`all`。`mode: all` 会分别展开 mode 1 和 mode 3；如果某个 mode 在当前 `data.group_dirs` 下没有可用数据，eval 会给出 warn 并跳过。

## 3. 环境检查

```bash
cd /home/hmq/haier_jk
source /home/hmq/miniconda3/etc/profile.d/conda.sh
conda activate haier-jk

python -m simu.temperature.example_usage
python -m simu.temperature.tools.evaluate_online
python -m simu.frequency.freq_response_model
python -m simu.energy.pinn
python -m unittest tests.test_baseline_controllers tests.test_closed_loop_interfaces
```

## 4. 运行实验

Python 入口：

```bash
python -m control.HanWAM.train \
  --config control/HanWAM/config/hanwam_smoke.yml

python -m control.HanWAM.eval \
  --config control/HanWAM/config/hanwam_smoke.yml

python -m control.MiniController.main \
  --config control/config/experiments/pid.yml \
  --stage eval
```

Shell 入口：

```bash
CONFIG=control/HanWAM/config/hanwam.yml SWANLAB=1 scripts/train_hanwam.sh
CONFIG=control/HanWAM/config/hanwam_smoke.yml SWANLAB=0 scripts/train_hanwam.sh
```

Shell 只负责激活环境和传递少量 override：`CONFIG`、`SWANLAB`。训练/评估参数应优先写在 YAML。

## 5. 输出

默认输出目录由 YAML 控制：

```text
<experiment.output_root>/<experiment.name>/<scenario>/mode<mode>/
```

核心文件：

- `summary.csv`：每条 run 的核心指标。
- `trajectories.csv`：闭环轨迹。
- `controller_log.csv`：逐步控制器日志。
- `controller_history.csv`：控制器可选内部历史。
- `*_debug.png`：单条 run 调试图。
- `summary_compare.png`：当前方法在多 run 上的轨迹对比图。

核心指标：

- `reach_time_s`：首次进入目标温度带的时间。
- `final_T_in_c`：实验结束时的室内温度。
- `energy_efficiency`：`thermal_kwh / electric_kwh`。

## 6. 推荐调试顺序

1. 在 YAML 中把 `train.epochs` 设为 `1`、`train.limit_transitions` 设为 `4096`。
2. 运行 `STAGE=train`，确认 WAM checkpoint 能生成。
3. 正式 `STAGE=eval` 默认使用 5 秒控制周期和 4 小时 horizon。
4. 若只是检查代码链路，可以临时复制一份 YAML，把 `eval.scenarios[].horizon_seconds` 缩短；不要把短 horizon 当作正式结果。
5. 需要上传时，在 `experiment.tracking.swanlab.enabled` 打开 SwanLab。
