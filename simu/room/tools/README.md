# 工具脚本

- `data_pipeline.py`：读取、修复时间戳并按完整实验划分数据；
- `train_continuous.py`：重新训练 V3；
- `evaluate_all.py`：逐条评估并生成图片和指标表。

从项目根目录执行：

```powershell
python -m simu.room.tools.train_continuous
python -m simu.room.tools.evaluate_all
python -m simu.room.tools.evaluate_all --data-dir data/dataset_eval --output-dir simu/room/output/dataset_eval_v3
```

重新训练会覆盖模块根目录的 `continuous_cooling_model.joblib`，操作前请备份。

