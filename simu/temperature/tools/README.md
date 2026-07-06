# 工具脚本

- `data_pipeline.py`：读取、清洗并按实验轮次切分训练CSV；
- `online_utils.py`：窗口构造和评估指标；
- `train_online_history.py`：重新训练当前在线模型；
- `evaluate_online.py`：复现1小时在线评估；
- `evaluate_long_horizon.py`：复现2/3小时外推评估。

从项目根目录执行，例如：

```powershell
python tools/evaluate_online.py
```

重新训练会覆盖根目录的 `online_history_ensemble.joblib`，操作前请先备份正式模型。
