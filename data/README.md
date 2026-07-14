# 数据约定

本仓库只保留数据读取代码，不提交真实 CSV、xlsx、截图或整理结果。训练和评估时请在运行环境外部提供数据目录。

默认布局：

```text
data/
  dataset/
    X1/
      status_data_*.csv
    X2/
    ...
    X27/
    unclassified/
  raw/
  split_manifest.csv
```

`data/dataset` 是 HanWAM 默认训练入口，按工况目录保存整理后的 `status_data_*.csv`。`split_manifest.csv` 用于从整理版文件恢复 train/val/test 来源划分。`data/raw` 只作为原始采集归档，不是默认训练入口。

测试不会读取仓库内真实数据。需要数据相关单测时，测试会在临时目录生成 mock CSV。
