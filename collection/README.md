# CARLA 数据采集

`collector/` 保存当前 v5 原始采集、道路走廊、校验及预览代码；`tests/` 是离线回归，`notebooks/` 是逐帧联合预览。`kradar_research/` 保存独立的 K-Radar 研究脚本。`weights/` 是本地权重，Git 忽略，目录整理未改动其内容。

采集格式见[数据契约](../docs/collection/DATA_COLLECTION_SPEC.md)，各阶段命令见[runbook](../docs/collection/RAW_COLLECTION_RUNBOOK.md)，最近一次实测见[v5 验收报告](../docs/collection/reports/v5_smoke_20261014.md)。真实路线与场景文件由运行命令显式传入；此目录不附带自写路线或触发点。
