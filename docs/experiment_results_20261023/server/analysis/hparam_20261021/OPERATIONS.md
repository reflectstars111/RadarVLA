# 本轮执行与结果位置

最新指令：三seed验证暂缓。本轮后续仅seed42；以`SINGLE_SEED_AMENDMENT.json`及`hparam_campaign_single_seed.py`为准。原三seed协议作为历史记录保留，不修改活动训练的哈希或checkpoint。当前40轮worker继续，seed43/44没有排入实际队列。

用户于2026-10-22授权继续执行至最后。主源码保持原状态；全部改动在隔离分支`codex/hparam-20261021`，提交`c6c1d43a4ea9bd19aa3fe3ab25c0c88c4f8fa3a1`。

- 服务器分析与持续报告：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/hparam_20261021/REPORT.md`。
- 精确实时状态：同目录`campaign_state.json`；`campaign_launch.json`记录coordinator PID。
- 每组配置：`jobs/<job>.json`；进度、显存/CPU设置、日志和逐轮诊断：`runs/<job>/`。
- 模型与原生训练输出：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/hparam_20261021/campaign/<job>/attempt_XX/`。
- 原始moving-v1 manifest路径和SHA、源码、运行脚本SHA已固化于`campaign_protocol.json`。不要在活动实验中编辑受哈希保护的源码或脚本。
- 最终判断依据：`MULTISEED_COMPARISON.json`、`FINAL_SELECTION_LOCK.json`；最终test存在时见`FINAL_TEST.json`。最终test不用于再次调参。

队列依次执行4组LR/调度、batch对照、最多2个40轮配方和三seed确认；100轮仅在预定条件满足时执行。运行条件见`EXECUTION_RULES.md`，并非承诺所有阶段必定产生精度收益。

coordinator持有`campaign.lock`排他锁；重复启动会被拒绝。它的PID退出后，worker在心跳失联后等待并在300秒内自行退出，仅影响本worker。技术失败会写入`campaign_error.json`或state的failed字段；不会忽略失败直接进入后续阶段。资源退出最多重排3次；恢复要求相同协议，不要更改旧实验的epochs、配置或指纹。

只读报告观察器持续更新REPORT.md，不操作GPU、训练或选择规则。它的PID记录于`report_observer_pid.json`。本地报告是同步时快照；服务器内容在运行中会继续更新。

训练是Grounder阶段，未扩大到SFT、Planner训练、闭环验证或论文效果结论。CPU线程1、data workers0、nice19、ionice idle；显存预算和GPU占空比动态控制，不管理其他用户进程。
