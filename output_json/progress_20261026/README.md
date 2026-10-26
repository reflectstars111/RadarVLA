# RadarVLA 进度归档：2026-10-26

本轮训练11/11项完成，完整规划validation/test评估12/12项完成。数据版本、checkpoint选择及评估定义均已固定，没有利用test调整模型或阈值。

## 从这里开始

- [结果结论与下一步](RESULTS_SUMMARY.md)
- [完整评估报告](EVALUATION_REPORT.md)：六组模型，validation1451帧、test1522帧；规划、覆盖率、SHORT、危险目标、Risk与离线物理指标。
- [评估口径](METRIC_SCOPE.md)：checkpoint选择、3秒轨迹与2秒Risk几何窗口、无效输出及动作优先推理范围。
- [原始完整汇总](FULL_RESULTS.json)与[相同覆盖点配对比较](PAIRED_COMPARISON.json)。
- [初始化审计](INITIALIZATION_AUDIT.json)：Risk SFT和等预算对照的planner初始化不同，因此其SFT差异不能完全归因于Risk输入。

## 关键test结果

|组别|ADE(m)|3s FDE(m)|有效输出|SHORT召回|2s危险目标检出|
|---|---:|---:|---:|---:|---:|
|baseline|1.493|3.183|100%|2/40|11/40|
|action|0.822|1.698|97.31%|3/40|11/40|
|assignment|1.810|3.851|100%|5/40|5/40|
|risk|2.184|4.958|100%|1/40|11/40|
|combined|0.794|1.639|97.31%|3/40|5/40|
|baseline readout对照|1.653|3.435|100%|1/40|11/40|

误差只计有效覆盖点；配对比较仍支持action的轨迹改善，但危险识别、SHORT触发与加速度超限样本尚未解决。combined整体均值略低，不代表危险场景更好。离线碰撞结果不等价于闭环安全。

## 文件与复核

- `training_records/`：11项训练的完整epoch指标、原生protocol、初始化、状态、execution与监督统计。较大的`metrics.jsonl`和`protocol.json`使用gzip压缩；`MANIFEST.json`记录压缩前后SHA256，gzip解压即可查看。`regression_test_result.txt`记录121项回归通过。
- `results/<arm>/<val或test>/`：可直接浏览的原生报告、分时窗统计及危险帧诊断。
- [逐样本完整结果包](full_planning_eval_20261026.tar.gz)：约46.2 MiB，包含12组逐样本`predictions.jsonl`及评估审计；其SHA256为`c1c140a33191e6cad56cd7483e7fefc8b0e71e104208a02eae2477d616b0fad7`，见`ARCHIVE_READY.json`。
- `ARCHIVE_MANIFEST.json`与`LOCAL_VERIFICATION.json`：归档内149个文件已经校验。此清单针对原始结果包，不包含发布时新增的本README与结论附录。
- `training_progress/`和`quota_recovery/`：历史阶段记录。最新结论以上面的完整评估报告为准。
- 根目录评估脚本、`plan.json`、`ENVIRONMENT.json`与调度审计记录保留实际执行过程。运行路径是服务器路径快照，复现时需要将数据与本地权重放到对应位置或显式配置等价路径。

## 代码和数据范围

训练源码提交：`2e176e8e6bd47d7d1142c5f971db70b060c1faea`，已包含在本进度分支历史中。发布提交只增加结果和文档，未改动训练算法。

manifest SHA256：`c6fa46a048fd0577ed0b2dc7a811131eac5d345e75f571749b2b3a82283e1672`。原始雷达、Qwen权重、checkpoint和Conda环境仍保留在服务器，不属于这次Git进度归档。

本轮使用单seed；按用户要求未新增三seed训练。更早的实验记录保留在远端`codex/progress-20261023`分支。
