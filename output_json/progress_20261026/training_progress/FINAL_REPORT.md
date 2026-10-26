# 最终训练进度（2026-10-26 10:00核实）

后续更新：完整validation与test评估已于2026-10-26 20:20全部完成，见[完整评估结果](../EVALUATION_REPORT.md)及[结论分析](../RESULTS_SUMMARY.md)。下文是训练结束时的历史记录。

本轮11项任务全部完成：Grounding两组各20轮、Risk readout三组各10轮、SFT六组各5轮。

最后完成的是risk_sft：配额清理后从第2轮checkpoint恢复，在2026-10-25 11:20:44完成第5轮，worker退出码0，恢复监督器和总队列均为complete。原中断的第3轮从轮次起点重跑，没有把半轮当作完成轮。

## 第5轮规划验证

固定32条验证样本，误差仅针对有效生成的轨迹点；不是完整验证集生成评估、test评估或闭环安全结论。各组预测short/long数量及有效点覆盖存在差异，不能只按ADE/FDE排序。

| SFT组 | Ego ADE(m) | Ego FDE(m) | 输出有效率 | 全3秒网格点覆盖率 | 教师要求点覆盖率 | SHORT召回 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.897 | 4.807 | 100% | 83.33% | 100% | 50.0% |
| action | 1.181 | 2.816 | 96.875% | 80.21% | 98.4375% | 50.0% |
| assignment | 2.177 | 5.377 | 100% | 85.42% | 100% | 43.75% |
| risk | 2.012 | 5.218 | 100% | 83.33% | 100% | 50.0% |
| combined | 1.213 | 2.924 | 96.875% | 82.29% | 98.4375% | 43.75% |
| baseline readout对照 | 2.036 | 5.289 | 100% | 83.33% | 100% | 50.0% |

以上均为固定末轮，不能当作各组最佳checkpoint的公平最终比较。原生best_planning对应epoch分别为baseline=3，其余五组=5。

## 当前可支持的结论

- action是最值得继续验证的改动：这32条样本上末轮ADE/FDE最低，但存在1条无效输出。
- combined的末轮常规ADE/FDE没有超过action，其SHORT召回更低；教师要求点ADE则为0.621m，低于action的0.653m，说明不同覆盖口径会改变排序，不能笼统判定全面更好或更差。
- risk与等预算baseline readout对照的Risk AP仍接近；其规划常规ADE/FDE略好，但教师要求点ADE更差（1.305 vs 1.058m）。当前没有一致的新增几何输入收益。
- assignment检测召回下降的既有问题仍需定位，SFT冻结Grounder不会自动修复该问题。
- test未评估，尚无本轮完整验证集生成报告、危险样本分组和闭环碰撞评估，不宣称最终安全性或泛化提升。

建议下一步按验证集预先固定checkpoint选择及比较口径，优先完成baseline/action的完整规划验证，并同时保留有效率、教师要求点覆盖率、SHORT召回及危险目标召回；完成选择后再做test评估。

机器可读状态见final_progress_20261026.json。服务器控制目录仍为`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.runs/RadarVLA-training-20261024`；模型输出仍在SSD的`/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/model_runs/bottleneck_localcache_20261024`。
