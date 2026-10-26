# 完整规划验证与test评估执行计划

执行已获授权。沿用训练提交`2e176e8e6bd47d7d1142c5f971db70b060c1faea`，不修改主仓库或冻结worktree。

## 固定模型与数据

- 六组：baseline、action、assignment、risk、combined、baseline_readout等预算对照。
- 每组固定原训练`best_planning.pt`，baseline为SFT epoch3，其余epoch5；具体SHA256见plan.json。
- 先完成六组完整validation，再对相同checkpoint做六组test。完整validation不重新选checkpoint，test不参与模型或阈值选择。
- 同一已验收数据版本：validation1451条、test1522条，无截取、无新增标签或数据划分。
- 全程仅访问SSD雷达、Qwen和checkpoint缓存；保留原始训练结果。

## 计算与接口

原生`pipeline.evaluate_checkpoint`，batch1，预测Risk、checkpoint默认risk projection；不做优化器更新。保存原生报告和全部逐样本predictions.jsonl。

调度从2个worker启动，发现更多空闲卡后由独立控制器接管，最多4个独立worker。每个worker独占一个调度名额，不使用DDP；准入显存余量≥26GiB、利用率≤60%，优先无其他计算进程的卡，连续两次观测通过才准入。18GiB显存预算；共享卡按既有策略节流；显存余量低于8GiB只退出自身worker；CPU/BLAS线程1，data workers0，nice19，ionice idle。不会因出现其他用户进程直接暂停或终止其任务。

控制器接管没有重启已运行评估；资源并行度变更单独记入CONTROL_CHANGE.json，不改变checkpoint、源码或评估定义。评估脚本、原训练源码指纹和checkpoint指纹均校验。

## 指标与产物

原生：Risk AP/校准、检测与运动误差、输出有效率、ADE/FDE、轨迹点覆盖率、教师要求点覆盖率与ADE、SHORT召回、离线物理碰撞与指令指标。

补充：同一预测在0.5/1/2/2.5/3秒的原生轨迹误差、0.5/1/2秒危险目标召回及“漏检且Risk<0.5”的逐帧诊断。已有per-agent标签按其known mask使用，不将未知当安全。匹配遵循原生2m Hungarian，汇总需与原生总TP/FN完全一致。

补充统计已通过合成样本检查：目标匹配、未知标签、漏检与Risk联合诊断、短轨迹的分时窗FDE与覆盖率。

GPU共享时的时延不用于宣称独占推理性能；离线碰撞不等价闭环安全。当前预处理仍为既有离线数据版本。

## 查看结果

服务器根目录：`/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/evaluations/full_planning_20261026`

- STATUS.json：12项评估状态。
- 各任务attempt目录：worker_status、progress、resource与日志。
- results/<arm>/<val或test>/report.json：原生完整指标。
- predictions.jsonl：逐样本预测。
- supplemental.json与dangerous_frames.json：分时窗与危险目标诊断。
- SUMMARY.json、REPORT.md：随任务完成更新。
- FULL_RESULTS.json、FINAL_REPORT.md：补充统计与最终汇总。
- FINALIZER_STATUS.json：补充统计是否完成。

本地目录为本计划所在位置。任务完成后同步完整报告与逐组指标；没有最终报告时不能声称全量评估完成。
