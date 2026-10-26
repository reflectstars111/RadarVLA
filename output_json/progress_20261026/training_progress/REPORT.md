# 训练恢复检查（2026-10-24）

## 08:03 进度更新

全量 9,841 条数据已在 SSD 完成原生验证；两组 Grounding 均完成 20/20 轮，三组 Risk readout 均完成 10/10 轮。初始化与训练样本顺序配对审计通过。测试集尚未评估。

Qwen 缓存于 01:26 因 SSD 剩余空间低于 150 GiB 预留而停止，导致六组 SFT 等待。08:02 检查时空间回升至约 156.4 GiB，两个大权重文件已完整缓存，仅缺四个小文件。按原限速与 150 GiB 保护续传后，新增读取 9,851,287 字节，九个文件指纹全部匹配，未重传大权重或雷达数据。

08:02:43 调度状态：baseline_sft 在 GPU 5 启动，其他五组 SFT 等待调度；启动状态包含模型初始化与 smoke 阶段，尚未据此确认优化器更新。

| 最后一轮验证指标 | baseline Grounding | assignment Grounding |
|---|---:|---:|
| Recall | 43.40% | 15.05% |
| F1 | 0.576 | 0.256 |
| 0.5s AP | 0.652 | 0.397 |
| 1s AP | 0.712 | 0.555 |
| 2s AP | 0.753 | 0.713 |
| 速度 RMSE (m/s) | 2.719 | 2.315 |

以上是末轮、当前解码阈值下的验证指标，不是最佳 checkpoint 或 test 结论。assignment 运动误差降低但检测召回明显下降，目前不能称为整体改进。三组 readout 的末轮 2s AP 分别为 baseline 0.774、risk 0.773、combined 0.759。

下方内容为启动时历史记录，其中“缓存进行中”不代表当前状态。

当前状态：单次限速 NAS→SSD 缓存进行中，正式训练尚未开始。缓存、全量本地验证、GPU 真实样本 smoke 依次通过后自动开始训练；任一前置检查失败不会绕过。

## 已确认

- SSH 可连接，现场可见多张空闲 L40S；GPU 可用性以启动时调度器的实时检查为准。
- 修复后的数据、pipeline、CLI 与 storage 已复制到独立 worktree，原有工作区及历史实验保持不动。
- 新分支 `codex/bottleneck-localcache-training-20261024`，源码提交 `2e176e8e6bd47d7d1142c5f971db70b060c1faea`。
- 独立虚拟环境导入路径已核对，指向新 worktree。
- 服务器相关回归检查 **121 passed in 40.01s**，包含数据、pipeline、训练启动、调度、检测分配、缓存与共享资源运行器。
- 已有 24 条样本的链路试跑不能替代完整 9,841 条数据验收。

## 当前执行与自动衔接

准备进程 PID：1076099；单个导入进程 PID：1076100。

1. 以 4 MiB/s、最多 20 次文件打开/秒单进程读取 NAS，去重缓存雷达文件；比对此前已验收实验保存的文件 SHA256，不修改 manifest、标签或划分。
2. 9,841 条记录、30,795 个唯一雷达文件的指纹全部匹配后，仅在 SSD 上运行原生全量验证。
3. 原生验证 PASS 后启动原共享 GPU 调度器；每个任务先检查真实样本的有限 loss/反向梯度，再运行正式训练。
4. Qwen 权重同样只复制一次并校验，SFT 等待权重缓存就绪。

不重新运行旧的 NAS 全量验证/恢复循环。保留旧 IO_PAUSE。发现 NFS 状态恢复计数持续异常时停止本导入进程，不自动重试。SSD 导入预留 150 GiB；不修改系统挂载，不操作其他人的进程。

## 训练参数与资源约束

沿用已定方案：Grounding 20 轮、Risk readout 10 轮、SFT 5 轮；不扩大为多卡 DDP，不更改模型、损失或学习率。独立实验可使用不同空闲卡。

CPU/BLAS 线程 1、data workers 0、nice 19、ionice idle；单 worker 显存预算与共享运行器的动态节流保持原配置，显存余量低于 8 GiB 只停止本 worker。

## 路径与查看方式

服务器控制目录：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.runs/RadarVLA-training-20261024`

- `PREPARATION.json`：准备阶段、子进程 PID。
- `NAS_PROGRESS.json`：实际读取字节、当前文件、更新时间。
- `STAGING.json`、`staging.log`：导入与权重缓存状态。
- `local_validation.log`：SSD 全量验证结果。
- `STATUS.json`、`coordinator.log`：正式训练队列启动后生成。
- `regression.log`：121 项回归结果。

SSD 缓存：`/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/mirrors/krsv2_c6fa46a048fd`

SSD 训练输出：`/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/model_runs/bottleneck_localcache_20261024`

本目录 `plan.json` 与 `paths.json` 是本次启动配置快照。准备进程启动不等于已执行优化器更新；应以任务训练日志及 epoch 指标为准。
