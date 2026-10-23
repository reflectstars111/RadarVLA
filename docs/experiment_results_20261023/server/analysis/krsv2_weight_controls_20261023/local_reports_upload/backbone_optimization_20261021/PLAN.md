# 2026-10-21 骨干训练优化

目标：在原有 CNN / HG / T-RODNet 五轮对照基础上，检查更低学习率与真实 batch 大小是否能改善泛化，并同时衡量检测、运动和五维 Risk。所有新训练从头开始，保留原始实验。

## 已有证据

原实验学习率为 3e-4、真实 batch=1、梯度累积=8。三组训练损失持续下降，但验证损失在第2–3轮达到最低后回升。T-RODNet 第5轮 Pcol3 AP=0.3420，但检测误报1771，precision=0.2745，因此不能只依据召回或 Risk 的单项指标选模型。

batch=4 的代表性真实样本前向/反向均通过有限数值检查，包含空目标和14目标样本。峰值分配显存：HG 1131 MiB，T-RODNet 2207 MiB；尚未包含 Adam 状态，估计分别再需495 / 242 MiB。这些是容量估算，不代表正式训练全过程上限。

## 实验矩阵

| 实验 | 骨干 | 学习率 | 真实 batch | 累积步数 | PyTorch 分配器预算 |
| --- | --- | ---: | ---: | ---: | ---: |
| hg_lr1e4_b1 | HG | 1e-4 | 1 | 8 | 4 GiB |
| hg_lr1e4_b4 | HG | 1e-4 | 4 | 2 | 4 GiB |
| trodnet_lr1e4_b1 | T-RODNet | 1e-4 | 1 | 8 | 4 GiB |
| trodnet_lr1e4_b4 | T-RODNet | 1e-4 | 4 | 2 | 6 GiB |

另一个 GPU worker 对已有 T-RODNet 第3、5轮执行完整验证集诊断：阈值0.5 / 0.7 / 0.9下的召回、误报、precision，以及误报与静止车、行人、骑行者、已匹配运动目标的几何关联。关联只是定位线索，不能证明错误原因。诊断不更新权重、不改变主要评估阈值0.5。

所有新训练：moving-v1同一 manifest、seed=42、有效 batch=8、5 epochs、BF16、原生 AdamW 和损失。固定第5轮作主要比较，保留各轮完整指标；完成后用同一批共同检出目标比较速度、径向速度、轨迹误差，并统计危险车辆召回。真实 batch=4 同时改变 BatchNorm 统计与原生 masked loss 的聚合，不能解释为单独的 BN 消融。

## 资源与他人任务保护

- 每张卡最多一个本项目新 worker；优先选没有其他计算进程的卡，连续两次检查确认后启动。取消固定32 GiB空闲、固定10%利用率和固定40%分配上限，改用上述实测预算；启动要求空闲显存至少为该预算加2 GiB。
- 约每8–10秒记录 GPU 进程和显存；其他用户计算进程的出现本身不触发暂停。优先使用空闲卡，已有训练继续运行；观察到实际资源争用时再调整本项目任务。保留显存压力退出和调度器失联保护。
- 独立守护线程每3秒检查显存；空闲低于2 GiB仅退出当前 worker。最多自动重排3次；有完整 last.pt 则从上个完整 epoch 恢复，否则新建 attempt，保留失败目录。不会发信号给其他用户进程。
- 调度器超过45秒无心跳则 worker 暂停计算；显存守护仍工作。
- CPU线程1，data workers=0，nice=19，ionice=idle；各任务启动间隔至少30秒，减少同时读取元数据的突发压力。
- 分配器预算不包含 CUDA 上下文等框架外开销；实时整卡空闲监控补充保护。共享GPU和存储无法保证零干扰，因此使用空闲卡、低优先级和持续让行。

## 源码与输出

原项目主目录及旧训练目录不修改。继续使用独立 worktree 提交 `943167941bf0d75c9bcd3614f7f931c4349645c9` 与独立虚拟环境；实验脚本、协议和日志存放于数据目录，校验源码、manifest、配置、实验脚本指纹后训练。

服务器诊断目录：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/backbone_optimization_20261021`

服务器权重目录：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/backbone_optimization_20261021`

`status.json` 记录运行卡号、PID和任务状态；各任务 `active.json` 指向当前 attempt；`REPORT.md` / `comparison.json` 自动汇总训练、检测、运动、Risk结果。这里只训练 Grounder，不能据此宣称 Planner、SHORT 模式或闭环驾驶已经改善。数据使用双向 Doppler 离线预处理，moving-v1 中静止车、行人、骑行者为背景。
