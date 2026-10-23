# 20轮训练与学习率衰减受控验证

目标：验证候选范围统一后，第5轮是否尚未收敛，以及训练更久、学习率衰减各自带来的变化。保持真实batch=4、累积=2、有效batch=8，不同时做batch扩展。

## 四组实验

| 任务 | 骨干 | 总轮数 | 学习率 |
| --- | --- | ---: | --- |
| hg_constant20 | HG | 20 | 始终1e-4 |
| hg_cosine20 | HG | 20 | 前5轮1e-4，后15轮余弦衰减，第20轮1e-5 |
| trodnet_constant20 | T-RODNet | 20 | 始终1e-4 |
| trodnet_cosine20 | T-RODNet | 20 | 前5轮1e-4，后15轮余弦衰减，第20轮1e-5 |

全部从头初始化，seed=42，BF16，原生AdamW、weight decay和梯度裁剪保持不变；同一moving-v1数据、同一候选范围契约、同一训练采样顺序。已有5轮实验与权重完整保留，不伪造resume，也不修改原协议中的预算和源码指纹。

比较顺序：每组第5→20轮观察延长训练；同骨干constant20与cosine20比较衰减；第10/15轮作为辅助趋势。固定第20轮为主要终点，同时报告验证loss和检测F1选择的最佳轮次，不事后只挑最好的一轮。前5轮两个计划的学习率完全一致；完成后比较第5轮权重与已有5轮实验的逐位一致性，记录可能的数值差异。

每轮约856次参数更新；20轮约17120次。余弦组第5/10/15/20轮学习率分别为1e-4、7.75e-5、3.25e-5、1e-5。计划由总预算和epoch解析计算，写入协议；改变计划的resume被拒绝，恢复不丢失衰减进度。

## 每轮验证

- 完整1472帧验证集；不评估test，不训练LLM或Planner。
- 检测TP/FP/FN、precision/recall/F1，以及35个危险目标实例的检出数。
- Pcol1/2/3的AP、Brier等原生指标，dmin/areq误差；解码匹配目标的速度、径向速度和轨迹误差。
- 复用原生验证的同一批前向结果记录对象ID；另存每轮逐样本诊断，便于对第5/20轮共同检出集合做配对比较，不用各模型不同的检出集合冒充运动改善。

## 代码、数据与回退

主代码根目录始终是 `/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA`。本次使用该仓库的独立worktree：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-long-training-20261021`，分支 `codex/long-training-20261021`，基于候选契约提交 `3b90c785be32adb5350f1be959a80036d2be6107`。

数据根目录始终是 `/home/luxinlan/data/lxl_data/RadarVLA`。

- manifest：`prepared/taskwise_moving_v1/manifest.jsonl`，SHA256 `6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e`。
- 任务、协议、日志和报告：`analysis/long_training_20261021`。
- 权重与独立Python环境：`model_runs/long_training_20261021`。

不修改主项目已有脏文件，不修改前两次隔离实验的源码和输出；回退可直接使用它们原来的环境和分支。

## 资源与验证

优先使用空闲GPU并行四组，每卡最多一个本次worker。HG显存分配器预算4 GiB、T-RODNet 6 GiB；启动多留2 GiB，实时空闲低于2 GiB仅退出本worker。其他用户进程出现本身不触发暂停。CPU线程1、data workers=0、nice19、ionice idle；错开启动，保留断点与失败attempt，自动内存压力重排最多3次。

96项相关测试通过，包含解析学习率端点、无效配置、实际训练的连续/恢复逐位一致性、前缀固定学习率一致性，以及候选范围、检测、指标、启动器和骨干回归。真实样本验证观察器已检查：不增加模型前向次数，原生验证指标保持逐项一致，且不执行优化器更新。

这些是实验准备与正确性验证，不代表训练20轮一定更好。单个种子、35个危险目标实例的结果只能作为下一步依据。
