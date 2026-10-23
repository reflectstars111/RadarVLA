# 共享服务器上的瓶颈对照训练

本方案补充 `BOTTLENECK_8XL40S_RUNBOOK.md`。服务器上使用空闲或可共享的卡；不要求八卡同时空闲，也不把八卡 DDP 当作正式训练默认方式。当前只完成配置与启动命令验收，尚未启动本轮正式训练。

## 1. 代码与配置

代码工作区：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-bottleneck-fixes-20261023`。

数据根：`/home/luxinlan/data/lxl_data/RadarVLA`。

新增 profile：`configs/bottleneck_shared_single_gpu.json`。保留官方八卡 profile，五组模型配置、标签及初始化依赖不变。共享 profile 的 GPU 列表故意留空，必须显式传入调度器当前选中的一张卡；不指定 GPU 或选择多卡会在启动训练前报错。

| 阶段 | 轮数 | 每次前向 batch | 梯度累积 | 单卡有效 batch | LR |
|---|---:|---:|---:|---:|---:|
| Grounding | 20 | 4 | 2 | 8 | 2e-4 |
| Risk readout | 10 | 4 | 2 | 8 | 2e-4 |
| SFT | 5 | 1 | 32 | 32 | 5e-5 |

所有阶段 BF16、seed42、data workers=0；按 optimizer update 调度，前 5/2/1 轮恒定后余弦衰减到初始 LR 的 0.1。SFT 冻结 Grounder，启用 LoRA 与 gradient checkpointing。其余实验超参固定，测试集不参与选择。

有效 batch 一致不代表等价八卡训练：物理 batch、BatchNorm、DDP 样本补齐和更新顺序不同。所有新组必须采用相同共享 profile 重建基线，不能把历史八卡或旧单卡成绩直接当作单变量对照。实际尾部累积窗口与样本顺序以原生 protocol/metrics 为准。

## 2. 根据空闲资源并行独立实验

一个实验固定 world_size=1，一张卡最多一个本项目 worker。空闲卡多时，把不同实验放到不同卡并行；不把临时空出的卡动态加进正在运行的实验。

依赖顺序：

1. `baseline/grounding` 与 `assignment/grounding` 可以并行，各20轮，分别从头初始化。
2. baseline Grounder 完成后，`baseline/risk-readout` 与 `risk/risk-readout` 从同一个最终 checkpoint 开始，各10轮；assignment Grounder 完成后，`combined/risk-readout` 从它开始。
3. SFT 各5轮：baseline/action 使用 baseline Grounder，assignment 使用 assignment Grounder，risk/combined 使用对应读出结果。还需保留从 baseline 读出10轮结果开始的 token SFT 对照，才能区分额外读出训练和物理融合本身的规划收益。

默认输出统一位于 `<run_root>/bottleneck_shared_20261023/`，不读取八卡目录中的初始化权重。比较最终固定轮次和共同目标集合，并核对初始化哈希与样本顺序。combined 不能独立证明三项改动的各自收益。

## 3. 资源管理要求

profile 和原生 `bottleneck` CLI **不提供自动空闲卡调度、显存守卫或计算限速**。本配置不能单独视为安全调度器；正式训练须由共享 worker 包装运行，继续采用此前的保护机制：

- 采样多次 GPU UUID、剩余显存、利用率和现有任务负载，优先选择负载低且满足本阶段预算的卡。一次 `nvidia-smi` 快照不等于已获得资源。
- 不固定要求32GiB空闲或利用率≤10%；按真实样本实测的本阶段峰值和优化器预算，加8GiB余量准入。Grounding/Risk 可先参考旧6GiB预算，但新模块和SFT必须重新测量，不能套用旧峰值。
- CPU/BLAS线程1、data workers=0、nice=19、ionice idle。共享卡通过每批后同步并休息控制本worker的计算占用；nice本身不限制GPU，休息比例也不等于驱动级GPU配额。
- 监控GPU剩余显存，低于8GiB只退出本worker。其他用户进程存在本身不触发停止；若计算争用明显，则降低自身运行比例或等待。不操作其他用户进程。
- 一个worker的资源中断不能停止其他独立实验。各自保留日志与checkpoint，只按原协议恢复。正式迁移到另一张卡仍固定world_size=1、batch、累积、初始化和数据；先验证原生恢复契约，不能修改历史指纹绕过检查。
- 不在训练中更改源码、profile或运行包装脚本。新分支需要独立editable环境；不得把共享Conda或已有实验环境重定向到这里。

正式启动前还有待完成的验收：共享worker适配与保护测试、新结构真实样本前向/反向、SFT本地Qwen路径和显存实测。当前dry-run只证明参数与依赖路径可解析，不证明显存足够、训练正确或已实现上述保护。

## 4. 只打印启动命令

下例的 `3` 是占位示例；正式执行由调度器根据实时资源选择，不能照抄后裸跑。

```bash
W=/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-bottleneck-fixes-20261023
cd "$W"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /usr/bin/python3 scripts/radarvla_cli.py --project-root "$W" \
  --paths configs/local_paths.json bottleneck \
  --profile configs/bottleneck_shared_single_gpu.json \
  --arm baseline --stage grounding --gpus 3 --dry-run
```

此命令不导入模型、不申请GPU、不创建训练结果。正式运行前，另设服务器路径JSON指向新分支独立环境和主目录现有Qwen权重；分支不含本地权重。不要修改旧配置或安装环境后直接启动八卡命令。

## 5. 本轮代码的验证边界

检测分配、预测物理Risk融合、连续动作头是三个显式实验项，读取新增实现文档时需注意：已有683项测试来自开发环境，不能替代服务器真实样本验收；`max_agents=8`只限制语言描述，不裁剪检测/Risk候选；连续动作action/combined目前支持SFT与评估，不支持现有token RL。

本次只验证共享profile的十条阶段启动路径、有效batch、初始化路径和单卡保护校验；不宣称新增真实训练结果。
