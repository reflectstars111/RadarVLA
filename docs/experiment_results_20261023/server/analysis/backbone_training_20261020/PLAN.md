# CNN / HG / T-RODNet 正式训练对照

队列于2026-10-20深夜（Asia/Shanghai）启动，实验ID保留为`backbone_training_20261020`；后续状态时间在JSON中用UTC记录。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 实现分支 | `codex/spatiotemporal-backbones-20261020` |
| 实现提交 | `943167941bf0d75c9bcd3614f7f931c4349645c9` |
| 训练代码 | 原生`pipeline.run_training`，不修改损失或训练循环 |
| 顺序 | `cnn2d` → `rodnet_hg` → `trodnet`，串行 |
| 数据 | 已验收`prepared/taskwise_moving_v1/manifest.jsonl` |
| 数据SHA256 | `6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e` |
| 训练/验证 | 完整训练6847帧、验证1472帧；不执行测试集推理 |
| 初始化 | 每组从头初始化，不加载旧模型 |
| 预算 | 每组5个epoch，seed=42 |
| 优化 | 原生AdamW，lr=0.0003，weight decay=0.0001，梯度范数裁剪5 |
| 有效batch | 单卡batch=1，累积8步；epoch末不足8帧的窗口由原生代码处理 |
| 精度 | BF16 |
| 主要比较 | 固定第5轮，同时展示所有轮次，不挑选最有利epoch |

当前阶段只训练Grounder，不训练LLM/SFT。每轮由原生代码计算检测、运动及五维Risk指标；每组第5轮结束后，在完整验证集额外记录逐目标匹配与Risk，汇总危险车辆召回及跨模型共同检出集合的速度/轨迹误差。训练使用相同的独立DataLoader种子，原生配置默认检测阈值0.5、NMS=0.3、匹配距离2m均保持一致。

## 资源与源码保护

每次只运行一个GPU训练进程，每组启动前重新查询GPU，要求空闲显存≥32GiB且利用率≤10%。worker分配显存上限40%，余量低于8GiB时只停止本worker；CPU线程1、data workers=0、nice=19、ionice=idle。失败时停止后续队列，不静默重试或覆盖实验。

首组CNN选择GPU 6，启动时空闲44238MiB、利用率0%。协调进程PID为338548，首组worker PID为338555；PID仅为启动记录，检查或操作前应重新核实进程身份。

代码worktree：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-backbones-20261020`。训练执行器和报告脚本放在数据分析目录中，不修改已验收、已提交的源代码。

分析目录：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/backbone_training_20261020`。

模型目录：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/backbone_training_20261020/{cnn2d,rodnet_hg,trodnet}`。

`protocol.json`固定源码和配置哈希。每组均会核验manifest，原生训练另外保存传感器文件指纹及checkpoint来源。原项目main目录、原Conda环境、旧实验和检查点保持独立。

## 查看进度与结果

- 总队列：分析目录的`status.json`、`queue.log`。
- 单组进度：分析目录对应组的`worker_status.json`、`training.log`，模型目录对应组的`status.json`。
- 每轮结果：模型目录对应组的`metrics.jsonl`，以及分析目录自动更新的`REPORT.md`、`comparison.json`。
- 第5轮逐样本审计：分析目录对应组的`final_val_samples.jsonl`、`final_validation.json`。

`preparing_training`包含数据加载、完整校验和传感器哈希，不表示已经执行优化器更新。原生状态进入`training`后才开始训练循环。队列按30秒间隔更新状态和报告。

## 解释边界

当前manifest只将当前速度≥0.1m/s的车辆作为Agent；静止车辆、行人、自行车作为背景，Risk也已重新计算。结论仅对应这一任务定义。双向Doppler预处理仍是离线数据，不宣称严格在线因果。

比较检测召回时同时检查误报和Precision；比较速度/轨迹误差时明确匹配集合。五维Risk分别看Pcol1/2/3的AP、校准及dmin/areq的有效标签误差。Grounder的Pcol阈值召回不是LLM的SHORT召回，也不能代表rollout改善。单个种子仅提供初步对照，不能替代多种子统计。
